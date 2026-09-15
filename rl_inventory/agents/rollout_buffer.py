"""
agents/rollout_buffer.py
========================
Bo dem Rollout cho IPPO (Independent Multi-Agent PPO).

PHIEN BAN v2:
  [V2-12] XU LY CAT EPISODE (truncation) NAM GON TRONG BUFFER.
          Ban cu bat train.py tu cong gamma*V(s_T) vao phan thuong cuoi, roi
          trong compute_gae lai nhan bootstrap voi (1 - episode_end) de "khu"
          no di. Hai cho bu tru cho nhau nen dung, nhung cuc ky de hong khi
          sua mot ben. Nay buffer nhan truc tiep next_value cua tung buoc;
          cong thuc GAE la cong thuc chuan, khong con meo mo.

  [V2-13] Chuan hoa advantage theo TUNG CAP (mac dinh) hoac toan cuc. Them
          tuy chon dua `values` vao batch de ho tro value clipping.
"""

import torch
import numpy as np
from typing import Generator, Dict, Optional, Union


class RolloutBuffer:
    """Bo dem thu thap du lieu rollout cho IPPO."""

    def __init__(self, buffer_size: int, obs_per_pair: int, n_pairs: int,
                 device: str = "cpu", normalize_per_pair: bool = True):
        self.buffer_size = buffer_size
        self.obs_per_pair = obs_per_pair
        self.n_pairs = n_pairs
        self.device = torch.device(device)
        self.normalize_per_pair = normalize_per_pair
        self.reset()

    def reset(self):
        T, P, O = self.buffer_size, self.n_pairs, self.obs_per_pair
        self.states      = np.zeros((T, P, O), dtype=np.float32)
        self.actions     = np.zeros((T, P), dtype=np.int64)
        self.log_probs   = np.zeros((T, P), dtype=np.float32)
        self.rewards     = np.zeros((T, P), dtype=np.float32)
        self.values      = np.zeros((T, P), dtype=np.float32)
        self.next_values = np.zeros((T, P), dtype=np.float32)
        self.has_next    = np.zeros(T, dtype=bool)
        self.terminated  = np.zeros((T, P), dtype=np.float32)
        self.episode_end = np.zeros((T, P), dtype=np.float32)

        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None
        self.ptr = 0
        self._flat_cache: Optional[Dict[str, Union[torch.Tensor, int]]] = None

    def add(self, state, action, log_prob, reward, value,
            terminated: bool = False, episode_end: bool = False,
            next_value: Optional[np.ndarray] = None):
        t = self.ptr
        self.states[t]      = state
        self.actions[t]     = action
        self.log_probs[t]   = log_prob
        self.rewards[t]     = reward
        self.values[t]      = value
        self.terminated[t]  = float(terminated)
        self.episode_end[t] = float(episode_end)
        if next_value is not None:
            self.next_values[t] = next_value
            self.has_next[t] = True
        self.ptr += 1

    def compute_gae(self, last_values: np.ndarray,
                    gamma: float = 0.99, gae_lambda: float = 0.95):
        """[V2-12] GAE chuan, doc lap cho tung cap kho-SKU.

            delta_t = r_t + gamma * V(s_{t+1}) * (1 - terminated_t) - V(s_t)
            A_t     = delta_t + gamma * lambda * (1 - episode_end_t) * A_{t+1}

        V(s_{t+1}) lay tu next_values[t] neu duoc cung cap (buoc cuoi episode,
        hoac buoc cuoi buffer), nguoc lai lay values[t+1].
        """
        T = self.ptr
        rewards = self.rewards[:T]
        values  = self.values[:T]
        term    = self.terminated[:T]
        ep_end  = self.episode_end[:T]

        advantages = np.zeros((T, self.n_pairs), dtype=np.float32)
        gae = np.zeros(self.n_pairs, dtype=np.float32)

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = (self.next_values[t] if self.has_next[t] else last_values)
            elif self.has_next[t]:
                next_value = self.next_values[t]
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value * (1.0 - term[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1.0 - ep_end[t]) * gae
            advantages[t] = gae

        returns = advantages + values

        if self.normalize_per_pair:
            mu = advantages.mean(axis=0, keepdims=True)
            sd = advantages.std(axis=0, keepdims=True)
        else:
            mu = advantages.mean()
            sd = advantages.std()
        advantages = (advantages - mu) / (sd + 1e-8)

        self.advantages = advantages
        self.returns = returns
        self._build_flat_cache(T)

    def _build_flat_cache(self, T: int):
        total = T * self.n_pairs
        to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(self.device)
        self._flat_cache = {
            "states":     to(self.states[:T].reshape(total, self.obs_per_pair)),
            "actions":    to(self.actions[:T].reshape(total)),
            "log_probs":  to(self.log_probs[:T].reshape(total)),
            "advantages": to(self.advantages.reshape(total)),
            "returns":    to(self.returns.reshape(total)),
            "values":     to(self.values[:T].reshape(total)),
            "total":      total,
        }

    def get_batches(self, batch_size: int) -> Generator[Dict[str, torch.Tensor], None, None]:
        if self._flat_cache is None:
            raise RuntimeError("Phai goi compute_gae() truoc khi goi get_batches().")
        total = self._flat_cache["total"]
        indices = torch.randperm(total, device=self.device)
        for start in range(0, total, batch_size):
            idx = indices[start:start + batch_size]
            yield {k: self._flat_cache[k][idx]
                   for k in ("states", "actions", "log_probs",
                             "advantages", "returns", "values")}
