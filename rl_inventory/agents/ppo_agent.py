"""
agents/ppo_agent.py
===================
IPPO (Independent Multi-Agent PPO) voi chia se tham so hoan toan:
mot bo mang Actor-Critic duy nhat dung chung cho toan bo n_pairs tac tu.

PHIEN BAN v2 - sua cac nguyen nhan chinh khien huan luyen khong hoi tu:

  [V2-8]  TACH ACTOR VA CRITIC THANH HAI MANG RIENG.
          Ban cu dung chung feature_net. Value loss co do lon ~1.100 con policy
          loss ~0,004; gradient tu value loss lon gap ~290 lan. Vi
          clip_grad_norm_ ap cho TOAN BO tham so mang (chuan tong ~28 > 0,5),
          moi gradient bi nhan 0,018 -> lr_actor hieu dung tut tu 1e-4 xuong
          ~1,8e-6. Do la ly do entropy dung yen o muc toi da luc dau roi sup
          ve 0 mot cach that thuong sau do. Tach mang la cach sua trieu de.

  [V2-9]  CAT GRADIENT RIENG cho actor va critic.

  [V2-10] CHUAN HOA MUC TIEU GIA TRI (value normalization, kieu MAPPO/PopArt
          rut gon). Critic hoc tren return DA CHUAN HOA bang trung binh/do lech
          chuan truot; khi tra ve moi truong thi khu chuan hoa. Giu value loss
          o do lon O(1) bat ke thang do chi phi.

  [V2-11] VALUE CLIPPING (tuy chon) de mot update khong keo critic di qua xa.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Optional, Dict, Any

from agents.rollout_buffer import RolloutBuffer


def _mlp(in_dim: int, hidden: int, out_dim: int, out_gain: float) -> nn.Sequential:
    net = nn.Sequential(
        nn.Linear(in_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)
    nn.init.orthogonal_(net[-1].weight, gain=out_gain)
    return net


class ValueNormalizer:
    """[V2-10] Chuan hoa muc tieu gia tri bang trung binh/phuong sai truot."""

    def __init__(self, beta: float = 0.999, eps: float = 1e-5):
        self.mean = 0.0
        self.var = 1.0
        self.beta = beta
        self.eps = eps
        self.count = 0

    def update(self, x: torch.Tensor):
        m = float(x.mean().item())
        v = float(x.var(unbiased=False).item())
        self.count += 1
        b = min(self.beta, 1.0 - 1.0 / self.count)   # bias correction luc dau
        self.mean = b * self.mean + (1 - b) * m
        self.var = b * self.var + (1 - b) * v

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.var, self.eps)))

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d):
        self.mean = d.get("mean", 0.0)
        self.var = d.get("var", 1.0)
        self.count = d.get("count", 0)


class SharedActorCriticNetwork(nn.Module):
    """[V2-8] Actor va Critic la HAI mang doc lap (van chia se giua cac tac tu)."""

    def __init__(self, obs_per_pair: int, n_action_levels: int, hidden_dim: int = 128,
                 shared_trunk: bool = False):
        super().__init__()
        self.obs_per_pair = obs_per_pair
        self.n_action_levels = n_action_levels
        self.shared_trunk = shared_trunk
        if shared_trunk:
            # [P2-7] ABLATION: tai hien kien truc v1 - Actor va Critic DUNG CHUNG
            # than mang (2 lop an), chi khac lop dau ra. Dung de chung minh
            # bang thuc nghiem vi sao [V2-8] phai tach hai mang.
            trunk = _mlp(obs_per_pair, hidden_dim, 1, out_gain=1.0)[:-1]  # 2 lop an + Tanh
            self.actor = nn.Sequential(trunk, nn.Linear(hidden_dim, n_action_levels))
            self.critic = nn.Sequential(trunk, nn.Linear(hidden_dim, 1))
            nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
            nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
            for head in (self.actor[-1], self.critic[-1]):
                nn.init.constant_(head.bias, 0.0)
        else:
            self.actor  = _mlp(obs_per_pair, hidden_dim, n_action_levels, out_gain=0.01)
            self.critic = _mlp(obs_per_pair, hidden_dim, 1, out_gain=1.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.actor(obs), self.critic(obs).squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor,
                             action: Optional[torch.Tensor] = None,
                             deterministic: bool = False):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


class PPOAgent:
    """Tac tu IPPO voi chia se tham so."""

    def __init__(self, obs_per_pair: int, n_pairs: int, n_action_levels: int,
                 config: Optional[Dict[str, Any]] = None, device: str = "cpu"):
        self.obs_per_pair = obs_per_pair
        self.n_pairs = n_pairs
        self.n_action_levels = n_action_levels

        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        cfg = config or {}
        self.lr_actor        = float(cfg.get("lr_actor", 3.0e-4))
        self.lr_critic       = float(cfg.get("lr_critic", 1.0e-3))
        self.gamma           = float(cfg.get("gamma", 0.99))
        self.gae_lambda      = float(cfg.get("gae_lambda", 0.95))
        self.clip_eps        = float(cfg.get("clip_eps", 0.2))
        self.ent_coef_start  = float(cfg.get("ent_coef", 0.01))
        self.ent_coef_end    = float(cfg.get("ent_coef_end", self.ent_coef_start))
        self.ent_coef        = self.ent_coef_start
        self.vf_coef         = float(cfg.get("vf_coef", 0.5))
        self.ppo_epochs      = int(cfg.get("ppo_epochs", 4))
        self.max_grad_norm   = float(cfg.get("max_grad_norm", 0.5))
        self.target_kl       = cfg.get("target_kl", 0.02)
        self.mini_batch_size = int(cfg.get("mini_batch_size", 16384))
        self.hidden_dim      = int(cfg.get("hidden_dim", 128))
        self.use_value_norm  = bool(cfg.get("use_value_norm", True))
        self.clip_value_loss = bool(cfg.get("clip_value_loss", True))
        self.shared_trunk    = bool(cfg.get("shared_trunk", False))

        self.network = SharedActorCriticNetwork(
            obs_per_pair, n_action_levels, self.hidden_dim,
            shared_trunk=self.shared_trunk).to(self.device)

        if self.shared_trunk:
            # [P2-7] Nhu v1: MOT optimizer, MOT lan cat gradient cho toan mang.
            self.opt_actor  = optim.Adam(self.network.parameters(), lr=self.lr_actor, eps=1e-5)
            self.opt_critic = None
        else:
            # [V2-9] Hai optimizer rieng -> cat gradient doc lap
            self.opt_actor  = optim.Adam(self.network.actor.parameters(),  lr=self.lr_actor,  eps=1e-5)
            self.opt_critic = optim.Adam(self.network.critic.parameters(), lr=self.lr_critic, eps=1e-5)

        self.value_norm = ValueNormalizer() if self.use_value_norm else None

    # ------------------------------------------------------------------ #
    def set_lr_scale(self, scale: float):
        scale = float(np.clip(scale, 0.0, 1.0))
        for g in self.opt_actor.param_groups:
            g["lr"] = self.lr_actor * scale
        for g in (self.opt_critic.param_groups if self.opt_critic else []):
            g["lr"] = self.lr_critic * scale

    def set_ent_scale(self, progress: float):
        progress = float(np.clip(progress, 0.0, 1.0))
        self.ent_coef = (self.ent_coef_end
                         + (self.ent_coef_start - self.ent_coef_end) * progress)

    # ------------------------------------------------------------------ #
    def select_action(self, obs: np.ndarray, deterministic: bool = False):
        """Tra ve (action, log_prob, value) - value da KHU chuan hoa."""
        with torch.inference_mode():
            obs_t = torch.from_numpy(
                np.ascontiguousarray(obs, dtype=np.float32)).to(self.device)
            action, log_prob, _, value = self.network.get_action_and_value(
                obs_t, deterministic=deterministic)
            v = value.cpu().numpy()
            if self.value_norm is not None:
                v = self.value_norm.denormalize(v)
        return action.cpu().numpy(), log_prob.cpu().numpy(), v

    # ------------------------------------------------------------------ #
    def update(self, rollout_buffer: RolloutBuffer) -> Dict[str, float]:
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                   "approx_kl": 0.0, "clip_frac": 0.0,
                   "grad_norm_actor": 0.0, "grad_norm_critic": 0.0}
        n_updates = 0

        fc = rollout_buffer._flat_cache
        with torch.no_grad():
            v_pred, v_true = fc["values"], fc["returns"]
            var_true = v_true.var()
            explained_var = (float("nan") if var_true < 1e-8 else
                             (1.0 - (v_true - v_pred).var() / var_true).item())
        metrics["explained_variance"] = explained_var

        # [V2-10] Cap nhat thong ke chuan hoa gia tri TRUOC khi hoc
        if self.value_norm is not None:
            self.value_norm.update(fc["returns"])

        stop_early = False
        for epoch_idx in range(self.ppo_epochs):
            if stop_early:
                break
            epoch_kls = []
            for batch in rollout_buffer.get_batches(batch_size=self.mini_batch_size):
                _, new_log_prob, entropy, new_value = self.network.get_action_and_value(
                    batch["states"], action=batch["actions"])

                log_ratio = new_log_prob - batch["log_probs"]
                ratio = torch.exp(log_ratio)
                adv = batch["advantages"]

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean() - self.ent_coef * entropy.mean()

                # ---- Critic --------------------------------------------
                target = batch["returns"]
                old_v  = batch["values"]
                if self.value_norm is not None:
                    target = torch.as_tensor(self.value_norm.normalize(target))
                    old_v  = torch.as_tensor(self.value_norm.normalize(old_v))
                if self.clip_value_loss:
                    v_clipped = old_v + torch.clamp(new_value - old_v,
                                                    -self.clip_eps, self.clip_eps)
                    value_loss = torch.max(F.mse_loss(new_value, target, reduction="none"),
                                           F.mse_loss(v_clipped, target, reduction="none")).mean()
                else:
                    value_loss = F.mse_loss(new_value, target)
                value_loss = self.vf_coef * value_loss

                if self.shared_trunk:
                    # [P2-7] Mot buoc toi uu chung. Van do rieng chuan gradient
                    # cua tung loss (tren than chung) de thay su chenh lech.
                    self.opt_actor.zero_grad(set_to_none=True)
                    policy_loss.backward(retain_graph=True)
                    gn_a = float(torch.norm(torch.stack([
                        q.grad.norm() for q in self.network.parameters() if q.grad is not None])))
                    self.opt_actor.zero_grad(set_to_none=True)
                    value_loss.backward(retain_graph=True)
                    gn_c = float(torch.norm(torch.stack([
                        q.grad.norm() for q in self.network.parameters() if q.grad is not None])))
                    self.opt_actor.zero_grad(set_to_none=True)
                    (policy_loss + value_loss).backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                    self.opt_actor.step()
                else:
                    # ---- [V2-9] Hai buoc toi uu rieng ------------------
                    self.opt_actor.zero_grad(set_to_none=True)
                    policy_loss.backward()
                    gn_a = nn.utils.clip_grad_norm_(self.network.actor.parameters(),
                                                    self.max_grad_norm).item()
                    self.opt_actor.step()

                    self.opt_critic.zero_grad(set_to_none=True)
                    value_loss.backward()
                    gn_c = nn.utils.clip_grad_norm_(self.network.critic.parameters(),
                                                    self.max_grad_norm).item()
                    self.opt_critic.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"]  += value_loss.item()
                metrics["entropy"]     += entropy.mean().item()
                metrics["approx_kl"]   += approx_kl
                metrics["clip_frac"]   += clip_frac
                metrics["grad_norm_actor"]  += gn_a
                metrics["grad_norm_critic"] += gn_c
                n_updates += 1
                epoch_kls.append(approx_kl)

            if self.target_kl is not None and np.mean(epoch_kls) > 1.5 * float(self.target_kl):
                stop_early = True
                metrics["stopped_early_at_epoch"] = epoch_idx

        for k in list(metrics.keys()):
            if k in ("explained_variance", "stopped_early_at_epoch"):
                continue
            metrics[k] /= max(n_updates, 1)
        return metrics

    # ------------------------------------------------------------------ #
    def save(self, filepath: str):
        # [P0-8] Luu them trang thai 2 optimizer (Adam moments) de --resume
        # trong train.py la mot lan warm-start THAT SU, khong phai khoi tao
        # lai Adam tu dau. Khong anh huong checkpoint cu: cac noi chi doc
        # "network"/"value_norm" (evaluate.py, iso_service.py, ...) van chay
        # binh thuong vi khong dong cham toi 2 key moi nay.
        torch.save({"network": self.network.state_dict(),
                    "value_norm": (self.value_norm.state_dict()
                                   if self.value_norm is not None else None),
                    "opt_actor": self.opt_actor.state_dict(),
                    "opt_critic": (self.opt_critic.state_dict()
                                   if self.opt_critic else None)},
                   filepath)

    def load(self, filepath: str, load_optimizer: bool = False):
        blob = torch.load(filepath, map_location=self.device, weights_only=False)
        if isinstance(blob, dict) and "network" in blob:
            self.network.load_state_dict(blob["network"])
            if self.value_norm is not None and blob.get("value_norm"):
                self.value_norm.load_state_dict(blob["value_norm"])
            # [P0-8] load_optimizer=False la mac dinh (dung cho evaluate.py,
            # iso_service.py, analyze_scale_groups.py - chi can trong so).
            # Checkpoint cu (truoc [P0-8]) khong co 2 key nay -> .get() tra
            # None, bo qua an toan, chi mat dong luong Adam chu khong loi.
            if load_optimizer:
                if blob.get("opt_actor"):
                    self.opt_actor.load_state_dict(blob["opt_actor"])
                if blob.get("opt_critic") and self.opt_critic:
                    self.opt_critic.load_state_dict(blob["opt_critic"])
        else:                                     # tuong thich checkpoint cu
            self.network.load_state_dict(blob)
