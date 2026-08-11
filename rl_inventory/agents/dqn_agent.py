"""
agents/dqn_agent.py
====================
Double DQN Agent for Multi-Warehouse Inventory Management.

Architecture Overview:
-----------------------
  Input (obs_dim) -> Linear(256) -> ReLU -> Linear(256) -> ReLU
                 -> Linear(n_pairs × n_action_levels)
                 -> Reshape(n_pairs, n_action_levels)
                 -> argmax per pair -> actions (n_pairs,)

Key Algorithmic Components (thesis-citations):
-----------------------------------------------
1. Double DQN (van Hasselt et al., 2016 -- DDQN):
   Standard DQN overestimates action values because the same network is
   used for both action selection and value evaluation (maximization bias).
   Double DQN decouples these:
     - Online network  theta  : selects the best action (argmax)
     - Target network  theta⁻ : evaluates the value of that action
   Target: y = r + gamma * Q(s', argmax_a Q(s',a; theta); theta⁻)
   This is CITATION-WORTHY in thesis: van Hasselt, H., Guez, A., & Silver, D.
   (2016). Deep Reinforcement Learning with Double Q-learning. AAAI-2016.

2. Experience Replay (Mnih et al., 2015 -- DQN Nature):
   Random mini-batch sampling from replay buffer to break temporal correlations.

3. Target Network with Soft Update (Polyak averaging):
   theta⁻ ← tau*theta + (1-tau)*theta⁻   (tau = 0.005 default)
   Stabilizes training by providing a slowly-changing regression target.

4. eps-Greedy Exploration with Exponential Decay:
   eps_t = eps_min + (eps_start - eps_min) × exp(-t / eps_decay)

GPU Support:
   Automatically uses CUDA if available (RTX 3050 4GB VRAM compatible
   with batch_size=128, hidden_dim=256).
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Tuple

from agents.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


# ===========================================================================
# Q-Network
# ===========================================================================

class QNetwork(nn.Module):
    """
    Multi-layer perceptron Q-Network for factorized inventory control.

    For each warehouse-SKU pair, outputs Q-values for all n_action_levels
    discrete order quantities. The network shares parameters across all pairs
    (parameter sharing) to improve generalization and sample efficiency.

    Architecture:
        Linear(state_dim, hidden_dim) -> LayerNorm -> ReLU
        -> Linear(hidden_dim, hidden_dim) -> LayerNorm -> ReLU
        -> Linear(hidden_dim, n_pairs × n_action_levels)
        -> Reshape -> (n_pairs, n_action_levels)

    Parameters
    ----------
    state_dim : int
        Input observation dimension.
    n_pairs : int
        Number of warehouse-SKU pairs.
    n_action_levels : int
        Number of discrete order quantities per pair.
    hidden_dim : int
        Width of hidden layers. Default 256 (fits RTX 3050 4GB).
    """

    def __init__(
        self,
        state_dim: int,
        n_pairs: int,
        n_action_levels: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.n_pairs = n_pairs
        self.n_action_levels = n_action_levels
        out_dim = n_pairs * n_action_levels

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        # Xavier initialization for stable gradients
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        state : torch.Tensor, shape (batch, state_dim)

        Returns
        -------
        q_values : torch.Tensor, shape (batch, n_pairs, n_action_levels)
        """
        x = self.net(state)                                    # (B, n_pairs * n_action_levels)
        return x.view(-1, self.n_pairs, self.n_action_levels)  # (B, n_pairs, n_levels)


# ===========================================================================
# Double DQN Agent
# ===========================================================================

class DoubleDQNAgent:
    """
    Double DQN Agent for Multi-Warehouse Inventory Optimization.

    Implements:
    - Double DQN loss (van Hasselt et al., 2016) -- see module docstring
    - eps-greedy exploration with exponential decay
    - Soft target network update (Polyak averaging)
    - TensorBoard logging (reward, loss, epsilon per episode)

    Parameters
    ----------
    state_dim : int
        Observation vector dimension.
    n_pairs : int
        Number of warehouse-SKU pairs (= n_warehouses × n_skus).
    n_action_levels : int
        Discrete action levels per pair. Default 6.
    hidden_dim : int
        Hidden layer size. Default 256.
    lr : float
        Adam learning rate. Default 3e-4.
    gamma : float
        Discount factor. Default 0.99.
    tau : float
        Soft update coefficient. Default 5e-3.
    buffer_capacity : int
        Replay buffer size. Default 100,000.
    batch_size : int
        Training batch size. Default 128 (RTX 3050 optimized).
    eps_start : float
        Initial exploration rate. Default 1.0.
    eps_min : float
        Minimum exploration rate. Default 0.05.
    eps_decay : int
        Decay rate (steps). Default 50,000.
    target_update_freq : int
        Hard target update frequency (steps). Used if tau=None. Default 1000.
    use_per : bool
        Use Prioritized Experience Replay. Default False.
    log_dir : str
        TensorBoard log directory. Default 'runs/'.
    device : str
        'cuda' or 'cpu'. Auto-detected if 'auto'.

    Example
    -------
    >>> agent = DoubleDQNAgent(state_dim=267, n_pairs=60)
    >>> action = agent.select_action(obs)
    >>> agent.store_transition(obs, action, reward, next_obs, done)
    >>> loss = agent.update()
    """

    def __init__(
        self,
        state_dim: int,
        n_pairs: int,
        n_action_levels: int = 6,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 5e-3,
        buffer_capacity: int = 100_000,
        batch_size: int = 128,
        eps_start: float = 1.0,
        eps_min: float = 0.05,
        eps_decay: int = 50_000,
        use_per: bool = False,
        log_dir: str = "runs/",
        device: str = "auto",
    ):
        # Device setup -- RTX 3050 auto-detected
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"[DoubleDQNAgent] Device: {self.device}")

        self.state_dim       = state_dim
        self.n_pairs         = n_pairs
        self.n_action_levels = n_action_levels
        self.gamma           = gamma
        self.tau             = tau
        self.batch_size      = batch_size
        self.eps_start       = eps_start
        self.eps_min         = eps_min
        self.eps_decay       = eps_decay
        self.use_per         = use_per

        # ---- Networks -------------------------------------------------------
        self.online_net = QNetwork(
            state_dim, n_pairs, n_action_levels, hidden_dim
        ).to(self.device)

        self.target_net = QNetwork(
            state_dim, n_pairs, n_action_levels, hidden_dim
        ).to(self.device)

        # Initialize target net with same weights as online net
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # Target net is never trained directly

        # ---- Optimizer & LR Scheduler ----------------------------------------
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20_000, gamma=0.5
        )

        # ---- Replay Buffer ---------------------------------------------------
        if use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=buffer_capacity,
                state_dim=state_dim,
                action_dim=n_pairs,
                device=str(self.device),
            )
        else:
            self.replay_buffer = ReplayBuffer(
                capacity=buffer_capacity,
                state_dim=state_dim,
                action_dim=n_pairs,
                device=str(self.device),
            )

        # ---- Counters & Logging ----------------------------------------------
        self.global_step     = 0
        self.episode_count   = 0
        self.total_loss      = 0.0
        self.update_count    = 0

        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    # ------------------------------------------------------------------
    # Exploration rate
    # ------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        """
        Current epsilon (exploration rate), exponentially decayed.

        eps_t = eps_min + (eps_start - eps_min) × exp(-t / decay)
        """
        return self.eps_min + (self.eps_start - self.eps_min) * np.exp(
            -self.global_step / self.eps_decay
        )

    # ------------------------------------------------------------------
    # Action Selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        state: np.ndarray,
        greedy: bool = False,
    ) -> np.ndarray:
        """
        eps-greedy action selection.

        For each of the n_pairs warehouse-SKU pairs, independently selects
        an action index in [0, n_action_levels - 1].

        Parameters
        ----------
        state : np.ndarray, shape (state_dim,)
        greedy : bool
            If True, always select greedy action (used during evaluation).

        Returns
        -------
        actions : np.ndarray, shape (n_pairs,), dtype int
        """
        eps = 0.0 if greedy else self.epsilon

        if np.random.rand() < eps:
            # Explore: random action per pair
            return np.random.randint(0, self.n_action_levels, size=self.n_pairs)
        else:
            # Exploit: argmax Q per pair
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q = self.online_net(s)          # (1, n_pairs, n_levels)
                actions = q.argmax(dim=-1)      # (1, n_pairs)
                return actions.squeeze(0).cpu().numpy().astype(np.int32)

    # ------------------------------------------------------------------
    # Store Transition
    # ------------------------------------------------------------------

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Push one transition into the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.global_step += 1

    # ------------------------------------------------------------------
    # Learning Step (Double DQN Loss)
    # ------------------------------------------------------------------

    def update(self) -> Optional[float]:
        """
        Sample a mini-batch and perform one gradient step.

        Double DQN Loss (van Hasselt et al., 2016) -- CITATION:
        --------------------------------------------------------
        Standard DQN:  y = r + gamma * max_a Q(s', a; theta⁻)
        Double DQN:    y = r + gamma * Q(s', argmax_a Q(s', a; theta); theta⁻)

        By using the online net theta for action selection and the target net theta⁻
        for value evaluation, Double DQN avoids the maximization bias that
        causes DQN to overestimate action values.

        Returns
        -------
        loss : float or None (None if buffer not yet ready)
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # --- Sample from buffer -------------------------------------------
        if self.use_per:
            beta = min(1.0, 0.4 + self.global_step * (1.0 - 0.4) / 200_000)
            (states, actions, rewards, next_states, dones), is_weights, indices = \
                self.replay_buffer.sample(self.batch_size, beta=beta)
            is_weights = torch.FloatTensor(is_weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.batch_size)
            is_weights = None
            indices = None

        # Convert to tensors
        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)   # (B, n_pairs)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)  # (B, 1)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)    # (B, 1)

        # --- Compute current Q values -------------------------------------
        q_online = self.online_net(states_t)       # (B, n_pairs, n_levels)

        # Gather Q-values for the actions taken
        # actions_t: (B, n_pairs) -> unsqueeze -> (B, n_pairs, 1)
        current_q = q_online.gather(
            dim=2,
            index=actions_t.unsqueeze(2)
        ).squeeze(2)                               # (B, n_pairs)

        # --- Compute Double DQN target ------------------------------------
        with torch.no_grad():
            # ACTION SELECTION: use online net
            next_q_online = self.online_net(next_states_t)        # (B, n_pairs, n_levels)
            best_actions  = next_q_online.argmax(dim=2, keepdim=True)  # (B, n_pairs, 1)

            # VALUE EVALUATION: use target net
            next_q_target = self.target_net(next_states_t)        # (B, n_pairs, n_levels)
            next_q_values = next_q_target.gather(
                dim=2, index=best_actions
            ).squeeze(2)                                          # (B, n_pairs)

            # y = r + gamma * Q_target(s', a*)  ×  (1 - done)
            # rewards_t: (B, 1) broadcast to (B, n_pairs)
            target_q = rewards_t + self.gamma * next_q_values * (1.0 - dones_t)

        # --- Compute loss -------------------------------------------------
        td_errors = current_q - target_q                           # (B, n_pairs)

        if is_weights is not None:
            # PER: weight loss by importance-sampling weights
            loss = (is_weights.unsqueeze(1) * td_errors.pow(2)).mean()
            # Update priorities in PER buffer
            per_errors = td_errors.detach().abs().mean(dim=1).cpu().numpy()
            self.replay_buffer.update_priorities(indices, per_errors)
        else:
            loss = F.mse_loss(current_q, target_q)

        # --- Backpropagation ----------------------------------------------
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for training stability
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.scheduler.step()

        # --- Soft update target network (Polyak averaging) ----------------
        # theta⁻ ← tau*theta + (1-tau)*theta⁻
        self._soft_update_target()

        loss_val = loss.item()
        self.total_loss += loss_val
        self.update_count += 1

        # TensorBoard logging (per step)
        if self.update_count % 100 == 0:
            self.writer.add_scalar("train/loss", loss_val, self.global_step)
            self.writer.add_scalar("train/epsilon", self.epsilon, self.global_step)
            self.writer.add_scalar(
                "train/avg_loss",
                self.total_loss / self.update_count,
                self.global_step,
            )

        return loss_val

    # ------------------------------------------------------------------
    # Target Network Update
    # ------------------------------------------------------------------

    def _soft_update_target(self) -> None:
        """
        Soft (Polyak) update: theta⁻ ← tau*theta + (1-tau)*theta⁻

        This gradually blends online weights into target weights,
        providing a smoother and more stable regression target than
        hard periodic updates.
        """
        for param, target_param in zip(
            self.online_net.parameters(),
            self.target_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def hard_update_target(self) -> None:
        """Full copy of online weights to target (alternative to soft update)."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ------------------------------------------------------------------
    # Episode Logging
    # ------------------------------------------------------------------

    def log_episode(
        self,
        episode_reward: float,
        episode_cost: float,
        service_level: float,
        stockout_rate: float,
    ) -> None:
        """
        Log per-episode metrics to TensorBoard.

        Parameters
        ----------
        episode_reward : float -- cumulative reward for the episode
        episode_cost   : float -- total inventory cost
        service_level  : float -- fraction of demand fulfilled
        stockout_rate  : float -- fraction of demand lost
        """
        ep = self.episode_count
        self.writer.add_scalar("episode/reward",        episode_reward, ep)
        self.writer.add_scalar("episode/cost",          episode_cost,   ep)
        self.writer.add_scalar("episode/service_level", service_level,  ep)
        self.writer.add_scalar("episode/stockout_rate", stockout_rate,  ep)
        self.writer.add_scalar("episode/epsilon",       self.epsilon,   ep)
        self.episode_count += 1

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save agent weights and training state to a checkpoint file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(
            {
                "online_net":    self.online_net.state_dict(),
                "target_net":    self.target_net.state_dict(),
                "optimizer":     self.optimizer.state_dict(),
                "global_step":   self.global_step,
                "episode_count": self.episode_count,
            },
            path,
        )
        print(f"[DoubleDQNAgent] Saved checkpoint -> {path}")

    def load(self, path: str) -> None:
        """Load agent from a checkpoint file."""
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step   = ckpt.get("global_step", 0)
        self.episode_count = ckpt.get("episode_count", 0)
        print(f"[DoubleDQNAgent] Loaded checkpoint ← {path}")

    def close(self) -> None:
        """Close TensorBoard writer."""
        self.writer.close()


# ===========================================================================
# Quick test
# ===========================================================================
if __name__ == "__main__":
    print("Testing DoubleDQNAgent (CPU mode)...")
    state_dim = 267
    n_pairs = 60

    agent = DoubleDQNAgent(
        state_dim=state_dim,
        n_pairs=n_pairs,
        n_action_levels=6,
        hidden_dim=128,
        batch_size=32,
        log_dir="runs/test",
        device="cpu",
    )

    # Fill buffer with dummy transitions
    for _ in range(100):
        s  = np.random.randn(state_dim).astype(np.float32)
        a  = np.random.randint(0, 6, size=n_pairs).astype(np.int32)
        r  = float(np.random.randn())
        ns = np.random.randn(state_dim).astype(np.float32)
        d  = False
        agent.store_transition(s, a, r, ns, d)

    # Select action
    obs = np.random.randn(state_dim).astype(np.float32)
    action = agent.select_action(obs)
    print(f"  Action shape: {action.shape}, sample: {action[:5]}")
    print(f"  Epsilon: {agent.epsilon:.4f}")

    # Update step
    loss = agent.update()
    print(f"  Loss: {loss:.6f}")

    agent.save("runs/test/test_ckpt.pth")
    agent.close()
    print("DoubleDQNAgent OK!")
