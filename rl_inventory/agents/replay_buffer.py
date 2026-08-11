"""
agents/replay_buffer.py
========================
Experience Replay Buffer for DQN training.

Implementation Notes (for thesis):
------------------------------------
Experience Replay (Lin, 1992) breaks temporal correlations in the training
data by storing transitions (s, a, r, s', done) in a circular buffer and
sampling random mini-batches. This is a critical component for stable DQN
training (Mnih et al., 2015 — DQN Nature paper).

Two implementations provided:
  1. ReplayBuffer       — Standard uniform random replay (default)
  2. PrioritizedReplayBuffer — Prioritized Experience Replay (PER)
     Schaul et al., 2016 — "Prioritized Experience Replay"
     Samples transitions proportional to |TD-error|^α, corrected by
     importance-sampling weights β (annealed from β₀ → 1).
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Tuple, Optional


# ===========================================================================
# Standard Replay Buffer
# ===========================================================================

class ReplayBuffer:
    """
    Circular experience replay buffer using pre-allocated numpy arrays.

    Stores transitions (state, action, reward, next_state, done) and
    supports uniform random batch sampling for DQN training.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store (older ones are overwritten).
    state_dim : int
        Dimensionality of the observation/state vector.
    action_dim : int
        Number of warehouse-SKU pairs (each has an integer action).
    device : str
        'cuda' or 'cpu' — returned tensors will be on this device.

    Example
    -------
    >>> buf = ReplayBuffer(capacity=10000, state_dim=100, action_dim=60)
    >>> buf.push(state, action, reward, next_state, done)
    >>> batch = buf.sample(batch_size=64)
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Pre-allocate numpy arrays for speed (avoids Python list overhead)
        self.states      = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.actions     = np.zeros((capacity, action_dim), dtype=np.int32)
        self.rewards     = np.zeros((capacity, 1),          dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.dones       = np.zeros((capacity, 1),          dtype=np.float32)

        self._ptr = 0       # write pointer (circular)
        self._size = 0      # current number of valid entries

    # ------------------------------------------------------------------

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store a single transition in the buffer.

        Parameters
        ----------
        state : np.ndarray, shape (state_dim,)
        action : np.ndarray, shape (action_dim,) of int
        reward : float
        next_state : np.ndarray, shape (state_dim,)
        done : bool — True if episode ended
        """
        self.states[self._ptr]      = state
        self.actions[self._ptr]     = action
        self.rewards[self._ptr]     = reward
        self.next_states[self._ptr] = next_state
        self.dones[self._ptr]       = float(done)

        # Circular overwrite
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample a random mini-batch of transitions.

        Parameters
        ----------
        batch_size : int

        Returns
        -------
        Tuple of (states, actions, rewards, next_states, dones)
        All as np.ndarray — caller is responsible for converting to tensors.
        """
        assert self._size >= batch_size, \
            f"Not enough samples: {self._size} < {batch_size}"

        indices = np.random.randint(0, self._size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        return self._size

    @property
    def is_ready(self) -> bool:
        """True if enough samples are available to draw a batch of 64."""
        return self._size >= 64


# ===========================================================================
# Prioritized Experience Replay (PER)
# Reference: Schaul et al., 2016 — "Prioritized Experience Replay"
#            https://arxiv.org/abs/1511.05952
# ===========================================================================

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) Buffer.

    Transitions are sampled with probability proportional to |TD-error|^α,
    enabling the agent to learn more from surprising (high-error) transitions.
    Importance-sampling (IS) weights correct for the introduced bias.

    Key Hyperparameters
    -------------------
    alpha : float (0 = uniform, 1 = full prioritization)
        Controls how much prioritization is used. Default: 0.6.
    beta_start : float
        Initial IS exponent (0 = no correction, 1 = full correction).
        Annealed linearly from beta_start → 1.0 over training. Default: 0.4.
    eps : float
        Small constant added to priorities to ensure non-zero sampling.

    Citation (thesis-worthy)
    ------------------------
    Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016).
    Prioritized Experience Replay. ICLR 2016.

    Example
    -------
    >>> buf = PrioritizedReplayBuffer(capacity=50000, state_dim=100, action_dim=60)
    >>> buf.push(state, action, reward, next_state, done)
    >>> batch, weights, indices = buf.sample(64, beta=0.4)
    >>> buf.update_priorities(indices, td_errors)
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        eps: float = 1e-6,
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.eps = eps
        self.device = device

        # Storage
        self.states      = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.actions     = np.zeros((capacity, action_dim), dtype=np.int32)
        self.rewards     = np.zeros((capacity, 1),          dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.dones       = np.zeros((capacity, 1),          dtype=np.float32)

        # Priority storage — use max-priority for new transitions
        self.priorities = np.zeros(capacity, dtype=np.float64)

        self._ptr = 0
        self._size = 0

    # ------------------------------------------------------------------

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition with max current priority."""
        max_priority = self.priorities[:self._size].max() if self._size > 0 else 1.0

        self.states[self._ptr]      = state
        self.actions[self._ptr]     = action
        self.rewards[self._ptr]     = reward
        self.next_states[self._ptr] = next_state
        self.dones[self._ptr]       = float(done)
        self.priorities[self._ptr]  = max_priority

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # ------------------------------------------------------------------

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> Tuple[Tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
        """
        Sample a prioritized batch.

        Returns
        -------
        batch : tuple of (states, actions, rewards, next_states, dones)
        weights : np.ndarray, shape (batch_size,) — IS correction weights
        indices : np.ndarray, shape (batch_size,) — for priority updates
        """
        priorities = self.priorities[:self._size]
        probs = (priorities ** self.alpha)
        probs /= probs.sum()

        indices = np.random.choice(self._size, size=batch_size, replace=False, p=probs)

        # Importance-sampling weights
        weights = (self._size * probs[indices]) ** (-beta)
        weights /= weights.max()  # normalize

        batch = (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )
        return batch, weights.astype(np.float32), indices

    # ------------------------------------------------------------------

    def update_priorities(
        self, indices: np.ndarray, td_errors: np.ndarray
    ) -> None:
        """
        Update priorities based on new TD errors.

        Parameters
        ----------
        indices : np.ndarray — indices returned by sample()
        td_errors : np.ndarray — absolute TD errors from the learning step
        """
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = float(abs(err)) + self.eps

    def __len__(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing ReplayBuffer...")
    buf = ReplayBuffer(capacity=1000, state_dim=10, action_dim=4)
    for _ in range(200):
        s  = np.random.randn(10).astype(np.float32)
        a  = np.random.randint(0, 6, size=(4,)).astype(np.int32)
        r  = float(np.random.randn())
        ns = np.random.randn(10).astype(np.float32)
        d  = False
        buf.push(s, a, r, ns, d)

    states, actions, rewards, next_states, dones = buf.sample(32)
    print(f"  states shape:      {states.shape}")
    print(f"  actions shape:     {actions.shape}")
    print(f"  rewards shape:     {rewards.shape}")
    print("ReplayBuffer OK!\n")

    print("Testing PrioritizedReplayBuffer...")
    per = PrioritizedReplayBuffer(capacity=1000, state_dim=10, action_dim=4)
    for _ in range(200):
        s  = np.random.randn(10).astype(np.float32)
        a  = np.random.randint(0, 6, size=(4,)).astype(np.int32)
        r  = float(np.random.randn())
        ns = np.random.randn(10).astype(np.float32)
        d  = False
        per.push(s, a, r, ns, d)

    (batch_s, batch_a, batch_r, batch_ns, batch_d), weights, indices = per.sample(32, beta=0.4)
    per.update_priorities(indices, np.abs(np.random.randn(32)))
    print(f"  batch_s shape: {batch_s.shape}")
    print(f"  weights range: [{weights.min():.4f}, {weights.max():.4f}]")
    print("PrioritizedReplayBuffer OK!")
