"""
agents/replay_buffer.py
========================
Bộ đệm phát lại kinh nghiệm (Experience Replay Buffer) cho huấn luyện DQN.

Ghi chú triển khai (cho khóa luận):
------------------------------------
Kỹ thuật phát lại kinh nghiệm (Lin, 1992) giúp phá vỡ sự tương quan theo thời gian
trong dữ liệu huấn luyện bằng cách lưu trữ các bước chuyển trạng thái (s, a, r, s', done)
vào một bộ đệm vòng và lấy mẫu các mini-batch ngẫu nhiên. Đây là thành phần quan trọng
để huấn luyện DQN ổn định (Mnih et al., 2015 — bài báo DQN Nature).

Cung cấp hai phiên bản triển khai:
  1. ReplayBuffer            — Lấy mẫu ngẫu nhiên đồng đều tiêu chuẩn (mặc định)
  2. PrioritizedReplayBuffer — Bộ đệm phát lại có ưu tiên (PER)
     Schaul et al., 2016 — "Prioritized Experience Replay"
     Lấy mẫu các bước chuyển trạng thái tỷ lệ với |TD-error|^α, được điều chỉnh bởi
     trọng số lấy mẫu tầm quan trọng (importance-sampling) β (tăng dần từ β₀ → 1).
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Tuple, Optional


# ===========================================================================
# Bộ đệm phát lại tiêu chuẩn (Standard Replay Buffer)
# ===========================================================================

class ReplayBuffer:
    """
    Bộ đệm phát lại kinh nghiệm dạng vòng sử dụng mảng numpy được cấp phát trước.

    Lưu trữ các bước chuyển trạng thái (state, action, reward, next_state, done)
    và hỗ trợ lấy mẫu lô ngẫu nhiên đồng đều cho quá trình huấn luyện DQN.

    Tham số
    ------
    capacity : int
        Số lượng tối đa các chuyển trạng thái lưu trữ (mẫu cũ hơn sẽ bị ghi đè).
    state_dim : int
        Kích thước của vectơ trạng thái/quan sát.
    action_dim : int
        Số lượng cặp kho - SKU (mỗi cặp có một chỉ số hành động số nguyên).
    device : str
        'cuda' hoặc 'cpu' — các tensor trả về sẽ thuộc thiết bị này.

    Ví dụ
    -----
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

        # Cấp phát trước các mảng numpy để tối ưu tốc độ (tránh chi phí danh sách Python)
        self.states      = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.actions     = np.zeros((capacity, action_dim), dtype=np.int32)
        self.rewards     = np.zeros((capacity, 1),          dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.dones       = np.zeros((capacity, 1),          dtype=np.float32)

        self._ptr = 0       # con trỏ ghi (dạng vòng)
        self._size = 0      # số lượng phần tử hợp lệ hiện tại

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
        Lưu một bước chuyển trạng thái vào bộ đệm.

        Tham số
        ------
        state : np.ndarray, shape (state_dim,)
        action : np.ndarray, shape (action_dim,) kiểu int
        reward : float
        next_state : np.ndarray, shape (state_dim,)
        done : bool — True nếu tập huấn luyện kết thúc
        """
        self.states[self._ptr]      = state
        self.actions[self._ptr]     = action
        self.rewards[self._ptr]     = reward
        self.next_states[self._ptr] = next_state
        self.dones[self._ptr]       = float(done)

        # Ghi đè dạng vòng
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Lấy mẫu một mini-batch ngẫu nhiên của các bước chuyển trạng thái.

        Tham số
        ------
        batch_size : int

        Trả về
        -----
        Tuple của (states, actions, rewards, next_states, dones)
        Tất cả dưới dạng np.ndarray — bên gọi có trách nhiệm chuyển sang Tensors.
        """
        assert self._size >= batch_size, \
            f"Chưa đủ mẫu: {self._size} < {batch_size}"

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
        """True nếu có đủ từ 64 mẫu trở lên để rút một batch."""
        return self._size >= 64


# ===========================================================================
# Bộ đệm phát lại có ưu tiên (PER - Prioritized Experience Replay)
# Trích dẫn: Schaul et al., 2016 — "Prioritized Experience Replay"
# ===========================================================================

class PrioritizedReplayBuffer:
    """
    Bộ đệm phát lại kinh nghiệm có ưu tiên (PER).

    Các bước chuyển trạng thái được lấy mẫu với xác suất tỷ lệ với |TD-error|^α,
    giúp tác tử học nhiều hơn từ các chuyển trạng thái có độ lỗi cao (bất ngờ).
    Trọng số lấy mẫu tầm quan trọng (IS) điều chỉnh độ lệch được đưa vào.

    Hyperparameters Chính
    --------------------
    alpha : float (0 = đồng đều, 1 = ưu tiên hoàn toàn)
        Điều khiển mức độ ưu tiên được sử dụng. Mặc định: 0.6.
    beta_start : float
        Mũ số IS ban đầu (0 = không hiệu chỉnh, 1 = hiệu chỉnh hoàn toàn).
        Tăng tuyến tính từ beta_start → 1.0 trong quá trình huấn luyện. Mặc định: 0.4.
    eps : float
        Hằng số nhỏ thêm vào độ ưu tiên để đảm bảo xác suất không bằng 0.

    Trích dẫn khoa học (cho khóa luận)
    ----------------------------------
    Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016).
    Prioritized Experience Replay. ICLR 2016.

    Ví dụ
    -----
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

        # Lưu trữ
        self.states      = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.actions     = np.zeros((capacity, action_dim), dtype=np.int32)
        self.rewards     = np.zeros((capacity, 1),          dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.dones       = np.zeros((capacity, 1),          dtype=np.float32)

        # Lưu trữ độ ưu tiên — sử dụng max-priority cho chuyển trạng thái mới
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
        """Lưu bước chuyển trạng thái với độ ưu tiên tối đa hiện tại."""
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
        Lấy mẫu một lô được ưu tiên.

        Trả về
        -----
        batch : tuple của (states, actions, rewards, next_states, dones)
        weights : np.ndarray, shape (batch_size,) — trọng số hiệu chỉnh IS
        indices : np.ndarray, shape (batch_size,) — dùng để cập nhật độ ưu tiên
        """
        priorities = self.priorities[:self._size]
        probs = (priorities ** self.alpha)
        probs /= probs.sum()

        indices = np.random.choice(self._size, size=batch_size, replace=False, p=probs)

        # Trọng số lấy mẫu tầm quan trọng (Importance-sampling weights)
        weights = (self._size * probs[indices]) ** (-beta)
        weights /= weights.max()  # chuẩn hóa

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
        Cập nhật độ ưu tiên dựa trên độ lỗi TD (TD error) mới.

        Tham số
        ------
        indices : np.ndarray — các chỉ số được trả về bởi sample()
        td_errors : np.ndarray — độ lỗi TD tuyệt đối từ bước cập nhật học tập
        """
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = float(abs(err)) + self.eps

    def __len__(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# Kiểm tra nhanh
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Đang thử nghiệm ReplayBuffer...")
    buf = ReplayBuffer(capacity=1000, state_dim=10, action_dim=4)
    for _ in range(200):
        s  = np.random.randn(10).astype(np.float32)
        a  = np.random.randint(0, 6, size=(4,)).astype(np.int32)
        r  = float(np.random.randn())
        ns = np.random.randn(10).astype(np.float32)
        d  = False
        buf.push(s, a, r, ns, d)

    states, actions, rewards, next_states, dones = buf.sample(32)
    print(f"  kích thước states:      {states.shape}")
    print(f"  kích thước actions:     {actions.shape}")
    print(f"  kích thước rewards:     {rewards.shape}")
    print("ReplayBuffer THÀNH CÔNG!\n")

    print("Đang thử nghiệm PrioritizedReplayBuffer...")
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
    print(f"  kích thước batch_s: {batch_s.shape}")
    print(f"  khoảng weights: [{weights.min():.4f}, {weights.max():.4f}]")
    print("PrioritizedReplayBuffer THÀNH CÔNG!")
