"""
agents/dqn_agent.py
====================
Tác tử Double DQN cho bài toán Quản lý Tồn kho Đa Kho.

Tổng quan kiến trúc:
--------------------
  Đầu vào (obs_dim) -> Linear(256) -> ReLU -> Linear(256) -> ReLU
                    -> Linear(n_pairs × n_action_levels)
                    -> Reshape(n_pairs, n_action_levels)
                    -> argmax trên từng cặp -> hành động (n_pairs,)

Các thành phần thuật toán chính (Trích dẫn cho khóa luận):
----------------------------------------------------------
1. Double DQN (van Hasselt et al., 2016 -- DDQN):
   DQN tiêu chuẩn có xu hướng đánh giá quá cao giá trị hành động (overestimation bias)
   do cùng một mạng neural vừa dùng để chọn hành động vừa dùng để đánh giá giá trị.
   Double DQN tách biệt hai nhiệm vụ này:
     - Mạng trực tuyến (Online network theta): chọn hành động tốt nhất (argmax)
     - Mạng mục tiêu (Target network theta⁻): đánh giá giá trị của hành động đó
   Mục tiêu: y = r + gamma * Q(s', argmax_a Q(s',a; theta); theta⁻)
   Trích dẫn khoa học: van Hasselt, H., Guez, A., & Silver, D. (2016).
   Deep Reinforcement Learning with Double Q-learning. AAAI-2016.

2. Experience Replay (Mnih et al., 2015 -- DQN Nature):
   Lấy mẫu mini-batch ngẫu nhiên từ bộ đệm phát lại để phá vỡ sự tương quan theo thời gian.

3. Target Network với Soft Update (Trung bình Polyak):
   theta⁻ ← tau*theta + (1-tau)*theta⁻   (mặc định tau = 0.005)
   Ổn định quá trình huấn luyện bằng cách cung cấp mục tiêu hồi quy thay đổi chậm.

4. Khám phá eps-Greedy giảm theo hàm mũ:
   eps_t = eps_min + (eps_start - eps_min) × exp(-t / eps_decay)

Hỗ trợ GPU:
   Tự động sử dụng CUDA nếu sẵn có (Tương thích với RTX 3050 4GB VRAM
   với batch_size=128, hidden_dim=256).
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
    Mạng Perceptron Đa Lớp (MLP) Q-Network cho điều khiển tồn kho phân rã.

    Với mỗi cặp kho - SKU, xuất ra các giá trị Q cho tất cả n_action_levels
    mức số lượng đặt hàng rời rạc. Mạng chia sẻ tham số trên tất cả các cặp
    (parameter sharing) để tăng khả năng tổng quát hóa và hiệu quả lấy mẫu.

    Kiến trúc:
        Linear(state_dim, hidden_dim) -> LayerNorm -> ReLU
        -> Linear(hidden_dim, hidden_dim) -> LayerNorm -> ReLU
        -> Linear(hidden_dim, n_pairs × n_action_levels)
        -> Reshape -> (n_pairs, n_action_levels)

    Tham số
    ------
    state_dim : int
        Kích thước vectơ quan sát đầu vào.
    n_pairs : int
        Số lượng cặp kho - SKU.
    n_action_levels : int
        Số mức hành động rời rạc cho mỗi cặp.
    hidden_dim : int
        Độ rộng của các lớp ẩn. Mặc định 256 (phù hợp với RTX 3050 4GB).
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

        # Khởi tạo trọng số Xavier cho đạo hàm ổn định
        self._init_weights()

    def _init_weights(self):
        """Khởi tạo Xavier uniform cho tất cả các lớp tuyến tính."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Lan truyền tiến (Forward pass).

        Tham số
        ------
        state : torch.Tensor, shape (batch, state_dim)

        Trả về
        -----
        q_values : torch.Tensor, shape (batch, n_pairs, n_action_levels)
        """
        x = self.net(state)                                    # (B, n_pairs * n_action_levels)
        return x.view(-1, self.n_pairs, self.n_action_levels)  # (B, n_pairs, n_levels)


# ===========================================================================
# Double DQN Agent
# ===========================================================================

class DoubleDQNAgent:
    """
    Tác tử Double DQN cho Tối ưu hóa Quản lý Tồn kho Đa Kho.

    Triển khai:
    - Hàm tổn thất Double DQN (van Hasselt et al., 2016)
    - Khám phá eps-greedy với giảm hàm mũ
    - Cập nhật mềm mạng mục tiêu (Trung bình Polyak)
    - Ghi log TensorBoard (reward, loss, epsilon theo từng tập)

    Tham số
    ------
    state_dim : int
        Chiều của vectơ quan sát.
    n_pairs : int
        Số lượng cặp kho - SKU (= n_warehouses × n_skus).
    n_action_levels : int
        Các mức hành động rời rạc cho mỗi cặp. Mặc định 6.
    hidden_dim : int
        Kích thước lớp ẩn. Mặc định 256.
    lr : float
        Tốc độ học Adam. Mặc định 3e-4.
    gamma : float
        Hệ số chiết khấu. Mặc định 0.99.
    tau : float
        Hệ số cập nhật mềm. Mặc định 5e-3.
    buffer_capacity : int
        Dung lượng bộ đệm phát lại. Mặc định 100,000.
    batch_size : int
        Kích thước lô huấn luyện. Mặc định 128 (tối ưu cho RTX 3050).
    eps_start : float
        Tỷ lệ khám phá ban đầu. Mặc định 1.0.
    eps_min : float
        Tỷ lệ khám phá tối thiểu. Mặc định 0.05.
    eps_decay : int
        Tốc độ suy giảm (theo bước). Mặc định 50,000.
    use_per : bool
        Có sử dụng Prioritized Experience Replay hay không. Mặc định False.
    log_dir : str
        Thư mục ghi log TensorBoard. Mặc định 'runs/'.
    device : str
        'cuda' hoặc 'cpu'. Tự động phát hiện nếu là 'auto'.

    Ví dụ
    -----
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
        # Thiết lập thiết bị tính toán -- Tự động phát hiện GPU RTX 3050
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"[DoubleDQNAgent] Thiết bị tính toán: {self.device}")

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

        # ---- Các Mạng Neural ------------------------------------------------
        self.online_net = QNetwork(
            state_dim, n_pairs, n_action_levels, hidden_dim
        ).to(self.device)

        self.target_net = QNetwork(
            state_dim, n_pairs, n_action_levels, hidden_dim
        ).to(self.device)

        # Khởi tạo mạng mục tiêu với cùng trọng số như mạng trực tuyến
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # Mạng mục tiêu không được huấn luyện trực tiếp

        # ---- Optimizer & LR Scheduler ----------------------------------------
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20_000, gamma=0.5
        )

        # ---- Bộ đệm phát lại (Replay Buffer) --------------------------------
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

        # ---- Đếm & Ghi log ---------------------------------------------------
        self.global_step     = 0
        self.episode_count   = 0
        self.total_loss      = 0.0
        self.update_count    = 0

        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    # ------------------------------------------------------------------
    # Tỷ lệ khám phá
    # ------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        """
        Tỷ lệ epsilon khám phá hiện tại, giảm theo hàm mũ.

        eps_t = eps_min + (eps_start - eps_min) × exp(-t / decay)
        """
        return self.eps_min + (self.eps_start - self.eps_min) * np.exp(
            -self.global_step / self.eps_decay
        )

    # ------------------------------------------------------------------
    # Lựa chọn hành động
    # ------------------------------------------------------------------

    def select_action(
        self,
        state: np.ndarray,
        greedy: bool = False,
    ) -> np.ndarray:
        """
        Lựa chọn hành động theo chiến lược eps-greedy.

        Đối với mỗi cặp kho - SKU trong n_pairs cặp, chọn độc lập
        một chỉ số hành động trong [0, n_action_levels - 1].

        Tham số
        ------
        state : np.ndarray, shape (state_dim,)
        greedy : bool
            Nếu True, luôn chọn hành động tham lam (dùng khi đánh giá).

        Trả về
        -----
        actions : np.ndarray, shape (n_pairs,), dtype int
        """
        eps = 0.0 if greedy else self.epsilon

        if np.random.rand() < eps:
            # Khám phá: chọn hành động ngẫu nhiên cho từng cặp
            return np.random.randint(0, self.n_action_levels, size=self.n_pairs)
        else:
            # Khai thác: chọn argmax Q cho từng cặp
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q = self.online_net(s)          # (1, n_pairs, n_levels)
                actions = q.argmax(dim=-1)      # (1, n_pairs)
                return actions.squeeze(0).cpu().numpy().astype(np.int32)

    # ------------------------------------------------------------------
    # Lưu chuyển trạng thái
    # ------------------------------------------------------------------

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Thêm một bước chuyển trạng thái vào bộ đệm phát lại."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.global_step += 1

    # ------------------------------------------------------------------
    # Bước cập nhật trọng số (Double DQN Loss)
    # ------------------------------------------------------------------

    def update(self) -> Optional[float]:
        """
        Lấy mẫu một mini-batch và thực hiện một bước tối ưu lan truyền ngược.

        Hàm tổn thất Double DQN (van Hasselt et al., 2016) -- TRÍCH DẪN:
        -----------------------------------------------------------------
        DQN tiêu chuẩn:  y = r + gamma * max_a Q(s', a; theta⁻)
        Double DQN:      y = r + gamma * Q(s', argmax_a Q(s', a; theta); theta⁻)

        Bằng cách sử dụng mạng trực tuyến theta để chọn hành động và mạng mục tiêu theta⁻
        để đánh giá giá trị, Double DQN tránh được hiện tượng đánh giá giá trị quá cao.

        Trả về
        -----
        loss : float hoặc None (None nếu bộ đệm chưa đủ mẫu)
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # --- Lấy mẫu từ bộ đệm --------------------------------------------
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

        # Chuyển đổi thành Tensors
        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)   # (B, n_pairs)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)  # (B, 1)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)    # (B, 1)

        # --- Tính giá trị Q hiện tại --------------------------------------
        q_online = self.online_net(states_t)       # (B, n_pairs, n_levels)

        # Lấy giá trị Q ứng với các hành động đã thực hiện
        # actions_t: (B, n_pairs) -> unsqueeze -> (B, n_pairs, 1)
        current_q = q_online.gather(
            dim=2,
            index=actions_t.unsqueeze(2)
        ).squeeze(2)                               # (B, n_pairs)

        # --- Tính mục tiêu Double DQN (Double DQN Target) -----------------
        with torch.no_grad():
            # LỰA CHỌN HÀNH ĐỘNG: sử dụng mạng trực tuyến (online net)
            next_q_online = self.online_net(next_states_t)        # (B, n_pairs, n_levels)
            best_actions  = next_q_online.argmax(dim=2, keepdim=True)  # (B, n_pairs, 1)

            # ĐÁNH GIÁ GIÁ TRỊ: sử dụng mạng mục tiêu (target net)
            next_q_target = self.target_net(next_states_t)        # (B, n_pairs, n_levels)
            next_q_values = next_q_target.gather(
                dim=2, index=best_actions
            ).squeeze(2)                                          # (B, n_pairs)

            # y = r + gamma * Q_target(s', a*)  ×  (1 - done)
            # rewards_t: (B, 1) broadcast tới (B, n_pairs)
            target_q = rewards_t + self.gamma * next_q_values * (1.0 - dones_t)

        # --- Tính hàm tổn thất --------------------------------------------
        td_errors = current_q - target_q                           # (B, n_pairs)

        if is_weights is not None:
            # PER: nhân trọng số tầm quan trọng (importance-sampling weights)
            loss = (is_weights.unsqueeze(1) * td_errors.pow(2)).mean()
            # Cập nhật độ ưu tiên trong bộ đệm PER
            per_errors = td_errors.detach().abs().mean(dim=1).cpu().numpy()
            self.replay_buffer.update_priorities(indices, per_errors)
        else:
            loss = F.mse_loss(current_q, target_q)

        # --- Lan truyền ngược ---------------------------------------------
        self.optimizer.zero_grad()
        loss.backward()
        # Cắt xới đạo hàm (Gradient clipping) để ổn định huấn luyện
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.scheduler.step()

        # --- Cập nhật mềm mạng mục tiêu (Polyak averaging) ----------------
        # theta⁻ ← tau*theta + (1-tau)*theta⁻
        self._soft_update_target()

        loss_val = loss.item()
        self.total_loss += loss_val
        self.update_count += 1

        # Ghi log TensorBoard (cho mỗi bước)
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
    # Cập nhật Mạng Mục tiêu
    # ------------------------------------------------------------------

    def _soft_update_target(self) -> None:
        """
        Cập nhật mềm (Polyak): theta⁻ ← tau*theta + (1-tau)*theta⁻

        Dần dần cập nhật trọng số của mạng mục tiêu theo mạng trực tuyến,
        giúp mục tiêu hồi quy mượt mà và ổn định hơn so với cập nhật cứng định kỳ.
        """
        for param, target_param in zip(
            self.online_net.parameters(),
            self.target_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def hard_update_target(self) -> None:
        """Sao chép toàn bộ trọng số mạng trực tuyến sang mạng mục tiêu."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ------------------------------------------------------------------
    # Ghi log Tập huấn luyện
    # ------------------------------------------------------------------

    def log_episode(
        self,
        episode_reward: float,
        episode_cost: float,
        service_level: float,
        stockout_rate: float,
    ) -> None:
        """
        Ghi lại các chỉ số của mỗi tập vào TensorBoard.

        Tham số
        ------
        episode_reward : float -- phần thưởng tích lũy của tập
        episode_cost   : float -- tổng chi phí tồn kho
        service_level  : float -- tỷ lệ nhu cầu được đáp ứng
        stockout_rate  : float -- tỷ lệ nhu cầu bị thiếu hàng
        """
        ep = self.episode_count
        self.writer.add_scalar("episode/reward",        episode_reward, ep)
        self.writer.add_scalar("episode/cost",          episode_cost,   ep)
        self.writer.add_scalar("episode/service_level", service_level,  ep)
        self.writer.add_scalar("episode/stockout_rate", stockout_rate,  ep)
        self.writer.add_scalar("episode/epsilon",       self.epsilon,   ep)
        self.episode_count += 1

    # ------------------------------------------------------------------
    # Lưu / Tải Mô hình
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Lưu trọng số tác tử và trạng thái huấn luyện vào file checkpoint."""
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
        print(f"[DoubleDQNAgent] Đã lưu checkpoint -> {path}")

    def load(self, path: str) -> None:
        """Tải tác tử từ file checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step   = ckpt.get("global_step", 0)
        self.episode_count = ckpt.get("episode_count", 0)
        print(f"[DoubleDQNAgent] Đã tải checkpoint ← {path}")

    def close(self) -> None:
        """Đóng TensorBoard writer."""
        self.writer.close()


# ===========================================================================
# Kiểm tra nhanh
# ===========================================================================
if __name__ == "__main__":
    print("Đang thử nghiệm DoubleDQNAgent (chế độ CPU)...")
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

    # Thêm dữ liệu chuyển trạng thái mẫu vào bộ đệm
    for _ in range(100):
        s  = np.random.randn(state_dim).astype(np.float32)
        a  = np.random.randint(0, 6, size=n_pairs).astype(np.int32)
        r  = float(np.random.randn())
        ns = np.random.randn(state_dim).astype(np.float32)
        d  = False
        agent.store_transition(s, a, r, ns, d)

    # Chọn hành động
    obs = np.random.randn(state_dim).astype(np.float32)
    action = agent.select_action(obs)
    print(f"  Kích thước hành động: {action.shape}, mẫu: {action[:5]}")
    print(f"  Epsilon: {agent.epsilon:.4f}")

    # Bước cập nhật
    loss = agent.update()
    print(f"  Hàm tổn thất (Loss): {loss:.6f}")

    agent.save("runs/test/test_ckpt.pth")
    agent.close()
    print("DoubleDQNAgent HOÀN THÀNH TỐT!")
