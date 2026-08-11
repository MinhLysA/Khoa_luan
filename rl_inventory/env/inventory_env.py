"""
env/inventory_env.py
====================
Môi trường Gymnasium tùy chỉnh cho bài toán Quản lý Tồn kho Đa Kho.

Lựa chọn thiết kế (ghi chú cho khóa luận tốt nghiệp):
-----------------------------------------------------
1. KHÔNG GIAN HÀNH ĐỘNG — Rời rạc (Phân rã - Factorized):
   Mỗi cặp nhà kho - SKU chọn độc lập từ N_ORDER_LEVELS mức số lượng đặt hàng
   rời rạc (ví dụ: {0, 10, 20, 30, 40, 50} đơn vị). Việc này tránh không gian hành
   động liên hợp quá lớn (6^60 cho 2 kho × 30 SKU) bằng cách sử dụng chiến lược
   phân rã/chọn hành động độc lập — một kỹ thuật tiêu chuẩn trong tài liệu Học sâu
   tăng cường đa tác tử (Sunehag et al., 2018 — VDN; Rashid et al., 2018 — QMIX).
   Trong mô hình tác tử đơn của chúng ta, mô hình DQN tính toán giá trị Q-value cho
   từng cặp độc lập và chọn argmax trên từng cặp.

2. KHÔNG GIAN TRẠNG THÁI:
   - Mức tồn kho hiện tại (n_warehouses × n_skus)
   - Lịch sử nhu cầu gần đây (lookback=7 ngày) cho mỗi cặp kho - SKU
   - Tồn kho trên đường vận chuyển (Pipeline inventory): các đơn hàng đã đặt nhưng chưa về đến kho (lead time)
   - Mã hóa ngày trong tuần (mã hóa one-hot 7 chiều) để bắt tính mùa vụ

3. HÀM THƯỞNG (REWARD FUNCTION):
   R_t = -(h * sum(inventory_t) + p * sum(stockout_t) + K * sum(I(order_t > 0)))
   trong đó:
     h = chi phí lưu kho trên mỗi đơn vị mỗi ngày (holding cost)
     p = chi phí thiếu hàng trên mỗi đơn vị nhu cầu không được đáp ứng (stockout penalty)
     K = chi phí cố định cho mỗi đơn đặt hàng được tạo (ordering cost)
   Ngoài ra còn có hình phạt bổ sung nếu lượng tồn kho vượt quá dung lượng tối đa (max_capacity).

4. ĐỘNG LỰC HỌC CHUYỂN TRẠNG THÁI (TRANSITION DYNAMICS):
   - Nhu cầu được lấy từ dữ liệu thực tế M5 Walmart (hoặc dữ liệu tổng hợp nếu không có M5)
   - Đơn hàng đặt tại thời điểm t sẽ về kho tại t + lead_time (thời gian cung ứng ngẫu nhiên)
   - Tồn kho bị giới hạn bởi max_capacity (ràng buộc dung tích nhà kho)
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


# ---------------------------------------------------------------------------
# Cấu hình mặc định — ghi đè thông qua dict EnvConfig
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # Kích thước hệ thống tồn kho
    "n_warehouses": 2,
    "n_skus": 30,

    # Các tham số chi phí
    "holding_cost": 1.0,        # h: chi phí lưu kho trên mỗi đơn vị mỗi bước thời gian
    "stockout_cost": 10.0,      # p: chi phí thiếu hàng trên mỗi đơn vị không đáp ứng
    "ordering_cost": 50.0,      # K: chi phí cố định cho mỗi đơn đặt hàng (trên mỗi SKU)
    "overflow_penalty": 5.0,    # hình phạt cho mỗi đơn vị vượt quá dung lượng tối đa

    # Ràng buộc nhà kho
    "max_inventory": 500,       # mức tồn kho tối đa cho mỗi vị trí kho-SKU
    "initial_inventory": 100,   # mức tồn kho ban đầu

    # Các mức số lượng đặt hàng (ánh xạ hành động rời rạc)
    # Tác tử chọn chỉ số 0..5 → ánh xạ tới số đơn vị bên dưới
    "order_levels": [0, 10, 20, 30, 40, 50],

    # Thời gian cung ứng (ngày): ngẫu nhiên U[min, max]
    "lead_time_min": 1,
    "lead_time_max": 3,

    # Cửa sổ lịch sử nhu cầu (ngày)
    "lookback": 7,

    # Độ dài một tập huấn luyện/đánh giá (ngày)
    "episode_length": 112,      # 16 tuần × 7 ngày

    # Hạt giống ngẫu nhiên
    "seed": 42,
}


class MultiWarehouseInventoryEnv(gym.Env):
    """
    Môi trường Quản lý Tồn kho Đa Kho.

    Mô phỏng các quyết định tồn kho cho (n_warehouses × n_skus) cặp kho-SKU
    trong một tập có thời hạn hữu hạn. Nhu cầu được lấy từ dữ liệu thực tế
    M5 Walmart hoặc dữ liệu tổng hợp nếu M5 không sẵn có.

    Thuộc tính
    ----------
    config : dict
        Cấu hình môi trường (chi phí, kích thước, lead time, v.v.)
    demand_data : np.ndarray, shape (T, n_warehouses, n_skus)
        Chuỗi dữ liệu nhu cầu hàng ngày đã tải trước. T phải >= episode_length + lookback.
    n_pairs : int
        Tổng số cặp kho-SKU = n_warehouses × n_skus.
    n_action_levels : int
        Số lượng lựa chọn rời rạc về số lượng đặt hàng cho mỗi cặp.
    observation_space : gym.Space
        Không gian quan sát dạng Box phẳng.
    action_space : gym.Space
        MultiDiscrete — một lựa chọn rời rạc cho mỗi cặp kho-SKU.

    Ví dụ
    -----
    >>> import numpy as np
    >>> from env.inventory_env import MultiWarehouseInventoryEnv
    >>> env = MultiWarehouseInventoryEnv()  # sử dụng nhu cầu tổng hợp
    >>> obs, info = env.reset()
    >>> action = env.action_space.sample()
    >>> obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        demand_data: Optional[np.ndarray] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Khởi tạo môi trường.

        Tham số
        ------
        config : dict, optional
            Ghi đè bất kỳ khóa nào trong DEFAULT_CONFIG.
        demand_data : np.ndarray, optional
            Hình dạng (T, n_warehouses, n_skus). Nếu là None, sử dụng nhu cầu
            tổng hợp (Poisson với trung bình ngẫu nhiên) để thử nghiệm/gỡ lỗi.
        render_mode : str, optional
            'human' để in ra console, 'ansi' để trả về dạng chuỗi.
        """
        super().__init__()

        # Kết hợp cấu hình người dùng với cấu hình mặc định
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.render_mode = render_mode

        # Các biến viết tắt thuận tiện
        self.n_w = self.config["n_warehouses"]
        self.n_s = self.config["n_skus"]
        self.n_pairs = self.n_w * self.n_s
        self.lookback = self.config["lookback"]
        self.episode_len = self.config["episode_length"]
        self.order_levels = np.array(self.config["order_levels"], dtype=np.float32)
        self.n_action_levels = len(self.order_levels)
        self.lead_min = self.config["lead_time_min"]
        self.lead_max = self.config["lead_time_max"]
        self.max_inv = self.config["max_inventory"]
        self.init_inv = self.config["initial_inventory"]

        # Thiết lập dữ liệu nhu cầu
        self._setup_demand(demand_data)

        # -----------------------------------------------------------------
        # Không gian Hành động: MultiDiscrete
        # Mỗi cặp trong số n_pairs cặp kho-SKU độc lập lựa chọn
        # một chỉ số trong [0, n_action_levels - 1].
        # MultiDiscrete([n_action_levels] * n_pairs)
        # -----------------------------------------------------------------
        self.action_space = spaces.MultiDiscrete(
            [self.n_action_levels] * self.n_pairs
        )

        # -----------------------------------------------------------------
        # Không gian Quan sát: Box phẳng
        # Các thành phần:
        #   [0 : n_pairs]                    — tồn kho hiện tại (đã chuẩn hóa)
        #   [n_pairs : n_pairs*(lookback+1)] — lịch sử nhu cầu (đã chuẩn hóa)
        #   [... : ... + n_pairs*lead_max]   — hàng trên đường về (đã chuẩn hóa)
        #   [... : ... + 7]                  — ngày trong tuần one-hot
        # -----------------------------------------------------------------
        self.inv_dim = self.n_pairs                           # tồn kho hiện tại
        self.dem_dim = self.n_pairs * self.lookback           # lịch sử nhu cầu
        self.pip_dim = self.n_pairs * self.lead_max           # hàng trên đường về
        self.dow_dim = 7                                      # ngày trong tuần
        self.obs_dim = self.inv_dim + self.dem_dim + self.pip_dim + self.dow_dim

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # Hằng số chuẩn hóa
        self._norm_inv = float(self.max_inv)
        self._norm_dem = float(self.order_levels.max())

        # Trạng thái nội bộ (được khởi tạo trong reset)
        self.inventory: np.ndarray = None       # (n_w, n_s)
        self.pipeline: np.ndarray = None        # (lead_max, n_w, n_s) — đang vận chuyển
        self.demand_history: np.ndarray = None  # (lookback, n_w, n_s)
        self.t: int = 0                         # bước thời gian hiện tại
        self.start_idx: int = 0                 # chỉ số bắt đầu tập trong demand_data
        self.episode_cost: float = 0.0

        # Lưu chỉ số để hiển thị / đánh giá
        self._episode_stockouts: float = 0.0
        self._episode_holding: float = 0.0
        self._episode_ordering: float = 0.0

        # Bộ tạo số ngẫu nhiên RNG
        self.np_random = np.random.default_rng(self.config["seed"])

    # ------------------------------------------------------------------
    # Thiết lập nhu cầu
    # ------------------------------------------------------------------

    def _setup_demand(self, demand_data: Optional[np.ndarray]) -> None:
        """
        Tải hoặc tạo dữ liệu nhu cầu.

        Tham số
        ------
        demand_data : np.ndarray hoặc None
            Nếu cung cấp, hình dạng phải là (T, n_warehouses, n_skus).
            Nếu là None, tạo nhu cầu tổng hợp theo phân phối Poisson.
        """
        required_len = self.episode_len + self.lookback + self.lead_max + 10

        if demand_data is not None:
            assert demand_data.ndim == 3, "demand_data phải là 3D: (T, n_w, n_s)"
            assert demand_data.shape[1] == self.n_w, \
                f"demand_data.shape[1]={demand_data.shape[1]} != n_warehouses={self.n_w}"
            assert demand_data.shape[2] == self.n_s, \
                f"demand_data.shape[2]={demand_data.shape[2]} != n_skus={self.n_s}"
            assert demand_data.shape[0] >= required_len, \
                f"Cần ít nhất {required_len} bước thời gian, nhận được {demand_data.shape[0]}"
            self.demand_data = demand_data.astype(np.float32)
            self.T = demand_data.shape[0]
        else:
            # Nhu cầu Poisson tổng hợp -- dùng thử nghiệm khi không có dữ liệu M5
            print("[MultiWarehouseInventoryEnv] Không có demand_data được cung cấp."
                  " Sử dụng nhu cầu Poisson tổng hợp (trung bình ~ U[5, 30]) để gỡ lỗi.")
            self.T = required_len + 200
            means = self.np_random.uniform(5, 30, size=(self.n_w, self.n_s))
            self.demand_data = self.np_random.poisson(
                means, size=(self.T, self.n_w, self.n_s)
            ).astype(np.float32)

        # Nhu cầu tối đa dùng cho chuẩn hóa
        self._norm_dem = max(float(self.demand_data.max()), 1.0)

    # ------------------------------------------------------------------
    # Gymnasium API Cốt lõi
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Đặt lại môi trường về thời điểm bắt đầu của một tập mới.

        Tham số
        ------
        seed : int, optional
            Ghi đè seed cấu hình để tái lập kết quả.
        options : dict, optional
            'start_idx': int — chỉ định chỉ số bắt đầu tập trong demand_data.

        Trả về
        -----
        observation : np.ndarray, shape (obs_dim,)
        info : dict
        """
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # Chọn chỉ số bắt đầu ngẫu nhiên trong chuỗi nhu cầu
        max_start = self.T - self.episode_len - self.lead_max - 10
        if options and "start_idx" in options:
            self.start_idx = int(options["start_idx"])
        else:
            self.start_idx = int(self.np_random.integers(self.lookback, max_start))

        self.t = 0

        # Khởi tạo tồn kho quanh giá trị init_inv với độ lệch nhỏ
        self.inventory = np.full(
            (self.n_w, self.n_s),
            fill_value=self.init_inv,
            dtype=np.float32,
        )
        noise = self.np_random.uniform(-10, 10, size=(self.n_w, self.n_s))
        self.inventory = np.clip(self.inventory + noise, 0, self.max_inv).astype(np.float32)

        # Pipeline: số đơn vị đang trên đường vận chuyển, đánh chỉ số [ngày_cho_đến_khi_về, w, s]
        self.pipeline = np.zeros((self.lead_max, self.n_w, self.n_s), dtype=np.float32)

        # Khởi tạo lịch sử nhu cầu từ dữ liệu (lookback ngày trước start_idx)
        hist_start = self.start_idx - self.lookback
        if hist_start < 0:
            pad = np.zeros((-hist_start, self.n_w, self.n_s), dtype=np.float32)
            real = self.demand_data[0:self.start_idx]
            self.demand_history = np.concatenate([pad, real], axis=0)
        else:
            self.demand_history = self.demand_data[hist_start:self.start_idx].copy()

        # Đặt lại các chỉ số tập
        self.episode_cost = 0.0
        self._episode_stockouts = 0.0
        self._episode_holding = 0.0
        self._episode_ordering = 0.0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Thực thi một bước thời gian (một ngày) trong môi trường.

        Tham số
        ------
        action : np.ndarray, shape (n_pairs,), dtype int
            Mỗi phần tử là chỉ số trong self.order_levels.
            Ví dụ: action[i] = 2 → đặt order_levels[2] = 20 đơn vị cho cặp i.

        Trả về
        -----
        observation : np.ndarray
        reward : float
        terminated : bool (luôn là False — sử dụng truncated cho độ dài tập)
        truncated : bool (True khi đạt đến episode_length)
        info : dict
        """
        assert self.inventory is not None, "Hãy gọi reset() trước khi step()."
        action = np.asarray(action, dtype=np.int32)
        assert action.shape == (self.n_pairs,), \
            f"Kích thước hành động mong đợi ({self.n_pairs},), nhận được {action.shape}"

        # ---- 1. Nhận hàng về từ pipeline (bước lead_time) -----------------
        # Đơn hàng đặt với lead_time=1 sẽ về vào hôm nay (chỉ số pipeline 0)
        arrivals = self.pipeline[0].copy()          # (n_w, n_s)
        self.inventory = np.minimum(
            self.inventory + arrivals, self.max_inv
        )

        # Dịch chuyển pipeline: ngày 1 thành ngày 0, v.v.
        self.pipeline = np.roll(self.pipeline, shift=-1, axis=0)
        self.pipeline[-1] = 0.0                     # xóa vị trí cuối

        # ---- 2. Đặt đơn hàng mới ------------------------------------------
        order_units = self.order_levels[action]     # (n_pairs,) float
        order_matrix = order_units.reshape(self.n_w, self.n_s)  # (n_w, n_s)

        # Lấy mẫu thời gian cung ứng ngẫu nhiên cho từng cặp kho-SKU
        lead_times = self.np_random.integers(
            self.lead_min, self.lead_max + 1, size=(self.n_w, self.n_s)
        )  # (n_w, n_s), mỗi giá trị trong [lead_min, lead_max]

        # Thêm đơn hàng vào pipeline tại vị trí lead-time tương ứng
        for w in range(self.n_w):
            for s in range(self.n_s):
                lt = lead_times[w, s] - 1           # chỉ số từ 0
                self.pipeline[lt, w, s] += order_matrix[w, s]

        # ---- 3. Ghi nhận nhu cầu thực tế ----------------------------------
        data_idx = self.start_idx + self.t
        demand = self.demand_data[data_idx].copy()  # (n_w, n_s)

        # Đáp ứng nhu cầu
        fulfilled = np.minimum(self.inventory, demand)
        stockout = demand - fulfilled               # nhu cầu chưa đáp ứng (n_w, n_s)
        self.inventory = self.inventory - fulfilled

        # Cập nhật lịch sử nhu cầu (cửa sổ trượt FIFO)
        self.demand_history = np.concatenate(
            [self.demand_history[1:], demand[np.newaxis, :, :]], axis=0
        )  # (lookback, n_w, n_s)

        # ---- 4. Tính toán phần thưởng (chi phí tổng âm) ------------------
        holding_cost = (
            self.config["holding_cost"] * np.sum(self.inventory)
        )
        stockout_cost = (
            self.config["stockout_cost"] * np.sum(stockout)
        )
        # Chi phí đặt hàng cố định: tính cho mỗi đơn hàng được đặt (>0 đơn vị)
        n_orders_placed = np.sum(order_matrix > 0)
        ordering_cost = self.config["ordering_cost"] * n_orders_placed

        # Phạt vượt dung lượng: số đơn vị vượt quá max_inv
        overflow = np.sum(np.maximum(self.inventory + arrivals - self.max_inv, 0.0))
        overflow_cost = self.config["overflow_penalty"] * overflow

        total_cost = holding_cost + stockout_cost + ordering_cost + overflow_cost
        reward = -total_cost

        # ---- 5. Cập nhật các chỉ số của tập ------------------------------
        self.episode_cost += total_cost
        self._episode_stockouts += float(np.sum(stockout))
        self._episode_holding += holding_cost
        self._episode_ordering += ordering_cost

        # ---- 6. Chuyển sang bước thời gian tiếp theo ---------------------
        self.t += 1
        terminated = False
        truncated = (self.t >= self.episode_len)

        obs = self._get_obs()
        info = self._get_info(
            demand=demand,
            stockout=stockout,
            order=order_matrix,
            holding_cost=holding_cost,
            stockout_cost=stockout_cost,
            ordering_cost=ordering_cost,
        )

        if self.render_mode == "human":
            self._render_human(info)

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Xây dựng Vectơ Quan sát
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """
        Xây dựng và trả về vectơ quan sát đã chuẩn hóa.

        Trả về
        -----
        obs : np.ndarray, shape (obs_dim,), dtype float32
            Tất cả các thành phần được chuẩn hóa về khoảng xấp xỉ [0, 1].
        """
        # (1) Tồn kho hiện tại — chuẩn hóa theo max_inventory
        inv_flat = (self.inventory.flatten() / self._norm_inv).astype(np.float32)

        # (2) Lịch sử nhu cầu — shape (lookback, n_w, n_s) → làm phẳng → chuẩn hóa
        dem_flat = (
            self.demand_history.reshape(-1) / self._norm_dem
        ).astype(np.float32)

        # (3) Tồn kho pipeline — tổng trên chiều lead-time cho mỗi cặp (w, s)
        pipeline_total = self.pipeline.sum(axis=0)          # (n_w, n_s)
        pip_flat = (
            pipeline_total.flatten() / self._norm_inv
        ).astype(np.float32)
        # Pad/trim về pip_dim
        pip_flat_full = np.zeros(self.pip_dim, dtype=np.float32)
        fill_len = min(len(pip_flat), self.pip_dim)
        pip_flat_full[:fill_len] = pip_flat[:fill_len]

        # (4) Ngày trong tuần one-hot
        day_idx = (self.start_idx + self.t) % 7
        dow = np.zeros(7, dtype=np.float32)
        dow[day_idx] = 1.0

        obs = np.concatenate([inv_flat, dem_flat, pip_flat_full, dow])
        assert obs.shape == (self.obs_dim,), \
            f"Kích thước quan sát không khớp: {obs.shape} vs {(self.obs_dim,)}"
        return obs

    # ------------------------------------------------------------------
    # Dict Info
    # ------------------------------------------------------------------

    def _get_info(self, **kwargs) -> Dict[str, Any]:
        """Tạo dictionary thông tin cho ghi log và đánh giá."""
        info = {
            "timestep": self.t,
            "episode_cost": self.episode_cost,
            "episode_stockouts": self._episode_stockouts,
            "episode_holding_cost": self._episode_holding,
            "episode_ordering_cost": self._episode_ordering,
            "inventory_mean": float(self.inventory.mean()),
            "inventory_total": float(self.inventory.sum()),
        }
        info.update(kwargs)
        return info

    # ------------------------------------------------------------------
    # Hiển thị (Render)
    # ------------------------------------------------------------------

    def render(self) -> Optional[str]:
        """
        Hiển thị trạng thái hiện tại của môi trường.

        Trả về
        -----
        str hoặc None tùy thuộc vào render_mode.
        """
        if self.render_mode == "human":
            self._render_human({})
        elif self.render_mode == "ansi":
            return self._render_ansi()

    def _render_human(self, info: Dict) -> None:
        """In tóm tắt ngắn gọn ra stdout."""
        print(
            f"[t={self.t:3d}] "
            f"inv_mean={self.inventory.mean():.1f}  "
            f"stockout={self._episode_stockouts:.1f}  "
            f"ep_cost={self.episode_cost:.1f}"
        )

    def _render_ansi(self) -> str:
        """Trả về chuỗi biểu diễn trạng thái hiện tại."""
        lines = [
            "=" * 50,
            f"Bước thời gian: {self.t} / {self.episode_len}",
            f"Tồn kho (kho 0, SKU 0..4): {self.inventory[0, :5]}",
            f"Chi phí tập tính đến hiện tại: {self.episode_cost:.2f}",
            f"Tổng số đơn vị thiếu hàng: {self._episode_stockouts:.1f}",
            "=" * 50,
        ]
        return "\n".join(lines)

    def close(self):
        """Dọn dẹp tài nguyên."""
        pass

    # ------------------------------------------------------------------
    # Hàm hỗ trợ: tính mức độ phục vụ (Service Level) để đánh giá
    # ------------------------------------------------------------------

    def get_service_level(self) -> float:
        """
        Tính mức độ phục vụ = (tổng_nhu_cầu - tổng_thiếu_hàng) / tổng_nhu_cầu.

        Nên được gọi sau khi hoàn thành một tập.

        Trả về
        -----
        float trong khoảng [0, 1]
        """
        if self.t == 0:
            return 1.0
        total_dem = float(self.demand_data[
            self.start_idx:self.start_idx + self.t
        ].sum())
        if total_dem == 0:
            return 1.0
        return max(0.0, 1.0 - self._episode_stockouts / total_dem)


# ---------------------------------------------------------------------------
# Kiểm tra nhanh
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    print("Đang tạo môi trường...")
    env = MultiWarehouseInventoryEnv(
        config={"n_warehouses": 2, "n_skus": 5, "episode_length": 14}
    )

    print(f"Không gian quan sát: {env.observation_space}")
    print(f"Không gian hành động:  {env.action_space}")
    print(f"Chiều quan sát:       {env.obs_dim}")

    print("\nĐang chạy kiểm tra gymnasium check_env...")
    check_env(env, warn=True)
    print("check_env THÀNH CÔNG!")

    print("\nĐang chạy 1 tập với chiến lược ngẫu nhiên...")
    obs, info = env.reset(seed=0)
    total_reward = 0.0
    for _ in range(14):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"Tổng phần thưởng của tập: {total_reward:.2f}")
    print(f"Mức độ phục vụ (Service level): {env.get_service_level():.4f}")
    print("HOÀN THÀNH.")
