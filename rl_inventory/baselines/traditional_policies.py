"""
baselines/traditional_policies.py
===================================
Các phương pháp quản lý tồn kho truyền thống (Baseline) để so sánh với DQN.

Triển khai 3 chiến lược cổ điển (Bắt buộc cho đánh giá chuẩn trong khóa luận):
  1. EOQ  -- Lượng đặt hàng kinh tế (Harris, 1913)
  2. Chiến lược (s,S) -- Điểm đặt hàng lại (Reorder-point) / Mức đặt hàng lên đến S (Order-up-to)
  3. Mô hình Newsvendor -- Bài toán người bán báo (Critical fractile đơn thời kỳ)

Tất cả các chiến lược đều cung cấp cùng một giao diện chuẩn:
    policy.get_action(env_state) -> np.ndarray (n_pairs,) chứa các chỉ số hành động đặt hàng

Điều này cho phép đánh giá dạng "plug-and-play" trong cùng một vòng lặp môi trường
với tác tử DQN, đảm bảo sự so sánh công bằng.

Tài liệu tham khảo (Trích dẫn khóa luận):
------------------------------------------
- EOQ: Harris, F.W. (1913). How many parts to make at once.
       Factory, The Magazine of Management, 10(2), 135-136.
- (s,S): Scarf, H. (1960). The optimality of (s,S) policies in dynamic
         inventory problems. Mathematical Methods in the Social Sciences.
- Newsvendor: Arrow, K.J., Harris, T., Marschak, J. (1951).
              Optimal inventory policy. Econometrica, 19(3), 250-272.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any
from scipy import stats


# ===========================================================================
# Lớp cơ sở (Interface chung)
# ===========================================================================

class BasePolicy:
    """
    Lớp cơ sở trừu tượng cho tất cả các chiến lược tồn kho.

    Tất cả các chiến lược nhận tóm tắt lịch sử nhu cầu và trả về một mảng
    hành động tương thích với không gian hành động của MultiWarehouseInventoryEnv.
    """

    def __init__(self, n_pairs: int, n_action_levels: int = 6,
                 order_levels: Optional[list] = None):
        """
        Tham số
        ------
        n_pairs : int
            Số lượng cặp kho - SKU.
        n_action_levels : int
            Số mức số lượng đặt hàng rời rạc (phải khớp với môi trường).
        order_levels : list
            Số lượng đặt hàng thực tế tương ứng với các chỉ số hành động.
            Mặc định: [0, 10, 20, 30, 40, 50].
        """
        self.n_pairs = n_pairs
        self.n_action_levels = n_action_levels
        self.order_levels = np.array(
            order_levels if order_levels else [0, 10, 20, 30, 40, 50],
            dtype=np.float32,
        )

    def _qty_to_action(self, order_qty: np.ndarray) -> np.ndarray:
        """
        Chuyển đổi số lượng đặt hàng liên tục sang chỉ số hành động rời rạc.

        Tìm mức gần nhất trong self.order_levels cho từng cặp.

        Tham số
        ------
        order_qty : np.ndarray, shape (n_pairs,)
            Số lượng đặt hàng mong muốn (đơn vị).

        Trả về
        -----
        actions : np.ndarray, shape (n_pairs,), dtype int
        """
        actions = np.zeros(self.n_pairs, dtype=np.int32)
        for i, qty in enumerate(order_qty):
            # Chọn mức hành động rời rạc gần nhất
            idx = int(np.argmin(np.abs(self.order_levels - qty)))
            actions[i] = idx
        return actions

    def get_action(self, inventory: np.ndarray, demand_hist: np.ndarray,
                   **kwargs) -> np.ndarray:
        """
        Tính toán hành động đặt hàng.

        Tham số
        ------
        inventory : np.ndarray, shape (n_pairs,)
            Mức tồn kho hiện tại (đã làm phẳng).
        demand_hist : np.ndarray, shape (n_pairs, lookback)
            Lịch sử nhu cầu cho từng cặp.

        Trả về
        -----
        actions : np.ndarray, shape (n_pairs,), dtype int
        """
        raise NotImplementedError

    def reset(self):
        """Đặt lại trạng thái nội bộ tại thời điểm bắt đầu một tập mới."""
        pass


# ===========================================================================
# 1. EOQ -- Economic Order Quantity (Mô hình Lượng đặt hàng Kinh tế)
# ===========================================================================

class EOQPolicy(BasePolicy):
    """
    Chiến lược Lượng đặt hàng Kinh tế (EOQ).

    Công thức EOQ (Harris, 1913) tối thiểu hóa tổng chi phí đặt hàng và chi phí lưu kho
    trong điều kiện nhu cầu xác định và cố định:

        Q* = sqrt(2 × D × K / h)

    Trong đó:
        D = nhu cầu trung bình hàng ngày (đơn vị/ngày)
        K = chi phí đặt hàng cố định cho mỗi đơn hàng ($)
        h = chi phí lưu kho trên mỗi đơn vị mỗi ngày ($)

    Áp dụng cho môi trường đa nhà kho hành động rời rạc:
    - Ước tính D từ lịch sử nhu cầu (trung bình trượt)
    - Tính Q* bằng công thức EOQ
    - Đặt hàng khi tồn kho xuống dưới điểm đặt hàng lại ROP = D × lead_time + safety_stock
    - Làm tròn Q* về mức hành động rời rạc gần nhất

    Trích dẫn (dùng cho khóa luận):
        Harris, F.W. (1913). How many parts to make at once.
        Factory, The Magazine of Management, 10(2), 135-136.

    Tham số
    ------
    n_pairs : int
    holding_cost : float -- h, chi phí lưu kho trên đơn vị mỗi ngày
    ordering_cost : float -- K, chi phí đặt hàng cố định
    lead_time : float -- thời gian cung ứng trung bình (ngày)
    safety_stock_k : float -- hệ số tồn kho an toàn (sigma × k)
    """

    def __init__(
        self,
        n_pairs: int,
        holding_cost: float = 1.0,
        ordering_cost: float = 50.0,
        lead_time: float = 2.0,
        safety_stock_k: float = 1.65,   # 95% service level
        n_action_levels: int = 6,
        order_levels: Optional[list] = None,
    ):
        super().__init__(n_pairs, n_action_levels, order_levels)
        self.h = holding_cost
        self.K = ordering_cost
        self.lead_time = lead_time
        self.safety_stock_k = safety_stock_k

    def get_action(
        self,
        inventory: np.ndarray,
        demand_hist: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Tính số lượng đặt hàng dựa trên EOQ.

        Tham số
        ------
        inventory : np.ndarray, shape (n_pairs,)
        demand_hist : np.ndarray, shape (n_pairs, lookback)

        Trả về
        -----
        actions : np.ndarray, shape (n_pairs,), dtype int
        """
        order_qty = np.zeros(self.n_pairs, dtype=np.float32)

        for i in range(self.n_pairs):
            hist = demand_hist[i]                # (lookback,)
            D = float(hist.mean()) + 1e-6        # nhu cầu hàng ngày trung bình

            # Tồn kho an toàn: k × sigma × sqrt(lead_time)
            sigma = float(hist.std()) + 1e-6
            safety_stock = self.safety_stock_k * sigma * np.sqrt(self.lead_time)

            # Điểm đặt hàng lại (Reorder Point - ROP)
            ROP = D * self.lead_time + safety_stock

            if inventory[i] <= ROP:
                # Công thức EOQ: Q* = sqrt(2DK/h)
                eoq = np.sqrt(2.0 * D * self.K / (self.h + 1e-6))
                order_qty[i] = eoq
            else:
                order_qty[i] = 0.0

        return self._qty_to_action(order_qty)

    def get_eoq(self, avg_demand: float) -> float:
        """Trả về EOQ cho nhu cầu trung bình cho trước (dùng cho phân tích)."""
        return np.sqrt(2.0 * avg_demand * self.K / (self.h + 1e-6))


# ===========================================================================
# 2. Chiến lược (s, S) -- Điểm đặt hàng lại / Mức đặt hàng lên đến S
# ===========================================================================

class SsPolicyOptimized(BasePolicy):
    """
    Chiến lược Tồn kho (s, S).

    Một trong những kết quả quan trọng nhất trong lý thuyết quản lý tồn kho (Scarf, 1960):
    Đối với quản lý tồn kho kiểm tra định kỳ có chi phí đặt hàng cố định,
    chiến lược tối ưu có dạng (s, S):
      - Nếu tồn kho ≤ s (điểm đặt hàng lại): đặt hàng sao cho tồn kho đạt mức S (Order-up-to)
      - Ngược lại: không đặt hàng

    Các tham số được ước tính từ lịch sử nhu cầu:
      s = mu_LT + z_alpha × sigma_LT
      S = s + EOQ
    trong đó mu_LT, sigma_LT là giá trị trung bình/độ lệch chuẩn của nhu cầu trong lead time,
    và z_alpha là z-score tương ứng với mức độ phục vụ.

    Trích dẫn (dùng cho khóa luận):
        Scarf, H. (1960). The optimality of (s,S) policies in the dynamic
        inventory problem. Mathematical Methods in the Social Sciences, 196-202.

    Tham số
    ------
    n_pairs : int
    s_params : dict, optional -- cài sẵn {pair_idx: (s, S)} cho mỗi cặp
    service_level : float -- mức độ phục vụ mục tiêu (ví dụ: 0.95)
    lead_time : float -- thời gian cung ứng trung bình (ngày)
    holding_cost : float
    ordering_cost : float
    """

    def __init__(
        self,
        n_pairs: int,
        service_level: float = 0.95,
        lead_time: float = 2.0,
        holding_cost: float = 1.0,
        ordering_cost: float = 50.0,
        n_action_levels: int = 6,
        order_levels: Optional[list] = None,
        s_params: Optional[Dict[int, tuple]] = None,
    ):
        super().__init__(n_pairs, n_action_levels, order_levels)
        self.service_level = service_level
        self.lead_time = lead_time
        self.h = holding_cost
        self.K = ordering_cost
        self.z = float(stats.norm.ppf(service_level))  # z-score cho mức độ phục vụ
        self.s_params = s_params  # {pair_idx: (s_val, S_val)}

    def _compute_s_S(
        self, avg_demand: float, std_demand: float
    ) -> tuple:
        """
        Tính toán tham số (s, S) từ thống kê nhu cầu.

        s = điểm đặt hàng lại = mu_LT + z × sigma_LT
        S = mức đặt hàng lên đến = s + EOQ
        """
        mu_lt = avg_demand * self.lead_time
        sigma_lt = std_demand * np.sqrt(self.lead_time)
        s = mu_lt + self.z * sigma_lt
        eoq = np.sqrt(2.0 * avg_demand * self.K / (self.h + 1e-6))
        S = s + eoq
        return float(s), float(S)

    def get_action(
        self,
        inventory: np.ndarray,
        demand_hist: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Áp dụng chiến lược (s, S).

        Tham số
        ------
        inventory : np.ndarray, shape (n_pairs,)
        demand_hist : np.ndarray, shape (n_pairs, lookback)

        Trả về
        -----
        actions : np.ndarray, shape (n_pairs,), dtype int
        """
        order_qty = np.zeros(self.n_pairs, dtype=np.float32)

        for i in range(self.n_pairs):
            hist = demand_hist[i]
            D = float(hist.mean()) + 1e-6
            sigma = float(hist.std()) + 1e-6

            # Sử dụng tham số cài sẵn hoặc tính từ lịch sử
            if self.s_params and i in self.s_params:
                s, S = self.s_params[i]
            else:
                s, S = self._compute_s_S(D, sigma)

            cur_inv = float(inventory[i])
            if cur_inv <= s:
                # Đặt hàng để đạt mức S
                order_qty[i] = max(0.0, S - cur_inv)
            else:
                order_qty[i] = 0.0

        return self._qty_to_action(order_qty)


# ===========================================================================
# 3. Mô hình Newsvendor (Bài toán Người bán báo)
# ===========================================================================

class NewsvendorPolicy(BasePolicy):
    """
    Mô hình Newsvendor (mô hình phân số tới hạn - critical fractile).

    Mô hình Newsvendor (Arrow et al., 1951) tìm số lượng đặt hàng tối ưu
    cho bài toán nhu cầu ngẫu nhiên đơn thời kỳ bằng cách cân bằng:
      - Chi phí thừa hàng c_o (overage cost - chi phí lưu kho dư thừa)
      - Chi phí thiếu hàng c_u (underage cost - chi phí đền bù/mất doanh thu)

    Số lượng đặt hàng tối ưu:
        Q* = F⁻¹(c_u / (c_u + c_o))   (tỷ lệ tới hạn - critical fractile)

    trong đó F là hàm phân phối tích lũy CDF của nhu cầu (giả định Chuẩn, ước tính từ lịch sử).

    Được mở rộng cho bài toán đa thời kỳ theo kiểu cửa sổ trượt bằng cách
    áp dụng tối ưu đơn thời kỳ tại mỗi bước.

    Trích dẫn (dùng cho khóa luận):
        Arrow, K.J., Harris, T., & Marschak, J. (1951).
        Optimal inventory policy. Econometrica, 19(3), 250-272.

    Tham số
    ------
    n_pairs : int
    holding_cost : float -- c_o (chi phí thừa hàng trên đơn vị)
    stockout_cost : float -- c_u (chi phí thiếu hàng trên đơn vị)
    lead_time : float -- thời gian cung ứng trung bình để tổng hợp nhu cầu
    """

    def __init__(
        self,
        n_pairs: int,
        holding_cost: float = 1.0,
        stockout_cost: float = 10.0,
        lead_time: float = 2.0,
        n_action_levels: int = 6,
        order_levels: Optional[list] = None,
    ):
        super().__init__(n_pairs, n_action_levels, order_levels)
        self.c_o = holding_cost
        self.c_u = stockout_cost
        self.lead_time = lead_time

        # Tỷ lệ tới hạn (mức độ phục vụ hàm ý từ chi phí)
        self.critical_ratio = self.c_u / (self.c_u + self.c_o)
        # z-score tương ứng với tỷ lệ tới hạn
        self.z_star = float(stats.norm.ppf(self.critical_ratio))

    def get_action(
        self,
        inventory: np.ndarray,
        demand_hist: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Tính số lượng đặt hàng tối ưu theo Newsvendor.

        Q* = mu_LT + z* × sigma_LT  (đặt hàng đến mức này nếu thấp hơn)
        Số lượng đặt hàng = max(0, Q* - inventory_hien_tai)

        Tham số
        ------
        inventory : np.ndarray, shape (n_pairs,)
        demand_hist : np.ndarray, shape (n_pairs, lookback)

        Trả về
        -----
        actions : np.ndarray, shape (n_pairs,), dtype int
        """
        order_qty = np.zeros(self.n_pairs, dtype=np.float32)

        for i in range(self.n_pairs):
            hist = demand_hist[i]
            D = float(hist.mean()) + 1e-6
            sigma = float(hist.std()) + 1e-6

            # Mức tồn kho tối ưu (tính theo nhu cầu trong lead-time)
            mu_lt    = D * self.lead_time
            sigma_lt = sigma * np.sqrt(self.lead_time)
            q_star   = mu_lt + self.z_star * sigma_lt

            # Đặt hàng để đưa tồn kho lên mức q_star
            cur_inv = float(inventory[i])
            order_qty[i] = max(0.0, q_star - cur_inv)

        return self._qty_to_action(order_qty)

    @property
    def implied_service_level(self) -> float:
        """Mức độ phục vụ hàm ý từ tỷ lệ chi phí."""
        return self.critical_ratio


# ===========================================================================
# Trích xuất trạng thái -- lấy trạng thái từ dict info của môi trường
# ===========================================================================

def extract_state_for_policy(
    obs: np.ndarray,
    env_config: Dict[str, Any],
) -> tuple:
    """
    Trích xuất tồn kho và lịch sử nhu cầu từ vectơ quan sát phẳng.

    Cấu trúc quan sát của môi trường:
      [inventory (n_pairs) | demand_hist (n_pairs × lookback) | pipeline | dow]

    Tham số
    ------
    obs : np.ndarray, shape (obs_dim,)
        Vectơ quan sát thô từ môi trường.
    env_config : dict
        Phải chứa 'n_warehouses', 'n_skus', 'lookback', 'max_inventory'.

    Trả về
    -----
    inventory : np.ndarray, shape (n_pairs,) -- số đơn vị thô (đã giải chuẩn hóa)
    demand_hist : np.ndarray, shape (n_pairs, lookback) -- số đơn vị thô
    """
    n_w = env_config["n_warehouses"]
    n_s = env_config["n_skus"]
    n_pairs = n_w * n_s
    lookback = env_config["lookback"]
    max_inv = env_config["max_inventory"]

    # Giải chuẩn hóa từ [0,1] về đơn vị thực tế
    inv_norm = obs[:n_pairs]
    inventory = inv_norm * max_inv

    dem_norm = obs[n_pairs: n_pairs + n_pairs * lookback]
    demand_hist = dem_norm.reshape(n_pairs, lookback)
    # Giải chuẩn hóa nhu cầu (xấp xỉ -- dùng mức đặt hàng tối đa làm thang đo)
    max_dem = max(env_config.get("order_levels", [50]))
    demand_hist = demand_hist * max_dem

    return inventory, demand_hist


# ===========================================================================
# Kiểm tra nhanh
# ===========================================================================

if __name__ == "__main__":
    import numpy as np

    N_PAIRS = 60
    LOOKBACK = 7

    # Trạng thái giả định
    inv = np.random.uniform(50, 200, size=N_PAIRS).astype(np.float32)
    dem = np.random.poisson(15, size=(N_PAIRS, LOOKBACK)).astype(np.float32)

    print("Đang thử nghiệm EOQ Policy...")
    eoq = EOQPolicy(n_pairs=N_PAIRS)
    actions = eoq.get_action(inv, dem)
    print(f"  Kích thước hành động: {actions.shape}, mẫu: {actions[:5]}")

    print("\nĐang thử nghiệm (s,S) Policy...")
    ss = SsPolicyOptimized(n_pairs=N_PAIRS)
    actions = ss.get_action(inv, dem)
    print(f"  Kích thước hành động: {actions.shape}, mẫu: {actions[:5]}")

    print("\nĐang thử nghiệm Newsvendor Policy...")
    nv = NewsvendorPolicy(n_pairs=N_PAIRS)
    actions = nv.get_action(inv, dem)
    print(f"  Kích thước hành động: {actions.shape}, mẫu: {actions[:5]}")
    print(f"  Mức độ phục vụ hàm ý: {nv.implied_service_level:.4f}")

    print("\nTất cả các baseline HOÀN THÀNH TỐT!")
