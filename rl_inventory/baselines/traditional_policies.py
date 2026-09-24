"""
baselines/traditional_policies.py
=================================
Ba chinh sach ton kho co dien dung lam moc so sanh.

  1. EOQ         (Harris, 1913)
  2. (s, S)      (Scarf, 1960)
  3. Newsvendor  (Arrow, Harris & Marschak, 1951)

[FIX-4]  Luong dat hang duoc luong tu hoa qua env.qty_to_action(), vi bang muc
         dat hang nay phu thuoc tung cap khi dung muc dat tuong doi.
[FIX-14] Moi chinh sach deu co THAM SO TINH CHINH DUOC, de scripts/tune_baselines.py
         chay grid search tren chinh moi truong mo phong. Ban cu hardcode
         service_level = 0.95 va lead_time = 2.0, khien so sanh voi RL khong
         cong bang (khoa luan cam ket baseline duoc toi uu bang tim kiem luoi).

Giao dien chung:
    policy.get_action(env_state) -> np.ndarray (n_pairs,) chi so hanh dong
trong do env_state phai chua: inventory, demand_history, pipeline_orders,
qty_to_action (callable), mean_demand.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional


class BasePolicy:
    """Lop co so cho cac chinh sach baseline."""

    def __init__(self, n_pairs: int):
        self.n_pairs = n_pairs

    @staticmethod
    def _stats(env_state: Dict[str, Any]):
        """Uoc luong cau trung binh / do lech chuan tu cua so lich su."""
        dh = env_state["demand_history"]                     # (lookback, n_pairs)
        mean_d = np.mean(dh, axis=0)
        std_d  = np.std(dh, axis=0)
        # Khi cua so lich su con rong (dau episode), lui ve mean_demand da uoc
        # luong tu du lieu huan luyen thay vi de bang 0.
        fallback = env_state.get("mean_demand")
        if fallback is not None:
            mean_d = np.where(mean_d > 1e-6, mean_d, fallback)
        mean_d = np.maximum(mean_d, 1e-6)
        std_d  = np.maximum(std_d, 1e-6)
        return mean_d, std_d

    def get_action(self, env_state: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError


class EOQPolicy(BasePolicy):
    """
    EOQ (Harris, 1913):  Q* = sqrt(2 * D * K / h)

    Dat hang khi vi the ton kho xuong duoi diem dat lai ROP = mu_L.
    Tham so tinh chinh: q_factor (he so nhan Q*), lead_time (gia dinh).
    """

    def __init__(self, n_pairs: int, ordering_cost: float = 50.0,
                 holding_cost: float = 1.0, lead_time: float = 2.0,
                 q_factor: float = 1.0, **kwargs):
        super().__init__(n_pairs)
        self.K = ordering_cost
        self.h = holding_cost
        self.lead_time = lead_time
        self.q_factor = q_factor

    def get_action(self, env_state: Dict[str, Any]) -> np.ndarray:
        mean_d, _ = self._stats(env_state)
        inv_pos = env_state["inventory"] + env_state["pipeline_orders"].sum(axis=0)

        eoq = self.q_factor * np.sqrt(2.0 * mean_d * self.K / self.h)
        rop = mean_d * self.lead_time
        target = np.where(inv_pos <= rop, eoq, 0.0)
        return env_state["qty_to_action"](target)


class SsPolicy(BasePolicy):
    """
    Chinh sach (s, S) (Scarf, 1960).

        s = mu_L + z * sigma_L
        S = s + q_factor * EOQ
        Neu vi the ton kho <= s: dat S - vi the ton kho, nguoc lai khong dat.

    Tham so tinh chinh: service_level (qua z), lead_time, q_factor.
    """

    def __init__(self, n_pairs: int, service_level: float = 0.90,
                 lead_time: float = 2.0, q_factor: float = 1.0,
                 ordering_cost: float = 50.0, holding_cost: float = 1.0, **kwargs):
        super().__init__(n_pairs)
        self.service_level = service_level
        self.lead_time = lead_time
        self.q_factor = q_factor
        self.K = ordering_cost
        self.h = holding_cost
        self.z = stats.norm.ppf(service_level)

    def get_action(self, env_state: Dict[str, Any]) -> np.ndarray:
        mean_d, std_d = self._stats(env_state)
        inv_pos = env_state["inventory"] + env_state["pipeline_orders"].sum(axis=0)

        mu_L    = mean_d * self.lead_time
        sigma_L = std_d * np.sqrt(self.lead_time)
        s = mu_L + self.z * sigma_L

        eoq = self.q_factor * np.sqrt(2.0 * mean_d * self.K / self.h)
        S = s + eoq

        target = np.where(inv_pos <= s, np.maximum(0.0, S - inv_pos), 0.0)
        return env_state["qty_to_action"](target)


class NewsvendorPolicy(BasePolicy):
    """
    Mo hinh Newsvendor (Arrow, Harris & Marschak, 1951).

        Critical ratio CR = p / (p + h)
        S* = mu_L + z_CR * sigma_L

    LUU Y LY THUYET: CR tuong ung voi CYCLE SERVICE LEVEL (alpha), khong phai
    fill rate (beta). Voi p = 10, h = 1 thi CR = 10/11 ~ 0.909 la xac suat
    khong het hang trong mot chu ky, khong the so sanh truc tiep voi nguong
    fill rate muc_dv = 0.85.

    Tham so tinh chinh: cr_override (thay CR ly thuyet), lead_time.
    """

    def __init__(self, n_pairs: int, holding_cost: float = 1.0,
                 stockout_cost: float = 10.0, lead_time: float = 2.0,
                 cr_override: Optional[float] = None, **kwargs):
        super().__init__(n_pairs)
        self.h = holding_cost
        self.p = stockout_cost
        self.lead_time = lead_time
        cr = cr_override if cr_override is not None else self.p / (self.p + self.h)
        # Cho phep CR la mang theo tung cap (baseline tinh chinh theo nhom quy mo cau)
        self.critical_ratio = np.clip(np.asarray(cr, dtype=float), 0.01, 0.999)
        self.z_cr = stats.norm.ppf(self.critical_ratio)

    def get_action(self, env_state: Dict[str, Any]) -> np.ndarray:
        mean_d, std_d = self._stats(env_state)
        inv_pos = env_state["inventory"] + env_state["pipeline_orders"].sum(axis=0)

        mu_L    = mean_d * self.lead_time
        sigma_L = std_d * np.sqrt(self.lead_time)
        S_star = mu_L + self.z_cr * sigma_L

        target = np.maximum(0.0, S_star - inv_pos)
        return env_state["qty_to_action"](target)


# --------------------------------------------------------------------------- #
# Tien ich dung chung cho evaluate.py va tune_baselines.py
# --------------------------------------------------------------------------- #
def build_env_state(env) -> Dict[str, Any]:
    """Dong goi trang thai moi truong theo dung giao dien ma baseline can."""
    return {
        "inventory":       env.inventory.copy(),
        "demand_history":  env.demand_history.copy(),
        "pipeline_orders": env.pipeline_orders.copy(),
        "mean_demand":     env.mean_demand.copy(),
        "qty_to_action":   env.qty_to_action,
    }


# --------------------------------------------------------------------------- #
# Baseline tinh chinh THEO NHOM QUY MO CAU: moi nhom co bo tham so rieng.
# Moi tham so (service_level, q_factor, lead_time, cr_override) deu nhan duoc
# mang (n_pairs,), nen chi can "trai" tham so cua nhom ra tung cap.
# --------------------------------------------------------------------------- #
DEMAND_GROUP_EDGES = [0.5, 2.0, 10.0]       # nguong cau TB/ngay (mien train)
DEMAND_GROUP_NAMES = ["Rất thấp (<0,5)", "Thấp (0,5-2)", "Trung bình (2-10)", "Cao (>=10)"]


def demand_group_index(mean_demand) -> np.ndarray:
    """Chi so nhom (0..3) cua tung cap theo cau trung binh."""
    return np.digitize(np.asarray(mean_demand), DEMAND_GROUP_EDGES)


def expand_group_params(params_by_group: Dict[str, Dict[str, float]],
                        group_idx: np.ndarray) -> Dict[str, np.ndarray]:
    """{'0': {...}, '1': {...}, ...} -> {ten_tham_so: mang (n_pairs,)}."""
    keys = set().union(*[p.keys() for p in params_by_group.values()])
    return {k: np.array([params_by_group[str(g)][k] for g in group_idx], dtype=float)
            for k in keys}


POLICY_REGISTRY = {
    "EOQ": EOQPolicy,
    "(s,S)": SsPolicy,
    "Newsvendor": NewsvendorPolicy,
}
