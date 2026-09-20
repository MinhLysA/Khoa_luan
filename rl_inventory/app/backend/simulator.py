"""
app/backend/simulator.py
=========================
Bọc MultiWarehouseInventoryEnv cho demo app: tạo env từ dữ liệu thật hoặc từ
kịch bản giả định (nhập tay), chạy 1 episode với một hàm chính sách, hoặc chạy
SONG SONG nhiều chính sách trên CÙNG một chuỗi cầu (cùng seed) để so sánh.

Không sửa env/inventory_env.py - chỉ gọi lại đúng API đã có.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.inventory_env import MultiWarehouseInventoryEnv

DATA_DIR = ROOT / "data" / "processed"

ActionFn = Callable[[np.ndarray, MultiWarehouseInventoryEnv], np.ndarray]


# --------------------------------------------------------------------------- #
def load_env_bundle():
    """Đọc dữ liệu đã tiền xử lý. Trả về None cho phần nào chưa có."""
    def _load(name):
        p = DATA_DIR / name
        return np.load(p) if p.exists() else None

    demand = _load("demand_data.npy")
    calf = _load("calendar_features.npy")
    if calf is not None and calf.size == 0:
        calf = None
    price = _load("price_per_pair.npy")
    price_series = _load("price_series.npy")

    import json
    meta_p = DATA_DIR / "env_config.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}

    return {"demand": demand, "calendar": calf, "price": price,
            "price_series": price_series, "meta": meta}


def make_env(cfg: dict, bundle: dict, mode: str = "test",
            env_overrides: Optional[dict] = None) -> MultiWarehouseInventoryEnv:
    """Env DÙNG DỮ LIỆU THẬT (300 cặp), giống hệt cách scripts/evaluate.py tạo."""
    e = dict(cfg["env"])
    e.update(env_overrides or {})
    return MultiWarehouseInventoryEnv(
        config=e, demand_data=bundle["demand"], calendar_features=bundle["calendar"],
        price_per_pair=bundle["price"], price_series=bundle["price_series"], mode=mode)


def make_scenario_env(cfg: dict, mean_demand: float, std_demand: Optional[float] = None,
                      current_inventory: Optional[float] = None,
                      env_overrides: Optional[dict] = None) -> MultiWarehouseInventoryEnv:
    """Env 1 kho - 1 SKU cho kịch bản GIẢ ĐỊNH (người dùng tự nhập tồn kho/dự
    báo cầu), dùng cho trang 'Đề xuất đặt hàng'. Không gắn với ngày tháng
    lịch sử thật - chỉ để TÍNH TOÁN một lần, không dùng để train."""
    e = dict(cfg["env"])
    e.update(env_overrides or {})
    e["n_warehouses"] = 1
    e["n_skus"] = 1
    env = MultiWarehouseInventoryEnv(config=e, demand_data=None, mode="train")
    env.reset(seed=0)
    env.mean_demand[:] = max(float(mean_demand), env.min_mean_demand)
    if std_demand is not None:
        env.std_demand[:] = max(float(std_demand), 1e-3)
    if current_inventory is not None:
        env.inventory[:] = float(current_inventory)
    return env


def current_obs(env: MultiWarehouseInventoryEnv) -> np.ndarray:
    """Tính lại quan sát cho trạng thái HIỆN TẠI của env (sau khi ghi đè tay
    inventory/mean_demand). Dùng hàm nội bộ _get_observation() đã có sẵn."""
    return env._get_observation()


# --------------------------------------------------------------------------- #
def run_episode(env: MultiWarehouseInventoryEnv, action_fn: ActionFn, seed: int,
                max_days: Optional[int] = None) -> dict:
    """Chạy 1 episode, trả về số liệu cả theo NGÀY (tổng hệ thống) lẫn MA TRẬN
    (kho x SKU) để vẽ heatmap."""
    obs, _ = env.reset(seed=seed)
    n_days = min(max_days or env.episode_length, env.episode_length)

    daily_rows = []
    inv_mat, dem_mat, stockout_mat, util_mat = [], [], [], []

    for t in range(n_days):
        action = action_fn(obs, env)
        obs, _, terminated, truncated, info = env.step(action)

        daily_rows.append({
            "Ngày": t, "Cầu": info["demand"], "Bán được": info["sold"],
            "Thiếu hàng": info["stockout"], "Đặt hàng": info["order_qty"],
            "Số lần đặt": info["n_orders"], "Tràn kho": info["overflow"],
            "Tồn kho": info["inventory"], "Fill rate": info["fill_rate_mean"],
            "Chi phí lưu kho": info["cost_holding"],
            "Chi phí thiếu hàng": info["cost_stockout"],
            "Chi phí đặt hàng": info["cost_ordering"],
            "Chi phí tràn kho": info["cost_overflow"],
            "Chi phí ngày": (info["cost_holding"] + info["cost_stockout"]
                            + info["cost_ordering"] + info["cost_overflow"]),
        })
        inv_mat.append(info["inventory_pairs"].reshape(env.n_warehouses, env.n_skus))
        dem_mat.append(info["demand_pairs"].reshape(env.n_warehouses, env.n_skus))
        stockout_mat.append(info["stockout_pairs"].reshape(env.n_warehouses, env.n_skus))
        util_mat.append(info["util_wh"])

        if terminated or truncated:
            break

    daily = pd.DataFrame(daily_rows)
    return {
        "daily": daily,
        "inventory_matrix": np.stack(inv_mat),
        "demand_matrix": np.stack(dem_mat),
        "stockout_matrix": np.stack(stockout_mat),
        "util_wh": np.stack(util_mat),
    }


def run_parallel(env_factory: Callable[[], MultiWarehouseInventoryEnv],
                 policies: Dict[str, ActionFn], seed: int,
                 max_days: Optional[int] = None) -> Dict[str, dict]:
    """Chạy CÙNG một seed (=CÙNG một chuỗi cầu/lead time ngẫu nhiên) cho nhiều
    chính sách, để so sánh công bằng trên đúng 1 kịch bản."""
    return {name: run_episode(env_factory(), fn, seed, max_days)
            for name, fn in policies.items()}


def summarize(ket_qua: Dict[str, dict]) -> pd.DataFrame:
    """Tổng hợp kết quả run_parallel() thành 1 bảng: tổng chi phí, fill rate,
    và 4 thành phần chi phí cho từng chính sách. Dùng chung cho trang So sánh
    policy và What-if để không lặp code."""
    rows = []
    for ten, kq in ket_qua.items():
        d = kq["daily"]
        fr = 1 - d["Thiếu hàng"].sum() / max(d["Cầu"].sum(), 1e-6)
        rows.append({
            "Chính sách": ten, "Tổng chi phí": d["Chi phí ngày"].sum(),
            "Fill rate": fr * 100, "Lưu kho": d["Chi phí lưu kho"].sum(),
            "Thiếu hàng": d["Chi phí thiếu hàng"].sum(),
            "Đặt hàng": d["Chi phí đặt hàng"].sum(),
            "Tràn kho": d["Chi phí tràn kho"].sum(),
        })
    return pd.DataFrame(rows).set_index("Chính sách")


def apply_demand_shock(env: MultiWarehouseInventoryEnv, shock_factor: float):
    """[What-if] Nhân toàn bộ demand_data ĐÃ NẠP của env với 1 hệ số (vd 1.5 =
    tăng cầu 50%). Sửa trực tiếp trên mảng demand_data của chính env này, KHÔNG
    dùng chung với mảng gốc của cfg (gọi sau khi make_env, trước reset)."""
    if env.demand_data is not None:
        env.demand_data = env.demand_data * float(shock_factor)
