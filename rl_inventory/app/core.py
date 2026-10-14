"""
app/core.py
===========
Lõi của demo: nạp dữ liệu rút gọn (app/demo_data/), mô hình IPPO và 3 baseline,
chạy mô phỏng INFERENCE trên miền test; đọc số liệu báo cáo từ results/.
Không có đường code nào gọi huấn luyện.
"""
import sys
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP = Path(__file__).resolve().parent
ROOT = APP.parent
DEMO = APP / "demo_data"
RESULTS = ROOT / "results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.inventory_env import MultiWarehouseInventoryEnv
from scripts.common import make_policies

TEAL, ORANGE, GREY, INK, PAPER = "#0F6E63", "#C2571A", "#7E8C86", "#16201C", "#F3F4F1"
CATEGORIES = ["FOODS", "HOBBIES", "HOUSEHOLD"]
COST_PARTS = [("holding_cost", "Lưu kho"), ("stockout_cost", "Thiếu hàng"),
              ("ordering_cost", "Đặt hàng"), ("overflow_cost", "Tràn kho")]
WEEKDAYS = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ nhật"]
# 20 chiều lịch, đúng bố cục scripts/data_preprocessing.generate_calendar_features
CALENDAR_NAMES = (["SNAP CA (0/1)", "SNAP TX (0/1)", "SNAP WI (0/1)", "Không có sự kiện (0/1)",
                   "Sự kiện thể thao (0/1)", "Sự kiện văn hóa (0/1)", "Ngày lễ quốc gia (0/1)",
                   "Ngày lễ tôn giáo (0/1)"] + [f"Tháng {m} (0/1)" for m in range(1, 13)])
ROLE = {"inventory": "Thông tin trạng thái", "demand_history": "Thông tin trạng thái",
        "pipeline": "Thông tin trạng thái", "fill_rate": "Thông tin trạng thái",
        "scale": "Thông tin trạng thái", "warehouse": "Tín hiệu cấp kho",
        "price": "Tín hiệu giá", "event_lookahead": "Tín hiệu lịch",
        "calendar": "Tín hiệu lịch"}


# --------------------------------------------------------------------------- #
# Dữ liệu demo, mô hình, mô phỏng
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_demo():
    """(meta, arrays) từ app/demo_data/ - do scripts/build_demo_data.py tạo."""
    meta = json.loads((DEMO / "meta.json").read_text(encoding="utf-8"))
    with np.load(DEMO / "demo.npz") as z:
        return meta, {k: z[k] for k in z.files}


def make_env(meta, arr) -> MultiWarehouseInventoryEnv:
    """Dựng lại môi trường gốc từ dữ liệu rút gọn.

    Env tra dữ liệu theo CHỈ SỐ NGÀY TUYỆT ĐỐI và tự ước lượng mean_demand từ
    demand[:split_day], nên ta dựng mảng đủ dài: miền test đặt đúng vị trí, miền
    train là các dòng hằng số = mean_demand gốc. Để float64 thì trung bình cộng
    lại ĐÚNG TỪNG BIT -> sức chứa, bảng đặt hàng, thang chuẩn hóa obs giống hệt
    lúc đánh giá (tests/test_demo.py kiểm tra)."""
    e = meta["env"]
    W, S, T, v0 = e["n_warehouses"], e["n_skus"], meta["n_days_total"], meta["val_day"]
    demand = np.zeros((T, W, S))
    demand[:e["split_day"]] = arr["mean_demand"].reshape(W, S)
    demand[v0:] = arr["demand"]
    cal = np.zeros((T, arr["calendar"].shape[1]), np.float32)
    cal[v0:] = arr["calendar"]
    price = np.zeros((T, W, S), np.float32)
    price[v0:] = arr["price"]
    return MultiWarehouseInventoryEnv(e, demand, cal, arr["price_per_pair"], price, mode="test")


@st.cache_resource
def base_env() -> MultiWarehouseInventoryEnv:
    """Một env chỉ để tra hằng số (sức chứa, bảng đặt hàng, nhóm obs) - không step."""
    return make_env(*load_demo())


@st.cache_resource
def load_policies():
    """Tên -> action_fn(obs, env): IPPO (deterministic) + 3 baseline đã tinh chỉnh."""
    meta, _ = load_demo()
    cfg = {"env": meta["env"], "ppo": meta["ppo"],
           "paths": {"results_dir": str(DEMO.relative_to(ROOT))}}
    return make_policies(cfg, base_env(), str((DEMO / "best_model.pth").relative_to(ROOT)),
                         include_iso=False, include_group=False)


@st.cache_data(show_spinner="Đang chạy mô phỏng tới ngày đã chọn…", max_entries=64)
def state_at(policy: str, day: int) -> dict:
    """Chạy mô phỏng từ đầu miền test tới ĐẦU ngày `day` (0 = ngày đầu tiên),
    trả về trạng thái lúc ra quyết định và hành động của chính sách."""
    meta, arr = load_demo()
    env = make_env(meta, arr)
    act = load_policies()[policy]
    obs, _ = env.reset(seed=meta["seed"])
    for _ in range(day):
        obs, *_ = env.step(act(obs, env))
    a = np.asarray(act(obs, env))
    return {"obs": obs, "action": a, "qty": env.action_to_qty(a),
            "inv": env.inventory.copy(), "pipe": env.pipeline_orders.sum(axis=0),
            "dem7": env.demand_history.mean(axis=0)}


def rollout(policy: str) -> dict:
    """Kết quả rollout đã chạy sẵn trên toàn miền test (mảng theo ngày)."""
    _, arr = load_demo()
    pre = f"{policy}|"
    return {k[len(pre):]: v for k, v in arr.items() if k.startswith(pre)}


def test_dates(meta) -> list:
    start = date.fromisoformat(meta["test_start"])
    return [start + timedelta(days=i) for i in range(meta["n_days_total"] - meta["val_day"])]


def pairs(meta) -> pd.DataFrame:
    """Một dòng cho mỗi cặp (kho, SKU), đúng thứ tự cặp trong env."""
    S = len(meta["skus"])
    return pd.DataFrame([{"Kho": f"K{w + 1:02d}", "Cửa hàng": st_id, "SKU": sku,
                          "Ngành": sku.split("_")[0], "w": w, "s": s, "pair": w * S + s}
                         for w, st_id in enumerate(meta["stores"])
                         for s, sku in enumerate(meta["skus"])])


def obs_labels(env, meta) -> list:
    """(vai trò, nhóm, tên) cho từng chiều obs, theo env.obs_groups."""
    L, LT, cover = env.lookback, env.lead_time_max, env.obs_cover_days
    day0 = date.fromisoformat(meta["test_start"]) - timedelta(days=meta["val_day"])
    names = {
        "inventory": [f"Tồn kho / (cầu TB × {cover:g} ngày)",
                      f"Vị thế tồn kho (tồn + hàng đang về) / (cầu TB × {cover:g} ngày)"],
        "demand_history": [f"Nhu cầu ngày t−{L - i}: log1p(d) / log1p(10·cầu TB + 1)"
                           for i in range(L)],
        "pipeline": [f"Hàng về {'hôm nay' if k == 0 else f'sau {k} ngày'} / (cầu TB × {cover:g} ngày)"
                     for k in range(LT)],
        "fill_rate": [f"Fill rate {env.cua_so_dv} ngày gần nhất"],
        "scale": ["Quy mô cầu: log1p(cầu TB) / log1p(150)"],
        "warehouse": ["util: tổng tồn kho của kho / sức chứa kho",
                      f"share: tồn của cặp / tổng tồn của kho × {env.n_skus} / 3"],
        "price": ["Mức giảm giá: clip(1 − giá hôm nay / giá TB miền train, 0, 1)"],
        "event_lookahead": ["Số ngày tới sự kiện kế tiếp / 30", "Có sự kiện trong 7 ngày tới"],
        "calendar": ([f"Ngày trong tuần: {WEEKDAYS[(day0 + timedelta(days=k)).weekday()]} "
                      f"(ô {k} = chỉ số ngày mod 7)" for k in range(7)] + CALENDAR_NAMES),
    }
    out = [(ROLE[g], g, names[g][i]) for g, idx in env.obs_groups.items()
           for i in range(len(idx))]
    if len(out) != env.obs_per_pair:
        raise RuntimeError(f"Nhãn obs ({len(out)}) lệch số chiều env ({env.obs_per_pair})")
    return out


# --------------------------------------------------------------------------- #
# Số liệu báo cáo (results/) - không hardcode, thiếu file thì trả None
# --------------------------------------------------------------------------- #
def read_results(name: str):
    p = RESULTS / name
    if not p.exists():
        return None
    return pd.read_csv(p) if p.suffix == ".csv" else json.loads(p.read_text(encoding="utf-8"))


def summary(meta):
    """results/summary_<tag>.json -> (DataFrame theo chính sách, stat, dict gốc) hoặc None."""
    s = read_results(f"summary_{meta['results_tag']}.json")
    return None if s is None else (pd.DataFrame(s["summary"]).set_index("policy"), s["stat"], s)


def shock_results():
    """Kết quả kịch bản sốc cầu (scripts/regime_analysis.py) nếu đã chạy."""
    for p in sorted(RESULTS.glob("regime_analysis*.json")):
        b2 = json.loads(p.read_text(encoding="utf-8")).get("B2_shock")
        if b2 and b2.get("results"):
            return p.name, b2
    return None


# --------------------------------------------------------------------------- #
# Giao diện dùng chung
# --------------------------------------------------------------------------- #
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
.stApp *:not([data-testid="stIconMaterial"]) {{ font-family: 'IBM Plex Sans', sans-serif; }}
.stApp {{ background: {PAPER}; color: {INK}; }}
[data-testid="stSidebar"] {{ background: {INK}; }}
[data-testid="stSidebar"] * {{ color: {PAPER} !important; }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ background: {TEAL}; }}
.stApp .num, .stApp code {{ font-family: 'IBM Plex Mono', monospace; }}
.kpi {{ background: #fff; border: 1px solid #dfe3dd; border-radius: 6px; padding: 14px 16px; height: 100%; }}
.kpi .lab {{ color: {GREY}; font-size: .85rem; }}
.kpi .num {{ font-size: 1.6rem; font-weight: 500; margin: 4px 0; color: {INK}; }}
.kpi .note {{ font-size: .85rem; color: {GREY}; }}
.kpi .warn {{ color: {ORANGE}; }} .kpi .good {{ color: {TEAL}; }}
.box {{ background: #fff; border: 1px solid #dfe3dd; border-radius: 6px; padding: 10px 14px; text-align: center; }}
</style>
"""


def kpi(label: str, value: str, note: str = "", tone: str = "") -> str:
    return (f'<div class="kpi"><div class="lab">{label}</div><div class="num">{value}</div>'
            f'<div class="note {tone}">{note}</div></div>')


def num(x, d: int = 0) -> str:
    """Định dạng số kiểu Việt Nam: 1.234.567,8"""
    return f"{x:,.{d}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def pct(x, d: int = 1) -> str:
    return num(100 * x, d) + "%"


def style_fig(fig, height: int = 380):
    fig.update_layout(height=height, margin=dict(l=70, r=20, t=50, b=50),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff",
                      font=dict(family="IBM Plex Sans, sans-serif", color=INK, size=13),
                      hoverlabel=dict(font_family="IBM Plex Mono, monospace"),
                      legend=dict(orientation="h", y=1.08, x=0))
    fig.update_xaxes(gridcolor="#e6e9e4", zerolinecolor="#e6e9e4", automargin=True)
    fig.update_yaxes(gridcolor="#e6e9e4", zerolinecolor="#e6e9e4", automargin=True)
    return fig


def policy_color(name: str) -> str:
    return TEAL if name == "IPPO" else ORANGE


def no_data(what: str):
    st.info(f"Chưa có dữ liệu: {what}")
