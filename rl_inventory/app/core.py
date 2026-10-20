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

# Màu nét vẽ plotly: tông trung bình, đọc được trên cả theme sáng và tối.
# TEAL (tên cũ) = màu nhấn của IPPO; ORANGE = chính sách đang so / cảnh báo; GREY = baseline; INK = đường mốc.
TEAL, ORANGE, GREY, INK = "#4361F5", "#EF8A3C", "#AEB6C8", "#8A93A8"
SHADE, ORANGE_SHADE = "rgba(138,147,168,.16)", "rgba(239,138,60,.14)"
# Màu cho HTML: biến CSS định nghĩa trong CSS bên dưới, tự đổi theo theme sáng/tối
MUTED, BORDER, TEAL_SOFT, ORANGE_SOFT = "var(--muted)", "var(--line)", "var(--accent-soft)", "var(--warn-soft)"
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


SOC_NGAY_M5 = 1650   # ngày bắt đầu sốc mặc định (--shock_start) của scripts/regime_analysis.py
SOC_CUA_SO = 28      # số ngày trước / sau sốc dùng để so sánh


@st.cache_data(show_spinner=False, max_entries=64)
def shock_rollout(policy: str, start: int, length: int, factor: float, kho=None, nganh=None) -> dict:
    """Chạy lại `policy` (deterministic, toàn miền test, hạt giống meta) với nhu cầu các ngày
    [start, start + length) nhân `factor` trên các cặp thuộc `kho` / `nganh` (None = toàn
    mạng). Chỉ suy luận. factor = 1 cho đúng lượt không sốc. Trả về mảng theo ngày, cộng
    toàn mạng: demand, stockout, order, overflow, cost (lưu kho + thiếu + đặt + tràn)."""
    meta, arr = load_demo()
    P = pairs(meta)
    m = np.ones(len(P), bool)
    if kho:
        m &= P["Kho"].isin(kho).to_numpy()
    if nganh:
        m &= P["Ngành"].isin(nganh).to_numpy()
    dem = arr["demand"].copy()
    dem[start:start + length] *= np.where(m, factor, 1.0).reshape(dem.shape[1:]).astype(dem.dtype)
    env = make_env(meta, {**arr, "demand": dem})
    act = load_policies()[policy]
    obs, _ = env.reset(seed=meta["seed"])
    rec = []
    while True:
        obs, _, te, tr, i = env.step(act(obs, env))
        rec.append((i["demand"], i["stockout"], i["order_qty"], i["overflow"],
                    i["cost_holding"] + i["cost_stockout"] + i["cost_ordering"] + i["cost_overflow"]))
        if te or tr:
            break
    return dict(zip(("demand", "stockout", "order", "overflow", "cost"), np.array(rec).T))


def shock_metrics(s: dict, b: dict, start: int, length: int, muc_dv: float, pad: int = SOC_CUA_SO) -> dict:
    """Chỉ số kịch bản sốc của một chính sách; s, b = mảng theo ngày có sốc / không sốc.
    Fill rate `pad` ngày trước, trong, `pad` ngày sau sốc (cắt ở mép miền test); thiếu hàng
    và chi phí tăng thêm trên [start, end + pad); ve_lai = số ngày từ lúc sốc bắt đầu tới
    khi fill rate trượt 7 ngày về lại >= muc_dv sau lần đầu xuống dưới (None: không xuống
    dưới mục tiêu, inf: chưa về lại trong cửa sổ)."""
    end = start + length
    hi = min(end + pad, len(s["demand"]))

    def fr(a, z):
        d = s["demand"][a:z].sum()
        return 1 - s["stockout"][a:z].sum() / d if d > 0 else float("nan")

    roll = np.array([fr(max(k - 6, 0), k + 1) for k in range(start, hi)])
    duoi = np.flatnonzero(roll < muc_dv)
    lai = np.flatnonzero(roll[duoi[0]:] >= muc_dv) if len(duoi) else []
    w = slice(start, hi)
    return {"truoc": fr(max(start - pad, 0), start), "trong": fr(start, end), "sau": fr(end, hi),
            "thieu_them": float(s["stockout"][w].sum() - b["stockout"][w].sum()),
            "chi_phi_them": float(s["cost"][w].sum() - b["cost"][w].sum()),
            "ve_lai": None if not len(duoi) else int(duoi[0] + lai[0]) if len(lai) else float("inf")}


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
# Token màu theo hướng thiết kế demo gợi ý sản phẩm; light-dark() đọc color-scheme mà
# Streamlit đặt trên .stApp theo theme đang chọn, nên mọi khối HTML tự đổi sáng/tối.
CSS = f"""
<style>
.stApp {{
  --bg: #F5F6FB; --surface: #FFFFFF; --surface-2: #EEF1F8;
  --ink: #11141F; --muted: #5C6378; --line: #E2E5EE;
  --accent: #3150F5; --accent-soft: #E7ECFF; --bar: #CCD2DF;
  --good: #0F9A4C; --good-soft: #DEF5E8;
  --warn: #B87300; --warn-soft: #FFF1D3;
  --bad: #E0434A; --bad-soft: #FFE5E5;
  --shadow: 0 1px 2px rgba(17,20,31,.04), 0 8px 24px -12px rgba(17,20,31,.12);
}}
@supports (color: light-dark(#000, #fff)) {{
.stApp {{
  --bg: light-dark(#F5F6FB, #0C0F17); --surface: light-dark(#FFFFFF, #141927); --surface-2: light-dark(#EEF1F8, #1B2133);
  --ink: light-dark(#11141F, #EEF0F8); --muted: light-dark(#5C6378, #9BA3B9); --line: light-dark(#E2E5EE, #252C41);
  --accent: light-dark(#3150F5, #7D93FF); --accent-soft: light-dark(#E7ECFF, #1D264D); --bar: light-dark(#CCD2DF, #374059);
  --good: light-dark(#0F9A4C, #3FD07F); --good-soft: light-dark(#DEF5E8, #10301E);
  --warn: light-dark(#B87300, #F2B543); --warn-soft: light-dark(#FFF1D3, #34270B);
  --bad: light-dark(#E0434A, #FF6E73); --bad-soft: light-dark(#FFE5E5, #3B191B);
  --shadow: 0 1px 2px light-dark(rgba(17,20,31,.04), rgba(0,0,0,.3)), 0 8px 24px -12px light-dark(rgba(17,20,31,.12), rgba(0,0,0,.6));
}}
}}
.stApp, .stApp button, .stApp input {{ font-family: "Plus Jakarta Sans", "Segoe UI", system-ui, sans-serif; }}
[data-testid="stMainBlockContainer"] {{ padding-top: 5.5rem; max-width: 1240px; }}
/* thanh trên cùng: nền thẻ, đường kẻ dưới, mục đang mở gạch chân màu nhấn */
[data-testid="stHeader"] {{ background: var(--surface); border-bottom: 1px solid var(--line); }}
[data-testid="stTopNavLink"] {{ background: transparent !important; border-radius: 0; padding: 0 4px; margin: 0 8px;
  border-bottom: 2px solid transparent; }}
[data-testid="stTopNavLink"] p {{ color: var(--muted); font-weight: 600; font-size: 14.5px; }}
[data-testid="stTopNavLink"][aria-current="page"] {{ border-bottom-color: var(--accent); }}
[data-testid="stTopNavLink"][aria-current="page"] p, [data-testid="stTopNavLink"]:hover p {{ color: var(--accent); }}
/* tiêu đề trang */
.eyebrow {{ color: var(--accent); font-size: 13px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
  margin-bottom: .1rem; }}
.stApp h1 {{ font-size: 44px; font-weight: 800; letter-spacing: -.02em; line-height: 1.12; padding: 0 0 .35rem; }}
.mota {{ font-size: 17px; color: var(--muted); max-width: 62ch; margin: 0 0 1.5rem; line-height: 1.55; }}
.stApp h2, .stApp h3 {{ font-size: 19px; font-weight: 700; letter-spacing: -.01em; }}
.stApp [data-testid="stCaptionContainer"] {{ color: var(--muted); }}
/* thẻ: mọi container có key bắt đầu bằng the_ */
[data-testid="stForm"], [class*="st-key-the_"] {{ background: var(--surface); border: 1px solid var(--line) !important; border-radius: 16px !important;
  box-shadow: var(--shadow); padding: 20px 22px !important; margin-top: .6rem; }}
[data-testid="stAlert"] {{ background: var(--accent-soft); border: none; border-radius: 12px; }}
[data-testid="stAlert"] * {{ color: var(--ink); }}
[data-testid="stColumn"] > div, [data-testid="stColumn"] [data-testid="stElementContainer"]:has(.kpi),
[data-testid="stColumn"] [data-testid="stMarkdown"]:has(.kpi) {{ height: 100%; }}
.kpi {{ background: var(--surface); border: 1px solid var(--line); border-radius: 16px; padding: 18px 20px; height: 100%;
  box-shadow: var(--shadow); }}
.kpi .lab {{ color: var(--muted); font-size: 13px; font-weight: 600; }}
.kpi .num {{ font-size: 32px; font-weight: 800; letter-spacing: -.02em; margin: 6px 0 8px; color: var(--ink);
  font-variant-numeric: tabular-nums; }}
.kpi .note {{ font-size: 13px; color: var(--muted); }}
.kpi .warn, .kpi .good, .chip {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-weight: 700;
  font-size: 12.5px; }}
.kpi .warn, .chip.warn {{ background: var(--warn-soft); color: var(--warn); }}
.kpi .good, .chip.good {{ background: var(--good-soft); color: var(--good); }}
.chip {{ background: var(--surface-2); color: var(--muted); white-space: nowrap; }}
.chip.bad {{ background: var(--bad-soft); color: var(--bad); }}
.box {{ background: var(--surface); border: 1px solid var(--line); border-radius: 16px; padding: 14px 16px;
  text-align: center; box-shadow: var(--shadow); }}
/* đầu mục đánh số như demo gợi ý: vòng tròn số + tiêu đề + ghi chú bên phải */
.muc {{ display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; margin: .25rem 0 .9rem; }}
.muc .so {{ width: 26px; height: 26px; border-radius: 50%; background: var(--accent); color: #fff; font-weight: 700;
  font-size: 14px; display: inline-flex; align-items: center; justify-content: center; transform: translateY(3px); }}
.muc .t {{ font-size: 21px; font-weight: 800; letter-spacing: -.01em; }}
.muc .g {{ margin-left: auto; color: var(--muted); font-size: 13.5px; }}
/* thẻ hỏi–đáp với thanh ngang, như trang Kết quả đánh giá của demo gợi ý */
.hoi {{ font-size: 19px; font-weight: 800; letter-spacing: -.01em; margin-bottom: 4px; }}
.dap {{ color: var(--muted); font-size: 14.5px; line-height: 1.55; margin-bottom: 16px; }}
.dap b {{ color: var(--ink); }}
.thanh {{ display: grid; gap: 12px; }}
.thanh .r {{ display: grid; grid-template-columns: minmax(120px, 170px) minmax(0, 1fr) auto; gap: 6px 14px; align-items: center; }}
.thanh .n {{ font-size: 14.5px; font-weight: 600; line-height: 1.25; }}
.thanh .n small {{ display: block; color: var(--muted); font-weight: 500; font-size: 12px; margin-top: 3px; }}
.thanh .n small .chip {{ font-size: 11.5px; padding: 2px 8px; }}
.thanh .tr {{ position: relative; height: 12px; border-radius: 99px; background: var(--surface-2); }}
.thanh .tr i {{ position: absolute; inset: 0 auto 0 0; border-radius: 99px; background: var(--bar); }}
.thanh .tr u {{ position: absolute; top: -5px; bottom: -5px; width: 2px; background: var(--bad); }}
.thanh .r.me .n {{ color: var(--accent); font-weight: 800; }}
.thanh .r.me .tr i {{ background: var(--accent); }}
.thanh .v {{ text-align: right; font-weight: 800; font-variant-numeric: tabular-nums; white-space: nowrap; min-width: 64px; }}
.thanh .c {{ grid-column: 2 / 4; margin-top: -4px; }}
.thanh .nhom {{ font-size: 12.5px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: .05em;
  margin-top: 6px; }}
.ghi {{ color: var(--muted); font-size: 12.5px; margin-top: 14px; }}
.stk-ds {{ display: grid; gap: 12px; }}
.stk-r {{ display: grid; grid-template-columns: minmax(110px, 170px) minmax(0, 1fr) 86px; gap: 14px; align-items: center; }}
.stk-r .n {{ font-size: 14px; font-weight: 600; }}
.stk-r.me .n {{ color: var(--accent); font-weight: 800; }}
.stk-r .v {{ text-align: right; font-weight: 800; font-variant-numeric: tabular-nums; }}
.stk {{ display: flex; height: 16px; border-radius: 6px; overflow: hidden; }}
.stk span {{ height: 100%; }}
.chu-mau {{ display: flex; gap: 16px; flex-wrap: wrap; color: var(--muted); font-size: 12.5px; margin-bottom: 14px; }}
.chu-mau i {{ display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }}
table.bang {{ width: 100%; border-collapse: collapse; font-size: 14px; font-variant-numeric: tabular-nums; }}
table.bang th, table.bang td {{ padding: 11px 8px; border-bottom: 1px solid var(--line); text-align: right; }}
table.bang thead th {{ color: var(--muted); font-size: 12.5px; font-weight: 600; }}
table.bang th:first-child {{ text-align: left; }}
table.bang tbody th {{ font-weight: 700; }}
.chu-giai {{ display: flex; gap: 16px; flex-wrap: wrap; color: var(--muted); font-size: 12.5px; margin-bottom: 12px; }}
.chu-giai span::before {{ content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 50%;
  margin-right: 6px; background: var(--bar); vertical-align: -1px; }}
.chu-giai span.me::before {{ background: var(--accent); }}
.chu-giai span.tg::before {{ width: 2px; border-radius: 0; height: 12px; background: var(--bad); }}
</style>
"""


def kpi(label: str, value: str, note: str = "", tone: str = "") -> str:
    return (f'<div class="kpi"><div class="lab">{label}</div><div class="num">{value}</div>'
            f'<div class="note {tone}">{note}</div></div>')


def tieu_de(nhan: str, tieu_de: str, mo_ta: str = None):
    """Nhãn nhỏ (eyebrow) + tiêu đề trang + mô tả một dòng."""
    st.markdown(f'<div class="eyebrow">{nhan}</div>', unsafe_allow_html=True)
    st.title(tieu_de)
    if mo_ta:
        st.markdown(f'<div class="mota">{mo_ta}</div>', unsafe_allow_html=True)


def muc(so, tieu_de: str, ghi_chu: str = ""):
    """Đầu mục trong thẻ: vòng tròn số + tiêu đề, ghi chú mờ bên phải (như demo gợi ý sản phẩm)."""
    st.markdown(f'<div class="muc"><span class="so">{so}</span><span class="t">{tieu_de}</span>'
                f'<span class="g">{ghi_chu}</span></div>', unsafe_allow_html=True)


TEN_CS = {"IPPO": "mô hình của đề tài", "EOQ": "đặt một lượng cố định", "(s,S)": "đặt bù khi tồn kho thấp",
          "Newsvendor": "đặt theo nhu cầu dự kiến"}


def ten_cs(p: str) -> tuple:
    """Tên chính sách + một dòng mô tả bằng lời thường (cho nhãn biểu đồ)."""
    goc = p.replace(" theo nhóm", "")
    return p, TEN_CS.get(goc, "") + (", chỉnh riêng theo nhóm hàng" if p.endswith("theo nhóm") else "")


def thanh(dong: list, lo: float = 0.0, hi: float = None, moc: float = None) -> str:
    """Thanh ngang kiểu demo gợi ý. dong: dict(ten, phu, v, nhan, me, chip) hoặc dict(nhom=...) làm dòng tiêu đề nhóm.
    Thanh tính từ lo đến hi; moc: vạch đỏ (vd mục tiêu mức phục vụ)."""
    vs = [d["v"] for d in dong if "v" in d]
    hi = max(vs) if hi is None else hi
    x = lambda v: max(0.0, min(100.0, (v - lo) / (hi - lo) * 100))  # noqa: E731
    out = []
    for d in dong:
        if "nhom" in d:
            out.append(f'<div class="nhom">{d["nhom"]}</div>')
            continue
        vach = f'<u style="left:{x(moc):.1f}%"></u>' if moc is not None else ""
        out.append(f'<div class="r{" me" if d.get("me") else ""}"><div class="n">{d["ten"]}'
                   + (f'<small>{d["phu"]}</small>' if d.get("phu") else "")
                   + f'</div><div class="tr"><i style="width:{x(d["v"]):.1f}%"></i>{vach}</div>'
                   f'<div class="v">{d["nhan"]}</div>'
                   + (f'<div class="c">{d["chip"]}</div>' if d.get("chip") else "") + "</div>")
    return '<div class="thanh">' + "".join(out) + "</div>"


def the_hoi(key: str, hoi: str, dap: str, than: str, ghi: str = ""):
    """Một thẻ hỏi–đáp: câu hỏi, câu trả lời ngắn từ số liệu, phần thân (thường là thanh())."""
    with st.container(key=f"the_{key}"):
        st.markdown(f'<div class="hoi">{hoi}</div><div class="dap">{dap}</div>{than}'
                    + (f'<div class="ghi">{ghi}</div>' if ghi else ""), unsafe_allow_html=True)


def chip(text: str, tone: str = "") -> str:
    return f'<span class="chip {tone}">{text}</span>'


def nguon(*parts):
    """Một dòng nguồn số liệu ở cuối trang."""
    st.caption("Nguồn số liệu: " + " · ".join(parts))


def num(x, d: int = 0) -> str:
    """Định dạng số kiểu Việt Nam: 1.234.567,8"""
    return f"{x:,.{d}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def pct(x, d: int = 1) -> str:
    return num(100 * x, d) + "%"


def style_fig(fig, height: int = 380):
    fig.update_layout(height=height, margin=dict(l=70, r=20, t=50, b=50), separators=",.",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Plus Jakarta Sans, Segoe UI, sans-serif", size=13),
                      hoverlabel=dict(font_family="Plus Jakarta Sans, Segoe UI, sans-serif"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_xaxes(showgrid=False, zerolinecolor=SHADE, automargin=True)
    fig.update_yaxes(gridcolor=SHADE, zerolinecolor=SHADE, automargin=True)
    return fig


def chart(fig, height: int = 380):
    # theme="streamlit": chữ, trục, hover theo theme sáng/tối đang chọn; màu nét giữ như đã đặt
    st.plotly_chart(style_fig(fig, height), theme="streamlit", config={"displayModeBar": False})


def policy_color(name: str) -> str:
    return TEAL if name == "IPPO" else ORANGE


def no_data(what: str):
    st.info(f"Chưa có dữ liệu: {what}")
