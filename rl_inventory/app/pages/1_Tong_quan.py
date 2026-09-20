"""
app/pages/1_Tong_quan.py
=========================
Trang 1 - Tổng quan mạng lưới kho: chọn ngày mô phỏng (slider), xem heatmap
tồn kho kho x SKU (đỏ = thiếu hàng, vàng = tồn dư), và 3 KPI hệ thống.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
APP_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT, APP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend import simulator, policy_runner  # noqa: E402

st.set_page_config(page_title="Tổng quan mạng lưới kho", page_icon="📊", layout="wide")
st.title("📊 Tổng quan mạng lưới kho")

with open(ROOT / "config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

bundle = simulator.load_env_bundle()
if bundle["demand"] is None:
    st.warning("Chưa có dữ liệu đã tiền xử lý. Chạy `python run.py data` trước.")
    st.stop()

meta = bundle["meta"]
ten_kho = meta.get("stores", None)
ten_sku = meta.get("top_items", None)

c1, c2, c3 = st.columns(3)
ckpts = sorted((ROOT / "checkpoints").glob("*.pth"))
ten_ckpt = c1.selectbox("Checkpoint IPPO", [p.name for p in ckpts] or ["(chưa có)"])
seed = c2.number_input("Seed", 0, 10**6, 1000)
so_ngay = c3.number_input("Số ngày mô phỏng", 10, 365, 90, step=10)

if st.button("▶️ Chạy mô phỏng", type="primary"):
    env = simulator.make_env(cfg, bundle, mode="test")
    policies = policy_runner.load_all_policies(
        env, str(ROOT / "checkpoints" / ten_ckpt) if ckpts else None,
        cfg["ppo"], ROOT / "results")
    ten_chinh_sach = "IPPO" if "IPPO" in policies else next(iter(policies))
    with st.spinner(f"Đang chạy {ten_chinh_sach} qua {so_ngay} ngày..."):
        ket_qua = simulator.run_episode(env, policies[ten_chinh_sach],
                                        seed=int(seed), max_days=int(so_ngay))
    st.session_state["tq"] = {
        "ket_qua": ket_qua, "chinh_sach": ten_chinh_sach,
        "n_wh": env.n_warehouses, "n_sku": env.n_skus,
        "mean_demand": env.mean_demand.reshape(env.n_warehouses, env.n_skus).copy(),
    }

if "tq" not in st.session_state:
    st.info("Bấm **Chạy mô phỏng** để xem tổng quan.")
    st.stop()

state = st.session_state["tq"]
kq = state["ket_qua"]
daily = kq["daily"]
n_days = len(daily)
n_wh, n_sku = state["n_wh"], state["n_sku"]
ten_kho = ten_kho or [f"Kho {i}" for i in range(n_wh)]
ten_sku = ten_sku or [f"SKU{i}" for i in range(n_sku)]

st.success(f"Đang xem chính sách **{state['chinh_sach']}**, {n_days} ngày đã mô phỏng.")

ngay = st.slider("Ngày mô phỏng", 1, n_days, n_days) - 1

c1, c2, c3 = st.columns(3)
c1.metric("Fill rate (cửa sổ 30 ngày, tại ngày này)", f"{daily['Fill rate'].iloc[ngay]:.1%}")
c2.metric("Tổng chi phí lũy kế", f"{daily['Chi phí ngày'].iloc[:ngay+1].sum():,.0f}")
so_luot_thieu = int((kq['stockout_matrix'][:ngay+1] > 0).sum())
c3.metric("Số lượt thiếu hàng lũy kế (kho,SKU,ngày)", f"{so_luot_thieu:,}")

st.subheader(f"Heatmap tồn kho — ngày {ngay + 1}")
st.caption("🟥 đỏ = ngày này CÓ thiếu hàng tại cặp đó. 🟨 vàng = tồn dư nhiều "
          "(tồn kho > 10 lần cầu trung bình của chính cặp đó). Trắng = bình thường.")

inv_day = kq["inventory_matrix"][ngay]
stockout_day = kq["stockout_matrix"][ngay]
surplus_day = inv_day > 10 * np.maximum(state["mean_demand"], 1e-6)

df_heat = pd.DataFrame(inv_day, index=ten_kho, columns=ten_sku)


def _to_mau(_):
    mau = np.where(stockout_day > 0, "background-color:#f8cecc",
           np.where(surplus_day, "background-color:#fff2cc", ""))
    return pd.DataFrame(mau, index=df_heat.index, columns=df_heat.columns)


st.dataframe(df_heat.style.apply(_to_mau, axis=None).format("{:.0f}"),
            width="stretch", height=min(38 * (n_wh + 1), 460))

st.subheader("Đường tổng hệ thống theo thời gian")
c1, c2 = st.columns(2)
c1.markdown("**Tồn kho vs Cầu (toàn hệ thống)**")
c1.line_chart(daily.iloc[:ngay + 1].set_index("Ngày")[["Tồn kho", "Cầu"]], height=260)
c2.markdown("**Fill rate cửa sổ trượt 30 ngày**")
c2.line_chart(daily.iloc[:ngay + 1].set_index("Ngày")[["Fill rate"]], height=260)

with st.expander("❓ Ví dụ đọc trang này"):
    st.markdown(
        "Mỗi ô trong bảng là tồn kho CUỐI ngày của 1 cặp (kho, SKU). Nếu 1 ô "
        "tô đỏ, nghĩa là NGÀY ĐÓ khách cần hàng nhưng kho không đủ để bán hết "
        "— dấu hiệu cần đặt hàng sớm hơn hoặc đặt nhiều hơn cho cặp đó. Nếu 1 "
        "ô tô vàng, nghĩa là đang giữ nhiều hơn mức cần — vốn bị 'chôn' vào "
        "hàng tồn thay vì dùng cho cặp khác đang thiếu.")
