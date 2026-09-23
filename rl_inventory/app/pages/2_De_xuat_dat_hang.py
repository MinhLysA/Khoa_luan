"""
app/pages/2_De_xuat_dat_hang.py
=================================
Trang 2 (trang chính) - nhập 1 tình huống (kho, SKU, tồn hiện tại, dự báo
cầu), xem IPPO và 3 baseline đề xuất đặt bao nhiêu. Có thể áp dụng 1 đề xuất
và chạy tiếp để xem diễn biến.
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

st.set_page_config(page_title="Đề xuất đặt hàng", page_icon="🧭", layout="wide")
st.title("🧭 Đề xuất đặt hàng cho 1 kho - 1 mặt hàng")

with open(ROOT / "config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
bundle = simulator.load_env_bundle()

nguon = st.radio("Nguồn tình huống", ["Cặp kho-SKU thật (từ dữ liệu M5)",
                                      "Tự nhập kịch bản giả định"], horizontal=True)

c1, c2, c3 = st.columns(3)
if nguon.startswith("Cặp") and bundle["demand"] is not None:
    meta = bundle["meta"]
    ten_kho = meta.get("stores", [f"Kho{i}" for i in range(bundle["demand"].shape[1])])
    ten_sku = meta.get("top_items", [f"SKU{i}" for i in range(bundle["demand"].shape[2])])
    w_idx = c1.selectbox("Kho", range(len(ten_kho)), format_func=lambda i: ten_kho[i])
    s_idx = c1.selectbox("Mặt hàng", range(len(ten_sku)), format_func=lambda i: ten_sku[i])
    md_that = float(bundle["demand"][:, w_idx, s_idx].mean())
    du_bao = c2.number_input("Dự báo cầu (đv/ngày)", 0.0, 1000.0, round(md_that, 1),
                             help=f"Mặc định = cầu trung bình lịch sử thực tế ({md_that:.1f})")
    ton_hien_tai = c3.number_input("Tồn kho hiện tại (đv)", 0.0, 100000.0,
                                   round(du_bao * 3, 0))
else:
    du_bao = c1.number_input("Dự báo cầu (đv/ngày)", 0.1, 1000.0, 10.0)
    ton_hien_tai = c2.number_input("Tồn kho hiện tại (đv)", 0.0, 100000.0, 30.0)
    c3.caption("Kịch bản tự nhập — không gắn với SKU thật nào.")

if st.button("🔄 Khởi tạo / Làm lại tình huống", type="primary"):
    env2 = simulator.make_scenario_env(cfg, mean_demand=du_bao,
                                       current_inventory=ton_hien_tai)
    st.session_state["dx"] = {"env": env2, "history": []}

if "dx" not in st.session_state:
    st.info("Nhập tình huống rồi bấm **Khởi tạo / Làm lại tình huống**.")
    st.stop()

env2 = st.session_state["dx"]["env"]
obs2 = simulator.current_obs(env2)

ckpt = policy_runner.default_checkpoint(ROOT / "checkpoints")
ckpt_path = str(ckpt) if ckpt else None
policies = policy_runner.load_all_policies(env2, ckpt_path, cfg["ppo"], ROOT / "results")

st.subheader(f"Đề xuất cho tình huống: tồn kho {float(env2.inventory[0]):.0f} đv, "
            f"dự báo cầu {float(env2.mean_demand[0]):.1f} đv/ngày")

hang = []
for ten, fn in policies.items():
    a = fn(obs2, env2)
    qty = float(env2.action_to_qty(a)[0])
    hang.append({"Chính sách": ten, "Mức hành động": int(a[0]), "Số lượng đề xuất": qty})
df_dx = pd.DataFrame(hang).set_index("Chính sách")
st.dataframe(df_dx, width="stretch")
st.bar_chart(df_dx[["Số lượng đề xuất"]], height=260)

st.divider()
st.subheader("Áp dụng 1 đề xuất và xem chuyện gì xảy ra")
ten_ap_dung = st.selectbox("Áp dụng đề xuất của chính sách nào?", list(policies.keys()))
c1, c2 = st.columns(2)
if c1.button("▶️ Áp dụng — qua 1 ngày"):
    a = policies[ten_ap_dung](obs2, env2)
    _, _, te, tr, inf = env2.step(a)
    st.session_state["dx"]["history"].append({
        "Ngày": env2.current_step, "Chính sách": ten_ap_dung,
        "Đặt hàng": inf["order_qty"], "Cầu": inf["demand"], "Bán được": inf["sold"],
        "Thiếu hàng": inf["stockout"], "Tồn kho cuối ngày": inf["inventory"],
        "Chi phí ngày": (inf["cost_holding"] + inf["cost_stockout"]
                        + inf["cost_ordering"] + inf["cost_overflow"]),
    })
    st.rerun()

n_tu_dong = c2.number_input("...hoặc chạy tự động bao nhiêu ngày", 1, 200, 14)
if c2.button(f"⏩ Chạy tự động {n_tu_dong} ngày theo {ten_ap_dung}"):
    for _ in range(int(n_tu_dong)):
        obs2 = simulator.current_obs(env2)
        a = policies[ten_ap_dung](obs2, env2)
        _, _, te, tr, inf = env2.step(a)
        st.session_state["dx"]["history"].append({
            "Ngày": env2.current_step, "Chính sách": ten_ap_dung,
            "Đặt hàng": inf["order_qty"], "Cầu": inf["demand"], "Bán được": inf["sold"],
            "Thiếu hàng": inf["stockout"], "Tồn kho cuối ngày": inf["inventory"],
            "Chi phí ngày": (inf["cost_holding"] + inf["cost_stockout"]
                            + inf["cost_ordering"] + inf["cost_overflow"]),
        })
        if te or tr:
            break
    st.rerun()

hist = st.session_state["dx"]["history"]
if hist:
    dfh = pd.DataFrame(hist)
    c = st.columns(3)
    c[0].metric("Tổng chi phí đến giờ", f"{dfh['Chi phí ngày'].sum():,.0f}")
    fr = 1 - dfh["Thiếu hàng"].sum() / max(dfh["Cầu"].sum(), 1e-6)
    c[1].metric("Fill rate đến giờ", f"{fr:.1%}")
    c[2].metric("Số ngày đã chạy", len(dfh))
    st.line_chart(dfh.set_index("Ngày")[["Tồn kho cuối ngày", "Cầu"]], height=260)
    with st.expander("Bảng chi tiết"):
        st.dataframe(dfh, width="stretch")

with st.expander("❓ Ví dụ đọc trang này"):
    st.markdown(
        "Mỗi chính sách nhìn CÙNG một tình huống (tồn kho, dự báo cầu) nhưng "
        "tính ra số lượng khác nhau vì CÁCH TÍNH khác nhau: EOQ tối thiểu chi "
        "phí đặt hàng+lưu kho giả định cầu ổn định; (s,S) và Newsvendor thêm "
        "vùng đệm theo độ biến động cầu; IPPO học từ dữ liệu lịch sử, có thể "
        "phản ứng khác tùy trạng thái (vd sắp đến ngày lễ, kho đang gần đầy). "
        "Không có đáp án 'đúng nhất' cố định — đó là lý do cần so sánh nhiều "
        "kịch bản ở trang **So sánh policy** và **What-if**.")
