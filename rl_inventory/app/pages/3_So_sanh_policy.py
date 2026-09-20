"""
app/pages/3_So_sanh_policy.py
================================
Trang 3 - chạy IPPO + 3 baseline trên CÙNG một chuỗi cầu (cùng seed, dữ liệu
thật) để so sánh công bằng: đường tồn kho, tổng chi phí, phân rã chi phí.
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

st.set_page_config(page_title="So sánh policy", page_icon="🏁", layout="wide")
st.title("🏁 So sánh policy trên CÙNG một chuỗi cầu")
st.caption("Không phải lấy số trung bình nhiều episode như Chương 4 — đây là "
          "MỘT kịch bản cụ thể, chạy song song, để nhìn thấy TRỰC TIẾP từng "
          "chính sách xử lý thế nào.")

with open(ROOT / "config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
bundle = simulator.load_env_bundle()
if bundle["demand"] is None:
    st.warning("Chưa có dữ liệu. Chạy `python run.py data` trước.")
    st.stop()

c1, c2, c3 = st.columns(3)
seed = c1.number_input("Seed", 0, 10**6, 1000)
so_ngay = c2.number_input("Số ngày", 10, 365, 120, step=10)
ckpts = sorted((ROOT / "checkpoints").glob("*.pth"))
ten_ckpt = c3.selectbox("Checkpoint IPPO", [p.name for p in ckpts] or ["(chưa có)"])

if st.button("▶️ Chạy so sánh", type="primary"):
    def env_factory():
        return simulator.make_env(cfg, bundle, mode="test")

    env_tmp = env_factory()
    policies = policy_runner.load_all_policies(
        env_tmp, str(ROOT / "checkpoints" / ten_ckpt) if ckpts else None,
        cfg["ppo"], ROOT / "results")

    with st.spinner(f"Đang chạy {len(policies)} chính sách x {so_ngay} ngày..."):
        ket_qua = simulator.run_parallel(env_factory, policies, seed=int(seed),
                                         max_days=int(so_ngay))
    st.session_state["ss"] = ket_qua

if "ss" not in st.session_state:
    st.info("Bấm **Chạy so sánh** để xem kết quả.")
    st.stop()

ket_qua = st.session_state["ss"]

duong_ton_kho = {ten: kq["daily"].set_index("Ngày")["Tồn kho"]
                for ten, kq in ket_qua.items()}
duong_chi_phi = {ten: kq["daily"].set_index("Ngày")["Chi phí ngày"].cumsum()
                for ten, kq in ket_qua.items()}

df_tong = simulator.summarize(ket_qua)
st.subheader("Tổng hợp")
st.dataframe(df_tong.style.format({"Tổng chi phí": "{:,.0f}", "Fill rate": "{:.1f}%",
                                   "Lưu kho": "{:,.0f}", "Thiếu hàng": "{:,.0f}",
                                   "Đặt hàng": "{:,.0f}", "Tràn kho": "{:,.0f}"}),
            width="stretch")
st.caption("So chi phí CHỈ CÔNG BẰNG khi fill rate gần nhau. Xem cột Fill rate "
          "cạnh cột Tổng chi phí trước khi kết luận chính sách nào 'rẻ hơn'.")

c1, c2 = st.columns(2)
c1.markdown("**Tổng chi phí**")
c1.bar_chart(df_tong[["Tổng chi phí"]], height=280)
c2.markdown("**Fill rate (%)**")
c2.bar_chart(df_tong[["Fill rate"]], height=280)

st.markdown("**Phân rã chi phí** (không có chi phí chuyển kho — dự án này mỗi "
           "kho hoạt động độc lập, chưa xét cấu trúc đa cấp/multi-echelon)")
st.bar_chart(df_tong[["Lưu kho", "Thiếu hàng", "Đặt hàng", "Tràn kho"]],
            height=300, stack=True)

st.subheader("Tồn kho theo thời gian (toàn hệ thống)")
st.line_chart(pd.DataFrame(duong_ton_kho), height=300)

st.subheader("Chi phí lũy kế theo thời gian")
st.line_chart(pd.DataFrame(duong_chi_phi), height=300)

with st.expander("❓ Ví dụ đọc trang này"):
    st.markdown(
        "Cả 4 chính sách cùng nhìn ĐÚNG một chuỗi cầu lịch sử thật (cùng "
        "seed) — khác nhau là ở cách MỖI chính sách phản ứng. Nếu 1 đường "
        "tồn kho luôn ở mức thấp và sát 0 nhiều đoạn, chính sách đó đang "
        "'liều lĩnh', dễ thiếu hàng khi cầu tăng đột ngột. Nếu 1 đường luôn "
        "cao và bằng phẳng, chính sách đó đang giữ dư phòng dày nhưng tốn chi "
        "phí lưu kho. Kết quả CHÍNH THỨC (trung bình nhiều episode, kiểm định "
        "thống kê) nằm ở `results/`, `TONG_HOP_SO_LIEU.txt` — trang này chỉ "
        "để MINH HỌA trực quan một kịch bản cụ thể.")
