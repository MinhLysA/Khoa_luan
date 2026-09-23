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
_mac_dinh = policy_runner.default_checkpoint(ROOT / "checkpoints")
_ten = [p.name for p in ckpts]
ten_ckpt = c3.selectbox("Checkpoint IPPO", _ten or ["(chưa có)"],
                        index=_ten.index(_mac_dinh.name) if _mac_dinh else 0)

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

st.subheader("Chi tiết từng ngày cho một cặp kho - SKU")
st.caption("Mỗi ngày: cầu xuất hiện → tác tử đọc trạng thái → quyết định đặt "
          "hàng → kho cập nhật → tính reward → sang ngày kế tiếp.")
c1, c2, c3 = st.columns(3)
ten_cs = c1.selectbox("Chính sách", list(ket_qua))
kq = ket_qua[ten_cs]
n_kho, n_sku = kq["inventory_matrix"].shape[1:]
kho = c2.number_input("Kho", 0, n_kho - 1, 0)
sku = c3.number_input("SKU", 0, n_sku - 1, 0)

chi_tiet = pd.DataFrame({
    "Cầu": kq["demand_matrix"][:, kho, sku],
    "Tồn kho cuối ngày": kq["inventory_matrix"][:, kho, sku],
    "Hàng đang về": kq["incoming_matrix"][:, kho, sku],
    "Đặt hàng": kq["order_matrix"][:, kho, sku],
    "Thiếu hàng": kq["stockout_matrix"][:, kho, sku],
    "Chi phí": kq["cost_matrix"][:, kho, sku],
    "Reward": kq["reward_matrix"][:, kho, sku],
})
chi_tiet.index.name = "Ngày"
chi_tiet["Cảnh báo"] = np.where(
    chi_tiet["Thiếu hàng"] > 0, "⛔ Thiếu hàng",
    np.where(chi_tiet["Tồn kho cuối ngày"] + chi_tiet["Hàng đang về"]
             < chi_tiet["Cầu"].rolling(7, min_periods=1).mean() * 2,
             "⚠️ Sắp hết (< 2 ngày cầu)", ""))

k1, k2, k3, k4 = st.columns(4)
k1.metric("Số ngày thiếu hàng", int((chi_tiet["Thiếu hàng"] > 0).sum()))
k4.metric("Tổng chi phí của cặp", f"{chi_tiet['Chi phí'].sum():,.0f}")
k2.metric("Fill rate của cặp",
          f"{100 * (1 - chi_tiet['Thiếu hàng'].sum() / max(chi_tiet['Cầu'].sum(), 1e-6)):.1f}%")
k3.metric("Tổng reward", f"{chi_tiet['Reward'].sum():,.0f}")
st.line_chart(chi_tiet[["Cầu", "Tồn kho cuối ngày", "Hàng đang về", "Đặt hàng"]], height=280)
st.dataframe(chi_tiet.style.format({c: "{:,.1f}" for c in chi_tiet.columns if c != "Cảnh báo"}),
             width="stretch", height=320)
st.caption("Chi phí = lưu kho + thiếu hàng + đặt hàng + tràn kho của riêng cặp này. "
          "Reward = −(chi phí + phạt mức phục vụ), nên luôn ≤ 0. Reward càng gần 0 thì càng tốt; "
          "đừng đọc reward âm là mô hình xấu.")

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
