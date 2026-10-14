"""
app/streamlit_app.py
====================
Demo "Mô phỏng chính sách đặt hàng lại đa kho bằng IPPO" - CHỈ INFERENCE:
nạp app/demo_data/best_model.pth, chạy deterministic trên miền test, không có
đường code nào gọi huấn luyện.

Tệp này chỉ làm điều hướng. Mỗi trang một tệp trong app/pages_/; phần nạp mô
hình và mô phỏng nằm trong app/core.py.

    python run.py demo_data      # một lần: đóng gói app/demo_data/
    python run.py app
"""
import streamlit as st

import core

st.set_page_config(page_title="Mô phỏng chính sách đặt hàng", layout="wide")
st.markdown(core.CSS, unsafe_allow_html=True)

if not (core.DEMO / "meta.json").exists():
    st.error("Chưa có app/demo_data/. Chạy `python run.py demo_data` một lần rồi mở lại.")
    st.stop()

meta, _ = core.load_demo()
st.sidebar.markdown(f"**Mô phỏng chính sách đặt hàng**  \n"
                    f"{len(meta['stores'])} kho × {len(meta['skus'])} SKU · dữ liệu M5 Walmart  \n"
                    f"Chỉ suy luận trên miền test")

st.navigation([
    st.Page("pages_/tong_quan.py", title="Tổng quan", default=True),
    st.Page("pages_/don_dat_hang.py", title="Đơn đặt hàng đề xuất"),
    st.Page("pages_/chi_tiet_cap.py", title="Chi tiết cặp (kho, SKU)"),
    st.Page("pages_/so_sanh.py", title="So sánh chính sách"),
    st.Page("pages_/mo_hinh.py", title="Về mô hình"),
]).run()
