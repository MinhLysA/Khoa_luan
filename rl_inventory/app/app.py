"""
app/app.py
==========
Điểm vào Demo IPPO - Tối ưu chính sách đặt hàng đa kho.

    streamlit run app/app.py

4 trang (Streamlit tự động nhận trong app/pages/):
    1. Tổng quan mạng lưới kho  - heatmap tồn kho kho x SKU theo ngày
    2. Đề xuất đặt hàng         - IPPO vs 3 baseline đề xuất bao nhiêu cho 1 cặp
    3. So sánh policy           - chạy song song trên CÙNG 1 chuỗi cầu
    4. What-if                  - đổi lead time/chi phí, tạo sốc cầu, xem ai chịu tốt hơn

Bảng điều khiển VẬN HÀNH dự án (sửa config, bấm chạy từng bước pipeline, xem
đường học training) vẫn còn nguyên ở app/streamlit_app.py - không bị xóa, chỉ
không còn là mặc định của `python run.py app` (xem pham_vi_khoa_luan.md muc 6).
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Demo IPPO Tồn kho đa kho", page_icon="📦", layout="wide")

st.title("📦 Demo: Tối ưu chính sách đặt hàng lại đa kho bằng IPPO")
st.caption("Đa kho - đa SKU | dữ liệu M5 Walmart | Independent Multi-Agent PPO "
           "với chia sẻ tham số")

st.markdown("""
Dùng menu bên trái để chuyển trang:

1. **Tổng quan mạng lưới kho** — xem tình trạng tồn kho toàn hệ thống theo từng ngày.
2. **Đề xuất đặt hàng** — nhập 1 tình huống (kho, SKU, tồn hiện tại, dự báo cầu),
   xem IPPO và 3 chính sách cổ điển (EOQ, (s,S), Newsvendor) đề xuất đặt bao nhiêu.
3. **So sánh policy** — chạy IPPO và 3 baseline trên CÙNG một chuỗi cầu để so
   sánh công bằng: đường tồn kho, chi phí, phân rã chi phí.
4. **What-if** — đổi lead time, chi phí, hoặc tạo cú sốc cầu tăng đột biến, xem
   chính sách nào thích nghi tốt hơn.
""")

st.info(
    "**Lưu ý trung thực**: so thẳng tổng chi phí giữa các chính sách có fill "
    "rate khác nhau là không công bằng — IPPO thường đẩy fill rate cao hơn "
    "mức baseline dừng lại. Trang *So sánh policy* có ghi rõ fill rate cạnh "
    "chi phí để đọc đúng; kết quả chính thức (so ở cùng mức phục vụ, kiểm "
    "định thống kê) nằm ở `results/iso_service.json` và `TONG_HOP_SO_LIEU.txt`, "
    "không phải ở app demo này.")

CKPT = ROOT / "checkpoints" / "best_model.pth"
DATA = ROOT / "data" / "processed" / "demand_data.npy"
c1, c2 = st.columns(2)
c1.metric("Dữ liệu đã tiền xử lý", "✅ Sẵn sàng" if DATA.exists() else "❌ Chưa có")
c2.metric("Checkpoint IPPO", "✅ Sẵn sàng" if CKPT.exists() else "❌ Chưa có")
if not DATA.exists() or not CKPT.exists():
    st.warning("Chạy `python run.py all` (hoặc từng bước data/train) trước khi "
               "dùng các trang demo.")
