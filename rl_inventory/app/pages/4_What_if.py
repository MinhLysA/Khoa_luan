"""
app/pages/4_What_if.py
========================
Trang 4 - đổi lead time / chi phí, hoặc tạo cú sốc cầu tăng đột biến, rồi
chạy lại so sánh 4 chính sách để xem ai thích nghi tốt hơn. KHÔNG ghi đè
config.yaml thật - chỉ override cục bộ trong phiên demo.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
APP_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT, APP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend import simulator, policy_runner  # noqa: E402

st.set_page_config(page_title="What-if", page_icon="🌪️", layout="wide")
st.title("🌪️ Kịch bản What-if")
st.caption("Đổi tham số hoặc tạo cú sốc cầu RIÊNG CHO PHIÊN DEMO NÀY — không "
          "đụng chạm config.yaml thật. Đây là kịch bản để tìm lợi thế của RL: "
          "cầu biến động mạnh, ràng buộc chặt, điều mà công thức tính (EOQ, "
          "(s,S)) khó thích nghi kịp.")

with open(ROOT / "config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
bundle = simulator.load_env_bundle()
if bundle["demand"] is None:
    st.warning("Chưa có dữ liệu. Chạy `python run.py data` trước.")
    st.stop()

e0 = cfg["env"]
st.subheader("Đổi tham số")
c1, c2, c3 = st.columns(3)
lead_min, lead_max = c1.slider("Lead time (ngày)", 1, 10,
                               (int(e0["lead_time_min"]), int(e0["lead_time_max"])))
cp_lk = c2.number_input("Chi phí lưu kho / đv / ngày", 0.0, 100.0, float(e0["cp_lk"]))
capacity_cover = c3.number_input("Sức chứa kho (số ngày cầu)", 1.0, 30.0,
                                 float(e0["capacity_cover_days"]), step=0.5)

st.subheader("Cú sốc cầu")
soc = st.slider("Hệ số nhân cầu (1,0 = bình thường, 1,5 = tăng đột biến 50%)",
                1.0, 3.0, 1.0, step=0.1)
st.caption("Cú sốc áp dụng cho TOÀN BỘ chuỗi cầu trong kịch bản — mô phỏng "
          "tình huống không ai báo trước (sức chứa kho, bảng mức đặt hàng, "
          "các tham số công thức baseline VẪN tính theo cầu CŨ, giống thực "
          "tế: không ai kịp xây thêm kho khi cầu đột ngột tăng).")

c1, c2 = st.columns(2)
seed = c1.number_input("Seed", 0, 10**6, 1000)
so_ngay = c2.number_input("Số ngày", 10, 365, 90, step=10)
ckpts = sorted((ROOT / "checkpoints").glob("*.pth"))
ten_ckpt = ckpts[0].name if ckpts else None

if st.button("▶️ Chạy so sánh what-if", type="primary"):
    overrides = {"lead_time_min": int(lead_min), "lead_time_max": int(lead_max),
                "cp_lk": float(cp_lk), "capacity_cover_days": float(capacity_cover)}

    def env_factory():
        env = simulator.make_env(cfg, bundle, mode="test", env_overrides=overrides)
        if soc != 1.0:
            simulator.apply_demand_shock(env, soc)
        return env

    env_tmp = env_factory()
    policies = policy_runner.load_all_policies(
        env_tmp, str(ROOT / "checkpoints" / ten_ckpt) if ten_ckpt else None,
        cfg["ppo"], ROOT / "results")

    with st.spinner("Đang chạy kịch bản what-if..."):
        kq_shock = simulator.run_parallel(env_factory, policies, seed=int(seed),
                                          max_days=int(so_ngay))

        def env_factory_goc():
            return simulator.make_env(cfg, bundle, mode="test")
        kq_goc = simulator.run_parallel(env_factory_goc, policies, seed=int(seed),
                                        max_days=int(so_ngay))

    st.session_state["wi"] = {"shock": kq_shock, "goc": kq_goc}

if "wi" not in st.session_state:
    st.info("Chỉnh tham số rồi bấm **Chạy so sánh what-if**.")
    st.stop()

df_goc = simulator.summarize(st.session_state["wi"]["goc"])
df_shock = simulator.summarize(st.session_state["wi"]["shock"])

_fmt = {"Tổng chi phí": "{:,.0f}", "Fill rate": "{:.1f}%", "Lưu kho": "{:,.0f}",
       "Thiếu hàng": "{:,.0f}", "Đặt hàng": "{:,.0f}", "Tràn kho": "{:,.0f}"}

st.subheader("Trước vs Sau khi áp dụng kịch bản")
c1, c2 = st.columns(2)
c1.markdown("**Cấu hình gốc (config.yaml, không sốc)**")
c1.dataframe(df_goc.style.format(_fmt), width="stretch")
c2.markdown(f"**Kịch bản what-if (tham số mới + sốc cầu x{soc:.1f})**")
c2.dataframe(df_shock.style.format(_fmt), width="stretch")

st.subheader("Chênh lệch fill rate khi có sốc (điểm %, âm = giảm mạnh hơn)")
chenh = (df_shock["Fill rate"] - df_goc["Fill rate"]).to_frame("Chênh lệch fill rate (điểm %)")
st.bar_chart(chenh, height=280)
st.caption("Chính sách nào TỤT fill rate ÍT HƠN khi bị sốc — chính sách đó "
          "'chịu sốc' tốt hơn. Đây là loại bằng chứng đáng đưa vào khóa luận "
          "hơn là chỉ số tổng chi phí đơn thuần, vì nó cho thấy khả năng "
          "THÍCH NGHI chứ không chỉ tối ưu trên dữ liệu ổn định.")

with st.expander("❓ Ví dụ đọc trang này"):
    st.markdown(
        "VD (s,S) và Newsvendor đặt tham số (ngưỡng đặt hàng, mức đặt tối đa) "
        "dựa trên cầu LỊCH SỬ (trước sốc) — khi cầu thực tế tăng 50% mà "
        "không ai báo trước, công thức cũ đặt KHÔNG ĐỦ, fill rate sụt "
        "mạnh. IPPO quan sát được lịch sử cầu GẦN NHẤT (7 ngày, thang log) "
        "nên có thể phản ứng nhanh hơn trong phạm vi dữ liệu đã từng thấy — "
        "nhưng nếu sốc vượt xa mọi thứ đã học trong lúc train, IPPO cũng có "
        "thể phản ứng kém, thậm chí tệ hơn nếu quan sát bị bão hòa (vượt "
        "ngưỡng chuẩn hóa đã tính từ cầu CŨ). Đây là điểm đáng trình bày "
        "trung thực thay vì chỉ nói 'AI luôn thắng'.")
