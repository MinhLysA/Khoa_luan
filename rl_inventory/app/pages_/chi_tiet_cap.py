"""Trang 3 - Chi tiết một cặp (kho, SKU): diễn biến trên miền test và quan sát 44 chiều của tác tử."""
import numpy as np
import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots

import core

meta, arr = core.load_demo()
P = core.pairs(meta)
dates = core.test_dates(meta)
env = core.base_env()
store_of = dict(zip(P["Kho"], P["Cửa hàng"]))

st.title("Chi tiết một cặp (kho, SKU) trong mô phỏng IPPO")
c1, c2, c3 = st.columns([1.2, 2, 1.2])
kho = c1.selectbox("Kho", list(store_of), format_func=lambda k: f"{k} · {store_of[k]}")
sku = c2.selectbox("SKU", meta["skus"])
ngay = c3.date_input("Ngày xem quan sát", value=dates[-1], min_value=dates[0],
                     max_value=dates[-1], format="DD/MM/YYYY")
row = P[(P["Kho"] == kho) & (P["SKU"] == sku)].iloc[0]
p, w, day = int(row["pair"]), int(row["w"]), (ngay - dates[0]).days
r = core.rollout("IPPO")
cap = float(env.suc_chua_kho[w])

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38],
                    vertical_spacing=0.08,
                    subplot_titles=("", f"Toàn kho {kho}: tổng tồn kho và sức chứa"))
fig.add_bar(x=dates, y=arr["demand"][:, w, row["s"]], name="Nhu cầu",
            marker_color=core.GREY, opacity=0.55, row=1, col=1)
fig.add_scatter(x=dates, y=r["inv"][:, p], name="Tồn kho đầu ngày",
                line=dict(color=core.TEAL, width=2), row=1, col=1)
dat = r["order"][:, p] > 0
fig.add_scatter(x=np.array(dates)[dat], y=r["order"][dat, p], mode="markers",
                name="Ngày đặt hàng (số lượng)", row=1, col=1,
                marker=dict(color=core.ORANGE, size=7, symbol="triangle-up"))
fig.add_scatter(x=dates, y=r["util"][:, w] * cap, name="Tổng tồn kho của kho",
                line=dict(color=core.INK, width=1.5), row=2, col=1)
fig.add_hline(y=cap, line=dict(color=core.ORANGE, dash="dash"), row=2, col=1,
              annotation_text=f"Sức chứa {core.num(cap)}", annotation_position="top left")
fig.add_vline(x=ngay, line=dict(color=core.GREY, dash="dot"))
st.plotly_chart(core.style_fig(fig, 560), theme=None)
st.caption("Sức chứa áp cho TỔNG tồn kho của cả kho (30 SKU dùng chung), không áp riêng cho "
           "từng cặp, nên đường sức chứa nằm ở biểu đồ dưới. Đường chấm dọc: ngày đang xem.")

s = core.state_at("IPPO", day)
a = int(s["action"][p])
mult = float(env.order_levels[a])
st.subheader(f"Hành động deterministic ngày {ngay:%d/%m/%Y}")
k = st.columns(3)
k[0].markdown(core.kpi("Mức hành động", f"{a} / {env.n_action_levels - 1}",
                       f"argmax của Actor trên {env.n_action_levels} mức"), unsafe_allow_html=True)
k[1].markdown(core.kpi("Hệ số nhân", f"× {core.num(mult, 1)}",
                       f"cầu TB ({core.num(env.mean_demand[p], 2)}/ngày) × lead time TB "
                       f"({core.num(env.tg_giao_tb, 1)} ngày)"), unsafe_allow_html=True)
k[2].markdown(core.kpi("Số lượng đặt", core.num(s["qty"][p]),
                       "làm tròn lên theo bảng đặt hàng của cặp"), unsafe_allow_html=True)

lab = core.obs_labels(env, meta)
tbl = pd.DataFrame(lab, columns=["Vai trò", "Nhóm", "Đặc trưng"]).assign(**{"Giá trị": s["obs"][p]})
tbl.index.name = "Chiều"
st.subheader(f"Quan sát của tác tử ({len(tbl)} chiều)")
GHI_CHU = {
    "Thông tin trạng thái": "Tồn kho, hàng đang về, lịch sử nhu cầu 7 ngày, fill rate gần đây, "
                            "quy mô cầu của chính cặp này.",
    "Tín hiệu cấp kho": "Kênh ghép nối gián tiếp giữa các cặp: các cặp cùng kho thấy chung util, "
                        "nên hàng mà cặp khác đặt về làm thay đổi quan sát của cặp này.",
    "Tín hiệu giá": "Mã hóa LIÊN TỤC trong [0, 1], không phải 0/1. Bằng 0 khi giá hôm nay bằng "
                    "hoặc cao hơn giá bán TB miền train của chính cặp (bình thường hoặc tăng giá); "
                    "càng gần 1 là giảm giá càng sâu (0,2 = đang giảm 20%).",
    "Tín hiệu lịch": "Ngày trong tuần one-hot theo chỉ số ngày mod 7 (nhãn thứ đã quy đổi theo "
                     "lịch M5); tiếp theo là SNAP 3 bang, loại sự kiện và tháng (one-hot).",
}
for vai_tro, ghi_chu in GHI_CHU.items():
    t = tbl[tbl["Vai trò"] == vai_tro]
    with st.expander(f"{vai_tro} · {len(t)} chiều", expanded=vai_tro != "Tín hiệu lịch"):
        st.caption(ghi_chu)
        st.dataframe(t[["Đặc trưng", "Giá trị"]], height=36 * (len(t) + 1),
                     column_config={"Giá trị": st.column_config.NumberColumn(format="%.4f")})
st.caption(f"Tổng {len(tbl)} chiều = env.obs_per_pair ({env.obs_per_pair}); mọi giá trị đã "
           f"chuẩn hóa về [0, 1].")
