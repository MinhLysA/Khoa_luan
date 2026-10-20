"""Trang 3 - Chi tiết một cặp (kho, SKU): diễn biến trên miền test và quyết định của IPPO ở ngày chọn."""
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

core.tieu_de(f"Chi tiết · {len(store_of)} cửa hàng × {len(meta['skus'])} mặt hàng",
             "Chi tiết một mặt hàng",
             "Tồn kho, nhu cầu và các lần đặt hàng của IPPO cho một mặt hàng ở một cửa hàng.")
c1, c2, c3 = st.container(key="the_chon").columns([1.2, 2, 1.2])
kho = c1.selectbox("Cửa hàng", list(store_of), format_func=lambda k: f"{k} · {store_of[k]}")
sku = c2.selectbox("Mặt hàng", meta["skus"])
ngay = c3.date_input("Ngày xem quyết định", value=dates[-1], min_value=dates[0],
                     max_value=dates[-1], format="DD/MM/YYYY")
row = P[(P["Kho"] == kho) & (P["SKU"] == sku)].iloc[0]
p, w, day = int(row["pair"]), int(row["w"]), (ngay - dates[0]).days
r = core.rollout("IPPO")
cap = float(env.suc_chua_kho[w])
cau = arr["demand"][:, w, row["s"]]
nua = pd.Timedelta(hours=12)                                # nửa ngày: mép cột / mép dải tô


def mep(i, j):
    """Khoảng thời gian phủ trọn các ngày dates[i..j]."""
    return pd.Timestamp(dates[i]) - nua, pd.Timestamp(dates[j]) + nua


fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.08, 0.32], vertical_spacing=0.07,
                    subplot_titles=("Tồn kho đầu ngày (đường bậc) và nhu cầu (cột) · đơn vị hàng",
                                    "Ngày đặt hàng · rê chuột để xem số lượng (đơn vị hàng)",
                                    f"Toàn kho {kho}: tổng tồn kho cuối ngày và sức chứa · đơn vị hàng"))
fig.update_annotations(font=dict(size=13), x=0, xanchor="left")
fig.add_bar(x=dates, y=cau, name="Nhu cầu", marker_color=core.GREY, opacity=0.55, row=1, col=1,
            hovertemplate="Nhu cầu: %{y:,.0f}<extra></extra>")
fig.add_scatter(x=dates, y=r["inv"][:, p], name="Tồn kho đầu ngày", line=dict(color=core.TEAL, width=2, shape="hv"),
                hovertemplate="Tồn kho đầu ngày: %{y:,.0f}<extra></extra>", row=1, col=1)
dat = r["order"][:, p] > 0
fig.add_scatter(x=np.array(dates)[dat], y=np.zeros(dat.sum()), customdata=r["order"][dat, p], mode="markers",
                name="Đặt hàng", hovertemplate="Đặt hàng: %{customdata:,.0f} đơn vị<extra></extra>",
                marker=dict(symbol="line-ns-open", size=16, color=core.TEAL, line=dict(width=2, color=core.TEAL)),
                row=2, col=1)
fig.add_scatter(x=dates, y=r["util"][:, w] * cap, name="Tổng tồn kho cuối ngày", line=dict(color=core.INK, width=1.5),
                hovertemplate="Tổng tồn kho cuối ngày: %{y:,.0f}<extra></extra>", row=3, col=1)
fig.add_hline(y=cap, line=dict(color=core.INK, dash="dash", width=1.5), row=3, col=1,
              annotation_text=f"Sức chứa {core.num(cap)}", annotation_position="top left")
# Đoạn >= 7 ngày liền M5 ghi nhận 0 bán: tô xám để không bị hiểu là mô phỏng đứng yên
z = np.r_[False, cau == 0, False]
doan = np.flatnonzero(z[1:] != z[:-1]).reshape(-1, 2)      # [đầu, cuối) mỗi đoạn 0 bán
doan = doan[doan[:, 1] - doan[:, 0] >= 7]
for a, b in doan:
    x0, x1 = mep(a, b - 1)
    fig.add_vrect(x0=x0, x1=x1, fillcolor=core.SHADE, line_width=0, layer="below", row=1, col=1)
fig.add_vline(x=ngay, line=dict(color=core.INK, dash="dot", width=1))
# Mặc định xem ngày đã chọn −60 … +30 ngày; thanh trượt dưới biểu đồ để mở rộng
fig.update_xaxes(range=mep(max(day - 60, 0), min(day + 30, len(dates) - 1)),
                 tickformat="%d/%m/%Y", hoverformat="%d/%m/%Y")
fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06), row=3, col=1)
fig.update_yaxes(title="Đơn vị hàng", row=1, col=1)
fig.update_yaxes(visible=False, range=[-1, 1], row=2, col=1)
fig.update_yaxes(title="Đơn vị hàng", row=3, col=1)
fig.update_layout(showlegend=False, hovermode="x unified", bargap=0.15)
with st.container(border=True, key="the_dien_bien"):
    core.muc(1, "Diễn biến trong giai đoạn test", f"{kho} · {store_of[kho]} · {sku}")
    core.chart(fig, 640)
    st.caption(("Vùng xám: những ngày liền không bán được sản phẩm nào · " if len(doan) else "")
               + "đường chấm dọc: ngày đang xem.")

s = core.state_at("IPPO", day)
a = int(s["action"][p])
mult = float(env.order_levels[a])
core.muc(2, "Quyết định của IPPO", f"ngày {ngay:%d/%m/%Y}")
k = st.columns(3)
k[0].markdown(core.kpi("Mức đặt hàng", f"{a} / {env.n_action_levels - 1}",
                       f"trong {env.n_action_levels} mức đặt hàng (0 = không đặt)"), unsafe_allow_html=True)
k[1].markdown(core.kpi("Đặt bằng", f"× {core.num(mult, 1)}",
                       f"lượng bán trung bình trong {core.num(env.tg_giao_tb, 1)} ngày chờ hàng về"), unsafe_allow_html=True)
k[2].markdown(core.kpi("Số lượng đặt", core.num(s["qty"][p]),
                       "đơn vị hàng, đã làm tròn"), unsafe_allow_html=True)

