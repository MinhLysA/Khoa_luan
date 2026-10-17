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

core.tieu_de(f"Chi tiết cặp · {len(store_of)} kho × {len(meta['skus'])} SKU",
             "Chi tiết một cặp (kho, SKU) trong mô phỏng IPPO",
             "Diễn biến trên miền test và quan sát của tác tử.")
c1, c2, c3 = st.columns([1.2, 2, 1.2])
kho = c1.selectbox("Kho", list(store_of), format_func=lambda k: f"{k} · {store_of[k]}")
sku = c2.selectbox("SKU", meta["skus"])
ngay = c3.date_input("Ngày xem quan sát", value=dates[-1], min_value=dates[0],
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
fig.update_annotations(font=dict(size=13, color=core.INK), x=0, xanchor="left")
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
    fig.add_vrect(x0=x0, x1=x1, fillcolor="#E6E9E4", line_width=0, layer="below", row=1, col=1)
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
    st.subheader(f"Diễn biến cặp {kho} · {sku}")
    core.chart(fig, 640)
    st.caption(("Vùng xám: dữ liệu M5 ghi nhận 0 bán (≥ 7 ngày liền) · " if len(doan) else "")
               + "đường chấm dọc: ngày đang xem · sức chứa áp cho tổng tồn kho của cả kho, không riêng từng cặp.")

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
with st.container(border=True, key="the_quan_sat"):
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
core.nguon(f"{len(tbl)} chiều = env.obs_per_pair ({env.obs_per_pair}), đã chuẩn hóa về [0, 1]",
           "rollout IPPO đóng gói trong app/demo_data")
