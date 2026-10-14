"""Trang 1 - Tổng quan: KPI (results/), heatmap fill rate và sức chứa (rollout trong demo_data)."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import core

UTIL_CANH_BAO = 0.90            # kho dùng từ 90% sức chứa trở lên tô cam

meta, arr = core.load_demo()
P = core.pairs(meta)
W, S = len(meta["stores"]), len(meta["skus"])
muc_dv, pt_tk = float(meta["env"]["muc_dv"]), float(meta["env"]["pt_tk"])
nhan_kho = [f"{k} · {c}" for k, c in P[["Kho", "Cửa hàng"]].drop_duplicates().values]

st.title("Tổng quan mô phỏng trên miền test")

sm = core.summary(meta)
if sm is None:
    core.no_data(f"results/summary_{meta['results_tag']}.json")
else:
    df, stat, raw = sm
    ip = df.loc["IPPO"]
    c = st.columns(4)
    if "gap_percent" in stat:
        gap = stat["gap_percent"]
        so_sanh = (f"{'+' if gap > 0 else ''}{core.num(gap, 1)}% so với "
                   f"{stat['best_baseline']} (baseline tốt nhất)")
        c[0].markdown(core.kpi("Tổng chi phí IPPO (đơn vị mô phỏng)", core.num(ip.total_cost_mean),
                               so_sanh, "warn" if gap > 0 else "good"), unsafe_allow_html=True)
    else:
        c[0].markdown(core.kpi("Tổng chi phí IPPO (đơn vị mô phỏng)", core.num(ip.total_cost_mean),
                               "chưa có dữ liệu so sánh"), unsafe_allow_html=True)
    c[1].markdown(core.kpi("Fill rate trung bình", core.pct(ip.fill_rate_mean),
                           f"Mục tiêu {core.pct(muc_dv, 0)}",
                           "good" if ip.fill_rate_mean >= muc_dv else "warn"), unsafe_allow_html=True)
    c[2].markdown(core.kpi("Đơn vị bị từ chối do tràn kho", core.num(ip.overflow_cost_mean / pt_tk),
                           f"Chi phí tràn kho / pt_tk ({core.num(pt_tk)} mỗi đơn vị)"),
                  unsafe_allow_html=True)
    c[3].markdown(core.kpi("Mức sử dụng sức chứa trung bình",
                           core.pct(ip.avg_inventory_mean / sum(raw["capacity_per_warehouse"])),
                           "Tồn kho TB toàn mạng / tổng sức chứa"), unsafe_allow_html=True)
    st.caption(f"Trung bình {raw['meta']['n_episodes']} lượt đánh giá trên miền test · "
               f"results/summary_{meta['results_tag']}.json · checkpoint {raw['meta']['checkpoint']}")

st.divider()
cs = st.selectbox("Chính sách cho bản đồ nhiệt và sức chứa", meta["policies"])
r = core.rollout(cs)
dem = arr["demand"].reshape(len(arr["demand"]), -1).sum(axis=0)
with np.errstate(invalid="ignore", divide="ignore"):
    fr = np.where(dem > 0, 1 - r["stockout"].sum(axis=0) / dem, np.nan).reshape(W, S)
st.caption(f"Một lượt mô phỏng đại diện trên toàn miền test (hạt giống {meta['seed']}, "
           f"trùng lượt đánh giá đầu tiên trong results/).")

st.subheader("Fill rate theo cặp (kho, SKU)")
fig = go.Figure(go.Heatmap(
    z=fr, x=meta["skus"], y=nhan_kho, zmin=2 * muc_dv - 1, zmax=1,
    colorscale=[[0, core.ORANGE], [0.5, "#ECEEE9"], [1, core.TEAL]],
    colorbar=dict(tickformat=".0%", title="Fill rate"),
    hovertemplate="Kho %{y}<br>SKU %{x}<br>Fill rate %{z:.1%}<extra></extra>"))
fig.update_yaxes(autorange="reversed")
fig.update_xaxes(tickangle=-50)
st.plotly_chart(core.style_fig(fig, 460), theme=None)
st.caption(f"Màu nhạt quanh mức mục tiêu {core.pct(muc_dv, 0)}; cam: thấp hơn; teal: cao hơn. "
           f"Ô trống: cặp không có nhu cầu trong miền test.")

st.subheader("Mức sử dụng sức chứa theo kho")
u = r["util"].mean(axis=0)
fig = go.Figure(go.Bar(
    x=u, y=nhan_kho, orientation="h", text=[core.pct(x) for x in u], textposition="outside",
    marker_color=[core.ORANGE if x >= UTIL_CANH_BAO else core.TEAL for x in u],
    hovertemplate="%{y}<br>%{x:.1%}<extra></extra>"))
fig.add_vline(x=UTIL_CANH_BAO, line=dict(color=core.GREY, dash="dot"))
fig.update_yaxes(autorange="reversed")
fig.update_xaxes(tickformat=".0%", range=[0, max(1.0, float(u.max()) * 1.15)])
st.plotly_chart(core.style_fig(fig, 400), theme=None)
st.caption(f"Trung bình theo ngày của tổng tồn kho cuối ngày / sức chứa kho. "
           f"Kho từ {core.pct(UTIL_CANH_BAO, 0)} trở lên tô màu cam.")
