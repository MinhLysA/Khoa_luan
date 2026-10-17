"""Trang 1 - Tổng quan: KPI (results/), heatmap fill rate và sức chứa (rollout trong demo_data)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import core

UTIL_CANH_BAO = 0.90            # kho dùng từ 90% sức chứa trở lên tô cam
CAU_LE_TE = 3                   # cầu TB/ngày từ mức này trở xuống: "lẻ tẻ" (chỉ để mô tả)
BAC = [-0.10, 0.0, 0.05, 0.10]  # cận các bậc màu fill rate, tính so với mục tiêu muc_dv
MAU_BAC = ["#C2571A", "#EDBE9F", "#DDE3DC", "#8CC2B7", "#0F6E63"]

meta, arr = core.load_demo()
P = core.pairs(meta)
W, S = len(meta["stores"]), len(meta["skus"])
muc_dv, pt_tk = float(meta["env"]["muc_dv"]), float(meta["env"]["pt_tk"])
nhan_kho = [f"{k} · {c}" for k, c in P[["Kho", "Cửa hàng"]].drop_duplicates().values]

sm = core.summary(meta)
nhan = f"Tổng quan · {W} kho × {S} SKU · dữ liệu M5"
if sm is None:
    core.tieu_de(nhan, "Tổng quan mô phỏng trên miền test")
    core.no_data(f"results/summary_{meta['results_tag']}.json")
else:
    df, stat, raw = sm
    ip = df.loc["IPPO"]
    if "gap_percent" in stat:
        gap = stat["gap_percent"]
        cao_nhat = ip.fill_rate_mean >= df["fill_rate_mean"].max()
        core.tieu_de(nhan, f"IPPO giữ fill rate {core.pct(ip.fill_rate_mean)} — "
                           f"{'đổi lại ' if gap > 0 else ''}chi phí {'+' if gap > 0 else ''}"
                           f"{core.num(gap, 1)}% so với {stat['best_baseline']}",
                     f"Fill rate cao nhất trong {len(df)} chính sách." if cao_nhat else None)
    else:
        core.tieu_de(nhan, "Tổng quan mô phỏng trên miền test")
    c = st.columns(4)
    if "gap_percent" in stat:
        so_sanh = f"{'+' if gap > 0 else ''}{core.num(gap, 1)}% so với {stat['best_baseline']}"
        c[0].markdown(core.kpi("Tổng chi phí IPPO", core.num(ip.total_cost_mean),
                               so_sanh, "warn" if gap > 0 else "good"), unsafe_allow_html=True)
    else:
        c[0].markdown(core.kpi("Tổng chi phí IPPO", core.num(ip.total_cost_mean),
                               "chưa có dữ liệu so sánh"), unsafe_allow_html=True)
    c[1].markdown(core.kpi("Fill rate trung bình", core.pct(ip.fill_rate_mean),
                           f"Mục tiêu {core.pct(muc_dv, 0)}",
                           "good" if ip.fill_rate_mean >= muc_dv else "warn"), unsafe_allow_html=True)
    c[2].markdown(core.kpi("Hàng bị từ chối do tràn kho", core.num(ip.overflow_cost_mean / pt_tk),
                           f"Chi phí tràn kho / pt_tk ({core.num(pt_tk)} mỗi đơn vị)"),
                  unsafe_allow_html=True)
    c[3].markdown(core.kpi("Mức dùng sức chứa",
                           core.pct(ip.avg_inventory_mean / sum(raw["capacity_per_warehouse"])),
                           "Tồn kho TB toàn mạng / tổng sức chứa"), unsafe_allow_html=True)

with st.container(border=True, key="the_heatmap"):
    h1, h2 = st.columns([3, 2])
    h1.subheader("Fill rate theo cặp (kho, SKU)")
    with h2:
        cs = st.segmented_control("Chính sách", meta["policies"], default="IPPO",
                                  label_visibility="collapsed", key="cs_tq") or "IPPO"
    r = core.rollout(cs)
    dem = arr["demand"].reshape(len(arr["demand"]), -1).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        fr = np.where(dem > 0, 1 - r["stockout"].sum(axis=0) / dem, np.nan)
    co = ~np.isnan(fr)                                       # cặp có nhu cầu
    nguong = [round(muc_dv + d, 4) for d in BAC]
    bac = np.where(co, np.digitize(fr, nguong), np.nan)      # 0..4, NaN = ô trống
    h1.caption(f"{int((fr >= muc_dv).sum())}/{int(co.sum())} cặp đạt mục tiêu {core.pct(muc_dv, 0)} · "
               "rê chuột vào ô để xem tên SKU đầy đủ")
    hover = np.array([f"{nhan_kho[p // S]} · {meta['skus'][p % S]}<br>"
                      + (f"Fill rate {core.pct(fr[p])}" if co[p] else "Không có nhu cầu")
                      for p in range(W * S)]).reshape(W, S)
    nhom = [[s for s, sku in enumerate(meta["skus"]) if sku.split("_")[0] == c] for c in core.CATEGORIES]
    fig = make_subplots(rows=1, cols=len(nhom), shared_yaxes=True, horizontal_spacing=0.025,
                        column_widths=[len(g) for g in nhom], subplot_titles=core.CATEGORIES)
    fig.update_annotations(font=dict(size=12, color=core.INK))
    thang_mau = [[round((i + e) / 5, 2), m] for i, m in enumerate(MAU_BAC) for e in (0, 1)]
    for j, g in enumerate(nhom, 1):
        ten = [meta["skus"][s] for s in g]                   # x = tên đầy đủ để không trùng ô
        fig.add_heatmap(z=bac.reshape(W, S)[:, g], x=ten,
                        y=nhan_kho, text=hover[:, g], hovertemplate="%{text}<extra></extra>",
                        colorscale=thang_mau, zmin=-0.5, zmax=4.5, showscale=False, xgap=2, ygap=2,
                        row=1, col=j)
        fig.update_xaxes(tickvals=ten, ticktext=[t.rsplit("_", 1)[1] for t in ten], row=1, col=j)  # "090"
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(type="category", tickangle=0, tickfont=dict(size=10.5))
    core.chart(fig, 420)
    nhan_bac = ([f"< {core.pct(nguong[0], 0)}"]
                + [f"{core.num(100 * a)}–{core.pct(b, 0)}" for a, b in zip(nguong, nguong[1:])]
                + [f"≥ {core.pct(nguong[-1], 0)}"])
    so_cap = np.bincount(bac[co].astype(int), minlength=len(MAU_BAC))
    st.markdown('<div style="font-size:13px">' + " &nbsp;&nbsp; ".join(
        f'<span style="display:inline-block;width:14px;height:14px;border-radius:3px;'
        f'vertical-align:-2px;margin-right:6px;background:{MAU_BAC[i]}"></span>{nhan_bac[i]} '
        f'<span style="color:{core.MUTED}">· {so_cap[i]} cặp</span>' for i in reversed(range(len(MAU_BAC))))
        + ("" if co.all() else f' &nbsp;&nbsp; <span style="color:{core.MUTED}">ô trống: cặp không có nhu cầu</span>')
        + "</div>", unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1.container(border=True, key="the_suc_chua"):
    st.subheader(f"Mức sử dụng sức chứa theo kho · {cs}")
    u = r["util"].mean(axis=0)
    fig = go.Figure(go.Bar(
        x=u, y=nhan_kho, orientation="h", text=[core.pct(x) for x in u], textposition="outside",
        marker_color=[core.ORANGE if x >= UTIL_CANH_BAO else core.TEAL for x in u],
        hovertemplate="%{y}<br>%{x:.1%}<extra></extra>"))
    fig.add_vline(x=UTIL_CANH_BAO, line=dict(color=core.GREY, dash="dot"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickformat=".0%", range=[0, max(1.0, float(u.max()) * 1.15)])
    core.chart(fig, 400)
    st.caption(f"Kho dùng từ {core.pct(UTIL_CANH_BAO, 0)} sức chứa trở lên tô cam.")

with c2.container(border=True, key="the_thap_nhat"):
    st.subheader(f"5 cặp có fill rate thấp nhất · {cs}")
    cau_tb = arr["demand"].mean(axis=0).reshape(-1)          # cầu TB/ngày trên miền test
    thap = [p for p in np.argsort(fr) if co[p]][:5]
    F = fr.reshape(W, S)
    mo_ta = ([f"Đều là mặt hàng cầu lẻ tẻ (≤ {CAU_LE_TE} đơn vị/ngày)"]
             if all(cau_tb[p] <= CAU_LE_TE for p in thap) else [])
    mo_ta += [f"{sku} dưới {core.pct(nguong[0], 0)} ở cả {W} kho"
              for s, sku in enumerate(meta["skus"]) if np.all(F[:, s] < nguong[0])]
    if mo_ta:
        st.caption("; ".join(mo_ta))
    bang = pd.DataFrame({"Kho": [nhan_kho[p // S] for p in thap], "SKU": [meta["skus"][p % S] for p in thap],
                         "Fill rate": [fr[p] for p in thap],
                         "Cầu TB/ngày": ["< 0,1" if cau_tb[p] < 0.05 else core.num(cau_tb[p], 1) for p in thap]})
    st.dataframe(bang.style.format({"Fill rate": core.pct}).map(
        lambda v: "color: #A84A14; font-weight: 600" if v < muc_dv else "", subset=["Fill rate"]),
        hide_index=True)

core.nguon(*([f"results/summary_{meta['results_tag']}.json",
              f"trung bình {raw['meta']['n_episodes']} lượt đánh giá",
              f"checkpoint {raw['meta']['checkpoint']}"] if sm else []),
           f"biểu đồ: một lượt mô phỏng đại diện (hạt giống {meta['seed']})")
