"""Trang 4 - So sánh chính sách: số liệu đọc từ results/ (không hardcode)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core

meta, _ = core.load_demo()
muc_dv = float(meta["env"]["muc_dv"])
sm = core.summary(meta)
iso = core.read_results("iso_service.json")
shock = core.shock_results()

st.title("So sánh chính sách trên miền test (mô phỏng)")
tabs = st.tabs(["So sánh"] + (["Sốc cầu"] if shock else []))

with tabs[0]:
    che_do = st.radio("Chế độ", ["So thẳng", "Cùng mức phục vụ"], horizontal=True)
    rows = None
    if che_do == "So thẳng":
        if sm is None:
            core.no_data(f"results/summary_{meta['results_tag']}.json")
        else:
            df, stat, raw = sm
            rows = [(n, df.loc[n, "total_cost_mean"], df.loc[n, "fill_rate_mean"])
                    for n in meta["policies"] if n in df.index]
            st.caption(f"Mỗi chính sách giữ tham số đã tinh chỉnh; trung bình "
                       f"{raw['meta']['n_episodes']} lượt đánh giá · "
                       f"results/summary_{meta['results_tag']}.json")
    elif iso is None:
        core.no_data("results/iso_service.json (so sánh ở cùng mức phục vụ chưa được chạy)")
    else:
        rows = [(n, r["cost"], r["fill"]) for n, r in iso["ket_qua"].items()]
        st.caption(f"Baseline được chỉnh để đạt fill rate ≥ {core.pct(iso['target_fill'], 2)} "
                   f"(mức của IPPO trừ dung sai) · results/iso_service.json")

    if rows:
        for col, (n, cost, fill) in zip(st.columns(len(rows)), rows):
            if cost is None:
                col.markdown(core.kpi(n, "—", "không đạt được mức phục vụ này", "warn"),
                             unsafe_allow_html=True)
            else:
                col.markdown(core.kpi(n, core.num(cost), f"Fill rate {core.pct(fill)}",
                                      "good" if fill >= muc_dv else "warn"), unsafe_allow_html=True)

        co = [r for r in rows if r[1] is not None]
        fig = go.Figure(go.Scatter(
            x=[r[2] for r in co], y=[r[1] for r in co], text=[r[0] for r in co],
            mode="markers+text", textposition="top center",
            marker=dict(size=14, color=[core.policy_color(r[0]) for r in co]),
            hovertemplate="%{text}<br>Fill rate %{x:.2%}<br>Chi phí %{y:,.0f}<extra></extra>"))
        fig.add_vline(x=muc_dv, line=dict(color=core.ORANGE, dash="dash"),
                      annotation_text=f"Mục tiêu {core.pct(muc_dv, 0)}")
        fig.update_xaxes(title="Fill rate", tickformat=".0%")
        fig.update_yaxes(title="Tổng chi phí (đơn vị mô phỏng)")
        fig.update_layout(title="Chi phí theo fill rate (tốt = phía dưới, bên phải)")

    if che_do == "So thẳng" and sm is not None:
        df = sm[0].loc[[r[0] for r in rows]]
        bar = go.Figure([go.Bar(x=df.index, y=df[f"{c}_mean"], name=ten, marker_color=mau)
                         for (c, ten), mau in zip(core.COST_PARTS,
                                                  [core.GREY, core.ORANGE, core.TEAL, core.INK])])
        bar.update_layout(barmode="stack", title="Cơ cấu chi phí")
        bar.update_yaxes(title="Chi phí (đơn vị mô phỏng)")
        g1, g2 = st.columns(2)
        g1.plotly_chart(core.style_fig(bar, 420), theme=None)
        g2.plotly_chart(core.style_fig(fig, 420), theme=None)
    elif rows:
        st.plotly_chart(core.style_fig(fig, 420), theme=None)
        st.caption("Cơ cấu chi phí: chưa có dữ liệu ở chế độ này "
                   "(iso_service.json chỉ lưu tổng chi phí và fill rate).")

    if che_do == "So thẳng" and sm is not None:
        per = sm[1].get("per_baseline")
        st.subheader("Kiểm định thống kê: IPPO so với từng baseline")
        if not per:
            core.no_data("kết quả kiểm định trong results/")
        else:
            st.dataframe(pd.DataFrame([{
                "Baseline": n,
                "Chênh chi phí": f"{'+' if t['gap_percent'] > 0 else ''}{core.num(t['gap_percent'], 2)}%",
                "Δ TB (IPPO − baseline)": core.num(t["mean_diff"]),
                "CI 95% của Δ": f"[{core.num(t['ci95_low'])}; {core.num(t['ci95_high'])}]",
                "p (t-test theo cặp)": f"{t['p_paired']:.1e}",
                "p (Wilcoxon)": f"{t['p_wilcoxon']:.1e}",
                "Cohen's d_z": core.num(t["d_z"], 2),
                "Số lượt IPPO rẻ hơn": f"{t['n_ippo_cheaper']}/{t['n']}",
            } for n, t in per.items()]), hide_index=True)
            st.caption("Mọi chính sách được đánh giá trên CÙNG các hạt giống (cùng chuỗi cầu, cùng "
                       "lead time ngẫu nhiên), nên dùng kiểm định THEO CẶP trên Δ = chi phí IPPO − "
                       "chi phí baseline: t-test theo cặp, Wilcoxon signed-rank, d_z = Δ̄ / s_Δ. "
                       "Δ > 0: IPPO đắt hơn. Ý nghĩa thống kê khi p < 0,05.")

if shock:
    ten_file, b2 = shock
    with tabs[1]:
        st.caption(f"Nguồn: results/{ten_file} · sốc bắt đầu ngày thứ {b2['start']} của miền test, "
                   f"kéo dài {b2['len']} ngày")
        for he_so, theo_cs in b2["results"].items():
            st.subheader(f"Nhu cầu × {he_so}")
            st.dataframe(pd.DataFrame([
                {"Chính sách": cs, "Giai đoạn": gd, "Fill rate": core.pct(v["fill_rate"]),
                 "Chi phí / ngày": core.num(v["cost_per_day"]), "Số ngày": v["n_days"]}
                for cs, theo_gd in theo_cs.items() for gd, v in theo_gd.items()]), hide_index=True)
