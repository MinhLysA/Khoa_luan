"""Trang 5 - Kịch bản tăng cầu: nhân nhu cầu một khoảng ngày của miền test rồi chạy lại các
chính sách đã lưu (CHỈ SUY LUẬN, deterministic). Số ở đây là mô phỏng trực tiếp 1 lượt,
không phải số báo cáo (số báo cáo nhiều lượt nằm ở results/regime_analysis*.json)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core

meta, _ = core.load_demo()
dates = core.test_dates(meta)
P = core.pairs(meta)
store_of = dict(zip(P["Kho"], P["Cửa hàng"]))
muc_dv = float(meta["env"]["muc_dv"])
pols = meta["policies"]
bases = [p for p in pols if p != "IPPO"]
PAD = core.SOC_CUA_SO
s0 = int(np.clip(core.SOC_NGAY_M5 - meta["val_day"], 0, len(dates) - 1))   # chỉ số trong miền test
PRESETS = {"Đột biến ngắn": (2.0, 14), "Tăng kéo dài": (1.5, 28)}           # (hệ số, số ngày)
PHAM_VI = ["Toàn mạng", "Theo kho", "Theo ngành"]
HE_SO = [round(1 + k / 10, 1) for k in range(21)]                             # 1,0 … 3,0 bước 0,1


def pc(x):
    return "—" if x != x else core.pct(x)


def dau(x):
    return ("+" if x >= 0 else "") + core.num(x)


def ve_lai(v):
    return "không giảm dưới mục tiêu" if v is None else "chưa về lại" if v == float("inf") else f"{v} ngày"


def diem(v, d):
    """Điểm xếp hạng (lớn hơn = tốt hơn); d = 1: càng cao càng tốt, -1: càng thấp càng tốt."""
    return float("inf") if v is None else float("-inf") if v != v else d * v


def cham(mau, w=8):
    return (f'<span style="display:inline-block;width:{w}px;height:{w}px;border-radius:50%;'
            f'background:{mau};margin-right:8px"></span>')


def noi(xs):
    """["a"] -> "a"; ["a", "b", "c"] -> "a, b và c"."""
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " và " + xs[-1]


# (khóa trong shock_metrics, tên cột, hướng tốt, định dạng)
COT = [("truoc", "Fill rate trước", 1, pc), ("trong", "Trong sốc", 1, pc), ("sau", "Sau sốc", 1, pc),
       ("thieu_them", "Thiếu hàng tăng thêm", -1, core.num), ("chi_phi_them", "Chi phí tăng thêm", -1, dau),
       ("ve_lai", f"Về lại ≥ {core.pct(muc_dv, 0)} sau", -1, ve_lai)]
FMT = {k: f for k, _, _, f in COT}
HUONG = {k: d for k, _, d, _ in COT}

core.tieu_de("Kịch bản · nếu nhu cầu tăng bất thường", "Giả lập nhu cầu tăng đột biến",
             f"Nhân nhu cầu trong một khoảng ngày của miền test, rồi chạy lại cả {len(pols)} chính sách "
             f"với cùng chuỗi cầu. Không chính sách nào được báo trước: IPPO chỉ nhận ra qua lịch sử "
             f"nhu cầu {meta['env']['lookback']} ngày trong quan sát.")

f0, n0 = next(iter(PRESETS.values()))
for k, v in dict(sc_heso=f0, sc_ngay=dates[s0], sc_dai=n0, sc_pv=PHAM_VI[0]).items():
    st.session_state.setdefault(k, v)

trai, phai = st.columns([1, 2.6], gap="large")
with trai.container(border=True, key="the_thiet_lap"):
    st.markdown("**Kịch bản có sẵn**")
    chay = False
    for i, (ten, (f, n)) in enumerate(PRESETS.items()):
        if st.button(f"{ten} · × {core.num(f, 1)} · {n} ngày", key=f"ps_{i}", width="stretch"):
            st.session_state.update(sc_heso=f, sc_ngay=dates[s0], sc_dai=n, sc_pv=PHAM_VI[0])
            chay = True
    hien = tuple(st.session_state[k] for k in ("sc_heso", "sc_dai", "sc_ngay", "sc_pv"))
    for i, (f, n) in enumerate(PRESETS.values()):
        if hien == (f, n, dates[s0], PHAM_VI[0]):                                  # tô kịch bản đang chọn
            st.markdown(f"<style>.st-key-ps_{i} button {{background:{core.TEAL_SOFT};"
                        f"border:1.5px solid {core.TEAL}}}</style>", unsafe_allow_html=True)
    st.caption(f"Cả hai bắt đầu {dates[s0]:%d/%m/%Y} (ngày {core.SOC_NGAY_M5} theo chỉ số ngày M5, "
               f"mặc định của phân tích sốc cầu).")
    st.divider()
    he_so = st.select_slider("Hệ số nhân nhu cầu", HE_SO, format_func=lambda x: f"× {core.num(x, 1)}",
                             key="sc_heso")
    c1, c2 = st.columns(2)
    ngay = c1.date_input("Bắt đầu", min_value=dates[0], max_value=dates[-1], format="DD/MM/YYYY",
                         key="sc_ngay")
    dai = c2.number_input("Kéo dài (ngày)", 3, 56, step=1, key="sc_dai")
    pv = st.segmented_control("Phạm vi", PHAM_VI, required=True, key="sc_pv", width="stretch")
    chon = []
    if pv == "Theo kho":
        chon = st.multiselect("Kho", list(store_of), format_func=lambda k: f"{k} · {store_of[k]}", key="sc_kho")
    elif pv == "Theo ngành":
        chon = st.multiselect("Ngành hàng", core.CATEGORIES, key="sc_nganh")
    chay |= st.button("Chạy mô phỏng", type="primary", width="stretch")
    st.caption(f"Khoảng 3–5 giây cho {len(pols)} chính sách · chỉ suy luận, không huấn luyện")

    if chay:
        s, n = ((ngay - dates[0]).days if ngay else -1), int(dai)
        if s < 0 or s + n > len(dates):
            st.warning(f"Khoảng sốc phải nằm trong miền test ({dates[0]:%d/%m/%Y} – {dates[-1]:%d/%m/%Y}): "
                       f"chọn ngày bắt đầu sớm hơn hoặc rút ngắn số ngày.")
        elif pv != PHAM_VI[0] and not chon:
            st.warning("Chọn ít nhất một kho hoặc ngành hàng cho phạm vi đã chọn.")
        else:
            kho = tuple(chon) if pv == "Theo kho" else None
            nganh = tuple(chon) if pv == "Theo ngành" else None
            with st.spinner(f"Đang chạy mô phỏng {len(pols)} chính sách…"):
                kq = {p: (core.shock_rollout(p, s, n, float(he_so), kho, nganh),
                          core.shock_rollout(p, 0, 0, 1.0)) for p in pols}       # (có sốc, không sốc)
            st.session_state.soc = dict(f=float(he_so), s=s, n=n, kq=kq, pham_vi=(
                "toàn mạng" if not chon else ("kho " if kho else "ngành ") + ", ".join(chon)))

sh = core.shock_results()
if sh:
    st.page_link("pages_/so_sanh.py", label="Xem số báo cáo nhiều lượt: trang So sánh chính sách, thẻ “Sốc cầu”")
core.nguon(f"mô phỏng trực tiếp 1 lượt (hạt giống {meta['seed']}) trên app/demo_data/, không phải số báo cáo",
           f"số báo cáo nhiều lượt: results/{sh[0]}" if sh
           else "số báo cáo nhiều lượt (results/regime_analysis_*.json): chưa có dữ liệu")

soc = st.session_state.get("soc")
if soc is None:
    phai.info("Chọn một kịch bản có sẵn, hoặc tự đặt thông số rồi bấm “Chạy mô phỏng”.")
    st.stop()

# --------------------------------------------------------------------------- #
# Kết quả (giữ trong session_state: đổi baseline so sánh không chạy lại)
# --------------------------------------------------------------------------- #
f, s, n, kq = soc["f"], soc["s"], soc["n"], soc["kq"]
e = s + n
m = {p: core.shock_metrics(*kq[p], s, n, muc_dv, PAD) for p in pols}
tot_nhat = {k: max(diem(m[p][k], d) for p in pols) for k, d in HUONG.items()}
sm = core.summary(meta)
tot = sm[1].get("best_baseline") if sm else None
mac_dinh = tot if tot in bases else "Newsvendor" if "Newsvendor" in bases else bases[0]


def la_tot(p, k):
    return diem(m[p][k], HUONG[k]) == tot_nhat[k]


# Câu kết luận theo luật: ai tốt nhất ở từng tiêu chí (kể cả đồng hạng), gộp lời khi trùng người
nhom = {}
for k, cau in [("trong", "giữ fill rate trong sốc cao nhất ({})"),
               ("chi_phi_them", "phát sinh thêm ít chi phí nhất ({})"),
               ("thieu_them", "thiếu thêm ít hàng nhất ({} đơn vị)")]:
    ai = tuple(p for p in pols if la_tot(p, k))
    nhom.setdefault(ai, []).append(cau.format(FMT[k](m[ai[0]][k])))
ket_luan = "; ".join(f"{noi(ai)} {noi(c)}" for ai, c in nhom.items()) + "."

with phai:
    st.markdown(f"""<style>
.kpi .num {{ font-size: 1.6rem; }}
.st-key-sc_so_voi button[kind="segmented_controlActive"]:not(:disabled) {{ background: {core.ORANGE};
  border-color: {core.ORANGE}; color: #FFFFFF; }}
table.bang {{ width: 100%; border-collapse: collapse; border: none; font-size: 14px;
  font-variant-numeric: tabular-nums; }}
table.bang tr {{ border: none; }}
table.bang th, table.bang td {{ border: none; border-bottom: 1px solid #EEF0EC; padding: 10px 8px;
  text-align: right; font-weight: 400; }}
table.bang thead th {{ color: {core.MUTED}; font-size: 12px; font-weight: 500;
  border-bottom: 1px solid {core.BORDER}; }}
table.bang th:first-child {{ text-align: left; }}
table.bang tbody th {{ font-weight: 600; }}
table.bang .tot {{ font-weight: 700; }}
</style>
<div style="background:{core.TEAL_SOFT};border-radius:10px;padding:16px 20px">
<div style="font-size:12px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#0A4F47">
Nhu cầu × {core.num(f, 1)} · {n} ngày từ {dates[s]:%d/%m/%Y} · {soc['pham_vi']}</div>
<div style="font-size:17px;font-weight:500;line-height:1.5">{ket_luan}</div></div>""",
                unsafe_allow_html=True)

    g1, g2 = st.columns([3, 2], vertical_alignment="bottom")
    base = g2.segmented_control("So với", bases, default=mac_dinh, required=True, key="sc_so_voi")
    g1.markdown(" &nbsp;&nbsp; ".join(
        f'<span style="display:inline-block;width:18px;height:{h}px;background:{mau};border-radius:2px;'
        f'vertical-align:middle;margin-right:6px"></span>{ten}'
        for mau, h, ten in [(core.TEAL, 3, "IPPO"), (core.ORANGE, 3, base),
                            (core.GREY, 2, f"{len(bases) - 1} baseline còn lại")]), unsafe_allow_html=True)

    for col, k, nhan in zip(st.columns(4), ["trong", "thieu_them", "chi_phi_them", "ve_lai"],
                            ["Fill rate trong sốc", "Thiếu hàng tăng thêm", "Chi phí tăng thêm",
                             f"Về lại fill rate ≥ {core.pct(muc_dv, 0)} sau"]):
        col.markdown(core.kpi(nhan, cham(core.TEAL) + FMT[k](m["IPPO"][k]),
                              cham(core.ORANGE) + f"{base}: {FMT[k](m[base][k])}"), unsafe_allow_html=True)

    lo, hi = max(s - PAD, 0), min(e + PAD, len(dates))
    x = dates[lo:hi]
    nua = pd.Timedelta(hours=12)                                    # mép dải sốc phủ trọn ngày
    x0, x1 = pd.Timestamp(dates[s]) - nua, pd.Timestamp(dates[e - 1]) + nua

    def khung(fig, y_title, **y):
        """Dải sốc (nền cam nhạt) + trục chung cho hai biểu đồ."""
        fig.add_vrect(x0=x0, x1=x1, fillcolor="#F6E3D6", line_width=0, layer="below",
                      annotation_text=f"Sốc × {core.num(f, 1)}", annotation_position="top left",
                      annotation_font=dict(color="#A84A14", size=12))
        fig.update_xaxes(tickformat="%d/%m", hoverformat="%d/%m/%Y")
        fig.update_yaxes(title=y_title, **y)
        fig.update_layout(hovermode="x unified")
        return fig

    with st.container(border=True, key="the_fill_soc"):
        st.subheader("Fill rate toàn mạng theo ngày")
        st.caption(f"{dates[lo]:%d/%m} – {dates[hi - 1]:%d/%m/%Y} · {PAD} ngày trước và sau sốc")
        fig = go.Figure()
        for p in [q for q in bases if q != base] + [base, "IPPO"]:          # IPPO vẽ sau cùng (trên cùng)
            mau, w = (core.TEAL, 2.5) if p == "IPPO" else (core.ORANGE, 2) if p == base else (core.GREY, 1.2)
            y = 1 - kq[p][0]["stockout"][lo:hi] / kq[p][0]["demand"][lo:hi]
            fig.add_scatter(x=x, y=y, name=p, line=dict(color=mau, width=w),
                            hovertemplate=f"{p}: %{{y:.1%}}<extra></extra>")
        fig.add_hline(y=muc_dv, line=dict(color=core.INK, dash="dash", width=1.5),
                      annotation_text=f"mục tiêu {core.pct(muc_dv, 0)}", annotation_position="top right")
        core.chart(khung(fig, "Fill rate (%)", tickformat=".0%"), 340)

    with st.container(border=True, key="the_dat_hang_soc"):
        st.subheader("Lượng đặt hàng so với nhu cầu")
        st.caption(f"Tổng {len(P)} cặp, đơn vị hàng/ngày · cho thấy chính sách phản ứng nhanh hay chậm")
        fig = go.Figure(go.Scatter(x=x, y=kq["IPPO"][0]["demand"][lo:hi], name="Nhu cầu toàn mạng",
                                   fill="tozeroy", fillcolor="#E2E6E0", line=dict(width=0),
                                   hovertemplate="Nhu cầu: %{y:,.0f}<extra></extra>"))
        for p, mau, w in [(base, core.ORANGE, 2), ("IPPO", core.TEAL, 2.5)]:
            fig.add_scatter(x=x, y=kq[p][0]["order"][lo:hi], name=f"Đặt hàng · {p}", line=dict(color=mau, width=w),
                            hovertemplate=f"Đặt hàng · {p}: %{{y:,.0f}}<extra></extra>")
        core.chart(khung(fig, "Đơn vị hàng/ngày", tickformat=",d"), 300)

    with st.container(border=True, key="the_bang_soc"):
        st.subheader(f"Cả {len(pols)} chính sách")
        dong = ""
        for p in pols:
            mau = core.TEAL if p == "IPPO" else core.ORANGE if p == base else core.GREY
            nen = core.TEAL_SOFT if p == "IPPO" else "#FBF1EA" if p == base else "transparent"
            o = "".join(f'<td class="{"tot" if la_tot(p, k) else ""}">{fmt(m[p][k])}</td>'
                        for k, _, _, fmt in COT)
            dong += f'<tr style="background:{nen}"><th>{cham(mau, 10)}{p}</th>{o}</tr>'
        dau_bang = "".join(f"<th>{t}</th>" for _, t, _, _ in COT)
        st.markdown(f'<table class="bang"><thead><tr><th>Chính sách</th>{dau_bang}</tr></thead>'
                    f'<tbody>{dong}</tbody></table>', unsafe_allow_html=True)
        st.caption(f"Đậm: tốt nhất ở cột đó. “Tăng thêm” so với chính chính sách đó khi không có sốc, tính từ "
                   f"ngày sốc bắt đầu tới {PAD} ngày sau khi sốc kết thúc. “Về lại ≥ {core.pct(muc_dv, 0)}”: "
                   f"số ngày từ lúc sốc bắt đầu tới khi fill rate trung bình 7 ngày trở lại mục tiêu sau lần "
                   f"đầu xuống dưới.")
