"""Trang 1 - Kết quả: các câu hỏi chính của đề tài, mỗi câu một thẻ thanh ngang.
Số liệu đọc từ results/ (đánh giá mô hình đã huấn luyện), câu trả lời sinh từ số liệu, không hardcode."""
import numpy as np
import streamlit as st

import core

meta, _ = core.load_demo()
muc_dv = float(meta["env"]["muc_dv"])
so_ngay = len(core.test_dates(meta))
sm = core.summary(meta)
ms = core.read_results("multiseed_main.json")
qm = core.read_results("scale_group_v2_main.json")
rq3 = core.read_results("rq3_multiseed.json")
soc = core.shock_results()

# Nhãn lời thường cho nhóm quy mô cầu và biến thể phần thưởng (chỉ là tên, số vẫn đọc từ file)
NHOM = {"Rất thấp (<0,5)": ("Bán rất chậm", "dưới 0,5 sản phẩm/ngày"), "Thấp (0,5-2)": ("Bán chậm", "0,5–2 sản phẩm/ngày"),
        "Trung bình (2-10)": ("Bán vừa", "2–10 sản phẩm/ngày"), "Cao (>=10)": ("Bán chạy", "từ 10 sản phẩm/ngày")}
RQ3 = {"abl_ref": ("Cách của đề tài", "thưởng riêng từng mặt hàng"),
       "abl_global": ("Thưởng chung", "cả mạng chung một điểm thưởng"),
       "abl_q3": ("Bỏ chuẩn hóa thưởng", "không cân theo quy mô mặt hàng")}


def dau(x, d=1):
    return ("+" if x > 0 else "−" if x < 0 else "") + core.num(abs(x), d) + "%"


core.tieu_de(f"Kết quả · {len(meta['stores'])} cửa hàng × {len(meta['skus'])} mặt hàng",
             "Kết quả đề tài",
             f"Mô hình IPPO đã huấn luyện tự đặt hàng trong {so_ngay} ngày bán hàng chưa dùng khi học, "
             f"đặt cạnh {len(sm[0]) - 1 if sm else 6} cách đặt hàng truyền thống trong cùng điều kiện.")
if sm is None:
    core.no_data("kết quả đánh giá mô hình chính")
    st.stop()

df, stat, raw = sm
per = stat.get("per_baseline", {})
bases = [p for p in df.index if p != "IPPO"]
ip = df.loc["IPPO"]
re_nhat = min(bases, key=lambda p: df.loc[p, "total_cost_mean"])
cao_nhat = ip.fill_rate_mean >= df["fill_rate_mean"].max()

# --------------------------------------------------------------------------- #
# Bốn số chính
# --------------------------------------------------------------------------- #
k = st.columns(4)
k[0].markdown(core.kpi("Đáp ứng nhu cầu khách", core.pct(ip.fill_rate_mean),
                       f"Cao nhất trong {len(df)} cách" if cao_nhat else f"Mục tiêu {core.pct(muc_dv, 0)}",
                       "good" if ip.fill_rate_mean >= muc_dv else "warn"), unsafe_allow_html=True)
k[1].markdown(core.kpi("Tổng chi phí", core.num(ip.total_cost_mean),
                       "đơn vị chi phí trong mô phỏng"), unsafe_allow_html=True)
for col, b in zip(k[2:], ["EOQ", "(s,S)"]):
    if b in per:
        g = per[b]["gap_percent"]
        col.markdown(core.kpi(f"So với {b}", f"{'rẻ hơn' if g < 0 else 'đắt hơn'} {core.num(abs(g), 1)}%",
                              f"rẻ hơn ở {per[b]['n_ippo_cheaper']}/{per[b]['n']} lượt chạy",
                              "good" if g < 0 else "warn"), unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# 1. Chi phí   2. Đáp ứng nhu cầu
# --------------------------------------------------------------------------- #
c1, c2 = st.columns(2, gap="medium")
with c1:
    re_, dat_, ngang = [], [], []
    dong = []
    for p in df.sort_values("total_cost_mean").index:
        t, phu = core.ten_cs(p)
        chip = ""
        if p != "IPPO" and p in per:
            g, ro = per[p]["gap_percent"], per[p]["p_paired"] < 0.05
            if not ro:
                chip = core.chip("ngang IPPO")
                ngang.append(p)
            elif g < 0:
                chip = core.chip(f"IPPO rẻ hơn {core.num(-g, 1)}%", "good")
                re_.append(p)
            else:
                chip = core.chip(f"IPPO đắt hơn {core.num(g, 1)}%", "warn")
                dat_.append(p)
        dong.append(dict(ten=t, phu=chip or "mô hình của đề tài", v=df.loc[p, "total_cost_mean"],
                         nhan=core.num(df.loc[p, "total_cost_mean"]), me=p == "IPPO"))
    dap = []
    if re_:
        dap.append(f"IPPO <b>rẻ hơn rõ</b> {', '.join(re_)}")
    if dat_:
        dap.append(f"<b>đắt hơn</b> {', '.join(dat_)}")
    if ngang:
        dap.append(f"ngang {', '.join(ngang)}")
    core.the_hoi("q1", "IPPO có đặt hàng rẻ hơn không?", "; ".join(dap) + ".", core.thanh(dong),
                 f"Tổng chi phí lưu kho, thiếu hàng, đặt hàng và hàng bị trả do kho đầy; trung bình "
                 f"{raw['meta']['n_episodes']} lượt chạy. Thanh ngắn hơn là rẻ hơn.")

with c2:
    lo = 0.7
    dong = [dict(ten=core.ten_cs(p)[0], phu=core.ten_cs(p)[1], v=df.loc[p, "fill_rate_mean"],
                 nhan=core.pct(df.loc[p, "fill_rate_mean"]), me=p == "IPPO")
            for p in df.sort_values("fill_rate_mean", ascending=False).index]
    khac = df.loc[bases, "fill_rate_mean"]
    core.the_hoi("q2", "IPPO có phục vụ khách tốt hơn không?",
                 f"IPPO đáp ứng <b>{core.pct(ip.fill_rate_mean)}</b> lượng hàng khách cần"
                 + (", <b>cao nhất</b>" if cao_nhat else "")
                 + f"; các cách truyền thống từ {core.pct(khac.min())} đến {core.pct(khac.max())}.",
                 '<div class="chu-giai"><span class="me">IPPO</span><span>cách truyền thống</span>'
                 f'<span class="tg">mục tiêu {core.pct(muc_dv, 0)}</span></div>'
                 + core.thanh(dong, lo=lo, hi=1.0, moc=muc_dv),
                 f"Thanh tính từ {core.pct(lo, 0)} để thấy rõ chênh lệch. Thanh dài hơn là ít hết hàng hơn.")

# --------------------------------------------------------------------------- #
# 3. Theo tốc độ bán   4. Khi nhu cầu tăng đột ngột
# --------------------------------------------------------------------------- #
c1, c2 = st.columns(2, gap="medium")
with c1:
    if qm:
        so = next(p for p in qm if p != "IPPO")
        dong, tot, kem = [], [], []
        for gr, v in qm["IPPO"].items():
            ten, phu = NHOM.get(gr, (gr, ""))
            a, b = v["fill"], qm[so][gr]["fill"]
            (tot if a > b else kem).append(f"{ten.lower()} ({'+' if a > b else '−'}{core.num(abs(a - b) * 100, 1)} điểm)")
            dong += [dict(nhom=f"{ten} · {phu} · {v['n_pairs']} mặt hàng-cửa hàng"),
                     dict(ten="IPPO", v=a, nhan=core.pct(a), me=True), dict(ten=so, v=b, nhan=core.pct(b))]
        dap = []
        if tot:
            dap.append(f"IPPO <b>tốt hơn</b> {so} ở hàng {', '.join(tot)}")
        if kem:
            dap.append(f"<b>kém hơn</b> ở hàng {', '.join(kem)}")
        core.the_hoi("q3", "Hàng bán chậm và hàng bán chạy khác nhau thế nào?", "; ".join(dap) + ".",
                     core.thanh(dong, lo=0.6, hi=1.0, moc=muc_dv),
                     f"Mức đáp ứng nhu cầu theo tốc độ bán của mặt hàng; thanh tính từ 60%. Vạch đỏ: mục tiêu {core.pct(muc_dv, 0)}.")
with c2:
    if soc:
        _, b2 = soc
        f = next(iter(b2["results"]))
        kq = {p: v for p, v in b2["results"][f].items() if "cùng mức" not in p}
        trong = {p: v["Trong sốc"]["fill_rate"] for p, v in kq.items()}
        cpd = {p: v["Trong sốc"]["cost_per_day"] for p, v in kq.items()}
        dong = [dict(ten=core.ten_cs(p)[0], v=trong[p], nhan=core.pct(trong[p]), me=p == "IPPO",
                     phu=core.chip(f"chi phí/ngày {core.num(cpd[p])}", "good" if cpd[p] == min(cpd.values()) else ""))
                for p in sorted(kq, key=lambda p: -trong[p])]
        hang = sorted(trong.values(), reverse=True).index(trong["IPPO"]) + 1
        dap = (f"Khi nhu cầu tăng ×{core.num(float(f), 1)} trong {b2['len']} ngày, IPPO vẫn đáp ứng "
               f"<b>{core.pct(trong['IPPO'])}</b> (đứng thứ {hang}/{len(trong)})"
               + (" với <b>chi phí mỗi ngày thấp nhất</b>" if cpd["IPPO"] == min(cpd.values()) else "") + ".")
        core.the_hoi("q4", "Khi nhu cầu tăng đột ngột thì sao?", dap, core.thanh(dong, lo=0.6, hi=1.0, moc=muc_dv),
                     "Mức đáp ứng trong những ngày nhu cầu tăng; không cách nào được báo trước. "
                     "Thử kịch bản khác ở trang Kịch bản tăng cầu.")

# --------------------------------------------------------------------------- #
# 5. Độ ổn định   6. Thiết kế phần thưởng
# --------------------------------------------------------------------------- #
c1, c2 = st.columns(2, gap="medium")
with c1:
    if ms:
        cs = [ms["seeds"][s]["cost"] for s in ms["seeds"]]
        dong = [dict(ten=f"Lần huấn luyện {i}", v=c, nhan=core.num(c), me=True) for i, c in enumerate(cs, 1)]
        ca3 = [b for b, g in ms["gap_vs_baseline"].items() if g["n_seeds_re_hon"] == g["n_seeds"]]
        core.the_hoi("q5", "Huấn luyện lại thì kết quả có giữ được không?",
                     f"{len(cs)} lần huấn luyện độc lập cho chi phí chênh nhau tối đa "
                     f"<b>{core.num((max(cs) - min(cs)) / np.mean(cs) * 100, 1)}%</b>"
                     + (f"; IPPO rẻ hơn {', '.join(ca3)} ở <b>cả {len(cs)} lần</b>." if ca3 else "."),
                     core.thanh(dong, lo=0, hi=max(cs) * 1.05),
                     "Mỗi lần huấn luyện bắt đầu từ một khởi tạo ngẫu nhiên khác.")
with c2:
    if rq3:
        tb = {t: np.mean([r["cost"] for r in v["seeds"].values()]) for t, v in rq3.items()}
        goc = tb.get("abl_ref")
        dong = [dict(ten=RQ3.get(t, (t, ""))[0], v=c, nhan=core.num(c), me=t == "abl_ref",
                     phu=RQ3.get(t, (t, ""))[1] + (" " + core.chip(f"gấp {core.num(c / goc, 1)} lần", "warn")
                                                   if goc and t != "abl_ref" else ""))
                for t, c in tb.items()]
        xau = [(RQ3.get(t, (t, ""))[0].lower(), c / goc) for t, c in tb.items() if goc and t != "abl_ref"]
        core.the_hoi("q6", "Cách tính điểm thưởng khi học có quan trọng không?",
                     "Có: " + "; ".join(f"{t} làm chi phí <b>gấp {core.num(r, 1)} lần</b>" for t, r in xau) + ".",
                     core.thanh(dong),
                     f"Các biến thể học cùng số lượt, trung bình {len(next(iter(rq3.values()))['seeds'])} lần huấn luyện.")
