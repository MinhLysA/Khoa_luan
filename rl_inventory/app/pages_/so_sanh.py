"""Trang 4 - So sánh chi tiết: cơ cấu chi phí và chênh lệch IPPO so với từng chính sách truyền thống.
Số liệu đọc từ results/ (không hardcode)."""
import streamlit as st

import core

meta, _ = core.load_demo()
muc_dv = float(meta["env"]["muc_dv"])
sm = core.summary(meta)
iso = core.read_results("iso_service.json")
MAU = ["var(--bar)", "#EF8A3C", "#4361F5", "#9B7BE0"]          # lưu kho, thiếu hàng, đặt hàng, tràn kho

core.tieu_de("So sánh · số liệu chi tiết", "So sánh chi tiết",
             "Chi phí của từng cách đặt hàng gồm những gì, và IPPO chênh bao nhiêu so với từng cách.")
if sm is None:
    core.no_data("kết quả đánh giá mô hình chính")
    st.stop()
df, stat, raw = sm
per = stat.get("per_baseline", {})

che_do = st.segmented_control("So sánh", ["Mỗi cách dùng thông số tốt nhất", "Các cách khác phục vụ ngang IPPO"],
                              default="Mỗi cách dùng thông số tốt nhất", required=True, label_visibility="collapsed")

if che_do.startswith("Mỗi cách"):
    c1, c2 = st.columns([1.1, 1], gap="medium")
    with c1:
        thu_tu = df.sort_values("total_cost_mean").index
        cao = df["total_cost_mean"].max()
        dong = ""
        for p in thu_tu:
            r = df.loc[p]
            doan = "".join(f'<span title="{ten}: {core.num(r[f"{k}_mean"])}" style="width:{r[f"{k}_mean"] / r.total_cost_mean * 100:.2f}%;'
                           f'background:{m}"></span>' for (k, ten), m in zip(core.COST_PARTS, MAU))
            dong += (f'<div class="stk-r{" me" if p == "IPPO" else ""}"><div class="n">{p}</div>'
                     f'<div class="stk" style="width:{r.total_cost_mean / cao * 100:.1f}%">{doan}</div>'
                     f'<div class="v">{core.num(r.total_cost_mean)}</div></div>')
        chu = "".join(f'<span><i style="background:{m}"></i>{ten}</span>' for (_, ten), m in zip(core.COST_PARTS, MAU))
        ip = df.loc["IPPO"]
        core.the_hoi("co_cau", "Chi phí gồm những gì?",
                     f"IPPO giữ hàng nhiều hơn (lưu kho {core.pct(ip.holding_cost_mean / ip.total_cost_mean, 0)} "
                     f"tổng chi phí) để ít bị thiếu hàng (thiếu hàng {core.pct(ip.stockout_cost_mean / ip.total_cost_mean, 0)}).",
                     f'<div class="chu-mau">{chu}</div><div class="stk-ds">{dong}</div>',
                     f"Trung bình {raw['meta']['n_episodes']} lượt chạy. Rê chuột vào từng đoạn để xem số.")
    with c2:
        hang = ""
        for p, t in sorted(per.items(), key=lambda x: x[1]["gap_percent"]):
            g, ro = t["gap_percent"], t["p_paired"] < 0.05
            kl = (core.chip("rẻ hơn rõ", "good") if ro and g < 0 else
                  core.chip("đắt hơn rõ", "warn") if ro else core.chip("ngang nhau"))
            hang += (f"<tr><th>{p}</th><td>{'+' if g > 0 else '−'}{core.num(abs(g), 1)}%</td>"
                     f"<td>{t['n_ippo_cheaper']}/{t['n']}</td><td>{kl}</td></tr>")
        core.the_hoi("bang", "IPPO chênh bao nhiêu so với từng cách?",
                     "Âm là IPPO rẻ hơn. “Rõ” nghĩa là chênh lệch lặp lại qua các lượt chạy, không phải do may rủi.",
                     '<table class="bang"><thead><tr><th>Cách đặt hàng</th><th>Chi phí IPPO chênh</th>'
                     f'<th>Số lượt IPPO rẻ hơn</th><th>Kết luận</th></tr></thead><tbody>{hang}</tbody></table>',
                     f"Mỗi lượt chạy, mọi cách đặt hàng gặp cùng một chuỗi nhu cầu; {raw['meta']['n_episodes']} lượt.")
elif iso is None:
    core.no_data("so sánh khi các cách khác phục vụ ngang IPPO")
else:
    ip = iso["ket_qua"]["IPPO"]
    dong, vuot = [dict(ten="IPPO", phu=f"đáp ứng {core.pct(ip['fill'])}", v=ip["cost"], nhan=core.num(ip["cost"]), me=True)], False
    for p, r in iso["ket_qua"].items():
        if p == "IPPO":
            continue
        if r.get("cost") is None:
            dong.append(dict(ten=p, phu="không đạt được mức đáp ứng này", v=0, nhan="—"))
            continue
        vuot |= r["fill"] > iso["target_fill"] + 0.01
        g = (ip["cost"] / r["cost"] - 1) * 100
        dong.append(dict(ten=p, phu=f"đáp ứng {core.pct(r['fill'])}", v=r["cost"], nhan=core.num(r["cost"]),
                         chip=core.chip(f"IPPO {'rẻ hơn' if g < 0 else 'đắt hơn'} {core.num(abs(g), 1)}%",
                                        "good" if g < 0 else "warn")))
    core.the_hoi("iso", "Nếu các cách khác cũng phục vụ khách tốt như IPPO thì sao?",
                 f"Các cách truyền thống được chỉnh để đáp ứng ít nhất {core.pct(iso['target_fill'])} nhu cầu như IPPO; "
                 "khi đó chúng tốn hơn nhiều.",
                 core.thanh(dong),
                 "Các cách này đáp ứng cao hơn mục tiêu một chút (bước chỉnh thông số còn thô), "
                 "nên mức chênh ở đây có phần nghiêng về IPPO." if vuot else "")
