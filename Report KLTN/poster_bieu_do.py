"""Hình kết quả cho poster: tỷ lệ đáp ứng (Bảng 4.8) và mức PS-IPPO rẻ hơn ở cùng mức phục vụ (Bảng 4.10).

Chạy: python poster_bieu_do.py  ->  media/poster_ket_qua.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
ORANGE, PURPLE, GREY, TEXT, MUTED = "#E0812A", "#4A1A45", "#CDBFCB", "#2A2A2A", "#6E6E6E"

# (chính sách, tỷ lệ đáp ứng %, PS-IPPO rẻ hơn thấp nhất %, cao nhất % trên 3 hạt giống)
ROWS = [("EOQ", 89.5, 22.3, 29.2), ("EOQ theo nhóm", 85.8, 9.8, 16.3), ("(s,S)", 86.2, 12.6, 13.7),
        ("Newsvendor theo nhóm", 85.7, 8.2, 10.8), ("(s,S) biết sức chứa", 85.6, 8.6, 10.6),
        ("Newsvendor", 87.4, 8.0, 10.3), ("(s,S) theo nhóm", 85.9, 7.3, 10.5)]
IPPO_FR = 90.9
fmt = lambda v: f"{v:.1f}".replace(".", ",")

plt.rcParams.update({"font.family": "Arial", "font.size": 15})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.35, 4.05), dpi=300, sharey=True,
                             gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.1})
y = list(range(len(ROWS)))[::-1]
YI = len(ROWS) + 0.15  # hàng PS-IPPO ở trên cùng, tách nhẹ khỏi các chính sách cổ điển

# Trái: tỷ lệ đáp ứng khi mỗi chính sách dùng cấu hình tốt nhất; PS-IPPO cùng màu, chỉ in đậm nhãn
a1.barh(y, [r[1] - 80 for r in ROWS], left=80, height=0.62, color=GREY)
a1.barh(YI, IPPO_FR - 80, left=80, height=0.62, color=GREY)
for yi, r in zip(y, ROWS):
    a1.text(r[1] - 0.25, yi, fmt(r[1]), ha="right", va="center", fontsize=16, color=TEXT)
a1.text(IPPO_FR - 0.25, YI, fmt(IPPO_FR), ha="right", va="center", fontsize=16, color=TEXT, fontweight="bold")
a1.set_xlim(80, 91.4)
a1.set_ylim(-0.6, YI + 0.55)
a1.set_yticks(y + [YI], [r[0] for r in ROWS] + ["PS-IPPO"], fontsize=16)
a1.get_yticklabels()[-1].set(fontweight="bold")
a2.text(0.6, YI, "mốc so sánh", va="center", fontsize=15, color=MUTED, style="italic")
a1.set_xticks([])
a1.set_title("Tỷ lệ đáp ứng (%)", fontsize=17, color=PURPLE, fontweight="bold", loc="left")

# Phải: PS-IPPO rẻ hơn bao nhiêu khi chính sách cổ điển phải phục vụ ngang PS-IPPO
for yi, r in zip(y, ROWS):
    a2.barh(yi, r[2], height=0.62, color=ORANGE)                                # rẻ hơn ít nhất
    a2.barh(yi, r[3] - r[2], left=r[2], height=0.62, color=ORANGE, alpha=0.35)  # tới mức cao nhất
    a2.text(r[3] + 0.6, yi, f"{fmt(r[2])}–{fmt(r[3])}%", va="center", fontsize=17, color=TEXT)
a2.set_xlim(0, 44)
a2.set_xticks([0, 10, 20, 30], ["0", "10%", "20%", "30%"], fontsize=14, color=MUTED)
a2.tick_params(axis="y", length=0)
a2.text(1.0, 1.03, "PS-IPPO rẻ hơn ở cùng mức phục vụ", transform=a2.transAxes, ha="right", va="bottom",
        fontsize=17, color=PURPLE, fontweight="bold")

for ax in (a1, a2):
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#BBBBBB")
a2.spines["bottom"].set_visible(True)
a2.spines["bottom"].set_color("#BBBBBB")
fig.subplots_adjust(left=0.268, right=0.985, top=0.9, bottom=0.08)
out = os.path.join(ROOT, "media", "poster_ket_qua.png")
fig.savefig(out, facecolor="white")
print("saved", out)
