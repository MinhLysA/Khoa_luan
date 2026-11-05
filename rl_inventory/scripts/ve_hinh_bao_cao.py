"""Ve cac hinh dua vao bao cao (Report KLTN/media) tu ket qua trong results/. Khong chay mo phong.

    py -3.12 scripts/ve_hinh_bao_cao.py

Ba hinh ve truc tiep tu so lieu da luu: duong_cong_hoc, so_sanh_baseline, danh_doi_chi_phi_dich_vu.
Hai hinh chep tu results/ (do regime_analysis.py va policy_behavior.py ve): hinh_theo_giai_doan_nhu_cau,
hinh_soc_cau, hinh_hanh_vi_chinh_sach.
"""
import json
import shutil
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MEDIA = ROOT.parent / "Report KLTN" / "media"
SEEDS = ("42", "1", "2")

# Bang mau phan loai da kiem dinh mu mau (thu tu co dinh); xam cho doi chung khi chi can lam noi IPPO
XANH, CAM, NGOC, VANG, HONG, LUC, TIM = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"
XAM, CHU = "#a3a29c", "#52514e"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titlesize": 11, "axes.edgecolor": CHU, "savefig.dpi": 200})
PHAY = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))  # dau thap phan kieu Viet


def luu(fig, ten):
    for a in fig.axes:
        a.yaxis.set_major_formatter(PHAY)
        a.xaxis.set_major_formatter(PHAY)
    fig.tight_layout()
    fig.savefig(MEDIA / ten)
    plt.close(fig)


def duong_cong_hoc():
    d = pd.read_csv(RES / "train_log_main_s42.csv")
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax = ax.ravel()
    ax[0].plot(d.episode, d.reward_raw.rolling(20, min_periods=1).mean() / 1e6, color=XANH, lw=1.2)
    ax[0].set(title="(a) Phần thưởng tích lũy, trung bình trượt 20 tập", ylabel="Phần thưởng (triệu)")
    ax[1].plot(d.episode, d.fill_rate * 100, color="#86b6ef", lw=0.6, label="Huấn luyện (có khám phá)")
    ev = d.dropna(subset=["eval_fill"])
    ax[1].plot(ev.episode, ev.eval_fill * 100, color=XANH, lw=1.8, label="Đánh giá trên miền xác thực")
    ax[1].axhline(85, color=CHU, ls="--", lw=0.8)
    ax[1].text(d.episode.max(), 85.6, "mục tiêu 85%", ha="right", va="bottom", fontsize=8, color=CHU)
    ax[1].set(title="(b) Tỷ lệ đáp ứng", ylabel="Tỷ lệ đáp ứng (%)", ylim=(60, 100))
    ax[1].legend(fontsize=8, loc="lower right")
    ax[2].plot(d.episode, d.entropy, color=CAM, lw=0.8)
    ax[2].set(title="(c) Entropy của chính sách", ylabel="Entropy")
    ax[3].plot(d.episode, d.explained_variance, color=NGOC, lw=0.6)
    ax[3].set(title="(d) Phương sai giải thích được của Critic", ylabel="Phương sai giải thích được",
              ylim=(-0.05, 1.0))
    for a in ax:
        a.set_xlabel("Tập huấn luyện")
        a.grid(alpha=0.3)
    luu(fig, "duong_cong_hoc.png")


def so_sanh_baseline():
    bang = [pd.read_csv(RES / f"baseline_comparison_main_s{s}.csv").set_index("policy") for s in SEEDS]
    ippo = pd.concat([b.loc[["IPPO"]] for b in bang]).mean(numeric_only=True)
    goc = bang[0].drop(index=["IPPO", "Newsvendor dự báo"])   # NV du bao khac loai thong tin, so o Cau 4
    goc.loc["IPPO"] = ippo
    goc = goc.sort_values("total_cost_mean")
    ten = [t.replace(" theo nhóm", "\ntheo nhóm").replace(" biết sức chứa", "\nbiết sức chứa") for t in goc.index]
    mau = [XANH if t == "IPPO" else XAM for t in goc.index]
    x = np.arange(len(goc))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    ax[0].bar(x, goc.total_cost_mean / 1e6, color=mau, width=0.7)
    ax[0].set(title="(a) Tổng chi phí vận hành", ylabel="Triệu đơn vị", ylim=(1.9, 2.55))
    for i, v in enumerate(goc.total_cost_mean / 1e6):
        ax[0].text(i, v + 0.008, f"{v:.2f}".replace(".", ","), ha="center", fontsize=8, color=CHU)
    ax[1].bar(x, goc.fill_rate_mean * 100, color=mau, width=0.7)
    ax[1].axhline(85, color=CHU, ls="--", lw=0.8)
    ax[1].set(title="(b) Tỷ lệ đáp ứng", ylabel="%", ylim=(80, 93))
    for i, v in enumerate(goc.fill_rate_mean * 100):
        ax[1].text(i, v + 0.15, f"{v:.1f}".replace(".", ","), ha="center", fontsize=8, color=CHU)
    day = np.zeros(len(goc))
    for cot, nhan, m in [("holding_cost_mean", "Lưu kho", XANH), ("stockout_cost_mean", "Thiếu hàng", CAM),
                         ("ordering_cost_mean", "Đặt hàng", NGOC), ("overflow_cost_mean", "Tràn kho", VANG)]:
        v = goc[cot].values / 1e6
        ax[2].bar(x, v, bottom=day, color=m, width=0.7, label=nhan, edgecolor="white", linewidth=1)
        day += v
    ax[2].set(title="(c) Cơ cấu chi phí", ylabel="Triệu đơn vị")
    ax[2].legend(fontsize=8, loc="upper left", ncol=2)
    ax[2].set_ylim(0, 3.0)
    for a in ax:
        a.set_xticks(x, ten, rotation=40, ha="right", fontsize=8)
        a.grid(axis="y", alpha=0.3)
    for a in ax:
        a.yaxis.set_major_formatter(PHAY)
    fig.tight_layout()
    fig.savefig(MEDIA / "so_sanh_baseline.png")
    plt.close(fig)


def danh_doi():
    bien = json.loads((RES / "baseline_frontier.json").read_text(encoding="utf-8"))
    bang = pd.read_csv(RES / "baseline_comparison_main_s42.csv").set_index("policy")
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    # Duong bien: cau hinh khong bi troi tren mien xac thuc, ve chi phi va ty le dap ung tren mien kiem thu
    for ten, m in [("(s,S)", CAM), ("Newsvendor", NGOC), ("Newsvendor dự báo", TIM)]:
        p = sorted(bien[ten], key=lambda q: q["test_fill"])
        ax.plot([q["test_fill"] * 100 for q in p], [q["test_cost"] / 1e6 for q in p], ".-", color=m,
                lw=1.0, ms=3, label=f"Đường biên của {ten}")
    for ten in bang.index.drop("IPPO"):
        m = {"(s,S)": CAM, "Newsvendor": NGOC, "Newsvendor dự báo": TIM}.get(ten, XAM)
        ax.plot(bang.loc[ten, "fill_rate_mean"] * 100, bang.loc[ten, "total_cost_mean"] / 1e6, "x",
                color=m, ms=8, mew=2)
    ax.plot([], [], "x", color=XAM, ms=8, mew=2, label="Cấu hình tối thiểu chi phí của đối chứng")
    for tag, nhan, mk in [("main", "IPPO", "o"), ("forecast_safety", "IPPO kết hợp dự báo", "s")]:
        f, c = [], []
        for s in SEEDS:
            r = json.loads((RES / f"same_service_{tag}_s{s}.json").read_text(encoding="utf-8"))["ket_qua"]["IPPO"]
            f.append(r["fill"] * 100)
            c.append(r["cost"] / 1e6)
        ax.plot(f, c, mk, color=XANH if tag == "main" else "#184f95", ms=9, mec="white", mew=1.5,
                label=f"{nhan} (ba hạt giống)")
    ax.set(xlabel="Tỷ lệ đáp ứng trên miền kiểm thử (%)", ylabel="Tổng chi phí vận hành (triệu đơn vị)",
           xlim=(85, 93), ylim=(1.85, 2.6))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    luu(fig, "danh_doi_chi_phi_dich_vu.png")


def chep():
    for nguon, dich in [("regime_analysis_main.png", "hinh_theo_giai_doan_nhu_cau.png"),
                        ("shock_test_main.png", "hinh_soc_cau.png"),
                        ("policy_behavior_main.png", "hinh_hanh_vi_chinh_sach.png")]:
        shutil.copyfile(RES / nguon, MEDIA / dich)


if __name__ == "__main__":
    duong_cong_hoc()
    so_sanh_baseline()
    danh_doi()
    chep()
    print("Da ve/chep hinh vao", MEDIA)
