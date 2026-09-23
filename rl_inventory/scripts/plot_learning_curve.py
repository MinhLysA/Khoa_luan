"""
scripts/plot_learning_curve.py
===============================
Ve duong cong huan luyen tu results/train_log*.csv (4 panel: phan thuong,
fill rate, entropy, explained variance). Dung cho Hinh "Duong cong hoc" o
Chuong 4 cua Report KLTN.

Cach dung:
    python scripts/plot_learning_curve.py                # doc train_log.csv
    python scripts/plot_learning_curve.py --tag seed1     # doc train_log_seed1.csv
    # So sanh nhieu lan chay (vd ablation) tren cung hinh, cat o 1000 episode:
    python scripts/plot_learning_curve.py --compare seed42 ablq3_seed42 \n        --labels "Chuan hoa theo cap" "Khong chuan hoa" --max_episode 1000
"""
import sys
import argparse
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--compare", nargs="+", default=None,
                    help="Nhieu tag de ve chong len nhau")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--max_episode", type=int, default=None)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(ROOT / args.config, encoding="utf-8"))
    results_dir = ROOT / cfg["paths"]["results_dir"]
    suffix = f"_{args.tag}" if args.tag else ""

    if args.compare:
        return compare(results_dir, args.compare, args.labels or args.compare,
                       args.max_episode)

    log_path = results_dir / f"train_log{suffix}.csv"
    if not log_path.exists():
        print(f"Khong tim thay {log_path} - chay python run.py train truoc.")
        return
    df = pd.read_csv(log_path)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    axes[0, 0].plot(df["episode"], df["reward_smooth"], color="#4C72B0")
    axes[0, 0].set_title("Phần thưởng trung bình trượt (20 episode)")
    axes[0, 0].set_xlabel("Episode"); axes[0, 0].set_ylabel("Reward")

    axes[0, 1].plot(df["episode"], df["fill_rate"] * 100, color="#55A868",
                     alpha=0.4, label="Huấn luyện (explore)")
    ev = df.dropna(subset=["eval_fill"])
    axes[0, 1].plot(ev["episode"], ev["eval_fill"] * 100, color="#1B5E20",
                     lw=2, label="Đánh giá deterministic (miền val)")
    axes[0, 1].axhline(85, ls="--", c="k", lw=1)
    axes[0, 1].set_title("Tỷ lệ đáp ứng nhu cầu (fill rate)")
    axes[0, 1].set_xlabel("Episode"); axes[0, 1].set_ylabel("Fill rate (%)")
    axes[0, 1].legend(fontsize=8)

    ent = df.dropna(subset=["entropy"])
    axes[1, 0].plot(ent["episode"], ent["entropy"], color="#C44E52")
    axes[1, 0].set_title("Entropy của chính sách")
    axes[1, 0].set_xlabel("Episode"); axes[1, 0].set_ylabel("Entropy")

    ex = df.dropna(subset=["explained_variance"])
    axes[1, 1].plot(ex["episode"], ex["explained_variance"], color="#8172B2")
    axes[1, 1].axhline(0, ls="--", c="k", lw=0.8)
    axes[1, 1].set_title("Explained variance của Critic")
    axes[1, 1].set_xlabel("Episode"); axes[1, 1].set_ylabel("Explained variance")

    plt.tight_layout()
    out_path = results_dir / f"learning_curve{suffix}.png"
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f"Da luu {out_path}")


def compare(results_dir, tags, labels, max_episode=None):
    """Ve chong tong chi phi, fill rate, entropy, explained variance cua
    nhieu lan chay (trung binh truot 20 episode)."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    panels = [("cost_total", 1e-6, "Tổng chi phí mỗi episode (triệu)"),
              ("fill_rate", 100, "Fill rate (%)"),
              ("entropy", 1, "Entropy của chính sách"),
              ("explained_variance", 1, "Explained variance của Critic")]
    for tag, lab in zip(tags, labels):
        df = pd.read_csv(results_dir / f"train_log_{tag}.csv")
        if max_episode:
            df = df[df["episode"] <= max_episode]
        for ax, (col, k, title) in zip(axes.flat, panels):
            d = df.dropna(subset=[col])
            ax.plot(d["episode"], d[col].rolling(20, min_periods=1).mean() * k, label=lab)
            ax.set_title(title); ax.set_xlabel("Episode")
    axes[0, 1].axhline(85, ls="--", c="k", lw=1)
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend(fontsize=8)
    plt.tight_layout()
    out_path = results_dir / f"learning_curve_compare_{'_vs_'.join(tags)}.png"
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f"Da luu {out_path}")


if __name__ == "__main__":
    main()
