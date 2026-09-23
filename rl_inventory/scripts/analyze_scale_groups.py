"""
scripts/analyze_scale_groups.py
================================
Phan tich hau nghiem (post-hoc): hieu nang cua chinh sach IPPO da huan luyen
(chia se tham so) tren TUNG NHOM quy mo cau, phuc vu Cau hoi nghien cuu 2
(parameter sharing co tong quat hoa duoc qua cac SKU co quy mo cau khac nhau
khong). Khong huan luyen lai - chi danh gia checkpoint da co san tren mien TEST.

Chia 300 cap thanh 3 nhom tam phan vi theo mean_demand (uoc luong tu mien
train, giong cach chia trong bao cao): thap / trung binh / cao. Voi moi nhom,
tinh fill rate va chi phi cuc bo trung binh MOI CAP MOI NGAY (info["local_rewards"]
- dung dai luong ma tac tu thuc su toi uu hoa, cong thuc (localReward) trong
Chuong 3), de so sanh cong bang giua cac nhom co quy mo chi phi khac nhau.

Cach dung:
    python scripts/analyze_scale_groups.py --checkpoint checkpoints/best_model.pth
"""
import sys
import json
import yaml
import argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.inventory_env import MultiWarehouseInventoryEnv
from agents.ppo_agent import PPOAgent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config, encoding="utf-8"))
    env_cfg, paths = cfg["env"], cfg["paths"]
    base_seed = cfg["eval"].get("seed", 1000)

    demand = calf = price = price_series = None
    p = ROOT / paths["data_dir"] / "demand_data.npy"
    if p.exists():
        demand = np.load(str(p))
    p = ROOT / paths["data_dir"] / "calendar_features.npy"
    if p.exists():
        calf = np.load(str(p))
        if calf.size == 0:
            calf = None
    p = ROOT / paths["data_dir"] / "price_per_pair.npy"
    if p.exists():
        price = np.load(str(p))
    p = ROOT / paths["data_dir"] / "price_series.npy"
    if p.exists():
        price_series = np.load(str(p))

    env = MultiWarehouseInventoryEnv(env_cfg, demand, calf, price, price_series, mode="test")
    n_pairs = env.n_pairs

    ckpt = ROOT / args.checkpoint
    agent = PPOAgent(env.obs_per_pair, n_pairs, env.n_action_levels,
                     config=cfg["ppo"], device="cpu")
    agent.load(str(ckpt))
    agent.network.eval()

    # -- 3 nhom tam phan vi theo mean_demand (uoc luong tu mien train) --------
    md = env.mean_demand.copy()
    order = np.argsort(md)
    n = len(md)
    edges = [0, n // 3, 2 * n // 3, n]
    group_names = ["Thấp (tercile 1)", "Trung bình (tercile 2)", "Cao (tercile 3)"]
    group_idx = [order[edges[i]:edges[i + 1]] for i in range(3)]

    print("=" * 72)
    print(f"Phan nhom theo mean_demand (mien train), n_pairs={n_pairs}")
    for name, idx in zip(group_names, group_idx):
        print(f"  {name:<24s}: {len(idx):3d} cap, mean_demand trong "
              f"[{md[idx].min():.3f}, {md[idx].max():.3f}]  (tb={md[idx].mean():.3f})")
    print("=" * 72)

    # -- Chay danh gia deterministic, tich luy theo tung cap -------------------
    sum_cost   = np.zeros(n_pairs, dtype=np.float64)   # -local_rewards (don vi tien)
    sum_demand = np.zeros(n_pairs, dtype=np.float64)
    sum_stock  = np.zeros(n_pairs, dtype=np.float64)
    n_days = 0

    for i in range(args.episodes):
        obs, _ = env.reset(seed=base_seed + i)
        while True:
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            sum_cost   += -info["local_rewards"].astype(np.float64)
            sum_demand += info["demand_pairs"].astype(np.float64)
            sum_stock  += info["stockout_pairs"].astype(np.float64)
            n_days += 1
            if term or trunc:
                break

    fill_pair = 1.0 - sum_stock / np.maximum(sum_demand, 1e-6)
    cost_per_day_pair = sum_cost / (args.episodes * env.episode_length)

    print(f"\nKET QUA THEO NHOM (trung binh {args.episodes} episode, mien TEST)")
    print(f"{'Nhom':<24}{'Fill rate TB':>14}{'Fill rate min':>15}"
          f"{'Chi phi/cap/ngay':>18}")
    rows = []
    for name, idx in zip(group_names, group_idx):
        fr_mean = float(fill_pair[idx].mean())
        fr_min = float(fill_pair[idx].min())
        c_mean = float(cost_per_day_pair[idx].mean())
        print(f"{name:<24}{fr_mean:>13.2%} {fr_min:>14.2%} {c_mean:>18,.3f}")
        rows.append({"group": name, "n_pairs": int(len(idx)),
                     "fill_rate_mean": fr_mean, "fill_rate_min": fr_min,
                     "fill_rate_std": float(fill_pair[idx].std()),
                     "cost_per_pair_day_mean": c_mean,
                     "cost_per_pair_day_std": float(cost_per_day_pair[idx].std())})

    out = {"n_pairs": n_pairs, "episodes": args.episodes, "groups": rows,
           "fill_rate_all_pairs_std": float(fill_pair.std())}
    sfx = f"_{args.tag}" if args.tag else ""
    out_path = ROOT / paths["results_dir"] / f"scale_group_analysis{sfx}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDa luu {out_path}")

    # -- Bieu do ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    names = [r["group"] for r in rows]
    fr = [r["fill_rate_mean"] * 100 for r in rows]
    fr_std = [r["fill_rate_std"] * 100 for r in rows]
    cst = [r["cost_per_pair_day_mean"] for r in rows]
    cst_std = [r["cost_per_pair_day_std"] for r in rows]

    axes[0].bar(names, fr, yerr=fr_std, capsize=4, color="#55A868")
    axes[0].axhline(env.muc_dv * 100, ls="--", c="k", lw=1, label=f"Ngưỡng {env.muc_dv:.0%}")
    axes[0].set_ylabel("Fill rate trung bình (%)")
    axes[0].set_title("Fill rate theo nhóm quy mô cầu")
    axes[0].legend(fontsize=8)
    axes[0].tick_params(axis="x", rotation=12)

    axes[1].bar(names, cst, yerr=cst_std, capsize=4, color="#4C72B0")
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[1].set_ylabel("Chi phí cục bộ / cặp / ngày (thang log)")
    axes[1].set_title("Chi phí cục bộ theo nhóm quy mô cầu")
    axes[1].tick_params(axis="x", rotation=12)

    plt.tight_layout()
    fig_path = ROOT / paths["results_dir"] / f"scale_group_analysis{sfx}.png"
    plt.savefig(fig_path, dpi=140)
    plt.close()
    print(f"Da luu {fig_path}")

    bang_4_nhom(cfg, env, args, ROOT / paths["results_dir"])


NHOM_4 = [("Rất thấp (<0,5)", 0.0, 0.5), ("Thấp (0,5-2)", 0.5, 2.0),
          ("Trung bình (2-10)", 2.0, 10.0), ("Cao (>=10)", 10.0, np.inf)]


def bang_4_nhom(cfg, env, args, results_dir):
    """Bang theo 4 nhom quy mo cau (nguong co dinh) cho IPPO va (s,S) tinh
    chinh truc tiep - dung cho Bang 'Hieu nang theo nhom quy mo' o Chuong 4.
    fill = fill rate GOP cua nhom (tong thieu / tong cau); cost_per_pair = chi
    phi cuc bo (gom phat muc phuc vu) trung binh moi cap moi episode."""
    from common import make_policies
    pols = make_policies(cfg, env, args.checkpoint, include_iso=False)
    base_seed = cfg["eval"].get("seed", 1000)
    md = env.mean_demand
    out = {}
    for name in ["IPPO", "(s,S)"]:
        cost = np.zeros(env.n_pairs); dem = np.zeros(env.n_pairs); st = np.zeros(env.n_pairs)
        for i in range(args.episodes):
            obs, _ = env.reset(seed=base_seed + i)
            while True:
                obs, _, te, tr, info = env.step(pols[name](obs, env))
                cost += -info["local_rewards"]; dem += info["demand_pairs"]
                st += info["stockout_pairs"]
                if te or tr:
                    break
        out[name] = {}
        for ten, lo, hi in NHOM_4:
            m = (md >= lo) & (md < hi)
            out[name][ten] = {"n_pairs": int(m.sum()),
                              "fill": float(1 - st[m].sum() / max(dem[m].sum(), 1e-9)),
                              "cost_per_pair": float(cost[m].mean() / args.episodes)}
        print(name, {k: f"{v['fill']:.1%}" for k, v in out[name].items()})
    suffix = f"_{args.tag}" if args.tag else ""
    path = results_dir / f"scale_group_v2{suffix}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Da luu {path}")


if __name__ == "__main__":
    main()
