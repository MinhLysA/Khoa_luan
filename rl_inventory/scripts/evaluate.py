"""
scripts/evaluate.py
===================
Danh gia va so sanh IPPO voi ba chinh sach co dien (EOQ, (s,S), Newsvendor).

PHIEN BAN v2:
  [V2-23] Luu them results/episode_traces.json - dien bien theo NGAY cua mot
          episode dai dien cho tung chinh sach (ton kho, dat hang, thieu hang,
          muc su dung suc chua tung kho). App Streamlit dung file nay de chieu
          lai "chuyen gi da xay ra".
  [V2-24] Bang ket qua them cot chi phi tran kho va do lech chuan, xuat ca
          results/summary.json de app doc truc tiep.

PHIEN BAN v3:
  [V3-5] Them --tag: moi file ket qua duoc ghi voi hau to rieng
         (vd summary_seed1.json) thay vi de --checkpoint danh de len
         results/summary.json mac dinh. Can cho `python run.py multiseed`
         chay nhieu seed ma khong lam mat ket qua cua seed truoc.
"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.inventory_env import MultiWarehouseInventoryEnv
from agents.ppo_agent import PPOAgent
from baselines.traditional_policies import (
    EOQPolicy, SsPolicy, NewsvendorPolicy, build_env_state)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_episode(env, policy, is_ppo=False, seed=None, deterministic=True,
                collect_trace=False, start_day=None):
    obs, _ = env.reset(seed=seed, options={"start_day": start_day})
    hold = stock = order = over = svc = 0.0
    demand_tot = stockout_tot = 0.0
    n_order_events = 0
    inv_trace, trace = [], []

    while True:
        if is_ppo:
            action, _, _ = policy.select_action(obs, deterministic=deterministic)
        else:
            action = policy.get_action(build_env_state(env))

        obs, _, terminated, truncated, info = env.step(action)

        hold  += info["cost_holding"]
        stock += info["cost_stockout"]
        order += info["cost_ordering"]
        over  += info["cost_overflow"]
        svc   += info["cost_service_penalty"]

        demand_tot     += info["demand"]
        stockout_tot   += info["stockout"]
        n_order_events += info["n_orders"]
        inv_trace.append(info["inventory"])

        if collect_trace:
            trace.append({
                "inventory": float(info["inventory"]),
                "demand": float(info["demand"]),
                "sold": float(info["sold"]),
                "stockout": float(info["stockout"]),
                "order_qty": float(info["order_qty"]),
                "n_orders": int(info["n_orders"]),
                "overflow": float(info["overflow"]),
                "fill_rate": float(info["fill_rate_mean"]),
                "util_wh": [round(float(x), 4) for x in info["util_wh"]],
                "inv_wh": [round(float(x), 1) for x in info["inv_wh"]],
                "cost_day": float(info["cost_holding"] + info["cost_stockout"]
                                  + info["cost_ordering"] + info["cost_overflow"]),
            })

        if terminated or truncated:
            break

    fill_rate = (demand_tot - stockout_tot) / max(demand_tot, 1e-6)
    res = {
        "total_cost":    hold + stock + order + over,
        "holding_cost":  hold, "stockout_cost": stock,
        "ordering_cost": order, "overflow_cost": over,
        "service_penalty_shaping": svc,
        "fill_rate": fill_rate, "stockout_units": stockout_tot,
        "demand_units": demand_tot, "n_order_events": n_order_events,
        "avg_inventory": float(np.mean(inv_trace)),
        "inventory_std": float(np.std(inv_trace)),
    }
    return (res, trace) if collect_trace else (res, None)


def paired_test(a, b):
    """So sanh theo cap hai day chi phi cung hat giong: hieu trung binh, khoang
    tin cay 95% (phan phoi t), kiem dinh t theo cap, Wilcoxon, hieu ung d_z."""
    n = min(len(a), len(b))
    a, b = np.asarray(a[:n], float), np.asarray(b[:n], float)
    d = a - b
    se = d.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    try:
        p_w = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        p_w = float("nan")
    return {"n": int(n), "gap_percent": float(100 * (a.mean() / b.mean() - 1)),
            "mean_diff": float(d.mean()), "ci95_low": float(d.mean() - h),
            "ci95_high": float(d.mean() + h),
            "p_paired": float(stats.ttest_rel(a, b).pvalue), "p_wilcoxon": p_w,
            "d_z": float(d.mean() / (d.std(ddof=1) + 1e-9)),
            "n_ippo_cheaper": int((d < 0).sum())}


def evaluate_policy(env, policy, n_episodes, is_ppo=False, name="", base_seed=1000,
                    traces=None, start_days=None):
    rows = []
    for i in range(n_episodes):
        want_trace = (traces is not None and i == 0)
        r, tr = run_episode(env, policy, is_ppo=is_ppo, seed=base_seed + i,
                            collect_trace=want_trace,
                            start_day=None if start_days is None else start_days[i])
        if want_trace:
            traces[name] = tr
        r["episode"] = i
        r["policy"] = name
        rows.append(r)
    df = pd.DataFrame(rows)
    print(f"  {name:<12s} cost={df.total_cost.mean():>13,.0f} "
          f"(+/-{df.total_cost.std():>11,.0f})  fill={df.fill_rate.mean():6.1%}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--mode", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--tag", type=str, default="",
                        help="Hau to ten file ket qua, vd --tag seed1 -> summary_seed1.json")
    # [P1-3] Ngay bat dau CO DINH, trai deu tren toan mien test (moi chinh
    # sach dung dung cac cua so nay) thay vi rut ngau nhien theo seed.
    parser.add_argument("--fixed_windows", action="store_true")
    args = parser.parse_args()

    with open(ROOT / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    env_cfg, paths = cfg["env"], cfg["paths"]
    n_episodes = args.episodes or cfg["eval"]["n_episodes"]
    base_seed = cfg["eval"].get("seed", 1000)
    suffix = f"_{args.tag}" if args.tag else ""

    results_dir = ROOT / paths["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    demand_data = calendar_features = price_per_pair = price_series = None
    p = ROOT / paths["data_dir"] / "demand_data.npy"
    if p.exists():
        demand_data = np.load(str(p))
    p = ROOT / paths["data_dir"] / "calendar_features.npy"
    if p.exists():
        calendar_features = np.load(str(p))
        if calendar_features.size == 0:
            calendar_features = None
    p = ROOT / paths["data_dir"] / "price_per_pair.npy"
    if p.exists():
        price_per_pair = np.load(str(p))
    p = ROOT / paths["data_dir"] / "price_series.npy"
    if p.exists():
        price_series = np.load(str(p))

    env = MultiWarehouseInventoryEnv(config=env_cfg, demand_data=demand_data,
                                     calendar_features=calendar_features,
                                     price_per_pair=price_per_pair,
                                     price_series=price_series,
                                     mode=args.mode)

    print("=" * 70)
    print(f"DANH GIA | mode={args.mode} | {n_episodes} episodes | n_pairs={env.n_pairs}")
    print(f"Suc chua tung kho: {np.round(env.suc_chua_kho).astype(int).tolist()}")
    print("=" * 70)

    params_path = results_dir / "baseline_params.json"
    tuned = {}
    if params_path.exists():
        tuned = json.loads(params_path.read_text(encoding="utf-8"))
        print(f"Da nap tham so baseline da tinh chinh tu {params_path.name}")
    else:
        print("CANH BAO: chua co baseline_params.json -> chay "
              "scripts/tune_baselines.py truoc de so sanh cong bang.")

    all_dfs, traces = [], {}
    start_days = None
    if args.fixed_windows and env.demand_data is not None:
        lo = env.val_day if args.mode == "test" else 0
        hi = (env.demand_data.shape[0] if args.mode == "test" else env.split_day) - env.episode_length
        start_days = np.linspace(lo, max(hi, lo), n_episodes).round().astype(int).tolist()
        print(f"Cua so co dinh: ngay bat dau {start_days[0]}..{start_days[-1]}")

    ckpt = ROOT / args.checkpoint
    if ckpt.exists():
        agent = PPOAgent(obs_per_pair=env.obs_per_pair, n_pairs=env.n_pairs,
                         n_action_levels=env.n_action_levels, config=cfg.get("ppo", {}))
        try:
            agent.load(str(ckpt))
            agent.network.eval()
            all_dfs.append(evaluate_policy(env, agent, n_episodes, is_ppo=True,
                                           name="IPPO", base_seed=base_seed,
                                           traces=traces, start_days=start_days))
        except (RuntimeError, KeyError) as e:
            print(f"  [IPPO] khong nap duoc checkpoint: {e}")
    else:
        print(f"  [IPPO] khong tim thay {ckpt}")

    specs = [
        ("EOQ", EOQPolicy, dict(ordering_cost=env.cp_dh, holding_cost=env.cp_lk,
                                lead_time=env.tg_giao_tb, q_factor=1.0)),
        ("(s,S)", SsPolicy, dict(service_level=0.90, lead_time=env.tg_giao_tb,
                                 q_factor=1.0, ordering_cost=env.cp_dh,
                                 holding_cost=env.cp_lk)),
        ("Newsvendor", NewsvendorPolicy, dict(holding_cost=env.cp_lk,
                                              stockout_cost=env.cp_th,
                                              lead_time=env.tg_giao_tb)),
    ]
    for name, cls, kw in specs:
        kw.update(tuned.get(name, {}))
        pol = cls(n_pairs=env.n_pairs, **kw)
        all_dfs.append(evaluate_policy(env, pol, n_episodes, name=name,
                                       base_seed=base_seed, traces=traces,
                                       start_days=start_days))

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv(results_dir / f"all_episodes{suffix}.csv", index=False)

    metrics = ["total_cost", "holding_cost", "stockout_cost", "ordering_cost",
               "overflow_cost", "fill_rate", "avg_inventory", "n_order_events",
               "inventory_std", "service_penalty_shaping"]
    summary = df_all.groupby("policy")[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(results_dir / f"baseline_comparison{suffix}.csv", index=False)

    print()
    print(f"{'Policy':<12}{'Tong CP':>15}{'Luu kho':>14}{'Thieu hang':>14}"
          f"{'Dat hang':>13}{'Tran kho':>12}{'FillRate':>10}")
    print("-" * 90)
    for _, r in summary.iterrows():
        print(f"{r['policy']:<12}{r['total_cost_mean']:>15,.0f}"
              f"{r['holding_cost_mean']:>14,.0f}{r['stockout_cost_mean']:>14,.0f}"
              f"{r['ordering_cost_mean']:>13,.0f}{r['overflow_cost_mean']:>12,.0f}"
              f"{r['fill_rate_mean']:>9.1%}")

    stat_out = {}
    if "IPPO" in df_all.policy.values:
        ppo_cost = df_all[df_all.policy == "IPPO"].sort_values("episode").total_cost.values
        others = summary[summary.policy != "IPPO"]
        best_name = others.loc[others.total_cost_mean.idxmin(), "policy"]

        # [P3-4] Moi lan danh gia cua IPPO va baseline dung CUNG hat giong
        # (cung nhu cau, cung thoi gian giao ngau nhien) -> chi dung kiem dinh
        # THEO CAP tren hieu so Delta_j = C_j^IPPO - C_j^baseline.
        per = {name: paired_test(ppo_cost, df_all[df_all.policy == name]
                                 .sort_values("episode").total_cost.values)
               for name in others.policy}
        stat_out = {"best_baseline": best_name, **per[best_name],
                    "per_baseline": per, "fixed_windows": bool(args.fixed_windows),
                    "n_episodes": int(n_episodes), "mode": args.mode}
        print()
        for name, r in per.items():
            print(f"IPPO - {name:<11s}: {r['gap_percent']:+6.2f}%  "
                  f"Delta={r['mean_diff']:+,.0f} [{r['ci95_low']:+,.0f}; {r['ci95_high']:+,.0f}]  "
                  f"p_t={r['p_paired']:.2e}  p_W={r['p_wilcoxon']:.2e}  d_z={r['d_z']:+.2f}  "
                  f"IPPO re hon {r['n_ippo_cheaper']}/{r['n']}")
        json.dump(stat_out, open(results_dir / f"statistical_test{suffix}.json", "w"), indent=2)

    # [V2-24] xuat summary.json cho app
    json.dump({"summary": summary.to_dict(orient="records"),
               "stat": stat_out,
               "capacity_per_warehouse": [float(x) for x in env.suc_chua_kho],
               "n_pairs": int(env.n_pairs), "mode": args.mode},
              open(results_dir / f"summary{suffix}.json", "w"), indent=2)
    # [V2-23] dien bien theo ngay
    json.dump(traces, open(results_dir / f"episode_traces{suffix}.json", "w"))

    _plot(summary, results_dir, suffix)
    print(f"\nDa luu ket qua vao {results_dir}")


def _plot(summary, results_dir, suffix=""):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    pol = summary["policy"].tolist()

    axes[0].bar(pol, summary["total_cost_mean"],
                yerr=summary["total_cost_std"], capsize=4, color="#4C72B0")
    axes[0].set_title("Tong chi phi van hanh")
    axes[0].set_ylabel("Chi phi")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(pol, summary["fill_rate_mean"] * 100,
                yerr=summary["fill_rate_std"] * 100, capsize=4, color="#55A868")
    axes[1].axhline(85, ls="--", c="k", lw=1)
    axes[1].set_title("Fill rate (%)")
    axes[1].tick_params(axis="x", rotation=20)

    bottom = np.zeros(len(pol))
    for col, lab in [("holding_cost_mean", "Luu kho"),
                     ("stockout_cost_mean", "Thieu hang"),
                     ("ordering_cost_mean", "Dat hang"),
                     ("overflow_cost_mean", "Tran kho")]:
        axes[2].bar(pol, summary[col], bottom=bottom, label=lab)
        bottom += summary[col].values
    axes[2].set_title("Co cau chi phi")
    axes[2].legend(fontsize=8)
    axes[2].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(results_dir / f"baseline_comparison{suffix}.png", dpi=140)
    plt.close()

    plt.figure(figsize=(6, 5))
    for _, r in summary.iterrows():
        plt.scatter(r["fill_rate_mean"] * 100, r["total_cost_mean"], s=110)
        plt.annotate(r["policy"], (r["fill_rate_mean"] * 100, r["total_cost_mean"]),
                     textcoords="offset points", xytext=(6, 6))
    plt.xlabel("Fill rate (%)")
    plt.ylabel("Tong chi phi van hanh")
    plt.title("Danh doi chi phi - muc phuc vu\n(tot = phia duoi ben phai)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / f"cost_service_frontier{suffix}.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    main()
