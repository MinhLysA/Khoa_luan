"""
scripts/policy_behavior.py
==========================
Kiem tra STATE va REWARD bang thuc nghiem (khong huan luyen lai):

1. Hanh vi dat hang co hop ly ve nghiep vu khong?
   Muc hanh dong trung binh va xac suat dat hang theo SO NGAY TON KHO (vi the
   ton kho / cau trung binh). Chinh sach hop ly phai dat IT khi ton kho CAO
   (tuong quan Spearman am).
2. Tac tu co "lach" phan thuong (reward hacking) khong?
   * Fill rate tung cap co don cuc sat nguong 85% (chi vua du tranh phat
     muc phuc vu) khong - so voi (s,S) cung muc phuc vu.
   * Ty trong chi phi tran kho, ty le dung muc dat hang lon nhat.
3. State co bien du thua khong? - Permutation importance: xao tron mot nhom
   dac trung giua cac cap o moi buoc (pha lien he dac trung <-> cap) roi do
   muc tang chi phi / giam fill rate. Nhom tang it ~ gan nhu khong duoc dung.

Cach dung:
    python scripts/policy_behavior.py --checkpoint checkpoints/best_model_seed42.pth
"""
import argparse
import json

import numpy as np
from scipy import stats

from common import ROOT, load_config, load_data, make_env, make_policies

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COVER_BINS = [0, 1, 2, 3, 5, 8, np.inf]


TEN_NHOM = {"inventory": "Tồn kho & vị thế tồn kho", "demand_history": "Lịch sử cầu 7 ngày",
            "pipeline": "Hàng đang về (pipeline)", "fill_rate": "Fill rate 30 ngày",
            "scale": "Đặc trưng quy mô", "warehouse": "Tín hiệu cấp kho",
            "price": "Tín hiệu giá", "event_lookahead": "Sự kiện sắp tới",
            "calendar": "Lịch (thứ, SNAP, sự kiện, tháng)"}


def obs_groups(env):
    """Vi tri cac nhom dac trung (lay tu env.obs_groups - mot nguon su that),
    bo qua nhom da bi tat bang obs_drop."""
    return {TEN_NHOM.get(k, k): v for k, v in env.obs_groups.items()
            if k not in env.obs_drop}


def run(env, fn, seeds, perm_cols=None, rng=None, collect=False):
    cost = demand = stock = overflow = 0.0
    dem_pair = np.zeros(env.n_pairs)
    st_pair = np.zeros(env.n_pairs)
    covers, acts = [], []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        while True:
            if perm_cols is not None:
                obs = obs.copy()
                # Nhom lich / su kien sap toi giong nhau o moi cap -> xao tron
                # giua cac cap khong doi gi; thay bang gia tri cua ngay ngau nhien.
                if perm_cols == env.obs_groups["calendar"]:
                    d = int(rng.integers(len(env.calendar_features)))
                    dow = np.eye(7, dtype=np.float32)[d % 7]
                    obs[:, perm_cols] = np.concatenate([dow, env.calendar_features[d]])
                elif perm_cols == env.obs_groups.get("event_lookahead"):
                    g = env.event_gap[int(rng.integers(len(env.event_gap)))]
                    obs[:, perm_cols] = [min(g / 30.0, 1.0), float(g <= 7)]
                else:
                    obs[:, perm_cols] = obs[rng.permutation(env.n_pairs)][:, perm_cols]
            if collect:
                inv_pos = env.inventory + env.pipeline_orders.sum(axis=0)
                covers.append(inv_pos / np.maximum(env.mean_demand, 1e-6))
            a = fn(obs, env)
            if collect:
                acts.append(np.asarray(a))
            obs, _, term, trunc, info = env.step(a)
            cost += (info["cost_holding"] + info["cost_stockout"]
                     + info["cost_ordering"] + info["cost_overflow"])
            overflow += info["cost_overflow"]
            demand += info["demand"]
            stock += info["stockout"]
            dem_pair += info["demand_pairs"]
            st_pair += info["stockout_pairs"]
            if term or trunc:
                break
    n = len(seeds)
    out = {"cost": cost / n, "fill": 1 - stock / demand, "overflow_share": overflow / cost,
           "fill_pair": 1 - st_pair / np.maximum(dem_pair, 1e-9)}
    if collect:
        out["cover"] = np.concatenate(covers)
        out["action"] = np.concatenate(acts)
    return out


def action_profile(r, n_levels):
    rows = []
    for lo, hi in zip(COVER_BINS[:-1], COVER_BINS[1:]):
        m = (r["cover"] >= lo) & (r["cover"] < hi)
        if m.sum() == 0:
            continue
        rows.append({"cover_days": f"[{lo}, {hi})", "share": float(m.mean()),
                     "mean_action": float(r["action"][m].mean()),
                     "p_order": float((r["action"][m] > 0).mean())})
    rho = stats.spearmanr(r["cover"], r["action"]).correlation
    return {"bins": rows, "spearman_cover_action": float(rho),
            "share_max_action": float((r["action"] == n_levels - 1).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/best_model_seed42.pth")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=None,
                    help="Nguong fill rate de kiem tra don cuc (mac dinh env.muc_dv)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    cfg = load_config(a.config)
    data = load_data(cfg)
    env = make_env(cfg, data)
    pols = make_policies(cfg, env, a.checkpoint)
    seeds = [cfg["eval"].get("seed", 1000) + i for i in range(a.episodes)]
    suffix = f"_{a.tag}" if a.tag else ""
    thr = a.threshold or env.muc_dv
    ref_name = "(s,S) (cùng mức PV)" if "(s,S) (cùng mức PV)" in pols else "(s,S)"

    # 1 + 2: hanh vi va dau hieu lach phan thuong
    beh = {}
    for name in ["IPPO", ref_name]:
        r = run(env, pols[name], seeds, collect=True)
        fp = r["fill_pair"]
        beh[name] = {
            "total_cost": r["cost"], "fill_rate": r["fill"],
            "overflow_cost_share": r["overflow_share"],
            **action_profile(r, env.n_action_levels),
            "pairs_fill_below_threshold": float((fp < thr).mean()),
            "pairs_fill_within_3pp_above_threshold": float(((fp >= thr) & (fp < thr + 0.03)).mean()),
            "pairs_fill_quantiles": {q: float(np.quantile(fp, q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]},
            "_fill_pair": fp,
        }
        b = beh[name]
        print(f"{name}: cost={b['total_cost']:,.0f} fill={b['fill_rate']:.1%} "
              f"rho(cover,action)={b['spearman_cover_action']:.3f} "
              f"max-action={b['share_max_action']:.1%} tran kho={b['overflow_cost_share']:.2%} "
              f"cap sat nguong={b['pairs_fill_within_3pp_above_threshold']:.1%}")

    # 3: permutation importance (IPPO)
    rng = np.random.default_rng(0)
    base = run(env, pols["IPPO"], seeds)
    perm = {}
    for g, cols in obs_groups(env).items():
        r = run(env, pols["IPPO"], seeds, perm_cols=cols, rng=rng)
        perm[g] = {"n_features": len(cols),
                   "delta_cost_percent": float(100 * (r["cost"] / base["cost"] - 1)),
                   "delta_fill_pp": float(100 * (r["fill"] - base["fill"]))}
        print(f"  xao tron {g:<34s}: chi phi {perm[g]['delta_cost_percent']:+6.1f}%  "
              f"fill {perm[g]['delta_fill_pp']:+5.1f} diem %")

    out = {"checkpoint": a.checkpoint, "episodes": a.episodes, "threshold": thr,
           "behavior": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                        for k, v in beh.items()},
           "permutation_importance": perm}
    p = ROOT / cfg["paths"]["results_dir"] / f"policy_behavior{suffix}.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Da luu {p}")
    _plot(beh, perm, thr, ROOT / cfg["paths"]["results_dir"] / f"policy_behavior{suffix}.png")


def _plot(beh, perm, thr, path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for name, b in beh.items():
        labels = [r["cover_days"] for r in b["bins"]]
        axes[0].plot(labels, [r["p_order"] * 100 for r in b["bins"]], "o-", label=name)
    axes[0].set_xlabel("Số ngày tồn kho (vị thế tồn kho / cầu TB)")
    axes[0].set_ylabel("Xác suất đặt hàng (%)")
    axes[0].set_title("Quyết định đặt hàng theo mức tồn kho")
    axes[0].legend(fontsize=8)

    bins = np.linspace(0.4, 1.0, 31)
    for name, b in beh.items():
        axes[1].hist(b["_fill_pair"], bins=bins, alpha=0.55, label=name)
    axes[1].axvline(thr, c="k", ls="--", lw=1, label=f"Ngưỡng {thr:.0%}")
    axes[1].set_xlabel("Fill rate của từng cặp kho-SKU")
    axes[1].set_ylabel("Số cặp")
    axes[1].set_title("Phân bố fill rate theo cặp")
    axes[1].legend(fontsize=8)

    items = sorted(perm.items(), key=lambda kv: kv[1]["delta_cost_percent"])
    axes[2].barh([k for k, _ in items], [v["delta_cost_percent"] for _, v in items],
                 color="#4C72B0")
    axes[2].set_xlabel("Chi phí tăng khi xáo trộn nhóm đặc trưng (%)")
    axes[2].set_title("Mức quan trọng của các nhóm trạng thái")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


if __name__ == "__main__":
    main()
