"""
scripts/build_demo_data.py
==========================
Dong goi du lieu RUT GON cho app demo vao app/demo_data/. App chi doc thu muc
nay (va results/ de lay so lieu bao cao), khong doc bo M5 goc hay data/processed.

  demo.npz              cau, gia ban (sell_prices.csv -> price_series.npy), lich
                        cua MIEN TEST; mean_demand / price_per_pair (uoc luong tu
                        mien train, env can de dung lai suc chua, bang dat hang...);
                        rollout cua IPPO va 3 baseline tren toan mien test
  meta.json             cau hinh env/ppo luc dong goi, ten kho/SKU, ngay bat dau test
  best_model.pth        ban sao checkpoint IPPO
  baseline_params.json  ban sao tham so baseline da tinh chinh

Chi chay INFERENCE (deterministic), khong huan luyen.

    python run.py demo_data
    python scripts/build_demo_data.py --checkpoint checkpoints/best_model_seed1.pth --tag seed1
"""
import sys
import json
import shutil
import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common import load_config, load_data, make_env, make_policies

M5_DAY0 = date(2011, 1, 29)        # d_1 trong calendar.csv cua M5
OUT = ROOT / "app" / "demo_data"


def rollout(env, act, seed):
    """Chay tron mien test. inv/pipe ghi LUC RA QUYET DINH (truoc step)."""
    obs, _ = env.reset(seed=seed)
    rec = {k: [] for k in ("inv", "pipe", "order", "stockout", "util", "overflow_wh")}
    cost = 0.0
    while True:
        rec["inv"].append(env.inventory.copy())
        rec["pipe"].append(env.pipeline_orders.sum(axis=0))
        obs, _, te, tr, info = env.step(act(obs, env))
        rec["order"].append(info["order_qty_pairs"])
        rec["stockout"].append(info["stockout_pairs"])
        rec["util"].append(info["util_wh"])
        rec["overflow_wh"].append(info["overflow_wh"])
        cost += (info["cost_holding"] + info["cost_stockout"]
                 + info["cost_ordering"] + info["cost_overflow"])
        if te or tr:
            break
    return {k: np.asarray(v, dtype=np.float32) for k, v in rec.items()}, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best_model_seed42.pth")
    ap.add_argument("--tag", default="seed42",
                    help="Hau to file trong results/ ma app doc (summary_<tag>.json...)")
    a = ap.parse_args()

    cfg = load_config()
    data = load_data(cfg)
    if data["demand_data"] is None:
        sys.exit("LOI: chua co data/processed/demand_data.npy -> chay `python run.py data` truoc.")
    if not (ROOT / a.checkpoint).exists():
        sys.exit(f"LOI: khong tim thay {a.checkpoint}")
    params = ROOT / cfg["paths"]["results_dir"] / "baseline_params.json"
    if not params.exists():
        sys.exit(f"LOI: chua co {params} -> chay `python run.py baseline` truoc.")

    env = make_env(cfg, data, mode="test")
    pols = make_policies(cfg, env, a.checkpoint, include_iso=False, include_group=False)
    seed = int(cfg["eval"].get("seed", 1000))
    v0 = env.val_day
    meta_pre = json.loads((ROOT / cfg["paths"]["data_dir"] / "env_config.json")
                          .read_text(encoding="utf-8"))

    arrays = {
        "demand": data["demand_data"][v0:].astype(np.float32),
        "price": data["price_series"][v0:].astype(np.float32),
        "calendar": data["calendar_features"][v0:].astype(np.float32),
        "mean_demand": env.mean_demand,
        "price_per_pair": np.asarray(data["price_per_pair"], dtype=np.float32),
    }
    ref = ROOT / cfg["paths"]["results_dir"] / f"all_episodes_{a.tag}.csv"
    ref = pd.read_csv(ref) if ref.exists() else None
    print(f"Mien test: ngay {v0}..{env.demand_data.shape[0] - 1} "
          f"({env.episode_length} ngay) | obs {env.obs_per_pair} chieu | seed {seed}")
    for name, act in pols.items():
        rec, cost = rollout(env, act, seed)
        arrays.update({f"{name}|{k}": v for k, v in rec.items()})
        # Doi chieu voi lan danh gia 0 trong results/ (cung hat giong 1000)
        goc = "" if ref is None else ref[(ref.policy == name) & (ref.episode == 0)].total_cost
        goc = f" | results/: {float(goc.iloc[0]):,.0f}" if len(goc) else ""
        print(f"  {name:<11s} tong chi phi {cost:,.0f}{goc}")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "demo.npz", **arrays)
    shutil.copy(ROOT / a.checkpoint, OUT / "best_model.pth")
    shutil.copy(params, OUT / "baseline_params.json")
    meta = {"checkpoint": a.checkpoint, "results_tag": a.tag, "seed": seed,
            "env": cfg["env"], "ppo": cfg["ppo"],
            "stores": meta_pre["stores"], "skus": meta_pre["top_items"],
            "val_day": int(v0), "n_days_total": int(env.demand_data.shape[0]),
            "test_start": (M5_DAY0 + timedelta(days=int(v0))).isoformat(),
            "policies": list(pols)}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    size = sum(p.stat().st_size for p in OUT.iterdir()) / 1e6
    print(f"Da luu {OUT} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
