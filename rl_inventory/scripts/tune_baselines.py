"""
scripts/tune_baselines.py
=========================
[FIX-14] Tim kiem luoi tham so cho ba chinh sach baseline TREN CHINH MOI TRUONG
MO PHONG, dung mien thoi gian HUAN LUYEN.

Ly do phai co script nay: khoa luan cam ket "tham so baseline duoc toi uu bang
tim kiem luoi tren chinh moi truong mo phong nham bao dam tinh cong bang cua so
sanh". Neu baseline bi hardcode (vd service_level = 0.95) thi phan bien
"anh so voi baseline bi lam yeu" la hoan toan chinh dang va rat kho go.

Ket qua luu vao results/baseline_params.json va duoc evaluate.py tu dong nap.

Cach dung:
    python scripts/tune_baselines.py --episodes 3
"""

import sys
import json
import yaml
import argparse
import itertools
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.inventory_env import MultiWarehouseInventoryEnv
from baselines.traditional_policies import (
    EOQPolicy, SsPolicy, NewsvendorPolicy, build_env_state)


def danh_gia(env, policy, n_episodes, base_seed):
    """Tra ve (chi phi trung binh, fill rate trung binh)."""
    costs, fills = [], []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + i)
        hold = stock = order = over = 0.0
        dem = so = 0.0
        while True:
            action = policy.get_action(build_env_state(env))
            obs, _, te, tr, info = env.step(action)
            # [V2-26] Lay chi phi TRUC TIEP tu info (env._calculate_reward) thay
            # vi tinh lai o day - truoc day hai noi tinh doc lap nen rat de lech
            # nhau khi sua env. Gio chi con MOT nguon su that.
            hold  += info["cost_holding"]
            stock += info["cost_stockout"]
            order += info["cost_ordering"]
            over  += info["cost_overflow"]
            dem += info["demand"]; so += info["stockout"]
            if te or tr:
                break
        costs.append(hold + stock + order + over)
        fills.append((dem - so) / max(dem, 1e-6))
    return float(np.mean(costs)), float(np.mean(fills))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config, encoding="utf-8"))
    env_cfg, paths = cfg["env"], cfg["paths"]

    demand_data = calendar_features = None
    p = ROOT / paths["data_dir"] / "demand_data.npy"
    if p.exists():
        demand_data = np.load(str(p))
    p = ROOT / paths["data_dir"] / "calendar_features.npy"
    if p.exists():
        calendar_features = np.load(str(p))

    # Tinh chinh tren mien HUAN LUYEN, khong dung mien danh gia
    env = MultiWarehouseInventoryEnv(config=env_cfg, demand_data=demand_data,
                                     calendar_features=calendar_features, mode="train")
    LT = env.tg_giao_tb
    seed = 7777

    results = {}

    # -- EOQ -----------------------------------------------------------------
    print("\n[EOQ] tim kiem luoi tren (q_factor, lead_time)")
    best = (float("inf"), None)
    for qf, lt in itertools.product([0.5, 0.75, 1.0, 1.5, 2.0], [1.0, LT, 3.0]):
        pol = EOQPolicy(env.n_pairs, ordering_cost=env.cp_dh, holding_cost=env.cp_lk,
                        lead_time=lt, q_factor=qf)
        c, f = danh_gia(env, pol, args.episodes, seed)
        print(f"   q_factor={qf:<5} lead_time={lt:<4} -> cost={c:>13,.0f}  fill={f:6.1%}")
        if c < best[0]:
            best = (c, dict(q_factor=qf, lead_time=lt))
    results["EOQ"] = best[1]
    print(f"   => tot nhat: {best[1]}  cost={best[0]:,.0f}")

    # -- (s,S) ---------------------------------------------------------------
    print("\n[(s,S)] tim kiem luoi tren (service_level, q_factor, lead_time)")
    best = (float("inf"), None)
    for sl, qf, lt in itertools.product([0.70, 0.80, 0.90, 0.95, 0.99],
                                        [0.5, 1.0, 1.5], [1.0, LT, 3.0]):
        pol = SsPolicy(env.n_pairs, service_level=sl, lead_time=lt, q_factor=qf,
                       ordering_cost=env.cp_dh, holding_cost=env.cp_lk)
        c, f = danh_gia(env, pol, args.episodes, seed)
        if c < best[0]:
            best = (c, dict(service_level=sl, q_factor=qf, lead_time=lt))
    print(f"   => tot nhat: {best[1]}  cost={best[0]:,.0f}")
    results["(s,S)"] = best[1]

    # -- Newsvendor ----------------------------------------------------------
    print("\n[Newsvendor] tim kiem luoi tren (critical_ratio, lead_time)")
    best = (float("inf"), None)
    cr_ly_thuyet = env.cp_th / (env.cp_th + env.cp_lk)
    for cr, lt in itertools.product([0.60, 0.75, 0.85, cr_ly_thuyet, 0.95],
                                    [1.0, LT, 3.0]):
        pol = NewsvendorPolicy(env.n_pairs, holding_cost=env.cp_lk,
                               stockout_cost=env.cp_th, lead_time=lt, cr_override=cr)
        c, f = danh_gia(env, pol, args.episodes, seed)
        if c < best[0]:
            best = (c, dict(cr_override=float(cr), lead_time=lt))
    print(f"   => tot nhat: {best[1]}  cost={best[0]:,.0f}")
    print(f"   (critical ratio ly thuyet p/(p+h) = {cr_ly_thuyet:.4f})")
    results["Newsvendor"] = best[1]

    out = ROOT / paths["results_dir"] / "baseline_params.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nDa luu tham so baseline da tinh chinh vao {out}")


if __name__ == "__main__":
    main()
