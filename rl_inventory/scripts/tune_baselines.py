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
    # [P1-1] Tinh chinh tren mien VAL (mac dinh) - cung mien IPPO dung de chon
    # checkpoint - thay vi mien train. "train" giu lai de tai lap ket qua cu.
    ap.add_argument("--tune_mode", choices=["train", "val"], default="val")
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

    # Tinh chinh tren mien train/val, KHONG BAO GIO tren mien test
    env = MultiWarehouseInventoryEnv(config=env_cfg, demand_data=demand_data,
                                     calendar_features=calendar_features,
                                     mode=args.tune_mode)
    print(f"Tinh chinh baseline tren mien {args.tune_mode.upper()}")
    LT = env.tg_giao_tb
    seed = 7777

    # [P3-3] Cung tieu chi voi viec chon checkpoint IPPO: chi phi THAP NHAT
    # trong so cac cau hinh dat fill rate >= muc_dv (85%) tren mien tinh chinh.
    # Neu khong cau hinh nao dat, lay cau hinh co fill rate cao nhat.
    luoi = {
        "EOQ": (EOQPolicy, [dict(q_factor=qf, lead_time=lt, ordering_cost=env.cp_dh,
                                 holding_cost=env.cp_lk)
                            for qf, lt in itertools.product([0.5, 1.0, 1.5, 2.0, 3.0],
                                                            [1.0, LT, 3.0, 4.0, 5.0])]),
        "(s,S)": (SsPolicy, [dict(service_level=sl, q_factor=qf, lead_time=lt,
                                  ordering_cost=env.cp_dh, holding_cost=env.cp_lk)
                             for sl, qf, lt in itertools.product(
                                 [0.70, 0.80, 0.90, 0.95, 0.99], [0.5, 1.0, 1.5],
                                 [1.0, LT, 3.0, 4.0])]),
        "Newsvendor": (NewsvendorPolicy, [dict(cr_override=float(cr), lead_time=lt,
                                               holding_cost=env.cp_lk, stockout_cost=env.cp_th)
                                          for cr, lt in itertools.product(
                                              [0.60, 0.75, 0.85, 0.90, 0.95, 0.99],
                                              [1.0, LT, 3.0, 4.0])]),
    }
    an = {"ordering_cost", "holding_cost", "stockout_cost"}
    results, chi_tiet = {}, {}
    for ten, (cls, grid) in luoi.items():
        print(f"\n[{ten}] quet {len(grid)} cau hinh")
        ung_vien = []
        for kw in grid:
            c, f = danh_gia(env, cls(env.n_pairs, **kw), args.episodes, seed)
            ung_vien.append((c, f, {k: v for k, v in kw.items() if k not in an}))
        dat = [u for u in ung_vien if u[1] >= env.muc_dv]
        c, f, kw = min(dat, key=lambda u: u[0]) if dat else max(ung_vien, key=lambda u: u[1])
        results[ten] = kw
        chi_tiet[ten] = {"cost": c, "fill": f, "feasible": bool(dat),
                         "n_feasible": len(dat), "n_grid": len(grid)}
        print(f"   => {kw}  cost={c:,.0f}  fill={f:.1%}  "
              f"({'dat' if dat else 'KHONG dat'} fill >= {env.muc_dv:.0%}; "
              f"{len(dat)}/{len(grid)} cau hinh dat)")
    (ROOT / paths["results_dir"] / "baseline_tuning.json").write_text(
        json.dumps({"tune_mode": args.tune_mode, "muc_dv": env.muc_dv,
                    "chon": chi_tiet}, indent=2, ensure_ascii=False), encoding="utf-8")
    out = ROOT / paths["results_dir"] / "baseline_params.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nDa luu tham so baseline da tinh chinh vao {out}")


if __name__ == "__main__":
    main()
