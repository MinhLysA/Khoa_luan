"""
scripts/iso_service.py
======================
[V2-25] SO SANH O CUNG MUC PHUC VU (iso-service comparison).

Vi sao can script nay: so sanh tong chi phi giua hai chinh sach co fill rate
KHAC NHAU la vo nghia. IPPO bi phat khi fill rate < muc_dv nen no tu day muc
phuc vu len ~92%, con EOQ/(s,S)/Newsvendor duoc tinh chinh de TOI THIEU CHI PHI
nen dung lai o ~84%. Bang so sanh thang se ket luan sai rang "IPPO dat hon".

Script nay lam dung viec ma mot phan bien se yeu cau:
  1. Do fill rate ma IPPO dat duoc tren mien TEST.
  2. Voi TUNG baseline, tim kiem luoi tren mien TRAIN de tim cau hinh RE NHAT
     ma van dat fill rate >= muc do (cong dung sai cho phep).
  3. Danh gia lai cac cau hinh do tren mien TEST va bao cao canh IPPO.

Ket qua luu vao results/iso_service.json (app Streamlit doc file nay).

Cach dung:
    python scripts/iso_service.py --checkpoint checkpoints/best_model.pth
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
from agents.ppo_agent import PPOAgent
from baselines.traditional_policies import (
    EOQPolicy, SsPolicy, NewsvendorPolicy, build_env_state)


def chay(env, policy, n, seed, is_ppo=False):
    costs, fills = [], []
    for i in range(n):
        obs, _ = env.reset(seed=seed + i)
        h = s = o = ov = 0.0
        dm = so = 0.0
        while True:
            if is_ppo:
                a, _, _ = policy.select_action(obs, deterministic=True)
            else:
                a = policy.get_action(build_env_state(env))
            obs, _, te, tr, inf = env.step(a)
            h += inf["cost_holding"]; s += inf["cost_stockout"]
            o += inf["cost_ordering"]; ov += inf["cost_overflow"]
            dm += inf["demand"]; so += inf["stockout"]
            if te or tr:
                break
        costs.append(h + s + o + ov)
        fills.append((dm - so) / max(dm, 1e-6))
    return float(np.mean(costs)), float(np.mean(fills))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    ap.add_argument("--tune_episodes", type=int, default=2)
    ap.add_argument("--eval_episodes", type=int, default=10)
    ap.add_argument("--tolerance", type=float, default=0.005,
                    help="Cho phep baseline thap hon IPPO toi da bao nhieu fill rate")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config, encoding="utf-8"))
    env_cfg, paths = cfg["env"], cfg["paths"]

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

    env_tr = MultiWarehouseInventoryEnv(env_cfg, demand, calf, price, price_series, mode="train")
    env_te = MultiWarehouseInventoryEnv(env_cfg, demand, calf, price, price_series, mode="test")
    base_seed = cfg["eval"].get("seed", 1000)

    # -- B1: fill rate cua IPPO ---------------------------------------------
    ck = ROOT / args.checkpoint
    if not ck.exists():
        print(f"Khong tim thay checkpoint {ck}")
        return
    agent = PPOAgent(env_te.obs_per_pair, env_te.n_pairs, env_te.n_action_levels,
                     config=cfg["ppo"], device="cpu")
    agent.load(str(ck))
    agent.network.eval()
    ippo_cost, ippo_fill = chay(env_te, agent, args.eval_episodes, base_seed, is_ppo=True)
    target = ippo_fill - args.tolerance
    print("=" * 72)
    print(f"IPPO tren mien TEST : cost={ippo_cost:,.0f}   fill={ippo_fill:.2%}")
    print(f"Muc phuc vu muc tieu cho baseline: >= {target:.2%}")
    print("=" * 72)

    # -- B2: tim kiem luoi tren mien TRAIN ----------------------------------
    luoi = {
        "EOQ": (EOQPolicy,
                [dict(q_factor=qf, lead_time=lt,
                      ordering_cost=env_tr.cp_dh, holding_cost=env_tr.cp_lk)
                 for qf, lt in itertools.product([0.5, 1.0, 2.0, 3.0],
                                                 [2.0, 3.0, 4.0, 5.0, 6.0])]),
        "(s,S)": (SsPolicy,
                  [dict(service_level=sl, q_factor=qf, lead_time=lt,
                        ordering_cost=env_tr.cp_dh, holding_cost=env_tr.cp_lk)
                   for sl, qf, lt in itertools.product([0.90, 0.95, 0.99, 0.999],
                                                       [0.5, 1.0, 2.0],
                                                       [3.0, 4.0, 5.0])]),
        "Newsvendor": (NewsvendorPolicy,
                       [dict(cr_override=cr, lead_time=lt,
                             holding_cost=env_tr.cp_lk, stockout_cost=env_tr.cp_th)
                        for cr, lt in itertools.product([0.90, 0.95, 0.99, 0.999],
                                                        [3.0, 4.0, 5.0, 6.0])]),
    }

    ket_qua = {"IPPO": {"cost": ippo_cost, "fill": ippo_fill, "params": None}}
    for ten, (cls, grid) in luoi.items():
        print(f"\n[{ten}] quet {len(grid)} cau hinh tren mien TRAIN...")
        ung_vien = []
        for kw in grid:
            c, f = chay(env_tr, cls(n_pairs=env_tr.n_pairs, **kw),
                        args.tune_episodes, 7777)
            if f >= target:
                ung_vien.append((c, f, kw))
        if not ung_vien:
            print(f"   khong cau hinh nao dat fill >= {target:.1%} "
                  f"-> baseline nay khong the sanh ngang IPPO ve muc phuc vu")
            ket_qua[ten] = {"cost": None, "fill": None, "params": None,
                            "khong_dat": True}
            continue
        ung_vien.sort(key=lambda x: x[0])
        _, _, kw_tot = ung_vien[0]
        c_te, f_te = chay(env_te, cls(n_pairs=env_te.n_pairs, **kw_tot),
                          args.eval_episodes, base_seed)
        kw_in = {k: v for k, v in kw_tot.items()
                 if k not in ("ordering_cost", "holding_cost", "stockout_cost")}
        print(f"   re nhat dat muc phuc vu: {kw_in}")
        print(f"   -> TEST: cost={c_te:,.0f}  fill={f_te:.2%}")
        ket_qua[ten] = {"cost": c_te, "fill": f_te, "params": kw_in}

    # -- B3: bang tong hop ---------------------------------------------------
    print("\n" + "=" * 72)
    print("SO SANH O CUNG MUC PHUC VU (mien TEST)")
    print("=" * 72)
    print(f"{'Chinh sach':<14}{'Tong chi phi':>16}{'Fill rate':>12}{'So voi IPPO':>14}")
    print("-" * 72)
    for ten, r in ket_qua.items():
        if r.get("khong_dat"):
            print(f"{ten:<14}{'khong dat duoc muc phuc vu nay':>42}")
            continue
        delta = "" if ten == "IPPO" else f"{100*(ippo_cost-r['cost'])/r['cost']:+.1f}%"
        print(f"{ten:<14}{r['cost']:>16,.0f}{r['fill']:>11.2%}{delta:>14}")
    print("\n(So voi IPPO: so am = IPPO RE HON o cung muc phuc vu)")

    out = ROOT / paths["results_dir"] / "iso_service.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"target_fill": target, "ket_qua": ket_qua},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDa luu {out}")


if __name__ == "__main__":
    main()
