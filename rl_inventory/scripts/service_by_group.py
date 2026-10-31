"""
scripts/service_by_group.py
========================
[Fill theo nhom] Phan ra muc phuc vu theo 4 nhom quy mo cau: moi chinh sach, moi nhom co so
cap, fill trung binh, ty le cap duoi 85%, chi phi cuc bo moi cap moi ngay.
Dong thoi kiem tra cong quyet dinh cho C' (Lagrange theo cap): tu 1/3 so cap cua
hai nhom cau thap nhat duoi 85% VA IPPO kem baseline o do.

Cach dung:
    python scripts/service_by_group.py --checkpoint checkpoints/best_model_main_s42.pth --tag main_s42
"""
import argparse
import json

import numpy as np

from common import ROOT, load_config, load_data, make_env, make_policies
from baselines.traditional_policies import DEMAND_GROUP_NAMES, demand_group_index

BASE_SO_SANH = ["(s,S)", "(s,S) theo nhóm", "Newsvendor theo nhóm"]


def chay(env, fn, seeds):
    dem, st, cost = np.zeros(env.n_pairs), np.zeros(env.n_pairs), np.zeros(env.n_pairs)
    days = 0
    for s in seeds:
        obs, _ = env.reset(seed=s)
        while True:
            obs, _, te, tr, info = env.step(fn(obs, env))
            dem += info["demand_pairs"]
            st += info["stockout_pairs"]
            cost += info["op_cost_pairs"]
            days += 1
            if te or tr:
                break
    return 1 - st / np.maximum(dem, 1e-9), cost / max(days, 1)


def bang_nhom(fill, cost, nhom, thr):
    out = {}
    for g, ten in enumerate(DEMAND_GROUP_NAMES):
        m = nhom == g
        out[ten] = {"n_pairs": int(m.sum()), "fill_tb": float(fill[m].mean()),
                    "ty_le_duoi_nguong": float((fill[m] < thr).mean()),
                    "chi_phi_cap_ngay": float(cost[m].mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/best_model_main_s42.pth")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--tag", default="main_s42")
    a = ap.parse_args()

    cfg = load_config(a.config)
    env = make_env(cfg, load_data(cfg))
    pols = make_policies(cfg, env, a.checkpoint, iso_tag=a.tag)
    seeds = [cfg["eval"].get("seed", 1000) + i for i in range(a.episodes)]
    nhom = demand_group_index(env.mean_demand)

    kq = {}
    for ten, fn in pols.items():
        kq[ten] = bang_nhom(*chay(env, fn, seeds), nhom, env.muc_dv)
        print(ten, {g: f"{v['fill_tb']:.1%}/{v['ty_le_duoi_nguong']:.0%}" for g, v in kq[ten].items()})

    thap = DEMAND_GROUP_NAMES[:2]
    n = sum(kq["IPPO"][g]["n_pairs"] for g in thap)
    duoi = sum(kq["IPPO"][g]["ty_le_duoi_nguong"] * kq["IPPO"][g]["n_pairs"] for g in thap) / n
    refs = [b for b in BASE_SO_SANH if b in kq]
    kem = any(kq["IPPO"][g]["fill_tb"] < kq[b][g]["fill_tb"] for b in refs for g in thap)
    cong = {"ty_le_duoi_nguong_2_nhom_thap": duoi, "ippo_kem_baseline": bool(kem),
            "kich_hoat_C_phay": bool(duoi >= 1 / 3 and kem)}
    print(f"Cong C': {cong}")

    p = ROOT / cfg["paths"]["results_dir"] / f"service_by_group_{a.tag}.json"
    p.write_text(json.dumps({"checkpoint": a.checkpoint, "episodes": a.episodes, "nhom": kq, "cong_C": cong},
                            indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Da luu {p}")


if __name__ == "__main__":
    main()
