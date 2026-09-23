"""
scripts/sensitivity.py
======================
[P1-5] Do nhay cua KET LUAN theo tham so chi phi.

Moi chinh sach (IPPO + baseline) duoc chay MOT lan tren mien test; ghi lai
tong luong vat ly: don vi-ngay luu kho, don vi thieu hang, so lan dat, don vi
tran kho. Sau do tinh lai tong chi phi voi mot luoi don gia (c_lk, c_th, c_dh)
khac nhau de xem thu hang cac chinh sach co doi khong.

GIOI HAN: chinh sach KHONG duoc huan luyen/tinh chinh lai cho tung bo don
gia - ket qua tra loi cau "neu doanh nghiep dinh gia chi phi khac di thi cac
chinh sach DA CO xep hang the nao", khong phai "IPPO huan luyen voi don gia
moi se ra sao".

Cach dung:
    python scripts/sensitivity.py --checkpoint checkpoints/best_model_seed42.pth
"""
import argparse
import itertools
import json

import numpy as np

from common import ROOT, load_config, load_data, make_env, make_policies


def physical_totals(env, fn, seeds):
    tot = {"holding_units": 0.0, "stockout_units": 0.0, "orders": 0.0,
           "overflow_units": 0.0, "demand": 0.0}
    for s in seeds:
        obs, _ = env.reset(seed=s)
        while True:
            obs, _, term, trunc, info = env.step(fn(obs, env))
            tot["holding_units"] += info["cost_holding"] / env.cp_lk
            tot["stockout_units"] += info["stockout"]
            tot["orders"] += info["n_orders"]
            tot["overflow_units"] += info["overflow"]
            tot["demand"] += info["demand"]
            if term or trunc:
                break
    return {k: v / len(seeds) for k, v in tot.items()}


def cost(t, c_lk, c_th, c_dh, p_tk):
    return (c_lk * t["holding_units"] + c_th * t["stockout_units"]
            + c_dh * t["orders"] + p_tk * t["overflow_units"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/best_model_seed42.pth")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    cfg = load_config(a.config)
    env = make_env(cfg, load_data(cfg))
    pols = make_policies(cfg, env, a.checkpoint)
    seeds = [cfg["eval"].get("seed", 1000) + i for i in range(a.episodes)]

    totals = {}
    for name, fn in pols.items():
        totals[name] = physical_totals(env, fn, seeds)
        t = totals[name]
        print(f"{name:<26s} fill={1 - t['stockout_units'] / t['demand']:.1%} "
              f"cost(goc)={cost(t, env.cp_lk, env.cp_th, env.cp_dh, env.pt_tk):,.0f}")

    grid = []
    for c_lk, c_th, c_dh in itertools.product([0.5, 1.0, 2.0], [5.0, 10.0, 20.0, 40.0],
                                              [2.0, 5.0, 10.0]):
        costs = {n: cost(t, c_lk, c_th, c_dh, env.pt_tk) for n, t in totals.items()}
        rank = sorted(costs, key=costs.get)
        row = {"c_lk": c_lk, "c_th": c_th, "c_dh": c_dh, "ratio_th_lk": c_th / c_lk,
               "ranking": rank, "costs": costs}
        for ref in [r for r in costs if r != "IPPO"]:
            row[f"gap_vs_{ref}"] = 100 * (costs["IPPO"] / costs[ref] - 1)
        grid.append(row)

    refs = [r for r in totals if r != "IPPO"]
    print(f"\n{'c_lk':>5}{'c_th':>6}{'c_dh':>6}  xep hang (re -> dat)")
    for r in grid:
        print(f"{r['c_lk']:>5}{r['c_th']:>6}{r['c_dh']:>6}  " + " < ".join(r["ranking"][:3]))
    summary = {ref: {"ippo_cheaper_in": int(sum(r[f"gap_vs_{ref}"] < 0 for r in grid)),
                     "of": len(grid),
                     "gap_min": float(min(r[f"gap_vs_{ref}"] for r in grid)),
                     "gap_max": float(max(r[f"gap_vs_{ref}"] for r in grid))}
               for ref in refs}
    for ref, v in summary.items():
        print(f"IPPO re hon {ref:<26s} o {v['ippo_cheaper_in']}/{v['of']} bo don gia "
              f"(chenh lech {v['gap_min']:+.1f}% .. {v['gap_max']:+.1f}%)")

    suffix = f"_{a.tag}" if a.tag else ""
    out = ROOT / cfg["paths"]["results_dir"] / f"sensitivity{suffix}.json"
    out.write_text(json.dumps({"checkpoint": a.checkpoint, "episodes": a.episodes,
                               "physical_totals": totals, "grid": grid,
                               "summary": summary}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"Da luu {out}")


if __name__ == "__main__":
    main()
