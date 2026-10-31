"""
scripts/hybrid_policy.py
========================
Hoc tang cuong them duoc gi so voi du bao roi toi uu: tach chi phi cua IPPO va Newsvendor
du bao theo nhom quy mo cau, roi tron hai chinh sach: moi nhom dung mot chinh sach. Cach
tron chon tren mien val, chay chung trong mot mo phong vi cac cap van chia suc chua.

Cach dung:
    python scripts/hybrid_policy.py --checkpoint checkpoints/best_model_main_s42.pth --tag main_s42
"""
import argparse
import json

import numpy as np

from common import ROOT, load_config, load_data, make_env, make_policies
from baselines.traditional_policies import DEMAND_GROUP_NAMES, demand_group_index
from compare_same_service import chon_theo_nhom
from evaluate import paired_test


def tron(fa, fb, la_a):
    """Cac cap co la_a = True theo chinh sach a, con lai theo chinh sach b."""
    return lambda obs, e: np.where(la_a, fa(obs, e), fb(obs, e))


def chay(env, fn, n, seed, nhom):
    """(chi phi tung lan, chi phi / thieu hang / nhu cau theo nhom, trung binh n lan)."""
    k = nhom.max() + 1
    tong, C, T, D = np.zeros(n), np.zeros(k), np.zeros(k), np.zeros(k)
    for i in range(n):
        obs, _ = env.reset(seed=seed + i)
        while True:
            obs, _, te, tr, inf = env.step(fn(obs, env))
            tong[i] += inf["op_cost_pairs"].sum()
            C += np.bincount(nhom, inf["op_cost_pairs"], k)
            T += np.bincount(nhom, inf["stockout_pairs"], k)
            D += np.bincount(nhom, inf["demand_pairs"], k)
            if te or tr:
                break
    return tong, C / n, T / n, D / n


def fill(T, D):
    return float(1 - T.sum() / max(D.sum(), 1e-6))


def chon_cach_tron(env, fa, fb, nhom, n, seed):
    """Quet cac cach tron tren val, lay cach re nhat khong thap hon fill val cua IPPO qua 0,2 diem."""
    _, ca, ta, da = chay(env, fa, n, seed, nhom)
    _, cb, tb, _ = chay(env, fb, n, seed, nhom)
    C, T = np.stack([ca, cb]), np.stack([ta, tb])
    muc = fill(ta, da) - 0.002
    diem = []
    for idx in chon_theo_nhom(C, T) | {(0,) * len(ca), (1,) * len(ca)}:
        la_ippo = np.array(idx)[nhom] == 0
        tong, _, t, d = chay(env, tron(fa, fb, la_ippo), n, seed, nhom)
        diem.append({"idx": idx, "cost": float(tong.mean()), "fill": fill(t, d)})
    dat = [d for d in diem if d["fill"] >= muc]
    return min(dat, key=lambda d: d["cost"]) if dat else max(diem, key=lambda d: d["fill"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/best_model_main_s42.pth")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--tune_episodes", type=int, default=3)
    ap.add_argument("--tag", default="main_s42")
    a = ap.parse_args()

    cfg = load_config(a.config)
    data = load_data(cfg)
    env_te, env_va = make_env(cfg, data), make_env(cfg, data, mode="val")
    pols = make_policies(cfg, env_te, a.checkpoint, include_iso=False, include_group=False)
    fa, fb = pols["IPPO"], pols["Newsvendor dự báo"]
    nhom = demand_group_index(env_te.mean_demand)
    seed = cfg["eval"].get("seed", 1000)

    chon = chon_cach_tron(env_va, fa, fb, nhom, a.tune_episodes, 7777)
    fm = tron(fa, fb, np.array(chon["idx"])[nhom] == 0)
    kq = {ten: chay(env_te, fn, a.episodes, seed, nhom)
          for ten, fn in [("IPPO", fa), ("Newsvendor dự báo", fb), ("Kết hợp", fm)]}
    out = {
        "chon_tren_val": {DEMAND_GROUP_NAMES[g]: ("IPPO" if i == 0 else "Newsvendor dự báo")
                          for g, i in enumerate(chon["idx"])},
        "tong": {ten: {"cost": float(v[0].mean()), "fill": fill(v[2], v[3])} for ten, v in kq.items()},
        "so_voi": {ten: paired_test(kq["Kết hợp"][0], kq[ten][0]) for ten in ("IPPO", "Newsvendor dự báo")},
        "theo_nhom": {DEMAND_GROUP_NAMES[g]: {ten: {"cost": float(v[1][g]), "fill": 1 - float(v[2][g] / max(v[3][g], 1e-6))}
                                              for ten, v in kq.items()} for g in range(nhom.max() + 1)},
    }
    p = ROOT / cfg["paths"]["results_dir"] / f"hybrid_policy_{a.tag}.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("chon_tren_val", "tong")}, indent=1, ensure_ascii=False))
    print(f"Da luu {p}")


if __name__ == "__main__":
    main()
