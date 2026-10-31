"""
scripts/hybrid_policy.py
========================
Hoc tang cuong them duoc gi so voi du bao roi toi uu: tach chi phi cua IPPO va Newsvendor
du bao theo nhom quy mo cau, roi tron: moi nhom dung IPPO hoac mot cau hinh du bao tren
duong bien. Doi chung cong bang la Newsvendor du bao theo nhom, cung chon cau hinh theo
nhom nhung khong co IPPO. Hai ho deu chon tren mien val va chay chung trong mot mo phong
vi cac cap van chia suc chua. Can results/baseline_frontier.json.

Cach dung:
    python scripts/hybrid_policy.py --checkpoint checkpoints/best_model_main_s42.pth --tag main_s42
"""
import argparse
import json

import numpy as np

from common import ROOT, load_config, load_data, make_env, make_policies
from baselines.traditional_policies import (DEMAND_GROUP_NAMES, ForecastNewsvendor, build_env_state,
                                            demand_group_index)
from compare_same_service import chon_theo_nhom, doc_tai, duong_test, pareto
from env.inventory_env import load_forecast
from evaluate import paired_test


def tron(cac_fn, chi_so):
    """Moi cap theo chinh sach cac_fn[chi_so[cap]]."""
    def fn(obs, e):
        a = np.zeros(len(chi_so), dtype=np.int64)
        for k in np.unique(chi_so):
            m = chi_so == k
            a[m] = np.asarray(cac_fn[k](obs, e))[m]
        return a
    return fn


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


def duong_bien_ho(env_va, env_te, cac_fn, chon_tu, quet, nhom, n_val, n_test, seed):
    """Duong bien cua mot ho: cach gan nhom tu gia bong, Pareto tren val, doc tren test."""
    C, T = (np.stack([quet[i][j] for i in chon_tu]) for j in (1, 2))
    diem = []
    for r in chon_theo_nhom(C, T):
        idx = np.array([chon_tu[i] for i in r])
        tong, _, t, d = chay(env_va, tron(cac_fn, idx[nhom]), n_val, 7777, nhom)
        diem.append({"idx": idx, "val_cost": float(tong.mean()), "val_fill": fill(t, d)})
    pts = pareto(diem)
    for d in pts:
        d["tong"], _, t, n = chay(env_te, tron(cac_fn, d["idx"][nhom]), n_test, seed, nhom)
        d["test_cost"], d["test_fill"] = float(d["tong"].mean()), fill(t, n)
    return pts


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
    res = ROOT / cfg["paths"]["results_dir"]
    env_te, env_va = make_env(cfg, data), make_env(cfg, data, mode="val")
    pols = make_policies(cfg, env_te, a.checkpoint, include_iso=False, include_group=False)
    nhom = demand_group_index(env_te.mean_demand)
    seed = cfg["eval"].get("seed", 1000)

    bien = json.loads((res / "baseline_frontier.json").read_text(encoding="utf-8"))["Newsvendor dự báo"]
    F, sigma = load_forecast(res)

    def nv(p):
        pol = ForecastNewsvendor(env_te.n_pairs, forecast=F, sigma=sigma, **p)
        return lambda obs, e, pol=pol: pol.get_action(build_env_state(e))

    cac_fn = [nv(d["params"]) for d in bien] + [pols["IPPO"]]
    I = len(cac_fn) - 1
    quet = [chay(env_va, fn, a.tune_episodes, 7777, nhom) for fn in cac_fn]
    ket_hop = duong_bien_ho(env_va, env_te, cac_fn, list(range(I + 1)), quet, nhom, a.tune_episodes, a.episodes, seed)
    theo_nhom = duong_bien_ho(env_va, env_te, cac_fn, list(range(I)), quet, nhom, a.tune_episodes, a.episodes, seed)

    kq = {ten: chay(env_te, pols[ten], a.episodes, seed, nhom) for ten in ("IPPO", "Newsvendor dự báo")}
    muc = fill(kq["IPPO"][2], kq["IPPO"][3])
    gan = min(ket_hop, key=lambda d: abs(d["test_fill"] - muc))
    out = {
        "muc_ippo": muc,
        "tong": {ten: {"cost": float(v[0].mean()), "fill": fill(v[2], v[3])} for ten, v in kq.items()},
        "tai_muc_ippo": {"Kết hợp": doc_tai(*duong_test(ket_hop), muc),
                         "Newsvendor dự báo theo nhóm": doc_tai(*duong_test(theo_nhom), muc),
                         "Newsvendor dự báo": doc_tai(*duong_test(bien), muc)},
        "cau_hinh_gan_nhat": {"cost": gan["test_cost"], "fill": gan["test_fill"],
                              "nhom_dung_ippo": [DEMAND_GROUP_NAMES[g] for g, i in enumerate(gan["idx"]) if i == I]},
        "so_voi_ippo": paired_test(gan["tong"], kq["IPPO"][0]),
        "theo_nhom": {DEMAND_GROUP_NAMES[g]: {ten: {"cost": float(v[1][g]), "fill": 1 - float(v[2][g] / max(v[3][g], 1e-6))}
                                              for ten, v in kq.items()} for g in range(nhom.max() + 1)},
    }
    p = res / f"hybrid_policy_{a.tag}.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("tong", "tai_muc_ippo", "cau_hinh_gan_nhat")}, indent=1, ensure_ascii=False))
    print(f"Da luu {p}")


if __name__ == "__main__":
    main()
