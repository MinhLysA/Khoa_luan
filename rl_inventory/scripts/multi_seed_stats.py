"""
scripts/multi_seed_stats.py
===========================
[Nhieu seed] Thong ke chinh cho Chuong 4, tinh tren chuoi chi phi THEO NGAY cua mot lan
chay lien tuc trong mien test (cung nhu cau, cung hat giong lead time cho moi
chinh sach), de tranh dung cac episode chong lan lam quan sat doc lap.

Voi moi baseline:
  * trung binh +- do lech chuan chenh lech chi phi giua cac seed huan luyen;
  * IQM (trung binh lien tu phan vi, Agarwal 2021) giua cac seed;
  * CI 95% bootstrap phan tang hai cap: lay mau lai seed huan luyen va lay mau lai
    khoi ngay, thu voi khoi 7 va 30 ngay. Khoi cuoi ngan hon van duoc giu.
Bootstrap chi giam chu khong loai bo tu tuong quan, va khong thay duoc viec them seed.

Cach dung:
    python scripts/multi_seed_stats.py --seeds 42 1 2 3 4
"""
import argparse
import json

import numpy as np
from scipy.stats import trim_mean

from common import ROOT, load_config, load_data, make_env, make_policies
from regime_analysis import run_daily


def iqm(x, axis=-1):
    return trim_mean(x, 0.25, axis=axis)


def bootstrap_phan_tang(C_ippo, C_base, block, n_boot, rng):
    """C_ippo (n_seed, n_ngay), C_base (n_ngay). Tra ve (TB, IQM, CI cua IQM) tinh bang %."""
    khoi = np.arange(C_base.size) // block
    n_khoi = khoi.max() + 1
    A = np.stack([np.bincount(khoi, c, n_khoi) for c in C_ippo])     # (n_seed, n_khoi)
    B = np.bincount(khoi, C_base, n_khoi)
    n_seed = len(C_ippo)
    picks = rng.integers(n_khoi, size=(n_boot, n_khoi))
    seeds = rng.integers(n_seed, size=(n_boot, n_seed))
    gaps = np.empty((n_boot, n_seed))
    for i in range(n_boot):
        gaps[i] = A[seeds[i]][:, picks[i]].sum(axis=1) / B[picks[i]].sum() - 1
    diem = A.sum(axis=1) / B.sum() - 1
    lo, hi = np.quantile(iqm(gaps, axis=1), [0.025, 0.975])
    return {"tb_percent": float(100 * diem.mean()), "sd_percent": float(100 * diem.std(ddof=1)) if n_seed > 1 else 0.0,
            "iqm_percent": float(100 * iqm(diem)), "ci95_low": float(100 * lo), "ci95_high": float(100 * hi),
            "co_y_nghia": bool(lo > 0 or hi < 0), "cac_seed_percent": (100 * diem).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--ten", default="main", help="Tien to lan chay, vd u5")
    a = ap.parse_args()

    cfg = load_config(a.config)
    env = make_env(cfg, load_data(cfg))
    lead_seeds = [cfg["eval"].get("seed", 1000) + i for i in range(a.episodes)]
    ck = {s: ROOT / f"checkpoints/best_model_{a.ten}_s{s}.pth" for s in a.seeds}
    ck = {s: p.relative_to(ROOT).as_posix() for s, p in ck.items() if p.exists()}
    if not ck:
        raise SystemExit(f"Chua co checkpoint best_model_{a.ten}_s*.pth")

    lo, hi = env.val_day, env.demand_data.shape[0]
    ippo, base = [], {}
    for i, (s, p) in enumerate(ck.items()):
        pols = make_policies(cfg, env, p, include_iso=False)
        ippo.append(run_daily(env, pols["IPPO"], lead_seeds)["cost"][lo:hi])
        if i == 0:                            # baseline khong phu thuoc seed huan luyen
            base = {n: run_daily(env, fn, lead_seeds)["cost"][lo:hi]
                    for n, fn in pols.items() if n != "IPPO"}
    ippo = np.stack(ippo)

    rng = np.random.default_rng(0)
    kq = {n: {f"khoi_{k}": bootstrap_phan_tang(ippo, c, k, a.n_boot, rng) for k in (7, 30)}
          for n, c in base.items()}
    for n, v in kq.items():
        r = v["khoi_7"]
        print(f"{n:<26} TB {r['tb_percent']:+.2f}% (SD {r['sd_percent']:.2f}) IQM {r['iqm_percent']:+.2f}% "
              f"CI [{r['ci95_low']:+.2f}; {r['ci95_high']:+.2f}]")
    p = ROOT / cfg["paths"]["results_dir"] / f"multi_seed_stats{'' if a.ten == 'main' else '_' + a.ten}.json"
    p.write_text(json.dumps({"seeds": list(ck), "so_ngay": int(hi - lo), "lan_danh_gia": a.episodes,
                             "gap_chi_phi": kq}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Da luu {p}")


if __name__ == "__main__":
    main()
