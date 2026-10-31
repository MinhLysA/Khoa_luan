"""
scripts/perfect_demand_benchmark.py
===================================
Moc biet truoc nhu cau: dat hang theo chinh sach du bao nhung thay du bao bang nhu
cau that cua cac ngay sap toi. Thoi gian giao hang van ngau nhien va luong dat van
duoc lam tron, nen day khong phai can duoi chinh xac. Moc nay chi cho biet thong tin
ve tuong lai dang gia bao nhieu. L duoc chon tren mien val.

Cach dung:
    python scripts/perfect_demand_benchmark.py
"""
import argparse
import json

import numpy as np

from common import ROOT, load_config, load_data, make_env
from baselines.traditional_policies import ForecastNewsvendor
from compare_same_service import chay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--episodes", type=int, default=30)
    a = ap.parse_args()

    cfg = load_config(a.config)
    data = load_data(cfg)
    D = data["demand_data"].astype(np.float32)
    D = D.reshape(D.shape[0], -1)
    F = np.stack([D, np.vstack([D[1:], D[-1:]]), np.vstack([D[2:], D[-2:]])], axis=-1)
    sigma = np.zeros(D.shape, dtype=np.float32)
    env_v, env_t = make_env(cfg, data, mode="val"), make_env(cfg, data, mode="test")

    def pol(env, L):
        return ForecastNewsvendor(env.n_pairs, F, sigma, cr_override=0.5, lead_time=float(L))

    diem = {L: chay(env_v, pol(env_v, L), 2, 7777) for L in range(1, 5)}
    dat = {L: d for L, d in diem.items() if d[1] >= env_v.muc_dv} or diem
    L = min(dat, key=lambda k: dat[k][0])
    c, f = chay(env_t, pol(env_t, L), a.episodes, cfg["eval"].get("seed", 1000))
    p = ROOT / cfg["paths"]["results_dir"] / "perfect_demand.json"
    p.write_text(json.dumps({"L": L, "cost": c, "fill": f}, indent=2), encoding="utf-8")
    print(f"Biet truoc nhu cau, L={L}: chi phi {c:,.0f}, fill {f:.2%}. Da luu {p}")


if __name__ == "__main__":
    main()
