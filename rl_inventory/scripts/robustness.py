"""
scripts/robustness.py
=====================
[P4-3] Do ben cua ket luan khi DIEU KIEN VAN HANH thay doi (zero-shot: giu
nguyen moi chinh sach da hoc / da tinh chinh, chi doi moi truong).

Kich ban (trong pham vi mo hinh con nhan duoc dau vao):
  * Suc chua kho 4 ngay / 6 ngay cau (goc 5).
  * Lead time luon 3 ngay; lead time 2-3 ngay (goc 1-3). Lead time toi da >3
    ngay KHONG danh gia duoc zero-shot: so chieu quan sat phu thuoc lead time
    toi da (hang dang ve theo tung ngay) -> can huan luyen lai (huong phat trien).
  * Nhu cau toan mien test +20% / -20% (xu huong tang / giam dai han).
Muc tieu muc phuc vu 80/90% KHONG danh gia o day: IPPO duoc huan luyen cho 85%,
doi muc tieu doi hoi huan luyen lai (huong phat trien).

Moi kich ban: danh gia tren toan quy dao test voi cung cac hat giong thoi gian
giao; IPPO so voi tung baseline bang kiem dinh theo cap.

Cach dung:
    python scripts/robustness.py --checkpoint checkpoints/best_model_main_s42.pth --tag main
"""
import argparse
import copy
import json

import numpy as np

from common import ROOT, load_config, load_data, make_env, make_policies
from evaluate import paired_test

KICH_BAN = {
    "Gốc": ({}, 1.0),
    "Sức chứa 4 ngày": ({"capacity_cover_days": 4.0}, 1.0),
    "Sức chứa 6 ngày": ({"capacity_cover_days": 6.0}, 1.0),
    "Lead time luôn 3 ngày": ({"lead_time_min": 3}, 1.0),
    "Lead time 2–3 ngày": ({"lead_time_min": 2}, 1.0),
    "Nhu cầu +20%": ({}, 1.2),
    "Nhu cầu −20%": ({}, 0.8),
}


def chay(env, fn, seeds):
    costs, fills = [], []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        c = dem = st = 0.0
        while True:
            obs, _, te, tr, info = env.step(fn(obs, env))
            c += (info["cost_holding"] + info["cost_stockout"]
                  + info["cost_ordering"] + info["cost_overflow"])
            dem += info["demand"]
            st += info["stockout"]
            if te or tr:
                break
        costs.append(c)
        fills.append(1 - st / max(dem, 1e-9))
    return np.array(costs), float(np.mean(fills))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/best_model_main_s42.pth")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    cfg0 = load_config(a.config)
    data = load_data(cfg0)
    seeds = [cfg0["eval"].get("seed", 1000) + i for i in range(a.episodes)]
    out = {}
    for ten, (env_over, he_so_cau) in KICH_BAN.items():
        cfg = copy.deepcopy(cfg0)
        cfg["env"].update(env_over)
        dem = data["demand_data"].astype(np.float32).copy()
        dem[cfg["env"]["val_day"]:] *= he_so_cau          # chi doi nhu cau mien test
        env = make_env(cfg, data, demand_override=dem)
        pols = make_policies(cfg, env, a.checkpoint, include_iso=False)
        res = {n: chay(env, fn, seeds) for n, fn in pols.items()}
        c_ippo = res["IPPO"][0]
        out[ten] = {n: {"cost": float(c.mean()), "fill": f,
                        **({} if n == "IPPO" else {"ippo_vs": paired_test(c_ippo, c)})}
                    for n, (c, f) in res.items()}
        dong = "  ".join(f"{n}: {v['ippo_vs']['gap_percent']:+.1f}%"
                         for n, v in out[ten].items() if n != "IPPO")
        print(f"{ten:<22} IPPO fill {res['IPPO'][1]:.1%} | IPPO so voi -> {dong}")

    suffix = f"_{a.tag}" if a.tag else ""
    p = ROOT / cfg0["paths"]["results_dir"] / f"robustness{suffix}.json"
    p.write_text(json.dumps({"checkpoint": a.checkpoint, "episodes": a.episodes,
                             "kich_ban": out}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Da luu {p}")


if __name__ == "__main__":
    main()
