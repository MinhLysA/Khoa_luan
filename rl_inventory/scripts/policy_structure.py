"""
scripts/policy_structure.py
===========================
[Cau truc chinh sach] Chinh sach IPPO da hoc co dang gi so voi chinh sach co cau truc (khong train):
  1. Xac suat dat hang va luong dat ky vong theo so ngay ton, tung nhom quy mo cau.
  2. Nguong s trung vi cua (s,S) tinh chinh theo nhom (quy ve so ngay ton) de chong hinh.
  3. Muc do xap xi order-up-to: hoi quy luong dat theo (S* - vi the ton kho).
  4. Anh huong cua muc day kho: o cung so ngay ton, so sanh kho vang va kho chat.

Cach dung:
    python scripts/policy_structure.py --checkpoint checkpoints/best_model_main_s42.pth --tag main
"""
import argparse
import json

import numpy as np

from common import ROOT, load_config, load_data, make_env, make_policies
from baselines.traditional_policies import DEMAND_GROUP_NAMES, demand_group_index

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COVER = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 12, np.inf]


def thu_thap(env, fn, seeds):
    """Moi dong la mot quyet dinh: so ngay ton, luong dat, muc day kho, vi the ton kho."""
    cover, qty, util, pos, day_mean = [], [], [], [], []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        while True:
            inv_pos = env.inventory + env.pipeline_orders.sum(axis=0)
            kho = env.inventory.reshape(env.n_warehouses, env.n_skus).sum(axis=1) / env.suc_chua_kho
            a = fn(obs, env)
            cover.append(inv_pos / np.maximum(env.mean_demand, 1e-6))
            qty.append(env.action_to_qty(a))
            util.append(np.repeat(kho, env.n_skus))
            pos.append(inv_pos)
            day_mean.append(np.maximum(env.demand_history.mean(axis=0), env.mean_demand * 0.1))
            obs, _, te, tr, _ = env.step(a)
            if te or tr:
                break
    return {k: np.concatenate(v) for k, v in dict(cover=cover, qty=qty, util=util, pos=pos,
                                                  day_mean=day_mean).items()}


def ho_so(r, nhom_pair, g, n_pairs):
    """Xac suat dat, luong dat quy ve ngay cau theo tung khoang so ngay ton cua nhom g."""
    m = np.tile(nhom_pair == g, len(r["cover"]) // n_pairs)
    rows = []
    for lo, hi in zip(COVER[:-1], COVER[1:]):
        c = m & (r["cover"] >= lo) & (r["cover"] < hi)
        if c.sum() >= 30:
            rows.append({"khoang": [lo, None if np.isinf(hi) else hi], "n": int(c.sum()),
                         "p_dat": float((r["qty"][c] > 0).mean()),
                         "luong_dat_ngay_cau": float((r["qty"][c] / r["day_mean"][c]).mean())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/best_model_main_s42.pth")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()

    cfg = load_config(a.config)
    env = make_env(cfg, load_data(cfg))
    pols = make_policies(cfg, env, a.checkpoint)
    seeds = [cfg["eval"].get("seed", 1000) + i for i in range(a.episodes)]
    nhom = demand_group_index(env.mean_demand)
    res = ROOT / cfg["paths"]["results_dir"]

    r = thu_thap(env, pols["IPPO"], seeds)
    ho = {DEMAND_GROUP_NAMES[g]: ho_so(r, nhom, g, env.n_pairs) for g in range(4)}

    # (2)(3): tham so (s,S) theo nhom -> s, S* quy ve so ngay ton cua nhom
    p = res / "baseline_params_group.json"
    s_ngay, hoi_quy = {}, {}
    if p.exists():
        ts = json.loads(p.read_text(encoding="utf-8"))["params"]["(s,S)"]
        from scipy import stats
        n_mau = len(r["cover"]) // env.n_pairs
        for g in range(4):
            q = ts[str(g)]
            z = float(stats.norm.ppf(q["service_level"]))
            md = env.mean_demand[nhom == g]
            std = env.std_demand[nhom == g]
            s_i = md * q["lead_time"] + z * std * np.sqrt(q["lead_time"])
            eoq = q["q_factor"] * np.sqrt(2 * md * env.cp_dh / env.cp_lk)
            S_i = s_i + eoq
            s_ngay[DEMAND_GROUP_NAMES[g]] = float(np.median(s_i / md))
            m = np.tile(nhom == g, n_mau)
            x = np.tile(S_i, n_mau) - r["pos"][m]
            y = r["qty"][m]
            co_dat = x > 0
            hs = stats.linregress(x[co_dat], y[co_dat])
            hoi_quy[DEMAND_GROUP_NAMES[g]] = {"he_so": float(hs.slope), "chan": float(hs.intercept),
                                              "r2": float(hs.rvalue ** 2)}

    # (4): anh huong cua muc day kho o cung so ngay ton
    kho = []
    q1, q3 = np.quantile(r["util"], [0.25, 0.75])
    for lo, hi in zip(COVER[:-1], COVER[1:]):
        c = (r["cover"] >= lo) & (r["cover"] < hi)
        v, h = c & (r["util"] <= q1), c & (r["util"] >= q3)
        if v.sum() >= 30 and h.sum() >= 30:
            kho.append({"khoang": [lo, None if np.isinf(hi) else hi],
                        "p_dat_kho_vang": float((r["qty"][v] > 0).mean()),
                        "p_dat_kho_chat": float((r["qty"][h] > 0).mean())})

    out = {"checkpoint": a.checkpoint, "episodes": a.episodes, "ho_so": ho, "s_so_ngay_ton": s_ngay,
           "hoi_quy_order_up_to": hoi_quy, "muc_day_kho": {"q1": float(q1), "q3": float(q3), "bang": kho}}
    (res / f"policy_structure_{a.tag}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for i, (ten, rows) in enumerate(ho.items()):
        x = [(row["khoang"][0]) for row in rows]
        axes[0].plot(x, [row["p_dat"] * 100 for row in rows], "o-", label=ten)
        axes[1].plot(x, [row["luong_dat_ngay_cau"] for row in rows], "o-", label=ten)
        if ten in s_ngay:
            axes[0].axvline(s_ngay[ten], color=f"C{i}", ls=":", lw=1)
    axes[0].set_xlabel("Số ngày tồn kho")
    axes[0].set_ylabel("Xác suất đặt hàng (%)")
    axes[0].set_title("Đường chấm: ngưỡng s của (s,S) theo nhóm")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Số ngày tồn kho")
    axes[1].set_ylabel("Lượng đặt kỳ vọng (ngày cầu)")
    axes[2].plot([b["khoang"][0] for b in kho], [b["p_dat_kho_vang"] * 100 for b in kho], "o-", label="Kho vắng")
    axes[2].plot([b["khoang"][0] for b in kho], [b["p_dat_kho_chat"] * 100 for b in kho], "s-", label="Kho chật")
    axes[2].set_xlabel("Số ngày tồn kho")
    axes[2].set_ylabel("Xác suất đặt hàng (%)")
    axes[2].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(res / f"policy_structure_{a.tag}.png", dpi=140)
    plt.close()
    print(f"Da luu {res / f'policy_structure_{a.tag}.json'}")


if __name__ == "__main__":
    main()
