"""
scripts/compare_same_service.py
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
    python scripts/compare_same_service.py --checkpoint checkpoints/best_model_main_s42.pth

[Cung muc phuc vu] --grid fine: luoi min va dai doi xung +-0,2 diem % theo tung seed. Cau hinh
duoc chon tren val; IPPO duoc doi chieu voi chinh no tren val. Duong bien baseline
khong phu thuoc seed nen chi quet mot lan, luu results/baseline_frontier.json.
Neu fill test cua cau hinh chon lech IPPO qua 0,2 diem, chi phi duoc noi suy tren
duong bien test va ghi noi_suy=True.
    python scripts/compare_same_service.py --grid fine --checkpoint checkpoints/best_model_main_s42.pth --tag main_s42
"""

import os
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

from env.inventory_env import MultiWarehouseInventoryEnv, load_forecast
from agents.ppo_agent import PPOAgent
from baselines.traditional_policies import (
    EOQPolicy, SsPolicy, NewsvendorPolicy, CapacityAwareSs, ForecastNewsvendor,
    build_env_state,
    demand_group_index, expand_group_params
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _plot_iso(ket_qua, target_fill, out_path):
    """Bieu do cot: tong chi phi cua tung chinh sach o cung muc phuc vu."""
    names = [n for n in ket_qua if not ket_qua[n].get("khong_dat")]
    costs = [ket_qua[n]["cost"] for n in names]
    fills = [ket_qua[n]["fill"] * 100 for n in names]
    colors = ["#2ca02c" if n == "IPPO" else "#4C72B0" for n in names]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(names, costs, color=colors)
    for b, f in zip(bars, fills):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.01,
                f"{f:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("Tổng chi phí vận hành (miền test)")
    ax.set_title(f"So sánh ở cùng mức phục vụ (ngưỡng ≥ {target_fill*100:.2f}%)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def chay(env, policy, n, seed, is_ppo=False):
    costs, fills = [], []
    for i in range(n):
        obs, _ = env.reset(seed=seed + i)
        h = s = o = ov = 0.0
        dm = so = 0.0
        while True:
            if is_ppo:
                a, _, _ = policy.select_action(obs, deterministic=True)
                a = policy.env_action(a, env)
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


def luoi_min(env, results_dir):
    """[Cung muc phuc vu] Luoi min cua moi baseline: ten -> (lop, danh sach tham so)."""
    tc = dict(ordering_cost=env.cp_dh, holding_cost=env.cp_lk)
    sl = [round(x, 2) for x in np.arange(0.80, 1.0, 0.01)] + [0.995, 0.999]
    qf = [round(x, 2) for x in np.arange(0.25, 3.01, 0.25)]
    cr = [round(x, 2) for x in np.arange(0.60, 1.0, 0.01)]
    out = {
        "EOQ": (EOQPolicy, [dict(q_factor=q, lead_time=float(l), **tc) for q, l in itertools.product(
            qf, np.arange(1.0, 6.01, 0.5))]),
        "(s,S)": (SsPolicy, [dict(service_level=s, q_factor=q, lead_time=float(l), **tc)
                             for s, q, l in itertools.product(sl, qf, range(1, 6))]),
        "Newsvendor": (NewsvendorPolicy, [
            dict(cr_override=c, lead_time=float(l), holding_cost=env.cp_lk, stockout_cost=env.cp_th)
            for c, l in itertools.product(cr, range(1, 7))]),
        "(s,S) biết sức chứa": (CapacityAwareSs, [
            dict(service_level=s, q_factor=q, lead_time=float(l), kappa=k, **tc)
            # kappa > 1 vi hang dang ve duoc tinh vao cho chiem nhung ve dan, trong khi ton kho ban dan
            for s, q, l, k in itertools.product(sl, qf, range(1, 6), [0.8, 1.0, 1.1, 1.2, 1.5])]),
    }
    if (results_dir / "forecast_test.npy").exists():
        F, sigma = load_forecast(results_dir)
        out["Newsvendor dự báo"] = (ForecastNewsvendor, [
            dict(cr_override=c, lead_time=float(l), forecast=F, sigma=sigma)
            for c, l in itertools.product(cr, range(1, 5))])
    if os.environ.get("KLTN_THU"):                   # chay thu day chuyen: luoi thua
        out = {k: (c, g[::max(1, len(g) // 6)]) for k, (c, g) in out.items()}
    return out


AN = ("ordering_cost", "holding_cost", "stockout_cost", "forecast", "sigma")
# Cac tham so nay chi tac dong IPPO, nen cac bien the dung chung mot duong bien baseline
KHONG_DOI_BIEN = ("phi_dv", "lagrange_per_pair", "lagrange", "include_forecast")


def pareto(diem):
    """Giu cac diem khong bi diem khac vua re hon vua fill cao hon (tren val)."""
    tot, cost_min = [], float("inf")
    for d in sorted(diem, key=lambda x: -x["val_fill"]):
        if d["val_cost"] < cost_min:
            tot.append(d)
            cost_min = d["val_cost"]
    return tot


def chi_phi_theo_nhom(env, policy, n, seed, nhom):
    """(chi phi van hanh, luong thieu hang) cua tung nhom cau, trung binh tren n lan chay."""
    k = nhom.max() + 1
    cost, thieu = np.zeros(k), np.zeros(k)
    for i in range(n):
        env.reset(seed=seed + i)
        while True:
            *_, te, tr, inf = env.step(policy.get_action(build_env_state(env)))
            cost += np.bincount(nhom, inf["op_cost_pairs"], k)
            thieu += np.bincount(nhom, inf["stockout_pairs"], k)
            if te or tr:
                break
    return cost / n, thieu / n


def chon_theo_nhom(C, T, gia_bong=np.logspace(-2, 3, 40)):
    """C, T: (cau hinh, nhom). Voi moi gia bong mu cua mot don vi thieu hang, tung
    nhom lay cau hinh nho nhat C + mu * T. Fill toan he thong tuyen tinh theo luong
    thieu hang, nen day la loi giai Lagrange cua min chi phi voi fill cho truoc."""
    return {tuple(int(i) for i in np.argmin(C + mu * T, axis=0)) for mu in gia_bong}


def duong_bien_nhom(cls, grid, env_val, env_test, tune_eps, eval_eps, base_seed):
    """Duong bien cua ban theo nhom: moi nhom cau mot cau hinh rieng, de doi chung
    o cung muc phuc vu manh nhu ban theo nhom o phep so sanh truc tiep."""
    nhom = demand_group_index(env_val.mean_demand)
    bang = [chi_phi_theo_nhom(env_val, cls(env_val.n_pairs, **kw), tune_eps, 7777, nhom) for kw in grid]
    diem = []
    for idx in chon_theo_nhom(np.array([b[0] for b in bang]), np.array([b[1] for b in bang])):
        moi_nhom = {str(g): grid[i] for g, i in enumerate(idx)}
        kw = expand_group_params(moi_nhom, nhom)
        c, f = chay(env_val, cls(env_val.n_pairs, **kw), tune_eps, 7777)
        diem.append({"val_cost": c, "val_fill": f, "moi_nhom": moi_nhom})
    pts = pareto(diem)
    for d in pts:
        moi_nhom = d.pop("moi_nhom")
        d["test_cost"], d["test_fill"] = chay(env_test, cls(env_test.n_pairs, **expand_group_params(moi_nhom, nhom)),
                                              eval_eps, base_seed)
        d["params"] = {g: {k: v for k, v in p.items() if k not in AN} for g, p in moi_nhom.items()}
    return pts


def quet_duong_bien(env_val, env_test, luoi, tune_eps, eval_eps, base_seed, path, van_tay):
    """Quet luoi tren val, lay bien Pareto, danh gia bien tren test. Cache bi bo
    khi van_tay (moi truong, so tap, du bao) khac lan quet truoc."""
    khoa = path.with_suffix(".key")
    con_dung = path.exists() and khoa.exists() and khoa.read_text(encoding="utf-8") == van_tay
    if path.exists() and not con_dung:
        print("Cau hinh da doi so voi duong bien da luu -> quet lai.", flush=True)
        path.unlink()
    bien = json.loads(path.read_text(encoding="utf-8")) if con_dung else {}
    khoa.write_text(van_tay, encoding="utf-8")
    for ten, (cls, grid) in luoi.items():
        if ten in bien:
            continue
        print(f"[{ten}] quet {len(grid)} cau hinh tren val...", flush=True)
        diem = []
        for kw in grid:
            c, f = chay(env_val, cls(env_val.n_pairs, **kw), tune_eps, 7777)
            diem.append({"val_cost": c, "val_fill": f, "kw": kw})
        pts = pareto(diem)
        for d in pts:
            d["test_cost"], d["test_fill"] = chay(env_test, cls(env_test.n_pairs, **d["kw"]),
                                                  eval_eps, base_seed)
            d["params"] = {k: v for k, v in d.pop("kw").items() if k not in AN}
        bien[ten] = pts
        path.write_text(json.dumps(bien, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"   {len(pts)} diem Pareto", flush=True)
    for ten in ("EOQ", "(s,S)", "Newsvendor"):
        nhan = f"{ten} theo nhóm"
        if ten in luoi and nhan not in bien:
            print(f"[{nhan}] quet theo nhom...", flush=True)
            bien[nhan] = duong_bien_nhom(*luoi[ten], env_val, env_test, tune_eps, eval_eps, base_seed)
            path.write_text(json.dumps(bien, indent=2, ensure_ascii=False), encoding="utf-8")
    return bien


def duong_test(diem):
    """Duong bien test: chi phi thap nhat de dat fill >= f (khong giam theo f)."""
    d = sorted(diem, key=lambda x: x["test_fill"])
    f = [x["test_fill"] for x in d]
    c = np.minimum.accumulate([x["test_cost"] for x in d][::-1])[::-1]
    return f, c.tolist()


def doc_tai(f, c, muc):
    """Noi suy chi phi tai fill = muc; None neu muc nam ngoai duong bien."""
    return float(np.interp(muc, f, c)) if f[0] <= muc <= f[-1] else None


def chon_cung_muc(diem, ippo_cost, fill_val, fill_test, dai=0.002):
    """Cau hinh re nhat tren val trong dai +-dai quanh fill val cua IPPO, roi
    kiem tra lai tren test; lech qua dai thi noi suy tren duong bien test."""
    if not diem:
        return {"cost": None, "fill": None, "params": None, "khong_dat": True}
    trong = [d for d in diem if abs(d["val_fill"] - fill_val) <= dai]
    d = (min(trong, key=lambda x: x["val_cost"]) if trong
         else min(diem, key=lambda x: abs(x["val_fill"] - fill_val)))
    r = {"cost": d["test_cost"], "fill": d["test_fill"], "params": d["params"], "noi_suy": False}
    if abs(d["test_fill"] - fill_test) > dai:
        c = doc_tai(*duong_test(diem), fill_test)
        if c is None:
            return {**r, "cost": None, "khong_dat": True}
        # Cac phan tich theo giai doan can mot cau hinh that: lay diem gan nhat tren duong bien
        gan = min(diem, key=lambda x: abs(x["test_fill"] - fill_test))
        r.update(cost=c, fill=fill_test, noi_suy=True, params=gan["params"], fill_cau_hinh=gan["test_fill"])
    r["delta_ippo_percent"] = float(100 * (ippo_cost / r["cost"] - 1))
    return r


def chay_luoi_min(args, cfg, agent, env_tr, env_te, base_seed):
    res = ROOT / cfg["paths"]["results_dir"]
    res.mkdir(parents=True, exist_ok=True)
    ippo_cost, ippo_fill = chay(env_te, agent, args.eval_episodes, base_seed, is_ppo=True)
    _, fill_val = chay(env_tr, agent, args.tune_episodes, 7777, is_ppo=True)
    print(f"IPPO: test cost={ippo_cost:,.0f} fill={ippo_fill:.2%}; val fill={fill_val:.2%}", flush=True)
    du_bao = res / "forecast_test.npy"
    van_tay = json.dumps([{k: v for k, v in cfg["env"].items() if k not in KHONG_DOI_BIEN},
                          args.tune_episodes, args.eval_episodes,
                          du_bao.stat().st_mtime if du_bao.exists() else 0],
                         sort_keys=True, default=str)
    bien = quet_duong_bien(env_tr, env_te, luoi_min(env_tr, res), args.tune_episodes,
                           args.eval_episodes, base_seed, res / "baseline_frontier.json", van_tay)
    ket_qua = {"IPPO": {"cost": ippo_cost, "fill": ippo_fill, "params": None}}
    for ten, diem in bien.items():
        ket_qua[ten] = chon_cung_muc(diem, ippo_cost, fill_val, ippo_fill)
        r = ket_qua[ten]
        print(f"{ten:<24}" + ("khong dat muc nay" if r.get("khong_dat") else
              f"cost={r['cost']:,.0f} fill={r['fill']:.2%} delta={r['delta_ippo_percent']:+.1f}%"
              + (" (noi suy)" if r["noi_suy"] else "")))
    tag = f"_{args.tag}" if args.tag else ""
    out = res / f"same_service{tag}.json"
    out.write_text(json.dumps({"target_fill": ippo_fill, "fill_val_ippo": fill_val, "ket_qua": ket_qua},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Da luu {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/best_model_main_s42.pth")
    ap.add_argument("--tune_episodes", type=int, default=2)
    ap.add_argument("--eval_episodes", type=int, default=10)
    ap.add_argument("--tolerance", type=float, default=0.005,
                    help="Cho phep baseline thap hon IPPO toi da bao nhieu fill rate")
    ap.add_argument("--tune_mode", choices=["train", "val"], default="val",
                    help="[P1-1] Mien dung de quet luoi baseline")
    ap.add_argument("--grid", choices=["coarse", "fine"], default="coarse",
                    help="[Cung muc phuc vu] coarse: luoi cu, tai lap so cu; fine: luoi min, dai doi xung")
    ap.add_argument("--tag", default="")
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

    env_tr = MultiWarehouseInventoryEnv(env_cfg, demand, calf, price, price_series,
                                        mode=args.tune_mode)
    env_te = MultiWarehouseInventoryEnv(env_cfg, demand, calf, price, price_series, mode="test")
    base_seed = cfg["eval"].get("seed", 1000)

    # -- B1: fill rate cua IPPO ---------------------------------------------
    ck = ROOT / args.checkpoint
    if not ck.exists():
        print(f"Khong tim thay checkpoint {ck}")
        return
    agent = PPOAgent(env_te.obs_per_pair, env_te.n_pairs, env_te.n_action_levels,
                     config=cfg["ppo"], device="cpu")
    agent.load(str(ck), env=env_te)
    agent.network.eval()
    if env_te.lagrange_dim:
        env_tr.set_lagrange(env_te.get_lagrange())
    if args.grid == "fine":
        chay_luoi_min(args, cfg, agent, env_tr, env_te, base_seed)
        return
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

    fig_path = ROOT / paths["results_dir"] / "iso_service_chart.png"
    _plot_iso(ket_qua, target, fig_path)
    print(f"Da luu {fig_path}")


if __name__ == "__main__":
    main()
