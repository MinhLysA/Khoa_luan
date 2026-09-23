"""
scripts/regime_analysis.py
==========================
THUC NGHIEM B - hieu nang theo GIAI DOAN NHU CAU (khong huan luyen lai, chi
danh gia checkpoint da co tren mien TEST).

Thuc nghiem A (evaluate.py, iso_service.py) tinh trung binh ca 365 ngay. O
day ta tach ra theo tung loai ngay de tra loi: IPPO co loi the o giai doan
nhu cau bien dong khong, va chinh sach co dien co tot hon khi nhu cau on dinh
khong?

B1 - Giai doan tu nhien trong du lieu M5 (moi ngay mien test gan nhan):
     * Do bien dong: dev_t = trung binh qua 10 kho |D_wt / MA28_wt - 1|, voi
       MA28 la trung binh 28 ngay TRUOC do. Chia tam phan vi -> On dinh /
       Trung binh / Bien dong.
     * Lich: ngay co su kien (le, the thao...), ngay SNAP, cuoi tuan, mua le
       cuoi nam (thang 11-12).
B2 - Soc cau nhan tao: nhan cau cua mot cua so ngay trong mien test voi mot
     he so (vd x1,5 tang dot bien, x0,5 sut giam), do fill rate / chi phi
     truoc - trong - sau soc va toc do hoi phuc.

Cac chinh sach: IPPO, 3 baseline tinh chinh truc tiep, va cac baseline da
chinh o CUNG MUC PHUC VU voi IPPO (tu iso_service.json) - chi so sanh chi phi
giua cac chinh sach co fill rate tuong duong moi cong bang.

Cach dung:
    python scripts/regime_analysis.py --checkpoint checkpoints/best_model_seed42.pth
    python scripts/regime_analysis.py --episodes 10 --shock_factors 1.5 0.5
"""
import argparse
import json

import numpy as np

from common import ROOT, load_config, load_data, make_env, make_policies

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VOL_GROUPS = ["Ổn định", "Trung bình", "Biến động"]


def run_daily(env, action_fn, seeds):
    """Chay cac episode, cong don theo CHI SO NGAY THAT cua du lieu."""
    T = env.demand_data.shape[0]
    acc = {k: np.zeros(T) for k in ["cost", "demand", "stockout", "visits"]}
    for s in seeds:
        obs, _ = env.reset(seed=s)
        while True:
            obs, _, term, trunc, info = env.step(action_fn(obs, env))
            d = info["day_index"]
            acc["cost"][d] += (info["cost_holding"] + info["cost_stockout"]
                               + info["cost_ordering"] + info["cost_overflow"])
            acc["demand"][d] += info["demand"]
            acc["stockout"][d] += info["stockout"]
            acc["visits"][d] += 1
            if term or trunc:
                break
    return acc


def summarize(acc, mask):
    m = mask & (acc["visits"] > 0)
    dem = acc["demand"][m].sum()
    return {"n_days": int(m.sum()),
            "fill_rate": float(1 - acc["stockout"][m].sum() / max(dem, 1e-9)),
            "cost_per_day": float(acc["cost"][m].sum() / max(acc["visits"][m].sum(), 1))}


def block_bootstrap_gap(acc_a, acc_b, mask, n_boot=2000, block=7, seed=0):
    """Khoang tin cay 95% cho chenh lech chi phi (A/B - 1) tren cac ngay thuoc
    mask, lay mau lai theo KHOI 7 ngay lien tiep (giu tu tuong quan giua cac
    ngay gan nhau - cac ngay khong doc lap nen khong dung bootstrap tung ngay)."""
    days = np.flatnonzero(mask & (acc_a["visits"] > 0) & (acc_b["visits"] > 0))
    if len(days) == 0:
        return None
    # Tong chi phi (khong chia so lan ghe tung ngay) -> uoc luong diem trung
    # voi cot "Chenh" trong bang (hai chinh sach dung cung seed nen so lan ghe
    # moi ngay bang nhau, ty so tong chi phi = ty so chi phi trung binh/ngay).
    ca = acc_a["cost"][days]
    cb = acc_b["cost"][days]
    blocks = days // block
    uniq = np.unique(blocks)
    idx = [np.flatnonzero(blocks == u) for u in uniq]
    rng = np.random.default_rng(seed)
    gaps = np.empty(n_boot)
    for k in range(n_boot):
        pick = np.concatenate([idx[j] for j in rng.integers(len(idx), size=len(idx))])
        gaps[k] = ca[pick].sum() / cb[pick].sum() - 1
    lo, hi = np.quantile(gaps, [0.025, 0.975])
    return {"gap_percent": float(100 * (ca.sum() / cb.sum() - 1)),
            "ci95_low": float(100 * lo), "ci95_high": float(100 * hi),
            "n_blocks": int(len(uniq)), "significant": bool(lo > 0 or hi < 0)}


def label_days(demand, cal, lo, hi):
    """Nhan cho moi ngay trong [lo, hi). Tra ve dict ten nhom -> mask (T,)."""
    T = demand.shape[0]
    wh = demand.sum(axis=2)                                  # (T, n_kho)
    cs = np.cumsum(np.vstack([np.zeros((1, wh.shape[1])), wh]), axis=0)
    ma = (cs[28:-1] - cs[:-29]) / 28.0                       # trung binh 28 ngay truoc t
    dev = np.full(T, np.nan)
    dev[28:] = np.abs(wh[28:] / np.maximum(ma, 1e-9) - 1.0).mean(axis=1)

    in_test = np.zeros(T, bool)
    in_test[lo:hi] = True
    q1, q2 = np.nanquantile(dev[in_test], [1 / 3, 2 / 3])
    groups = {
        "Ổn định": in_test & (dev <= q1),
        "Trung bình": in_test & (dev > q1) & (dev <= q2),
        "Biến động": in_test & (dev > q2),
    }
    if cal is not None:
        day = np.arange(T)
        month = cal[:, 8:20].argmax(axis=1) + 1
        groups.update({
            "Ngày có sự kiện": in_test & (cal[:, 4:8].sum(axis=1) > 0),
            "Ngày SNAP": in_test & (cal[:, 0:3].sum(axis=1) > 0),
            "Cuối tuần": in_test & (day % 7 < 2),       # ngay 0 cua M5 la thu Bay
            "Mùa lễ (T11-T12)": in_test & np.isin(month, [11, 12]),
            "Ngày thường": in_test & (cal[:, 4:8].sum(axis=1) == 0)
                           & (cal[:, 0:3].sum(axis=1) == 0) & (day % 7 >= 2),
        })
    return groups, dev, (float(q1), float(q2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/best_model_seed42.pth")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--shock_factors", type=float, nargs="*", default=[1.5, 0.5])
    ap.add_argument("--shock_start", type=int, default=1650,
                    help="Ngay bat dau soc (phai nam trong moi episode test)")
    ap.add_argument("--shock_len", type=int, default=28)
    ap.add_argument("--tag", default="")
    ap.add_argument("--n_boot", type=int, default=2000,
                    help="So lan bootstrap theo khoi tuan cho khoang tin cay")
    a = ap.parse_args()

    cfg = load_config(a.config)
    data = load_data(cfg)
    base_seed = cfg["eval"].get("seed", 1000)
    seeds = [base_seed + i for i in range(a.episodes)]
    suffix = f"_{a.tag}" if a.tag else ""
    out_dir = ROOT / cfg["paths"]["results_dir"]

    env = make_env(cfg, data)
    policies = make_policies(cfg, env, a.checkpoint)
    T = env.demand_data.shape[0]
    groups, dev, (q1, q2) = label_days(env.demand_data, data["calendar_features"],
                                       env.val_day, T)

    # ------------------------- B1: giai doan tu nhien -----------------------
    print(f"B1 | {a.episodes} episode | nguong bien dong: q1={q1:.3f}, q2={q2:.3f}")
    b1, accs = {}, {}
    for name, fn in policies.items():
        acc = accs[name] = run_daily(env, fn, seeds)
        b1[name] = {g: summarize(acc, m) for g, m in groups.items()}
        b1[name]["Toàn miền test"] = summarize(acc, np.ones(T, bool))
        print(f"  {name:<26s}" + "  ".join(
            f"{g}: {b1[name][g]['fill_rate']:.1%}/{b1[name][g]['cost_per_day']:,.0f}"
            for g in VOL_GROUPS))

    # IPPO so voi baseline cung muc phuc vu: chenh lech chi phi theo nhom
    gaps = {}
    for ref in [p for p in policies if p != "IPPO"]:
        gaps[ref] = {g: 100 * (b1["IPPO"][g]["cost_per_day"] / b1[ref][g]["cost_per_day"] - 1)
                     for g in b1["IPPO"]}

    # Khoang tin cay 95% (bootstrap khoi tuan) cho chenh lech chi phi IPPO
    ci = {ref: {g: block_bootstrap_gap(accs["IPPO"], accs[ref], m, a.n_boot)
                for g, m in groups.items()}
          for ref in policies if ref != "IPPO"}
    for ref, per_g in ci.items():
        print(f"  CI95 IPPO vs {ref}: " + "  ".join(
            f"{g}: [{v['ci95_low']:+.1f}, {v['ci95_high']:+.1f}]"
            for g, v in per_g.items() if v and g in VOL_GROUPS))

    # ------------------------- B2: soc cau nhan tao -------------------------
    s0, L = a.shock_start, a.shock_len
    pre, post = (s0 - 28, s0), (s0 + L, s0 + L + 28)
    b2, traj = {}, {}
    for f in a.shock_factors:
        dem = data["demand_data"].astype(np.float32).copy()
        dem[s0:s0 + L] *= f
        env_s = make_env(cfg, data, demand_override=dem)
        pols = make_policies(cfg, env_s, a.checkpoint)
        b2[str(f)], traj[str(f)] = {}, {}
        print(f"B2 | soc x{f} tai ngay [{s0}, {s0 + L})")
        for name, fn in pols.items():
            acc = run_daily(env_s, fn, seeds)
            win = {}
            for lab, (x, y) in [("Trước sốc", pre), ("Trong sốc", (s0, s0 + L)),
                                ("Sau sốc", post)]:
                m = np.zeros(T, bool)
                m[x:y] = True
                win[lab] = summarize(acc, m)
            b2[str(f)][name] = win
            x, y = pre[0], post[1]
            with np.errstate(invalid="ignore", divide="ignore"):
                daily = 1 - acc["stockout"][x:y] / acc["demand"][x:y]
            traj[str(f)][name] = daily.tolist()
            print(f"  {name:<26s}" + "  ".join(
                f"{k}: {v['fill_rate']:.1%}" for k, v in win.items()))

    out = {"checkpoint": a.checkpoint, "episodes": a.episodes,
           "volatility_thresholds": {"q1": q1, "q2": q2},
           "n_days_per_group": {g: int(m.sum()) for g, m in groups.items()},
           "B1_regimes": b1, "B1_cost_gap_IPPO_vs_percent": gaps,
           "B1_cost_gap_ci95_block_bootstrap": ci,
           "B2_shock": {"start": s0, "len": L, "results": b2}}
    p = out_dir / f"regime_analysis{suffix}.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Da luu {p}")

    _plot(b1, gaps, traj, s0, L, pre, out_dir, suffix)


def _plot(b1, gaps, traj, s0, L, pre, out_dir, suffix):
    names = list(b1)
    cats = VOL_GROUPS + [g for g in b1["IPPO"] if g not in VOL_GROUPS
                         and g != "Toàn miền test"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    w = 0.8 / len(names)
    x = np.arange(len(cats))
    for i, n in enumerate(names):
        axes[0].bar(x + i * w, [b1[n][c]["fill_rate"] * 100 for c in cats], w, label=n)
    axes[0].set_xticks(x + 0.4 - w / 2)
    axes[0].set_xticklabels(cats, rotation=25, ha="right")
    axes[0].set_ylim(60, 100)
    axes[0].set_ylabel("Fill rate (%)")
    axes[0].set_title("Fill rate theo giai đoạn nhu cầu")
    axes[0].legend(fontsize=7)

    refs = [r for r in gaps if "cùng mức" in r] or list(gaps)
    w2 = 0.8 / len(refs)
    for i, r in enumerate(refs):
        axes[1].bar(x + i * w2, [gaps[r][c] for c in cats], w2, label=f"IPPO so với {r}")
    axes[1].axhline(0, c="k", lw=0.8)
    axes[1].set_xticks(x + 0.4 - w2 / 2)
    axes[1].set_xticklabels(cats, rotation=25, ha="right")
    axes[1].set_ylabel("Chênh lệch chi phí/ngày (%)\n(âm = IPPO rẻ hơn)")
    axes[1].set_title("Chênh lệch chi phí IPPO theo giai đoạn")
    lo = min(min(gaps[r][c] for c in cats) for r in refs)
    axes[1].set_ylim(min(lo * 1.35, -1), max(0, max(max(gaps[r][c] for c in cats) for r in refs)) + 1)
    axes[1].legend(fontsize=7, loc="lower left")
    plt.tight_layout()
    plt.savefig(out_dir / f"regime_analysis{suffix}.png", dpi=140)
    plt.close()

    if not traj:
        return
    fig, axes = plt.subplots(1, len(traj), figsize=(7 * len(traj), 4.2), squeeze=False)
    for ax, (f, tr) in zip(axes[0], traj.items()):
        for n, v in tr.items():
            v = np.asarray(v, float)
            k = 7
            sm = np.convolve(np.nan_to_num(v, nan=1.0), np.ones(k) / k, mode="valid")
            ax.plot(np.arange(len(sm)) + pre[0] + k - 1, sm * 100, label=n, lw=1.3)
        ax.axvspan(s0, s0 + L, color="red", alpha=0.1, label="Cửa sổ sốc")
        ax.set_title(f"Sốc cầu x{f}: fill rate ngày (trượt 7 ngày)")
        ax.set_xlabel("Chỉ số ngày (M5)")
        ax.set_ylabel("Fill rate (%)")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / f"shock_test{suffix}.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    main()
