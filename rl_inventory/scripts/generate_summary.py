"""
scripts/generate_summary.py
============================
Xuat TONG_HOP_SO_LIEU.txt TU DONG bang cach doc lai config.yaml va cac file
ket qua da co (results/*.json, results/train_log.csv, checkpoints/,
data/processed/env_config.json).

Ly do can script nay: file TONG_HOP_SO_LIEU.txt truoc day duoc go tay, nen so
lieu de bi lech voi ket qua chay thuc te (vd iso_service.json tung ghi nhan
hai con so khac nhau giua cac lan chay ma khong ai phat hien kip). Chay lai
script nay sau moi lan train/eval/iso se luon cho mot file khop 100% voi
trang thai hien tai cua results/ va checkpoints/.

Cach dung:
    python scripts/generate_summary.py
"""

import sys
import json
import yaml
from pathlib import Path
from datetime import datetime

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.inventory_env import MultiWarehouseInventoryEnv

MISSING = "  (chua co - chay buoc lien quan roi chay lai script nay)"


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _fmt_money(x):
    return f"{x:,.0f}"


def section_env(cfg, env):
    e = cfg["env"]
    lines = [
        "PHAN 1: CAU HINH MOI TRUONG (env)",
        "-" * 78,
        f"  So kho / So SKU / So cap    : {e['n_warehouses']} / {e['n_skus']} / {env.n_pairs}",
        f"  Do dai episode              : {e['episode_length']} ngay",
        f"  Cua so nhin lai             : {e['lookback']} ngay",
        f"  Lead time                   : {e['lead_time_min']}-{e['lead_time_max']} ngay",
        f"  Suc chua kho (capacity_cover_days = {e['capacity_cover_days']})",
    ]
    for i in range(env.n_warehouses):
        lines.append(f"    Kho {i:<2d} : {env.suc_chua_kho[i]:>10,.1f}")
    lines += [
        "",
        f"  cp_lk (luu kho/dv/ngay)     : {e['cp_lk']}",
        f"  cp_th (thieu hang/dv)       : {e['cp_th']}",
        f"  cp_dh (dat hang/lan)        : {e['cp_dh']}",
        f"  pt_tk (phat tran kho/dv)    : {e['pt_tk']}",
        f"  phi_dv (he so phat dich vu) : {e['phi_dv']}",
        f"  muc_dv (nguong fill rate)   : {e['muc_dv']}",
        f"  normalize_reward_per_pair   : {e['normalize_reward_per_pair']}",
        f"  Bang muc dat hang (nhan doi cau): {e['order_multipliers']}",
        f"  Ngay tach train/test        : {e['split_day']}",
        "",
    ]
    return lines


def section_ppo(cfg):
    p = cfg["ppo"]
    lines = [
        "PHAN 2: CAU HINH THUAT TOAN PPO (IPPO)",
        "-" * 78,
        f"  hidden_dim                  : {p['hidden_dim']}",
        f"  n_steps / ppo_epochs        : {p['n_steps']} / {p['ppo_epochs']}",
        f"  mini_batch_size             : {p['mini_batch_size']}",
        f"  lr_actor / lr_critic        : {p['lr_actor']} / {p['lr_critic']}",
        f"  gamma / gae_lambda          : {p['gamma']} / {p['gae_lambda']}",
        f"  clip_eps / target_kl        : {p['clip_eps']} / {p['target_kl']}",
        f"  ent_coef (dau->cuoi)        : {p['ent_coef']} -> {p['ent_coef_end']}",
        f"  use_value_norm/clip_value   : {p['use_value_norm']} / {p['clip_value_loss']}",
        f"  normalize_adv_per_pair      : {p['normalize_adv_per_pair']}",
        f"  total_episodes (config)     : {p['total_episodes']}",
        f"  Seed                        : {cfg.get('seed')}",
    ]

    log_path = ROOT / cfg["paths"]["results_dir"] / "train_log.csv"
    if log_path.exists():
        import pandas as pd
        df = pd.read_csv(log_path)
        last = df.iloc[-1]
        first_ent = df["entropy"].dropna().iloc[0] if df["entropy"].notna().any() else float("nan")
        lines += [
            "",
            f"  Episode da huan luyen thuc te: {int(last['episode'])} "
            f"(thoi gian: {last['elapsed_s']/60:.1f} phut)",
            f"  Entropy dau -> cuoi          : {first_ent:.3f} -> {last['entropy']:.3f}",
            f"  Explained variance (cuoi)    : {last['explained_variance']:.3f}",
            f"  Approx KL (cuoi)             : {last['approx_kl']:.5f}",
            f"  Eval fill rate (cuoi, deterministic): {last['eval_fill']:.3f}",
        ]
    else:
        lines.append(MISSING)
    lines.append("")
    return lines


def section_baselines(cfg):
    lines = ["PHAN 3: CAU HINH CAC BASELINE", "-" * 78]
    tuned = _load_json(ROOT / cfg["paths"]["results_dir"] / "baseline_params.json")
    if tuned:
        for name, params in tuned.items():
            lines.append(f"  {name:<12s}: {params}")
    else:
        lines.append("  Chua tinh chinh -> dang dung tham so mac dinh trong evaluate.py")
        lines.append(MISSING)
    lines.append("")
    return lines


def section_eval(cfg):
    lines = ["PHAN 4: KET QUA DANH GIA TRUC TIEP (mien test)", "-" * 78]
    summary = _load_json(ROOT / cfg["paths"]["results_dir"] / "summary.json")
    if not summary:
        lines.append(MISSING)
        lines.append("")
        return lines

    rows = summary["summary"]
    lines.append(f"  So episode danh gia: {summary['stat'].get('n_episodes', '?')}")
    lines.append(f"  {'Chinh sach':<12}{'Tong chi phi':>16}{'Fill rate':>12}"
                 f"{'Luu kho':>13}{'Thieu hang':>13}{'Dat hang':>12}{'Tran kho':>11}")
    for r in sorted(rows, key=lambda r: r["total_cost_mean"]):
        lines.append(
            f"  {r['policy']:<12}{_fmt_money(r['total_cost_mean']):>16}"
            f"{r['fill_rate_mean']*100:>11.2f}%"
            f"{_fmt_money(r['holding_cost_mean']):>13}"
            f"{_fmt_money(r['stockout_cost_mean']):>13}"
            f"{_fmt_money(r['ordering_cost_mean']):>12}"
            f"{_fmt_money(r['overflow_cost_mean']):>11}")
    lines.append("")
    return lines


def section_iso(cfg):
    lines = ["PHAN 5: SO SANH O CUNG MUC PHUC VU (iso-service)", "-" * 78]
    iso = _load_json(ROOT / cfg["paths"]["results_dir"] / "iso_service.json")
    if not iso:
        lines.append(MISSING)
        lines.append("")
        return lines

    lines.append(f"  Nguong fill rate muc tieu: {iso['target_fill']*100:.2f}%")
    lines.append(f"  {'Chinh sach':<12}{'Tong chi phi':>16}{'Fill rate':>12}  {'Tham so tot nhat'}")
    for name, r in iso["ket_qua"].items():
        if r.get("khong_dat"):
            lines.append(f"  {name:<12}{'khong dat duoc nguong nay':>28}")
            continue
        lines.append(f"  {name:<12}{_fmt_money(r['cost']):>16}{r['fill']*100:>11.2f}%  {r.get('params') or ''}")
    lines.append("")
    return lines


def section_stat(cfg):
    lines = ["PHAN 6: KIEM DINH THONG KE (IPPO vs baseline tot nhat)", "-" * 78]
    st = _load_json(ROOT / cfg["paths"]["results_dir"] / "statistical_test.json")
    if not st:
        lines.append(MISSING)
        lines.append("")
        return lines

    lines += [
        f"  Baseline tot nhat  : {st['best_baseline']}",
        f"  Chenh lech chi phi : {st['gap_percent']:+.2f}%",
        f"  Welch t-test       : t={st['t_stat']:.3f}, p={st['p_ttest']:.3e}",
        f"  Wilcoxon           : p={st['p_wilcoxon']:.3e}",
        f"  Cohen's d          : {st['cohen_d']:.3f}",
        f"  So episode / mien  : {st['n_episodes']} / {st['mode']}",
        "",
    ]
    return lines


def section_preprocess(cfg, env):
    lines = ["PHAN 7: CAU HINH TIEN XU LY DU LIEU", "-" * 78]
    meta = _load_json(ROOT / cfg["paths"]["data_dir"] / "env_config.json")
    if meta:
        lines += [
            f"  Nguon du lieu            : {meta.get('mode')}",
            f"  Cua hang su dung         : {meta.get('stores')}",
            f"  So SKU sau loc           : {len(meta.get('top_items', []))}",
            f"  Phuong phap chon SKU     : {meta.get('sku_selection')}",
            f"  min_mean_demand (tien xu ly): {meta.get('min_mean_demand')}",
        ]
    else:
        lines.append(MISSING)

    md = env.mean_demand
    lines += [
        "",
        f"  Cau trung binh moi cap   : {md.mean():.2f} dv/ngay",
        f"  Trung vi cau             : {np.median(md):.2f} dv/ngay",
        f"  Cap gan nhu khong ban (<0.1 dv/ngay): {int((md < 0.1).sum())}/{env.n_pairs}",
        "",
    ]
    return lines


def section_files(cfg):
    lines = ["PHAN 8: FILE KET QUA HIEN CO", "-" * 78]
    results_dir = ROOT / cfg["paths"]["results_dir"]
    checkpoint_dir = ROOT / cfg["paths"]["checkpoint_dir"]
    for label, d in [("results/", results_dir), ("checkpoints/", checkpoint_dir)]:
        lines.append(f"  {label}")
        if d.exists():
            for f in sorted(d.iterdir()):
                lines.append(f"    [OK] {f.name}")
        else:
            lines.append("    (thu muc chua ton tai)")
    lines.append("")
    return lines


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))

    demand_data = calendar_features = None
    p = ROOT / cfg["paths"]["data_dir"] / "demand_data.npy"
    if p.exists():
        demand_data = np.load(str(p))
    p = ROOT / cfg["paths"]["data_dir"] / "calendar_features.npy"
    if p.exists():
        calendar_features = np.load(str(p))
        if calendar_features.size == 0:
            calendar_features = None

    env = MultiWarehouseInventoryEnv(config=cfg["env"], demand_data=demand_data,
                                     calendar_features=calendar_features, mode="test")

    header = [
        "=" * 80,
        "  TONG HOP SO LIEU -- DU AN IPPO QUAN LY TON KHO",
        f"  Tu dong xuat luc: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
        "  (File nay duoc sinh boi scripts/generate_summary.py - KHONG sua tay,",
        "   chay lai script sau moi lan train/eval/iso de cap nhat.)",
        "=" * 80,
        "",
    ]

    body = []
    for fn in (section_env, section_ppo, section_baselines, section_eval,
              section_iso, section_stat, section_preprocess, section_files):
        args = (cfg, env) if fn in (section_env, section_preprocess) else (cfg,)
        body += fn(*args)
        body.append("")

    out_path = ROOT / "TONG_HOP_SO_LIEU.txt"
    out_path.write_text("\n".join(header + body), encoding="utf-8")
    print(f"Da xuat: {out_path}")


if __name__ == "__main__":
    main()
