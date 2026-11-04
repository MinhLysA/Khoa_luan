"""
scripts/run_experiments.py
==========================
MOT noi dinh nghia TOAN BO dot thuc nghiem cuoi cung: moi lan huan luyen va pipeline
danh gia. Ten lan chay (tag) -> ten day du: xem TEN_DAY_DU trong scripts/build_results.py.

    python scripts/run_experiments.py status              # tien do tung lan chay
    python scripts/run_experiments.py train main_s42      # train 1 lan chay (tu resume)
    python scripts/run_experiments.py train ablations     # lan luot moi ablation
    python scripts/run_experiments.py train full          # toan bo lan chay, tuan tu
    python scripts/run_experiments.py train forecast_safety  # mot nhom mo rong, xem NHOM_LAN_CHAY_MO_RONG
    python scripts/run_experiments.py eval                # toan bo pipeline danh gia
    python scripts/run_experiments.py tonghop             # chi dung lai KET_QUA.txt

Train va eval deu CHAY NOI TIEP: bi ngat thi chay lai cung lenh. Train resume tu checkpoint
gan nhat; eval bo qua buoc da co ket qua moi hon dau vao (checkpoint, tham so baseline,
ket qua cung muc phuc vu). Muon chay lai mot buoc: xoa file ket qua cua buoc do.
Bien KLTN_THU=N: moi lan train chi N episode, dung de thu ca day chuyen.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PY = sys.executable

# Hat giong cho mo hinh chinh, cau hoi 3 (phan thuong) va cau hoi 4 (du bao). Seed dau = seed chinh,
# cac ablation khac chi chay seed nay.
SEEDS = (42, 1, 2)
MAIN_SEEDS = SEEDS

# So episode mo hinh chinh = ppo.total_episodes trong config.yaml (mot nguon duy nhat)
MAIN_EPISODES = int(yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
                    ["ppo"]["total_episodes"])

# ten lan chay -> (config, seed, so episode, mo ta)
MAIN = {f"main_s{s}": ("config.yaml", s, MAIN_EPISODES, f"Mo hinh chinh, seed {s}")
        for s in MAIN_SEEDS}
ABLATIONS = {
    "abl_ref":       ("config.yaml", 42, 1000, "Moc so sanh: cau hinh chinh, 1.000 episode"),
    "abl_global":    ("configs/ablation_global_reward.yaml", 42, 1000, "Cau 3: phan thuong toan mang luoi"),
    "abl_q3":        ("configs/ablation_q3_no_reward_norm.yaml", 42, 1000, "Cau 3: tat chuan hoa phan thuong theo cap"),
    "abl_reward_cu": ("configs/ablation_reward_demand_scaled.yaml", 42, 1000, "Phat muc phuc vu kieu cu"),
    "abl_trunk":     ("configs/ablation_shared_trunk.yaml", 42, 1000, "Actor/Critic chung than mang"),
    "abl_nocal":     ("configs/ablation_drop_calendar.yaml", 42, 1000, "Bo dac trung lich"),
    "abl_nowh":      ("configs/ablation_drop_warehouse.yaml", 42, 1000, "Bo tin hieu cap kho"),
    "abl_event":     ("configs/ablation_event_lookahead.yaml", 42, 1000, "Them dac trung su kien sap toi"),
    "abl_warm":      ("configs/ablation_warm_start.yaml", 42, 1000, "Khoi tao lich su cau bang du lieu that"),
    "holdout":       ("configs/holdout_train.yaml", 42, 1000, "Mat hang chua gap: train 24 SKU"),
}
# Cau 3 lap lai tren moi hat giong: moc, phan thuong toan mang luoi, khong chuan hoa
RQ3_CFG = {"abl_ref": "config.yaml", "abl_global": "configs/ablation_global_reward.yaml",
           "abl_q3": "configs/ablation_q3_no_reward_norm.yaml"}
for _base, _cfg in RQ3_CFG.items():
    for _s in SEEDS[1:]:
        ABLATIONS[f"{_base}_s{_s}"] = (_cfg, _s, 1000, f"Cau 3 {_base}, seed {_s}")

# Lan chay mo rong (khong sua mo hinh chinh):
#  - penalty*_trial: 5 lan chay thu he so phat muc phuc vu phi_dv (1.000 episode) -> duong danh doi chi phi-fill
#  - forecast_safety_s*: Cau 4, tac tu chon he so an toan tren nen du bao LightGBM
PHI = ("05", "10", "20", "60", "100")
LAN_CHAY_MO_RONG = {f"penalty{p}_trial": (f"configs/service_penalty_{p}.yaml", 42, 1000,
                                          f"Chay thu he so phat phi_dv {p}") for p in PHI}
for _s in MAIN_SEEDS:
    LAN_CHAY_MO_RONG[f"forecast_safety_s{_s}"] = ("configs/forecast_safety_factor.yaml", _s, MAIN_EPISODES,
                                                  f"He so an toan tren nen du bao, seed {_s}")
NHOM_LAN_CHAY_MO_RONG = {"penalty_trial": [t for t in LAN_CHAY_MO_RONG if t.startswith("penalty")],
                         "forecast_safety": [t for t in LAN_CHAY_MO_RONG if t.startswith("forecast_safety_")]}
# Cau 5: dieu kien van hanh KHONG train lai, chi danh gia zero-shot mo hinh chinh (main_s42)
DIEU_KIEN_ZERO_SHOT = ("shared_supply_tight", "shared_supply_mild")
RUNS = {**MAIN, **ABLATIONS, **LAN_CHAY_MO_RONG}
MO_HINH_DU_BAO = RESULTS / "forecast_model.txt"
THU = int(os.environ.get("KLTN_THU", 0))
if THU:
    RUNS = {t: (c, s, THU, d) for t, (c, s, _, d) in RUNS.items()}


def sh(cmd, check=True):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=str(ROOT))
    if check and r.returncode != 0:
        sys.exit(r.returncode)
    return r.returncode


# ---------------------------------------------------------------------------
# Huan luyen
# ---------------------------------------------------------------------------
def progress(tag):
    st = ROOT / "checkpoints" / f"train_state_{tag}.json"
    return json.loads(st.read_text(encoding="utf-8")).get("episode_count", 0) if st.exists() else 0


def latest_checkpoint(tag):
    """Checkpoint moi nhat de resume: checkpoint_latest (luu moi ~25 episode) hoac
    checkpoint_epNNN, lay file ghi sau cung."""
    cands = [p for p in (ROOT / "checkpoints").glob(f"checkpoint_*_{tag}.pth")
             if re.fullmatch(rf"checkpoint_(ep\d+|latest)_{re.escape(tag)}\.pth", p.name)]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def done(tag):
    return (ROOT / "checkpoints" / f"final_model_{tag}.pth").exists() and progress(tag) >= RUNS[tag][2]


def ghi_phien_ban(tag):
    """Commit code cua tung lan train; hau to -dirty = co thay doi chua commit."""
    r = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=str(ROOT),
                       capture_output=True, text=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    with open(ROOT / "logs" / "phien_ban_code.txt", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {tag}  {r.stdout.strip() or '?'}\n")


def train(tag):
    cfg, seed, eps, desc = RUNS[tag]
    if done(tag):
        print(f"[{tag}] da xong ({eps} episode) - bo qua.")
        return
    if tag.startswith("forecast_safety_") and not MO_HINH_DU_BAO.exists():
        sh([PY, "scripts/forecast_demand.py"])
    cmd = [PY, "scripts/train.py", "--config", cfg, "--seed", seed, "--episodes", eps, "--tag", tag]
    ck = latest_checkpoint(tag)
    if ck:
        cmd += ["--resume", ck.relative_to(ROOT).as_posix()]
        print(f"[{tag}] resume tu {ck.name} (da co {progress(tag)} episode)")
    print(f"[{tag}] {desc}")
    ghi_phien_ban(tag)
    sh(cmd)


def train_full():
    """Moi lan chay, viec quyet dinh ket luan truoc: mo hinh chinh, du bao, phi_dv, roi ablation."""
    for tag in [*MAIN, *LAN_CHAY_MO_RONG, *ABLATIONS]:
        train(tag)


def status():
    print(f"{'Lan chay':<22}{'Tien do':>14}  Mo ta")
    for tag, (_, _, eps, desc) in RUNS.items():
        print(f"{tag:<22}{'XONG' if done(tag) else f'{progress(tag)}/{eps}':>14}  {desc}")


# ---------------------------------------------------------------------------
# Danh gia
# ---------------------------------------------------------------------------
def duong_checkpoint(tag):
    p = ROOT / "checkpoints" / f"best_model_{tag}.pth"
    return p.relative_to(ROOT).as_posix() if p.exists() else None


def can_chay(ket_qua, *dau_vao):
    """True neu file ket qua chua co hoac cu hon mot file dau vao dang co (buoc can chay lai)."""
    ket_qua = Path(ket_qua)
    if not ket_qua.exists():
        return True
    moc = ket_qua.stat().st_mtime
    return any(Path(p).exists() and Path(p).stat().st_mtime > moc for p in dau_vao if p)


def buoc(ket_qua, dau_vao, cmd):
    ket_qua = ROOT / ket_qua          # chap nhan ca duong dan tuong doi ("results/...") lan tuyet doi
    if can_chay(ket_qua, *[ROOT / p for p in dau_vao if p]):
        sh(cmd)
    else:
        print(f"[bo qua] {ket_qua.relative_to(ROOT).as_posix()} da moi hon dau vao")


def chep_du_bao_sang(thu_muc):
    """Dieu kien van hanh dung chung du bao cua mien du lieu goc."""
    (ROOT / thu_muc).mkdir(parents=True, exist_ok=True)
    for m in ("train_oof", "val", "test", "sigma"):
        shutil.copy(RESULTS / f"forecast_{m}.npy", ROOT / thu_muc)


def evaluate_all(n_eval=30):
    R = "results/"
    THAM_SO_BASELINE = (R + "baseline_params.json", R + "baseline_params_group.json")
    it = ["--episodes", 1] if THU else []
    tune_eps = 1 if THU else 3
    miss = [t for t in RUNS if not duong_checkpoint(t)]
    if miss:
        print(f"CANH BAO: chua co checkpoint cho {miss} - cac buoc lien quan se bo qua.")

    # 0. Du bao LightGBM dung chung cho Newsvendor du bao va Cau 4; tham so baseline tinh chinh tren VAL.
    #    Muon tinh chinh lai: xoa results/baseline_params*.json
    if not MO_HINH_DU_BAO.exists():
        sh([PY, "scripts/forecast_demand.py"])
    if not all((ROOT / p).exists() for p in THAM_SO_BASELINE):
        sh([PY, "scripts/tune_baselines.py", "--episodes", tune_eps, "--per_group"])

    def danh_gia(tag, cfg="config.yaml", out_tag=None, thu_muc=R):
        ck = duong_checkpoint(tag)
        if ck:
            out_tag = out_tag or tag
            buoc(ROOT / thu_muc / f"summary_{out_tag}.json", [ck, *THAM_SO_BASELINE],
                 [PY, "scripts/evaluate.py", "--config", cfg, "--checkpoint", ck,
                  "--episodes", n_eval, "--tag", out_tag])

    # 1. Cau 1: mo hinh chinh moi seed; Cau 4: he so an toan tren nen du bao moi seed
    for s in MAIN_SEEDS:
        danh_gia(f"main_s{s}")
        buoc(R + f"learning_curve_main_s{s}.png", [R + f"train_log_main_s{s}.csv"],
             [PY, "scripts/plot_learning_curve.py", "--tag", f"main_s{s}"])
        danh_gia(f"forecast_safety_s{s}", "configs/forecast_safety_factor.yaml")

    # 2. Cau 5: zero-shot mo hinh chinh duoi nguon cung chung (khong train lai, so sanh truc tiep)
    for ten in DIEU_KIEN_ZERO_SHOT if duong_checkpoint("main_s42") else ():
        cfg = f"configs/{ten}.yaml"
        thu_muc = yaml.safe_load(open(ROOT / cfg, encoding="utf-8"))["paths"]["results_dir"] + "/"
        if not (ROOT / thu_muc / "forecast_test.npy").exists():
            chep_du_bao_sang(thu_muc)
        if not (ROOT / thu_muc / "baseline_params_group.json").exists():
            sh([PY, "scripts/tune_baselines.py", "--config", cfg, "--episodes", tune_eps, "--per_group"])
        danh_gia("main_s42", cfg, f"zeroshot_{ten}", thu_muc)

    # 3. Cung muc phuc vu (luoi min). Duong bien baseline quet MOT lan, cache o baseline_frontier.json
    #    (lau nhat: vai gio), cac seed va Cau 4 dung chung.
    for tag, cfg in [*[(f"main_s{s}", "config.yaml") for s in MAIN_SEEDS],
                     *[(f"forecast_safety_s{s}", "configs/forecast_safety_factor.yaml") for s in MAIN_SEEDS]]:
        ck = duong_checkpoint(tag)
        if ck:
            buoc(R + f"same_service_{tag}.json", [ck],
                 [PY, "scripts/compare_same_service.py", "--config", cfg, "--checkpoint", ck,
                  "--tune_episodes", 1 if THU else 2, "--eval_episodes", n_eval, "--tag", tag])

    # 4. Phan tich phu mo hinh chinh. "Cung muc PV" trong cac phan tich nay lay tu same_service_main_s42.json
    main = duong_checkpoint("main_s42")
    if main:
        cung_muc = R + "same_service_main_s42.json"
        for script, ket_qua, them in [("regime_analysis.py", "regime_analysis_main.json", it),
                                      ("sensitivity.py", "sensitivity_main.json", it),
                                      ("robustness.py", "robustness_main.json", it),
                                      ("analyze_scale_groups.py", "scale_group_v2_main.json", []),
                                      ("policy_behavior.py", "policy_behavior_main.json", it)]:
            buoc(R + ket_qua, [main, cung_muc, *THAM_SO_BASELINE],
                 [PY, f"scripts/{script}", "--checkpoint", main, "--tag", "main", *them])
    if duong_checkpoint("forecast_safety_s42"):
        buoc(R + "regime_analysis_forecast_safety.json", [duong_checkpoint("forecast_safety_s42"), *THAM_SO_BASELINE],
             [PY, "scripts/regime_analysis.py", "--config", "configs/forecast_safety_factor.yaml",
              "--checkpoint", duong_checkpoint("forecast_safety_s42"), "--tag", "forecast_safety", *it])

    # 5. Ablation, Cau 3, phi_dv: danh gia tren mien test voi DUNG config da train
    for tag, (cfg, *_rest) in {**ABLATIONS, **{t: LAN_CHAY_MO_RONG[t] for t in NHOM_LAN_CHAY_MO_RONG["penalty_trial"]}}.items():
        if tag != "holdout":
            danh_gia(tag, cfg)
    for tag in ("abl_ref", "abl_reward_cu"):          # phat muc phuc vu moi vs cu: fill tung cap
        if duong_checkpoint(tag):
            buoc(R + f"policy_behavior_{tag}.json", [duong_checkpoint(tag)],
                 [PY, "scripts/policy_behavior.py", "--config", RUNS[tag][0],
                  "--checkpoint", duong_checkpoint(tag), "--tag", tag])
    cmp = [t for t in ABLATIONS if t != "holdout" and not re.search(r"_s\d+$", t) and duong_checkpoint(t)]
    if "abl_ref" in cmp and len(cmp) > 1:
        sh([PY, "scripts/plot_learning_curve.py", "--compare", *cmp, "--max_episode", 1000])

    # 6. Cau 2: mat hang chua gap (so voi mo hinh cung 1.000 episode va mo hinh chinh da thay ca 30 SKU)
    for tag, out in [("holdout", "holdout_unseen"), ("abl_ref", "holdout_seen_ref"),
                     ("main_s42", "holdout_seen_main")]:
        danh_gia(tag, "configs/holdout_eval.yaml", out)

    # 7. Moc biet truoc nhu cau; duong danh doi chi phi - muc phuc vu
    buoc(R + "perfect_demand.json", [R + "forecast_test.npy"],
         [PY, "scripts/perfect_demand_benchmark.py", "--episodes", n_eval])
    if (RESULTS / "baseline_frontier.json").exists():
        diem = [t for t in ["main_s42", "forecast_safety_s42", "abl_ref", *NHOM_LAN_CHAY_MO_RONG["penalty_trial"]]
                if (RESULTS / f"summary_{t}.json").exists()]
        sh([PY, "scripts/cost_service_frontier.py", "--tags", *diem])

    tong_hop()
    print("\nXONG. So lieu tong: results/KET_QUA.txt")


def tong_hop():
    tong_hop_nhieu_seed("main", "multiseed_main.json")
    tong_hop_nhieu_seed("forecast_safety", "multiseed_forecast_safety.json")
    tong_hop_rq3()
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_results import ghi_ket_qua
    print("Da luu", ghi_ket_qua(ROOT))


def tong_hop_rq3():
    """Cau 3 tren moi hat giong: moi bien the (toan mang luoi, khong chuan hoa) so voi moc abl_ref
    CUNG hat giong. Hieu nang: kiem dinh theo cap tren 30 lan danh gia test. On dinh huan luyen:
    tu train_log (episode dau tien dat fill >= 85% tren val; chi phi, entropy, explained variance
    100 episode cuoi)."""
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(ROOT / "scripts"))
    from evaluate import paired_test

    def tag(base, s):
        return base if s == 42 else f"{base}_s{s}"

    def ippo_costs(t):
        p = RESULTS / f"all_episodes_{t}.csv"
        if not p.exists():
            return None
        d = pd.read_csv(p)
        return d[d.policy == "IPPO"].sort_values("episode")

    def on_dinh(t):
        p = RESULTS / f"train_log_{t}.csv"
        if not p.exists():
            return {}
        d = pd.read_csv(p)
        dat = d.dropna(subset=["eval_fill"])
        dat = dat[dat.eval_fill >= 0.85]
        cuoi = d.tail(100)
        return {"episode_dat_85_val": int(dat.episode.iloc[0]) if len(dat) else None,
                "cost_100_ep_cuoi": float(cuoi.cost_total.mean()),
                "entropy_cuoi": float(cuoi.entropy.dropna().mean()),
                "explained_var_cuoi": float(cuoi.explained_variance.dropna().mean())}

    out = {}
    for base in RQ3_CFG:
        out[base] = {"seeds": {}}
        for s in SEEDS:
            t = tag(base, s)
            e = ippo_costs(t)
            if e is None:
                continue
            row = {"cost": float(e.total_cost.mean()), "fill": float(e.fill_rate.mean()), **on_dinh(t)}
            ref = ippo_costs(tag("abl_ref", s))
            if base != "abl_ref" and ref is not None:
                row["so_voi_ref"] = paired_test(e.total_cost.values, ref.total_cost.values)
            out[base]["seeds"][s] = row
        g = [r["so_voi_ref"]["gap_percent"] for r in out[base]["seeds"].values() if "so_voi_ref" in r]
        if g:
            out[base]["gap_vs_ref_mean"] = float(np.mean(g))
            out[base]["n_seeds_ref_re_hon_co_y_nghia"] = int(sum(
                r["so_voi_ref"]["gap_percent"] > 0 and r["so_voi_ref"]["p_paired"] < 0.05
                for r in out[base]["seeds"].values() if "so_voi_ref" in r))
            out[base]["n_seeds"] = len(g)
    (RESULTS / "rq3_multiseed.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("Da luu results/rq3_multiseed.json")


def tong_hop_nhieu_seed(goc, ten_file):
    """Tong hop cac hat giong DUNG CACH: kiem dinh theo cap tinh RIENG tung hat giong
    (evaluate.py), roi bao cao trung binh +- do lech chuan GIUA cac hat giong - khong gop moi
    lan danh gia cua cac seed nhu quan sat doc lap. goc: 'main' hoac 'forecast_safety'."""
    import numpy as np
    rows = {}
    for s in MAIN_SEEDS:
        p = RESULTS / f"summary_{goc}_s{s}.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            ippo = next(r for r in d["summary"] if r["policy"] == "IPPO")
            rows[s] = {"cost": ippo["total_cost_mean"], "fill": ippo["fill_rate_mean"],
                       "per_baseline": d["stat"].get("per_baseline", {})}
    if not rows:
        return
    out = {"seeds": rows,
           "ippo_cost_mean": float(np.mean([r["cost"] for r in rows.values()])),
           "ippo_cost_sd": float(np.std([r["cost"] for r in rows.values()], ddof=1)) if len(rows) > 1 else 0.0,
           "ippo_fill_mean": float(np.mean([r["fill"] for r in rows.values()])),
           "gap_vs_baseline": {}}
    for ref in sorted(set().union(*[r["per_baseline"] for r in rows.values()])):
        g = [r["per_baseline"][ref]["gap_percent"] for r in rows.values() if ref in r["per_baseline"]]
        sig = [r["per_baseline"][ref]["p_paired"] < 0.05 for r in rows.values() if ref in r["per_baseline"]]
        out["gap_vs_baseline"][ref] = {"mean": float(np.mean(g)), "min": float(min(g)),
                                       "max": float(max(g)), "n_seeds_significant": int(sum(sig)),
                                       "n_seeds_re_hon": int(sum(s and x < 0 for s, x in zip(sig, g))),
                                       "n_seeds_dat_hon": int(sum(s and x > 0 for s, x in zip(sig, g))),
                                       "n_seeds": len(g)}
    (RESULTS / ten_file).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Da luu results/{ten_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lenh", choices=["status", "train", "eval", "tonghop"])
    ap.add_argument("ten", nargs="?", help="ten lan chay, 'main', 'ablations', 'full' hoac nhom mo rong")
    ap.add_argument("--n_eval", type=int, default=30)
    a = ap.parse_args()
    if a.lenh == "status":
        status()
    elif a.lenh == "eval":
        evaluate_all(a.n_eval)
    elif a.lenh == "tonghop":
        tong_hop()
    elif a.ten == "full":
        train_full()
    else:
        nhom = {"main": list(MAIN), "ablations": list(ABLATIONS), **NHOM_LAN_CHAY_MO_RONG}
        tags = nhom.get(a.ten, [a.ten])
        bad = [t for t in tags if t not in RUNS]
        if bad:
            sys.exit(f"Khong co lan chay {bad}. Chon: {list(RUNS)} hoac main/ablations/full/"
                     f"{'/'.join(NHOM_LAN_CHAY_MO_RONG)}")
        for t in tags:
            train(t)


if __name__ == "__main__":
    main()
