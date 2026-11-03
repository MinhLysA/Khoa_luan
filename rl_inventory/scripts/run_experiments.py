"""
scripts/run_experiments.py
===================
MOT noi dinh nghia TOAN BO dot thuc nghiem cuoi cung: moi lan huan luyen (mo
hinh chinh + ablation) va pipeline danh gia. Moi lenh train TU RESUME tu
checkpoint gan nhat neu bi ngat giua chung.

    python scripts/run_experiments.py status              # tien do tung lan chay
    python scripts/run_experiments.py train main_s42      # train 1 lan chay (tu resume)
    python scripts/run_experiments.py train ablations     # lan luot moi ablation
    python scripts/run_experiments.py train auto          # tu lay lan chay tiep theo (mo nhieu terminal)
    python scripts/run_experiments.py eval                # toan bo pipeline danh gia
    python scripts/run_experiments.py train full          # toan bo dot thuc nghiem, viec quan trong nhat truoc
    python scripts/run_experiments.py train forecast_safety  # mot nhom mo rong, xem NHOM_MO_RONG
    python scripts/run_experiments.py eval_mo_rong        # chi phan bo sung, khong chay lai ablation

Nhom mo rong khong nam trong "main/ablations/all/auto". "full" chay ca chung (phuong an
rut gon: 3 seed, bo U3/U4/C'/E2/E3, dieu kien van hanh chi danh gia zero-shot khong train).
"eval" tu danh gia moi lan chay da co checkpoint.
Bien KLTN_THU=N: moi lan train chi N episode, dung de thu ca day chuyen.
"""
import argparse
import json
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

# Hat giong cho mo hinh chinh va RQ3 (seed dau = seed chinh, cac ablation khac
# chi chay seed nay). Them/bot seed chi can sua dong nay.
SEEDS = (42, 1, 2)
# [Nhieu seed] Phuong an rut gon: 3 seed. Muon them seed cho mo hinh chinh: dien vao day (vd (3, 4))
EXTRA_MAIN_SEEDS = ()
MAIN_SEEDS = SEEDS + EXTRA_MAIN_SEEDS

# So episode mo hinh chinh = ppo.total_episodes trong config.yaml (mot nguon duy nhat)
MAIN_EPISODES = int(yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
                    ["ppo"]["total_episodes"])

# ten lan chay -> (config, seed, so episode, mo ta)
MAIN = {f"main_s{s}": ("config.yaml", s, MAIN_EPISODES, f"Mo hinh chinh, seed {s}")
        for s in MAIN_SEEDS}
ABLATIONS = {
    "abl_ref":       ("config.yaml", 42, 1000, "Tham chieu 1000 episode (cung lich lr/entropy voi ablation)"),
    "abl_global":    ("configs/ablation_global_reward.yaml", 42, 1000, "RQ3: phan thuong toan cuc (team reward)"),
    "abl_q3":        ("configs/ablation_q3_no_reward_norm.yaml", 42, 1000, "RQ3: tat chuan hoa reward theo cap"),
    "abl_reward_cu": ("configs/ablation_reward_demand_scaled.yaml", 42, 1000, "Phat SLA kieu cu"),
    "abl_trunk":     ("configs/ablation_shared_trunk.yaml", 42, 1000, "Actor/Critic chung than"),
    "abl_nocal":     ("configs/ablation_drop_calendar.yaml", 42, 1000, "Bo dac trung lich"),
    "abl_nowh":      ("configs/ablation_drop_warehouse.yaml", 42, 1000, "Bo tin hieu cap kho"),
    "abl_event":     ("configs/ablation_event_lookahead.yaml", 42, 1000, "Them dac trung su kien sap toi"),
    "abl_warm":      ("configs/ablation_warm_start.yaml", 42, 1000, "Warm-start lich su cau"),
    "holdout":       ("configs/holdout_train.yaml", 42, 1000, "Hold-out: train 24 SKU"),
}
# [P4-2] RQ3 lap lai tren moi hat giong trong SEEDS (42 o tren + cac seed con
# lai): moc, toan cuc, khong chuan hoa
RQ3_CFG = {"abl_ref": "config.yaml", "abl_global": "configs/ablation_global_reward.yaml",
           "abl_q3": "configs/ablation_q3_no_reward_norm.yaml"}
for _base, _cfg in RQ3_CFG.items():
    for _s in SEEDS[1:]:
        ABLATIONS[f"{_base}_s{_s}"] = (_cfg, _s, 1000, f"RQ3 {_base}, seed {_s}")

# Cac lan chay mo rong theo phuong an trien khai (khong sua mo hinh chinh).
PHI = ("05", "10", "20", "60", "100")
MO_RONG = {}
for _p in PHI:
    _cfg = f"configs/service_penalty_{_p}.yaml"
    MO_RONG[f"penalty{_p}_trial"] = (_cfg, 42, 1000, f"Chay thu he so phat phi_dv {_p}")
    # Phuong an rut gon: chi giu 5 lan chay thu 1.000 episode, bo buoc train lai hai gia tri x 3 seed
for _s in SEEDS:
    MO_RONG[f"forecast_obs_s{_s}"] = ("configs/forecast_observation.yaml", _s, MAIN_EPISODES, f"IPPO co du bao, seed {_s}")
    MO_RONG[f"safety_factor_s{_s}"] = ("configs/safety_factor.yaml", _s, MAIN_EPISODES, f"He so an toan, seed {_s}")
for _s in MAIN_SEEDS:
    MO_RONG[f"forecast_safety_s{_s}"] = ("configs/forecast_safety_factor.yaml", _s, MAIN_EPISODES, f"He so an toan tren nen du bao, seed {_s}")
# Dieu kien van hanh: KHONG train lai, chi danh gia zero-shot mo hinh chinh (main_s42) o thu muc ket qua rieng
DIEU_KIEN = ("shared_supply_tight", "shared_supply_mild", "capacity_3", "capacity_8")
NHOM_MO_RONG = {"penalty_trial": [t for t in MO_RONG if t.endswith("_trial") and t.startswith("penalty")],
                "forecast_obs": [t for t in MO_RONG if t.startswith("forecast_obs_")],
                "safety_factor": [t for t in MO_RONG if t.startswith("safety_factor_")],
                "forecast_safety": [t for t in MO_RONG if t.startswith("forecast_safety_")]}
RUNS = {**MAIN, **ABLATIONS, **MO_RONG}
MO_HINH_DU_BAO = ROOT / "results" / "forecast_model.txt"
THU = int(os.environ.get("KLTN_THU", 0))
if THU:
    RUNS = {t: (c, s, THU, d) for t, (c, s, _, d) in RUNS.items()}


def sh(cmd, check=True):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=str(ROOT))
    if check and r.returncode != 0:
        sys.exit(r.returncode)
    return r.returncode


def progress(tag):
    st = ROOT / "checkpoints" / f"train_state_{tag}.json"
    return json.loads(st.read_text(encoding="utf-8")).get("episode_count", 0) if st.exists() else 0


def latest_checkpoint(tag):
    """Checkpoint moi nhat de resume: file checkpoint_latest (luu moi ~25
    episode) neu co, nguoc lai checkpoint_epNNN lon nhat; chon file ghi sau cung."""
    cands = []
    for p in (ROOT / "checkpoints").glob(f"checkpoint_*_{tag}.pth"):
        if re.fullmatch(rf"checkpoint_(ep\d+|latest)_{re.escape(tag)}\.pth", p.name):
            cands.append(p)
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def done(tag):
    cfg, _, eps, _ = RUNS[tag]
    return (ROOT / "checkpoints" / f"final_model_{tag}.pth").exists() and progress(tag) >= eps


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
    cmd = [PY, "scripts/train.py", "--config", cfg, "--seed", seed,
           "--episodes", eps, "--tag", tag]
    if tag.startswith(("forecast_obs_", "forecast_safety_")) and not MO_HINH_DU_BAO.exists():
        sh([PY, "scripts/forecast_demand.py"])
    ck = latest_checkpoint(tag)
    if ck:
        cmd += ["--resume", ck.relative_to(ROOT).as_posix()]
        print(f"[{tag}] resume tu {ck.name} (da co {progress(tag)} episode)")
    print(f"[{tag}] {desc}")
    ghi_phien_ban(tag)
    sh(cmd)


# ---------------------------------------------------------------------------
# Che do "auto": mo nhieu terminal tren cung may. Moi phien tu lay lan chay chua
# xong va chua co phien nao dang chay, theo thu tu uu tien.
# ---------------------------------------------------------------------------
PRIORITY = (["main_s42", "abl_ref", "abl_global", "abl_q3"]
            + [f"main_s{s}" for s in MAIN_SEEDS[1:]]
            + [f"{b}_s{s}" for s in SEEDS[1:] for b in RQ3_CFG]
            + ["abl_reward_cu", "holdout", "abl_trunk", "abl_nocal", "abl_nowh",
               "abl_event", "abl_warm"])
assert sorted(PRIORITY) == sorted({**MAIN, **ABLATIONS}), "PRIORITY phai chua dung MAIN + ABLATIONS"
STALE_MIN = 20          # khoa khong co dau hieu song qua 20 phut = phien da chet
ME = f"{socket.gethostname()}-{os.getpid()}"


def _lock(tag):
    return ROOT / "checkpoints" / f"lock_{tag}.json"


def lock_owner(tag):
    """Chu so huu khoa neu khoa con 'song', nguoc lai None. Dau hieu song: file
    khoa hoac train_state (ghi moi 25 episode) duoc cap nhat trong STALE_MIN phut."""
    lk = _lock(tag)
    if not lk.exists():
        return None
    st = ROOT / "checkpoints" / f"train_state_{tag}.json"
    last = max(lk.stat().st_mtime, st.stat().st_mtime if st.exists() else 0)
    if time.time() - last > STALE_MIN * 60:
        return None
    try:
        return json.loads(lk.read_text(encoding="utf-8")).get("owner")
    except (json.JSONDecodeError, OSError):
        return "?"


def acquire(tag):
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    _lock(tag).write_text(json.dumps({"owner": ME, "since": time.ctime()}), encoding="utf-8")
    time.sleep(0 if THU else 2)        # roi kiem tra lai co bi phien khac chen khong
    return lock_owner(tag) == ME


def release(tag):
    try:
        if lock_owner(tag) in (ME, None):
            _lock(tag).unlink()
    except OSError:
        pass


def train_auto():
    time.sleep(random.uniform(0, 3))        # tranh 2 phien khoi dong cung luc lay trung
    while True:
        con_lai = [t for t in PRIORITY if not done(t)]
        if not con_lai:
            print("\nTAT CA LAN CHAY DA XONG. Chay: python scripts/run_experiments.py eval")
            return
        rank = [t for t in con_lai if lock_owner(t) in (None, ME)]
        if not rank:
            print(f"Cac lan chay con lai deu dang duoc phien khac chay: {con_lai}. Dung.")
            return
        tag = rank[0]
        if not acquire(tag):
            continue
        print(f"\n===== [{ME}] nhan lan chay: {tag} =====", flush=True)
        try:
            train(tag)
        finally:
            release(tag)


def fill_val_cuoi(tag):
    """Fill trung binh 5 lan danh gia val cuoi cua mot lan chay; None neu chua co."""
    import pandas as pd
    p = ROOT / "results" / f"train_log_{tag}.csv"
    if not p.exists():
        return None
    e = pd.read_csv(p).eval_fill.dropna()
    e = e[e != e.shift()]                         # moi dong lap lai ket qua danh gia gan nhat
    return float(e.tail(5).mean()) if len(e) else None


def train_co_khoa(tag):
    """Lan chay dang co terminal khac giu thi bo qua, nen mo duoc nhieu terminal."""
    if done(tag) or lock_owner(tag) not in (None, ME) or not acquire(tag):
        return
    try:
        train(tag)
    finally:
        release(tag)


def cho_xong(tags):
    """Doi cac terminal khac chay xong. Lan chay cua terminal da chet thi khoa het
    hieu luc sau STALE_MIN phut va duoc nhan lai tai day."""
    while not all(done(t) for t in tags):
        for t in tags:
            train_co_khoa(t)
        if not all(done(t) for t in tags):
            time.sleep(60)


def train_full():
    """Toan bo dot thuc nghiem (phuong an rut gon). Viec quyet dinh ket luan dung truoc de neu
    phai dung giua chung van co so lieu quan trong nhat. Bi ngat thi chay lai, tu noi tiep."""
    if not MO_HINH_DU_BAO.exists():
        sh([PY, "scripts/forecast_demand.py"])
    # U1 (forecast_obs) va U2 (safety_factor) tam bo khoi dot train; muon train lai thi them
    # *NHOM_MO_RONG["forecast_obs"], *NHOM_MO_RONG["safety_factor"] vao day (danh gia tu bo qua nhom chua co checkpoint)
    dot1 = [*MAIN, *NHOM_MO_RONG["forecast_safety"], *NHOM_MO_RONG["penalty_trial"]]
    for t in dot1:
        train_co_khoa(t)
    dot2 = [x for x in PRIORITY if x not in MAIN]           # ablation, RQ3, hold-out
    for t in dot2:
        train_co_khoa(t)
    cho_xong(dot1 + dot2)                         # de buoc danh gia thay du moi lan chay


def status():
    print(f"{'Lan chay':<15}{'Tien do':>14}  {'Dang chay boi':<22} Mo ta")
    for tag, (cfg, seed, eps, desc) in RUNS.items():
        if tag in MO_RONG and not progress(tag):
            continue
        mark = "XONG" if done(tag) else f"{progress(tag)}/{eps}"
        print(f"{tag:<15}{mark:>14}  {(lock_owner(tag) or '-'):<22} {desc}")


def ck(tag):
    p = ROOT / "checkpoints" / f"best_model_{tag}.pth"
    return p.relative_to(ROOT).as_posix() if p.exists() else None


def chep_du_bao(thu_muc):
    """Dieu kien van hanh dung chung du bao cua mien du lieu goc."""
    (ROOT / thu_muc).mkdir(parents=True, exist_ok=True)
    for m in ("train_oof", "val", "test", "sigma"):
        shutil.copy(ROOT / "results" / f"forecast_{m}.npy", ROOT / thu_muc)


def evaluate_all(n_eval=30):
    miss = [t for t in [*MAIN, *ABLATIONS] if not ck(t)]
    if miss:
        print(f"CANH BAO: chua co checkpoint cho {miss} - cac buoc lien quan se bo qua.")
    main = ck("main_s42")

    # 0. [Du bao] Du bao dung chung cho baseline du bao va U1 (chi chay mot lan)
    if not MO_HINH_DU_BAO.exists():
        sh([PY, "scripts/forecast_demand.py"])
    # 1. Baseline tinh chinh tren mien VAL
    sh([PY, "scripts/tune_baselines.py", "--episodes", 1 if THU else 3, "--per_group"])
    # 2. Danh gia moi seed mo hinh chinh (+ cua so co dinh cho seed 42)
    for s in MAIN_SEEDS:
        if ck(f"main_s{s}"):
            sh([PY, "scripts/evaluate.py", "--checkpoint", ck(f"main_s{s}"),
                "--episodes", n_eval, "--tag", f"main_s{s}"])
            sh([PY, "scripts/plot_learning_curve.py", "--tag", f"main_s{s}"])
    if main:
        # 3. Cung muc phuc vu -> 4. giai doan -> 5. hanh vi -> 6. quy mo -> 7. do nhay
        it = ["--episodes", 1] if THU else []
        # Luoi min truoc: cac phan tich sau dung baseline cung muc phuc vu tu ket qua nay
        sh([PY, "scripts/compare_same_service.py", "--grid", "fine", "--checkpoint", main,
            "--tune_episodes", 1 if THU else 2, "--eval_episodes", n_eval, "--tag", "main_s42"])
        sh([PY, "scripts/compare_same_service.py", "--checkpoint", main,
            "--tune_episodes", 1 if THU else 3, "--eval_episodes", n_eval])
        sh([PY, "scripts/regime_analysis.py", "--checkpoint", main, "--tag", "main", *it])
        sh([PY, "scripts/policy_behavior.py", "--checkpoint", main, "--tag", "main", *it])
        sh([PY, "scripts/analyze_scale_groups.py", "--checkpoint", main, "--tag", "main"])
        sh([PY, "scripts/sensitivity.py", "--checkpoint", main, "--tag", "main", *it])
        sh([PY, "scripts/robustness.py", "--checkpoint", main, "--tag", "main", *it])
    # 8. Ablation: danh gia tren mien test voi DUNG config da train
    for tag, (cfg, *_rest) in ABLATIONS.items():
        if tag == "holdout" or not ck(tag):
            continue
        sh([PY, "scripts/evaluate.py", "--config", cfg, "--checkpoint", ck(tag),
            "--episodes", n_eval, "--tag", tag])
    for tag in ("abl_ref", "abl_reward_cu"):          # phat SLA moi vs cu: fill tung cap
        if ck(tag):
            sh([PY, "scripts/policy_behavior.py", "--config", RUNS[tag][0],
                "--checkpoint", ck(tag), "--tag", tag])
    cmp = [t for t in ABLATIONS if t != "holdout" and not re.search(r"_s\d+$", t) and ck(t)]
    if "abl_ref" in cmp and len(cmp) > 1:
        sh([PY, "scripts/plot_learning_curve.py", "--compare", *cmp, "--max_episode", 1000])
    # 9. Hold-out: 6 SKU chua gap (so voi mo hinh cung 1000 episode da thay ca 30 SKU)
    for tag, out in [("holdout", "holdout_unseen"), ("abl_ref", "holdout_seen_ref"),
                     ("main_s42", "holdout_seen_main")]:
        if ck(tag):
            sh([PY, "scripts/evaluate.py", "--config", "configs/holdout_eval.yaml",
                "--checkpoint", ck(tag), "--episodes", n_eval, "--tag", out])
    danh_gia_mo_rong(n_eval)
    tong_hop_da_hat_giong()
    tong_hop_rq3()
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_results import ghi_ket_qua
    print("Da luu", ghi_ket_qua(ROOT))
    print("\nXONG. So lieu tong: results/KET_QUA.txt")


def eval_mo_rong(n_eval=30):
    """Ket qua mo hinh chinh va ablation da co: chi tinh lai baseline moi va phan
    bo sung, khong chay lai cac phan tich cu."""
    if not MO_HINH_DU_BAO.exists():
        sh([PY, "scripts/forecast_demand.py"])
    sh([PY, "scripts/tune_baselines.py", "--episodes", 1 if THU else 3, "--per_group"])
    for s in MAIN_SEEDS:
        if ck(f"main_s{s}"):
            sh([PY, "scripts/evaluate.py", "--checkpoint", ck(f"main_s{s}"),
                "--episodes", n_eval, "--tag", f"main_s{s}"])
    danh_gia_mo_rong(n_eval)
    tong_hop_da_hat_giong()
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_results import ghi_ket_qua
    print("Da luu", ghi_ket_qua(ROOT))


def danh_gia_mo_rong(n_eval):
    """[Cung muc phuc vu][Fill theo nhom][Nhieu seed][Cau truc chinh sach] cho mo hinh chinh. Moi lan chay mo rong da co checkpoint:
    danh gia voi dung config da train, so sanh cung muc phuc vu tren duong bien
    dung chung, phan ra fill theo nhom."""
    for s in MAIN_SEEDS:
        if ck(f"main_s{s}"):
            sh([PY, "scripts/compare_same_service.py", "--grid", "fine", "--checkpoint", ck(f"main_s{s}"),
                "--tune_episodes", 1 if THU else 2, "--eval_episodes", n_eval, "--tag", f"main_s{s}"])
            sh([PY, "scripts/service_by_group.py", "--checkpoint", ck(f"main_s{s}"),
                "--episodes", n_eval, "--tag", f"main_s{s}"])
    if ck("main_s42"):
        sh([PY, "scripts/policy_structure.py", "--checkpoint", ck("main_s42"), "--tag", "main"])
        sh([PY, "scripts/multi_seed_stats.py", "--seeds", *MAIN_SEEDS, "--episodes", n_eval,
            "--n_boot", 50 if THU else 2000])
    sh([PY, "scripts/perfect_demand_benchmark.py", "--episodes", n_eval])
    for s in MAIN_SEEDS:
        if ck(f"main_s{s}"):
            sh([PY, "scripts/hybrid_policy.py", "--checkpoint", ck(f"main_s{s}"), "--tag", f"main_s{s}",
                "--episodes", n_eval, "--tune_episodes", 1 if THU else 3])
    if ck("forecast_safety_s42"):
        sh([PY, "scripts/regime_analysis.py", "--config", "configs/forecast_safety_factor.yaml", "--checkpoint",
            ck("forecast_safety_s42"), "--tag", "forecast_safety", *(["--episodes", 1] if THU else [])])
    co = [t for t in MO_RONG if ck(t)]
    da_tinh_chinh = set()
    for tag in co:
        cfg = MO_RONG[tag][0]
        thu_muc = yaml.safe_load(open(ROOT / cfg, encoding="utf-8"))["paths"]["results_dir"]
        if thu_muc != "results" and thu_muc not in da_tinh_chinh:
            da_tinh_chinh.add(thu_muc)
            sh([PY, "scripts/tune_baselines.py", "--config", cfg, "--episodes", 1 if THU else 3, "--per_group"])
        sh([PY, "scripts/evaluate.py", "--config", cfg, "--checkpoint", ck(tag),
            "--episodes", n_eval, "--tag", tag])
        if not tag.endswith("_trial"):
            sh([PY, "scripts/compare_same_service.py", "--grid", "fine", "--config", cfg, "--checkpoint", ck(tag),
                "--tune_episodes", 1 if THU else 2, "--eval_episodes", n_eval, "--tag", tag])
            sh([PY, "scripts/service_by_group.py", "--config", cfg, "--checkpoint", ck(tag),
                "--episodes", n_eval, "--tag", tag])
    if sum(bool(ck(f"forecast_safety_s{s}")) for s in MAIN_SEEDS) >= 2:
        sh([PY, "scripts/multi_seed_stats.py", "--config", "configs/forecast_safety_factor.yaml", "--ten", "forecast_safety",
            "--seeds", *MAIN_SEEDS, "--episodes", n_eval, "--n_boot", 50 if THU else 2000])
    # [Cau 5] Dieu kien van hanh: danh gia ZERO-SHOT mo hinh chinh (main_s42), KHONG train lai
    if ck("main_s42"):
        for ten in DIEU_KIEN:
            cfg = f"configs/{ten}.yaml"
            thu_muc = yaml.safe_load(open(ROOT / cfg, encoding="utf-8"))["paths"]["results_dir"]
            chep_du_bao(thu_muc)
            sh([PY, "scripts/tune_baselines.py", "--config", cfg, "--episodes", 1 if THU else 3, "--per_group"])
            sh([PY, "scripts/evaluate.py", "--config", cfg, "--checkpoint", ck("main_s42"),
                "--episodes", n_eval, "--tag", f"zeroshot_{ten}"])
            sh([PY, "scripts/compare_same_service.py", "--grid", "fine", "--config", cfg, "--checkpoint", ck("main_s42"),
                "--tune_episodes", 1 if THU else 2, "--eval_episodes", n_eval, "--tag", f"zeroshot_{ten}"])
    diem = ["main_s42"] + [t for t in co if t.startswith(("penalty", "forecast_obs_s42", "forecast_safety_s42")) and not t.endswith("_trial")]
    if (ROOT / "results" / "baseline_frontier.json").exists() and ck("main_s42"):
        sh([PY, "scripts/cost_service_frontier.py", "--tags", *diem])


def tong_hop_rq3():
    """[P4-2] RQ3 tren moi hat giong (SEEDS): moi bien the (toan cuc, khong chuan hoa) so
    voi moc abl_ref CUNG hat giong. Hieu nang: kiem dinh theo cap tren 30 lan
    danh gia test. On dinh huan luyen: tu train_log (episode dau tien dat fill
    >= 85% tren val, chi phi / entropy / explained variance 100 episode cuoi)."""
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(ROOT / "scripts"))
    from evaluate import paired_test

    def tag(base, s):
        return base if s == 42 else f"{base}_s{s}"

    def ippo_costs(t):
        p = ROOT / "results" / f"all_episodes_{t}.csv"
        if not p.exists():
            return None
        d = pd.read_csv(p)
        return d[d.policy == "IPPO"].sort_values("episode")

    def on_dinh(t):
        p = ROOT / "results" / f"train_log_{t}.csv"
        if not p.exists():
            return {}
        d = pd.read_csv(p)
        ev = d.dropna(subset=["eval_fill"])
        dat = ev[ev.eval_fill >= 0.85]
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
            row = {"cost": float(e.total_cost.mean()), "fill": float(e.fill_rate.mean()),
                   **on_dinh(t)}
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
    (ROOT / "results" / "rq3_multiseed.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("Da luu results/rq3_multiseed.json")
    for base, v in out.items():
        if "gap_vs_ref_mean" in v:
            print(f"  {base:<11}: chi phi so voi abl_ref {v['gap_vs_ref_mean']:+.1f}% (TB {v['n_seeds']} seed); "
                  f"abl_ref re hon co y nghia o {v['n_seeds_ref_re_hon_co_y_nghia']}/{v['n_seeds']} seed")


def tong_hop_da_hat_giong():
    """Tong hop cac hat giong (SEEDS) DUNG CACH: kiem dinh theo cap tinh RIENG
    tung hat giong (evaluate.py), roi bao cao trung binh +- do lech chuan GIUA
    cac hat giong - khong gop moi lan danh gia IPPO cua cac seed voi 30 lan
    baseline nhu quan sat doc lap."""
    import numpy as np
    rows = {}
    for s in MAIN_SEEDS:
        p = ROOT / "results" / f"summary_main_s{s}.json"
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
    refs = set().union(*[r["per_baseline"] for r in rows.values()])
    for ref in sorted(refs):
        g = [r["per_baseline"][ref]["gap_percent"] for r in rows.values() if ref in r["per_baseline"]]
        sig = [r["per_baseline"][ref]["p_paired"] < 0.05 for r in rows.values() if ref in r["per_baseline"]]
        out["gap_vs_baseline"][ref] = {"mean": float(np.mean(g)), "min": float(min(g)),
                                       "max": float(max(g)), "n_seeds_significant": int(sum(sig)),
                                       # tach theo chieu: co y nghia nhung IPPO re hon / dat hon
                                       "n_seeds_re_hon": int(sum(s and x < 0 for s, x in zip(sig, g))),
                                       "n_seeds_dat_hon": int(sum(s and x > 0 for s, x in zip(sig, g))),
                                       "n_seeds": len(g)}
    (ROOT / "results" / "multiseed_main.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Da luu results/multiseed_main.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lenh", choices=["status", "train", "eval", "eval_mo_rong", "tonghop"])
    ap.add_argument("ten", nargs="?", help="ten lan chay, 'main', 'ablations' hoac 'all'")
    ap.add_argument("--n_eval", type=int, default=30)
    a = ap.parse_args()
    if a.lenh == "status":
        status()
    elif a.lenh == "eval":
        evaluate_all(a.n_eval)
    elif a.lenh == "eval_mo_rong":
        eval_mo_rong(a.n_eval)
    elif a.lenh == "tonghop":
        tong_hop_da_hat_giong()
        tong_hop_rq3()
        sys.path.insert(0, str(ROOT / "scripts"))
        from build_results import ghi_ket_qua
        print("Da luu", ghi_ket_qua(ROOT))
    elif a.ten == "auto":
        train_auto()
    elif a.ten == "full":
        train_full()
    else:
        # "all": mo hinh chinh truoc (RQ1/RQ2), roi ablation theo thu tu uu tien
        nhom = {"main": list(MAIN), "ablations": list(ABLATIONS),
                "all": list(MAIN) + [t for t in PRIORITY if t not in MAIN], **NHOM_MO_RONG}
        tags = nhom.get(a.ten, [a.ten])
        bad = [t for t in tags if t not in RUNS]
        if bad:
            sys.exit(f"Khong co lan chay {bad}. Chon: {list(RUNS)} hoac "
                     f"main/ablations/all/{'/'.join(NHOM_MO_RONG)}")
        for t in tags:
            train(t)


if __name__ == "__main__":
    main()
