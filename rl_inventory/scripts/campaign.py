"""
scripts/campaign.py
===================
MOT noi dinh nghia TOAN BO dot thuc nghiem cuoi cung: moi lan huan luyen (mo
hinh chinh + ablation) va pipeline danh gia. Chay duoc ca local lan Colab;
moi lenh train TU RESUME tu checkpoint gan nhat neu bi ngat giua chung.

    python scripts/campaign.py status              # tien do tung lan chay
    python scripts/campaign.py train main_s42      # train 1 lan chay (tu resume)
    python scripts/campaign.py train ablations     # lan luot moi ablation
    python scripts/campaign.py train auto          # tu lay lan chay tiep theo (nhieu phien song song)
    python scripts/campaign.py eval                # toan bo pipeline danh gia
"""
import argparse
import json
import os
import random
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

# ten lan chay -> (config, seed, so episode, mo ta)
MAIN = {f"main_s{s}": ("config.yaml", s, 4000, f"Mo hinh chinh, seed {s}")
        for s in (42, 1, 2)}
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
RUNS = {**MAIN, **ABLATIONS}


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


def train(tag):
    cfg, seed, eps, desc = RUNS[tag]
    if done(tag):
        print(f"[{tag}] da xong ({eps} episode) - bo qua.")
        return
    cmd = [PY, "scripts/train.py", "--config", cfg, "--seed", seed,
           "--episodes", eps, "--tag", tag]
    ck = latest_checkpoint(tag)
    if ck:
        cmd += ["--resume", ck.relative_to(ROOT).as_posix()]
        print(f"[{tag}] resume tu {ck.name} (da co {progress(tag)} episode)")
    print(f"[{tag}] {desc}")
    sh(cmd)


# ---------------------------------------------------------------------------
# Che do "auto": nhieu phien Colab cung chay tren MOT thu muc Drive. Moi phien
# tu lay lan chay chua xong va chua co phien nao dang chay, theo thu tu uu tien.
# ---------------------------------------------------------------------------
PRIORITY = ["main_s42", "abl_ref", "abl_global", "abl_q3", "main_s1", "main_s2",
            "abl_reward_cu", "holdout", "abl_trunk", "abl_nocal", "abl_nowh",
            "abl_event", "abl_warm"]
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
    time.sleep(15)          # cho Drive dong bo, roi kiem tra lai co bi phien khac chen khong
    return lock_owner(tag) == ME


def release(tag):
    try:
        if lock_owner(tag) in (ME, None):
            _lock(tag).unlink()
    except OSError:
        pass


def train_auto():
    time.sleep(random.uniform(0, 20))       # tranh 2 phien khoi dong cung luc lay trung
    while True:
        con_lai = [t for t in PRIORITY if not done(t)]
        if not con_lai:
            print("\nTAT CA LAN CHAY DA XONG. Chay: python scripts/campaign.py eval")
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


def status():
    print(f"{'Lan chay':<15}{'Tien do':>14}  {'Dang chay boi':<22} Mo ta")
    for tag, (cfg, seed, eps, desc) in RUNS.items():
        mark = "XONG" if done(tag) else f"{progress(tag)}/{eps}"
        print(f"{tag:<15}{mark:>14}  {(lock_owner(tag) or '-'):<22} {desc}")


def ck(tag):
    p = ROOT / "checkpoints" / f"best_model_{tag}.pth"
    return p.relative_to(ROOT).as_posix() if p.exists() else None


def evaluate_all(n_eval=30):
    miss = [t for t in RUNS if not ck(t)]
    if miss:
        print(f"CANH BAO: chua co checkpoint cho {miss} - cac buoc lien quan se bo qua.")
    main = ck("main_s42")

    # 1. Baseline tinh chinh tren mien VAL
    sh([PY, "scripts/tune_baselines.py", "--episodes", 3])
    # 2. Danh gia 3 seed mo hinh chinh (+ cua so co dinh cho seed 42)
    for s in (42, 1, 2):
        if ck(f"main_s{s}"):
            sh([PY, "scripts/evaluate.py", "--checkpoint", ck(f"main_s{s}"),
                "--episodes", n_eval, "--tag", f"main_s{s}"])
            sh([PY, "scripts/plot_learning_curve.py", "--tag", f"main_s{s}"])
    if main:
        # 3. Cung muc phuc vu -> 4. giai doan -> 5. hanh vi -> 6. quy mo -> 7. do nhay
        sh([PY, "scripts/iso_service.py", "--checkpoint", main,
            "--tune_episodes", 3, "--eval_episodes", n_eval])
        sh([PY, "scripts/regime_analysis.py", "--checkpoint", main, "--tag", "main"])
        sh([PY, "scripts/policy_behavior.py", "--checkpoint", main, "--tag", "main"])
        sh([PY, "scripts/analyze_scale_groups.py", "--checkpoint", main, "--tag", "main"])
        sh([PY, "scripts/sensitivity.py", "--checkpoint", main, "--tag", "main"])
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
    cmp = [t for t in ABLATIONS if t != "holdout" and ck(t)]
    if "abl_ref" in cmp and len(cmp) > 1:
        sh([PY, "scripts/plot_learning_curve.py", "--compare", *cmp, "--max_episode", 1000])
    # 9. Hold-out: 6 SKU chua gap (so voi mo hinh cung 1000 episode da thay ca 30 SKU)
    for tag, out in [("holdout", "holdout_unseen"), ("abl_ref", "holdout_seen_ref"),
                     ("main_s42", "holdout_seen_main")]:
        if ck(tag):
            sh([PY, "scripts/evaluate.py", "--config", "configs/holdout_eval.yaml",
                "--checkpoint", ck(tag), "--episodes", n_eval, "--tag", out])
    tong_hop_da_hat_giong()
    print("\nXONG. Ket qua trong results/*_main*, *_abl_*, *holdout*.")


def tong_hop_da_hat_giong():
    """Tong hop 3 hat giong DUNG CACH: kiem dinh theo cap tinh RIENG tung hat
    giong (evaluate.py), roi bao cao trung binh +- do lech chuan GIUA cac hat
    giong - khong gop 90 lan danh gia IPPO voi 30 lan baseline nhu quan sat
    doc lap."""
    import numpy as np
    rows = {}
    for s in (42, 1, 2):
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
                                       "n_seeds": len(g)}
    (ROOT / "results" / "multiseed_main.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Da luu results/multiseed_main.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lenh", choices=["status", "train", "eval", "tonghop"])
    ap.add_argument("ten", nargs="?", help="ten lan chay, 'main', 'ablations' hoac 'all'")
    ap.add_argument("--n_eval", type=int, default=30)
    a = ap.parse_args()
    if a.lenh == "status":
        status()
    elif a.lenh == "eval":
        evaluate_all(a.n_eval)
    elif a.lenh == "tonghop":
        tong_hop_da_hat_giong()
    elif a.ten == "auto":
        train_auto()
    else:
        nhom = {"main": list(MAIN), "ablations": list(ABLATIONS), "all": list(RUNS)}
        tags = nhom.get(a.ten, [a.ten])
        bad = [t for t in tags if t not in RUNS]
        if bad:
            sys.exit(f"Khong co lan chay {bad}. Chon: {list(RUNS)} hoac main/ablations/all")
        for t in tags:
            train(t)


if __name__ == "__main__":
    main()
