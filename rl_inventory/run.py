#!/usr/bin/env python
"""
run.py — mot cua duy nhat de chay ca du an:  python run.py <lenh>
cd D:\\KhoaLuanTotNghiep\\KhoaLuan_RL_Inventory_FIXED\\KhoaLuan_RL_Inventory_v2\\Khoa_luan\\rl_inventory; py -3.12 run.py full

Dot thuc nghiem cuoi (so lieu bao cao; seed o scripts/campaign.py):
    full             test -> train 19 lan chay -> danh gia -> results/KET_QUA.txt
                     Bi ngat thi chay lai, tu resume. Log tu ghi vao logs/<lenh>.txt
                     (full, train_*, eval_all) - khong can 2>&1 / Tee-Object.
    status           tien do tung lan chay
    train_main       chi train mo hinh chinh (3 seed x ppo.total_episodes)
    train_ablations  chi train ablation, RQ3, hold-out (1.000 ep)
    eval_all         danh gia + moi phan tich -> KET_QUA.txt
    tonghop          dung lai KET_QUA.txt tu ket qua da co

Chuan bi:  check | data (tien xu ly M5) | test
Demo:      demo_data (dong goi app/demo_data/) | app (Streamlit, chi inference)

Giao thuc cu / thu nhanh (khong dung cho ban nop):
    all | baseline | train | eval | iso | summary | multiseed | regime | behavior
    Tuy chon: --episodes N, --quick (~5 phut), --checkpoint (regime/behavior)
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
CAMPAIGN = [PY, "scripts/campaign.py"]
os.environ.setdefault("PYTHONUNBUFFERED", "1")     # log tien trinh con hien ngay
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # tat thong bao TensorFlow (tu tensorboard)
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
sys.stdout.reconfigure(errors="replace")
LOG = None   # file log cua cac lenh chay dai, mo trong main()

MULTISEED_SEEDS = [42, 1, 2]   # [V3-6] cho lenh multiseed (giao thuc cu)


def sh(cmd):
    """Chay lenh con. Khi co LOG: gop stderr vao stdout, vua in vua ghi log, nen
    PowerShell khong to do stderr thanh loi (NativeCommandError)."""
    dong = "\n$ " + " ".join(str(c) for c in cmd)
    print(dong, flush=True)
    if LOG is None:
        code = subprocess.run(cmd, cwd=str(ROOT)).returncode
    else:
        LOG.write(dong + "\n")
        p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace")
        for s in p.stdout:
            sys.stdout.write(s)
            sys.stdout.flush()
            LOG.write(s)
            LOG.flush()
        code = p.wait()
    if code != 0:
        msg = f"\n[LOI] lenh tren thoat voi ma {code} - dung lai."
        print(msg)
        if LOG:
            LOG.write(msg + "\n")
        sys.exit(code)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lenh", choices=["full", "status", "train_main", "train_ablations",
                                     "eval_all", "tonghop",
                                     "app", "demo_data", "all", "data", "baseline", "train",
                                     "eval", "iso", "summary", "multiseed",
                                     "regime", "behavior", "test", "check"])
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--quick", action="store_true",
                    help="Ban rut gon: 60 episode, 5 episode danh gia")
    ap.add_argument("--n_skus", type=int, default=None)
    ap.add_argument("--n_warehouses", type=int, default=None)
    ap.add_argument("--checkpoint", default="checkpoints/best_model_main_s42.pth",
                    help="Checkpoint IPPO cho lenh regime/behavior")
    a = ap.parse_args()

    global LOG
    if a.lenh in ("full", "train_main", "train_ablations", "eval_all"):
        (ROOT / "logs").mkdir(exist_ok=True)
        LOG = open(ROOT / "logs" / f"{a.lenh}.txt", "a", encoding="utf-8")
        LOG.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  python run.py {a.lenh} =====\n")
        print(f"Log: logs/{a.lenh}.txt")

    import yaml
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    n_wh = a.n_warehouses or cfg["env"]["n_warehouses"]
    n_sku = a.n_skus or cfg["env"]["n_skus"]
    eps = a.episodes or (60 if a.quick else cfg["ppo"]["total_episodes"])
    eval_eps = 5 if a.quick else cfg["eval"]["n_episodes"]
    tune_eps = 1 if a.quick else 3

    cmd_data = [PY, "scripts/data_preprocessing.py", "--m5",
                "--n_warehouses", str(n_wh), "--n_skus", str(n_sku),
                "--min_mean_demand", str(cfg["preprocess"].get("min_mean_demand", 0.2)),
                # phai khop env.split_day: chi uoc luong gia/cau tu mien train
                "--split_day", str(cfg["env"].get("split_day", 1050))]
    cmd_base = [PY, "scripts/tune_baselines.py", "--episodes", str(tune_eps)]
    cmd_train = [PY, "scripts/train.py", "--episodes", str(eps)]
    cmd_eval = [PY, "scripts/evaluate.py", "--episodes", str(eval_eps)]
    cmd_iso = [PY, "scripts/iso_service.py", "--tune_episodes", str(tune_eps),
               "--eval_episodes", str(eval_eps)]
    cmd_summary = [PY, "scripts/generate_summary.py"]

    if a.lenh == "full":
        sh([PY, "-m", "pytest", "tests/", "-q"])
        sh(CAMPAIGN + ["train", "all"])
        sh(CAMPAIGN + ["eval", "--n_eval", str(eval_eps)])
        print("\nXONG. So lieu cho bao cao: results/KET_QUA.txt (chi tiet: results/*.json, *.png)")
    elif a.lenh == "status":
        sh(CAMPAIGN + ["status"])
    elif a.lenh == "train_main":
        sh(CAMPAIGN + ["train", "main"])
    elif a.lenh == "train_ablations":
        sh(CAMPAIGN + ["train", "ablations"])
    elif a.lenh == "eval_all":
        sh(CAMPAIGN + ["eval", "--n_eval", str(eval_eps)])
    elif a.lenh == "tonghop":
        sh(CAMPAIGN + ["tonghop"])
    elif a.lenh == "app":
        # chi inference tu app/demo_data/ (can chay demo_data truoc)
        sh([PY, "-m", "streamlit", "run", "app/streamlit_app.py"])
    elif a.lenh == "demo_data":
        sh([PY, "scripts/build_demo_data.py"])
    elif a.lenh == "data":
        sh(cmd_data)
    elif a.lenh == "baseline":
        sh(cmd_base)
    elif a.lenh == "train":
        sh(cmd_train)
    elif a.lenh == "eval":
        sh(cmd_eval)
    elif a.lenh == "iso":
        sh(cmd_iso)
    elif a.lenh == "summary":
        sh(cmd_summary)
    elif a.lenh == "multiseed":
        # moi seed mot --tag rieng -> khong ghi de nhau
        ms_episodes = a.episodes or 1000
        for s in MULTISEED_SEEDS:
            tag = f"seed{s}"
            sh([PY, "scripts/train.py", "--seed", str(s),
               "--episodes", str(ms_episodes), "--tag", tag])
            sh([PY, "scripts/evaluate.py",
               "--checkpoint", f"checkpoints/best_model_{tag}.pth",
               "--episodes", str(cfg["eval"]["n_episodes"]), "--tag", tag])
        sh(cmd_summary)
        print("\nXong da hat giong. Xem muc 'DA HAT GIONG' trong TONG_HOP_SO_LIEU.txt.")
    elif a.lenh == "regime":
        sh([PY, "scripts/regime_analysis.py", "--checkpoint", a.checkpoint]
           + (["--episodes", "5"] if a.quick else []))
    elif a.lenh == "behavior":
        sh([PY, "scripts/policy_behavior.py", "--checkpoint", a.checkpoint]
           + (["--episodes", "3"] if a.quick else []))
    elif a.lenh == "test":
        sh([PY, "-m", "pytest", "tests/", "-q"])
    elif a.lenh == "all":
        sh(cmd_data); sh(cmd_base); sh(cmd_train); sh(cmd_eval); sh(cmd_iso); sh(cmd_summary)
        print("\nXong. Mo bang dieu khien:  python run.py app")
    elif a.lenh == "check":
        ok = True
        for mod in ["torch", "gymnasium", "numpy", "pandas", "yaml",
                    "scipy", "matplotlib", "streamlit"]:
            try:
                __import__(mod)
                print(f"  [OK] {mod}")
            except ImportError:
                print(f"  [THIEU] {mod}  ->  pip install -r requirements.txt")
                ok = False
        for p, mo_ta in [("data/raw/sales_train_evaluation.csv", "du lieu M5 goc"),
                         ("data/processed/demand_data.npy", "du lieu da tien xu ly"),
                         ("checkpoints/best_model.pth", "checkpoint da huan luyen"),
                         ("results/baseline_comparison.csv", "ket qua danh gia"),
                         ("results/iso_service.json", "so sanh cung muc phuc vu")]:
            print(f"  [{'OK' if (ROOT / p).exists() else '--'}] {mo_ta}: {p}")
        print("\nSan sang." if ok else "\nCai them thu vien con thieu roi chay lai.")


if __name__ == "__main__":
    main()
