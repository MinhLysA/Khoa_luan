#!/usr/bin/env python
"""
run.py — mot cua duy nhat de chay ca du an.

    python run.py app          # mo bang dieu khien Streamlit (khuyen dung)
    python run.py all          # chay tron bo: du lieu -> baseline -> train -> danh gia
    python run.py data         # buoc 1: tien xu ly M5
    python run.py baseline     # buoc 2: tim kiem luoi tham so baseline
    python run.py train        # buoc 3: huan luyen IPPO
    python run.py eval         # buoc 4: danh gia & so sanh
    python run.py iso          # buoc 5: so sanh O CUNG MUC PHUC VU (quan trong)
    python run.py test         # chay unit test moi truong
    python run.py check        # kiem tra nhanh moi thu da san sang chua

Tuy chon hay dung:
    python run.py train --episodes 800
    python run.py all --quick          # ban rut gon de thu duong ong (~5 phut)
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def sh(cmd):
    print("\n$ " + " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lenh", choices=["app", "all", "data", "baseline",
                                     "train", "eval", "iso", "test", "check"])
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--quick", action="store_true",
                    help="Ban rut gon: 60 episode, 5 episode danh gia")
    ap.add_argument("--n_skus", type=int, default=None)
    ap.add_argument("--n_warehouses", type=int, default=None)
    a = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    n_wh = a.n_warehouses or cfg["env"]["n_warehouses"]
    n_sku = a.n_skus or cfg["env"]["n_skus"]
    eps = a.episodes or (60 if a.quick else cfg["ppo"]["total_episodes"])
    eval_eps = 5 if a.quick else cfg["eval"]["n_episodes"]
    tune_eps = 1 if a.quick else 3

    cmd_data = [PY, "scripts/data_preprocessing.py", "--m5",
                "--n_warehouses", str(n_wh), "--n_skus", str(n_sku),
                "--min_mean_demand", str(cfg["preprocess"].get("min_mean_demand", 0.2))]
    cmd_base = [PY, "scripts/tune_baselines.py", "--episodes", str(tune_eps)]
    cmd_train = [PY, "scripts/train.py", "--episodes", str(eps)]
    cmd_eval = [PY, "scripts/evaluate.py", "--episodes", str(eval_eps)]
    cmd_iso = [PY, "scripts/iso_service.py", "--tune_episodes", str(tune_eps),
               "--eval_episodes", str(eval_eps)]

    if a.lenh == "app":
        sh([PY, "-m", "streamlit", "run", "app/streamlit_app.py"])
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
    elif a.lenh == "test":
        sh([PY, "-m", "pytest", "tests/", "-q"])
    elif a.lenh == "all":
        sh(cmd_data); sh(cmd_base); sh(cmd_train); sh(cmd_eval); sh(cmd_iso)
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
