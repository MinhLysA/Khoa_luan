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
    python run.py summary      # buoc 6: xuat TONG_HOP_SO_LIEU.txt tu ket qua hien co
    python run.py multiseed    # train + eval 3 SEED (do tin cay thong ke), roi xuat summary
    python run.py test         # chay unit test moi truong
    python run.py check        # kiem tra nhanh moi thu da san sang chua

Tuy chon hay dung:
    python run.py train --episodes 800
    python run.py all --quick          # ban rut gon de thu duong ong (~5 phut)
    python run.py multiseed --episodes 1000   # doi so episode moi seed (mac dinh 1000)
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

# [V3-6] 3 seed dung de danh gia do tin cay thong ke (python run.py multiseed).
# Co dinh (khong random moi lan chay) de ket qua tai lap duoc.
MULTISEED_SEEDS = [42, 1, 2]


def sh(cmd):
    print("\n$ " + " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lenh", choices=["app", "all", "data", "baseline", "train",
                                     "eval", "iso", "summary", "multiseed",
                                     "test", "check"])
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
                "--min_mean_demand", str(cfg["preprocess"].get("min_mean_demand", 0.2)),
                # split_day phai KHOP voi config.yaml env.split_day, vi day la
                # ranh gioi loc SKU va uoc luong gia/cau chi tu mien train.
                "--split_day", str(cfg["env"].get("split_day", 1050))]
    cmd_base = [PY, "scripts/tune_baselines.py", "--episodes", str(tune_eps)]
    cmd_train = [PY, "scripts/train.py", "--episodes", str(eps)]
    cmd_eval = [PY, "scripts/evaluate.py", "--episodes", str(eval_eps)]
    cmd_iso = [PY, "scripts/iso_service.py", "--tune_episodes", str(tune_eps),
               "--eval_episodes", str(eval_eps)]
    cmd_summary = [PY, "scripts/generate_summary.py"]

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
    elif a.lenh == "summary":
        sh(cmd_summary)
    elif a.lenh == "multiseed":
        # [V3-6] Train + danh gia MULTISEED_SEEDS (mac dinh 3 seed), moi seed
        # gan --tag rieng nen khong ghi de ket qua cua seed truoc. Mac dinh
        # 1000 episode/seed (nhanh hon 5000 de kiem tra do tin cay thong ke,
        # doi bang --episodes).
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
