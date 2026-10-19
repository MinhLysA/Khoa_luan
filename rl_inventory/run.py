#!/usr/bin/env python
"""
run.py — mot cua duy nhat de chay ca du an.

=== CHAY QUA DEM: MOT LENH CHO TOAN BO SO LIEU BAO CAO ===
    python run.py full             # test -> train TOAN BO dot cuoi (3 seed mo hinh chinh
                                   # x 4.000 ep, RQ3 x 3 seed, cac ablation, hold-out)
                                   # -> danh gia + phan tich -> results/KET_QUA.txt
    Bi ngat giua chung (tat may, mat dien): chay lai DUNG lenh do - lan train da
    xong duoc bo qua, lan dang do tu resume; buoc danh gia chay lai tu dau.
    Luu log:  python run.py full 2>&1 | Tee-Object logs\\full.txt      (PowerShell)

=== DOT THUC NGHIEM CUOI, TUNG BUOC (scripts/campaign.py, seed: campaign.SEEDS) ===
    python run.py status           # tien do tung lan chay (main_s*, abl_*, holdout)
    python run.py train_main       # chi train mo hinh chinh (moi seed, 4.000 ep)
    python run.py train_ablations  # chi train ablation + RQ3 nhieu seed + hold-out (1.000 ep)
    python run.py eval_all         # tune baseline -> danh gia moi mo hinh -> iso, regime,
                                   # behavior, quy mo, do nhay, do ben, hold-out -> KET_QUA.txt
    python run.py tonghop          # chi dung lai multiseed_main.json, rq3_multiseed.json,
                                   # KET_QUA.txt tu ket qua da co

=== CHUAN BI ===
    python run.py check        # kiem tra thu vien + file can thiet
    python run.py data         # tien xu ly M5 -> data/processed/
    python run.py test         # chay unit test

=== TUNG BUOC LE (giao thuc cu / thu nhanh, KHONG dung cho ban nop) ===
    python run.py all          # du lieu -> baseline -> train -> danh gia (1 seed)
    python run.py baseline     # tim kiem luoi tham so baseline
    python run.py train        # huan luyen IPPO mot lan
    python run.py eval         # danh gia & so sanh
    python run.py iso          # so sanh O CUNG MUC PHUC VU
    python run.py summary      # xuat TONG_HOP_SO_LIEU.txt tu ket qua hien co
    python run.py multiseed    # train + eval 3 seed x 1.000 ep, roi xuat summary
    python run.py regime       # thuc nghiem B: giai doan nhu cau + soc cau
    python run.py behavior     # hanh vi policy, reward hacking, tam quan trong state

=== DEMO ===
    python run.py demo_data    # dong goi du lieu rut gon cho demo vao app/demo_data/
    python run.py app          # mo demo Streamlit (chi inference, doc app/demo_data/)

Tuy chon hay dung:
    python run.py train --episodes 800
    python run.py all --quick          # ban rut gon de thu duong ong (~5 phut)
    python run.py multiseed --episodes 1000   # doi so episode moi seed (mac dinh 1000)
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
CAMPAIGN = [PY, "scripts/campaign.py"]
# Log cua cac tien trinh con hien ngay ca khi chuyen huong ra file (Tee-Object).
os.environ.setdefault("PYTHONUNBUFFERED", "1")

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

    if a.lenh == "full":
        # Test truoc de khong mat ca dem train tren mot moi truong dang loi.
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
        # Demo 5 trang, CHI INFERENCE: nap app/demo_data/best_model.pth, khong
        # co duong code nao goi huan luyen. Can `python run.py demo_data` truoc.
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
