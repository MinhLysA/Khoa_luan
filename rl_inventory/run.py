#!/usr/bin/env python
"""
run.py — mot cua duy nhat de chay ca du an:  py -3.12 run.py <lenh>   (trong thu muc rl_inventory)

Dot thuc nghiem cuoi (so lieu bao cao; danh sach lan chay o scripts/run_experiments.py):
    full             test -> train toan bo -> danh gia -> results/KET_QUA.txt
                     --quick: thu ca day chuyen tren ban sao, moi lan train 3 episode
    eval_all         chi danh gia + moi phan tich -> KET_QUA.txt (bo qua buoc da co ket qua moi)
    tonghop          dung lai KET_QUA.txt tu ket qua da co (vai giay)
    status           tien do tung lan chay
    train_main       chi train mo hinh chinh (3 seed x ppo.total_episodes)
    train_ablations  chi train ablation, Cau 3, mat hang chua gap (1.000 episode)
    forecast         du bao LightGBM (truoc khi tinh chinh baseline va train Cau 4)
    Bi ngat thi chay lai cung lenh, tu noi tiep. Log ghi vao logs/<lenh>.txt.

Chuan bi:  check | data (tien xu ly M5) | test
Demo:      demo_data (dong goi app/demo_data/) | app (Streamlit, chi suy luan)
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
CAMPAIGN = [PY, "scripts/run_experiments.py"]
os.environ.setdefault("PYTHONUNBUFFERED", "1")     # log tien trinh con hien ngay
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # tat thong bao TensorFlow (tu tensorboard)
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
sys.stdout.reconfigure(errors="replace")
LOG = None   # file log cua cac lenh chay dai, mo trong main()



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
    ap.add_argument("lenh", choices=["full", "status", "train_main", "train_ablations", "eval_all",
                                     "tonghop", "forecast", "app", "demo_data", "data", "test", "check"])
    ap.add_argument("--quick", action="store_true", help="full: thu ca day chuyen tren ban sao")
    a = ap.parse_args()

    if a.quick and a.lenh == "full" and not os.environ.get("KLTN_THU"):
        # Chay tren ban sao de ket qua thu khong lan vao ket qua that
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            w = Path(tmp) / "rl_inventory"
            shutil.copytree(ROOT, w, ignore=shutil.ignore_patterns(
                "checkpoints", "results", "runs", "logs", "__pycache__", ".pytest_cache", "*.pdf"))
            code = subprocess.run([PY, "run.py", "full", "--quick"], cwd=w,
                                  env=dict(os.environ, KLTN_THU="3")).returncode
        print("\nChay thu " + ("THANH CONG, co the train that." if code == 0 else "LOI, xem thong bao o tren."))
        sys.exit(code)

    global LOG
    if a.lenh in ("full", "train_main", "train_ablations", "eval_all"):
        (ROOT / "logs").mkdir(exist_ok=True)
        LOG = open(ROOT / "logs" / f"{a.lenh}.txt", "a", encoding="utf-8")
        LOG.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  python run.py {a.lenh} =====\n")
        print(f"Log: logs/{a.lenh}.txt")

    import yaml
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    eval_eps = 2 if a.quick else cfg["eval"]["n_episodes"]

    if a.lenh == "full":
        sh([PY, "-c", "import lightgbm"])      # buoc eval can; bao loi ngay, khong doi het train
        sh([PY, "-m", "pytest", "tests/", "-q"])
        sh(CAMPAIGN + ["train", "full"])
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
    elif a.lenh == "forecast":
        sh([PY, "scripts/forecast_demand.py"])
    elif a.lenh == "app":
        # chi inference tu app/demo_data/ (can chay demo_data truoc)
        sh([PY, "-m", "streamlit", "run", "app/streamlit_app.py"])
    elif a.lenh == "demo_data":
        sh([PY, "scripts/build_demo_data.py"])
    elif a.lenh == "data":
        sh([PY, "scripts/data_preprocessing.py", "--m5",
            "--n_warehouses", str(cfg["env"]["n_warehouses"]), "--n_skus", str(cfg["env"]["n_skus"]),
            "--min_mean_demand", str(cfg["preprocess"].get("min_mean_demand", 0.2)),
            # phai khop env.split_day: chi uoc luong gia/cau tu mien train
            "--split_day", str(cfg["env"].get("split_day", 1050))])
    elif a.lenh == "test":
        sh([PY, "-m", "pytest", "tests/", "-q"])
    elif a.lenh == "check":
        ok = True
        for mod in ["torch", "gymnasium", "numpy", "pandas", "yaml",
                    "scipy", "matplotlib", "streamlit", "lightgbm"]:
            try:
                __import__(mod)
                print(f"  [OK] {mod}")
            except ImportError:
                print(f"  [THIEU] {mod}  ->  pip install -r requirements.txt")
                ok = False
        try:
            import torch
            if torch.cuda.is_available():
                print(f"  [OK] GPU {torch.cuda.get_device_name(0)}: train chay tren cuda")
            else:
                print(f"  [--] Khong co GPU CUDA (torch {torch.__version__}): train chay tren CPU")
        except ImportError:
            pass
        for p, mo_ta in [("data/raw/sales_train_evaluation.csv", "du lieu M5 goc"),
                         ("data/processed/demand_data.npy", "du lieu da tien xu ly"),
                         ("checkpoints/best_model_main_s42.pth", "mo hinh chinh seed 42"),
                         ("results/forecast_test.npy", "du bao LightGBM"),
                         ("results/KET_QUA.txt", "so lieu tong")]:
            print(f"  [{'OK' if (ROOT / p).exists() else '--'}] {mo_ta}: {p}")
        print("\nSan sang." if ok else "\nCai them thu vien con thieu roi chay lai.")


if __name__ == "__main__":
    main()
