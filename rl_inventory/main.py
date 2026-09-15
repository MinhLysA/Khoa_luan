"""
main.py — giữ lại cho tương thích ngược. Điểm vào chính thức bây giờ là run.py.

    python run.py app       # bảng điều khiển Streamlit
    python run.py all       # chạy trọn bộ
    python run.py --help
"""
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANH_XA = {"preprocess": "data", "train": "train", "evaluate": "eval"}

if __name__ == "__main__":
    print("[!] main.py đã được thay bằng run.py. Xem README.md.\n")
    if len(sys.argv) > 1 and sys.argv[1] in ANH_XA:
        lenh = ANH_XA[sys.argv[1]]
        print(f"    Đang chuyển tiếp sang:  python run.py {lenh}\n")
        sys.exit(subprocess.run([sys.executable, "run.py", lenh] + sys.argv[2:],
                                cwd=str(ROOT)).returncode)
    subprocess.run([sys.executable, "run.py", "--help"], cwd=str(ROOT))
