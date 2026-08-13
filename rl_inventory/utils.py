"""
utils.py
========
Các hàm tiện ích dùng chung cho toàn bộ project.

- load_config      : Đọc config.yaml và merge với DEFAULT_CONFIG
- load_demand_data : Tải dữ liệu nhu cầu (M5 thực tế hoặc synthetic fallback)
- set_seed         : Đảm bảo tính tái lập (reproducibility)
"""

from __future__ import annotations

import sys
import os

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import numpy as np
import torch
from pathlib import Path
from typing import Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[str] = None) -> dict:
    """
    Đọc config.yaml và trả về dict cấu hình hoàn chỉnh.

    Nếu config_path là None hoặc file không tồn tại, trả về dict rỗng
    (các script sẽ dùng giá trị mặc định của riêng mình).

    Tham số
    ------
    config_path : str hoặc None
        Đường dẫn tới file config.yaml.

    Trả về
    -----
    dict — toàn bộ cấu hình từ YAML, hoặc {} nếu không tìm thấy file.
    """
    if config_path is None:
        return {}

    cfg_file = Path(config_path)
    if not cfg_file.exists():
        print(f"[Config] Không tìm thấy {config_path}, dùng giá trị mặc định.")
        return {}

    if not HAS_YAML:
        raise ImportError(
            "Cần cài đặt PyYAML để đọc config.yaml:\n"
            "  pip install pyyaml"
        )

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    return cfg


# ---------------------------------------------------------------------------
# load_demand_data
# ---------------------------------------------------------------------------

def load_demand_data(
    data_dir: str,
    n_warehouses: int = 2,
    n_skus: int = 30,
    n_days: int = 800,
    seed: int = 42,
    synthetic: bool = False,
) -> tuple:
    """
    Tải dữ liệu nhu cầu và cấu hình môi trường.

    Ưu tiên dùng dữ liệu M5 đã xử lý trong data_dir. Nếu không tìm thấy
    hoặc synthetic=True, tạo dữ liệu giả lập bằng generate_synthetic_fallback.

    Tham số
    ------
    data_dir : str
        Thư mục chứa demand_data.npy và env_config.json.
    n_warehouses, n_skus, n_days : int
        Tham số cho dữ liệu synthetic (chỉ dùng khi không có M5).
    seed : int
        Hạt giống cho dữ liệu synthetic.
    synthetic : bool
        Nếu True, bỏ qua M5 và dùng dữ liệu giả lập.

    Trả về
    -----
    demand_data : np.ndarray, shape (T, n_w, n_s)
    env_config  : dict
    """
    # Import ở đây để tránh circular import
    from env.inventory_env import DEFAULT_CONFIG
    from scripts.data_preprocessing import generate_synthetic_fallback

    data_dir = Path(data_dir)
    np_path  = data_dir / "demand_data.npy"
    cfg_path = data_dir / "env_config.json"

    if not synthetic and np_path.exists() and cfg_path.exists():
        print(f"[Data] Tải dữ liệu thực tế từ {data_dir}")
        demand_data = np.load(str(np_path))
        with open(cfg_path) as f:
            env_config = json.load(f)
        print(f"  Kích thước demand_data: {demand_data.shape}")
    else:
        if not synthetic:
            print("[Data] Không tìm thấy dữ liệu M5 → dùng synthetic fallback")
        else:
            print("[Data] Dùng dữ liệu synthetic (--synthetic flag)")
        demand_data = generate_synthetic_fallback(
            n_warehouses=n_warehouses,
            n_skus=n_skus,
            n_days=n_days,
            seed=seed,
        )
        env_config = {
            **DEFAULT_CONFIG,
            "n_warehouses": n_warehouses,
            "n_skus": n_skus,
        }

    return demand_data, env_config


# ---------------------------------------------------------------------------
# set_seed
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    Đặt seed cho numpy và torch để đảm bảo tính tái lập.

    Tham số
    ------
    seed : int
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
