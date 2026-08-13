"""
demo_app/backend/model_loader.py
================================
Quản lý nạp mô hình Double DQN đã huấn luyện từ các tệp checkpoint (.pth).

Sử dụng sys.path để tái sử dụng trực tiếp các class từ module `rl_inventory` gốc.
"""

from __future__ import annotations

import os
import sys
import torch
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

# Đảm bảo thư mục rl_inventory nằm trong sys.path để import trực tiếp
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RL_INVENTORY_DIR = PROJECT_ROOT / "rl_inventory"

if str(RL_INVENTORY_DIR) not in sys.path:
    sys.path.insert(0, str(RL_INVENTORY_DIR))

from agents.dqn_agent import DoubleDQNAgent


def discover_checkpoints(checkpoint_dir: Optional[str] = None) -> List[Path]:
    """
    Tự động tìm tất cả các file checkpoint .pth trong thư mục chỉ định.

    Tham số
    ------
    checkpoint_dir : str hoặc None
        Đường dẫn tới thư mục checkpoint. Nếu None, dùng rl_inventory/checkpoints/

    Trả về
    -----
    List[Path] — Danh sách đường dẫn file .pth được sắp xếp theo thời gian sửa đổi gần nhất.
    """
    if checkpoint_dir is None:
        target_dir = RL_INVENTORY_DIR / "checkpoints"
    else:
        target_dir = Path(checkpoint_dir)

    if not target_dir.exists():
        return []

    ckpts = list(target_dir.glob("*.pth"))
    # Sắp xếp ưu tiên best_model.pth và final_model.pth lên đầu, sau đó theo mtime
    def sort_key(p: Path):
        name = p.name.lower()
        if name == "best_model.pth":
            return (0, -p.stat().st_mtime)
        elif name == "final_model.pth":
            return (1, -p.stat().st_mtime)
        return (2, -p.stat().st_mtime)

    ckpts.sort(key=sort_key)
    return ckpts


def load_trained_agent(
    checkpoint_path: str,
    env_config: Dict[str, Any],
    device: str = "auto",
) -> DoubleDQNAgent:
    """
    Khởi tạo và nạp trọng số mô hình Double DQN từ file checkpoint.

    Tham số
    ------
    checkpoint_path : str
        Đường dẫn tới tệp checkpoint .pth
    env_config : dict
        Cấu hình môi trường (chứa n_warehouses, n_skus, order_levels, v.v.)
    device : str
        Thiết bị tính toán ('cuda', 'cpu', hoặc 'auto')

    Trả về
    -----
    DoubleDQNAgent — Tác tử DQN đã nạp trọng số hoàn chỉnh.
    """
    n_w = env_config.get("n_warehouses", 2)
    n_s = env_config.get("n_skus", 30)
    n_pairs = n_w * n_s
    lookback = env_config.get("lookback", 7)
    lead_max = env_config.get("lead_time_max", 3)

    # Tính kích thước obs_dim theo đúng logic của MultiWarehouseInventoryEnv
    inv_dim = n_pairs
    dem_dim = n_pairs * lookback
    pip_dim = n_pairs * lead_max
    dow_dim = 7
    obs_dim = inv_dim + dem_dim + pip_dim + dow_dim

    order_levels = env_config.get("order_levels", [0, 10, 20, 30, 40, 50])
    n_action_levels = len(order_levels)
    hidden_dim = env_config.get("hidden_dim", 256)

    # Xác định device
    if device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = device

    agent = DoubleDQNAgent(
        state_dim=obs_dim,
        n_pairs=n_pairs,
        n_action_levels=n_action_levels,
        hidden_dim=hidden_dim,
        log_dir=str(PROJECT_ROOT / "demo_app" / "logs"),
        device=dev,
    )

    if os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
    else:
        raise FileNotFoundError(f"Không tìm thấy file checkpoint tại: {checkpoint_path}")

    return agent
