"""
demo_app/backend/inference.py
==============================
Hàm suy luận (Inference): Nhận quan sát trạng thái (State), tính toán hành động rời rạc
và chuyển đổi thành số lượng đặt hàng thực tế (units).
"""

from __future__ import annotations

import sys
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RL_INVENTORY_DIR = PROJECT_ROOT / "rl_inventory"

if str(RL_INVENTORY_DIR) not in sys.path:
    sys.path.insert(0, str(RL_INVENTORY_DIR))

from agents.dqn_agent import DoubleDQNAgent
from baselines.traditional_policies import (
    EOQPolicy,
    SsPolicyOptimized,
    NewsvendorPolicy,
    extract_state_for_policy,
)


def get_order_units_from_action(action_indices: np.ndarray, order_levels: list) -> np.ndarray:
    """
    Ánh xạ chỉ số hành động rời rạc sang số đơn vị đặt hàng thực tế.

    Tham số
    ------
    action_indices : np.ndarray, shape (n_pairs,)
        Chỉ số rời rạc trong [0 .. len(order_levels)-1]
    order_levels : list
        Danh sách mức đặt hàng (VD: [0, 10, 20, 30, 40, 50])

    Trả về
    -----
    np.ndarray — Số lượng đặt hàng tương ứng (đơn vị).
    """
    levels = np.array(order_levels, dtype=np.float32)
    return levels[action_indices]


def predict_dqn_orders(
    agent: DoubleDQNAgent,
    obs: np.ndarray,
    order_levels: list,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chạy inference với tác tử Double DQN theo chiến lược tham lam (greedy).

    Tham số
    ------
    agent : DoubleDQNAgent
    obs : np.ndarray, shape (obs_dim,)
    order_levels : list

    Trả về
    -----
    action_indices : np.ndarray, shape (n_pairs,)
    order_units    : np.ndarray, shape (n_pairs,)
    """
    action_indices = agent.select_action(obs, greedy=True)
    order_units = get_order_units_from_action(action_indices, order_levels)
    return action_indices, order_units


def predict_all_baseline_orders(
    baselines: Dict[str, Any],
    obs: np.ndarray,
    env_config: Dict[str, Any],
    order_levels: list,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Chạy inference cho tất cả các chiến lược baseline (EOQ, (s,S), Newsvendor).

    Tham số
    ------
    baselines : dict
        Dictionary chứa các đối tượng Policy instance
    obs : np.ndarray
    env_config : dict
    order_levels : list

    Trả về
    -----
    Dict[str, Tuple[action_indices, order_units]]
    """
    inventory, demand_hist = extract_state_for_policy(obs, env_config)
    results = {}

    for name, policy in baselines.items():
        action_indices = policy.get_action(inventory, demand_hist)
        order_units = get_order_units_from_action(action_indices, order_levels)
        results[name] = (action_indices, order_units)

    return results
