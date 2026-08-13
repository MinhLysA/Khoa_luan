"""
demo_app/backend/env_wrapper.py
===============================
Môi trường Mô phỏng (Simulation Wrapper): Quản lý vòng lặp chạy mô phỏng tập (episode)
cho cả Double DQN lẫn các phương pháp baseline truyền thống (EOQ, (s,S), Newsvendor)
trên cùng một chuỗi nhu cầu để so sánh công bằng.
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RL_INVENTORY_DIR = PROJECT_ROOT / "rl_inventory"

if str(RL_INVENTORY_DIR) not in sys.path:
    sys.path.insert(0, str(RL_INVENTORY_DIR))

from env.inventory_env import MultiWarehouseInventoryEnv, DEFAULT_CONFIG
from agents.dqn_agent import DoubleDQNAgent
from baselines.traditional_policies import (
    EOQPolicy,
    SsPolicyOptimized,
    NewsvendorPolicy,
    extract_state_for_policy,
)
from utils import load_demand_data


from backend.excel_adapter import RealDataMasterAdapter


class SimulationEngine:
    """
    Quản lý việc tạo môi trường và thực thi mô phỏng so sánh nhiều chiến lược.
    Hỗ trợ 3 nguồn dữ liệu: Synthetic, M5 Walmart, và Excel Master Data Nhà máy.
    """

    def __init__(
        self,
        synthetic: bool = True,
        data_source_type: str = "synthetic",
        excel_source: Optional[Any] = None,
        seed: int = 42,
        n_warehouses: int = 2,
        n_skus: int = 30,
        episode_length: int = 112,
        holding_cost: float = 1.0,
        stockout_cost: float = 10.0,
        ordering_cost: float = 50.0,
    ):
        self.synthetic = synthetic
        self.data_source_type = data_source_type
        self.excel_source = excel_source
        self.seed = seed
        self.n_w = n_warehouses
        self.n_s = n_skus
        self.episode_length = episode_length
        self.sku_metadata_df: Optional[pd.DataFrame] = None
        self.cleaning_stats: Dict[str, Any] = {}

        # 1. Tải dữ liệu nhu cầu theo nguồn chỉ định
        if self.data_source_type == "excel" and self.excel_source is not None:
            adapter = RealDataMasterAdapter(self.excel_source)
            self.demand_data, self.sku_metadata_df, self.cleaning_stats = adapter.build_simulation_dataset(
                n_warehouses=self.n_w,
                n_skus=self.n_s,
                n_days=800,
                seed=self.seed,
            )
            loaded_cfg = {}
        else:
            is_synth = (self.data_source_type == "synthetic") if self.data_source_type else self.synthetic
            data_dir = str(RL_INVENTORY_DIR / "data" / "processed")
            self.demand_data, loaded_cfg = load_demand_data(
                data_dir=data_dir,
                n_warehouses=self.n_w,
                n_skus=self.n_s,
                seed=self.seed,
                synthetic=is_synth,
            )

        # Xây dựng cấu hình môi trường
        self.env_config = {
            **DEFAULT_CONFIG,
            **loaded_cfg,
            "n_warehouses": self.n_w,
            "n_skus": self.n_s,
            "episode_length": self.episode_length,
            "holding_cost": holding_cost,
            "stockout_cost": stockout_cost,
            "ordering_cost": ordering_cost,
        }

        # Khởi tạo môi trường
        self.env = MultiWarehouseInventoryEnv(
            config=self.env_config,
            demand_data=self.demand_data,
        )

        # Khởi tạo các baseline
        self.n_pairs = self.env.n_pairs
        order_levels = self.env_config.get("order_levels", [0, 10, 20, 30, 40, 50])

        self.baselines = {
            "EOQ": EOQPolicy(
                n_pairs=self.n_pairs,
                holding_cost=holding_cost,
                ordering_cost=ordering_cost,
                order_levels=order_levels,
            ),
            "(s,S)": SsPolicyOptimized(
                n_pairs=self.n_pairs,
                holding_cost=holding_cost,
                ordering_cost=ordering_cost,
                order_levels=order_levels,
            ),
            "Newsvendor": NewsvendorPolicy(
                n_pairs=self.n_pairs,
                holding_cost=holding_cost,
                stockout_cost=stockout_cost,
                order_levels=order_levels,
            ),
        }

    def run_simulation(
        self,
        agent: DoubleDQNAgent,
        sim_seed: int = 100,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Dict[str, Any]:
        """
        Chạy 1 tập mô phỏng đầy đủ cho tất cả chiến lược (DQN + 3 Baselines)
        trên cùng seed khởi đầu và ghi lại chi tiết từng ngày.

        Tham số
        ------
        agent : DoubleDQNAgent
        sim_seed : int — Hạt giống ngẫu nhiên cho tập mô phỏng
        progress_callback : Callable — Hàm callback cập nhật tiến độ UI (0.0 đến 1.0)

        Trả về
        -----
        dict chứa:
          - 'daily_data': Dict[policy_name, List[step_dict]]
          - 'summary': Dict[policy_name, summary_dict]
          - 'env_config': dict
        """
        policies_to_run = ["Double DQN"] + list(self.baselines.keys())
        total_steps_all = len(policies_to_run) * self.episode_length
        current_step_count = 0

        daily_data = {}
        summary = {}

        for pol_idx, pol_name in enumerate(policies_to_run):
            obs, info = self.env.reset(seed=sim_seed)
            step_history = []
            daily_rewards = []
            daily_inventory = []

            for t in range(self.episode_length):
                # Lấy hành động
                if pol_name == "Double DQN":
                    action_idx = agent.select_action(obs, greedy=True)
                else:
                    inv_flat, dem_hist = extract_state_for_policy(obs, self.env_config)
                    action_idx = self.baselines[pol_name].get_action(inv_flat, dem_hist)

                # Chuyển đổi action index sang số lượng đơn vị đặt hàng thực tế
                order_units_flat = self.env.order_levels[action_idx]
                order_matrix = order_units_flat.reshape(self.n_w, self.n_s)

                # Lưu lại trạng thái tồn kho trước bước step
                current_inv_matrix = self.env.inventory.copy()

                # Bước môi trường
                obs, reward, terminated, truncated, step_info = self.env.step(action_idx)
                daily_rewards.append(reward)
                daily_inventory.append(step_info["inventory_total"])

                # Ghi lại dữ liệu từng ngày
                step_record = {
                    "day": t + 1,
                    "inventory_matrix": current_inv_matrix,     # shape (n_w, n_s)
                    "order_matrix": order_matrix,               # shape (n_w, n_s)
                    "demand_matrix": step_info["demand"].copy(), # shape (n_w, n_s)
                    "stockout_matrix": step_info["stockout"].copy(), # shape (n_w, n_s)
                    "inventory_total": step_info["inventory_total"],
                    "inventory_mean": step_info["inventory_mean"],
                    "order_total": float(order_matrix.sum()),
                    "demand_total": float(step_info["demand"].sum()),
                    "stockout_total": float(step_info["stockout"].sum()),
                    "holding_cost": step_info["holding_cost"],
                    "ordering_cost": step_info["ordering_cost"],
                    "stockout_cost": step_info["stockout_cost"],
                    "cost": step_info["holding_cost"] + step_info["ordering_cost"] + step_info["stockout_cost"],
                    "reward": reward,
                }
                step_history.append(step_record)

                current_step_count += 1
                if progress_callback:
                    progress_callback(min(1.0, current_step_count / total_steps_all))

                if terminated or truncated:
                    break

            # Tính toán summary chỉ số cho chiến lược
            daily_data[pol_name] = step_history
            summary[pol_name] = {
                "policy": pol_name,
                "total_cost": step_info["episode_cost"],
                "total_reward": float(np.sum(daily_rewards)),
                "holding_cost": step_info["episode_holding_cost"],
                "ordering_cost": step_info["episode_ordering_cost"],
                "stockout_cost": step_info["episode_cost"] - step_info["episode_holding_cost"] - step_info["episode_ordering_cost"],
                "total_stockout": step_info["episode_stockouts"],
                "service_level": self.env.get_service_level(),
                "avg_inventory": float(np.mean(daily_inventory)),
            }

        return {
            "daily_data": daily_data,
            "summary": summary,
            "env_config": self.env_config,
            "sku_metadata_df": self.sku_metadata_df,
            "cleaning_stats": self.cleaning_stats,
        }
