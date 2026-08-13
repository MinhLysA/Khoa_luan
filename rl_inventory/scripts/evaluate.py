"""
scripts/evaluate.py
====================
Kịch bản đánh giá: So sánh mô hình DQN với các baseline EOQ, (s,S) và Newsvendor.

Chạy tất cả các chiến lược trên cùng các tập thử nghiệm và tạo ra:
  1. Bảng tổng hợp chỉ số (tổng chi phí, mức độ phục vụ, tỷ lệ thiếu hàng cho mỗi chiến lược)
  2. Các biểu đồ so sánh Matplotlib/Seaborn (biểu đồ cột + đường cong phần thưởng)
  3. Tệp CSV lưu trữ các chỉ số đo lường

Cách sử dụng khuyến nghị (qua main.py):
  python main.py evaluate
  python main.py evaluate --checkpoint checkpoints/best_model.pth --synthetic

Hoặc chạy trực tiếp (tương thích ngược):
  python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
  python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --synthetic
"""

from __future__ import annotations

import os
import sys

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend không tương tác (hoạt động tốt trong môi trường headless)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
from pathlib import Path
from tqdm import tqdm

# Đảm bảo thư mục gốc project nằm trong sys.path (khi chạy trực tiếp)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.inventory_env import MultiWarehouseInventoryEnv
from agents.dqn_agent import DoubleDQNAgent
from baselines.traditional_policies import (
    EOQPolicy,
    SsPolicyOptimized,
    NewsvendorPolicy,
    extract_state_for_policy,
)
from utils import load_demand_data


# ---------------------------------------------------------------------------
# Phân tích tham số CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Đánh giá và so sánh DQN với các chiến lược baseline truyền thống."
    )
    parser.add_argument("--checkpoint",  type=str,
                        default=str(ROOT / "checkpoints" / "best_model.pth"),
                        help="Đường dẫn tới file checkpoint của mô hình DQN đã huấn luyện")
    parser.add_argument("--n_episodes",  type=int, default=20,
                        help="Số tập thử nghiệm cho mỗi chiến lược (mặc định: 20)")
    parser.add_argument("--data_dir",    type=str,
                        default=str(ROOT / "data" / "processed"),
                        help="Thư mục chứa dữ liệu đã xử lý")
    parser.add_argument("--output_dir",  type=str,
                        default=str(ROOT / "results"),
                        help="Thư mục xuất biểu đồ và tệp CSV kết quả")
    parser.add_argument("--synthetic",   action="store_true",
                        help="Sử dụng dữ liệu giả lập")
    parser.add_argument("--seed",        type=int, default=100,
                        help="Hạt giống khởi đầu cho các tập thử nghiệm (mặc định: 100)")
    parser.add_argument("--hidden_dim",  type=int, default=256,
                        help="Kích thước lớp ẩn (phải khớp với lúc huấn luyện)")
    return parser.parse_args()



# ---------------------------------------------------------------------------
# Trình chạy từng tập chiến lược
# ---------------------------------------------------------------------------

def run_dqn_episode(
    env: MultiWarehouseInventoryEnv,
    agent: DoubleDQNAgent,
    seed: int,
) -> dict:
    """Chạy 1 tập DQN tham lam (greedy) và thu thập các chỉ số."""
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    daily_rewards = []
    daily_inventory = []

    for _ in range(env.episode_len):
        action = agent.select_action(obs, greedy=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        daily_rewards.append(reward)
        daily_inventory.append(info["inventory_total"])

        if terminated or truncated:
            break

    return {
        "policy":          "Double DQN",
        "total_reward":    total_reward,
        "total_cost":      info["episode_cost"],
        "service_level":   env.get_service_level(),
        "total_stockout":  info["episode_stockouts"],
        "avg_inventory":   float(np.mean(daily_inventory)),
        "holding_cost":    info["episode_holding_cost"],
        "ordering_cost":   info["episode_ordering_cost"],
        "daily_rewards":   daily_rewards,
    }


def run_baseline_episode(
    env: MultiWarehouseInventoryEnv,
    policy,
    env_config: dict,
    policy_name: str,
    seed: int,
) -> dict:
    """Chạy 1 tập của chiến lược baseline truyền thống."""
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    daily_rewards = []
    daily_inventory = []

    for _ in range(env.episode_len):
        # Trích xuất tồn kho và lịch sử nhu cầu từ quan sát phẳng
        inventory, demand_hist = extract_state_for_policy(obs, env_config)

        # Lấy hành động từ chiến lược
        action = policy.get_action(inventory, demand_hist)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        daily_rewards.append(reward)
        daily_inventory.append(info["inventory_total"])

        if terminated or truncated:
            break

    return {
        "policy":         policy_name,
        "total_reward":   total_reward,
        "total_cost":     info["episode_cost"],
        "service_level":  env.get_service_level(),
        "total_stockout": info["episode_stockouts"],
        "avg_inventory":  float(np.mean(daily_inventory)),
        "holding_cost":   info["episode_holding_cost"],
        "ordering_cost":  info["episode_ordering_cost"],
        "daily_rewards":  daily_rewards,
    }


# ---------------------------------------------------------------------------
# Đánh giá chính
# ---------------------------------------------------------------------------

def evaluate(args):
    """Chạy đánh giá toàn bộ so sánh DQN vs 3 phương pháp baseline."""

    print("=" * 65)
    print("  Đánh giá So sánh: DQN vs EOQ vs (s,S) vs Newsvendor")
    print("=" * 65)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Môi trường ---------------------------------------------------------
    demand_data, env_config = load_demand_data(
        data_dir=args.data_dir,
        n_warehouses=2,
        n_skus=30,
        seed=42,
        synthetic=args.synthetic,
    )
    env = MultiWarehouseInventoryEnv(config=env_config, demand_data=demand_data)
    n_pairs = env.n_pairs
    obs_dim = env.obs_dim
    order_levels = env.order_levels.tolist()

    print(f"Môi trường: {env.n_w} nhà kho × {env.n_s} SKUs = {n_pairs} cặp kho-SKU")
    print(f"Số tập thử nghiệm: {args.n_episodes} tập mỗi chiến lược")

    # ---- Tải tác tử DQN ------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DoubleDQNAgent(
        state_dim       = obs_dim,
        n_pairs         = n_pairs,
        n_action_levels = len(order_levels),
        hidden_dim      = args.hidden_dim,
        log_dir         = os.path.join(args.output_dir, "tb_eval"),
        device          = device,
    )

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        agent.load(str(ckpt_path))
        print(f"Đã tải checkpoint DQN: {ckpt_path}")
    else:
        print(f"[CẢNH BÁO] Không tìm thấy checkpoint: {ckpt_path}")
        print("          DQN sẽ sử dụng trọng số ngẫu nhiên (chưa huấn luyện).")

    # ---- Các chiến lược Baseline --------------------------------------------
    baselines = {
        "EOQ":        EOQPolicy(
            n_pairs        = n_pairs,
            holding_cost   = env_config.get("holding_cost", 1.0),
            ordering_cost  = env_config.get("ordering_cost", 50.0),
            order_levels   = order_levels,
        ),
        "(s,S)":      SsPolicyOptimized(
            n_pairs        = n_pairs,
            holding_cost   = env_config.get("holding_cost", 1.0),
            ordering_cost  = env_config.get("ordering_cost", 50.0),
            order_levels   = order_levels,
        ),
        "Newsvendor": NewsvendorPolicy(
            n_pairs        = n_pairs,
            holding_cost   = env_config.get("holding_cost", 1.0),
            stockout_cost  = env_config.get("stockout_cost", 10.0),
            order_levels   = order_levels,
        ),
    }

    # ---- Thống kê thực thi --------------------------------------------------
    all_results = []
    all_daily_rewards = {}

    policies_to_run = ["Double DQN"] + list(baselines.keys())

    for pol_name in policies_to_run:
        print(f"\nĐang đánh giá: {pol_name}")
        pol_results = []
        daily_rew_list = []

        for ep in tqdm(range(args.n_episodes), desc=f"  {pol_name}", leave=False):
            seed = args.seed + ep

            if pol_name == "Double DQN":
                result = run_dqn_episode(env, agent, seed=seed)
            else:
                result = run_baseline_episode(
                    env, baselines[pol_name], env_config, pol_name, seed=seed
                )

            pol_results.append(result)
            daily_rew_list.append(result["daily_rewards"])

        all_results.extend(pol_results)
        all_daily_rewards[pol_name] = daily_rew_list

    agent.close()

    # ---- Tạo bảng tổng hợp chỉ số -------------------------------------------
    df = pd.DataFrame(all_results)
    df_summary = df.groupby("policy").agg(
        avg_total_cost    = ("total_cost",    "mean"),
        std_total_cost    = ("total_cost",    "std"),
        avg_service_level = ("service_level", "mean"),
        std_service_level = ("service_level", "std"),
        avg_stockout      = ("total_stockout","mean"),
        avg_reward        = ("total_reward",  "mean"),
        avg_holding_cost  = ("holding_cost",  "mean"),
        avg_ordering_cost = ("ordering_cost", "mean"),
        avg_inventory     = ("avg_inventory", "mean"),
    ).reset_index()

    # Sắp xếp danh sách chiến lược đẹp mắt
    order = ["Double DQN", "EOQ", "(s,S)", "Newsvendor"]
    df_summary["policy"] = pd.Categorical(df_summary["policy"], categories=order, ordered=True)
    df_summary = df_summary.sort_values("policy").reset_index(drop=True)

    print("\n" + "=" * 65)
    print("BẢNG TỔNG HỢP KẾT QUẢ")
    print("=" * 65)
    cols_show = ["policy", "avg_total_cost", "avg_service_level", "avg_stockout", "avg_reward"]
    print(df_summary[cols_show].to_string(index=False, float_format="{:.2f}".format))

    # Lưu tệp CSV
    csv_path = os.path.join(args.output_dir, "evaluation_results.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\nĐã lưu tệp kết quả: {csv_path}")

    # ---- Vẽ các biểu đồ so sánh ---------------------------------------------
    plot_comparison(df_summary, all_daily_rewards, args.output_dir)

    return df_summary, all_daily_rewards


# ---------------------------------------------------------------------------
# Trực quan hóa Biểu đồ (Plotting)
# ---------------------------------------------------------------------------

POLICY_COLORS = {
    "Double DQN": "#4C72B0",
    "EOQ":        "#DD8452",
    "(s,S)":      "#55A868",
    "Newsvendor": "#C44E52",
}


def plot_comparison(
    df_summary: pd.DataFrame,
    daily_rewards: dict,
    output_dir: str,
) -> None:
    """
    Tạo các biểu đồ so sánh:
      1. Biểu đồ cột: So sánh tổng chi phí
      2. Biểu đồ cột: So sánh mức độ phục vụ (Service level)
      3. Biểu đồ cột: Phân rã chi phí (lưu kho + đặt hàng)
      4. Biểu đồ đường: Đường cong phần thưởng hàng ngày (trung bình ± độ lệch chuẩn)
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)
    policies = df_summary["policy"].tolist()
    colors   = [POLICY_COLORS.get(p, "#888888") for p in policies]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        "So sánh RL và Các Phương pháp Baseline Truyền thống — Quản lý Tồn kho Đa Kho",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # --- (1) Tổng Chi Phí ----------------------------------------------------
    ax = axes[0, 0]
    bars = ax.bar(
        policies,
        df_summary["avg_total_cost"],
        yerr=df_summary["std_total_cost"],
        color=colors,
        capsize=5,
        edgecolor="white",
        linewidth=1.2,
    )
    ax.set_title("Chi phí Tồn kho Trung bình (thấp hơn là tốt hơn)", fontweight="bold")
    ax.set_ylabel("Chi phí trên mỗi Tập")
    ax.set_xlabel("")
    max_cost = df_summary["avg_total_cost"].max()
    ax.set_ylim(0, max_cost * 1.2 if max_cost > 0 else 1.0)
    ax.tick_params(axis="x", rotation=15)
    # Chú thích giá trị trên các cột
    for bar, val in zip(bars, df_summary["avg_total_cost"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{val:,.0f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # --- (2) Mức độ Phục vụ (Service Level) ----------------------------------
    ax = axes[0, 1]
    bars = ax.bar(
        policies,
        df_summary["avg_service_level"] * 100,
        yerr=df_summary["std_service_level"] * 100,
        color=colors,
        capsize=5,
        edgecolor="white",
        linewidth=1.2,
    )
    ax.set_title("Mức độ Phục vụ Trung bình % (cao hơn là tốt hơn)", fontweight="bold")
    ax.set_ylabel("Mức độ Phục vụ (%)")
    ax.set_ylim(0, 118)
    ax.axhline(95, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Mục tiêu 95%")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=15)
    for bar, val in zip(bars, df_summary["avg_service_level"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 100 + 0.5,
            f"{val*100:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # --- (3) Phân Rã Chi Phí -------------------------------------------------
    ax = axes[1, 0]
    x = np.arange(len(policies))
    w = 0.35
    hold_bars = ax.bar(
        x - w / 2,
        df_summary["avg_holding_cost"],
        width=w,
        label="Chi phí Lưu kho",
        color="#4C72B0",
        alpha=0.85,
    )
    order_bars = ax.bar(
        x + w / 2,
        df_summary["avg_ordering_cost"],
        width=w,
        label="Chi phí Đặt hàng",
        color="#DD8452",
        alpha=0.85,
    )
    ax.set_title("Phân rã Chi phí: Lưu kho vs Đặt hàng", fontweight="bold")
    ax.set_ylabel("Chi phí Trung bình trên mỗi Tập")
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=15)
    ax.legend(fontsize=9)

    # --- (4) Đường Cong Phần Thưởng Hàng Ngày ----------------                 
    ax = axes[1, 1]
    for pol_name, ep_list in daily_rewards.items():
        # Dóng hàng độ dài các tập
        min_len = min(len(ep) for ep in ep_list)
        arr = np.array([ep[:min_len] for ep in ep_list])
        mean_r = arr.mean(axis=0)
        std_r  = arr.std(axis=0)

        # Phần thưởng tích lũy theo tập
        cum = np.cumsum(mean_r)
        cum_std = np.cumsum(std_r)  # xấp xỉ

        color = POLICY_COLORS.get(pol_name, "#888888")
        ax.plot(cum, label=pol_name, color=color, linewidth=2)
        ax.fill_between(
            range(len(cum)),
            cum - cum_std,
            cum + cum_std,
            color=color,
            alpha=0.15,
        )

    ax.set_title("Phần thưởng Tích lũy mỗi Tập (trung bình ± std)", fontweight="bold")
    ax.set_xlabel("Ngày (trong tập)")
    ax.set_ylabel("Phần thưởng Tích lũy")
    ax.legend(fontsize=9)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.95, hspace=0.3, wspace=0.25)
    chart_path = os.path.join(output_dir, "comparison_chart.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Đã lưu biểu đồ: {chart_path}")

    # ---- Biểu đồ cột về lượng thiếu hàng (riêng biệt) ----------------------
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(
        policies,
        df_summary["avg_stockout"],
        color=colors,
        edgecolor="white",
        linewidth=1.2,
    )
    ax2.set_title(
        "Tổng số Đơn vị Thiếu hàng Trung bình mỗi Tập\n(thấp hơn là tốt hơn)",
        fontweight="bold",
    )
    ax2.set_ylabel("Số đơn vị thiếu hàng")
    ax2.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    stockout_path = os.path.join(output_dir, "stockout_comparison.png")
    plt.savefig(stockout_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Đã lưu biểu đồ thiếu hàng: {stockout_path}")


# ---------------------------------------------------------------------------
# Điểm vào chương trình
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    df_summary, daily_rewards = evaluate(args)

    print("\n" + "=" * 65)
    print("Đánh giá hoàn tất! Tất cả biểu đồ được lưu tại:", args.output_dir)
    print("=" * 65)
