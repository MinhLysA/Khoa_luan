"""
scripts/evaluate.py
====================
Evaluation script: compare DQN vs EOQ vs (s,S) vs Newsvendor.

Runs all policies on the same held-out test episodes and produces:
  1. Summary table (total cost, service level, stockout rate per policy)
  2. matplotlib comparison charts (bar charts + episode reward curves)
  3. Saved CSV of metrics

Usage:
  # After training:
  python scripts/evaluate.py --checkpoint checkpoints/best_model.pth

  # With synthetic data (no M5):
  python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --synthetic
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (works in headless environments)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.inventory_env import MultiWarehouseInventoryEnv, DEFAULT_CONFIG
from agents.dqn_agent import DoubleDQNAgent
from baselines.traditional_policies import (
    EOQPolicy,
    SsPolicyOptimized,
    NewsvendorPolicy,
    extract_state_for_policy,
)
from scripts.data_preprocessing import generate_synthetic_fallback


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare DQN vs baseline policies."
    )
    parser.add_argument("--checkpoint",  type=str,
                        default=str(ROOT / "checkpoints" / "best_model.pth"),
                        help="Path to trained DQN checkpoint")
    parser.add_argument("--n_episodes",  type=int, default=20,
                        help="Number of test episodes per policy (default: 20)")
    parser.add_argument("--data_dir",    type=str,
                        default=str(ROOT / "data" / "processed"),
                        help="Processed data directory")
    parser.add_argument("--output_dir",  type=str,
                        default=str(ROOT / "results"),
                        help="Directory for output charts and CSV")
    parser.add_argument("--synthetic",   action="store_true",
                        help="Use synthetic data")
    parser.add_argument("--seed",        type=int, default=100,
                        help="Starting seed for test episodes (default: 100)")
    parser.add_argument("--hidden_dim",  type=int, default=256,
                        help="Must match training hidden_dim")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data & env setup
# ---------------------------------------------------------------------------

def load_env(args) -> tuple:
    """Load environment with real or synthetic demand data."""
    data_dir = Path(args.data_dir)
    np_path  = data_dir / "demand_data.npy"
    cfg_path = data_dir / "env_config.json"

    if not args.synthetic and np_path.exists():
        demand_data = np.load(str(np_path))
        with open(cfg_path) as f:
            env_config = json.load(f)
    else:
        demand_data = generate_synthetic_fallback(n_warehouses=2, n_skus=30, seed=42)
        env_config = {**DEFAULT_CONFIG, "n_warehouses": 2, "n_skus": 30}

    env = MultiWarehouseInventoryEnv(config=env_config, demand_data=demand_data)
    return env, env_config


# ---------------------------------------------------------------------------
# Policy runners
# ---------------------------------------------------------------------------

def run_dqn_episode(
    env: MultiWarehouseInventoryEnv,
    agent: DoubleDQNAgent,
    seed: int,
) -> dict:
    """Run one greedy DQN episode and collect metrics."""
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    daily_rewards = []
    daily_stockouts = []
    daily_inventory = []

    for _ in range(env.episode_len):
        action = agent.select_action(obs, greedy=True)
        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        daily_rewards.append(reward)

        if "stockout" in step_info:
            daily_stockouts.append(float(np.sum(step_info["stockout"])))
        daily_inventory.append(step_info["inventory_total"])

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
    """Run one episode of a traditional baseline policy."""
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    daily_rewards = []
    daily_inventory = []

    for _ in range(env.episode_len):
        # Extract inventory and demand history from flat obs
        inventory, demand_hist = extract_state_for_policy(obs, env_config)

        # Get policy action
        action = policy.get_action(inventory, demand_hist)

        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        daily_rewards.append(reward)
        daily_inventory.append(step_info["inventory_total"])

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
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(args):
    """Run full evaluation comparing DQN vs 3 baselines."""

    print("=" * 65)
    print("  Evaluation: DQN vs EOQ vs (s,S) vs Newsvendor")
    print("=" * 65)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Environment --------------------------------------------------------
    env, env_config = load_env(args)
    n_pairs = env.n_pairs
    obs_dim = env.obs_dim
    order_levels = env.order_levels.tolist()

    print(f"Environment: {env.n_w} warehouses × {env.n_s} SKUs = {n_pairs} pairs")
    print(f"Test episodes: {args.n_episodes} per policy")

    # ---- Load DQN agent -----------------------------------------------------
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
        print(f"Loaded DQN checkpoint: {ckpt_path}")
    else:
        print(f"[WARNING] Checkpoint not found: {ckpt_path}")
        print("         DQN will use random weights (untrained agent).")

    # ---- Baseline policies ---------------------------------------------------
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

    # ---- Run evaluation episodes --------------------------------------------
    all_results = []
    all_daily_rewards = {}

    policies_to_run = ["Double DQN"] + list(baselines.keys())

    for pol_name in policies_to_run:
        print(f"\nEvaluating: {pol_name}")
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

    # ---- Build summary DataFrame -------------------------------------------
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

    # Order policies nicely
    order = ["Double DQN", "EOQ", "(s,S)", "Newsvendor"]
    df_summary["policy"] = pd.Categorical(df_summary["policy"], categories=order, ordered=True)
    df_summary = df_summary.sort_values("policy").reset_index(drop=True)

    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    cols_show = ["policy", "avg_total_cost", "avg_service_level", "avg_stockout", "avg_reward"]
    print(df_summary[cols_show].to_string(index=False, float_format="{:.2f}".format))

    # Save CSV
    csv_path = os.path.join(args.output_dir, "evaluation_results.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")

    # ---- Generate plots -----------------------------------------------------
    plot_comparison(df_summary, all_daily_rewards, args.output_dir)

    return df_summary, all_daily_rewards


# ---------------------------------------------------------------------------
# Plotting
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
    Generate comparison charts:
      1. Bar chart: Total cost comparison
      2. Bar chart: Service level comparison
      3. Bar chart: Cost breakdown (holding + ordering)
      4. Line chart: Daily reward curves (mean ± std) per policy
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)
    policies = df_summary["policy"].tolist()
    colors   = [POLICY_COLORS.get(p, "#888888") for p in policies]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        "RL vs Traditional Baselines — Multi-Warehouse Inventory Management",
        fontsize=15, fontweight="bold", y=1.01,
    )

    # --- (1) Total Cost ------------------------------------------------------
    ax = axes[0, 0]
    bars = ax.bar(
        policies,
        df_summary["avg_total_cost"],
        yerr=df_summary["std_total_cost"],
        color=colors,
        capsize=6,
        edgecolor="white",
        linewidth=1.2,
    )
    ax.set_title("Average Total Inventory Cost (lower is better)", fontweight="bold")
    ax.set_ylabel("Cost per Episode")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    # Annotate bars
    for bar, val in zip(bars, df_summary["avg_total_cost"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + df_summary["std_total_cost"].max() * 0.05,
            f"{val:,.0f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # --- (2) Service Level ---------------------------------------------------
    ax = axes[0, 1]
    bars = ax.bar(
        policies,
        df_summary["avg_service_level"] * 100,
        yerr=df_summary["std_service_level"] * 100,
        color=colors,
        capsize=6,
        edgecolor="white",
        linewidth=1.2,
    )
    ax.set_title("Average Service Level % (higher is better)", fontweight="bold")
    ax.set_ylabel("Service Level (%)")
    ax.set_ylim(0, 115)
    ax.axhline(95, color="red", linestyle="--", linewidth=1, alpha=0.7, label="95% target")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=15)
    for bar, val in zip(bars, df_summary["avg_service_level"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 100 + 0.5,
            f"{val*100:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # --- (3) Cost Breakdown --------------------------------------------------
    ax = axes[1, 0]
    x = np.arange(len(policies))
    w = 0.35
    hold_bars = ax.bar(
        x - w / 2,
        df_summary["avg_holding_cost"],
        width=w,
        label="Holding Cost",
        color="#4C72B0",
        alpha=0.85,
    )
    order_bars = ax.bar(
        x + w / 2,
        df_summary["avg_ordering_cost"],
        width=w,
        label="Ordering Cost",
        color="#DD8452",
        alpha=0.85,
    )
    ax.set_title("Cost Breakdown: Holding vs Ordering", fontweight="bold")
    ax.set_ylabel("Average Cost per Episode")
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=15)
    ax.legend(fontsize=9)

    # --- (4) Daily Reward Curves ---------------------------------------------
    ax = axes[1, 1]
    for pol_name, ep_list in daily_rewards.items():
        # Align episode lengths
        min_len = min(len(ep) for ep in ep_list)
        arr = np.array([ep[:min_len] for ep in ep_list])
        mean_r = arr.mean(axis=0)
        std_r  = arr.std(axis=0)

        # Cumulative reward per episode
        cum = np.cumsum(mean_r)
        cum_std = np.cumsum(std_r)  # approximate

        color = POLICY_COLORS.get(pol_name, "#888888")
        ax.plot(cum, label=pol_name, color=color, linewidth=2)
        ax.fill_between(
            range(len(cum)),
            cum - cum_std,
            cum + cum_std,
            color=color,
            alpha=0.15,
        )

    ax.set_title("Cumulative Reward per Episode (mean ± std)", fontweight="bold")
    ax.set_xlabel("Day (within episode)")
    ax.set_ylabel("Cumulative Reward")
    ax.legend(fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "comparison_chart.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved: {chart_path}")

    # ---- Stockout rate bar chart (separate) ----------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(
        policies,
        df_summary["avg_stockout"],
        color=colors,
        edgecolor="white",
        linewidth=1.2,
    )
    ax2.set_title(
        "Average Total Stockout Units per Episode\n(lower is better)",
        fontweight="bold",
    )
    ax2.set_ylabel("Stockout Units")
    ax2.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    stockout_path = os.path.join(output_dir, "stockout_comparison.png")
    plt.savefig(stockout_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Stockout chart saved: {stockout_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    df_summary, daily_rewards = evaluate(args)

    print("\n" + "=" * 65)
    print("Evaluation complete! Charts saved to:", args.output_dir)
    print("=" * 65)
