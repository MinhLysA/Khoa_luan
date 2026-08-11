"""
scripts/train.py
=================
Training script for Double DQN agent on Multi-Warehouse Inventory problem.

Usage (local GPU — RTX 3050):
  python scripts/train.py --episodes 500 --use_m5

Usage (synthetic data — quick test):
  python scripts/train.py --episodes 50 --synthetic

TensorBoard monitoring:
  tensorboard --logdir runs/

GPU memory usage (RTX 3050 4GB):
  - Model: ~2MB
  - Replay buffer (100K): ~500MB
  - Batch (128): ~1MB
  Total: ~0.5GB — well within 4GB VRAM limit
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.inventory_env import MultiWarehouseInventoryEnv, DEFAULT_CONFIG
from agents.dqn_agent import DoubleDQNAgent
from scripts.data_preprocessing import generate_synthetic_fallback


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Double DQN for multi-warehouse inventory management."
    )
    parser.add_argument("--episodes",      type=int,   default=500,
                        help="Number of training episodes (default: 500)")
    parser.add_argument("--hidden_dim",    type=int,   default=256,
                        help="Hidden layer size (default: 256)")
    parser.add_argument("--batch_size",    type=int,   default=128,
                        help="Training batch size (default: 128, RTX 3050 optimized)")
    parser.add_argument("--lr",            type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--gamma",         type=float, default=0.99,
                        help="Discount factor (default: 0.99)")
    parser.add_argument("--buffer_cap",    type=int,   default=100_000,
                        help="Replay buffer capacity (default: 100000)")
    parser.add_argument("--eps_start",     type=float, default=1.0,
                        help="Initial epsilon (default: 1.0)")
    parser.add_argument("--eps_min",       type=float, default=0.05,
                        help="Minimum epsilon (default: 0.05)")
    parser.add_argument("--eps_decay",     type=int,   default=50_000,
                        help="Epsilon decay steps (default: 50000)")
    parser.add_argument("--n_skus",        type=int,   default=30,
                        help="Number of SKUs per warehouse (default: 30)")
    parser.add_argument("--use_per",       action="store_true",
                        help="Use Prioritized Experience Replay")
    parser.add_argument("--synthetic",     action="store_true",
                        help="Force synthetic data (ignore M5)")
    parser.add_argument("--data_dir",      type=str,   default=str(ROOT / "data" / "processed"),
                        help="Processed data directory")
    parser.add_argument("--log_dir",       type=str,   default=str(ROOT / "runs"),
                        help="TensorBoard log directory")
    parser.add_argument("--checkpoint_dir", type=str,  default=str(ROOT / "checkpoints"),
                        help="Checkpoint save directory")
    parser.add_argument("--save_every",    type=int,   default=100,
                        help="Save checkpoint every N episodes (default: 100)")
    parser.add_argument("--eval_every",    type=int,   default=50,
                        help="Run evaluation episode every N episodes (default: 50)")
    parser.add_argument("--seed",          type=int,   default=42,
                        help="Random seed (default: 42)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_demand_data(data_dir: str, args) -> tuple:
    """
    Load demand data and env config from processed directory.

    Falls back to synthetic data if files not found.

    Returns
    -------
    demand_data : np.ndarray, shape (T, n_w, n_s)
    env_config : dict
    """
    data_dir = Path(data_dir)
    np_path  = data_dir / "demand_data.npy"
    cfg_path = data_dir / "env_config.json"

    if not args.synthetic and np_path.exists() and cfg_path.exists():
        print(f"[Train] Loading real demand data from {data_dir}")
        demand_data = np.load(str(np_path))
        with open(cfg_path) as f:
            env_config = json.load(f)
        print(f"  demand_data shape: {demand_data.shape}")
    else:
        print("[Train] Using synthetic demand data (run data_preprocessing.py for real M5 data)")
        demand_data = generate_synthetic_fallback(
            n_warehouses=2,
            n_skus=args.n_skus,
            n_days=800,
            seed=args.seed,
        )
        env_config = {**DEFAULT_CONFIG, "n_warehouses": 2, "n_skus": args.n_skus}

    return demand_data, env_config


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    """Main training loop for Double DQN inventory agent."""

    print("=" * 65)
    print("  Double DQN — Multi-Warehouse Inventory Optimization")
    print("=" * 65)

    # ---- Data & Environment ------------------------------------------------
    demand_data, env_config = load_demand_data(args.data_dir, args)

    # Override config with CLI args
    env_config["seed"] = args.seed

    env = MultiWarehouseInventoryEnv(
        config=env_config,
        demand_data=demand_data,
    )

    n_pairs = env.n_pairs
    obs_dim = env.obs_dim

    print(f"\nEnvironment:")
    print(f"  Warehouses:       {env.n_w}")
    print(f"  SKUs/warehouse:   {env.n_s}")
    print(f"  Pairs (n_pairs):  {n_pairs}")
    print(f"  Obs dim:          {obs_dim}")
    print(f"  Episode length:   {env.episode_len} days")
    print(f"  Action levels:    {env.order_levels.tolist()}")

    # ---- Agent ---------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"\n[GPU] {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"dqn_{timestamp}")

    agent = DoubleDQNAgent(
        state_dim       = obs_dim,
        n_pairs         = n_pairs,
        n_action_levels = len(env.order_levels),
        hidden_dim      = args.hidden_dim,
        lr              = args.lr,
        gamma           = args.gamma,
        buffer_capacity = args.buffer_cap,
        batch_size      = args.batch_size,
        eps_start       = args.eps_start,
        eps_min         = args.eps_min,
        eps_decay       = args.eps_decay,
        use_per         = args.use_per,
        log_dir         = log_dir,
        device          = device,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ---- Tracking -----------------------------------------------------------
    best_reward = -float("inf")
    episode_rewards  = []
    episode_costs    = []
    episode_sl       = []   # service levels
    losses           = []

    print(f"\n[Train] Starting training for {args.episodes} episodes...")
    print(f"  Log dir:   {log_dir}")
    print(f"  Ckpt dir:  {args.checkpoint_dir}")
    print(f"  PER:       {args.use_per}")
    print("=" * 65)

    pbar = tqdm(range(1, args.episodes + 1), desc="Training", unit="ep")

    for episode in pbar:
        obs, info = env.reset(seed=args.seed + episode)
        ep_reward = 0.0
        ep_loss   = []

        # ---- Episode loop ---------------------------------------------------
        for step in range(env.episode_len):
            # ε-greedy action selection
            action = agent.select_action(obs, greedy=False)

            # Environment step
            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            # Store transition
            agent.store_transition(obs, action, reward, next_obs, done)
            ep_reward += reward
            obs = next_obs

            # Learning step
            loss = agent.update()
            if loss is not None:
                ep_loss.append(loss)
                losses.append(loss)

            if done:
                break

        # ---- Episode bookkeeping --------------------------------------------
        service_level = env.get_service_level()
        ep_cost = info["episode_cost"]
        total_demand = env.demand_data[
            env.start_idx:env.start_idx + env.t
        ].sum()
        stockout_rate = (
            info["episode_stockouts"] / (total_demand + 1e-6)
        )

        episode_rewards.append(ep_reward)
        episode_costs.append(ep_cost)
        episode_sl.append(service_level)

        avg_loss = np.mean(ep_loss) if ep_loss else 0.0

        # Log to TensorBoard
        agent.log_episode(ep_reward, ep_cost, service_level, stockout_rate)

        # Progress bar update
        pbar.set_postfix({
            "rew":   f"{ep_reward:.0f}",
            "cost":  f"{ep_cost:.0f}",
            "SL":    f"{service_level:.3f}",
            "ε":     f"{agent.epsilon:.3f}",
            "loss":  f"{avg_loss:.4f}",
        })

        # ---- Periodic evaluation (greedy policy) ----------------------------
        if episode % args.eval_every == 0:
            eval_reward, eval_sl = evaluate_episode(env, agent)
            agent.writer.add_scalar("eval/reward",        eval_reward, episode)
            agent.writer.add_scalar("eval/service_level", eval_sl,     episode)

            tqdm.write(
                f"\n[Eval ep={episode:4d}] reward={eval_reward:.1f}  "
                f"service_level={eval_sl:.4f}  ε={agent.epsilon:.3f}"
            )

        # ---- Save best checkpoint -------------------------------------------
        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(os.path.join(args.checkpoint_dir, "best_model.pth"))

        # ---- Periodic checkpoint --------------------------------------------
        if episode % args.save_every == 0:
            agent.save(
                os.path.join(args.checkpoint_dir, f"ckpt_ep{episode:04d}.pth")
            )

    # ---- Training summary ---------------------------------------------------
    agent.save(os.path.join(args.checkpoint_dir, "final_model.pth"))
    agent.close()

    # Print final stats
    last_100 = slice(-min(100, args.episodes), None)
    print("\n" + "=" * 65)
    print("Training Complete!")
    print(f"  Best episode reward:         {best_reward:.2f}")
    print(f"  Last 100 episodes avg reward: {np.mean(episode_rewards[last_100]):.2f}")
    print(f"  Last 100 episodes avg cost:   {np.mean(episode_costs[last_100]):.2f}")
    print(f"  Last 100 episodes avg SL:     {np.mean(episode_sl[last_100]):.4f}")
    print(f"  Total gradient updates:       {agent.update_count}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print("=" * 65)

    # Save training curves to numpy for plotting
    np.save(
        os.path.join(args.checkpoint_dir, "training_rewards.npy"),
        np.array(episode_rewards)
    )
    np.save(
        os.path.join(args.checkpoint_dir, "training_service_levels.npy"),
        np.array(episode_sl)
    )

    return episode_rewards, episode_sl


def evaluate_episode(
    env: MultiWarehouseInventoryEnv,
    agent: DoubleDQNAgent,
    seed: int = 9999,
) -> tuple:
    """
    Run one greedy evaluation episode (no exploration).

    Parameters
    ----------
    env : MultiWarehouseInventoryEnv
    agent : DoubleDQNAgent
    seed : int

    Returns
    -------
    total_reward : float
    service_level : float
    """
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0

    for _ in range(env.episode_len):
        action = agent.select_action(obs, greedy=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    return total_reward, env.get_service_level()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    train(args)
