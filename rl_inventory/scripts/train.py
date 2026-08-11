"""
scripts/train.py
=================
Kịch bản huấn luyện tác tử Double DQN cho bài toán Quản lý Tồn kho Đa Kho.

Cách sử dụng (GPU cục bộ — RTX 3050):
  python scripts/train.py --episodes 500 --use_m5

Cách sử dụng (Dữ liệu giả lập — thử nghiệm nhanh):
  python scripts/train.py --episodes 50 --synthetic

Theo dõi TensorBoard:
  tensorboard --logdir runs/

Sử dụng bộ nhớ GPU (RTX 3050 4GB):
  - Mô hình: ~2MB
  - Bộ đệm phát lại (100K): ~500MB
  - Batch (128): ~1MB
  Tổng cộng: ~0.5GB — hoàn toàn nằm trong giới hạn VRAM 4GB
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

# Đảm bảo đường dẫn thư mục gốc dự án nằm trong sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.inventory_env import MultiWarehouseInventoryEnv, DEFAULT_CONFIG
from agents.dqn_agent import DoubleDQNAgent
from scripts.data_preprocessing import generate_synthetic_fallback


# ---------------------------------------------------------------------------
# Phân tích tham số dòng lệnh CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Huấn luyện Double DQN cho bài toán tối ưu hóa quản lý tồn kho đa nhà kho."
    )
    parser.add_argument("--episodes",      type=int,   default=500,
                        help="Số lượng tập huấn luyện (mặc định: 500)")
    parser.add_argument("--hidden_dim",    type=int,   default=256,
                        help="Kích thước lớp ẩn (mặc định: 256)")
    parser.add_argument("--batch_size",    type=int,   default=128,
                        help="Kích thước lô huấn luyện (mặc định: 128, tối ưu cho RTX 3050)")
    parser.add_argument("--lr",            type=float, default=3e-4,
                        help="Tốc độ học (mặc định: 3e-4)")
    parser.add_argument("--gamma",         type=float, default=0.99,
                        help="Hệ số chiết khấu gamma (mặc định: 0.99)")
    parser.add_argument("--buffer_cap",    type=int,   default=100_000,
                        help="Dung lượng bộ đệm phát lại (mặc định: 100000)")
    parser.add_argument("--eps_start",     type=float, default=1.0,
                        help="Tỷ lệ khám phá epsilon ban đầu (mặc định: 1.0)")
    parser.add_argument("--eps_min",       type=float, default=0.05,
                        help="Tỷ lệ khám phá epsilon tối thiểu (mặc định: 0.05)")
    parser.add_argument("--eps_decay",     type=int,   default=50_000,
                        help="Số bước suy giảm epsilon (mặc định: 50000)")
    parser.add_argument("--n_skus",        type=int,   default=30,
                        help="Số lượng SKU trên mỗi nhà kho (mặc định: 30)")
    parser.add_argument("--use_per",       action="store_true",
                        help="Sử dụng Prioritized Experience Replay")
    parser.add_argument("--synthetic",     action="store_true",
                        help="Bắt buộc sử dụng dữ liệu giả lập (bỏ qua M5)")
    parser.add_argument("--data_dir",      type=str,   default=str(ROOT / "data" / "processed"),
                        help="Thư mục chứa dữ liệu đã xử lý")
    parser.add_argument("--log_dir",       type=str,   default=str(ROOT / "runs"),
                        help="Thư mục lưu log TensorBoard")
    parser.add_argument("--checkpoint_dir", type=str,  default=str(ROOT / "checkpoints"),
                        help="Thư mục lưu điểm kiểm tra checkpoint")
    parser.add_argument("--save_every",    type=int,   default=100,
                        help="Lưu checkpoint sau mỗi N tập (mặc định: 100)")
    parser.add_argument("--eval_every",    type=int,   default=50,
                        help="Chạy tập đánh giá sau mỗi N tập (mặc định: 50)")
    parser.add_argument("--seed",          type=int,   default=42,
                        help="Hạt giống ngẫu nhiên (mặc định: 42)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Tải dữ liệu
# ---------------------------------------------------------------------------

def load_demand_data(data_dir: str, args) -> tuple:
    """
    Tải dữ liệu nhu cầu và cấu hình môi trường từ thư mục đã xử lý.

    Chuyển sang dùng dữ liệu giả lập nếu không tìm thấy tệp.

    Trả về
    -----
    demand_data : np.ndarray, shape (T, n_w, n_s)
    env_config : dict
    """
    data_dir = Path(data_dir)
    np_path  = data_dir / "demand_data.npy"
    cfg_path = data_dir / "env_config.json"

    if not args.synthetic and np_path.exists() and cfg_path.exists():
        print(f"[Train] Đang tải dữ liệu nhu cầu thực tế từ {data_dir}")
        demand_data = np.load(str(np_path))
        with open(cfg_path) as f:
            env_config = json.load(f)
        print(f"  Kích thước demand_data: {demand_data.shape}")
    else:
        print("[Train] Sử dụng dữ liệu nhu cầu giả lập (chạy data_preprocessing.py để dùng M5 thực tế)")
        demand_data = generate_synthetic_fallback(
            n_warehouses=2,
            n_skus=args.n_skus,
            n_days=800,
            seed=args.seed,
        )
        env_config = {**DEFAULT_CONFIG, "n_warehouses": 2, "n_skus": args.n_skus}

    return demand_data, env_config


# ---------------------------------------------------------------------------
# Vòng lặp huấn luyện chính
# ---------------------------------------------------------------------------

def train(args):
    """Vòng lặp huấn luyện chính cho tác tử tồn kho Double DQN."""

    print("=" * 65)
    print("  Double DQN — Tối ưu hóa Quản lý Tồn kho Đa Kho")
    print("=" * 65)

    # ---- Dữ liệu & Môi trường -----------------------------------------------
    demand_data, env_config = load_demand_data(args.data_dir, args)

    # Ghi đè cấu hình từ tham số CLI
    env_config["seed"] = args.seed

    env = MultiWarehouseInventoryEnv(
        config=env_config,
        demand_data=demand_data,
    )

    n_pairs = env.n_pairs
    obs_dim = env.obs_dim

    print(f"\nMôi trường:")
    print(f"  Số nhà kho:               {env.n_w}")
    print(f"  SKU / nhà kho:            {env.n_s}")
    print(f"  Số cặp kho-SKU (n_pairs): {n_pairs}")
    print(f"  Chiều quan sát (obs_dim): {obs_dim}")
    print(f"  Độ dài tập (episode):     {env.episode_len} ngày")
    print(f"  Các mức đặt hàng:         {env.order_levels.tolist()}")

    # ---- Tác tử Agent --------------------------------------------------------
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

    # ---- Theo dõi chỉ số ----------------------------------------------------
    best_reward = -float("inf")
    episode_rewards  = []
    episode_costs    = []
    episode_sl       = []   # service levels (mức độ phục vụ)
    losses           = []

    print(f"\n[Train] Bắt đầu huấn luyện trong {args.episodes} tập...")
    print(f"  Thư mục Log:        {log_dir}")
    print(f"  Thư mục Checkpoint: {args.checkpoint_dir}")
    print(f"  Sử dụng PER:        {args.use_per}")
    print("=" * 65)

    pbar = tqdm(range(1, args.episodes + 1), desc="Đang huấn luyện", unit="ep")

    for episode in pbar:
        obs, info = env.reset(seed=args.seed + episode)
        ep_reward = 0.0
        ep_loss   = []

        # ---- Vòng lặp tập (Episode loop) -----------------------------------
        for step in range(env.episode_len):
            # Lựa chọn hành động ε-greedy
            action = agent.select_action(obs, greedy=False)

            # Bước môi trường
            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            # Lưu chuyển trạng thái
            agent.store_transition(obs, action, reward, next_obs, done)
            ep_reward += reward
            obs = next_obs

            # Bước cập nhật mô hình
            loss = agent.update()
            if loss is not None:
                ep_loss.append(loss)
                losses.append(loss)

            if done:
                break

        # ---- Ghi nhận kết quả tập -------------------------------------------
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

        # Ghi log TensorBoard
        agent.log_episode(ep_reward, ep_cost, service_level, stockout_rate)

        # Cập nhật thanh tiến trình progress bar
        pbar.set_postfix({
            "rew":   f"{ep_reward:.0f}",
            "cost":  f"{ep_cost:.0f}",
            "SL":    f"{service_level:.3f}",
            "ε":     f"{agent.epsilon:.3f}",
            "loss":  f"{avg_loss:.4f}",
        })

        # ---- Đánh giá định kỳ (greedy policy) --------------------------------
        if episode % args.eval_every == 0:
            eval_reward, eval_sl = evaluate_episode(env, agent)
            agent.writer.add_scalar("eval/reward",        eval_reward, episode)
            agent.writer.add_scalar("eval/service_level", eval_sl,     episode)

            tqdm.write(
                f"\n[Đánh giá ep={episode:4d}] phần_thưởng={eval_reward:.1f}  "
                f"service_level={eval_sl:.4f}  ε={agent.epsilon:.3f}"
            )

        # ---- Lưu checkpoint tốt nhất ----------------------------------------
        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(os.path.join(args.checkpoint_dir, "best_model.pth"))

        # ---- Lưu checkpoint định kỳ ----------------------------------------
        if episode % args.save_every == 0:
            agent.save(
                os.path.join(args.checkpoint_dir, f"ckpt_ep{episode:04d}.pth")
            )

    # ---- Tổng kết huấn luyện ------------------------------------------------
    agent.save(os.path.join(args.checkpoint_dir, "final_model.pth"))
    agent.close()

    # In thống kê cuối cùng
    last_100 = slice(-min(100, args.episodes), None)
    print("\n" + "=" * 65)
    print("Huấn luyện HOÀN THÀNH!")
    print(f"  Phần thưởng tập tốt nhất:         {best_reward:.2f}")
    print(f"  Phần thưởng trung bình 100 tập cuối: {np.mean(episode_rewards[last_100]):.2f}")
    print(f"  Chi phí trung bình 100 tập cuối:     {np.mean(episode_costs[last_100]):.2f}")
    print(f"  Mức độ phục vụ trung bình 100 tập:   {np.mean(episode_sl[last_100]):.4f}")
    print(f"  Tổng số lần cập nhật gradient:       {agent.update_count}")
    print(f"  Thư mục checkpoint: {args.checkpoint_dir}")
    print("=" * 65)

    # Lưu đường cong huấn luyện ra file numpy để vẽ biểu đồ
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
    Chạy 1 tập đánh giá tham lam (không khám phá).

    Tham số
    ------
    env : MultiWarehouseInventoryEnv
    agent : DoubleDQNAgent
    seed : int

    Trả về
    -----
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
# Điểm vào chương trình
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    # Đảm bảo tính khả lặp lại (Reproducibility)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    train(args)
