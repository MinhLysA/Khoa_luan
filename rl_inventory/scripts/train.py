"""
scripts/train.py
================
Vong lap huan luyen IPPO cho bai toan ton kho da kho - da SKU.

PHIEN BAN v2:
  [V2-12] Khong con "vá" phan thuong cuoi episode bang gamma*V(s_T); buffer
          nhan next_value truc tiep (xem agents/rollout_buffer.py).
  [V2-14] GHI NHAT KY RA results/train_log.csv sau MOI episode. App Streamlit
          doc file nay de ve duong hoc theo thoi gian thuc.
  [V2-15] CHON CHECKPOINT TOT NHAT BANG DANH GIA DETERMINISTIC dinh ky, thay vi
          bang trung binh truot cua reward luc dang lay mau ngau nhien. Reward
          khi dang explore khong phai chat luong chinh sach.
  [V2-16] In ra canh bao chan doan (entropy khong giam, explained_variance am,
          KL qua lon, fill rate ket) ngay trong luc chay.

PHIEN BAN v3:
  [V3-1] eval_env DUNG mode="val" (khong con "test"): chon best_model bang
         mien VALIDATION rieng, khong con dong tram vao mien test dung de bao
         cao ket qua cuoi (xem env/inventory_env.py mo ta [V3-1]).
  [V3-2] Nap them price_per_pair.npy (neu co) de dua vao chi phi thieu hang
         theo gia that.
"""

import os
import sys
import csv
import yaml
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.inventory_env import MultiWarehouseInventoryEnv
from agents.ppo_agent import PPOAgent
from agents.rollout_buffer import RolloutBuffer

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


LOG_FIELDS = ["episode", "global_step", "elapsed_s", "reward_raw", "reward_smooth",
              "fill_rate", "stockout", "cost_holding", "cost_stockout",
              "cost_ordering", "cost_overflow", "cost_service_penalty",
              "cost_total", "entropy", "approx_kl", "clip_frac", "value_loss",
              "explained_variance", "lr_scale", "ent_coef",
              "eval_reward", "eval_fill"]


def load_data(paths):
    demand_data = None
    p = ROOT / paths["data_dir"] / "demand_data.npy"
    if p.exists():
        demand_data = np.load(str(p))
    calendar_features = None
    p = ROOT / paths["data_dir"] / "calendar_features.npy"
    if p.exists():
        calendar_features = np.load(str(p))
    price_per_pair = None
    p = ROOT / paths["data_dir"] / "price_per_pair.npy"
    if p.exists():
        price_per_pair = np.load(str(p))
    price_series = None
    p = ROOT / paths["data_dir"] / "price_series.npy"
    if p.exists():
        price_series = np.load(str(p))
    return demand_data, calendar_features, price_per_pair, price_series


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    # [V2-27] Huan luyen tren CPU rat lau; cho phep chay tiep tu checkpoint cu
    # thay vi bat dau lai tu dau moi khi may bi ngat.
    parser.add_argument("--resume", type=str, default=None,
                        help="Duong dan checkpoint de hoc tiep")
    args = parser.parse_args()

    with open(ROOT / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env_cfg, ppo_cfg, paths = cfg["env"], cfg["ppo"], cfg["paths"]
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    total_episodes = args.episodes or ppo_cfg["total_episodes"]
    n_steps = int(ppo_cfg["n_steps"])

    checkpoint_dir = ROOT / paths["checkpoint_dir"]
    log_dir        = ROOT / paths["log_dir"]
    results_dir    = ROOT / paths["results_dir"]
    for d in (checkpoint_dir, log_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)

    demand_data, calendar_features, price_per_pair, price_series = load_data(paths)
    if demand_data is None:
        print("Khong tim thay demand_data.npy -> dung Poisson ngau nhien")

    env = MultiWarehouseInventoryEnv(config=env_cfg, demand_data=demand_data,
                                     calendar_features=calendar_features,
                                     price_per_pair=price_per_pair,
                                     price_series=price_series, mode="train")
    # [V3-1] mode="val": chon best_model bang mien validation, KHONG dong tram
    # vao mien "test" dung de bao cao ket qua cuoi (scripts/evaluate.py).
    eval_env = MultiWarehouseInventoryEnv(config=env_cfg, demand_data=demand_data,
                                          calendar_features=calendar_features,
                                          price_per_pair=price_per_pair,
                                          price_series=price_series, mode="val")
    eval_every      = int(ppo_cfg.get("eval_every_episodes", 25))
    eval_n_episodes = int(ppo_cfg.get("eval_n_episodes", 2))

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    agent = PPOAgent(obs_per_pair=env.obs_per_pair, n_pairs=env.n_pairs,
                     n_action_levels=env.n_action_levels, config=ppo_cfg, device=device)
    buffer = RolloutBuffer(buffer_size=n_steps, obs_per_pair=env.obs_per_pair,
                           n_pairs=env.n_pairs, device=device,
                           normalize_per_pair=ppo_cfg.get("normalize_adv_per_pair", True))

    if args.resume:
        rp = ROOT / args.resume
        if rp.exists():
            agent.load(str(rp))
            print(f"[V2-27] Da nap trong so tu {rp} de hoc tiep.")
        else:
            print(f"[V2-27] Khong tim thay {rp}, bat dau tu dau.")

    def run_deterministic_eval():
        rewards, fills = [], []
        for i in range(eval_n_episodes):
            o, _ = eval_env.reset(seed=90000 + i)
            ep_r, ep_f = 0.0, []
            while True:
                a, _, _ = agent.select_action(o, deterministic=True)
                o, _, term, trunc, inf = eval_env.step(a)
                ep_r += inf["raw_reward"]
                ep_f.append(inf["fill_rate_mean"])
                if term or trunc:
                    break
            rewards.append(ep_r)
            fills.append(float(np.mean(ep_f)))
        return float(np.mean(rewards)), float(np.mean(fills))

    suffix = f"_{args.tag}" if args.tag else ""
    writer = None
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(
            log_dir=str(log_dir / f"PPO{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))

    log_path = results_dir / f"train_log{suffix}.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
    log_writer.writeheader()

    print("=" * 68)
    print("HUAN LUYEN IPPO (parameter sharing) - phien ban v2")
    print(f"  device={device}  obs_per_pair={env.obs_per_pair}  n_pairs={env.n_pairs}")
    print(f"  n_actions={env.n_action_levels}  order_relative={env.use_relative_orders}")
    print(f"  suc chua tung kho = {np.round(env.suc_chua_kho).astype(int).tolist()}")
    print(f"  mau/update = {n_steps} x {env.n_pairs} = {n_steps*env.n_pairs:,}")
    print(f"  nhat ky -> {log_path}")
    print("=" * 68)

    best_score = -float("inf")
    episode_count = 0
    global_step = 0
    reward_window, fill_window = [], []
    anneal     = bool(ppo_cfg.get("anneal_lr", True))
    anneal_ent = bool(ppo_cfg.get("anneal_ent", True))
    min_fill_to_save = float(ppo_cfg.get("min_fill_to_save", 0.0))
    t0 = time.time()

    obs, _ = env.reset(seed=seed)
    ep_reward, ep_stockout, ep_fill = 0.0, 0.0, []
    ep_costs = {"cost_holding": 0.0, "cost_stockout": 0.0, "cost_ordering": 0.0,
                "cost_overflow": 0.0, "cost_service_penalty": 0.0}
    last_metrics = {}
    last_eval = (float("nan"), float("nan"))

    while episode_count < total_episodes:
        buffer.reset()

        for _ in range(n_steps):
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # [V2-12] O buoc cuoi episode (cat vi het gio), V(s_{t+1}) van ton
            # tai va phai duoc bootstrap -> tinh va dua thang cho buffer.
            nv = None
            if truncated and not terminated:
                _, _, nv = agent.select_action(next_obs)

            buffer.add(obs, action, log_prob, info["local_scaled_rewards"], value,
                       terminated=terminated,
                       episode_end=(terminated or truncated),
                       next_value=nv)

            ep_reward   += info["raw_reward"]
            ep_stockout += info["stockout"]
            ep_fill.append(info["fill_rate_mean"])
            for k in ep_costs:
                ep_costs[k] += info[k]
            global_step += 1
            obs = next_obs

            if terminated or truncated:
                episode_count += 1
                reward_window.append(ep_reward)
                fill_window.append(float(np.mean(ep_fill)))
                if len(reward_window) > 20:
                    reward_window.pop(0)
                    fill_window.pop(0)

                if episode_count % eval_every == 0:
                    last_eval = run_deterministic_eval()
                    det_r, det_f = last_eval
                    print(f"  [Eval] Ep {episode_count:5d}  det_reward={det_r:>14,.0f}"
                          f"  det_fill={det_f:.3f}")
                    # [V2-15] Chon checkpoint theo danh gia deterministic
                    if det_f >= min_fill_to_save and det_r > best_score:
                        best_score = det_r
                        agent.save(str(checkpoint_dir / f"best_model{suffix}.pth"))
                        print(f"         -> luu best_model (score={det_r:,.0f})")

                row = {f: "" for f in LOG_FIELDS}
                row.update({
                    "episode": episode_count, "global_step": global_step,
                    "elapsed_s": round(time.time() - t0, 1),
                    "reward_raw": ep_reward,
                    "reward_smooth": float(np.mean(reward_window)),
                    "fill_rate": float(np.mean(ep_fill)),
                    "stockout": ep_stockout,
                    "cost_total": sum(ep_costs.values()),
                    "entropy": last_metrics.get("entropy", ""),
                    "approx_kl": last_metrics.get("approx_kl", ""),
                    "clip_frac": last_metrics.get("clip_frac", ""),
                    "value_loss": last_metrics.get("value_loss", ""),
                    "explained_variance": last_metrics.get("explained_variance", ""),
                    "ent_coef": agent.ent_coef,
                    "eval_reward": last_eval[0], "eval_fill": last_eval[1],
                })
                row.update(ep_costs)
                log_writer.writerow(row)
                log_file.flush()

                if episode_count % 10 == 0:
                    print(f"Ep {episode_count:5d}/{total_episodes}"
                          f"  reward={ep_reward:>14,.0f}"
                          f"  smooth={np.mean(reward_window):>14,.0f}"
                          f"  fill={np.mean(ep_fill):.3f}"
                          f"  ent={last_metrics.get('entropy', float('nan')):.3f}"
                          f"  ev={last_metrics.get('explained_variance', float('nan')):.2f}")

                if writer:
                    writer.add_scalar("train/episode_reward", ep_reward, episode_count)
                    writer.add_scalar("train/fill_rate", np.mean(ep_fill), episode_count)
                    for k, v in ep_costs.items():
                        writer.add_scalar(f"cost/{k}", v, episode_count)

                ep_reward, ep_stockout, ep_fill = 0.0, 0.0, []
                ep_costs = {k: 0.0 for k in ep_costs}
                obs, _ = env.reset()
                if episode_count >= total_episodes:
                    break

        frac = 1.0 - episode_count / max(total_episodes, 1)
        if anneal:
            agent.set_lr_scale(max(0.1, frac))
        if anneal_ent:
            agent.set_ent_scale(max(0.0, frac))

        _, _, last_value = agent.select_action(obs)
        buffer.compute_gae(last_values=last_value,
                           gamma=ppo_cfg["gamma"], gae_lambda=ppo_cfg["gae_lambda"])
        last_metrics = agent.update(buffer)

        if writer:
            for k, v in last_metrics.items():
                writer.add_scalar(f"loss/{k}", v, global_step)

        # [V2-16] Canh bao chan doan
        if last_metrics.get("explained_variance", 0) < -0.5:
            print("  [!] explained_variance am manh -> critic chua bam duoc return.")
        if last_metrics.get("approx_kl", 0) > 0.05:
            print("  [!] approx_kl > 0.05 -> policy nhay qua manh, giam lr_actor.")

        if episode_count % 200 == 0 and episode_count > 0:
            agent.save(str(checkpoint_dir / f"checkpoint_ep{episode_count}{suffix}.pth"))

    agent.save(str(checkpoint_dir / f"final_model{suffix}.pth"))
    if not (checkpoint_dir / f"best_model{suffix}.pth").exists():
        agent.save(str(checkpoint_dir / f"best_model{suffix}.pth"))
    log_file.close()
    if writer:
        writer.close()
    print("=" * 68)
    print(f"HOAN TAT sau {time.time()-t0:.0f}s. Best eval reward: {best_score:,.0f}")
    print(f"  checkpoint : {checkpoint_dir / f'best_model{suffix}.pth'}")
    print(f"  nhat ky    : {log_path}")


if __name__ == "__main__":
    train()
