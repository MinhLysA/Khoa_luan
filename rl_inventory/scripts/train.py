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

PHIEN BAN P0 (xem README_CLAUDE_CODE_KLTN.md):
  [P0-4] "fill_rate" ghi nhat ky (ca train lan eval) nay la TRUE EPISODE FILL
         RATE = 1 - tong_thieu_hang/tong_cau CUA CA EPISODE, khong con la
         trung binh cong cua tin hieu fill rate cua so truot 30 ngay tai
         moi buoc (ban cu: trung binh cua N so da trung lap chong lan nhau,
         thien lech nhat luc dau episode khi cua so con rong).
  [P0-5] Chon best_model theo CHI PHI VAN HANH THAP NHAT trong nhom cac
         checkpoint DAT duoc rang buoc fill rate >= ppo.min_fill_to_save
         (truoc: chon theo REWARD cao nhat, kem theo nguong fill_to_save
         mac dinh 0.0 tuc moi checkpoint deu "dat"). Neu chua checkpoint nao
         dat rang buoc, tam luu checkpoint co fill rate cao nhat lam
         fallback va IN RO feasible=False.
  [P0-8] SUA LOI --resume KHONG THUC SU "RESUME": ban truoc --resume chi nap
         lai TRONG SO mang, con episode_count/global_step luon bat dau lai tu
         0, file log mo o che do "w" (GHI DE, mat toan bo lich su cu), va
         optimizer Adam khoi tao lai tu dau (mat dong luong). Hau qua: lich
         trinh suy giam LR/entropy chay lai tu dau thay vi noi tiep, va cac
         file checkpoint_epNNN co the bi ghi de boi mot giai doan huan luyen
         khac voi cung so N. Nay moi episode deu ghi kem
         checkpoints/train_state{tag}.json (episode_count, global_step,
         elapsed_s, best_feasible_cost, best_feasible_found,
         fallback_best_fill, reward_window); --resume doc lai file nay (neu
         co, khop voi --tag) de noi tiep dung episode_count va tieu chi chon
         best_model, mo log o che do "a" (noi tiep) thay vi "w".
"""

import os
import sys
import csv
import json
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
              "eval_reward", "eval_cost", "eval_fill", "eval_feasible"]


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

    suffix = f"_{args.tag}" if args.tag else ""
    state_path = checkpoint_dir / f"train_state{suffix}.json"

    # [P0-8] Trang thai noi tiep khi --resume: episode_count/global_step,
    # tieu chi chon best_model, va do lech elapsed_s cua (cac) lan chay truoc.
    # Mac dinh nhu chua tung resume; se bi ghi de neu doc duoc state_path.
    resumed_episode_count = 0
    resumed_global_step   = 0
    resumed_elapsed_s     = 0.0
    resumed_reward_window = []
    resumed_best_feasible_cost  = float("inf")
    resumed_best_feasible_found = False
    resumed_fallback_best_fill  = -float("inf")
    da_resume = False

    if args.resume:
        rp = ROOT / args.resume
        if rp.exists():
            agent.load(str(rp), load_optimizer=True)
            print(f"[V2-27] Da nap trong so + optimizer tu {rp} de hoc tiep.")
            if state_path.exists():
                st = json.loads(state_path.read_text(encoding="utf-8"))
                resumed_episode_count = int(st.get("episode_count", 0))
                resumed_global_step   = int(st.get("global_step", 0))
                resumed_elapsed_s     = float(st.get("elapsed_s", 0.0))
                resumed_reward_window = list(st.get("reward_window", []))
                # None trong json = "chua co gia tri" (da ghi lai tu +-inf
                # luc luu, vi JSON khong co Infinity chuan) -> doi lai +-inf.
                v = st.get("best_feasible_cost")
                resumed_best_feasible_cost = float("inf") if v is None else float(v)
                resumed_best_feasible_found = bool(st.get("best_feasible_found", False))
                v = st.get("fallback_best_fill")
                resumed_fallback_best_fill = -float("inf") if v is None else float(v)
                da_resume = True
                print(f"[P0-8] Da nap {state_path.name}: noi tiep tu episode "
                      f"{resumed_episode_count}, best_feasible_found="
                      f"{resumed_best_feasible_found}.")
            else:
                print(f"[P0-8] KHONG THAY {state_path.name} -> checkpoint nay "
                      f"duoc luu TRUOC ban vá [P0-8] (hoac khac --tag). Tiep "
                      f"tuc voi trong so da nap nhung episode_count, lich su "
                      f"log va tieu chi chon best_model se BAT DAU LAI TU 0 -"
                      f" kiem tra ky --tag truoc khi chay dai.")
        else:
            print(f"[V2-27] Khong tim thay {rp}, bat dau tu dau.")

    def run_deterministic_eval():
        """[P0-4] Tra ve (reward, chi phi van hanh, TRUE fill rate) - ca ba
        deu tinh tren TOAN BO episode, khong phai trung binh cua tin hieu
        cua so truot tai tung buoc. Chi phi van hanh KHONG gom phat SLA
        (chi_phi_service_penalty) vi do la tin hieu dinh hinh hanh vi cho
        huan luyen, khong phai chi phi van hanh thuc te (dung cong thuc
        giong scripts/evaluate.py de hai noi nhat quan)."""
        rewards, costs, fills = [], [], []
        for i in range(eval_n_episodes):
            o, _ = eval_env.reset(seed=90000 + i)
            ep_r, ep_cost = 0.0, 0.0
            ep_demand, ep_stockout = 0.0, 0.0
            while True:
                a, _, _ = agent.select_action(o, deterministic=True)
                o, _, term, trunc, inf = eval_env.step(a)
                ep_r    += inf["raw_reward"]
                ep_cost += (inf["cost_holding"] + inf["cost_stockout"]
                           + inf["cost_ordering"] + inf["cost_overflow"])
                ep_demand   += inf["demand"]
                ep_stockout += inf["stockout"]
                if term or trunc:
                    break
            rewards.append(ep_r)
            costs.append(ep_cost)
            fills.append(1.0 - ep_stockout / max(ep_demand, 1e-6))
        return float(np.mean(rewards)), float(np.mean(costs)), float(np.mean(fills))

    writer = None
    if TENSORBOARD_AVAILABLE:
        # TensorBoard/TensorFlow gfile co the loi tren duong dan chua ky tu
        # Unicode (dau tieng Viet trong ten thu muc du an) o mot so may Windows -
        # day chi la log phu (nguon su that la results/train_log*.csv), nen
        # khong de loi nay lam hong ca qua trinh huan luyen.
        try:
            writer = SummaryWriter(
                log_dir=str(log_dir / f"PPO{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
        except Exception as e:
            print(f"[!] Khong khoi tao duoc TensorBoard SummaryWriter ({e!r}); "
                  f"bo qua, chi dung results/train_log{suffix}.csv.")
            writer = None

    # [P0-8] Noi tiep (append) log cu khi resume THAT SU (co state_path va
    # file log da ton tai); nguoc lai ghi moi nhu cu. Tranh mat lich su cu
    # khi keo dai mot hat giong da huan luyen mot phan.
    log_path = results_dir / f"train_log{suffix}.csv"
    tiep_noi_log = da_resume and log_path.exists()
    log_file = open(log_path, "a" if tiep_noi_log else "w",
                    newline="", encoding="utf-8")
    log_writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
    if not tiep_noi_log:
        log_writer.writeheader()

    print("=" * 68)
    print("HUAN LUYEN IPPO (parameter sharing) - phien ban v2")
    print(f"  device={device}  obs_per_pair={env.obs_per_pair}  n_pairs={env.n_pairs}")
    print(f"  n_actions={env.n_action_levels}  order_relative={env.use_relative_orders}")
    print(f"  suc chua tung kho = {np.round(env.suc_chua_kho).astype(int).tolist()}")
    print(f"  mau/update = {n_steps} x {env.n_pairs} = {n_steps*env.n_pairs:,}")
    print(f"  nhat ky -> {log_path}")
    print("=" * 68)

    # [P0-5] Chon checkpoint theo chi phi thap nhat TRONG SO cac lan danh
    # gia dat rang buoc fill rate; fallback theo fill rate cao nhat neu chua
    # co lan nao dat rang buoc.
    # [P0-8] Khoi tao tu trang thai da resume (neu co) thay vi luon bat dau
    # tu gia tri "trong".
    best_feasible_cost  = resumed_best_feasible_cost
    best_feasible_found = resumed_best_feasible_found
    fallback_best_fill  = resumed_fallback_best_fill
    episode_count = resumed_episode_count
    global_step = resumed_global_step
    reward_window = resumed_reward_window
    anneal     = bool(ppo_cfg.get("anneal_lr", True))
    anneal_ent = bool(ppo_cfg.get("anneal_ent", True))
    min_fill_to_save = float(ppo_cfg.get("min_fill_to_save", 0.0))
    t0 = time.time() - resumed_elapsed_s

    obs, _ = env.reset(seed=seed)
    ep_reward, ep_stockout = 0.0, 0.0
    ep_demand_true, ep_stockout_true = 0.0, 0.0
    ep_costs = {"cost_holding": 0.0, "cost_stockout": 0.0, "cost_ordering": 0.0,
                "cost_overflow": 0.0, "cost_service_penalty": 0.0}
    last_metrics = {}
    last_eval = (float("nan"), float("nan"), float("nan"))

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
            ep_demand_true   += info["demand"]
            ep_stockout_true += info["stockout"]
            for k in ep_costs:
                ep_costs[k] += info[k]
            global_step += 1
            obs = next_obs

            if terminated or truncated:
                episode_count += 1
                # [P0-6] fill rate CUA EPISODE (khong phai trung binh cong
                # cua tin hieu cua so truot tung buoc).
                episode_fill = 1.0 - ep_stockout_true / max(ep_demand_true, 1e-6)
                reward_window.append(ep_reward)
                if len(reward_window) > 20:
                    reward_window.pop(0)

                eval_feasible = ""
                if episode_count % eval_every == 0:
                    last_eval = run_deterministic_eval()
                    det_r, det_cost, det_f = last_eval
                    feasible = det_f >= min_fill_to_save
                    eval_feasible = feasible
                    print(f"  [Eval] Ep {episode_count:5d}  det_cost={det_cost:>14,.0f}"
                          f"  det_fill={det_f:.3f}  feasible={feasible}")
                    # [P0-5] Chon checkpoint theo chi phi thap nhat TRONG SO
                    # cac lan dat rang buoc fill rate; neu chua co lan nao
                    # dat, tam giu checkpoint co fill rate cao nhat (fallback).
                    if feasible and det_cost < best_feasible_cost:
                        best_feasible_cost = det_cost
                        best_feasible_found = True
                        agent.save(str(checkpoint_dir / f"best_model{suffix}.pth"))
                        print(f"         -> luu best_model (feasible, cost={det_cost:,.0f})")
                    elif not best_feasible_found and det_f > fallback_best_fill:
                        fallback_best_fill = det_f
                        agent.save(str(checkpoint_dir / f"best_model{suffix}.pth"))
                        print(f"         -> luu best_model (fallback, CHUA feasible, fill={det_f:.3f})")

                row = {f: "" for f in LOG_FIELDS}
                row.update({
                    "episode": episode_count, "global_step": global_step,
                    "elapsed_s": round(time.time() - t0, 1),
                    "reward_raw": ep_reward,
                    "reward_smooth": float(np.mean(reward_window)),
                    "fill_rate": episode_fill,
                    "stockout": ep_stockout,
                    "cost_total": sum(ep_costs.values()),
                    "entropy": last_metrics.get("entropy", ""),
                    "approx_kl": last_metrics.get("approx_kl", ""),
                    "clip_frac": last_metrics.get("clip_frac", ""),
                    "value_loss": last_metrics.get("value_loss", ""),
                    "explained_variance": last_metrics.get("explained_variance", ""),
                    "ent_coef": agent.ent_coef,
                    "eval_reward": last_eval[0], "eval_cost": last_eval[1],
                    "eval_fill": last_eval[2], "eval_feasible": eval_feasible,
                })
                row.update(ep_costs)
                log_writer.writerow(row)
                log_file.flush()

                # [P0-8] Ghi lai trang thai de --resume sau nay noi tiep dung
                # (episode_count, tieu chi chon best_model, ...). Re, ghi moi
                # episode la du (khong can dong bo voi nhip luu checkpoint).
                state_path.write_text(json.dumps({
                    "episode_count": episode_count,
                    "global_step": global_step,
                    "elapsed_s": round(time.time() - t0, 1),
                    "reward_window": reward_window,
                    "best_feasible_cost": (None if best_feasible_cost == float("inf")
                                           else best_feasible_cost),
                    "best_feasible_found": best_feasible_found,
                    "fallback_best_fill": (None if fallback_best_fill == -float("inf")
                                           else fallback_best_fill),
                }), encoding="utf-8")

                if episode_count % 10 == 0:
                    print(f"Ep {episode_count:5d}/{total_episodes}"
                          f"  reward={ep_reward:>14,.0f}"
                          f"  smooth={np.mean(reward_window):>14,.0f}"
                          f"  fill={episode_fill:.3f}"
                          f"  ent={last_metrics.get('entropy', float('nan')):.3f}"
                          f"  ev={last_metrics.get('explained_variance', float('nan')):.2f}")

                if writer:
                    writer.add_scalar("train/episode_reward", ep_reward, episode_count)
                    writer.add_scalar("train/fill_rate", episode_fill, episode_count)
                    for k, v in ep_costs.items():
                        writer.add_scalar(f"cost/{k}", v, episode_count)

                ep_reward, ep_stockout = 0.0, 0.0
                ep_demand_true, ep_stockout_true = 0.0, 0.0
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
    if best_feasible_found:
        print(f"HOAN TAT sau {time.time()-t0:.0f}s. "
              f"Best FEASIBLE checkpoint (fill>={min_fill_to_save:.0%}): "
              f"cost={best_feasible_cost:,.0f}")
    else:
        print(f"HOAN TAT sau {time.time()-t0:.0f}s.")
        print(f"  [!] KHONG checkpoint nao dat fill >= {min_fill_to_save:.0%} trong "
              f"suot qua trinh huan luyen; best_model{suffix}.pth la FALLBACK theo "
              f"fill rate cao nhat (fill={fallback_best_fill:.3f}), feasible=False. "
              f"Can xem xet tang so episode, giam min_fill_to_save, hoac tang phi_dv.")
    print(f"  checkpoint : {checkpoint_dir / f'best_model{suffix}.pth'}")
    print(f"  nhat ky    : {log_path}")


if __name__ == "__main__":
    train()
