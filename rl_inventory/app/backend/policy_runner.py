"""
app/backend/policy_runner.py
=============================
Nạp tác tử IPPO đã huấn luyện (checkpoint .pth) và 3 chính sách cổ điển
(EOQ, (s,S), Newsvendor), đóng gói thành cùng MỘT giao diện:

    action_fn(obs, env) -> np.ndarray (n_pairs,)  chỉ số hành động

để app/backend/simulator.py chạy được bất kỳ chính sách nào như nhau.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.inventory_env import MultiWarehouseInventoryEnv
from baselines.traditional_policies import (
    EOQPolicy, SsPolicy, NewsvendorPolicy, build_env_state)

ActionFn = Callable[[np.ndarray, MultiWarehouseInventoryEnv], np.ndarray]


def load_ippo(checkpoint_path: str, env: MultiWarehouseInventoryEnv,
             ppo_cfg: dict):
    """Nạp PPOAgent từ checkpoint. Trả về (action_fn, agent) - agent trả về
    để tái dùng (vd trang 'Đề xuất' cần gọi lại nhiều lần với obs khác nhau)."""
    from agents.ppo_agent import PPOAgent
    agent = PPOAgent(obs_per_pair=env.obs_per_pair, n_pairs=env.n_pairs,
                     n_action_levels=env.n_action_levels, config=ppo_cfg, device="cpu")
    agent.load(checkpoint_path)
    agent.network.eval()

    def action_fn(obs, _env):
        a, _, _ = agent.select_action(obs, deterministic=True)
        return a

    return action_fn, agent


# Thu tu uu tien checkpoint mac dinh: mo hinh chinh moi -> 3 seed ban truoc -> ban cu.
DEFAULT_CHECKPOINTS = ["best_model_main_s42.pth", "best_model_seed42.pth", "best_model.pth"]


def default_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Checkpoint IPPO mac dinh cho app/MCP (khong lay theo thu tu ten file,
    vi best_model.pth la ban chay thu cu)."""
    for name in DEFAULT_CHECKPOINTS:
        if (checkpoint_dir / name).exists():
            return checkpoint_dir / name
    return None


def load_baseline_params(results_dir: Path) -> dict:
    p = results_dir / "baseline_params.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load_baselines(env: MultiWarehouseInventoryEnv,
                   tuned: Optional[dict] = None) -> Dict[str, ActionFn]:
    """3 chính sách cổ điển, dùng tham số đã tinh chỉnh (baseline_params.json)
    nếu có, không thì dùng mặc định hợp lý."""
    tuned = tuned or {}
    specs = [
        ("EOQ", EOQPolicy, dict(ordering_cost=env.cp_dh, holding_cost=env.cp_lk,
                                lead_time=env.tg_giao_tb, q_factor=1.0)),
        ("(s,S)", SsPolicy, dict(service_level=0.90, lead_time=env.tg_giao_tb,
                                 q_factor=1.0, ordering_cost=env.cp_dh,
                                 holding_cost=env.cp_lk)),
        ("Newsvendor", NewsvendorPolicy, dict(holding_cost=env.cp_lk,
                                              stockout_cost=env.cp_th,
                                              lead_time=env.tg_giao_tb)),
    ]
    out = {}
    for name, cls, kw in specs:
        kw.update(tuned.get(name, {}))
        pol = cls(n_pairs=env.n_pairs, **kw)
        out[name] = (lambda obs, e, pol=pol: pol.get_action(build_env_state(e)))
    return out


def load_all_policies(env: MultiWarehouseInventoryEnv, checkpoint_path: Optional[str],
                      ppo_cfg: dict, results_dir: Path) -> Dict[str, ActionFn]:
    """Trả về dict tên -> action_fn, gồm cả IPPO (nếu có checkpoint) và 3
    baseline. Đây là hàm chính các trang demo gọi."""
    tuned = load_baseline_params(results_dir)
    policies = load_baselines(env, tuned)
    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            ippo_fn, _ = load_ippo(checkpoint_path, env, ppo_cfg)
            policies = {"IPPO": ippo_fn, **policies}
        except (RuntimeError, KeyError):
            pass
    return policies
