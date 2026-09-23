"""
scripts/common.py
=================
Tien ich dung chung cho cac script phan tich hau nghiem (regime_analysis.py,
policy_behavior.py): nap du lieu, tao moi truong, nap IPPO va cac baseline
(ca ban tinh chinh truc tiep lan ban cung muc phuc vu tu iso_service.json).
"""
import sys
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.inventory_env import MultiWarehouseInventoryEnv
from agents.ppo_agent import PPOAgent
from baselines.traditional_policies import POLICY_REGISTRY, build_env_state


def load_config(path="config.yaml"):
    return yaml.safe_load(open(ROOT / path, encoding="utf-8"))


def load_data(cfg):
    """Tra ve dict cac mang du lieu da tien xu ly (None neu thieu file)."""
    d = ROOT / cfg["paths"]["data_dir"]
    out = {}
    for key in ["demand_data", "calendar_features", "price_per_pair", "price_series"]:
        p = d / f"{key}.npy"
        out[key] = np.load(str(p)) if p.exists() else None
    if out["calendar_features"] is not None and out["calendar_features"].size == 0:
        out["calendar_features"] = None
    return out


def make_env(cfg, data, mode="test", demand_override=None):
    demand = data["demand_data"] if demand_override is None else demand_override
    return MultiWarehouseInventoryEnv(
        config=cfg["env"], demand_data=demand,
        calendar_features=data["calendar_features"],
        price_per_pair=data["price_per_pair"],
        price_series=data["price_series"], mode=mode)


def load_agent(cfg, env, checkpoint):
    agent = PPOAgent(env.obs_per_pair, env.n_pairs, env.n_action_levels,
                     config=cfg["ppo"], device="cpu")
    agent.load(str(ROOT / checkpoint))
    agent.network.eval()
    return agent


def make_policies(cfg, env, checkpoint, include_iso=True):
    """dict ten -> action_fn(obs, env). Gom IPPO, 3 baseline tinh chinh truc
    tiep (baseline_params.json) va - neu co iso_service.json - cac baseline
    duoc chinh de dat CUNG MUC PHUC VU voi IPPO (so sanh chi phi cong bang)."""
    results = ROOT / cfg["paths"]["results_dir"]
    agent = load_agent(cfg, env, checkpoint)
    pols = {"IPPO": lambda obs, e: agent.select_action(obs, deterministic=True)[0]}

    defaults = {
        "EOQ": dict(ordering_cost=env.cp_dh, holding_cost=env.cp_lk,
                    lead_time=env.tg_giao_tb),
        "(s,S)": dict(service_level=0.90, lead_time=env.tg_giao_tb,
                      ordering_cost=env.cp_dh, holding_cost=env.cp_lk),
        "Newsvendor": dict(holding_cost=env.cp_lk, stockout_cost=env.cp_th,
                           lead_time=env.tg_giao_tb),
    }

    def build(name, params):
        pol = POLICY_REGISTRY[name](n_pairs=env.n_pairs, **{**defaults[name], **params})
        return lambda obs, e, pol=pol: pol.get_action(build_env_state(e))

    p = results / "baseline_params.json"
    tuned = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    for name in defaults:
        pols[name] = build(name, tuned.get(name, {}))

    p = results / "iso_service.json"
    if include_iso and p.exists():
        iso = json.loads(p.read_text(encoding="utf-8"))
        for name, r in iso.get("ket_qua", {}).items():
            if name in defaults and r.get("params"):
                pols[f"{name} (cùng mức PV)"] = build(name, r["params"])
    return pols
