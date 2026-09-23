"""
configs/make_configs.py
=======================
Sinh cac config ablation tu config.yaml: moi file CHI KHAC config chinh dung
MOT thay doi (nguyen tac ablation). Chay lai moi khi sua config.yaml:

    python configs/make_configs.py
"""
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
HOLDOUT = [3, 8, 9, 10, 23, 25]          # 6 SKU trai deu thu hang quy mo cau
TRAIN_SKUS = [i for i in range(30) if i not in HOLDOUT]

ABLATIONS = {
    # ten file              : (mo ta, ham sua config)
    "ablation_global_reward": ("RQ3: phan thuong toan cuc (team reward) thay vi cuc bo",
                               lambda c: c["env"].update(reward_mode="global")),
    "ablation_q3_no_reward_norm": ("RQ3: tat chuan hoa phan thuong theo cap",
                                   lambda c: c["env"].update(normalize_reward_per_pair=False)),
    "ablation_reward_demand_scaled": ("Phat SLA kieu cu (ty le mean_demand)",
                                      lambda c: c["env"].update(service_penalty_mode="demand")),
    "ablation_shared_trunk": ("Actor/Critic dung chung than mang (v1)",
                              lambda c: c["ppo"].update(shared_trunk=True)),
    "ablation_drop_calendar": ("Tat 27 chieu dac trung lich",
                               lambda c: c["env"].update(obs_drop=["calendar"])),
    "ablation_drop_warehouse": ("Tat 2 tin hieu cap kho (muc day, ty trong)",
                                lambda c: c["env"].update(obs_drop=["warehouse"])),
    "ablation_event_lookahead": ("Them 2 dac trung su kien sap toi",
                                 lambda c: c["env"].update(include_event_lookahead=True)),
    "ablation_warm_start": ("Khoi tao lich su cau bang du lieu that",
                            lambda c: c["env"].update(warm_start_history=True)),
    "holdout_train": ("Hold-out: huan luyen tren 24 SKU",
                      lambda c: c["env"].update(sku_indices=TRAIN_SKUS)),
    "holdout_eval": ("Hold-out: danh gia tren 6 SKU chua gap khi huan luyen",
                     lambda c: c["env"].update(sku_indices=HOLDOUT)),
}


def main():
    base = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    for name, (desc, fn) in ABLATIONS.items():
        cfg = copy.deepcopy(base)
        fn(cfg)
        out = ROOT / "configs" / f"{name}.yaml"
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"# SINH TU DONG boi configs/make_configs.py - KHONG sua tay.\n"
                    f"# {desc}. Moi thu khac giong het config.yaml.\n")
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        print(f"  {out.relative_to(ROOT)}  <- {desc}")


if __name__ == "__main__":
    main()
