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
    # Tat ca value norm: loi v1 la gradient value loss CHUA chuan hoa lan at
    # policy loss tren than chung; con value norm thi loi do bi che mat.
    "ablation_shared_trunk": ("Actor/Critic dung chung than mang, khong chuan hoa value (tai hien v1)",
                              lambda c: c["ppo"].update(shared_trunk=True, use_value_norm=False)),
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

# [Goi C] Duong bien chi phi - muc phuc vu: chi doi phi_dv
PHI = {"05": 0.5, "10": 1.0, "20": 2.0, "60": 6.0, "100": 10.0}
for _k, _v in PHI.items():
    ABLATIONS[f"service_penalty_{_k}"] = (f"He so phat muc phuc vu phi_dv = {_v}",
                              lambda c, v=_v: c["env"].update(phi_dv=v))

ABLATIONS["forecast_observation"] = ("IPPO nhan du bao trong quan sat",
                            lambda c: c["env"].update(include_forecast=True))
# z_levels co 6 muc de giu nguyen n_action_levels; lead_time, q_factor nhu (s,S) tinh chinh
ABLATIONS["safety_factor"] = ("Tac tu chon he so an toan z cua chinh sach (s,S)",
                              lambda c: c["ppo"].update(
                                  action_mode="base_stock",
                                  base_stock=dict(z_levels=[-0.5, 0.0, 0.5, 1.0, 1.5, 2.5],
                                                  lead_time=3.0, q_factor=1.0,
                                                  ordering_cost=c["env"]["cp_dh"],
                                                  holding_cost=c["env"]["cp_lk"])))
# [He so an toan tren nen du bao] Co dinh z thi trung Newsvendor du bao, nen RL chi phai hoc phan dieu chinh theo
# trang thai. z_levels, entropy va phi_dv chon tren mien val: luoi hep va entropy thap
# de tac tu khong chon z gan ngau nhien khi tin hieu loi the nho
ABLATIONS["forecast_safety_factor"] = ("Tac tu chon he so an toan tren nen du bao",
                            lambda c: (c["env"].update(include_forecast=True, phi_dv=1.0),
                                       c["ppo"].update(action_mode="base_stock", ent_coef=0.005,
                                                       ent_coef_end=0.0005,
                                                       base_stock=dict(z_levels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.2],
                                                                       lead_time=3.0, du_bao=True))))

def dieu_kien(ten, mo_ta, **env):
    """Dieu kien van hanh chi dung de DANH GIA zero-shot mo hinh chinh (khong train lai), ket qua o thu muc rieng.
    supply_hide_obs: giu 44 chieu quan sat cua mo hinh chinh, tran nguon cung van ap dung."""
    def cap_nhat(c):
        c["env"].update(env)
        c["paths"].update(results_dir=f"results/{ten}")
    ABLATIONS[ten] = (mo_ta, cap_nhat)


# Muc nguon cung do bang chinh sach co san: 2 lan cau TB lam fill giam khoang 4 diem, duoi 1,5 lan
# thi khong chinh sach nao giu duoc 85% vi don dat theo lo hay vuot tran ngay
dieu_kien("shared_supply_tight", "Nguon cung chung cang", supply_cover=2.0, supply_hide_obs=True)
dieu_kien("shared_supply_mild", "Nguon cung chung vua", supply_cover=3.0, supply_hide_obs=True)
dieu_kien("capacity_3", "Suc chua bang 3 ngay cau", capacity_cover_days=3.0)
dieu_kien("capacity_8", "Suc chua bang 8 ngay cau", capacity_cover_days=8.0)


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
