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

# [E3] Do nhay sieu tham so PPO, moi lan doi mot tham so so voi abl_ref
for _ten, _h in [("05", 0.5), ("2", 2.0)]:
    ABLATIONS[f"tuning_lr_x{_ten}"] = (f"Do nhay: lr_actor x{_h}",
                                   lambda c, h=_h: c["ppo"].update(lr_actor=c["ppo"]["lr_actor"] * h))
    ABLATIONS[f"tuning_entropy_x{_ten}"] = (f"Do nhay: ent_coef x{_h}",
                                    lambda c, h=_h: c["ppo"].update(ent_coef=c["ppo"]["ent_coef"] * h,
                                                                    ent_coef_end=c["ppo"]["ent_coef_end"] * h))
for _n in (512, 2048):
    ABLATIONS[f"tuning_rollout_{_n}"] = (f"Do nhay: n_steps = {_n}",
                                    lambda c, n=_n: c["ppo"].update(n_steps=n))

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
ABLATIONS["warehouse_critic"] = ("Critic co ngu canh theo kho",
                              lambda c: c["ppo"].update(critic_context="warehouse", batch_mode="warehouse",
                                                        warehouse_skus=c["env"]["n_skus"]))
for _k in (1, 4, 10, 300):
    ABLATIONS[f"networks_{_k}"] = (f"{_k} mang theo nhom, lay mau theo buoc thoi gian",
                              lambda c, k=_k: c["ppo"].update(param_groups=k, batch_mode="time"))
# [Phat theo tung cap] Ba bo (KP, KI, KD) chay thu truoc khi chay du seed
# KI nho vi lambda giu qua cac tap va cap nhat 12 lan moi tap: phai cham hon toc do
# chinh sach thay doi (khoang 50-100 tap), neu khong se cham tran truoc khi kip phan ung.
for _ten, _p in {"a": (0.5, 0.02, 0.0), "b": (1.0, 0.05, 0.0), "c": (1.0, 0.05, 0.5)}.items():
    ABLATIONS[f"pair_penalty_{_ten}"] = (f"Lagrange PID theo cap (KP, KI, KD) = {_p}",
                                   lambda c, p=_p: c["env"].update(
                                       lagrange_per_pair=True,
                                       lagrange=dict(kp=p[0], ki=p[1], kd=p[2], lambda_max=20.0)))
ABLATIONS["scale_1000_agents"] = ("Mo rong 10 kho x 100 SKU (1.000 tac tu), du lieu va ket qua rieng",
                          lambda c: (c["env"].update(n_skus=100),
                                     c["preprocess"].update(n_skus=100),
                                     c["paths"].update(data_dir="data/processed_100sku",
                                                       results_dir="results/scale_1000")))



def dieu_kien(ten, mo_ta, **env):
    """Dieu kien van hanh: mo hinh chinh va ban he so an toan tren nen du bao, thu muc ket qua rieng."""
    def cap_nhat(c):
        c["env"].update(env)
        c["paths"].update(results_dir=f"results/{ten}")
    ABLATIONS[ten] = (mo_ta, cap_nhat)
    ABLATIONS[f"{ten}_fs"] = (f"{mo_ta}, tac tu chon he so an toan tren nen du bao",
                              lambda c: (ABLATIONS["forecast_safety_factor"][1](c), cap_nhat(c)))


# Muc nguon cung do bang chinh sach co san: 2 lan cau TB lam fill giam khoang 4 diem, duoi 1,5 lan
# thi khong chinh sach nao giu duoc 85% vi don dat theo lo hay vuot tran ngay
dieu_kien("shared_supply_tight", "Nguon cung chung cang", supply_cover=2.0)
dieu_kien("shared_supply_mild", "Nguon cung chung vua", supply_cover=3.0)
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
