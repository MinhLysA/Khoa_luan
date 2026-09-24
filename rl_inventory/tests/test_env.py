"""
tests/test_env.py
=================
Kiem thu don vi cho MultiWarehouseInventoryEnv (phien ban v2).

Chay:  python -m pytest tests/ -v      hoac      python run.py test

Cac test danh dau [V2-n] la test HOI QUY: chung se THAT BAI tren ban cu, dung
de chung minh loi da duoc sua.
"""

import sys
import numpy as np
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.inventory_env import MultiWarehouseInventoryEnv

CFG = dict(n_warehouses=3, n_skus=4, episode_length=40, lookback=5,
           lead_time_min=1, lead_time_max=3,
           warehouse_capacity="auto", capacity_cover_days=5.0,
           initial_cover_days=2.0, cp_lk=1.0, cp_th=10.0, cp_dh=5.0,
           pt_tk=5.0, phi_dv=3.0, muc_dv=0.85, fill_rate_window=10,
           use_relative_orders=True, min_mean_demand=0.05,
           order_multipliers=[0.0, 0.5, 1.0, 2.0, 3.0, 5.0])


def make_env(price_per_pair=None, price_series=None, mode="train", **kw):
    cfg = dict(CFG); cfg.update(kw)
    T, W, S = 200, cfg["n_warehouses"], cfg["n_skus"]
    rng = np.random.default_rng(0)
    # cau CHENH LECH manh giua cac kho -> de bat loi suc chua dung chung
    he_so = np.array([12.0, 4.0, 1.0])[:W, None]
    demand = rng.poisson(lam=np.broadcast_to(he_so, (W, S)),
                         size=(T, W, S)).astype(np.float32)
    cfg.setdefault("split_day", 150)
    return MultiWarehouseInventoryEnv(config=cfg, demand_data=demand,
                                      price_per_pair=price_per_pair,
                                      price_series=price_series, mode=mode)


# --------------------------------------------------------------------------- #
def test_khong_gian_quan_sat_khop_kich_thuoc():
    env = make_env()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (env.n_pairs, env.obs_per_pair)
    assert env.observation_space.contains(obs)


def test_quan_sat_luon_nam_trong_0_1():
    env = make_env()
    obs, _ = env.reset(seed=1)
    for _ in range(30):
        obs, _, te, tr, _ = env.step(env.action_space.sample())
        assert obs.min() >= 0.0 and obs.max() <= 1.0
        assert np.isfinite(obs).all()
        if te or tr:
            break


def test_bao_toan_hang_hoa():
    """[V2-5] ton_dau + nhap - tran - ban = ton_cuoi.

    Tran kho xay ra NGAY LUC NHAP (hang vuot suc chua bi tu choi), khong phai
    sau khi ban nhu ban cu.
    """
    env = make_env()
    env.reset(seed=2)
    for _ in range(30):
        inv_truoc = env.inventory.copy()
        nhap = env.pipeline_orders[0].copy()
        _, _, te, tr, info = env.step(env.action_space.sample())
        ban = float((info["demand_pairs"] - info["stockout_pairs"]).sum())
        du_kien = (float(inv_truoc.sum()) + float(nhap.sum())
                   - info["overflow"] - ban)
        assert abs(du_kien - float(env.inventory.sum())) < 1e-2
        if te or tr:
            break


def test_ton_kho_khong_am():
    env = make_env()
    env.reset(seed=3)
    for _ in range(40):
        _, _, te, tr, _ = env.step(env.action_space.sample())
        assert env.inventory.min() >= -1e-6
        if te or tr:
            break


def test_hanh_dong_0_khong_dat_hang():
    env = make_env()
    env.reset(seed=4)
    _, _, _, _, info = env.step(np.zeros(env.n_pairs, dtype=np.int64))
    assert info["order_qty"] == 0.0
    assert info["n_orders"] == 0
    assert info["cost_ordering"] == 0.0


# --------------------------------------------------------------------------- #
# [V2-1] Suc chua theo TUNG kho
# --------------------------------------------------------------------------- #
def test_suc_chua_khac_nhau_giua_cac_kho():
    """[V2-1] Kho ban nhieu phai co suc chua lon hon kho ban it.

    Ban cu dung MOT con so chung (trung binh cac kho) -> test nay that bai.
    """
    env = make_env()
    assert env.suc_chua_kho.shape == (env.n_warehouses,)
    assert env.suc_chua_kho[0] > env.suc_chua_kho[1] > env.suc_chua_kho[2]


def test_suc_chua_dung_bang_so_ngay_cau_cua_chinh_kho():
    env = make_env(capacity_cover_days=5.0)
    cau_kho = env.mean_demand.reshape(env.n_warehouses, env.n_skus).sum(axis=1)
    assert np.allclose(env.suc_chua_kho, cau_kho * 5.0, rtol=1e-5)


def test_tran_kho_chi_xay_ra_khi_vuot_suc_chua():
    env = make_env()
    env.reset(seed=5)
    for _ in range(30):
        _, _, te, tr, info = env.step(np.full(env.n_pairs, 5, dtype=np.int64))
        assert info["inventory"] <= env.suc_chua_kho.sum() + 1e-2
        if te or tr:
            break


def test_tran_kho_chi_tu_choi_hang_moi_khong_dung_ton_kho_cu():
    """[P0-1] Tran kho CHI duoc tu choi tu phan hang MOI VE vuot dung luong
    con trong, KHONG duoc "an" vao ton kho cu dang nam hop le trong suc chua.

    Kich ban P0 (pham_vi_khoa_luan.md muc 9): 1 kho 2 SKU, suc chua
    = 100. Ton kho cu (SKU0=90, SKU1=0). Hang moi ve = 20, toan bo o SKU1
    (SKU0=0, SKU1=20). Tong sau nhap = 110 -> vuot 10.
    Ky vong: CHI 10/20 (50%) hang MOI cua SKU1 bi tu choi; SKU0 (ton kho cu)
    khong doi.
    """
    T = 20
    demand_zero = np.zeros((T, 1, 2), dtype=np.float32)
    cfg = dict(CFG)
    cfg.update(n_warehouses=1, n_skus=2, warehouse_capacity=100.0,
              lead_time_min=1, lead_time_max=1, split_day=15, val_day=18)
    env = MultiWarehouseInventoryEnv(config=cfg, demand_data=demand_zero, mode="train")
    env.reset(seed=0)
    env.inventory = np.array([90.0, 0.0], dtype=np.float32)
    env.pipeline_orders[:] = 0.0
    env.pipeline_orders[0] = np.array([0.0, 20.0], dtype=np.float32)

    _, _, _, _, info = env.step(np.zeros(env.n_pairs, dtype=np.int64))

    assert np.isclose(info["overflow"], 10.0, atol=1e-3)
    assert np.isclose(env.inventory[0], 90.0, atol=1e-3), "ton kho cu khong duoc bi dung"
    assert np.isclose(env.inventory[1], 10.0, atol=1e-3), "20 hang moi ve, 10 bi tu choi"
    assert np.isclose(info["received_accepted"], 10.0, atol=1e-3)
    assert env.inventory.sum() <= env.suc_chua_kho.sum() + 1e-6
    # Bao toan dong chay: ton_cu + hang_moi = ton_sau_nhap + tran_kho (cau bang 0)
    assert np.isclose(90.0 + 20.0, env.inventory.sum() + info["overflow"], atol=1e-3)


# --------------------------------------------------------------------------- #
# [V2-3] Bang muc dat hang
# --------------------------------------------------------------------------- #
def test_bang_dat_hang_tang_nghiem_ngat():
    """[V2-3] Khong duoc co hai hanh dong cho cung mot luong dat hang.

    Ban cu dung np.round() -> voi SKU ban cham, bang thanh [0,0,1,1,2,3]:
    35,6% so cap chi con 2 muc phan biet duoc trong 6.
    """
    env = make_env()
    tbl = env.order_qty_table
    assert np.all(np.diff(tbl, axis=1) > 0), "bang muc dat hang co muc trung nhau"
    assert np.all(tbl[:, 0] == 0.0), "muc 0 phai la khong dat hang"
    for hang in tbl:
        assert len(np.unique(hang)) == env.n_action_levels


def test_muc_dat_hang_ti_le_voi_cau():
    env = make_env()
    md = env.mean_demand
    lon = np.argmax(md)
    nho = np.argmin(md)
    assert env.order_qty_table[lon, -1] > env.order_qty_table[nho, -1]


# --------------------------------------------------------------------------- #
# [V2-2] Chuan hoa phan thuong theo tung cap
# --------------------------------------------------------------------------- #
def test_phan_thuong_chuan_hoa_co_cung_do_lon():
    """[V2-2] Sau chuan hoa, cap ban nhanh va cap ban cham phai co phan thuong
    cung bac do lon. Truoc khi chuan hoa chung lech toi 4 bac -> critic dung
    chung trong so khong the khop, value loss ~1.100, gradient ap dao actor.
    """
    env = make_env()
    env.reset(seed=6)
    _, _, _, _, info = env.step(np.zeros(env.n_pairs, dtype=np.int64))
    tho = np.abs(info["local_rewards"])
    chuan = np.abs(info["local_scaled_rewards"])
    ty_le_tho = tho.max() / max(tho.min(), 1e-6)
    ty_le_chuan = chuan.max() / max(chuan.min(), 1e-6)
    assert ty_le_chuan < ty_le_tho
    assert chuan.max() < 50.0


def test_tat_chuan_hoa_thi_quay_ve_thang_cu():
    env = make_env(normalize_reward_per_pair=False)
    env.reset(seed=6)
    _, _, _, _, info = env.step(np.zeros(env.n_pairs, dtype=np.int64))
    assert np.allclose(info["local_scaled_rewards"],
                       info["local_rewards"] / 100.0, atol=1e-4)


# --------------------------------------------------------------------------- #
# [V2-4] Tin hieu cap kho trong quan sat
# --------------------------------------------------------------------------- #
def test_quan_sat_chua_muc_su_dung_suc_chua():
    """[V2-4] Ham thuong phat tran kho theo TONG ton kho ca kho, nen quan sat
    bat buoc phai chua thong tin do, neu khong moi truong mat tinh Markov.
    """
    env = make_env()
    obs, _ = env.reset(seed=7)
    i_util = 1 + 1 + env.lookback + env.lead_time_max + 1 + 1
    util_thuc = (env.inventory.reshape(env.n_warehouses, env.n_skus).sum(axis=1)
                 / env.suc_chua_kho)
    util_obs = obs[:, i_util].reshape(env.n_warehouses, env.n_skus)
    for w in range(env.n_warehouses):
        assert np.allclose(util_obs[w], min(util_thuc[w], 1.0), atol=1e-4)
        assert np.allclose(util_obs[w], util_obs[w][0])   # dung chung ca kho


def test_fill_rate_nam_trong_quan_sat():
    env = make_env()
    obs, _ = env.reset(seed=8)
    i_fr = 1 + 1 + env.lookback + env.lead_time_max
    for _ in range(15):
        obs, _, te, tr, _ = env.step(np.zeros(env.n_pairs, dtype=np.int64))
        assert np.allclose(obs[:, i_fr], np.clip(env.fill_rate_per_pair, 0, 1), atol=1e-4)
        if te or tr:
            break


def test_dac_trung_quy_mo_phan_biet_duoc_sku():
    env = make_env()
    obs, _ = env.reset(seed=9)
    i_scale = 1 + 1 + env.lookback + env.lead_time_max + 1
    assert obs[:, i_scale].std() > 1e-3


# --------------------------------------------------------------------------- #
def test_cong_thuc_chi_phi_dung():
    env = make_env()
    env.reset(seed=10)
    _, _, _, _, info = env.step(np.full(env.n_pairs, 2, dtype=np.int64))
    assert np.isclose(info["cost_holding"], env.cp_lk * env.inventory.sum(), atol=1e-3)
    assert np.isclose(info["cost_stockout"], env.cp_th * info["stockout"], atol=1e-3)
    assert np.isclose(info["cost_ordering"], env.cp_dh * info["n_orders"], atol=1e-3)
    assert np.isclose(info["cost_overflow"], env.pt_tk * info["overflow"], atol=1e-3)


def test_chi_phi_dat_hang_theo_tung_cap():
    """[FIX-1 cu] info['order_qty_pairs'] phai la MANG per-pair."""
    env = make_env()
    env.reset(seed=11)
    _, _, _, _, info = env.step(np.full(env.n_pairs, 3, dtype=np.int64))
    assert isinstance(info["order_qty_pairs"], np.ndarray)
    assert info["order_qty_pairs"].shape == (env.n_pairs,)
    assert info["n_orders"] == int((info["order_qty_pairs"] > 0).sum())


def test_tach_train_test_khong_chong_lan():
    env_tr = make_env()
    env_te = make_env()
    env_te.mode = "test"
    ngay_tr, ngay_te = [], []
    for i in range(25):
        env_tr.reset(seed=i); ngay_tr.append(env_tr.start_day)
        env_te.reset(seed=i); ngay_te.append(env_te.start_day)
    assert max(ngay_tr) + env_tr.episode_length <= env_tr.split_day
    assert min(ngay_te) >= env_te.split_day


def test_mean_demand_chi_uoc_luong_tu_mien_train():
    env = make_env()
    md_train = env.demand_data[:env.split_day].reshape(env.split_day, -1).mean(axis=0)
    assert np.allclose(env.mean_demand, np.maximum(md_train, env.min_mean_demand),
                       rtol=1e-4)


def test_do_dai_episode():
    env = make_env(episode_length=25)
    env.reset(seed=12)
    n = 0
    while True:
        _, _, te, tr, _ = env.step(np.zeros(env.n_pairs, dtype=np.int64))
        n += 1
        if te or tr:
            break
    assert n == 25


def test_reset_lap_lai_duoc():
    e1, e2 = make_env(), make_env()
    o1, _ = e1.reset(seed=123)
    o2, _ = e2.reset(seed=123)
    assert np.allclose(o1, o2)
    assert e1.start_day == e2.start_day


def test_info_co_du_lieu_cap_kho_cho_truc_quan():
    """[V2-7] App Streamlit can cac truong nay de ve mo phong tung ngay."""
    env = make_env()
    env.reset(seed=13)
    _, _, _, _, info = env.step(env.action_space.sample())
    for k in ("inv_wh", "cap_wh", "util_wh", "overflow_wh", "order_wh"):
        assert k in info and len(info[k]) == env.n_warehouses


# --------------------------------------------------------------------------- #
# [V3-1] Chia 3 mien train / val / test
# --------------------------------------------------------------------------- #
def test_ba_mien_train_val_test_khong_chong_lan():
    """[V3-1] train < val < test, khong ngay nao bi dung 2 lan."""
    L = 40  # episode_length cua CFG
    env_tr = make_env(split_day=100, val_day=150, mode="train")
    env_va = make_env(split_day=100, val_day=150, mode="val")
    env_te = make_env(split_day=100, val_day=150, mode="test")

    ngay_tr, ngay_va, ngay_te = [], [], []
    for i in range(30):
        env_tr.reset(seed=i); ngay_tr.append(env_tr.start_day)
        env_va.reset(seed=i); ngay_va.append(env_va.start_day)
        env_te.reset(seed=i); ngay_te.append(env_te.start_day)

    assert max(ngay_tr) + L <= 100
    assert min(ngay_va) >= 100 and max(ngay_va) + L <= 150
    assert min(ngay_te) >= 150


def test_val_day_mac_dinh_bang_split_day_tuong_thich_nguoc():
    """Khong khai bao val_day -> mode="test" hanh xu giong het ban v2 cu."""
    env = make_env(split_day=150)  # khong co val_day trong kw
    assert env.val_day == env.split_day


# --------------------------------------------------------------------------- #
# [V3-2] Chi phi thieu hang theo gia ban that
# --------------------------------------------------------------------------- #
def test_cp_th_pair_mac_dinh_la_hang_so_khi_tat_gia_that():
    env = make_env()
    assert np.allclose(env.cp_th_pair, env.cp_th)


def test_cp_th_pair_theo_gia_khi_bat_use_real_price_stockout():
    """[V3-2] Cap gia cao phai co don gia thieu hang cao hon cap gia thap,
    va reward_norm_pair van quy duoc SKU dat/re ve cung thang do (khong can
    thiet ke lai co che chuan hoa da co san o [V2-2])."""
    env = make_env()
    gia = np.linspace(1.0, 50.0, env.n_pairs).astype(np.float32)
    env2 = make_env(price_per_pair=gia, use_real_price_stockout=True,
                    margin_ratio=0.3, cp_th_min=0.5)

    assert np.isclose(env2.cp_th_pair[-1], gia[-1] * 0.3)
    assert np.all(env2.cp_th_pair >= 0.5)                  # san cp_th_min
    assert env2.cp_th_pair[-1] > env2.cp_th_pair[0]         # cap mac tien hon -> don gia cao hon

    # ty le lech gan chuan hoa van nho hon truoc chuan hoa, dung y [V2-2]
    env2.reset(seed=20)
    _, _, _, _, info = env2.step(np.zeros(env2.n_pairs, dtype=np.int64))
    tho = np.abs(info["local_rewards"])
    chuan = np.abs(info["local_scaled_rewards"])
    assert chuan.max() / max(chuan.min(), 1e-6) < tho.max() / max(tho.min(), 1e-6)


def test_phat_sla_khong_ty_le_theo_gia():
    """[V3-3] phi_phat_dv phai dung cp_th CO DINH, KHONG dung cp_th_pair -
    "phai dat 85% fill rate" la muc tieu quan tri, khong nen ty le theo gia
    SKU. Neu tinh sai (dung cp_th_pair), tat gia that (cp_th_pair nho) se lam
    hinh phat SLA yeu di dung bang ty le gia giam - day la bug da gap phai."""
    gia_thap = np.full(12, 1.0, dtype=np.float32)   # cp_th_pair ~ 1.0*0.3=0.3 (duoi san)
    env = make_env(price_per_pair=gia_thap, use_real_price_stockout=True,
                  margin_ratio=0.3, cp_th_min=0.5, muc_dv=0.85)
    env.reset(seed=21)
    fill_rate_thap = np.zeros(env.n_pairs, dtype=np.float32)  # ep shortfall toi da
    _, _, breakdown_metadata = env._calculate_reward(
        tk=np.zeros(env.n_pairs, dtype=np.float32),
        th=np.zeros(env.n_pairs, dtype=np.float32),
        sl_dh=np.zeros(env.n_pairs, dtype=np.float32),
        tran=np.zeros(env.n_pairs, dtype=np.float32),
        fill_rate=fill_rate_thap)
    expected = env.phi_dv * env.cp_th * env.muc_dv * env.mean_demand
    assert np.allclose(breakdown_metadata["cost_service_penalty"], expected.sum(), rtol=1e-4)


# --------------------------------------------------------------------------- #
# [V3-4] Tin hieu giam gia trong quan sat
# --------------------------------------------------------------------------- #
def test_tin_hieu_giam_gia_khong_bat_neu_thieu_co_include_price_signal():
    # [P0-7] price_signal_dim chi bat khi config.include_price_signal=True,
    # khong con suy ra tu viec truyen price_per_pair/price_series hay khong -
    # tranh am tham doi obs_per_pair (xem config.yaml, muc [P0-7]).
    env_khong_gia = make_env()
    env_co_du_lieu_gia = make_env(
        price_per_pair=np.full(12, 5.0, dtype=np.float32),
        price_series=np.full((200, 3, 4), 5.0, dtype=np.float32))
    assert env_co_du_lieu_gia.obs_per_pair == env_khong_gia.obs_per_pair


def test_tin_hieu_giam_gia_them_1_chieu_quan_sat_khi_bat_co():
    env_khong_gia = make_env()
    env_co_gia = make_env(price_per_pair=np.full(12, 5.0, dtype=np.float32),
                          price_series=np.full((200, 3, 4), 5.0, dtype=np.float32),
                          include_price_signal=True)
    assert env_co_gia.obs_per_pair == env_khong_gia.obs_per_pair + 1


def test_tin_hieu_giam_gia_phan_anh_dung_muc_giam():
    gia_tb = np.full(12, 10.0, dtype=np.float32)
    series = np.full((200, 3, 4), 10.0, dtype=np.float32)
    series[5] = 5.0     # ngay thu 5: giam gia 50%
    env = make_env(price_per_pair=gia_tb, price_series=series,
                   include_price_signal=True)
    env.reset(seed=0)
    env.start_day = 0
    env.current_step = 5
    obs = env._get_observation()
    i_gia = 1 + 1 + env.lookback + env.lead_time_max + 1 + 1 + 1 + 1
    assert np.allclose(obs[:, i_gia], 0.5, atol=1e-4)   # giam 50% -> tin hieu 0.5

    env.current_step = 6   # gia binh thuong tro lai
    obs2 = env._get_observation()
    assert np.allclose(obs2[:, i_gia], 0.0, atol=1e-4)


# --------------------------------------------------------------------------- #
# Script phan tich hau nghiem (scripts/policy_behavior.py, regime_analysis.py)
# --------------------------------------------------------------------------- #
sys.path.insert(0, str(ROOT / "scripts"))


def test_nhom_dac_trung_phu_kin_vector_quan_sat():
    """obs_groups phai phu DUNG moi chieu quan sat, khong trung, khong sot -
    neu _get_observation doi bo cuc ma quen sua, permutation importance se
    xao tron nham cot."""
    from policy_behavior import obs_groups
    for kw in [{}, {"include_price_signal": True}]:
        env = make_env(price_per_pair=np.ones(12, np.float32),
                       price_series=np.ones((200, 12), np.float32), **kw)
        cols = sorted(c for g in obs_groups(env).values() for c in g)
        assert cols == list(range(env.obs_per_pair))


def test_gan_nhan_giai_doan_chia_ba_khong_chong_lan():
    from regime_analysis import label_days
    rng = np.random.default_rng(1)
    demand = rng.poisson(5.0, size=(300, 3, 4)).astype(np.float32)
    groups, dev, _ = label_days(demand, None, 100, 300)
    vol = [groups[g] for g in ["Ổn định", "Trung bình", "Biến động"]]
    assert (sum(m.astype(int) for m in vol)[100:300] == 1).all()   # moi ngay test dung 1 nhom
    assert not any(m[:100].any() for m in vol)                     # khong gan nhan ngoai mien
    assert dev[groups["Biến động"]].min() > dev[groups["Ổn định"]].max()


# --------------------------------------------------------------------------- #
# [P2-*] Cac tuy chon cho ablation / hold-out
# --------------------------------------------------------------------------- #
def test_sku_indices_cat_dung_tap_con():
    env = make_env(sku_indices=[1, 3])
    full = make_env()
    assert env.n_skus == 2 and env.n_pairs == 6
    # cau trung binh cua cap (kho 0, SKU 1) phai giong ban day du
    assert np.isclose(env.mean_demand[0], full.mean_demand[1])
    env.reset(seed=0)
    env.step(np.zeros(env.n_pairs, dtype=int))


def test_phat_sla_normalized_bang_nhau_sau_chuan_hoa():
    """Che do normalized: phat SLA chia reward_norm_pair phai nhu nhau o moi cap."""
    env = make_env(service_penalty_mode="normalized")
    env.reset(seed=0)
    fr = np.full(env.n_pairs, 0.5, dtype=np.float32)
    z = np.zeros(env.n_pairs, dtype=np.float32)
    _, local, _ = env._calculate_reward(z, z, z, z, fr)
    per_norm = -local / env.reward_norm_pair
    assert np.allclose(per_norm, per_norm[0])
    # che do cu thi SKU cau nho bi phat nhe hon han
    env2 = make_env(service_penalty_mode="demand")
    _, local2, _ = env2._calculate_reward(z, z, z, z, fr)
    assert np.ptp(-local2 / env2.reward_norm_pair) > 0.1


def test_obs_drop_dat_nhom_bang_0_va_giu_kich_thuoc():
    env = make_env(obs_drop=["warehouse"])
    obs, _ = env.reset(seed=0)
    obs, *_ = env.step(np.full(env.n_pairs, 3))
    assert obs.shape[1] == make_env().obs_per_pair
    assert (obs[:, env.obs_groups["warehouse"]] == 0).all()
    with pytest.raises(ValueError):
        make_env(obs_drop=["khong_ton_tai"])


def test_warm_start_nap_lich_su_cau_that():
    env = make_env(warm_start_history=True)
    env.reset(seed=0)
    s0 = env.start_day
    kv = env.demand_data.reshape(env.demand_data.shape[0], -1)
    assert s0 >= env.lookback
    assert np.allclose(env.demand_history, kv[s0 - env.lookback:s0])


def test_reset_start_day_co_dinh():
    env = make_env()
    env.reset(seed=0, options={"start_day": 7})
    assert env.start_day == 7


def test_shared_trunk_cap_nhat_duoc():
    from agents.ppo_agent import PPOAgent
    from agents.rollout_buffer import RolloutBuffer
    env = make_env()
    ag = PPOAgent(env.obs_per_pair, env.n_pairs, env.n_action_levels,
                  config={"shared_trunk": True, "n_steps": 8, "mini_batch_size": 32})
    buf = RolloutBuffer(buffer_size=8, obs_per_pair=env.obs_per_pair, n_pairs=env.n_pairs)
    obs, _ = env.reset(seed=0)
    for _ in range(8):
        a, lp, v = ag.select_action(obs)
        nobs, r, te, tr, info = env.step(a)
        buf.add(obs, a, lp, info["local_scaled_rewards"], v)
        obs = nobs
    buf.compute_gae(last_values=ag.select_action(obs)[2], gamma=ag.gamma,
                    gae_lambda=ag.gae_lambda)
    before = [q.clone() for q in ag.network.parameters()]
    m = ag.update(buf)
    assert any(not torch_equal(b, q) for b, q in zip(before, ag.network.parameters()))
    assert m["grad_norm_actor"] > 0 and m["grad_norm_critic"] > 0


def torch_equal(a, b):
    return bool((a == b).all())


# --------------------------------------------------------------------------- #
# [P3-*] Giao thuc danh gia cuoi va ablation RQ3
# --------------------------------------------------------------------------- #
def test_reward_global_moi_cap_nhan_cung_tin_hieu():
    env = make_env(reward_mode="global")
    env.reset(seed=0)
    *_, info = env.step(np.full(env.n_pairs, 2))
    r = info["local_scaled_rewards"]
    assert np.allclose(r, r[0])
    expect = info["local_rewards"].sum() / env.reward_norm_pair.sum()
    assert np.isclose(r[0], expect, rtol=1e-5)


def test_test_full_horizon_chay_tron_mien_test():
    env = make_env(mode="test", test_full_horizon=True, split_day=100, val_day=150)
    for seed in (0, 1):
        env.reset(seed=seed)
        assert env.start_day == 150 and env.episode_length == 200 - 150
    # mien train khong bi anh huong
    assert make_env(test_full_horizon=True).episode_length == CFG["episode_length"]


def test_paired_test_dau_va_khoang_tin_cay():
    sys.path.insert(0, str(ROOT / "scripts"))
    from evaluate import paired_test
    a = np.array([90.0, 95, 100, 105, 110])
    r = paired_test(a, a + 10)
    assert r["mean_diff"] == -10 and r["n_ippo_cheaper"] == 5
    assert r["ci95_low"] <= -10 <= r["ci95_high"]


# --------------------------------------------------------------------------- #
# [P4-1] Baseline tinh chinh theo nhom quy mo cau
# --------------------------------------------------------------------------- #
def test_tham_so_baseline_theo_nhom_trai_dung_tung_cap():
    from baselines.traditional_policies import (demand_group_index, expand_group_params,
                                                SsPolicy, NewsvendorPolicy)
    md = np.array([0.1, 1.0, 5.0, 20.0, 0.3])
    gi = demand_group_index(md)
    assert gi.tolist() == [0, 1, 2, 3, 0]
    p = {str(g): {"service_level": 0.7 + 0.05 * g, "q_factor": 1.0, "lead_time": 3.0}
         for g in range(4)}
    kw = expand_group_params(p, gi)
    assert np.allclose(kw["service_level"], [0.7, 0.75, 0.8, 0.85, 0.7])
    s = SsPolicy(5, **kw)
    assert s.z.shape == (5,) and s.z[0] == s.z[4] and s.z[3] > s.z[0]
    n = NewsvendorPolicy(5, cr_override=np.array([0.6, 0.7, 0.8, 0.9, 0.6]))
    assert n.z_cr.shape == (5,)
    # tham so vo huong van chay nhu cu
    assert np.ndim(NewsvendorPolicy(5, cr_override=0.9).z_cr) == 0
