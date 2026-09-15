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


def make_env(**kw):
    cfg = dict(CFG); cfg.update(kw)
    T, W, S = 200, cfg["n_warehouses"], cfg["n_skus"]
    rng = np.random.default_rng(0)
    # cau CHENH LECH manh giua cac kho -> de bat loi suc chua dung chung
    he_so = np.array([12.0, 4.0, 1.0])[:W, None]
    demand = rng.poisson(lam=np.broadcast_to(he_so, (W, S)),
                         size=(T, W, S)).astype(np.float32)
    cfg["split_day"] = 150
    return MultiWarehouseInventoryEnv(config=cfg, demand_data=demand, mode="train")


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
