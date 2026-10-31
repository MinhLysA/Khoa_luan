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


def test_config_ablation_dong_bo_voi_config_chinh():
    """configs/*.yaml phai dung bang config.yaml + dung MOT thay doi. Neu fail:
    chay `python configs/make_configs.py` sau khi sua config.yaml."""
    import copy, yaml
    sys.path.insert(0, str(ROOT / "configs"))
    from make_configs import ABLATIONS
    base = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    for name, (_, fn) in ABLATIONS.items():
        cfg = copy.deepcopy(base)
        fn(cfg)
        assert yaml.safe_load(open(ROOT / "configs" / f"{name}.yaml", encoding="utf-8")) == cfg, \
            f"configs/{name}.yaml lech voi config.yaml - chay python configs/make_configs.py"


# --------------------------------------------------------------------------- #
# Mo rong: baseline cung thong tin, du bao, base-stock, nhom tham so, Lagrange
# --------------------------------------------------------------------------- #
def _trang_thai_chay(env, n=8):
    from baselines.traditional_policies import build_env_state
    env.reset(seed=0)
    for _ in range(n):
        env.step(np.full(env.n_pairs, 1))
    return build_env_state(env)


def test_ss_biet_suc_chua_khong_vuot_phan_con_trong():
    from baselines.traditional_policies import CapacityAwareSs
    env = make_env()
    st = _trang_thai_chay(env)
    pol = CapacityAwareSs(env.n_pairs, kappa=0.9, service_level=0.99, q_factor=3.0, lead_time=5.0,
                          ordering_cost=5.0, holding_cost=1.0)
    q = env.action_to_qty(pol.get_action(st)).reshape(env.n_warehouses, env.n_skus)
    ton = (st["inventory"] + st["pipeline_orders"].sum(axis=0)).reshape(q.shape).sum(axis=1)
    room = np.maximum(0.9 * env.suc_chua_kho - ton, 0.0)
    # sai so luong tu hoa: toi da nua buoc lon nhat cua bang muc dat moi cap
    buoc = np.diff(env.order_qty_table, axis=1).max(axis=1).reshape(q.shape).sum(axis=1)
    assert (q.sum(axis=1) <= room + 0.5 * buoc).all()


def test_ss_biet_suc_chua_rong_cho_trung_ss():
    from baselines.traditional_policies import CapacityAwareSs, SsPolicy
    env = make_env(warehouse_capacity=1e9)
    st = _trang_thai_chay(env)
    kw = dict(service_level=0.9, q_factor=1.0, lead_time=3.0, ordering_cost=5.0, holding_cost=1.0)
    assert np.array_equal(CapacityAwareSs(env.n_pairs, **kw).get_action(st),
                          SsPolicy(env.n_pairs, **kw).get_action(st))


def _du_bao(env, T=200):
    F = np.zeros((T, env.n_pairs, 3), dtype=np.float32)
    F[:] = env.mean_demand[None, :, None]
    return F, np.ones((T, env.n_pairs), dtype=np.float32)


def test_newsvendor_du_bao_tra_dung_ngay():
    from baselines.traditional_policies import ForecastNewsvendor, build_env_state
    env = make_env()
    F, sigma = _du_bao(env)
    F[:] = 0.0
    env.reset(seed=0, options={"start_day": 10})
    F[10] = 7.0
    pol = ForecastNewsvendor(env.n_pairs, F, sigma * 0, cr_override=0.5, lead_time=4.0)
    env.inventory[:] = 0.0
    q = env.action_to_qty(pol.get_action(build_env_state(env)))
    # 5 ngay bao phu x 7/ngay = 35, lam tron ve muc gan nhat cua bang
    assert np.array_equal(q, env.action_to_qty(env.qty_to_action(np.full(env.n_pairs, 35.0))))


def test_quan_sat_du_bao_them_2_chieu_va_nam_trong_0_1():
    goc = make_env()
    F, sigma = _du_bao(goc)
    env = MultiWarehouseInventoryEnv(config=goc.config | {"include_forecast": True},
                                     demand_data=goc.demand_data, forecast=(F, sigma), mode="train")
    assert env.obs_per_pair == goc.obs_per_pair + 2
    obs, _ = env.reset(seed=0)
    c = env.obs_groups["forecast"]
    assert obs.shape[1] == env.obs_per_pair and (obs[:, c] > 0).all() and (obs[:, c] <= 1).all()
    assert list(range(env.obs_per_pair)) == sorted(i for v in env.obs_groups.values() for i in v)


def test_base_stock_z_co_dinh_trung_ss():
    from baselines.traditional_policies import BaseStockActions, SsPolicy, build_env_state
    env = make_env()
    st = _trang_thai_chay(env)
    pol = SsPolicy(env.n_pairs, service_level=0.9, lead_time=3.0, q_factor=1.0,
                   ordering_cost=5.0, holding_cost=1.0)
    bs = BaseStockActions(env.n_pairs, z_levels=[float(pol.z)] * 6, lead_time=3.0, q_factor=1.0)
    a = bs(np.zeros(env.n_pairs, dtype=int), env)
    assert np.array_equal(a, pol.get_action(st))


def _thu_thap(env, ag, buf, n=8):
    obs, _ = env.reset(seed=0)
    for _ in range(n):
        a, lp, v = ag.select_action(obs)
        nobs, r, te, tr, info = env.step(ag.env_action(a, env))
        buf.add(obs, a, lp, info["local_scaled_rewards"], v)
        obs = nobs
    buf.compute_gae(last_values=ag.select_action(obs)[2], gamma=ag.gamma, gae_lambda=ag.gae_lambda)


def test_u5_z_co_dinh_trung_newsvendor_du_bao():
    from baselines.traditional_policies import BaseStockActions, ForecastNewsvendor, build_env_state
    env = make_env()
    env.reset(seed=0)
    T = env.demand_data.shape[0] if env.demand_data is not None else 2000
    rng = np.random.default_rng(0)
    env.forecast = rng.uniform(0, 5, (T, env.n_pairs, 3)).astype(np.float32)
    env.forecast_sigma = rng.uniform(0, 2, (T, env.n_pairs)).astype(np.float32)
    bao = BaseStockActions(env.n_pairs, [0.0, 0.5], lead_time=3.0, du_bao=True)
    a = bao(np.ones(env.n_pairs, dtype=int), env)
    pol = ForecastNewsvendor(env.n_pairs, env.forecast, env.forecast_sigma, lead_time=3.0)
    pol.z = 0.5
    assert np.array_equal(a, pol.get_action(build_env_state(env)))


def test_buffer_u2_luu_chi_so_z_goc():
    from agents.ppo_agent import PPOAgent
    from agents.rollout_buffer import RolloutBuffer
    env = make_env()
    ag = PPOAgent(env.obs_per_pair, env.n_pairs, env.n_action_levels,
                  config={"action_mode": "base_stock", "mini_batch_size": 32,
                          "base_stock": {"z_levels": [-0.5, 0, 0.5, 1, 1.5, 2.5]}})
    buf = RolloutBuffer(buffer_size=8, obs_per_pair=env.obs_per_pair, n_pairs=env.n_pairs)
    obs, _ = env.reset(seed=0)
    a, lp, v = ag.select_action(obs)
    buf.add(obs, a, lp, np.zeros(env.n_pairs), v)
    assert np.array_equal(buf.actions[0], a)         # luu hanh dong goc, khong phai muc dat
    assert ag.env_action(a, env).shape == a.shape


def test_grouped_mlp_k1_trung_mang_chinh():
    import torch
    from agents.ppo_agent import GroupedMLP, _mlp
    net = _mlp(9, 16, 6, 0.01)
    g = GroupedMLP(1, 12, 9, 16, 6, 0.01)
    for l, i in enumerate((0, 2, 4)):
        g.w[l].data[0] = net[i].weight.data
        g.b[l].data[0] = net[i].bias.data
    x = torch.rand(5, 12, 9)
    assert torch.allclose(g(x), net(x), atol=1e-6)
    assert g(x[0]).shape == (12, 6)                  # luc chon hanh dong khong co truc buoc


def test_u4_k_mang_cap_nhat_duoc():
    from agents.ppo_agent import PPOAgent, pair_groups
    from agents.rollout_buffer import RolloutBuffer
    env = make_env()
    ag = PPOAgent(env.obs_per_pair, env.n_pairs, env.n_action_levels,
                  config={"param_groups": 3, "batch_mode": "time", "mini_batch_size": 48})
    ag.set_groups(pair_groups(3, env))
    assert ag.network.actor.group_of.tolist() == pair_groups(3, env).tolist()
    buf = RolloutBuffer(buffer_size=8, obs_per_pair=env.obs_per_pair, n_pairs=env.n_pairs)
    _thu_thap(env, ag, buf)
    trc = [q.clone() for q in ag.network.parameters()]
    ag.update(buf)
    assert any(not torch_equal(a, b) for a, b in zip(trc, ag.network.parameters()))
    with pytest.raises(ValueError):
        PPOAgent(env.obs_per_pair, env.n_pairs, env.n_action_levels, config={"param_groups": 3})


def test_buffer_lay_mau_theo_nhom_giu_tron_nhom():
    from agents.rollout_buffer import RolloutBuffer
    env = make_env()
    buf = RolloutBuffer(buffer_size=6, obs_per_pair=2, n_pairs=12, group_size=4)
    for t in range(6):
        s = np.zeros((12, 2), dtype=np.float32)
        s[:, 0] = t                                  # danh dau buoc
        s[:, 1] = np.repeat(np.arange(3), 4)         # danh dau kho
        buf.add(s, np.zeros(12, int), np.zeros(12), np.zeros(12), np.zeros(12))
    buf.compute_gae(np.zeros(12))
    tong = 0
    for b in buf.get_batches(batch_size=16, mode="warehouse"):
        assert b["states"].shape[1:] == (4, 2) and b["actions"].shape[1:] == (4,)
        assert (b["states"][:, :, 0] == b["states"][:, :1, 0]).all()
        assert (b["states"][:, :, 1] == b["states"][:, :1, 1]).all()
        tong += b["states"].shape[0]
    assert tong == 18
    assert all(b["states"].shape[1:] == (12, 2) for b in buf.get_batches(batch_size=24, mode="time"))
    assert all(b["states"].ndim == 2 for b in buf.get_batches(batch_size=24))


def test_critic_ngu_canh_bat_bien_hoan_vi_sku():
    import torch
    from agents.ppo_agent import ContextCritic
    c = ContextCritic(5, 4, 16)
    x = torch.rand(2, 4, 5)
    p = torch.tensor([2, 0, 3, 1])
    assert torch.allclose(c(x)[:, p], c(x[:, p]), atol=1e-6)     # V cua tung SKU di theo SKU do
    assert c(x.reshape(8, 5)).shape == (8, 1)                    # lo phang tu select_action


def test_u3_cap_nhat_duoc():
    from agents.ppo_agent import PPOAgent
    from agents.rollout_buffer import RolloutBuffer
    env = make_env()
    ag = PPOAgent(env.obs_per_pair, env.n_pairs, env.n_action_levels,
                  config={"critic_context": "warehouse", "batch_mode": "warehouse",
                          "warehouse_skus": env.n_skus, "mini_batch_size": 16})
    buf = RolloutBuffer(buffer_size=8, obs_per_pair=env.obs_per_pair, n_pairs=env.n_pairs,
                        group_size=env.n_skus)
    _thu_thap(env, ag, buf)
    assert ag.update(buf)["grad_norm_critic"] > 0


def test_lagrange_lambda_tang_khi_fill_thap_va_giu_trong_chan():
    env = make_env(lagrange_per_pair=True, fill_rate_window=5,
                   lagrange=dict(kp=1.0, ki=1.0, kd=0.0, lambda_max=4.0))
    assert env.obs_per_pair == make_env().obs_per_pair + 1
    obs, _ = env.reset(seed=0)
    assert np.allclose(env.lam, env.phi_dv)
    assert np.allclose(obs[:, env.obs_groups["lagrange"]], env.phi_dv / 4.0)
    for _ in range(30):                              # khong dat hang -> het hang -> fill thap
        obs, *_, info = env.step(np.zeros(env.n_pairs, dtype=int))
    assert (info["lambda_pairs"][env.mean_demand > 1] > env.phi_dv).all()
    assert info["lambda_pairs"].max() <= 4.0


def test_lagrange_giu_qua_tap_train_luu_checkpoint_va_khoi_phuc_khi_danh_gia(tmp_path):
    from agents.ppo_agent import PPOAgent
    cfg = dict(lagrange_per_pair=True, fill_rate_window=5,
               lagrange=dict(kp=1.0, ki=1.0, kd=0.0, lambda_max=4.0))
    khong_dat = lambda e: e.step(np.zeros(e.n_pairs, dtype=int))
    env = make_env(**cfg)
    env.reset(seed=0)
    for _ in range(30):
        khong_dat(env)
    lam = env.lam.copy()
    assert not np.allclose(lam, env.phi_dv)
    env.reset(seed=1)
    assert np.allclose(env.lam, lam)                 # mien train: nhan tu giu qua cac tap

    ag = PPOAgent(env.obs_per_pair, env.n_pairs, env.n_action_levels, config={})
    ag.lagrange_env = env
    ag.save(str(tmp_path / "m.pth"))
    te = make_env(mode="test", **cfg)
    ag.load(str(tmp_path / "m.pth"), env=te)
    for s in (0, 1):                                 # moi lan danh gia cung diem xuat phat
        te.reset(seed=s)
        assert np.allclose(te.lam, lam)
        for _ in range(30):
            khong_dat(te)


def test_duong_bien_quet_lai_khi_van_tay_doi(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import compare_same_service as iso
    env, goi = make_env(), []
    monkeypatch.setattr(iso, "chay", lambda *a, **k: goi.append(1) or (1.0, 0.9))
    luoi = {"(s,S)": (iso.SsPolicy, [dict(service_level=0.9, lead_time=2.0)])}
    quet = lambda vt: iso.quet_duong_bien(env, env, luoi, 1, 1, 0, tmp_path / "bien.json", vt)
    quet("a")
    n = len(goi)
    assert n and (quet("a"), len(goi))[1] == n       # cung van tay: dung lai cache
    quet("b")
    assert len(goi) == 2 * n


def test_chon_theo_nhom_moi_nhom_lay_cau_hinh_re_nhat_theo_gia_bong():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import compare_same_service as iso
    C = np.array([[1.0, 10.0], [5.0, 2.0]])          # (cau hinh, nhom): chi phi
    T = np.array([[9.0, 9.0], [0.0, 0.0]])           # cau hinh 0 thieu nhieu hang hon
    assert iso.chon_theo_nhom(C, T, [0.0]) == {(0, 1)}   # khong phat thieu hang: re nhat tung nhom
    assert iso.chon_theo_nhom(C, T, [100.0]) == {(1, 1)}  # gia bong cao: ca hai nhom giu hang


def test_chon_phi_dung_cac_lan_danh_gia_val_khac_nhau(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import run_experiments as campaign
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    for p, f in zip(campaign.PHI, (0.70, 0.86, 0.90, 0.93, 0.97)):
        (tmp_path / "results" / f"train_log_penalty{p}_trial.csv").write_text(
            f"episode,eval_fill\n1,\n2,{f}\n3,{f}\n", encoding="utf-8")
    assert campaign.chon_phi() == ["10", "60"]
    (tmp_path / "results" / "train_log_x.csv").write_text(
        "episode,eval_fill\n1,0.5\n2,0.5\n3,0.5\n4,0.9\n", encoding="utf-8")
    assert campaign.fill_val_cuoi("x") == pytest.approx(0.7)   # 0.5 lap lai chi tinh mot lan


def test_chi_phi_van_hanh_theo_cap_khong_gom_phat_sla():
    env = make_env()
    env.reset(seed=0)
    *_, info = env.step(np.zeros(env.n_pairs, dtype=int))
    assert info["op_cost_pairs"].sum() == pytest.approx(
        info["cost_total"] - info["cost_service_penalty"], rel=1e-5)


def test_lagrange_tat_giu_nguyen_phat():
    env = make_env()
    env.reset(seed=0)
    assert env.lam == env.phi_dv


def test_forecast_khong_nhin_tuong_lai():
    from forecast_demand import features
    rng = np.random.default_rng(0)
    T, P = 80, 20
    D = rng.poisson(2.0, size=(T, P)).astype(np.float32)
    cal = np.zeros((T, 20), dtype=np.float32)
    cal[:, 8] = 1.0
    price = np.ones((T, P), dtype=np.float32)
    md, snap = np.ones(P), np.zeros(P, dtype=int)
    t = 50
    a = features(D, cal, price, price[0], md, snap, [t], W=2)
    D2 = D.copy()
    D2[t:] += 100.0                                  # doi nhu cau tu ngay t tro di
    b = features(D2, cal, price, price[0], md, snap, [t], W=2)
    assert np.array_equal(a, b)
    D3 = D.copy()
    D3[t - 1] += 100.0                               # ngay t-1 phai co tac dung
    assert not np.array_equal(a, features(D3, cal, price, price[0], md, snap, [t], W=2))


def test_duong_bien_noi_suy_va_chon_cung_muc():
    from compare_same_service import duong_test, doc_tai, chon_cung_muc, pareto
    diem = [{"val_fill": f, "val_cost": c, "test_fill": f, "test_cost": c, "params": {"x": f}}
            for f, c in [(0.86, 100.0), (0.90, 120.0), (0.94, 160.0)]]
    assert len(pareto(diem + [{"val_fill": 0.87, "val_cost": 130.0}])) == 3
    f, c = duong_test(diem)
    assert doc_tai(f, c, 0.92) == pytest.approx(140.0)
    assert doc_tai(f, c, 0.99) is None
    r = chon_cung_muc(diem, 110.0, 0.90, 0.90)
    assert r["noi_suy"] is False and r["delta_ippo_percent"] == pytest.approx(100 * (110 / 120 - 1))
    r = chon_cung_muc(diem, 110.0, 0.90, 0.92)       # val khop nhung test lech -> noi suy
    assert r["noi_suy"] is True and r["cost"] == pytest.approx(140.0)
    r = chon_cung_muc(diem, 110.0, 0.90, 0.91)       # noi suy van tra ve cau hinh that gan nhat
    assert r["params"] == {"x": 0.90} and r["fill_cau_hinh"] == 0.90
    assert chon_cung_muc(diem, 110.0, 0.90, 0.99).get("khong_dat")


def test_cung_muc_noi_suy_chi_dung_cau_hinh_du_gan(tmp_path, monkeypatch):
    import json
    import common
    env = make_env()
    ket_qua = {
        "(s,S)": {"cost": 1.0, "fill": 0.90, "noi_suy": True, "fill_cau_hinh": 0.895,
                  "params": {"service_level": 0.9, "q_factor": 1.0, "lead_time": 2.0}},
        "EOQ": {"cost": 1.0, "fill": 0.90, "noi_suy": True, "fill_cau_hinh": 0.88,
                "params": {"q_factor": 1.0, "lead_time": 2.0}},
        "Newsvendor": {"cost": None, "khong_dat": True},
    }
    (tmp_path / "same_service_main_s42.json").write_text(json.dumps({"ket_qua": ket_qua}), encoding="utf-8")
    monkeypatch.setattr(common, "load_agent", lambda *a: None)
    pols = common.make_policies({"paths": {"results_dir": str(tmp_path)}}, env, None, du_bao=False)
    assert "(s,S) (cùng mức PV)" in pols
    assert "EOQ (cùng mức PV)" not in pols and "Newsvendor (cùng mức PV)" not in pols


def test_nguon_cung_chung_gioi_han_toan_mang_va_tin_hieu():
    env = make_env(supply_cover=0.5)
    goc = make_env()
    assert env.obs_per_pair == goc.obs_per_pair + 1 and env.obs_groups["supply"]
    obs, _ = env.reset(seed=0)
    assert np.all(obs[:, env.obs_groups["supply"][0]] == 1.0)
    a = np.full(env.n_pairs, env.n_action_levels - 1)
    obs, _, _, _, inf = env.step(a)
    don = inf["order_qty_pairs"].reshape(env.n_warehouses, env.n_skus)
    assert np.all(don.sum(axis=0) <= env.nguon_cung + 1e-3)
    assert inf["order_requested"] > inf["order_qty"]
    # moi kho bi cat cung ty le tren cung mat hang
    cot = obs[:, env.obs_groups["supply"][0]].reshape(env.n_warehouses, env.n_skus)
    assert np.allclose(cot, cot[0]) and cot.max() < 1.0
    assert np.allclose(goc.action_to_qty(a).reshape(don.shape) * cot, don, atol=1e-3)


def test_khong_bat_nguon_cung_thi_khong_cat_don():
    env = make_env()
    env.reset(seed=0)
    *_, inf = env.step(np.full(env.n_pairs, env.n_action_levels - 1))
    assert inf["order_requested"] == pytest.approx(inf["order_qty"]) and "supply" not in env.obs_groups


def test_tron_chinh_sach_theo_nhom():
    from hybrid_policy import tron
    fn = tron(lambda o, e: np.zeros(3, int), lambda o, e: np.ones(3, int), np.array([True, False, True]))
    assert fn(None, None).tolist() == [0, 1, 0]


def test_du_bao_lai_phan_ung_voi_soc_tu_ngay_hom_sau(tmp_path):
    pytest.importorskip("lightgbm")
    from forecast_demand import du_bao_lai, features, fit, targets, sai_so_truot, HORIZONS
    rng = np.random.default_rng(0)
    T, W, S = 120, 2, 2
    P = W * S
    muc = 2.0 + 8.0 * (np.arange(T) // 15 % 2)        # muc cau doi theo khoi de mo hinh hoc tu lich su
    D = rng.poisson(muc[:, None, None], size=(T, W, S)).astype(np.float32)
    cal = np.zeros((T, 20), dtype=np.float32)
    cal[:, 8] = 1.0
    data = dict(demand_data=D, calendar_features=cal, price_per_pair=np.ones((W, S), dtype=np.float32),
                price_series=np.ones((T, W, S), dtype=np.float32))
    E = dict(split_day=60, val_day=90, warehouse_stores=["CA_1", "TX_1"])
    cfg = dict(env=E, paths=dict(results_dir=str(tmp_path)))
    Dp, md = D.reshape(T, P), np.full(P, 5.0)
    snap = np.repeat([0, 1], S)
    ngay = np.arange(28, 90 - max(HORIZONS))
    m = fit({"objective": "regression"}, features(Dp, cal, np.ones((T, P)), np.ones(P), md, snap, ngay, W),
            targets(Dp, ngay), 20)
    m.save_model(str(tmp_path / "forecast_model.txt"))
    F = np.full((T, P, 3), 5.0, dtype=np.float32)
    for ten, x in (("train_oof", F[:60]), ("val", F[60:90]), ("test", F[90:]), ("sigma", np.ones((T, P)))):
        np.save(tmp_path / f"forecast_{ten}.npy", x)
    sot = D.copy()
    sot[100:] *= 3.0
    F2, sigma2 = du_bao_lai(cfg, data, sot)
    F0, sigma0 = du_bao_lai(cfg, data, D)
    assert np.array_equal(F2[:101], F0[:101])         # ngay 100 chua thay soc
    assert not np.allclose(F2[101:], F0[101:])
    assert sigma2[115:].mean() > sigma0[115:].mean()
    assert du_bao_lai(dict(cfg, paths=dict(results_dir=str(tmp_path / "khong"))), data, sot) is None


def test_bootstrap_phan_tang_dau_dung():
    from multi_seed_stats import bootstrap_phan_tang
    C_base = np.ones(70) * 100
    C_ippo = np.stack([C_base * 0.9, C_base * 0.92, C_base * 0.88])
    r = bootstrap_phan_tang(C_ippo, C_base, 7, 200, np.random.default_rng(0))
    assert r["tb_percent"] == pytest.approx(-10.0)
    assert r["co_y_nghia"] and r["ci95_high"] < 0


# --------------------------------------------------------------------------- #
# [P5-1] SNAP theo bang cua kho
# --------------------------------------------------------------------------- #
def test_snap_chi_hien_co_cua_bang_chua_kho():
    cal = np.zeros((200, 20), dtype=np.float32)
    cal[:, 0:3] = 1.0                                  # ca 3 bang deu dang SNAP
    cfg = dict(CFG, split_day=150, warehouse_stores=["CA_1", "TX_2", "WI_3"])
    T, W, S = 200, cfg["n_warehouses"], cfg["n_skus"]
    demand = np.ones((T, W, S), dtype=np.float32)
    env = MultiWarehouseInventoryEnv(config=cfg, demand_data=demand,
                                     calendar_features=cal, mode="train")
    obs, _ = env.reset(seed=0)
    c0 = env.obs_groups["calendar"][0] + 7
    snap = obs[:, c0:c0 + 3].reshape(W, S, 3)
    assert np.array_equal(snap, np.broadcast_to(np.eye(3)[:, None, :], (W, S, 3)))
    # khong co warehouse_stores -> giu hanh vi cu (ca 3 co)
    cfg.pop("warehouse_stores")
    env_cu = MultiWarehouseInventoryEnv(config=cfg, demand_data=demand,
                                        calendar_features=cal, mode="train")
    assert (env_cu.reset(seed=0)[0][:, c0:c0 + 3] == 1.0).all()


def test_warehouse_stores_khop_thu_tu_kho_cua_du_lieu():
    """config.yaml phai liet ke kho dung thu tu truc kho cua demand_data."""
    import json, yaml
    meta = ROOT / "data" / "processed" / "env_config.json"
    if not meta.exists():
        pytest.skip("chua co du lieu da tien xu ly")
    stores = json.loads(meta.read_text(encoding="utf-8"))["stores"]
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    assert cfg["env"]["warehouse_stores"] == stores
