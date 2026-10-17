"""
tests/test_demo.py
==================
Kiem tra app demo (app/): du lieu rut gon nap duoc, mo phong chay dung hinh
dang, obs 44 chieu, kich ban soc cau (chi suy luan), va app khong co duong code
nao goi huan luyen.
Can app/demo_data/ (tao bang `python run.py demo_data`).
"""
import re
import sys
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP))

import core  # noqa: E402


def test_demo_data_nap_duoc():
    meta, arr = core.load_demo()
    W, S = len(meta["stores"]), len(meta["skus"])
    n = meta["n_days_total"] - meta["val_day"]
    assert arr["demand"].shape == arr["price"].shape == (n, W, S)
    for name in meta["policies"]:
        assert core.rollout(name)["order"].shape == (n, W * S)


def test_rollout_mot_ngay_dung_shape():
    s = core.state_at("IPPO", 1)
    assert s["action"].shape == (300,)
    assert s["qty"].shape == (300,)


def test_obs_44_chieu():
    meta, _ = core.load_demo()
    env = core.base_env()
    assert env.obs_per_pair == 44
    assert core.state_at("IPPO", 0)["obs"].shape == (300, 44)
    assert len(core.obs_labels(env, meta)) == 44


def test_mo_phong_truc_tiep_khop_rollout_da_chay_san():
    # Rollout san duoc chay tren env dung tu data/processed day du; mo phong
    # trong app dung env dung lai tu du lieu rut gon -> phai ra cung hanh dong.
    for day in (0, 3, 30):
        for name in ("IPPO", "(s,S)"):
            assert np.array_equal(core.state_at(name, day)["qty"], core.rollout(name)["order"][day])


def test_soc_cau_he_so_1_khop_rollout():
    # He so 1 = khong soc -> phai ra dung rollout da chay san (lam moc "tang them")
    for name in ("IPPO", "(s,S)"):
        s, r = core.shock_rollout(name, 200, 14, 1.0), core.rollout(name)
        assert np.allclose(s["stockout"], r["stockout"].sum(axis=1))
        assert np.allclose(s["order"], r["order"].sum(axis=1))


def test_soc_cau_chi_nhan_trong_cua_so_va_pham_vi():
    # x2 tren kho K02 x nganh HOBBIES, ngay [100, 110): nhu cau toan mang tang dung bang
    # cau cu cua cac cap do trong cua so, ngoai ra khong doi
    meta, arr = core.load_demo()
    b = core.shock_rollout("EOQ", 0, 0, 1.0)
    s = core.shock_rollout("EOQ", 100, 10, 2.0, kho=("K02",), nganh=("HOBBIES",))
    hob = [i for i, k in enumerate(meta["skus"]) if k.startswith("HOBBIES")]
    them = np.zeros(len(b["demand"]))
    them[100:110] = arr["demand"][100:110, 1][:, hob].sum(axis=1)
    assert np.allclose(s["demand"] - b["demand"], them)


def test_chi_so_soc_ngay_ve_lai():
    # Cau 10/ngay; ngay 20..24 thieu 5/ngay -> fill rate truot 7 ngay < 85% tu ngay 22,
    # ve lai >= 85% ngay 29 (thieu trong 7 ngay con 10/70) -> 9 ngay ke tu luc soc bat dau
    dem, khong = np.full(60, 10.0), np.zeros(60)
    so = khong.copy()
    so[20:25] = 5
    b = {"demand": dem, "stockout": khong, "cost": khong}
    m = core.shock_metrics({"demand": dem, "stockout": so, "cost": so}, b, 20, 5, 0.85)
    assert (m["ve_lai"], m["trong"], m["truoc"], m["thieu_them"], m["chi_phi_them"]) == (9, 0.5, 1.0, 25, 25)
    # Thieu 1/ngay: khong xuong duoi muc tieu -> None; thieu 5/ngay toi het: chua ve lai -> inf
    it = {"demand": dem, "stockout": np.r_[np.zeros(20), np.ones(40)], "cost": khong}
    mai = {"demand": dem, "stockout": np.r_[np.zeros(20), np.full(40, 5.0)], "cost": khong}
    assert core.shock_metrics(it, b, 20, 5, 0.85)["ve_lai"] is None
    assert core.shock_metrics(mai, b, 20, 5, 0.85)["ve_lai"] == float("inf")


def test_app_khong_import_train():
    mau = re.compile(r"^\s*(from|import)\s+(scripts\.)?train\b|scripts[/\\.]train\b", re.M)
    tep = list(APP.rglob("*.py"))
    assert APP / "pages_" / "soc_cau.py" in tep          # trang kich ban soc cau cung duoc quet
    vi_pham = [str(p) for p in tep if mau.search(p.read_text(encoding="utf-8"))]
    assert not vi_pham, vi_pham
