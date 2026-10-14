"""
tests/test_demo.py
==================
Kiem tra app demo (app/): du lieu rut gon nap duoc, mo phong chay dung hinh
dang, obs 44 chieu, va app khong co duong code nao goi huan luyen.
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


def test_app_khong_import_train():
    mau = re.compile(r"^\s*(from|import)\s+(scripts\.)?train\b|scripts[/\\.]train\b", re.M)
    vi_pham = [str(p) for p in APP.rglob("*.py") if mau.search(p.read_text(encoding="utf-8"))]
    assert not vi_pham, vi_pham
