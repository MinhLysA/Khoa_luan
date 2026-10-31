"""
scripts/forecast_demand.py
==========================
[Du bao] Du bao nhu cau 1-3 ngay toi cho moi cap bang mot mo hinh toan cuc LightGBM.

Tai ngay t (dau ngay, luc ra quyet dinh) chi dung nhu cau toi t-1: moi dac trung
lag / trung binh truot ket thuc o t-1; lich va gia la thong tin biet truoc.
Cot h = 0, 1, 2 cua mang ket qua la du bao cho ngay t, t+1, t+2.

Tranh lech phan phoi giua huan luyen va danh gia cua U1:
  * Mien train: du bao NGOAI MAU theo thoi gian (cua so mo rong): moi khoi ngay
    duoc du bao bang mo hinh chi hoc cac muc tieu truoc khoi do. Nam dau tien
    khong du du lieu nen dung trung binh truot 28 ngay.
  * Mien val: mo hinh hoc tren mien train, dung de chon sieu tham so.
  * Mien test: hoc lai tren train + val voi sieu tham so da chon.

Dau ra (results/):
    forecast_train_oof.npy, forecast_val.npy, forecast_test.npy   (ngay, cap, 3)
    forecast_sigma.npy   (toan bo ngay, cap): do lech chuan sai so 28 ngay truoc t
    forecast_metrics.json: MAE, RMSSE theo nhom cau va rieng mua le, so voi seasonal naive

Cach dung:
    python scripts/forecast_demand.py
"""
import argparse
import json

import numpy as np

from common import ROOT, load_config, load_data
from env.inventory_env import load_forecast
from baselines.traditional_policies import DEMAND_GROUP_NAMES, demand_group_index

HORIZONS = (0, 1, 2)
LAGS = (1, 7, 14, 28)
BURN_IN = 365                      # so ngay dau mien train khong du bao ngoai mau duoc
CAT_COLS = [15, 16]                # cot ma kho, ma SKU trong features()


def rolling_mean(D, k):
    """Trung binh k ngay ket thuc o t-1 cho moi t (nan khi chua du k ngay)."""
    cs = np.vstack([np.zeros((1, D.shape[1])), np.cumsum(D, axis=0)])
    out = np.full(D.shape, np.nan, dtype=np.float32)
    out[k:] = (cs[k:-1] - cs[:-k - 1]) / k
    return out


def features(D, cal, price, price_mean, md, snap_col, days, W=10):
    """Ma tran dac trung cho moi (ngay goc t, cap, h). D: (T, n_pairs). Khong
    dong toi D[t:], test_forecast_khong_nhin_tuong_lai kiem tra dieu nay."""
    T, P = D.shape
    S = P // W
    days = np.asarray(days)
    ma7, ma28 = rolling_mean(D, 7), rolling_mean(D, 28)
    cols = []
    for k in LAGS:
        lag = np.full((len(days), P), np.nan, dtype=np.float32)
        ok = days >= k
        lag[ok] = D[days[ok] - k]
        cols.append(lag)
    cols += [ma7[days], ma28[days]]
    tile_gia = price[days] / np.maximum(price_mean, 1e-6)
    cols.append(tile_gia)
    co_ban = np.stack(cols, axis=-1)                       # (n_ngay, P, 7)

    khoi = []
    for h in HORIZONS:
        muc_tieu = np.minimum(days + h, T - 1)
        c = cal[muc_tieu]
        lich = np.stack([np.full(len(days), h), (days + h) % 7, c[:, 8:20].argmax(axis=1)],
                        axis=1)                            # h, thu, thang
        su_kien = c[:, 4:8]
        snap = c[:, snap_col]                              # (n_ngay, P): SNAP bang cua kho
        ma_cap = np.stack([np.repeat(np.arange(W), S), np.tile(np.arange(S), W), md],
                          axis=1)
        x = np.concatenate([
            co_ban,
            np.repeat(np.concatenate([lich, su_kien], axis=1)[:, None], P, axis=1),
            snap[..., None],
            np.repeat(ma_cap[None], len(days), axis=0)], axis=-1)
        khoi.append(x.reshape(-1, x.shape[-1]))
    return np.concatenate(khoi).astype(np.float32)


def targets(D, days):
    T, P = D.shape
    days = np.asarray(days)
    return np.concatenate([D[np.minimum(days + h, T - 1)].reshape(-1) for h in HORIZONS])


def to_cube(pred, n_days, P):
    return np.clip(pred, 0, None).reshape(len(HORIZONS), n_days, P).transpose(1, 2, 0)


def fit(params, X, y, n_round, X_val=None, y_val=None):
    import lightgbm as lgb
    ds = lgb.Dataset(X, y, categorical_feature=CAT_COLS, free_raw_data=False)
    p = {"learning_rate": 0.05, "verbosity": -1, "seed": 42, "feature_pre_filter": False, **params}
    if X_val is None:
        return lgb.train(p, ds, n_round)
    return lgb.train(p, ds, n_round, valid_sets=[lgb.Dataset(X_val, y_val, reference=ds)],
                     valid_names=["val"], callbacks=[lgb.early_stopping(30, verbose=False)])


def metrics(y, f, naive, md, scale, mask, nhom):
    """y, f, naive: (n_ngay, P, 3). RMSSE tinh theo cap roi lay trung binh nhom."""
    def rmsse(p):
        return np.sqrt(((y - p) ** 2)[mask].reshape(-1, y.shape[1], 3).mean(axis=(0, 2))) / scale
    r_f, r_n = rmsse(f), rmsse(naive)
    a_f = np.abs(y - f)[mask].reshape(-1, y.shape[1], 3).mean(axis=(0, 2))
    a_n = np.abs(y - naive)[mask].reshape(-1, y.shape[1], 3).mean(axis=(0, 2))
    out = {}
    for g, ten in enumerate(DEMAND_GROUP_NAMES):
        m = nhom == g
        out[ten] = {"n_pairs": int(m.sum()), "mae": float(a_f[m].mean()), "mae_naive": float(a_n[m].mean()),
                    "rmsse": float(r_f[m].mean()), "rmsse_naive": float(r_n[m].mean())}
    out["Toàn bộ"] = {"n_pairs": len(nhom), "mae": float(a_f.mean()), "mae_naive": float(a_n.mean()),
                      "rmsse": float(r_f.mean()), "rmsse_naive": float(r_n.mean())}
    return out


def sai_so_truot(D, F, split, k=28):
    """Do lech chuan sai so du bao 1 ngay, chi dung sai so cua k ngay truoc t."""
    T, P = D.shape
    err1 = D - F[:, :, 0]
    cs1 = np.vstack([np.zeros((1, P)), np.cumsum(err1, axis=0)])
    cs2 = np.vstack([np.zeros((1, P)), np.cumsum(err1 ** 2, axis=0)])
    sigma = np.tile(D[:split].std(axis=0), (T, 1)).astype(np.float32)
    bien = ((cs2[k:-1] - cs2[:-k - 1]) / k - ((cs1[k:-1] - cs1[:-k - 1]) / k) ** 2)
    sigma[k:] = np.sqrt(np.maximum(bien, 0.0))
    return sigma


def du_bao_lai(cfg, data, dem):
    """(F, sigma) cho chuoi nhu cau da doi: mo hinh da hoc giu nguyen, chi dac trung tu lich
    su ban hang duoc tinh lai, nen du bao phan ung voi cu soc chi tu ngay hom sau.
    None neu chua co mo hinh du bao."""
    out = ROOT / cfg["paths"]["results_dir"]
    if not (out / "forecast_model.txt").exists():
        return None
    import lightgbm as lgb
    E = cfg["env"]
    T, W, S = dem.shape
    P = W * S
    D = dem.reshape(T, P).astype(np.float32)
    D0 = data["demand_data"].reshape(T, P)
    split, val = E["split_day"], E["val_day"]
    md = np.maximum(D0[:split].mean(axis=0), E.get("min_mean_demand", 0.05))
    bang = {"CA": 0, "TX": 1, "WI": 2}
    snap_col = np.repeat([bang[str(s).split("_")[0]] for s in E["warehouse_stores"]], S)
    te = np.arange(val, T)
    X = features(D, data["calendar_features"], data["price_series"].reshape(T, P).astype(np.float32),
                 data["price_per_pair"].reshape(-1), md, snap_col, te, W)
    F, _ = load_forecast(out)
    F = F.copy()
    F[te] = to_cube(lgb.Booster(model_file=str(out / "forecast_model.txt")).predict(X), len(te), P)
    return F, sai_so_truot(D, F, split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--folds", type=int, default=4)
    a = ap.parse_args()

    cfg = load_config(a.config)
    data, E = load_data(cfg), cfg["env"]
    T, W, S = data["demand_data"].shape
    P = W * S
    D = data["demand_data"].reshape(T, P).astype(np.float32)
    cal = data["calendar_features"]
    price = data["price_series"].reshape(T, P).astype(np.float32)
    price_mean = data["price_per_pair"].reshape(-1)
    split, val = E["split_day"], E["val_day"]
    md = np.maximum(D[:split].mean(axis=0), E.get("min_mean_demand", 0.05))
    bang = {"CA": 0, "TX": 1, "WI": 2}
    snap_col = np.repeat([bang[str(s).split("_")[0]] for s in E["warehouse_stores"]], S)

    def X_of(days, m=md):
        return features(D, cal, price, price_mean, m, snap_col, days, W)

    def goc(lo, hi):
        """Ngay goc t co moi muc tieu t+h con nam trong [lo, hi)."""
        return np.arange(max(lo, 28), hi - max(HORIZONS))

    # 1. Chon sieu tham so tren val: hoc tren train, muc tieu khong vuot split_day
    tr = goc(0, split)
    X_tr, y_tr = X_of(tr), targets(D, tr)
    chon = goc(split, val)                         # muc tieu khong vuot sang mien test
    X_ch, y_ch = X_of(chon), targets(D, chon)
    va = np.arange(split, val)
    X_va = X_of(va)
    luoi = [{"objective": o, "num_leaves": n}
            for o in ("regression", "tweedie") for n in (31, 63)]
    tot, best_iter, best_err = None, a.rounds, np.inf
    for p in luoi:
        m = fit(p, X_tr, y_tr, a.rounds, X_ch, y_ch)
        err = np.sqrt(np.mean((np.clip(m.predict(X_ch), 0, None) - y_ch) ** 2))
        print(f"  {p} iter={m.best_iteration} rmse_val={err:.4f}", flush=True)
        if err < best_err:
            tot, best_iter, best_err = p, m.best_iteration or a.rounds, err
    print(f"Chon {tot}, {best_iter} vong")

    # 2. Mien val: mo hinh chi hoc tren train
    m_tr = fit(tot, X_tr, y_tr, best_iter)
    F_val = to_cube(m_tr.predict(X_va), len(va), P)

    # 3. Mien train: ngoai mau theo thoi gian, nam dau dung trung binh truot
    ma28 = rolling_mean(D, 28)
    F_oof = np.repeat(np.nan_to_num(ma28[:split], nan=0.0)[..., None], 3, axis=2)
    cuts = np.linspace(BURN_IN, split, a.folds + 1).astype(int)
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        hoc = goc(0, lo)
        md_lo = np.maximum(D[:lo].mean(axis=0), E.get("min_mean_demand", 0.05))
        m = fit(tot, X_of(hoc, md_lo), targets(D, hoc), best_iter)
        khoi = np.arange(lo, hi)
        F_oof[lo:hi] = to_cube(m.predict(X_of(khoi, md_lo)), len(khoi), P)
        print(f"  OOF [{lo}, {hi}) xong", flush=True)

    # 4. Mien test: hoc lai tren train + val
    hoc = goc(0, val)
    m_all = fit(tot, X_of(hoc), targets(D, hoc), best_iter)
    te = np.arange(val, T)
    F_test = to_cube(m_all.predict(X_of(te)), len(te), P)

    # 5. Sai so du bao truot: chi dung sai so cua cac ngay truoc t
    F = np.concatenate([F_oof, F_val, F_test]).astype(np.float32)
    sigma = sai_so_truot(D, F, split)

    out = ROOT / cfg["paths"]["results_dir"]
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "forecast_train_oof.npy", F[:split])
    np.save(out / "forecast_val.npy", F[split:val])
    np.save(out / "forecast_test.npy", F[val:])
    np.save(out / "forecast_sigma.npy", sigma)
    m_all.save_model(str(out / "forecast_model.txt"))

    # 6. Do chinh xac tren val, test va rieng mua le trong test
    nhom = demand_group_index(md)
    scale = np.sqrt((np.diff(D[:split], axis=0) ** 2).mean(axis=0))
    scale = np.where(scale > 0, scale, 1.0)     # cap khong ban gi tren mien train
    thang = cal[:, 8:20].argmax(axis=1) + 1

    def do(ngay):
        ngay = ngay[ngay < T - 2]
        y = np.stack([D[ngay + h] for h in HORIZONS], axis=-1)
        naive = np.stack([D[ngay + h - 7] for h in HORIZONS], axis=-1)
        mask = np.ones(y.shape, bool)
        return metrics(y, F[ngay], naive, md, scale, mask, nhom)

    ket_qua = {"params": tot, "n_round": int(best_iter), "val": do(va), "test": do(te),
               "test_mua_le": do(te[np.isin(thang[te], [11, 12])])}
    (out / "forecast_metrics.json").write_text(
        json.dumps(ket_qua, indent=2, ensure_ascii=False), encoding="utf-8")
    for ten in ("val", "test", "test_mua_le"):
        print(ten, {k: f"{v['rmsse']:.3f}/{v['rmsse_naive']:.3f}" for k, v in ket_qua[ten].items()})
    print(f"Da luu du bao vao {out}")


if __name__ == "__main__":
    main()
