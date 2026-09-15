"""
scripts/data_preprocessing.py
==============================
Tien xu ly du lieu nhu cau (Demand Data).

PHIEN BAN v2:
  [V2-17] SUA LOI CHONG CHI SO TRONG calendar_features.
          Ban cu: no_event ghi vao cot 3 - dung cot cua "Sporting"; month ghi
          tu cot 7 - dung cot cua "Religious". Ket qua: cot 3 co 1.799 ngay
          (gop no_event + Sporting), cot 7 gop Religious + thang Mot, va cot 19
          luon bang 0. Agent nhan tin hieu lich sai suot qua trinh huan luyen.
          Nay bo cuc dung: [0-2] SNAP, [3-7] event one-hot, [8-19] thang.

  [V2-18] LOC BO CAC SKU GAN NHU KHONG BAN.
          Trong tap 50 SKU phan tang co 142/500 cap (28%) cau trung binh
          < 0,1 dv/ngay - mat hang cua hang do khong kinh doanh. Cac "tac tu"
          nay chi them nhieu vao gradient chung. Tham so --min_mean_demand loai
          cac SKU khong dat nguong o DA SO cua hang.

  [V2-19] Metadata luu them ten SKU, ten kho, cau trung binh tung cap - app
          Streamlit dung de hien thi thay vi chi so 0..N.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_data(n_days=365, n_warehouses=4, n_skus=40, seed=42):
    """Sinh du lieu nhu cau gia lap (Poisson + mua vu theo ngay trong tuan)."""
    rng = np.random.default_rng(seed)
    base_rates = rng.uniform(2.0, 25.0, size=(n_warehouses, n_skus))
    demand = np.zeros((n_days, n_warehouses, n_skus), dtype=np.float32)
    for day in range(n_days):
        seasonality = 1.3 if (day % 7) in (5, 6) else 0.9
        demand[day] = rng.poisson(lam=base_rates * seasonality)
    return demand


def generate_calendar_features(raw_dir: str, n_days: int = 1941) -> np.ndarray:
    """[V2-17] Vector dac trung lich (n_days, 20) - bo cuc DUNG:

        [0]     snap_CA
        [1]     snap_TX
        [2]     snap_WI
        [3]     no_event
        [4]     Sporting
        [5]     Cultural
        [6]     National
        [7]     Religious
        [8-19]  thang 1..12 (one-hot)
    """
    cal = pd.read_csv(Path(raw_dir) / "calendar.csv").iloc[:n_days].reset_index(drop=True)
    EVENT_TYPES = ["Sporting", "Cultural", "National", "Religious"]
    features = np.zeros((n_days, 20), dtype=np.float32)

    for i, row in cal.iterrows():
        features[i, 0] = float(row["snap_CA"])
        features[i, 1] = float(row["snap_TX"])
        features[i, 2] = float(row["snap_WI"])

        et = row["event_type_1"] if pd.notna(row["event_type_1"]) else None
        if et is None and pd.notna(row.get("event_type_2")):
            et = row["event_type_2"]
        if et is not None and et in EVENT_TYPES:
            features[i, 4 + EVENT_TYPES.index(et)] = 1.0   # cot 4..7
        else:
            features[i, 3] = 1.0                           # cot 3 = no_event

        features[i, 8 + (int(row["month"]) - 1)] = 1.0     # cot 8..19

    print(f"  Calendar features shape: {features.shape}")
    print(f"  SNAP CA/TX/WI: {int(features[:,0].sum())}/{int(features[:,1].sum())}"
          f"/{int(features[:,2].sum())}")
    print(f"  Ngay khong su kien: {int(features[:,3].sum())} | co su kien: "
          f"{int(features[:,4:8].sum())}")
    print(f"  Tong one-hot thang: {int(features[:,8:20].sum())} (phai = {n_days})")
    return features


def preprocess_m5_data(raw_dir, n_warehouses=10, n_skus=30,
                       sku_selection="stratified", min_mean_demand=0.2,
                       split_day=1450):
    """Doc M5 Kaggle -> mang (T, n_warehouses, n_skus)."""
    raw_dir = Path(raw_dir)
    sales_path = raw_dir / "sales_train_evaluation.csv"
    print(f"Dang doc {sales_path} ({sales_path.stat().st_size / 1e6:.1f} MB)...")

    df = pd.read_csv(sales_path)
    day_cols = [c for c in df.columns if c.startswith("d_")]
    print(f"  -> {len(df)} items x {len(day_cols)} ngay")

    store_sales = df.groupby("store_id")[day_cols].sum().sum(axis=1)
    top_stores = store_sales.nlargest(n_warehouses).index.tolist()
    print(f"  Chon {n_warehouses} cua hang: {top_stores}")
    df_f = df[df["store_id"].isin(top_stores)].copy()

    # [V2-18] Chi giu SKU co cau du lon o DA SO cua hang (tren mien huan luyen)
    train_cols = day_cols[:split_day]
    per_store_mean = (df_f.groupby(["item_id", "store_id"])[train_cols]
                      .sum().mean(axis=1).unstack())
    du_dieu_kien = (per_store_mean >= min_mean_demand).sum(axis=1)
    hop_le = du_dieu_kien[du_dieu_kien >= int(np.ceil(0.8 * n_warehouses))].index
    print(f"  SKU dat nguong cau >= {min_mean_demand} dv/ngay o >=80% cua hang: "
          f"{len(hop_le)}/{df_f['item_id'].nunique()}")

    item_total = df_f[df_f["item_id"].isin(hop_le)].groupby("item_id")[train_cols].sum().sum(axis=1)

    if sku_selection == "stratified":
        cat_of = df_f.drop_duplicates("item_id").set_index("item_id")["cat_id"]
        cats = sorted(cat_of.reindex(item_total.index).unique())
        top_items, per_cat = [], max(1, n_skus // len(cats))
        for ci, cat in enumerate(cats):
            items_c = item_total[cat_of.reindex(item_total.index) == cat]
            items_c = items_c[items_c > 0].sort_values(ascending=False)
            if len(items_c) == 0:
                continue
            k = per_cat if ci < len(cats) - 1 else n_skus - len(top_items)
            k = min(k, len(items_c))
            idx = np.linspace(0, len(items_c) - 1, k).round().astype(int)
            top_items += items_c.index[idx].tolist()
        top_items = top_items[:n_skus]
        print(f"  Chon {len(top_items)} SKU phan tang tren {len(cats)} nhom nganh hang")
    else:
        top_items = item_total.nlargest(n_skus).index.tolist()
        print(f"  Chon {len(top_items)} SKU ban chay nhat (che do 'top')")

    df_f = df_f[df_f["item_id"].isin(top_items)].copy()

    T = len(day_cols)
    demand = np.zeros((T, n_warehouses, n_skus), dtype=np.float32)
    for w_idx, store in enumerate(top_stores):
        sdf = df_f[df_f["store_id"] == store].set_index("item_id").reindex(top_items)
        demand[:, w_idx, :] = sdf[day_cols].values.astype(np.float32).T

    demand = np.nan_to_num(demand, nan=0.0)

    flat = demand[:split_day].reshape(split_day, -1)
    md = flat.mean(axis=0)
    print(f"  Demand array shape: {demand.shape}")
    print(f"  Cau TB/cap: {md.mean():.2f}  trung vi: {np.median(md):.2f}  "
          f"min: {md.min():.2f}  max: {md.max():.1f}")
    print(f"  Ty le ngay bang 0: {(demand == 0).mean():.1%}")
    print(f"  So cap co cau < 0.1 dv/ngay: {(md < 0.1).sum()} (cang it cang tot)")
    return demand, top_stores, top_items


def main():
    p = argparse.ArgumentParser(description="Tien xu ly du lieu nhu cau ton kho.")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--m5", action="store_true")
    p.add_argument("--n_days", type=int, default=365)
    p.add_argument("--n_warehouses", type=int, default=10)
    p.add_argument("--n_skus", type=int, default=30)
    p.add_argument("--raw_dir", type=str, default="data/raw")
    p.add_argument("--output_dir", type=str, default="data/processed")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split_day", type=int, default=1450)
    p.add_argument("--min_mean_demand", type=float, default=0.2)
    p.add_argument("--sku_selection", type=str, default="stratified",
                   choices=["stratified", "top"])
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = Path(args.output_dir) / "demand_data.npy"

    if args.m5:
        print("=== CHE DO: M5 Walmart Dataset ===")
        demand, stores, items = preprocess_m5_data(
            args.raw_dir, args.n_warehouses, args.n_skus,
            args.sku_selection, args.min_mean_demand, args.split_day)
        print("Dang tao calendar features...")
        cal_feat = generate_calendar_features(args.raw_dir, n_days=demand.shape[0])
        np.save(Path(args.output_dir) / "calendar_features.npy", cal_feat)
        sd = min(args.split_day, demand.shape[0])
        md = demand[:sd].reshape(sd, -1).mean(axis=0)
        meta = {"mode": "m5", "n_days": int(demand.shape[0]),
                "n_warehouses": args.n_warehouses, "n_skus": args.n_skus,
                "stores": stores, "top_items": items,
                "calendar_features_dim": 20,
                "split_day": args.split_day,
                "min_mean_demand": args.min_mean_demand,
                "sku_selection": args.sku_selection,
                # [V2-19] cho app Streamlit
                "mean_demand_pairs": [round(float(x), 4) for x in md],
                "mean_demand_per_warehouse": [round(float(x), 2) for x in
                                              md.reshape(args.n_warehouses, args.n_skus).sum(1)]}
    elif args.synthetic:
        print("=== CHE DO: Synthetic Data ===")
        demand = generate_synthetic_data(args.n_days, args.n_warehouses,
                                         args.n_skus, args.seed)
        cal_feat = np.zeros((demand.shape[0], 0), dtype=np.float32)
        np.save(Path(args.output_dir) / "calendar_features.npy", cal_feat)
        md = demand.reshape(demand.shape[0], -1).mean(axis=0)
        meta = {"mode": "synthetic", "n_days": int(demand.shape[0]),
                "n_warehouses": args.n_warehouses, "n_skus": args.n_skus,
                "stores": [f"WH_{i+1}" for i in range(args.n_warehouses)],
                "top_items": [f"SKU_{j+1}" for j in range(args.n_skus)],
                "calendar_features_dim": 0,
                "split_day": min(args.split_day, int(demand.shape[0] * 0.75)),
                "mean_demand_pairs": [round(float(x), 4) for x in md],
                "mean_demand_per_warehouse": [round(float(x), 2) for x in
                                              md.reshape(args.n_warehouses, args.n_skus).sum(1)]}
    else:
        print("Vui long them --synthetic hoac --m5")
        return

    np.save(out_path, demand)
    print(f"Da luu: {out_path}  shape={demand.shape}")
    with open(Path(args.output_dir) / "env_config.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
    print(f"Da luu metadata: {Path(args.output_dir) / 'env_config.json'}")


if __name__ == "__main__":
    main()
