"""
scripts/data_preprocessing.py
================================
M5 Forecasting Dataset Preprocessing for RL Inventory Environment.

Downloads/loads the M5 Walmart dataset, samples 2 stores and N SKUs,
computes daily demand, handles missing values, and saves a clean
numpy array + metadata JSON for use by MultiWarehouseInventoryEnv.

M5 Dataset Files Required (place in data/raw/):
  - sales_train_evaluation.csv   (~100MB, all items × 1969 days)
  - calendar.csv                 (~30KB, date metadata)
  - sell_prices.csv              (~140MB, optional for cost info)

Kaggle download (if not done yet):
  kaggle competitions download -c m5-forecasting-accuracy
  Unzip to data/raw/

Output (saved to data/processed/):
  - demand_data.npy     shape (T, n_warehouses, n_skus)  float32
  - env_config.json     environment config dict
  - sku_metadata.csv    SKU names and store mapping

Usage:
  python scripts/data_preprocessing.py --stores CA_1 TX_1 --n_skus 30 --output_dir data/processed
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR  = ROOT / "data" / "raw"
OUT_DIR  = ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------

def load_m5_sales(
    raw_dir: Path,
    stores: List[str],
    n_skus: int = 30,
    seed: int = 42,
    chunk_size: int = 5000,
) -> pd.DataFrame:
    """
    Load and filter the M5 sales_train_evaluation.csv.

    Uses chunked reading to handle the large file (~100MB) without
    loading everything into RAM at once.

    Parameters
    ----------
    raw_dir : Path
        Directory containing M5 raw files.
    stores : list of str
        Store IDs to keep (e.g., ['CA_1', 'TX_1']).
        Available stores: CA_1, CA_2, CA_3, CA_4, TX_1, TX_2, TX_3,
                          WI_1, WI_2, WI_3.
    n_skus : int
        Number of unique SKUs to sample per store.
    seed : int
        Random seed for SKU sampling.
    chunk_size : int
        Rows per chunk for memory-efficient reading.

    Returns
    -------
    pd.DataFrame
        Filtered long-format dataframe with columns:
        [item_id, store_id, dept_id, cat_id, d_1 ... d_1969]
    """
    sales_path = raw_dir / "sales_train_evaluation.csv"
    if not sales_path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] M5 file not found: {sales_path}\n"
            "Please download from Kaggle:\n"
            "  kaggle competitions download -c m5-forecasting-accuracy\n"
            "  Unzip to data/raw/\n"
        )

    print(f"[Preprocessing] Reading M5 sales data (chunked)...")
    print(f"  File: {sales_path}")
    print(f"  Filtering stores: {stores}, SKUs per store: {n_skus}")

    chunks = []
    total_rows = 0

    for chunk in pd.read_csv(sales_path, chunksize=chunk_size):
        # Filter to target stores only
        filtered = chunk[chunk["store_id"].isin(stores)]
        if len(filtered) > 0:
            chunks.append(filtered)
        total_rows += len(chunk)

    print(f"  Total rows in M5: {total_rows:,}")

    if not chunks:
        raise ValueError(
            f"No rows found for stores {stores}. "
            f"Check store IDs (e.g., CA_1, TX_1, WI_1)."
        )

    df = pd.concat(chunks, ignore_index=True)
    print(f"  Rows after store filter: {len(df):,}")

    # Sample n_skus per store (reproducible)
    rng = np.random.default_rng(seed)
    sampled_parts = []
    for store in stores:
        store_df = df[df["store_id"] == store]
        items = store_df["item_id"].unique()
        if len(items) <= n_skus:
            sampled_parts.append(store_df)
        else:
            chosen = rng.choice(items, size=n_skus, replace=False)
            sampled_parts.append(store_df[store_df["item_id"].isin(chosen)])

    df_sampled = pd.concat(sampled_parts, ignore_index=True)
    print(f"  Rows after SKU sampling: {len(df_sampled):,}")
    print(f"  Unique items per store:")
    for store in stores:
        cnt = df_sampled[df_sampled["store_id"] == store]["item_id"].nunique()
        print(f"    {store}: {cnt} SKUs")

    return df_sampled


def load_calendar(raw_dir: Path) -> pd.DataFrame:
    """
    Load M5 calendar file for date alignment.

    Returns a DataFrame mapping d_1..d_1969 to real dates.
    """
    cal_path = raw_dir / "calendar.csv"
    if not cal_path.exists():
        print("[WARNING] calendar.csv not found. Skipping date alignment.")
        return None

    cal = pd.read_csv(cal_path, parse_dates=["date"])
    return cal


def compute_daily_demand(
    df: pd.DataFrame,
    stores: List[str],
    calendar: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Convert M5 wide-format sales to a 3D demand array.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered M5 data (item_id, store_id, d_1 ... d_T columns).
    stores : list of str
        Ordered list of stores (determines warehouse axis).
    calendar : pd.DataFrame or None
        Used to trim to a consistent date range.

    Returns
    -------
    demand_array : np.ndarray, shape (T, n_warehouses, n_skus), float32
        T = number of days; warehouses ordered per `stores` list.
    sku_meta : pd.DataFrame
        Metadata: item_id, store_id, pair_idx, dept_id, cat_id.
    """
    # Identify day columns
    day_cols = [c for c in df.columns if c.startswith("d_")]
    T = len(day_cols)
    print(f"[Preprocessing] Day columns: {len(day_cols)} days (d_1 to d_{T})")

    # Determine n_skus (ensure consistent across stores via reindexing)
    skus_per_store = {}
    for store in stores:
        items = sorted(df[df["store_id"] == store]["item_id"].unique().tolist())
        skus_per_store[store] = items

    # Find minimum SKU count across stores for alignment
    n_skus = min(len(v) for v in skus_per_store.values())
    print(f"[Preprocessing] Using {n_skus} SKUs per store (aligned)")

    n_w = len(stores)
    demand_array = np.zeros((T, n_w, n_skus), dtype=np.float32)
    meta_records = []

    for w_idx, store in enumerate(stores):
        store_df = df[df["store_id"] == store].reset_index(drop=True)
        items = skus_per_store[store][:n_skus]

        for s_idx, item_id in enumerate(items):
            row = store_df[store_df["item_id"] == item_id]
            if len(row) == 0:
                continue
            sales = row[day_cols].values.flatten().astype(np.float32)
            demand_array[:, w_idx, s_idx] = sales

            # Metadata
            meta_records.append({
                "pair_idx":  w_idx * n_skus + s_idx,
                "w_idx":     w_idx,
                "s_idx":     s_idx,
                "store_id":  store,
                "item_id":   item_id,
                "dept_id":   row["dept_id"].values[0] if "dept_id" in row else "",
                "cat_id":    row["cat_id"].values[0]  if "cat_id"  in row else "",
            })

    sku_meta = pd.DataFrame(meta_records)
    return demand_array, sku_meta


def clean_demand(demand_array: np.ndarray) -> np.ndarray:
    """
    Handle missing values and outliers in demand data.

    Steps:
      1. Replace NaN with 0 (missing sales = 0 demand assumed)
      2. Clip negative values to 0
      3. Clip extreme outliers (> 99th percentile × 3) to reduce noise
      4. Optional: forward-fill zeros in interior (if >3 consecutive zeros)

    Parameters
    ----------
    demand_array : np.ndarray, shape (T, n_w, n_s)

    Returns
    -------
    np.ndarray, same shape, cleaned
    """
    arr = demand_array.copy()

    # 1. NaN → 0
    nan_count = np.isnan(arr).sum()
    if nan_count > 0:
        print(f"[Preprocessing] Filling {nan_count} NaN values with 0")
    arr = np.nan_to_num(arr, nan=0.0)

    # 2. Clip negatives
    arr = np.clip(arr, 0.0, None)

    # 3. Clip extreme outliers per pair
    T, n_w, n_s = arr.shape
    for w in range(n_w):
        for s in range(n_s):
            series = arr[:, w, s]
            p99 = np.percentile(series[series > 0], 99) if (series > 0).any() else 1.0
            arr[:, w, s] = np.clip(series, 0.0, p99 * 3.0)

    print(f"[Preprocessing] Demand stats after cleaning:")
    print(f"  Global mean:  {arr.mean():.2f}")
    print(f"  Global std:   {arr.std():.2f}")
    print(f"  Global max:   {arr.max():.2f}")
    print(f"  Global min:   {arr.min():.2f}")
    print(f"  Non-zero %:   {(arr > 0).mean() * 100:.1f}%")

    return arr


def save_outputs(
    demand_array: np.ndarray,
    sku_meta: pd.DataFrame,
    stores: List[str],
    out_dir: Path,
    env_config_overrides: Optional[dict] = None,
) -> dict:
    """
    Save processed demand data and environment config to disk.

    Parameters
    ----------
    demand_array : np.ndarray, shape (T, n_w, n_s)
    sku_meta : pd.DataFrame
    stores : list of str
    out_dir : Path
    env_config_overrides : dict, optional — override default env config values

    Returns
    -------
    env_config : dict — saved to env_config.json
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    T, n_w, n_s = demand_array.shape

    # Build env config
    env_config = {
        "n_warehouses":   n_w,
        "n_skus":         n_s,
        "episode_length": 112,
        "lookback":       7,
        "lead_time_min":  1,
        "lead_time_max":  3,
        "max_inventory":  500,
        "initial_inventory": 100,
        "holding_cost":   1.0,
        "stockout_cost":  10.0,
        "ordering_cost":  50.0,
        "overflow_penalty": 5.0,
        "order_levels":   [0, 10, 20, 30, 40, 50],
        "stores":         stores,
        "n_days":         T,
        "seed":           42,
    }
    if env_config_overrides:
        env_config.update(env_config_overrides)

    # Save files
    np_path   = out_dir / "demand_data.npy"
    meta_path = out_dir / "sku_metadata.csv"
    cfg_path  = out_dir / "env_config.json"

    np.save(str(np_path), demand_array)
    sku_meta.to_csv(str(meta_path), index=False)
    with open(cfg_path, "w") as f:
        json.dump(env_config, f, indent=2)

    print(f"\n[Preprocessing] Saved outputs to {out_dir}:")
    print(f"  demand_data.npy   shape={demand_array.shape}  ({np_path.stat().st_size / 1024:.1f} KB)")
    print(f"  sku_metadata.csv  rows={len(sku_meta)}")
    print(f"  env_config.json")

    return env_config


def generate_synthetic_fallback(
    n_warehouses: int = 2,
    n_skus: int = 30,
    n_days: int = 800,
    seed: int = 42,
    out_dir: Optional[Path] = None,
) -> np.ndarray:
    """
    Generate synthetic demand data when M5 is unavailable.

    Creates realistic demand with:
      - Poisson base with random means (5–30 units/day)
      - Weekly seasonality (weekend peaks)
      - Occasional stockout days (demand = 0)

    Parameters
    ----------
    n_warehouses, n_skus, n_days : int
    seed : int
    out_dir : Path, optional — if set, saves the data

    Returns
    -------
    np.ndarray, shape (n_days, n_warehouses, n_skus)
    """
    print(f"\n[Preprocessing] Generating synthetic demand data...")
    print(f"  Shape: ({n_days}, {n_warehouses}, {n_skus})")
    rng = np.random.default_rng(seed)

    # Base means per pair
    base_means = rng.uniform(5, 30, size=(n_warehouses, n_skus))

    # Weekly seasonality weights (Mon=0 .. Sun=6)
    week_weights = np.array([0.8, 0.9, 1.0, 1.0, 1.1, 1.3, 1.2])

    demand = np.zeros((n_days, n_warehouses, n_skus), dtype=np.float32)
    for t in range(n_days):
        dow = t % 7
        seasonal_means = base_means * week_weights[dow]
        demand[t] = rng.poisson(seasonal_means).astype(np.float32)

    print(f"  Demand stats: mean={demand.mean():.2f}, std={demand.std():.2f}, max={demand.max():.2f}")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(out_dir / "demand_data.npy"), demand)

        # Build synthetic env_config
        config = {
            "n_warehouses": n_warehouses,
            "n_skus": n_skus,
            "episode_length": 112,
            "lookback": 7,
            "lead_time_min": 1,
            "lead_time_max": 3,
            "max_inventory": 500,
            "initial_inventory": 100,
            "holding_cost": 1.0,
            "stockout_cost": 10.0,
            "ordering_cost": 50.0,
            "overflow_penalty": 5.0,
            "order_levels": [0, 10, 20, 30, 40, 50],
            "n_days": n_days,
            "seed": seed,
            "synthetic": True,
        }
        with open(out_dir / "env_config.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"  Saved to {out_dir}")

    return demand


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess M5 Forecasting data for RL inventory env."
    )
    parser.add_argument(
        "--stores", nargs="+",
        default=["CA_1", "TX_1"],
        help="Store IDs to use as warehouses (default: CA_1 TX_1)",
    )
    parser.add_argument(
        "--n_skus", type=int, default=30,
        help="Number of SKUs to sample per store (default: 30)",
    )
    parser.add_argument(
        "--raw_dir", type=str, default=str(RAW_DIR),
        help=f"Directory with M5 raw CSVs (default: {RAW_DIR})",
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(OUT_DIR),
        help=f"Output directory (default: {OUT_DIR})",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Generate synthetic data even if M5 files exist (for testing)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)

    print("=" * 60)
    print("M5 Data Preprocessing for RL Inventory Environment")
    print("=" * 60)
    print(f"Stores (warehouses): {args.stores}")
    print(f"SKUs per store:      {args.n_skus}")
    print(f"Raw data dir:        {raw_dir}")
    print(f"Output dir:          {out_dir}")
    print("=" * 60)

    sales_path = raw_dir / "sales_train_evaluation.csv"
    if args.synthetic or not sales_path.exists():
        if not args.synthetic:
            print("\n[WARNING] M5 files not found in data/raw/. Using synthetic data.")
            print("  To use real M5 data, download from Kaggle and place files in data/raw/")
        demand_array = generate_synthetic_fallback(
            n_warehouses=len(args.stores),
            n_skus=args.n_skus,
            seed=args.seed,
            out_dir=out_dir,
        )
        # Create dummy metadata
        meta = pd.DataFrame([
            {
                "pair_idx": w * args.n_skus + s,
                "w_idx": w,
                "s_idx": s,
                "store_id": args.stores[w] if w < len(args.stores) else f"STORE_{w}",
                "item_id": f"ITEM_{s:03d}",
                "dept_id": "SYNTHETIC",
                "cat_id": "SYNTHETIC",
            }
            for w in range(len(args.stores))
            for s in range(args.n_skus)
        ])
        meta.to_csv(out_dir / "sku_metadata.csv", index=False)
        print("\nDone! Synthetic data ready.")
        return

    # --- Real M5 preprocessing ---
    df = load_m5_sales(raw_dir, args.stores, n_skus=args.n_skus, seed=args.seed)
    calendar = load_calendar(raw_dir)
    demand_array, sku_meta = compute_daily_demand(df, args.stores, calendar)
    demand_array = clean_demand(demand_array)
    env_config = save_outputs(demand_array, sku_meta, args.stores, out_dir)

    print("\n" + "=" * 60)
    print("Preprocessing COMPLETE!")
    print(f"Demand shape: {demand_array.shape}")
    print(f"Env config:   {env_config}")
    print("=" * 60)


if __name__ == "__main__":
    main()
