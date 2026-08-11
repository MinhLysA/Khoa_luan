"""
scripts/data_preprocessing.py
================================
Tiền xử lý Bộ dữ liệu M5 Forecasting cho Môi trường Tồn kho RL.

Tải/đọc dữ liệu M5 Walmart, lấy mẫu 2 cửa hàng và N mặt hàng (SKU),
tính toán nhu cầu hàng ngày, xử lý các giá trị thiếu và lưu mảng numpy sạch
+ tệp metadata JSON phục vụ cho MultiWarehouseInventoryEnv.

Các file dữ liệu M5 bắt buộc (đặt trong data/raw/):
  - sales_train_evaluation.csv   (~100MB, tất cả mặt hàng × 1969 ngày)
  - calendar.csv                 (~30KB, dữ liệu thời gian)
  - sell_prices.csv              (~140MB, tùy chọn cho thông tin chi phí)

Tải xuống từ Kaggle (nếu chưa có):
  kaggle competitions download -c m5-forecasting-accuracy
  Giải nén vào data/raw/

Đầu ra (lưu tại data/processed/):
  - demand_data.npy     dạng (T, n_warehouses, n_skus)  float32
  - env_config.json     dict cấu hình môi trường
  - sku_metadata.csv    tên SKU và ánh xạ cửa hàng

Cách sử dụng:
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
# Đường dẫn
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR  = ROOT / "data" / "raw"
OUT_DIR  = ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Hàm tiền xử lý chính
# ---------------------------------------------------------------------------

def load_m5_sales(
    raw_dir: Path,
    stores: List[str],
    n_skus: int = 30,
    seed: int = 42,
    chunk_size: int = 5000,
) -> pd.DataFrame:
    """
    Tải và lọc dữ liệu M5 sales_train_evaluation.csv.

    Sử dụng đọc theo từng khối (chunked reading) để xử lý tệp lớn (~100MB) mà không
    tải toàn bộ vào RAM cùng một lúc.

    Tham số
    ------
    raw_dir : Path
        Thư mục chứa các tệp thô M5.
    stores : list của str
        Danh sách ID cửa hàng cần giữ lại (ví dụ: ['CA_1', 'TX_1']).
        Các cửa hàng hiện có: CA_1, CA_2, CA_3, CA_4, TX_1, TX_2, TX_3,
                              WI_1, WI_2, WI_3.
    n_skus : int
        Số lượng SKU duy nhất cần lấy mẫu trên mỗi cửa hàng.
    seed : int
        Hạt giống ngẫu nhiên cho việc lấy mẫu SKU.
    chunk_size : int
        Số dòng mỗi khối để tiết kiệm bộ nhớ khi đọc.

    Trả về
    -----
    pd.DataFrame
        Dataframe định dạng dài đã được lọc với các cột:
        [item_id, store_id, dept_id, cat_id, d_1 ... d_1969]
    """
    sales_path = raw_dir / "sales_train_evaluation.csv"
    if not sales_path.exists():
        raise FileNotFoundError(
            f"\n[LỖI] Không tìm thấy file M5: {sales_path}\n"
            "Vui lòng tải về từ Kaggle:\n"
            "  kaggle competitions download -c m5-forecasting-accuracy\n"
            "  Giải nén vào data/raw/\n"
        )

    print(f"[Tiền xử lý] Đang đọc dữ liệu bán hàng M5 (theo khối)...")
    print(f"  Tệp: {sales_path}")
    print(f"  Lọc cửa hàng: {stores}, số SKU mỗi cửa hàng: {n_skus}")

    chunks = []
    total_rows = 0

    for chunk in pd.read_csv(sales_path, chunksize=chunk_size):
        # Lọc duy nhất các cửa hàng mục tiêu
        filtered = chunk[chunk["store_id"].isin(stores)]
        if len(filtered) > 0:
            chunks.append(filtered)
        total_rows += len(chunk)

    print(f"  Tổng số dòng trong M5: {total_rows:,}")

    if not chunks:
        raise ValueError(
            f"Không tìm thấy dòng nào cho các cửa hàng {stores}. "
            f"Hãy kiểm tra lại ID cửa hàng (ví dụ: CA_1, TX_1, WI_1)."
        )

    df = pd.concat(chunks, ignore_index=True)
    print(f"  Số dòng sau khi lọc cửa hàng: {len(df):,}")

    # Lấy mẫu n_skus mỗi cửa hàng (có thể tái lập)
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
    print(f"  Số dòng sau khi lấy mẫu SKU: {len(df_sampled):,}")
    print(f"  Số mặt hàng duy nhất mỗi cửa hàng:")
    for store in stores:
        cnt = df_sampled[df_sampled["store_id"] == store]["item_id"].nunique()
        print(f"    {store}: {cnt} SKUs")

    return df_sampled


def load_calendar(raw_dir: Path) -> pd.DataFrame:
    """
    Tải file lịch M5 calendar.csv để dóng hàng thời gian.

    Trả về DataFrame ánh xạ d_1..d_1969 sang ngày thực tế.
    """
    cal_path = raw_dir / "calendar.csv"
    if not cal_path.exists():
        print("[CẢNH BÁO] Không tìm thấy calendar.csv. Bỏ qua căn chỉnh ngày.")
        return None

    cal = pd.read_csv(cal_path, parse_dates=["date"])
    return cal


def compute_daily_demand(
    df: pd.DataFrame,
    stores: List[str],
    calendar: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Chuyển đổi dữ liệu M5 dạng ngang sang mảng nhu cầu 3D.

    Tham số
    ------
    df : pd.DataFrame
        Dữ liệu M5 đã lọc (các cột item_id, store_id, d_1 ... d_T).
    stores : list của str
        Danh sách cửa hàng sắp xếp (quy định trục nhà kho).
    calendar : pd.DataFrame hoặc None
        Dùng để cắt theo khoảng ngày thống nhất.

    Trả về
    -----
    demand_array : np.ndarray, shape (T, n_warehouses, n_skus), float32
        T = số ngày; các nhà kho được sắp xếp theo danh sách `stores`.
    sku_meta : pd.DataFrame
        Metadata: item_id, store_id, pair_idx, dept_id, cat_id.
    """
    # Xác định các cột ngày
    day_cols = [c for c in df.columns if c.startswith("d_")]
    T = len(day_cols)
    print(f"[Tiền xử lý] Các cột ngày: {len(day_cols)} ngày (d_1 tới d_{T})")

    # Xác định n_skus (đảm bảo đồng nhất giữa các cửa hàng qua reindexing)
    skus_per_store = {}
    for store in stores:
        items = sorted(df[df["store_id"] == store]["item_id"].unique().tolist())
        skus_per_store[store] = items

    # Tìm số lượng SKU nhỏ nhất giữa các cửa hàng để đồng bộ
    n_skus = min(len(v) for v in skus_per_store.values())
    print(f"[Tiền xử lý] Sử dụng {n_skus} SKUs cho mỗi cửa hàng (đã dóng hàng)")

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
    Xử lý các giá trị khuyết và ngoại lệ trong dữ liệu nhu cầu.

    Các bước:
      1. Thay thế NaN bằng 0 (giả định doanh số thiếu = nhu cầu bằng 0)
      2. Giới hạn các giá trị âm về 0
      3. Cắt xới ngoại lệ cực đoan (> percentile thứ 99 × 3) để giảm nhiễu

    Tham số
    ------
    demand_array : np.ndarray, shape (T, n_w, n_s)

    Trả về
    -----
    np.ndarray, cùng hình dạng, đã làm sạch
    """
    arr = demand_array.copy()

    # 1. NaN → 0
    nan_count = np.isnan(arr).sum()
    if nan_count > 0:
        print(f"[Tiền xử lý] Thay thế {nan_count} giá trị NaN bằng 0")
    arr = np.nan_to_num(arr, nan=0.0)

    # 2. Cắt giá trị âm
    arr = np.clip(arr, 0.0, None)

    # 3. Cắt ngoại lệ cực đoan cho từng cặp
    T, n_w, n_s = arr.shape
    for w in range(n_w):
        for s in range(n_s):
            series = arr[:, w, s]
            p99 = np.percentile(series[series > 0], 99) if (series > 0).any() else 1.0
            arr[:, w, s] = np.clip(series, 0.0, p99 * 3.0)

    print(f"[Tiền xử lý] Thống kê nhu cầu sau khi làm sạch:")
    print(f"  Trung bình toàn cục: {arr.mean():.2f}")
    print(f"  Độ lệch chuẩn:       {arr.std():.2f}")
    print(f"  Giá trị lớn nhất:    {arr.max():.2f}")
    print(f"  Giá trị nhỏ nhất:    {arr.min():.2f}")
    print(f"  Tỷ lệ khác 0 %:      {(arr > 0).mean() * 100:.1f}%")

    return arr


def save_outputs(
    demand_array: np.ndarray,
    sku_meta: pd.DataFrame,
    stores: List[str],
    out_dir: Path,
    env_config_overrides: Optional[dict] = None,
) -> dict:
    """
    Lưu dữ liệu nhu cầu đã xử lý và cấu hình môi trường ra đĩa.

    Tham số
    ------
    demand_array : np.ndarray, shape (T, n_w, n_s)
    sku_meta : pd.DataFrame
    stores : list của str
    out_dir : Path
    env_config_overrides : dict, optional — ghi đè các giá trị cấu hình mặc định

    Trả về
    -----
    env_config : dict — đã lưu vào env_config.json
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    T, n_w, n_s = demand_array.shape

    # Xây dựng cấu hình môi trường
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

    # Lưu các tệp
    np_path   = out_dir / "demand_data.npy"
    meta_path = out_dir / "sku_metadata.csv"
    cfg_path  = out_dir / "env_config.json"

    np.save(str(np_path), demand_array)
    sku_meta.to_csv(str(meta_path), index=False)
    with open(cfg_path, "w") as f:
        json.dump(env_config, f, indent=2)

    print(f"\n[Tiền xử lý] Đã lưu kết quả tại {out_dir}:")
    print(f"  demand_data.npy   kích thước={demand_array.shape}  ({np_path.stat().st_size / 1024:.1f} KB)")
    print(f"  sku_metadata.csv  số dòng={len(sku_meta)}")
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
    Tạo dữ liệu nhu cầu giả lập khi không có bộ dữ liệu M5.

    Tạo nhu cầu thực tế với:
      - Phân phối Poisson cơ bản với trung bình ngẫu nhiên (5–30 đơn vị/ngày)
      - Tính mùa vụ theo tuần (đỉnh điểm cuối tuần)
      - Các ngày thiếu hàng ngẫu nhiên (demand = 0)

    Tham số
    ------
    n_warehouses, n_skus, n_days : int
    seed : int
    out_dir : Path, optional — nếu cung cấp, lưu dữ liệu ra đĩa

    Trả về
    -----
    np.ndarray, shape (n_days, n_warehouses, n_skus)
    """
    print(f"\n[Tiền xử lý] Đang tạo dữ liệu nhu cầu giả lập (synthetic demand)...")
    print(f"  Kích thước: ({n_days}, {n_warehouses}, {n_skus})")
    rng = np.random.default_rng(seed)

    # Trung bình cơ bản cho mỗi cặp
    base_means = rng.uniform(5, 30, size=(n_warehouses, n_skus))

    # Trọng số mùa vụ theo tuần (T2=0 .. CN=6)
    week_weights = np.array([0.8, 0.9, 1.0, 1.0, 1.1, 1.3, 1.2])

    demand = np.zeros((n_days, n_warehouses, n_skus), dtype=np.float32)
    for t in range(n_days):
        dow = t % 7
        seasonal_means = base_means * week_weights[dow]
        demand[t] = rng.poisson(seasonal_means).astype(np.float32)

    print(f"  Thống kê nhu cầu: trung bình={demand.mean():.2f}, std={demand.std():.2f}, max={demand.max():.2f}")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(out_dir / "demand_data.npy"), demand)

        # Xây dựng cấu hình env_config giả lập
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
        print(f"  Đã lưu tại {out_dir}")

    return demand


# ---------------------------------------------------------------------------
# Điểm vào CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tiền xử lý dữ liệu M5 Forecasting cho môi trường tồn kho RL."
    )
    parser.add_argument(
        "--stores", nargs="+",
        default=["CA_1", "TX_1"],
        help="Danh sách ID cửa hàng dùng làm nhà kho (mặc định: CA_1 TX_1)",
    )
    parser.add_argument(
        "--n_skus", type=int, default=30,
        help="Số lượng mặt hàng SKU lấy mẫu trên mỗi cửa hàng (mặc định: 30)",
    )
    parser.add_argument(
        "--raw_dir", type=str, default=str(RAW_DIR),
        help=f"Thư mục chứa các tệp CSV thô M5 (mặc định: {RAW_DIR})",
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(OUT_DIR),
        help=f"Thư mục lưu dữ liệu đầu ra (mặc định: {OUT_DIR})",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Tạo dữ liệu giả lập ngay cả khi có tệp M5 (phục vụ thử nghiệm)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Hạt giống ngẫu nhiên (mặc định: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)

    print("=" * 60)
    print("Tiền xử lý Dữ liệu M5 cho Môi trường Quản lý Tồn kho RL")
    print("=" * 60)
    print(f"Cửa hàng (Nhà kho): {args.stores}")
    print(f"Số SKU mỗi cửa hàng: {args.n_skus}")
    print(f"Thư mục dữ liệu thô:  {raw_dir}")
    print(f"Thư mục đầu ra:       {out_dir}")
    print("=" * 60)

    sales_path = raw_dir / "sales_train_evaluation.csv"
    if args.synthetic or not sales_path.exists():
        if not args.synthetic:
            print("\n[CẢNH BÁO] Không tìm thấy các tệp M5 trong data/raw/. Sử dụng dữ liệu giả lập.")
            print("  Để dùng dữ liệu M5 thực tế, hãy tải về từ Kaggle và đặt vào data/raw/")
        demand_array = generate_synthetic_fallback(
            n_warehouses=len(args.stores),
            n_skus=args.n_skus,
            seed=args.seed,
            out_dir=out_dir,
        )
        # Tạo metadata mẫu
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
        print("\nHoàn thành! Dữ liệu giả lập đã sẵn sàng.")
        return

    # --- Tiền xử lý dữ liệu M5 thực tế ---
    df = load_m5_sales(raw_dir, args.stores, n_skus=args.n_skus, seed=args.seed)
    calendar = load_calendar(raw_dir)
    demand_array, sku_meta = compute_daily_demand(df, args.stores, calendar)
    demand_array = clean_demand(demand_array)
    env_config = save_outputs(demand_array, sku_meta, args.stores, out_dir)

    print("\n" + "=" * 60)
    print("Tiền xử lý HOÀN THÀNH!")
    print(f"Kích thước dữ liệu nhu cầu: {demand_array.shape}")
    print(f"Cấu hình môi trường:        {env_config}")
    print("=" * 60)


if __name__ == "__main__":
    main()
