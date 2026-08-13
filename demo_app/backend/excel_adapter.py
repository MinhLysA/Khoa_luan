"""
demo_app/backend/excel_adapter.py
=================================
Data Adapter xử lý dữ liệu thật từ file Excel Master Data của Nhà máy (Export_20260808_121740.xlsx).

Chức năng:
1. Đọc & Làm sạch Master Data: Lọc bỏ dòng ngưng/hủy/bỏ.
2. Chuẩn hóa & Ánh xạ: Nhóm theo Vị trí lưu trữ & Nhóm nguyên liệu thành Kho và SKU.
3. Xử lý Tồn an toàn: Điền giá trị thiếu (Imputation) bằng trung vị nhóm.
4. Sinh Demand Giả lập Neo vào Dữ liệu Thật (Anchored Demand Generation):
   Dùng Tồn an toàn thực tế làm neo (anchor) để suy ra nhu cầu trung bình D_mean hợp lý,
   tạo ra chuỗi demand thời gian (n_days, n_warehouses, n_skus) tương thích 100% với env RL.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union


# Từ khóa nhận diện các dòng ngưng/hủy/không sử dụng
CANCEL_KEYWORDS = [
    "hủy", "huy",
    "không sử dụng", "khong su dung", "không sd", "khong sd",
    "ko sd", "kosd", "k sd",
    "ngưng", "ngung",
    "ướt hư", "bị hư", "bỏ mã", "bỏ",
]


class RealDataMasterAdapter:
    """
    Adapter biến đổi Master Data tĩnh từ Excel thành dữ liệu nhu cầu động cho Simulation Engine.
    """

    def __init__(self, excel_source: Union[str, Path, Any]):
        """
        Tham số
        ------
        excel_source : str, Path hoặc UploadedFile của Streamlit
        """
        self.excel_source = excel_source
        self.raw_df: Optional[pd.DataFrame] = None
        self.clean_df: Optional[pd.DataFrame] = None
        self.cleaning_stats: Dict[str, Any] = {}

    def load_and_clean(self) -> pd.DataFrame:
        """
        Đọc sheet ExportData (header=1) và làm sạch dữ liệu.
        """
        # Read excel
        try:
            self.raw_df = pd.read_excel(self.excel_source, sheet_name="ExportData", header=1)
        except Exception:
            # Fallback if header is 0
            self.raw_df = pd.read_excel(self.excel_source, header=0)

        original_count = len(self.raw_df)

        # Đảm bảo các cột cần thiết có mặt
        col_map = {
            "STT": "stt",
            "Mã nguyên liệu": "sku_code",
            "Tên Tiếng Việt": "sku_name",
            "Nhóm nguyên liệu [F3]": "category",
            "Vị trí lưu trữ [F3]": "location",
            "Loại nguyên liệu [F3]": "material_type",
            "ĐVT kho [F3]": "unit",
            "Qui cách": "spec",
            "Tồn an toàn": "safety_stock_raw",
            "Ghi chú": "note",
        }

        # Rename columns that exist
        df = self.raw_df.copy()
        for orig_c, new_c in col_map.items():
            if orig_c in df.columns:
                df[new_c] = df[orig_c]

        # Fill missing string columns
        for text_col in ["sku_name", "category", "location", "spec", "note"]:
            if text_col not in df.columns:
                df[text_col] = ""
            else:
                df[text_col] = df[text_col].astype(str).fillna("")

        # 1. Lọc bỏ các dòng hủy / ngưng sử dụng
        pattern = "|".join([re.escape(k) for k in CANCEL_KEYWORDS])
        cancel_mask = (
            df["spec"].str.contains(pattern, flags=re.IGNORECASE, regex=True) |
            df["note"].str.contains(pattern, flags=re.IGNORECASE, regex=True) |
            df["sku_name"].str.contains(pattern, flags=re.IGNORECASE, regex=True)
        )

        df_filtered = df[~cancel_mask].copy()
        cancelled_count = int(cancel_mask.sum())

        # 2. Xử lý Tồn an toàn (Safety Stock)
        ss_numeric = pd.to_numeric(df_filtered["safety_stock_raw"], errors="coerce")
        df_filtered["ss_raw_numeric"] = ss_numeric
        missing_ss_count = int(ss_numeric.isna().sum())

        # Impute missing safety stock bằng trung vị theo Nhóm nguyên liệu (Category)
        cat_medians = df_filtered.groupby("category")["ss_raw_numeric"].transform("median")
        overall_median = ss_numeric.median()
        if pd.isna(overall_median) or overall_median <= 0:
            overall_median = 1000.0

        df_filtered["ss_clean"] = (
            df_filtered["ss_raw_numeric"]
            .fillna(cat_medians)
            .fillna(overall_median)
        )

        # Clip safety stock tối thiểu 1.0
        df_filtered["ss_clean"] = np.maximum(df_filtered["ss_clean"], 1.0)

        self.clean_df = df_filtered
        self.cleaning_stats = {
            "original_rows": original_count,
            "cancelled_rows_dropped": cancelled_count,
            "valid_rows_remaining": len(df_filtered),
            "real_ss_count": int(ss_numeric.notna().sum()),
            "imputed_ss_count": missing_ss_count,
            "overall_ss_median": float(overall_median),
        }

        return self.clean_df

    def build_simulation_dataset(
        self,
        n_warehouses: int = 2,
        n_skus: int = 30,
        n_days: int = 800,
        seed: int = 42,
    ) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Any]]:
        """
        Xây dựng ma trận demand (n_days, n_warehouses, n_skus) và metadata SKU thật từ Excel.

        Tham số
        ------
        n_warehouses : int
        n_skus : int
        n_days : int
        seed : int

        Trả về
        -----
        demand_matrix : np.ndarray, shape (n_days, n_warehouses, n_skus)
        sku_metadata_df : pd.DataFrame chứa thông tin thật của từng SKU được chọn
        stats : dict
        """
        if self.clean_df is None:
            self.load_and_clean()

        df = self.clean_df

        # 1. Chọn Warehouses (dựa vào vị trí lưu trữ phổ biến nhất hoặc phân vùng)
        loc_counts = df["location"].value_counts()
        # Loại bỏ chuỗi rỗng / nan string nếu có
        valid_locs = [loc for loc in loc_counts.index if loc and loc.lower() not in ["nan", "none", ""]]
        
        selected_whs = valid_locs[:n_warehouses]
        # Nếu không đủ vị trí thật, đặt tên mặc định
        while len(selected_whs) < n_warehouses:
            selected_whs.append(f"Kho Thật {len(selected_whs) + 1}")

        selected_sku_metadata = []
        demand_means_matrix = np.zeros((n_warehouses, n_skus), dtype=np.float32)

        rng = np.random.default_rng(seed)

        # Lấy mẫu n_skus cho mỗi nhà kho
        for w_idx, wh_name in enumerate(selected_whs):
            sub_df = df[df["location"] == wh_name]
            if len(sub_df) < n_skus:
                # Lấy thêm từ danh mục chung nếu kho không đủ SKU
                needed = n_skus - len(sub_df)
                extra_df = df[~df.index.isin(sub_df.index)].sample(n=min(needed, len(df)), random_state=seed)
                sub_df = pd.concat([sub_df, extra_df])

            sample_skus = sub_df.head(n_skus).copy()

            for s_idx, (_, row) in enumerate(sample_skus.iterrows()):
                ss_val = float(row["ss_clean"])
                
                # NEO DEMAND VÀO TỒN AN TOÀN THỰC TẾ:
                # Nhu cầu trung bình hàng ngày D_mean được suy ra từ Tồn an toàn (SS)
                # Giả định SS bao phủ khoảng 5-10 ngày nhu cầu, đồng thời scale mượt về khoảng [5.0, 35.0]
                # để khớp với quan sát & hành động rời rạc {0, 10, 20, 30, 40, 50} của agent RL
                scaled_d_mean = np.clip(ss_val / 500.0, 5.0, 35.0)

                demand_means_matrix[w_idx, s_idx] = scaled_d_mean

                selected_sku_metadata.append({
                    "warehouse_idx": w_idx,
                    "warehouse_name": wh_name,
                    "sku_idx": s_idx,
                    "sku_code": row.get("sku_code", f"SKU_{s_idx}"),
                    "sku_name": row.get("sku_name", f"Nguyên liệu {s_idx}"),
                    "category": row.get("category", "Vật tư"),
                    "unit": row.get("unit", "Cái"),
                    "real_safety_stock": ss_val,
                    "anchored_d_mean": float(scaled_d_mean),
                })

        sku_metadata_df = pd.DataFrame(selected_sku_metadata)

        # 2. Sinh chuỗi nhu cầu Poisson theo thời gian với tính mùa vụ theo tuần
        week_weights = np.array([0.8, 0.9, 1.0, 1.0, 1.1, 1.3, 1.2], dtype=np.float32)
        demand_matrix = np.zeros((n_days, n_warehouses, n_skus), dtype=np.float32)

        for t in range(n_days):
            dow = t % 7
            daily_means = demand_means_matrix * week_weights[dow]
            demand_matrix[t] = rng.poisson(daily_means).astype(np.float32)

        full_stats = {
            **self.cleaning_stats,
            "selected_warehouses": selected_whs,
            "n_warehouses": n_warehouses,
            "n_skus": n_skus,
            "n_days": n_days,
            "overall_derived_d_mean": float(demand_means_matrix.mean()),
        }

        return demand_matrix, sku_metadata_df, full_stats
