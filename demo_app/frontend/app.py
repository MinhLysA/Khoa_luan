"""
demo_app/frontend/app.py
========================
Ứng dụng Web Demo Streamlit: Trực quan hóa & Giả lập Tác tử RL (Double DQN)
cho Bài toán Quản lý Tồn kho Đa Kho.

Tách biệt hoàn toàn với logic backend và code huấn luyện gốc.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as bg
import plotly.subplots as sp
from pathlib import Path

# Thêm thư mục demo_app vào sys.path để import backend
DEMO_APP_DIR = Path(__file__).resolve().parent.parent
if str(DEMO_APP_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_APP_DIR))

# Thêm rl_inventory vào sys.path
PROJECT_ROOT = DEMO_APP_DIR.parent
RL_INVENTORY_DIR = PROJECT_ROOT / "rl_inventory"
if str(RL_INVENTORY_DIR) not in sys.path:
    sys.path.insert(0, str(RL_INVENTORY_DIR))

from backend.model_loader import discover_checkpoints, load_trained_agent
from backend.env_wrapper import SimulationEngine


# ---------------------------------------------------------------------------
# Cấu hình Trang Streamlit & Style
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RL Inventory Management Demo",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS cho giao diện hiện đại
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Caching Model Loading (Yêu cầu kỹ thuật #2)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Đang nạp mô hình Double DQN...")
def get_cached_agent(checkpoint_path: str, env_config: dict):
    return load_trained_agent(checkpoint_path, env_config)


# ---------------------------------------------------------------------------
# Giao diện Sidebar (Yêu cầu chức năng #1, #5)
# ---------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/isometric/100/warehouse.png", width=70)
st.sidebar.title("⚙️ Cấu hình Mô phỏng")

# 1. Chọn Checkpoint Model
st.sidebar.markdown("### 🤖 Checkpoint Model")
discovered_ckpts = discover_checkpoints()

ckpt_options = {}
for p in discovered_ckpts:
    label = f"{p.name} ({p.stat().st_size / 1024:.0f} KB)"
    ckpt_options[label] = str(p)

selected_ckpt_label = st.sidebar.selectbox(
    "Chọn file weights (.pth):",
    options=list(ckpt_options.keys()) if ckpt_options else ["Không tìm thấy file .pth"],
)

uploaded_ckpt = st.sidebar.file_uploader("Hoặc tải lên checkpoint (.pth) mới:", type=["pth"])

# Xác định đường dẫn checkpoint cuối cùng
if uploaded_ckpt is not None:
    temp_ckpt_path = DEMO_APP_DIR / "temp_uploaded.pth"
    with open(temp_ckpt_path, "wb") as f:
        f.write(uploaded_ckpt.getbuffer())
    active_ckpt_path = str(temp_ckpt_path)
    st.sidebar.success("Đã tải lên checkpoint tùy chỉnh!")
elif ckpt_options:
    active_ckpt_path = ckpt_options[selected_ckpt_label]
else:
    active_ckpt_path = str(RL_INVENTORY_DIR / "checkpoints" / "best_model.pth")

# 2. Tham số Môi trường & Nhu cầu
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Tham số Môi trường")

data_source = st.sidebar.radio(
    "Nguồn dữ liệu nhu cầu:",
    options=[
        "Dữ liệu Giả lập (Synthetic)",
        "Dữ liệu M5 Walmart (Real)",
        "Dữ liệu Thật Nhà máy (Excel Master Data)",
    ],
    index=0,
)

excel_file_source = None
if data_source == "Dữ liệu Giả lập (Synthetic)":
    data_source_type = "synthetic"
elif data_source == "Dữ liệu M5 Walmart (Real)":
    data_source_type = "m5"
else:
    data_source_type = "excel"
    default_excel_path = PROJECT_ROOT / "Export_20260808_121740.xlsx"
    uploaded_excel = st.sidebar.file_uploader("Tải lên file Excel (.xlsx):", type=["xlsx"])
    if uploaded_excel is not None:
        excel_file_source = uploaded_excel
    elif default_excel_path.exists():
        excel_file_source = str(default_excel_path)
        st.sidebar.caption(f"📁 Tự động dùng file: `Export_20260808_121740.xlsx`")
    else:
        st.sidebar.error("Không tìm thấy file Excel mặc định!")

sim_seed = st.sidebar.number_input("Hạt giống ngẫu nhiên (Seed):", min_value=1, max_value=9999, value=100)
episode_len = st.sidebar.slider("Số ngày mô phỏng (Episode Days):", min_value=14, max_value=365, value=112, step=7)

st.sidebar.markdown("### 💰 Tham số Chi phí ($)")
holding_cost = st.sidebar.number_input("Holding Cost (lưu kho / đơn vị / ngày):", value=1.0, step=0.5)
stockout_cost = st.sidebar.number_input("Stockout Cost (thiếu hàng / đơn vị):", value=10.0, step=1.0)
ordering_cost = st.sidebar.number_input("Ordering Cost (đặt hàng / lần):", value=50.0, step=5.0)

# Nút Run Simulation (Yêu cầu chức năng #4)
st.sidebar.markdown("---")
run_sim_btn = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Tiêu đề & Thông tin chính
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">📦 Demo Quản lý Tồn kho Đa Kho với Double DQN</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Trực quan hóa và so sánh mô hình Reinforcement Learning với các baseline EOQ, (s,S) và Newsvendor</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Thực thi Simulation Engine
# ---------------------------------------------------------------------------
if "sim_result" not in st.session_state or run_sim_btn:
    progress_bar = st.progress(0, text="Đang chuẩn bị môi trường mô phỏng...")

    try:
        # 1. Khởi tạo Engine
        engine = SimulationEngine(
            data_source_type=data_source_type,
            excel_source=excel_file_source,
            seed=sim_seed,
            n_warehouses=2,
            n_skus=30,
            episode_length=episode_len,
            holding_cost=holding_cost,
            stockout_cost=stockout_cost,
            ordering_cost=ordering_cost,
        )

        # 2. Nạp Agent
        def update_progress(val):
            progress_bar.progress(val, text=f"Đang chạy mô phỏng tập... ({int(val*100)}%)")

        agent = get_cached_agent(active_ckpt_path, engine.env_config)

        # 3. Chạy Simulation
        sim_data = engine.run_simulation(
            agent=agent,
            sim_seed=sim_seed,
            progress_callback=update_progress,
        )

        st.session_state["sim_result"] = sim_data
        progress_bar.empty()
        st.toast("✅ Chạy mô phỏng thành công!", icon="🎉")

    except Exception as e:
        progress_bar.empty()
        st.error(f"Lỗi khi chạy mô phỏng: {e}")
        st.stop()

sim_result = st.session_state["sim_result"]
daily_data = sim_result["daily_data"]
summary = sim_result["summary"]
env_config = sim_result["env_config"]
sku_metadata_df = sim_result.get("sku_metadata_df", None)
cleaning_stats = sim_result.get("cleaning_stats", {})


# Display Data Cleaning Banner if Excel data is active
if cleaning_stats and "original_rows" in cleaning_stats:
    with st.expander("🏭 Thông tin Data Adapter: Làm sạch Dữ liệu Thật Nhà máy (Excel Master Data)", expanded=True):
        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        sc1.metric("Dòng Master Data Gốc", f"{cleaning_stats['original_rows']:,}")
        sc2.metric("Đã loại (Hủy / Ko SD)", f"{cleaning_stats['cancelled_rows_dropped']:,}", delta=f"-{cleaning_stats['cancelled_rows_dropped']}", delta_color="inverse")
        sc3.metric("Số dòng Hợp lệ", f"{cleaning_stats['valid_rows_remaining']:,}")
        sc4.metric("SKU có Tồn An toàn Thực", f"{cleaning_stats['real_ss_count']:,}")
        sc5.metric("Demand Trung bình Neo", f"{cleaning_stats.get('overall_derived_d_mean', 0.0):.1f} đơn vị/ngày")
        
        if sku_metadata_df is not None:
            st.caption("📋 Danh mục Nguyên vật liệu Thực tế được chọn mô phỏng:")
            st.dataframe(
                sku_metadata_df[[
                    "warehouse_name", "sku_code", "sku_name", "category", "unit", "real_safety_stock", "anchored_d_mean"
                ]].rename(columns={
                    "warehouse_name": "Vị trí / Kho",
                    "sku_code": "Mã NVL",
                    "sku_name": "Tên Nguyên vật liệu",
                    "category": "Nhóm NVL",
                    "unit": "ĐVT",
                    "real_safety_stock": "Tồn an toàn thực",
                    "anchored_d_mean": "Demand Neo (units/day)",
                }),
                use_container_width=True,
                hide_index=True,
            )


# ---------------------------------------------------------------------------
# 1. KPI Cards (Yêu cầu chức năng #3)
# ---------------------------------------------------------------------------
st.markdown("### 📌 Chỉ số Tổng hợp (Double DQN vs Baseline tốt nhất)")

col1, col2, col3, col4 = st.columns(4)

dqn_summary = summary.get("Double DQN", {})
best_baseline_name = min(
    [k for k in summary.keys() if k != "Double DQN"],
    key=lambda k: summary[k]["total_cost"]
)
best_base = summary[best_baseline_name]

with col1:
    cost_diff = dqn_summary["total_cost"] - best_base["total_cost"]
    st.metric(
        label="TỔNG CHI PHÍ (TOTAL COST)",
        value=f"${dqn_summary['total_cost']:,.0f}",
        delta=f"{cost_diff:+,.0f} vs {best_baseline_name}",
        delta_color="inverse",
    )

with col2:
    st.metric(
        label="MỨC ĐỘ PHỤC VỤ (SERVICE LEVEL)",
        value=f"{dqn_summary['service_level']*100:.1f}%",
        delta=f"{(dqn_summary['service_level'] - best_base['service_level'])*100:+.1f}% vs {best_baseline_name}",
    )

with col3:
    st.metric(
        label="TỒN KHO TRUNG BÌNH (AVG INVENTORY)",
        value=f"{dqn_summary['avg_inventory']:,.1f} đơn vị",
        delta=f"{dqn_summary['avg_inventory'] - best_base['avg_inventory']:+,.1f} vs {best_baseline_name}",
        delta_color="inverse",
    )

with col4:
    st.metric(
        label="TỔNG THIẾU HÀNG (STOCKOUT UNITS)",
        value=f"{dqn_summary['total_stockout']:,.0f} đơn vị",
        delta=f"{dqn_summary['total_stockout'] - best_base['total_stockout']:+,.0f} vs {best_baseline_name}",
        delta_color="inverse",
    )

st.markdown("---")


# ---------------------------------------------------------------------------
# Bộ lọc Kho & SKU (Yêu cầu chức năng #2)
# ---------------------------------------------------------------------------
st.markdown("### 🎯 Bộ lọc Trực quan hóa")
f_col1, f_col2 = st.columns(2)

with f_col1:
    if sku_metadata_df is not None and "warehouse_name" in sku_metadata_df.columns:
        unique_wh_names = sku_metadata_df["warehouse_name"].unique().tolist()
        wh_options = ["Tất cả Nhà kho (Tổng hợp)"] + [f"Kho {i}: {name}" for i, name in enumerate(unique_wh_names)]
    else:
        wh_options = ["Tất cả Nhà kho (Tổng hợp)"] + [f"Nhà kho {w}" for w in range(env_config["n_warehouses"])]
    selected_wh_label = st.selectbox("Chọn Nhà kho:", wh_options, index=0)

with f_col2:
    if sku_metadata_df is not None:
        sku_options = ["Tất cả SKUs (Tổng hợp)"]
        for _, row in sku_metadata_df.iterrows():
            s_idx = row["sku_idx"]
            code = row["sku_code"]
            name = row["sku_name"]
            sku_options.append(f"SKU {s_idx}: {code} - {name}")
    else:
        sku_options = ["Tất cả SKUs (Tổng hợp)"] + [f"SKU {s}" for s in range(env_config["n_skus"])]
    selected_sku_label = st.selectbox("Chọn Mặt hàng (SKU):", sku_options, index=0)

# Xác định index để lọc dữ liệu
if selected_wh_label == "Tất cả Nhà kho (Tổng hợp)":
    wh_idx = None
else:
    # Trích xuất số index từ chuỗi chọn
    wh_idx = int(re.search(r"\d+", selected_wh_label).group(0))

if selected_sku_label == "Tất cả SKUs (Tổng hợp)":
    sku_idx = None
else:
    sku_idx = int(re.search(r"\d+", selected_sku_label).group(0))


# ---------------------------------------------------------------------------
# Hàm tạo Time Series DataFrame cho Plotly
# ---------------------------------------------------------------------------
def extract_time_series(daily_data_dict: dict, wh_i: Optional[int], sku_i: Optional[int]) -> pd.DataFrame:
    rows = []
    for policy, steps in daily_data_dict.items():
        for step in steps:
            day = step["day"]
            inv_m = step["inventory_matrix"]
            ord_m = step["order_matrix"]
            dem_m = step["demand_matrix"]

            # Lọc theo wh và sku
            if wh_i is not None and sku_i is not None:
                inv_val = inv_m[wh_i, sku_i]
                ord_val = ord_m[wh_i, sku_i]
                dem_val = dem_m[wh_i, sku_i]
            elif wh_i is not None:
                inv_val = inv_m[wh_i, :].sum()
                ord_val = ord_m[wh_i, :].sum()
                dem_val = dem_m[wh_i, :].sum()
            elif sku_i is not None:
                inv_val = inv_m[:, sku_i].sum()
                ord_val = ord_m[:, sku_i].sum()
                dem_val = dem_m[:, sku_i].sum()
            else:
                inv_val = step["inventory_total"]
                ord_val = step["order_total"]
                dem_val = step["demand_total"]

            rows.append({
                "policy": policy,
                "day": day,
                "inventory": inv_val,
                "order": ord_val,
                "demand": dem_val,
            })
    return pd.DataFrame(rows)


df_ts = extract_time_series(daily_data, wh_idx, sku_idx)


# ---------------------------------------------------------------------------
# Giao diện Tabs (Yêu cầu chức năng #3)
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Tồn kho theo Thời gian",
    "📦 Quyết định Đặt hàng",
    "📊 So sánh Bảng Tổng hợp",
    "🔍 Chi tiết từng Ngày (Inspector)",
])

# --- Tab 1: Biểu đồ Tồn kho --------------------------------------------------
with tab1:
    st.markdown("#### Diễn biến Mức Tồn kho (Inventory Level) qua các ngày")
    fig_inv = px.line(
        df_ts,
        x="day",
        y="inventory",
        color="policy",
        labels={"day": "Ngày (Time step)", "inventory": "Mức Tồn kho (Đơn vị)", "policy": "Chiến lược"},
        title="Biểu đồ Tồn kho theo Thời gian giữa các Chiến lược",
        color_discrete_map={
            "Double DQN": "#2563EB",
            "EOQ": "#D97706",
            "(s,S)": "#059669",
            "Newsvendor": "#DC2626",
        },
    )
    fig_inv.update_traces(line=dict(width=2.5))
    fig_inv.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_inv, use_container_width=True)

# --- Tab 2: So sánh Order Quantity --------------------------------------------
with tab2:
    st.markdown("#### Quyết định Đặt hàng (Order Quantity) vs Nhu cầu (Demand)")
    
    # Biểu đồ Order Quantity
    fig_ord = px.bar(
        df_ts,
        x="day",
        y="order",
        color="policy",
        barmode="group",
        labels={"day": "Ngày", "order": "Lượng Đặt hàng (Units)", "policy": "Chiến lược"},
        title="So sánh Lượng Đặt hàng từng Ngày giữa các Chiến lược",
        color_discrete_map={
            "Double DQN": "#2563EB",
            "EOQ": "#D97706",
            "(s,S)": "#059669",
            "Newsvendor": "#DC2626",
        },
    )
    fig_ord.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_ord, use_container_width=True)

    # Biểu đồ Demand overlay
    df_demand_unique = df_ts[df_ts["policy"] == "Double DQN"]
    fig_dem = px.line(
        df_demand_unique,
        x="day",
        y="demand",
        labels={"day": "Ngày", "demand": "Nhu cầu Thực tế (Units)"},
        title="Đường Nhu cầu Thực tế (Actual Demand)",
    )
    fig_dem.update_traces(line=dict(color="#475569", dash="dash", width=2))
    st.plotly_chart(fig_dem, use_container_width=True)

# --- Tab 3: Bảng Tổng hợp & Cost Breakdown -----------------------------------
with tab3:
    st.markdown("#### Bảng So sánh Chỉ số Hiệu năng (Performance Metrics)")
    df_summary = pd.DataFrame(list(summary.values()))
    
    # Format hiển thị đẹp mắt
    df_display = df_summary.copy()
    df_display["total_cost"] = df_display["total_cost"].map("${:,.2f}".format)
    df_display["holding_cost"] = df_display["holding_cost"].map("${:,.2f}".format)
    df_display["ordering_cost"] = df_display["ordering_cost"].map("${:,.2f}".format)
    df_display["stockout_cost"] = df_display["stockout_cost"].map("${:,.2f}".format)
    df_display["service_level"] = df_display["service_level"].map("{:.2%}".format)
    df_display["avg_inventory"] = df_display["avg_inventory"].map("{:,.1f}".format)
    df_display["total_stockout"] = df_display["total_stockout"].map("{:,.0f}".format)

    st.dataframe(
        df_display[[
            "policy", "total_cost", "service_level", "avg_inventory",
            "total_stockout", "holding_cost", "ordering_cost", "stockout_cost"
        ]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Phân rã Chi phí (Cost Breakdown)")
    fig_cost = px.bar(
        df_summary,
        x="policy",
        y=["holding_cost", "ordering_cost", "stockout_cost"],
        title="Phân rã Thành phần Chi phí theo Chiến lược ($)",
        labels={"value": "Chi phí ($)", "policy": "Chiến lược", "variable": "Loại chi phí"},
        color_discrete_map={
            "holding_cost": "#3B82F6",
            "ordering_cost": "#F59E0B",
            "stockout_cost": "#EF4444",
        },
    )
    st.plotly_chart(fig_cost, use_container_width=True)

# --- Tab 4: Inspector từng Ngày -----------------------------------------------
with tab4:
    st.markdown("#### 🔍 Soi Trạng thái & Quyết định theo Ngày cụ thể")
    selected_day = st.slider("Chọn Ngày trong tập:", min_value=1, max_value=episode_len, value=1)
    
    inspect_col1, inspect_col2 = st.columns(2)

    with inspect_col1:
        st.markdown(f"##### 📋 Ma trận Đặt hàng ở Ngày {selected_day} (Double DQN)")
        dqn_day_step = daily_data["Double DQN"][selected_day - 1]
        df_dqn_ord = pd.DataFrame(
            dqn_day_step["order_matrix"],
            index=[f"Kho {w}" for w in range(env_config["n_warehouses"])],
            columns=[f"SKU {s}" for s in range(env_config["n_skus"])],
        )
        st.dataframe(df_dqn_ord.style.highlight_max(axis=None, color="#DBEAFE"), use_container_width=True)

    with inspect_col2:
        st.markdown(f"##### 📦 Ma trận Tồn kho ở Ngày {selected_day} (Double DQN)")
        df_dqn_inv = pd.DataFrame(
            dqn_day_step["inventory_matrix"],
            index=[f"Kho {w}" for w in range(env_config["n_warehouses"])],
            columns=[f"SKU {s}" for s in range(env_config["n_skus"])],
        )
        st.dataframe(df_dqn_inv.style.highlight_min(axis=None, color="#FEE2E2"), use_container_width=True)
