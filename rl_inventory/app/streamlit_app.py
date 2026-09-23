"""
app/streamlit_app.py
====================
Bảng điều khiển trực quan cho khóa luận "Tối ưu hóa chính sách đặt hàng lại
trong quản lý tồn kho đa kho bằng học tăng cường (IPPO)".

Chạy:
    streamlit run app/streamlit_app.py

Mục đích: nhìn thấy CHUYỆN GÌ ĐANG XẢY RA, thay vì đọc log chữ chạy.
  Tab 1  Dữ liệu        - cầu M5 sau tiền xử lý trông như thế nào
  Tab 2  Cấu hình & Chạy- sửa tham số và bấm nút chạy từng bước
  Tab 3  Huấn luyện     - đường học theo thời gian thực (đọc results/train_log*.csv)
  Tab 4  So sánh        - IPPO vs EOQ / (s,S) / Newsvendor
  Tab 5  Mô phỏng       - chiếu lại 365 ngày của một chính sách, theo từng ngày
  Tab 6  Giả lập tay    - tự tay đặt hàng cho 1 kho-SKU qua vài ngày, để hiểu
                          cơ chế trước khi nhìn cả 300 cặp cùng lúc

[V3-7] Các tab số liệu (3, 4, 5) có thêm expander "Ví dụ đọc số liệu" giải
       thích bằng một ví dụ tính toán cụ thể, không chỉ nói suông tên chỉ số.
"""

import io
import os
import sys
import json
import time
import yaml
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="IPPO Tồn kho đa kho", page_icon="📦", layout="wide")

CFG_PATH = ROOT / "config.yaml"
DATA_DIR = ROOT / "data" / "processed"
RES_DIR = ROOT / "results"
CKPT_DIR = ROOT / "checkpoints"


# --------------------------------------------------------------------------- #
# Tiện ích
# --------------------------------------------------------------------------- #
def load_cfg():
    with open(CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_cfg(cfg):
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


@st.cache_data(show_spinner=False)
def load_demand(mtime: float):
    p = DATA_DIR / "demand_data.npy"
    if not p.exists():
        return None, None
    d = np.load(p)
    meta = {}
    mp = DATA_DIR / "env_config.json"
    if mp.exists():
        meta = json.loads(mp.read_text(encoding="utf-8"))
    return d, meta


def data_mtime():
    p = DATA_DIR / "demand_data.npy"
    return p.stat().st_mtime if p.exists() else 0.0


def chay_lenh(cmd, placeholder, max_dong=400):
    """Chạy lệnh con và đổ log ra màn hình theo thời gian thực."""
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"})
    dong = []
    for line in proc.stdout:
        dong.append(line.rstrip())
        placeholder.code("\n".join(dong[-max_dong:]), language="text")
    proc.wait()
    return proc.returncode


def doc_train_log():
    files = sorted(RES_DIR.glob("train_log*.csv"))
    if not files:
        return None, None
    ten = st.session_state.get("chon_log")
    path = next((f for f in files if f.name == ten), files[-1])
    try:
        df = pd.read_csv(path)
    except Exception:
        return None, [f.name for f in files]
    return (df, path), [f.name for f in files]


# --------------------------------------------------------------------------- #
st.title("📦 Tối ưu chính sách đặt hàng lại bằng IPPO")
st.caption("Đa kho - đa SKU | dữ liệu M5 Walmart | Independent Multi-Agent PPO "
           "với chia sẻ tham số")

cfg = load_cfg()
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📊 Dữ liệu", "⚙️ Cấu hình & Chạy", "📈 Huấn luyện",
     "🏁 So sánh chính sách", "🎬 Mô phỏng từng ngày", "🧪 Giả lập tay"])


# =========================================================================== #
# TAB 1 - DỮ LIỆU
# =========================================================================== #
with tab1:
    d, meta = load_demand(data_mtime())
    if d is None:
        st.warning("Chưa có dữ liệu. Sang tab **Cấu hình & Chạy** để bấm "
                   "*Tiền xử lý dữ liệu*.")
    else:
        T, W, S = d.shape
        split = int(meta.get("split_day", cfg["env"]["split_day"]))
        flat = d[:split].reshape(split, -1)
        md = flat.mean(axis=0)

        c = st.columns(5)
        c[0].metric("Số ngày", f"{T:,}")
        c[1].metric("Số kho x SKU", f"{W} x {S} = {W*S}")
        c[2].metric("Cầu TB / cặp / ngày", f"{md.mean():.2f}")
        c[3].metric("Tỷ lệ ngày bằng 0", f"{(d == 0).mean():.1%}")
        c[4].metric("Cặp gần như không bán", f"{(md < 0.1).sum()}")

        st.info(f"Chia thời gian: huấn luyện ngày 0–{split}, đánh giá ngày "
                f"{split}–{T}. Hai miền không chồng lấn.")

        st.subheader("Cầu trung bình một ngày theo từng kho")
        ten_kho = meta.get("stores", [f"WH_{i}" for i in range(W)])
        cau_kho = md.reshape(W, S).sum(axis=1)
        cov = cfg["env"].get("capacity_cover_days", 5.0)
        df_kho = pd.DataFrame({
            "Kho": ten_kho,
            "Cầu/ngày": np.round(cau_kho, 1),
            "Sức chứa (auto)": np.round(cau_kho * cov, 0),
        }).set_index("Kho")
        col1, col2 = st.columns([2, 1])
        col1.bar_chart(df_kho[["Cầu/ngày"]])
        col2.dataframe(df_kho, width="stretch")
        st.caption(f"Sức chứa mỗi kho = {cov} ngày cầu CỦA CHÍNH KHO ĐÓ. Đây là "
                   "điểm sửa quan trọng: bản cũ dùng một con số chung cho cả "
                   f"{W} kho, nên kho lớn nhất chỉ chứa được ~1,4 ngày cầu.")

        with st.expander("❓ Ví dụ đọc con số này"):
            vd_kho = df_kho.index[0]
            vd_cau = df_kho["Cầu/ngày"].iloc[0]
            vd_sc = df_kho["Sức chứa (auto)"].iloc[0]
            st.markdown(
                f"Kho **{vd_kho}** bán trung bình **{vd_cau:.0f} đơn vị/ngày** "
                f"(cộng dồn cả {S} SKU trong kho). Với capacity_cover_days = "
                f"**{cov}**, sức chứa tính ra là {vd_cau:.0f} × {cov} = "
                f"**{vd_sc:.0f} đơn vị** — dự trữ cho {cov} ngày bán trung bình. "
                f"Nếu tổng tồn kho CẢ {S} SKU trong kho **{vd_kho}** cộng lại "
                f"vượt quá {vd_sc:.0f}, phần vượt bị TỪ CHỐI ngay lúc nhập hàng "
                f"(không được lưu kho), dù đơn đã đặt và đã trả tiền đặt hàng.")

        st.subheader("Phân bố quy mô cầu của 300 cặp (kho, SKU)")
        hist = pd.DataFrame({"cau_tb": md})
        st.bar_chart(np.histogram(np.log10(np.maximum(md, 1e-2)), bins=25)[0])
        st.caption("Trục hoành: log10(cầu trung bình/ngày). Đuôi càng dài thì "
                   "mạng dùng chung trọng số càng khó tổng quát hóa.")

        st.subheader("Nhu cầu theo thời gian (tổng toàn hệ thống)")
        n_smooth = st.slider("Làm mượt (số ngày)", 1, 60, 28, key="sm_data")
        ts = pd.Series(d.reshape(T, -1).sum(axis=1)).rolling(n_smooth).mean()
        st.line_chart(ts, height=240)


# =========================================================================== #
# TAB 2 - CẤU HÌNH & CHẠY
# =========================================================================== #
with tab2:
    st.subheader("Tham số chính")
    st.caption("Sửa ở đây rồi bấm *Lưu cấu hình*; các script đều đọc từ "
               "config.yaml nên không cần sửa code.")

    e = cfg["env"]
    p = cfg["ppo"]
    pre = cfg["preprocess"]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Môi trường**")
        e["n_warehouses"] = st.number_input("Số kho", 1, 10, int(e["n_warehouses"]))
        e["n_skus"] = st.number_input("Số SKU", 5, 100, int(e["n_skus"]))
        e["episode_length"] = st.number_input("Độ dài episode (ngày)", 30, 730,
                                              int(e["episode_length"]), step=5)
        e["capacity_cover_days"] = st.number_input(
            "Sức chứa kho (số ngày cầu)", 1.0, 30.0,
            float(e["capacity_cover_days"]), step=0.5)
        pre["min_mean_demand"] = st.number_input(
            "Ngưỡng lọc SKU (đv/ngày)", 0.0, 5.0,
            float(pre.get("min_mean_demand", 0.2)), step=0.05)
    with c2:
        st.markdown("**Chi phí**")
        e["cp_lk"] = st.number_input("Lưu kho / đv / ngày", 0.0, 100.0, float(e["cp_lk"]))
        e["cp_th"] = st.number_input("Thiếu hàng / đv", 0.0, 500.0, float(e["cp_th"]))
        e["cp_dh"] = st.number_input("Đặt hàng / lần", 0.0, 200.0, float(e["cp_dh"]))
        e["pt_tk"] = st.number_input("Phạt tràn kho / đv", 0.0, 200.0, float(e["pt_tk"]))
        e["phi_dv"] = st.number_input("Hệ số phạt mức phục vụ", 0.0, 50.0, float(e["phi_dv"]))
        e["muc_dv"] = st.number_input("Fill rate mục tiêu", 0.0, 1.0,
                                      float(e["muc_dv"]), step=0.01)
    with c3:
        st.markdown("**PPO**")
        p["total_episodes"] = st.number_input("Số episode huấn luyện", 10, 20000,
                                              int(p["total_episodes"]), step=50)
        p["n_steps"] = st.number_input("n_steps / update", 128, 8192,
                                       int(p["n_steps"]), step=128)
        p["lr_actor"] = st.number_input("lr actor", 1e-6, 1e-2,
                                        float(p["lr_actor"]), format="%.6f")
        p["lr_critic"] = st.number_input("lr critic", 1e-6, 1e-2,
                                         float(p["lr_critic"]), format="%.6f")
        p["ent_coef"] = st.number_input("ent_coef (đầu)", 0.0, 0.2,
                                        float(p["ent_coef"]), step=0.001, format="%.3f")
        e["normalize_reward_per_pair"] = st.checkbox(
            "Chuẩn hóa phần thưởng theo từng cặp (BẬT = hội tụ)",
            bool(e.get("normalize_reward_per_pair", True)))

    if st.button("💾 Lưu cấu hình", type="primary"):
        cfg["env"], cfg["ppo"], cfg["preprocess"] = e, p, pre
        save_cfg(cfg)
        st.success("Đã lưu config.yaml")

    st.divider()
    st.subheader("Chạy từng bước")
    st.caption("Thứ tự chuẩn: 1 → 2 → 3 → 4. Bước 3 là bước lâu nhất.")

    b1, b2, b3, b4 = st.columns(4)
    out = st.empty()
    if b1.button("1. Tiền xử lý dữ liệu"):
        chay_lenh([sys.executable, "scripts/data_preprocessing.py", "--m5",
                   "--n_warehouses", str(e["n_warehouses"]),
                   "--n_skus", str(e["n_skus"]),
                   "--min_mean_demand", str(pre.get("min_mean_demand", 0.2)),
                   "--sku_selection", str(pre.get("sku_selection", "stratified"))], out)
        st.cache_data.clear()
    if b2.button("2. Tinh chỉnh baseline"):
        chay_lenh([sys.executable, "scripts/tune_baselines.py", "--episodes", "2"], out)
    if b3.button("3. Huấn luyện IPPO"):
        chay_lenh([sys.executable, "scripts/train.py",
                   "--episodes", str(p["total_episodes"])], out)
    if b4.button("4. Đánh giá & so sánh"):
        chay_lenh([sys.executable, "scripts/evaluate.py"], out)

    if st.button("5. So sánh ở cùng mức phục vụ (iso-service)"):
        chay_lenh([sys.executable, "scripts/iso_service.py",
                   "--tune_episodes", "2", "--eval_episodes", "10"], out)

    st.divider()
    st.caption("Checkpoint hiện có: " +
               (", ".join(f.name for f in sorted(CKPT_DIR.glob('*.pth')))
                or "chưa có"))


# =========================================================================== #
# TAB 3 - HUẤN LUYỆN
# =========================================================================== #
with tab3:
    kq, ten_files = doc_train_log()
    if kq is None:
        st.warning("Chưa có results/train_log*.csv. Hãy chạy huấn luyện ở tab 2.")
    else:
        df, path = kq
        if ten_files and len(ten_files) > 1:
            st.selectbox("Chọn lần chạy", ten_files, key="chon_log")
        c = st.columns(4)
        c[0].metric("Episode đã chạy", f"{int(df.episode.max()):,}")
        c[1].metric("Fill rate (20 ep cuối)", f"{df.fill_rate.tail(20).mean():.1%}")
        c[2].metric("Chi phí/episode (20 ep cuối)",
                    f"{df.cost_total.tail(20).mean():,.0f}")
        c[3].metric("Thời gian", f"{df.elapsed_s.max()/60:.1f} phút")

        if st.checkbox("Tự động làm mới mỗi 10 giây (khi đang chạy)"):
            time.sleep(10)
            st.rerun()

        st.subheader("Đường học")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Phần thưởng (càng cao càng tốt)**")
            st.line_chart(df.set_index("episode")[["reward_raw", "reward_smooth"]],
                          height=250)
            st.markdown("**Fill rate**")
            st.line_chart(df.set_index("episode")[["fill_rate"]], height=200)
        with c2:
            st.markdown("**Entropy (phải giảm dần → chính sách định hình)**")
            st.line_chart(df.set_index("episode")[["entropy"]].dropna(), height=250)
            st.markdown("**Explained variance (critic; phải tiến về 1)**")
            st.line_chart(df.set_index("episode")[["explained_variance"]].dropna(),
                          height=200)

        with st.expander("❓ Ví dụ đọc 3 chỉ số trên"):
            st.markdown(
                "- **Fill rate**: tỷ lệ nhu cầu được đáp ứng trong cửa sổ 30 "
                "ngày gần nhất. VD 30 ngày qua khách cần tổng 300 đơn vị, kho "
                "bán được 255 -> fill rate = 255/300 = **85%**. Mục tiêu của "
                f"khóa luận là {cfg['env']['muc_dv']*100:.0f}%.\n"
                "- **Entropy**: độ 'phân vân' của chính sách khi chọn 1 trong "
                "6 mức đặt hàng. Chọn đều cả 6 mức (ngẫu nhiên hoàn toàn) cho "
                "entropy = ln(6) ≈ **1,79** — đúng ở đầu quá trình học, lúc "
                "còn đang thăm dò. Entropy -> 0 nghĩa là agent gần như LUÔN "
                "chọn 1 mức cố định cho mọi trạng thái — đã 'chốt' chính sách. "
                "Nếu nó giảm về gần 0 CHỈ SAU VÀI CHỤC episode đầu (chưa kịp "
                "học gì) thì đó là dấu hiệu đông cứng sớm vào một nghiệm tồi, "
                "không phải hội tụ thật.\n"
                "- **Explained variance**: critic dự đoán giá trị trạng thái "
                "tốt tới đâu. VD return thật sự là [-100, -80, -120], nếu "
                "critic đoán đúng xu hướng thì explained variance gần **1**; "
                "nếu dự đoán không hơn gì việc đoán trung bình (-100 cho cả 3) "
                "thì nó gần **0**; âm nghĩa là dự đoán CÒN TỆ HƠN cả đoán bừa "
                "trung bình.")

        st.subheader("Cơ cấu chi phí qua quá trình học")
        cols = ["cost_holding", "cost_stockout", "cost_ordering",
                "cost_overflow", "cost_service_penalty"]
        st.area_chart(df.set_index("episode")[cols], height=260)

        with st.expander("Chẩn đoán ổn định (approx_kl, clip_frac)"):
            st.line_chart(df.set_index("episode")[["approx_kl", "clip_frac"]].dropna(),
                          height=220)
            st.caption("approx_kl ổn định quanh 0,005–0,02 là tốt. Vọt lên > 0,05 "
                       "liên tục = lr_actor quá lớn.")

        if "eval_fill" in df and df.eval_fill.notna().any():
            st.subheader("Đánh giá deterministic (argmax) trên miền TEST")
            ev = df[["episode", "eval_reward", "eval_fill"]].dropna()
            ev = ev.drop_duplicates(subset=["episode"]).set_index("episode")
            st.line_chart(ev, height=230)
            st.caption("Đường này mới là chất lượng chính sách thật. Nếu nó ổn "
                       "định dần trong khi reward lúc lấy mẫu vẫn nhảy, đó chỉ "
                       "là nhiễu do explore.")


# =========================================================================== #
# TAB 4 - SO SÁNH
# =========================================================================== #
with tab4:
    sp = RES_DIR / "summary.json"
    bp = RES_DIR / "baseline_comparison.csv"
    if not bp.exists():
        st.warning("Chưa có kết quả đánh giá. Chạy bước 4 ở tab 2.")
    else:
        summ = pd.read_csv(bp)
        stat = {}
        if sp.exists():
            stat = json.loads(sp.read_text(encoding="utf-8")).get("stat", {})

        bang = pd.DataFrame({
            "Chính sách": summ.policy,
            "Tổng chi phí": summ.total_cost_mean.round(0),
            "± độ lệch": summ.total_cost_std.round(0),
            "Lưu kho": summ.holding_cost_mean.round(0),
            "Thiếu hàng": summ.stockout_cost_mean.round(0),
            "Đặt hàng": summ.ordering_cost_mean.round(0),
            "Tràn kho": summ.overflow_cost_mean.round(0),
            "Fill rate": (summ.fill_rate_mean * 100).round(1),
            "Tồn kho TB": summ.avg_inventory_mean.round(0),
        }).set_index("Chính sách")
        st.dataframe(bang, width="stretch")

        with st.expander("❓ Ví dụ đọc bảng này"):
            hang0 = bang.index[0]
            r0 = bang.iloc[0]
            st.markdown(
                f"Đọc hàng **{hang0}**: tổng chi phí vận hành trung bình một "
                f"episode (365 ngày) là **{r0['Tổng chi phí']:,.0f}**, trong đó "
                f"lưu kho {r0['Lưu kho']:,.0f} + thiếu hàng {r0['Thiếu hàng']:,.0f} "
                f"+ đặt hàng {r0['Đặt hàng']:,.0f} + tràn kho {r0['Tràn kho']:,.0f} "
                f"= tổng. Fill rate {r0['Fill rate']:.1f}% nghĩa là chính sách "
                f"này đáp ứng được chừng đó % nhu cầu trong cả năm.\n\n"
                "So SAI lầm phổ biến: thấy IPPO có tổng chi phí cao hơn 1 "
                "baseline rồi kết luận 'IPPO kém hơn' — trong khi hai chính "
                "sách có thể có fill rate khác hẳn nhau (chi phí thấp dễ đạt "
                "được nếu chấp nhận thiếu hàng nhiều). Phải so ở **cùng một "
                "mức fill rate** mới công bằng — xem bảng 'So sánh ở CÙNG MỨC "
                "PHỤC VỤ' phía dưới.")

        if stat:
            gap = stat.get("gap_percent", 0.0)
            c = st.columns(4)
            c[0].metric("Baseline tốt nhất", stat.get("best_baseline", "-"))
            c[1].metric("Chênh lệch chi phí", f"{gap:+.1f}%",
                        delta=f"{'IPPO tốt hơn' if gap < 0 else 'IPPO kém hơn'}",
                        delta_color="normal" if gap < 0 else "inverse")
            c[2].metric("p (paired t-test)",
                        f"{stat.get('p_paired', stat.get('p_ttest', float('nan'))):.2e}")
            c[3].metric("Hiệu ứng d_z", f"{stat.get('d_z', float('nan')):.2f}")

        c1, c2 = st.columns(2)
        c1.markdown("**Tổng chi phí vận hành**")
        c1.bar_chart(bang[["Tổng chi phí"]], height=280)
        c2.markdown("**Fill rate (%) — đường mục tiêu 85%**")
        c2.bar_chart(bang[["Fill rate"]], height=280)

        st.markdown("**Cơ cấu chi phí**")
        st.bar_chart(bang[["Lưu kho", "Thiếu hàng", "Đặt hàng", "Tràn kho"]],
                     height=300, stack=True)

        st.markdown("**Đánh đổi chi phí — mức phục vụ** (tốt = dưới, bên phải)")
        st.scatter_chart(bang.reset_index(), x="Fill rate", y="Tổng chi phí",
                         color="Chính sách", height=320)

        iso_p = RES_DIR / "iso_service.json"
        if iso_p.exists():
            st.divider()
            st.subheader("So sánh ở CÙNG MỨC PHỤC VỤ")
            iso = json.loads(iso_p.read_text(encoding="utf-8"))
            st.caption(
                f"So chi phí giữa hai chính sách có fill rate khác nhau là vô "
                f"nghĩa. Bảng dưới đây tìm cấu hình RẺ NHẤT của từng baseline "
                f"mà vẫn đạt fill rate >= {iso['target_fill']:.1%} (mức IPPO đạt "
                f"được), rồi mới so chi phí.")
            hang, ippo_cost = [], iso["ket_qua"].get("IPPO", {}).get("cost")
            for ten, r in iso["ket_qua"].items():
                if r.get("khong_dat"):
                    hang.append({"Chính sách": ten, "Tổng chi phí": None,
                                 "Fill rate": None,
                                 "Chênh lệch vs IPPO": "không đạt được mức này"})
                    continue
                d = ("" if ten == "IPPO" else
                     f"{100*(ippo_cost - r['cost'])/r['cost']:+.1f}%")
                hang.append({"Chính sách": ten,
                             "Tổng chi phí": round(r["cost"]),
                             "Fill rate": round(r["fill"] * 100, 2),
                             "Chênh lệch vs IPPO": d})
            st.dataframe(pd.DataFrame(hang).set_index("Chính sách"),
                         width="stretch")
            st.caption("Cột cuối: số ÂM nghĩa là IPPO RẺ HƠN ở cùng mức phục vụ. "
                       "Đây mới là bảng nên đưa vào Chương 4.")
            with st.expander("❓ Ví dụ đọc bảng iso-service"):
                st.markdown(
                    f"IPPO tự nó đạt fill rate {iso['target_fill']*100:.1f}%. "
                    "Với TỪNG baseline, script `iso_service.py` dò tìm tham số "
                    "rẻ nhất (vd tăng service_level, tăng q_factor) để baseline "
                    "đó CŨNG đạt được mức fill rate này, rồi mới so tổng chi "
                    "phí. VD nếu (s,S) cần chi 1.250.000 để đạt 89% fill rate "
                    "còn IPPO chỉ cần 990.000 để đạt 89% — thì ghi là IPPO rẻ "
                    "hơn (990.000-1.250.000)/1.250.000 = **-21%**, tức IPPO rẻ "
                    "hơn 21%. Dòng 'không đạt được mức này' nghĩa là dù quét "
                    "hết lưới tham số, baseline đó vẫn không có cấu hình nào "
                    "vươn tới mức fill rate của IPPO.")

        for img, cap in [("baseline_comparison.png", "Biểu đồ tổng hợp"),
                         ("cost_service_frontier.png", "Đường đánh đổi")]:
            f = RES_DIR / img
            if f.exists():
                with st.expander(cap):
                    st.image(str(f))


# =========================================================================== #
# TAB 5 - MÔ PHỎNG TỪNG NGÀY
# =========================================================================== #
with tab5:
    st.subheader("Chiếu lại một episode 365 ngày")
    st.caption("Chạy trực tiếp môi trường với chính sách được chọn, rồi xem "
               "diễn biến từng ngày: tồn kho chạm trần sức chứa lúc nào, "
               "khi nào hết hàng, agent đặt hàng bao nhiêu.")

    d, meta = load_demand(data_mtime())
    if d is None:
        st.warning("Chưa có dữ liệu.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    ckpts = sorted(f.name for f in CKPT_DIR.glob("*.pth"))
    chinh_sach = c1.selectbox("Chính sách", ["IPPO", "(s,S)", "EOQ", "Newsvendor"])
    ckpt = c2.selectbox("Checkpoint", ckpts or ["(không có)"])
    mien = c3.selectbox("Miền thời gian", ["test", "train"])
    seed = c4.number_input("Seed", 0, 10**6, 1000)

    if st.button("▶️ Chạy mô phỏng", type="primary"):
        from env.inventory_env import MultiWarehouseInventoryEnv
        from baselines.traditional_policies import (
            EOQPolicy, SsPolicy, NewsvendorPolicy, build_env_state)

        calf = None
        cp = DATA_DIR / "calendar_features.npy"
        if cp.exists():
            calf = np.load(cp)
            if calf.size == 0:
                calf = None
        gia_tb = None
        pp = DATA_DIR / "price_per_pair.npy"
        if pp.exists():
            gia_tb = np.load(pp)
        gia_series = None
        ps = DATA_DIR / "price_series.npy"
        if ps.exists():
            gia_series = np.load(ps)
        env = MultiWarehouseInventoryEnv(config=cfg["env"], demand_data=d,
                                         calendar_features=calf,
                                         price_per_pair=gia_tb,
                                         price_series=gia_series, mode=mien)

        agent = None
        if chinh_sach == "IPPO":
            if not ckpts:
                st.error("Chưa có checkpoint — hãy huấn luyện trước.")
                st.stop()
            from agents.ppo_agent import PPOAgent
            agent = PPOAgent(env.obs_per_pair, env.n_pairs, env.n_action_levels,
                             config=cfg["ppo"], device="cpu")
            agent.load(str(CKPT_DIR / ckpt))
            agent.network.eval()
        elif chinh_sach == "EOQ":
            pol = EOQPolicy(env.n_pairs, ordering_cost=env.cp_dh,
                            holding_cost=env.cp_lk, lead_time=env.tg_giao_tb)
        elif chinh_sach == "(s,S)":
            pol = SsPolicy(env.n_pairs, service_level=0.90, lead_time=env.tg_giao_tb,
                           ordering_cost=env.cp_dh, holding_cost=env.cp_lk)
        else:
            pol = NewsvendorPolicy(env.n_pairs, holding_cost=env.cp_lk,
                                   stockout_cost=env.cp_th, lead_time=env.tg_giao_tb)

        obs, _ = env.reset(seed=int(seed))
        rows, util_rows = [], []
        bar = st.progress(0.0, "Đang mô phỏng...")
        for t in range(env.episode_length):
            if agent is not None:
                a, _, _ = agent.select_action(obs, deterministic=True)
            else:
                a = pol.get_action(build_env_state(env))
            obs, _, te, tr, inf = env.step(a)
            rows.append({
                "Ngày": t, "Tồn kho": inf["inventory"], "Cầu": inf["demand"],
                "Bán được": inf["sold"], "Thiếu hàng": inf["stockout"],
                "Đặt hàng": inf["order_qty"], "Số lần đặt": inf["n_orders"],
                "Tràn kho": inf["overflow"], "Fill rate": inf["fill_rate_mean"],
                "Chi phí ngày": (inf["cost_holding"] + inf["cost_stockout"]
                                 + inf["cost_ordering"] + inf["cost_overflow"]),
            })
            util_rows.append(inf["util_wh"].tolist())
            if t % 20 == 0:
                bar.progress(t / env.episode_length, f"Ngày {t}/{env.episode_length}")
            if te or tr:
                break
        bar.empty()
        st.session_state["sim"] = (pd.DataFrame(rows),
                                   pd.DataFrame(util_rows,
                                                columns=meta.get("stores",
                                                                 [f"WH{i}" for i in
                                                                  range(env.n_warehouses)])),
                                   chinh_sach)

    if "sim" in st.session_state:
        df, util, ten = st.session_state["sim"]
        st.success(f"Chính sách **{ten}** — {len(df)} ngày mô phỏng")

        c = st.columns(5)
        c[0].metric("Tổng chi phí", f"{df['Chi phí ngày'].sum():,.0f}")
        fr = 1 - df["Thiếu hàng"].sum() / max(df["Cầu"].sum(), 1)
        c[1].metric("Fill rate cả kỳ", f"{fr:.1%}")
        c[2].metric("Tồn kho TB", f"{df['Tồn kho'].mean():,.0f}")
        c[3].metric("Số lần đặt hàng", f"{int(df['Số lần đặt'].sum()):,}")
        c[4].metric("Đơn vị bị từ chối nhập", f"{df['Tràn kho'].sum():,.0f}")

        ngay = st.slider("Xem đến ngày", 1, len(df), len(df))
        dfx = df.iloc[:ngay]

        c1, c2 = st.columns(2)
        c1.markdown("**Tồn kho vs Cầu mỗi ngày**")
        c1.line_chart(dfx.set_index("Ngày")[["Tồn kho", "Cầu"]], height=250)
        c2.markdown("**Thiếu hàng vs Lượng đặt hàng**")
        c2.line_chart(dfx.set_index("Ngày")[["Thiếu hàng", "Đặt hàng"]], height=250)

        st.markdown("**Mức sử dụng sức chứa từng kho** (1,0 = đầy kho, "
                    "vượt 1,0 thì hàng bị từ chối nhập)")
        st.line_chart(util.iloc[:ngay], height=280)
        st.caption("Đây chính là chỗ các tác tử 'va nhau': 30 SKU trong cùng "
                   "một kho tranh nhau một sức chứa có hạn. Nếu đường này ép "
                   "sát 1,0 liên tục thì ràng buộc đang căng — đó là điều cần "
                   "có để bài toán thực sự là đa tác tử.")
        with st.expander("❓ Ví dụ đọc chỉ số này"):
            st.markdown(
                "Kho sức chứa 500 đơn vị (cộng dồn cả 30 SKU trong kho). Nếu "
                "hôm nay tổng tồn kho cả 30 SKU là 450 -> mức sử dụng = "
                "450/500 = **0,90**. Nếu 1 SKU trong kho đặt thêm 80 đơn vị "
                "(450+80=530 > 500) -> **30 đơn vị vượt bị từ chối ngay lúc "
                "nhập** (vẫn bị tính phí tràn kho), 500 đơn vị còn lại được "
                "lưu bình thường. Mức sử dụng > 1,0 trên đồ thị nghĩa là "
                "NGÀY ĐÓ có từ chối nhập hàng xảy ra.")

        c1, c2 = st.columns(2)
        c1.markdown("**Fill rate cửa sổ trượt 30 ngày**")
        c1.line_chart(dfx.set_index("Ngày")[["Fill rate"]], height=230)
        c2.markdown("**Chi phí phát sinh mỗi ngày**")
        c2.line_chart(dfx.set_index("Ngày")[["Chi phí ngày"]], height=230)

        with st.expander("Bảng số liệu từng ngày"):
            st.dataframe(dfx, width="stretch", height=320)
        st.download_button("⬇️ Tải CSV mô phỏng",
                           df.to_csv(index=False).encode("utf-8"),
                           file_name=f"mo_phong_{ten}.csv", mime="text/csv")


# =========================================================================== #
# TAB 6 - GIẢ LẬP TAY (1 kho - 1 SKU, tự tay đặt hàng từng ngày)
# =========================================================================== #
with tab6:
    st.subheader("Tự tay đặt hàng cho 1 kho - 1 mặt hàng, xem chuyện gì xảy ra")
    st.caption("Tab 5 chạy sẵn 1 chính sách rồi xem lại cả 365 ngày cùng lúc. "
               "Ở đây bạn TỰ MÌNH quyết định đặt bao nhiêu MỖI NGÀY cho DUY "
               "NHẤT 1 cặp kho-SKU, để cảm nhận trực tiếp cơ chế trước khi "
               "nhìn cả 300 cặp chạy tự động.")

    d6, meta6 = load_demand(data_mtime())
    dung_du_lieu_that = d6 is not None

    cs = st.columns(3)
    if dung_du_lieu_that:
        ten_kho_list = meta6.get("stores", [f"WH{i}" for i in range(d6.shape[1])])
        ten_sku_list = meta6.get("top_items", [f"SKU{i}" for i in range(d6.shape[2])])
        w_idx = cs[0].selectbox("Kho", range(len(ten_kho_list)),
                                format_func=lambda i: ten_kho_list[i], key="g6_w")
        s_idx = cs[1].selectbox("Mặt hàng (SKU)", range(len(ten_sku_list)),
                                format_func=lambda i: ten_sku_list[i], key="g6_s")
    else:
        st.info("Chưa có dữ liệu M5 → dùng cầu giả lập ngẫu nhiên "
                "(Poisson, trung bình ~10 đơn vị/ngày).")
        w_idx = s_idx = 0
    so_ngay6 = cs[2].number_input("Số ngày mô phỏng", 10, 90, 30, key="g6_len")

    if st.button("🔄 Bắt đầu / Làm lại", key="g6_reset"):
        from env.inventory_env import MultiWarehouseInventoryEnv
        e6 = dict(cfg["env"])
        e6["n_warehouses"] = 1
        e6["n_skus"] = 1
        e6["episode_length"] = int(so_ngay6)
        e6["warehouse_capacity"] = "auto"
        if dung_du_lieu_that:
            demand_1 = d6[:, w_idx:w_idx + 1, s_idx:s_idx + 1]
            e6["split_day"] = max(demand_1.shape[0] - int(so_ngay6) - 1, 1)
        else:
            demand_1 = None
        env6 = MultiWarehouseInventoryEnv(config=e6, demand_data=demand_1, mode="train")
        obs6, _ = env6.reset(seed=42)
        st.session_state["g6"] = {"env": env6, "history": [], "done": False}

    if "g6" in st.session_state:
        g = st.session_state["g6"]
        env6 = g["env"]

        if g["done"]:
            st.success(f"Đã mô phỏng xong {len(g['history'])} ngày. Bấm "
                       "*Bắt đầu / Làm lại* ở trên để thử lại.")
        else:
            ton_dau_ngay = float(env6.inventory[0])
            sap_ve = float(env6.pipeline_orders[0, 0])
            dong_trang_thai = (f"**Ngày {env6.current_step + 1}/{env6.episode_length}** "
                               f"— tồn kho đầu ngày: **{ton_dau_ngay:.0f}** đơn vị")
            if sap_ve > 0:
                dong_trang_thai += f", hôm nay nhận thêm **{sap_ve:.0f}** đơn vị từ đơn đã đặt trước đó"
            st.markdown(dong_trang_thai)
            st.caption(f"Cầu trung bình lịch sử của cặp này: khoảng "
                       f"{float(env6.mean_demand[0]):.1f} đơn vị/ngày — dùng con "
                       "số này để ước lượng nên đặt bao nhiêu.")

            muc_luong = env6.order_qty_table[0]
            nhan_muc = [f"Mức {i}: đặt {int(q)} đơn vị" for i, q in enumerate(muc_luong)]
            chon = st.radio("Bạn muốn đặt hàng bao nhiêu cho HÔM NAY?",
                            range(len(nhan_muc)), format_func=lambda i: nhan_muc[i],
                            horizontal=True, key=f"g6_act_{env6.current_step}")

            if st.button("✅ Xác nhận — qua ngày tiếp theo",
                         key=f"g6_step_{env6.current_step}", type="primary"):
                _, _, te, tr, inf = env6.step(np.array([chon], dtype=np.int64))
                g["history"].append({
                    "Ngày": env6.current_step, "Tồn kho đầu ngày": ton_dau_ngay,
                    "Đặt hàng": inf["order_qty"], "Cầu": inf["demand"],
                    "Bán được": inf["sold"], "Thiếu hàng": inf["stockout"],
                    "Tồn kho cuối ngày": inf["inventory"],
                    "Chi phí lưu kho": inf["cost_holding"],
                    "Chi phí thiếu hàng": inf["cost_stockout"],
                    "Chi phí đặt hàng": inf["cost_ordering"],
                    "Chi phí tràn kho": inf["cost_overflow"],
                    "Chi phí ngày": (inf["cost_holding"] + inf["cost_stockout"]
                                     + inf["cost_ordering"] + inf["cost_overflow"]),
                })
                g["done"] = bool(te or tr)
                st.rerun()

        if g["history"]:
            dfh = pd.DataFrame(g["history"])
            c = st.columns(4)
            c[0].metric("Tổng chi phí đến giờ", f"{dfh['Chi phí ngày'].sum():,.0f}")
            fr6 = 1 - dfh["Thiếu hàng"].sum() / max(dfh["Cầu"].sum(), 1e-6)
            c[1].metric("Fill rate đến giờ", f"{fr6:.1%}")
            c[2].metric("Số lần đã đặt hàng", int((dfh["Đặt hàng"] > 0).sum()))
            c[3].metric("Tổng đơn vị thiếu hàng", f"{dfh['Thiếu hàng'].sum():,.0f}")

            st.markdown("**Tồn kho vs Cầu qua các ngày bạn đã chơi**")
            st.line_chart(dfh.set_index("Ngày")[["Tồn kho cuối ngày", "Cầu"]], height=240)
            with st.expander("Bảng chi tiết từng ngày bạn đã đặt"):
                st.dataframe(dfh, width="stretch")

        with st.expander("❓ Ví dụ đọc các chỉ số ở tab này"):
            st.markdown(
                "- **Tồn kho đầu ngày**: số hàng còn lại TRƯỚC khi bán hàng "
                "hôm nay (hàng đặt trước có thể đã về thêm vào đây).\n"
                "- **Fill rate đến giờ**: VD tính đến ngày hiện tại khách cần "
                "tổng 40 đơn vị, bạn bán được 34 → fill rate = 34/40 = **85%**. "
                "Thiếu hàng 1-2 ngày không sao, miễn cả kỳ không thiếu triền "
                "miên.\n"
                f"- **Chi phí lưu kho** = tồn kho cuối ngày × cp_lk. VD còn "
                f"20 đơn vị tồn, cp_lk = {cfg['env']['cp_lk']} → chi phí lưu "
                f"kho ngày đó = {20*cfg['env']['cp_lk']:.0f}.\n"
                "- **Chi phí tràn kho**: nếu bạn đặt quá tay và tổng tồn kho "
                "vượt sức chứa, phần vượt bị TỪ CHỐI NGAY LÚC NHẬP — vừa mất "
                "tiền đặt hàng (cp_dh) vừa không có hàng để bán, còn bị tính "
                "thêm phí phạt trên số đơn vị bị từ chối đó.")
