"""
app/streamlit_app.py
====================
Bang dieu khien truc quan cho khoa luan "Toi uu hoa chinh sach dat hang lai
trong quan ly ton kho da kho bang hoc tang cuong (IPPO)".

Chay:
    streamlit run app/streamlit_app.py

Muc dich: nhin thay CHUYEN GI DANG XAY RA, thay vi doc log chu chay.
  Tab 1  Du lieu        - cau M5 sau tien xu ly trong nhu the nao
  Tab 2  Cau hinh & Chay- sua tham so va bam nut chay tung buoc
  Tab 3  Huan luyen     - duong hoc theo thoi gian thuc (doc results/train_log*.csv)
  Tab 4  So sanh        - IPPO vs EOQ / (s,S) / Newsvendor
  Tab 5  Mo phong       - chieu lai 365 ngay cua mot chinh sach, theo tung ngay
  Tab 6  Gia lap tay    - tu tay dat hang cho 1 kho-SKU qua vai ngay, de hieu
                          co che truoc khi nhin ca 300 cap cung luc

[V3-7] Cac tab so lieu (3, 4, 5) co them expander "Vi du doc so lieu" giai
       thich bang mot vi du tinh toan cu the, khong chi noi suong ten chi so.
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

st.set_page_config(page_title="IPPO Ton kho da kho", page_icon="📦", layout="wide")

CFG_PATH = ROOT / "config.yaml"
DATA_DIR = ROOT / "data" / "processed"
RES_DIR = ROOT / "results"
CKPT_DIR = ROOT / "checkpoints"


# --------------------------------------------------------------------------- #
# Tien ich
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
    """Chay lenh con va do log ra man hinh theo thoi gian thuc."""
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
st.title("📦 Toi uu chinh sach dat hang lai bang IPPO")
st.caption("Da kho - da SKU | du lieu M5 Walmart | Independent Multi-Agent PPO "
           "voi chia se tham so")

cfg = load_cfg()
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📊 Du lieu", "⚙️ Cau hinh & Chay", "📈 Huan luyen",
     "🏁 So sanh chinh sach", "🎬 Mo phong tung ngay", "🧪 Gia lap tay"])


# =========================================================================== #
# TAB 1 - DU LIEU
# =========================================================================== #
with tab1:
    d, meta = load_demand(data_mtime())
    if d is None:
        st.warning("Chua co du lieu. Sang tab **Cau hinh & Chay** de bam "
                   "*Tien xu ly du lieu*.")
    else:
        T, W, S = d.shape
        split = int(meta.get("split_day", cfg["env"]["split_day"]))
        flat = d[:split].reshape(split, -1)
        md = flat.mean(axis=0)

        c = st.columns(5)
        c[0].metric("So ngay", f"{T:,}")
        c[1].metric("So kho x SKU", f"{W} x {S} = {W*S}")
        c[2].metric("Cau TB / cap / ngay", f"{md.mean():.2f}")
        c[3].metric("Ty le ngay bang 0", f"{(d == 0).mean():.1%}")
        c[4].metric("Cap gan nhu khong ban", f"{(md < 0.1).sum()}")

        st.info(f"Chia thoi gian: huan luyen ngay 0–{split}, danh gia ngay "
                f"{split}–{T}. Hai mien khong chong lan.")

        st.subheader("Cau trung binh mot ngay theo tung kho")
        ten_kho = meta.get("stores", [f"WH_{i}" for i in range(W)])
        cau_kho = md.reshape(W, S).sum(axis=1)
        cov = cfg["env"].get("capacity_cover_days", 5.0)
        df_kho = pd.DataFrame({
            "Kho": ten_kho,
            "Cau/ngay": np.round(cau_kho, 1),
            "Suc chua (auto)": np.round(cau_kho * cov, 0),
        }).set_index("Kho")
        col1, col2 = st.columns([2, 1])
        col1.bar_chart(df_kho[["Cau/ngay"]])
        col2.dataframe(df_kho, width="stretch")
        st.caption(f"Suc chua moi kho = {cov} ngay cau CUA CHINH KHO DO. Day la "
                   "diem sua quan trong: ban cu dung mot con so chung cho ca "
                   f"{W} kho, nen kho lon nhat chi chua duoc ~1,4 ngay cau.")

        with st.expander("❓ Vi du doc con so nay"):
            vd_kho = df_kho.index[0]
            vd_cau = df_kho["Cau/ngay"].iloc[0]
            vd_sc = df_kho["Suc chua (auto)"].iloc[0]
            st.markdown(
                f"Kho **{vd_kho}** ban trung binh **{vd_cau:.0f} don vi/ngay** "
                f"(cong don ca {S} SKU trong kho). Voi capacity_cover_days = "
                f"**{cov}**, suc chua tinh ra la {vd_cau:.0f} × {cov} = "
                f"**{vd_sc:.0f} don vi** — du tru cho {cov} ngay ban trung binh. "
                f"Neu tong ton kho CA {S} SKU trong kho **{vd_kho}** cong lai "
                f"vuot qua {vd_sc:.0f}, phan vuot bi TU CHOI ngay luc nhap hang "
                f"(khong duoc luu kho), du don da dat va da tra tien dat hang.")

        st.subheader("Phan bo quy mo cau cua 300 cap (kho, SKU)")
        hist = pd.DataFrame({"cau_tb": md})
        st.bar_chart(np.histogram(np.log10(np.maximum(md, 1e-2)), bins=25)[0])
        st.caption("Truc hoanh: log10(cau trung binh/ngay). Duoi cang dai thi "
                   "mang dung chung trong so cang kho tong quat hoa.")

        st.subheader("Nhu cau theo thoi gian (tong toan he thong)")
        n_smooth = st.slider("Lam muot (so ngay)", 1, 60, 28, key="sm_data")
        ts = pd.Series(d.reshape(T, -1).sum(axis=1)).rolling(n_smooth).mean()
        st.line_chart(ts, height=240)


# =========================================================================== #
# TAB 2 - CAU HINH & CHAY
# =========================================================================== #
with tab2:
    st.subheader("Tham so chinh")
    st.caption("Sua o day roi bam *Luu cau hinh*; cac script deu doc tu "
               "config.yaml nen khong can sua code.")

    e = cfg["env"]
    p = cfg["ppo"]
    pre = cfg["preprocess"]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Moi truong**")
        e["n_warehouses"] = st.number_input("So kho", 1, 10, int(e["n_warehouses"]))
        e["n_skus"] = st.number_input("So SKU", 5, 100, int(e["n_skus"]))
        e["episode_length"] = st.number_input("Do dai episode (ngay)", 30, 730,
                                              int(e["episode_length"]), step=5)
        e["capacity_cover_days"] = st.number_input(
            "Suc chua kho (so ngay cau)", 1.0, 30.0,
            float(e["capacity_cover_days"]), step=0.5)
        pre["min_mean_demand"] = st.number_input(
            "Nguong loc SKU (dv/ngay)", 0.0, 5.0,
            float(pre.get("min_mean_demand", 0.2)), step=0.05)
    with c2:
        st.markdown("**Chi phi**")
        e["cp_lk"] = st.number_input("Luu kho / dv / ngay", 0.0, 100.0, float(e["cp_lk"]))
        e["cp_th"] = st.number_input("Thieu hang / dv", 0.0, 500.0, float(e["cp_th"]))
        e["cp_dh"] = st.number_input("Dat hang / lan", 0.0, 200.0, float(e["cp_dh"]))
        e["pt_tk"] = st.number_input("Phat tran kho / dv", 0.0, 200.0, float(e["pt_tk"]))
        e["phi_dv"] = st.number_input("He so phat muc phuc vu", 0.0, 50.0, float(e["phi_dv"]))
        e["muc_dv"] = st.number_input("Fill rate muc tieu", 0.0, 1.0,
                                      float(e["muc_dv"]), step=0.01)
    with c3:
        st.markdown("**PPO**")
        p["total_episodes"] = st.number_input("So episode huan luyen", 10, 20000,
                                              int(p["total_episodes"]), step=50)
        p["n_steps"] = st.number_input("n_steps / update", 128, 8192,
                                       int(p["n_steps"]), step=128)
        p["lr_actor"] = st.number_input("lr actor", 1e-6, 1e-2,
                                        float(p["lr_actor"]), format="%.6f")
        p["lr_critic"] = st.number_input("lr critic", 1e-6, 1e-2,
                                         float(p["lr_critic"]), format="%.6f")
        p["ent_coef"] = st.number_input("ent_coef (dau)", 0.0, 0.2,
                                        float(p["ent_coef"]), step=0.001, format="%.3f")
        e["normalize_reward_per_pair"] = st.checkbox(
            "Chuan hoa phan thuong theo tung cap (BAT = hoi tu)",
            bool(e.get("normalize_reward_per_pair", True)))

    if st.button("💾 Luu cau hinh", type="primary"):
        cfg["env"], cfg["ppo"], cfg["preprocess"] = e, p, pre
        save_cfg(cfg)
        st.success("Da luu config.yaml")

    st.divider()
    st.subheader("Chay tung buoc")
    st.caption("Thu tu chuan: 1 → 2 → 3 → 4. Buoc 3 la buoc lau nhat.")

    b1, b2, b3, b4 = st.columns(4)
    out = st.empty()
    if b1.button("1. Tien xu ly du lieu"):
        chay_lenh([sys.executable, "scripts/data_preprocessing.py", "--m5",
                   "--n_warehouses", str(e["n_warehouses"]),
                   "--n_skus", str(e["n_skus"]),
                   "--min_mean_demand", str(pre.get("min_mean_demand", 0.2)),
                   "--sku_selection", str(pre.get("sku_selection", "stratified"))], out)
        st.cache_data.clear()
    if b2.button("2. Tinh chinh baseline"):
        chay_lenh([sys.executable, "scripts/tune_baselines.py", "--episodes", "2"], out)
    if b3.button("3. Huan luyen IPPO"):
        chay_lenh([sys.executable, "scripts/train.py",
                   "--episodes", str(p["total_episodes"])], out)
    if b4.button("4. Danh gia & so sanh"):
        chay_lenh([sys.executable, "scripts/evaluate.py"], out)

    if st.button("5. So sanh o cung muc phuc vu (iso-service)"):
        chay_lenh([sys.executable, "scripts/iso_service.py",
                   "--tune_episodes", "2", "--eval_episodes", "10"], out)

    st.divider()
    st.caption("Checkpoint hien co: " +
               (", ".join(f.name for f in sorted(CKPT_DIR.glob('*.pth')))
                or "chua co"))


# =========================================================================== #
# TAB 3 - HUAN LUYEN
# =========================================================================== #
with tab3:
    kq, ten_files = doc_train_log()
    if kq is None:
        st.warning("Chua co results/train_log*.csv. Hay chay huan luyen o tab 2.")
    else:
        df, path = kq
        if ten_files and len(ten_files) > 1:
            st.selectbox("Chon lan chay", ten_files, key="chon_log")
        c = st.columns(4)
        c[0].metric("Episode da chay", f"{int(df.episode.max()):,}")
        c[1].metric("Fill rate (20 ep cuoi)", f"{df.fill_rate.tail(20).mean():.1%}")
        c[2].metric("Chi phi/episode (20 ep cuoi)",
                    f"{df.cost_total.tail(20).mean():,.0f}")
        c[3].metric("Thoi gian", f"{df.elapsed_s.max()/60:.1f} phut")

        if st.checkbox("Tu dong lam moi moi 10 giay (khi dang chay)"):
            time.sleep(10)
            st.rerun()

        st.subheader("Duong hoc")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Phan thuong (cang cao cang tot)**")
            st.line_chart(df.set_index("episode")[["reward_raw", "reward_smooth"]],
                          height=250)
            st.markdown("**Fill rate**")
            st.line_chart(df.set_index("episode")[["fill_rate"]], height=200)
        with c2:
            st.markdown("**Entropy (phai giam dan → chinh sach dinh hinh)**")
            st.line_chart(df.set_index("episode")[["entropy"]].dropna(), height=250)
            st.markdown("**Explained variance (critic; phai tien ve 1)**")
            st.line_chart(df.set_index("episode")[["explained_variance"]].dropna(),
                          height=200)

        with st.expander("❓ Vi du doc 3 chi so tren"):
            st.markdown(
                "- **Fill rate**: ty le nhu cau duoc dap ung trong cua so 30 "
                "ngay gan nhat. VD 30 ngay qua khach can tong 300 don vi, kho "
                "ban duoc 255 -> fill rate = 255/300 = **85%**. Muc tieu cua "
                f"khoa luan la {cfg['env']['muc_dv']*100:.0f}%.\n"
                "- **Entropy**: do 'phan van' cua chinh sach khi chon 1 trong "
                "6 muc dat hang. Chon deu ca 6 muc (ngau nhien hoan toan) cho "
                "entropy = ln(6) ≈ **1,79** — dung o dau qua trinh hoc, luc "
                "con dang tham do. Entropy -> 0 nghia la agent gan nhu LUON "
                "chon 1 muc co dinh cho moi trang thai — da 'chot' chinh sach. "
                "Neu no giam ve gan 0 CHI SAU VAI CHUC episode dau (chua kip "
                "hoc gi) thi do la dau hieu dong cung som vao mot nghiem toi, "
                "khong phai hoi tu that.\n"
                "- **Explained variance**: critic du doan gia tri trang thai "
                "tot toi dau. VD return that su la [-100, -80, -120], neu "
                "critic doan dung xu huong thi explained variance gan **1**; "
                "neu du doan khong hon gi viec doan trung binh (-100 cho ca 3) "
                "thi no gan **0**; am nghia la du doan CON TE HON ca doan bua "
                "trung binh.")

        st.subheader("Co cau chi phi qua qua trinh hoc")
        cols = ["cost_holding", "cost_stockout", "cost_ordering",
                "cost_overflow", "cost_service_penalty"]
        st.area_chart(df.set_index("episode")[cols], height=260)

        with st.expander("Chan doan on dinh (approx_kl, clip_frac)"):
            st.line_chart(df.set_index("episode")[["approx_kl", "clip_frac"]].dropna(),
                          height=220)
            st.caption("approx_kl on dinh quanh 0,005–0,02 la tot. Vot len > 0,05 "
                       "lien tuc = lr_actor qua lon.")

        if "eval_fill" in df and df.eval_fill.notna().any():
            st.subheader("Danh gia deterministic (argmax) tren mien TEST")
            ev = df[["episode", "eval_reward", "eval_fill"]].dropna()
            ev = ev.drop_duplicates(subset=["episode"]).set_index("episode")
            st.line_chart(ev, height=230)
            st.caption("Duong nay moi la chat luong chinh sach that. Neu no on "
                       "dinh dan trong khi reward luc lay mau van nhay, do chi "
                       "la nhieu do explore.")


# =========================================================================== #
# TAB 4 - SO SANH
# =========================================================================== #
with tab4:
    sp = RES_DIR / "summary.json"
    bp = RES_DIR / "baseline_comparison.csv"
    if not bp.exists():
        st.warning("Chua co ket qua danh gia. Chay buoc 4 o tab 2.")
    else:
        summ = pd.read_csv(bp)
        stat = {}
        if sp.exists():
            stat = json.loads(sp.read_text(encoding="utf-8")).get("stat", {})

        bang = pd.DataFrame({
            "Chinh sach": summ.policy,
            "Tong chi phi": summ.total_cost_mean.round(0),
            "± do lech": summ.total_cost_std.round(0),
            "Luu kho": summ.holding_cost_mean.round(0),
            "Thieu hang": summ.stockout_cost_mean.round(0),
            "Dat hang": summ.ordering_cost_mean.round(0),
            "Tran kho": summ.overflow_cost_mean.round(0),
            "Fill rate": (summ.fill_rate_mean * 100).round(1),
            "Ton kho TB": summ.avg_inventory_mean.round(0),
        }).set_index("Chinh sach")
        st.dataframe(bang, width="stretch")

        with st.expander("❓ Vi du doc bang nay"):
            hang0 = bang.index[0]
            r0 = bang.iloc[0]
            st.markdown(
                f"Doc hang **{hang0}**: tong chi phi van hanh trung binh mot "
                f"episode (365 ngay) la **{r0['Tong chi phi']:,.0f}**, trong do "
                f"lưu kho {r0['Luu kho']:,.0f} + thieu hang {r0['Thieu hang']:,.0f} "
                f"+ dat hang {r0['Dat hang']:,.0f} + tran kho {r0['Tran kho']:,.0f} "
                f"= tong. Fill rate {r0['Fill rate']:.1f}% nghia la chinh sach "
                f"nay dap ung duoc chung do % nhu cau trong ca nam.\n\n"
                "So SAI lam pho bien: thay IPPO co tong chi phi cao hon 1 "
                "baseline roi ket luan 'IPPO kem hon' — trong khi hai chinh "
                "sach co the co fill rate khac han nhau (chi phi thap de dat "
                "duoc neu chap nhan thieu hang nhieu). Phai so o **cung mot "
                "muc fill rate** moi cong bang — xem bang 'So sanh o CUNG MUC "
                "PHUC VU' phia duoi.")

        if stat:
            gap = stat.get("gap_percent", 0.0)
            c = st.columns(4)
            c[0].metric("Baseline tot nhat", stat.get("best_baseline", "-"))
            c[1].metric("Chenh lech chi phi", f"{gap:+.1f}%",
                        delta=f"{'IPPO tot hon' if gap < 0 else 'IPPO kem hon'}",
                        delta_color="normal" if gap < 0 else "inverse")
            c[2].metric("p (Welch t-test)", f"{stat.get('p_ttest', float('nan')):.2e}")
            c[3].metric("Cohen's d", f"{stat.get('cohen_d', float('nan')):.2f}")

        c1, c2 = st.columns(2)
        c1.markdown("**Tong chi phi van hanh**")
        c1.bar_chart(bang[["Tong chi phi"]], height=280)
        c2.markdown("**Fill rate (%) — duong muc tieu 85%**")
        c2.bar_chart(bang[["Fill rate"]], height=280)

        st.markdown("**Co cau chi phi**")
        st.bar_chart(bang[["Luu kho", "Thieu hang", "Dat hang", "Tran kho"]],
                     height=300, stack=True)

        st.markdown("**Danh doi chi phi — muc phuc vu** (tot = duoi, ben phai)")
        st.scatter_chart(bang.reset_index(), x="Fill rate", y="Tong chi phi",
                         color="Chinh sach", height=320)

        iso_p = RES_DIR / "iso_service.json"
        if iso_p.exists():
            st.divider()
            st.subheader("So sanh o CUNG MUC PHUC VU")
            iso = json.loads(iso_p.read_text(encoding="utf-8"))
            st.caption(
                f"So chi phi giua hai chinh sach co fill rate khac nhau la vo "
                f"nghia. Bang duoi day tim cau hinh RE NHAT cua tung baseline "
                f"ma van dat fill rate >= {iso['target_fill']:.1%} (muc IPPO dat "
                f"duoc), roi moi so chi phi.")
            hang, ippo_cost = [], iso["ket_qua"].get("IPPO", {}).get("cost")
            for ten, r in iso["ket_qua"].items():
                if r.get("khong_dat"):
                    hang.append({"Chinh sach": ten, "Tong chi phi": None,
                                 "Fill rate": None,
                                 "Chenh lech vs IPPO": "khong dat duoc muc nay"})
                    continue
                d = ("" if ten == "IPPO" else
                     f"{100*(ippo_cost - r['cost'])/r['cost']:+.1f}%")
                hang.append({"Chinh sach": ten,
                             "Tong chi phi": round(r["cost"]),
                             "Fill rate": round(r["fill"] * 100, 2),
                             "Chenh lech vs IPPO": d})
            st.dataframe(pd.DataFrame(hang).set_index("Chinh sach"),
                         width="stretch")
            st.caption("Cot cuoi: so AM nghia la IPPO RE HON o cung muc phuc vu. "
                       "Day moi la bang nen dua vao Chuong 4.")
            with st.expander("❓ Vi du doc bang iso-service"):
                st.markdown(
                    f"IPPO tu no dat fill rate {iso['target_fill']*100:.1f}%. "
                    "Voi TUNG baseline, script `iso_service.py` do tim tham so "
                    "re nhat (vd tang service_level, tang q_factor) de baseline "
                    "do CUNG dat duoc muc fill rate nay, roi moi so tong chi "
                    "phi. VD neu (s,S) can chi 1.250.000 de dat 89% fill rate "
                    "con IPPO chi can 990.000 de dat 89% — thi ghi la IPPO re "
                    "hon (990.000-1.250.000)/1.250.000 = **-21%**, tuc IPPO re "
                    "hon 21%. Dong 'khong dat duoc muc nay' nghia la du quet "
                    "het luoi tham so, baseline do van khong co cau hinh nao "
                    "vuon toi muc fill rate cua IPPO.")

        for img, cap in [("baseline_comparison.png", "Bieu do tong hop"),
                         ("cost_service_frontier.png", "Duong danh doi")]:
            f = RES_DIR / img
            if f.exists():
                with st.expander(cap):
                    st.image(str(f))


# =========================================================================== #
# TAB 5 - MO PHONG TUNG NGAY
# =========================================================================== #
with tab5:
    st.subheader("Chieu lai mot episode 365 ngay")
    st.caption("Chay truc tiep moi truong voi chinh sach duoc chon, roi xem "
               "dien bien tung ngay: ton kho cham tran suc chua luc nao, "
               "khi nao het hang, agent dat hang bao nhieu.")

    d, meta = load_demand(data_mtime())
    if d is None:
        st.warning("Chua co du lieu.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    ckpts = sorted(f.name for f in CKPT_DIR.glob("*.pth"))
    chinh_sach = c1.selectbox("Chinh sach", ["IPPO", "(s,S)", "EOQ", "Newsvendor"])
    ckpt = c2.selectbox("Checkpoint", ckpts or ["(khong co)"])
    mien = c3.selectbox("Mien thoi gian", ["test", "train"])
    seed = c4.number_input("Seed", 0, 10**6, 1000)

    if st.button("▶️ Chay mo phong", type="primary"):
        from env.inventory_env import MultiWarehouseInventoryEnv
        from baselines.traditional_policies import (
            EOQPolicy, SsPolicy, NewsvendorPolicy, build_env_state)

        calf = None
        cp = DATA_DIR / "calendar_features.npy"
        if cp.exists():
            calf = np.load(cp)
            if calf.size == 0:
                calf = None
        env = MultiWarehouseInventoryEnv(config=cfg["env"], demand_data=d,
                                         calendar_features=calf, mode=mien)

        agent = None
        if chinh_sach == "IPPO":
            if not ckpts:
                st.error("Chua co checkpoint — hay huan luyen truoc.")
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
        bar = st.progress(0.0, "Dang mo phong...")
        for t in range(env.episode_length):
            if agent is not None:
                a, _, _ = agent.select_action(obs, deterministic=True)
            else:
                a = pol.get_action(build_env_state(env))
            obs, _, te, tr, inf = env.step(a)
            rows.append({
                "ngay": t, "Ton kho": inf["inventory"], "Cau": inf["demand"],
                "Ban duoc": inf["sold"], "Thieu hang": inf["stockout"],
                "Dat hang": inf["order_qty"], "So lan dat": inf["n_orders"],
                "Tran kho": inf["overflow"], "Fill rate": inf["fill_rate_mean"],
                "Chi phi ngay": (inf["cost_holding"] + inf["cost_stockout"]
                                 + inf["cost_ordering"] + inf["cost_overflow"]),
            })
            util_rows.append(inf["util_wh"].tolist())
            if t % 20 == 0:
                bar.progress(t / env.episode_length, f"Ngay {t}/{env.episode_length}")
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
        st.success(f"Chinh sach **{ten}** — {len(df)} ngay mo phong")

        c = st.columns(5)
        c[0].metric("Tong chi phi", f"{df['Chi phi ngay'].sum():,.0f}")
        fr = 1 - df["Thieu hang"].sum() / max(df["Cau"].sum(), 1)
        c[1].metric("Fill rate ca ky", f"{fr:.1%}")
        c[2].metric("Ton kho TB", f"{df['Ton kho'].mean():,.0f}")
        c[3].metric("So lan dat hang", f"{int(df['So lan dat'].sum()):,}")
        c[4].metric("Don vi bi tu choi nhap", f"{df['Tran kho'].sum():,.0f}")

        ngay = st.slider("Xem den ngay", 1, len(df), len(df))
        dfx = df.iloc[:ngay]

        c1, c2 = st.columns(2)
        c1.markdown("**Ton kho vs Cau moi ngay**")
        c1.line_chart(dfx.set_index("ngay")[["Ton kho", "Cau"]], height=250)
        c2.markdown("**Thieu hang vs Luong dat hang**")
        c2.line_chart(dfx.set_index("ngay")[["Thieu hang", "Dat hang"]], height=250)

        st.markdown("**Muc su dung suc chua tung kho** (1,0 = day kho, "
                    "vuot 1,0 thi hang bi tu choi nhap)")
        st.line_chart(util.iloc[:ngay], height=280)
        st.caption("Day chinh la cho cac tac tu 'va nhau': 30 SKU trong cung "
                   "mot kho tranh nhau mot suc chua co han. Neu duong nay ep "
                   "sat 1,0 lien tuc thi rang buoc dang can — do la dieu can "
                   "co de bai toan thuc su la da tac tu.")
        with st.expander("❓ Vi du doc chi so nay"):
            st.markdown(
                "Kho suc chua 500 don vi (cong don ca 30 SKU trong kho). Neu "
                "hom nay tong ton kho ca 30 SKU la 450 -> muc su dung = "
                "450/500 = **0,90**. Neu 1 SKU trong kho dat them 80 don vi "
                "(450+80=530 > 500) -> **30 don vi vuot bi tu choi ngay luc "
                "nhap** (van bi tinh phi tran kho), 500 don vi con lai duoc "
                "luu binh thuong. Muc su dung > 1,0 tren do thi nghia la "
                "NGAY DO co tu choi nhap hang xay ra.")

        c1, c2 = st.columns(2)
        c1.markdown("**Fill rate cua so truot 30 ngay**")
        c1.line_chart(dfx.set_index("ngay")[["Fill rate"]], height=230)
        c2.markdown("**Chi phi phat sinh moi ngay**")
        c2.line_chart(dfx.set_index("ngay")[["Chi phi ngay"]], height=230)

        with st.expander("Bang so lieu tung ngay"):
            st.dataframe(dfx, width="stretch", height=320)
        st.download_button("⬇️ Tai CSV mo phong",
                           df.to_csv(index=False).encode("utf-8"),
                           file_name=f"mo_phong_{ten}.csv", mime="text/csv")


# =========================================================================== #
# TAB 6 - GIA LAP TAY (1 kho - 1 SKU, tu tay dat hang tung ngay)
# =========================================================================== #
with tab6:
    st.subheader("Tu tay dat hang cho 1 kho - 1 mat hang, xem chuyen gi xay ra")
    st.caption("Ban 5 chay san 1 chinh sach roi xem lai ca 365 ngay cung luc. "
               "O day ban TU MINH quyet dinh dat bao nhieu MOI NGAY cho DUY "
               "NHAT 1 cap kho-SKU, de cam nhan truc tiep co che truoc khi "
               "nhin ca 300 cap chay tu dong.")

    d6, meta6 = load_demand(data_mtime())
    dung_du_lieu_that = d6 is not None

    cs = st.columns(3)
    if dung_du_lieu_that:
        ten_kho_list = meta6.get("stores", [f"WH{i}" for i in range(d6.shape[1])])
        ten_sku_list = meta6.get("top_items", [f"SKU{i}" for i in range(d6.shape[2])])
        w_idx = cs[0].selectbox("Kho", range(len(ten_kho_list)),
                                format_func=lambda i: ten_kho_list[i], key="g6_w")
        s_idx = cs[1].selectbox("Mat hang (SKU)", range(len(ten_sku_list)),
                                format_func=lambda i: ten_sku_list[i], key="g6_s")
    else:
        st.info("Chua co du lieu M5 → dung cau gia lap ngau nhien "
                "(Poisson, trung binh ~10 don vi/ngay).")
        w_idx = s_idx = 0
    so_ngay6 = cs[2].number_input("So ngay mo phong", 10, 90, 30, key="g6_len")

    if st.button("🔄 Bat dau / Lam lai", key="g6_reset"):
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
            st.success(f"Da mo phong xong {len(g['history'])} ngay. Bam "
                       "*Bat dau / Lam lai* o tren de thu lai.")
        else:
            ton_dau_ngay = float(env6.inventory[0])
            sap_ve = float(env6.pipeline_orders[0, 0])
            dong_trang_thai = (f"**Ngay {env6.current_step + 1}/{env6.episode_length}** "
                               f"— ton kho dau ngay: **{ton_dau_ngay:.0f}** don vi")
            if sap_ve > 0:
                dong_trang_thai += f", hom nay nhan them **{sap_ve:.0f}** don vi tu don da dat truoc do"
            st.markdown(dong_trang_thai)
            st.caption(f"Cau trung binh lich su cua cap nay: khoang "
                       f"{float(env6.mean_demand[0]):.1f} don vi/ngay — dung con "
                       "so nay de uoc luong nen dat bao nhieu.")

            muc_luong = env6.order_qty_table[0]
            nhan_muc = [f"Muc {i}: dat {int(q)} don vi" for i, q in enumerate(muc_luong)]
            chon = st.radio("Ban muon dat hang bao nhieu cho HOM NAY?",
                            range(len(nhan_muc)), format_func=lambda i: nhan_muc[i],
                            horizontal=True, key=f"g6_act_{env6.current_step}")

            if st.button("✅ Xac nhan — qua ngay tiep theo",
                         key=f"g6_step_{env6.current_step}", type="primary"):
                _, _, te, tr, inf = env6.step(np.array([chon], dtype=np.int64))
                g["history"].append({
                    "Ngay": env6.current_step, "Ton kho dau ngay": ton_dau_ngay,
                    "Dat hang": inf["order_qty"], "Cau": inf["demand"],
                    "Ban duoc": inf["sold"], "Thieu hang": inf["stockout"],
                    "Ton kho cuoi ngay": inf["inventory"],
                    "Chi phi luu kho": inf["cost_holding"],
                    "Chi phi thieu hang": inf["cost_stockout"],
                    "Chi phi dat hang": inf["cost_ordering"],
                    "Chi phi tran kho": inf["cost_overflow"],
                    "Chi phi ngay": (inf["cost_holding"] + inf["cost_stockout"]
                                     + inf["cost_ordering"] + inf["cost_overflow"]),
                })
                g["done"] = bool(te or tr)
                st.rerun()

        if g["history"]:
            dfh = pd.DataFrame(g["history"])
            c = st.columns(4)
            c[0].metric("Tong chi phi den gio", f"{dfh['Chi phi ngay'].sum():,.0f}")
            fr6 = 1 - dfh["Thieu hang"].sum() / max(dfh["Cau"].sum(), 1e-6)
            c[1].metric("Fill rate den gio", f"{fr6:.1%}")
            c[2].metric("So lan da dat hang", int((dfh["Dat hang"] > 0).sum()))
            c[3].metric("Tong don vi thieu hang", f"{dfh['Thieu hang'].sum():,.0f}")

            st.markdown("**Ton kho vs Cau qua cac ngay ban da choi**")
            st.line_chart(dfh.set_index("Ngay")[["Ton kho cuoi ngay", "Cau"]], height=240)
            with st.expander("Bang chi tiet tung ngay ban da dat"):
                st.dataframe(dfh, width="stretch")

        with st.expander("❓ Vi du doc cac chi so o tab nay"):
            st.markdown(
                "- **Ton kho dau ngay**: so hang con lai TRUOC khi ban hang "
                "hom nay (hang dat truoc co the da ve them vao day).\n"
                "- **Fill rate den gio**: VD tinh den ngay hien tai khach can "
                "tong 40 don vi, ban ban duoc 34 → fill rate = 34/40 = **85%**. "
                "Thieu hang 1-2 ngay khong sao, mien ca ky khong thieu trien "
                "mien.\n"
                f"- **Chi phi luu kho** = ton kho cuoi ngay × cp_lk. VD con "
                f"20 don vi ton, cp_lk = {cfg['env']['cp_lk']} → chi phi luu "
                f"kho ngay do = {20*cfg['env']['cp_lk']:.0f}.\n"
                "- **Chi phi tran kho**: neu ban dat qua tay va tong ton kho "
                "vuot suc chua, phan vuot bi TU CHOI NGAY LUC NHAP — vua mat "
                "tien dat hang (cp_dh) vua khong co hang de ban, con bi tinh "
                "them phi phat tren so don vi bi tu choi do.")
