"""Trang 2 - Phiếu mua sắm: mỗi kho mỗi ngày một phiếu, duyệt cả phiếu hoặc bỏ từng dòng."""
import streamlit as st

import core

meta, arr = core.load_demo()
dates = core.test_dates(meta)
P = core.pairs(meta)
store_of = dict(zip(P["Kho"], P["Cửa hàng"]))

LINE_COLS = ["SKU", "Ngành", "Tồn kho", "Hàng đang về", "Nhu cầu TB 7 ngày", "Số lượng đề nghị",
             "Ngày cần hàng", "Đơn giá ($)", "Thành tiền ($)", "Trạng thái"]
FMT = {"Tồn kho": "%.1f", "Hàng đang về": "%.1f", "Nhu cầu TB 7 ngày": "%.1f",
       "Số lượng đề nghị": "%d", "Đơn giá ($)": "%.2f", "Thành tiền ($)": "%.2f",
       "Số dòng": "%d", "Tổng số lượng": "%d"}
COLCFG = {c: st.column_config.NumberColumn(c, format=f) for c, f in FMT.items()}
COLCFG["Bỏ dòng"] = st.column_config.CheckboxColumn("Bỏ dòng", width="small")

core.tieu_de(f"Phiếu mua sắm · {len(store_of)} cửa hàng × {len(meta['skus'])} mặt hàng",
             "Phiếu đề nghị đặt hàng từ mô hình IPPO",
             f"Chọn một ngày trong giai đoạn {dates[0]:%d/%m/%Y} – {dates[-1]:%d/%m/%Y}: mô phỏng chạy tới ngày đó, "
             "mỗi kho có một phiếu gồm các mặt hàng chính sách đề nghị đặt.")

with st.form("loc"):
    c1, c2, c3, c4 = st.columns([1.1, 2.2, 2, 1.2])
    ngay = c1.date_input("Ngày lập", value=dates[-1], min_value=dates[0], max_value=dates[-1],
                         format="DD/MM/YYYY")
    kho = c2.multiselect("Cửa hàng", list(store_of), default=list(store_of),
                         format_func=lambda k: f"{k} · {store_of[k]}")
    nganh = c3.multiselect("Ngành hàng", core.CATEGORIES, default=core.CATEGORIES)
    cs = c4.selectbox("Chính sách", meta["policies"])
    sinh = st.form_submit_button("Lập phiếu", type="primary")

if sinh:
    if not kho or not nganh:
        st.warning("Chọn ít nhất một kho và một ngành hàng.")
    else:
        dong = core.dong_de_nghi(cs, (ngay - dates[0]).days)
        dong = dong[dong["Kho"].isin(kho) & dong["Ngành"].isin(nganh)]
        st.session_state.phieu = {"ngay": ngay, "cs": cs, "df": core.lap_phieu(dong, ngay)}
        st.session_state.ver = st.session_state.get("ver", 0) + 1

phieu = st.session_state.get("phieu")
if phieu is None:
    st.info("Chọn bộ lọc rồi bấm “Lập phiếu”.")
    st.stop()

df = phieu["df"]
if df.empty:
    st.info("Chính sách không đề nghị đặt hàng cho kho và ngành hàng đã chọn trong ngày này.")
    st.stop()

with st.container(border=True, key="ds_phieu"):
    core.muc(1, "Danh sách phiếu", f"{phieu['cs']} · lập ngày {phieu['ngay']:%d/%m/%Y}")
    st.dataframe(core.tom_tat_phieu(df), hide_index=True, column_config=COLCFG)

ma = st.selectbox("Mở phiếu", list(df["Mã phiếu"].unique()))
mo = df[df["Mã phiếu"] == ma]
if thong_bao := st.session_state.pop("thong_bao", None):
    st.toast(thong_bao)
with st.container(border=True, key="chi_tiet_phieu"):
    core.muc(2, "Chi tiết phiếu", f"{ma} · {core.tom_tat_phieu(mo)['Trạng thái'].iloc[0]}")
    view = mo[LINE_COLS].copy()
    view.insert(0, "Bỏ dòng", mo["Trạng thái"] == "Đã bỏ")
    ed = st.data_editor(view, hide_index=True, column_config=COLCFG,
                        disabled=[c for c in view.columns if c != "Bỏ dòng"],
                        key=f"ed_{ma}_{st.session_state.get('ver', 0)}")

b1, b2, _ = st.columns([1, 1, 5])
duyet_bam = b1.button("Duyệt phiếu", type="primary")
tu_choi_bam = b2.button("Từ chối phiếu")
if duyet_bam:
    bo = ed["Bỏ dòng"].to_numpy()
    df.loc[mo.index[bo], "Trạng thái"] = "Đã bỏ"
    df.loc[mo.index[~bo], "Trạng thái"] = "Đã duyệt"
    st.session_state.thong_bao = f"Đã duyệt {ma}: {int((~bo).sum())} dòng gửi phòng mua hàng, bỏ {int(bo.sum())} dòng."
if tu_choi_bam:
    # Tu choi ca phieu: khong dong nao duoc gui, bo qua cac o "Bo dong"
    df.loc[mo.index, "Trạng thái"] = "Từ chối"
    st.session_state.thong_bao = f"Đã từ chối {ma}: không dòng nào được gửi phòng mua hàng."
if duyet_bam or tu_choi_bam:
    st.session_state.ver += 1
    st.rerun()

st.caption("Phạm vi dừng ở phiếu gửi phòng mua hàng. Báo giá, chọn nhà cung cấp và gửi đơn do phòng mua hàng thực hiện.")
duyet = df[df["Trạng thái"] == "Đã duyệt"].assign(**{"Ngày lập": f"{phieu['ngay']:%Y-%m-%d}",
                                                     "Chính sách": phieu["cs"]})
st.download_button("Tải các dòng đã duyệt (CSV)", disabled=duyet.empty,
                   data=duyet[["Mã phiếu", "Ngày lập", "Kho", "Chính sách"] + LINE_COLS]
                   .to_csv(index=False).encode("utf-8-sig"),
                   file_name=f"phieu_da_duyet_{phieu['ngay']:%Y%m%d}.csv", mime="text/csv")
