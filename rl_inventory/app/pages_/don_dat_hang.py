"""Trang 2 - Đơn đặt hàng đề xuất: chọn ngày, kho -> bảng cần đặt gì, kèm tính tiền, duyệt."""
import streamlit as st

import core

meta, arr = core.load_demo()
dates = core.test_dates(meta)
P = core.pairs(meta)
cp_dh = float(meta["env"]["cp_dh"])
store_of = dict(zip(P["Kho"], P["Cửa hàng"]))

LINE_COLS = ["Kho", "SKU", "Ngành", "Tồn kho", "Hàng đang về", "Nhu cầu TB 7 ngày",
             "Số lượng đề xuất", "Đơn giá ($)", "Thành tiền ($)", "Chi phí đặt hàng", "Trạng thái"]
FMT = {"Tồn kho": "%.1f", "Hàng đang về": "%.1f", "Nhu cầu TB 7 ngày": "%.1f",
       "Số lượng đề xuất": "%d", "Đơn giá ($)": "%.2f", "Thành tiền ($)": "%.2f",
       "Chi phí đặt hàng": "%.0f", "Số dòng": "%d"}
COLCFG = {c: st.column_config.NumberColumn(c, format=f) for c, f in FMT.items()}
COLCFG["Chọn"] = st.column_config.CheckboxColumn("Chọn", width="small")

core.tieu_de(f"Đơn đặt hàng · {len(store_of)} kho × {len(meta['skus'])} SKU",
             "Mô phỏng đơn đặt hàng đề xuất",
             f"Miền test {dates[0]:%d/%m/%Y} – {dates[-1]:%d/%m/%Y}; mô phỏng chạy tới ngày đã chọn "
             f"rồi lấy hành động deterministic của chính sách.")

with st.form("loc"):
    c1, c2, c3, c4 = st.columns([1.1, 2.2, 2, 1.2])
    ngay = c1.date_input("Ngày", value=dates[-1], min_value=dates[0], max_value=dates[-1],
                         format="DD/MM/YYYY")
    kho = c2.multiselect("Kho", list(store_of), default=list(store_of)[:1],
                         format_func=lambda k: f"{k} · {store_of[k]}")
    nganh = c3.multiselect("Ngành hàng", core.CATEGORIES, default=core.CATEGORIES)
    cs = c4.selectbox("Chính sách", meta["policies"])
    sinh = st.form_submit_button("Sinh đơn nháp", type="primary")

if sinh:
    if not kho or not nganh:
        st.warning("Chọn ít nhất một kho và một ngành hàng.")
    else:
        day = (ngay - dates[0]).days
        s = core.state_at(cs, day)
        df = P.assign(**{"Tồn kho": s["inv"], "Hàng đang về": s["pipe"],
                         "Nhu cầu TB 7 ngày": s["dem7"], "Số lượng đề xuất": s["qty"],
                         "Đơn giá ($)": arr["price"][day].reshape(-1)})
        df = df[(df["Số lượng đề xuất"] > 0) & df["Kho"].isin(kho) & df["Ngành"].isin(nganh)]
        so = ["Tồn kho", "Hàng đang về", "Nhu cầu TB 7 ngày", "Số lượng đề xuất", "Đơn giá ($)"]
        df = df.astype({c: float for c in so}).round(2)
        df = df.assign(**{"Thành tiền ($)": (df["Số lượng đề xuất"] * df["Đơn giá ($)"]).round(2),
                          "Chi phí đặt hàng": cp_dh, "Trạng thái": "Nháp"})
        st.session_state.don = {"ngay": ngay, "cs": cs, "df": df.reset_index(drop=True)}
        st.session_state.ver = st.session_state.get("ver", 0) + 1

don = st.session_state.get("don")
if don is None:
    st.info("Chọn bộ lọc rồi bấm “Sinh đơn nháp”.")
    st.stop()

df = don["df"]
the = st.container(border=True, key="the_don_nhap")
the.subheader(f"Đơn nháp · {don['cs']} · ngày {don['ngay']:%d/%m/%Y}")
if df.empty:
    st.info("Chính sách không đề xuất đặt hàng cho kho và ngành hàng đã chọn trong ngày này.")
    st.stop()

with the:
    gom = st.toggle("Gom phiếu theo kho × ngành hàng "
                    "(ngành hàng là giả định thay cho nhà cung cấp vì M5 không có dữ liệu nhà cung cấp)")
    if gom:
        view = (df.groupby(["Kho", "Ngành"], as_index=False)
                  .agg(**{"Số dòng": ("SKU", "size"),
                          "Số lượng đề xuất": ("Số lượng đề xuất", "sum"),
                          "Thành tiền ($)": ("Thành tiền ($)", "sum"),
                          "Chi phí đặt hàng": ("Chi phí đặt hàng", "sum"),
                          "Trạng thái": ("Trạng thái",
                                         lambda x: x.iloc[0] if x.nunique() == 1 else "Hỗn hợp")}))
        view.insert(0, "Phiếu", "PH-" + view["Kho"] + "-" + view["Ngành"])
    else:
        view = df[LINE_COLS].copy()
    view.insert(0, "Chọn", False)

    ed = st.data_editor(view, hide_index=True, column_config=COLCFG,
                        disabled=[c for c in view.columns if c != "Chọn"],
                        key=f"ed_{gom}_{st.session_state.get('ver', 0)}")
    st.dataframe(
        [{"": "Tổng", "Số dòng": len(df), "Số lượng đề xuất": int(df["Số lượng đề xuất"].sum()),
          "Thành tiền ($)": df["Thành tiền ($)"].sum(), "Chi phí đặt hàng": df["Chi phí đặt hàng"].sum()}],
        hide_index=True, column_config=COLCFG)
    st.caption("Gom phiếu chỉ là lớp trình bày; chi phí đặt hàng vẫn tính theo từng cặp.")

picked = ed[ed["Chọn"]]
b1, b2, b3 = st.columns([1, 1, 5])
for col, nut, trang_thai in ((b1, "Duyệt", "Đã duyệt"), (b2, "Từ chối", "Từ chối")):
    if col.button(nut, disabled=picked.empty, type="primary" if nut == "Duyệt" else "secondary"):
        if gom:
            chon = set(zip(picked["Kho"], picked["Ngành"]))
            mask = [k in chon for k in zip(df["Kho"], df["Ngành"])]
        else:
            mask = df.index.isin(picked.index)
        df.loc[mask, "Trạng thái"] = trang_thai
        st.session_state.ver += 1
        st.rerun()
dem = df["Trạng thái"].value_counts()
b3.markdown(" · ".join(f"{k}: <span class='num'>{dem.get(k, 0)}</span>"
                       for k in ("Nháp", "Đã duyệt", "Từ chối")), unsafe_allow_html=True)

st.caption("Phạm vi dừng ở bước gửi yêu cầu sang phòng mua hàng.")
duyet = df[df["Trạng thái"] == "Đã duyệt"].assign(Ngày=f"{don['ngay']:%Y-%m-%d}",
                                                   **{"Chính sách": don["cs"]})
st.download_button("Tải đơn đã duyệt (CSV)", disabled=duyet.empty,
                   data=duyet[["Ngày", "Chính sách"] + LINE_COLS].to_csv(index=False).encode("utf-8-sig"),
                   file_name=f"don_da_duyet_{don['ngay']:%Y%m%d}.csv", mime="text/csv")
core.nguon("đơn giá = giá bán lẻ sell_prices M5 tại tuần chứa ngày đã chọn (M5 không có giá nhập)",
           f"chi phí đặt hàng cp_dh = {core.num(cp_dh)} mỗi cặp có đặt hàng (config)")
