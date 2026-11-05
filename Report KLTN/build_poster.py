"""Dựng poster KLTN trên khung mẫu Khoa CNTT TDTU (Report KLTN/Poster.pptx)."""
import copy
import io
import os
import sys

import qrcode
from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
SRC, OUT = f"{ROOT}/Poster.pptx", sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/Poster_KLTN.pptx"
REPO_URL = "https://github.com/MinhLysA/Khoa_luan"

# Bảng màu lấy từ khung nền Khoa CNTT: tím mận làm chủ đạo, cam hổ phách làm điểm nhấn
PURPLE = RGBColor(0x4A, 0x1A, 0x45)   # tím mận đậm: tiêu đề, khối chính
PLUM_MID = RGBColor(0x7A, 0x3B, 0x6E)  # tím mận vừa: biểu tượng phụ
ORANGE = RGBColor(0xE0, 0x81, 0x2A)   # cam hổ phách: số nổi bật, IPPO
TINT = RGBColor(0xF7, 0xF1, 0xF6)     # nền thẻ tím rất nhạt
TEXT = RGBColor(0x2A, 0x2A, 0x2A)
MUTED = RGBColor(0x66, 0x66, 0x66)
GREY_BAR = RGBColor(0xCD, 0xBF, 0xCB)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Arial"
BODY = 21

prs = Presentation(SRC)
s = prs.slides[0]

# 1. Bỏ nội dung đề tài cũ (bảng + 3 hình), giữ ảnh nền và các ô tiêu đề
for sh in list(s.shapes):
    if sh.has_table or sh.shape_type == 13:
        if sh.shape_type == 13:
            rid = sh._element.blipFill.blip.rEmbed
            s.part.drop_rel(rid)
        sh._element.getparent().remove(sh._element)


# 2. Ô tiêu đề: giữ định dạng run đầu, thay chữ
def set_runs(shape, parts, bold=False):
    """parts: list[(text, superscript?)] — dùng rPr của run đầu tiên.
    Font SVN-Gilroy của mẫu thường không có trên máy in, thay bằng Arial và bỏ giãn chữ (spc)."""
    p = shape.text_frame.paragraphs[0]
    runs = p.runs
    rpr = copy.deepcopy(runs[0]._r.find(qn("a:rPr")))
    rpr.attrib.pop("spc", None)
    rpr.set("b", "1" if bold else "0")
    for tag in ("a:latin", "a:ea", "a:cs"):
        el = rpr.find(qn(tag))
        if el is not None:
            el.set("typeface", "Arial")
            el.attrib.pop("panose", None)
    for r in runs:
        p._p.remove(r._r)
    for extra in shape.text_frame.paragraphs[1:]:
        extra._p.getparent().remove(extra._p)
    end = p._p.find(qn("a:endParaRPr"))
    for text, sup in parts:
        r = etree.SubElement(p._p, qn("a:r"))
        rp = copy.deepcopy(rpr)
        rp.attrib.pop("baseline", None)
        if sup:
            rp.set("baseline", "30000")
        r.append(rp)
        t = etree.SubElement(r, qn("a:t"))
        t.text = text
        if end is not None:
            p._p.remove(end)
            p._p.append(end)


boxes = [sh for sh in s.shapes if sh.has_text_frame]
head = lambda pre: next(b for b in boxes if b.text_frame.text.strip().startswith(pre))
set_runs(head("DỰ ÁN CÔNG N"), [("KHÓA LUẬN TỐT NGHIỆP", False)], bold=True)
set_runs(head("Năm triển kh"), [("Năm triển khai: 2026", False)])
# Tên đề tài như poster mẫu của Khoa: viết hoa đầu câu, không có chữ "Đề tài:"
title = head("Đề tài: Phát")
set_runs(title, [("Xây dựng mô hình học tăng cường tối ưu hóa chính sách đặt hàng lại "
                  "trong quản lý tồn kho đa kho", False)], bold=True)
for r in title.text_frame.paragraphs[0].runs:
    r.font.size = Pt(42)
title.top, title.height = Inches(3.3), Inches(2.1)
authors = head("Nguyễn Hoàng")
set_runs(authors, [("Lê Minh Lý", False), ("1", True), (", Sử Thị Yến Linh", False),
                   ("1", True), (", TS. Hồ Thị Linh", False), ("*", True)])
authors.top = Inches(5.55)
aff = head("1 Sinh viên n")
set_runs(aff, [("1", True), (" Sinh viên ngành Mạng máy tính và Truyền thông dữ liệu, ", False),
               ("*", True), (" GVHD Khoa Công nghệ thông tin", False)])
aff.top, aff.height = Inches(6.6), Inches(1.0)

# 3. QR vào ô "Xem chi tiết" (ô trắng x 18,65–21,97; y 3,65–6,99)
qr = qrcode.QRCode(border=1, box_size=20)
qr.add_data(REPO_URL)
buf = io.BytesIO()
qr.make_image(fill_color="black", back_color="white").save(buf, format="PNG")
buf.seek(0)
pic = s.shapes.add_picture(buf, Inches(18.96), Inches(3.97), Inches(2.7), Inches(2.7))
pic.name = "QR kho mã nguồn"


# ---------- tiện ích ----------
def textbox(x, y, w, h, name):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tb, tf


def style(run, size=BODY, bold=False, color=TEXT, italic=False):
    f = run.font
    f.name, f.size, f.bold, f.italic = FONT, Pt(size), bold, italic
    f.color.rgb = color


def add_par(tf, segments, size=BODY, align=PP_ALIGN.JUSTIFY, bullet=False, after=6, first=False):
    """segments: str hoặc list[(text, dict style)]"""
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_after = Pt(after)
    if isinstance(segments, str):
        segments = [(segments, {})]
    for text, st in segments:
        r = p.add_run()
        r.text = text
        style(r, size=st.get("size", size), bold=st.get("bold", False),
              color=st.get("color", TEXT), italic=st.get("italic", False))
    if bullet:
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(Inches(0.38)))
        pPr.set("indent", str(-Inches(0.38)))
        bf = etree.SubElement(pPr, qn("a:buFont"))
        bf.set("typeface", FONT)
        bc = etree.SubElement(pPr, qn("a:buChar"))
        bc.set("char", "–")
    return p


def card(x, y, w, h, name, fill=TINT, line=None):
    c = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    c.name = name
    c.adjustments[0] = 0.08
    c.fill.solid()
    c.fill.fore_color.rgb = fill
    if line is None:
        c.line.fill.background()
    else:
        c.line.color.rgb = line
        c.line.width = Pt(2)
    c.shadow.inherit = False
    tf = c.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.15)
    tf.margin_top = tf.margin_bottom = Inches(0.08)
    return c, tf


def arrow(x1, y1, x2, y2, name):
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    ln.name = name
    ln.line.color.rgb = ORANGE
    ln.line.width = Pt(4)
    tail = etree.SubElement(ln.line._get_or_add_ln(), qn("a:tailEnd"))
    tail.set("type", "triangle")
    tail.set("w", "med")
    tail.set("len", "med")


# ---------- bố cục (theo poster mẫu của Khoa: dải băng tiêu đề, không đánh số) ----------
XL, XR, CW = 2.05, 11.95, 9.35
FULL = XR + CW - XL
ROSE = RGBColor(0x8E, 0x5C, 0x60)      # nâu hồng của dải băng tiêu đề trong poster mẫu
B = {"bold": True}


def ribbon(x, y, text):
    """Tiêu đề mục: dải băng hình mũi nhọn màu nâu hồng, chữ trắng in đậm."""
    w = 0.95 + 0.205 * len(text)
    sh = s.shapes.add_shape(MSO_SHAPE.PENTAGON, Inches(x), Inches(y), Inches(w), Inches(0.62))
    sh.name = f"Tiêu đề {text}"
    sh.fill.solid()
    sh.fill.fore_color.rgb = ROSE
    sh.line.fill.background()
    sh.shadow.inherit = False
    tf = sh.text_frame
    tf.margin_left, tf.margin_right, tf.margin_top, tf.margin_bottom = Inches(0.3), Inches(0.2), 0, 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    add_par(tf, [(text, {"bold": True, "color": WHITE, "size": 24})], align=PP_ALIGN.LEFT, after=0, first=True)
    return y + 0.8


def bullets(x, y, w, h, name, items, size=BODY):
    """items: phần tử là list[(text, style)]; phần tử dạng ("+", [...]) là ý con."""
    _, tf = textbox(x, y, w, h, name)
    for i, it in enumerate(items):
        con = isinstance(it, tuple) and it[0] == "+"
        p = add_par(tf, it[1] if con else it, size=size - (2 if con else 0), bullet=True,
                    align=PP_ALIGN.LEFT, after=5, first=(i == 0))
        if con:
            pPr = p._p.get_or_add_pPr()
            pPr.set("marL", str(Inches(0.8)))
            pPr.find(qn("a:buChar")).set("char", "+")
    return tf


def anh(path, x, y, w, name, khung=None):
    im = Image.open(path)
    h = w * im.height / im.width
    if khung is not None:
        fr = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x - 0.12), Inches(y - 0.12),
                                Inches(w + 0.24), Inches(h + 0.24))
        fr.name = f"Khung {name}"
        fr.fill.solid()
        fr.fill.fore_color.rgb = WHITE
        fr.line.color.rgb = khung
        fr.line.width = Pt(3)
        fr.shadow.inherit = False
    pic = s.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))
    pic.name = name
    return y + h


def chu_thich(x, y, w, text):
    _, tf = textbox(x, y, w, 0.45, f"Chú thích {text[:20]}")
    add_par(tf, [(text, {"size": 15, "color": MUTED, "italic": True})], align=PP_ALIGN.CENTER, first=True)


# ===== CỘT TRÁI: TÓM TẮT, DỮ LIỆU, KẾT QUẢ =====
y = ribbon(XL, 8.6, "TÓM TẮT")
_, tf = textbox(XL, y, CW, 3.4, "Tóm tắt")
add_par(tf, "Quản lý tồn kho cho mạng lưới nhiều kho, nhiều mặt hàng phải cân bằng chi phí lưu kho, thiếu hàng, "
            "đặt hàng và tràn kho, trong khi nhu cầu thưa, khó đoán và các mặt hàng dùng chung sức chứa kho. "
            "Đề tài xây dựng mô hình PS-IPPO: 300 tác tử học tăng cường dùng chung tham số, mỗi tác tử quyết "
            "định lượng đặt cho một cặp kho – mặt hàng. Trên dữ liệu M5 Walmart, PS-IPPO đạt tỷ lệ đáp ứng "
            "90,9%, rẻ hơn (s,S) và EOQ; khi kết hợp dự báo nhu cầu, mô hình rẻ hơn mọi chính sách cổ điển.",
        size=20, first=True)

y = ribbon(XL, 12.25, "DỮ LIỆU")
bullets(XL, y, CW, 3.3, "Dữ liệu", [
    [("M5 Walmart: ", B), ("doanh số theo ngày, 29/01/2011 – 22/05/2016 (1.941 ngày)", {})],
    ("+", [("10 cửa hàng × 30 mặt hàng = 300 cặp kho – mặt hàng", {})]),
    ("+", [("57,4% số ngày không bán; giá bán và lịch sự kiện, SNAP", {})]),
    [("Chia theo thời gian: ", B), ("huấn luyện 1.050 ngày · xác thực 400 · kiểm thử 491", {})],
    [("Mô phỏng: ", B), ("sức chứa mỗi kho = 5 ngày nhu cầu, giao hàng sau 1–3 ngày", {})]], size=20)

y = ribbon(XL, 16.3, "KẾT QUẢ THỰC NGHIỆM")
_, tf = textbox(XL, y, CW, 0.45, "Tên biểu đồ")
add_par(tf, [("Tổng chi phí vận hành trên 491 ngày kiểm thử (triệu đơn vị)", {"bold": True, "size": 18})],
        align=PP_ALIGN.LEFT, first=True)
data = [("EOQ", 2500198), ("(s,S)", 2091665), ("EOQ nhóm", 2074745), ("Newsvendor", 2062190),
        ("PS-IPPO", 2055442), ("(s,S) nhóm", 2048883), ("(s,S) biết sức chứa", 2039654),
        ("Newsvendor nhóm", 2021883), ("PS-IPPO + dự báo", 1897517), ("Newsvendor dự báo", 1892876)]
CH_H = 6.0
cd = CategoryChartData()
cd.categories = [d[0] for d in data]
cd.add_series("Tổng chi phí", [round(v / 1e6, 3) for _, v in data])
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(XL), Inches(y + 0.45), Inches(CW), Inches(CH_H), cd)
gf.name = "Biểu đồ tổng chi phí"
ch_ = gf.chart
ch_.has_legend = False
ch_.has_title = False
ch_.font.name, ch_.font.size = FONT, Pt(16)
plot = ch_.plots[0]
plot.gap_width = 40
ser = plot.series[0]
for i, (name, v) in enumerate(data):
    la_ippo = name.startswith("PS-IPPO")
    pt = ser.points[i]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = ORANGE if la_ippo else GREY_BAR
    dtf = pt.data_label.text_frame
    pt.data_label.position = XL_LABEL_POSITION.OUTSIDE_END
    dtf.text = f"{v / 1e6:.2f}".replace(".", ",")
    style(dtf.paragraphs[0].runs[0], size=16, bold=la_ippo, color=ORANGE if la_ippo else TEXT)
va = ch_.value_axis
va.minimum_scale, va.maximum_scale = 1.6, 2.6
va.visible = False
va.has_major_gridlines = False
ca = ch_.category_axis
ca.tick_labels.font.size = Pt(16)
ca.tick_labels.font.name = FONT
ca.format.line.color.rgb = RGBColor(0xBF, 0xBF, 0xBF)
_, tf = textbox(XL, y + 0.5 + CH_H, CW, 0.75, "Ghi chú biểu đồ")
add_par(tf, [("Trung bình 3 hạt giống, trục bắt đầu từ 1,6 triệu; “nhóm”: tinh chỉnh theo nhóm mặt hàng. "
              "Tỷ lệ đáp ứng: PS-IPPO 90,9%, kết hợp dự báo 86,7%, chính sách cổ điển 85,6–89,5%.",
              {"size": 14, "color": MUTED, "italic": True})], align=PP_ALIGN.LEFT, first=True)

# ===== CỘT PHẢI: MÔ HÌNH ĐỀ XUẤT, KẾT LUẬN =====
y = ribbon(XR, 8.6, "MÔ HÌNH ĐỀ XUẤT: PS-IPPO")
y = anh(f"{ROOT}/media/poster_mo_hinh.png", XR + 0.15, y + 0.15, CW - 0.3, "Sơ đồ PS-IPPO", khung=ROSE)
bullets(XR, y + 0.35, CW, 2.9, "Điểm thiết kế", [
    [("IPPO: ", B), ("mỗi cặp kho – mặt hàng là một tác tử PPO, tự quyết từ quan sát 44 chiều của mình.", {})],
    [("Chia sẻ tham số (PS): ", B),
     ("hai mạng Actor và Critic tách rời (2 lớp × 128), một bộ trọng số cho cả 300 tác tử.", {})],
    [("Phần thưởng cục bộ ", B), ("chuẩn hóa theo quy mô từng cặp; mục tiêu đáp ứng 85% là phạt mềm.", {})]],
    size=20)

y = ribbon(XR, 16.5, "KẾT LUẬN")
stats = [("90,9%", "tỷ lệ đáp ứng, mục tiêu 85%", PURPLE),
         ("−1,7%", "chi phí so với (s,S), cả 3 hạt giống", PURPLE),
         ("−17,8%", "chi phí so với EOQ, cả 3 hạt giống", PURPLE),
         ("−7,7%", "chi phí khi kết hợp dự báo (đáp ứng 86,7%)", ORANGE)]
sw = (CW - 0.25) / 2
for i, (big, lab, col) in enumerate(stats):
    c, ctf = card(XR + (i % 2) * (sw + 0.25), y + (i // 2) * 1.42, sw, 1.3, f"Chỉ số {i}")
    add_par(ctf, [(big, {"bold": True, "color": col, "size": 32})], align=PP_ALIGN.CENTER, after=0, first=True)
    add_par(ctf, [(lab, {"color": TEXT, "size": 15})], align=PP_ALIGN.CENTER, after=0)
y += 2 * 1.42 + 0.12
bullets(XR, y, CW, 4.4, "Kết luận", [
    [("Phục vụ tốt hơn, chi phí tương đương: ", B),
     ("tỷ lệ đáp ứng cao hơn chính sách cổ điển 1,4–5,3 điểm; rẻ hơn (s,S), EOQ ở cả 3 hạt giống, "
      "chỉ đắt hơn Newsvendor theo nhóm 1,7%.", {})],
    [("Ở cùng mức phục vụ, ", B), ("chính sách cổ điển phải tốn thêm 7–16% chi phí, EOQ trên 22%.", {})],
    [("Dự báo nhu cầu là thông tin đáng giá nhất: ", B),
     ("kết hợp dự báo rẻ hơn mọi chính sách cổ điển 6,2–24,1%, ngang Newsvendor dùng cùng dự báo.", {})],
    [("Bền khi các kho dùng chung nguồn cung: ", B),
     ("không huấn luyện lại vẫn rẻ hơn mọi chính sách cổ điển 1,1–22,4%, nguồn cung càng chặt càng rõ.", {})],
    [("Thiết kế phần thưởng quyết định việc học: ", B),
     ("dùng phần thưởng chung toàn mạng hoặc bỏ chuẩn hóa theo cặp làm chi phí tăng gấp ba đến gấp năm.", {})]],
    size=19)

# ===== HÀNG DƯỚI CẢ BỀ NGANG: MÔ PHỎNG KẾT QUẢ (ảnh chụp ứng dụng) =====
Y_MP = 24.55
y = ribbon(XL, Y_MP, "MÔ PHỎNG KẾT QUẢ")
yb = anh(f"{ROOT}/media/poster_phieu.png", XL, y + 0.05, CW, "Ảnh phiếu đề nghị", khung=GREY_BAR)
chu_thich(XL, yb + 0.12, CW, "Phiếu đề nghị đặt hàng do mô hình lập cho kho K01")
yb = anh(f"{ROOT}/media/poster_chi_tiet.png", XR, y + 0.05, CW, "Ảnh diễn biến mặt hàng", khung=GREY_BAR)
chu_thich(XR, yb + 0.12, CW, "Tồn kho, nhu cầu và các lần đặt hàng của một mặt hàng")

prs.save(OUT)
print("saved", OUT)
