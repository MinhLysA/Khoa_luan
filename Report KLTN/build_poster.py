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
    add_par(tf, [(text, {"size": 16, "color": MUTED, "italic": True})], align=PP_ALIGN.CENTER, first=True)


# ===== HÀNG 1: TÓM TẮT | DỮ LIỆU =====
y = ribbon(XL, 8.6, "TÓM TẮT")
_, tf = textbox(XL, y, CW, 2.4, "Tóm tắt")
add_par(tf, "Quản lý tồn kho cho mạng lưới nhiều kho, nhiều mặt hàng phải cân bằng chi phí lưu kho, thiếu hàng, "
            "đặt hàng và tràn kho, trong khi nhu cầu thưa, khó đoán và các mặt hàng dùng chung sức chứa kho. "
            "Đề tài xây dựng mô hình PS-IPPO: 300 tác tử học tăng cường dùng chung tham số, mỗi tác tử quyết "
            "định lượng đặt cho một cặp kho – mặt hàng. Trên dữ liệu M5 Walmart, PS-IPPO đạt tỷ lệ đáp ứng "
            "90,9% và rẻ hơn mọi chính sách cổ điển khi so ở cùng mức phục vụ.",
        size=19, first=True)
y = ribbon(XR, 8.6, "DỮ LIỆU")
bullets(XR, y, CW, 2.4, "Dữ liệu", [
    [("M5 Walmart: ", B), ("doanh số theo ngày, 29/01/2011 – 22/05/2016 (1.941 ngày)", {})],
    ("+", [("10 cửa hàng × 30 mặt hàng = 300 cặp; 57,4% số ngày không bán", {})]),
    [("Chia theo thời gian: ", B), ("huấn luyện 1.050 ngày · xác thực 400 · kiểm thử 491", {})],
    [("Mô phỏng: ", B), ("sức chứa mỗi kho = 5 ngày nhu cầu, giao hàng sau 1–3 ngày", {})]], size=19)

# ===== HÀNG 2: MÔ HÌNH ĐỀ XUẤT (sơ đồ kiến trúc đa tác tử, trải hết bề ngang) =====
y = ribbon(XL, 11.5, "MÔ HÌNH ĐỀ XUẤT: PS-IPPO")
WM = 17.6
y = anh(f"{ROOT}/media/kien_truc_da_tac_tu.png", XL + (FULL - WM) / 2, y + 0.12, WM, "Sơ đồ kiến trúc PS-IPPO", khung=ROSE)
_, tf = textbox(XL, y + 0.25, FULL, 0.4, "Ghi chú mô hình")
add_par(tf, [("300 tác tử = 10 kho × 30 mặt hàng  ·  mỗi tác tử quan sát 44 chiều, chọn 1 trong 6 mức đặt  ·  "
              "Actor và Critic 2 lớp × 128 dùng chung  ·  phần thưởng chuẩn hóa theo từng cặp", {"size": 17, "color": TEXT})],
        align=PP_ALIGN.CENTER, first=True)

# ===== HÀNG 3: KẾT QUẢ THỰC NGHIỆM | KẾT LUẬN =====
Y3 = 19.1
y = ribbon(XL, Y3, "KẾT QUẢ THỰC NGHIỆM")
yb = anh(f"{ROOT}/media/poster_ket_qua.png", XL, y + 0.05, CW, "Biểu đồ kết quả")   # vẽ bằng poster_bieu_do.py
_, tf = textbox(XL, yb + 0.05, CW, 0.3, "Ghi chú biểu đồ")
add_par(tf, [("Trung bình 3 hạt giống, 491 ngày kiểm thử · thanh đậm/nhạt: mức rẻ hơn thấp nhất/cao nhất",
              {"size": 14, "color": MUTED, "italic": True})], align=PP_ALIGN.LEFT, first=True)

y = ribbon(XR, Y3, "KẾT LUẬN")
bullets(XR, y, CW, 4.4, "Kết luận", [
    [("Phục vụ tốt hơn, chi phí tương đương: ", B),
     ("tỷ lệ đáp ứng 90,9%, cao hơn chính sách cổ điển 1,4–5,3 điểm; tổng chi phí rẻ hơn (s,S) 1,7% và EOQ 17,8%.", {})],
    [("Ở cùng mức phục vụ, ", B), ("PS-IPPO rẻ hơn mọi chính sách cổ điển 7,3–16,3%, rẻ hơn EOQ trên 22%.", {})],
    [("Chống sốc cầu tốt hơn: ", B),
     ("khi nhu cầu tăng 50% trong 28 ngày, chi phí mỗi ngày ít hơn Newsvendor 13,2%, ít hơn (s,S) 16,8%; "
      "sau sốc tỷ lệ đáp ứng lên 93,3%.", {})],
    [("Kết hợp dự báo nhu cầu: ", B),
     ("rẻ hơn mọi chính sách cổ điển 6,2–24,1%, ngang Newsvendor dùng cùng dự báo.", {})],
    [("Bền khi các kho dùng chung nguồn cung: ", B),
     ("không huấn luyện lại vẫn rẻ hơn mọi chính sách cổ điển 1,1–22,4%.", {})],
    [("Thiết kế phần thưởng quyết định việc học: ", B),
     ("dùng phần thưởng chung toàn mạng hoặc bỏ chuẩn hóa theo cặp làm chi phí tăng gấp ba đến gấp năm.", {})]],
    size=18)

# ===== HÀNG 4: MÔ PHỎNG SỐC CẦU TRÊN ỨNG DỤNG (hai biểu đồ chụp từ demo) =====
y = ribbon(XL, 24.4, "MÔ PHỎNG SỐC CẦU TRÊN ỨNG DỤNG")
H4 = 4.0
w1 = H4 * 1340 / 625
w2 = FULL - w1 - 0.5
yb = anh(f"{ROOT}/media/poster_soc_dap_ung.png", XL, y + 0.1, w1, "Ảnh tỷ lệ đáp ứng khi sốc", khung=GREY_BAR)
chu_thich(XL, yb + 0.2, w1, "Tỷ lệ đáp ứng theo ngày, nhu cầu tăng ×1,5 trong 28 ngày (vùng cam)")
yb = anh(f"{ROOT}/media/poster_soc_dat_hang.png", XL + w1 + 0.5, y + 0.1, w2, "Ảnh đặt hàng khi sốc", khung=GREY_BAR)
chu_thich(XL + w1 + 0.5, yb + 0.2, w2, "Lượng đặt hàng của IPPO và Newsvendor bám theo nhu cầu toàn mạng")

prs.save(OUT)
print("saved", OUT)
