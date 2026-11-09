"""Sinh file draw.io bản sinh động: biểu tượng màu, khung đánh số, mũi tên đánh số theo luồng."""
import sys
import xml.etree.ElementTree as ET
from icons import uri

NAVY, CAM, XANH, DO, LUC, TIM = "#1F3A5F", "#E07B24", "#2A9D8F", "#C0392B", "#2E7D32", "#6A4C93"
BLUE = "#1E6FC2"
F = "fontFamily=Segoe UI;"
EDGE = (F + "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeWidth=2.5;endArrow=block;endFill=1;endSize=7;"
        "fontSize=12;labelBackgroundColor=#FFFFFF;")


def num(n, color=BLUE):
    return (f"<span style='background:{color};color:#fff;border-radius:11px;padding:1px 7px;"
            f"font-weight:bold;font-size:12px'>{n}</span>")


def sub(s, color="#5A6575"):
    return f"<span style='font-size:11.5px;color:{color}'>{s}</span>"


class Page:
    def __init__(self, name, w, h):
        self.name, self.w, self.h, self.cells, self.n, self.geo = name, w, h, [], 0, {}

    def v(self, x, y, w, h, label, style):
        self.n += 1
        cid = f"c{self.n}"
        c = ET.Element("mxCell", id=cid, value=label, style=style, vertex="1", parent="1")
        ET.SubElement(c, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"})
        self.cells.append(c)
        self.geo[cid] = (x, y, w, h)
        return cid

    # ---- hình cơ bản
    def text(self, x, y, w, h, label, size=13, color="#1A1A1A", extra=""):
        return self.v(x, y, w, h, label, F + "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                      f"align=left;verticalAlign=middle;fontSize={size};fontColor={color};{extra}")

    def rect(self, x, y, w, h, fill="#FFFFFF", stroke="#B8C7D9", extra="", label=""):
        return self.v(x, y, w, h, label, F + f"rounded=1;arcSize=6;html=1;whiteSpace=wrap;fillColor={fill};"
                      f"strokeColor={stroke};strokeWidth=1.5;fontSize=13;{extra}")

    def icon(self, x, y, name, s=40):
        return self.v(x, y, s, s, "", f"shape=image;html=1;image={uri(name)};imageAspect=1;aspect=fixed;")

    def vcard(self, x, y, w, h, icon, title, desc="", fill="#FFFFFF", stroke="#B8C7D9", s=44, extra=""):
        """Thẻ dọc: biểu tượng trên, chữ dưới."""
        lab = f"<b style='font-size:14px'>{title}</b>" + (f"<br>{sub(desc)}" if desc else "")
        return self.v(x, y, w, h, lab, F + f"shape=label;html=1;whiteSpace=wrap;rounded=1;arcSize=8;image={uri(icon)};"
                      f"imageWidth={s};imageHeight={s};imageAlign=center;imageVerticalAlign=top;spacingTop={s + 8};"
                      f"spacing=8;verticalAlign=top;align=center;fillColor={fill};strokeColor={stroke};"
                      f"strokeWidth=1.5;fontSize=13;fontColor=#1A1A1A;{extra}")

    def hcard(self, x, y, w, h, icon, title, desc="", fill="#FFFFFF", stroke="#B8C7D9", s=38, extra="", color="#1A1A1A"):
        """Thẻ ngang: biểu tượng trái, chữ phải."""
        lab = f"<b style='font-size:14px'>{title}</b>" + (f"<br>{sub(desc, '#D6E2F0' if color == '#FFFFFF' else '#5A6575')}" if desc else "")
        return self.v(x, y, w, h, lab, F + f"shape=label;html=1;whiteSpace=wrap;rounded=1;arcSize=10;image={uri(icon)};"
                      f"imageWidth={s};imageHeight={s};imageAlign=left;imageVerticalAlign=middle;spacingLeft={s + 12};"
                      f"spacing=8;verticalAlign=middle;align=left;fillColor={fill};strokeColor={stroke};"
                      f"strokeWidth=1.5;fontSize=13;fontColor={color};{extra}")

    def section(self, x, y, w, h, n, title, color=BLUE, fill="#FFFFFF"):
        """Khung nét đứt có số thứ tự, như ảnh mẫu."""
        self.rect(x, y, w, h, fill=fill, stroke="#8DB3DA", extra="dashed=1;dashPattern=6 4;arcSize=3;")
        self.v(x + 10, y + 10, 26, 26, f"<b>{n}</b>", F + f"ellipse;html=1;fillColor={color};strokeColor=none;"
               "fontColor=#FFFFFF;fontSize=13;")
        self.text(x + 42, y + 8, w - 50, 30, f"<b>{title}</b>", size=14, color=NAVY)

    def layer(self, x, y, w, h, title, desc, fill, stroke=NAVY):
        self.rect(x, y, w, h, fill=fill, stroke=stroke, extra="strokeWidth=2.5;arcSize=3;")
        self.text(x + 20, y + 8, w - 40, 32, f"<b>{title}</b>", size=19, color="#111111")
        self.text(x + 20, y + 38, w - 40, 22, desc, size=13, color="#333333")

    def up(self, cx, y):
        return self.v(cx - 16, y, 32, 24, "", "triangle;direction=north;html=1;fillColor=#3D4F66;strokeColor=none;")

    # ---- cạnh nối vào hình, giữ liên kết khi kéo thả
    def link(self, src, tgt, p1, p2, label="", color=NAVY, extra="", pts=None):
        def frac(cid, p):
            x, y, w, h = self.geo[cid]
            return round((p[0] - x) / w, 4), round((p[1] - y) / h, 4)
        ex, ey = frac(src, p1)
        nx, ny = frac(tgt, p2)
        self.n += 1
        cid = f"c{self.n}"
        if p1[1] == p2[1] and "</span> " in label:  # nhãn có chữ trên cạnh ngang: đặt lên trên đường
            extra = "verticalAlign=bottom;labelBackgroundColor=none;spacingBottom=3;" + extra
        st = EDGE + f"strokeColor={color};fontColor={color};exitX={ex};exitY={ey};entryX={nx};entryY={ny};exitDx=0;exitDy=0;entryDx=0;entryDy=0;{extra}"
        c = ET.Element("mxCell", id=cid, value=label, style=st, edge="1", parent="1", source=src, target=tgt)
        g = ET.SubElement(c, "mxGeometry", relative="1", **{"as": "geometry"})
        if pts:
            arr = ET.SubElement(g, "Array", **{"as": "points"})
            for px, py in pts:
                ET.SubElement(arr, "mxPoint", x=str(px), y=str(py))
        self.cells.append(c)
        return cid

    def xml(self):
        d = ET.Element("diagram", name=self.name, id=f"p{abs(hash(self.name)) % 10**8}")
        m = ET.SubElement(d, "mxGraphModel", dx="1200", dy="800", grid="1", gridSize="10", guides="1", tooltips="1",
                          connect="1", arrows="1", fold="1", page="1", pageScale="1", pageWidth=str(self.w),
                          pageHeight=str(self.h), math="0", shadow="0")
        root = ET.SubElement(m, "root")
        ET.SubElement(root, "mxCell", id="0")
        ET.SubElement(root, "mxCell", id="1", parent="0")
        root.extend(self.cells)
        return d


def title(p, t, s=None):
    p.text(20, 10, p.w - 40, 50, f"<b>{t}</b>", size=30, color="#111111", extra="align=center;")
    if s:
        p.text(20, 54, p.w - 40, 24, s, size=14, color="#555555", extra="align=center;")


# ================================================================ 1. Kiến trúc phân lớp
def trang1():
    p = Page("1. Kiến trúc phân lớp", 1400, 1000)

    def band(y, h, t, fill, col, fs=15):
        p.rect(0, y, 1370, h, fill=fill, stroke=col, extra="strokeWidth=2.5;arcSize=2;")
        p.v(0, y, 50, h, f"<b>{t}</b>", F + f"rounded=1;arcSize=6;html=1;horizontal=0;fillColor={col};strokeColor={col};"
            f"fontColor=#FFFFFF;fontSize={fs};whiteSpace=wrap;")

    title(p, "KIẾN TRÚC HỆ THỐNG PS-IPPO",
          "Ba lớp nối với nhau bằng các luồng dữ liệu: xanh lá là luồng huấn luyện, cam là luồng sử dụng")
    G, U = LUC, CAM  # màu luồng huấn luyện / sử dụng

    # ---- Người dùng
    mgr = p.hcard(560, 92, 280, 66, "user", "Người quản lý", "xem kết quả, duyệt phiếu", fill="#EAF3FD",
                  stroke=BLUE, s=44, extra="strokeWidth=2;")
    pmh = p.hcard(1060, 92, 280, 66, "receipt", "Phòng mua hàng", "nhận phiếu · ngoài phạm vi đề tài", stroke=NAVY, s=40,
                  extra="strokeWidth=2;")
    p.link(mgr, pmh, (840, 125), (1060, 125), "tải CSV các dòng đã duyệt", NAVY)

    # ---- Lớp ứng dụng
    band(195, 190, "① LỚP ỨNG DỤNG", "#FBF1E4", "#B5651D")
    st = p.rect(60, 210, 1280, 160, fill="#FFFFFF", stroke=NAVY, extra="strokeWidth=2;arcSize=4;")
    p.icon(75, 216, "monitor", 24)
    p.text(106, 214, 1200, 26, "<b>Ứng dụng Streamlit</b> " + sub("· 5 màn hình, chỉ nạp mô hình đã huấn luyện và chạy tất định trên miền kiểm thử"),
           size=14, color=NAVY)
    screens = [("monitor", "Kết quả", "thẻ cho 5 câu hỏi nghiên cứu"),
               ("receipt", "Phiếu mua sắm", "duyệt phiếu, bỏ dòng, tải CSV"),
               ("chart", "Chi tiết mặt hàng", "tồn kho, nhu cầu, lượng đặt"),
               ("coins", "So sánh chi tiết", "cơ cấu chi phí từng chính sách"),
               ("forecast", "Kịch bản tăng cầu", "nhân nhu cầu, chạy lại chính sách")]
    w = (1250 - 4 * 10) / 5
    for k, (ic, t, d) in enumerate(screens):
        p.vcard(75 + k * (w + 10), 252, w, 104, ic, t, d, fill="#F6F8FB", stroke="#C9D3DF", s=30,
                extra="spacingTop=40;spacing=6;fontSize=12.5;")
    p.link(mgr, st, (700, 158), (700, 210), "xem, duyệt phiếu", NAVY)

    # ---- Lớp tác tử và mô phỏng
    band(425, 300, "② LỚP TÁC TỬ VÀ MÔ PHỎNG", "#E5F3E8", "#2E7D32")
    env = p.vcard(60, 485, 320, 215, "warehouse", "Môi trường mô phỏng",
                  "10 kho × 30 mặt hàng, mỗi bước 1 ngày<br>các mặt hàng dùng chung sức chứa kho<br>giao tiếp reset() / step() theo Gymnasium",
                  fill="#FFF4F2", stroke=DO, s=60, extra="strokeWidth=2.5;")
    ag = p.rect(500, 475, 380, 235, fill="#FFFFFF", stroke=TIM, extra="strokeWidth=2.5;arcSize=5;")
    p.text(515, 480, 350, 26, "<b>300 tác tử PS-IPPO</b> " + sub("· mỗi tác tử một cặp (kho, mặt hàng)"), size=14, color=TIM)
    for x, t in [(530, "Tác tử 1"), (610, "Tác tử 2"), (775, "Tác tử 300")]:
        p.icon(x, 512, "robot", 44)
        p.text(x - 18, 557, 80, 18, f"<b>{t}</b>", size=11.5, extra="align=center;")
    p.text(680, 515, 60, 40, "<b>⋯</b>", size=24, color=NAVY, extra="align=center;")
    pol = p.hcard(520, 592, 340, 100, "share", "Chính sách dùng chung",
                  "Actor π<sub>θ</sub> chọn mức đặt · Critic V<sub>φ</sub><br>một bộ trọng số cho cả 300 tác tử",
                  fill=NAVY, stroke=NAVY, color="#FFFFFF", s=50)
    agg = p.hcard(1000, 480, 340, 95, "chart", "Lớp tổng hợp báo cáo",
                  "cộng kết quả theo kho, ngành hàng, toàn mạng<br>chỉ đọc, không gửi ngược cho tác tử", stroke=U, s=44,
                  extra="strokeWidth=2;")
    trn = p.hcard(1000, 610, 340, 95, "gear", "Huấn luyện PPO",
                  "GAE riêng từng cặp, PPO-clip<br>10.000 tập × 365 ngày", fill="#EAF6EC", stroke=G, s=44,
                  extra="strokeWidth=2;")
    p.link(env, ag, (380, 530), (500, 530), "quan sát o<br>phần thưởng r", NAVY)
    p.link(ag, env, (500, 660), (380, 660), "hành động a<br>(1 trong 6 mức)", CAM)
    p.link(ag, agg, (880, 528), (1000, 528), "quyết định,<br>chi phí", U)
    p.link(ag, trn, (880, 640), (1000, 640), "mẫu (o, a, r, V)", G)
    p.link(trn, ag, (1000, 685), (880, 685), "θ, φ mới", G, "dashed=1;")

    # ---- Luồng từ lớp tác tử lên lớp ứng dụng
    p.link(agg, st, (1170, 480), (1170, 370), "chi phí, tỷ lệ đáp ứng, phiếu", U)

    # ---- Lớp dữ liệu và hạ tầng
    band(765, 150, "③ DỮ LIỆU, HẠ TẦNG", "#E4EEF9", "#1F3A5F", 13)
    m5 = p.hcard(60, 795, 250, 80, "db", "Dữ liệu M5 Walmart", "doanh số, giá bán, lịch<br>1.941 ngày", stroke=NAVY, s=44,
                 extra="strokeWidth=2;")
    lgb = p.hcard(330, 795, 250, 80, "forecast", "Dự báo LightGBM", "nhu cầu ngày t, t+1, t+2<br>dùng ở biến thể dự báo",
                  stroke=NAVY, s=44, extra="strokeWidth=2;dashed=1;")
    pt = p.hcard(600, 795, 230, 80, "chip", "PyTorch", "cài đặt mạng Actor, Critic", stroke=NAVY, s=44, extra="strokeWidth=2;")
    sto = p.hcard(850, 795, 490, 80, "save", "Lưu trữ",
                  "checkpoint .pth của PS-IPPO, tham số đã tinh chỉnh<br>của chính sách cổ điển, kết quả đánh giá",
                  stroke=NAVY, s=44, extra="strokeWidth=2;")
    p.link(m5, env, (180, 795), (180, 700), "nhu cầu, giá, lịch", NAVY)
    p.link(lgb, env, (350, 795), (350, 700), "dự báo", NAVY, "dashed=1;")
    p.link(pt, ag, (715, 795), (715, 710), "mạng nơ-ron", NAVY)
    p.link(trn, sto, (1170, 705), (1170, 795), "lưu checkpoint tốt nhất", G)
    p.link(sto, ag, (900, 795), (840, 710), "nạp mô hình đã chọn", U, pts=[(900, 745), (840, 745)])

    # ---- Chú giải
    p.text(0, 925, 1370, 30,
           f"<span style='color:{G}'><b>━ luồng huấn luyện</b></span> &nbsp;&nbsp; "
           f"<span style='color:{U}'><b>━ luồng sử dụng</b></span> &nbsp;&nbsp; "
           f"<span style='color:{NAVY}'><b>━ dữ liệu và tương tác</b></span> &nbsp;&nbsp; "
           f"<span style='color:#555'>- - - nét đứt: tùy chọn hoặc cập nhật</span>",
           size=13, extra="align=center;")
    return p


# ================================================================ 2. Luồng hoạt động
def trang2():
    p = Page("2. Luồng hoạt động", 1560, 980)
    title(p, "LUỒNG HOẠT ĐỘNG CỦA MÔ HÌNH PS-IPPO", "Đọc theo số trên mũi tên: 1 → 6 là vòng huấn luyện, 7 → 10 là giai đoạn sử dụng")

    # Vùng A: huấn luyện
    p.rect(15, 90, 1530, 510, fill="#F4F9FF", stroke="#1E6FC2", extra="dashed=1;strokeWidth=2;arcSize=2;")
    p.text(30, 96, 900, 30, "<b>A · GIAI ĐOẠN HUẤN LUYỆN</b>  " + sub("miền huấn luyện 1.050 ngày · 10.000 tập × 365 ngày"),
           size=16, color=BLUE)

    p.section(35, 135, 220, 445, 1, "Dữ liệu đầu vào")
    s1 = [p.hcard(50, 185 + k * 94, 190, 76, ic, a, b) for k, (ic, a, b) in enumerate(
        [("db", "Nhu cầu M5", "1.941 ngày, 300 chuỗi"), ("tag", "Giá bán", "tín hiệu giảm giá"),
         ("calendar", "Lịch", "thứ, sự kiện, SNAP"), ("forecast", "Dự báo LightGBM", "chỉ ở biến thể dự báo")])]

    p.section(365, 135, 320, 445, 2, "Môi trường mô phỏng")
    env = p.hcard(385, 182, 280, 84, "warehouse", "10 kho × 30 mặt hàng", "mỗi bước = 1 ngày", fill="#FFF4F2", stroke=DO,
                  s=50, extra="strokeWidth=2;")
    steps = ["Đổi a thành lượng đặt q", "Nhận hàng đến hạn", "Áp sức chứa, từ chối phần tràn",
             "Ghi đơn mới, về sau 1–3 ngày", "Bán hàng theo nhu cầu M5", "Tính chi phí, tỷ lệ đáp ứng, r"]
    for k, t in enumerate(steps):
        p.rect(385, 282 + k * 47, 280, 39, fill="#F3F5F8", stroke="#C9D3DF",
               label=f"<span style='color:{DO};font-weight:bold'>{k + 1}.</span>&nbsp; {t}",
               extra="align=left;spacingLeft=10;fontSize=12.5;")
    envbox = p.v(365, 135, 320, 445, "", "html=1;fillColor=none;strokeColor=none;")

    p.section(845, 135, 320, 445, 3, "300 tác tử PS-IPPO")
    pol = p.hcard(865, 182, 280, 84, "network", "Chính sách dùng chung", "Actor π<sub>θ</sub> · Critic V<sub>φ</sub>",
                  fill=NAVY, stroke=NAVY, color="#FFFFFF", s=50)
    for x, t in [(875, "Tác tử 1"), (960, "Tác tử 2"), (1085, "Tác tử 300")]:
        r = p.icon(x, 300, "robot", 50)
        p.text(x - 15, 352, 80, 20, f"<b>{t}</b>", size=12, extra="align=center;")
        p.link(pol, r, (x + 25, 266), (x + 25, 300), "", "#7A8CA5", "endArrow=none;strokeWidth=1.5;dashed=1;")
    p.text(1025, 305, 50, 40, "<b>⋯</b>", size=24, color=NAVY, extra="align=center;")
    p.rect(865, 395, 280, 165, fill="#FFFFFF", stroke="#C9D3DF", extra="align=left;spacingLeft=14;fontSize=13;",
           label="<b>Mỗi tác tử = một cặp (kho, mặt hàng)</b><br><br>▸ nhận quan sát <b>44 chiều</b><br>"
                 "▸ chọn <b>1 trong 6 mức</b> đặt hàng<br>▸ nhận phần thưởng của riêng cặp<br>▸ cùng một bộ trọng số θ, φ")
    agbox = p.v(845, 135, 320, 445, "", "html=1;fillColor=none;strokeColor=none;")

    p.section(1285, 135, 245, 195, 4, "Bộ đệm kinh nghiệm")
    buf = p.hcard(1300, 185, 215, 125, "stack", "1024 bước<br>× 300 tác tử", "lưu (o, a, r, V)", fill="#FFF3E6", stroke=CAM,
                  s=48, extra="strokeWidth=2;")
    p.section(1285, 365, 245, 215, 5, "Cập nhật PPO")
    upd = p.hcard(1300, 410, 215, 70, "gear", "Cập nhật θ, φ", "lặp tối đa 4 lượt", fill="#EAF6EC", stroke=LUC, s=44,
                  extra="strokeWidth=2;")
    p.rect(1300, 490, 215, 75, fill="#FFFFFF", stroke="#C9D3DF", extra="align=left;spacingLeft=10;fontSize=12;",
           label="▸ GAE riêng từng cặp<br>▸ chuẩn hóa lợi thế<br>▸ PPO-clip, ε = 0,2")

    # Luồng huấn luyện 1 → 6
    p.link(s1[0], env, (240, 224), (385, 224), num(1, NAVY) + " dữ liệu", NAVY)
    p.link(envbox, agbox, (685, 224), (845, 224), num(2, NAVY) + " quan sát o<br>phần thưởng r", NAVY)
    p.link(agbox, envbox, (845, 470), (685, 470), num(3, CAM) + " hành động a (1/6 mức)", CAM)
    p.link(agbox, buf, (1165, 247), (1300, 247), num(4, CAM) + " lưu mẫu", CAM)
    p.link(buf, upd, (1490, 310), (1490, 410), num(5, CAM), CAM)
    p.text(1340, 330, 140, 30, sub("đủ 1024 bước"), extra="align=right;")
    p.link(upd, agbox, (1300, 445), (1165, 445), num(6, LUC) + " trọng số mới", LUC, "dashed=1;")
    p.icon(1208, 500, "loop", 34)
    p.text(1185, 534, 80, 20, sub("lặp lại"), extra="align=center;")

    # Vùng B: sử dụng
    p.rect(15, 660, 1530, 290, fill="#FFF9F1", stroke=CAM, extra="dashed=1;strokeWidth=2;arcSize=3;")
    p.text(330, 666, 900, 30, "<b>B · GIAI ĐOẠN SỬ DỤNG</b>  " + sub("miền kiểm thử 491 ngày · chạy tất định, không huấn luyện lại"),
           size=16, color=CAM)
    cards = [("save", "Mô hình đã chọn", "checkpoint có chi phí thấp nhất<br>trên miền xác thực, FR ≥ 85%", "#FFFFFF", NAVY),
             ("receipt", "Phiếu đề nghị đặt hàng", "mỗi kho mỗi ngày một phiếu<br>số lượng, ngày cần hàng", "#FFFFFF", NAVY),
             ("monitor", "Ứng dụng Streamlit", "5 màn hình<br>chi phí, tỷ lệ đáp ứng, phiếu", "#FFFFFF", NAVY),
             ("user", "Người quản lý", "duyệt phiếu, bỏ dòng,<br>tải CSV gửi phòng mua hàng", "#EAF3FD", BLUE)]
    d = [p.vcard(60 + k * 380, 715, 270, 200, ic, a, b, fill=f, stroke=s, s=56, extra="strokeWidth=2;")
         for k, (ic, a, b, f, s) in enumerate(cards)]
    p.link(agbox, d[0], (1005, 580), (195, 715), num(7, NAVY) + " lưu checkpoint tốt nhất", NAVY, pts=[(1005, 630), (195, 630)])
    for k in range(3):
        x1 = 60 + k * 380 + 270
        p.link(d[k], d[k + 1], (x1, 815), (x1 + 110, 815), num(8 + k, NAVY), NAVY)
    return p


# ================================================================ 3. Mô hình khái niệm
def trang3():
    p = Page("3. Mô hình khái niệm", 1320, 920)
    title(p, "MÔ HÌNH KHÁI NIỆM: MỘT NGÀY VẬN HÀNH",
          "Mỗi ngày, mỗi tác tử nhìn tình trạng kho, quyết định lượng đặt, rồi nhận phần thưởng theo chi phí phát sinh")

    # ---- Khối tác tử: quan sát vào, hành động ra
    ag = p.rect(360, 95, 600, 180, fill="#F3EEFB", stroke=TIM, extra="strokeWidth=2.5;arcSize=5;")
    p.icon(378, 103, "robot", 44)
    p.text(430, 103, 520, 44, "<b>300 tác tử PS-IPPO</b> " + sub("· mỗi tác tử phụ trách một cặp (kho, mặt hàng), dùng chung một mạng"),
           size=16, color=TIM)
    obs = p.rect(378, 158, 200, 102, fill="#FFFFFF", stroke="#C7B8E0", extra="align=left;spacingLeft=10;fontSize=12;",
                 label="<b style='color:#1F3A5F'>Quan sát o · 44 chiều</b><br>"
                       "▸ tồn kho, hàng đang về<br>▸ nhu cầu 7 ngày gần nhất<br>▸ mức đầy kho, giảm giá, lịch")
    net = p.icon(600, 178, "network", 62)
    act = p.rect(700, 158, 242, 102, fill="#FFFFFF", stroke="#C7B8E0")
    p.text(700, 160, 242, 22, "<b style='color:#1F3A5F'>Hành động a · 1 trong 6 mức đặt</b>", size=12, extra="align=center;")
    for k, m in enumerate(["0", "0,5", "1", "2", "3", "5"]):
        p.v(710 + k * 38, 188, 34, 34, f"<b>{m}</b>", F + f"rounded=1;arcSize=20;html=1;fillColor=#FCE3C7;"
            f"strokeColor={CAM};fontSize=12;fontColor={NAVY};")
    p.text(700, 228, 242, 26, sub("q = ⌈m · D̄ · L̄⌉, với L̄ = 2 ngày"), extra="align=center;")
    p.link(obs, net, (578, 209), (600, 209), "", TIM, "strokeWidth=2;")
    p.link(net, act, (662, 209), (700, 209), "", TIM, "strokeWidth=2;")

    # ---- Hàng giữa: khách hàng ← kho ← nhà cung cấp
    kh = p.vcard(60, 360, 240, 180, "people", "Khách hàng",
                 "nhu cầu D<sub>t</sub> thực tế từ M5<br>từ dưới 0,1 đến gần 143 đơn vị/ngày", fill="#F5F0FA", stroke=TIM, s=70,
                 extra="strokeWidth=2;")
    kho = p.rect(450, 335, 420, 240, fill="#FFF4F2", stroke=DO, extra="strokeWidth=2.5;arcSize=5;")
    p.icon(468, 350, "warehouse", 78)
    p.text(560, 352, 300, 70, "<b style='font-size:17px'>Kho w</b><br>" + sub("30 mặt hàng dùng chung sức chứa W<sub>w</sub>"),
           size=14, color=DO)
    p.text(468, 440, 190, 22, sub("Mức đầy kho u<sub>t</sub><sup>(w)</sup>"))
    p.v(468, 464, 190, 24, "<span style='font-size:11.5px'>tổng tồn kho của kho / W<sub>w</sub></span>",
        F + f"rounded=1;arcSize=30;html=1;fillColor=#FFFFFF;strokeColor={NAVY};strokeWidth=1.5;")
    p.text(682, 440, 175, 22, sub("Hàng đang về (đã đặt)"))
    for k in range(3):
        p.v(682 + k * 60, 464, 54, 24, f"<span style='font-size:11px'>+{k + 1} ngày</span>",
            F + "rounded=1;arcSize=20;html=1;fillColor=#FCE3C7;strokeColor=#E0A060;")
    p.rect(468, 505, 384, 54, fill="#FFFFFF", stroke="#E6C9C4", extra="fontSize=12;",
           label="Tồn kho I<sub>t</sub> + hàng đang về = <b>vị thế tồn kho</b><br>"
                 + sub("hàng về vượt chỗ trống thì bị từ chối tại cửa nhập"))
    ncc = p.vcard(1020, 360, 240, 180, "truck", "Nhà cung cấp",
                  "thời gian giao L ∈ {1, 2, 3} ngày<br>ngẫu nhiên, không biết trước", fill="#FFF3E6", stroke=CAM, s=70,
                  extra="strokeWidth=2;")

    # ---- Khối chi phí
    cp = p.rect(360, 650, 600, 150, fill="#FDECEC", stroke=DO, extra="strokeWidth=2.5;arcSize=5;")
    p.icon(378, 658, "coins", 40)
    p.text(426, 658, 520, 40, "<b>Chi phí và mức phục vụ trong ngày</b> " + sub("· tính riêng cho từng cặp"),
           size=16, color=DO)
    items = [("Lưu kho", "1,0 /đv/ngày", "#E3F2FD", "#1E6FC2"), ("Thiếu hàng", "10,0 /đv", "#FDE2E1", DO),
             ("Đặt hàng", "5,0 /lần", "#FFF3E0", CAM), ("Tràn kho", "5,0 /đv", "#F3E5F5", TIM),
             ("Phạt mức<br>phục vụ", "FR &lt; 85%", "#E8F5E9", LUC)]
    w = (564 - 4 * 8) / 5
    for k, (a, b, f, s) in enumerate(items):
        p.rect(378 + k * (w + 8), 710, w, 72, fill=f, stroke=s, extra="fontSize=12.5;strokeWidth=1.5;",
               label=f"<b>{a}</b><br>{sub(b)}")

    # ---- Luồng 1 → 6
    p.link(kho, ag, (660, 335), (660, 275), num(1, NAVY) + " kiểm tra tình trạng", NAVY)
    p.link(ag, ncc, (960, 185), (1140, 360), num(2, CAM) + " đơn đặt q", CAM, pts=[(1140, 185)])
    p.link(ncc, kho, (1020, 430), (870, 430), num(3, CAM) + " hàng về", CAM)
    p.link(kho, kh, (450, 430), (300, 430), num(4, NAVY) + " bán min(I, D)", NAVY)
    p.text(310, 438, 135, 36, sub("phần thiếu bị mất,<br>khách sang nơi khác"), extra="align=center;")
    p.text(875, 438, 140, 36, sub("sau 1–3 ngày,<br>theo chỗ trống"), extra="align=center;")
    p.link(kho, cp, (660, 575), (660, 650), num(5, DO) + " ghi nhận", DO)
    p.link(cp, ag, (360, 725), (360, 185), num(6, LUC), LUC, "dashed=1;strokeWidth=3;", pts=[(25, 725), (25, 185)])
    p.text(40, 145, 310, 36, f"<b style='color:{LUC}'>phần thưởng r = −(chi phí + phạt)</b><br>"
           + sub("chuẩn hóa theo cặp · sang ngày t + 1"), size=13)
    p.icon(40, 600, "loop", 40)
    p.text(86, 600, 230, 40, f"<b style='color:{LUC}'>lặp lại mỗi ngày</b><br>" + sub("365 ngày một tập"), size=13)

    # ---- Dải đối chiếu nghiệp vụ ↔ học tăng cường
    p.text(20, 818, 1280, 24, "<b>Mỗi bước nghiệp vụ ứng với một thành phần học tăng cường</b>", size=13, color=NAVY)
    steps = [("Kiểm tra tình trạng", "quan sát o", NAVY), ("Ra quyết định", "hành động a", CAM),
             ("Nhận hàng", "chuyển trạng thái", CAM), ("Bán hàng", "nhu cầu thực tế", NAVY),
             ("Ghi nhận", "chi phí trong ngày", DO), ("Học từ kết quả", "phần thưởng r", LUC)]
    w = (1280 - 5 * 10) / 6
    for k, (a, b, c) in enumerate(steps):
        p.rect(20 + k * (w + 10), 846, w, 54, fill="#FFFFFF", stroke=c, extra="fontSize=12.5;strokeWidth=1.5;",
               label=f"{num(k + 1, c)} <b>{a}</b><br>{sub('→ ' + b)}")
    return p


# ================================================================ 4. Vòng học tăng cường
def trang4():
    p = Page("4. Vòng học tăng cường", 1120, 640)
    title(p, "VÒNG TƯƠNG TÁC VÀ HUẤN LUYỆN CỦA MỘT TÁC TỬ")
    for off, col in ((16, "#D7CCE8"), (8, "#BFAEDB")):
        p.rect(60 + off, 90 + off, 300, 230, fill="#F7F3FC", stroke=col)
    ag = p.vcard(60, 90, 300, 230, "robot", "Tác tử i · cặp (kho w, mặt hàng j)",
                 "Actor π<sub>θ</sub>(a | o) chọn mức đặt<br>Critic V<sub>φ</sub>(o) ước lượng giá trị<br><b>× 300 tác tử, cùng θ, φ</b>",
                 fill="#EDE7F6", stroke=TIM, s=70, extra="strokeWidth=2.5;")
    env = p.rect(640, 80, 430, 290, fill="#FFF4F2", stroke=DO, extra="strokeWidth=2.5;arcSize=4;")
    p.icon(660, 92, "warehouse", 44)
    p.text(712, 96, 340, 36, "<b>Môi trường kho · một ngày</b>", size=16, color=DO)
    for k, t in enumerate(["đổi a thành lượng đặt q", "nhận hàng đến hạn", "áp sức chứa, từ chối phần tràn",
                           "ghi đơn mới, về sau 1–3 ngày", "bán hàng theo nhu cầu M5", "tính chi phí, tỷ lệ đáp ứng, r"]):
        p.rect(660, 145 + k * 36, 390, 30, fill="#FFFFFF", stroke="#E6C9C4",
               label=f"<span style='color:{DO};font-weight:bold'>{k + 1}.</span>&nbsp; {t}", extra="align=left;spacingLeft=10;fontSize=12.5;")
    p.link(env, ag, (640, 130), (360, 130), num(1, NAVY) + " quan sát o<sub>t</sub>", NAVY)
    p.link(ag, env, (360, 205), (640, 205), num(2, CAM) + " hành động a<sub>t</sub>", CAM)
    p.link(env, ag, (640, 280), (360, 280), num(3, XANH) + " phần thưởng r<sub>t</sub>, o<sub>t+1</sub>", XANH)

    buf = p.hcard(60, 450, 430, 100, "stack", "Bộ đệm 1024 bước × 300 tác tử", "307.200 mẫu (o, a, r, V) mỗi vòng",
                  fill="#FFF3E6", stroke=CAM, s=56, extra="strokeWidth=2.5;")
    upd = p.hcard(640, 450, 430, 100, "gear", "Cập nhật PPO",
                  "GAE riêng từng cặp → chuẩn hóa lợi thế<br>→ PPO-clip ε = 0,2 → θ, φ mới", fill="#EAF6EC", stroke=LUC, s=56,
                  extra="strokeWidth=2.5;")
    p.link(ag, buf, (150, 320), (150, 450), num(4, CAM) + " lưu", CAM)
    p.link(buf, upd, (490, 500), (640, 500), num(5, CAM) + " đủ vòng", CAM)
    p.link(upd, ag, (855, 450), (300, 320), num(6, LUC) + " trọng số mới", LUC, "dashed=1;", pts=[(855, 400), (300, 400)])
    return p


# ================================================================ 5. Actor–Critic
def trang5():
    p = Page("5. Kiến trúc Actor–Critic", 1240, 620)
    title(p, "KIẾN TRÚC ACTOR–CRITIC DÙNG CHUNG")
    groups = [("thứ trong tuần, lịch · 27", 27, "#B9C4D3"), ("đáp ứng, quy mô, giảm giá · 3", 3, "#E8918A"),
              ("mức đầy kho, tỷ trọng · 2", 2, "#7FC4BA"), ("nhu cầu 7 ngày · 7", 7, "#F2A65A"),
              ("tồn kho, vị thế, hàng về · 5", 5, "#4C6585")]
    y = 120
    for t, d, c in groups:
        h = 8 * d
        p.v(210, y, 46, h, "", f"html=1;fillColor={c};strokeColor=#FFFFFF;strokeWidth=1.5;")
        p.text(10, y + h / 2 - 10, 192, 20, t, size=12, color=NAVY, extra="align=right;")
        y += h
    obs = p.v(210, 120, 46, 352, "", f"html=1;fillColor=none;strokeColor={NAVY};strokeWidth=2;")
    p.text(160, 477, 146, 26, "<b>o<sub>i</sub><sup>t</sup> · 44 chiều</b>", size=13, color=NAVY, extra="align=center;")

    act = p.hcard(350, 120, 260, 120, "network", "Actor · θ", "2 lớp ẩn × 128, Tanh", fill="#E0F2F1", stroke=XANH, s=64,
                  extra="strokeWidth=2.5;")
    cri = p.hcard(350, 360, 260, 120, "network", "Critic · φ", "2 lớp ẩn × 128, Tanh", fill="#FFF0E0", stroke=CAM, s=64,
                  extra="strokeWidth=2.5;")
    p.link(obs, act, (256, 210), (350, 180), num(1, NAVY), NAVY)
    p.link(obs, cri, (256, 390), (350, 420), num(1, NAVY), NAVY)

    dist = p.rect(690, 105, 250, 150, fill="#FFFFFF", stroke="#B8C7D9")
    p.text(690, 108, 250, 24, "<b>softmax → π<sub>θ</sub>(a | o)</b>", size=13, color=XANH, extra="align=center;")
    p.text(690, 136, 250, 22, sub("xác suất chọn từng mức m"), extra="align=center;")
    for k, m in enumerate(["0", "0,5", "1", "2", "3", "5"]):
        p.v(704 + k * 38, 165, 34, 40, f"<b>{m}</b>", F + f"rounded=1;arcSize=20;html=1;fillColor=#D7EFEC;"
            f"strokeColor={XANH};fontSize=13;fontColor={NAVY};")
    p.text(690, 212, 250, 36, sub("lượng đặt q = ⌈m · D̄ · L̄⌉<br>với L̄ = 2 ngày"), extra="align=center;")
    v = p.v(780, 380, 80, 80, "<b style='font-size:20px'>V</b>", F + f"ellipse;html=1;fillColor=#FFE0C2;strokeColor={CAM};strokeWidth=2.5;fontColor={NAVY};")
    p.text(760, 462, 120, 24, "V<sub>φ</sub>(o) · giá trị", size=13, color=NAVY, extra="align=center;")
    p.link(act, dist, (610, 180), (690, 180), num(2, XANH), XANH)
    p.link(cri, v, (610, 420), (780, 420), num(2, CAM), CAM)

    upd = p.v(1000, 190, 210, 215,
              "<b style='font-size:15px'>Cập nhật sau mỗi vòng</b><br><br><div style='text-align:left'>"
              "<b>1.</b> GAE riêng từng cặp<br><b>2.</b> chuẩn hóa lợi thế<br><b>3.</b> làm phẳng 307.200 mẫu<br>"
              "<b>4.</b> cập nhật θ, φ bằng<br>&nbsp;&nbsp;&nbsp;&nbsp;hai bộ tối ưu riêng</div>",
              F + f"shape=label;html=1;whiteSpace=wrap;rounded=1;arcSize=6;image={uri('gear')};imageWidth=48;imageHeight=48;"
              "imageAlign=center;imageVerticalAlign=top;spacingTop=58;spacing=10;verticalAlign=top;align=center;"
              f"fillColor=#EAF6EC;strokeColor={LUC};strokeWidth=2.5;fontSize=13;")
    p.link(upd, act, (1105, 190), (480, 120), num(3, LUC) + " cập nhật θ", LUC, "dashed=1;", pts=[(1105, 85), (480, 85)])
    p.link(upd, cri, (1105, 405), (480, 480), num(3, LUC) + " cập nhật φ", LUC, "dashed=1;", pts=[(1105, 560), (480, 560)])
    return p


pages = [trang1(), trang2(), trang3(), trang4(), trang5()]
mx = ET.Element("mxfile", host="app.diagrams.net", type="device")
for pg in pages:
    mx.append(pg.xml())
ET.indent(mx)
ET.ElementTree(mx).write(sys.argv[1], encoding="utf-8", xml_declaration=True)
print("ok", [(pg.name, pg.w, pg.h) for pg in pages])
