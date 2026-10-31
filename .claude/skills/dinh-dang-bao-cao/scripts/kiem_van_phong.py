"""Kiểm văn phong và định dạng báo cáo LaTeX theo quy tắc trong SKILL.md.

Cách dùng: python kiem_van_phong.py <thư mục hoặc tệp .tex> [...]
Mỗi cảnh báo in dạng tệp:dòng: loại: nội dung. Bảng sinh tự động (dòng đầu chứa "Sinh tự động") được bỏ qua.
Script chỉ báo chỗ nghi vấn; người viết quyết định có sửa hay không.
"""
import re
import sys
from pathlib import Path

MAX_WORDS = 45          # câu dài hơn thì tách
MAX_CAPTION = 60        # chú thích bảng/hình là cụm danh từ ngắn
AI_PHRASES = [", tức", "Đáng chú ý", "Như vậy", " vốn "]
SKIP_ENVS = ("table", "figure", "algorithm", "algorithmic", "equation", "align", "tikzpicture",
             "tabular", "longtable", "xltabular", "minipage", "titlepage", "center", "flushright")
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0d]")


def strip_latex(s):
    """Bỏ lệnh LaTeX để đếm từ: giữ chữ trong ngoặc của \\textit{...}, bỏ tham chiếu, trích dẫn, công thức."""
    s = re.sub(r"\$[^$]*\$", " X ", s)
    s = re.sub(r"\\(label|ref|eqref|cite|parencite|textcite|bangref\w*|hinhref)\*?(\[[^\]]*\])?\{[^}]*\}(\{[^}]*\})?", " X ", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = re.sub(r"[{}~]", " ", s)
    return s


def words(s):
    return [w for w in strip_latex(s).split() if re.search(r"\w", w)]


def check_file(path, out):
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if lines and "Sinh tự động" in lines[0]:
        return
    is_abstract = "abstract" in path.stem.lower()

    def warn(no, kind, msg):
        out.append(f"{path}:{no}: {kind}: {msg}")

    depth = 0             # đang ở trong môi trường không phải văn xuôi
    para, para_start = [], 0
    item_buf, item_line = None, 0

    def flush_item():
        nonlocal item_buf
        if item_buf is None:
            return
        body = " ".join(item_buf).strip()
        plain = strip_latex(body).strip()
        if plain and not body.startswith(("\\texttt", "$")):
            first = plain[0]
            if first.isalpha() and not first.isupper():
                warn(item_line, "liệt kê", "dòng \\item nên viết hoa chữ đầu")
            if not body.rstrip().endswith((".", ":", "?")):
                warn(item_line, "liệt kê", "dòng \\item nên kết thúc bằng dấu chấm")
        for sent in re.split(r"(?<=[.!?])\s+", body):
            n = len(words(sent))
            if n > MAX_WORDS:
                warn(item_line, "câu dài", f"{n} từ: {strip_latex(sent).strip()[:70]}...")
        item_buf = None

    def flush_para():
        nonlocal para
        if not para:
            return
        body = " ".join(para)
        if re.match(r"\s*(Bảng|Hình)~?\\ref\{[^}]*\}\s+(liệt kê|trình bày|cho thấy|tổng hợp)", body):
            warn(para_start, "mở đoạn", "không mở đoạn bằng \"Bảng/Hình X liệt kê/trình bày...\"; mở bằng nhận xét")
        for sent in re.split(r"(?<=[.!?])\s+", body):
            n = len(words(sent))
            if n > MAX_WORDS:
                warn(para_start, "câu dài", f"{n} từ: {strip_latex(sent).strip()[:70]}...")
        para = []

    for no, raw in enumerate(lines, 1):
        if CTRL.search(raw):
            warn(no, "ký tự lạ", "có ký tự điều khiển (thường do sửa hàng loạt qua heredoc)")
        line = re.sub(r"(?<!\\)%.*", "", raw)
        for env in re.findall(r"\\begin\{(\w+)\*?\}", line):
            if env in SKIP_ENVS:
                depth += 1
        for env in re.findall(r"\\end\{(\w+)\*?\}", line):
            if env in SKIP_ENVS:
                depth = max(0, depth - 1)

        for cap in re.findall(r"\\caption\{(.*)\}", line):
            n = len(strip_latex(cap).strip())
            if n > MAX_CAPTION:
                warn(no, "chú thích dài", f"{n} ký tự; chuyển chi tiết xuống ghi chú hoặc thân bài")
        if "\\textbf" in line and depth == 0 and "frontmatter" not in str(path):
            warn(no, "in đậm", "không \\textbf trong thân bài, nhãn liệt kê hay chú thích")
        if depth:
            continue
        if is_abstract and not line.lstrip().startswith(("{", "\\vspace")):
            if re.search(r"\\(cite|parencite|textcite)", line):
                warn(no, "tóm tắt", "tóm tắt không trích dẫn")
            if re.search(r"\d+[,.]\d+|\d+\s*%|\\%", line):
                warn(no, "tóm tắt", "tóm tắt không nêu số liệu")

        stripped = line.strip()
        # kiểm theo dòng: áp dụng cho cả đoạn văn và dòng liệt kê
        for ph in AI_PHRASES:
            if ph in line:
                warn(no, "văn AI", f"tránh \"{ph.strip()}\"")
        if re.search(r"không phải\b.{1,80}\bmà\b", line):
            warn(no, "văn AI", "tránh cấu trúc \"không phải ... mà ...\"")
        if re.search(r"[^\s\\]\s*;\s", strip_latex(line)):
            warn(no, "chấm phẩy", "không nối mệnh đề bằng dấu chấm phẩy trong văn xuôi")
        if " -- " in line or " --- " in line:
            warn(no, "gạch ngang", "hạn chế gạch ngang chèn ý; tách thành câu riêng")

        if stripped.startswith("\\item"):
            flush_item()
            item_buf, item_line = [stripped[5:]], no
            continue
        if item_buf is not None and stripped and not stripped.startswith(("\\end", "\\begin")):
            item_buf.append(stripped)
            continue
        flush_item()

        if not stripped or stripped.startswith(("\\chapter", "\\section", "\\subsection", "\\subsubsection",
                                                "\\label", "\\begin", "\\end", "\\input", "\\include",
                                                "\\FloatBarrier", "\\newpage", "\\clearpage", "\\vspace",
                                                "\\noindent\\rule")):
            flush_para()
            continue
        if not para:
            para_start = no
        para.append(stripped)
    flush_item()
    flush_para()


def main(args):
    files = []
    for a in args or ["."]:
        p = Path(a)
        files += sorted(p.rglob("*.tex")) if p.is_dir() else [p]
    out = []
    for f in files:
        check_file(f, out)
    print("\n".join(out) if out else "Không có cảnh báo.")
    print(f"\n{len(out)} cảnh báo trên {len(files)} tệp.")


if __name__ == "__main__":
    main(sys.argv[1:])
