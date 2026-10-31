"""
scripts/export_tex_numbers.py
=============================
Mot nguon so lieu duy nhat cho bao cao: sinh Report KLTN/so_lieu.tex tu results/*.json
thanh cac lenh LaTeX (\\fillIPPO, \\deltaSS, ...), roi quet tai lieu tim cac so cu.

Cach dung:
    python scripts/export_tex_numbers.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
REPORT = ROOT.parent / "Report KLTN"
# So da loi thoi: chi la cac moc can thay sau khi co ket qua moi
SO_CU = ["25,7", "17,3", "90,15", "90,65", "92,65", "92,28", "0/19"]
TAI_LIEU = [ROOT.parent / "README.md", ROOT.parent / "docs" / "van_dap.md", ROOT.parent / "docs" / "TIEN_DO.md",
            *REPORT.glob("content/*.tex"), *REPORT.glob("frontmatter/*.tex")]


def doc(ten):
    p = RES / ten
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def so(x, d=1):
    """Dinh dang kieu Viet: dau phay thap phan, dau cham hang nghin."""
    s = f"{x:,.{d}f}"
    return s.replace(",", "@").replace(".", ",").replace("@", ".")


def lenh(multi, iso, tk):
    L = {}
    if multi:
        L["chiPhiIPPO"] = so(multi["ippo_cost_mean"], 0)
        L["fillIPPO"] = so(100 * multi["ippo_fill_mean"]) + r"\%"
        L["soHatGiong"] = str(len(multi["seeds"]))
        for ten, ma in [("(s,S)", "SS"), ("Newsvendor theo nhóm", "NVNhom"),
                        ("(s,S) biết sức chứa", "SSKho"), ("Newsvendor dự báo", "NVDuBao")]:
            g = multi["gap_vs_baseline"].get(ten)
            if g:
                L[f"delta{ma}"] = so(g["mean"]) + r"\%"
    if iso:
        for ten, ma in [("(s,S)", "SS"), ("Newsvendor", "NV"), ("(s,S) biết sức chứa", "SSKho"),
                        ("Newsvendor dự báo", "NVDuBao")]:
            r = iso["ket_qua"].get(ten)
            if r and r.get("delta_ippo_percent") is not None:
                L[f"cungMuc{ma}"] = so(r["delta_ippo_percent"]) + r"\%"
    if tk:
        for ten, ma in [("(s,S)", "SS"), ("Newsvendor theo nhóm", "NVNhom")]:
            r = tk["gap_chi_phi"].get(ten, {}).get("khoi_7")
            if r:
                L[f"iqm{ma}"] = so(r["iqm_percent"]) + r"\%"
                L[f"ciThap{ma}"] = so(r["ci95_low"]) + r"\%"
                L[f"ciCao{ma}"] = so(r["ci95_high"]) + r"\%"
    return L


def quet():
    dem = 0
    for p in TAI_LIEU:
        if not p.exists():
            continue
        for i, dong in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for x in SO_CU:
                if re.search(rf"(?<![\d,]){re.escape(x)}(?![\d,])", dong):
                    print(f"  {p.name}:{i} con so cu {x}")
                    dem += 1
    print(f"Quet xong: {dem} cho can xem lai.")


def main():
    L = lenh(doc("multiseed_main.json"), doc("same_service_main_s42.json"), doc("multi_seed_stats.json"))
    if L:
        out = REPORT / "so_lieu.tex"
        out.write_text("% SINH TU DONG boi scripts/export_tex_numbers.py - KHONG sua tay.\n"
                       + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in L.items()), encoding="utf-8")
        print(f"Da luu {out} ({len(L)} lenh)")
    else:
        print("Chua co results/*.json cuoi cung: khong sinh so_lieu.tex.")
    quet()


if __name__ == "__main__":
    main()
