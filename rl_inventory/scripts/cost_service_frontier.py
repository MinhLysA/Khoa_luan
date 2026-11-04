"""
scripts/cost_service_frontier.py
=====================
[Goi C] Dat cac diem IPPO (mo hinh chinh va cac gia tri phi_dv) len cung mat
phang chi phi - fill voi duong bien cua tung baseline, va doc chi phi tai cac muc
phuc vu 85, 88, 90, 92%. Doc results/baseline_frontier.json (compare_same_service.py) va results/summary_<tag>.json.

Cach dung:
    python scripts/cost_service_frontier.py --tags main_s42 phi20_s42 phi60_s42
"""
import argparse
import json

import numpy as np

from common import ROOT, load_config

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MUC = (0.85, 0.88, 0.90, 0.92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--tags", nargs="+", required=True)
    a = ap.parse_args()

    from compare_same_service import duong_test, doc_tai

    res = ROOT / load_config(a.config)["paths"]["results_dir"]
    bien = json.loads((res / "baseline_frontier.json").read_text(encoding="utf-8"))
    ippo = {}
    for t in a.tags:
        p = res / f"summary_{t}.json"
        if p.exists():
            r = next(x for x in json.loads(p.read_text(encoding="utf-8"))["summary"] if x["policy"] == "IPPO")
            ippo[t] = (r["fill_rate_mean"], r["total_cost_mean"])
    if not ippo:
        raise SystemExit("Chua co summary cua cac tag da chon")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    bang = {}
    for ten, diem in bien.items():
        f, c = duong_test(diem)
        ax.plot(np.array(f) * 100, c, "-", lw=1.5, label=ten)
        bang[ten] = {f"{m:.0%}": doc_tai(f, c, m) for m in MUC}
    f = [v[0] * 100 for v in ippo.values()]
    c = [v[1] for v in ippo.values()]
    ax.scatter(f, c, c="k", marker="*", s=130, zorder=5, label="IPPO")
    for t, x, y in zip(ippo, f, c):
        ax.annotate(t, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Tỷ lệ đáp ứng (%)")
    ax.set_ylabel("Tổng chi phí vận hành (miền kiểm thử)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(res / "cost_service_frontier.png", dpi=140)
    plt.close()

    # Chenh lech cua tung diem IPPO so voi baseline doc tai dung fill cua diem do
    so_sanh = {t: {ten: None if (x := doc_tai(*duong_test(d), v[0])) is None
                   else float(100 * (v[1] / x - 1)) for ten, d in bien.items()}
               for t, v in ippo.items()}
    (res / "cost_service_frontier.json").write_text(json.dumps(
        {"ippo": {t: {"fill": v[0], "cost": v[1]} for t, v in ippo.items()},
         "baseline_tai_muc": bang, "ippo_so_voi_baseline_percent": so_sanh},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print("Da luu results/cost_service_frontier.json, cost_service_frontier.png")
    for t, v in so_sanh.items():
        print(t, {k: None if x is None else round(x, 1) for k, x in v.items()})


if __name__ == "__main__":
    main()
