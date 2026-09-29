"""
scripts/ket_qua.py
==================
MOT file so lieu duy nhat: results/KET_QUA.txt.

File duoc DUNG LAI TOAN BO tu cac file ket qua goc moi khi train xong
(train.py), danh gia xong (evaluate.py) hoac tong hop (campaign.py tonghop/eval):
    results/train_log_<tag>.csv          -> qua trinh huan luyen
    checkpoints/train_state_<tag>.json   -> config, seed cua lan train
    results/summary_<tag>.json           -> danh gia tren mien test
    results/multiseed_main.json          -> RQ1 (3 hat giong)
    results/rq3_multiseed.json           -> RQ3 (3 hat giong)
Khong ghi noi tiep -> chay lai khong sinh doan trung, nhieu phien song song
khong ghi de mat phan cua nhau, so lieu luon khop file goc.

Chay tay:  python scripts/ket_qua.py
"""
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MUC_DV = 0.85


def _doc_json(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def dinh_dang_train(log_path, state=None, min_fill=MUC_DV):
    """Tom tat mot lan train tu train_log_<tag>.csv (gom ca cac doan resume)."""
    import pandas as pd
    d = pd.read_csv(log_path)
    if d.empty:
        return ["  (log train rong)"]
    L = []
    if state and state.get("config"):
        L.append(f"Config: {state['config']} | seed: {state.get('seed')} | "
                 f"episode muc tieu: {state.get('total_episodes')} | device: {state.get('device')} | "
                 f"quan sat: {state.get('obs_per_pair')} chieu | {state.get('n_pairs')} cap")
    ev = d.dropna(subset=["eval_cost"]).drop_duplicates("episode")
    dat = ev[ev.eval_fill >= min_fill]
    L.append(f"So episode da train: {int(d.episode.max())}"
             f" | thoi gian (phien cuoi): {d.elapsed_s.iloc[-1] / 60:.1f} phut")
    if len(dat):
        b = dat.loc[dat.eval_cost.idxmin()]
        L.append(f"Mo hinh duoc chon (best_model): episode {int(b.episode)}, "
                 f"chi phi val {b.eval_cost:,.0f}, fill val {b.eval_fill:.1%}")
        L.append(f"Episode dau tien dat fill val >= {min_fill:.0%}: {int(dat.episode.iloc[0])}")
        moc = d.episode.max() * 0.8
        truoc, sau = dat[dat.episode <= moc], dat[dat.episode > moc]
        if len(dat) >= 5 and len(truoc) and len(sau):
            g = (sau.eval_cost.min() / truoc.eval_cost.min() - 1) * 100
            L.append(f"Hoi tu: chi phi val tot nhat 20% episode cuoi so voi truoc do {g:+.1f}% "
                     f"({'VAN CON GIAM RO' if g < -2 else 'da di ngang'})")
    else:
        L.append(f"[!] KHONG lan danh gia val nao dat fill >= {min_fill:.0%} "
                 "-> best_model la ban du phong, KHONG dung cho bao cao.")

    cuoi = d.tail(100)
    tong = cuoi.cost_total.mean()
    L.append(f"Trung binh {len(cuoi)} episode train cuoi (chinh sach con kham pha):")
    for c, ten in [("cost_holding", "Luu kho"), ("cost_stockout", "Thieu hang"),
                   ("cost_ordering", "Dat hang"), ("cost_overflow", "Tran kho"),
                   ("cost_service_penalty", "Phat SLA (tin hieu huan luyen)")]:
        v = cuoi[c].mean()
        L.append(f"  {ten:<32}{v:>14,.0f}  ({v / tong:6.1%})")
    L.append(f"  {'Tong':<32}{tong:>14,.0f}")
    L.append(f"  Fill: {cuoi.fill_rate.mean():.1%} | entropy: {cuoi.entropy.dropna().mean():.3f}"
             f" | explained variance: {cuoi.explained_variance.dropna().mean():.3f}")
    return L


def dinh_dang_danh_gia(s):
    """Bang so sanh + kiem dinh theo cap tu summary_<tag>.json (evaluate.py)."""
    meta = s.get("meta", {})
    L = []
    if meta:
        L.append(f"Mien {s.get('mode', 'test')} | {meta.get('n_episodes')} lan danh gia (cung hat giong"
                 f" cho moi chinh sach) | checkpoint {meta.get('checkpoint')} | config {meta.get('config')}"
                 f" | {meta.get('thoi_diem')}")
    L.append(f"{'Chinh sach':<22}{'Tong CP':>13}{'Luu kho':>12}{'Thieu hang':>12}"
             f"{'Dat hang':>11}{'Tran kho':>11}{'Fill':>8}")
    L.append("-" * 89)
    for r in s["summary"]:
        L.append(f"{r['policy']:<22}{r['total_cost_mean']:>13,.0f}{r['holding_cost_mean']:>12,.0f}"
                 f"{r['stockout_cost_mean']:>12,.0f}{r['ordering_cost_mean']:>11,.0f}"
                 f"{r['overflow_cost_mean']:>11,.0f}{r['fill_rate_mean']:>8.1%}")
    per = (s.get("stat") or {}).get("per_baseline", {})
    if per:
        L.append("Kiem dinh theo cap (Delta = IPPO - baseline, CI 95%) va ket luan (p_t < 0,05):")
        for name, r in per.items():
            if r["p_paired"] >= 0.05:
                kl = "khong khac biet co y nghia"
            elif r["mean_diff"] < 0:
                kl = "IPPO RE HON"
            else:
                kl = "IPPO DAT HON"
            L.append(f"  vs {name:<20}: {r['gap_percent']:+7.2f}%  "
                     f"[{r['ci95_low']:+,.0f}; {r['ci95_high']:+,.0f}]  p_t={r['p_paired']:.1e}  "
                     f"p_W={r['p_wilcoxon']:.1e}  d_z={r['d_z']:+.2f}  "
                     f"re hon {r['n_ippo_cheaper']}/{r['n']}  -> {kl}")
    duoi = [r["policy"] for r in s["summary"] if r["fill_rate_mean"] < MUC_DV]
    L.append(f"Chinh sach duoi muc phuc vu {MUC_DV:.0%}: {', '.join(duoi) if duoi else 'khong co'}")
    return L


def _thu_tu(tag):
    tt = tag.replace("_s42", "_s0")          # seed 42 (chinh) dung truoc seed 1, 2
    for i, p in enumerate(["main", "abl_ref", "abl_global", "abl_q3", "abl", "holdout"]):
        if tag.startswith(p):
            return (i, tt)
    return (9, tt)


def ghi_ket_qua(root=ROOT):
    """Dung lai results/KET_QUA.txt tu moi file ket qua dang co."""
    res, ck = Path(root) / "results", Path(root) / "checkpoints"
    L = [f"KET QUA THUC NGHIEM - cap nhat {datetime.now():%Y-%m-%d %H:%M}",
         "Moi con so doc tu file ket qua goc trong results/ (khong go tay).",
         "Chi phi = don vi mo phong chuan hoa. Kiem dinh theo cap, y nghia khi p < 0,05.", ""]

    L += ["=" * 89, "PHAN 1. TONG HOP CAU HOI NGHIEN CUU", "=" * 89]
    d = _doc_json(res / "multiseed_main.json")
    L.append("RQ1 - Mo hinh chinh, 3 hat giong, so voi tung baseline tren mien test:")
    if d:
        L.append(f"  IPPO: chi phi TB {d['ippo_cost_mean']:,.0f} +- {d['ippo_cost_sd']:,.0f} (SD giua seed)"
                 f" | fill TB {d['ippo_fill_mean']:.1%} | {len(d['seeds'])} seed")
        for ref, g in d["gap_vs_baseline"].items():
            L.append(f"  vs {ref:<22}: TB {g['mean']:+7.2f}% (tu {g['min']:+.2f} den {g['max']:+.2f})"
                     f" | co y nghia o {g['n_seeds_significant']}/{g['n_seeds']} seed")
    else:
        L.append("  (chua co - can danh gia main_s42/s1/s2 roi chay campaign.py tonghop)")
    d = _doc_json(res / "rq3_multiseed.json")
    L.append("RQ3 - Thiet ke phan thuong, 3 hat giong, so voi abl_ref cung seed:")
    if d:
        for base, v in d.items():
            for sd, r in v["seeds"].items():
                ss = r.get("so_voi_ref")
                L.append(f"  {base:<11} seed {sd:>2}: chi phi {r['cost']:>12,.0f} fill {r['fill']:.1%}"
                         f" | dat 85% val tu ep {r.get('episode_dat_85_val') or 'chua dat'}"
                         + (f" | vs abl_ref {ss['gap_percent']:+.2f}% (p={ss['p_paired']:.1e})" if ss else ""))
            if "gap_vs_ref_mean" in v:
                L.append(f"  => {base}: TB {v['gap_vs_ref_mean']:+.2f}% so voi abl_ref; abl_ref re hon co y"
                         f" nghia o {v['n_seeds_ref_re_hon_co_y_nghia']}/{v['n_seeds']} seed")
    else:
        L.append("  (chua co - can danh gia abl_ref/abl_global/abl_q3 x 3 seed roi chay campaign.py tonghop)")

    train_tags = {p.stem[len("train_log_"):] for p in res.glob("train_log_*.csv")}
    eval_tags = {p.stem[len("summary_"):] for p in res.glob("summary_*.json")}
    L += ["", "=" * 89, "PHAN 2. TUNG MO HINH (huan luyen -> danh gia tren mien test)", "=" * 89]
    if not (train_tags | eval_tags):
        L.append("  (chua co lan chay nao)")
    for tag in sorted(train_tags | eval_tags, key=_thu_tu):
        L += ["", f"##### {tag} " + "#" * max(3, 82 - len(tag))]
        if tag in train_tags:
            L.append("[Huan luyen]")
            try:
                L += dinh_dang_train(res / f"train_log_{tag}.csv",
                                     _doc_json(ck / f"train_state_{tag}.json"))
            except Exception as e:                  # log dang ghi do / hong
                L.append(f"  (khong doc duoc log train: {e})")
        s = _doc_json(res / f"summary_{tag}.json") if tag in eval_tags else None
        L.append("[Danh gia tren mien test]")
        L += dinh_dang_danh_gia(s) if s else ["  (chua danh gia)"]

    out = res / "KET_QUA.txt"
    res.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    print("Da luu", ghi_ket_qua())
