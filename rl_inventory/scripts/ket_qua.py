"""
scripts/ket_qua.py
==================
MOT file so lieu duy nhat: results/KET_QUA.txt.

File duoc DUNG LAI TOAN BO tu cac file ket qua goc moi khi train xong
(train.py), danh gia xong (evaluate.py) hoac tong hop (campaign.py tonghop/eval):
    results/train_log_<tag>.csv          -> qua trinh huan luyen
    checkpoints/train_state_<tag>.json   -> config, seed cua lan train
    results/summary_<tag>.json           -> danh gia tren mien test
    results/all_episodes_<tag>.csv       -> kiem dinh theo cap giua 2 lan chay (ablation, hold-out)
    results/multiseed_main.json          -> RQ1 (da hat giong, campaign.SEEDS)
    results/rq3_multiseed.json           -> RQ3 (da hat giong, campaign.SEEDS)
    results/iso_service.json, scale_group_v2_main.json, regime_analysis_main.json,
    robustness_main.json, sensitivity_main.json -> phan tich phu cua mo hinh chinh
Moi muc la mot bang gon; bang baseline chi in mot lan. Chi tiet tung lan danh gia
van nam trong summary_<tag>.json.
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


def _train_info(d, min_fill=MUC_DV):
    """Cac chi so chinh cua mot lan train tu DataFrame train_log (dung chung cho
    dinh_dang_train va bang huan luyen trong KET_QUA.txt)."""
    ev = d.dropna(subset=["eval_cost"]).drop_duplicates("episode")
    dat = ev[ev.eval_fill >= min_fill]
    cuoi = d.tail(100)
    i = {"ep": int(d.episode.max()), "phut": d.elapsed_s.iloc[-1] / 60, "best": None,
         "hoi_tu": None, "fill": cuoi.fill_rate.mean(),
         "entropy": cuoi.entropy.dropna().mean(), "ev": cuoi.explained_variance.dropna().mean()}
    if len(dat):
        b = dat.loc[dat.eval_cost.idxmin()]
        i.update(best=int(b.episode), cp_val=b.eval_cost, fill_val=b.eval_fill,
                 dat85=int(dat.episode.iloc[0]))
        moc = d.episode.max() * 0.8
        truoc, sau = dat[dat.episode <= moc], dat[dat.episode > moc]
        if len(dat) >= 5 and len(truoc) and len(sau):
            i["hoi_tu"] = (sau.eval_cost.min() / truoc.eval_cost.min() - 1) * 100
    return i


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
    i = _train_info(d, min_fill)
    L.append(f"So episode da train: {i['ep']} | thoi gian (phien cuoi): {i['phut']:.1f} phut")
    if i["best"] is not None:
        L.append(f"Mo hinh duoc chon (best_model): episode {i['best']}, "
                 f"chi phi val {i['cp_val']:,.0f}, fill val {i['fill_val']:.1%}")
        L.append(f"Episode dau tien dat fill val >= {min_fill:.0%}: {i['dat85']}")
        if i["hoi_tu"] is not None:
            g = i["hoi_tu"]
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


BASE = ["(s,S)", "(s,S) theo nhóm", "EOQ", "EOQ theo nhóm", "Newsvendor", "Newsvendor theo nhóm"]
NGAN = {"(s,S)": "(s,S)", "(s,S) theo nhóm": "(s,S)-N", "EOQ": "EOQ", "EOQ theo nhóm": "EOQ-N",
        "Newsvendor": "NV", "Newsvendor theo nhóm": "NV-N",
        "(s,S) (cùng mức PV)": "(s,S)=PV", "Newsvendor (cùng mức PV)": "NV=PV"}
ABL = ["abl_reward_cu", "abl_trunk", "abl_nocal", "abl_nowh", "abl_event", "abl_warm"]
MO_TA = {"abl_ref": "Mốc: cấu hình chính, 1.000 ep", "abl_reward_cu": "Phạt SLA kiểu cũ",
         "abl_trunk": "Chung thân + tắt value norm", "abl_nocal": "Bỏ đặc trưng lịch",
         "abl_nowh": "Bỏ tín hiệu cấp kho", "abl_event": "Thêm đặc trưng sự kiện tới",
         "abl_warm": "Warm-start lịch sử cầu"}
GACH = "=" * 100


def _ippo(s):
    return next(r for r in s["summary"] if r["policy"] == "IPPO")


def _sao(p):
    return "*" if p < 0.05 else " "


def _o_delta(r):
    """Mot o 'Δ%' 9 ky tu tu mot muc per_baseline."""
    return f"{r['gap_percent']:+8.1f}{_sao(r['p_paired'])}" if r else f"{'—':>9}"


def _bang_cp(rows):
    """[(ten, dong summary)] -> bang chi phi theo thanh phan + fill."""
    L = [f"  {'Chính sách':<28}{'Tổng CP':>11}{'Lưu kho':>11}{'Thiếu hàng':>12}"
         f"{'Đặt hàng':>10}{'Tràn kho':>10}{'Fill':>7}"]
    for ten, r in rows:
        L.append(f"  {ten:<28}{r['total_cost_mean']:>11,.0f}{r['holding_cost_mean']:>11,.0f}"
                 f"{r['stockout_cost_mean']:>12,.0f}{r['ordering_cost_mean']:>10,.0f}"
                 f"{r['overflow_cost_mean']:>10,.0f}{r['fill_rate_mean']:>7.1%}")
    return L


def _chi_phi_ippo(res, tag):
    import pandas as pd
    p = res / f"all_episodes_{tag}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    return d[d.policy == "IPPO"].sort_values("episode").total_cost.values


def _so_cap(res, a, b):
    """Kiem dinh theo cap IPPO cua lan chay a so voi b (cung hat giong danh gia) -> (Δ%, p)."""
    from scipy.stats import ttest_rel
    x, y = _chi_phi_ippo(res, a), _chi_phi_ippo(res, b)
    if x is None or y is None or len(x) != len(y):
        return None
    return (x.mean() / y.mean() - 1) * 100, ttest_rel(x, y).pvalue


def _ket_luan(re_hon, dat_hon, n):
    if re_hon == n:
        return f"IPPO rẻ hơn ({n}/{n} seed)"
    if dat_hon == n:
        return f"IPPO đắt hơn ({n}/{n} seed)"
    if re_hon and dat_hon:
        return f"không nhất quán (rẻ hơn {re_hon}, đắt hơn {dat_hon} seed)"
    if re_hon:
        return f"rẻ hơn {re_hon}/{n} seed, còn lại không khác biệt"
    if dat_hon:
        return f"đắt hơn {dat_hon}/{n} seed, còn lại không khác biệt"
    return "không khác biệt có ý nghĩa"


def _hoi_tu(i):
    if not i:
        return "—"
    if i["best"] is None:
        return "[!] val chưa đạt 85%"
    if i["hoi_tu"] is None:
        return "[!] 20% cuối không đạt 85%"
    return f"{'còn giảm' if i['hoi_tu'] < -2 else 'đi ngang'} ({i['hoi_tu']:+.1f}%)"


def _rq1(res):
    d = _doc_json(res / "multiseed_main.json")
    if not d:
        return ["  (chưa có - cần đánh giá main_s* rồi chạy campaign.py tonghop)"]
    seeds = list(d["seeds"])
    sm = {s: _doc_json(res / f"summary_main_s{s}.json") for s in seeds}
    s0 = sm[seeds[0]]
    L = [f"  {s0.get('n_pairs')} cặp kho–SKU, {s0['meta'].get('n_episodes')} lần đánh giá/chính sách"
         " (cùng hạt giống cho mọi chính sách).", ""]
    L += _bang_cp([(f"IPPO seed {s}", _ippo(sm[s])) for s in seeds if sm[s]]
                  + [(r["policy"], r) for r in s0["summary"] if r["policy"] != "IPPO"])
    L += [f"  → IPPO trung bình {len(seeds)} seed: {d['ippo_cost_mean']:,.0f} ± {d['ippo_cost_sd']:,.0f}"
          f" (SD giữa seed), fill {d['ippo_fill_mean']:.1%}", "",
          f"  {'Δ% IPPO so với':<24}" + "".join(f"{'seed ' + s:>9}" for s in seeds) + f"{'TB':>8}   Kết luận"]
    for b in BASE:
        g = d["gap_vs_baseline"].get(b)
        if g:
            L.append(f"  {b:<24}" + "".join(_o_delta(d["seeds"][s]["per_baseline"].get(b)) for s in seeds)
                     + f"{g['mean']:+8.1f}   "
                     + _ket_luan(g.get("n_seeds_re_hon", 0), g.get("n_seeds_dat_hon", 0), g["n_seeds"]))
    duoi = [r["policy"] for r in s0["summary"] if r["fill_rate_mean"] < MUC_DV]
    L.append(f"  Chính sách dưới fill {MUC_DV:.0%}: {', '.join(duoi) if duoi else 'không có'}")

    iso = _doc_json(res / "iso_service.json")
    if iso:
        t, kq = iso["target_fill"], iso["ket_qua"]
        ip = kq["IPPO"]["cost"]
        L += ["", f"  Phân tích phụ - cùng mức phục vụ: baseline chỉnh tham số để đạt fill ≥ {t:.1%}"
                  f" (≈ IPPO seed {seeds[0]}):"]
        vuot = False
        for n, r in kq.items():
            if n == "IPPO":
                continue
            if r.get("khong_dat") or r.get("cost") is None:
                L.append(f"    {n:<22}không đạt được mức fill này")
                continue
            L.append(f"    {n:<22}{r['cost']:>11,.0f}  fill {r['fill']:.1%}   Δ IPPO {(ip / r['cost'] - 1) * 100:+.1f}%")
            vuot |= r["fill"] > t + 0.01
        if vuot:
            L.append("    [!] Fill baseline vượt mục tiêu > 1 điểm % (lưới tham số thô) → lợi thế IPPO bị phóng đại.")
    return L


def _rq2(res):
    L = []
    q = _doc_json(res / "scale_group_v2_main.json")
    if q:
        pols = [p for p in ["IPPO"] + BASE if p in q]
        L += ["  a) Fill theo quy mô cầu của cặp (cầu TB/ngày), mô hình main_s42:",
              f"    {'Nhóm':<22}{'cặp':>5}" + "".join(f"{NGAN.get(p, p):>9}" for p in pols)]
        for g in q["IPPO"]:
            L.append(f"    {g:<22}{q['IPPO'][g]['n_pairs']:>5}" + "".join(f"{q[p][g]['fill']:>9.1%}" for p in pols))
        thap = [g for g in q["IPPO"] if q["IPPO"][g]["fill"] < MUC_DV]
        if thap:
            L.append(f"    [!] IPPO dưới fill {MUC_DV:.0%} ở nhóm: {', '.join(thap)}")

    tags = [("holdout_unseen", "IPPO chưa gặp (holdout)"), ("holdout_seen_ref", "IPPO đã gặp (abl_ref)"),
            ("holdout_seen_main", "IPPO đã gặp (main_s42)")]
    sh = {t: _doc_json(res / f"summary_{t}.json") for t, _ in tags}
    co = [(t, ten) for t, ten in tags if sh[t]]
    if co:
        s0 = sh[co[0][0]]
        L += ["", f"  b) Hold-out: {s0.get('n_pairs')} cặp = 6 SKU không có trong lúc train holdout × 10 kho (1 seed).",
              "     holdout: train 24 SKU, 1.000 ep | abl_ref: 30 SKU, 1.000 ep | main_s42: 30 SKU, 10.000 ep", ""]
        L += _bang_cp([(ten, _ippo(sh[t])) for t, ten in co]
                      + [(r["policy"], r) for r in s0["summary"] if r["policy"] != "IPPO"])
        L += ["", f"  {'Δ% so với baseline':<28}" + "".join(f"{NGAN[b]:>9}" for b in BASE)]
        for t, ten in co:
            per = sh[t]["stat"]["per_baseline"]
            L.append(f"  {ten:<28}" + "".join(_o_delta(per.get(b)) for b in BASE))
        for ref, ten in [("holdout_seen_ref", "đã gặp cùng 1.000 ep (abl_ref)"),
                         ("holdout_seen_main", "đã gặp 10.000 ep (main_s42)")]:
            k = _so_cap(res, "holdout_unseen", ref)
            if k:
                L.append(f"  Chưa gặp so với {ten}: {k[0]:+.1f}% (p = {k[1]:.1e})")
        duoi = [r["policy"] for r in s0["summary"] if r["policy"] != "IPPO" and r["fill_rate_mean"] < MUC_DV]
        if duoi:
            L.append(f"  [!] {len(duoi)} baseline dưới fill {MUC_DV:.0%} trên tập này → so chi phí KHÔNG cùng mức phục vụ.")
    return L or ["  (chưa có)"]


def _rq3(res):
    d = _doc_json(res / "rq3_multiseed.json")
    if not d:
        return ["  (chưa có - cần đánh giá abl_ref/abl_global/abl_q3 × mọi seed rồi chạy campaign.py tonghop)"]
    L = [f"  {'Lần chạy':<12}{'seed':>5}{'Tổng CP':>13}{'Fill':>7}{'Δ abl_ref':>12}{'p':>10}   Fill val ≥ 85%"]
    for base, v in d.items():
        for sd, r in v["seeds"].items():
            ss, dat = r.get("so_voi_ref"), r.get("episode_dat_85_val")
            L.append(f"  {base:<12}{sd:>5}{r['cost']:>13,.0f}{r['fill']:>7.1%}"
                     + (f"{ss['gap_percent']:+11.1f}{_sao(ss['p_paired'])}{ss['p_paired']:>10.1e}" if ss
                        else f"{'—':>12}{'—':>10}")
                     + (f"   từ ep {dat}" if dat else "   [!] chưa đạt → mô hình dự phòng, chỉ dùng định tính"))
        if "gap_vs_ref_mean" in v:
            L.append(f"  → {base}: TB {v['gap_vs_ref_mean']:+.1f}% so với abl_ref; abl_ref rẻ hơn có ý nghĩa ở "
                     f"{v['n_seeds_ref_re_hon_co_y_nghia']}/{v['n_seeds']} seed")
        L.append("")
    c = [r["cost"] for r in d.get("abl_ref", {}).get("seeds", {}).values()]
    if len(c) > 1:
        L.append(f"  Dao động chi phí giữa {len(c)} seed của abl_ref: {(max(c) - min(c)) / (sum(c) / len(c)) * 100:.1f}%"
                 " (dùng làm ngưỡng đọc mục 4)")
    return L


def _ablation(res, info):
    ref = _doc_json(res / "summary_abl_ref.json")
    if not ref:
        return ["  (chưa có abl_ref)"]
    ss_ref = next(r for r in ref["summary"] if r["policy"] == "(s,S)")["total_cost_mean"]
    L = [f"  {'Lần chạy':<15}{'Thay đổi so với abl_ref':<30}{'Tổng CP':>11}{'Fill':>7}"
         f"{'Δ abl_ref':>11}{'Δ (s,S)':>10}  Hội tụ"]
    khac_mt = False
    for t in ["abl_ref"] + ABL:
        s = _doc_json(res / f"summary_{t}.json")
        if not s:
            continue
        ip = _ippo(s)
        k = None if t == "abl_ref" else _so_cap(res, t, "abl_ref")
        ss = next(r for r in s["summary"] if r["policy"] == "(s,S)")["total_cost_mean"]
        khac = abs(ss - ss_ref) > 0.5
        khac_mt |= khac
        L.append(f"  {t:<15}{MO_TA.get(t, ''):<30}{ip['total_cost_mean']:>11,.0f}{ip['fill_rate_mean']:>7.1%}"
                 + (f"{k[0]:+10.1f}{_sao(k[1])}" if k else f"{'—':>11}")
                 + f" {_o_delta(s['stat']['per_baseline'].get('(s,S)'))}  {_hoi_tu(info.get(t))}"
                 + (" †" if khac else ""))
    L.append("  Chỉ 1 seed: chênh lệch nhỏ hơn mức dao động giữa seed của abl_ref (mục 3) chưa đủ để kết luận.")
    if khac_mt:
        L.append("  † Môi trường đánh giá khác abl_ref (baseline cũng đổi) → đọc thêm cột Δ (s,S) cùng môi trường.")
    return L


def _phan_tich(res):
    L = []
    g = _doc_json(res / "regime_analysis_main.json")
    if g:
        ci = g["B1_cost_gap_ci95_block_bootstrap"]
        cols, nd = list(ci), g.get("n_days_per_group", {})
        L += ["  a) Theo giai đoạn nhu cầu - Δ% chi phí IPPO (* = CI 95% bootstrap khối 7 ngày không chứa 0):",
              f"    {'Giai đoạn':<20}{'ngày':>5}" + "".join(f"{NGAN.get(b, b):>9}" for b in cols)]
        for r in ci[cols[0]]:
            L.append(f"    {r:<20}{nd.get(r, ''):>5}" + "".join(
                f"{ci[b][r]['gap_percent']:+8.1f}{'*' if ci[b][r]['significant'] else ' '}" for b in cols))
        b2 = g.get("B2_shock")
        if b2:
            L += ["", f"  b) Sốc cầu {b2['len']} ngày từ ngày {b2['start']} - fill trước / trong / sau sốc, CP/ngày trong sốc:"]
            for f, kq in b2["results"].items():
                L.append(f"    Cầu ×{f}:{'':<17}{'trước':>7}{'trong':>7}{'sau':>7}{'CP/ngày':>10}")
                for p, ph in kq.items():
                    a, b, c = ph["Trước sốc"], ph["Trong sốc"], ph["Sau sốc"]
                    L.append(f"      {p:<24}{a['fill_rate']:>7.1%}{b['fill_rate']:>7.1%}{c['fill_rate']:>7.1%}"
                             f"{b['cost_per_day']:>10,.0f}")
    r = _doc_json(res / "robustness_main.json")
    if r:
        L += ["", f"  c) Độ bền zero-shot (không train lại), {r.get('episodes')} lần đánh giá/kịch bản - Δ% IPPO:",
              f"    {'Kịch bản':<22}{'IPPO CP':>11}{'Fill':>7}" + "".join(f"{NGAN[b]:>9}" for b in BASE)]
        td = False
        for k, v in r["kich_ban"].items():
            o = []
            for b in BASE:
                x = v.get(b, {}).get("ippo_vs")
                tat_dinh = bool(x) and x["ci95_low"] == x["ci95_high"]
                td |= tat_dinh
                o.append(f"{x['gap_percent']:+8.1f}~" if tat_dinh else _o_delta(x))
            L.append(f"    {k:<22}{v['IPPO']['cost']:>11,.0f}{v['IPPO']['fill']:>7.1%}" + "".join(o))
        if td:
            L.append("    ~ = kịch bản tất định (các lần đánh giá giống hệt nhau) → không kiểm định được, chỉ đọc mức chênh.")
    s = _doc_json(res / "sensitivity_main.json")
    if s:
        L += ["", f"  d) Độ nhạy đơn giá ({len(s.get('grid', []))} bộ đơn giá lưu kho / thiếu hàng / đặt hàng):"]
        for b, v in s["summary"].items():
            L.append(f"    vs {b:<26}IPPO rẻ hơn ở {v['ippo_cheaper_in']:>2}/{v['of']} bộ"
                     f"  (Δ từ {v['gap_min']:+.1f}% đến {v['gap_max']:+.1f}%)")
    return L or ["  (chưa có)"]


def _huan_luyen(info):
    if not info:
        return ["  (chưa có lần chạy nào)"]
    L = [f"  {'Lần chạy':<15}{'Ep':>6}{'Phút':>6}{'Best ep':>8}{'CP val':>11}{'Fill val':>9}{'Đạt 85%':>8}"
         f"  {'Hội tụ':<27}{'Entropy':>8}{'EV':>6}"]
    for t in sorted(info, key=_thu_tu):
        i = info[t]
        mid = (f"{i['best']:>8}{i['cp_val']:>11,.0f}{i['fill_val']:>9.1%}{i['dat85']:>8}" if i["best"] is not None
               else f"{'—':>8}{'—':>11}{'—':>9}{'chưa':>8}")
        L.append(f"  {t:<15}{i['ep']:>6}{i['phut']:>6.0f}{mid}  {_hoi_tu(i):<27}{i['entropy']:>8.3f}{i['ev']:>6.2f}")
    L += ["  Best ep = mô hình được chọn (CP val thấp nhất trong các lần đánh giá val đạt fill ≥ 85%).",
          "  Hội tụ = CP val tốt nhất ở 20% episode cuối so với trước đó (giảm > 2% = còn giảm).",
          "  Phút = phiên train cuối; Entropy, EV (explained variance) = TB 100 episode cuối."]
    chuan = info.get("main_s42", {}).get("state", {})
    for t, i in info.items():
        st = i.get("state", {})
        if chuan and st and (st.get("obs_per_pair"), st.get("n_pairs")) != (chuan.get("obs_per_pair"), chuan.get("n_pairs")):
            L.append(f"  {t}: {st.get('obs_per_pair')} chiều quan sát, {st.get('n_pairs')} cặp"
                     f" (mô hình chính: {chuan.get('obs_per_pair')} chiều, {chuan.get('n_pairs')} cặp)")
    return L


def _phu_luc(res):
    d = _doc_json(res / "multiseed_main.json")
    if not d:
        return ["  (chưa có)"]
    L = [f"  {'Seed':<6}{'So với':<22}{'Δ%':>8}{'Δ tuyệt đối [CI 95%]':>36}{'p (t)':>10}{'p (W)':>10}"
         f"{'d_z':>8}{'IPPO rẻ hơn':>13}"]
    for s, v in d["seeds"].items():
        for b in BASE:
            r = v["per_baseline"].get(b)
            if r:
                ci = f"{r['mean_diff']:+,.0f} [{r['ci95_low']:+,.0f}; {r['ci95_high']:+,.0f}]"
                L.append(f"  {s:<6}{b:<22}{r['gap_percent']:+8.2f}{ci:>36}{r['p_paired']:>10.1e}"
                         f"{r['p_wilcoxon']:>10.1e}{r['d_z']:>+8.2f}{str(r['n_ippo_cheaper']) + '/' + str(r['n']):>13}")
    L.append("  Chi tiết các lần chạy khác: results/summary_<tag>.json (mục stat.per_baseline).")
    return L


def _an_toan(f, *a):
    try:
        return f(*a)
    except Exception as e:      # file dang ghi do / thieu truong -> khong lam hong train/eval
        return [f"  (không đọc được: {e!r})"]


def ghi_ket_qua(root=ROOT):
    """Dung lai results/KET_QUA.txt tu moi file ket qua dang co."""
    import pandas as pd
    res, ck = Path(root) / "results", Path(root) / "checkpoints"
    info = {}
    for p in res.glob("train_log_*.csv"):
        t = p.stem[len("train_log_"):]
        try:
            d = pd.read_csv(p)
            if len(d):
                info[t] = {**_train_info(d), "state": _doc_json(ck / f"train_state_{t}.json") or {}}
        except Exception:       # log dang ghi do / hong
            pass

    L = [f"KẾT QUẢ THỰC NGHIỆM - PS-IPPO quản lý tồn kho đa kho (cập nhật {datetime.now():%Y-%m-%d %H:%M})",
         "Số liệu đọc tự động từ results/ (không gõ tay). Chi phí: đơn vị mô phỏng, TB các lần đánh giá trên miền test.",
         "Δ% = (IPPO − baseline) / baseline: âm = IPPO rẻ hơn.   * = kiểm định theo cặp có ý nghĩa (p < 0,05).",
         "Tên tắt: -N = tham số tinh chỉnh theo nhóm quy mô cầu | =PV = cùng mức phục vụ với IPPO | NV = Newsvendor."]
    for ten, f, a in [("1. RQ1 - MÔ HÌNH CHÍNH (10.000 episode × 3 seed) so với baseline", _rq1, (res,)),
                      ("2. RQ2 - QUY MÔ CẦU KHÁC NHAU VÀ MẶT HÀNG CHƯA GẶP (hold-out)", _rq2, (res,)),
                      ("3. RQ3 - THIẾT KẾ PHẦN THƯỞNG (1.000 episode × 3 seed, so với abl_ref cùng seed)", _rq3, (res,)),
                      ("4. ABLATION KHÁC (1.000 episode, seed 42, so với abl_ref)", _ablation, (res, info)),
                      ("5. PHÂN TÍCH PHỤ CỦA MÔ HÌNH CHÍNH (main_s42)", _phan_tich, (res,)),
                      ("6. HUẤN LUYỆN - MỌI LẦN CHẠY", _huan_luyen, (info,)),
                      ("PHỤ LỤC - CHI TIẾT KIỂM ĐỊNH MÔ HÌNH CHÍNH (30 lần đánh giá theo cặp)", _phu_luc, (res,))]:
        L += ["", GACH, ten, GACH] + _an_toan(f, *a)

    out = res / "KET_QUA.txt"
    res.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    print("Da luu", ghi_ket_qua())
