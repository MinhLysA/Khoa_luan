"""
scripts/build_results.py
========================
MOT file so lieu duy nhat: results/KET_QUA.txt, sap theo 5 cau hoi nghien cuu cua bao cao.

File duoc DUNG LAI TOAN BO tu cac file ket qua goc moi khi train xong (train.py), danh gia
xong (evaluate.py) hoac tong hop (run_experiments.py tonghop/eval):
    results/train_log_<tag>.csv          -> qua trinh huan luyen
    checkpoints/train_state_<tag>.json   -> config, seed cua lan train
    results/summary_<tag>.json           -> danh gia tren mien test
    results/all_episodes_<tag>.csv       -> kiem dinh theo cap giua 2 lan chay (ablation, hold-out)
    results/multiseed_main.json, multiseed_forecast_safety.json -> Cau 1, Cau 4 (nhieu hat giong)
    results/rq3_multiseed.json           -> Cau 3
    results/same_service_<tag>.json      -> cung muc phuc vu (luoi min)
    results/scale_group_v2_main.json     -> Cau 2 (fill theo quy mo cau)
    results/robustness_main.json, shared_supply_*/summary_zeroshot_*.json -> Cau 5
    results/regime_analysis_main.json, sensitivity_main.json, cost_service_frontier.json,
    forecast_metrics.json, perfect_demand.json -> Cau 4 va phan tich bo sung
Ten lan chay (tag) -> ten day du: TEN_DAY_DU ben duoi, in o dau KET_QUA.txt.
Khong ghi noi tiep -> chay lai khong sinh doan trung, so lieu luon khop file goc.

Chay tay:  python scripts/build_results.py
"""
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MUC_DV = 0.85

# Ten day du cua tung lan chay. Tag giu nguyen vi da gan vao ten file checkpoint/ket qua.
TEN_DAY_DU = {
    "main": "Mô hình chính PS-IPPO: quan sát 44 chiều, 10.000 tập, không dùng dự báo",
    "forecast_safety": "PS-IPPO + dự báo (Câu 4): chọn hệ số an toàn trên nền dự báo LightGBM, 10.000 tập",
    "abl_ref": "Mốc ablation: cấu hình chính, 1.000 tập",
    "abl_global": "Câu 3: phần thưởng toàn mạng lưới thay cho phần thưởng riêng từng cặp",
    "abl_q3": "Câu 3: tắt chuẩn hóa phần thưởng theo cặp",
    "abl_reward_cu": "Phạt mức phục vụ kiểu cũ (tỷ lệ theo cầu trung bình)",
    "abl_trunk": "Actor và Critic dùng chung thân mạng, tắt chuẩn hóa giá trị",
    "abl_nocal": "Bỏ đặc trưng lịch",
    "abl_nowh": "Bỏ tín hiệu cấp kho (mức đầy kho, tỷ trọng của cặp trong kho)",
    "abl_event": "Thêm đặc trưng sự kiện sắp tới",
    "abl_warm": "Khởi tạo lịch sử cầu bằng dữ liệu thật",
    "holdout": "Câu 2: huấn luyện trên 24 SKU, giữ lại 6 SKU làm mặt hàng chưa gặp, 1.000 tập",
    "holdout_unseen": "Câu 2: mô hình holdout trên 6 SKU chưa gặp × 10 kho",
    "holdout_seen_ref": "Câu 2: abl_ref (đã thấy 6 SKU khi huấn luyện) trên cùng 60 cặp",
    "holdout_seen_main": "Câu 2: mô hình chính (đã thấy 6 SKU) trên cùng 60 cặp",
    "penalty05_trial": "Chạy thử hệ số phạt mức phục vụ φ = 0,5 (1.000 tập)",
    "penalty10_trial": "Chạy thử hệ số phạt mức phục vụ φ = 1,0 (1.000 tập)",
    "penalty20_trial": "Chạy thử hệ số phạt mức phục vụ φ = 2,0 (1.000 tập)",
    "penalty60_trial": "Chạy thử hệ số phạt mức phục vụ φ = 6,0 (1.000 tập)",
    "penalty100_trial": "Chạy thử hệ số phạt mức phục vụ φ = 10,0 (1.000 tập)",
    "zeroshot_shared_supply_tight": "Câu 5: mô hình chính, nguồn cung chung chặt (2 × cầu TB), không huấn luyện lại",
    "zeroshot_shared_supply_mild": "Câu 5: mô hình chính, nguồn cung chung vừa (3 × cầu TB), không huấn luyện lại",
}
TEN_CHINH_SACH = {
    "IPPO": "PS-IPPO: một mạng Actor–Critic dùng chung cho 300 tác tử (cặp kho–SKU)",
    "IPPO+DB": "PS-IPPO + dự báo (Câu 4)",
    "EOQ": "Lượng đặt hàng kinh tế (Economic Order Quantity)",
    "(s,S)": "Điểm đặt lại s, đặt lên mức tối đa S",
    "Newsvendor (NV)": "Người bán báo: mức đặt theo tỷ lệ tới hạn",
    "… theo nhóm (-N)": "Cùng chính sách, tham số riêng cho 4 nhóm quy mô cầu (đối chứng mạnh hơn)",
    "(s,S) biết sức chứa (-K)": "(s,S) cắt lượng đặt theo chỗ trống còn lại của kho",
    "Newsvendor dự báo (NV-DB)": "Dự báo rồi tối ưu: Newsvendor dùng dự báo LightGBM 1–3 ngày",
    "… (cùng mức PV) (=PV)": "Chỉnh tham số để đạt cùng mức phục vụ (fill) với IPPO rồi so chi phí",
}


def ten_day_du(tag):
    """Ten day du cua mot tag; tag co hau to _s<seed> thi them seed."""
    if tag in TEN_DAY_DU:
        return TEN_DAY_DU[tag]
    m = re.fullmatch(r"(.+)_s(\d+)", tag)
    if m and m.group(1) in TEN_DAY_DU:
        return f"{TEN_DAY_DU[m.group(1)]}, seed {m.group(2)}"
    return tag


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
    for i, p in enumerate(["main", "forecast_safety", "abl_ref", "abl_global", "abl_q3", "abl", "holdout",
                           "penalty", "zeroshot"]):
        if tag.startswith(p):
            return (i, tt)
    return (99, tt)


BASE = ["(s,S)", "(s,S) theo nhóm", "EOQ", "EOQ theo nhóm", "Newsvendor", "Newsvendor theo nhóm",
        "(s,S) biết sức chứa", "Newsvendor dự báo"]
NGAN = {"(s,S)": "(s,S)", "(s,S) theo nhóm": "(s,S)-N", "EOQ": "EOQ", "EOQ theo nhóm": "EOQ-N",
        "Newsvendor": "NV", "Newsvendor theo nhóm": "NV-N",
        "(s,S) biết sức chứa": "(s,S)-K", "Newsvendor dự báo": "NV-DB",
        "(s,S) (cùng mức PV)": "(s,S)=PV", "Newsvendor (cùng mức PV)": "NV=PV",
        "(s,S) biết sức chứa (cùng mức PV)": "(s,S)-K=PV", "Newsvendor dự báo (cùng mức PV)": "NV-DB=PV",
        "EOQ (cùng mức PV)": "EOQ=PV", "EOQ theo nhóm (cùng mức PV)": "EOQ-N=PV",
        "(s,S) theo nhóm (cùng mức PV)": "(s,S)-N=PV", "Newsvendor theo nhóm (cùng mức PV)": "NV-N=PV"}
ABL = ["abl_reward_cu", "abl_trunk", "abl_nocal", "abl_nowh", "abl_event", "abl_warm"]
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
        return f"rẻ hơn ({n}/{n} seed)"
    if dat_hon == n:
        return f"đắt hơn ({n}/{n} seed)"
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


def _bang_tra_ten(res, info):
    tags = set(info) | {p.stem[len("summary_"):] for p in res.glob("summary_*.json")}
    tags |= {"zeroshot_shared_supply_tight", "zeroshot_shared_supply_mild"}
    L = ["  a) Lần chạy (tag dùng trong tên file checkpoint và kết quả):"]
    L += [f"    {t:<30}{ten_day_du(t)}" for t in sorted(tags, key=_thu_tu)]
    L += ["", "  b) Chính sách (tên tắt trong các bảng):"] + [f"    {k:<30}{v}" for k, v in TEN_CHINH_SACH.items()]
    return L


def _so_voi_baseline(res, goc="main", nhan="IPPO"):
    """Cau 1 (goc='main') va Cau 4 (goc='forecast_safety'): moi seed so voi moi baseline, so truc
    tiep va o cung muc phuc vu."""
    d = _doc_json(res / f"multiseed_{goc}.json")
    if not d:
        return [f"  (chưa có - cần đánh giá {goc}_s* rồi chạy run_experiments.py tonghop)"]
    seeds = list(d["seeds"])
    sm = {s: _doc_json(res / f"summary_{goc}_s{s}.json") for s in seeds}
    s0 = sm[seeds[0]]
    L = [f"  {s0.get('n_pairs')} cặp kho–SKU, {s0['meta'].get('n_episodes')} lần đánh giá/chính sách"
         " (cùng hạt giống cho mọi chính sách).", ""]
    L += _bang_cp([(f"{nhan} seed {s}", _ippo(sm[s])) for s in seeds if sm[s]]
                  + [(r["policy"], r) for r in s0["summary"] if r["policy"] != "IPPO"])
    L += [f"  → {nhan} trung bình {len(seeds)} seed: {d['ippo_cost_mean']:,.0f} ± {d['ippo_cost_sd']:,.0f}"
          f" (SD giữa seed), fill {d['ippo_fill_mean']:.1%}", "",
          f"  {'Δ% ' + nhan + ' so với':<24}" + "".join(f"{'seed ' + s:>9}" for s in seeds) + f"{'TB':>8}   Kết luận"]
    for b in BASE:
        g = d["gap_vs_baseline"].get(b)
        if g:
            L.append(f"  {b:<24}" + "".join(_o_delta(d["seeds"][s]["per_baseline"].get(b)) for s in seeds)
                     + f"{g['mean']:+8.1f}   " + nhan + " "
                     + _ket_luan(g.get("n_seeds_re_hon", 0), g.get("n_seeds_dat_hon", 0), g["n_seeds"]))
    duoi = [r["policy"] for r in s0["summary"] if r["fill_rate_mean"] < MUC_DV]
    L.append(f"  Chính sách dưới fill {MUC_DV:.0%}: {', '.join(duoi) if duoi else 'không có'}")
    if goc == "main" and "Newsvendor dự báo" in d["gap_vs_baseline"]:
        L.append("  Lưu ý: Newsvendor dự báo dùng dự báo LightGBM mà mô hình chính không nhận (khác loại thông tin);"
                 " so sánh cùng loại thông tin ở Câu 4.")

    ss = {s: _doc_json(res / f"same_service_{goc}_s{s}.json") for s in seeds}
    ss = {s: m for s, m in ss.items() if m}
    if not ss:
        return L + ["", "  Cùng mức phục vụ: chưa có (chạy run.py eval_all)."]
    ten = [n for n in next(iter(ss.values()))["ket_qua"] if n != "IPPO"]
    L += ["", f"  Cùng mức phục vụ (lưới mịn, dải ±0,2 điểm %, tham số chọn trên val) - Δ% chi phí {nhan},"
          f" âm = {nhan} rẻ hơn:",
          f"    {'Seed':<8}{'Fill':>8}" + "".join(f"{NGAN.get(n, n):>10}" for n in ten)]
    for s, m in ss.items():
        o = []
        for n in ten:
            r = m["ket_qua"].get(n)
            o.append(f"{'—':>9} " if not r or r.get("khong_dat") else
                     f"{r['delta_ippo_percent']:+9.1f}{'~' if r['noi_suy'] else ' '}")
        L.append(f"    {s:<8}{m['target_fill']:>8.1%}" + "".join(o))
    L.append("    ~ = fill test lệch quá 0,2 điểm nên chi phí được nội suy trên đường biên;"
             " — = baseline không đạt mức fill này.")
    return L


def _cau2(res):
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
        L += ["", f"  b) Mặt hàng chưa gặp: {s0.get('n_pairs')} cặp = 6 SKU không có lúc huấn luyện holdout × 10 kho.",
              "     holdout: train 24 SKU, 1.000 tập | abl_ref: 30 SKU, 1.000 tập | main_s42: 30 SKU, 10.000 tập", ""]
        L += _bang_cp([(ten, _ippo(sh[t])) for t, ten in co]
                      + [(r["policy"], r) for r in s0["summary"] if r["policy"] != "IPPO"])
        L += ["", f"  {'Δ% so với baseline':<28}" + "".join(f"{NGAN[b]:>9}" for b in BASE)]
        for t, ten in co:
            per = sh[t]["stat"]["per_baseline"]
            L.append(f"  {ten:<28}" + "".join(_o_delta(per.get(b)) for b in BASE))
        for ref, ten in [("holdout_seen_ref", "đã gặp cùng 1.000 tập (abl_ref)"),
                         ("holdout_seen_main", "đã gặp 10.000 tập (main_s42)")]:
            k = _so_cap(res, "holdout_unseen", ref)
            if k:
                L.append(f"  Chưa gặp so với {ten}: {k[0]:+.1f}% (p = {k[1]:.1e})")
        duoi = [r["policy"] for r in s0["summary"] if r["policy"] != "IPPO" and r["fill_rate_mean"] < MUC_DV]
        if duoi:
            L.append(f"  [!] {len(duoi)} baseline dưới fill {MUC_DV:.0%} trên tập này → so chi phí KHÔNG cùng mức phục vụ.")
    return L or ["  (chưa có)"]


def _cau3(res):
    d = _doc_json(res / "rq3_multiseed.json")
    if not d:
        return ["  (chưa có - cần đánh giá abl_ref/abl_global/abl_q3 × mọi seed rồi chạy run_experiments.py tonghop)"]
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
                 " (dùng làm ngưỡng đọc ablation ở mục 6)")
    return L


def _cau4(res, info):
    L = _so_voi_baseline(res, "forecast_safety", "IPPO+DB")
    best = [f"seed {t[len('forecast_safety_s'):]}: tập {i['best']}" for t, i in sorted(info.items())
            if t.startswith("forecast_safety_s") and i.get("best") is not None]
    if best:
        L += ["", f"  Mô hình được chọn (chi phí val thấp nhất trong các lần fill val ≥ 85%): {', '.join(best)}.",
              "  Huấn luyện tiếp làm fill tăng và chi phí tăng (đánh đổi chi phí - fill, mục 6d). Cố định hệ số"
              " an toàn thì chính sách trùng Newsvendor dự báo: phần RL đóng góp là chọn hệ số theo trạng thái."]
    m = _doc_json(res / "perfect_demand.json")
    if m:
        L += ["", f"  Mốc biết trước nhu cầu (thay dự báo bằng nhu cầu thật, L = {m['L']}): chi phí {m['cost']:,.0f},"
              f" fill {m['fill']:.1%} - cho biết thông tin tương lai đáng giá bao nhiêu."]
    d = _doc_json(res / "forecast_metrics.json")
    if d:
        L += ["", "  Độ chính xác dự báo LightGBM trên test, RMSSE (càng nhỏ càng tốt; trong ngoặc: dự báo"
              f" ngây thơ theo mùa), tham số {d['params']}:"]
        for g, v in d["test"].items():
            L.append(f"    {g:<20}{v['rmsse']:>7.3f} ({v['rmsse_naive']:.3f})")
    return L


def _bang_kich_ban(rows):
    """[(ten, IPPO cp, IPPO fill, {baseline: muc per_baseline})] -> bang Δ%."""
    L = [f"    {'Kịch bản':<22}{'IPPO CP':>11}{'Fill':>7}" + "".join(f"{NGAN[b]:>9}" for b in BASE)]
    td = False
    for ten, cp, fill, per in rows:
        o = []
        for b in BASE:
            x = per.get(b)
            tat_dinh = bool(x) and x["ci95_low"] == x["ci95_high"]
            td |= tat_dinh
            o.append(f"{x['gap_percent']:+8.1f}~" if tat_dinh else _o_delta(x))
        L.append(f"    {ten:<22}{cp:>11,.0f}{fill:>7.1%}" + "".join(o))
    if td:
        L.append("    ~ = kịch bản tất định (các lần đánh giá giống hệt nhau) → không kiểm định được, chỉ đọc mức chênh.")
    return L


def _cau5(res):
    L = []
    r = _doc_json(res / "robustness_main.json")
    if r:
        L += [f"  a) Đổi sức chứa, lead time, mức cầu - mô hình chính không huấn luyện lại, {r.get('episodes')}"
              " lần đánh giá/kịch bản, Δ% IPPO:"]
        L += _bang_kich_ban([(k, v["IPPO"]["cost"], v["IPPO"]["fill"],
                              {b: v.get(b, {}).get("ippo_vs") for b in BASE}) for k, v in r["kich_ban"].items()])
    rows, roi_da = [], []
    for p in sorted(res.glob("shared_supply_*/summary_zeroshot_*.json")):
        s = _doc_json(p)
        if s:
            ip, ten = _ippo(s), p.stem[len("summary_zeroshot_"):]
            rows.append((ten, ip["total_cost_mean"], ip["fill_rate_mean"], s["stat"].get("per_baseline", {})))
            roi_da.append(f"    {ten:<22}" + "  ".join(f"{NGAN.get(r['policy'], r['policy'])} {r['bullwhip_mean']:.2f}"
                                                  for r in s["summary"] if "bullwhip_mean" in r))
    L += ["", "  b) Nguồn cung chung giữa các kho (tổng lượng giao mỗi ngày bị giới hạn) - mô hình chính không"
          " huấn luyện lại, Δ% IPPO:"]
    L += _bang_kich_ban(rows) if rows else ["    chưa có (chạy run.py eval_all)."]
    if roi_da:
        L += ["", "  Hệ số roi da (phương sai lượng đặt / phương sai nhu cầu toàn mạng, > 1 = khuếch đại dao động):"] + roi_da
    return L


def _ablation(res, info):
    ref = _doc_json(res / "summary_abl_ref.json")
    if not ref:
        return ["  (chưa có abl_ref)"]
    ss_ref = next(r for r in ref["summary"] if r["policy"] == "(s,S)")["total_cost_mean"]
    L = [f"  {'Lần chạy':<15}{'Thay đổi so với abl_ref':<46}{'Tổng CP':>11}{'Fill':>7}"
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
        L.append(f"  {t:<15}{ten_day_du(t)[:45]:<46}{ip['total_cost_mean']:>11,.0f}{ip['fill_rate_mean']:>7.1%}"
                 + (f"{k[0]:+10.1f}{_sao(k[1])}" if k else f"{'—':>11}")
                 + f" {_o_delta(s['stat']['per_baseline'].get('(s,S)'))}  {_hoi_tu(info.get(t))}"
                 + (" †" if khac else ""))
    L.append("  Chỉ 1 seed: chênh lệch nhỏ hơn mức dao động giữa seed của abl_ref (Câu 3) chưa đủ để kết luận.")
    if khac_mt:
        L.append("  † Môi trường đánh giá khác abl_ref (baseline cũng đổi) → đọc thêm cột Δ (s,S) cùng môi trường.")
    return L


def _bo_sung(res, info):
    L = []
    g = _doc_json(res / "regime_analysis_main.json")
    if g:
        ci = g["B1_cost_gap_ci95_block_bootstrap"]
        cols, nd = list(ci), g.get("n_days_per_group", {})
        L += ["  a) Theo giai đoạn nhu cầu (main_s42) - Δ% chi phí IPPO (* = CI 95% bootstrap khối 7 ngày không chứa 0):",
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
    s = _doc_json(res / "sensitivity_main.json")
    if s:
        L += ["", f"  c) Độ nhạy đơn giá ({len(s.get('grid', []))} bộ đơn giá lưu kho / thiếu hàng / đặt hàng):"]
        for b, v in s["summary"].items():
            L.append(f"    vs {b:<26}IPPO rẻ hơn ở {v['ippo_cheaper_in']:>2}/{v['of']} bộ"
                     f"  (Δ từ {v['gap_min']:+.1f}% đến {v['gap_max']:+.1f}%)")
    bien = _doc_json(res / "cost_service_frontier.json")
    if bien:
        L += ["", "  d) Đường đánh đổi chi phí - mức phục vụ: Δ% của từng điểm IPPO so với đường biên baseline"
              " tại cùng fill (— = ngoài đường biên):"]
        for t2, v in bien["ippo_so_voi_baseline_percent"].items():
            L.append(f"    {t2:<20} fill {bien['ippo'][t2]['fill']:.1%}  " + "  ".join(
                f"{NGAN.get(n, n)} {'—' if x is None else format(x, '+.1f')}" for n, x in v.items()))
    L += ["", "  e) Ablation khác (1.000 tập, seed 42, so với abl_ref):"] + _ablation(res, info)
    return L


def _huan_luyen(info):
    if not info:
        return ["  (chưa có lần chạy nào)"]
    L = [f"  {'Lần chạy':<20}{'Tập':>6}{'Phút':>6}{'Chọn tập':>9}{'CP val':>11}{'Fill val':>9}{'Đạt 85%':>8}"
         f"  {'Hội tụ':<27}{'Entropy':>8}{'EV':>6}"]
    for t in sorted(info, key=_thu_tu):
        i = info[t]
        mid = (f"{i['best']:>9}{i['cp_val']:>11,.0f}{i['fill_val']:>9.1%}{i['dat85']:>8}" if i["best"] is not None
               else f"{'—':>9}{'—':>11}{'—':>9}{'chưa':>8}")
        L.append(f"  {t:<20}{i['ep']:>6}{i['phut']:>6.0f}{mid}  {_hoi_tu(i):<27}{i['entropy']:>8.3f}{i['ev']:>6.2f}")
    L += ["  Chọn tập = mô hình được giữ lại (CP val thấp nhất trong các lần đánh giá val đạt fill ≥ 85%).",
          "  Hội tụ = CP val tốt nhất ở 20% số tập cuối so với trước đó (giảm > 2% = còn giảm).",
          "  Phút = phiên train cuối; Entropy, EV (explained variance) = TB 100 tập cuối."]
    chuan = info.get("main_s42", {}).get("state", {})
    for t, i in info.items():
        st = i.get("state", {})
        if chuan and st and (st.get("obs_per_pair"), st.get("n_pairs")) != (chuan.get("obs_per_pair"), chuan.get("n_pairs")):
            L.append(f"  {t}: {st.get('obs_per_pair')} chiều quan sát, {st.get('n_pairs')} cặp"
                     f" (mô hình chính: {chuan.get('obs_per_pair')} chiều, {chuan.get('n_pairs')} cặp)")
    return L


def _phu_luc(res):
    L = []
    for goc, nhan in (("main", "mô hình chính"), ("forecast_safety", "PS-IPPO + dự báo")):
        d = _doc_json(res / f"multiseed_{goc}.json")
        if not d:
            continue
        L += [f"  {nhan}:", f"  {'Seed':<6}{'So với':<22}{'Δ%':>8}{'Δ tuyệt đối [CI 95%]':>36}{'p (t)':>10}{'p (W)':>10}"
              f"{'d_z':>8}{'rẻ hơn':>9}"]
        for s, v in d["seeds"].items():
            for b in BASE:
                r = v["per_baseline"].get(b)
                if r:
                    ci = f"{r['mean_diff']:+,.0f} [{r['ci95_low']:+,.0f}; {r['ci95_high']:+,.0f}]"
                    L.append(f"  {s:<6}{b:<22}{r['gap_percent']:+8.2f}{ci:>36}{r['p_paired']:>10.1e}"
                             f"{r['p_wilcoxon']:>10.1e}{r['d_z']:>+8.2f}{str(r['n_ippo_cheaper']) + '/' + str(r['n']):>9}")
        L.append("")
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
         "Fill = tỷ lệ nhu cầu được đáp ứng; mục tiêu nghiệp vụ ≥ 85%. Tên đầy đủ: mục 0."]
    for ten, f, a in [("0. BẢNG TRA TÊN", _bang_tra_ten, (res, info)),
                      ("1. CÂU HỎI 1 - MÔ HÌNH CHÍNH (không dự báo) SO VỚI CHÍNH SÁCH ĐỐI CHỨNG", _so_voi_baseline, (res,)),
                      ("2. CÂU HỎI 2 - QUY MÔ CẦU KHÁC NHAU VÀ MẶT HÀNG CHƯA GẶP", _cau2, (res,)),
                      ("3. CÂU HỎI 3 - THIẾT KẾ PHẦN THƯỞNG (1.000 tập × 3 seed, so với abl_ref cùng seed)", _cau3, (res,)),
                      ("4. CÂU HỎI 4 - PS-IPPO KẾT HỢP DỰ BÁO (hệ số an toàn trên nền dự báo, 3 seed)", _cau4, (res, info)),
                      ("5. CÂU HỎI 5 - ĐỔI ĐIỀU KIỆN VẬN HÀNH, KHÔNG HUẤN LUYỆN LẠI (zero-shot)", _cau5, (res,)),
                      ("6. PHÂN TÍCH BỔ SUNG CỦA MÔ HÌNH CHÍNH", _bo_sung, (res, info)),
                      ("7. HUẤN LUYỆN - MỌI LẦN CHẠY", _huan_luyen, (info,)),
                      ("PHỤ LỤC - CHI TIẾT KIỂM ĐỊNH THEO CẶP (30 lần đánh giá)", _phu_luc, (res,))]:
        L += ["", GACH, ten, GACH] + _an_toan(f, *a)

    out = res / "KET_QUA.txt"
    res.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    print("Da luu", ghi_ket_qua())
