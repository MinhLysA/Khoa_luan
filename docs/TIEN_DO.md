# Theo dõi tiến độ khóa luận (PS-IPPO tồn kho đa kho)

Cập nhật: 05/11/2026 (tối) · Chú thích: ✅ xong · 🟡 đang triển khai / chờ train · ⬜ chưa làm · ⚠️ cần tự kiểm tra

> **[25/09/2026] Quyết định thiết kế:** Bật `include_price_signal: true` → môi trường dùng **44 chiều** (thêm tín hiệu giảm giá [V3-4]). Cần cập nhật Chương 3 báo cáo (Bảng đối chiếu biến trạng thái, công thức obs\_per\_pair, mô tả đặc trưng giá) trước khi nộp.

> Từ 31/10 nhóm train local trên máy MinhLysA (RTX 3050); `checkpoints/`, `results/` chỉ có trên máy đó (không đưa lên git). Máy còn lại chỉ có `app/demo_data/` (mô hình 23/10) và `results/` cũ của đợt chạy thử 31/10, không dùng cho báo cáo.

## 0. Trạng thái 05/11 (tối) — đã xong toàn bộ số liệu, chuẩn bị bảo vệ

**Train:** xong 27/27 lần chạy. **Đánh giá:** `eval_all` xong 04/11; `nv_lai` (tinh chỉnh lại Newsvendor với lưới tỷ lệ tới hạn 0,30–0,99) xong 05/11 20:47, `KET_QUA.txt` sinh lại cùng lúc.

**Newsvendor sau `nv_lai`:** Newsvendor giữ CR 0,60; Newsvendor dự báo chọn CR 0,50 (trước 0,60, đúng cận dưới cũ); Newsvendor theo nhóm chọn 0,40 / 0,75 / 0,85 / 0,60. Không tham số nào còn nằm ở cận lưới.

**Kết quả đã chốt (KET_QUA.txt):**
- Câu 1, so trực tiếp, 3 seed: IPPO 2.055.442, fill 90,9%. Rẻ hơn (s,S) −1,7% và EOQ −17,8% (3/3 seed); đắt hơn NV theo nhóm +1,7%, (s,S) biết sức chứa +0,8%, NV dự báo +8,6% (khác loại thông tin).
- Câu 1, cùng mức phục vụ (lưới mịn), 3/3 seed: đối chứng cổ điển đắt hơn IPPO 7,3–16,3% (EOQ 22,3% và 29,2%; seed 1 EOQ không đạt mức fill). NV dự báo rẻ hơn IPPO 3,3–4,3%. Giữ ở 36/36 bộ đơn giá và mọi giai đoạn nhu cầu.
- Câu 4: 1.897.517, fill 86,7%. Rẻ hơn mọi chính sách cổ điển 6,2–24,1% (so trực tiếp) và 6,0–17,8% (cùng mức phục vụ). Ngang NV dự báo: +0,2% so trực tiếp (2 seed đắt hơn 0,4% nhưng fill cao hơn 1,6–1,8 điểm, 1 seed rẻ hơn dưới 0,1%); cùng mức phục vụ +0,3 / +0,2 / −0,2%.
- Câu 5, nguồn cung chung (zero-shot, đối chứng tinh chỉnh lại): rẻ hơn mọi chính sách cổ điển 1,1–15,3% (vừa) và 2,8–22,4% (chặt); khoảng cách với NV dự báo thu hẹp 8,6% → 6,1% → 3,5%.

**Đã cập nhật theo số sau `nv_lai`:** báo cáo C4, C5 (15 chỗ `[SO-MOI-NV]`, đã xóa comment), hình C4 (`ve_hinh_bao_cao.py`), PDF 109 trang biên dịch sạch; poster (`build_poster.py`, `ROOT` nay tự lấy theo vị trí script); slide bảo vệ (Slides artifact, 15 trang); `app/demo_data/`; `van_dap.md` Phần V câu 29 và Phần VI. Demo: nút "Từ chối phiếu" từ chối cả phiếu, có thông báo và trạng thái phiếu cạnh mã phiếu.

### Việc còn lại

| # | Việc | Ghi chú |
|---|---|---|
| 1 | Commit, push báo cáo, poster, `app/demo_data/`, `docs/` | Chưa commit |
| 2 | Chia sẻ slide cho Linh và cô | Slide đang ở chế độ riêng tư |
| 3 | Tập trình bày, đối chiếu `van_dap.md` Phần VI | Khoảng 15 phút cho 15 slide |

## 0a. Phương án rút gọn 01/11 (chờ giảng viên duyệt)

Để đủ thời gian train, nhóm chọn: mô hình chính và U5 chỉ **3 seed** (42, 1, 2); **bỏ** U3 (Critic theo kho), U4 (số mạng), C' (phạt Lagrange theo cặp), E2 (1.000 tác tử), E3 (độ nhạy siêu tham số) và train điều kiện vận hành. **Giữ** mô hình chính, U1, U2, U5, ablation, hold-out và mọi phép đánh giá không train. Câu 5 (điều kiện vận hành) chuyển thành đánh giá zero-shot mô hình chính dưới nguồn cung chung và sức chứa 3, 8 ngày. `phi_dv` chỉ giữ 5 lần chạy thử 1.000 tập (lựa chọn rút gọn, có thể đổi lại). Khối lượng khoảng 141.000 tập (khoảng 43 giờ GPU cỡ T4) thay vì 480.000 tập (khoảng 160 giờ). **Các dòng U3, U4, C', E2, E3 ở bảng mục 0 bên dưới là kế hoạch cũ, nay đã bỏ.**

| Việc | Trạng thái |
|---|---|
| Sửa code theo phương án (3 seed, gỡ nhóm đã bỏ, bước zero-shot) | ✅ |
| `run.py check` | ✅ 01/11: GPU RTX 3050 Laptop (cuda), đã cài `lightgbm`; thiếu `data/raw` (không chặn dự báo) |
| `run.py full --quick` trên code cuối | ⬜ lượt 01/11 dừng giữa chừng ở bước U5 theo chủ ý, chưa chạy trọn |
| Train thật theo phương án | ⬜ |
| Thay số vào C4, C5, abstract (các chỗ `[SO-MOI]`) | ✅ 05/11 |

## 0. Phương án 31/10: code đã xong, chờ chạy

Mọi tính năng mới nằm sau cờ, tắt mặc định; mô hình chính giữ nguyên 44 chiều và không bị ghi đè. 74 test pass (54 cũ + 20 mới). Hướng dẫn chạy: `../rl_inventory/README.md`.

| Việc | Code, báo cáo | Còn lại sau khi chạy |
|---|---|---|
| A1, A2, B6 | ✅ C3 (hình Actor–Critic, bảng công thức chi phí, hai đoạn kho/doanh số), C5 | — |
| A3 lịch, giá, mùa lễ | ✅ C4 mục mới, dùng số hiện có | Thêm so sánh có/không dự báo nếu làm U1 |
| A4 sốc ×2, ×3, ×4 | ✅ `regime_analysis.py` mặc định chạy thêm | Bảng C4 |
| A5 phiếu mua sắm | ✅ `app/pages_/don_dat_hang.py`, `core.py`, test | — |
| A6 đồng bộ số | ✅ số điền tay từ `KET_QUA.txt`, đối chiếu bằng script | — |
| B1 lưới mịn | ✅ `compare_same_service.py --grid fine` | Thay 25,7%, 17,3% ở C4, C5, abstract |
| B2 (s,S) biết sức chứa | ✅ `CapacityAwareSs`, tune, evaluate, common, test | Bảng, phương pháp đối chứng C4 |
| B3 fill theo nhóm | ✅ (04/11 đã xóa `service_by_group.py`) | — |
| B4 dự báo rồi tối ưu | ✅ `forecast_demand.py`, `ForecastNewsvendor` | Bảng C4 |
| B5 seed 3, 4, thống kê | ✅ `run_experiments.py` (04/11 đã xóa `multi_seed_stats.py`) | Đã đổi: chỉ 3 seed (bỏ seed 3, 4); mục thống kê C4; p-value xuống phụ lục |
| Gói C `phi_dv` | ✅ 5 config, `cost_service_frontier.py` | `train phi_thu` (chỉ chạy thử 1.000 tập); bỏ bước chọn 2 giá trị và train 10.000 tập (rút gọn) |
| Gói D tái lập | ✅ README, `.gitignore` (gỡ `graphify-out/` khỏi git), đánh dấu file lỗi thời | Release `v1.0-final` |
| U1, U2, E1 | Đã bỏ (04/11): U5 trả lời Câu 4; xóa config U1, U2 và `policy_structure.py` | — |
| ~~U4, U3, C', E2~~ | Đã bỏ khỏi đợt thực nghiệm (01/11). Code từng xong; không train, không báo cáo | — |
| C2, tài liệu tham khảo | ✅ thêm 8 nguồn, đã đối chiếu tác giả, trang, DOI; bảng so sánh có hàng chính sách đối chứng; sửa câu CD-PPO theo bài gốc | Maggiar 2025 là poster workshop NeurIPS, không viết là hội nghị chính |
| C5 hướng phát triển | ✅ bảng 9 hướng | Thêm mục X chưa làm kèm lý do; viết lại hạn chế theo kết quả mới |
| Lý thuyết và phương pháp C1–C4 | ✅ 31/10: thêm lý thuyết dự báo, CMDP, CTDE, biến thể mở rộng, hai đối thủ mới, giao thức đường biên, thống kê | — |
| Phần kết quả C4, C5, abstract | ✅ 05/11 | Đã điền số từ `KET_QUA.txt` |

---

## 1. Tổng quan nhanh

| Hạng mục | Trạng thái |
|---|---|
| Dữ liệu M5 + tiền xử lý (10 kho × 30 SKU = 300 cặp) | ✅ |
| Môi trường Gymnasium | ✅ |
| Thuật toán PS-IPPO | ✅ |
| 3 baseline (EOQ, (s,S), Newsvendor) + tune trên val | ✅ (Newsvendor tinh chỉnh lại với lưới CR 0,30–0,99, `run.py nv_lai` 05/11) |
| Script đánh giá / phân tích | ✅ `eval_all` xong 04/11, `nv_lai` xong 05/11 |
| Test tự động | ✅ 74/74 pass với **44 chiều** (31/10; thêm test cho phần mở rộng) |
| `run.py full` + `run_experiments.py` (train local) | ✅ |
| App Streamlit 4 trang + MCP prototype | ✅ |
| **Đợt train cuối** | ✅ 27/27 lần chạy (máy MinhLysA) |
| Báo cáo LaTeX (109 trang) | ✅ số 3 hạt giống và số Newsvendor sau `nv_lai`, biên dịch sạch |
| Poster | 🟡 04/11: `Report KLTN/Poster_KLTN.pptx` dựng trên khung Khoa CNTT (`Poster.pptx` là mẫu của đề tài khác), sinh lại bằng `build_poster.py`; 05/11 đã dựng lại theo số sau `nv_lai`, thêm ý Câu 5 ✅ |

---

## 2. Đã làm xong ✅

### Dữ liệu
- [x] Chọn 30 SKU phân tầng từ M5 Walmart, 10 cửa hàng, 1.941 ngày (`scripts/data_preprocessing.py`)
- [x] Chia 3 miền: train [0,1050) · val [1050,1450) · test [1450,1941)
- [x] Dữ liệu đã xử lý trong `data/processed/`; tiền xử lý lại khớp 100% bản GitHub

### Môi trường (`env/inventory_env.py`)
- [x] 300 tác tử vector hóa, quan sát **44 chiều** (bật `include_price_signal: true` từ 25/09/2026 — thêm tín hiệu giảm giá [V3-4]), 6 mức hành động
- [x] Lead time ngẫu nhiên 1–3 ngày, sức chứa riêng từng kho (5 ngày cầu)
- [x] Reward = −(chi phí vận hành + phạt SLA), chuẩn hóa theo cặp
- [x] Các cờ: `service_penalty_mode`, `reward_mode` (local/global), `obs_drop`, `include_event_lookahead`, `warm_start_history`, `sku_indices`, `test_full_horizon`
- [x] Sửa lỗi v2 / v3 / P0 / P2 / P3 (chi tiết trong `rl_inventory/pham_vi_khoa_luan.md` mục 9)

### Thuật toán (`agents/`)
- [x] PS-IPPO: Actor/Critic tách riêng, value norm, value clip, GAE theo cặp, chuẩn hóa advantage theo cặp
- [x] Chế độ `shared_trunk` cho ablation
- [x] Train trên GPU (tự quay về CPU), `--resume` từ checkpoint

### Baseline & đánh giá (`baselines/`, `scripts/`)
- [x] EOQ, (s,S), Newsvendor dùng cùng thông tin và cùng 6 mức đặt
- [x] Tune baseline trên val theo ràng buộc fill ≥ 85% (bộ chung **và** theo 4 nhóm quy mô cầu)
- [x] Đánh giá trọn 491 ngày test × 30 seed lead time, kiểm định theo cặp (paired t, Wilcoxon, d_z)
- [x] iso-service, regime (bootstrap khối 7 ngày), policy behavior, scale groups, sensitivity, robustness (zero-shot)
- [x] Tổng hợp đa hạt giống (`multiseed_main.json`), tổng hợp RQ3 theo seed (`rq3_multiseed.json`)
- [x] `generate_summary.py` → `TONG_HOP_SO_LIEU.txt` (04/11 đã xóa script)

### Hạ tầng chạy
- [x] `scripts/run_experiments.py` (train tự resume, eval toàn pipeline, status, chế độ nhiều phiên song song)
- [x] Bỏ notebook Colab (31/10): train local bằng `run.py full` và `run_experiments.py`
- [x] Configs ablation sinh bằng `configs/make_configs.py` (10 file)

### Demo
- [x] App Streamlit 4 trang: Tổng quan, Đề xuất đặt hàng, So sánh policy, What-if (chi tiết theo ngày từng cặp)
- [x] MCP prototype (5 tool, 1 resource, có `--selftest`)

### Kết quả sơ bộ (protocol cũ — chỉ để định hướng, **không dùng cho bản nộp**)
- [x] 3 seed × 1.000 episode: IPPO fill 90,5%, đắt hơn (s,S) 4,5–5,5%
- [x] Cùng mức phục vụ ~90%: IPPO rẻ hơn (s,S) 10,2%, Newsvendor 14,4%
- [x] Seed 42 train kéo dài ~4.083 ep: −0,5% so với (s,S), p = 0,050
- [x] Thực nghiệm B (giai đoạn nhu cầu, sốc cầu), hành vi policy, theo quy mô cầu, độ nhạy 36 bộ đơn giá
- [x] RQ3 sơ bộ (625 ep): tắt chuẩn hóa → chi phí ×3,2

### Báo cáo LaTeX (`Report KLTN/`)
- [x] C1–C5, phụ lục A (nguồn) và B (MCP), biên dịch sạch 91 trang
- [x] RQ1–RQ3 bản chốt; methodology 3 miền; thuật ngữ (local factored reward, Dec-POMDP, ràng buộc mềm)
- [x] Hình TikZ 3.1–3.3; bibliography IEEE đã sửa
- [x] Đáp ứng 26/27 yêu cầu giảng viên (mục MCP ở mức "tìm hiểu", Phụ lục B)
- [x] `van_dap.md`: trả lời 27 câu hỏi của hội đồng

---

## 3. Đang triển khai — chờ train full 🟡

**Kiểm tra trước khi train (29/09):**
- Chạy thử nhanh trên Colab T4 (commit `c6a92a6`): code, GPU, 44 chiều, lưu Drive đều chạy; ~1 giây/episode. Bước tune baseline của notebook test lỗi (IPython không thay `{TUNE_EPISODES}`) → **đã sửa**.
- Máy local: 44/44 test pass, config ablation đồng bộ.
- **Chạy thử toàn bộ pipeline (26 episode/lần):** 19/19 lần train OK (44 chiều; `abl_event` 46 chiều). `run_experiments.py eval --n_eval 2` chạy trọn 34/34 lệnh, không lỗi, sinh đủ `multiseed_main.json`, `rq3_multiseed.json`, hold-out. Số liệu chạy thử vô nghĩa (mô hình 26 episode chưa đặt hàng).
- Colab test nhanh lần 2 (commit `6437894`): tune baseline OK, 6 baseline fill ≥ 85%; IPPO 300 ep chưa hội tụ (tràn kho ~32% chi phí).
- Đã sửa thêm: `evaluate.py` dừng hẳn khi thiếu `baseline_params.json`; `run_experiments.py` ghi commit mỗi lần train vào `logs/phien_ban_code.txt`; `.gitignore` cho checkpoints/results/runs/pyc/LaTeX build.
- Mọi số liệu gộp vào **một file `results/KET_QUA.txt`** (RQ1, RQ3, từng mô hình: huấn luyện + đánh giá test), tự dựng lại sau mỗi lần train/đánh giá (`scripts/build_results.py`).

Chạy: `python scripts/run_experiments.py train main` → `train ablations` → `eval` (≈ 8–10 giờ GPU T4).

Đợt cuối đã chạy xong 19/19 lần (20/10), xem mục 1. Các biến thể và phương án rút gọn: mục 0a và 0.

Rủi ro đã biết: Newsvendor tune theo nhóm có thể rẻ hơn IPPO → kết luận RQ1 phải viết theo số liệu thật.

---

## 4. Chưa làm — sau khi có kết quả đợt cuối ⬜

### Cập nhật báo cáo cho 44 chiều (do bật include_price_signal)
- [x] Cập nhật Chương 3 §3.1.2: thêm đặc trưng tín hiệu giá vào Bảng đối chiếu biến trạng thái (29/09; kèm C2, Phụ lục B)
- [x] Sửa công thức `obs_per_pair` trong báo cáo (43 → 44)
- [x] Chạy lại test suite (`pytest`) để xác nhận 44 chiều pass (44/44)
- [x] Cập nhật dòng mô tả "44 chiều" ở TIEN_DO.md mục 2 sau khi test pass
- [x] Sinh lại 10 config ablation (trước đó vẫn `include_price_signal: false` → ablation sẽ train 43 chiều, lệch `abl_ref`)


### Nguyên tắc viết kết quả (29/09)
- Chỉ tin số liệu của chính khóa luận: mọi kết luận trỏ tới bảng/hình của khóa luận, kèm CI 95% và p.
- Nguồn tham khảo chỉ để đặt bối cảnh ("phù hợp với" / "khác với"), không làm bằng chứng; SSRN (Nam et al.) là preprint chưa bình duyệt → chỉ so phạm vi.
- IPPO không thắng baseline thì viết đúng như vậy. Không so hiệu năng/thời gian với công trình khác bài toán/phần cứng.
- Đã đánh dấu `% VIET LAI THEO SO LIEU CUOI` tại: C4 (dòng 15: CPU/thời gian vs Zhu & Wu; dòng 187: Zipkin; dòng 1028: "đủ để vượt chính sách cổ điển" mâu thuẫn số sơ bộ), C5 dòng 258.

### Báo cáo
- [ ] Pull kết quả, chép hình mới vào `Report KLTN/media/`
- [ ] Thay toàn bộ bảng/hình C4 bằng số `main_*`, `abl_*`, `holdout_*`, `multiseed_main.json`, `rq3_multiseed.json`
- [ ] Chuyển bảng kết quả C4 sang booktabs, sửa 4 bảng tràn lề
- [ ] Điền mục RQ3 (local vs global + chuẩn hóa, 3 seed) và hold-out
- [ ] Viết lại Abstract VN/EN (`abstract.tex:29` có comment `CAP NHAT SAU DOT CHAY CUOI`)
- [ ] Viết lại C5 §5.1–5.2; chuyển thí nghiệm đã chạy từ "hướng phát triển" sang C4 (`C5.tex:183` bảng "đang chờ huấn luyện")
- [ ] Xóa mọi câu "sơ bộ", "đang chờ train" (C4.tex dòng 328, 375, 388, 515, 529, 582–602, 1007; C5.tex 101, 183)
- [ ] Quyết định TODO `C3.tex:631`: có thêm cấu hình chi phí phụ (vd c_th = 3) hay không
- [ ] Đối chiếu mọi % trong Abstract/Poster/slide với bảng/hình
- [ ] Cập nhật số liệu Poster

### Kiểm tra
- [x] Tác giả bài SSRN (Nam et al.): đã đối chiếu trang SSRN 7068293 (05/11)
- [ ] Commit + push kết quả và báo cáo cuối

### Tùy chọn (nếu còn thời gian)
- [x] Val dùng cùng các cửa sổ ở mọi lần đánh giá (hạt giống cố định 90000 + i)
- [x] Ghi phiên bản code của từng lần chạy vào `logs/phien_ban_code.txt`
- [x] Bỏ lệnh `run.py multiseed` của giao thức cũ

---

## 5. Ngoài phạm vi (ghi vào "Hướng phát triển")
Xem bảng hướng phát triển ở C5 (9 hướng cần dữ liệu, hệ thống thật hoặc đổi lớp bài toán). Dự báo đã chuyển vào phạm vi thực hiện. Lagrange theo cặp, Critic theo kho, 1.000 tác tử, số mạng bỏ khỏi đợt thực nghiệm (phương án rút gọn 01/11), có thể nêu thành hướng phát triển.

---

## 6. Lưu ý file cũ
- `results/` không hậu tố, `TONG_HOP_SO_LIEU.txt`: sót lại từ chạy thử — **không dùng cho báo cáo**
- `best_model_seed42.pth`: bản seed 42 train tiếp ~4.083 ep (tag `seed42_resume`)
- `results_pre_p0/`, `checkpoints_pre_p0/`: trước bản P0, chỉ tham khảo

---

## 7. Chi tiết kết quả sơ bộ (protocol cũ — chỉ để định hướng, sẽ thay toàn bộ)

- **So thẳng (3 seed × 1.000 episode):** IPPO fill 90,5% nhưng chi phí cao hơn
  (s,S) 4,5–5,5% (paired p < 10⁻¹⁶ ở cả 3 seed; đắt hơn ở 30/30 episode).
- **Cùng mức phục vụ ~90% (seed 42):** IPPO rẻ hơn (s,S) 10,2%, Newsvendor 14,4%; EOQ không đạt.
- **Seed 42 train kéo dài (~4.083 ep):** chi phí 1.512.076 (fill 89,9%) so với
  (s,S) 1.520.337 (84,1%) → −0,5%, paired p = 0,050; cửa sổ cố định −0,2%, p = 0,51.
- **Thực nghiệm B:** ở cùng mức phục vụ, IPPO rẻ hơn (s,S) 12,9–17,6% ở cả 8 nhóm
  ngày (CI 95% đều dưới 0). So thẳng: đắt hơn ở ngày ổn định (+2,5%) và ngày thường
  (+4,2%); rẻ hơn ở ngày sự kiện (−9,6%) và cuối tuần (−9,7%); mùa lễ +8,7% nhưng
  không có ý nghĩa (chỉ 9 tuần). Sốc cầu ×1,5: IPPO tốn ít hơn 29–32%; sau sốc ×0,5
  baseline tụt còn 87–88%, IPPO giữ 90%.
- **Hành vi:** xác suất đặt giảm đơn điệu theo số ngày tồn (Spearman −0,42). Không
  dồn sát ngưỡng 85%, nhưng **39% cặp dưới 85%** do phạt SLA tỷ lệ D̄.
  Permutation importance: quy mô (+98%), tồn kho (+94%), lịch sử cầu (+50%); lịch
  và tín hiệu cấp kho chỉ khoảng 3%.
- **Theo quy mô cầu:** IPPO tốt hơn (s,S) ở nhóm cầu ≥ 2/ngày, kém ở nhóm < 2/ngày.
- **Độ nhạy đơn giá (36 bộ):** rẻ hơn (s,S) cùng mức phục vụ ở 36/36 bộ; so thẳng
  chỉ 22/36, phụ thuộc tỷ số c_th/c_lk.
- **RQ3 sơ bộ (dừng ở 625 ep):** tắt chuẩn hóa → chi phí gấp khoảng 3,2 lần, fill
  80,7% so với 90,5%, dù explained variance ≈ 0,997.
- **Baseline theo nhóm (chạy thử 1 episode):** EOQ 1,886M → 1,603M, (s,S) 1,629M →
  1,610M, Newsvendor 1,626M → 1,601M, fill ~85%.
- **Độ bền (chạy thử):** IPPO tốt hơn khi lead time dài và cầu +20%, kém hơn khi
  cầu −20% (+6,2% so với (s,S)).

## 8. Checklist báo cáo trước hội đồng

Đáp ứng 26/27 yêu cầu giảng viên; MCP ở mức "tìm hiểu" (prototype, Phụ lục B).

| Mục | Trạng thái |
|---|---|
| RQ1–RQ3 viết lại; RQ3 có thí nghiệm local vs global thật | ✅ |
| 3 miền + bảng vai trò; baseline tune trên val | ✅ |
| Đánh giá trọn quỹ đạo test, kiểm định theo cặp | ✅ protocol · ⏳ số liệu |
| Ngân sách thống nhất 3 × 10.000 | ✅ cài đặt · ⏳ chạy |
| Quan sát 44 chiều (thêm tín hiệu giảm giá) trong Chương 3 | ✅ (C4/C5 "27/43 chiều" sửa khi viết lại kết quả) |
| 85% là mục tiêu chính, ~90% là phụ | ✅ |
| Thuật ngữ: local factored reward, Dec-POMDP team objective, ràng buộc mềm, chi phí vận hành ≠ phạt | ✅ |
| Bibliography sạch (IEEE); DOI Zhu & Wu | ✅ |
| Tác giả bài SSRN (Nam et al.) | ✅ 05/11 |
| Phụ lục A dùng `\ref` tự động; MCP ở Phụ lục B | ✅ |
| booktabs cho mọi bảng | 🟡 bảng kết quả C4 chuyển khi sinh lại |
| Viết lại Abstract, C4, C5; xóa "sơ bộ"/"đang chờ train" | ⏳ sau đợt cuối |


---

## Góp ý GVHD họp 04/10/2026 (cập nhật 23/10/2026)

| Mục | Việc | Trạng thái |
|---|---|---|
| K1, K2 | Bảng 6+ thành phần RL, câu '300 tác tử, một π_θ' (C3) | ✅ |
| K3 | Lớp tổng hợp báo cáo (C3) + bảng tổng hợp theo cửa hàng/ngành trên demo | ✅ |
| K4 | Cột Nhóm + Mã hóa trong bảng biến trạng thái (C3) | ✅ |
| K6 | Áp dụng cho Việt Nam (C5, van_dap) | ✅ |
| K7 | Ngày lịch 3 miền + hình trục thời gian (C3), bảng (C4) | ✅ |
| K8, K9 | Phát biểu bài toán (C1); kết quả theo giai đoạn nhu cầu, không kết luận RL thắng mọi nơi (C4, C5) | ✅ |
| K10 | Không dùng số '40–50%'; dùng số KET_QUA.txt | ✅ |
| K11 | Đợt train cuối + cập nhật C4, C5, Abstract | ✅ |
| D1, D6 | Gọi là 'ứng dụng mô phỏng'; phạm vi dừng ở phiếu gửi phòng mua hàng | ✅ |
| D2, D3 | Demo chỉ suy luận, dùng `demo_data` | ✅ |
| D4 | Bộ lọc kho/ngành + bảng KPI tổng hợp (trang Kết quả, trang Đơn đặt hàng) | ✅ |
| D5 | Luồng phiếu: nháp → duyệt/từ chối, gom phiếu | ✅ |
| Poster | `Report KLTN/Poster_KLTN.pptx`, số sau `nv_lai` (05/11) | ✅ |
| K5 | Chuẩn bị số thời gian train/ablation để trả lời (có trong van_dap) | 🟡 |
