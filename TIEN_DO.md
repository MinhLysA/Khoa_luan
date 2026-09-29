# Theo dõi tiến độ khóa luận (PS-IPPO tồn kho đa kho)

Cập nhật: 29/09/2026 · Chú thích: ✅ xong · 🟡 đang triển khai / chờ train · ⬜ chưa làm · ⚠️ cần tự kiểm tra

> **[25/09/2026] Quyết định thiết kế:** Bật `include_price_signal: true` → môi trường dùng **44 chiều** (thêm tín hiệu giảm giá [V3-4]). Cần cập nhật Chương 3 báo cáo (Bảng đối chiếu biến trạng thái, công thức obs\_per\_pair, mô tả đặc trưng giá) trước khi nộp.

> Trạng thái đợt cuối lấy từ `python scripts/campaign.py status` **trên máy local** (0/19 lần chạy).
> Nếu đang chạy trên Colab thì tiến độ thật nằm ở `MyDrive/KLTN_final` — chạy lại lệnh status trên Colab để cập nhật bảng mục 3.

---

## 1. Tổng quan nhanh

| Hạng mục | Trạng thái |
|---|---|
| Dữ liệu M5 + tiền xử lý (10 kho × 30 SKU = 300 cặp) | ✅ |
| Môi trường Gymnasium | ✅ |
| Thuật toán PS-IPPO | ✅ |
| 3 baseline (EOQ, (s,S), Newsvendor) + tune trên val | ✅ code · 🟡 chạy lại trong đợt cuối |
| Script đánh giá / phân tích | ✅ code · 🟡 chờ số liệu đợt cuối |
| Test tự động | ✅ 44/44 pass với **44 chiều** (29/09; thêm test config ablation đồng bộ với config.yaml) |
| Notebook Colab + `campaign.py` | ✅ |
| App Streamlit 4 trang + MCP prototype | ✅ |
| **Đợt train cuối (3 main + 16 ablation/hold-out)** | 🟡 **0/19 — chưa chạy** |
| Báo cáo LaTeX (91 trang) | ✅ khung · 🟡 C4/C5/Abstract chờ số liệu cuối |
| Poster | ✅ có file · ⬜ cập nhật số liệu cuối |

---

## 2. Đã làm xong ✅

### Dữ liệu
- [x] Chọn 30 SKU phân tầng từ M5 Walmart, 10 cửa hàng, 1.941 ngày (`scripts/data_preprocessing.py`)
- [x] Chia 3 miền: train [0,1050) · val [1050,1450) · test [1450,1941)
- [x] Dữ liệu đã xử lý trong `data/processed/`; tiền xử lý trên Colab khớp 100% bản GitHub

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
- [x] `generate_summary.py` → `TONG_HOP_SO_LIEU.txt`

### Hạ tầng chạy
- [x] `scripts/campaign.py` (train tự resume, eval toàn pipeline, status, chế độ nhiều phiên song song)
- [x] `Khoa_luan_Colab.ipynb` 11 bước (29/09): code clone vào `/content`, dữ liệu M5 đọc từ `MyDrive/KLTN_data`, **chỉ kết quả** (checkpoints/results/runs/logs) lưu vào `MyDrive/KLTN_final`
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
- Chạy thử nhanh trên Colab T4 (commit `c6a92a6`): code, GPU, 44 chiều, lưu Drive đều chạy; ~1 giây/episode. Bước tune baseline của notebook test lỗi (IPython không thay `{TUNE_EPISODES}`) → **đã sửa**, cần push `Khoa_luan_Colab_test_nhanh.ipynb`.
- Máy local: 44/44 test pass, config ablation đồng bộ, cả 2 notebook thay biến đúng ở mọi dòng `!`. Chạy thử 26 episode: `main_s42`, `main_s1`, `main_s2`, `abl_ref` train OK (dừng giữa chừng, chưa chạy thử 15 lần còn lại và `campaign.py eval`).

Chạy: `python scripts/campaign.py train main` → `train ablations` → `eval` (≈ 8–10 giờ GPU T4).

| Lần chạy | Episode | Phục vụ | Tiến độ |
|---|---|---|---|
| `main_s42` | 4.000 | RQ1, RQ2 | ⬜ 0/4000 |
| `main_s1` | 4.000 | RQ1, RQ2 | ⬜ 0/4000 |
| `main_s2` | 4.000 | RQ1, RQ2 | ⬜ 0/4000 |
| `abl_ref`, `abl_ref_s1`, `abl_ref_s2` | 1.000 | mốc ablation | ⬜ 0/1000 × 3 |
| `abl_global`, `_s1`, `_s2` | 1.000 | RQ3 local vs global | ⬜ 0/1000 × 3 |
| `abl_q3`, `_s1`, `_s2` | 1.000 | RQ3 chuẩn hóa | ⬜ 0/1000 × 3 |
| `abl_reward_cu` | 1.000 | phạt SLA kiểu cũ | ⬜ 0/1000 |
| `abl_trunk` | 1.000 | tách Actor/Critic | ⬜ 0/1000 |
| `abl_nocal`, `abl_nowh` | 1.000 | biến dư thừa trong state | ⬜ 0/1000 × 2 |
| `abl_event` | 1.000 | mùa lễ | ⬜ 0/1000 |
| `abl_warm` | 1.000 | lịch sử rỗng đầu episode | ⬜ 0/1000 |
| `holdout` | 1.000 | RQ2 mở rộng (6 SKU chưa gặp) | ⬜ 0/1000 |
| **`campaign.py eval`** (tune → eval → iso → regime → behavior → quy mô → độ nhạy → độ bền → ablation → hold-out → multiseed → RQ3) | — | tất cả | ⬜ chờ train xong |

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
- [ ] Chạy lại `generate_summary.py` → `TONG_HOP_SO_LIEU.txt` (bản hiện tại là số cũ)
- [ ] Cập nhật số liệu Poster

### Kiểm tra
- [ ] ⚠️ Tác giả bài SSRN (Nam et al.) — tự kiểm tra trên trang SSRN
- [ ] Commit + push kết quả và báo cáo cuối

### Tùy chọn (nếu còn thời gian)
- [ ] Val trên toàn quỹ đạo cố định (hiện val vẫn rút cửa sổ 365 ngày)
- [ ] `run_metadata.json` lưu snapshot config mỗi lần chạy
- [ ] `run.py multiseed` vẫn mặc định 1.000 episode (đợt cuối dùng `campaign.py`, không ảnh hưởng)

---

## 5. Ngoài phạm vi (ghi vào "Hướng phát triển")
Lead time > 3 ngày, mục tiêu 80/90%, Lagrange theo cặp, chuyển hàng liên kho / multi-echelon, MAPPO, communication giữa tác tử, action liên tục, dự báo LSTM/Transformer, 3.049 SKU, triển khai thực tế, giao tiếp ngôn ngữ qua MCP.

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
| Ngân sách thống nhất 3 × 4.000 | ✅ cài đặt · ⏳ chạy |
| Quan sát 44 chiều (thêm tín hiệu giảm giá) trong Chương 3 | ✅ (C4/C5 "27/43 chiều" sửa khi viết lại kết quả) |
| 85% là mục tiêu chính, ~90% là phụ | ✅ |
| Thuật ngữ: local factored reward, Dec-POMDP team objective, ràng buộc mềm, chi phí vận hành ≠ phạt | ✅ |
| Bibliography sạch (IEEE); DOI Zhu & Wu | ✅ |
| Tác giả bài SSRN (Nam et al.) | ⚠️ tự kiểm tra |
| Phụ lục A dùng `\ref` tự động; MCP ở Phụ lục B | ✅ |
| booktabs cho mọi bảng | 🟡 bảng kết quả C4 chuyển khi sinh lại |
| Viết lại Abstract, C4, C5; xóa "sơ bộ"/"đang chờ train" | ⏳ sau đợt cuối |
