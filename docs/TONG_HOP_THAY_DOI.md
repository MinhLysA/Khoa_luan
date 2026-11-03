# Báo cáo thay đổi đề tài khóa luận: trước và sau

- **Đề tài:** Xây dựng mô hình học tăng cường tối ưu hóa chính sách đặt hàng lại trong quản lý tồn kho đa kho
- **Sinh viên:** Lê Minh Lý (52300220), Sử Thị Yến Linh (52300218)
- **Giảng viên hướng dẫn:** TS. Hồ Thị Linh
- **Cập nhật:** 01/11/2026. "Trước" là bản đến commit `fcd1673`, "sau" là bản hiện tại.
- **Phương án rút gọn (01/11/2026):** nhóm chọn 3 hạt giống và bỏ sáu nhóm thí nghiệm để đủ thời gian train. **Chưa được giảng viên duyệt**, xem mục 1, 4 và 7. Lệnh làm theo phương án: `PROMPT_PHUONG_AN_3SEED.md`.

## 1. Tóm tắt

- **Đã đổi hướng câu hỏi chính.** Từ "IPPO có tốt hơn chính sách cổ điển không" sang "so với dự báo rồi tối ưu, học tăng cường thêm được gì và trong điều kiện nào".
- **Lý do đổi:** khi thêm chính sách đối chứng dùng dự báo nhu cầu, chính sách này rẻ hơn IPPO. Giữ luận điểm cũ sẽ không bảo vệ được trước hội đồng.
- **Đã bổ sung:** so sánh ở cùng mức phục vụ, chính sách đối chứng mạnh hơn, các biến thể có dự báo và thống kê chặt hơn. Nguồn cung chung giữa các kho giờ chỉ dùng để đánh giá zero-shot (không train lại).
- **Phương án rút gọn:** mô hình chính và U5 chạy 3 hạt giống (42, 1, 2), không phải 5. Bỏ U3 (Critic theo kho), U4 (số mạng), C' (phạt Lagrange theo cặp), E2 (1.000 tác tử), E3 (độ nhạy siêu tham số) và việc train điều kiện vận hành. Giữ U1, U2, U5, ablation, hold-out và mọi phép đánh giá không cần train. Khối lượng train giảm từ khoảng 480.000 xuống khoảng 141.000 tập (xem mục 4).
- **Trạng thái:** lý thuyết, phương pháp và code đã xong. **Kết quả cuối chưa có**, mọi số trong tài liệu này là số sơ bộ của một hạt giống (seed 42), 10.000 tập, chỉ đánh giá lại trên mô hình đã có.
- **Kết quả sơ bộ:** ở cùng mức phục vụ, IPPO rẻ hơn mọi chính sách cổ điển từ 7,3% đến 22,3%, nhưng đắt hơn chính sách dự báo 4,3%. Kết hợp IPPO với dự báo rút khoảng cách còn 1,0%, chưa vượt được.

## 2. Trước và sau khi thay đổi

| Nội dung | Trước | Sau |
|---|---|---|
| Câu hỏi nghiên cứu | 3 câu, câu chính hỏi IPPO có rẻ hơn ở mức 85% không | 5 câu. Câu 1 so với đối chứng có cùng thông tin, ở cùng mức phục vụ. Câu 2 chỉ còn quy mô nhu cầu và mặt hàng chưa gặp. Thêm Câu 4 về dự báo và cấu trúc hành động. Câu 5 là độ bền khi đổi điều kiện vận hành, đánh giá zero-shot |
| Chính sách đối chứng | EOQ, (s,S), Newsvendor | Thêm (s,S) biết sức chứa, Newsvendor dùng dự báo LightGBM, mốc biết trước nhu cầu |
| Cách so sánh | Tổng chi phí, mỗi chính sách ở một mức phục vụ khác nhau; lưới tham số thô | Thêm so sánh ở cùng mức phục vụ trên đường biên chi phí–mức phục vụ; lưới mịn; chỉ chọn tham số trên miền xác thực |
| Tính đa kho | Các kho chỉ chia sức chứa nội bộ, không ảnh hưởng lẫn nhau | Thêm điều kiện nguồn cung chung: mỗi mặt hàng có trần cung ứng toàn mạng, thiếu thì mọi kho bị cắt theo tỷ lệ. Chỉ **đánh giá zero-shot** mô hình chính, không train lại trong điều kiện này |
| Mô hình | Một mô hình PS-IPPO | Giữ mô hình chính làm mốc, thêm 3 biến thể (U1, U2, U5), mỗi biến thể đổi một thành phần. Kế hoạch đầu là 6 biến thể, nay bỏ U3, U4, C' |
| Sốc cầu | Chưa so với chính sách dự báo | Dự báo được tính lại trên nhu cầu đã đổi để so được |
| Thống kê | 3 hạt giống, p-value theo cặp | 3 hạt giống (phương án rút gọn; kế hoạch đầu là 5), IQM, bootstrap phân tầng theo khối 7 và 30 ngày. Với 3 seed khoảng tin cậy rộng hơn |
| Quy trình chạy | Chạy tay trên Colab | Train local bằng một lệnh, tự nối tiếp khi bị ngắt |
| Báo cáo | Kết luận IPPO rẻ hơn rõ rệt, dựa trên lưới thô | Trình bày theo từng câu hỏi nghiên cứu, kết luận theo số đo mới; các chỗ chờ số có đánh dấu `[SO-MOI]` |

## 3. Kết quả sơ bộ

Đo trên miền kiểm thử 491 ngày, 30 lần đánh giá, một hạt giống. Đây chưa phải số cuối.

**So sánh trực tiếp**, mỗi chính sách dùng cấu hình tinh chỉnh của mình:

| Chính sách | Tổng chi phí | Tỷ lệ đáp ứng | IPPO so với chính sách này |
|---|---:|---:|---:|
| IPPO | 2.052.496 | 90,65% | — |
| Newsvendor dự báo | 1.899.732 | 88,1% | đắt hơn 8,0% |
| Newsvendor theo nhóm | 2.022.942 | 85,8% | đắt hơn 1,5% |
| (s,S) theo nhóm | 2.048.883 | 85,9% | ngang, không có ý nghĩa thống kê |
| Newsvendor | 2.062.190 | 87,4% | rẻ hơn 0,5% |
| EOQ theo nhóm | 2.074.745 | 85,8% | rẻ hơn 1,1% |
| (s,S) | 2.091.665 | 86,2% | rẻ hơn 1,9% |
| (s,S) biết sức chứa | 2.153.173 | 86,1% | rẻ hơn 4,7% |
| EOQ | 2.500.198 | 89,5% | rẻ hơn 17,9% |
| Biết trước nhu cầu (mốc lý tưởng) | 1.358.864 | 93,6% | đắt hơn 51% |

**So ở cùng mức phục vụ 90,65%**, đọc trên đường biên. Đây là phép so sánh công bằng hơn vì IPPO đang giữ mức phục vụ cao hơn các chính sách khác:

| Chính sách | Tổng chi phí | IPPO so với chính sách này |
|---|---:|---:|
| EOQ | 2.640.010 | rẻ hơn 22,3% |
| (s,S) | 2.366.664 | rẻ hơn 13,3% |
| EOQ theo nhóm | 2.275.502 | rẻ hơn 9,8% |
| (s,S) biết sức chứa (lưới rút gọn) | 2.260.819 | rẻ hơn 9,2% |
| Newsvendor theo nhóm | 2.243.057 | rẻ hơn 8,5% |
| Newsvendor | 2.230.265 | rẻ hơn 8,0% |
| (s,S) theo nhóm | 2.213.800 | rẻ hơn 7,3% |
| Newsvendor dự báo | 1.968.561 | đắt hơn 4,3% |

**Tách theo nhóm nhu cầu** (chi phí / tỷ lệ đáp ứng):

| Nhóm nhu cầu | IPPO | Newsvendor dự báo |
|---|---:|---:|
| Rất thấp (dưới 0,5) | 248.587 / 74,2% | 254.448 / 78,2% |
| Thấp (0,5–2) | 470.140 / 83,2% | 443.355 / 85,3% |
| Trung bình (2–10) | 485.451 / 89,5% | 445.004 / 86,7% |
| Cao (từ 10) | 848.318 / 95,8% | 756.926 / 91,1% |

**Kết hợp IPPO và dự báo.** Mỗi nhóm nhu cầu dùng IPPO hoặc một cấu hình dự báo, chọn trên miền xác thực, không cần huấn luyện lại. Tại mức 90,65%:

| Chính sách | Tổng chi phí | So với Newsvendor dự báo |
|---|---:|---:|
| IPPO | 2.052.496 | đắt hơn 4,3% |
| Kết hợp (IPPO cho nhóm cao, dự báo cho ba nhóm còn lại) | 1.987.598 | đắt hơn 1,0% |
| Newsvendor dự báo theo nhóm | 1.965.309 | rẻ hơn 0,2% |
| Newsvendor dự báo | 1.968.561 | — |

**Nhận định từ số sơ bộ:**

- IPPO vượt mọi chính sách cổ điển khi so công bằng, kể cả bản tinh chỉnh theo nhóm và bản biết sức chứa.
- Thông tin về nhu cầu tương lai là phần giá trị nhất. Dự báo LightGBM có RMSSE 0,87 so với 1,19 của cách lấy nhu cầu tuần trước, và mốc biết trước nhu cầu rẻ hơn IPPO khoảng một phần ba.
- IPPO chưa thêm được lợi thế chi phí so với dự báo. Ở nhóm bán chậm, đặt hàng theo dự báo là đủ. Nhóm bán nhanh là nơi IPPO tiến gần nhất.
- Biến thể hứa hẹn nhất là cho tác tử chọn hệ số an toàn trên nền dự báo: chạy thử 600 tập chỉ kém Newsvendor dự báo 0,7% trên miền xác thực. Chưa chắc vượt được ở 10.000 tập.

**Kiểm tra đi kèm:** không phát hiện rò rỉ dữ liệu; làm tròn lượng đặt về 6 mức không phải nguyên nhân chênh lệch (ảnh hưởng tối đa khoảng 1,6%); code hiện tại tái lập đúng số của mô hình cũ.

## 4. Thí nghiệm sẽ chạy

Theo phương án rút gọn 01/11/2026 (chờ giảng viên duyệt):

| Câu hỏi | Thí nghiệm | Mục đích |
|---|---|---|
| 1 | Mô hình chính, 3 hạt giống | Có số cuối, có khoảng tin cậy (rộng hơn bản 5 seed) |
| 2 | Theo quy mô nhu cầu, mặt hàng chưa gặp (hold-out) | Kiểm tra chia sẻ tham số. Bỏ phần 1.000 tác tử (E2) |
| 3 | Phần thưởng toàn cục, bỏ chuẩn hóa, 3 hạt giống | Vai trò của thiết kế phần thưởng |
| 4 | 3 biến thể: U1 (quan sát có dự báo), U2 (hệ số an toàn của (s,S)), U5 (hệ số an toàn trên nền dự báo), mỗi biến thể 3 hạt giống | Dự báo và cấu trúc hành động có giúp không |
| 5 | **Zero-shot, không train:** mô hình chính đánh giá dưới nguồn cung chung ρ = 2 và 3, sức chứa 3 và 8 ngày; giữ phần độ bền zero-shot sẵn có (sức chứa, lead time, nhu cầu ±20%) | Lợi thế thay đổi thế nào khi điều kiện vận hành đổi. Đây là giới hạn: mô hình chưa từng được huấn luyện trong các điều kiện này |

**Đã bỏ so với kế hoạch đầu:** U3 (Critic theo kho), U4 (1, 4, 10, 300 mạng), C' (phạt Lagrange theo cặp), E2 (1.000 tác tử), E3 (độ nhạy siêu tham số) và train điều kiện vận hành.

**Lựa chọn rút gọn có thể đổi lại (`phi_dv`):** chỉ giữ 5 lần chạy thử 1.000 tập để làm bằng chứng, bỏ giai đoạn train lại hai giá trị đã chọn × 3 hạt giống (khoảng 60.000 tập, 18 giờ T4). Đây là lựa chọn của nhóm khi rút gọn, không phải yêu cầu của giảng viên. Nếu muốn khôi phục thì thêm lại giai đoạn này vào `train_full`.

Hai mức ρ chọn bằng cách chạy thử chính sách có sẵn: ρ = 3 làm tỷ lệ đáp ứng giảm khoảng 2 điểm, ρ = 2 giảm khoảng 4 điểm, ρ = 1,5 giảm gần 6 điểm.

Khối lượng (giờ T4 tính theo 1,1 giây mỗi tập):

| Phương án | Số tập | GPU cỡ T4 | CPU 4 nhân |
|---|---:|---:|---:|
| Kế hoạch đầu (5 seed, đủ các nhóm) | 480.000 | khoảng 160 giờ | khoảng 320 giờ |
| Rút gọn, bỏ train lại `phi_dv` (mặc định) | khoảng 141.000 | khoảng 43 giờ | khoảng 86 giờ |
| Rút gọn, giữ train lại `phi_dv` | khoảng 201.000 | khoảng 61 giờ | khoảng 122 giờ |

Chưa kể phần quét lưới chính sách đối chứng và các phép đánh giá. Cận dưới khoảng 41 giờ (không chạy thử `phi_dv`), cận trên khoảng 61 giờ (giữ train lại `phi_dv`).

## 5. Hạn chế và rủi ro

- **Chưa có kết quả cuối.** Số hiện tại của một hạt giống. Trung bình 3 hạt giống và kết quả loại trừ trong báo cáo cũ không còn file gốc, phải chạy lại.
- **IPPO có thể không vượt được chính sách dự báo.** Khi đó kết luận là dự báo rồi tối ưu đã đủ tốt trong bối cảnh này, học tăng cường không thêm được về chi phí. Kết luận này vẫn có giá trị vì được so sánh công bằng.
- **Nguồn cung chung có thể làm IPPO tệ đi,** vì cách chia theo tỷ lệ tạo động cơ đặt dư. Nhóm đo thêm hệ số roi da để phân tích hiện tượng này. Theo phương án rút gọn, mô hình chính chỉ được đánh giá zero-shot trong điều kiện này, nên kết quả xấu không chứng minh được IPPO không học được nếu train lại.
- **GPU:** `run.py check` ngày 01/11 báo máy hiện tại có NVIDIA RTX 3050 Laptop và torch chạy trên cuda (ghi chú cũ "chỉ có CPU" đã lỗi thời). Chưa đo tốc độ thật, các ước tính giờ vẫn tính theo T4.
- **Toàn bộ dây chuyền chưa chạy trọn trên bản code cuối.** Sẽ chạy thử nhanh trước khi train.
- **Phương án rút gọn có giá phải trả:**
  - Mất Câu 5 dạng train, mất train nguồn cung chung, mất 1.000 tác tử, mất U3, U4, C', E3.
  - Nguồn cung chung là thí nghiệm duy nhất làm các kho ảnh hưởng lẫn nhau. Mô hình chính chỉ chia sức chứa nội bộ, nên hội đồng có thể hỏi "đa kho ở chỗ nào". Câu trả lời dựa vào đánh giá zero-shot và tín hiệu cấp kho trong quan sát (xem `van_dap.md`).
  - 3 hạt giống làm khoảng tin cậy và IQM kém chặt hơn 5 hạt giống. Giao thức cũ cũng dùng 3 hạt giống.
  - Số "6 biến thể" của kế hoạch đầu giả định gồm U1–U5 và C'. Nếu cách đếm khác thì số biến thể còn lại có thể khác 3.
- **Phạm vi còn giới hạn:** chưa xét mạng đa cấp và điều chuyển hàng giữa các kho, dữ liệu M5 là doanh số chứ không phải nhu cầu thật, sức chứa và đơn giá chi phí là giả định.

## 6. Kế hoạch tiếp theo

1. Xin ý kiến giảng viên về phương án rút gọn ở mục 7.
2. Sửa code theo phương án rút gọn (`PROMPT_PHUONG_AN_3SEED.md`), chạy thử nhanh cả dây chuyền.
3. Train theo thứ tự: mô hình chính (3 seed) → U5 → U1 → U2 → chạy thử `phi_dv` → ablation. Bị ngắt thì chạy lại, tự nối tiếp.
4. Đánh giá zero-shot dưới nguồn cung chung và sức chứa 3, 8 ngày.
5. Thay số vào báo cáo, viết phần kết quả Câu 4 và 5, chốt kết luận.
6. Hoàn thiện ứng dụng demo, poster, tài liệu vấn đáp. Hạn nộp 23/01/2027.

## 7. Điểm cần xin ý kiến giảng viên

- Đồng ý đổi câu hỏi chính sang "so với dự báo rồi tối ưu, học tăng cường thêm được gì" không?
- Phương án rút gọn có chấp nhận được không: mô hình chính và U5 chỉ 3 hạt giống; bỏ U3 (Critic theo kho), U4 (số mạng), C' (phạt theo cặp), E2 (1.000 tác tử), E3 (độ nhạy siêu tham số) và train điều kiện vận hành?
- Câu 5 chuyển thành đánh giá zero-shot (mô hình chính chạy dưới nguồn cung chung và sức chứa khác, không train lại) có chấp nhận được không?
- Nếu bỏ train nguồn cung chung thì tính "đa kho" của đề tài có đủ không?
- Nếu IPPO không vượt được chính sách dự báo, kết luận theo hướng ở mục 5 có chấp nhận được không?
- Giả định nguồn cung chung (trần theo từng mặt hàng, chia theo tỷ lệ) dùng cho đánh giá zero-shot có phù hợp không?

---

# Phụ lục kỹ thuật

Phần này dành cho nhóm, ghi chi tiết để tra cứu.

## A. Đổi tên để nộp bài

`TIEN_DO.md` và phụ lục này vẫn dùng mã trong phương án (U1, U5...). Trong code, các tên đã đổi như sau.

| Mã trong phương án | Nội dung | Nhóm chạy | File cấu hình |
|---|---|---|---|
| U5 | Hệ số an toàn trên nền dự báo | `forecast_safety` | `forecast_safety_factor.yaml` |
| U1 | Quan sát có dự báo | `forecast_obs` | `forecast_observation.yaml` |
| U2 | Hệ số an toàn của (s,S) | `safety_factor` | `safety_factor.yaml` |
| Gói C | Hệ số phạt mức phục vụ (chỉ chạy thử 1.000 tập) | `penalty_trial` | `service_penalty_*.yaml` |

Theo phương án rút gọn 01/11, các mã **U3, U4, C', E2, E3** và train điều kiện vận hành bị bỏ khỏi đợt thực nghiệm (nhóm `wh_critic`, `networks`, `pair_penalty_trial`, `scale_1000`, `tuning`, `dieu_kien`).

| Script cũ | Script mới |
|---|---|
| `campaign.py` | `run_experiments.py` |
| `iso_service.py` | `compare_same_service.py` |
| `ket_qua.py` | `build_results.py` |
| `thong_ke_da_seed.py` | `multi_seed_stats.py` |
| `duong_bien.py` | `cost_service_frontier.py` |
| `fill_by_group.py` | `service_by_group.py` |

Cấu trúc thư mục: thêm `README.md` ở gốc; `TIEN_DO.md`, `TONG_HOP_THAY_DOI.md`, `van_dap.md`, `pham_vi_khoa_luan.md` và ba bài báo PDF chuyển vào `docs/`.

## B. Thay đổi về code

### Tính năng mới

Mọi tính năng nằm sau cờ, tắt mặc định, mô hình chính giữ 44 chiều. Các mã U3, U4, C', E2 đã bỏ khỏi đợt thực nghiệm (phương án rút gọn 01/11) nên không còn trong bảng này. Danh sách lỗi đã sửa bên dưới là lịch sử và vẫn nhắc tới C', E2.

| Mã | Nội dung | File chính |
|---|---|---|
| B1 | Lưới mịn, đường biên Pareto, chọn cùng mức phục vụ, nội suy | `scripts/compare_same_service.py`, `scripts/cost_service_frontier.py` |
| B1 | Đường biên bản theo nhóm bằng giá bóng của lượng thiếu hàng | `scripts/compare_same_service.py` |
| B2 | (s,S) biết sức chứa | `baselines/traditional_policies.py` |
| B3 | Tỷ lệ đáp ứng và chi phí vận hành theo nhóm cầu | `scripts/service_by_group.py` |
| B4 | Dự báo LightGBM ngoài mẫu, Newsvendor dự báo | `scripts/forecast_demand.py` |
| B5 | 3 seed (phương án rút gọn; code hỗ trợ nhiều hơn), IQM, bootstrap phân tầng | `scripts/multi_seed_stats.py` |
| E1 | Cấu trúc chính sách học được | `scripts/policy_structure.py` |
| U1 | Thêm dự báo vào quan sát (46 chiều) | `env/inventory_env.py` |
| U2 | Tác tử chọn hệ số an toàn của (s,S) | `agents/ppo_agent.py` |
| U5 | **Mới:** tác tử chọn hệ số an toàn trên nền dự báo | `baselines/traditional_policies.py`, `configs/forecast_safety_factor.yaml` |
| A5 | Trang phiếu mua sắm | `app/core.py`, `app/pages_/don_dat_hang.py` |
| Mới | Mốc biết trước nhu cầu | `scripts/perfect_demand_benchmark.py` |
| Mới | Nguồn cung chung (`supply_cover`), thêm 1 chiều quan sát; chỉ dùng để đánh giá zero-shot | `env/inventory_env.py` |
| Mới | Dự báo tính lại khi đổi nhu cầu (`du_bao_lai`) | `scripts/forecast_demand.py` |
| Mới | Chính sách kết hợp theo nhóm | `scripts/hybrid_policy.py` |
| Mới | Hệ số roi da | `scripts/evaluate.py` |
| Mới | 4 điều kiện vận hành (nguồn cung chung ρ = 2 và 3, sức chứa 3 và 8 ngày), chỉ đánh giá zero-shot, không train | `configs/make_configs.py`, `scripts/run_experiments.py` |

### Lỗi đã sửa

**Vòng rà 1 (10 lỗi):**

1. Hold-out làm `run.py full` dừng trước bước tổng hợp.
2. C' đo mức phục vụ trên miền xác thực bằng λ sai.
3. Baseline dự báo dùng dự báo nhu cầu gốc trong kịch bản sốc cầu.
4. E2 không xuất hiện trong `KET_QUA.txt`.
5. Tổng phiếu mua sắm tính cả dòng đã bỏ.
6. "Cùng mức phục vụ" dùng cấu hình nội suy và kết quả của lần chạy khác.
7. Đường biên baseline dùng lại bản cũ khi cấu hình đổi.
8. `evaluate.py --mode val --fixed_windows` lấy nhầm ngày của miền huấn luyện.
9. Chi phí theo nhóm tính cả phạt mức phục vụ.
10. `run.py check` lỗi khi thiếu torch.

**Vòng rà 2 (8 lỗi, 2 góp ý về hiệu năng và kiểu code chưa làm vì không ảnh hưởng kết quả):**

1. Trang demo lỗi khi chọn chính sách dự báo.
2. Terminal chết giữa chừng làm `run.py full` treo, hoặc đánh giá chạy khi terminal khác chưa xong.
3. Bị ngắt khi quét lưới làm đường biên cũ trông như còn đúng.
4. Chọn `phi_dv` chỉ dựa trên một lần đánh giá.
5. Lần đánh giá đầu dùng lưới thô, lần sau dùng lưới mịn, làm số không khớp.
6. `--quick` có thể ghi đè kết quả thật, nay chạy trên bản sao.
7. Dự báo dùng 2 ngày của miền kiểm thử khi chọn siêu tham số.
8. Ký tự BOM đầu `run.py`.

Ngoài ra, chạy thử trọn luồng phát hiện checkpoint C' không lưu λ, đã sửa và thêm chặn cứng.

**Vòng rà 3 (trước khi train):**

1. Lưới của (s,S) biết sức chứa bị chặn ở κ = 1, nên chính sách này dừng ở 86,4% và bị ghi là "không đạt" mức của IPPO. Lý do: hàng đang về được tính vào chỗ chiếm nhưng về dần, trong khi tồn kho bán dần. Lưới mới là κ ∈ {0,8; 1; 1,1; 1,2; 1,5}. Quét lại cho thấy chính sách đạt được mức này và IPPO vẫn rẻ hơn 9,2%.
2. Chính sách cùng mức phục vụ được nội suy thì không còn cấu hình thật, nên biến mất khỏi phân tích theo giai đoạn nhu cầu. Nay lấy cấu hình gần nhất trên đường biên, lệch quá 1 điểm phần trăm thì bỏ.
3. `export_tex_numbers.py` không quét README sau khi file này chuyển ra thư mục gốc.
4. Bỏ lệnh `run.py multiseed` của giao thức cũ để tránh chạy nhầm.
5. `perfect_demand_benchmark.py` import `baselines` trước khi `common` thiết lập đường dẫn, làm `run.py full` dừng sau khi đã train xong. Chạy thử đã bắt được lỗi này.

### Train và chạy

- `python run.py full`: chạy test, dự báo, train toàn bộ, đánh giá, ghi `results/KET_QUA.txt`.
- `python run.py full --quick`: thử cả dây chuyền trên bản sao, không đụng kết quả thật.
- `python run.py check`: báo GPU, `lightgbm`, dữ liệu.
- `logs/phien_ban_code.txt`: ghi commit của từng lần train.
- Mở nhiều terminal `python scripts/run_experiments.py train full` được, không giẫm chân nhau.

## C. Bỏ Colab

| File, nội dung | Trạng thái |
|---|---|
| `rl_inventory/Khoa_luan_Colab.ipynb` | Đã xóa |
| `rl_inventory/Khoa_luan_Colab_test_nhanh.ipynb` | Đã xóa |
| README, `run_experiments.py`, `train.py`, `evaluate.py`, `.gitignore` | Đã bỏ nội dung Colab, viết lại hướng dẫn local |
| `TIEN_DO.md` | Còn 4 dòng ghi lịch sử, không ảnh hưởng gì |

## D. Cách train

```powershell
cd rl_inventory
# Cài PyTorch bản CUDA trước: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
python run.py check          # dòng GPU phải là [OK]
python run.py full --quick   # thử cả dây chuyền
python run.py full           # train thật (phương án rút gọn: khoảng 43 giờ T4)
```

Bị ngắt thì chạy lại đúng lệnh, tự nối tiếp. Kết quả nằm ở `results/KET_QUA.txt`.

Thứ tự train (phương án rút gọn): mô hình chính → U5 → U1 → U2 → `phi_dv` (chỉ chạy thử) → ablation. Sau đó đánh giá, trong đó có phần zero-shot dưới nguồn cung chung và sức chứa 3, 8 ngày.

| Nhóm | Episode | GPU cỡ T4 | CPU 4 nhân |
|---|---:|---:|---:|
| Mô hình chính, 3 seed | 30.000 | 9 giờ | 18 giờ |
| U5, 3 seed | 30.000 | 9 giờ | 18 giờ |
| U1, U2, mỗi biến thể 3 seed | 60.000 | 18 giờ | 37 giờ |
| `phi_dv`, chỉ chạy thử 5 giá trị × 1.000 tập | 5.000 | 1,5 giờ | 3 giờ |
| Ablation, RQ3, hold-out | 16.000 | 5 giờ | 10 giờ |
| **Tổng (mặc định)** | **141.000** | **khoảng 43 giờ** | **khoảng 86 giờ** |
| Nếu giữ train lại `phi_dv` (2 giá trị × 3 seed) | +60.000 | +18 giờ | +37 giờ |
| **Tổng (giữ train lại `phi_dv`)** | **201.000** | **khoảng 61 giờ** | **khoảng 122 giờ** |

Giờ T4 tính theo 1,1 giây mỗi tập. So với kế hoạch đầu: bỏ siêu tham số (2 giờ), U4 (12 giờ), U3 (11 giờ), C' (10 giờ), E2 (9–25 giờ), điều kiện vận hành (36 giờ); mô hình chính và U5 xuống 3 seed (tiết kiệm 12 giờ).

## E. Chạy thử trên miền xác thực

So với Newsvendor dự báo ở cùng tỷ lệ đáp ứng:

| Lần chạy | Episode | Chi phí, tỷ lệ đáp ứng | Chênh so với Newsvendor dự báo |
|---|---:|---:|---:|
| U5 lưới rộng, `phi_dv` = 3 | 500 | 1.631.557 ở 91,09% | +5,0% |
| U5 lưới rộng, `phi_dv` = 1 | 500 | 1.553.522 ở 89,79% | +3,1% |
| U1 | 500 | 1.746.085 ở 90,60% | +14,0% |
| U5 lưới hẹp, entropy mặc định | 600 | 1.532.124 ở 89,77% | +1,7% |
| **U5 lưới hẹp, entropy thấp (đã chốt)** | 600 | 1.492.070 ở 88,85% | **+0,7%** |

Cấu hình đã chốt: `z` thuộc {0; 0,2; 0,4; 0,6; 0,8; 1,2}, hệ số entropy 0,005 giảm về 0,0005, `phi_dv` = 1. Chỉ chọn trên miền xác thực. Chưa chắc thắng ở 10.000 episode, nhưng chiều hướng tốt nhất trong các biến thể.

## F. Trạng thái kiểm tra trước khi train

- [x] Chốt cấu hình `forecast_safety_factor` trên miền xác thực.
- [x] Đường biên đầy đủ của seed 42, gồm bản theo nhóm.
- [x] Quét lại (s,S) biết sức chứa trên lưới κ mới. Số 2.260.819 đo trên 240 cấu hình, sẽ được thay bằng lưới đầy đủ khi train.
- [x] Chạy riêng `forecast_safety_factor` qua train, đánh giá, cùng mức phục vụ, phân tích giai đoạn.
- [x] `export_tex_numbers.py` đọc được định dạng kết quả thật.
- [x] `build_results.py` sinh được `KET_QUA.txt` từ kết quả thật của seed 42.
- [x] Mọi script trong `scripts/` khởi động được, tham số `run_experiments.py` truyền vào khớp với từng script.
- [x] 82 test pass, báo cáo 105 trang, không tràn lề, không tham chiếu lỗi.
- [ ] Dự báo tính lại mới có test trên dữ liệu giả, chưa chạy trên mô hình dự báo thật vì `forecast_model.txt` chỉ sinh ra khi chạy lại `forecast_demand.py`.
- [ ] Nguồn cung chung mới chạy với chính sách có sẵn. Theo phương án rút gọn IPPO không được train trong điều kiện này, chỉ đánh giá zero-shot mô hình chính; phần zero-shot này chưa chạy.
- [ ] Code đổi theo phương án rút gọn (3 seed, gỡ các nhóm đã bỏ, bước zero-shot): làm theo `PROMPT_PHUONG_AN_3SEED.md`, chưa kiểm tra.
- [ ] `run.py full --quick` chưa chạy trọn trên code cuối. Lượt gần nhất qua 121 lệnh rồi dừng ở lỗi số 5 của vòng rà 3. Phần đánh giá thí nghiệm mở rộng và tổng hợp sau chỗ đó mới được đọc code, chưa chạy nối tiếp. Lượt chạy lại đã dừng theo yêu cầu không train.

Bằng chứng số đo seed 42 nằm ở `docs/ket_qua_seed42_31-10/`.
