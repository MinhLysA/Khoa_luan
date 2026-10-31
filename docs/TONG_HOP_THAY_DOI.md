# Tổng hợp thay đổi đề tài PS-IPPO

Cập nhật: 31/10/2026, 12:05. "Trước" là trạng thái đầu phiên làm việc ngày 31/10 (commit `fcd1673`). "Hiện tại" là bản đang nằm trên máy, **chưa commit**.

## 1. Tóm tắt

- **Hướng đề tài đổi.** Trên mô hình seed 42, ở cùng mức phục vụ, IPPO rẻ hơn mọi chính sách cổ điển 7,3–22,3% nhưng đắt hơn chính sách dự báo 4,3%. Luận điểm mới: "So với dự báo rồi tối ưu, học tăng cường thêm được gì, trong điều kiện nào?".
- **Thêm biến thể U5** (tác tử chọn hệ số an toàn trên nền dự báo). Đây là ứng viên tốt nhất để có kết quả dương thật.
- **Code sẵn sàng train:** 77 test pass, đã rà hai vòng và sửa 18 lỗi.
- **Train local:** đã bỏ hai notebook Colab, có một lệnh duy nhất `python run.py full`.
- **Báo cáo:** lý thuyết, phương pháp, bố cục đã xong (102 trang). Kết luận ở C4, C5, tóm tắt đã viết lại theo số seed 42; số cuối chờ train xong.
- **Chưa có:** số liệu cuối, và máy hiện tại không có GPU NVIDIA (xem mục 9).

## 2. Trước và hiện tại

| Hạng mục | Trước | Hiện tại |
|---|---|---|
| Câu hỏi nghiên cứu | 3 câu; Câu 1 hỏi IPPO có rẻ hơn ở mức 85% không | 4 câu; Câu 1 so với đối chứng có cùng thông tin, ở cùng mức phục vụ; Câu 4 về dự báo và cấu trúc hành động |
| Chính sách đối chứng | EOQ, (s,S), Newsvendor | Thêm (s,S) biết sức chứa, Newsvendor dự báo, mốc biết trước nhu cầu |
| So sánh cùng mức phục vụ | Lưới thô, ngưỡng lấy từ miền kiểm thử, chỉ bản một bộ tham số | Lưới mịn, chọn trên miền xác thực, dải ±0,2 điểm, có đường biên cho bản theo nhóm |
| Thống kê | 3 seed, p-value theo cặp | 5 seed, IQM, bootstrap phân tầng theo khối 7 và 30 ngày |
| Mô hình | Chỉ mô hình chính | Thêm U1, U2, U3, U4, U5, C', E2, mỗi biến thể đổi một thành phần |
| Cách chạy | Notebook Colab, chạy tay từng bước | `python run.py full`, tự chọn tham số trên miền xác thực, tự nối tiếp khi bị ngắt |
| Kiểm tra trước khi train | Chạy thử thủ công | `python run.py full --quick` trên bản sao |
| Báo cáo | Kết luận IPPO "rẻ hơn rõ rệt" dựa trên lưới thô | Kết luận theo số đo mới: ngang chính sách cổ điển, chưa vượt chính sách dự báo; chỗ cần đổi số có đánh dấu `[SO-MOI]` |
| Trình bày C4 | Theo loại thí nghiệm | Theo từng câu hỏi nghiên cứu |

## 2a. Khai thác thêm đề tài (đợt 31/10/2026)

Bốn chỗ còn thiếu và cách xử lý:

| Chỗ thiếu | Trước | Hiện tại |
|---|---|---|
| "Đa kho" chưa thật sự đa kho | Kho chỉ chia sức chứa nội bộ, các kho không ảnh hưởng nhau | Thêm nguồn cung chung theo từng mặt hàng (`supply_cover`), chia theo tỷ lệ đơn, tác tử thấy tỷ lệ được cấp hôm trước |
| Chưa so với dự báo khi sốc cầu | Newsvendor dự báo bị loại khỏi sốc cầu và kịch bản đổi mức cầu | `du_bao_lai`: giữ mô hình dự báo, tính lại đặc trưng trên nhu cầu đã đổi |
| Chưa biết IPPO thắng ở đâu | Chỉ có fill theo nhóm | `scripts/hybrid_policy.py`: tách chi phí theo nhóm, kết hợp hai chính sách, chọn trên val |
| Chưa có "trong điều kiện nào" | Độ nhạy không học lại | 12 lần train mới: nguồn cung ρ = 2 và 3, sức chứa 3 và 8 ngày, mỗi điều kiện có mô hình chính và bản nền dự báo |

Thêm hệ số roi da (`bullwhip`) vào `evaluate.py`, Câu hỏi 5 vào C1, mô hình nguồn cung vào C3, thiết kế và kết quả kết hợp vào C4, sửa hạn chế và hướng phát triển ở C5.

Số đo seed 42 (chỉ đánh giá, không train, 30 lần):

| Nhóm nhu cầu | IPPO | Newsvendor dự báo |
|---|---:|---:|
| Rất thấp | 248.587 / 74,2% | 254.448 / 78,2% |
| Thấp | 470.140 / 83,2% | 443.355 / 85,3% |
| Trung bình | 485.451 / 89,5% | 445.004 / 86,7% |
| Cao | 848.318 / 95,8% | 756.926 / 91,1% |

Chính sách kết hợp (IPPO cho nhóm cao, dự báo cho ba nhóm còn lại): 2.005.411 / 90,70%. Rẻ hơn IPPO 2,3% ở cả 30/30 lần đánh giá. So với Newsvendor dự báo cùng mức phục vụ, khoảng cách còn 1,9% thay vì 4,3%. Chưa thắng được chính sách dự báo.

Mức ρ chọn bằng cách chạy thử chính sách đã có, chưa tinh chỉnh lại: ρ = 3 giảm fill khoảng 2 điểm, ρ = 2 giảm khoảng 4 điểm, ρ = 1,5 giảm gần 6 điểm.

Không làm điều chuyển hàng ngang: chưa cần vì nguồn cung chung đã tạo ràng buộc giữa các kho, và luật điều chuyển sẽ áp dụng cho mọi chính sách nên không trả lời câu hỏi về học tăng cường.

## 2b. Đổi tên để nộp bài

Tài liệu này và `TIEN_DO.md` vẫn dùng mã trong phương án (U1, U5...). Trong code, các tên đã đổi như sau.

| Mã trong phương án | Nội dung | Nhóm chạy | File cấu hình |
|---|---|---|---|
| U5 | Hệ số an toàn trên nền dự báo | `forecast_safety` | `forecast_safety_factor.yaml` |
| U1 | Quan sát có dự báo | `forecast_obs` | `forecast_observation.yaml` |
| U2 | Hệ số an toàn của (s,S) | `safety_factor` | `safety_factor.yaml` |
| Gói C | Hệ số phạt mức phục vụ | `penalty_trial` | `service_penalty_*.yaml` |
| E3 | Độ nhạy siêu tham số | `tuning` | `tuning_*.yaml` |
| U4 | Số mạng | `networks` | `networks_*.yaml` |
| U3 | Critic theo kho | `wh_critic` | `warehouse_critic.yaml` |
| C' | Phạt Lagrange theo cặp | `pair_penalty_trial` | `pair_penalty_*.yaml` |
| E2 | 1.000 tác tử | `scale_1000` | `scale_1000_agents.yaml` |

| Script cũ | Script mới |
|---|---|
| `campaign.py` | `run_experiments.py` |
| `iso_service.py` | `compare_same_service.py` |
| `ket_qua.py` | `build_results.py` |
| `thong_ke_da_seed.py` | `multi_seed_stats.py` |
| `duong_bien.py` | `cost_service_frontier.py` |
| `fill_by_group.py` | `service_by_group.py` |

Cấu trúc thư mục: thêm `README.md` ở gốc; `TIEN_DO.md`, `TONG_HOP_THAY_DOI.md`, `van_dap.md`, `pham_vi_khoa_luan.md` và ba bài báo PDF chuyển vào `docs/`.

## 3. Kết quả đo mới

Đo trên mô hình 10.000 episode seed 42, 30 lần đánh giá, miền kiểm thử. Code hiện tại tái lập đúng số IPPO trong C4. **Đây là một hạt giống, chưa phải số cuối.**

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
| Biết trước nhu cầu | 1.358.864 | 93,6% | đắt hơn 51% |

**Ở đúng mức phục vụ của IPPO (90,65%), lưới mịn, đọc trên đường biên:**

| Chính sách | Tổng chi phí | IPPO so với chính sách này |
|---|---:|---:|
| EOQ | 2.640.010 | rẻ hơn 22,3% |
| (s,S) | 2.366.664 | rẻ hơn 13,3% |
| EOQ theo nhóm | 2.275.502 | rẻ hơn 9,8% |
| Newsvendor theo nhóm | 2.243.057 | rẻ hơn 8,5% |
| Newsvendor | 2.230.265 | rẻ hơn 8,0% |
| (s,S) theo nhóm | 2.213.800 | rẻ hơn 7,3% |
| Newsvendor dự báo | 1.968.561 | đắt hơn 4,3% |
| (s,S) biết sức chứa | 2.260.819 (lưới rút gọn, 240 cấu hình) | rẻ hơn 9,2% |

Số cũ trong C4 là 25,7% và 17,3% do lưới thô, đã thay bằng số trên.

Các phép kiểm tra đi kèm:

- **Rò rỉ dữ liệu:** chỉ số ngày của dự báo đúng; test rò rỉ pass.
- **Làm tròn 6 mức:** bỏ làm tròn thì (s,S) theo nhóm rẻ thêm 1,6%, Newsvendor theo nhóm đắt hơn 0,7%. Làm tròn không phải nguyên nhân.
- **Dự báo LightGBM:** RMSSE trên kiểm thử 0,87, so với 1,19 của cách lấy nhu cầu tuần trước.

**Chạy thử trên miền xác thực, so với Newsvendor dự báo ở cùng tỷ lệ đáp ứng:**

| Lần chạy | Episode | Chi phí, tỷ lệ đáp ứng | Chênh so với Newsvendor dự báo |
|---|---:|---:|---:|
| U5 lưới rộng, `phi_dv` = 3 | 500 | 1.631.557 ở 91,09% | +5,0% |
| U5 lưới rộng, `phi_dv` = 1 | 500 | 1.553.522 ở 89,79% | +3,1% |
| U1 | 500 | 1.746.085 ở 90,60% | +14,0% |
| U5 lưới hẹp, entropy mặc định | 600 | 1.532.124 ở 89,77% | +1,7% |
| **U5 lưới hẹp, entropy thấp (đã chốt)** | 600 | 1.492.070 ở 88,85% | **+0,7%** |

Cấu hình đã chốt: `z` thuộc {0; 0,2; 0,4; 0,6; 0,8; 1,2}, hệ số entropy 0,005 giảm về 0,0005, `phi_dv` = 1. Chỉ chọn trên miền xác thực. Chưa chắc thắng ở 10.000 episode, nhưng chiều hướng tốt nhất trong các biến thể.

## 4. Thay đổi về code

### Tính năng mới

Mọi tính năng nằm sau cờ, tắt mặc định, mô hình chính giữ 44 chiều.

| Mã | Nội dung | File chính |
|---|---|---|
| B1 | Lưới mịn, đường biên Pareto, chọn cùng mức phục vụ, nội suy | `scripts/compare_same_service.py`, `scripts/cost_service_frontier.py` |
| B1 | Đường biên bản theo nhóm bằng giá bóng của lượng thiếu hàng | `scripts/compare_same_service.py` |
| B2 | (s,S) biết sức chứa | `baselines/traditional_policies.py` |
| B3 | Tỷ lệ đáp ứng và chi phí vận hành theo nhóm cầu | `scripts/service_by_group.py` |
| B4 | Dự báo LightGBM ngoài mẫu, Newsvendor dự báo | `scripts/forecast_demand.py` |
| B5 | 5 seed, IQM, bootstrap phân tầng | `scripts/multi_seed_stats.py` |
| E1 | Cấu trúc chính sách học được | `scripts/policy_structure.py` |
| U1 | Thêm dự báo vào quan sát (46 chiều) | `env/inventory_env.py` |
| U2 | Tác tử chọn hệ số an toàn của (s,S) | `agents/ppo_agent.py` |
| U3 | Critic có ngữ cảnh theo kho | `agents/ppo_agent.py` |
| U4 | 1, 4, 10, 300 mạng | `agents/ppo_agent.py` |
| U5 | **Mới:** tác tử chọn hệ số an toàn trên nền dự báo | `baselines/traditional_policies.py`, `configs/forecast_safety_factor.yaml` |
| C' | Lagrange PID theo cặp, λ giữ qua các tập, lưu kèm checkpoint | `env/inventory_env.py` |
| E2 | 1.000 tác tử | `configs/scale_1000_agents.yaml` |
| A5 | Trang phiếu mua sắm | `app/core.py`, `app/pages_/don_dat_hang.py` |
| Mới | Mốc biết trước nhu cầu | `scripts/perfect_demand_benchmark.py` |

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

## 5. Thay đổi về báo cáo

| Chương | Thay đổi |
|---|---|
| C1 | Viết lại bằng câu đơn giản, gạch đầu dòng; sửa câu hỏi nghiên cứu; bỏ tham chiếu tới chương sau; bỏ bảng chi phí trùng |
| C2 | Thêm mức đặt mục tiêu, dự báo rồi tối ưu, LightGBM, RMSSE, CMDP, PID Lagrange, CTDE, chia sẻ chọn lọc, phương trình Bellman; đối chiếu lại 13 nguồn mới; sửa câu CD-PPO; thêm hàng chính sách đối chứng trong bảng so sánh |
| C3 | Sắp lại mục MDP; chuyển hình Actor–Critic về đúng mục; thêm mục 3.4 cho 6 biến thể; thêm ví dụ bằng số; sửa các chỗ sai so với code (lịch hàng về, chuẩn hóa phần thưởng, đơn giá, "phối hợp") |
| C4 | Thêm hai chính sách đối chứng, so sánh trên đường biên, thống kê, bảng nối câu hỏi với thí nghiệm, bảng thiết kế thí nghiệm mở rộng; sắp lại theo câu hỏi nghiên cứu; bỏ mục hạn chế trùng C5 |
| C5 | Bỏ đoạn trùng bảng đối chiếu; bảng 9 hướng phát triển |
| Chung | Bỏ câu dẫn bảng cạnh bảng; bỏ tham chiếu tới mục của chương sau; bớt thuật ngữ dịch gượng |

## 6. Bỏ Colab

| File, nội dung | Trạng thái |
|---|---|
| `rl_inventory/Khoa_luan_Colab.ipynb` | Đã xóa |
| `rl_inventory/Khoa_luan_Colab_test_nhanh.ipynb` | Đã xóa |
| README, `run_experiments.py`, `train.py`, `evaluate.py`, `.gitignore` | Đã bỏ nội dung Colab, viết lại hướng dẫn local |
| `TIEN_DO.md` | Còn 4 dòng ghi lịch sử, không ảnh hưởng gì |

## 7. Cách train

```powershell
cd rl_inventory
# Cài PyTorch bản CUDA trước: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
python run.py check          # dòng GPU phải là [OK]
python run.py full --quick   # thử cả dây chuyền
python run.py full           # train thật
```

Bị ngắt thì chạy lại đúng lệnh, tự nối tiếp. Kết quả nằm ở `results/KET_QUA.txt`.

Thứ tự train: mô hình chính → U5 → U1 → U2 → `phi_dv` → siêu tham số → ablation → U4 → U3 → C' → E2.

| Nhóm | Episode | GPU cỡ T4 | CPU 4 nhân |
|---|---:|---:|---:|
| Mô hình chính, 5 seed | 50.000 | 15 giờ | 31 giờ |
| U5, 5 seed | 50.000 | 15 giờ | 31 giờ |
| U1, U2, mỗi biến thể 3 seed | 60.000 | 18 giờ | 37 giờ |
| `phi_dv` | 65.000 | 20 giờ | 40 giờ |
| Siêu tham số | 6.000 | 2 giờ | 4 giờ |
| Ablation | 16.000 | 5 giờ | 10 giờ |
| U4 | 40.000 | 12 giờ | 24 giờ |
| U3 | 30.000 | 11 giờ | 21 giờ |
| C' | 33.000 | 10 giờ | 20 giờ |
| E2 | 10.000 | 9–25 giờ | 18–50 giờ |
| **Tổng** | **360.000** | **khoảng 125 giờ** | **khoảng 250 giờ** |

## 8. Việc còn lại sau khi train

1. Thay số ở mọi chỗ có comment `% [SO-MOI]` (C4, C5, tóm tắt).
2. Viết lại kết luận theo luận điểm mới, thêm mục Câu hỏi 4.
3. Chạy `python scripts/export_tex_numbers.py` để quét số cũ.
4. Dựng lại demo bằng `python run.py demo_data`.
5. Cập nhật `van_dap.md`.
6. Tạo release `v1.0-final`.
7. Làm poster.

## 9. Rủi ro và điều cần biết

- **GPU:** máy i5-1135G7 này chỉ có Intel Iris Xe, PyTorch đang là bản CPU. Nếu train trên máy khác thì phải chuyển code sang trước (commit hoặc sao chép).
- **`app/demo_data/best_model.pth`** là bản seed 42 duy nhất còn lại của đợt 10.000 episode. Nên sao lưu trước khi chạy `run.py demo_data`.
- **Thiếu số gốc:** trung bình 3 seed trong C4 (2.055.442), ablation và log học không còn file gốc; `run.py full` sẽ tạo lại toàn bộ.
- **U5 chưa chắc thắng.** Nếu không thắng, đề tài vẫn bảo vệ được theo luận điểm mới, kết luận là dự báo rồi tối ưu đủ tốt trong bối cảnh này.
- **Câu hỏi nghiên cứu đã đổi:** nên báo giảng viên sớm.
- **Tác giả bài SSRN:** cần tự đối chiếu trên trình duyệt vì trang từ chối truy cập.
- **Hai file `phuluc_*.tex`** đã bị xóa từ trước, cần xác nhận là cố ý.
- **Toàn bộ thay đổi chưa commit,** chỉ nằm trên một máy.

## 10. Trạng thái trước khi train

- [x] Chốt cấu hình `forecast_safety_factor` trên miền xác thực.
- [x] Đường biên đầy đủ của seed 42, gồm bản theo nhóm.
- [x] Quét lại (s,S) biết sức chứa trên lưới κ mới. Số 2.260.819 đo trên 240 cấu hình, sẽ được thay bằng lưới đầy đủ khi train.
- [x] Chạy riêng `forecast_safety_factor` qua train, đánh giá, cùng mức phục vụ, phân tích giai đoạn.
- [x] `export_tex_numbers.py` đọc được định dạng kết quả thật.
- [x] `build_results.py` sinh được `KET_QUA.txt` từ kết quả thật của seed 42.
- [x] Mọi script trong `scripts/` khởi động được, tham số `run_experiments.py` truyền vào khớp với từng script.
- [x] 82 test pass, báo cáo 102 trang, không tràn lề, không tham chiếu lỗi.
- [ ] Dự báo tính lại mới có test trên dữ liệu giả, chưa chạy trên mô hình dự báo thật vì `forecast_model.txt` chỉ sinh ra khi chạy lại `forecast_demand.py`.
- [ ] Biến thể nguồn cung chung mới chạy với chính sách có sẵn, IPPO chưa được train trong điều kiện này.
- [ ] `run.py full --quick` chưa chạy trọn trên code cuối. Lượt gần nhất qua 121 lệnh rồi dừng ở lỗi số 5 của vòng rà 3. Phần đánh giá thí nghiệm mở rộng và tổng hợp sau chỗ đó mới được đọc code, chưa chạy nối tiếp. Lượt chạy lại đã dừng theo yêu cầu không train.

Bằng chứng số đo seed 42 nằm ở `docs/ket_qua_seed42_31-10/`.
