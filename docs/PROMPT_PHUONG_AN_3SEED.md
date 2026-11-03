# Prompt: chuyển sang phương án rút gọn (3 seed, bỏ 6 nhóm thí nghiệm)

Dán nguyên phần trong khung dưới đây vào một phiên Claude Code mở tại thư mục gốc của repo (`Khoa_luan`).

````text
# Bối cảnh
Repo khóa luận "Xây dựng mô hình học tăng cường tối ưu hóa chính sách đặt hàng lại trong quản lý tồn kho đa kho"
(PS-IPPO, dữ liệu M5 Walmart, 300 cặp kho-SKU). Đọc trước, theo thứ tự:
1. docs/TONG_HOP_THAY_DOI.md (hướng nghiên cứu mới, kết quả sơ bộ seed 42, mã U1-U5/E/C')
2. docs/TIEN_DO.md, docs/van_dap.md, docs/pham_vi_khoa_luan.md
3. rl_inventory/scripts/run_experiments.py (SEEDS, EXTRA_MAIN_SEEDS, MAIN_SEEDS, MO_RONG, NHOM_MO_RONG, train_full, danh_gia_mo_rong)
4. rl_inventory/run.py, rl_inventory/scripts/build_results.py, rl_inventory/scripts/export_tex_numbers.py
5. "Report KLTN/content/C1..C5.tex", "Report KLTN/frontmatter/abstract*.tex"

Dùng skill `dinh-dang-bao-cao` mỗi khi sửa file .tex.

# Quyết định đã chốt (không hỏi lại, không đảo ngược)
Giảng viên chưa duyệt nhưng nhóm chọn phương án rút gọn để đủ thời gian train (máy hiện có GPU RTX 3050 Laptop).

A. Mô hình chính chỉ dùng 3 seed (42, 1, 2), không phải 5.
   Áp dụng cho mô hình chính và U5 (forecast_safety). Các nhóm khác giữ số seed hiện tại trong code.
B. BỎ HẲN các nhóm thí nghiệm sau, kể cả train, đánh giá, báo cáo, demo, tài liệu:
   - U3 (wh_critic, Critic theo kho)
   - U4 (networks, số mạng 1/4/10/300)
   - C' (pair_penalty, phạt Lagrange theo cặp)
   - E2 (scale_1000, 1.000 tác tử)
   - E3 (tuning, độ nhạy siêu tham số)
   - Điều kiện vận hành TRAIN (shared_supply_*, capacity_3, capacity_8 và các bản *_fs): không train nữa.
C. GIỮ và phải chạy: mô hình chính; U5 (forecast_safety); U1 (forecast_obs); U2 (safety_factor);
   ablation + RQ3 (abl_*) + hold-out; mọi phần ĐÁNH GIÁ không train (cùng mức phục vụ, đường biên, theo nhóm nhu cầu,
   sốc cầu, giai đoạn nhu cầu, Newsvendor dự báo, mốc biết trước nhu cầu, chính sách kết hợp, E1 policy_structure).
D. phi_dv (Gói C): chỉ giữ 5 lần chạy thử 1.000 episode để làm bằng chứng; BỎ giai đoạn train lại hai giá trị đã chọn
   × 3 seed (khoảng 60.000 episode). Ghi rõ lựa chọn này trong tài liệu để nhóm đổi lại nếu muốn.
E. Thay Câu 5 (điều kiện vận hành) bằng đánh giá ZERO-SHOT, không train:
   chạy mô hình chính đã train dưới các cấu hình shared_supply_tight, shared_supply_mild, capacity_3, capacity_8
   (chỉ evaluate). Nếu evaluate.py chưa hỗ trợ truyền cấu hình môi trường khác khi đánh giá checkpoint,
   hãy bổ sung bằng thay đổi nhỏ nhất. Báo cáo ghi rõ đây là zero-shot, không huấn luyện lại, và đây là giới hạn.
   Giữ lại phần "độ bền zero-shot" hiện có (sức chứa, lead time, nhu cầu ±20%) nếu chưa có.
F. Số câu hỏi nghiên cứu: 5 câu nhưng Câu 5 chuyển thành "độ bền khi đổi điều kiện vận hành (zero-shot)".
   Câu 2 chỉ còn phần theo quy mô nhu cầu và mặt hàng chưa gặp (hold-out), bỏ phần 1.000 tác tử.

# Việc cần làm (làm lần lượt, mỗi bước tự kiểm tra)
1. Code: sửa rl_inventory/scripts/run_experiments.py
   - EXTRA_MAIN_SEEDS = () để MAIN_SEEDS = SEEDS (3 seed).
   - Bỏ khỏi train_full: tuning, networks, wh_critic, pair_penalty_trial, các lần chạy pair_penalty_*, scale_1000, dieu_kien,
     và giai đoạn "phi_dv đã chọn × SEEDS". Giữ penalty_trial.
   - Xóa hoặc vô hiệu các mục tương ứng trong MO_RONG/NHOM_MO_RONG/DIEU_KIEN sao cho không còn tham chiếu treo.
   - Sửa danh_gia_mo_rong, tong_hop_*, build_results.py, export_tex_numbers.py, run.py (hướng dẫn, danh sách lệnh)
     để KHÔNG lỗi và KHÔNG in "thiếu" khi các nhóm bị bỏ vắng mặt.
   - Xóa các file cấu hình không còn dùng trong rl_inventory/configs (warehouse_critic.yaml, networks_*.yaml, pair_penalty_*.yaml,
     scale_1000_agents.yaml, tuning_*.yaml) và sửa configs/make_configs.py cho khớp.
     GIỮ shared_supply_*.yaml và capacity_*.yaml (dùng cho đánh giá zero-shot).
   - Bước zero-shot ở mục E: thêm vào đánh giá mở rộng, kết quả vào results/ và vào KET_QUA.txt.
2. Kiểm tra code:
   - Chạy bộ test (hiện 82 test pass); sửa/bổ sung test cho phần vừa đổi. Dùng `py -3.12` (xem ghi chú dưới).
   - `py -3.12 run.py check`, rồi `py -3.12 run.py full --quick` (chạy trên bản sao, không đụng kết quả thật).
     Phải chạy xuyên suốt tới bước tổng hợp KET_QUA.txt mà không lỗi. Nếu lỗi thì sửa tận gốc rồi chạy lại.
   - Trước khi kết luận xong bước này, grep toàn repo các tên đã bỏ (wh_critic, networks_, pair_penalty, scale_1000,
     tuning_, dieu_kien) để chắc không còn tham chiếu.
3. Tài liệu:
   - docs/TONG_HOP_THAY_DOI.md: sửa mục 1, 2, 4, 5, 6, 7 và phụ lục A, B, D (bảng giờ train: tổng khoảng 41-61 giờ T4),
     phụ lục F. Ghi rõ phương án rút gọn, vì sao, cái giá phải trả (mất Câu 5 dạng train, mất nguồn cung chung train,
     mất 1.000 tác tử; 3 seed thì khoảng tin cậy rộng hơn). Mục 7 (điểm xin ý giảng viên) đổi thành xin duyệt phương án rút gọn.
   - docs/TIEN_DO.md, docs/van_dap.md, rl_inventory/README.md: đồng bộ. van_dap thêm câu trả lời cho
     "tại sao chỉ 3 seed" và "đa kho ở chỗ nào nếu bỏ thí nghiệm nguồn cung chung train".
   - docs/PROMPT_PHUONG_AN_3SEED.md này giữ nguyên.
4. Báo cáo LaTeX ("Report KLTN"):
   - C1: danh sách câu hỏi nghiên cứu (mục F), phạm vi, giới hạn.
   - C3: bỏ mô tả U3, U4, C', E2, E3 và các điều kiện vận hành train; giữ mô tả cái còn chạy.
   - C4, C5, abstract (vi, en): bỏ mọi mục/bảng/hình của phần đã cắt; ghi 3 seed; phần Câu 5 viết theo zero-shot.
     Số liệu CHƯA có thì GIỮ NGUYÊN dấu `[SO-MOI]`, tuyệt đối không bịa số, không dùng lại số của giao thức cũ.
   - Biên dịch xelatex + biber (2-3 lượt); không lỗi, không tham chiếu `undefined`, không tràn lề mới.
5. Demo (rl_inventory/app) và sơ đồ:
   - Bảo đảm demo không tham chiếu thứ đã bỏ; vẫn chạy tốt với các chính sách còn lại (kể cả chính sách dự báo).
   - Cập nhật pipeline.drawio ở gốc repo CHỈ nếu có trang nhắc nội dung đã bỏ; KHÔNG sửa trang 1.
6. Cuối cùng: in danh sách file đã sửa/xóa, kết quả test, kết quả `full --quick`, và LỆNH TRAIN THẬT cần chạy:
   `cd rl_inventory; py -3.12 run.py full` (tự nối tiếp khi bị ngắt). Ước tính giờ theo số lần chạy thực tế còn lại.

# Ràng buộc
- Viết tiếng Việt có dấu đầy đủ. Mã định danh, tên file, lệnh giữ nguyên.
- KHÔNG train thật trong phiên này (chỉ `--quick` trên bản sao). KHÔNG commit, KHÔNG push, trừ khi tôi yêu cầu.
- Số liệu chỉ lấy từ rl_inventory/results/ hoặc KET_QUA.txt. Số trong docs/TONG_HOP_THAY_DOI.md là sơ bộ (seed 42):
  chỉ dùng để mô tả hiện trạng và phải ghi rõ "sơ bộ".
- Mô hình chính giữ nguyên 44 chiều quan sát; không đổi thuật toán PS-IPPO.
- Demo chỉ suy luận, gọi là "ứng dụng mô phỏng" (xem memory demo-advisor-constraints).
- `python` trần là MSYS2 3.11; mọi lệnh dự án dùng `py -3.12`. lightgbm đã cài.
- Sửa nhỏ nhất đủ dùng, không thêm tính năng ngoài danh sách. Không refactor ngoài phạm vi.
- Nếu một yêu cầu mâu thuẫn với code (ví dụ nhóm nào đó được nhóm khác dùng), dừng lại, nêu rõ chỗ mâu thuẫn và đề xuất cách xử lý thay vì tự quyết.
````

## Ghi chú cho nhóm

- Prompt đã đối chiếu với `run_experiments.py`: đổi seed chỉ cần `EXTRA_MAIN_SEEDS = ()`. U5 dùng `MAIN_SEEDS` nên cũng xuống 3 seed theo.
- Thí nghiệm E2 (1.000 tác tử) vốn sẽ tự bị bỏ qua vì máy thiếu `data/raw/sales_train_evaluation.csv`.
- Mục D (`phi_dv`) là lựa chọn của tôi, không phải của bạn. Nếu muốn giữ phần train lại hai giá trị × 3 seed (khoảng 20 giờ T4), xóa mục D trước khi dán.
