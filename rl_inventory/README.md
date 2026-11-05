# PS-IPPO cho quản lý tồn kho đa kho — Hướng dẫn chạy

10 kho × 30 mặt hàng = 300 tác tử, dữ liệu M5 Walmart, quan sát 44 chiều.
Tiến độ: **[../docs/TIEN_DO.md](../docs/TIEN_DO.md)** · Vấn đáp: **[../docs/van_dap.md](../docs/van_dap.md)**.

Trên máy nhóm, `python` trần là bản MSYS2 thiếu thư viện: mọi lệnh dưới đây chạy bằng `py -3.12`.

---

## 1. Cài đặt

```bash
# Có card NVIDIA: cài PyTorch bản CUDA trước, lấy lệnh ở https://pytorch.org/get-started/locally/
pip install -r requirements.txt
py -3.12 run.py check                # dòng GPU: [OK] là train trên cuda, [--] là train trên CPU
py -3.12 run.py test                 # phải toàn passed
```

Dữ liệu đã xử lý có sẵn trong `data/processed/`. Muốn tiền xử lý lại: đặt 3 file CSV M5 vào
`data/raw/` rồi `py -3.12 run.py data`.

Demo, không cần train:

```bash
py -3.12 run.py demo_data            # đóng gói app/demo_data/ (một lần)
py -3.12 run.py app                  # ứng dụng mô phỏng Streamlit, chỉ suy luận
py -3.12 app/mcp_server.py --selftest
```

---

## 2. Đợt thực nghiệm

```bash
py -3.12 run.py full --quick         # thử cả dây chuyền trên bản sao, mỗi lần train 3 episode
py -3.12 run.py full                 # test -> train toàn bộ -> đánh giá -> results/KET_QUA.txt
py -3.12 run.py eval_all             # chỉ đánh giá (khi đã train xong)
py -3.12 run.py tonghop              # dựng lại KET_QUA.txt từ kết quả đã có (vài giây)
py -3.12 run.py status               # tiến độ từng lần chạy
```

**Train và đánh giá đều chạy nối tiếp.** Bị ngắt (tắt máy, mất điện, Ctrl+C) thì chạy lại đúng lệnh đó:
train tiếp từ checkpoint gần nhất; đánh giá bỏ qua bước nào đã có kết quả mới hơn đầu vào
(checkpoint, `baseline_params*.json`, `same_service_main_s42.json`). Muốn chạy lại một bước: xóa file
kết quả của bước đó. Muốn tinh chỉnh lại baseline: xóa `results/baseline_params*.json`.
Tắt chế độ ngủ của máy trước khi để chạy qua đêm. Log ở `logs/<lệnh>.txt`.

Chạy từng phần (`scripts/run_experiments.py`):

```bash
py -3.12 scripts/run_experiments.py train main              # mô hình chính, 3 seed × 10.000 episode
py -3.12 scripts/run_experiments.py train ablations         # 16 lần × 1.000 episode
py -3.12 scripts/run_experiments.py train forecast_safety   # Câu 4, 3 seed × 10.000 episode
py -3.12 scripts/run_experiments.py train penalty_trial     # 5 lần thử phi_dv × 1.000 episode
py -3.12 scripts/run_experiments.py train <tag>             # một lần chạy, vd main_s42
py -3.12 scripts/run_experiments.py eval                    # toàn bộ pipeline đánh giá
```

Thứ tự bước `eval` (bước quan trọng trước): tinh chỉnh baseline (nếu chưa có) → đánh giá mô hình chính
và Câu 4 (3 seed) → zero-shot nguồn cung chung (Câu 5) → cùng mức phục vụ, lưới mịn (lâu nhất, vài giờ;
đường biên baseline cache ở `results/baseline_frontier.json`, dùng chung cho mọi seed) → giai đoạn nhu cầu,
độ nhạy, độ bền, quy mô cầu, hành vi → ablation, Câu 3, phi_dv → mặt hàng chưa gặp → mốc biết trước nhu cầu,
đường đánh đổi → `KET_QUA.txt`.

### Thời gian train đã đo (RTX 3050 Laptop)

| Nhóm | Episode | Thời gian |
|---|---|---:|
| Mô hình chính, 3 seed | 30.000 | khoảng 4 giờ/seed |
| Hệ số an toàn trên nền dự báo (Câu 4), 3 seed | 30.000 | khoảng 8 giờ/seed |
| Chạy thử hệ số phạt (5 giá trị × 1.000 tập) | 5.000 | khoảng 45 phút/lần |
| Ablation, Câu 3, mặt hàng chưa gặp | 16.000 | khoảng 25 phút/lần |

---

## 3. Tên lần chạy

Tag giữ nguyên vì đã gắn vào tên file checkpoint/kết quả. Tên đầy đủ in ở mục 0 của
`results/KET_QUA.txt` (nguồn: `TEN_DAY_DU` trong `scripts/build_results.py`).

| Tag | Tên đầy đủ | Câu hỏi |
|---|---|---|
| `main_s42`, `main_s1`, `main_s2` | Mô hình chính PS-IPPO, 44 chiều, 10.000 tập, không dự báo | 1, 2, 5 |
| `forecast_safety_s*` | PS-IPPO + dự báo: chọn hệ số an toàn trên nền dự báo LightGBM | 4 |
| `abl_ref`, `abl_ref_s*` | Mốc ablation: cấu hình chính, 1.000 tập | 3 |
| `abl_global`, `abl_global_s*` | Phần thưởng toàn mạng lưới | 3 |
| `abl_q3`, `abl_q3_s*` | Tắt chuẩn hóa phần thưởng theo cặp | 3 |
| `abl_reward_cu` | Phạt mức phục vụ kiểu cũ | ablation |
| `abl_trunk` | Actor/Critic chung thân mạng, tắt chuẩn hóa giá trị | ablation |
| `abl_nocal` / `abl_nowh` | Bỏ đặc trưng lịch / bỏ tín hiệu cấp kho | ablation |
| `abl_event` / `abl_warm` | Thêm sự kiện sắp tới / khởi tạo lịch sử cầu thật | ablation |
| `holdout`, `holdout_unseen`, `holdout_seen_*` | Mặt hàng chưa gặp (train 24 SKU, đánh giá 6 SKU) | 2 |
| `penalty05_trial` … `penalty100_trial` | Chạy thử hệ số phạt φ = 0,5 … 10 | đường đánh đổi |
| `zeroshot_shared_supply_tight/mild` | Mô hình chính dưới nguồn cung chung 2 / 3 × cầu TB, không train lại | 5 |

Tên tắt chính sách trong bảng: `-N` theo nhóm quy mô cầu, `-K` biết sức chứa, `NV-DB` Newsvendor dự báo,
`=PV` chỉnh về cùng mức phục vụ với IPPO.

---

## 4. Cấu trúc thư mục

```
rl_inventory/
├── run.py                        điểm vào: check, full, eval_all, tonghop, app...
├── config.yaml                   toàn bộ tham số và cờ
├── configs/                      config biến thể (sinh bằng configs/make_configs.py)
├── env/inventory_env.py          môi trường Gymnasium (300 cặp vector hóa)
├── agents/                       PS-IPPO, rollout buffer (GAE theo cặp)
├── baselines/                    EOQ, (s,S), Newsvendor, (s,S) biết sức chứa, Newsvendor dự báo
├── scripts/
│   ├── run_experiments.py        danh sách lần chạy, train (tự resume), eval (nối tiếp), tổng hợp
│   ├── train.py / evaluate.py    huấn luyện / đánh giá (kiểm định theo cặp)
│   ├── tune_baselines.py         tinh chỉnh baseline trên val theo ràng buộc 85%
│   ├── compare_same_service.py   cùng mức phục vụ: lưới mịn, đường biên baseline
│   ├── cost_service_frontier.py  đường đánh đổi chi phí–mức phục vụ (điểm IPPO, phi_dv)
│   ├── forecast_demand.py        dự báo LightGBM (ngoài mẫu trên miền train)
│   ├── perfect_demand_benchmark.py  mốc biết trước nhu cầu
│   ├── regime_analysis.py        giai đoạn nhu cầu, sốc cầu, bootstrap khối 7 ngày
│   ├── policy_behavior.py        hành vi chính sách, permutation importance
│   ├── analyze_scale_groups.py   theo nhóm quy mô cầu
│   ├── sensitivity.py            độ nhạy đơn giá chi phí
│   ├── robustness.py             độ bền zero-shot (sức chứa, lead time, nhu cầu ±20%)
│   ├── build_results.py          sinh results/KET_QUA.txt (theo 5 câu hỏi) + bảng tra tên
│   ├── plot_learning_curve.py    đường học
│   └── data_preprocessing.py     tiền xử lý M5
├── app/                          Streamlit, mcp_server.py
├── data/processed/               dữ liệu đã xử lý
├── checkpoints/ results/ runs/   đầu ra
└── tests/                        test_env.py, test_demo.py
```

Kết quả cũ không dùng cho báo cáo: `results/iso_service*.{json,png}` (lưới thô, phóng đại lợi thế IPPO),
`results_pre_p0/`, `checkpoints_pre_p0/`, `best_model_seed42.pth`.

Báo cáo LaTeX: `../Report KLTN/`, biên dịch bằng `compile.bat` (XeLaTeX + Biber). Các chỗ còn chờ số
có comment `% [SO-MOI]`, mỗi comment ghi mục `KET_QUA.txt` cần lấy.

---

## 5. Ghi chú phương pháp

- Đã bỏ khỏi đợt thực nghiệm (phương án rút gọn 01/11 và 03/11): Critic theo kho, số mạng, phạt Lagrange
  theo cặp, 1.000 tác tử, độ nhạy siêu tham số, quan sát có dự báo (U1), hệ số an toàn của (s,S) (U2),
  zero-shot sức chứa 3/8 ngày, chính sách kết hợp theo nhóm, bootstrap phân tầng nhiều seed. Lý do: thời
  gian train, và các câu hỏi đã được trả lời bằng thí nghiệm còn giữ.
- Hệ số `phi_dv` chọn theo cột `eval_fill` trong `results/train_log_*.csv` (miền xác thực), không theo
  `KET_QUA.txt` (miền kiểm thử).
- Tái lập: clone sạch → `pip install -r requirements.txt` → `py -3.12 run.py test` →
  `py -3.12 scripts/evaluate.py --checkpoint checkpoints/best_model_main_s42.pth --episodes 30 --tag main_s42`
  → khớp số mục 1 của `KET_QUA.txt`.
