# PS-IPPO cho quản lý tồn kho đa kho — Hướng dẫn chạy

10 kho × 30 mặt hàng = 300 tác tử, dữ liệu M5 Walmart, quan sát 44 chiều.
Tiến độ: **[../docs/TIEN_DO.md](../docs/TIEN_DO.md)** · Tổng hợp thay đổi:
**[../docs/TONG_HOP_THAY_DOI.md](../docs/TONG_HOP_THAY_DOI.md)** · Vấn đáp: **[../docs/van_dap.md](../docs/van_dap.md)**.

---

## 1. Cài đặt

```bash
# Có card NVIDIA: cài PyTorch bản CUDA trước, lấy lệnh ở https://pytorch.org/get-started/locally/
pip install -r requirements.txt
python run.py check                  # xem dòng GPU: [OK] là train trên cuda, [--] là train trên CPU
python run.py test                   # phải toàn passed
```

`train.py` tự dùng GPU khi PyTorch thấy CUDA; các bước đánh giá luôn chạy trên CPU. Dữ liệu đã
xử lý có sẵn trong `data/processed/`. Ba file CSV gốc trong `data/raw/` không nằm trong git: nếu train
ở máy khác thì chép sang, chỉ cần khi tiền xử lý lại. Muốn tiền xử lý lại: đặt 3 file CSV M5 vào `data/raw/`
rồi `python run.py data`.

Demo, không cần train:

```bash
python run.py demo_data              # dong goi app/demo_data/ (mot lan)
python run.py app                    # demo Streamlit 5 trang, chi inference
python app/mcp_server.py --selftest  # thử MCP không cần LLM
```

---

## 2. Đợt thực nghiệm chính

```bash
python run.py full --quick           # thử cả dây chuyền, mỗi lần train 3 episode; chạy trước khi train thật
python run.py full                   # test -> dự báo -> train toàn bộ -> đánh giá -> results/KET_QUA.txt
```

`run.py full` tự làm hết, không bao giờ dùng miền kiểm thử để chọn tham số. **Phương án rút gọn 01/11 (3 seed):** thứ tự train theo mức quan trọng với kết luận: mô hình chính → hệ số an toàn trên nền dự báo → quan sát có dự báo → hệ số an toàn → chạy thử hệ số phạt mức phục vụ → ablation. Sau đó đánh giá, gồm phần zero-shot dưới nguồn cung chung và sức chứa 3, 8 ngày. Nếu phải dừng giữa chừng, phần quan trọng nhất đã có. Các nhóm U3, U4, C', E2, E3 và train điều kiện vận hành đã bỏ (xem `../docs/TONG_HOP_THAY_DOI.md` mục 4).

Bị ngắt (tắt máy, mất điện, Ctrl+C) thì chạy lại đúng lệnh đó: mỗi lần chạy tự train tiếp từ
checkpoint gần nhất, lần đã xong tự bỏ qua. Tắt chế độ ngủ của máy trước khi để chạy qua đêm.
Log ở `logs/full.txt`; `logs/phien_ban_code.txt` ghi commit của từng lần train (hậu tố `-dirty`
là có thay đổi chưa commit). Đường dẫn có dấu tiếng Việt làm TensorBoard báo không ghi được, không
ảnh hưởng train: số liệu nằm ở `results/train_log_<tag>.csv`.

Chạy từng phần:

```bash
python scripts/run_experiments.py status             # tiến độ từng lần chạy
python scripts/run_experiments.py train main         # 3 seed × 10.000 episode
python scripts/run_experiments.py train ablations    # 16 lần × 1.000 episode
python scripts/run_experiments.py train <ten>        # một lần chạy, vd main_s42, abl_ref
python scripts/run_experiments.py train full         # chỉ phần train của run.py full, mở được nhiều terminal
python scripts/run_experiments.py eval               # toàn bộ pipeline đánh giá
python scripts/run_experiments.py tonghop            # dựng lại KET_QUA.txt từ kết quả đã có
```

`eval` chạy lần lượt: dự báo (nếu chưa có) → tune baseline → eval 3 seed → iso-service →
regime → behavior → quy mô cầu → độ nhạy → độ bền → ablation → hold-out → phần mở rộng →
`results/multiseed_main.json` → `results/rq3_multiseed.json`. Lần đầu mất thêm khoảng 1,5 giờ để
quét đường biên baseline; kết quả lưu ở `results/baseline_frontier.json` và được dùng lại. Đổi
môi trường hoặc lưới thì xóa file này.

**Mọi số liệu nằm trong một file: `results/KET_QUA.txt`**, tự dựng lại sau mỗi lần train, đánh giá:
- Phần 1: RQ1 (các seed so với từng baseline), RQ3 (từng seed so với `abl_ref`).
- Phần 2: từng mô hình: huấn luyện (mô hình được chọn, episode đạt 85%, đã hội tụ chưa, cơ cấu
  chi phí) và đánh giá trên test (bảng chi phí/fill, kiểm định theo cặp, kết luận).

### Thời gian ước tính

Tính theo tốc độ đo được: 1,1 giây/episode trên GPU T4 và 2,2 giây/episode trên CPU Intel
i5-1135G7. Đo máy của bạn bằng `python scripts/train.py --episodes 30 --tag do_toc_do` (dòng
`HOAN TAT sau ...`), rồi xóa `checkpoints/*do_toc_do*` và `results/*do_toc_do*`.

| Nhóm | Episode | GPU cỡ T4 | CPU 4 nhân |
|---|---|---:|---:|
| Mô hình chính, 3 seed | 30.000 | 9 giờ | 18 giờ |
| Hệ số an toàn trên nền dự báo, 3 seed | 30.000 | 9 giờ | 18 giờ |
| Quan sát có dự báo, hệ số an toàn, mỗi biến thể 3 seed | 60.000 | 18 giờ | 37 giờ |
| Chạy thử hệ số phạt (5 giá trị × 1.000 tập) | 5.000 | 1,5 giờ | 3 giờ |
| Ablation, RQ3, hold-out | 16.000 | 5 giờ | 10 giờ |
| **Tổng cộng (phương án rút gọn)** | **141.000** | **khoảng 43 giờ** | **khoảng 86 giờ** |

Nếu khôi phục giai đoạn train lại hai giá trị `phi_dv` × 3 seed (lựa chọn rút gọn có thể đổi lại) thì thêm 60.000 tập, khoảng 18 giờ T4, tổng khoảng 61 giờ.

Thiếu thời gian thì dừng lại ở bất kỳ đâu rồi chạy `python run.py eval_all`: `eval` đánh giá mọi lần chạy đã có checkpoint.

Chạy song song rút ngắn được thời gian. Trên CPU, 4 terminal với mỗi tiến trình một luồng đạt
0,57 episode/giây, so với 0,39 khi chạy một tiến trình (i5-1135G7, 4 nhân):

```powershell
$env:OMP_NUM_THREADS=1; python scripts/run_experiments.py train full
```

Có GPU thì bỏ phần đặt luồng và thử 2–3 terminal. Mỗi terminal tự nhận lần chạy chưa có ai làm, nên
không giẫm chân nhau. Chạy `python run.py forecast` một lần trước khi mở nhiều terminal. Khi `status`
báo xong hết, chạy `python run.py eval_all`.

### Lệnh lẻ (`run.py`)

| Lệnh | Việc |
|---|---|
| `data` | tiền xử lý M5 |
| `baseline` | tìm lưới tham số baseline |
| `train [--episodes N]` | huấn luyện IPPO |
| `eval` / `iso` | đánh giá / so sánh ở cùng mức phục vụ |
| `regime` / `behavior` | thực nghiệm B / hành vi policy |
| `summary` | xuất `TONG_HOP_SO_LIEU.txt` (số cũ, không dùng cho báo cáo) |
| `forecast` | dự báo LightGBM cho baseline dự báo và U1 |
| `all [--quick]` | chạy trọn bộ (bản rút gọn ~5 phút) |

---

## 3. Cấu trúc thư mục

```
rl_inventory/
├── run.py                       điểm vào: check, full, eval_all, app...
├── config.yaml                  toàn bộ tham số và cờ
├── configs/                     config ablation (sinh bằng configs/make_configs.py)
├── env/inventory_env.py         môi trường Gymnasium (300 cặp vector hóa)
├── agents/                      PS-IPPO, rollout buffer (GAE theo cặp)
├── baselines/                   EOQ, (s,S), Newsvendor, (s,S) biết sức chứa, Newsvendor dự báo
├── scripts/
│   ├── run_experiments.py              đợt cuối: train (tự resume) + eval + status
│   ├── train.py / evaluate.py   huấn luyện / đánh giá (kiểm định theo cặp)
│   ├── tune_baselines.py        tune trên val theo ràng buộc 85%
│   ├── compare_same_service.py           cùng mức phục vụ; --grid fine: lưới mịn, đường biên baseline
│   ├── forecast_demand.py       dự báo LightGBM (ngoài mẫu trên miền train)
│   ├── service_by_group.py         fill theo nhóm cầu, cổng quyết định Lagrange theo cặp
│   ├── policy_structure.py      cấu trúc chính sách học được so với order-up-to
│   ├── multi_seed_stats.py      IQM, bootstrap phân tầng theo khối ngày
│   ├── cost_service_frontier.py            đường biên chi phí–mức phục vụ theo phi_dv
│   ├── regime_analysis.py       giai đoạn nhu cầu, sốc cầu, bootstrap
│   ├── policy_behavior.py       hành vi policy, permutation importance
│   ├── analyze_scale_groups.py  theo nhóm quy mô cầu
│   ├── sensitivity.py           độ nhạy đơn giá chi phí
│   ├── robustness.py            độ bền zero-shot
│   ├── perfect_demand_benchmark.py  mốc biết trước nhu cầu
│   ├── plot_learning_curve.py   đường học
│   ├── generate_summary.py      TONG_HOP_SO_LIEU.txt
│   └── data_preprocessing.py    tiền xử lý M5
├── app/                         Streamlit (app.py + pages/), mcp_server.py
├── data/processed/              dữ liệu đã xử lý
├── checkpoints/ results/ runs/  đầu ra
└── tests/                       test_env.py, test_demo.py
```

**File kết quả cũ:** file trong `results/` không hậu tố, `TONG_HOP_SO_LIEU.txt`,
`results_pre_p0/`, `checkpoints_pre_p0/`, `best_model_seed42.pth` là từ các lần
chạy thử trước — **không dùng cho báo cáo**.

Báo cáo LaTeX: `../Report KLTN/`, biên dịch bằng `compile.bat` (XeLaTeX + Biber).

---

## 4. Thí nghiệm mở rộng

Mỗi biến thể là một cờ trong `config.yaml` và một file trong `configs/` (sinh bằng `python configs/make_configs.py`). `run.py full` đã chạy hết các biến thể còn lại sau khi rút gọn. Muốn chạy riêng một nhóm: `python scripts/run_experiments.py train <nhóm>`, rồi `python scripts/run_experiments.py eval_mo_rong`.

| Nhóm | Cấu hình | Nội dung |
|---|---|---|
| `forecast_safety` | `forecast_safety_factor.yaml` | Tác tử chọn hệ số an toàn trên nền dự báo |
| `forecast_obs` | `forecast_observation.yaml` | IPPO nhận dự báo trong quan sát |
| `safety_factor` | `safety_factor.yaml` | Tác tử chọn hệ số an toàn của (s,S) |
| `penalty_trial` | `service_penalty_*.yaml` | Chạy thử hệ số phạt mức phục vụ |

Đã bỏ khỏi đợt thực nghiệm (phương án rút gọn 01/11): `tuning`, `networks`, `wh_critic`, `pair_penalty_trial`, `scale_1000` và train điều kiện vận hành. Điều kiện vận hành (`shared_supply_*`, `capacity_3`, `capacity_8`) chỉ dùng để **đánh giá zero-shot** mô hình chính, không train.

Hệ số `phi_dv` chọn từ cột `eval_fill` trong `results/train_log_*.csv`, tức miền xác thực. Không dùng số trong `KET_QUA.txt` để chọn vì đó là miền kiểm thử.

Kết quả nằm trong mục 5b của `results/KET_QUA.txt`, chi tiết ở `results/same_service_<tag>.json`,
`service_by_group_<tag>.json`, `baseline_frontier.json`, `cost_service_frontier.json`, `multi_seed_stats.json`,
`forecast_metrics.json`, `policy_structure_main.json`. `python scripts/export_tex_numbers.py` sinh
`Report KLTN/so_lieu.tex` và báo các số lệch trong tài liệu.

Khi có kết quả cuối, đính kèm `KET_QUA.txt`, `results/*.json`, `best_model_main_s*.pth` và
`baseline_params.json` vào GitHub release `v1.0-final`, rồi kiểm tra tái lập: clone sạch →
`pip install -r requirements.txt` → `python run.py test` → `python scripts/evaluate.py` trên
checkpoint trong release → khớp số bảng chính (sai số làm tròn).
