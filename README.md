# Quản lý Tồn kho Đa Kho bằng Học sâu Tăng cường (RL)
> Khóa luận tốt nghiệp — Tối ưu Quản lý Tồn kho Đa Kho bằng Deep Reinforcement Learning

## Cấu trúc thư mục

```
rl_inventory/
├── config.yaml            ← Tất cả hyperparameter & đường dẫn (chỉnh ở đây)
├── main.py                ← Entry point duy nhất (CLI)
├── utils.py               ← Hàm tiện ích dùng chung
├── data/
│   ├── raw/              ← Đặt file M5 ở đây (sales_train_evaluation.csv, calendar.csv)
│   └── processed/        ← Đầu ra: demand_data.npy, env_config.json
├── env/
│   └── inventory_env.py  ← Môi trường Gymnasium
├── agents/
│   ├── dqn_agent.py      ← Tác tử Double DQN
│   └── replay_buffer.py  ← Standard + Prioritized Experience Replay (PER)
├── baselines/
│   └── traditional_policies.py  ← EOQ, (s,S), Newsvendor
├── scripts/
│   ├── data_preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   └── colab_mvp.ipynb   ← Notebook chạy thử Colab end-to-end
├── checkpoints/          ← Lưu trọng số mô hình (.pth)
├── results/              ← Biểu đồ so sánh & kết quả CSV
└── runs/                 ← Log theo dõi TensorBoard
```

---

## Cài đặt

```bash
# 1. Tạo môi trường ảo virtual environment
python -m venv venv
venv\Scripts\activate          # Trên Windows
# source venv/bin/activate     # Trên Linux/macOS

# 2. Cài đặt PyTorch với CUDA 11.8 (tối ưu cho RTX 3050)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Cài đặt các thư viện phụ thuộc còn lại
pip install -r requirements.txt pyyaml
```

---

## Cách chạy

> Chạy từ thư mục gốc `Khoa_luan\` hoặc vào `rl_inventory\` trước đều được:
> ```bash
> # Từ Khoa_luan\:
> py rl_inventory\main.py <lệnh>
>
> # Hoặc cd vào rl_inventory\ rồi chạy:
> cd rl_inventory
> py main.py <lệnh>
> ```

---

### Bước 1 — Chuẩn bị dữ liệu *(chạy 1 lần)*

```bash
# Cách A: Dữ liệu giả lập (không cần tải gì thêm, dùng để thử nhanh)
py rl_inventory\main.py preprocess --synthetic

# Cách B: Dữ liệu M5 thực tế (Walmart Kaggle)
#   → Tải về từ: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
#   → Đặt sales_train_evaluation.csv + calendar.csv vào rl_inventory\data\raw\
py rl_inventory\main.py preprocess
```

### Bước 2 — Huấn luyện mô hình *(chạy 1 lần, ~10–15 phút với RTX 3050)*

```bash
py rl_inventory\main.py train
```

Model được lưu tự động vào `rl_inventory\checkpoints\`:
- `best_model.pth` — checkpoint có reward cao nhất trong quá trình train
- `final_model.pth` — trạng thái cuối cùng sau khi train xong

> Theo dõi tiến trình real-time (mở terminal riêng):
> ```bash
> tensorboard --logdir rl_inventory\runs\
> # Truy cập: http://localhost:6006
> ```

### Bước 3 — Đánh giá & so sánh kết quả *(chạy bất kỳ lúc nào sau khi train)*

```bash
# Đánh giá với checkpoint tốt nhất (mặc định)
py rl_inventory\main.py evaluate

# Hoặc chỉ định checkpoint cụ thể
py rl_inventory\main.py evaluate --checkpoint rl_inventory\checkpoints\final_model.pth
```

Kết quả xuất ra `rl_inventory\results\`:
- Biểu đồ so sánh DQN vs EOQ vs (s,S) vs Newsvendor (`.png`)
- Bảng chỉ số chi tiết (`.csv`)

---

### Tùy chỉnh hyperparameter

Chỉnh trực tiếp trong [`rl_inventory/config.yaml`](rl_inventory/config.yaml) rồi chạy lại — không cần sửa code:

```yaml
train:
  episodes: 500      # ← tăng/giảm số tập train
  lr: 3.0e-4         # ← learning rate
  use_per: false     # ← bật Prioritized Experience Replay
  batch_size: 128    # ← tùy theo VRAM GPU
```

Hoặc override tạm thời qua CLI mà không cần sửa file:

```bash
py rl_inventory\main.py train --episodes 200 --lr 1e-4 --use_per
```

---

## Thiết kế chính (Ghi chú Khóa luận tốt nghiệp)

### Môi trường (Environment)
| Thành phần | Chi tiết |
|---|---|
| Trạng thái (State) | Mức tồn kho hiện tại + Lịch sử nhu cầu (7 ngày) + Đơn hàng đang vận chuyển (Pipeline) + Mã hóa ngày trong tuần |
| Hành động (Action) | Rời rạc 6 mức: {0, 10, 20, 30, 40, 50} đơn vị (phân rã độc lập theo từng cặp Kho-SKU) |
| Phần thưởng (Reward) | -(chi phí lưu kho + chi phí thiếu hàng + chi phí đặt hàng cố định + phạt tràn kho) |
| Chuyển trạng thái | Nhu cầu lấy từ dữ liệu M5 thực tế, thời gian cung ứng lead time U[1,3] ngày |

### Tác tử — Double DQN
| Thành phần | Chi tiết |
|---|---|
| Mạng Neural | MLP: 256→256→(n_pairs × 6) |
| Thuật toán Double DQN | van Hasselt et al. (2016) — khắc phục hiện tượng đánh giá quá cao giá trị hành động (overestimation bias) |
| Bộ đệm Replay | Standard hoặc Prioritized Experience Replay (Schaul et al., 2016) |
| Mạng mục tiêu | Soft Polyak averaging (τ=0.005) |
| Khám phá | Chiến lược ε-greedy với suy giảm theo hàm mũ |
| Thiết bị tính toán | CUDA (NVIDIA RTX 3050 4GB) — batch_size=128 |

### Các phương pháp Baseline so sánh (Bắt buộc cho bài báo / khóa luận)
| Baseline | Tài liệu trích dẫn |
|---|---|
| EOQ (Lượng đặt hàng kinh tế) | Harris (1913) |
| Chiến lược (s,S) | Scarf (1960) |
| Mô hình Newsvendor | Arrow, Harris & Marschak (1951) |

---

## Chỉ số Đánh giá (Evaluation Metrics)

- **Chi phí Tổng (Total Cost)**: Chi phí lưu kho (Holding) + Chi phí thiếu hàng (Stockout) + Chi phí đặt hàng (Ordering) trên mỗi tập.
- **Mức độ Phục vụ (Service Level)**: Tỷ lệ % nhu cầu khách hàng được đáp ứng thành công.
- **Tỷ lệ Thiếu hàng (Stockout Rate)**: Tỷ lệ % nhu cầu bị thất thoát do thiếu hàng.
- **Tồn kho Trung bình (Average Inventory)**: Lượng hàng tồn kho bình quân duy trì tại các kho.

---

## Theo dõi với TensorBoard

```bash
tensorboard --logdir runs/
# Mở trình duyệt tại: http://localhost:6006
```

Các nhật ký theo dõi bao gồm:
- `episode/reward`, `episode/cost`, `episode/service_level` (theo từng tập episode)
- `train/loss`, `train/epsilon`, `train/avg_loss` (theo từng bước step)
- `eval/reward`, `eval/service_level` (đánh giá định kỳ theo chiến lược tham lam)

---

## Yêu cầu phần cứng GPU

- NVIDIA RTX 3050 4GB VRAM ✅
- Batch size: 128
- Replay Buffer: 100,000 bước chuyển trạng thái (~500MB RAM)
- Thời gian huấn luyện 500 tập: ~10–15 phút


### Môi trường (Environment)
| Thành phần | Chi tiết |
|---|---|
| Trạng thái (State) | Mức tồn kho hiện tại + Lịch sử nhu cầu (7 ngày) + Đơn hàng đang vận chuyển (Pipeline) + Mã hóa ngày trong tuần |
| Hành động (Action) | Rời rạc 6 mức: {0, 10, 20, 30, 40, 50} đơn vị (phân rã độc lập theo từng cặp Kho-SKU) |
| Phần thưởng (Reward) | -(chi phí lưu kho + chi phí thiếu hàng + chi phí đặt hàng cố định + phạt tràn kho) |
| Chuyển trạng thái | Nhu cầu lấy từ dữ liệu M5 thực tế, thời gian cung ứng lead time U[1,3] ngày |

### Tác tử — Double DQN
| Thành phần | Chi tiết |
|---|---|
| Mạng Neural | MLP: 256→256→(n_pairs × 6) |
| Thuật toán Double DQN | van Hasselt et al. (2016) — khắc phục hiện tượng đánh giá quá cao giá trị hành động (overestimation bias) |
| Bộ đệm Replay | Standard hoặc Prioritized Experience Replay (Schaul et al., 2016) |
| Mạng mục tiêu | Soft Polyak averaging (τ=0.005) |
| Khám phá | Chiến lược ε-greedy với suy giảm theo hàm mũ |
| Thiết bị tính toán | CUDA (NVIDIA RTX 3050 4GB) — batch_size=128 |

### Các phương pháp Baseline so sánh (Bắt buộc cho bài báo / khóa luận)
| Baseline | Tài liệu trích dẫn |
|---|---|
| EOQ (Lượng đặt hàng kinh tế) | Harris (1913) |
| Chiến lược (s,S) | Scarf (1960) |
| Mô hình Newsvendor | Arrow, Harris & Marschak (1951) |

---

## Chỉ số Đánh giá (Evaluation Metrics)

- **Chi phí Tổng (Total Cost)**: Chi phí lưu kho (Holding) + Chi phí thiếu hàng (Stockout) + Chi phí đặt hàng (Ordering) trên mỗi tập.
- **Mức độ Phục vụ (Service Level)**: Tỷ lệ % nhu cầu khách hàng được đáp ứng thành công.
- **Tỷ lệ Thiếu hàng (Stockout Rate)**: Tỷ lệ % nhu cầu bị thất thoát do thiếu hàng.
- **Tồn kho Trung bình (Average Inventory)**: Lượng hàng tồn kho bình quân duy trì tại các kho.

---

## Theo dõi với TensorBoard

```bash
tensorboard --logdir runs/
# Mở trình duyệt tại: http://localhost:6006
```

Các nhật ký theo dõi bao gồm:
- `episode/reward`, `episode/cost`, `episode/service_level` (theo từng tập episode)
- `train/loss`, `train/epsilon`, `train/avg_loss` (theo từng bước step)
- `eval/reward`, `eval/service_level` (đánh giá định kỳ theo chiến lược tham lam)

---

## Yêu cầu phần cứng GPU

- NVIDIA RTX 3050 4GB VRAM ✅
- Batch size: 128
- Replay Buffer: 100,000 bước chuyển trạng thái (~500MB RAM)
- Thời gian huấn luyện 500 tập: ~10–15 phút
