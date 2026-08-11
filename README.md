# Quản lý Tồn kho Đa Kho bằng Học sâu Tăng cường (RL)
> Khóa luận tốt nghiệp — Tối ưu Quản lý Tồn kho Đa Kho bằng Deep Reinforcement Learning

## Cấu trúc thư mục

```
D:\rl_inventory\
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
pip install -r requirements.txt
```

---

## Chạy nhanh (Dùng dữ liệu nhu cầu giả lập)

```bash
# 1. Tạo dữ liệu nhu cầu giả lập (Synthetic demand data)
python scripts/data_preprocessing.py --synthetic

# 2. Huấn luyện Double DQN (RTX 3050, 500 episodes ~10-15 phút)
python scripts/train.py --episodes 500

# 3. Đánh giá và so sánh mô hình với các phương pháp Baseline
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth

# 4. Theo dõi tiến trình bằng TensorBoard (mở ở cửa sổ terminal khác)
tensorboard --logdir runs/
```

---

## Chạy với Bộ dữ liệu M5 (Walmart Kaggle)

```bash
# 1. Tải bộ dữ liệu M5 từ Kaggle (sử dụng kaggle CLI hoặc trình duyệt)
# https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
# Đặt các tệp vào data/raw/:
#   - sales_train_evaluation.csv
#   - calendar.csv

# 2. Tiền xử lý dữ liệu (2 kho CA_1 + TX_1, 30 SKU mỗi kho)
python scripts/data_preprocessing.py --stores CA_1 TX_1 --n_skus 30

# 3. Huấn luyện tác tử
python scripts/train.py --episodes 500

# 4. Đánh giá mô hình
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
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
