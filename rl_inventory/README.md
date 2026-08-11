# RL Multi-Warehouse Inventory Management
> Khóa luận tốt nghiệp — Tối ưu Quản lý Tồn kho Đa Kho bằng Deep Reinforcement Learning

## Cấu trúc thư mục

```
D:\rl_inventory\
├── data/
│   ├── raw/              ← Đặt file M5 ở đây (sales_train_evaluation.csv, calendar.csv)
│   └── processed/        ← Output: demand_data.npy, env_config.json
├── env/
│   └── inventory_env.py  ← Gymnasium environment
├── agents/
│   ├── dqn_agent.py      ← Double DQN agent
│   └── replay_buffer.py  ← Standard + Prioritized Experience Replay
├── baselines/
│   └── traditional_policies.py  ← EOQ, (s,S), Newsvendor
├── scripts/
│   ├── data_preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   └── colab_mvp.ipynb   ← End-to-end Colab notebook
├── checkpoints/          ← Saved model weights
├── results/              ← Charts & CSV
└── runs/                 ← TensorBoard logs
```

---

## Cài đặt

```bash
# 1. Tạo virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

# 2. Cài PyTorch với CUDA 11.8 (RTX 3050)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Cài dependencies còn lại
pip install -r requirements.txt
```

---

## Chạy nhanh (không cần M5 dataset)

```bash
# 1. Tạo synthetic demand data
python scripts/data_preprocessing.py --synthetic

# 2. Train DQN (RTX 3050, 500 episodes ~10-15 phút)
python scripts/train.py --episodes 500

# 3. Đánh giá và so sánh
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth

# 4. TensorBoard monitoring (mở tab khác)
tensorboard --logdir runs/
```

---

## Chạy với M5 Dataset (Walmart Kaggle)

```bash
# 1. Download M5 từ Kaggle (cần kaggle CLI hoặc browser)
# https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
# Đặt vào data/raw/:
#   - sales_train_evaluation.csv
#   - calendar.csv

# 2. Preprocessing (2 kho CA_1 + TX_1, 30 SKU mỗi kho)
python scripts/data_preprocessing.py --stores CA_1 TX_1 --n_skus 30

# 3. Train
python scripts/train.py --episodes 500 --use_m5

# 4. Evaluate
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
```

---

## Thiết kế chính (Thesis notes)

### Environment
| Thành phần | Chi tiết |
|---|---|
| State | Tồn kho + demand history (7 ngày) + pipeline + day-of-week |
| Action | Discrete 6 mức: {0, 10, 20, 30, 40, 50} units (factorized per SKU-pair) |
| Reward | -(holding + stockout + ordering cost) |
| Transition | Demand từ M5 thực tế, lead time U[1,3] ngày |

### Agent — Double DQN
| Component | Chi tiết |
|---|---|
| Network | MLP: 256→256→(n_pairs × 6) |
| Double DQN | van Hasselt et al. (2016) — tránh overestimation bias |
| Replay | Standard hoặc Prioritized (Schaul et al., 2016) |
| Target update | Soft Polyak averaging (τ=0.005) |
| Exploration | ε-greedy với exponential decay |
| Device | CUDA (RTX 3050 4GB) — batch_size=128 |

### Baselines (để so sánh trong thesis)
| Baseline | Reference |
|---|---|
| EOQ | Harris (1913) |
| (s,S) Policy | Scarf (1960) |
| Newsvendor | Arrow, Harris & Marschak (1951) |

---

## Metrics đánh giá
- **Total Cost**: Holding + Stockout + Ordering cost per episode
- **Service Level**: % demand được đáp ứng
- **Stockout Rate**: % demand bị thiếu hàng
- **Average Inventory**: Tồn kho trung bình

---

## TensorBoard

```bash
tensorboard --logdir runs/
# Mở browser: http://localhost:6006
```

Logs bao gồm:
- `episode/reward`, `episode/cost`, `episode/service_level` (theo episode)
- `train/loss`, `train/epsilon`, `train/avg_loss` (theo step)
- `eval/reward`, `eval/service_level` (greedy evaluation)

---

## GPU Requirements
- NVIDIA RTX 3050 4GB VRAM ✅
- Batch size: 128
- Buffer: 100K transitions (~500MB RAM)
- Training 500 episodes: ~10–15 phút
