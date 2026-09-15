# Khảo Sát Tài Liệu & Xác Định Phạm Vi Khóa Luận

## Tên Đề Tài (Đề Xuất)

> **"Tối Ưu Hóa Chính Sách Đặt Hàng Lại Trong Quản Lý Tồn Kho Đa Kho Bằng Học Tăng Cường: Thuật Toán Independent Multi-Agent PPO"**

---

## 1. Bối Cảnh & Vấn Đề Nghiên Cứu

### 1.1. Bài Toán Thực Tiễn

Quản lý tồn kho trong chuỗi cung ứng đa điểm (multi-echelon) đặt ra bài toán **tối ưu hóa cân bằng đồng thời** giữa 5 nhóm chi phí xung đột nhau:

| Chi phí | Ký hiệu | Chiều tác động |
|---|---|---|
| Lưu kho (Holding Cost) | $c_{lk} = 1.0$/đvị/ngày | ↑ khi tồn kho cao |
| Thiếu hàng (Stockout Cost) | $c_{th} = 10.0$/đvị | ↑ khi hết hàng |
| Đặt hàng cố định (Ordering Cost) | $c_{dh} = 50.0$/lần đặt | ↑ theo tần suất đặt |
| Tràn kho (Overflow Penalty) | $p_{tk} = 5.0$/đvị tràn | ↑ khi vượt sức chứa |
| Vi phạm mức phục vụ | $\phi_{dv} = 1.5$ | ↑ khi fill rate < 85% |

Bài toán cực kỳ phức tạp do:
- Không gian hành động là **tích Descartes** của 500 cặp (kho × SKU)
- Nhu cầu ngẫu nhiên + thời gian giao hàng ngẫu nhiên (1–3 ngày)
- Các chiến lược truyền thống (EOQ, (s,S), Newsvendor) không tối ưu hóa đồng thời tất cả chi phí trên

---

## 2. Tổng Quan Tài Liệu Liên Quan

### 2.1. Phương Pháp Truyền Thống (Baselines)

| Phương pháp | Tác giả & Năm | Công thức | Hạn chế |
|---|---|---|---|
| **EOQ** (Economic Order Quantity) | Harris, 1913 | $Q^* = \sqrt{\frac{2 D c_{dh}}{c_{lk}}}$ | Giả định nhu cầu xác định, không có stockout |
| **(s, S) Policy** | Scarf, 1960 | $s = \mu_L + z\sigma_L$; đặt lên $S$ khi $I_t \le s$ | Tham số tĩnh, không thích ứng |
| **Newsvendor Model** | Arrow et al., 1951 | $Q^* = F^{-1}\left(\frac{c_{th}}{c_{th}+c_{lk}}\right)$ | Chỉ tối ưu 1 kỳ, không có pipeline |

> [!NOTE]
> Ba phương pháp trên **không thể** xử lý đồng thời: nhu cầu stochastic, lead time ngẫu nhiên, ràng buộc fill rate, và phạt tràn kho trong hệ thống đa kho–đa SKU.

### 2.2. Học Tăng Cường Áp Dụng Vào Tồn Kho

| Tác giả & Năm | Đóng góp |
|---|---|
| Schulman et al. (2017) | **PPO** – Proximal Policy Optimization, thuật toán RL on-policy hiện đại, ổn định, hiệu quả mẫu |
| Yu et al. (NeurIPS 2022) | **IPPO** – Independent Multi-Agent PPO, hiệu quả bất ngờ trong môi trường cooperative MARL |

### 2.3. Dữ Liệu Thực Tế

- **M5 Walmart Dataset**: Dữ liệu bán lẻ thực tế từ 10 cửa hàng Walmart (CA_1..4, TX_1..3, WI_1..3) với 50 SKU mỗi cửa hàng, bao gồm thông tin lịch, ngày lễ (SNAP events).

---

## 3. Phạm Vi Khóa Luận

### 3.1. Phạm Vi TRONG đề tài

```
┌─────────────────────────────────────────────────────────────────┐
│                     PHẠM VI NGHIÊN CỨU                         │
│                                                                 │
│  ✅ Xây dựng môi trường Gymnasium tùy chỉnh                    │
│     • N=10 kho × M=50 SKU → 500 agents đồng thời              │
│     • Nhu cầu stochastic (M5 Walmart hoặc Poisson Synthetic)   │
│     • Lead time ngẫu nhiên [1, 3] ngày                         │
│     • Hàm thưởng phân rã cục bộ (local reward decomposition)  │
│     • Rolling-window fill rate per-pair (30 ngày)              │
│                                                                 │
│  ✅ Thuật toán IPPO với Parameter Sharing                       │
│     • 1 mạng SharedActorCritic dùng chung cho 500 cặp          │
│     • GAE + per-pair advantage normalization                    │
│     • Sample efficiency: 4096×500 = ~2M mẫu/update            │
│                                                                 │
│  ✅ So sánh đối chứng với 3 Baselines truyền thống             │
│     • EOQ, (s,S) Policy, Newsvendor Model                      │
│                                                                 │
│  ✅ Đánh giá trên 4 chỉ số                                     │
│     • Tổng chi phí | Fill Rate | Holding/Stockout/Ordering     │
│                                                                 │
│  ✅ Hỗ trợ dữ liệu thực tế (M5 Walmart) + dữ liệu giả lập    │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2. Phạm Vi NGOÀI đề tài (Không xét)

- ❌ Multi-echelon (quan hệ cung cấp giữa các kho) — hiện tại mỗi kho độc lập
- ❌ Tối ưu hóa logistics / vận chuyển giữa các kho
- ❌ Dự báo nhu cầu bằng DL (LSTM, Transformer) — hiện dùng lịch sử trực tiếp
- ❌ Hệ thống thời gian thực / triển khai production
- ❌ Multi-Agent Communication (không có kênh giao tiếp giữa các agents)

---

## 4. Cấu Trúc Mã Nguồn Hiện Tại

```
KhoaLuan_RL_Inventory/
├── env/inventory_env.py        ← Môi trường Gymnasium (MultiWarehouseInventoryEnv)
├── agents/
│   ├── ppo_agent.py            ← PPOAgent + SharedActorCriticNetwork
│   └── rollout_buffer.py       ← RolloutBuffer (GAE, per-pair normalization)
├── baselines/traditional_policies.py  ← EOQ, (s,S), Newsvendor
├── scripts/
│   ├── data_preprocessing.py   ← Tiền xử lý M5 / Synthetic data
│   ├── train.py                ← Vòng lặp huấn luyện
│   └── evaluate.py             ← Đánh giá & vẽ biểu đồ so sánh
└── config.yaml                 ← Tham số tập trung
```

---

## 5. Các Thông Số Cấu Hình Chính

| Nhóm | Tham số | Giá trị hiện tại |
|---|---|---|
| **Môi trường** | n_warehouses × n_skus | 10 × 50 = **500 cặp** |
| | episode_length | **365 ngày** (1 năm) |
| | order_levels | {0, 10, 20, 30, 40, 50} |
| | lead_time | [1, 3] ngày (ngẫu nhiên) |
| | muc_dv (fill rate target) | **85%** |
| **PPO** | n_steps (rollout) | 4096 bước |
| | ppo_epochs | 6 |
| | lr_actor / lr_critic | 3e-5 / 3e-4 |
| | gamma / gae_lambda | 0.99 / 0.95 |
| | total_episodes | 6000 |
| **Mạng Neural** | hidden_dim | 128 (2 lớp ẩn) |
| | Activation | ReLU |

---

## 6. Câu Hỏi Nghiên Cứu

> [!IMPORTANT]
> Khóa luận cần trả lời rõ 3 câu hỏi chính:

1. **Câu hỏi 1**: Liệu IPPO có học được chính sách đặt hàng tốt hơn các phương pháp tối ưu hóa cổ điển (EOQ, (s,S), Newsvendor) trong môi trường đa kho – đa SKU với nhu cầu ngẫu nhiên và lead time biến động?

2. **Câu hỏi 2**: Parameter sharing giữa 500 agents có đủ để học một chính sách tổng quát hóa cho các SKU có quy mô nhu cầu khác nhau không?

3. **Câu hỏi 3**: Hàm thưởng phân rã cục bộ (local reward decomposition) với rolling-window fill rate có cải thiện quá trình học so với hàm thưởng tổng thể không?

---

## 7. Kết Quả Kỳ Vọng & Chỉ Số Đánh Giá

| Chỉ số | Mô tả | Kỳ vọng IPPO |
|---|---|---|
| **Total Cost** | Tổng chi phí sau 365 ngày | Thấp nhất trong 4 phương pháp |
| **Fill Rate** | Tỷ lệ đáp ứng nhu cầu | ≥ 85% (ràng buộc thiết kế) |
| **Holding Cost** | Chi phí tồn kho | Thấp hơn EOQ (tránh dự trữ quá nhiều) |
| **Stockout Cost** | Chi phí mất hàng | Thấp hơn Newsvendor (đơn kỳ) |
| **Ordering Cost** | Chi phí đặt hàng | Cân bằng hợp lý |

---

## 8. Hướng Mở Rộng & Cải Tiến Tiềm Năng

> [!TIP]
> Những hướng có thể bổ sung trong quá trình thực hiện khóa luận:

- **Ablation study**: So sánh IPPO có/không có parameter sharing, có/không có per-pair normalization
- **Sensitivity analysis**: Thay đổi tỷ lệ chi phí $c_{lk}/c_{th}$ để xem IPPO thích ứng như thế nào
- **Visualization**: Vẽ learning curve, phân phối hành động của từng cặp (kho, SKU)
- **Thêm baseline**: Demand-Driven Planning, Min-Max Policy
- **Thử nghiệm với dữ liệu M5 đầy đủ**: 10 cửa hàng × 3,049 SKU gốc

---

## 9. Tài Liệu Tham Khảo Chính

1. **Yu et al. (NeurIPS 2022)** — *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games* — Nền tảng lý thuyết cho IPPO
2. **Schulman et al. (2017)** — *Proximal Policy Optimization Algorithms* — Thuật toán PPO gốc
3. **Harris (1913)** — *How many parts to make at once* — EOQ baseline
4. **Scarf (1960)** — *The optimality of (s, S) policies* — (s,S) baseline
5. **Arrow, Harris & Marschak (1951)** — *Optimal inventory policy* — Newsvendor baseline
