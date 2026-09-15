# Tối ưu hóa chính sách đặt hàng lại trong quản lý tồn kho đa kho bằng IPPO

Môi trường Gymnasium đa kho, đa SKU trên dữ liệu M5 Walmart, huấn luyện bằng
**Independent Multi-Agent PPO với chia sẻ tham số**, so sánh đối chứng với EOQ,
(s,S) và Newsvendor.

> **Đây là bản v2.** Bản trước không hội tụ. Nguyên nhân và bằng chứng số nằm
> trong [`BAO_CAO_SUA_LOI.md`](BAO_CAO_SUA_LOI.md) — nên đọc file đó trước.

---

## Cài đặt

```bash
pip install -r requirements.txt
python run.py check          # kiểm tra thư viện và dữ liệu đã sẵn sàng chưa
```

Dữ liệu M5 gốc đặt tại `data/raw/`: `sales_train_evaluation.csv` và
`calendar.csv`. Hai file này bị lược khỏi gói tải về vì nặng khoảng 130 MB, bạn
copy từ bản cũ sang là được.

## Chạy — cách đơn giản nhất

```bash
python run.py app
```

Mở bảng điều khiển Streamlit, làm được mọi thứ bằng nút bấm và xem trực quan
từng bước. Nếu thích dòng lệnh:

```bash
python run.py all            # chạy trọn bộ 5 bước
python run.py all --quick    # bản rút gọn khoảng 5 phút, để thử đường ống
```

hoặc từng bước:

| Lệnh | Việc | Thời gian (1 nhân CPU) |
|---|---|---|
| `python run.py data` | Tiền xử lý M5 vào `data/processed/` | khoảng 1 phút |
| `python run.py baseline` | Tìm kiếm lưới tham số EOQ, (s,S), Newsvendor | khoảng 2 phút |
| `python run.py train` | Huấn luyện IPPO 800 episode | khoảng 35 phút |
| `python run.py eval` | Đánh giá và so sánh, xuất bảng và biểu đồ | khoảng 3 phút |
| `python run.py iso` | **So sánh ở cùng mức phục vụ** — bảng nên dùng cho Chương 4 | khoảng 4 phút |
| `python run.py test` | 22 unit test môi trường | vài giây |

Mọi tham số nằm ở **`config.yaml`**, không cần sửa code để chạy thí nghiệm.

---

## Bảng điều khiển Streamlit

```bash
streamlit run app/streamlit_app.py
```

| Tab | Nội dung |
|---|---|
| Dữ liệu | Cầu M5 sau tiền xử lý: phân bố quy mô của 300 cặp, cầu từng kho, sức chứa tương ứng, tỷ lệ ngày bằng 0, chuỗi thời gian |
| Cấu hình và Chạy | Sửa tham số bằng form rồi lưu `config.yaml`; 5 nút chạy từng bước, log hiện trực tiếp |
| Huấn luyện | Đường học đọc từ `results/train_log*.csv`: reward, fill rate, **entropy**, **explained variance**, cơ cấu chi phí, approx_kl. Có tự động làm mới 10 giây một lần để xem trong lúc đang chạy |
| So sánh | Bảng IPPO đối chứng 3 baseline, kiểm định Welch t-test, Wilcoxon, Cohen's d, biểu đồ đánh đổi chi phí và mức phục vụ, **và bảng so sánh ở cùng mức phục vụ** |
| Mô phỏng | **Chiếu lại 365 ngày** của một chính sách bất kỳ, có thanh trượt theo ngày: tồn kho đối chiếu cầu, thiếu hàng đối chiếu lượng đặt, **mức sử dụng sức chứa từng kho** (thấy rõ lúc nào hàng bị từ chối nhập), fill rate cửa sổ trượt, chi phí mỗi ngày. Tải được CSV |

Tab Mô phỏng chính là chỗ trả lời câu "chuyện gì đang xảy ra". Nếu đường mức sử
dụng sức chứa ép sát 1,0 liên tục thì ràng buộc đang cắn — đó là lúc 30 SKU
trong cùng một kho thực sự tranh nhau tài nguyên, tức bài toán đúng là đa tác tử
chứ không phải 300 bài toán độc lập ghép lại.

---

## Cấu trúc

```
KhoaLuan_RL_Inventory/
├── run.py                        <- một cửa duy nhất cho mọi lệnh
├── config.yaml                   <- toàn bộ tham số và các cờ ablation
├── BAO_CAO_SUA_LOI.md            <- chẩn đoán, bằng chứng số, việc còn lại
├── app/streamlit_app.py          <- bảng điều khiển trực quan
├── env/inventory_env.py          <- môi trường Gymnasium
├── agents/
│   ├── ppo_agent.py              <- actor và critic TÁCH RIÊNG, value normalization
│   └── rollout_buffer.py         <- GAE chuẩn, per-pair advantage normalization
├── baselines/traditional_policies.py
├── scripts/
│   ├── data_preprocessing.py     <- M5 hoặc synthetic, lọc SKU, calendar features
│   ├── tune_baselines.py         <- tìm kiếm lưới tham số baseline
│   ├── train.py                  <- vòng lặp huấn luyện, ghi train_log.csv
│   ├── evaluate.py               <- đánh giá, kiểm định thống kê, xuất traces
│   └── iso_service.py            <- so sánh ở cùng mức phục vụ (chống phản biện)
├── tests/test_env.py             <- 22 test, gồm test hồi quy cho từng lỗi đã sửa
├── data/{raw,processed}/
└── checkpoints/  results/  runs/
```

---

## Ablation — đổi giá trị trong `config.yaml` rồi chạy lại

| Cờ | Ý nghĩa |
|---|---|
| `env.normalize_reward_per_pair` | **Quan trọng nhất.** Đặt `false` sẽ tái hiện đúng hiện tượng không hội tụ của bản cũ |
| `env.capacity_cover_days` | Độ chặt của ràng buộc ghép nối giữa các tác tử |
| `env.phi_dv` | Độ nhạy của hệ số phạt mức phục vụ |
| `ppo.normalize_adv_per_pair` | Chuẩn hóa advantage theo từng cặp hay toàn cục |
| `ppo.use_value_norm` | Bật tắt value normalization |
| `ppo.anneal_ent` | Bật tắt entropy annealing |
| `env.use_relative_orders` | Mức đặt hàng tương đối theo cầu hay tuyệt đối |

Mỗi lần chỉ đổi một cờ, giữ nguyên seed, dùng `--tag` để tách nhật ký:

```bash
python scripts/train.py --seed 1 --tag khong_chuan_hoa_reward
```

---

## Ba câu hỏi nghiên cứu — trạng thái

1. *IPPO có tốt hơn EOQ, (s,S), Newsvendor không?* — Kết quả 173 episode: so
   thẳng thì IPPO đắt hơn 11,4% nhưng phục vụ cao hơn 8 điểm (92,1% so với
   84,1%). **So ở cùng mức phục vụ** (`python run.py iso`) thì IPPO **rẻ hơn
   (s,S) 11,7%, rẻ hơn Newsvendor 5,0%, và EOQ không đạt nổi mức phục vụ đó**.
   Đây mới là bảng đúng để đưa vào Chương 4, xem Mục 8 của báo cáo sửa lỗi.

2. *Parameter sharing có tổng quát hóa được qua các quy mô cầu không?* — quan
   sát đã có đặc trưng quy mô `log1p(mean_demand)` và thang "số ngày tồn kho"
   bất biến theo quy mô. Kiểm chứng bằng cách tách một nhóm SKU hold-out.

3. *Phần thưởng phân rã cục bộ có giúp học tốt hơn thưởng tổng thể không?* —
   chạy nhánh đối chứng, hướng dẫn trong `BAO_CAO_SUA_LOI.md`.

---

## Tham khảo

1. Schulman và cộng sự (2017), *Proximal Policy Optimization Algorithms*
2. Yu và cộng sự (NeurIPS 2022), *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*
3. Harris (1913) cho EOQ; Scarf (1960) cho (s,S); Arrow, Harris và Marschak (1951) cho Newsvendor
4. Wolpert và Tumer (2001), *Optimal Payoff Functions for Members of Collectives* — cơ sở cho cách phân bổ trách nhiệm tràn kho về từng SKU
