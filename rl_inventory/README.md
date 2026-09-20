# Tối ưu hóa chính sách đặt hàng lại trong quản lý tồn kho đa kho bằng IPPO

Môi trường Gymnasium đa kho, đa SKU trên dữ liệu M5 Walmart, huấn luyện bằng
**Independent Multi-Agent PPO với chia sẻ tham số**, so sánh đối chứng với EOQ,
(s,S) và Newsvendor.

> **Đây là bản v3.** Bản v2 đã sửa các lỗi khiến huấn luyện không hội tụ (xem
> [`BAO_CAO_SUA_LOI.md`](BAO_CAO_SUA_LOI.md)). Bản v3 thêm: chia dữ liệu 3
> miền train/val/test (thay vì 2 miền), và chi phí thiếu hàng theo **giá bán
> thật** của M5 (thay vì hằng số). Xem mục [Thay đổi trong bản v3](#thay-đổi-trong-bản-v3)
> ở cuối file — **cần huấn luyện lại** trước khi dùng số liệu cho báo cáo.

---

## Cài đặt

```bash
pip install -r requirements.txt
python run.py check          # kiểm tra thư viện và dữ liệu đã sẵn sàng chưa
```

Dữ liệu M5 gốc đặt tại `data/raw/`: `sales_train_evaluation.csv`,
`calendar.csv`, và `sell_prices.csv` (dùng cho chi phí thiếu hàng theo giá
thật — xem bên dưới). Ba file này bị lược khỏi gói tải về vì nặng, bạn copy từ
bản gốc trên Kaggle (hoặc bản cũ) sang là được.

## Chạy — cách đơn giản nhất

```bash
python run.py app
```

Mở bảng điều khiển Streamlit, làm được mọi thứ bằng nút bấm và xem trực quan
từng bước. Nếu thích dòng lệnh:

```bash
python run.py all            # chạy trọn bộ 6 bước
python run.py all --quick    # bản rút gọn khoảng 5 phút, để thử đường ống
```

hoặc từng bước:

| Lệnh | Việc | Thời gian (1 nhân CPU) |
|---|---|---|
| `python run.py data` | Tiền xử lý M5 (cầu + lịch + **giá thật**) vào `data/processed/` | khoảng 1-2 phút |
| `python run.py baseline` | Tìm kiếm lưới tham số EOQ, (s,S), Newsvendor | khoảng 2 phút |
| `python run.py train` | Huấn luyện IPPO | khoảng 78 phút cho 5000 episode |
| `python run.py eval` | Đánh giá và so sánh, xuất bảng và biểu đồ | khoảng 3 phút |
| `python run.py iso` | **So sánh ở cùng mức phục vụ** — bảng nên dùng cho Chương 4 | khoảng 4 phút |
| `python run.py summary` | Xuất `TONG_HOP_SO_LIEU.txt` tự động từ `config.yaml` + `results/*` | vài giây |
| `python run.py test` | Unit test môi trường | vài giây |

Mọi tham số nằm ở **`config.yaml`**, không cần sửa code để chạy thí nghiệm.

---

## App Streamlit

Có **2 app riêng biệt**, mục đích khác nhau:

### 1. Demo cho hội đồng (`app/app.py`, `python run.py app`)

```bash
streamlit run app/app.py
```

4 trang (Streamlit tự nhận trong `app/pages/`):

| Trang | Nội dung |
|---|---|
| Tổng quan mạng lưới kho | Slider chọn ngày mô phỏng, heatmap tồn kho kho × SKU (đỏ = thiếu hàng, vàng = tồn dư), KPI fill rate/chi phí lũy kế/số lượt thiếu hàng |
| Đề xuất đặt hàng | Nhập 1 tình huống (kho, SKU thật hoặc tự giả định, tồn hiện tại, dự báo cầu) → IPPO và 3 baseline đề xuất đặt bao nhiêu; có thể áp dụng đề xuất rồi chạy tiếp để xem diễn biến |
| So sánh policy | Chạy IPPO + 3 baseline trên **cùng một chuỗi cầu** (cùng seed) — đường tồn kho, tổng chi phí, phân rã chi phí |
| What-if | Đổi lead time / chi phí lưu kho / sức chứa, hoặc tạo cú sốc cầu (nhân toàn chuỗi cầu với 1 hệ số) — so sánh chính sách nào giữ được fill rate tốt hơn khi bị sốc |

Backend dùng chung: `app/backend/simulator.py` (bọc `MultiWarehouseInventoryEnv`,
chạy 1 hoặc nhiều chính sách song song trên cùng seed) và
`app/backend/policy_runner.py` (nạp IPPO từ checkpoint + 3 baseline thành
cùng một giao diện `action_fn(obs, env) -> action`).

**Lưu ý trung thực khi demo** (trang chủ có nhắc lại): so tổng chi phí giữa
các chính sách có fill rate khác nhau là không công bằng — luôn nhìn cột
fill rate cạnh chi phí. Kết quả chính thức (trung bình nhiều episode, kiểm
định thống kê) nằm ở `results/` và `TONG_HOP_SO_LIEU.txt`, không phải số
liệu từ 1 lần chạy demo.

### 2. Bảng điều khiển vận hành dự án (`app/streamlit_app.py`)

```bash
streamlit run app/streamlit_app.py
```

Dùng khi **làm khóa luận** (không phải lúc demo): sửa `config.yaml` bằng
form, bấm nút chạy từng bước pipeline, xem đường học huấn luyện theo thời
gian thực, và tab "Giả lập tay" để tự đặt hàng từng ngày cho 1 cặp kho-SKU.

| Tab | Nội dung |
|---|---|
| Dữ liệu | Cầu M5 sau tiền xử lý: phân bố quy mô của 300 cặp, cầu từng kho, sức chứa tương ứng, tỷ lệ ngày bằng 0, chuỗi thời gian |
| Cấu hình và Chạy | Sửa tham số bằng form rồi lưu `config.yaml`; các nút chạy từng bước, log hiện trực tiếp |
| Huấn luyện | Đường học đọc từ `results/train_log*.csv`: reward, fill rate, entropy, explained variance, cơ cấu chi phí, approx_kl — mỗi chỉ số có ví dụ tính tay |
| So sánh | Bảng IPPO đối chứng 3 baseline, kiểm định Welch t-test, Wilcoxon, Cohen's d, biểu đồ đánh đổi chi phí và mức phục vụ, và bảng so sánh ở cùng mức phục vụ |
| Mô phỏng | Chiếu lại 365 ngày của một chính sách bất kỳ theo ngày: tồn kho đối chiếu cầu, thiếu hàng đối chiếu lượng đặt, mức sử dụng sức chứa từng kho, fill rate cửa sổ trượt, chi phí mỗi ngày |
| Giả lập tay | Tự tay đặt hàng từng ngày cho 1 cặp kho-SKU (dữ liệu thật hoặc giả lập Poisson), xem ngay tồn kho/chi phí cập nhật — để cảm nhận cơ chế trước khi nhìn cả 300 cặp chạy tự động |

Tab Mô phỏng là chỗ trả lời câu "chuyện gì đang xảy ra". Nếu đường mức sử dụng
sức chứa ép sát 1,0 liên tục thì ràng buộc đang cắn — đó là lúc 30 SKU trong
cùng một kho thực sự tranh nhau tài nguyên, tức bài toán đúng là đa tác tử chứ
không phải 300 bài toán độc lập ghép lại.

---

## Cấu trúc

```
rl_inventory/
├── run.py                        <- một cửa duy nhất cho mọi lệnh
├── config.yaml                   <- toàn bộ tham số và các cờ ablation
├── BAO_CAO_SUA_LOI.md            <- chẩn đoán, bằng chứng số cho bản v2
├── app/
│   ├── app.py                    <- demo 4 trang cho hội đồng (python run.py app)
│   ├── pages/                    <- 1_Tong_quan, 2_De_xuat_dat_hang, 3_So_sanh_policy, 4_What_if
│   ├── backend/                  <- simulator.py + policy_runner.py (dung chung 4 trang)
│   └── streamlit_app.py          <- bảng điều khiển vận hành dự án (sửa config, chạy pipeline)
├── env/inventory_env.py          <- môi trường Gymnasium
├── agents/
│   ├── ppo_agent.py              <- actor và critic TÁCH RIÊNG, value normalization
│   └── rollout_buffer.py         <- GAE chuẩn, per-pair advantage normalization
├── baselines/traditional_policies.py
├── scripts/
│   ├── data_preprocessing.py     <- M5 hoặc synthetic: cầu, lịch, GIÁ THẬT, lọc SKU
│   ├── tune_baselines.py         <- tìm kiếm lưới tham số baseline (trên miền train)
│   ├── train.py                  <- vòng lặp huấn luyện, chọn checkpoint bằng miền VAL
│   ├── evaluate.py               <- đánh giá trên miền TEST, kiểm định thống kê
│   ├── iso_service.py            <- so sánh ở cùng mức phục vụ (chống phản biện)
│   └── generate_summary.py       <- xuất TONG_HOP_SO_LIEU.txt tự động, không lệch số
├── tests/test_env.py             <- test hồi quy cho từng lỗi/ tính năng đã thêm
├── data/{raw,processed}/
└── checkpoints/  results/  runs/
```

---

## Kiến trúc: 300 "tác tử" nhưng chỉ 1 mạng neural

10 kho × 30 SKU = 300 cặp (kho, SKU), mỗi cặp là một tác tử tự quyết định mỗi
ngày đặt hàng bao nhiêu. Đây **không phải 300 mạng riêng biệt** — toàn bộ 300
cặp dùng **chung một bộ trọng số Actor-Critic** (parameter sharing). Mỗi bước,
quan sát của 300 cặp được xếp thành một batch `(300, obs_per_pair)`, đi qua
mạng một lần duy nhất.

Sự "kết hợp" giữa các tác tử xảy ra qua đúng 2 kênh:

1. **Chia sẻ gradient**: mọi cặp học chung một bộ trọng số.
2. **Tín hiệu quan sát cấp kho**: mỗi cặp biết kho của nó đang đầy bao nhiêu %
   sức chứa và nó chiếm bao nhiêu % trong đó — kênh duy nhất để một cặp "cảm
   nhận" áp lực từ 29 cặp còn lại cùng kho (30 SKU tranh nhau 1 sức chứa
   chung). Không có giao tiếp trực tiếp giữa các tác tử.

GAE (advantage) được tính **độc lập theo từng cặp** theo trục thời gian — đây
là chữ "Independent" trong IPPO: độc lập về *return*, chia sẻ *tham số*.

Luồng dữ liệu đầy đủ:

```
data_preprocessing.py -> demand_data.npy, calendar_features.npy, price_per_pair.npy
        v
env/inventory_env.py: moi step() xu ly vector hoa ca 300 cap cung luc
        v
agents/ppo_agent.py: 1 mang forward 1 lan cho ca 300 cap (SharedActorCriticNetwork)
        v
agents/rollout_buffer.py: GAE doc lap theo cap -> lam PHANG thanh 1 batch lon
        v
scripts/train.py: vong lap huan luyen, chon best_model bang mien VAL
        v
scripts/evaluate.py + iso_service.py: danh gia tren mien TEST, so voi baseline
```

---

## Chia dữ liệu: train / val / test (không rò rỉ)

`config.yaml` có 2 mốc ngày, cắt trên cùng một chuỗi `demand_data` (1941 ngày
của M5):

```
train: [0, split_day)          -> huan luyen
val:   [split_day, val_day)    -> chon best_model TRONG LUC huan luyen
test:  [val_day, het du lieu)  -> CHI dung 1 lan de bao cao ket qua cuoi
```

Mặc định: `split_day: 1050`, `val_day: 1450` → **test = `[1450, 1941)`, giữ
nguyên đúng miền mà bản v2 đã dùng**, để các kết quả trước đó vẫn có thể so
sánh trên cùng một tập test. Miền val được cắt ra từ phần trước đó vốn là
"train" của bản v2.

Ba miền không bao giờ chồng lấn: `env/inventory_env.py::reset()` giới hạn
`start_day` sao cho cả episode 365 ngày luôn nằm gọn trong miền của nó (không
"tràn" sang miền kế bên). Ước lượng `mean_demand`, sức chứa kho, bảng mức đặt
hàng, và giá trung bình mỗi cặp đều **chỉ tính từ miền train**.

**Tại sao cần thêm miền val**: bản v2 chọn `best_model.pth` bằng cách đánh giá
định kỳ **trên chính miền test** (chỉ khác seed) — tức là việc chọn checkpoint
đã "nhìn thấy" hiệu năng trên đúng dữ liệu sau này dùng để báo cáo và so sánh
với baseline. Bản v3 tách hẳn: `scripts/train.py` dùng miền **val** để chọn
checkpoint, `scripts/evaluate.py`/`iso_service.py` dùng miền **test** — chỉ
chạm vào đúng 1 lần, ở cuối cùng.

---

## Chi phí thiếu hàng theo giá thật (tùy chọn)

Mặc định `env.cp_th = 10.0` là **hằng số dùng chung cho cả 300 cặp**, bất kể
SKU đó bán 50 xu hay 100 đô. `config.yaml` có tùy chọn dùng **giá bán thật**
của M5 (`sell_prices.csv`) để mỗi cặp có đơn giá thiếu hàng riêng:

```yaml
use_real_price_stockout: true   # false -> quay ve hang so cp_th cu
margin_ratio: 0.3                # gia dinh: thieu 1 don vi = mat 30% gia ban
cp_th_min:    0.5                # san toi thieu, tranh SKU re tien ve gan 0
```

Công thức: `cp_th_pair = max(giá_bán_trung_bình(cặp) * margin_ratio, cp_th_min)`,
với giá trung bình chỉ ước lượng **trên miền train** (tránh rò rỉ, giống cách
ước lượng `mean_demand`). `cp_th_pair` thay thế `cp_th` trong cả công thức chi
phí thiếu hàng lẫn phạt mức phục vụ.

**Giả định kinh tế cần nêu rõ trong báo cáo**: M5 không công bố giá vốn, nên
`margin_ratio` là một giả định (ở đây: thiếu hàng làm mất đúng phần lợi nhuận
biên, không phải toàn bộ giá bán). Đây là điểm phương pháp luận, không phải số
đo được — cần trình bày như một giả định tường minh, không phải sự thật.

**Vì sao không cần thiết kế lại hàm thưởng khi có SKU mắc/rẻ lẫn lộn**: cơ chế
chuẩn hóa phần thưởng theo từng cặp đã có sẵn từ bản v2 (`reward_norm_pair =
cp_th_pair * mean_demand + cp_dh`) để xử lý đúng vấn đề này — SKU mắc tiền có
mẫu số chuẩn hóa lớn hơn tương ứng nên vẫn được quy về cùng thang đo với SKU rẻ
tiền, không cần sửa kiến trúc.

Nguồn dữ liệu giá: `scripts/data_preprocessing.py` đọc `sell_prices.csv`
(giá theo tuần) + `calendar.csv` (map tuần → ngày) để tính giá trung bình mỗi
cặp trên các tuần thuộc miền train, lưu vào `data/processed/price_per_pair.npy`.

---

## Ablation — đổi giá trị trong `config.yaml` rồi chạy lại

| Cờ | Ý nghĩa |
|---|---|
| `env.normalize_reward_per_pair` | **Quan trọng nhất.** Đặt `false` sẽ tái hiện đúng hiện tượng không hội tụ của bản v1 |
| `env.use_real_price_stockout` | Bật/tắt chi phí thiếu hàng theo giá thật (v3) so với hằng số `cp_th` |
| `env.margin_ratio`, `env.cp_th_min` | Độ nhạy của giả định kinh tế khi bật giá thật |
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

1. *IPPO có tốt hơn EOQ, (s,S), Newsvendor không?*
2. *Parameter sharing có tổng quát hóa được qua các quy mô cầu không?*
3. *Phần thưởng phân rã cục bộ có giúp học tốt hơn thưởng tổng thể không?*

`results/`, `TONG_HOP_SO_LIEU.txt` có số liệu của bản **trước v3** (hằng số
`cp_th`, chia 2 miền train/test cũ). Sau khi thêm giá thật + chia lại dữ liệu,
**cần chạy lại `python run.py train` rồi `python run.py eval && python run.py
iso && python run.py summary`** để có số liệu khớp với cấu hình hiện tại — chi
phí thiếu hàng đã giảm mạnh về độ lớn (giá trung bình M5 × 30% margin ≈ 1,2,
thay vì hằng số 10 cũ), nên checkpoint cũ (`best_model.pth`) được huấn luyện
dưới thang chi phí cũ, không còn tối ưu cho thang chi phí mới.

---

## Thay đổi trong bản v3

So với bản v2 (mô tả trong `BAO_CAO_SUA_LOI.md`):

- **[V3-1] Chia 3 miền train/val/test** thay vì 2 miền. Bản v2 chọn
  `best_model.pth` bằng hiệu năng trên chính miền test — một dạng rò rỉ nhẹ
  trong quy trình đánh giá. Nay có miền val riêng
  (`env/inventory_env.py::reset()`, `config.yaml: val_day`), miền test giữ
  nguyên phạm vi cũ để không mất khả năng so sánh với kết quả đã có.
- **[V3-2] Chi phí thiếu hàng theo giá bán thật** (tùy chọn, mặc định bật) từ
  `sell_prices.csv` của M5, thay vì hằng số `cp_th` dùng chung. Xem mục
  [Chi phí thiếu hàng theo giá thật](#chi-phí-thiếu-hàng-theo-giá-thật-tùy-chọn).
- `scripts/data_preprocessing.py`: thêm đọc `sell_prices.csv`, xuất
  `data/processed/price_per_pair.npy`; đổi mặc định `--split_day` từ 1450
  xuống 1050 (khớp `config.yaml`).
- `run.py`: lệnh `data` nay truyền `--split_day` từ `config.yaml` thay vì để
  `data_preprocessing.py` dùng mặc định riêng — trước đây 2 nơi có thể lệch
  nhau âm thầm mà không ai biết.
- `demo_app/` (thư mục ngoài `rl_inventory/`) đã bị **xóa**: import
  `agents.dqn_agent.DoubleDQNAgent` không tồn tại, tàn dư kiến trúc Double DQN
  đời trước, không tương thích với `env/inventory_env.py` hiện tại.
- **Việc còn lại**: `results/`, `checkpoints/`, `TONG_HOP_SO_LIEU.txt` hiện
  tại vẫn là số liệu **trước** khi thêm v3 — cần chạy lại toàn bộ pipeline
  (mục "Ba câu hỏi nghiên cứu" ở trên) trước khi dùng cho báo cáo. Đồng thời
  `Report KLTN/content/*.tex` vẫn mô tả cấu hình đời v1 (500 tác tử, sức chứa
  dùng chung) và chưa được cập nhật theo v2/v3.

---

## Tham khảo

1. Schulman và cộng sự (2017), *Proximal Policy Optimization Algorithms*
2. Yu và cộng sự (NeurIPS 2022), *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*
3. Harris (1913) cho EOQ; Scarf (1960) cho (s,S); Arrow, Harris và Marschak (1951) cho Newsvendor
4. Wolpert và Tumer (2001), *Optimal Payoff Functions for Members of Collectives* — cơ sở cho cách phân bổ trách nhiệm tràn kho về từng SKU
