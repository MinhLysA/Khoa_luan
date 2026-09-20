# Tối ưu hóa chính sách đặt hàng lại trong quản lý tồn kho đa kho bằng IPPO

Môi trường Gymnasium đa kho, đa SKU trên dữ liệu M5 Walmart, huấn luyện bằng
**Independent Multi-Agent PPO với chia sẻ tham số**, so sánh đối chứng với EOQ,
(s,S) và Newsvendor.

> **Đây là bản P0** (xem [`README_CLAUDE_CODE_KLTN.md`](README_CLAUDE_CODE_KLTN.md)
> — bản kế hoạch sửa lỗi/nâng cấp đầy đủ, P0 là nhóm bắt buộc sửa trước khi
> huấn luyện lại lần cuối). Bản v2 sửa các lỗi khiến huấn luyện không hội tụ
> (xem [`BAO_CAO_SUA_LOI.md`](BAO_CAO_SUA_LOI.md)); bản v3 thêm chia 3 miền
> train/val/test và chi phí thiếu hàng theo giá bán thật; bản P0 sửa **một lỗi
> tràn kho** (có thể vô tình xóa oan tồn kho cũ hợp lệ), đổi cách chọn
> checkpoint sang theo chi phí thấp nhất trong số các checkpoint đạt fill rate
> ≥ 85%, và đổi cách tính chi phí thiếu hàng về hằng số dùng chung cho thí
> nghiệm chính (tránh khẳng định số liệu là USD thật khi M5 không công bố giá
> vốn). Xem mục [Thay đổi trong bản P0](#thay-đổi-trong-bản-p0) để biết chi
> tiết đầy đủ.
>
> **QUAN TRỌNG: `results/`, `checkpoints/` và `Report KLTN/` hiện tại vẫn là
> số liệu TRƯỚC bản P0** (đã archive nguyên vẹn vào `results_pre_p0/` và
> `checkpoints_pre_p0/`). Cần chạy lại toàn bộ pipeline huấn luyện + đánh giá
> với code hiện tại rồi mới có số liệu khớp để cập nhật báo cáo — xem mục
> [Trạng thái báo cáo LaTeX](#trạng-thái-báo-cáo-latex).

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
│   ├── analyze_scale_groups.py   <- hậu nghiệm: fill rate/chi phí theo nhóm quy mô cầu (Câu hỏi 2)
│   ├── plot_learning_curve.py    <- ve duong cong hoc tu results/train_log*.csv
│   └── generate_summary.py       <- xuất TONG_HOP_SO_LIEU.txt tự động, không lệch số
├── tests/test_env.py             <- test hồi quy cho từng lỗi/ tính năng đã thêm
├── data/{raw,processed}/
├── checkpoints/  results/  runs/
└── checkpoints_pre_p0/  results_pre_p0/   <- archive ket qua TRUOC ban P0 (xem canh bao dau file)
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

## Chi phí thiếu hàng theo giá thật (tùy chọn, TẮT mặc định từ bản P0)

Mặc định `env.cp_th = 10.0` là **hằng số dùng chung cho cả 300 cặp**, bất kể
SKU đó bán 50 xu hay 100 đô. `config.yaml` có tùy chọn dùng **giá bán thật**
của M5 (`sell_prices.csv`) để mỗi cặp có đơn giá thiếu hàng riêng:

```yaml
use_real_price_stockout: false  # [P0-3] TAT mac dinh cho thi nghiem chinh -
                                 # xem ly do trong muc "Thay doi trong ban P0"
margin_ratio: 0.3                # gia dinh: thieu 1 don vi = mat 30% gia ban
cp_th_min:    0.5                # san toi thieu, tranh SKU re tien ve gan 0
```

**[P0-3] Vì sao tắt mặc định**: M5 không công bố giá vốn của từng mặt hàng,
nên `cp_th_pair` suy ra từ giá bán × `margin_ratio` là một **giả định**, không
phải số đo được. Gọi thẳng kết quả mô phỏng là chi phí "USD thật" của Walmart
khi dựa trên giả định này là một khẳng định quá mức. Thí nghiệm chính vì vậy
dùng hằng số `cp_th` chung (đơn vị chi phí mô phỏng chuẩn hóa), còn tín hiệu
giá theo ngày (`price_series`) vẫn được đưa vào quan sát của tác tử (chỉ báo
giảm giá) bất kể cờ này bật hay tắt. Bật `true` để chạy như một ablation
riêng, không dùng cho số liệu báo cáo chính.

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
| `env.use_real_price_stockout` | Bật/tắt chi phí thiếu hàng theo giá thật (v3); **tắt mặc định từ P0** cho thí nghiệm chính |
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

> ⚠️ **Số liệu dưới đây là kết quả TRƯỚC bản P0** (đã archive ở
> `results_pre_p0/`), giữ lại để tham khảo xu hướng. Cấu hình đã đổi thực sự
> từ bản P0 (sửa lỗi tràn kho, đổi chọn checkpoint theo chi phí thấp nhất
> trong nhóm đạt fill ≥ 85%, tắt chi phí thiếu hàng theo giá thật) nên **cần
> chạy lại pipeline với code hiện tại** để có số liệu chính thức — xem mục
> [Trạng thái báo cáo LaTeX](#trạng-thái-báo-cáo-latex).

1. *IPPO có tốt hơn EOQ, (s,S), Newsvendor không?* — **Có, khi so cùng mức
   phục vụ** (kết quả trước P0). So thẳng, IPPO đắt hơn EOQ 3,8% (nhưng fill
   rate 85,2% so với 75,0%). Ở cùng ngưỡng phục vụ ≥ 84,71%, IPPO rẻ hơn EOQ
   47,5%, rẻ hơn (s,S) 21,2%, rẻ hơn Newsvendor 11,3% (`python run.py iso`,
   xem `results_pre_p0/iso_service.json`).
2. *Parameter sharing có tổng quát hóa được qua các quy mô cầu không?* —
   **Có điều kiện** (kết quả trước P0). Nhóm cầu cao/trung bình đạt fill rate
   86,3%/83,6%, nhưng nhóm cầu thấp và gián đoạn nhất chỉ đạt 72,8%
   (`python scripts/analyze_scale_groups.py`, xem
   `results_pre_p0/scale_group_analysis.json`).
3. *Phần thưởng phân rã cục bộ có giúp học tốt hơn thưởng tổng thể không?* —
   **Chưa kiểm chứng đầy đủ.** Thiết kế cục bộ + chuẩn hóa theo cặp là thiết
   kế được dùng xuyên suốt và có bằng chứng phát triển sơ bộ ủng hộ (xem
   `BAO_CAO_SUA_LOI.md`), nhưng chưa có thí nghiệm loại trừ (tắt
   `normalize_reward_per_pair`, hoặc thay bằng phần thưởng toàn cục dùng
   chung) chạy lại trên đúng cấu hình 300 cặp hiện tại — xem mục
   [Ablation](#ablation--đổi-giá-trị-trong-configyaml-rồi-chạy-lại) ở trên.

Số liệu chi tiết đầy đủ của lần chạy trước P0 (bảng so sánh, kiểm định thống
kê, đa hạt giống) nằm trong `results_pre_p0/` — `Report KLTN/` hiện vẫn phản
ánh đúng các số liệu này (chưa cập nhật theo P0).

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
- `main.py` (shim tương thích ngược trỏ sang `run.py`) đã bị **xóa** vì không
  còn nơi nào gọi tới; `run.py` là điểm vào duy nhất.

---

## Thay đổi trong bản P0

Theo kế hoạch chi tiết trong [`README_CLAUDE_CODE_KLTN.md`](README_CLAUDE_CODE_KLTN.md)
(nhóm P0 — bắt buộc sửa trước khi huấn luyện lại lần cuối):

- **[P0-1] Sửa lỗi tràn kho** (`env/inventory_env.py`): bản trước ([V2-5])
  tính tổng (tồn kho cũ + hàng mới về) rồi phân bổ phần vượt sức chứa theo
  **tỷ trọng trên cả hai nguồn** — có thể vô tình "ăn" vào tồn kho cũ đang nằm
  hợp lệ trong sức chứa (ví dụ tồn cũ 90, sức chứa 100, hàng mới về 20 → bản
  cũ từ chối 10/110 trên cả hai nguồn, làm tồn kho cũ cũng bị giảm oan). Nay
  chỉ tính đúng phần hàng **mới về** vượt quá dung lượng còn trống; tồn kho cũ
  không bao giờ bị buộc này giảm. Có test hồi quy
  `test_tran_kho_chi_tu_choi_hang_moi_khong_dung_ton_kho_cu`.
- **[P0-3] Chi phí thiếu hàng cho thí nghiệm chính quay về hằng số** dùng
  chung (`use_real_price_stockout: false` mặc định) — xem mục
  [Chi phí thiếu hàng theo giá thật](#chi-phí-thiếu-hàng-theo-giá-thật-tùy-chọn-tắt-mặc-định-từ-bản-p0).
- **[P0-4]/[P0-6] "fill_rate" trong `train_log*.csv` (cả lúc huấn luyện lẫn
  đánh giá) nay là fill rate THẬT của toàn episode** (1 − tổng thiếu
  hàng/tổng cầu), không còn là trung bình cộng của tín hiệu cửa sổ trượt 30
  ngày tại từng bước — cách tính cũ thiên lệch, đặc biệt ở đầu episode khi cửa
  sổ chưa đầy.
- **[P0-5] Chọn `best_model` theo chi phí vận hành thấp nhất** trong số các
  lần đánh giá đạt ràng buộc `fill_rate >= ppo.min_fill_to_save` (nay mặc
  định `0.85`, trước là `0.0` tức mọi checkpoint đều "đạt"). Nếu chưa lần nào
  đạt ràng buộc, tạm giữ checkpoint có fill rate cao nhất làm fallback và in
  rõ `feasible=False` — không âm thầm coi một checkpoint chưa đạt SLA là tốt
  nhất.
- Sửa thêm một lỗi môi trường: `SummaryWriter` (TensorBoard) có thể crash
  trên Windows khi đường dẫn dự án chứa dấu tiếng Việt (lỗi
  `FailedPreconditionError` từ `tensorflow.io.gfile`) — nay lỗi này chỉ in
  cảnh báo và bỏ qua TensorBoard, không làm hỏng cả quá trình huấn luyện (vẫn
  ghi đầy đủ `results/train_log*.csv`).
- Kết quả/checkpoint trước bản P0 được archive nguyên vẹn vào
  `results_pre_p0/` và `checkpoints_pre_p0/`, không ghi đè mất.
- **Việc còn lại**: cần chạy lại `python scripts/train.py --episodes 5000`
  (hoặc `python run.py train`) rồi `python run.py eval && python run.py iso
  && python run.py summary && python scripts/analyze_scale_groups.py` để có
  số liệu khớp với code hiện tại, sau đó cập nhật lại Chương 4/5 của
  `Report KLTN/`. Các hạng mục P1/P2 (tune baseline trên miền val thay vì
  train, kiểm định thống kê dạng paired, cửa sổ đánh giá cố định toàn miền,
  warm-start lịch sử cầu, thí nghiệm theo chế độ nhu cầu, phân tích độ nhạy)
  vẫn còn nguyên trong `README_CLAUDE_CODE_KLTN.md`, chưa thực hiện.

---

## Trạng thái báo cáo LaTeX

`Report KLTN/` hiện phản ánh đúng cấu hình **trước bản P0** (10 kho × 30 SKU
= 300 tác tử, kiến trúc Actor/Critic tách riêng, chuẩn hóa phần thưởng theo
cặp, chia 3 miền train/val/test, chi phí thiếu hàng theo giá thật) — Chương
4/5 đã điền số liệu thật (không còn placeholder `[cần điền]`), nhưng số liệu
đó lấy từ lần chạy **trước** các sửa lỗi P0 ở trên (đã archive ở
`results_pre_p0/`). Sau khi chạy lại pipeline với code hiện tại (mục
[Thay đổi trong bản P0](#thay-đổi-trong-bản-p0)), cần:

1. Copy các hình mới vào `Report KLTN/media/` (`hinh_duong_cong_hoc.png`,
   `hinh_so_sanh_baseline.png`, `hinh_danh_doi_chi_phi_dich_vu.png`,
   `hinh_so_sanh_cung_muc_phuc_vu.png`, `hinh_phan_tich_theo_quy_mo.png` —
   sinh bởi `scripts/plot_learning_curve.py`, `scripts/evaluate.py`,
   `scripts/iso_service.py`, `scripts/analyze_scale_groups.py`).
2. Cập nhật lại các bảng số liệu và phần thảo luận trong `content/C4.tex` và
   `content/C5.tex` theo số liệu mới (bảng so sánh trực tiếp, so sánh cùng
   mức phục vụ, phân tích theo nhóm quy mô cầu, đa hạt giống).
3. Cập nhật ngắn gọn mục "Các tham số chi phí" ở `content/C3.tex` nếu chạy
   chính thức với `use_real_price_stockout: false` (bỏ phần mô tả đơn giá suy
   ra từ giá thật khỏi số liệu chính, hoặc chuyển thành mục ablation).

Báo cáo **không** tự động đồng bộ với `results/` — đây là bước thủ công sau
mỗi lần huấn luyện lại.

Câu hỏi nghiên cứu 3 (local reward so với global reward) vẫn chưa có thí
nghiệm loại trừ chạy lại trên đúng cấu hình 300 cặp — xem mục "Ba câu hỏi
nghiên cứu" và mục "Ablation" ở trên.

---

## Tham khảo

1. Schulman và cộng sự (2017), *Proximal Policy Optimization Algorithms*
2. Yu và cộng sự (NeurIPS 2022), *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*
3. Harris (1913) cho EOQ; Scarf (1960) cho (s,S); Arrow, Harris và Marschak (1951) cho Newsvendor
4. Wolpert và Tumer (2001), *Optimal Payoff Functions for Members of Collectives* — cơ sở cho cách phân bổ trách nhiệm tràn kho về từng SKU
