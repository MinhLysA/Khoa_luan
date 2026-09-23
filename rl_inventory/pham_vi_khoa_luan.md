# Khóa luận: Tối ưu hóa chính sách đặt hàng lại trong quản lý tồn kho đa kho bằng PS-IPPO

*Tài liệu tổng hợp duy nhất của dự án — chốt ngày 24/09/2026.*
Gộp và thay thế: `README_CLAUDE_CODE_KLTN.md` (kế hoạch P0/P1/P2),
`BAO_CAO_SUA_LOI.md` (chẩn đoán bản v2), `Report KLTN/CHANGELOG_HIEUCHINH.md`
(nhật ký sửa báo cáo) và bản phạm vi cũ. `README.md` chỉ còn là trang dẫn tới
tài liệu này.

**Mục lục**
1. [Trạng thái hiện tại](#1-trạng-thái-hiện-tại)
2. [Đề tài, câu hỏi nghiên cứu, phạm vi](#2-đề-tài-câu-hỏi-nghiên-cứu-phạm-vi)
3. [Mô hình bài toán](#3-mô-hình-bài-toán)
4. [Dữ liệu và giao thức đánh giá](#4-dữ-liệu-và-giao-thức-đánh-giá)
5. [Thuật toán và siêu tham số](#5-thuật-toán-và-siêu-tham-số)
6. [Mã nguồn và cách chạy](#6-mã-nguồn-và-cách-chạy)
7. [Đợt thực nghiệm cuối](#7-đợt-thực-nghiệm-cuối)
8. [Kết quả hiện có](#8-kết-quả-hiện-có-trước-đợt-cuối)
9. [Lịch sử sửa lỗi quan trọng](#9-lịch-sử-sửa-lỗi-quan-trọng)
10. [Báo cáo LaTeX](#10-báo-cáo-latex)
11. [Việc còn lại và hướng phát triển](#11-việc-còn-lại-và-hướng-phát-triển)
12. [Câu hỏi thường gặp của hội đồng](#12-câu-hỏi-thường-gặp-của-hội-đồng)

---

## 1. Trạng thái hiện tại

| Hạng mục | Trạng thái |
|---|---|
| Code | **Sẵn sàng cho đợt train cuối.** 42/42 test pass. Protocol cuối đã cài đặt: 3 miền, tune baseline trên val theo ràng buộc 85%, đánh giá trên toàn bộ quỹ đạo test, kiểm định theo cặp, reward toàn cục cho RQ3. |
| Đợt thực nghiệm cuối | **Chưa chạy.** 12 lần chạy (khoảng 25 giờ CPU), chạy bằng `scripts/campaign.py` hoặc notebook Colab. Xem mục 7. |
| Số liệu trong báo cáo | Hiện đang là **bản cũ** (3 seed × 1.000 episode, phạt SLA kiểu cũ, 30 cửa sổ 365 ngày chồng lấn). Phải thay toàn bộ sau đợt cuối. |
| Báo cáo LaTeX | Khung logic đã chốt (RQ, methodology, thuật ngữ, trích dẫn). Biên dịch sạch, 91 trang. Xem mục 10. |
| Git | Chưa commit các thay đổi gần nhất. Colab `git pull` từ GitHub nên phải push trước khi chạy. |

**Những file kết quả cũ cần biết:**
- File `results/` **không có hậu tố** (`summary.json`, `train_log.csv`, `best_model.pth`…) và `TONG_HOP_SO_LIEU.txt` là phần sót lại của các lần chạy thử. Không dùng cho báo cáo.
- `best_model_seed42.pth` là bản seed 42 train tiếp trên Colab tới khoảng 4.083 episode (đánh giá với tag `seed42_resume`), không phải bản 1.000 episode của bảng chính.
- `results_pre_p0/`, `checkpoints_pre_p0/`: số liệu trước bản P0, chỉ để tham khảo.

---

## 2. Đề tài, câu hỏi nghiên cứu, phạm vi

**Đề tài:** Xây dựng mô hình học tăng cường tối ưu hóa chính sách đặt hàng lại
trong quản lý tồn kho đa kho, bằng IPPO với chia sẻ tham số (PS-IPPO).

### Câu hỏi nghiên cứu (bản chốt)

1. **RQ1 (chính):** Trong cùng điều kiện mức phục vụ — mọi chính sách phải đạt
   fill rate ≥ 85% — PS-IPPO có đạt tổng chi phí vận hành thấp hơn EOQ, (s,S),
   Newsvendor không, và khác biệt có ý nghĩa thống kê không? So sánh ở mức ~90%
   (bằng mức IPPO đạt) chỉ là phân tích phụ.
2. **RQ2:** Chia sẻ tham số có duy trì hiệu năng trên các cặp kho–mặt hàng có
   quy mô nhu cầu khác nhau không? Mở rộng: áp dụng được cho mặt hàng chưa gặp
   khi huấn luyện không (thí nghiệm hold-out)?
3. **RQ3:** Thiết kế tín hiệu phần thưởng — cục bộ theo cặp so với toàn cục cả
   mạng lưới, và có/không chuẩn hóa theo cặp — ảnh hưởng thế nào đến ổn định huấn
   luyện và hiệu năng?

### Phạm vi

**Trong phạm vi:** môi trường Gymnasium 10 kho × 30 SKU = 300 tác tử, nhu cầu
thật M5, lead time ngẫu nhiên 1–3 ngày, sức chứa dùng chung mỗi kho; PS-IPPO; ba
baseline cổ điển; đánh giá đa hạt giống có kiểm định; app Streamlit mô phỏng theo
ngày; prototype MCP (phụ lục, không phải đóng góp chính).

**Ngoài phạm vi:** multi-echelon và chuyển hàng liên kho (nên không có chi phí
điều chuyển); dự báo nhu cầu bằng học sâu; truyền thông trực tiếp giữa tác tử;
action liên tục; triển khai thời gian thực; toàn bộ 3.049 SKU.

---

## 3. Mô hình bài toán

### 3.1. Từ nghiệp vụ tới tác tử

Quy trình một ngày tại mỗi kho, mỗi mã hàng: nhận hàng (hàng vượt chỗ trống bị
từ chối) → kiểm tra tồn kho, hàng đang về, lịch sử bán, mức đầy kho → **quyết định
lượng đặt** → bán hàng (thiếu thì mất đơn) → ghi nhận chi phí và mức phục vụ.

Chỉ có **một** quyết định cần tối ưu (lượng đặt), lặp cho từng mã hàng tại từng
kho, do người phụ trách mã hàng đó chịu trách nhiệm. Vì vậy nhiệm vụ tổng được
chia thành 300 nhiệm vụ con cùng loại: **1 tác tử = 1 cặp (kho, mặt hàng)**.
Không có coordinator.

### 3.2. Kiến trúc đa tác tử

- **300 tác tử, dùng chung một hàm chính sách π_θ**: mỗi tác tử tự quyết định từ
  quan sát riêng và nhận phần thưởng riêng. Chia sẻ tham số không gộp 300 tác tử
  thành một tác tử trung tâm.
- **"Independent" trong IPPO:** GAE/advantage tính riêng cho từng tác tử; không
  có critic tập trung (khác MAPPO).
- **Phối hợp gián tiếp qua 3 kênh:** (1) mức đầy kho `u_t^(w)` do môi trường tính
  và phát lại vào quan sát của 30 tác tử cùng kho; (2) phạt tràn kho khi cùng
  tranh sức chứa; (3) trọng số dùng chung khi huấn luyện.

### 3.3. MDP / Dec-POMDP

| Thành phần | Nội dung |
|---|---|
| Mục tiêu đội | `R_team = Σ_i r_i`; tín hiệu huấn luyện được phân rã thành `r_i` cục bộ (local factored reward). Đây **không phải** difference reward (vốn cần phản thực `G(z) − G(z₋ᵢ)`). |
| Quan sát (43 chiều) | tồn kho, vị thế tồn kho (2) · cầu 7 ngày, thang log (7) · hàng đang về theo ngày (3) · fill rate 30 ngày (1) · quy mô log(1+D̄) (1) · mức đầy kho, tỷ trọng trong kho (2) · thứ trong tuần (7) · lịch: SNAP, sự kiện, tháng (20). **Không có tín hiệu giá** (`include_price_signal: false`). |
| Hành động | 6 mức: `q = ceil(m·D̄·L̄)`, m = (0; 0,5; 1; 2; 3; 5), ép tăng nghiêm ngặt. |
| Chuyển trạng thái | Nhận hàng → kiểm tra sức chứa (chỉ từ chối **hàng mới về** vượt chỗ trống, chia theo tỷ trọng hàng về) → ghi đơn mới (L ~ U{1,2,3}) → nhu cầu thật M5, mất đơn nếu thiếu → cập nhật lịch sử và fill rate. |
| Phần thưởng | `r_i = −(C_i + P_i)`. Chi phí vận hành `C = lưu kho + thiếu hàng + đặt hàng + tràn kho` (4 thành phần). `P` = phạt mức phục vụ, **là tín hiệu huấn luyện, không phải chi phí thứ năm**, không tính vào chi phí báo cáo. |
| Chuẩn hóa | `r_i / (c_th·D̄_i + c_dh)` để các cặp có quy mô chênh 4 bậc về cùng thang. |
| γ | 0,99 |

**Reward luôn âm** (= −chi phí); không có khoảng reward "chuẩn". Chỉ đọc xu hướng
(càng gần 0 càng tốt) và so tương đối trong cùng bài toán.

### 3.4. Mục tiêu 85% là ràng buộc mềm

Mục tiêu nghiệp vụ: `min E[Σ C]` sao cho `FR ≥ 85%`. Khi huấn luyện, ràng buộc
được đưa vào dưới dạng phạt mềm `P`. Tính khả thi được kiểm tra ở khâu chọn mô
hình: checkpoint IPPO = chi phí vận hành val thấp nhất trong số checkpoint có
`FR_val ≥ 85%`; baseline tune theo đúng tiêu chí đó.

**Hai cách định cỡ phạt SLA** (`env.service_penalty_mode`):
- `demand` (cũ, dùng cho mọi số liệu hiện có): `P = φ·c_th·D̄·max(τ−FR, 0)` → mặt
  hàng bán chậm bị phạt rất nhẹ, chính sách phân bổ mức phục vụ thấp cho chúng
  (39% cặp dưới 85%).
- `normalized` (**cấu hình chính của đợt cuối**): `P = φ·(c_th·D̄ + c_dh)·max(τ−FR, 0)`
  → sau chuẩn hóa, mọi cặp chịu cùng mức phạt `φ·max(τ−FR, 0)`.

### 3.5. Tham số chi phí

| c_lk | c_th | c_dh | p_tk | φ_dv | τ | Sức chứa |
|---|---|---|---|---|---|---|
| 1 | 10 | 5 | 5 | 3 | 85% | 5 ngày cầu của từng kho |

Đơn vị là chi phí mô phỏng chuẩn hóa, không phải USD (M5 không công bố giá vốn).
Sức chứa 5 ngày được hiệu chỉnh bằng thực nghiệm: ở 3 ngày mục tiêu 85% bất khả
thi (tối đa 79,5%); từ 12 ngày trở lên ràng buộc không còn hiệu lực.

---

## 4. Dữ liệu và giao thức đánh giá

### 4.1. Dữ liệu

M5 Walmart: 10 cửa hàng (CA_1..4, TX_1..3, WI_1..3), 1.941 ngày. Chọn 30 SKU
phân tầng, chỉ giữ SKU có cầu ≥ 0,2/ngày ở ≥ 80% cửa hàng (loại mặt hàng cửa
hàng không kinh doanh). Tập 300 cặp: cầu TB 3,24, trung vị 0,74, 57,4% ngày bằng
0 (vẫn giữ tính gián đoạn). Dữ liệu gốc (`data/raw/*.csv`) tải từ Kaggle; dữ liệu
đã xử lý có sẵn trong `data/processed/`.

### 4.2. Ba miền

| Miền | Ngày | Được dùng để |
|---|---|---|
| Train | [0, 1050) | rollout PPO; ước lượng D̄, sức chứa, mẫu số chuẩn hóa |
| Validation | [1050, 1450) | chọn checkpoint IPPO; tune baseline |
| Test | [1450, 1941) | **chỉ** báo cáo kết quả cuối |

### 4.3. Đánh giá cuối (`env.test_full_horizon: true`)

- Mỗi lần đánh giá chạy **trọn 491 ngày** test, nhu cầu thật cố định; lặp **30
  seed thời gian giao hàng**. Mọi chính sách dùng cùng seed → so sánh theo cặp.
  (Thay cho giao thức cũ: 30 cửa sổ 365 ngày trong miền 491 ngày, chồng lấn mạnh.)
- Kiểm định trên `ΔC_j = C_j^IPPO − C_j^baseline`: chênh lệch TB + CI 95%, paired
  t-test, Wilcoxon, `d_z`. Không dùng Welch; không gộp 90 lần IPPO với 30 lần
  baseline.
- 3 seed huấn luyện: báo cáo từng seed, rồi TB ± SD giữa các seed
  (`results/multiseed_main.json`).
- Theo giai đoạn nhu cầu: bootstrap theo khối 7 ngày (CI 95%).

### 4.4. Baseline và chỉ số

- EOQ (Harris 1913), (s,S) (Scarf 1960), Newsvendor (Arrow et al. 1951); dùng
  cùng thông tin với IPPO và cùng 6 mức đặt (ràng buộc lô chuẩn chung, để cô lập
  tác động của thuật toán).
- Chỉ số: tổng chi phí vận hành, fill rate (tổng bán/tổng cầu), cơ cấu chi phí,
  fill rate từng cặp, đường học (reward, entropy, explained variance, KL).

---

## 5. Thuật toán và siêu tham số

PS-IPPO: Actor và Critic **tách riêng** (mỗi mạng 2 lớp ẩn 128, Tanh), 2
optimizer Adam, cắt gradient riêng; value normalization; value clipping; GAE
theo từng cặp; chuẩn hóa advantage theo cặp.

| n_steps | epochs | minibatch | lr actor / critic | γ / λ | clip | target KL | entropy | episode |
|---|---|---|---|---|---|---|---|---|
| 1024 | 4 | 16.384 | 3e-4 / 1e-3 | 0,99 / 0,95 | 0,2 | 0,02 | 0,02 → 0,002 | 4.000 (chính), 1.000 (ablation) |

Đánh giá tất định trên val mỗi 25 episode (2 episode). Lưu checkpoint mỗi 200
episode; `--resume` train tiếp đúng từ trạng thái đã lưu.

---

## 6. Mã nguồn và cách chạy

### 6.1. Cấu trúc

```
rl_inventory/
├── run.py                       điểm vào: data/baseline/train/eval/iso/regime/behavior/summary/test/app
├── config.yaml                  toàn bộ tham số và cờ
├── configs/                     config ablation, sinh bằng configs/make_configs.py
├── Khoa_luan_Colab.ipynb        chạy đợt cuối trên Colab (gọi campaign.py)
├── env/inventory_env.py         môi trường Gymnasium (300 cặp vector hóa)
├── agents/ppo_agent.py          PS-IPPO (+ chế độ shared_trunk cho ablation)
├── agents/rollout_buffer.py     GAE theo cặp
├── baselines/traditional_policies.py
├── scripts/
│   ├── campaign.py              toàn bộ đợt cuối: train (tự resume) + eval
│   ├── train.py / evaluate.py   huấn luyện / đánh giá (kiểm định theo cặp)
│   ├── tune_baselines.py        tune trên val theo ràng buộc 85%
│   ├── iso_service.py           so sánh ở cùng mức phục vụ ~90% (phụ)
│   ├── regime_analysis.py       thực nghiệm B: giai đoạn nhu cầu, sốc cầu, bootstrap
│   ├── policy_behavior.py       hành vi policy, lách reward, permutation importance
│   ├── analyze_scale_groups.py  theo nhóm quy mô cầu (bảng 4 nhóm)
│   ├── sensitivity.py           độ nhạy theo đơn giá chi phí
│   ├── plot_learning_curve.py   đường học (--compare cho ablation)
│   ├── generate_summary.py      TONG_HOP_SO_LIEU.txt
│   └── common.py
├── app/app.py + pages/          demo Streamlit 4 trang (chi tiết theo ngày từng cặp: cầu, tồn, hàng đang về, đặt, thiếu, chi phí, reward, cảnh báo)
├── app/streamlit_app.py         bảng điều khiển khi làm khóa luận
├── app/mcp_server.py            prototype MCP (5 tool, 1 resource)
└── tests/test_env.py            42 test
```

### 6.2. Lệnh thường dùng

```bash
pip install -r requirements.txt
python run.py test                      # 42 test
python run.py app                       # demo Streamlit
python scripts/campaign.py status       # tiến độ đợt cuối
python scripts/campaign.py train main   # 3 seed × 4.000 episode (tự resume)
python scripts/campaign.py train ablations
python scripts/campaign.py eval         # toàn bộ pipeline đánh giá
python app/mcp_server.py --selftest     # thử MCP không cần LLM
```

**Colab (`Khoa_luan_Colab.ipynb`):** notebook trình bày quy trình theo 12 bước đánh
số: (1) kết nối Drive → (2) lấy code từ GitHub → (3) cài thư viện → (4) lấy **3 file
CSV gốc M5 từ `MyDrive/KLTN_data/`** (quá lớn cho GitHub), chạy tiền xử lý và đối
chiếu với bản đã xử lý trên GitHub → (5) kiểm tra dữ liệu và ba miền → (6) chạy 42
test → (7) tune baseline trên val → (8) train 3 seed mô hình chính → (9) train 10
ablation → (10) đánh giá (10.1 so sánh chính, 10.2 cùng mức phục vụ, 10.3 thực
nghiệm B / hành vi / quy mô / độ nhạy, 10.4 ablation, 10.5 hold-out) → (11) xem kết
quả → (12) tùy chọn đẩy lên GitHub. Kết quả lưu ở `MyDrive/KLTN_final`, log từng
bước ở `logs/buocN_*.log`. Phụ lục của notebook có chế độ chạy nhiều phiên song
song (`campaign.py train auto`).

**Chuẩn bị dữ liệu cho Colab:** tải `sales_train_evaluation.csv`, `calendar.csv`,
`sell_prices.csv` từ Kaggle (M5 Forecasting – Accuracy) và upload vào
`MyDrive/KLTN_data/`. Tiền xử lý từ 3 file này cho ra dữ liệu **trùng khớp hoàn
toàn** với `data/processed/` trên GitHub (đã kiểm tra: 4 file `.npy` và danh sách
kho/mặt hàng), nên kết quả tái lập được.

### 6.3. Cờ cấu hình chính

| Cờ | Ý nghĩa |
|---|---|
| `env.service_penalty_mode` | `normalized` (chính) / `demand` (cũ) |
| `env.reward_mode` | `local` (chính) / `global` (ablation RQ3) |
| `env.normalize_reward_per_pair` | chuẩn hóa theo cặp (ablation RQ3) |
| `env.test_full_horizon` | đánh giá trọn 491 ngày test |
| `env.obs_drop` | tắt nhóm quan sát (`calendar`, `warehouse`, …) |
| `env.include_event_lookahead` | thêm 2 đặc trưng sự kiện sắp tới |
| `env.warm_start_history` | lịch sử cầu khởi tạo bằng dữ liệu thật |
| `env.sku_indices` | tập con SKU (hold-out) |
| `env.include_price_signal` | tín hiệu giá (tắt; bật sẽ thành 44 chiều) |
| `ppo.shared_trunk` | Actor/Critic chung thân (ablation) |

---

## 7. Đợt thực nghiệm cuối

| Lần chạy | Config | Episode | Trả lời |
|---|---|---|---|
| `main_s42`, `main_s1`, `main_s2` | `config.yaml` | 4.000 | RQ1, RQ2; mặt hàng bán chậm còn bị phân bổ mức phục vụ thấp không |
| `abl_ref` | `config.yaml` | 1.000 | mốc cho mọi ablation (cùng lịch lr/entropy) |
| `abl_global` | `ablation_global_reward` | 1.000 | RQ3: cục bộ vs toàn cục |
| `abl_q3` | `ablation_q3_no_reward_norm` | 1.000 | RQ3: có/không chuẩn hóa |
| `abl_reward_cu` | `ablation_reward_demand_scaled` | 1.000 | tác động của việc định cỡ lại phạt SLA |
| `abl_trunk` | `ablation_shared_trunk` | 1.000 | vì sao phải tách Actor/Critic |
| `abl_nocal`, `abl_nowh` | `ablation_drop_*` | 1.000 | state có biến dư thừa không |
| `abl_event` | `ablation_event_lookahead` | 1.000 | cải thiện mùa lễ |
| `abl_warm` | `ablation_warm_start` | 1.000 | ảnh hưởng lịch sử rỗng đầu episode |
| `holdout` | `holdout_train` → `holdout_eval` | 1.000 | RQ2 mở rộng: 6 SKU chưa gặp (3, 8, 9, 10, 23, 25) |

`campaign.py eval` chạy lần lượt: tune baseline (val) → eval 3 seed → iso-service
→ regime → behavior → quy mô → độ nhạy → eval ablation → hold-out → tổng hợp đa
hạt giống.

---

## 8. Kết quả hiện có (trước đợt cuối)

*Protocol cũ; dùng để định hướng, sẽ được thay toàn bộ.*

- **So thẳng (3 seed × 1.000 episode):** IPPO fill 90,5% nhưng chi phí cao hơn
  (s,S) 4,5–5,5% (paired p < 10⁻¹⁶ ở cả 3 seed; đắt hơn ở 30/30 episode).
- **Cùng mức phục vụ ~90% (seed 42):** IPPO rẻ hơn (s,S) 10,2%, Newsvendor 14,4%;
  EOQ không đạt.
- **Seed 42 train kéo dài (~4.083 ep):** chi phí 1.512.076 (fill 89,9%) so với
  (s,S) 1.520.337 (84,1%) → −0,5%, paired p = 0,050; cửa sổ cố định −0,2%, p = 0,51.
- **Thực nghiệm B:** ở cùng mức phục vụ, IPPO rẻ hơn (s,S) 12,9–17,6% ở cả 8 nhóm
  ngày (CI 95% đều dưới 0). So với (s,S) trực tiếp: đắt hơn ở ngày ổn định
  (+2,5%) và ngày thường (+4,2%); rẻ hơn ở ngày sự kiện (−9,6%) và cuối tuần
  (−9,7%); mùa lễ +8,7% nhưng không có ý nghĩa (chỉ 9 tuần). Sốc cầu ×1,5: IPPO
  tốn ít hơn 29–32%; sau sốc ×0,5 baseline tụt còn 87–88%, IPPO giữ 90%.
- **Hành vi:** xác suất đặt giảm đơn điệu theo số ngày tồn (Spearman −0,42). Không
  dồn sát ngưỡng 85%, nhưng **39% cặp dưới 85%** do phạt SLA tỷ lệ D̄.
  Permutation importance: quy mô (+98%), tồn kho (+94%), lịch sử cầu (+50%); lịch
  và tín hiệu cấp kho chỉ khoảng 3%.
- **Theo quy mô cầu:** IPPO tốt hơn (s,S) ở nhóm cầu ≥ 2/ngày, kém ở nhóm < 2/ngày.
- **Độ nhạy đơn giá (36 bộ):** rẻ hơn (s,S) cùng mức phục vụ ở 36/36 bộ; so thẳng
  chỉ 22/36, phụ thuộc tỷ số c_th/c_lk.
- **RQ3 sơ bộ (dừng ở 625 ep):** tắt chuẩn hóa → chi phí gấp khoảng 3,2 lần, fill
  80,7% so với 90,5%, dù explained variance ≈ 0,997.

---

## 9. Lịch sử sửa lỗi quan trọng

Bằng chứng số dưới đây được trích trong Chương 3–4; mã nguồn có comment `[V2-n]`,
`[V3-n]`, `[P0-n]`, `[P2-n]`, `[P3-n]` tại đúng vị trí.

**v2 — vì sao bản đầu không hội tụ:**
- *Actor/Critic chung thân:* gradient value loss 28,13 so với policy loss 0,096
  (chênh 293 lần); cắt gradient chung làm lr actor hiệu dụng giảm khoảng 55 lần →
  entropy đứng yên ở ln 6. Sửa: tách hai mạng, hai optimizer, value norm, value clip.
- *Reward chênh 4 bậc độ lớn* giữa SKU chậm (−0,5) và nhanh (−14.000) → chuẩn hóa
  theo cặp `c_th·D̄ + c_dh`.
- *Sức chứa một con số cho cả 10 kho* → CA_3 chỉ chứa 1,38 ngày cầu, bài toán vô
  nghiệm. Sửa: sức chứa riêng từng kho (5 ngày cầu).
- *Bảng mức đặt bị trùng:* 35,6% cặp chỉ có 2 mức phân biệt → `ceil` + ép tăng
  nghiêm ngặt.
- *Quan sát thiếu tín hiệu cấp kho* → thêm mức đầy kho và tỷ trọng; lịch sử cầu
  thang log.
- *Lỗi chồng chỉ số `calendar_features`*; xử lý truncation; chọn checkpoint bằng
  đánh giá tất định; log CSV.
- *Quy mô:* 50 → 30 SKU (loại 142/500 cặp gần như không bán).

**v3:** chia 3 miền (bản v2 chọn checkpoint trên chính test); lệnh multiseed.

**P0:** tràn kho chỉ từ chối hàng mới về (bản cũ trừ oan tồn kho cũ); fill rate
trong log là fill rate thật của cả episode; chọn checkpoint theo chi phí thấp
nhất trong nhóm đạt 85%; tắt chi phí thiếu hàng theo giá (M5 không có giá vốn).

**P2/P3 (đợt cuối):** tune baseline trên val theo ràng buộc 85%; kiểm định theo
cặp; đánh giá trọn quỹ đạo test; bootstrap khối tuần; phạt SLA `normalized`;
reward toàn cục; warm-start; obs_drop; sự kiện sắp tới; hold-out SKU; shared
trunk; độ nhạy chi phí; MCP; app chi tiết theo ngày.

---

## 10. Báo cáo LaTeX

`../Report KLTN/`, biên dịch bằng `compile.bat` (XeLaTeX + Biber).

### 10.1. Đáp ứng 27 yêu cầu của giảng viên

Đủ 26/27. Mục còn lại (MCP) được đáp ứng ở mức giảng viên yêu cầu ("tìm hiểu,
chưa phải nội dung chính"): có prototype chạy được, trình bày ở Phụ lục B. Demo
theo ngày đã đủ mọi cột được yêu cầu, gồm cả chi phí từng cặp.

Các mục đã có trong báo cáo: IPPO và chữ "I" (C2); 300 tác tử, không coordinator,
đi từ nghiệp vụ (C3 §3.1.1); task decomposition; bảng đặc tả tác tử; bảng ánh xạ
MDP; reward âm và không có khoảng chuẩn; sơ đồ khái niệm, vòng RL, kiến trúc
(Hình 3.1–3.3, TikZ); đường học; kiểm tra state/reward bằng thực nghiệm; thực
nghiệm A/B theo giai đoạn; không mặc định RL thắng.

### 10.2. Checklist review trước hội đồng

| Mục | Trạng thái |
|---|---|
| RQ1–RQ3 viết lại; RQ3 có thí nghiệm local vs global thật | ✅ |
| 3 miền + bảng vai trò; baseline tune trên val | ✅ |
| Đánh giá trọn quỹ đạo test, kiểm định theo cặp | ✅ (protocol); ⏳ số liệu |
| Ngân sách thống nhất 3 × 4.000 | ✅ (cài đặt); ⏳ chạy |
| Quan sát 43 chiều, bỏ "giá" khỏi mô tả | ✅ |
| 85% là mục tiêu chính, ~90% là phụ | ✅ |
| Thuật ngữ: local factored reward, Dec-POMDP team objective, ràng buộc mềm, chi phí vận hành ≠ phạt | ✅ |
| Bibliography sạch (IEEE, [n]); sửa sai tác giả 4 mục; DOI Zhu & Wu | ✅ |
| Tác giả bài SSRN (Nam et al.) | ⚠️ cần tự kiểm tra trên trang SSRN |
| Phụ lục A dùng `\ref` tự động | ✅ |
| MCP thu gọn, chi tiết sang Phụ lục B | ✅ |
| Văn phong khoa học | ✅ (lượt 1) |
| booktabs cho mọi bảng | 🟡 các bảng mới đã dùng; bảng kết quả C4 chuyển khi sinh lại |
| Viết lại Abstract, C4, C5 theo số liệu cuối | ⏳ sau đợt cuối (Abstract có comment `CAP NHAT SAU DOT CHAY CUOI`) |
| Xóa mọi câu "đang chờ train", "sơ bộ" khỏi bản nộp | ⏳ sau đợt cuối |

### 10.3. Sau khi có kết quả đợt cuối

1. `git pull` kết quả; chép hình mới vào `Report KLTN/media/`.
2. Thay mọi bảng/hình C4 bằng số `*_main*`, `*abl_*`, `holdout_*`,
   `multiseed_main.json`; chuyển sang booktabs, sửa 4 bảng đang tràn lề.
3. Viết lại diễn giải nếu kết luận đổi; điền mục RQ3 (local vs global + chuẩn hóa)
   và hold-out.
4. Viết lại Abstract (VN/EN), C5 §5.1–5.2; chuyển các thí nghiệm đã chạy từ "hướng
   phát triển" sang C4.
5. Đối chiếu mọi % trong Abstract/slide với bảng/hình.

---

## 11. Việc còn lại và hướng phát triển

**Trước khi nộp:** chạy đợt cuối (mục 7) → cập nhật báo cáo (mục 10.3) → kiểm tra
tác giả SSRN → commit.

**Chưa làm (có thể bổ sung nếu còn thời gian):**
- Độ nhạy theo lead time (2–5 ngày), sức chứa (4/5/6 ngày), mục tiêu mức phục vụ
  (80/85/90%) — hiện mới có độ nhạy theo đơn giá chi phí.
- Val trên toàn quỹ đạo cố định (hiện val vẫn rút cửa sổ 365 ngày trong miền 400
  ngày).
- File `run_metadata.json` lưu config snapshot cho mỗi lần chạy.
- `run.py multiseed` vẫn mặc định 1.000 episode; đợt cuối dùng `campaign.py`.

**Hướng phát triển (ngoài phạm vi):** nhân tử Lagrange riêng từng cặp;
chuyển hàng liên kho, multi-echelon; MAPPO/critic tập trung; communication giữa
tác tử; action liên tục; dự báo LSTM/Transformer; 3.049 SKU; triển khai thực tế;
lớp giao tiếp ngôn ngữ qua MCP.

---

## 12. Câu hỏi thường gặp của hội đồng

Bản trả lời đầy đủ cho 27 yêu cầu của giảng viên, kèm bằng chứng (hình, bảng, dòng
code): **[van_dap.md](van_dap.md)**. Dưới đây là bản rút gọn.

- **"I" trong IPPO?** Independent: mỗi tác tử tính advantage riêng từ reward
  riêng, quyết định từ quan sát riêng; không có critic tập trung.
- **Dùng chung mạng thì sao còn "independent"?** Chung *hàm chính sách* π_θ,
  nhưng quyết định và reward là riêng của từng tác tử; không có tác tử nào nhìn
  toàn hệ thống hay quyết định thay.
- **Bao nhiêu tác tử, mỗi tác tử làm gì?** 300 = 10 kho × 30 SKU; mỗi tác tử chọn
  1 trong 6 mức đặt hàng mỗi ngày cho đúng một cặp.
- **Có coordinator không?** Không. Tổng hợp cấp kho do môi trường làm và phát lại
  vào quan sát.
- **Vì sao là RL?** Quyết định tuần tự, hệ quả trễ 1–3 ngày, hành động làm đổi
  trạng thái, không có nhãn hành động tối ưu, mục tiêu dài hạn.
- **Reward âm có xấu không?** Không; reward = −chi phí nên luôn âm, chỉ đọc xu hướng.
- **RL có chắc thắng EOQ/(s,S) không?** Không. Khi nhu cầu ổn định, (s,S) rẻ hơn;
  ưu thế của IPPO nằm ở giai đoạn biến động và khi so ở cùng mức phục vụ.
- **Local reward có phải difference reward không?** Không; difference reward cần
  phản thực. Khóa luận dùng local factored reward với Σ r_i = R_team, và kiểm
  chứng bằng thí nghiệm reward toàn cục.

### Tài liệu tham khảo chính

Schulman et al. 2017 (PPO); Schröder de Witt et al. 2020 (IPPO, arXiv); Yu et
al. 2022 (MAPPO); Gupta et al. 2017 (parameter sharing); Oliehoek & Amato 2016
(Dec-POMDP); Wolpert & Tumer 2001 (credit assignment); Harris 1913; Scarf 1960;
Arrow et al. 1951; Makridakis et al. 2022 (M5). Danh mục đầy đủ:
`Report KLTN/backmatter/references.bib`.
