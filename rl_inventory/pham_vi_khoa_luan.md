# Khóa luận: Tối ưu hóa chính sách đặt hàng lại trong quản lý tồn kho đa kho bằng PS-IPPO

File này chỉ chứa **lý thuyết và phạm vi đề tài**.
Cách chạy: **[README.md](README.md)** · Tiến độ, kết quả sơ bộ, checklist báo cáo: **[../TIEN_DO.md](../TIEN_DO.md)** · Trả lời 27 yêu cầu của giảng viên: **[van_dap.md](van_dap.md)**.

**Mục lục**
1. [Đề tài, câu hỏi nghiên cứu, phạm vi](#1-đề-tài-câu-hỏi-nghiên-cứu-phạm-vi)
2. [Mô hình bài toán](#2-mô-hình-bài-toán)
3. [Dữ liệu và giao thức đánh giá](#3-dữ-liệu-và-giao-thức-đánh-giá)
4. [Thuật toán và siêu tham số](#4-thuật-toán-và-siêu-tham-số)
5. [Thiết kế thực nghiệm](#5-thiết-kế-thực-nghiệm)
6. [Lịch sử thiết kế và sửa lỗi quan trọng](#6-lịch-sử-thiết-kế-và-sửa-lỗi-quan-trọng)
7. [Hướng phát triển (ngoài phạm vi)](#7-hướng-phát-triển-ngoài-phạm-vi)
8. [Câu hỏi thường gặp của hội đồng](#8-câu-hỏi-thường-gặp-của-hội-đồng)

---

## 1. Đề tài, câu hỏi nghiên cứu, phạm vi

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

## 2. Mô hình bài toán

### 2.1. Từ nghiệp vụ tới tác tử

Quy trình một ngày tại mỗi kho, mỗi mã hàng: nhận hàng (hàng vượt chỗ trống bị
từ chối) → kiểm tra tồn kho, hàng đang về, lịch sử bán, mức đầy kho → **quyết định
lượng đặt** → bán hàng (thiếu thì mất đơn) → ghi nhận chi phí và mức phục vụ.

Chỉ có **một** quyết định cần tối ưu (lượng đặt), lặp cho từng mã hàng tại từng
kho, do người phụ trách mã hàng đó chịu trách nhiệm. Vì vậy nhiệm vụ tổng được
chia thành 300 nhiệm vụ con cùng loại: **1 tác tử = 1 cặp (kho, mặt hàng)**.
Không có coordinator.

### 2.2. Kiến trúc đa tác tử

- **300 tác tử, dùng chung một hàm chính sách π_θ**: mỗi tác tử tự quyết định từ
  quan sát riêng và nhận phần thưởng riêng. Chia sẻ tham số không gộp 300 tác tử
  thành một tác tử trung tâm.
- **"Independent" trong IPPO:** GAE/advantage tính riêng cho từng tác tử; không
  có critic tập trung (khác MAPPO).
- **Phối hợp gián tiếp qua 3 kênh:** (1) mức đầy kho `u_t^(w)` do môi trường tính
  và phát lại vào quan sát của 30 tác tử cùng kho; (2) phạt tràn kho khi cùng
  tranh sức chứa; (3) trọng số dùng chung khi huấn luyện.

### 2.3. MDP / Dec-POMDP

| Thành phần | Nội dung |
|---|---|
| Mục tiêu đội | `R_team = Σ_i r_i`; tín hiệu huấn luyện được phân rã thành `r_i` cục bộ (local factored reward). Đây **không phải** difference reward (vốn cần phản thực `G(z) − G(z₋ᵢ)`). |
| Quan sát (44 chiều) | tồn kho, vị thế tồn kho (2) · cầu 7 ngày, thang log (7) · hàng đang về theo ngày (3) · fill rate 30 ngày (1) · quy mô log(1+D̄) (1) · mức đầy kho, tỷ trọng trong kho (2) · **tín hiệu giảm giá (1)** · thứ trong tuần (7) · lịch: SNAP, sự kiện, tháng (20). |
| Hành động | 6 mức: `q = ceil(m·D̄·L̄)`, m = (0; 0,5; 1; 2; 3; 5), ép tăng nghiêm ngặt. |
| Chuyển trạng thái | Nhận hàng → kiểm tra sức chứa (chỉ từ chối **hàng mới về** vượt chỗ trống, chia theo tỷ trọng hàng về) → ghi đơn mới (L ~ U{1,2,3}) → nhu cầu thật M5, mất đơn nếu thiếu → cập nhật lịch sử và fill rate. |
| Phần thưởng | `r_i = −(C_i + P_i)`. Chi phí vận hành `C = lưu kho + thiếu hàng + đặt hàng + tràn kho` (4 thành phần). `P` = phạt mức phục vụ, **là tín hiệu huấn luyện, không phải chi phí thứ năm**, không tính vào chi phí báo cáo. |
| Chuẩn hóa | `r_i / (c_th·D̄_i + c_dh)` để các cặp có quy mô chênh 4 bậc về cùng thang. |
| γ | 0,99 |

**Tín hiệu giảm giá [V3-4]** (`env.include_price_signal: true`, bật từ 25/09/2026):
`clip(1 − giá_hôm_nay / giá_TB_của_cặp, 0, 1)`, trong đó giá theo ngày lấy từ
`sell_prices.csv` (`price_series.npy`) và giá trung bình chỉ tính trên miền train.
Bằng 0 khi giá bình thường hoặc tăng, càng gần 1 khi giảm giá càng sâu. Lý do: giá
bán M5 đổi theo tuần (khuyến mãi) và đợt giảm giá thường đi trước hoặc trùng lúc
cầu tăng đột biến, nên đây là chỉ báo sớm. Tắt cờ thì quay về 43 chiều.
⚠️ Chương 3 báo cáo cần cập nhật tương ứng (bảng đối chiếu biến trạng thái, công
thức `obs_per_pair`).

**Reward luôn âm** (= −chi phí); không có khoảng reward "chuẩn". Chỉ đọc xu hướng
(càng gần 0 càng tốt) và so tương đối trong cùng bài toán.

### 2.4. Mục tiêu 85% là ràng buộc mềm

Mục tiêu nghiệp vụ: `min E[Σ C]` sao cho `FR ≥ 85%`. Khi huấn luyện, ràng buộc
được đưa vào dưới dạng phạt mềm `P`. Tính khả thi được kiểm tra ở khâu chọn mô
hình: checkpoint IPPO = chi phí vận hành val thấp nhất trong số checkpoint có
`FR_val ≥ 85%`; baseline tune theo đúng tiêu chí đó.

**Hai cách định cỡ phạt SLA** (`env.service_penalty_mode`):
- `demand` (cũ): `P = φ·c_th·D̄·max(τ−FR, 0)` → mặt hàng bán chậm bị phạt rất
  nhẹ, chính sách phân bổ mức phục vụ thấp cho chúng (39% cặp dưới 85%).
- `normalized` (**cấu hình chính**): `P = φ·(c_th·D̄ + c_dh)·max(τ−FR, 0)`
  → sau chuẩn hóa, mọi cặp chịu cùng mức phạt `φ·max(τ−FR, 0)`.

### 2.5. Tham số chi phí

| c_lk | c_th | c_dh | p_tk | φ_dv | τ | Sức chứa |
|---|---|---|---|---|---|---|
| 1 | 10 | 5 | 5 | 3 | 85% | 5 ngày cầu của từng kho |

Đơn vị là chi phí mô phỏng chuẩn hóa, không phải USD (M5 không công bố giá vốn).
Sức chứa 5 ngày được hiệu chỉnh bằng thực nghiệm: ở 3 ngày mục tiêu 85% bất khả
thi (tối đa 79,5%); từ 12 ngày trở lên ràng buộc không còn hiệu lực.

---

## 3. Dữ liệu và giao thức đánh giá

### 3.1. Dữ liệu

M5 Walmart: 10 cửa hàng (CA_1..4, TX_1..3, WI_1..3), 1.941 ngày. Chọn 30 SKU
phân tầng, chỉ giữ SKU có cầu ≥ 0,2/ngày ở ≥ 80% cửa hàng (loại mặt hàng cửa
hàng không kinh doanh). Tập 300 cặp: cầu TB 3,24, trung vị 0,74, 57,4% ngày bằng
0 (vẫn giữ tính gián đoạn).

### 3.2. Ba miền

| Miền | Ngày | Được dùng để |
|---|---|---|
| Train | [0, 1050) | rollout PPO; ước lượng D̄, sức chứa, mẫu số chuẩn hóa, giá TB |
| Validation | [1050, 1450) | chọn checkpoint IPPO; tune baseline |
| Test | [1450, 1941) | **chỉ** báo cáo kết quả cuối |

### 3.3. Đánh giá cuối (`env.test_full_horizon: true`)

- Mỗi lần đánh giá chạy **trọn 491 ngày** test, nhu cầu thật cố định; lặp **30
  seed thời gian giao hàng**. Mọi chính sách dùng cùng seed → so sánh theo cặp.
  (Thay cho giao thức cũ: 30 cửa sổ 365 ngày trong miền 491 ngày, chồng lấn mạnh.)
- Kiểm định trên `ΔC_j = C_j^IPPO − C_j^baseline`: chênh lệch TB + CI 95%, paired
  t-test, Wilcoxon, `d_z`. Không dùng Welch; không gộp 90 lần IPPO với 30 lần
  baseline.
- 3 seed huấn luyện: báo cáo từng seed, rồi TB ± SD giữa các seed.
- Theo giai đoạn nhu cầu: bootstrap theo khối 7 ngày (CI 95%).

### 3.4. Baseline và chỉ số

- EOQ (Harris 1913), (s,S) (Scarf 1960), Newsvendor (Arrow et al. 1951); dùng
  cùng thông tin với IPPO và cùng 6 mức đặt (ràng buộc lô chuẩn chung, để cô lập
  tác động của thuật toán).
- Hai phiên bản tham số baseline, cùng tune trên val theo ràng buộc 85%: **bộ
  chung** cho cả 300 cặp, và **theo 4 nhóm quy mô cầu** (<0,5; 0,5–2; 2–10; ≥10
  đơn vị/ngày; tìm theo tọa độ từ bộ chung) — baseline mạnh hơn, tránh so IPPO với
  đối thủ bị làm yếu.
- Chỉ số: tổng chi phí vận hành, fill rate (tổng bán/tổng cầu), cơ cấu chi phí,
  fill rate từng cặp, đường học (reward, entropy, explained variance, KL).

---

## 4. Thuật toán và siêu tham số

PS-IPPO: Actor và Critic **tách riêng** (mỗi mạng 2 lớp ẩn 128, Tanh), 2
optimizer Adam, cắt gradient riêng; value normalization; value clipping; GAE
theo từng cặp; chuẩn hóa advantage theo cặp.

| n_steps | epochs | minibatch | lr actor / critic | γ / λ | clip | target KL | entropy | episode |
|---|---|---|---|---|---|---|---|---|
| 1024 | 4 | 16.384 | 3e-4 / 1e-3 | 0,99 / 0,95 | 0,2 | 0,02 | 0,02 → 0,002 | 4.000 (chính), 1.000 (ablation) |

Đánh giá tất định trên val mỗi 25 episode (2 episode). Lưu checkpoint mỗi 200
episode; `--resume` train tiếp đúng từ trạng thái đã lưu.

### Cờ cấu hình (`config.yaml`)

| Cờ | Ý nghĩa |
|---|---|
| `env.service_penalty_mode` | `normalized` (chính) / `demand` (cũ) |
| `env.reward_mode` | `local` (chính) / `global` (ablation RQ3) |
| `env.normalize_reward_per_pair` | chuẩn hóa theo cặp (ablation RQ3) |
| `env.include_price_signal` | tín hiệu giảm giá (**bật**, 44 chiều; tắt → 43 chiều) |
| `env.test_full_horizon` | đánh giá trọn 491 ngày test |
| `env.obs_drop` | tắt nhóm quan sát (`calendar`, `warehouse`, …) |
| `env.include_event_lookahead` | thêm 2 đặc trưng sự kiện sắp tới |
| `env.warm_start_history` | lịch sử cầu khởi tạo bằng dữ liệu thật |
| `env.sku_indices` | tập con SKU (hold-out) |
| `ppo.shared_trunk` | Actor/Critic chung thân (ablation) |

---

## 5. Thiết kế thực nghiệm

Mỗi ablation chỉ khác `abl_ref` **đúng một thay đổi**, cùng ngân sách 1.000 episode.

| Lần chạy | Config | Episode | Trả lời |
|---|---|---|---|
| `main_s42`, `main_s1`, `main_s2` | `config.yaml` | 4.000 | RQ1, RQ2; mặt hàng bán chậm còn bị phân bổ mức phục vụ thấp không |
| `abl_ref` (+`_s1`, `_s2`) | `config.yaml` | 1.000 | mốc cho mọi ablation (cùng lịch lr/entropy) |
| `abl_global` (+`_s1`, `_s2`) | `ablation_global_reward` | 1.000 | RQ3: cục bộ vs toàn cục (3 seed) |
| `abl_q3` (+`_s1`, `_s2`) | `ablation_q3_no_reward_norm` | 1.000 | RQ3: có/không chuẩn hóa (3 seed) |
| `abl_reward_cu` | `ablation_reward_demand_scaled` | 1.000 | tác động của việc định cỡ lại phạt SLA |
| `abl_trunk` | `ablation_shared_trunk` | 1.000 | vì sao phải tách Actor/Critic |
| `abl_nocal`, `abl_nowh` | `ablation_drop_*` | 1.000 | state có biến dư thừa không |
| `abl_event` | `ablation_event_lookahead` | 1.000 | cải thiện mùa lễ |
| `abl_warm` | `ablation_warm_start` | 1.000 | ảnh hưởng lịch sử rỗng đầu episode |
| `holdout` | `holdout_train` → `holdout_eval` | 1.000 | RQ2 mở rộng: 6 SKU chưa gặp (3, 8, 9, 10, 23, 25) |

**Phân tích sau huấn luyện:** so sánh ở cùng mức phục vụ ~90% (phụ); thực nghiệm
B theo giai đoạn nhu cầu và sốc cầu; hành vi chính sách (permutation importance);
theo nhóm quy mô cầu; độ nhạy 36 bộ đơn giá chi phí; **độ bền zero-shot** (sức
chứa 4/6 ngày, lead time luôn 3 hoặc 2–3 ngày, nhu cầu ±20%); tổng hợp đa hạt
giống và RQ3 theo seed (kiểm định theo cặp với `abl_ref` cùng seed).

---

## 6. Lịch sử thiết kế và sửa lỗi quan trọng

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

**v3:** chia 3 miền (bản v2 chọn checkpoint trên chính test); lệnh multiseed;
[V3-4] tín hiệu giảm giá trong quan sát (bật cho đợt cuối, 44 chiều).

**P0:** tràn kho chỉ từ chối hàng mới về (bản cũ trừ oan tồn kho cũ); fill rate
trong log là fill rate thật của cả episode; chọn checkpoint theo chi phí thấp
nhất trong nhóm đạt 85%; tắt chi phí thiếu hàng theo giá (M5 không có giá vốn).

**P2/P3 (đợt cuối):** tune baseline trên val theo ràng buộc 85%; kiểm định theo
cặp; đánh giá trọn quỹ đạo test; bootstrap khối tuần; phạt SLA `normalized`;
reward toàn cục; warm-start; obs_drop; sự kiện sắp tới; hold-out SKU; shared
trunk; độ nhạy chi phí; MCP; app chi tiết theo ngày.

---

## 7. Hướng phát triển (ngoài phạm vi)

Lead time > 3 ngày và mục tiêu mức phục vụ 80/90% (cần huấn luyện lại: số chiều
quan sát phụ thuộc lead time tối đa, hệ số phạt hiệu chỉnh cho 85%); nhân tử
Lagrange riêng từng cặp; chuyển hàng liên kho, multi-echelon; MAPPO/critic tập
trung; communication giữa tác tử; action liên tục; dự báo LSTM/Transformer; 3.049
SKU; triển khai thực tế; lớp giao tiếp ngôn ngữ qua MCP.

---

## 8. Câu hỏi thường gặp của hội đồng

Bản đầy đủ, kèm bằng chứng (hình, bảng, dòng code): **[van_dap.md](van_dap.md)**.

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
