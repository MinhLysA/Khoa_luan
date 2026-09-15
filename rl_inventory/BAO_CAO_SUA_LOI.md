# Báo cáo chẩn đoán và sửa lỗi — bản v2

Tài liệu này ghi lại **vì sao bản cũ không hội tụ**, đã sửa những gì, và bằng
chứng số cho từng kết luận. Mọi thay đổi trong mã nguồn đều có comment `[V2-n]`
tại đúng vị trí, đối chiếu được với các mục dưới đây.

---

## 0. Kết quả trước và sau

| | Bản cũ | Bản v2 |
|---|---|---|
| Entropy chính sách | đứng ở mức tối đa `ln 6 = 1,792` rồi sập về 0,01 | giảm đều 1,79 xuống 0,33 |
| Explained variance (critic) | khoảng 0 | 0,69 |
| Chính sách học được | 494/500 cặp chọn "không đặt hàng" | phân hóa theo trạng thái |
| Fill rate (đánh giá) | 73,4% | **92,1%** |
| Chi phí tràn kho | 1.072.887 | 89.669 |
| So với baseline, so thẳng | **+71,2% (thua đậm)** | +11,4% ở mức phục vụ cao hơn 8 điểm |
| So với baseline, cùng mức phục vụ | — | **rẻ hơn (s,S) 11,7%** |

> Lưu ý: bản v2 mới chỉ chạy **173/800 episode** trên máy sandbox 1 nhân CPU.
> Xem Mục 9 để biết việc còn lại.

---

## 1. Lỗi số 1 — Actor và Critic dùng chung thân mạng

**File:** `agents/ppo_agent.py`

`SharedActorCriticNetwork` cũ có một `feature_net` dùng chung, và `update()` gộp
hai loss lại rồi gọi một lần `clip_grad_norm_(self.network.parameters(), 0.5)`.

Đo thực tế trên một minibatch 8.192 mẫu lúc khởi tạo:

```
policy_loss = 0,00362            value_loss = 1.142,77
grad_norm từ policy loss    = 0,096
grad_norm từ 0,5 * value loss = 28,130      ->  chênh 293 lần
```

Vì `max_grad_norm = 0,5` áp cho **tổng** gradient (chuẩn khoảng 28), cả cụm bị
nhân với `0,5 / 28 = 0,018`. `lr_actor = 1e-4` trở thành **1,8e-6 hiệu dụng**.
Đó là lý do entropy nằm im ở mức tối đa suốt giai đoạn đầu: actor gần như không
nhận được tín hiệu học, chỉ có critic học.

**Đã sửa:** tách `actor` và `critic` thành hai MLP độc lập (vẫn chia sẻ tham số
*giữa 300 tác tử* — luận điểm parameter sharing của khóa luận không đổi), hai
optimizer Adam riêng, cắt gradient riêng cho từng mạng.

Bổ sung kèm theo:

- **`[V2-10]` Value normalization** kiểu MAPPO/PopArt rút gọn: critic học trên
  return đã chuẩn hóa bằng trung bình và phương sai trượt, khử chuẩn hóa khi trả
  về môi trường. Giữ value loss ở độ lớn O(1) bất kể thang chi phí.
- **`[V2-11]` Value clipping** để một lần cập nhật không kéo critic đi quá xa.

---

## 2. Lỗi số 2 — Phần thưởng chênh nhau 4 bậc độ lớn giữa các cặp

**File:** `env/inventory_env.py`

Phần thưởng thô của một cặp (kho, SKU) trong một bước:

| Quy mô cặp | Cầu trung bình | Reward thô điển hình |
|---|---|---|
| SKU bán chậm | 0,1 đv/ngày | −0,5 |
| SKU bán nhanh | 143 đv/ngày | **−14.000** |

Một mạng critic dùng chung trọng số phải khớp mục tiêu trải 4 bậc độ lớn. Hệ quả
trực tiếp là `value_loss ≈ 1.143` ở Mục 1. Đây mới là nguyên nhân gốc, còn việc
dùng chung thân mạng là thứ khuếch đại nó lên.

**Đã sửa `[V2-2]`:** mỗi cặp được chia cho đơn vị chi phí riêng

```
reward_norm_pair = cp_th * mean_demand + cp_dh
```

tức "chi phí của một ngày xấu điển hình" của chính cặp đó. Hằng số `cp_dh` giữ
cho mẫu số không tiến về 0 ở các SKU gần như không bán.

Cờ `env.normalize_reward_per_pair: false` trong `config.yaml` tắt tính năng này.
Chạy ablation với cờ tắt sẽ **tái hiện lại đúng hiện tượng không hội tụ**, rất
đáng đưa vào Chương 4 làm bằng chứng cho đóng góp kỹ thuật.

---

## 3. Lỗi số 3 — Sức chứa kho dùng chung một con số cho cả 10 kho

**File:** `env/inventory_env.py`

Bản cũ:

```python
cau_moi_kho = self.mean_demand.reshape(n_wh, n_sku).sum(axis=1).mean()   # <- .mean()
self.suc_chua_kho = float(cau_moi_kho * capacity_cover_days)             # <- một số vô hướng
```

Lấy **trung bình cầu của các kho** rồi áp cùng một sức chứa cho tất cả. Với tập
dữ liệu cũ, sức chứa bằng 300 đơn vị cho mọi kho:

| Kho | Cầu/ngày | Sức chứa quy ra số ngày |
|---|---|---|
| CA_3 | 217,1 | **1,38 ngày** |
| CA_1 | 136,7 | 2,19 |
| WI_1 | 57,8 | 5,18 |

CA_3 chỉ chứa được 1,38 ngày cầu trong khi lead time tối đa là 3 ngày. Mục tiêu
fill rate 85% ở kho đó là **bài toán vô nghiệm**: mọi hành động đều bị phạt tràn
kho hoặc phạt mức phục vụ, thường là cả hai. Agent không có gradient nào dẫn tới
lời giải tốt hơn. Đây là lý do `overflow_cost` của IPPO bản cũ lên tới 1.072.887.

**Đã sửa `[V2-1]`:** sức chứa tính riêng cho từng kho theo cầu của chính kho đó.

Và `[V2-5]`: hàng vượt sức chứa nay **bị từ chối ngay lúc nhập** (đúng với thực
tế kho hàng), thay vì đã về kho rồi bị bốc hơi sau khi bán như bản cũ.

### Hiệu chuẩn `capacity_cover_days`

Quét bằng ba chính sách baseline trên môi trường v2:

| cover_days | Fill rate tốt nhất đạt được | Chi phí tràn kho |
|---|---|---|
| 3,0 | 79,5% — **không thể đạt mục tiêu 85%** | 503.667 |
| **5,0** | **85,7%** | 104.393 (khoảng 6% tổng chi phí) |
| 8,0 | 86,8% | 9.759 |
| 12,0 hoặc vô hạn | 87,0% | 0 — ràng buộc **không cắn**, mất luôn tính đa tác tử |

Chọn **5,0**: mục tiêu khả thi, mà 30 SKU trong cùng một kho vẫn thực sự tranh
nhau một sức chứa có hạn — điều kiện cần để bài toán là Dec-POMDP chứ không phải
300 bài toán độc lập ghép lại.

---

## 4. Lỗi số 4 — Bảng mức đặt hàng có các hành động trùng nhau

**File:** `env/inventory_env.py`

```python
order_qty_table = np.round(order_multipliers * mean_demand * lead_time_tb)
```

Với SKU bán chậm, `np.round` làm bẹp bảng. Ví dụ cặp có cầu 0,33 đv/ngày:

```
hệ số   [0,0   0,5   1,0   2,0   3,0   5,0]
-> bảng [0     0     1     1     2     3  ]     <- hành động 0 trùng 1, và 2 trùng 3
```

Thống kê trên toàn bộ 500 cặp của bản cũ:

| Số mức phân biệt được trên 6 | Số cặp |
|---|---|
| 2 | 178 (**35,6%**) |
| 3 | 44 |
| 4 | 56 |
| 5 | 74 |
| 6 | 148 |

Hai hành động khác nhau cho kết quả y hệt nhau thì advantage của chúng chỉ khác
nhau vì nhiễu, gradient triệt tiêu lẫn nhau, và entropy không có lý do gì để
giảm.

**Đã sửa `[V2-3]`:** dùng `np.ceil` rồi ép bảng tăng nghiêm ngặt (mức sau luôn
lớn hơn mức trước ít nhất 1 đơn vị), mức 0 luôn giữ nguyên là "không đặt". Có
unit test `test_bang_dat_hang_tang_nghiem_ngat` kiểm tra.

---

## 5. Lỗi số 5 — Quan sát không có tín hiệu cấp kho

**File:** `env/inventory_env.py`

Hàm thưởng phạt tràn kho theo **tổng tồn kho của cả kho**, nhưng vector quan sát
của một cặp không chứa bất kỳ thông tin nào về tổng đó. Môi trường mất tính
Markov đúng tại chỗ ghép nối giữa các tác tử, nên critic không thể học được hàm
giá trị nhất quán cho thành phần chi phí này.

**Đã sửa `[V2-4]`:** thêm 2 chiều quan sát

- mức sử dụng sức chứa của kho: `tổng tồn kho kho / sức chứa kho`
- tỷ trọng tồn kho của chính cặp trong kho

Kèm `[V2-6]`: lịch sử cầu chuyển sang thang log. Bản cũ chia cho
`mean_demand * 3` rồi `clip(0, 1)` nên mọi ngày bán trên 3 lần trung bình đều
bằng 1, mất sạch thông tin về các ngày cao điểm.

---

## 6. Các lỗi khác

### `[V2-17]` Lỗi chồng chỉ số trong `calendar_features`

**File:** `scripts/data_preprocessing.py`

Comment ghi bố cục `[3-7] event one-hot, [8-19] tháng`, nhưng code ghi
`features[i, 3 + EVENT_TYPES.index(et)]` (đưa Sporting vào cột 3) trong khi
`no_event` cũng ghi vào cột 3; one-hot tháng lại bắt đầu từ cột 7 thay vì cột 8.

Kiểm chứng trên file cũ: cột 3 có **1.799 ngày** (gộp no_event với Sporting),
cột 7 gộp Religious với tháng Một, và **cột 19 luôn bằng 0** (tháng 12 rơi vào
cột 18). Agent nhận tín hiệu lịch sai suốt quá trình huấn luyện. Đã sửa lại đúng
bố cục và in ra kiểm tra `tổng one-hot tháng = số ngày`.

### `[V2-12]` Xử lý truncation bù trừ chéo giữa hai file

**File:** `agents/rollout_buffer.py`, `scripts/train.py`

`train.py` cộng `gamma * V(s_T)` vào phần thưởng cuối, rồi `compute_gae` lại
nhân số hạng bootstrap với `(1 - episode_end)` để khử đi. Hai chỗ triệt tiêu
nhau nên kết quả đúng, nhưng sửa một bên là hỏng ngay mà không có lỗi nào báo.
Nay buffer nhận `next_value` trực tiếp và dùng công thức GAE chuẩn.

### `[V2-15]` Chọn checkpoint bằng reward lúc đang explore

**File:** `scripts/train.py`

Bản cũ chọn `best_model` theo trung bình trượt của reward *trong lúc đang lấy
mẫu ngẫu nhiên*. Đó không phải chất lượng chính sách. Nay chọn bằng đánh giá
deterministic (argmax) định kỳ trên **miền test**.

### `[V2-14]` Không có nhật ký đọc được bằng máy

Nay ghi `results/train_log*.csv` sau mỗi episode: reward, fill rate, 5 thành
phần chi phí, entropy, approx_kl, clip_frac, explained_variance, và kết quả
đánh giá deterministic. App Streamlit đọc trực tiếp file này.

### `[V2-16]` Cảnh báo chẩn đoán ngay trong lúc chạy

In cảnh báo khi `explained_variance < -0,5` hoặc `approx_kl > 0,05`.

---

## 7. Góp ý về quy mô bài toán

### Số kho: giữ **10**

Đúng 10 cửa hàng M5 (CA_1..4, TX_1..3, WI_1..3). Tự nhiên, dễ bảo vệ, không cần
giải thích thêm. Không thể tăng vì M5 chỉ có 10 store.

### Số SKU: **50 xuống 30**

Trong tập 50 SKU phân tầng cũ, **142/500 cặp có cầu trung bình dưới 0,1 đv/ngày**
— cửa hàng đó đơn giản là không kinh doanh mặt hàng đó. 28% "tác tử" chỉ đóng góp
nhiễu vào gradient của mạng dùng chung, và làm chậm huấn luyện tuyến tính.

`[V2-18]` Đã thêm bộ lọc `--min_mean_demand`: chỉ giữ SKU đạt ngưỡng cầu ở **ít
nhất 80% số cửa hàng**, tính trên miền huấn luyện.

| | Tập cũ (50 SKU) | Tập v2 (30 SKU) |
|---|---|---|
| Số cặp | 500 | 300 |
| Cặp gần như không bán (dưới 0,1) | 142 (28%) | **5 (1,7%)** |
| Cầu trung bình mỗi cặp | 2,01 | 3,24 |
| Trung vị | 0,40 | 0,74 |
| Tỷ lệ ngày bằng 0 | 70,6% | **57,4%** |
| Nhóm ngành hàng | FOODS, HOBBIES, HOUSEHOLD | đủ cả 3 |

57,4% ngày bằng 0 vẫn giữ nguyên đặc trưng gián đoạn và lumpy mà Chương 1 nhấn
mạnh, nên lập luận của Chương 1 không bị mâu thuẫn.

Nếu cần một ablation về khả năng mở rộng, chạy lại với `--n_skus 50` và báo cáo
thành mục "độ nhạy theo quy mô". Đừng dùng nó làm cấu hình chính.

### Siêu tham số PPO

| Tham số | Cũ | v2 | Lý do |
|---|---|---|---|
| `n_steps` | 2048 | 1024 | vẫn 307.200 mẫu mỗi update với 300 cặp, mà gấp đôi số lần cập nhật |
| `ppo_epochs` | 6 | 4 | giảm rủi ro off-policy |
| `lr_actor` | 1e-4 | 3e-4 | actor không còn bị critic ghì |
| `lr_critic` | 3e-4 | 1e-3 | đã có value normalization bảo vệ |
| `ent_coef` | 0,01 cố định | 0,02 xuống 0,002 (anneal) | explore đủ lúc đầu, dứt khoát lúc cuối |
| `target_kl` | tắt | 0,02 | chặn update kéo policy đi quá xa |
| `total_episodes` | 3000 | 800 | đủ; 800 episode khoảng 35 phút trên 1 nhân CPU |
| `capacity_cover_days` | 3,0 | **5,0** | xem Mục 3 |

---

## 8. Kết quả đánh giá bản v2

173 episode huấn luyện, 10 episode đánh giá, miền test.

```
Policy              Tổng CP       Lưu kho    Thiếu hàng     Đặt hàng    Tràn kho  FillRate
------------------------------------------------------------------------------------------
(s,S)             1.438.433       766.051       524.707      132.762      14.912    84,1%
Newsvendor        1.464.828       773.242       515.707      159.706      16.173    84,4%
IPPO              1.602.271     1.136.985       261.962      113.655      89.669    92,1%
EOQ               1.637.930       773.023       766.430       81.494      16.982    76,8%
```

IPPO **đạt fill rate cao nhất (92,1%)** và chi phí thiếu hàng thấp nhất (261.962,
bằng khoảng một nửa baseline), đổi lại chi phí lưu kho cao hơn. Nói cách khác nó
nằm ở một **điểm khác trên đường đánh đổi chi phí và mức phục vụ**, chứ không
phải thua về chất lượng.

### `[V2-25]` So sánh ở cùng mức phục vụ — đây mới là bảng để đưa vào Chương 4

So tổng chi phí giữa hai chính sách có fill rate khác nhau là vô nghĩa. IPPO bị
phạt khi fill rate dưới `muc_dv` nên nó tự đẩy mức phục vụ lên 92%, còn ba
baseline được tinh chỉnh để **tối thiểu chi phí** nên dừng lại ở khoảng 84%.
Bảng so thẳng ở trên sẽ dẫn tới kết luận sai là "IPPO đắt hơn".

`scripts/iso_service.py` làm đúng việc mà một phản biện sẽ yêu cầu: đo fill rate
IPPO đạt được trên miền test, rồi với **từng** baseline tìm kiếm lưới trên miền
train để lấy cấu hình **rẻ nhất vẫn đạt được mức phục vụ đó**, và đánh giá lại
trên miền test.

```
SO SÁNH Ở CÙNG MỨC PHỤC VỤ (miền test, ngưỡng >= 91,45%)

Chính sách        Tổng chi phí   Fill rate   So với IPPO
------------------------------------------------------------
IPPO                 1.598.726     91,95%
EOQ                    không đạt được mức phục vụ này
(s,S)                1.810.313     92,58%       -11,7%
Newsvendor           1.682.098     92,62%        -5,0%
```

**Ở cùng mức phục vụ, IPPO rẻ hơn (s,S) 11,7% và rẻ hơn Newsvendor 5,0%, còn EOQ
không có cấu hình nào vươn tới được mức đó.** Đây là kết luận đúng của thí
nghiệm, và nó vẫn chỉ mới dựa trên 173/800 episode huấn luyện.

Chạy lại bằng: `python run.py iso`

---

## 9. Việc còn lại — cần bạn chạy

1. **Chạy đủ 800 episode.** Checkpoint kèm theo mới chạy 173/800 trên sandbox
   1 nhân CPU. Lệnh: `python run.py train --episodes 800`, khoảng 35 phút.

2. **Chạy lại `python run.py iso` sau khi huấn luyện đủ**, với
   `--tune_episodes 3 --eval_episodes 30` để bảng iso-service ở Mục 8 có đủ độ
   tin cậy thống kê. Bảng đó, chứ không phải bảng so sánh thẳng, là bảng nên đưa
   vào Chương 4.

3. **Thử hiệu chỉnh `phi_dv`.** Hiện `phi_dv = 3,0` đẩy agent lên 92,1% fill
   rate, vượt xa mục tiêu 85%. Thử `phi_dv = 1,0` và `1,5` để xem IPPO có giữ
   được lợi thế chi phí khi hạ về đúng vùng 85 đến 87% hay không. Đây là thí
   nghiệm độ nhạy đáng có trong Chương 4.

4. **Đa hạt giống.** `python scripts/train.py --seed 1 --tag s1`, lặp 3 đến 5
   seed, báo cáo trung bình cộng trừ độ lệch chuẩn. Một seed duy nhất là điểm dễ
   bị phản biện nhất của mọi báo cáo RL.

5. **Ablation.** Các cờ đã sẵn trong `config.yaml`, đổi giá trị rồi chạy lại,
   không cần sửa code. Ablation đáng giá nhất cho Chương 4 là
   `normalize_reward_per_pair: false` — nó sẽ tái hiện đúng hiện tượng không hội
   tụ của bản cũ, tức bằng chứng trực tiếp cho đóng góp kỹ thuật của khóa luận.
