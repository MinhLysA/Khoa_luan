# Vấn đáp với giảng viên — trả lời 27 yêu cầu

*Cập nhật 24/09/2026. Mỗi câu gồm: **Trả lời ngắn** (nói trực tiếp) → **Giải thích**
→ **Bằng chứng** (vị trí trong báo cáo / code). Số liệu đánh dấu (★) là kết quả
hiện có theo protocol cũ (3 seed × 1.000 episode); sẽ thay bằng số của đợt thực
nghiệm cuối. Thông tin kỹ thuật đầy đủ: [pham_vi_khoa_luan.md](pham_vi_khoa_luan.md).*

> **Ba câu cần nói nhất quán trong mọi câu trả lời**
> 1. Có **300 tác tử**, mỗi tác tử là **một cặp (kho, mặt hàng)**, tự quyết định lượng đặt hàng từ quan sát của riêng nó.
> 2. 300 tác tử **dùng chung một hàm chính sách π_θ** (chia sẻ tham số), nhưng **mỗi tác tử ra quyết định riêng và nhận phần thưởng riêng**. Không có coordinator.
> 3. Chi phí vận hành gồm **4 thành phần** (lưu kho, thiếu hàng, đặt hàng, tràn kho). Mức phục vụ 85% là **ràng buộc**, đưa vào huấn luyện dưới dạng hình phạt, **không phải chi phí thứ năm**.

---

## Phần I — Thuật toán và kiến trúc tác tử

### 1. Thuật toán là gì? Chữ "I" nghĩa là gì?

**Trả lời ngắn.** Thuật toán là **IPPO — Independent Proximal Policy Optimization**,
dùng kèm **chia sẻ tham số**, nên nhóm gọi là **PS-IPPO**. Chữ "I" là
*Independent*: mỗi tác tử học như một bài toán PPO riêng, với quan sát riêng, phần
thưởng riêng và advantage tính riêng; không có thành phần nào nhìn toàn hệ thống.

**Giải thích.**
- **PPO** (Schulman et al., 2017) là thuật toán policy gradient on-policy: cập nhật
  chính sách theo hướng tăng lợi thế (advantage), nhưng **cắt** tỉ số xác suất
  mới/cũ trong khoảng [1−ε, 1+ε] (ε = 0,2) để mỗi lần cập nhật không đẩy chính sách
  đi quá xa. Dùng kiến trúc Actor (chọn hành động) – Critic (ước lượng giá trị).
- **IPPO khác PPO ở đâu:** PPO gốc dành cho **một** tác tử. IPPO áp PPO cho **nhiều**
  tác tử, mỗi tác tử coi các tác tử khác là một phần của môi trường.
- **"Independent" có ba tầng nghĩa:**
  | Tầng | Có độc lập không? |
  |---|---|
  | Ra quyết định: mỗi tác tử chọn hành động chỉ từ quan sát của nó | **Có** |
  | Học: phần thưởng riêng, GAE/advantage tính riêng theo từng tác tử | **Có** |
  | Tham số: mỗi tác tử một bộ trọng số riêng | **Không** — dùng chung (parameter sharing) |
- **Mỗi tác tử có policy riêng không?** Chính xác là: 300 tác tử **dùng chung một
  hàm chính sách π_θ(a | o)**, nhưng hàm đó được áp lên **quan sát riêng** của từng
  tác tử nên cho ra **quyết định riêng**. Hai cặp ở hai trạng thái khác nhau sẽ hành
  động khác nhau dù cùng θ. Chia sẻ tham số giúp 300 tác tử học từ kinh nghiệm
  chung; nó **không** gộp 300 tác tử thành một tác tử trung tâm (nếu gộp, không gian
  hành động là 6³⁰⁰, không học được).
- **Học độc lập như thế nào:** mỗi ngày 300 tác tử cùng hành động trong một môi
  trường chung; bộ đệm lưu (quan sát, hành động, phần thưởng, giá trị) theo từng
  tác tử; GAE tính theo trục thời gian riêng cho từng cột tác tử; sau đó mọi mẫu
  được gộp để cập nhật θ chung.
- **Dùng chung môi trường và dữ liệu không:** có. Cùng một môi trường mô phỏng 10
  kho, cùng chuỗi nhu cầu M5; các tác tử trong cùng kho chia sẻ **sức chứa**.
- **Vì sao IPPO hợp bài toán đa kho:** (1) quyết định thực tế diễn ra ở từng mã
  hàng tại từng kho, đúng dạng phi tập trung; (2) mỗi tác tử chỉ có 6 hành động dù
  hệ thống có 300 tác tử, nên mở rộng được; (3) chia sẻ tham số giúp mặt hàng ít dữ
  liệu học từ mặt hàng khác; (4) Yu et al. (2022) và Schröder de Witt et al. (2020)
  báo cáo IPPO cạnh tranh được với các phương pháp tập trung trên nhiều bài toán hợp
  tác.

**Bằng chứng.** Báo cáo §2.2.4 (PPO), §2.3.3 (IPPO, ba tầng nghĩa), §2.3.4 (chia
sẻ tham số). Code: `agents/ppo_agent.py` (`SharedActorCriticNetwork`, dòng 89),
`agents/rollout_buffer.py:68–97` (GAE theo từng tác tử).

---

### 2. Kiến trúc multi-agent cụ thể

| Câu hỏi | Trả lời |
|---|---|
| Có bao nhiêu agent? | **300** = 10 kho × 30 mặt hàng. |
| Mỗi agent đại diện cho gì? | **Một cặp (kho, mặt hàng)** — không phải một kho, không phải một mặt hàng, mà là giao điểm của cả hai. Ví dụ: tác tử (CA_3, FOODS_3_090). |
| Agent nhận dữ liệu gì? | Vector quan sát **43 chiều** (xem câu 7). |
| Nhiệm vụ? | Mỗi ngày quyết định **đặt thêm bao nhiêu** cho đúng cặp của mình, để giảm chi phí vận hành của cặp đó trong khi giữ fill rate ≥ 85%. |
| Action? | Chọn **1 trong 6 mức đặt**: q = ⌈m × cầu TB × lead time TB⌉, m ∈ {0; 0,5; 1; 2; 3; 5}. Toàn hệ thống: `MultiDiscrete([6]*300)`. |
| Policy riêng không? | **Dùng chung một hàm π_θ**, áp lên quan sát riêng → quyết định riêng (xem câu 1). |
| Agent nào ra quyết định đặt hàng? | **Cả 300 tác tử**, đồng thời, mỗi tác tử cho đúng cặp của mình. Không có tác tử nào quyết định thay tác tử khác. |
| Agent nào tổng hợp thông tin? | **Không có agent tổng hợp.** Việc tổng hợp cấp kho (cộng tồn kho của 30 mặt hàng để tính mức đầy kho) do **môi trường** làm, rồi phát lại con số đó vào quan sát của 30 tác tử cùng kho. |
| Trao đổi dữ liệu thế nào? | **Không trao đổi trực tiếp** (không message passing). Phối hợp **gián tiếp qua 3 kênh**: (1) **mức đầy kho** do môi trường phát lại; (2) **phạt tràn kho**: khi kho đầy, hàng mới về bị từ chối và phạt theo tỷ trọng hàng về của từng cặp; (3) **trọng số dùng chung** khi huấn luyện. Tác tử thấy hậu quả gộp của các tác tử cùng kho ở bước sau, nhưng không thấy hành động cụ thể của từng tác tử khác. |
| Có coordinator không? | **Không.** Không có coordinator, không có bộ điều khiển trung tâm, không có critic tập trung (khác MAPPO). |

**Bằng chứng.** Hình 3.3 (kiến trúc đa tác tử), Bảng 3.4 (đặc tả tác tử), §3.1.5
(phối hợp). Code: tính mức đầy kho trong `env/inventory_env.py` `_get_observation()`
(dòng 533, `util` ở dòng 553); từ chối hàng tràn kho ở dòng 411.

---

### 3. Không chia agent tùy ý — đi từ nghiệp vụ

**Trả lời ngắn.** Số tác tử được **suy ra từ quy trình nghiệp vụ**, không đặt trước.

**Trình tự lập luận (báo cáo §3.1.1):**
1. **Quy trình một ngày** tại mỗi kho, mỗi mã hàng: nhận hàng → kiểm tra tồn kho,
   hàng đang về, lượng bán gần đây, mức đầy kho → **quyết định lượng đặt** → bán
   hàng (thiếu thì mất đơn) → ghi nhận chi phí và mức phục vụ.
2. **Quyết định cần tối ưu:** chỉ có **một** — lượng đặt bổ sung. Các bước còn lại
   là hệ quả vật lý hoặc ghi nhận.
3. **Trách nhiệm:** quyết định đó thuộc về **người phụ trách mã hàng tại kho**;
   sức chứa kho là tài nguyên chung, không ai phân bổ tường minh.
4. **Task:** "tối thiểu chi phí vận hành toàn mạng dưới ràng buộc sức chứa và mức
   phục vụ" được chia thành 300 task con cùng loại.
5. **Agent:** mỗi task con là một tác tử = một cặp (kho, mặt hàng).

**Bằng chứng.** Báo cáo §3.1.1, Hình 3.1.

---

### 4. Task riêng hay task tổng chia nhỏ?

**Trả lời ngắn.** **Hướng 2 — task decomposition.** Task tổng được chia thành 300
task con **cùng loại** (cùng là "quyết định lượng đặt"), mỗi task con giao cho
một tác tử.

**Vì sao không chọn hướng 1** (agent quan sát / agent quyết định / agent điều phối):
- Trong thực tế không có "người quan sát" tách khỏi "người quyết định"; người phụ
  trách mã hàng tự làm cả hai.
- Chia theo chức năng buộc phải **thiết kế thêm** giao thức truyền tin giữa các
  agent — một thành phần không có trong bài toán gốc và làm thay đổi đối tượng
  nghiên cứu, đồng thời sinh câu hỏi ai chịu trách nhiệm khi hai agent bất đồng.
- Chia theo điểm ra quyết định thì mỗi hành động gắn đúng một đòn bẩy vận hành có
  thật, và chi phí của mỗi tác tử gắn đúng với quyết định của nó (dễ gán tín dụng).

**Bằng chứng.** Báo cáo §3.1.3 "Phân rã nhiệm vụ và thiết kế tác tử"; Bảng 3.4
dòng "Nhiệm vụ".

---

### 5. Có coordinator / agent tổng không?

**Trả lời ngắn.** **Không có.** Nhóm nói nhất quán: các tác tử độc lập, và **không
có** agent tổng.

**Vậy các tác tử tự phối hợp thế nào?** Qua 3 kênh gián tiếp ở câu 2. Bằng chứng
thực nghiệm cho phối hợp tự nổi lên: khi tăng mức đầy kho từ 20% lên 95% (giữ
nguyên các chiều quan sát khác), tác tử **không đổi ngưỡng đặt hàng** nhưng **thu
nhỏ lô hàng** (mức tối đa giảm từ 5,65 xuống 5,09 ngày cầu), và 4 cặp ngừng đặt
khi kho gần đầy (★). Hành vi này không được lập trình mà do tác tử tự học.

**Vì sao không thêm coordinator:** (1) nghiệp vụ không có vai trò đó; (2) thêm
critic tập trung sẽ biến bài toán thành MAPPO (CTDE), tức một đối tượng nghiên cứu
khác. So sánh với MAPPO được ghi vào hướng phát triển.

**Bằng chứng.** Báo cáo §3.1.5.

---

## Phần II — Đưa bài toán về học tăng cường

### 6. Vì sao đây là bài toán RL?

**Trả lời ngắn.** Vì nó có đủ 4 đặc điểm mà học có giám sát hay tối ưu một bước
không xử lý được:
1. **Quyết định tuần tự, hệ quả trễ:** đơn hôm nay 1–3 ngày sau mới về.
2. **Hành động làm đổi trạng thái:** đặt hàng thay đổi tồn kho, tức thay đổi bài
   toán của ngày mai.
3. **Không có nhãn hành động tối ưu:** không có dữ liệu "hôm đó lẽ ra nên đặt bao
   nhiêu"; chỉ quan sát được chi phí **sau khi** hành động.
4. **Mục tiêu dài hạn:** tối thiểu chi phí cả kỳ; tích trữ hôm nay tốn chi phí
   nhưng tránh thiếu hàng ngày mai.

Quy hoạch động không khả thi vì trạng thái liên tục 43 chiều × 300 tác tử và nhu
cầu M5 không có phân phối dạng đóng.

**Bằng chứng.** Báo cáo §2.3.1, Bảng 3.2 (ánh xạ MDP).

---

### 7. State cụ thể gồm những biến nào?

**Trả lời ngắn.** Quan sát của mỗi tác tử là vector **43 chiều**, mọi giá trị chuẩn hóa về [0, 1]:

| Biến | Số chiều | Có trong quan sát? | Ghi chú |
|---|---|---|---|
| Tồn kho hiện tại | 1 | Có | quy về số ngày cầu |
| Vị thế tồn kho (tồn + hàng đang về) | 1 | Có | |
| Lịch sử nhu cầu | 7 | Có | 7 ngày gần nhất, thang log |
| Hàng đang về (outstanding/incoming) | 3 | Có | theo từng ngày còn lại tới hạn |
| Tình trạng thiếu hàng | 1 | Có | fill rate cửa sổ 30 ngày |
| Quy mô mặt hàng | 1 | Có | log(1 + cầu TB) |
| Mức đầy kho (warehouse capacity) | 1 | Có | tín hiệu cấp kho |
| Tỷ trọng tồn kho của cặp trong kho | 1 | Có | tín hiệu cấp kho |
| Ngày trong tuần | 7 | Có | one-hot |
| Lịch: SNAP, loại sự kiện, tháng | 20 | Có | |
| Nhu cầu hiện tại | — | **Không** | quyết định diễn ra trước khi nhu cầu hôm đó phát sinh; đưa vào là rò rỉ tương lai |
| Lead time của từng đơn | — | **Gián tiếp** | nhà cung cấp không cam kết ngày giao; thể hiện qua pipeline |
| Thông tin kho khác | — | **Không** | các kho độc lập về nguồn cung |
| Giá bán | — | **Không** | cờ `include_price_signal: false` |

**Có biến dư thừa không:** kiểm tra bằng thực nghiệm ở câu 17.

**Bằng chứng.** Bảng 3.1 (đối chiếu biến trạng thái); code `env/inventory_env.py`
(`obs_per_pair` dòng 283, `_get_observation` dòng 533).

---

### 8. Action Space

**Trả lời ngắn.** Mỗi tác tử có **một** loại hành động: **chọn mức đặt bổ sung**
trong 6 mức rời rạc.

| Mức | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Hệ số m | 0 (không đặt) | 0,5 | 1 | 2 | 3 | 5 |
| Lượng đặt | 0 | ⌈m × D̄ × L̄⌉ … ép tăng nghiêm ngặt | | | | |

- Mức tính **tương đối theo cầu trung bình của chính cặp** (73% chuỗi M5 có cầu dưới
  1 đơn vị/ngày, nên một thang tuyệt đối dùng chung không hợp lý).
- Bảng được ép tăng nghiêm ngặt để không có hai hành động trùng lượng đặt.
- **Không có** hành động chuyển hàng giữa kho hay chọn nhà cung cấp — ngoài phạm vi
  (các kho độc lập về nguồn cung).
- Vì mọi tác tử cùng nhiệm vụ nên cùng không gian hành động.

**Bằng chứng.** Công thức (3.1), Bảng 3.2; code `_build_order_table` (dòng 323),
`action_space` (dòng 261).

---

### 9. Reward Function — tác tử đang tối ưu cái gì?

**Trả lời ngắn.** Tác tử tối thiểu **chi phí vận hành của cặp mình**, có kèm hình
phạt khi mức phục vụ dưới 85%:

```
r_i,t = −( c_lk·Tồn_kho  +  c_th·Thiếu_hàng  +  c_dh·1(có đặt)  +  p_tk·Hàng_tràn )  −  P_i,t
        └──────────────── chi phí vận hành C (4 thành phần) ───────────────┘     └ phạt mức phục vụ
```

| Thành phần | Ý nghĩa nghiệp vụ | Giá trị |
|---|---|---|
| Lưu kho | vốn bị giữ, mặt bằng | c_lk = 1 / đv / ngày |
| Thiếu hàng | doanh thu và thiện chí khách hàng bị mất (mất đơn) | c_th = 10 / đv |
| Đặt hàng | chi phí cố định mỗi lần phát đơn (xử lý, vận chuyển) | c_dh = 5 / lần |
| Tràn kho | hàng về bị từ chối vì kho đầy — tổn thất do đặt vượt sức chứa | p_tk = 5 / đv |
| Phạt mức phục vụ P | cam kết giữ fill rate ≥ 85% (**ràng buộc mềm, không phải chi phí**) | φ = 3 |

- **Không có chi phí chuyển kho (transfer cost)** vì mô hình không có chuyển hàng
  giữa kho.
- Chi phí báo cáo khi so sánh chỉ gồm **4 thành phần vận hành**; P không được cộng vào.
- Mục tiêu toàn đội là tổng phần thưởng 300 cặp; phần thưởng được **phân rã cục bộ**
  cho từng cặp (local factored reward) để giảm khó khăn gán tín dụng. RQ3 kiểm
  chứng lựa chọn này bằng cách so với **phần thưởng toàn cục**.
- Phần thưởng mỗi cặp được **chuẩn hóa** theo quy mô chi phí của cặp đó
  (c_th·D̄ + c_dh), vì phần thưởng thô giữa mặt hàng bán chậm và bán nhanh chênh
  nhau khoảng 4 bậc độ lớn.

**Bằng chứng.** Công thức (3.4), (3.6), (3.7); §3.2; code `_calculate_reward`
(dòng 592).

---

### 10. Reward âm không có nghĩa là mô hình xấu

**Trả lời ngắn.** Vì reward = −(chi phí + phạt) và mọi chi phí đều ≥ 0, reward
**luôn ≤ 0**. Reward bằng 0 chỉ xảy ra khi không lưu kho, không thiếu hàng, không
đặt hàng — điều không thể khi có nhu cầu. Nhóm chỉ đọc **xu hướng**: reward tăng
dần về phía 0 tương đương chi phí giảm dần.

**Ví dụ từ thực nghiệm (★):** tổng reward một episode tăng từ khoảng **−16,99 triệu**
ở episode đầu lên ổn định quanh **−1,69 triệu**, tức chi phí giảm khoảng 90% so với
chính sách khởi tạo ngẫu nhiên — dù reward vẫn âm.

**Bằng chứng.** Báo cáo §3.2.1 đoạn "Vì sao phần thưởng luôn âm"; app trang So
sánh policy có ghi chú này dưới bảng theo ngày.

---

### 11. Không có khoảng reward "chuẩn"

**Trả lời ngắn.** Độ lớn tuyệt đối của reward phụ thuộc vào đơn giá chi phí, quy
mô nhu cầu của dữ liệu, số tác tử (300) và độ dài episode (365 ngày). Vì thế nhóm
**không** so reward với một khoảng cố định như [0, 1] hay [−1, 1], mà chỉ đánh giá
**tương đối**: so với chính sách ngẫu nhiên ban đầu, giữa các giai đoạn huấn luyện,
và **quy đổi về chi phí vận hành** để so với chính sách cổ điển trên cùng thang đo.

**Bằng chứng.** Báo cáo §3.2.1.

---

### 12. Chứng minh agent thực sự học

**Trả lời ngắn.** Có đường học 4 chỉ số (Hình 4.1) cho thấy tác tử học thật, không
chỉ nhìn reward cuối:

| Chỉ số | Đầu huấn luyện | Sau hội tụ (★) | Ý nghĩa |
|---|---|---|---|
| Reward/episode | −16,99 triệu | ≈ −1,69 triệu (dao động −1,76 … −1,64 triệu) | chi phí giảm khoảng 90% |
| Mốc ổn định | — | khoảng episode 600 | hội tụ, không sụp đổ sau đó |
| Fill rate | thấp | vượt 85% từ episode ~66, ổn định quanh 90% | |
| Entropy chính sách | 1,778 (≈ ln 6, gần ngẫu nhiên) | 0,182 | chính sách đã định hình |
| Explained variance (Critic) | ≈ 0 | 0,70–0,87 | Critic dự đoán được giá trị |
| KL xấp xỉ | — | 2,4×10⁻⁴ | chính sách ổn định giữa các lần cập nhật |

Kết quả lặp lại trên **3 hạt giống huấn luyện** độc lập với cùng xu hướng. Đợt
thực nghiệm cuối sẽ vẽ lại với 3 seed × 4.000 episode.

**Bằng chứng.** Hình 4.1, §4.4.1; log `results/train_log_seed*.csv`.

---

### 13. MDP của hệ thống kho

| Thành phần | Trong đề tài |
|---|---|
| **Environment** | Mô phỏng 10 kho × 30 mặt hàng trên nhu cầu thật M5; 1 bước = 1 ngày; 1 episode = 365 ngày |
| **S** — trạng thái | Vector 43 chiều của từng cặp (câu 7) |
| **A** — hành động | Chọn 1 trong 6 mức đặt (câu 8) |
| **P** — chuyển trạng thái | (1) đổi hành động thành lượng đặt → (2) nhận hàng đến hạn, **kiểm tra sức chứa**, chỉ từ chối hàng mới về vượt chỗ trống → (3) ghi đơn mới, lead time ngẫu nhiên 1–3 ngày → (4) nhu cầu thật phát sinh, bán min(tồn, cầu), thiếu thì mất đơn → (5) cập nhật lịch sử và fill rate → trạng thái mới. Hai nguồn ngẫu nhiên: lead time và nhu cầu. |
| **R** — phần thưởng | −(chi phí vận hành + phạt mức phục vụ) của riêng cặp (câu 9) |
| **Policy** | π_θ(a \| o): mạng Actor dùng chung cho 300 tác tử |
| **Value** | V_φ(o): mạng Critic dùng chung |
| γ | 0,99 |

**Luồng:** `s_t` (tồn kho, hàng đang về, cầu 7 ngày, mức đầy kho, lịch) → tác tử
chọn `a_t` (mức đặt) → môi trường nhận hàng, bán hàng, tính chi phí → `r_t` (−chi
phí cục bộ) và `s_{t+1}` → tác tử lưu kinh nghiệm và cập nhật trọng số.

Về hình thức, hệ thống là **Dec-POMDP** (nhiều tác tử, mỗi tác tử quan sát một
phần) với mục tiêu đội `R_team = Σ r_i`.

**Bằng chứng.** Bảng 3.2 (ánh xạ MDP), §3.1.3, công thức (2.9).

---

### 14. Sơ đồ RL gắn với đề tài

**Đã có:** **Hình 3.2 — "Vòng tương tác học tăng cường của một tác tử trong môi
trường kho"** (vẽ bằng TikZ, không lấy từ internet). Thể hiện: tác tử cặp (kho w,
mặt hàng j) với Actor/Critic → **hành động** a_t ⇒ đặt q_i → **môi trường kho**
(nhận hàng, kiểm tra sức chứa, ghi đơn vào pipeline, nhu cầu phát sinh, tính chi
phí) → **phần thưởng** r_t và **quan sát mới** o_{t+1} → **bộ đệm rollout** (GAE
riêng từng tác tử) → **cập nhật PPO** cho θ, φ.

### 15. Sơ đồ kiến trúc multi-agent

**Đã có:** **Hình 3.3 — "Kiến trúc đa tác tử"**. Thể hiện: các kho (Kho 1, Kho 2,
…, Kho 10), mỗi kho chứa các tác tử A_{w,1} … A_{w,30} và **sức chứa W_w dùng
chung → mức đầy u_t^(w)**; **chính sách dùng chung** Actor π_θ + Critic V_φ ở trên
(mũi tên o_i lên, a_i xuống); **môi trường dùng chung** (nhu cầu M5, pipeline giao
hàng, chi phí, mức đầy kho); khách hàng (nhu cầu) và nhà cung cấp (đơn q_i, hàng
về); chú thích vai trò – đầu vào – đầu ra – cách phối hợp – **không có coordinator**.

Ngoài ra có **Hình 3.1 — mô hình khái niệm** (câu 23).

### 16. Bảng đặc tả tác tử

| Thành phần | Nội dung |
|---|---|
| **Agent** | Tác tử replenishment của cặp (kho w, mặt hàng j); 300 tác tử đồng nhất về vai trò |
| **Role** | Người phụ trách bổ sung hàng cho đúng một mặt hàng tại đúng một kho |
| **Input** | Vector quan sát cục bộ o_i^t ∈ [0,1]⁴³ |
| **State** | Tồn kho, vị thế tồn kho, cầu 7 ngày, hàng đang về theo ngày, fill rate 30 ngày, quy mô, mức đầy kho, tỷ trọng trong kho, thứ trong tuần, lịch |
| **Task** | Task con của task tổng "tối thiểu chi phí vận hành toàn mạng dưới ràng buộc sức chứa và mức phục vụ 85%" |
| **Action** | a_i^t ∈ {0,…,5} → lượng đặt q_i |
| **Policy** | π_θ(a \| o) — Actor dùng chung trọng số θ cho 300 tác tử |
| **Output** | Lượng đặt q_i gửi nhà cung cấp mỗi ngày |
| **Interaction** | Không truyền tin trực tiếp; gián tiếp qua mức đầy kho, phạt tràn kho và trọng số dùng chung |

**Bằng chứng.** Bảng 3.4.

---

## Phần III — Kiểm chứng và so sánh

### 17. Kiểm tra state và reward bằng thực nghiệm

**Trả lời ngắn.** Nhóm kiểm tra 3 câu hỏi bằng thực nghiệm (§4.5.5, Hình 4.6, Bảng 4.11):

1. **Hành vi có hợp lý về nghiệp vụ không?** Có (★). Xác suất đặt hàng giảm dần khi
   tồn kho tăng: 99% khi còn 1–2 ngày hàng → 51% (2–3 ngày) → 12% (5–8 ngày) → 7%
   (trên 8 ngày). Tương quan Spearman −0,42. Chỉ 1,4% quyết định dùng mức đặt lớn
   nhất; chi phí tràn kho chỉ 1,5% tổng chi phí.
2. **Tác tử có lách reward không?** Không dồn sát ngưỡng 85% (chỉ 13,7% cặp nằm
   trong [85%, 88%)). **Nhưng phát hiện một dạng khai thác:** 39% số cặp có fill rate
   dưới 85% (so với 6,3% của (s,S) cùng mức phục vụ) — chính sách phân bổ mức phục
   vụ thấp hơn cho mặt hàng bán chậm, vì phạt SLA tỷ lệ với cầu trung bình nên với
   mặt hàng bán chậm, phạt rất nhỏ so với chi phí đặt hàng. Nhóm đã **sửa cách định
   cỡ phạt** (`normalized`: mọi cặp chịu cùng mức phạt sau chuẩn hóa) và kiểm chứng
   trong đợt thực nghiệm cuối.
3. **State có biến dư thừa không?** Permutation importance — xáo trộn từng nhóm
   đặc trưng và đo chi phí tăng: quy mô (+98%), tồn kho (+94%), lịch sử cầu (+50%),
   fill rate (+17%), pipeline (+8%), lịch (+3,5%), tín hiệu cấp kho (+2,8%). Không
   nhóm nào vô dụng hoàn toàn; lịch và tín hiệu cấp kho đóng góp ít. Đợt cuối có
   thí nghiệm **huấn luyện lại khi bỏ hẳn** hai nhóm này để khẳng định.

---

### 18. So sánh với chính sách truyền thống — không mặc định RL thắng

**Trả lời ngắn.** Nhóm **không** đặt giả thuyết "RL chắc chắn tốt hơn". Ba baseline
EOQ, (s,S) (tương đương reorder point có lô), Newsvendor (dạng base-stock) được
**tinh chỉnh tham số** trên cùng môi trường, trên miền validation, theo **cùng tiêu
chí** với IPPO (chi phí thấp nhất trong số cấu hình đạt fill rate ≥ 85%).

**Kết quả hiện có cho thấy RL không thắng mọi nơi (★):**
- **So thẳng:** IPPO đạt fill rate 90,5% nhưng **chi phí cao hơn (s,S) 4,5–5,5%** ở
  cả 3 seed.
- **Khi nhu cầu ổn định:** (s,S) rẻ hơn IPPO 2,5%; ở ngày thường rẻ hơn 4,2% (có ý
  nghĩa thống kê).
- **Ở cùng mức phục vụ ~90%:** IPPO rẻ hơn (s,S) 10,2%, Newsvendor 14,4%; EOQ không
  đạt được mức đó.
- **Độ nhạy:** so thẳng, IPPO chỉ rẻ hơn (s,S) ở 22/36 bộ đơn giá; ở cùng mức phục
  vụ thì rẻ hơn ở 36/36.

**Bằng chứng.** §4.2 (baseline), §4.4.2, §4.4.4, Bảng 4.6, Bảng 4.12.

---

### 19–21. RL trong môi trường động — Thực nghiệm A và B

**Trả lời ngắn.** Nhóm làm đúng hai nhóm thực nghiệm giảng viên gợi ý:
- **Thực nghiệm A (dài hạn):** trung bình trên toàn kỳ đánh giá.
- **Thực nghiệm B (theo ngày / giai đoạn):** gán nhãn từng ngày trong miền test theo
  **mức biến động** (độ lệch cầu so với trung bình 28 ngày trước, chia 3 nhóm) và
  theo **lịch** (ngày có sự kiện, SNAP, cuối tuần, mùa lễ T11–T12, ngày thường); thêm
  **sốc cầu nhân tạo** ×1,5 và ×0,5 trong 28 ngày. Độ tin cậy bằng bootstrap theo
  khối 7 ngày.

**Kết quả (★) — đúng kiểu kết luận giảng viên mô tả ở câu 21:**
- **Nhu cầu ổn định → chính sách truyền thống rẻ hơn:** (s,S) rẻ hơn IPPO ở ngày ổn
  định (+2,5%) và ngày thường (+4,2%), có ý nghĩa thống kê.
- **Nhu cầu biến động → IPPO tốt hơn:** fill rate của IPPO chỉ giảm 2,4 điểm từ ngày
  ổn định sang ngày biến động, trong khi (s,S) giảm 7,3 điểm. Ở **ngày có sự kiện**
  và **cuối tuần**, IPPO vừa rẻ hơn (s,S) khoảng 10% vừa phục vụ tốt hơn khoảng 10
  điểm (có ý nghĩa thống kê).
- **Sốc tăng cầu 50%:** mọi chính sách đều mất khoảng 7–9 điểm fill rate, nhưng IPPO
  tốn ít hơn 29–32% chi phí để chống đỡ.
- **Sốc giảm cầu rồi hồi phục:** baseline tụt còn 87–88% (vì cửa sổ 7 ngày còn
  "nhớ" giai đoạn cầu thấp), IPPO giữ 90%.
- **Điểm yếu:** mùa lễ cuối năm là giai đoạn IPPO kém nhất (+8,7% chi phí so với
  (s,S)), nhưng chưa có ý nghĩa thống kê vì chỉ có 9 tuần dữ liệu.

Nhóm **không** chỉnh thực nghiệm để ép RL thắng; mọi trường hợp RL thua đều được
báo cáo.

**Bằng chứng.** §4.5.4, Bảng 4.8–4.10, Hình 4.4–4.5; script `scripts/regime_analysis.py`.

---

### 22. Business metrics

| Chỉ số | Dùng thế nào |
|---|---|
| **Tổng chi phí vận hành** | chỉ số chính (4 thành phần) |
| **Fill rate** (service level β) | ràng buộc ≥ 85%; báo cáo cả toàn hệ thống và **từng cặp** |
| Cơ cấu chi phí | lưu kho / thiếu hàng / đặt hàng / tràn kho → nhận diện chiến lược |
| Tồn kho trung bình, số lần đặt | trong bảng kết quả chi tiết |
| Số ngày–cặp thiếu hàng | trong app và MCP |

Phân biệt rõ: mức phục vụ theo chu kỳ (α, liên quan critical ratio Newsvendor) khác
fill rate (β, mục tiêu 85%) — §2.1.3.

### 23. Conceptual Model

**Đã có:** **Hình 3.1 — "Mô hình khái niệm của bài toán bổ sung hàng đa kho"**:
Nhu cầu khách hàng (M5) → Mạng lưới kho (10 × 30) → Trạng thái tồn kho → Hệ đa tác
tử IPPO → Quyết định bổ sung hàng → Chi phí & mức phục vụ → Trạng thái tồn kho mới
→ quay lại ngày t+1. Mỗi khối có chú thích nghiệp vụ bên cạnh.

---

## Phần IV — MCP và mô phỏng

### 24–25. MCP — Model Context Protocol

**Trả lời ngắn.** MCP là giao thức chuẩn để một mô hình ngôn ngữ lớn (LLM) gọi công
cụ bên ngoài. Nhóm đã tìm hiểu và làm một **prototype** (`app/mcp_server.py`), để
minh họa luồng điều phối — **không phải nội dung chính** của khóa luận (trình bày ở
Phụ lục B).

**Trả lời các câu hỏi điều phối theo đúng luồng Context → Task → Action/Tool →
thực hiện → kết quả:**

| Câu hỏi | Trong prototype |
|---|---|
| Task đến từ đâu, ai nhận? | Người quản lý kho hỏi bằng ngôn ngữ tự nhiên ("tuần tới kho CA_3 có mặt hàng nào sắp thiếu?"); **LLM (host)** nhận task |
| Task cần action nào? | LLM chọn **tool** phù hợp trong 5 tool: `canh_bao_thieu_hang`, `de_xuat_dat_hang`, `mo_phong`, `what_if`, `danh_sach_kho_va_mat_hang` |
| Agent nào thực hiện? | Máy chủ MCP gọi **engine IPPO** (và baseline) — IPPO vẫn là nơi ra quyết định định lượng |
| Context lấy từ đâu? | Từ môi trường mô phỏng: tồn kho, hàng đang về, lịch sử bán, lịch; và resource `ket-qua://tong-hop` (kết quả đánh giá chính thức) |
| Kết quả trả về thế nào? | Tool trả JSON (danh sách cặp sắp thiếu, lượng đặt đề xuất, KPI) → LLM tổng hợp thành câu trả lời |

**So với kiến trúc IPPO:** IPPO **không cần** lớp phân công task vì mỗi tác tử có
task cố định; MCP phù hợp khi cần giao tiếp ngôn ngữ và chọn công cụ linh hoạt. Hai
lớp bổ sung cho nhau: MCP lo giao tiếp, IPPO lo quyết định.

**Đã kiểm chứng:** client MCP chuẩn kết nối qua stdio, liệt kê đủ 5 tool và gọi
thành công.

### 26–27. Mô phỏng và dashboard theo SKU, theo ngày

**Trả lời ngắn.** Có đủ chuỗi **Model → Simulation → Interface**: app Streamlit
(`python run.py app`) gồm 4 trang:

| Trang | Nội dung |
|---|---|
| Tổng quan | chọn ngày; heatmap tồn kho kho × SKU (đỏ = thiếu hàng, vàng = tồn dư); KPI |
| Đề xuất đặt hàng | nhập tình huống một cặp → IPPO và 3 baseline cùng đề xuất; áp dụng rồi chạy tiếp từng ngày |
| So sánh policy | 4 chính sách chạy trên **cùng chuỗi cầu**; **bảng chi tiết từng ngày cho một cặp kho–SKU** |
| What-if | đổi lead time, sức chứa, chi phí hoặc tạo sốc cầu |

**Bảng theo ngày cho một cặp kho–SKU** đúng luồng ngày t → nhu cầu xuất hiện → tác
tử đọc state → quyết định đặt hàng → kho cập nhật → tính chi phí/reward → ngày t+1,
hiển thị đủ các cột giảng viên liệt kê:

| SKU | Inventory | Demand | Order quantity | Incoming | Stockout | Cost | Reward | Cảnh báo |
|---|---|---|---|---|---|---|---|---|
| chọn kho + SKU | Tồn kho cuối ngày | Cầu | Đặt hàng | Hàng đang về | Thiếu hàng | Chi phí của cặp | Reward của cặp | ⛔ thiếu hàng / ⚠️ sắp hết (< 2 ngày cầu) |

Kèm KPI của cặp (số ngày thiếu hàng, fill rate, tổng chi phí, tổng reward) và biểu
đồ cầu – tồn kho – hàng đang về – lượng đặt theo ngày.

---

## Phụ lục — Những điểm dễ bị hỏi vặn

| Câu hỏi | Trả lời gọn |
|---|---|
| "Nói agent độc lập mà lại dùng chung mạng?" | Độc lập về quyết định và phần thưởng; chung về tham số. Không có agent nào quyết định thay agent khác. |
| "Local reward có phải difference reward?" | Không. Difference reward cần tính phản thực G(z) − G(z₋ᵢ); nhóm dùng local factored reward (Σ r_i = R_team) và kiểm chứng bằng thí nghiệm reward toàn cục (RQ3). |
| "Phạt 85% có phải chi phí không?" | Không. Là ràng buộc mềm trong huấn luyện; chi phí báo cáo chỉ gồm 4 thành phần. Tính khả thi kiểm tra khi chọn checkpoint (FR_val ≥ 85%). |
| "Có rò rỉ dữ liệu test không?" | Không. 3 miền: train [0,1050) để học; validation [1050,1450) để chọn checkpoint và tune baseline; test [1450,1941) chỉ để báo cáo. |
| "Baseline có bị làm yếu đi không?" | Baseline được tune trên validation theo đúng tiêu chí của IPPO, dùng cùng thông tin quan sát và cùng 6 mức đặt (ràng buộc lô chuẩn chung). |
| "Kết quả có phụ thuộc may rủi một lần chạy?" | Đánh giá 3 hạt giống huấn luyện; kiểm định theo cặp (cùng hạt giống môi trường) với khoảng tin cậy 95%, paired t-test, Wilcoxon, d_z. |
| "Tại sao 30 SKU, không phải toàn bộ 3.049?" | Chọn phân tầng theo quy mô cầu, loại mặt hàng cửa hàng không kinh doanh; vẫn giữ tính gián đoạn (57% ngày bằng 0). Mở rộng quy mô là hướng phát triển. |
