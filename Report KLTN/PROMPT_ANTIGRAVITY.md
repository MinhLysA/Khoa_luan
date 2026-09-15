# NHIỆM VỤ: Hiệu chỉnh khóa luận LaTeX theo kết quả thẩm định học thuật

## 0. Bối cảnh & Cấu trúc Repository

Bạn đang làm việc trên repository chứa mã nguồn LaTeX của một khóa luận tốt nghiệp đại học (tiếng Việt), đề tài:

> **"Tối ưu hóa chính sách đặt hàng lại trong quản lý tồn kho đa kho bằng học tăng cường: thuật toán Independent Multi-Agent PPO (IPPO)"**

### Cấu hình bài toán:
- 10 kho song song (KHÔNG multi-echelon) × 50 SKU = 500 tác tử
- Parameter sharing một mạng Actor–Critic
- Hành động rời rạc 6 mức: `{0, 10, 20, 30, 40, 50}`
- Lead time: `Uniform(1,3)`
- Lost sales; ràng buộc sức chứa kho dùng chung
- Hàm thưởng phân rã cục bộ kèm phạt fill rate cửa sổ trượt 30 ngày
- Dữ liệu: M5 Walmart; Baseline: EOQ / (s,S) / Newsvendor

### Cấu trúc tệp tin trong Repository:
- Tệp cấu hình Preamble & Main: [`preamble.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/preamble.tex), [`main.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/main.tex)
- Tệp Tài liệu tham khảo BibLaTeX: [`backmatter/references.bib`](file:///d:/A.N%C4%82M%204/Report%20KLTN/backmatter/references.bib)
- Các tệp nội dung Chương:
  - Chương 1: [`content/C1.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C1.tex)
  - Chương 2: [`content/C2.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C2.tex)
  - Chương 3: [`content/C3.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C3.tex)
  - Chương 4: [`content/C4.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C4.tex)
  - Chương 5: [`content/C5.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C5.tex)
- Kịch bản biên dịch tự động: [`compile.bat`](file:///d:/A.N%C4%82M%204/Report%20KLTN/compile.bat) hoặc [`run.bat`](file:///d:/A.N%C4%82M%204/Report%20KLTN/run.bat)

**Engine biên dịch:** XeLaTeX (`fontspec` + `Times New Roman`) + Biber. Không đổi engine.  
**Ngôn ngữ:** Tiếng Việt. Giữ nguyên phong cách hành văn hiện có.

---

## 1. QUY TẮC BẮT BUỘC

1. **KHÔNG được bịa trích dẫn:** Chỉ dùng đúng các tài liệu được liệt kê tường minh trong nhiệm vụ này. Nếu cần một nguồn không có trong danh sách, hãy dừng lại và hỏi, tuyệt đối không tự tạo mục `.bib` mới từ trí nhớ.
2. **KHÔNG được bịa số liệu thực nghiệm:** Các chỗ đang có `[cần điền]`, `[cần phân tích]` trong [`content/C4.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C4.tex) và [`content/C5.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C5.tex) phải **giữ nguyên**. Không điền số, không viết kết luận thay.
3. **KHÔNG sửa những phần không được yêu cầu:** Không đổi bố cục chương, không đổi template, không "cải thiện" hành văn ngoài phạm vi các mục dưới đây.
4. **Mỗi thay đổi = một commit riêng** (hoặc ghi rõ từng bước thay đổi) với thông điệp mô tả rõ ràng.
5. **Sau khi sửa xong, biên dịch kiểm tra:** Chạy [`compile.bat`](file:///d:/A.N%C4%82M%204/Report%20KLTN/compile.bat) (hoặc lệnh `xelatex` → `biber` → `xelatex` × 2) và báo cáo: số lỗi, số cảnh báo citation/reference chưa phân giải. Nếu có lỗi, sửa cho đến khi sạch.
6. **Xuất tệp CHANGELOG:** Tạo tệp `CHANGELOG_HIEUCHINH.md` liệt kê từng thay đổi kèm vị trí file/dòng tương ứng.

---

## 2. NHIỆM VỤ A — Sửa lỗi trong tệp `backmatter/references.bib`

### A1. Sửa các mục sai hiện tại

Chỉnh sửa chính xác các entry trong [`backmatter/references.bib`](file:///d:/A.N%C4%82M%204/Report%20KLTN/backmatter/references.bib):

| Citation Key | Lỗi hiện tại | Sửa thành |
|---|---|---|
| `wolpert2001` | năm `2002` | năm **2001** (`Advances in Complex Systems` 4(2–3), 265–279, DOI `10.1142/S0219525901000188`) |
| `gijsbrechts2022` | thiếu phụ đề & tác giả 4 | Tiêu đề đầy đủ: `Can Deep Reinforcement Learning Improve Inventory Management? Performance on Lost Sales, Dual-Sourcing, and Multi-Echelon Problems`<br>Tác giả 4: `Dennis J. Zhang` |
| `oroojlooyjadid2022` | tên tác giả 2 | `MohammadReza Nazari` (viết liền, hoa chữ R) |
| `dewitt2020ippo` | họ tác giả đầu | `Schröder de Witt, Christian` (đảm bảo hiển thị và sắp xếp alphabet đúng theo họ `Schröder de Witt`) |
| `yu2022mappo` | thiếu track & trang | Bổ sung: `booktitle = {Advances in Neural Information Processing Systems 35 (NeurIPS 2022), Datasets and Benchmarks Track}`, `pages = {24611--24624}` |
| `agarwal2021` | thiếu trang | Bổ sung `pages = {29304--29320}` (NeurIPS 34) |
| `silver2016` | tên tác giả đầu | `Silver, Edward A.` (kiểm tra chuẩn `Edward A. Silver`), NXB `CRC Press` |
| `ng1999` | tiêu đề rút gọn | Tiêu đề đầy đủ: `Policy invariance under reward transformations: Theory and application to reward shaping` |

### A2. Tách trích dẫn M5 thành hai mục riêng

Trong [`backmatter/references.bib`](file:///d:/A.N%C4%82M%204/Report%20KLTN/backmatter/references.bib), tách trích dẫn M5 hiện tại (`m5dataset` / `makridakis2022background`) thành hai mục rõ ràng:

1. **Mục 1 (Bài báo khoa học - `makridakis2022background`):**  
   Makridakis, Spiliotis & Assimakopoulos (2022), *"The M5 competition: Background, organization, and implementation"*, *International Journal of Forecasting* 38(4), 1325–1336, DOI `10.1016/j.ijforecast.2021.07.007`
2. **Mục 2 (Bộ dữ liệu Kaggle - `m5dataset`):**  
   Trích theo định dạng competition của Kaggle. Ban tổ chức là Makridakis Open Forecasting Center (MOFC), Đại học Nicosia, dữ liệu do Walmart cung cấp, năm 2020.
   > ⚠️ **CẦN CON NGƯỜI XÁC NHẬN:** Chèn comment `% TODO: xác nhận danh sách tác giả từ nút Cite trên Kaggle` ngay trên mục `m5dataset`, không tự điền danh sách tác giả bịa.

### A3. Thêm các mục mới vào `backmatter/references.bib`

Thêm đúng 8 mục BibTeX sau vào [`backmatter/references.bib`](file:///d:/A.N%C4%82M%204/Report%20KLTN/backmatter/references.bib) (dùng cho Nhiệm vụ B, C, D):

1. **`kostenko2006`**: Kostenko & Hyndman (2006), *"A note on the categorization of demand patterns"*, *Journal of the Operational Research Society* 57(10), 1256–1257, DOI `10.1057/palgrave.jors.2602211`
2. **`yang2023mabim`**: Yang, Liu, Jiang, Zhang, Zhao, Song & Bian (2023), *"A Versatile Multi-Agent Reinforcement Learning Benchmark for Inventory Management"*, arXiv:2306.07542
3. **`ding2022cdppo`**: Ding, Feng, Liu, Jiang, Zhang, Zhao, Song, Li, Jin & Bian (2022), *"Multi-Agent Reinforcement Learning with Shared Resources for Inventory Management"*, arXiv:2212.07684 (NeurIPS 2022 RL4RealLife Workshop)
4. **`liu2025happo`**: Liu, Hu, Peng & Yang (2025), *"Multi-Agent Deep Reinforcement Learning for Multi-Echelon Inventory Management"*, *Production and Operations Management*, DOI `10.1177/10591478241305863`
5. **`kotecha2024decentralized`**: Kotecha & del Rio Chanona (2024), *"An analysis of multi-agent reinforcement learning for decentralized inventory control systems"*, *Computers & Chemical Engineering*, arXiv:2307.11432
6. **`hubbs2020orgym`**: Hubbs, Perez, Sarwar, Sahinidis, Grossmann & Wassick (2020), *"OR-Gym: A Reinforcement Learning Library for Operations Research Problems"*, arXiv:2008.06319
7. **`demoor2022reward`**: De Moor, Gijsbrechts & Boute (2022), *"Reward shaping to improve the performance of deep reinforcement learning in perishable inventory management"*, *European Journal of Operational Research* 301(2), 535–545
8. **`devlin2011potential`**: Devlin & Kudenko (2011), *"Theoretical considerations of potential-based reward shaping for multi-agent systems"*, AAMAS 2011

---

## 3. NHIỆM VỤ B — Viết lại Mục 2.4.2 (Khoảng trống nghiên cứu) & Bổ sung lược khảo trong `content/C2.tex`

### B1. Vấn đề hiện tại
Mục 2.4.2 trong [`content/C2.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C2.tex) hiện tuyên bố rằng chưa có công trình nào (a) mở rộng RL tồn kho tới hàng trăm cặp kho–SKU bằng parameter sharing, (b) đánh giá tường minh phân rã hàm thưởng ở quy mô đó, (c) hiệu chỉnh môi trường từ dữ liệu bán lẻ thực.

**Tuyên bố này SAI.** Các công trình sau đã thực hiện phần lớn các nội dung trên:
- **MABIM (`yang2023mabim`)**: benchmark MARL cho tồn kho, mỗi SKU tại mỗi kho là một tác tử, dùng dữ liệu cầu thực của >2000 SKU, 51 tác vụ gồm nhóm "scaling up", có đánh giá IPPO, có ràng buộc sức chứa dùng chung, giao diện Gym.
- **CD-PPO (`ding2022cdppo`)**: số lượng lớn SKU, mỗi SKU một tác tử, ràng buộc tài nguyên dùng chung (sức chứa kho) ghép nối các SKU độc lập (đúng cấu trúc đề tài). So sánh trực tiếp với IPPO/MAPPO parameter sharing.
- **Liu et al. 2025 (`liu2025happo`)**: dùng HAPPO, phát hiện phần thưởng kết hợp chi phí cục bộ + chi phí toàn hệ tốt hơn chi phí toàn hệ thuần.
- **Kotecha & del Rio Chanona 2024 (`kotecha2024decentralized`)**: so sánh IPPO / IPPO-shared-network / MAPPO, báo cáo hiệu năng tụt của IPPO-shared khi tăng số tác tử.
- **OR-Gym (`hubbs2020orgym`)**: benchmark môi trường RL cho OR, có newsvendor và multi-echelon lost-sales.

### B2. Việc cần làm trong `content/C2.tex` và `content/C3.tex`

1. **Bổ sung Lược khảo tại Mục 2.4.1 ([`content/C2.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C2.tex)):**  
   Thêm 2–4 câu cho mỗi công trình trên (MABIM, CD-PPO, Liu et al., Kotecha & del Rio Chanona, OR-Gym), nêu rõ quy mô, thuật toán và nguồn dữ liệu.
2. **Viết lại Mục 2.4.2 ([`content/C2.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C2.tex)):**  
   Bỏ các khẳng định tuyệt đối ("chưa có công trình nào", "đầu tiên"). Định vị lại đóng góp theo hướng **tái lập và mở rộng có kiểm soát**:
   - (i) Khảo sát cấu hình **đa kho song song thuần túy, không multi-echelon** (khác MABIM và Liu et al. vốn là multi-echelon);
   - (ii) **Ablation study tường minh** ảnh hưởng của phân rã phần thưởng cục bộ ở quy mô 500 tác tử;
   - (iii) Hiệu chỉnh môi trường lost-sales từ dữ liệu M5 Walmart với đối chứng cổ điển EOQ / (s,S) / Newsvendor.
3. **Mở rộng Bảng 2.1 (`tab:soSanhCongTrinh` trong [`content/C2.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C2.tex)):**  
   Đổi thành bảng so sánh 6 cột:  
   `Nghiên cứu | Cấu trúc mạng | Số SKU | Parameter sharing | Dữ liệu thực | Thuật toán`  
   Thêm các dòng tương ứng cho MABIM, CD-PPO, Liu et al. (2025), Kotecha & del Rio Chanona (2024).
4. **Bổ sung giải thích tại Mục 3.4 ([`content/C3.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C3.tex)):**  
   Giải thích lý do tự xây dựng môi trường thay vì dùng trực tiếp OR-Gym hay MABIM (do cần cấu hình đa kho song song với ràng buộc sức chứa kho dùng chung mà các benchmark đó không hỗ trợ sẵn). Nếu cần xác nhận thêm, chèn comment `% TODO` cho tác giả quyết định.

---

## 4. NHIỆM VỤ C — Sửa hai lỗi lý thuyết trong `content/C2.tex`, `content/C3.tex`, `content/C5.tex`

### C1. Lỗi Reward Shaping (Mục 2.3.5, 3.2.2 và 5.1) — NGHIÊM TRỌNG

- **Vấn đề:** Văn bản hiện tại trong [`content/C3.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C3.tex) (dòng 101-103) và [`content/C5.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C5.tex) (dòng 18-21) khẳng định sai rằng thành phần phạt fill rate cửa sổ trượt *"phù hợp với khung lý thuyết reward shaping của Ng và cộng sự (1999)"* và *"không làm thay đổi nghiệm tối ưu"*.
- **Vì sao sai:** Theo định lý của Ng, Harada & Russell (1999), hàm shaping bảo toàn chính sách tối ưu **khi và chỉ khi** có dạng thế năng $F(s,a,s') = \gamma\Phi(s') - \Phi(s)$. Phạt fill rate cửa sổ trượt 30 ngày không có dạng này. Hơn nữa, mục đích thiết kế thành phần phạt này chính là **cố ý thay đổi hành vi** hướng tới mức phục vụ cao hơn.
- **Việc cần làm:**
  - Xóa mọi viện dẫn Ng et al. (1999) như căn cứ bảo toàn nghiệm tối ưu tại Mục 3.2.2 ([`content/C3.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C3.tex)) và Mục 5.1 ([`content/C5.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C5.tex)).
  - Trình bày thành phần phạt fill rate dưới dạng **hình phạt ràng buộc (constraint penalty) theo tinh thần Lagrangian**, thừa nhận tường minh rằng nó **thay đổi nghiệm tối ưu của MDP gốc** để thỏa mãn mục tiêu kinh doanh.
  - Tại Mục 2.3.5 ([`content/C2.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C2.tex)), trình bày **chính xác** định lý Ng et al. (1999) về dạng $\gamma\Phi(s') - \Phi(s)$, rồi trích dẫn De Moor et al. (2022) (`demoor2022reward`) và Devlin & Kudenko (2011) (`devlin2011potential`) làm ví dụ ứng dụng potential-based reward shaping trong tồn kho và MARL.

### C2. Nhầm lẫn giữa Cycle Service Level ($\alpha$) và Fill Rate ($\beta$) (Mục 2.1 và 3.2.3)

- **Vấn đề:** Chi phí $c_{th} = 10$ và $c_{lk} = 1$ cho critical ratio $10/11 \approx 0{,}909$. Đây là **cycle service level $\alpha$** (xác suất không hết hàng trong kỳ), **KHÔNG phải fill rate $\beta$**. Ngưỡng $\tau = 0{,}85$ trong đề tài là fill rate ($\beta$). Không được đồng nhất hai khái niệm này.
- **Việc cần làm:**
  1. **Thêm Tiểu mục 2.1.3 "Các thước đo mức phục vụ"** vào [`content/C2.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C2.tex):  
     Định nghĩa rõ 3 chỉ số:  
     - $\alpha$ (Cycle Service Level / Type-1): xác suất không thiếu hàng trong chu kỳ bổ sung;  
     - $\beta$ (Fill Rate / Type-2): tỷ lệ nhu cầu được đáp ứng trực tiếp từ tồn kho;  
     - $\gamma$ (Ready Rate): tỷ lệ thời gian tồn kho dương.  
     Nêu rõ: Critical ratio tương ứng với $\alpha$. Trong điều kiện biến động vừa phải $\beta$ thường cao hơn $\alpha$, nhưng với cầu gián đoạn và lead time ngẫu nhiên (như M5), $\beta$ có thể thấp hơn $\alpha$. Trích Silver et al. (2016) (`silver2016`) và Zipkin (2000) (`zipkin2000`).
  2. **Thêm thảo luận tại Mục 3.2.3 ([`content/C3.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C3.tex)):**  
     Phân tích rằng vì bộ tham số chi phí ngầm định $\alpha \approx 0{,}909$, ràng buộc $\beta \ge 0{,}85$ có thể **không có hiệu lực (non-binding)** trong thực nghiệm. Ghi rõ đây là giả thuyết cần kiểm chứng qua mô phỏng, và chèn `% TODO` về việc cân nhắc thêm cấu hình chi phí phụ (ví dụ $c_{th} = 3$).
  3. **KHÔNG tự đổi giá trị $\tau$ hay $c_{th}$ trong bảng.**

---

## 5. NHIỆM VỤ D — Bổ sung tại Mục 1.1.1 trong `content/C1.tex`

Tại Mục 1.1.1 trong [`content/C1.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C1.tex) (xung quanh dòng 35-39), tại đoạn viện dẫn ngưỡng phân loại Syntetos–Boylan–Croston ($p = 1{,}32$; $CV^2 = 0{,}49$):
- Bổ sung ghi chú rằng Kostenko & Hyndman (2006) (`kostenko2006`) đã chứng minh kết quả biên chính xác với ngưỡng $p = 4/3 \approx 1{,}333$ và đề xuất quy tắc phân loại chỉnh sửa.
- Trích dẫn cả Syntetos et al. (2005) (`syntetos2005categorization`) và Kostenko & Hyndman (2006) (`kostenko2006`).
- Giữ nguyên toàn bộ số liệu thống kê M5 hiện có ($30.490$ chuỗi; trung bình $1{,}13$; trung vị $0{,}45$; $73{,}4\%$; $68\%$; $95{,}2\%$).

---

## 6. NHIỆM VỤ E — Chèn ghi chú TODO thiết kế

Chèn chính xác 2 đoạn comment LaTeX sau vào [`content/C3.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C3.tex):

1. **Tại Mục 3.1.2 (Không gian hành động):**
```latex
% TODO [QUYẾT ĐỊNH THIẾT KẾ]: Mức đặt hàng tuyệt đối {0,10,20,30,40,50} không
% tương thích quy mô cầu M5 — 73,4% chuỗi có cầu trung bình dưới 1 đơn vị/ngày,
% nên mức đặt nhỏ nhất khác 0 tương đương hơn 20 ngày cầu. Cân nhắc chuyển sang
% mức đặt tương đối theo bội số cầu trung bình của từng cặp. Liên quan trực tiếp
% tới Câu hỏi nghiên cứu 2.
```

2. **Tại Mục 3.2.1 (Không gian quan sát / Môi trường):**
```latex
% TODO [KIỂM TRA MÃ NGUỒN]: Fill rate cửa sổ trượt 30 ngày phải nằm trong vector
% quan sát (hoặc hai bộ tích lũy tổng cầu / tổng thiếu hụt), nếu không môi trường
% mất tính Markov và critic không học được hàm giá trị nhất quán.
```

---

## 7. ĐẦU RA MONG ĐỢI VÀ XÁC NHẬN

1. Tệp [`backmatter/references.bib`](file:///d:/A.N%C4%82M%204/Report%20KLTN/backmatter/references.bib) được cập nhật các mục sai và 8 mục mới.
2. Tệp [`content/C2.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C2.tex) được viết lại các Mục 2.1.3, 2.3.5, 2.4.1, 2.4.2 và Bảng 2.1.
3. Tệp [`content/C3.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C3.tex) được điều chỉnh Mục 3.2.2, 3.2.3, 3.4 và chèn các ghi chú `% TODO`.
4. Tệp [`content/C5.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C5.tex) được điều chỉnh Mục 5.1 (bỏ trích dẫn Ng et al. sai).
5. Tệp [`content/C1.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C1.tex) được bổ sung trích dẫn Kostenko & Hyndman (2006).
6. Tệp `CHANGELOG_HIEUCHINH.md` được khởi tạo liệt kê đầy đủ các thay đổi.
7. Kết quả biên dịch qua [`compile.bat`](file:///d:/A.N%C4%82M%204/Report%20KLTN/compile.bat): **0 lỗi (errors)**, **0 warning citation/reference chưa phân giải**.

