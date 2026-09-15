# CHANGELOG HIỆU CHỈNH KHÓA LUẬN LA-TEX

Tài liệu ghi nhận toàn bộ các chỉnh sửa, hiệu chỉnh và bổ sung được thực hiện theo kết quả thẩm định học thuật.

---

## 1. Tệp [`PROMPT_ANTIGRAVITY.md`](file:///d:/A.N%C4%82M%204/Report%20KLTN/PROMPT_ANTIGRAVITY.md)
- **Nội dung:** Rà soát và cập nhật đường dẫn các tệp nguồn trong repository ([`backmatter/references.bib`](file:///d:/A.N%C4%82M%204/Report%20KLTN/backmatter/references.bib), [`content/C1.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C1.tex) đến [`content/C5.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C5.tex)), ánh xạ citation key chính xác và câu lệnh biên dịch tự động qua [`compile.bat`](file:///d:/A.N%C4%82M%204/Report%20KLTN/compile.bat).

---

## 2. NHIỆM VỤ A — Tệp [`backmatter/references.bib`](file:///d:/A.N%C4%82M%204/Report%20KLTN/backmatter/references.bib)

### A1. Sửa các mục sai hiện tại:
- **`wolpert2001`**: Sửa năm từ `2002` thành `2001`, bổ sung quyển `4`, số `2--3`, trang `265--279`, DOI `10.1142/S0219525901000188`.
- **`gijsbrechts2022`**: Bổ sung tiêu đề đầy đủ `Can Deep Reinforcement Learning Improve Inventory Management? Performance on Lost Sales, Dual-Sourcing, and Multi-Echelon Problems` và tác giả thứ 4 `Dennis J. Zhang`.
- **`oroojlooyjadid2022`**: Xác nhận định dạng tác giả thứ 2 `Nazari, MohammadReza`.
- **`dewitt2020ippo`**: Chuẩn hóa tên tác giả đầu thành `{Schr{\"o}der de Witt}, Christian` để BibLaTeX hiển thị và sắp xếp alphabet chính xác theo họ `Schröder de Witt`.
- **`yu2022mappo`**: Bổ sung thông tin track `Advances in Neural Information Processing Systems 35 (NeurIPS 2022), Datasets and Benchmarks Track` và số trang `24611--24624`.
- **`silver2016`**: Chuẩn hóa họ tên tác giả `Silver, Edward A.`, nhà xuất bản `CRC Press`.
- **`ng1999`**: Cập nhật tiêu đề đầy đủ `Policy invariance under reward transformations: Theory and application to reward shaping`.

### A2. Tách trích dẫn M5:
- **`makridakis2022background`**: Giữ nguyên làm mục bài báo khoa học trên International Journal of Forecasting.
- **`m5dataset`**: Chuẩn hóa mục dữ liệu cuộc thi Kaggle và chèn ghi chú comment `% TODO: xác nhận danh sách tác giả từ nút Cite trên Kaggle`.

### A3. Bổ sung 8 mục BibTeX mới:
1. `kostenko2006`: Kostenko & Hyndman (2006) - Phân loại dạng nhu cầu.
2. `yang2023mabim`: Yang et al. (2023) - Benchmark MABIM.
3. `ding2022cdppo`: Ding et al. (2022) - CD-PPO trong tồn kho với tài nguyên dùng chung.
4. `liu2025happo`: Liu et al. (2025) - HAPPO cho tồn kho đa cấp.
5. `kotecha2024decentralized`: Kotecha & del Rio Chanona (2024) - Đánh giá MARL phân tán.
6. `hubbs2020orgym`: Hubbs et al. (2020) - Môi trường OR-Gym.
7. `demoor2022reward`: De Moor, Gijsbrechts & Boute (2022) - Reward shaping cho tồn kho.
8. `devlin2011potential`: Devlin & Kudenko (2011) - Potential-based reward shaping trong MARL.

---

## 3. NHIỆM VỤ D — Tệp [`content/C1.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C1.tex)
- **Vị trí:** Mục 1.1.1 (`\subsection{Thực trạng quản lý tồn kho trong mạng lưới bán lẻ đa điểm}`), xung quanh dòng 35–39.
- **Nội dung:** Thêm câu ghi chú cho thấy Kostenko & Hyndman (2006) (`\cite{kostenko2006}`) đã chứng minh kết quả biên chính xác với ngưỡng $p = 4/3 \approx 1{,}333$ cho quy tắc phân loại Syntetos–Boylan–Croston.

---

## 4. NHIỆM VỤ B & C — Tệp [`content/C2.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C2.tex)

### C2. Thêm Tiểu mục 2.1.3:
- **Vị trí:** Ngay sau Mục 2.1.2 (Lượng đặt hàng kinh tế, (s,S), Newsvendor).
- **Nội dung:** Tạo tiểu mục `\subsection{Các thước đo mức phục vụ trong quản lý tồn kho}` (`\label{subsec:thuocDoMucPhucVu}`), phân định rõ 3 chỉ số $\alpha$ (Cycle Service Level / Type-1), $\beta$ (Fill Rate / Type-2), $\gamma$ (Ready Rate). Nêu rõ critical ratio của Newsvendor tương ứng với $\alpha$, còn $\tau = 0{,}85$ là fill rate $\beta$; giải thích sự khác biệt và khả năng $\beta < \alpha$ dưới nhu cầu gián đoạn. Trích `silver2016` và `zipkin2000`.

### C1. Sửa lỗi lý thuyết Reward Shaping (Mục 2.3.5):
- **Vị trí:** Mục 2.3.5 (`\subsection{Phần thưởng phân rã cục bộ và bài toán gán tín dụng}`).
- **Nội dung:** Viết lại đúng lý thuyết Potential-Based Reward Shaping (PBRS) của Ng et al. (1999) với điều kiện cần và đủ dạng $\gamma\Phi(s') - \Phi(s)$, trích dẫn Devlin & Kudenko (2011) (`devlin2011potential`) và De Moor et al. (2022) (`demoor2022reward`). Xóa bỏ tuyên bố phạt fill rate cửa sổ trượt bảo toàn chính sách tối ưu.

### B. Viết lại Mục 2.4 và Bảng 2.1:
- **Mục 2.4.1:** Bổ sung lược khảo 5 nghiên cứu MABIM (`yang2023mabim`), CD-PPO (`ding2022cdppo`), Liu et al. (2025) (`liu2025happo`), Kotecha & del Rio Chanona (2024) (`kotecha2024decentralized`), OR-Gym (`hubbs2020orgym`).
- **Mục 2.4.2:** Viết lại hoàn toàn, loại bỏ khẳng định tuyệt đối ("chưa có công trình nào", "đầu tiên"). Định vị đóng góp là **tái lập và mở rộng có kiểm soát** trên 3 trọng tâm: (1) Cấu hình 10 kho song song không multi-echelon, (2) Ablation study phân rã phần thưởng cục bộ ở 500 tác tử, (3) Hiệu chỉnh từ M5 Walmart với đối chứng cổ điển.
- **Bảng 2.1 (`tab:soSanhCongTrinh`):** Mở rộng thành 6 cột (`Nghiên cứu | Cấu trúc mạng | Số SKU | Param sharing | Dữ liệu thực | Thuật toán`) và thêm các dòng cho tất cả các nghiên cứu mới.

---

## 5. NHIỆM VỤ C, B & E — Tệp [`content/C3.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C3.tex)

### E. Chèn các ghi chú TODO:
- **Mục 3.1.1 (Không gian quan sát):** Chèn `% TODO [KIỂM TRA MÃ NGUỒN]` về việc đưa fill rate trượt 30 ngày vào observation vector để đảm bảo tính Markov.
- **Mục 3.1.2 (Không gian hành động):** Chèn `% TODO [QUYẾT ĐỊNH THIẾT KẾ]` về mức đặt tuyệt đối `{0,10,20,30,40,50}` không tương thích với quy mô cầu M5.

### C1. Điều chỉnh Mục 3.2.2:
- **Nội dung:** Xóa bỏ viện dẫn Ng et al. (1999). Trình bày phần phạt fill rate dưới dạng **hình phạt ràng buộc (constraint penalty) theo tinh thần Lagrangian**, cố ý thay đổi nghiệm tối ưu của MDP gốc để đáp ứng yêu cầu dịch vụ.

### C2. Thêm thảo luận Mục 3.2.3:
- **Nội dung:** Phân tích rằng bộ chi phí ngầm định $c_{th}=10, c_{lk}=1$ cho theoretical $\alpha \approx 90{,}9\%$, làm cho ràng buộc $\beta \ge 85\%$ có nguy cơ không có hiệu lực (non-binding). Chèn comment `% TODO: quyết định có bổ sung cấu hình chi phí phụ (ví dụ c_th = 3) hay không`.

### B2. Bổ sung giải thích Mục 3.4:
- **Nội dung:** Thêm đoạn giải thích lý do tự xây dựng môi trường `MultiWarehouseInventoryEnv` thay vì dùng trực tiếp OR-Gym hay MABIM.

---

## 6. NHIỆM VỤ C1 — Tệp [`content/C5.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C5.tex)
- **Vị trí:** Mục 5.1 (Tổng kết các kết quả đạt được, Mục tiêu 2).
- **Nội dung:** Xóa viện dẫn Ng et al. (1999), viết lại thành *"hàm phần thưởng phân rã cục bộ (dựa trên nguyên lý phần thưởng chênh lệch Wolpert & Tumer 2001) kèm thành phần phạt fill rate ràng buộc theo tinh thần Lagrangian"*.

---

## 8. ĐỊNH DẠNG BẢNG SỐ LIỆU KHẾP KÍN
- **Nội dung:** Chuyển đổi toàn bộ các bảng số liệu trong báo cáo từ dạng mở sang dạng **bảng khép kín (closed borders)** có đường kẻ biên hai bên `|`, đường kẻ phân cách các cột `|`, và kẻ vạch ngang `\hline` giữa tất cả các hàng.
- **Danh sách bảng đã điều chỉnh:**
  - Bảng 1.1 ([`content/C1.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C1.tex)): Phân loại dạng nhu cầu M5.
  - Bảng 1.2 ([`content/C1.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C1.tex)): Các nhóm chi phí bài toán.
  - Bảng 2.1 ([`content/C2.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C2.tex)): So sánh tổng quan các công trình liên quan.
  - Bảng 3.1 ([`content/C3.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C3.tex)): Tham số chi phí trong môi trường mô phỏng.
  - Bảng 3.2 ([`content/C3.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C3.tex)): Siêu tham số huấn luyện IPPO.
  - Bảng 4.1 ([`content/C4.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C4.tex)): Cấu hình môi trường mô phỏng.
  - Bảng 4.2 ([`content/C4.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/content/C4.tex)): Tổng chi phí vận hành đối chứng.
  - Bảng Phụ lục A ([`phuluc_nguon.tex`](file:///d:/A.N%C4%82M%204/Report%20KLTN/phuluc_nguon.tex)): Longtable đối chiếu luận điểm và nguồn trích dẫn.

