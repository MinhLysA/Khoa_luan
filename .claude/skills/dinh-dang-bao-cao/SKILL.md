---
name: dinh-dang-bao-cao
description: Định dạng và văn phong cho khóa luận tốt nghiệp viết bằng LaTeX trong thư mục "Report KLTN" (XeLaTeX, Times New Roman 13pt, khổ A4, mẫu Khoa CNTT, Trường Đại học Tôn Đức Thắng). Dùng khi viết mới, sửa câu chữ, chỉnh format, kiểm tra trước khi nộp các tệp .tex của khóa luận (chương, tóm tắt, bảng, hình, mã giả, phụ lục, tài liệu tham khảo, trang bìa, cam đoan).
---

# Định dạng khóa luận tốt nghiệp (LaTeX, mẫu TDTU)

Các quy tắc dưới đây rút ra từ báo cáo Dự án CNTT 2026 đã được giảng viên hướng dẫn góp ý nhiều vòng, chỉnh lại
cho khóa luận "Xây dựng mô hình học tăng cường tối ưu hóa chính sách đặt hàng lại trong quản lý tồn kho đa kho".
Mã LaTeX nằm ở `Report KLTN/` (`main.tex`, `preamble.tex`, `frontmatter/`, `content/C1–C5.tex`, `phuluc_*.tex`,
`backmatter/references.bib`). Khi quy tắc ở đây mâu thuẫn với quy định khóa luận của Khoa, quy định của Khoa thắng.

Tệp kèm theo:
- `templates/preamble.tex`: toàn bộ thiết lập trang, tiêu đề, mục lục, bảng, mã giả, tài liệu tham khảo.
- `templates/main.tex`: thứ tự các phần của tài liệu.
- `scripts/kiem_van_phong.py`: kiểm câu dài, in đậm, chấm phẩy, văn AI, chú thích dài, dòng liệt kê, tóm tắt có số.

## 1. Thông số trang

| Thành phần | Giá trị |
|---|---|
| Trình biên dịch | XeLaTeX + Biber (`fontspec`, `babel` vietnamese) |
| Phông, cỡ chữ | Times New Roman 13pt (`\changefontsize[15.6]{13}`) |
| Khổ giấy, lề | A4; trên 3,0 cm, dưới 3,0 cm, trái 3,5 cm (đóng gáy), phải 2,0 cm |
| Dãn dòng | 1,5 dòng (`\onehalfspacing`) |
| Đoạn văn | thụt đầu dòng 1,27 cm, cả đoạn đầu tiên (`indentfirst`); không cách đoạn (`\parskip` 0) |
| Số trang | giữa, phía trên, cỡ 10pt, không kẻ đường header/footer; cả trang đầu chương |
| Đánh số | phần mở đầu số La Mã (i, ii...), từ Chương 1 số Ả Rập; trang bìa và phụ bìa không số |
| Tràn lề | `microtype` + `\emergencystretch 3em`; `xurl` và `biburlnumpenalty`/`biburlucpenalty` = 100 để ngắt DOI dài |

## 2. Thứ tự các phần

1. Trang bìa, trang phụ bìa (phụ bìa thêm "Giảng viên hướng dẫn" và tên GV).
2. Lời cảm ơn.
3. Trang cam đoan theo mẫu mới của Trường.
4. Phần xác nhận và đánh giá của giảng viên (GV hướng dẫn, GV chấm bài; mỗi phần 5 dòng kẻ dài + 1 dòng 70%).
5. Tóm tắt tiếng Việt, Abstract tiếng Anh (mỗi bản đúng một trang).
6. Mục lục, Danh mục hình vẽ, Danh mục bảng biểu, Danh mục các chữ viết tắt.
7. Chương 1 đến Chương 5.
8. Tài liệu tham khảo.
9. Phụ lục (Phụ lục A, B, ...), nếu có.

Thứ tự "tài liệu tham khảo rồi phụ lục" là thứ tự thông dụng; đối chiếu lại với hướng dẫn khóa luận của Khoa.
Trang THESIS PLEDGE (`frontmatter/fledge.tex`, trang xác nhận của hội đồng) chỉ bật khi Khoa yêu cầu.

Không có danh mục thuật toán.

### Phụ lục
- Mỗi phụ lục bắt đầu trang mới, tiêu đề `\chapter*{PHỤ LỤC A. TÊN PHỤ LỤC}` kèm
  `\addcontentsline{toc}{chapter}{PHỤ LỤC A. TÊN PHỤ LỤC}` và `\label{app:...}`.
- Phụ lục chỉ chứa nội dung bổ trợ mà thân bài có dẫn tới (bảng chi tiết, cấu hình, mô tả giao diện). Không đưa
  lệnh chạy, mã nguồn dài hay nhật ký chạy vào phụ lục nếu không được yêu cầu.
- Bảng dài trong phụ lục dùng `longtable` với `\endfirsthead`/`\endhead` để lặp tiêu đề cột khi sang trang.

### Trang bìa
- Căn giữa: "TỔNG LIÊN ĐOÀN LAO ĐỘNG VIỆT NAM" (14pt thường), "TRƯỜNG ĐẠI HỌC TÔN ĐỨC THẮNG", "KHOA CÔNG NGHỆ
  THÔNG TIN" (14pt đậm); logo rộng 4,13 cm cao 2,29 cm.
- Tên sinh viên kèm MSSV (13pt đậm, "HỌ TÊN - MSSV"), tên đề tài (18pt đậm, viết hoa, ngắt dòng cân đối, không
  tách một chữ ra dòng riêng), "KHÓA LUẬN TỐT NGHIỆP" và tên ngành (16pt đậm), cuối trang "THÀNH PHỐ HỒ CHÍ MINH,
  NĂM 20xx" (13pt đậm, đẩy xuống bằng `\vfill`).
- Tên đề tài trên bìa, phụ bìa, Tóm tắt phải giống nhau từng chữ; Abstract dùng bản dịch tiếng Anh thống nhất.

### Trang cam đoan
- Tiêu đề 16pt đậm, căn giữa: "CÔNG TRÌNH ĐƯỢC HOÀN THÀNH / TẠI TRƯỜNG ĐẠI HỌC TÔN ĐỨC THẮNG".
- Mở đầu "Chúng tôi xin cam đoan đây là công trình nghiên cứu của riêng chúng tôi và được sự hướng dẫn khoa học
  của TS. ..." (làm một mình thì đổi "chúng tôi" thành "tôi"). Gọi tài liệu là "Khóa luận tốt nghiệp" ở mọi chỗ.
- Câu "Nếu phát hiện có bất kỳ sự gian lận nào ... chịu trách nhiệm về nội dung ... của mình." là chỗ in đậm duy
  nhất trong thân trang.
- Khối ký tên căn phải (minipage 0,62\textwidth): ngày tháng in nghiêng, "Tác giả", "(Ký tên và ghi rõ họ tên)",
  khoảng trống 2,5 cm, tên in đậm.

### Tiêu đề các trang phần mở đầu
LỜI CẢM ƠN, TÓM TẮT, ABSTRACT, MỤC LỤC, DANH MỤC...: 16pt đậm, căn giữa, cách dưới 0,5 cm. Tóm tắt và Abstract có
tên đề tài 15pt đậm phía trên tiêu đề. Các danh mục được thêm vào mục lục bằng `\phantomsection` +
`\addcontentsline{toc}{chapter}{...}`.

## 3. Tiêu đề trong thân bài

| Cấp | Lệnh | Định dạng | Thụt lề | Cách trên / dưới |
|---|---|---|---|---|
| Chương | `\chapter` | 16pt đậm, viết hoa, "CHƯƠNG 1." + tên | 0 | 0 / 16pt |
| Mục | `\section` | 14pt đậm, "1.1" | 0 | 14pt / 6pt |
| Tiểu mục | `\subsection` | 13pt đậm nghiêng, "1.1.1" | 0,635 cm | 10pt / 4pt |
| Tiểu mục con | `\subsubsection` | 13pt nghiêng, "1.1.1.1" | 1,27 cm | 8pt / 3pt |

Mục lục: dòng chương in đậm có chữ "CHƯƠNG ", dòng mục chữ thường, dòng tiểu mục in nghiêng; dấu chấm dẫn sát
(`\cftdotsep` 0,5). Danh mục hình/bảng ghi "Hình 4.1: ...", "Bảng 4.1: ...", không có khoảng trắng giữa các chương.

Mọi chương, mục, bảng, hình, công thức, thuật toán đều có `\label` với tiền tố: `chap:`, `sec:`, `subsec:`, `tab:`,
`fig:`, `eq:`, `alg:`. Tham chiếu bằng `Chương~\ref{}`, `mục~\ref{}`, `Bảng~\ref{}`, `Hình~\ref{}`.

## 4. Bảng

- Kẻ lưới đầy đủ: cột ngăn bằng `|`, mọi hàng có `\hline`. `\extrarowheight` 2pt để chữ không sát đường kẻ.
- Chữ trong bảng `\small`, không nhỏ hơn 10pt. Không dùng `adjustbox`/`resizebox` co bảng làm chữ nhỏ đi; bảng
  rộng thì rút gọn tiêu đề cột, ngắt dòng trong ô (`\tbl`, `\tbc`, `\tbr` ở preamble) hoặc dùng `tabularx`.
- Chú thích đặt trên bảng, chú thích hình đặt dưới hình. Nhãn đậm, phân cách dấu hai chấm, chữ `small`.
- Chú thích là cụm danh từ ngắn, khoảng 20 đến 50 ký tự, không có dấu chấm cuối (vd. "Lọc $k$-core trên mẫu phát
  triển"). Giải thích ký hiệu, điều kiện đo đặt ở thân bài hoặc ghi chú dưới bảng.
- Bảng dài nhiều trang dùng `longtable`/`xltabular`. Đặt `[H]` để bảng nằm đúng chỗ được nhắc tới.
- Số thập phân dùng dấu phẩy (`0{,}0413` trong môi trường toán, `0,0413` trong chữ), hàng nghìn dùng dấu chấm
  (`31.788.324`), số âm trong bảng dùng `$-$`.
- Chữ đậm trong bảng chỉ dùng để đánh dấu giá trị tốt nhất.
- Bảng số liệu thực nghiệm nên sinh tự động từ script và chèn bằng `\input`; không gõ tay số kết quả.

## 5. Hình

- `figure[H]`, `\centering`, độ rộng tối đa `\textwidth`. Sơ đồ vẽ bằng TikZ hoặc draw.io xuất PDF/PNG.
- Ảnh lưu trong `media/` (`\graphicspath{{media/}}`).
- Sơ đồ có chữ đủ đọc khi in; nếu phải co sơ đồ TikZ thì giảm font node (`\small`, `\footnotesize`) thay vì co
  cả hình xuống quá nhỏ.
- Thân bài nhắc tới hình trước khi hình xuất hiện.

## 6. Công thức

- `equation` (một dòng) hoặc `align` (nhiều dòng), có `\label{eq:...}`, đánh số theo chương.
- Khoảng cách trên/dưới công thức 6pt (ngắn 3pt), đặt trong `\AtBeginDocument`.
- Câu dẫn vào công thức kết thúc bằng dấu hai chấm. Ký hiệu được giải thích ngay sau công thức ở lần đầu dùng.
- Biến một chữ in nghiêng trong `$...$`; tên hàm nhiều chữ dùng `\text{}` hoặc `\operatorname{}`.

## 7. Mã giả (thuật toán)

- Gói `algorithm` (đánh số theo chương: "Thuật toán 3.1") + `algpseudocode` với tuỳ chọn `noend`.
- Từ khóa tiếng Việt, chữ thường, không in đậm: Đầu vào, Đầu ra, với, với mọi, làm, trong khi, lặp, đến khi, nếu,
  thì, ngược lại, trả về, hàm, thủ tục. Chú thích dòng: `\Comment{}` in nghiêng, có dấu ▷.
- Chú thích thuật toán là cụm danh từ ngắn. Không có danh mục thuật toán.

## 8. Liệt kê

- `enumitem`: thụt nhãn 1,27 cm, cách nhãn 0,4 cm, cách dòng 2pt; nhãn cấp 1 là gạch ngang "--".
- Mỗi `\item` viết hoa chữ đầu, kết thúc bằng dấu chấm. Nhãn mục không in đậm.
- Danh sách đánh số dùng cho câu hỏi nghiên cứu (`[label=Câu hỏi \arabic*:]`) và mục tiêu ((1)–(5)).
- Không lạm dụng liệt kê; ý có quan hệ nhân quả thì viết thành đoạn.

## 9. Tài liệu tham khảo và trích dẫn

- `biblatex` style `ieee`, backend `biber`, `defernumbers=true`; trích dẫn dạng số [1], [2]. Biên dịch đủ
  xelatex → biber → xelatex → xelatex (lần ba để đánh số liên tục giữa các nhóm).
- Bắt buộc có `\DeclareLanguageMapping{vietnamese}{english}` và
  `\DefineBibliographyStrings{vietnamese}{andothers={và cộng sự}, and={và}}`; thiếu thì danh mục in từ khoá thô.
- Danh mục chia sáu nhóm chủ đề, khai báo bằng `\DeclareBibliographyCategory` trong `Report KLTN/preamble.tex`:
  Lý thuyết tồn kho (`tonkho`); Học tăng cường (`rl`); Học tăng cường đa tác tử (`marl`); Học tăng cường cho
  quản lý tồn kho (`rltonkho`); Dự báo nhu cầu và bộ dữ liệu M5 (`dubao`); Môi trường mô phỏng và đánh giá
  (`danhgia`). `main.tex` in từng nhóm bằng `\printbibliography[category=..., heading=subbib, title={...}]`.
- Thêm nguồn mới vào `references.bib` thì phải thêm khóa vào đúng một nhóm, nếu không nguồn đó không được in.
  Sau khi biên dịch, kiểm số nguồn in ra bằng số nguồn được trích. Cần `defernumbers=true` để số thứ tự liên
  tục giữa các nhóm. Mọi nguồn hiện là tiếng Anh nên không chia theo ngôn ngữ.
- Không dùng `\ref` tới nhãn đặt sau `\chapter*` (phụ lục): LaTeX sẽ in số của mục trước đó. Viết thẳng
  "Phụ lục A", "Phụ lục B".
- Chỉ trích dẫn khi định nghĩa hoặc dùng lại phương pháp, mô hình, dữ liệu, số liệu của một công trình cụ thể; câu
  chứa trích dẫn phải cho người đọc thấy đề tài dùng gì từ nguồn đó. Chương 1 không trích dẫn (để ở chỗ định nghĩa).
  Mỗi nguồn trích một lần ở chỗ liên quan nhất. Không trích cho nhận định chung, không dồn nhiều nguồn cho một câu,
  không trích trong tóm tắt.
- Không dùng preprint arXiv khi có bản đã xuất bản. Không trích mô hình, công trình mà đề tài không dùng.
- Danh mục chỉ chứa nguồn bên ngoài: không đưa kho mã nguồn cá nhân, đường dẫn GitHub, tài liệu nội bộ của nhóm.
- Bài báo là tham khảo về cách làm. Không đặt số liệu của đề tài cạnh số của bài báo để so sánh khi dữ liệu và
  giao thức khác nhau, trừ bảng đối chiếu mà giảng viên yêu cầu (khi đó ghi rõ các số không so được trực tiếp).

## 10. Văn phong

### Câu và đoạn
- Câu ngắn, mỗi câu một ý. Câu trên 45 từ thì tách.
- Không nối mệnh đề bằng dấu chấm phẩy trong văn xuôi. Hạn chế gạch ngang " -- " chèn ý và ngoặc lồng nhau.
- Không in đậm trong thân bài, nhãn liệt kê, chú thích, mã giả. Nhấn mạnh bằng cách đặt ý ở đầu câu.
- Mỗi ý nói một lần ở chỗ phù hợp nhất; chỗ khác dẫn "xem mục X.Y".
- Không mở đoạn bằng "Bảng X liệt kê/trình bày...". Mở bằng nhận xét hoặc mục đích, rồi dẫn bảng giữa câu.
- Chuẩn dẫn chiếu (người dùng chốt 03/11/2026, vì bản in phải tự đọc hiểu được):
  + Bảng, hình: mọi bảng/hình phải có một câu dẫn ngay trước nó, nói rõ bảng/hình chứa gì: "Chi phí và tỷ lệ đáp ứng
    của từng chính sách khi đạt mức phục vụ của IPPO được trình bày ở Bảng 4.9." Không dẫn trong ngoặc "(Bảng 4.9)".
  + Mục: không trỏ số mục trong văn xuôi ("ở mục 3.1.3", "(mục 4.5.1)"). Nhắc lại nội dung ngắn gọn, hoặc gọi tên
    phần ("như phần kiểm tra chính sách đã học cho thấy"), hoặc dẫn tới chương ("mô tả ở Chương 3"). Chỉ cột tra cứu
    trong bảng tổng hợp mới được ghi số mục.
  + Công thức: ghi số trong ngoặc tròn như chuẩn toán học, "theo công thức (3.4)".
- Tránh dấu hiệu văn AI: ", tức", "Đáng chú ý", "Như vậy", "vốn", cấu trúc "không phải ... mà ...", câu tự hạ thấp
  đóng góp.
- Chủ ngữ: "nhóm" cho việc đã làm ("Nhóm chọn ..."), "đề tài" cho kết quả và đóng góp ("Đề tài cho thấy ...").

### Thuật ngữ
- Thuật ngữ tiếng Việt nhất quán trong toàn bài, theo đúng chữ trong tên đề tài. Thuật ngữ tiếng Anh ghi trong
  ngoặc, in nghiêng, chỉ ở lần đầu: "lọc cộng tác (\textit{collaborative filtering}, CF)".
- Không trộn tiếng Anh và tiếng Việt cho cùng khái niệm (User/Item lẫn người dùng/sản phẩm).
- Dùng "huấn luyện", "tập/giai đoạn đánh giá", "kiểm thử", "tinh chỉnh"; không dùng "train/test/validation/tune"
  trong văn bản.
- Học tăng cường: ghi tiếng Anh ở lần đầu rồi dùng tiếng Việt nhất quán, vd. học tăng cường (\textit{reinforcement
  learning}, RL), tác tử (\textit{agent}), môi trường, trạng thái, hành động, phần thưởng, chính sách
  (\textit{policy}), tập (\textit{episode}). Chọn một từ cho mỗi khái niệm và giữ nguyên trong cả khóa luận.
- Dấu thanh kiểu mới theo tên đề tài ("TỐI ƯU HÓA", "KHÓA LUẬN"): hóa, họa, khóa, hòa, tùy, lũy, thỏa, khỏe,
  thúy. Không dùng lẫn kiểu cũ (hoá, khoá, hoà, tuỳ...).
- Mọi chữ viết tắt có trong Danh mục các chữ viết tắt: "Viết tắt & Tên đầy đủ tiếng Anh (Nghĩa tiếng Việt)", sắp
  theo bảng chữ cái.
- Ngày viết dd/mm/yyyy.
- Gạch: gạch ngang "–" (`--`) cho nhãn đầu dòng, khoảng số (1--3, 2011--2016), quan hệ hai đối tượng (kho--mặt hàng,
  Actor--Critic) và dấu ngắt có cách hai bên (HỌ TÊN -- MSSV ở bìa). Gạch nối "-" chỉ trong từ ghép, tên riêng
  (PS-IPPO, NeuMF-F, Top-$K$). Ô trống trong bảng dùng "---". Không gõ ký tự Unicode – — trực tiếp trong mã nguồn.

### Những gì không đưa vào báo cáo
- Tình trạng mã nguồn: MD5, mã băm, commit, nhật ký chạy, tên tệp nội bộ, bộ kiểm thử tự động, container.
- Chạy mấy lần, chạy lại, chạy trên máy nào, có GPU hay không, lệch giữa các máy. Chỉ nêu cách làm.
- Lịch sử sửa kế hoạch, "bản đầu/bản cuối", "lệch kế hoạch". Lịch sử phát triển chỉ gói trong một đoạn bài học.
- Ký hiệu nội bộ của nhóm (v1/v2, p0, mẫu A/B, tên thư mục kết quả...). Đặt tên có nghĩa, vd. "mô hình đề xuất",
  "chính sách đối chứng", "kịch bản nhu cầu".
- Câu rào đón, tuyên bố nguyên tắc (vd. "các công trình chỉ là cơ sở phương pháp, số liệu không dùng làm chuẩn so
  sánh"). Cứ làm đúng nguyên tắc, không cần nói ra.
- Nội dung ngoài trọng tâm tên đề tài. Nếu buộc phải nhắc thì chỉ ở một chỗ thiết kế và một câu hạn chế, không rải ở
  chương 1, phạm vi, mục tiêu.
- Các chi tiết trên để ở README hoặc tài liệu vấn đáp, dùng khi hội đồng hỏi.

## 11. Mạch nội dung theo chương

- Tóm tắt (Việt và Anh): không số liệu, không trích dẫn, một trang. Ba đoạn: bài toán và mô hình; cách đánh giá;
  một câu kết luận chung về đóng góp (nêu điểm mạnh trước, không liệt kê từng phép so sánh).
- Chương 1: bối cảnh, câu hỏi nghiên cứu đánh số (Câu hỏi 1–3), mục tiêu đánh số (1)–(5), phạm vi, phương pháp.
- Không viết đoạn giới thiệu chương ("Chương này trình bày...") hay câu "Mục này trình bày..."; vào thẳng mục đầu.
  Không có đề mục "Tóm tắt chương" (bài báo không có, báo cáo dự án đã duyệt cũng không có). Chương 2 trở đi kết
  bằng một đoạn 2–3 câu không đề mục: kết quả chính của chương rồi nối sang chương sau; Chương 1 không cần vì đã có
  "Bố cục". Đặt `\FloatBarrier` trước đoạn kết để bảng, hình không trôi qua.
- Chương kết quả: mỗi mục mở bằng câu kết luận rồi dẫn bảng. Tối đa 1–2 số mỗi đoạn; không chép lại chuỗi số đã có
  trong bảng, chỉ diễn giải. Có mục "Thảo luận và trả lời câu hỏi nghiên cứu" trả lời lần lượt từng câu.
- Chương kết luận viết thành đoạn liền mạch: kết quả đạt được (bảng đối chiếu với mục tiêu), kết luận trả lời câu
  hỏi nghiên cứu, các đóng góp ("Thứ nhất ... Thứ tư"), ứng dụng, hạn chế xếp theo mức ảnh hưởng tới kết luận (mỗi
  hạn chế nói rõ ảnh hưởng), hướng phát triển theo đúng thứ tự các hạn chế. Viết theo hướng đề cao đóng góp, nhưng
  vẫn giữ đủ kết quả bất lợi và phạm vi kết luận.

## 12. Quy trình khi sửa

1. Đọc tệp hiện tại trước khi sửa; chỉ đổi đúng chỗ được yêu cầu.
2. Sửa nội dung có dấu `\` (LaTeX) bằng công cụ Edit hoặc script Python ghi ra tệp, không qua heredoc của shell
   (heredoc có thể rút `\\` thành `\`, sinh ký tự điều khiển lạc).
3. Chạy `python .claude/skills/dinh-dang-bao-cao/scripts/kiem_van_phong.py "Report KLTN/content" "Report KLTN/frontmatter"`
   và xử lý cảnh báo. Kiểm dấu thanh kiểu cũ còn sót: `grep -n "hoá\|khoá\|hoà\|tuỳ\|thoả" "Report KLTN"/content/*.tex`.
4. Biên dịch bằng `Report KLTN/compile.bat` hoặc đủ bốn lượt (xelatex, biber, xelatex, xelatex). Đọc `main.log`:
   không còn `Overfull \hbox`, `undefined references`, `Citation ... undefined`, không có "??" trong PDF.
5. Mở PDF kiểm: số trang La Mã/Ả Rập đúng chỗ, mục lục, danh mục hình/bảng, không có trang chỉ chứa khoảng trắng
   lớn do float, bảng không tràn lề, tài liệu tham khảo in đủ các nhóm, phụ lục có trong mục lục.
6. Không tự `git commit/push`; liệt kê tệp đã đổi và để người dùng tự commit.
