# Khóa luận tốt nghiệp: Xây dựng mô hình học tăng cường tối ưu hóa chính sách đặt hàng lại trong quản lý tồn kho đa kho

Đề tài mô hình hóa mạng lưới 10 kho và 30 mặt hàng thành 300 tác tử, huấn luyện bằng IPPO với chia sẻ tham số trên dữ liệu bán lẻ M5 của Walmart. Mô hình được so sánh với các chính sách cổ điển và chính sách dự báo rồi tối ưu ở cùng mức phục vụ.

## Cấu trúc thư mục

```
Khoa_luan/
├── Report KLTN/       Báo cáo LaTeX, biên dịch bằng compile.bat
├── rl_inventory/      Mã nguồn: môi trường, tác tử, chính sách đối chứng, thí nghiệm, ứng dụng
└── docs/              Tài liệu của nhóm: tiến độ, tổng hợp thay đổi, vấn đáp, bài báo tham khảo
```

## Chạy

Hướng dẫn đầy đủ ở [rl_inventory/README.md](rl_inventory/README.md).

```powershell
cd rl_inventory
pip install -r requirements.txt
python run.py check          # kiểm tra GPU, thư viện, dữ liệu
python run.py full --quick   # thử cả dây chuyền trên bản sao
python run.py full           # huấn luyện và đánh giá toàn bộ
```

Mọi số liệu được ghi vào `rl_inventory/results/KET_QUA.txt`.
