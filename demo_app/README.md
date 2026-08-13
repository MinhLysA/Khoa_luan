# Streamlit Web Demo — Quản lý Tồn kho Đa Kho với Double DQN

Ứng dụng Web Demo tương tác bằng **Streamlit** để trực quan hóa, chạy mô phỏng real-time và so sánh tác tử **Double DQN** với các chiến lược tồn kho truyền thống (**EOQ**, **(s,S)**, **Newsvendor**).

---

## 🛠️ Cấu trúc thư mục

```
demo_app/
├── backend/
│   ├── model_loader.py      # Tải checkpoint weights (.pth) & nạp tác tử DoubleDQNAgent
│   ├── env_wrapper.py       # Engine mô phỏng vòng lặp so sánh các chiến lược
│   └── inference.py         # Hàm suy luận & ánh xạ hành động rời rạc sang số đơn vị
├── frontend/
│   └── app.py               # Ứng dụng Giao diện Streamlit (Plotly charts, KPI cards, Filters)
├── requirements.txt         # Các thư viện phụ thuộc cho Web App
└── README.md
```

---

## 🚀 Hướng dẫn Chạy Ứng dụng

### 1. Thư mục và File Checkpoint cần có

Đảm bảo bạn đã có ít nhất một tệp checkpoint `.pth` trong thư mục `rl_inventory/checkpoints/` (ví dụ: `best_model.pth` hoặc `final_model.pth`).

### 2. Cài đặt Phụ thuộc

```bash
# Từ thư mục gốc dự án:
pip install -r demo_app/requirements.txt
```

### 3. Lệnh Chạy Web App

Bạn có thể chạy lệnh Streamlit từ thư mục gốc dự án:

```bash
streamlit run demo_app/frontend/app.py
```

Hoặc di chuyển vào thư mục `demo_app/` và chạy:

```bash
cd demo_app
streamlit run frontend/app.py
```

Ứng dụng sẽ tự động mở giao diện tại địa chỉ `http://localhost:8501`.

---

## ✨ Các Chức năng Chính

1. **Chọn Checkpoint & Nạp Mô hình**: Tự động phát hiện các checkpoint `.pth` sẵn có hoặc cho phép tải file weights tùy chỉnh.
2. **Bộ lọc Kho & SKU**: Chọn từng cặp Nhà kho / Mặt hàng cụ thể để soi chi tiết hoặc xem tổng hợp toàn bộ hệ thống.
3. **Mô phỏng Real-time**: Nút **Run Simulation** với thanh tiến trình real-time chạy thử nghiệm trên cùng chuỗi nhu cầu cho 4 chiến lược.
4. **Biểu đồ Tương tác (Plotly)**:
   - Diễn biến tồn kho theo thời gian
   - Lượng đặt hàng từng ngày so với nhu cầu thực tế
   - Phân rã thành phần chi phí (Holding Cost vs Ordering Cost vs Stockout Cost)
5. **Inspector theo Ngày**: Slider chọn ngày cụ thể để xem ma trận Tồn kho và Ma trận Đặt hàng tức thời.
