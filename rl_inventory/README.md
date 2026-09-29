# PS-IPPO cho quản lý tồn kho đa kho — Hướng dẫn chạy

10 kho × 30 mặt hàng = 300 tác tử, dữ liệu M5 Walmart, quan sát 44 chiều.
Lý thuyết và phạm vi: **[pham_vi_khoa_luan.md](pham_vi_khoa_luan.md)** · Tiến độ:
**[../TIEN_DO.md](../TIEN_DO.md)** · Vấn đáp: **[van_dap.md](van_dap.md)**.

---

## 1. Chạy đợt thực nghiệm cuối trên Google Colab (khuyên dùng)

**Chạy thử nhanh trước** (~10–15 phút, T4): `Khoa_luan_Colab_test_nhanh.ipynb` → Run all.
Train 300 episode, đánh giá 5 lần trên test; kết quả lần mới nhất lưu ở
`MyDrive/KLTN_test/` (mỗi lần chạy ghi đè lần trước, không đụng `KLTN_final`).
Chỉ để kiểm tra quy trình, không dùng số liệu này cho báo cáo.

Notebook: `Khoa_luan_Colab.ipynb`. Code **không** chép lên Drive; Drive chỉ dùng
để đọc dữ liệu gốc và lưu kết quả.

| Nơi | Chứa gì |
|---|---|
| GitHub `MinhLysA/Khoa_luan` | mã nguồn + `config.yaml` (Colab clone mỗi phiên vào `/content/Khoa_luan`) |
| `MyDrive/KLTN_data/` | 3 file CSV gốc M5 (tự upload) |
| `MyDrive/KLTN_final/` | **chỉ kết quả**: `checkpoints/`, `results/`, `runs/`, `logs/` |

### Chuẩn bị (một lần)
1. Tải `sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv` từ
   [M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
   rồi upload vào `MyDrive/KLTN_data/`.
2. Code/config sửa ở máy phải **push lên GitHub trước**, vì Colab lấy code từ GitHub.
3. Nếu `MyDrive/KLTN_final` còn là bản cũ (chứa cả repo) → xóa hoặc đổi tên.
4. Colab: **Runtime → Change runtime type → T4 GPU**.

### Các bước trong notebook

| Bước | Việc | Kiểm tra |
|---|---|---|
| 1 | Mount Drive | — |
| 2 | Clone code, nối 4 thư mục kết quả sang Drive | in `include_price_signal = True` |
| 3 | Cài thư viện, kiểm tra GPU | in tên GPU |
| 4 | Tiền xử lý M5 từ `KLTN_data` | 4 dòng `TRUNG KHOP` |
| 5 | Kiểm tra dữ liệu, ba miền | shape `(1941, 10, 30)` |
| 6 | Chạy test | `44 passed` — nếu có `failed` thì dừng |
| 7 | Tune baseline trên val (~5–10 phút) | — |
| 8 | Train chính 3 seed × 4.000 ep (8.0 chạy thử 30 ep đo tốc độ) | `device=cuda`, ~1 giây/episode |
| 9 | Train 16 ablation × 1.000 ep; cuối bước chạy `status` | mọi dòng `XONG` |
| 10 | Đánh giá trên test (~30–40 phút) | — |
| 11 | Xem nhanh kết quả | — |

Bước 8–9 tổng khoảng 8–10 giờ trên T4. Phụ lục notebook có chế độ chạy nhiều phiên
song song (`campaign.py train auto`).

### Khi Colab ngắt / khởi động lại
Chạy lại **Bước 1–3** rồi chạy lại đúng cell đang dở. Mọi lần train tự train tiếp
từ checkpoint trên Drive (lưu khoảng mỗi 25 episode). Không cần chạy lại Bước 4–7.
Hết hạn mức GPU: có thể chuyển runtime sang CPU chạy tiếp, checkpoint dùng lẫn được.

### Lấy kết quả về máy
Tải thư mục `MyDrive/KLTN_final` từ Google Drive. **Mọi số liệu nằm trong một file:
`results/KET_QUA.txt`**, tự dựng lại mỗi khi train xong, đánh giá xong hoặc chạy
`campaign.py tonghop` (chạy tay: `python scripts/ket_qua.py`):
- Phần 1: RQ1 (3 seed so với từng baseline), RQ3 (từng seed so với `abl_ref`).
- Phần 2: từng mô hình — huấn luyện (mô hình được chọn, episode đạt 85%, đã hội tụ chưa,
  cơ cấu chi phí) và đánh giá trên test (bảng chi phí/fill, kiểm định theo cặp, kết luận).

`logs/phien_ban_code.txt` ghi commit code của từng phiên chạy.

---

## 2. Chạy trên máy local

```bash
pip install -r requirements.txt
python run.py check                  # kiểm tra môi trường đã sẵn sàng
python run.py test                   # 44 test
python run.py app                    # demo Streamlit 4 trang
python app/mcp_server.py --selftest  # thử MCP không cần LLM
```

Dữ liệu đã xử lý có sẵn trong `data/processed/`. Muốn tiền xử lý lại từ đầu: đặt 3
file CSV M5 vào `data/raw/` rồi `python run.py data`.

### Đợt thực nghiệm cuối bằng `campaign.py`

```bash
python scripts/campaign.py status             # tiến độ từng lần chạy
python scripts/campaign.py train main         # 3 seed × 4.000 episode (tự resume)
python scripts/campaign.py train ablations    # 16 lần × 1.000 episode
python scripts/campaign.py train <ten>        # một lần chạy, vd main_s42, abl_ref
python scripts/campaign.py eval               # toàn bộ pipeline đánh giá
python scripts/campaign.py tonghop            # tổng hợp đa hạt giống + RQ3
```

`eval` chạy lần lượt: tune baseline → eval 3 seed → iso-service → regime →
behavior → quy mô cầu → độ nhạy → độ bền → ablation → hold-out →
`results/multiseed_main.json` → `results/rq3_multiseed.json`.

### Lệnh lẻ (`run.py`)

| Lệnh | Việc |
|---|---|
| `data` | tiền xử lý M5 |
| `baseline` | tìm lưới tham số baseline |
| `train [--episodes N]` | huấn luyện IPPO |
| `eval` / `iso` | đánh giá / so sánh ở cùng mức phục vụ |
| `regime` / `behavior` | thực nghiệm B / hành vi policy |
| `summary` | xuất `TONG_HOP_SO_LIEU.txt` |
| `all [--quick]` | chạy trọn bộ (bản rút gọn ~5 phút) |

---

## 3. Cấu trúc thư mục

```
rl_inventory/
├── run.py                       điểm vào lệnh lẻ
├── config.yaml                  toàn bộ tham số và cờ
├── configs/                     config ablation (sinh bằng configs/make_configs.py)
├── Khoa_luan_Colab.ipynb        chạy đợt cuối trên Colab
├── env/inventory_env.py         môi trường Gymnasium (300 cặp vector hóa)
├── agents/                      PS-IPPO, rollout buffer (GAE theo cặp)
├── baselines/                   EOQ, (s,S), Newsvendor
├── scripts/
│   ├── campaign.py              đợt cuối: train (tự resume) + eval + status
│   ├── train.py / evaluate.py   huấn luyện / đánh giá (kiểm định theo cặp)
│   ├── tune_baselines.py        tune trên val theo ràng buộc 85%
│   ├── iso_service.py           so sánh ở cùng mức phục vụ ~90%
│   ├── regime_analysis.py       giai đoạn nhu cầu, sốc cầu, bootstrap
│   ├── policy_behavior.py       hành vi policy, permutation importance
│   ├── analyze_scale_groups.py  theo nhóm quy mô cầu
│   ├── sensitivity.py           độ nhạy đơn giá chi phí
│   ├── robustness.py            độ bền zero-shot
│   ├── plot_learning_curve.py   đường học
│   ├── generate_summary.py      TONG_HOP_SO_LIEU.txt
│   └── data_preprocessing.py    tiền xử lý M5
├── app/                         Streamlit (app.py + pages/), mcp_server.py
├── data/processed/              dữ liệu đã xử lý
├── checkpoints/ results/ runs/  đầu ra (trên Colab là liên kết sang Drive)
└── tests/test_env.py            44 test
```

**File kết quả cũ:** file trong `results/` không hậu tố, `TONG_HOP_SO_LIEU.txt`,
`results_pre_p0/`, `checkpoints_pre_p0/`, `best_model_seed42.pth` là từ các lần
chạy thử trước — **không dùng cho báo cáo**.

Báo cáo LaTeX: `../Report KLTN/`, biên dịch bằng `compile.bat` (XeLaTeX + Biber).
