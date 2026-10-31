# Dự báo đa bước giá heo hơi Việt Nam bằng Machine Learning và Deep Learning

Đồ án: *Dự báo đa bước giá heo hơi Việt Nam bằng Machine Learning và Deep Learning: vai trò của độ trễ giá thức ăn, lịch Tết và điều kiện thời tiết.*

- **Dữ liệu:** bộ giá heo hơi Việt Nam theo ngày do nhóm tự thu thập, 34 tỉnh gộp thành 3 miền (Bắc, Trung, Nam), 486 ngày từ 02/07/2025 đến 30/10/2026, kèm lịch âm và Tết, thời tiết, tỷ giá, giá thức ăn, nhập khẩu, dịch tả lợn châu Phi. Mô tả đầy đủ: `data/raw/README.md`, `data_dictionary.md`, `quality_report.md`.
- **Bài tham chiếu phương pháp:** Liu et al., 2025, PLOS ONE, DOI 10.1371/journal.pone.0324646. Nhóm tự viết code mới, chỉ áp dụng mô hình và thuật toán của bài.
- **Mô hình:** Naive, ARIMA/SARIMAX, XGBoost, LSTM, BiLSTM, STL-PCA-BiLSTM. **Horizon:** 7, 14, 30 ngày (dự báo trực tiếp, mỗi horizon một mô hình).
- Kế hoạch, phân công: drive dự án của nhóm https://drive.google.com/drive/folders/1T0f25lQLUYlO4BnWc8AyF4MyUeaFOt0F.

## 1. Cài đặt và chạy

Yêu cầu Python 3.10 trở lên. Chạy mọi lệnh **từ thư mục gốc của repo**.

```
python -m venv .venv
.venv\Scripts\activate          # Windows;  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python src/check_data.py        # kiểm tra bộ dữ liệu -> data/checks/data_check.md
python src/split.py             # chia Train/Val/Test -> data/split/split_config.json
```

Bộ dữ liệu đã crawl nằm nguyên vẹn trong `data/raw/` (giữ đúng cấu trúc thư mục gốc `parquet/`, `csv/` để `CHECKSUMS.sha256` kiểm được). **Không sửa** các file trong thư mục này.

## 2. Cấu trúc thư mục

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                       # bộ crawl nguyên vẹn, KHÔNG sửa
│   │   ├── parquet/               # 5 bảng .parquet
│   │   ├── csv/                   # cùng nội dung dạng CSV + feature_groups.csv
│   │   ├── README.md, data_dictionary.md, quality_report.md, CHECKSUMS.sha256
│   ├── checks/data_check.md       # kết quả check_data.py (P1)
│   ├── split/split_config.json    # mốc chia Train/Val/Test (P1)
│   └── features/                  # data_v1 (P2), data_v2 nếu có lỗi
├── src/
│   ├── check_data.py, split.py    # P1
│   ├── features.py                # P2
│   ├── evaluate.py, walkforward.py, naive_arima.py   # P3
│   ├── xgb_model.py               # P4
│   ├── lstm_model.py              # P5
│   └── stl_pca.py, dm_test.py     # P6
├── notebooks/                     # EDA, thử nghiệm: pN_noidung.ipynb
├── predictions/                   # file dự báo CSV của mọi mô hình
└── results/
    ├── tables/                    # bảng MAE, RMSE, MAPE, ablation, DM...
    └── figures/                   # hình dùng trong báo cáo (PNG)
```

## 3. Dữ liệu dùng để huấn luyện

- Bảng chính: `supervised_region_daily` (3 miền × 486 ngày, đã có sẵn feature và nhãn `y_h7`, `y_h14`, `y_h30`). Dự báo cấp **miền** (Bắc, Trung, Nam); kết quả báo cáo theo từng miền và trung bình.
- `price_ffill`/`price_t` chỉ dùng làm **đầu vào**. Nhãn là `price` quan sát thật tại ngày t + h.
- Khi đánh giá: bỏ dòng có nhãn NaN và dòng `is_sunday_copy == 1`.
- **Nhóm biến cho ablation** (`feature_groups.csv`): E0 giá lịch sử, E1 lịch và Tết, E2 thức ăn, E3 cung–cầu, E4 dịch bệnh, E5 thời tiết và vĩ mô. Ablation cộng dồn thành 6 cấu hình `c1` đến `c6` theo thứ tự đề xuất: E0 → E2 → E1 → E3 → E4 → E5.
- Không dùng `retail_idx_mean` làm feature: nguồn ngừng đăng từ 03/08/2026 nên thiếu 68% trong tập Test.

## 4. Quy ước đặt tên

- **Thư mục, file code, tên cột:** chữ thường, nối bằng gạch dưới (`snake_case`), không dấu, không khoảng trắng.
- **Tên cột dữ liệu:** dùng đúng tên trong bộ crawl (ví dụ `price`, `lag_7`, `wb_maize_usd_t_lag30`). Không đổi tên cột.
- **Tên mô hình** (cột `model`): `naive`, `arima`, `sarimax`, `xgboost`, `lstm`, `bilstm`, `stl_pca_bilstm`.
- **Tên cấu hình** (cột `config`): `c1` đến `c6`. Mặc định khi không chạy ablation: `c6`.
- **File dự báo:** `predictions/{model}_{config}.csv`. Ví dụ `predictions/xgboost_c3.csv`.
- **Phiên bản dữ liệu:** `data_v1`, nếu có lỗi thì `data_v2`. Không ghi đè, không ai tự tạo bản riêng.
- **Hình và bảng:** `results/figures/{muc}_{noidung}.png`, `results/tables/{muc}_{noidung}.csv`, với `muc` là mục trong báo cáo (ví dụ `4_2_xgboost_mae_horizon.csv`).
- **Notebook:** `p{số}_{noidung}.ipynb`.

## 5. Mốc chia dữ liệu và walk-forward

| Tập | Từ | Đến | Số ngày |
|---|---|---|---|
| Train | 02/07/2025 | 31/03/2026 | 273 |
| Val | 01/04/2026 | 30/06/2026 | 91 |
| Test | 01/07/2026 | 30/10/2026 | 122 |

- Cấu hình ở `data/split/split_config.json`. Trong code dùng `from split import assign_split, is_valid_origin`, không chép tay ngày.
- **Loại điểm gốc sát cuối tập:** nhãn `y_h` nhìn tới t + h ngày, nên mỗi tập chỉ dùng điểm gốc t có t + h ≤ ngày cuối của tập đó (hàm `is_valid_origin`).
- **Retrain walk-forward:** đầu mỗi tháng trong Test (01/07, 01/08, 01/09, 01/10/2026), 4 lần, cửa sổ mở rộng dần.
- Chọn tham số trên Val; sau đó huấn luyện lại trên Train + Val rồi chấm Test.

## 6. Định dạng file dự báo

Mỗi file là CSV có đúng các cột (P3 chốt trong `evaluate.py`, nếu đổi thì sửa tại đây):

`model, config, region, horizon, origin_date, target_date, y_true, y_pred`

- `horizon` tính bằng ngày (7, 14, 30). `origin_date`: ngày dùng để dự báo. `target_date` = `origin_date` + `horizon`.
- Mô hình học sâu chạy nhiều seed: file chính ghi trung bình các seed; file từng seed để trong `predictions/seeds/`.

## 7. Quy tắc chống rò rỉ dữ liệu (bắt buộc)

1. Tại ngày dự báo t, chỉ dùng dữ liệu đến hết ngày t. Biến ngoại sinh ghép theo thời điểm đã công bố (cột `*_age_days` luôn ≥ 0).
2. Chia **theo thứ tự thời gian**, không xáo trộn (mục 5), và loại điểm gốc sát cuối tập.
3. Scaler, STL, PCA chỉ **fit trên Train**; Val và Test chỉ `transform`.
4. Tham số mô hình chọn trên **Val**, không bao giờ chọn dựa trên Test.
5. Các cột lag đầu Train bị thiếu (`lag_60` thiếu 22%, `*_lag90` thiếu khoảng 12%): mô hình không nhận ô trống (LSTM, BiLSTM, PCA, ARIMA) phải bỏ các dòng đầu hoặc bỏ các cột đó. Thống nhất một cách cho mọi mô hình để cùng huấn luyện trên cùng tập dòng.
6. Biến tháng (World Bank, Cục Thống kê) chỉ có 16–17 giá trị khác nhau và trễ 32–65 ngày: ghi chú khi diễn giải kết quả về độ trễ thức ăn.
7. Mọi mô hình có ngẫu nhiên: cố định `seed = 42`; học sâu chạy thêm 5 seed, báo cáo trung bình và độ lệch chuẩn.
8. Naive luôn có mặt trong mọi bảng kết quả làm mốc so sánh.

## 8. Quy ước Git

- Nhánh `main` luôn chạy được. Mỗi người **chỉ sửa file của mình** (mục 9), nên hiếm khi xung đột.
- Tin nhắn commit: `P{số}: việc đã làm`. Ví dụ `P4: them xgb_model.py chay 3 horizon`.
- Không đẩy file lớn (trọng số mô hình, môi trường ảo). Xem `.gitignore`.
- Cần đổi file của người khác: nhắn cho người đó, không tự sửa.
- Mọi kết quả đưa vào báo cáo phải có code chạy lại được từ repo.
- Bộ dữ liệu crawl có nguồn từ VietnamBiz: kiểm tra điều khoản sử dụng trước khi công bố repo hoặc dữ liệu ra ngoài lớp.
