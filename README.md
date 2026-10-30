# Dự báo đa bước giá heo hơi Trung Quốc bằng Machine Learning và Deep Learning

Đồ án: *Dự báo đa bước giá heo hơi Trung Quốc bằng Machine Learning và Deep Learning: vai trò của độ trễ giá thức ăn, yếu tố thay thế–thương mại và chu kỳ thị trường.*

- Dữ liệu: AgriProdPrice_DataSet (Trung Quốc, theo tuần), 438 tuần, 03/01/2016 đến 19/05/2024. Nguồn: https://github.com/2523438346/AgriProdPrice_DataSet. Không thu thập thêm dữ liệu.
- Bài tham chiếu: Liu et al., 2025, PLOS ONE, DOI 10.1371/journal.pone.0324646. Nhóm tự viết code mới, chỉ áp dụng mô hình và thuật toán của bài.
- Mô hình: Naive, ARIMA/SARIMAX, XGBoost, LSTM, BiLSTM, STL-PCA-BiLSTM. Horizon dự báo: 1, 2, 4, 8, 12 tuần (dự báo trực tiếp, mỗi horizon một mô hình).

## 1. Cài đặt và chạy

Yêu cầu Python 3.10 trở lên. Chạy mọi lệnh **từ thư mục gốc của repo**.

```
python -m venv .venv
.venv\Scripts\activate          # Windows;  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python src/clean.py             # data/raw -> data/clean/clean.csv
python src/split.py             # -> data/split/split_config.json
```

Đặt file gốc `DataSet.xlsx` vào `data/raw/` trước khi chạy `clean.py`.

## 2. Cấu trúc thư mục

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── asf_config.json            # ngày bắt đầu, kết thúc ASF
├── data/
│   ├── raw/                       # file xlsx gốc, KHÔNG sửa
│   ├── clean/                     # clean.csv, column_mapping.json, clean_report.json
│   ├── split/                     # split_config.json
│   └── features/                  # data_v1.csv, data_v2.csv nếu có lỗi
├── docs/
│   └── data_dictionary.md         # tên cột, ý nghĩa, đơn vị, nguồn
├── src/
│   ├── clean.py                   # P1
│   ├── split.py                   # P1
│   ├── features.py                # P2
│   ├── evaluate.py, walkforward.py# P3
│   ├── naive_arima.py             # P3
│   ├── xgb_model.py               # P4
│   ├── lstm_model.py              # P5
│   ├── stl_pca.py, dm_test.py     # P6
├── notebooks/                     # EDA và thử nghiệm, đặt tên pN_noidung.ipynb
├── predictions/                   # file dự báo CSV của mọi mô hình
└── results/
    ├── tables/                    # bảng MAE, RMSE, MAPE, ablation, DM...
    └── figures/                   # hình dùng trong báo cáo (PNG)
```

## 3. Quy ước đặt tên

- **Thư mục, file code, tên cột:** chữ thường, nối bằng gạch dưới (`snake_case`), không dấu, không khoảng trắng. Ví dụ `live_pig`, `xgb_model.py`.
- **Tên cột dữ liệu:** dùng đúng tên trong `clean.csv` (ví dụ `live_pig`, `corn`, `soybean_meal`). Tên gốc và tên mới xem `data/clean/column_mapping.json`.
- **Biến trễ và trung bình trượt:** `{tên_cột}_lag{k}` và `{tên_cột}_roll{k}`. Ví dụ `corn_lag13`, `soybean_meal_roll4`. Mùa vụ: `sin_52`, `cos_52`. ASF: `asf_flag`. Nhãn tập: `split` (`train`, `val`, `test`).
- **Tên mô hình** (cột `model` trong file dự báo): `naive`, `arima`, `sarimax`, `xgboost`, `lstm`, `bilstm`, `stl_pca_bilstm`.
- **Tên cấu hình ablation** (cột `config`): `c1` đến `c7` theo thứ tự cộng dồn nhóm biến (xem `data/features/` do P2 ghi). Cấu hình mặc định khi không chạy ablation: `c7`.
- **File dự báo:** `predictions/{model}_{config}.csv`. Ví dụ `predictions/xgboost_c3.csv`.
- **Phiên bản dữ liệu:** `data_v1.csv`, nếu có lỗi thì `data_v2.csv`. Không ghi đè, không ai tự tạo bản riêng.
- **Hình và bảng:** `results/figures/{muc}_{noidung}.png`, `results/tables/{muc}_{noidung}.csv`, với `muc` là mục trong báo cáo (ví dụ `4_2_xgboost_mae_horizon.csv`).
- **Notebook:** `p{số}_{noidung}.ipynb`. Ví dụ `p1_eda.ipynb`.

## 4. Định dạng file dự báo

Mỗi file là CSV có đúng các cột (P3 chốt trong `evaluate.py`, nếu đổi thì sửa tại đây):

`model, config, horizon, origin_date, target_date, y_true, y_pred`

- `origin_date`: tuần dùng để dự báo (chỉ dùng dữ liệu đến hết tuần này).
- `target_date`: tuần được dự báo (`origin_date` cộng `horizon` tuần).
- Mô hình học sâu chạy nhiều seed: ghi trung bình các seed vào file chính, file từng seed để trong thư mục con `predictions/seeds/`.

## 5. Quy tắc chống rò rỉ dữ liệu (bắt buộc)

1. Tại tuần dự báo `t`, chỉ dùng dữ liệu đến hết tuần `t`.
2. Chia **theo thứ tự thời gian**, không xáo trộn: Train 286 tuần (03/01/2016 đến 20/06/2021), Val 52 tuần (27/06/2021 đến 19/06/2022), Test 100 tuần (26/06/2022 đến 19/05/2024). Cấu hình trong `data/split/split_config.json`, dùng `from split import assign_split` thay vì chép tay ngày.
3. Biến trễ tính trên **toàn bộ chuỗi** theo thứ tự thời gian rồi mới gán nhãn `split`. Không tách `train.csv`, `val.csv`, `test.csv` rồi tạo lag riêng từng file.
4. Scaler, STL, PCA chỉ **fit trên Train**; Val và Test chỉ `transform`.
5. Tham số mô hình chọn trên **Val**, không bao giờ chọn dựa trên Test.
6. Số liệu xuất nhập khẩu theo tháng: dùng giá trị của **tháng trước**.
7. Giá thịt heo hơi trắng (`white_striped_pork`): chỉ dùng trễ từ 1 tuần trở lên (gần trùng với giá heo hơi).
8. Walk-forward: huấn luyện lại mỗi 12 tuần trên tập Test (khoảng 8 lần).
9. Mọi mô hình có ngẫu nhiên: cố định `seed = 42` (học sâu: chạy thêm 5 seed khác nhau, báo cáo trung bình và độ lệch chuẩn).
10. Mô hình Naive luôn có mặt trong mọi bảng kết quả làm mốc so sánh.

## 6. Quy ước Git

- Nhánh `main` luôn chạy được. Mỗi người **chỉ sửa file của mình** (xem bảng phân công trong mục 2) nên hiếm khi xung đột.
- Tin nhắn commit: `P{số}: việc đã làm`. Ví dụ `P4: them xgb_model.py chay 5 horizon`.
- Không đẩy file lớn (trọng số mô hình, môi trường ảo). Xem `.gitignore`.
- Cần đổi file của người khác: nhắn cho người đó, không tự sửa.
- Mọi kết quả đưa vào báo cáo phải có code chạy lại được từ repo.
