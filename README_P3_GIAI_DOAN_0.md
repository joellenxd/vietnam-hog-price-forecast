# P3 – Nam Hiên: Bàn giao Giai đoạn 0 (dữ liệu P1 chính thức)

## Phạm vi
- `src/evaluate.py`: chuẩn CSV, MAE / RMSE / MAPE, loại nhãn NaN và dòng gốc `is_sunday_copy == 1` khi kèm quality; đếm `n` theo (model, config, region, horizon).
- `src/walkforward.py`: lịch 01/07, 01/08, 01/09, 01/10/2026, các cửa sổ dự báo theo tháng; chọn hàng huấn luyện đã lộ nhãn trước lúc retrain.
- `src/naive_arima.py`: **chỉ phần Naive trong Giai đoạn 0**, không chạy ARIMA.
- `predictions/naive_c6.csv`: CSV 8 cột chính thức.
- `results/tables/p3_naive_mae_by_region_horizon.csv`: MAE, RMSE, MAPE, n theo 3 miền × 3 horizon.
- `tests/test_p3_stage0.py`: kiểm thử đơn vị và tích hợp trên P1.

## Cách chạy (từ thư mục gốc dự án)
```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/walkforward.py
python src/naive_arima.py
python src/evaluate.py predictions/naive_c6.csv --output results/tables/p3_naive_mae_by_region_horizon.csv
```

## Quy ước quan trọng
- Nguồn: `data/raw/csv/supervised_region_daily.csv`, không chỉnh sửa dữ liệu gốc.
- Split lấy `src/split.py` và `data/split/split_config.json` của P1; Test 01/07–30/10/2026; lọc origin có t+h nằm trong Test.
- Dùng `price_ffill(t)` là dự báo Naive tại t, `y_h7/y_h14/y_h30` là giá nhãn t+h.
- Bỏ dòng *origin* `is_sunday_copy=1` và `y_true`/`y_pred` NaN; không tự động bỏ dòng chỉ vì ngày *đích* là Chủ nhật. Cần thống nhất với nhóm nếu muốn áp dụng thêm quy tắc lọc theo target.
- MAPE tính trên những giá thực tế khác 0, báo NaN nếu không có mẫu phù hợp.
- `n` là số cặp đánh giá hợp lệ (không phải tổng số ngày Test).
- Naive không có tham số cần fit/retrain; `walkforward.py` cung cấp lịch và quy tắc dùng chung cho các mô hình cần train ở giai đoạn sau.
- Nếu muốn các mô hình đánh giá cùng chính xác một tập origin, cần tạo giao của các origin hợp lệ trước khi so sánh bảng metrics.

## Cảnh báo sử dụng
Kết quả CSV là Naive trên dữ liệu thực tế do nhóm thu thập, không phải dự báo tương lai ngoài tập dữ liệu. Không sửa bản raw/P1/P2, chỉ thêm file P3.
