# P3 – Nam Hiên | Giai đoạn 1 – Việt Nam

## Sản phẩm
- `src/arima_sarimax.py`: ARIMA và SARIMAX cho ba miền, horizon 7/14/30 ngày.
- `predictions/p3_arima_sarimax.csv`: 1.626 dự báo Test với 8 cột chuẩn chung.
- `results/tables/p3_arima_sarimax_metrics.csv`: 18 nhóm MAE, RMSE, MAPE và số mẫu `n`.
- `results/tables/p3_arima_sarimax_val_selection.csv`: 12 cấu hình thử trên Validation, 6 cấu hình được chọn (theo miền × mô hình).
- `tests/test_p3_stage1.py`: kiểm tra cấu trúc, biên Test, tính khớp nhãn, bộ lọc Chủ nhật, đếm mẫu.
- `Bao_cao_P3_Giai_doan_1.docx`: bản thảo 1.1–1.6, 2.2 (Naive/ARIMA/SARIMAX), 2.3 và phụ lục kết quả.

## Yêu cầu
Python >=3.10; pandas, numpy, statsmodels. Chạy từ **thư mục gốc repo**, nơi có `data/features/data_v1.csv` và `src/evaluate.py`.

```powershell
py -m pip install numpy pandas statsmodels
py src/arima_sarimax.py
py -m unittest discover -s tests -v
```

Kết quả đã chạy: 1.626 dòng dự báo; 18 nhóm đánh giá; 25/25 tests đạt trên bộ Giai đoạn 0+1. Để tái lập 25 bài test, cần giữ cả `tests/test_p3_stage0.py` và các file P3 Giai đoạn 0 hiện có.

## Phương pháp / lưu ý quan trọng
- Tham số `(p,d,q)` được chọn trên Validation 01/04–30/06/2026, không xem sai số Test; ứng viên `(0,1,0)` và `(1,1,0)` (lưới nhỏ để có baseline thống kê tái lập nhanh).
- Tại mỗi đầu tháng Test (01/07, 01/08, 01/09, 01/10/2026), ước lượng lại tham số trên chuỗi giá quan sát đến ngày hôm trước. Trong tháng, chỉ cập nhật **trạng thái lọc** theo quan sát ngày hiện tại, không tái ước lượng tham số mỗi ngày.
- ARIMA: không dùng biến ngoại sinh. SARIMAX: hai biến lịch tuần sin/cos có thể tính trước cho ngày tương lai; thành phần ARIMA mùa vụ đặt bằng không. **Chưa sử dụng biến thức ăn/thời tiết tương lai** hay khẳng định đã chạy đầy đủ SARIMA mùa vụ. Không kết luận tác động của E1–E5 từ kết quả này.
- `data_v1.csv` dùng các cột ngày, miền, giá quan sát, cờ `is_sunday_copy`, nhãn y_h7/y_h14/y_h30 và `valid_h*`; với từng origin `t`, dự báo tại `t+h` chỉ được chấm khi target nằm trong Test, giá thực tế hợp lệ, và origin không thuộc Chủ nhật sao chép.
- Dữ liệu giá chưa báo được giữ là NaN để Kalman xử lý, không chuyển `price_ffill` thành nhãn giá thật. Không dùng giá tương lai khi tạo dự báo.
- Việc lựa chọn tham số bằng MAE gộp các horizon trên Validation có thể ưu tiên khoảng dự báo có nhiều mẫu hơn. Để có lựa chọn riêng theo horizon, nhóm cần mở rộng thử nghiệm sau, với quy trình cố định từ trước.
- Đối chiếu Giai đoạn 0: n/miền/horizon tương ứng 99, 93, 79; 813 dự báo/mô hình.

## Hướng dẫn đưa lên GitHub
**Chỉ copy phần file Giai đoạn 1** theo cùng đường dẫn, giữ nguyên code Giai đoạn 0 và `data` của P1/P2. Nhánh riêng đề nghị: `feature/p3-nam-hien-stage1` tạo từ `main` đã chứa PR #2. Nếu PR #2 chưa merge, có thể tạo nhánh từ `feature/p3-nam-hien`; không tự merge vào `main`.

1. Thêm `src/arima_sarimax.py`, `tests/test_p3_stage1.py`, hai CSV kết quả + một CSV chọn tham số, `README_P3_GIAI_DOAN_1.md` và bản thảo báo cáo nếu nhóm lưu báo cáo cùng Git.
2. Chạy test và so sánh đầu ra; xem `git diff --stat` trước khi `git add`.
3. Tạo PR, nhờ nhóm review cách chọn tham số, ngoại sinh lịch tuần, các lọc Chủ nhật và kết quả.
