# P2 – BÁO CÁO XÁC NHẬN DATA_V1 (GIAI ĐOẠN 1)

**Người thực hiện:** Lệ Khanh
**Ngày kiểm tra:** 03/11/2026
**Branch:** `feature/p2-le-khanh`
**Dữ liệu cần kiểm tra:** `data/features/data_v1.csv`
**Dữ liệu nguồn đối chiếu:** `data/raw/parquet/supervised_region_daily.parquet`,
`data/raw/parquet/panel_region_daily.parquet`

## 1. Mục tiêu và phạm vi

Kiểm tra tính toàn vẹn của dữ liệu nguồn, cấu trúc của `data_v1`, tính nhất quán của các đặc trưng và dấu hiệu rò rỉ dữ liệu theo thời gian (data leakage) trong phạm vi các script hiện có.

Mục tiêu cuối cùng là xác định có cần chỉnh sửa `data_v1` và tạo `data_v2` hay không.

Phạm vi kiểm tra chưa bao gồm kiểm toán độc lập toàn bộ quá trình thu thập dữ liệu, lịch công bố nguồn và tính khả dụng của mọi đặc trưng tại thời điểm dự báo.

## 2. Quy trình và công cụ kiểm tra

Các lệnh được thực hiện trong môi trường Python của dự án:

```powershell
python src/check_data.py
Get-Content data/checks/data_check.md
python src/check_features.py
python src/check_leakage.py
```

**Chức năng từng script:**

- `check_data.py`: Kiểm tra dữ liệu nguồn, checksum, số dòng và các ràng buộc; xuất báo cáo `data/checks/data_check.md`.
- `check_features.py`: Kiểm tra cấu trúc `data_v1`, các cột bắt buộc, phân chia Train/Validation/Test, cờ nhãn hợp lệ và giá trị thiếu.
- `check_leakage.py`: Đối chiếu một số công thức lag, rolling, return và các biến liên quan trên dữ liệu Parquet nguồn; kiểm tra các chỉ số `age_days` và sự hiện diện của các cột ngoại sinh.

**Minh chứng:** Output Terminal ngày 03/11/2026, file `data/checks/data_check.md`, các script và dữ liệu tương ứng trong repository.

## 3. Kết quả kiểm tra dữ liệu nguồn

| Tiêu chí                             | Kết quả                      |
| ------------------------------------ | ---------------------------- |
| Checksum                             | 15 file khớp, 0 sai, 0 thiếu |
| Số dòng của 5 bảng Parquet           | Khớp với README              |
| Kiểm tra `supervised_region_daily`   | 12/12 điều kiện đạt          |
| Ngày không có báo giá ở miền Bắc     | 22 ngày                      |
| Chủ nhật sao chép giá                | 61 ngày/miền                 |
| Khoảng giá quan sát                  | 43.909–80.200 đồng/kg        |
| `retail_idx_mean` thiếu ở Train      | Khoảng 2%                    |
| `retail_idx_mean` thiếu ở Validation | 0%                           |
| `retail_idx_mean` thiếu ở Test       | Khoảng 68%                   |

**Nhận xét:**

Các phép kiểm tra dữ liệu nguồn đều đạt theo ràng buộc được triển khai trong script.

Những ngày không có báo giá và các bản ghi Chủ nhật sao chép giá đã được nhận diện bằng thông tin hoặc cờ dữ liệu tương ứng.

Do tỷ lệ thiếu của `retail_idx_mean` ở tập Test cao, không khuyến nghị sử dụng biến này làm đặc trưng dự báo.

Kết quả kiểm tra chưa chứng minh độc lập rằng tất cả giá trị trong dữ liệu đều trùng khớp với từng nguồn xuất bản ban đầu.

## 4. Kết quả kiểm tra cấu trúc data_v1

| Tiêu chí                     | Kết quả               |
| ---------------------------- | --------------------- |
| Số dòng                      | 1.458                 |
| Số cột                       | 74                    |
| Phạm vi thời gian            | 02/07/2025–30/10/2026 |
| Số miền                      | 3 (Bắc, Trung, Nam)   |
| Khóa `date–region`           | Không trùng           |
| Train                        | 819 dòng              |
| Validation                   | 273 dòng              |
| Test                         | 366 dòng              |
| Cột bắt buộc                 | Đầy đủ                |
| Định dạng split và cờ nhãn   | Đạt                   |
| Số cột feature được thống kê | 41                    |

**Số dòng có cờ nhãn hợp lệ theo horizon:**

| Horizon | Train | Validation | Test |  Tổng |
| ------- | ----: | ---------: | ---: | ----: |
| 7 ngày  |   745 |        243 |  342 | 1.330 |
| 14 ngày |   730 |        222 |  321 | 1.273 |
| 30 ngày |   682 |        174 |  273 | 1.129 |

**Nhận xét:**

`data_v1` đáp ứng các điều kiện kiểm tra cấu trúc được triển khai trong `check_features.py`.

Số lượng nhãn hợp lệ khác nhau giữa các horizon do điều kiện sẵn có của nhãn và giới hạn ngày đích trong từng tập dữ liệu.

Cờ `valid_h*` không đồng nghĩa với việc đã loại bỏ mọi dòng Chủ nhật sao chép giá. Do đó, khi đánh giá mô hình cần áp dụng riêng quy tắc lọc `is_sunday_copy == 1` theo thiết kế của nhóm.

## 5. Kiểm tra giá trị thiếu (NaN)

| Đặc trưng                   | Số NaN | Nhận xét                                       |
| --------------------------- | -----: | ---------------------------------------------- |
| `lag_1`                     |     12 | Thiếu ở đầu chuỗi và sau gián đoạn báo giá     |
| `lag_60`                    |    189 | Train: 180; Validation: 9                      |
| `roll_mean_30`              |     42 | Có thể do chưa đủ lịch sử hoặc gián đoạn       |
| `wb_maize_usd_t_lag90`      |     96 | Thuộc Train                                    |
| `imp_corn_unit_usd_t_lag90` |    102 | Thuộc Train                                    |
| `spread_vs_ref`             |     66 | Tương ứng 22 ngày × 3 miền trong Parquet nguồn |
| `retail_age_days`           |    267 | Thiếu trong bảng dữ liệu nguồn được kiểm tra   |

**Phân tích:**

Các đặc trưng lag và rolling có thể xuất hiện NaN tại đầu chuỗi do chưa có đủ dữ liệu lịch sử.

Trong giai đoạn 20–22/02/2026, `price_ffill` bị thiếu ở cả ba miền. Điều này dẫn tới một số giá trị lag tiếp theo bị thiếu.

Những giá trị thiếu này chưa đủ căn cứ để xác định là lỗi dữ liệu vì có thể phản ánh đúng quy tắc giới hạn điền giá đã được thiết kế.

**Nguyên tắc xử lý:**

- Không tự động thay NaN bằng 0.
- Không sử dụng dữ liệu tương lai để điền giá trị thiếu.
- Không thay đổi quy tắc tiền xử lý khi chưa có căn cứ xác định lỗi.
- Các thành viên xây dựng mô hình cần thống nhất phương pháp xử lý NaN để bảo đảm khả năng so sánh kết quả.

## 6. Kiểm tra công thức và nguy cơ Data Leakage

### 6.1. Kết quả đối chiếu công thức

| Nội dung kiểm tra              | Kết quả                                  |
| ------------------------------ | ---------------------------------------- |
| 8 biến lag giá heo             | 0 dòng sai khác                          |
| 4 biến rolling                 | 0 dòng sai khác                          |
| `price_t` so với `price_ffill` | 0 dòng sai khác                          |
| `return_7d`                    | 0 sai khác trên 1.419 dòng được kiểm tra |
| `return_14d`                   | 0 sai khác trên 1.398 dòng được kiểm tra |
| `return_30d`                   | 0 sai khác trên 1.350 dòng được kiểm tra |
| `spread_vs_ref`                | 0 dòng sai khác                          |
| 29 cột ngoại sinh bắt buộc     | Đầy đủ tên cột                           |

Các công thức được đối chiếu trên dữ liệu Parquet nguồn đều đạt trong phạm vi kiểm tra của script.

### 6.2. Kiểm tra age_days

| Biến               | Min | Max | Số NaN | Giá trị âm |
| ------------------ | --: | --: | -----: | ---------: |
| `weather_age_days` |   5 |   5 |      0 |          0 |
| `fx_age_days`      |   1 |   3 |      0 |          0 |
| `wb_age_days`      |  32 |  65 |      0 |          0 |
| `nso_age_days`     |   7 |  37 |      0 |          0 |
| `retail_age_days`  |   0 |   7 |    267 |          0 |

Không phát hiện giá trị `age_days` âm trong các cột được kiểm tra.

Tuy nhiên, điều này chưa chứng minh độc lập rằng lịch công bố của tất cả nguồn ngoại sinh hoàn toàn chính xác.

### 6.3. Giới hạn kiểm tra leakage

**Giới hạn 1 – Thời điểm công bố dữ liệu ngoại sinh**

Việc kiểm tra `age_days` không thay thế cho đối chiếu độc lập thời điểm công bố thực tế của dữ liệu World Bank, NSO, tỷ giá, thời tiết hoặc các nguồn khác.

Cần phối hợp P1 để xác nhận quy tắc as-of.

**Giới hạn 2 – Thời điểm phát dự báo**

Một số đặc trưng như `price_t`, `spread_vs_ref` và rolling có thể sử dụng thông tin ngày `t`.

Việc sử dụng chỉ hợp lệ khi thông tin này thực sự sẵn có tại thời điểm mô hình phát dự báo.

Nếu dự báo được phát trước khi có giá của ngày `t`, cần điều chỉnh quy ước hoặc cách sử dụng đặc trưng.

**Giới hạn 3 – Quy trình huấn luyện và đánh giá**

Các script hiện tại chưa xác nhận toàn bộ quy trình xử lý dữ liệu trong mô hình, bao gồm:

- Fit scaler hoặc bộ biến đổi dữ liệu trên tập huấn luyện phù hợp.
- Lựa chọn đặc trưng không sử dụng thông tin tương lai.
- Retraining đúng quy tắc thời gian.
- Lọc nhãn và các bản ghi Chủ nhật khi đánh giá.
- Xử lý dữ liệu thiếu nhất quán giữa các mô hình.

**Giới hạn 4 – Phạm vi dữ liệu được kiểm tra**

`check_leakage.py` chủ yếu kiểm tra công thức trên Parquet nguồn P1. Kết quả này không đồng nghĩa với việc mọi phép biến đổi sau đó trong `data_v1` đã được kiểm toán độc lập.

## 7. Kết luận và quyết định phiên bản dữ liệu

Sau khi thực hiện ba script:

- `check_data.py`: Đạt.
- `check_features.py`: Đạt.
- `check_leakage.py`: Đạt trong phạm vi các phép kiểm tra đã triển khai.

Không phát hiện lỗi cấu trúc hoặc sai khác công thức thuộc phạm vi kiểm tra buộc phải chỉnh sửa `data_v1`.

**Quyết định:**

Tiếp tục sử dụng `data/features/data_v1.csv` và **chưa tạo `data_v2`**.

Quyết định này có điều kiện, không đồng nghĩa với việc đã loại trừ mọi nguy cơ leakage hoặc xác minh được tất cả giá trị từ nguồn ban đầu.

Nếu các bước kiểm toán tiếp theo phát hiện lỗi thực sự ảnh hưởng đến dữ liệu hoặc đặc trưng, nhóm cần:

1. Ghi nhận lỗi và bằng chứng.
2. Xác định phạm vi ảnh hưởng.
3. Tạo `data_v2` nếu cần.
4. Ghi rõ các thay đổi trong changelog.
5. Thông báo cho các thành viên sử dụng dữ liệu để đồng bộ phiên bản.

## 8. Công việc cần phối hợp và bàn giao

| Nội dung                 | Người phối hợp | Yêu cầu                                                          |
| ------------------------ | -------------- | ---------------------------------------------------------------- |
| Quy tắc as-of            | P1             | Xác nhận thời điểm công bố và tính khả dụng của nguồn ngoại sinh |
| Quy ước thời điểm dự báo | Nhóm mô hình   | Xác định thông tin ngày `t` được phép sử dụng                    |
| Xử lý NaN                | Nhóm mô hình   | Thống nhất phương pháp xử lý                                     |
| Lọc Chủ nhật             | Nhóm đánh giá  | Áp dụng đúng quy tắc đánh giá                                    |
| `retail_idx_mean`        | Nhóm mô hình   | Không sử dụng làm feature theo khuyến nghị của README            |
| Quản lý phiên bản        | P2             | Chỉ tạo `data_v2` khi có lỗi được xác nhận                       |

**Trạng thái Task 1 – P2:** Hoàn thành các bước kiểm tra hiện có, chưa phát hiện lỗi cần sửa `data_v1`. Các giới hạn kiểm toán còn lại đã được ghi nhận để phối hợp xác minh.
