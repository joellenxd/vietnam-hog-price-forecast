# P2 – Báo cáo kiểm tra và chuẩn bị dữ liệu đầu vào

**Thành viên phụ trách:** Lệ Khanh

**Nguồn dữ liệu:** `data/raw/parquet/supervised_region_daily.parquet`

**Phương pháp phân tập:** Sử dụng `assign_split()` và `is_valid_origin()` từ `src/split.py` của P1.

**Đối chiếu cấu hình:** Các mốc Train/Validation/Test và số lượng nhãn hợp lệ được đối chiếu với `data/split/split_config.json`.

## 1. Kết quả tạo dữ liệu `data_v1`

Dữ liệu đầu vào gồm 1.458 dòng × 70 cột, tương ứng ba miền Bắc, Trung và Nam trong 486 ngày, từ 02/07/2025 đến 30/10/2026.

P2 sử dụng các hàm `assign_split()` và `is_valid_origin()` của P1 để bổ sung bốn cột:

- `split`: xác định tập Train, Validation hoặc Test.
- `valid_h7`: cờ nhãn hợp lệ cho dự báo 7 ngày.
- `valid_h14`: cờ nhãn hợp lệ cho dự báo 14 ngày.
- `valid_h30`: cờ nhãn hợp lệ cho dự báo 30 ngày.

Sau xử lý, `data_v1.csv` có **1.458 dòng × 74 cột**.

| Tập dữ liệu | Khoảng thời gian      |   Số dòng | h7 hợp lệ | h14 hợp lệ | h30 hợp lệ |
| ----------- | --------------------- | --------: | --------: | ---------: | ---------: |
| Train       | 02/07/2025–31/03/2026 |       819 |       745 |        730 |        682 |
| Validation  | 01/04/2026–30/06/2026 |       273 |       243 |        222 |        174 |
| Test        | 01/07/2026–30/10/2026 |       366 |       342 |        321 |        273 |
| **Tổng**    |                       | **1.458** | **1.330** |  **1.273** |  **1.129** |

Một điểm gốc `t` chỉ có cờ nhãn hợp lệ khi nhãn `y_h` không thiếu và ngày đích `t+h` không vượt ngày cuối của tập dữ liệu tương ứng.

P2 không xóa điểm gốc khỏi bộ dữ liệu, mà sử dụng các cờ để các mô hình lọc dữ liệu theo từng horizon.

### Kiểm tra khả năng tái tạo

Chạy `python src/features.py` cho kết quả:

- Dữ liệu tạo lại khớp với `data_v1.csv` hiện có trong dung sai đối chiếu.
- Không ghi đè file `data_v1.csv` đã tồn tại.
- Không phát hiện trùng khóa `date–region`.

## 2. Kiểm tra các đặc trưng bắt buộc

P2 đã kiểm tra 41 cột đặc trưng theo các nhóm yêu cầu:

| Nhóm đặc trưng  | Nội dung                                       | Số cột |
| --------------- | ---------------------------------------------- | -----: |
| Lag giá heo     | 1, 2, 3, 7, 14, 21, 30, 60 ngày                |      8 |
| Rolling giá heo | Mean 7/14/30 ngày và Std 14 ngày               |      4 |
| Lag thức ăn     | Bốn chuỗi × độ trễ 30/60/90 ngày               |     12 |
| Lịch âm và Tết  | Tháng/ngày âm lịch, các giai đoạn Tết, ngày lễ |     10 |
| ASF             | `asf_active_share`                             |      1 |
| Thời tiết       | Nhiệt độ, độ ẩm, lượng mưa, THI, ngày nóng     |      6 |
| **Tổng**        |                                                | **41** |

Kết quả `check_features.py`: không phát hiện thiếu cột bắt buộc, trùng khóa `date–region` hoặc giá trị split không hợp lệ.

Các đặc trưng cần thiết đã có trong dữ liệu P1 nên **P2 không tạo thêm biến trùng lặp**, tuân thủ yêu cầu chỉ bổ sung khi thiếu.

41 cột trên là phạm vi kiểm tra các nhóm đặc trưng trọng tâm, không phải toàn bộ 59 feature dùng cho ablation.

## 3. Kiểm tra giá trị thiếu

| Cột                         | Số NaN | Ghi chú                                       |
| --------------------------- | -----: | --------------------------------------------- |
| `lag_1`                     |     12 | Đầu chuỗi và gián đoạn giá tháng 02/2026      |
| `lag_60`                    |    189 | 180 Train, 9 Validation                       |
| `roll_mean_30`              |     42 | Chủ yếu đầu chuỗi và giai đoạn thiếu giá      |
| `wb_maize_usd_t_lag90`      |     96 | Thuộc Train                                   |
| `imp_corn_unit_usd_t_lag90` |    102 | Thuộc Train                                   |
| `spread_vs_ref`             |     66 | 22 ngày × 3 miền                              |
| `retail_age_days`           |    267 | Cột kiểm tra tuổi dữ liệu thuộc bảng panel P1 |

Trong giai đoạn 20–22/02/2026, ba miền không có `price` và `price_ffill`; việc này làm một số lag tiếp tục thiếu ở các ngày kế tiếp.

P2 giữ nguyên NaN, không điền 0, không tự xóa dòng hoặc sử dụng dữ liệu tương lai để bổ sung.

Trước khi huấn luyện, nhóm phải thống nhất cách xử lý NaN và tập dòng dùng chung để bảo đảm so sánh mô hình công bằng.

## 4. Kiểm tra công thức và chống rò rỉ dữ liệu

Các phép kiểm tra được thực hiện bằng `src/check_leakage.py`.

| Nội dung             | Kết quả                                                 |
| -------------------- | ------------------------------------------------------- |
| 8 lag giá heo        | 0 dòng sai khác                                         |
| Rolling mean 7/14/30 | 0 dòng sai khác                                         |
| Rolling std 14       | 0 dòng sai khác trong dung sai `atol=0.01`, `rtol=1e-6` |
| `price_t`            | Khớp `price_ffill`                                      |
| `return_7d`          | 0 sai khác trên 1.419 dòng đủ điều kiện                 |
| `return_14d`         | 0 sai khác trên 1.398 dòng đủ điều kiện                 |
| `return_30d`         | 0 sai khác trên 1.350 dòng đủ điều kiện                 |
| `spread_vs_ref`      | 0 dòng sai khác; 66 NaN                                 |
| `*_age_days`         | Không phát hiện giá trị âm, sai định dạng hoặc vô cực   |
| Đặc trưng ngoại sinh | Đủ 29 cột được kiểm tra                                 |

Năm cột tuổi dữ liệu được kiểm tra gồm:

| Cột                | Khoảng giá trị | Số NaN |
| ------------------ | -------------- | -----: |
| `weather_age_days` | 5              |      0 |
| `fx_age_days`      | 1–3            |      0 |
| `wb_age_days`      | 32–65          |      0 |
| `nso_age_days`     | 7–37           |      0 |
| `retail_age_days`  | 0–7            |    267 |

Các phép kiểm tra không phát hiện sai khác ngoài dung sai cho phép.

**Giới hạn kiểm chứng:** Việc `*_age_days` không âm chưa chứng minh độc lập toàn bộ thời điểm công bố dữ liệu ngoại sinh. Quy trình dự báo phải bảo đảm chỉ sử dụng thông tin thực sự đã sẵn có đến hết ngày gốc `t`.

Đặc biệt, các biến dùng giá cùng ngày như `price_t` và `spread_vs_ref` chỉ phù hợp nếu thông tin giá liên quan đã được công bố trước thời điểm phát hành dự báo.

## 5. Phân nhóm biến và cấu hình ablation

P2 đã tạo bảng ánh xạ gồm 59 feature, phân vào sáu nhóm:

| Nhóm     | Nội dung           | Số feature |
| -------- | ------------------ | ---------: |
| E0       | Giá lịch sử        |         18 |
| E1       | Lịch và Tết        |         13 |
| E2       | Thức ăn            |         16 |
| E3       | Cung–cầu           |          3 |
| E4       | Dịch bệnh          |          1 |
| E5       | Thời tiết và vĩ mô |          8 |
| **Tổng** |                    |     **59** |

Thứ tự ablation theo README: **E0 → E2 → E1 → E3 → E4 → E5**.

| Cấu hình | Nhóm biến cộng dồn          | Số feature |
| -------- | --------------------------- | ---------: |
| c1       | E0                          |         18 |
| c2       | E0 + E2                     |         34 |
| c3       | E0 + E2 + E1                |         47 |
| c4       | E0 + E2 + E1 + E3           |         50 |
| c5       | E0 + E2 + E1 + E3 + E4      |         51 |
| c6       | E0 + E2 + E1 + E3 + E4 + E5 |         59 |

Bảng `data/features/feature_groups.csv` có **59 dòng × 9 cột**, bao gồm `feature`, `group`, `group_name` và sáu cột đánh dấu `c1`–`c6` với giá trị 1/0.

Cấu hình ablation được lưu trong `data/features/ablation_config.json`.

Biến `retail_idx_mean` không được sử dụng làm feature theo README.

## 6. Từ điển dữ liệu P2

Từ điển được lưu tại `data/features/data_dictionary_v1.md`, dựa trên cấu trúc dữ liệu và tài liệu P1.

74 cột được phân loại:

| Vai trò    | Số cột |
| ---------- | -----: |
| Feature    |     59 |
| Metadata   |      7 |
| Split/Flag |      4 |
| Target     |      3 |
| Excluded   |      1 |
| **Tổng**   | **74** |

Từ điển gồm tám mục: tổng quan, danh sách cột, lưu ý, nguồn và quy ước dữ liệu, các cột P2 bổ sung, quy tắc lựa chọn feature, quy tắc sử dụng dữ liệu và kết quả kiểm tra chất lượng.

Phần thống kê được tạo bằng `src/create_dictionary.py`. Nội dung giải thích bổ sung được lưu trong `data/features/data_dictionary_notes.md` và tự động ghép vào từ điển khi chạy script.

Đã kiểm tra chạy lại hai lần: tài liệu giữ đủ tám mục và không nhân đôi nội dung.

## 7. Quy tắc bàn giao cho các thành viên xây dựng mô hình

1. Sử dụng `data_v1` làm bộ dữ liệu đầu vào thống nhất, không tự tạo phiên bản riêng.
2. Chỉ sử dụng dữ liệu đã sẵn có đến hết ngày dự báo `t`.
3. Lọc điểm gốc bằng `valid_h7`, `valid_h14`, `valid_h30` tương ứng với từng horizon.
4. Khi đánh giá, bỏ dòng có nhãn thiếu và `is_sunday_copy == 1`.
5. Thống nhất cách xử lý NaN và tập dòng so sánh giữa các mô hình.
6. Scaler, STL, PCA chỉ fit trên Train; Val/Test chỉ transform theo quy tắc của nhóm.
7. Chọn tham số trên Validation, không lựa chọn dựa trên kết quả Test.
8. Tuân thủ cấu hình ablation c1–c6; cấu hình mặc định là c6.
9. Ghi chú độ trễ công bố và số lượng giá trị khác nhau của các biến kinh tế theo tháng khi diễn giải kết quả.

## 8. Quy trình tái tạo và kiểm chứng

Chạy từ thư mục gốc repository:

```powershell
python src/check_data.py
python src/split.py
python src/features.py
python src/check_features.py
python src/check_leakage.py
python src/check_feature_groups.py
python src/check_ablation.py
python src/create_dictionary.py
```

Các script P2 đã được chạy kiểm tra trên bộ dữ liệu hiện tại. Nếu `data_v1.csv` đã tồn tại, `features.py` đối chiếu dữ liệu tạo lại và không ghi đè.

Đầu ra chính của P2:

- `data/features/data_v1.csv`
- `data/features/feature_groups.csv`
- `data/features/ablation_config.json`
- `data/features/data_dictionary_v1.md`
- `data/features/data_dictionary_notes.md`
- `data/features/p2_check_report.md`

Các file Python phục vụ tạo dữ liệu và kiểm tra được lưu trong thư mục `src/`.

## 9. Kết luận

P2 đã hoàn thành việc chuẩn bị `data_v1` từ bảng `supervised_region_daily`, gắn split và cờ nhãn hợp lệ cho ba horizon, kiểm tra các đặc trưng bắt buộc và đối chiếu các công thức tạo đặc trưng giá.

Đã tạo bảng ánh xạ 59 feature vào E0–E5, cấu hình ablation c1–c6, từ điển dữ liệu và báo cáo kiểm tra. Không phát hiện sai khác ngoài dung sai cho phép trong các phép kiểm thử đã thực hiện.

Dữ liệu P1 và các giá trị thiếu được giữ nguyên. Bộ dữ liệu P2 đã sẵn sàng về cấu trúc để bàn giao; việc đưa vào thí nghiệm mô hình cần tuân thủ quy tắc xử lý NaN chung, kiểm soát thời điểm công bố dữ liệu ngoại sinh và quy trình đánh giá của README.
