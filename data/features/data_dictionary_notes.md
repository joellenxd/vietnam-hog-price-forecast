## 4. Nguồn và quy ước dữ liệu

Từ điển này được xây dựng từ:

- `data/raw/data_dictionary.md`
- `data/raw/csv/feature_groups.csv`
- `data/split/split_config.json`

Xem [Từ điển dữ liệu gốc P1](../raw/data_dictionary.md)
để tra cứu mô tả, đơn vị, nguồn và quy tắc as-of.

Bảng dữ liệu chính: `supervised_region_daily`.

Khóa dữ liệu: (`date`, `region`).

Đơn vị giá heo: đồng/kg.

Phạm vi: 02/07/2025 đến 30/10/2026.

## 5. Các cột do P2 bổ sung

| Cột       | Ý nghĩa                                              |
| --------- | ---------------------------------------------------- |
| split     | Tập train, val hoặc test của ngày gốc                |
| valid_h7  | True nếu y_h7 không NaN và t+7 không vượt cuối tập   |
| valid_h14 | True nếu y_h14 không NaN và t+14 không vượt cuối tập |
| valid_h30 | True nếu y_h30 không NaN và t+30 không vượt cuối tập |

Các cột trên không được sử dụng làm feature
đầu vào của mô hình.

## 6. Quy tắc lựa chọn feature

- E0: Giá heo lịch sử.
- E1: Lịch và Tết.
- E2: Thức ăn chăn nuôi.
- E3: Cung - cầu.
- E4: Dịch bệnh ASF.
- E5: Thời tiết và vĩ mô.

Thứ tự ablation: E0 -> E2 -> E1 -> E3 -> E4 -> E5.

Tổng số feature sử dụng: 59.

Không sử dụng retail_idx_mean trong các cấu hình
ablation theo quy định của README.

Danh sách feature chính thức:
`data/features/feature_groups.csv`.

Cấu hình thí nghiệm:
`data/features/ablation_config.json`.

## 7. Quy tắc sử dụng dữ liệu

- Giữ nguyên các giá trị NaN của dữ liệu nguồn.
- Không thay NaN bằng 0 khi chưa có cơ sở.
- Không sử dụng nhãn y_h7, y_h14, y_h30 làm feature.
- Chỉ dùng nhãn có valid_h tương ứng bằng True.
- Việc lọc is_sunday_copy == 0 thực hiện khi đánh giá
  theo quy ước của nhóm.
- Kiểm tra tính hợp lệ theo thời gian của lag,
  rolling và dữ liệu ngoại sinh trước khi huấn luyện.
- Scaler, PCA, STL chỉ được fit trên Train.
- Không điều chỉnh mô hình dựa trên kết quả Test.

## 8. Kết quả kiểm tra chất lượng P2

- Tổng dòng: 1458.
- Tổng cột: 74.
- Train: 819 dòng.
- Validation: 273 dòng.
- Test: 366 dòng.
- Nhóm feature: E0-E5, tổng 59 feature.
- Không phát hiện giá trị age_days âm.
- retail_age_days có 267 giá trị NaN.
- Dữ liệu lag có NaN ở đầu chuỗi và giai đoạn
  gián đoạn giá trong tháng 02/2026.
- Chưa thực hiện kiểm chứng toàn diện data leakage
  từ quy trình tạo đặc trưng và công bố dữ liệu.
