# Tu dien du lieu P2 - data_v1

## 1. Tong quan

- So dong: 1458
- So cot: 74
- Nguon: supervised_region_daily
- Split: train / val / test

## 2. Danh sach cot

| Column | Data type | Role | Group | NaN |
|---|---|---|---|---:|
| date | str | Metadata | - | 0 |
| region | str | Metadata | - | 0 |
| price | float64 | Metadata | - | 68 |
| price_ffill | float64 | Metadata | - | 9 |
| is_observed | int64 | Metadata | - | 0 |
| is_imputed | int64 | Metadata | - | 0 |
| is_sunday_copy | int64 | Metadata | - | 0 |
| price_t | float64 | Feature | E0 | 9 |
| lag_1 | float64 | Feature | E0 | 12 |
| lag_2 | float64 | Feature | E0 | 15 |
| lag_3 | float64 | Feature | E0 | 18 |
| lag_7 | float64 | Feature | E0 | 30 |
| lag_14 | float64 | Feature | E0 | 51 |
| lag_21 | float64 | Feature | E0 | 72 |
| lag_30 | float64 | Feature | E0 | 99 |
| lag_60 | float64 | Feature | E0 | 189 |
| roll_mean_7 | float64 | Feature | E0 | 6 |
| roll_mean_14 | float64 | Feature | E0 | 18 |
| roll_mean_30 | float64 | Feature | E0 | 42 |
| roll_std_14 | float64 | Feature | E0 | 18 |
| return_7d | float64 | Feature | E0 | 39 |
| return_14d | float64 | Feature | E0 | 60 |
| return_30d | float64 | Feature | E0 | 108 |
| spread_vs_ref | float64 | Feature | E0 | 66 |
| region_code | int64 | Feature | E0 | 0 |
| dow | int64 | Feature | E1 | 0 |
| month | int64 | Feature | E1 | 0 |
| day_of_year | int64 | Feature | E1 | 0 |
| lunar_month | int64 | Feature | E1 | 0 |
| lunar_day | int64 | Feature | E1 | 0 |
| days_until_tet | int64 | Feature | E1 | 0 |
| days_since_tet | int64 | Feature | E1 | 0 |
| tet_pre30 | int64 | Feature | E1 | 0 |
| tet_holiday | int64 | Feature | E1 | 0 |
| tet_post30 | int64 | Feature | E1 | 0 |
| lunar_1_or_15 | int64 | Feature | E1 | 0 |
| vu_lan | int64 | Feature | E1 | 0 |
| is_public_holiday | int64 | Feature | E1 | 0 |
| wb_maize_usd_t | float64 | Feature | E2 | 0 |
| wb_soybean_meal_usd_t | int64 | Feature | E2 | 0 |
| imp_corn_unit_usd_t | float64 | Feature | E2 | 0 |
| imp_soybean_unit_usd_t | float64 | Feature | E2 | 0 |
| wb_maize_usd_t_lag30 | float64 | Feature | E2 | 0 |
| wb_soybean_meal_usd_t_lag30 | float64 | Feature | E2 | 0 |
| imp_corn_unit_usd_t_lag30 | float64 | Feature | E2 | 0 |
| imp_soybean_unit_usd_t_lag30 | float64 | Feature | E2 | 0 |
| wb_maize_usd_t_lag60 | float64 | Feature | E2 | 6 |
| wb_soybean_meal_usd_t_lag60 | float64 | Feature | E2 | 6 |
| imp_corn_unit_usd_t_lag60 | float64 | Feature | E2 | 12 |
| imp_soybean_unit_usd_t_lag60 | float64 | Feature | E2 | 12 |
| wb_maize_usd_t_lag90 | float64 | Feature | E2 | 96 |
| wb_soybean_meal_usd_t_lag90 | float64 | Feature | E2 | 96 |
| imp_corn_unit_usd_t_lag90 | float64 | Feature | E2 | 102 |
| imp_soybean_unit_usd_t_lag90 | float64 | Feature | E2 | 102 |
| imp_corn_volume_kt | float64 | Feature | E3 | 0 |
| imp_soybean_volume_kt | float64 | Feature | E3 | 0 |
| imp_feed_value_musd | float64 | Feature | E3 | 0 |
| retail_idx_mean | float64 | Excluded | - | 267 |
| asf_active_share | float64 | Feature | E4 | 0 |
| temp_mean | float64 | Feature | E5 | 0 |
| temp_max | float64 | Feature | E5 | 0 |
| rh_mean | float64 | Feature | E5 | 0 |
| rain_mm | float64 | Feature | E5 | 0 |
| thi | float64 | Feature | E5 | 0 |
| heat_day_share | float64 | Feature | E5 | 0 |
| usd_vnd | float64 | Feature | E5 | 0 |
| wb_crude_usd_bbl | float64 | Feature | E5 | 0 |
| y_h7 | float64 | Target | - | 89 |
| y_h14 | float64 | Target | - | 104 |
| y_h30 | float64 | Target | - | 152 |
| split | str | Split/Flag | - | 0 |
| valid_h7 | bool | Split/Flag | - | 0 |
| valid_h14 | bool | Split/Flag | - | 0 |
| valid_h30 | bool | Split/Flag | - | 0 |

## 3. Luu y

- Khong dung y_h7, y_h14, y_h30 lam feature.
- retail_idx_mean khong duoc dua vao cau hinh ablation.
- Chi su dung nhan khi valid_h tuong ung bang True.
- Can kiem tra gia tri NaN va thoi diem cong bo du lieu.


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
