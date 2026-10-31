# Báo cáo chất lượng dữ liệu

Sinh tự động lúc 2026-10-30 20:06. Phạm vi: 2025-07-02 → 2026-10-30 (486 ngày), 34 tỉnh theo địa giới sau 01/07/2025.

## 1. Biến mục tiêu `price` (đồng/kg, trung bình các tỉnh trong miền)

| region | days_observed | pct_observed | max_gap_days | mean_provinces | y_mean | y_std | y_min | y_max |
|---|---|---|---|---|---|---|---|---|
| Bac | 464 | 95.47 | 6 | 15.00 | 62,018.68 | 7,066.01 | 44,533.33 | 80,200.00 |
| Nam | 462 | 95.06 | 6 | 8.00 | 61,623.11 | 6,691.83 | 45,000.00 | 76,750.00 |
| Trung | 464 | 95.47 | 6 | 11.00 | 60,782.13 | 7,047.73 | 43,909.09 | 77,454.55 |

Ngày không có báo giá (22): Tết, lễ và một số cuối tuần — nguồn không đăng bài, **không phải lỗi crawl**.

`2025-07-12, 2025-07-13, 2025-08-23, 2025-08-24, 2025-08-31, 2025-09-05, 2025-10-11, 2025-10-12, 2025-11-04, 2025-11-14, 2026-02-15, 2026-02-17, 2026-02-18, 2026-02-19, 2026-02-20, 2026-02-21, 2026-02-22, 2026-05-01, 2026-05-02, 2026-05-27, 2026-07-05, 2026-09-02`

## 2. Kiểm định khi chuẩn hóa

```json
{
 "articles_in_scope": 464,
 "articles_skipped_not_hog_price": 1,
 "articles_without_table": 0,
 "dates_with_duplicate_articles": 0,
 "rows_unmatched_province": {},
 "rows_total": 15774,
 "rows_out_of_range_dropped": 1,
 "rows_jump_flagged": 0,
 "change_checks": 15275,
 "change_mismatch_rate": 0.007201309328968904,
 "possible_stale_copy_days": 0,
 "sundays_with_prices": 62,
 "sunday_copy_days": 61
}
```

- `change_mismatch_rate`: tỷ lệ dòng mà cột *Tăng/giảm* của nguồn khác chênh lệch tự tính giữa hai ngày liên tiếp — thước đo nhất quán nội tại của nguồn.
- `rows_out_of_range_dropped`: giá ngoài [20.000; 120.000] đ/kg (lỗi gõ), xem `dropped_out_of_range_rows.csv`.
- `possible_stale_copy_days`: ngày mà mọi tỉnh giữ giá nhưng cột *Tăng/giảm* báo có đổi (ngày đầu chuỗi không xét).
- `sunday_copy_days`: Chủ nhật có bài nhưng bảng giá y hệt thứ Bảy; gắn cờ `flag_sunday_copy` / `is_sunday_copy`, không xóa.
- `rows_jump_flagged`: biến động > 8.000 đ/kg/ngày, gắn cờ `flag_jump`, không tự xóa.

## 3. Độ phủ theo tỉnh (% số ngày có giá)

| region | province | pct_days |
|---|---|---|
| Bac | Bắc Ninh | 95.5 |
| Bac | Cao Bằng | 95.5 |
| Bac | Hà Nội | 95.5 |
| Bac | Hưng Yên | 95.5 |
| Bac | Hải Phòng | 95.5 |
| Bac | Lai Châu | 95.5 |
| Bac | Lào Cai | 95.5 |
| Bac | Lạng Sơn | 95.5 |
| Bac | Ninh Bình | 95.5 |
| Bac | Phú Thọ | 95.5 |
| Bac | Quảng Ninh | 95.5 |
| Bac | Sơn La | 95.5 |
| Bac | Thái Nguyên | 95.5 |
| Bac | Tuyên Quang | 95.5 |
| Bac | Điện Biên | 95.5 |
| Nam | Cần Thơ | 95.1 |
| Nam | An Giang | 95.5 |
| Nam | Cà Mau | 95.5 |
| Nam | TP HCM | 95.5 |
| Nam | Tây Ninh | 95.5 |
| Nam | Vĩnh Long | 95.5 |
| Nam | Đồng Nai | 95.5 |
| Nam | Đồng Tháp | 95.5 |
| Trung | Huế | 95.3 |
| Trung | Gia Lai | 95.5 |
| Trung | Hà Tĩnh | 95.5 |
| Trung | Khánh Hòa | 95.5 |
| Trung | Lâm Đồng | 95.5 |
| Trung | Nghệ An | 95.5 |
| Trung | Quảng Ngãi | 95.5 |
| Trung | Quảng Trị | 95.5 |
| Trung | Thanh Hóa | 95.5 |
| Trung | Đà Nẵng | 95.5 |
| Trung | Đắk Lắk | 95.5 |

## 4. Độ phủ biến ngoại sinh trong panel (%)

| variable | pct_non_null |
|---|---|
| temp_mean | 100.0 |
| temp_max | 100.0 |
| rh_mean | 100.0 |
| rain_mm | 100.0 |
| thi | 100.0 |
| heat_day_share | 100.0 |
| usd_vnd | 100.0 |
| wb_maize_usd_t | 100.0 |
| wb_soybeans_usd_t | 100.0 |
| wb_soybean_meal_usd_t | 100.0 |
| wb_wheat_usd_t | 100.0 |
| wb_rice_usd_t | 100.0 |
| wb_crude_usd_bbl | 100.0 |
| asf_active_localities | 100.0 |
| asf_active_share | 100.0 |
| imp_corn_volume_kt | 100.0 |
| imp_corn_value_musd | 100.0 |
| imp_corn_unit_usd_t | 100.0 |
| imp_soybean_volume_kt | 100.0 |
| imp_soybean_value_musd | 100.0 |
| imp_soybean_unit_usd_t | 100.0 |
| imp_feed_value_musd | 100.0 |
| retail_idx_ha_hien | 78.0 |
| retail_idx_winmart | 80.2 |
| retail_idx_mean | 81.7 |

`retail_idx_*` dừng sau 03/08/2026 vì nguồn ngừng đăng; không điền tiếp quá 7 ngày.

## 5. Kiểm tra rò rỉ thông tin tương lai

Mỗi biến ngoại sinh được ghép bằng *as-of join*: tại ngày `date` chỉ dùng bản ghi có `available_at ≤ date`. `*_age_days` = khoảng cách từ kỳ tham chiếu tới `date`; giá trị âm sẽ là rò rỉ.

```json
{
 "weather_age_days": {
  "min": 5.0,
  "max": 5.0,
  "negative": 0
 },
 "fx_age_days": {
  "min": 1.0,
  "max": 3.0,
  "negative": 0
 },
 "wb_age_days": {
  "min": 32.0,
  "max": 65.0,
  "negative": 0
 },
 "nso_age_days": {
  "min": 7.0,
  "max": 37.0,
  "negative": 0
 },
 "retail_age_days": {
  "min": 0.0,
  "max": 7.0,
  "negative": 0
 }
}
```
