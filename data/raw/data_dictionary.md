# Từ điển dữ liệu

Đơn vị giá heo: đồng/kg. Khóa của panel miền: (`date`, `region`), `region` ∈ {Bac, Trung, Nam};
khóa của panel tỉnh: (`date`, `province`). Phạm vi 02/07/2025 – 30/10/2026 (486 ngày, lưới ngày đầy đủ).
"As-of" = giá trị mới nhất đã công khai tính đến `date` (xem README mục 2.5).

## `panel_region_daily` (1.458 dòng) và `panel_province_daily` (16.524 dòng)

Hai bảng có cùng cột ngoại sinh; bảng tỉnh dùng ngoại sinh của miền chứa tỉnh đó và không có các cột
`price_median`, `price_min`, `price_max`, `n_provinces`, `coverage`.

### Biến mục tiêu và giá
| Cột | Mô tả |
|---|---|
| `price` | **Biến mục tiêu.** Miền: trung bình giá các tỉnh trong miền, NaN nếu < 90% số tỉnh có giá hoặc không có báo giá. Tỉnh: giá tỉnh |
| `is_observed` | 1 nếu `price` là quan sát thật |
| `price_ffill` | `price` điền tiếp ≤ 3 ngày — chỉ dùng làm input |
| `is_imputed` | 1 nếu `price_ffill` là giá trị điền tiếp |
| `is_sunday_copy` | 1 nếu là ngày Chủ nhật có bài nhưng bảng giá y hệt thứ Bảy (không phải quan sát độc lập). Không xóa, lọc bằng `is_sunday_copy == 0` |
| `price_median`, `price_min`, `price_max` | (chỉ bảng miền) thống kê trên các tỉnh có giá trong ngày |
| `n_provinces`, `coverage` | (chỉ bảng miền) số tỉnh có giá; tỷ lệ trên số tỉnh của miền (Bắc 15, Trung 11, Nam 8) |
| `price_Bac`, `price_Trung`, `price_Nam` | `price` của cả 3 miền cùng ngày (để mô hình học quan hệ liên miền) |

### Lịch (biết trước, không rò rỉ)
| Cột | Mô tả |
|---|---|
| `dow`, `is_weekend`, `month`, `day_of_year`, `week_of_year` | Lịch dương |
| `lunar_day`, `lunar_month`, `lunar_year`, `lunar_leap_month` | Lịch âm Việt Nam (UTC+7) |
| `days_until_tet`, `days_since_tet` | Số ngày tới / từ mùng 1 Tết gần nhất |
| `tet_pre30`, `tet_holiday`, `tet_post30` | 30 ngày trước Tết; mùng 1–5; 30 ngày sau nghỉ Tết |
| `lunar_1_or_15`, `vu_lan`, `hung_kings` | Mùng 1/rằm; rằm tháng 7 (13–15/7 âm); Giỗ Tổ 10/3 âm |
| `public_holiday_fixed`, `is_public_holiday` | 1/1, 30/4, 1/5, 1–2/9; gộp với Tết và Giỗ Tổ |

### Thời tiết (Open-Meteo ERA5, trung bình các tỉnh trong miền, as-of trễ 5 ngày)
| Cột | Mô tả |
|---|---|
| `temp_mean`, `temp_max` | Nhiệt độ trung bình / cao nhất ngày (°C) |
| `rh_mean`, `rain_mm` | Độ ẩm tương đối trung bình (%), lượng mưa (mm) |
| `thi` | Chỉ số nhiệt–ẩm cho heo: 1.8T + 32 − (0.55 − 0.0055·RH)(1.8T − 26) |
| `heat_day_share` | Tỷ lệ tỉnh trong miền có Tmax ≥ 35 °C |
| `weather_age_days` | `date` − ngày đo (luôn = 5) |

### Vĩ mô, chi phí thức ăn
| Cột | Mô tả |
|---|---|
| `usd_vnd`, `fx_age_days` | Tỷ giá đóng cửa gần nhất (trễ 1–3 ngày) |
| `wb_maize_usd_t`, `wb_soybeans_usd_t`, `wb_soybean_meal_usd_t`, `wb_wheat_usd_t`, `wb_rice_usd_t` | Giá thế giới USD/tấn (World Bank) |
| `wb_crude_usd_bbl` | Dầu thô bình quân USD/thùng |
| `wb_age_days` | `date` − ngày đầu tháng tham chiếu (32–65) |
| `imp_corn_volume_kt`, `imp_corn_value_musd`, `imp_corn_unit_usd_t` | Nhập khẩu ngô trong tháng: nghìn tấn, triệu USD, USD/tấn (NSO) |
| `imp_soybean_volume_kt`, `imp_soybean_value_musd`, `imp_soybean_unit_usd_t` | Như trên cho đậu tương |
| `imp_feed_value_musd` | Trị giá nhập khẩu thức ăn gia súc và nguyên liệu (triệu USD) |

### Dịch bệnh, cầu
| Cột | Mô tả |
|---|---|
| `asf_active_localities` | Số địa phương có dịch tả lợn châu Phi chưa qua 21 ngày (NSO) |
| `asf_active_share` | Chia cho tổng số tỉnh (63 cho kỳ ≤ 06/2025, 34 từ 07/2025) — dùng biến này khi so sánh qua thời gian |
| `nso_age_days` | `date` − cuối tháng tham chiếu NSO (7–37) |
| `retail_idx_winmart`, `retail_idx_ha_hien`, `retail_idx_mean` | Trung bình giá ba rọi, nạc đùi, nạc vai (đ/kg); NaN sau 10/08/2026 (nguồn ngừng đăng 03/08/2026) |
| `retail_age_days` | Tuổi giá bán lẻ (0–7 ngày) |

## `supervised_region_daily` (1.458 dòng) và `supervised_province_daily` (16.524 dòng)

Chỉ giữ cột khóa, giá, feature theo nhóm (`feature_groups.csv`) và nhãn; không chứa toàn bộ cột panel.
Bảng tỉnh có thêm `region`, không có `region_code`.

| Cột | Mô tả |
|---|---|
| `price`, `price_ffill`, `is_observed`, `is_imputed`, `is_sunday_copy` | Như panel |
| `price_t` | Giá tại ngày dự báo (`price_ffill`) |
| `lag_L` (L = 1,2,3,7,14,21,30,60) | Giá tại t − L ngày lịch |
| `roll_mean_{7,14,30}`, `roll_std_14` | Trung bình / độ lệch chuẩn trượt kết thúc tại t |
| `return_{7,14,30}d` | Thay đổi % so với t − L |
| `spread_vs_ref` | Chênh lệch với mặt bằng chung cùng ngày: miền − trung bình 3 miền; tỉnh − giá miền của tỉnh |
| `region_code` | (chỉ bảng miền) Bắc 0, Trung 1, Nam 2 |
| Nhóm lịch E1 | `dow`, `month`, `day_of_year`, `lunar_month`, `lunar_day`, `days_until_tet`, `days_since_tet`, `tet_pre30`, `tet_holiday`, `tet_post30`, `lunar_1_or_15`, `vu_lan`, `is_public_holiday` |
| `<feed>` và `<feed>_lag{30,60,90}` | `feed` ∈ {`wb_maize_usd_t`, `wb_soybean_meal_usd_t`, `imp_corn_unit_usd_t`, `imp_soybean_unit_usd_t`}; biến lag tra as-of tại t − L từ lưới ngày bắt đầu 01/04/2025 |
| Nhóm cung–cầu E3, dịch bệnh E4 | `imp_corn_volume_kt`, `imp_soybean_volume_kt`, `imp_feed_value_musd`, `retail_idx_mean`; `asf_active_share` |
| Nhóm thời tiết–vĩ mô E5 | `temp_mean`, `temp_max`, `rh_mean`, `rain_mm`, `thi`, `heat_day_share`, `usd_vnd`, `wb_crude_usd_bbl` |
| `y_h{7,14,30}` | **Nhãn**: `price` tại t + h ngày lịch; NaN nếu ngày đó không có báo giá hoặc vượt cuối dữ liệu |

Nhãn chênh lệch `y_h{h} − price_t` không lưu sẵn trong bảng, tự tính khi huấn luyện.
Nhóm biến E0–E5: xem `data/processed/feature_groups.csv`.

## `pig_price_province_daily` (15.773 dòng, chỉ tỉnh-ngày có báo giá)

| Cột | Mô tả |
|---|---|
| `province`, `region` | Tỉnh theo địa giới 34 tỉnh, miền |
| `price`, `price_min`, `price_max` | Giá (trung điểm), cận dưới, cận trên của khoảng giá nguồn báo |
| `change_reported` | Cột "Tăng/giảm" của nguồn |
| `delta_calc` | Chênh lệch tự tính so với lần báo giá trước của tỉnh |
| `flag_jump` | Biến động > 8.000 đ/kg so với kỳ trước |
| `flag_change_mismatch` | `change_reported` ≠ `delta_calc` (chỉ kiểm khi kỳ trước là hôm qua) |
| `flag_possible_stale_copy` | Ngày nghi bảng giá bị chép lại từ hôm trước (cột "Tăng/giảm" báo đổi nhưng mọi tỉnh giữ giá); ngày đầu chuỗi không xét. Hiện = 0 dòng |
| `flag_sunday_copy` | Chủ nhật có bài, bảng giá y hệt thứ Bảy |
| `url`, `published_at` | Bài nguồn, thời điểm đăng |
