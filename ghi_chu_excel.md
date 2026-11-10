# Ghi chú: biểu đồ nào vẽ lại được bằng Excel (để tự kiểm tra)

| Hình (results/figures) | Excel? | Dữ liệu dùng (results/tables) | Cách làm |
|---|---|---|---|
| 4_1_price_by_region.png | Được (không cần vùng tô Tết) | 4_1_price_wide.csv | Chọn cột ngày + 3 vùng → Insert → Line. Vùng tô Tết/đường mốc split: bỏ qua hoặc vẽ tay |
| 4_1_price_distribution.png | Được | 4_1_price_wide.csv | Insert → Box & Whisker (3 cột vùng) |
| 4_1_tet_window.png | Được | 4_1_tet_effect.csv / price_wide | Thêm cột "ngày tương đối" = ngày − 17/02/2026, lọc −60..+60, vẽ Scatter có đường |
| 4_1_region_spread.png | Được | 4_1_price_wide.csv | Thêm cột chênh lệch (Bắc−Nam...), cột trung bình trượt 7 ngày (AVERAGE), vẽ Line |
| 4_1_lag_corr.png | Được | 4_1_lag_corr.csv | Pivot: hàng = lag, cột = chuỗi; Clustered Column |
| 4_1_corr_matrix_level.png / _change.png | Được | 4_1_corr_input.csv | =CORREL(cột i; cột j) cho từng cặp; tô màu bằng Conditional Formatting → Color Scale (3 màu, giữa = 0) |
| Bảng ADF (4_1_adf.csv) | KHÔNG | — | Chỉ chạy bằng code (statsmodels) |

Lưu ý: kiểm tra bằng Excel chỉ cần so vài giá trị (vd. 1–2 hệ số tương quan, 1 giá trị trung bình) khớp với bảng code là đủ.
