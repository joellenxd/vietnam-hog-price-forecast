# Bộ dữ liệu giá heo hơi Việt Nam theo ngày (sau sáp nhập tỉnh)

Giá heo hơi hằng ngày của 34 tỉnh/thành, gộp thành 3 miền (Bắc, Trung, Nam), kèm lịch Việt Nam
(có âm lịch, Tết), thời tiết, tỷ giá, giá nguyên liệu thức ăn chăn nuôi, nhập khẩu, dịch tả lợn châu Phi (ASF)
và giá thịt bán lẻ. Dùng cho bài toán dự báo giá 7/14/30 ngày.

- **Phạm vi:** 02/07/2025 → 30/10/2026, 486 ngày liên tục. Chỉ giai đoạn sau sáp nhập tỉnh (34 tỉnh).
- **Đơn vị giá:** đồng/kg. Giá nguồn rời rạc theo bước 1.000 đ.
- **Mã hóa:** CSV là UTF-8 có BOM (mở thẳng bằng Excel); `region` ∈ {`Bac`, `Trung`, `Nam`}.

## 1. Gồm những gì

| Bảng (có cả `csv/` và `parquet/`) | Số dòng | Mỗi dòng là | Dùng khi |
|---|---|---|---|
| `pig_price_province_daily` | 15.773 | 1 tỉnh × 1 ngày có báo giá (dữ liệu gốc đã chuẩn hóa) | kiểm tra nguồn, tự gộp theo cách khác |
| `panel_region_daily` | 1.458 | 3 miền × 486 ngày (lưới đầy đủ) + toàn bộ biến ngoại sinh | phân tích, dự báo cấp miền |
| `panel_province_daily` | 16.524 | 34 tỉnh × 486 ngày + biến ngoại sinh của miền | mô hình dùng chung cho 34 tỉnh |
| `supervised_region_daily` | 1.458 | như panel miền nhưng đã tạo sẵn feature và nhãn | huấn luyện mô hình ngay |
| `supervised_province_daily` | 16.524 | như trên, cấp tỉnh | huấn luyện mô hình ngay |
| `feature_groups.csv` | 6 | nhóm biến E0–E5 cho thí nghiệm ablation | chọn tập biến |
| `dropped_out_of_range_rows.csv` | 1 | dòng bị loại vì giá ngoài [20.000; 120.000] (lỗi gõ của nguồn) | đối chiếu |

Tài liệu đi kèm: `data_dictionary.md` (ý nghĩa từng cột), `quality_report.md` (kiểm định chi tiết),
`CHECKSUMS.sha256` (kiểm tra tệp không bị hỏng).

**Chưa biết dùng bảng nào?** Bắt đầu với `supervised_region_daily`: cột `y_h7`, `y_h14`, `y_h30` là nhãn
(giá sau 7/14/30 ngày), các cột còn lại là feature.

## 2. Các cột quan trọng

- `price` — **biến mục tiêu**: giá heo hơi trung bình các tỉnh trong miền (hoặc giá của tỉnh). Là `NaN` ở ngày nguồn không đăng giá.
- `is_observed` — 1 nếu `price` là quan sát thật.
- `price_ffill`, `is_imputed` — giá điền tiếp tối đa 3 ngày, **chỉ để làm input**, không dùng làm nhãn.
- `is_sunday_copy` — 1 nếu là Chủ nhật mà bảng giá y hệt thứ Bảy (xem mục 4).
- `y_h7`, `y_h14`, `y_h30` — nhãn: `price` tại ngày t + h (bảng `supervised_*`).
- Biến ngoại sinh có hậu tố `*_age_days` = số ngày từ kỳ tham chiếu đến `date`; luôn ≥ 0 (không dùng thông tin tương lai).

## 3. Cách đọc

```python
import pandas as pd
df = pd.read_parquet("parquet/supervised_region_daily.parquet")   # hoặc pd.read_csv("csv/...", parse_dates=["date"])
bac = df[(df.region_code == 0) & df.y_h7.notna()]
```

## 4. Chất lượng — điều cần biết trước khi dùng

**Đã kiểm và đạt**
- Không có khóa trùng; lưới ngày đầy đủ 486 ngày cho mỗi miền; CSV và parquet khớp nhau.
- Giá miền bằng đúng trung bình giá các tỉnh (sai số 0); mọi giá nằm trong 43.000–81.000 đ/kg.
- Nhãn `y_h*` và các cột `lag_*` được kiểm khớp với chuỗi giá dịch ngày; biến ngoại sinh ghép theo *as-of join*, không âm tuổi (không rò rỉ).
- Cột "Tăng/giảm" của nguồn khớp chênh lệch tự tính 99,28%; 110 dòng lệch được gắn cờ `flag_change_mismatch`, giữ nguyên.

**Đã làm sạch**
- Loại 1 dòng giá lỗi gõ (ngoài [20.000; 120.000]).
- Sửa cờ `flag_possible_stale_copy` (trước đó gắn nhầm cả ngày đầu chuỗi); hiện không có ngày nào nghi chép lại.
- Thêm cờ `is_sunday_copy` / `flag_sunday_copy`.

**Giới hạn — không tự sửa, người dùng phải xử lý**

| Vấn đề | Chi tiết | Gợi ý |
|---|---|---|
| Chủ nhật là bản chép của thứ Bảy | 61/61/60 ngày (Bắc/Trung/Nam) trong số 464/464/462 ngày có giá. Còn lại ~403 ngày độc lập mỗi miền | Lọc `is_sunday_copy == 0` khi đánh giá hoặc tính thống kê "không đổi giá" |
| Ngày không có giá (22 ngày) | Tết, lễ và một số ngày thường (vd 04/11/2025, 14/11/2025, 27/05/2026). Đã đối chiếu: nguồn không đăng bài giá heo hơi ngày đó | `price` = NaN, không phải lỗi thu thập |
| `retail_idx_*` (giá thịt bán lẻ) | Nguồn ngừng đăng sau 03/08/2026; có giá trị đến 10/08/2026 rồi NaN (18,3% dòng) | Không dùng hoặc loại khỏi feature ở giai đoạn cuối |
| Giá tỉnh đi ngang dài | Mỗi tỉnh có đoạn giá không đổi, trung vị 33 ngày liên tiếp, dài nhất 44 ngày. Giống đặc điểm báo giá của nguồn | Giá miền (trung bình) mượt hơn giá tỉnh |
| Mẫu ngắn, một nguồn giá | ~1,3 năm, chỉ VietnamBiz, chưa đối chiếu nguồn giá thứ hai | Ghi rõ trong phần hạn chế của bài báo |
| Giá cuối kỳ thấp bất thường | Tháng 10/2026 ~44–46k, thấp hơn mọi giá trước đó trong mẫu | Mô hình cây không ngoại suy được; dự báo chênh lệch (y_h − price_t) |
| Biến tháng | World Bank, NSO có ~17 giá trị khác nhau trong cả kỳ | Thận trọng khi diễn giải ablation |
| 4,7% ngày miền không có giá; nhãn thiếu | `y_h7/14/30` có 1.369 / 1.354 / 1.306 dòng khác NaN (trên 1.458) | Lọc `y_h*.notna()` khi huấn luyện |

## 5. Nguồn dữ liệu

| Dữ liệu | Nguồn |
|---|---|
| Giá heo hơi theo tỉnh, giá thịt bán lẻ | VietnamBiz (bài "giá heo hơi hôm nay", bảng giá theo tỉnh) |
| ASF, nhập khẩu ngô/đậu tương/thức ăn chăn nuôi | Báo cáo kinh tế–xã hội hằng tháng, Cục Thống kê (NSO) |
| Giá thế giới ngô, đậu tương, khô đậu, lúa mì, gạo, dầu thô | World Bank Commodity Price Data (Pink Sheet) |
| Thời tiết | Open-Meteo (ERA5), tọa độ các tỉnh |
| Tỷ giá USD/VND | Yahoo Finance |
| Âm lịch, Tết, ngày lễ | Tự tính (đã kiểm Tết 2018–2027) |

Giá heo hơi và bán lẻ được thu thập (crawl) từ website VietnamBiz. Kiểm tra điều khoản sử dụng của nguồn
trước khi phân phối lại hoặc công bố bảng `pig_price_province_daily`.

## 6. Tái tạo

Từ thư mục dự án gốc: `python pipeline.py --steps build,report,release` (parse lại từ cache, vài giây,
cho kết quả giống hệt). Mã nguồn ở `src/pigdata/`, README gốc ở `../../README.md`.
