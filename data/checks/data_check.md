# Bao cao kiem tra bo du lieu

Thu muc: `data\raw`

**Ket luan: DAT.**

## 1. Checksum (CHECKSUMS.sha256)


Khop: 15; sai: 0; khong co file: 0.

## 2. So dong cac bang

| Bang | Dong thuc te | Dong theo README | Ket qua |
|---|---|---|---|
| pig_price_province_daily | 15773 | 15773 | OK |
| panel_region_daily | 1458 | 1458 | OK |
| panel_province_daily | 16524 | 16524 | OK |
| supervised_region_daily | 1458 | 1458 | OK |
| supervised_province_daily | 16524 | 16524 | OK |

## 3. supervised_region_daily

- 12/12 kiem tra dat.
- Ngay khong co bao gia (mien Bac): 22 (Tet, le, mot so cuoi tuan; nguon khong dang bai, khong phai loi crawl).
- Chu nhat chep lai thu Bay (`is_sunday_copy` = 1): 61 ngay/mien. Loc `is_sunday_copy == 0` khi danh gia.
- Khoang gia: 43,909 - 80,200 dong/kg.
- Nhan `y_h*` va cot `price` bi NaN o ngay khong co bao gia, khi huan luyen loc `y_h*.notna()`.
- Cot `retail_idx_mean`: Train 2% NaN, Val 0% NaN, Test 68% NaN (nguon ngung dang; khong nen dung lam feature).

## 4. Don vi (theo data_dictionary.md cua bo crawl)

| Cot | Don vi |
|---|---|
| price, price_t, lag_*, roll_* | dong/kg |
| wb_maize/soybeans/soybean_meal/wheat/rice_usd_t | USD/tan (World Bank, theo thang) |
| wb_crude_usd_bbl | USD/thung (theo thang) |
| imp_*_volume_kt | nghin tan (NSO, theo thang) |
| imp_*_value_musd, imp_feed_value_musd | trieu USD (NSO, theo thang) |
| imp_*_unit_usd_t | USD/tan (NSO, theo thang) |
| temp_mean, temp_max | do C |
| rh_mean / rain_mm | % / mm |
| usd_vnd | VND moi USD |
| asf_active_localities / asf_active_share | so dia phuong co dich / ty le tren tong so tinh |
| retail_idx_* | dong/kg (gia ban le; ngung dang sau 03/08/2026) |
