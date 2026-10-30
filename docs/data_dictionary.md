# Từ điển dữ liệu – AgriProdPrice_DataSet (Trung Quốc, theo tuần)

- File gốc: `DataSet.xlsx` (Sheet1), 438 dòng × 15 cột; mỗi dòng là một tuần, cột `date` là Chủ nhật, cách nhau đúng 7 ngày, từ 03/01/2016 đến 19/05/2024. Không có ô trống.
- Nguồn dữ liệu theo bài gốc (Liu et al., 2025, PLOS ONE, DOI 10.1371/journal.pone.0324646, mục Dataset description): giá heo hơi, ngô, khô đậu lấy từ China Hog Net; các biến còn lại lấy từ Bộ Nông nghiệp và Nông thôn Trung Quốc (MARA). Dữ liệu tuần là trung bình theo tuần; giá thiếu (ngày nghỉ lễ) đã được tác giả điền bằng nội suy spline bậc ba.
- Bài gốc không ghi đơn vị. Đơn vị dưới đây do nhóm tự suy ra, kèm căn cứ.
- Lưu ý: bài gốc ghi 418 tuần, nhưng thống kê Bảng 1 của bài khớp chính xác với cả 438 dòng của file (Live pig: trung bình 19,04, độ lệch chuẩn 7,16), nên 418 là sai số của bài, file 438 dòng là đúng.

## Bảng cột

| Cột trong file | Ý nghĩa | Đơn vị | Nguồn dữ liệu |
|---|---|---|---|
| `date` | Ngày của tuần (Chủ nhật) | ngày (YYYY-MM-DD) | – |
| `live_pig` | Giá heo hơi (biến mục tiêu) | NDT/kg | China Hog Net |
| `white_striped_pork` | Giá thịt heo hơi trắng (thịt heo bán buôn đã làm sạch) | NDT/kg | MARA |
| `beef` | Giá thịt bò | NDT/kg | MARA |
| `dressed_chicken` | Giá gà làm sẵn | NDT/kg | MARA |
| `mutton` | Giá thịt cừu/dê | NDT/kg | MARA (bài gốc Bảng 1 không liệt kê cột này) |
| `large_yellow_croaker` | Giá cá hoàng ngư lớn | NDT/kg | MARA |
| `white_radish` | Giá củ cải trắng | NDT/kg | MARA |
| `apwp200` | Chỉ số giá bán buôn nông sản 200 (Agricultural Product Wholesale Price 200 Index) | chỉ số, không có đơn vị | MARA |
| `soybean_meal` | Giá khô đậu tương | NDT/kg | China Hog Net |
| `corn` | Giá ngô | NDT/kg | China Hog Net |
| `import_volume` | Khối lượng nhập khẩu heo sống, theo tháng | CHƯA XÁC ĐỊNH | MARA |
| `export_volume` | Khối lượng xuất khẩu heo sống, theo tháng | CHƯA XÁC ĐỊNH | MARA |
| `import_value` | Giá trị nhập khẩu heo sống, theo tháng | CHƯA XÁC ĐỊNH | MARA |
| `export_value` | Giá trị xuất khẩu heo sống, theo tháng | CHƯA XÁC ĐỊNH | MARA |

Ghi chú 4 cột xuất nhập khẩu: Chưa có căn cứ cho đơn vị nên CHƯA XÁC ĐỊNH, nhóm không đoán. chỉ có 101 giá trị khác nhau trên 438 dòng, đúng bằng 101 tháng từ 01/2016 đến 05/2024, nên là số liệu tháng lặp lại theo tuần. Khi tạo đặc trưng phải dùng giá trị của tháng trước để tránh rò rỉ. Trong báo cáo ghi "đơn vị theo dữ liệu gốc, không được nêu rõ trong bài".

## Căn cứ cho đơn vị

[1] **Live pig = NDT/kg.** Tạp chí Công Thương (15/07/2021, dẫn CGTN): giá lợn hơi giảm từ 36 xuống 22 NDT/kg trong 5 tháng đầu năm 2021, rồi còn 12 NDT/kg ngày 22/6; mức đỉnh lịch sử "gần 40 NDT/kg". Trong file: trung bình tháng 1/2021 là 35,5; tháng 5/2021 là 20,1; đỉnh 37,8 (tuần 19/07/2020). https://tapchicongthuong.vn/bai-viet/gia-thit-lon-tai-trung-quoc-cham-day-giam-hon-60-tu-dau-nam-den-nay-82325.htm

[2] **Thịt heo, bò, cừu, gà = NDT/kg và APWP200 = chỉ số.** Bản tin giá của MARA đăng trên Trung Quốc Chứng khoán Báo, 18/08/2021: thịt lợn 21,62; bò 75,60; cừu 69,80; gà trắng 16,70 (đơn vị 元/公斤); chỉ số "农产品批发价格200指数" 113,72. https://cs.com.cn/xwzx/hg/202108/t20210818_6195163.html

[3] Cùng nguồn, bản tin 12/07/2021: thịt lợn 22,68; bò 76,80; cừu 72,40; gà 16,52 (元/公斤); chỉ số 108,32. https://cs.com.cn/xwzx/hg/202107/t20210712_6183088.html

[4] Nguồn mô tả dữ liệu: Liu et al., 2025, PLOS ONE, 20(5), DOI 10.1371/journal.pone.0324646, mục Dataset description và Bảng 1.
