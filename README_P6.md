# P6 – Ngọc Hương: STL-PCA-BiLSTM và kiểm định Diebold–Mariano

## Sản phẩm
| File | Nội dung |
|---|---|
| `src/stl_pca.py` | STL cuốn chiếu (chu kỳ 7) + PCA (fit Train) + BiLSTM, dự báo trực tiếp 7/14/30 ngày, tune trên Val, 5 seed, walk-forward 4 mốc. Có `assert_no_leak` chống rò rỉ |
| `src/dm_test.py` | Diebold–Mariano có hiệu chỉnh Harvey–Leybourne–Newbold, thêm kernel Bartlett để kiểm tra độ bền |
| `src/p6_report_assets.py` | Sinh bảng và hình của báo cáo từ `predictions/` |
| `predictions/stl_pca_bilstm_c6.csv` | Dự báo chính (trung bình 5 seed), 8 cột chuẩn |
| `predictions/seeds/stl_pca_bilstm_c6_seed{42..46}.csv` | Dự báo từng seed |
| `results/tables/4_2_stl_pca_bilstm_mae_horizon.csv` | MAE, RMSE, MAPE trung bình ± độ lệch chuẩn qua 5 seed |
| `results/tables/4_3_model_comparison.csv` | MAE, RMSE, MAPE mọi mô hình × horizon × miền (TB = trung bình 3 miền) |
| `results/tables/4_3_dm_pvalues.csv`, `4_3_dm_matrix_all_ae.csv` | p-value DM theo cặp mô hình, horizon, miền |
| `results/tables/3_4_3_stl_pca_bilstm_val_grid.csv`, `..._params.json` | Lưới siêu tham số trên Val và bộ được chọn theo horizon |
| `results/figures/2_2_*, 3_4_3_*, 4_2_*, 4_3_*.png` | Hình cho báo cáo |
| `reports/p6/bao_cao_p6.md` | Bản thảo các mục 2.2, 3.4.3, 3.5, 4.2, 4.3 |
| `tests/test_p6_stl_pca.py`, `tests/test_p6_dm.py` | Kiểm thử |

## Cách chạy (từ thư mục gốc repo)
```powershell
python -m pip install -r requirements.txt     # cần torch, scikit-learn, statsmodels, scipy (scipy đi kèm scikit-learn)
python -m unittest discover -s tests -v

python src/stl_pca.py --horizons 7 30 --seeds 42 --no-tune   # chạy nhanh đầu–cuối (~2 phút)
python src/stl_pca.py                                         # đầy đủ: tune Val + 5 seed + 3 horizon (lâu hơn nhiều)
python src/dm_test.py                                         # DM cho mọi file trong predictions/*.csv
python src/p6_report_assets.py                                # bảng + hình cho báo cáo
```
Muốn đưa XGBoost của P4 vào bảng so sánh trước khi nhánh P4 được merge:
```powershell
git show origin/feature/p4-khanhngoc:predictions/xgboost_c6.csv > xgboost_c6.csv
python src/p6_report_assets.py --extra xgboost_c6.csv
```
Các bảng 4.3 trong báo cáo được sinh **có** file XGBoost này; chạy `src/dm_test.py` hoặc `p6_report_assets.py` mà thiếu nó sẽ ghi đè bằng bảng không có XGBoost. `predictions/sensitivity/` chứa biến thể epoch cố định 30 (chỉ để phân tích độ nhạy, không phải mô hình chính).
Sau khi P4/P5 merge, `predictions/*.csv` tự có đủ các mô hình, không cần `--extra`.

## Quy ước và lựa chọn của P6 (cần nhóm duyệt)
- **STL cuốn chiếu.** Tại ngày t chỉ phân rã tối đa 84 ngày gần nhất (≥ 28 ngày mới bắt đầu), lấy giá trị cuối. 3 ngày không có giá (20–22/02/2026) được điền bằng giá ngày trước, chỉ dùng quá khứ.
- **PCA** nén 37 biến ngoại sinh (E1–E5 của c6). E0 (giá, lag giá) không đưa vào PCA vì STL đã thay vai trò đó. **Bỏ 4 cột `*_lag90`** (thiếu 12% đầu Train; README mục 7.5 yêu cầu thống nhất một cách). Nếu nhóm quyết định dùng cách khác (bỏ dòng, hoặc giữ lag90) thì sửa `exog_columns()`.
- **Mục tiêu** là `Δ = y_h − price_t` (giống P4), dự báo giá = `price_t + Δ`. Khác bài gốc: không dự báo riêng từng thành phần rồi cộng.
- **Chọn siêu tham số** trên Val bằng MAE của 2 seed (42, 43), riêng từng horizon: cửa sổ, hidden, dropout, lr, ngưỡng phương sai PCA, số epoch. Sau đó huấn luyện lại trên các dòng đã lộ nhãn (t + h < ngày retrain) ở 4 mốc đầu tháng Test, số epoch cố định như đã chọn. Scaler/PCA giữ nguyên bản fit trên Train.
- **Tập điểm gốc** giống P3/P4: bỏ gốc Chủ nhật sao chép và nhãn NaN. Có test khẳng định số dòng khớp `naive_c6.csv` (297/279/237).
- **DM:** tổn thất |e| (chính) và e²; chuỗi hiệu tổn thất lập trên **lịch ngày** nên các ngày thiếu (Chủ nhật, nhãn NaN) không làm lệch tự tương quan. Miền `all` là trung bình hiệu tổn thất của ba miền tại mỗi ngày gốc (ba miền chịu chung cú sốc giá nên không gộp như 3 mẫu độc lập). Cột `*_bartlett` là kiểm tra độ bền vì DM gốc không ổn định khi h = 30 với n ≈ 80.
- Chưa có LSTM/BiLSTM (P5) trong repo tại thời điểm viết; `dm_test.py` và `p6_report_assets.py` tự nhận mọi file khi chúng được thêm.
