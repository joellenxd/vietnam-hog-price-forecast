# """clean.py - Lam sach AgriProdPrice_DataSet -> clean.csv
# Chi doi ten cot, dinh dang ngay, kiem tra chat luong. KHONG sua gia tri so,
# khong dien/xoa dong, khong doi thu tu tuan.
# """
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Ten cot goc -> ten cot moi (thuong, gach duoi, sua loi chinh ta)
COLUMN_MAPPING = {
    "date": "date",
    "Live pig": "live_pig",
    "White-striped pork": "white_striped_pork",
    "Beef": "beef",
    "Dressed chicken": "dressed_chicken",
    "Large yellow coraker": "large_yellow_croaker",
    "White radish": "white_radish",
    "APWP200": "apwp200",
    "Soybean meal": "soybean_meal",
    "Corn": "corn",
    "Import volume": "import_volume",
    "Export volume": "export_volume",
    "Import value": "import_value",
    "Export value": "export_value",
    "mutton": "mutton",
}

EXPECTED_ROWS = 438
EXPECTED_START = "2016-01-03"
EXPECTED_END = "2024-05-19"
MONTHLY_COLS = ["import_volume", "export_volume", "import_value", "export_value"]


def main():
    ap = argparse.ArgumentParser(description="Lam sach dataset -> clean.csv")
    ap.add_argument("--input", default="data/raw/DataSet.xlsx", help="file xlsx goc")
    ap.add_argument("--outdir", default="data/clean", help="thu muc xuat ket qua")
    ap.add_argument("--sheet", default=0, help="ten hoac so thu tu sheet (mac dinh: sheet dau)")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        sys.exit(f"Khong thay file: {src}. Dat file goc vao data/raw/ hoac dung --input.")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_excel(src, sheet_name=args.sheet)
    report = {"input": str(src), "rows_raw": int(len(raw))}

    # 1. Kiem tra du/thieu cot, roi doi ten
    missing_cols = set(COLUMN_MAPPING) - set(raw.columns)
    extra_cols = set(raw.columns) - set(COLUMN_MAPPING)
    assert not missing_cols, f"Thieu cot so voi du kien: {missing_cols}"
    assert not extra_cols, f"Cot la khong co trong bang doi chieu: {extra_cols}"
    df = raw.rename(columns=COLUMN_MAPPING)[list(COLUMN_MAPPING.values())].copy()

    # 2. Dinh dang ngay, giu nguyen thu tu dong
    df["date"] = pd.to_datetime(df["date"])

    # 3. Kieu du lieu: cac cot con lai phai la so
    value_cols = [c for c in df.columns if c != "date"]
    for c in value_cols:
        assert pd.api.types.is_numeric_dtype(df[c]), f"Cot {c} khong phai kieu so"

    # 4. Kiem tra chat luong (dung lai neu sai)
    n_missing = int(df.isna().sum().sum())
    assert n_missing == 0, f"Con {n_missing} o thieu:\n{df.isna().sum()[df.isna().sum() > 0]}"
    assert df["date"].is_unique, "Co ngay bi trung"
    assert df["date"].is_monotonic_increasing, "Ngay khong tang dan"
    gaps = df["date"].diff().dropna().dt.days
    assert (gaps == 7).all(), f"Co tuan bi nhay/khong cach 7 ngay: {gaps[gaps != 7].to_dict()}"
    assert df["date"].dt.day_of_week.eq(6).all(), "Co ngay khong phai Chu nhat"
    assert len(df) == EXPECTED_ROWS, f"So dong {len(df)} != {EXPECTED_ROWS}"
    assert df["date"].iloc[0] == pd.Timestamp(EXPECTED_START), "Ngay dau khac du kien"
    assert df["date"].iloc[-1] == pd.Timestamp(EXPECTED_END), "Ngay cuoi khac du kien"
    assert (df[value_cols] > 0).all().all(), "Co gia tri <= 0"

    # 6. Ghi chu chat luong
    report.update({
        "rows_clean": int(len(df)),
        "date_start": str(df["date"].iloc[0].date()),
        "date_end": str(df["date"].iloc[-1].date()),
        "missing_cells": n_missing,
        "week_gap_days": "tat ca = 7",
        "monthly_columns": MONTHLY_COLS,
        "unique_values_per_monthly_col": {c: int(df[c].nunique()) for c in MONTHLY_COLS},
        "min_max": {c: [float(df[c].min()), float(df[c].max())] for c in value_cols},
    })

    # 7. Xuat file
    df.to_csv(outdir / "clean.csv", index=False, date_format="%Y-%m-%d", encoding="utf-8")
    with open(outdir / "column_mapping.json", "w", encoding="utf-8") as f:
        json.dump(COLUMN_MAPPING, f, ensure_ascii=False, indent=2)
    with open(outdir / "clean_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 8. Doc lai clean.csv, doi chieu voi file xlsx goc (phat hien loi lam tron/ghi file)
    back = pd.read_csv(outdir / "clean.csv", parse_dates=["date"])
    assert (back["date"].values == raw["date"].values).all(), "Ngay trong clean.csv khac file goc"
    for old_name, new_name in COLUMN_MAPPING.items():
        if new_name != "date":
            assert (back[new_name] - raw[old_name]).abs().max() < 1e-9, f"Cot {new_name} lech so voi file goc"

    print(f"OK: {len(df)} dong x {df.shape[1]} cot, {report['date_start']} -> {report['date_end']}")
    print(f"    0 o thieu, cac tuan cach deu 7 ngay, clean.csv khop file goc")
    print(f"    Da ghi: {outdir/'clean.csv'}, {outdir/'column_mapping.json'}, {outdir/'clean_report.json'}")


if __name__ == "__main__":
    main()
