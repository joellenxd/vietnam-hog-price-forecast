
"""
KIEM TRA FEATURE VA CHAT LUONG DATA_V1 - P2

Chay tu thu muc goc:
    python src/check_features.py

Chi kiem tra, khong thay doi du lieu.
"""

from pathlib import Path
import pandas as pd


# ==========================================
# 1. DUONG DAN
# ==========================================

DATA_PATH = Path("data/features/data_v1.csv")


# ==========================================
# 2. DANH SACH FEATURE BAT BUOC
# ==========================================

# Lag gia heo
PRICE_LAGS = [
    "lag_1", "lag_2", "lag_3", "lag_7",
    "lag_14", "lag_21", "lag_30", "lag_60"
]

# Trung binh va do lech chuan truot
ROLLING = [
    "roll_mean_7",
    "roll_mean_14",
    "roll_mean_30",
    "roll_std_14"
]

# Lag gia thuc an: 30, 60, 90 ngay
FEED_LAGS = [
    f"{name}_lag{lag}"
    for name in [
        "wb_maize_usd_t",
        "wb_soybean_meal_usd_t",
        "imp_corn_unit_usd_t",
        "imp_soybean_unit_usd_t"
    ]
    for lag in [30, 60, 90]
]

# Lich am va Tet
CALENDAR = [
    "lunar_month",
    "lunar_day",
    "days_until_tet",
    "days_since_tet",
    "tet_pre30",
    "tet_holiday",
    "tet_post30",
    "lunar_1_or_15",
    "vu_lan",
    "is_public_holiday"
]

# Dich benh
ASF = [
    "asf_active_share"
]

# Thoi tiet
WEATHER = [
    "temp_mean",
    "temp_max",
    "rh_mean",
    "rain_mm",
    "thi",
    "heat_day_share"
]

# Nhom feature can kiem tra
GROUPS = {
    "Lag gia heo": PRICE_LAGS,
    "Rolling gia heo": ROLLING,
    "Lag thuc an": FEED_LAGS,
    "Lich am va Tet": CALENDAR,
    "Dich benh ASF": ASF,
    "Thoi tiet": WEATHER
}

# Cot du lieu va nhan bat buoc
CORE = [
    "date",
    "region",
    "price",
    "price_ffill",
    "is_observed",
    "is_imputed",
    "is_sunday_copy",
    "split",
    "y_h7",
    "y_h14",
    "y_h30",
    "valid_h7",
    "valid_h14",
    "valid_h30"
]


# ==========================================
# 3. HAM KIEM TRA CHINH
# ==========================================

def main():

    # --------------------------------------
    # 3.1. KIEM TRA FILE
    # --------------------------------------

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Khong tim thay {DATA_PATH}. "
            "Hay chay python src/features.py truoc."
        )

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"]
    )

    print("=== TONG QUAN DATA_V1 ===")
    print("So dong:", len(df))
    print("So cot:", len(df.columns))

    # --------------------------------------
    # 3.2. KIEM TRA CAC COT BAT BUOC
    # --------------------------------------

    required = CORE.copy()

    for cols in GROUPS.values():
        required.extend(cols)

    # Danh sach kiem tra khong duoc trung
    assert len(required) == len(set(required)), (
        "Danh sach cot bat buoc bi trung!"
    )

    missing = sorted(
        set(required) - set(df.columns)
    )

    if missing:
        raise AssertionError(
            "THIEU CAC COT BAT BUOC: "
            + ", ".join(missing)
        )

    print("\nDAT: Du cac cot bat buoc.")

    # --------------------------------------
    # 3.3. KIEM TRA KHOA DATE - REGION
    # --------------------------------------

    assert not df.empty, "Du lieu rong!"

    assert df["date"].notna().all(), (
        "Cot date co gia tri thieu!"
    )

    assert df["region"].notna().all(), (
        "Cot region co gia tri thieu!"
    )

    assert not df.duplicated(
        ["date", "region"]
    ).any(), "Trung khoa date-region!"

    print("DAT: Khong trung khoa date-region.")

    # --------------------------------------
    # 3.4. KIEM TRA TRAIN / VAL / TEST
    # --------------------------------------

    valid_splits = ["train", "val", "test"]

    assert df["split"].isin(valid_splits).all(), (
        "Co gia tri split khong hop le!"
    )

    assert set(df["split"].unique()) == set(
        valid_splits
    ), "Thieu mot trong ba tap du lieu!"

    print("\n=== SO DONG MOI TAP ===")

    print(
        df["split"]
        .value_counts()
        .reindex(valid_splits)
    )

    # --------------------------------------
    # 3.5. KIEM TRA CO NHAN HOP LE
    # --------------------------------------

    print("\n=== KIEM TRA NHAN HOP LE ===")

    for h in [7, 14, 30]:

        flag_col = f"valid_h{h}"
        label_col = f"y_h{h}"

        # Doc co duoi dang chuoi de kiem tra
        flag_text = (
            df[flag_col]
            .astype(str)
            .str.lower()
        )

        allowed = ["true", "false", "1", "0"]

        assert flag_text.isin(allowed).all(), (
            f"{flag_col} khong hop le!"
        )

        valid = flag_text.isin(
            ["true", "1"]
        )

        # Neu flag = True thi nhan khong duoc NaN
        invalid_rows = (
            valid & df[label_col].isna()
        )

        assert not invalid_rows.any(), (
            f"{flag_col}=True nhung "
            f"{label_col} bi thieu!"
        )

        print(f"\nHorizon {h} ngay:")

        print(
            valid.groupby(df["split"])
            .sum()
            .reindex(valid_splits)
        )

    # --------------------------------------
    # 3.6. KIEM TRA FEATURE VA NaN
    # --------------------------------------

    print("\n=== KIEM TRA FEATURE ===")

    total_checked = 0

    for group_name, cols in GROUPS.items():

        print(f"\n[{group_name}]")

        for col in cols:

            if col not in df.columns:
                raise AssertionError(
                    f"THIEU FEATURE: {col}"
                )

            missing_count = int(
                df[col].isna().sum()
            )

            print(
                f"{col}: {missing_count} NaN"
            )

            total_checked += 1

    print(
        "\nTong so cot feature da kiem tra:",
        total_checked
    )

    # --------------------------------------
    # 3.7. THONG KE NaN THEO TAP
    # --------------------------------------

    selected_cols = [
        "lag_1",
        "lag_7",
        "lag_30",
        "lag_60",
        "roll_mean_30",
        "wb_maize_usd_t_lag90",
        "imp_corn_unit_usd_t_lag90"
    ]

    print("\n=== NaN THEO TRAIN / VAL / TEST ===")

    missing_by_split = (
        df.groupby("split")[selected_cols]
        .apply(lambda x: x.isna().sum())
        .reindex(valid_splits)
    )

    print(missing_by_split.to_string())

    # --------------------------------------
    # 3.8. KIEM TRA NGAY THIEU LAG_1
    # --------------------------------------

    print("\n=== CAC NGAY THIEU LAG_1 ===")

    missing_lag1 = df.loc[
        df["lag_1"].isna(),
        ["date", "region", "split", "lag_1"]
    ]

    print(
        missing_lag1.to_string(index=False)
    )

    # --------------------------------------
    # 3.9. KIEM TRA GIA DIP TET 2026
    # --------------------------------------

    print("\n=== GIA VA LAG DIP TET 2026 ===")

    tet_rows = df.loc[
        df["date"].between(
            "2026-02-18",
            "2026-02-25"
        )
    ]

    tet_columns = [
        "date",
        "region",
        "price",
        "price_ffill",
        "is_observed",
        "is_imputed",
        "is_sunday_copy",
        "lag_1",
        "lag_2",
        "lag_3"
    ]

    print(
        tet_rows
        .sort_values(["region", "date"])[
            tet_columns
        ]
        .to_string(index=False)
    )

    # --------------------------------------
    # 3.10. KET LUAN
    # --------------------------------------

    print("\n================================")
    print("DAT: KIEM TRA FEATURE THANH CONG")
    print("================================")

    print(
        "Tat ca cot bat buoc deu ton tai."
    )

    print(
        "Khong trung khoa date-region."
    )

    print(
        "Split va co nhan dung dinh dang."
    )

    print(
        "NaN duoc thong ke, khong tu dong xu ly."
    )

    print(
        "Luu y: Cong thuc lag, rolling va "
        "quy tac as-of duoc kiem tra rieng "
        "trong check_leakage.py."
    )


# ==========================================
# 4. CHAY CHUONG TRINH
# ==========================================

if __name__ == "__main__":
    main()
