
"""
P2 - KIEM TRA LEAKAGE VA CONG THUC FEATURE

Chay tu thu muc goc:
    python src/check_leakage.py

Chi doc du lieu, khong ghi de file P1/P2.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================
# 1. DUONG DAN DU LIEU
# ==========================================

PANEL_PATH = Path(
    "data/raw/parquet/panel_region_daily.parquet"
)

SUPERVISED_PATH = Path(
    "data/raw/parquet/supervised_region_daily.parquet"
)

PRICE_LAGS = [1, 2, 3, 7, 14, 21, 30, 60]
HORIZONS = [7, 14, 30]

RTOL = 1e-6
ATOL = 1e-6


# ==========================================
# 2. HAM SO SANH
# ==========================================

def check_equal(actual, expected, name, atol=ATOL):
    """So sanh hai day so, cho phep NaN cung vi tri."""

    equal = np.isclose(
        actual,
        expected,
        equal_nan=True,
        rtol=RTOL,
        atol=atol
    )

    mismatches = int((~equal).sum())

    print(f"{name}: {mismatches} dong sai khac")

    if mismatches > 0:
        raise AssertionError(
            f"SAI {name}: {mismatches} dong"
        )


# ==========================================
# 3. KIEM TRA AGE_DAYS
# ==========================================

def check_age_days(panel):

    print("\n=== KIEM TRA AGE_DAYS ===")

    age_cols = [
        col for col in panel.columns
        if col.endswith("_age_days")
    ]

    if not age_cols:
        raise AssertionError(
            "Khong tim thay cot *_age_days!"
        )

    print("Cac cot:", age_cols)

    for col in age_cols:

        original = panel[col]

        values = pd.to_numeric(
            original,
            errors="coerce"
        )

        # Gia tri co san nhung khong doi duoc sang so
        invalid = (
            original.notna()
            & values.isna()
        )

        # Khong cho phep vo cuc
        infinite = pd.Series(
            np.isinf(values.to_numpy(dtype=float)),
            index=values.index
        )

        negative = int((values < 0).sum())
        missing = int(values.isna().sum())

        print(f"\nCot: {col}")
        print("Min:", values.min())
        print("Max:", values.max())
        print("So gia tri am:", negative)
        print("So gia tri thieu:", missing)
        print("So gia tri sai dinh dang:", int(invalid.sum()))
        print("So gia tri vo cuc:", int(infinite.sum()))

        if negative > 0:
            raise AssertionError(
                f"{col}: Co {negative} gia tri am!"
            )

        if invalid.any() or infinite.any():
            raise AssertionError(
                f"{col}: Co gia tri khong hop le!"
            )

    print("\nDAT: Khong phat hien age_days am.")
    print(
        "LUU Y: age_days khong am chua chung minh "
        "toan bo lich cong bo nguon da dung."
    )


# ==========================================
# 4. KIEM TRA LAG GIA HEO
# ==========================================

def check_price_lags(df):

    print("\n=== KIEM TRA LAG GIA HEO ===")

    for lag in PRICE_LAGS:

        col = f"lag_{lag}"

        expected = (
            df.groupby("region")["price_ffill"]
            .shift(lag)
        )

        check_equal(
            df[col],
            expected,
            col
        )

    print("DAT: 8 lag gia heo khop.")


# ==========================================
# 5. KIEM TRA ROLLING
# ==========================================

def check_rolling(df):

    print("\n=== KIEM TRA ROLLING ===")

    rolling_config = {
        7: 3,
        14: 7,
        30: 15
    }

    for window, min_periods in rolling_config.items():

        col = f"roll_mean_{window}"

        expected = (
            df.groupby("region")["price_ffill"]
            .transform(
                lambda x: x.rolling(
                    window=window,
                    min_periods=min_periods
                ).mean()
            )
        )

        check_equal(
            df[col],
            expected,
            col
        )

    expected_std = (
        df.groupby("region")["price_ffill"]
        .transform(
            lambda x: x.rolling(
                window=14,
                min_periods=7
            ).std()
        )
    )

    # Dung sai 0.01 cho do lech chuan
    check_equal(
        df["roll_std_14"],
        expected_std,
        "roll_std_14",
        atol=0.01
    )

    print("DAT: 4 bien rolling khop.")


# ==========================================
# 6. KIEM TRA PRICE_T
# ==========================================

def check_price_t(df):

    print("\n=== KIEM TRA PRICE_T ===")

    check_equal(
        df["price_t"],
        df["price_ffill"],
        "price_t"
    )

    print("DAT: price_t khop price_ffill.")


# ==========================================
# 7. KIEM TRA RETURN
# ==========================================

def check_returns(df):

    print("\n=== KIEM TRA CAC BIEN RETURN ===")

    for h in HORIZONS:

        col = f"return_{h}d"
        lag_col = f"lag_{h}"

        current = df["price_t"]
        past = df[lag_col]

        valid = (
            current.notna()
            & past.notna()
            & (past != 0)
        )

        expected = (
            current.loc[valid] - past.loc[valid]
        ) / past.loc[valid]

        actual = df.loc[valid, col]

        print(
            f"{col}: Kiem tra {int(valid.sum())} dong"
        )

        check_equal(
            actual,
            expected,
            col
        )

    print("DAT: 3 bien return khop cong thuc.")


# ==========================================
# 8. KIEM TRA SPREAD_VS_REF
# ==========================================

def check_spread(df):

    print("\n=== KIEM TRA SPREAD_VS_REF ===")

    # Gia trung binh quan sat cua cac mien cung ngay
    reference = (
        df.groupby("date")["price"]
        .transform("mean")
    )

    # Gia cua mien su dung price_ffill
    expected = df["price_ffill"] - reference

    check_equal(
        df["spread_vs_ref"],
        expected,
        "spread_vs_ref"
    )

    missing = df[
        df["spread_vs_ref"].isna()
    ]

    print(
        "So dong thieu spread:",
        len(missing)
    )

    print(
        "So ngay thieu spread:",
        missing["date"].nunique()
    )

    print("\nSo NaN theo mien:")

    print(
        df.groupby("region")["spread_vs_ref"]
        .apply(lambda x: x.isna().sum())
    )

    print("DAT: spread_vs_ref khop cong thuc.")


# ==========================================
# 9. KIEM TRA CAC COT NGOAI SINH
# ==========================================

def check_external_features(df):

    print("\n=== KIEM TRA FEATURE NGOAI SINH ===")

    feed_names = [
        "wb_maize_usd_t",
        "wb_soybean_meal_usd_t",
        "imp_corn_unit_usd_t",
        "imp_soybean_unit_usd_t"
    ]

    feed_lags = [
        f"{name}_lag{lag}"
        for name in feed_names
        for lag in [30, 60, 90]
    ]

    calendar_cols = [
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

    weather_cols = [
        "temp_mean",
        "temp_max",
        "rh_mean",
        "rain_mm",
        "thi",
        "heat_day_share"
    ]

    required = (
        feed_lags
        + calendar_cols
        + weather_cols
        + ["asf_active_share"]
    )

    missing_cols = sorted(
        set(required) - set(df.columns)
    )

    if missing_cols:
        raise AssertionError(
            "Thieu feature ngoai sinh: "
            + ", ".join(missing_cols)
        )

    print(
        "So cot ngoai sinh da kiem tra:",
        len(required)
    )

    print("DAT: Du cac cot ngoai sinh bat buoc.")
    print(
        "LUU Y: Can doi chieu quy tac cong bo "
        "as-of cua P1 khi danh gia leakage."
    )


# ==========================================
# 10. CHUONG TRINH CHINH
# ==========================================

def main():

    if not PANEL_PATH.is_file():
        raise FileNotFoundError(PANEL_PATH)

    if not SUPERVISED_PATH.is_file():
        raise FileNotFoundError(SUPERVISED_PATH)

    panel = pd.read_parquet(PANEL_PATH)

    df = pd.read_parquet(SUPERVISED_PATH)

    df["date"] = pd.to_datetime(df["date"])

    # Kiem tra cot co ban
    basic_cols = [
        "date",
        "region",
        "price",
        "price_ffill",
        "price_t",
        "spread_vs_ref",
        "roll_mean_7",
        "roll_mean_14",
        "roll_mean_30",
        "roll_std_14",
    ]

    basic_cols += [
        f"lag_{lag}" for lag in PRICE_LAGS
    ]

    basic_cols += [
        f"return_{h}d" for h in HORIZONS
    ]

    missing_cols = sorted(
        set(basic_cols) - set(df.columns)
    )

    if missing_cols:
        raise AssertionError(
            "Thieu cot de kiem tra: "
            + ", ".join(missing_cols)
        )

    if df["date"].isna().any():
        raise AssertionError("Ngay khong hop le!")

    if df[["date", "region"]].isna().any().any():
        raise AssertionError(
            "Thieu date hoac region!"
        )

    if df.duplicated(["date", "region"]).any():
        raise AssertionError(
            "Trung khoa date-region!"
        )

    df = df.sort_values(
        ["region", "date"]
    ).reset_index(drop=True)

    print("=== TONG QUAN ===")
    print("So dong supervised:", len(df))
    print("So mien:", df["region"].nunique())

    check_age_days(panel)
    check_price_lags(df)
    check_rolling(df)
    check_price_t(df)
    check_returns(df)
    check_spread(df)
    check_external_features(df)

    print("\n======================================")
    print("DAT: CAC PHEP KIEM TRA LEAKAGE P2")
    print("======================================")

    print(
        "Khong phat hien sai khac trong "
        "cac cong thuc da doi chieu."
    )

    print(
        "Khong phat hien age_days am."
    )

    print(
        "GIOI HAN: Chua kiem toan doc lap "
        "toan bo thoi diem cong bo du lieu."
    )


if __name__ == "__main__":
    main()
