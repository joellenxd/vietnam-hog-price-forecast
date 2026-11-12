
from pathlib import Path
import json
import ast

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from evaluate import PREDICTION_COLUMNS, evaluate_predictions
from split import is_valid_origin
from walkforward import (
    load_config,
    available_training_rows,
    get_walkforward_windows,
)


# ==========================================
# 1. CAU HINH DUONG DAN
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "features" / "data_v1.csv"
FEATURE_GROUPS_PATH = ROOT / "data" / "features" / "feature_groups.csv"
CONFIG_PATH = ROOT / "data" / "split" / "split_config.json"

BEST_PARAMS_PATH = (
    ROOT / "results" / "tables" / "xgboost_tuning_best_params.csv"
)

BASELINE_METRICS_PATH = (
    ROOT / "results" / "tables" / "xgboost_c6_metrics.csv"
)

PREDICTIONS_PATH = (
    ROOT / "predictions" / "xgboost_tuned_test.csv"
)

METRICS_PATH = (
    ROOT / "results" / "tables" / "xgboost_tuned_test_metrics.csv"
)

COMPARISON_PATH = (
    ROOT / "results" / "tables"
    / "xgboost_baseline_vs_tuned_test.csv"
)

HORIZONS = (7, 14, 30)
RANDOM_SEED = 42


# ==========================================
# 2. DOC DU LIEU VA THAM SO
# ==========================================

def load_data():
    df = pd.read_csv(DATA_PATH)
    feature_groups = pd.read_csv(FEATURE_GROUPS_PATH)

    df["date"] = pd.to_datetime(df["date"])

    feature_cols = feature_groups.loc[
        feature_groups["c6"] == 1, "feature"
    ].tolist()

    if len(feature_cols) != 59:
        raise ValueError(
            f"C6 co {len(feature_cols)} features, khong phai 59."
        )

    required_cols = {
        "date", "region", "price_t", "is_sunday_copy",
        "y_h7", "y_h14", "y_h30"
    }

    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Dataset thieu cot: {sorted(missing)}")

    missing_features = set(feature_cols) - set(df.columns)

    if missing_features:
        raise ValueError(
            f"Dataset thieu features: {sorted(missing_features)}"
        )

    df["price_t"] = pd.to_numeric(
        df["price_t"], errors="coerce"
    )
    df["is_sunday_copy"] = pd.to_numeric(
        df["is_sunday_copy"], errors="coerce"
    )

    print("Da doc dataset thanh cong!")
    print("So dong:", len(df))
    print("So features C6:", len(feature_cols))

    return df, feature_cols


def load_best_params():
    """Doc tham so da duoc chon tren Validation."""

    params_df = pd.read_csv(BEST_PARAMS_PATH)

    if "horizon" not in params_df.columns:
        raise ValueError("Bang best params thieu cot horizon.")

    params_col = (
        "params" if "params" in params_df.columns
        else "best_params" if "best_params" in params_df.columns
        else None
    )

    if params_col is None:
        raise ValueError(
            "Khong tim thay cot params trong bang best params."
        )

    best_params = {}

    for horizon in HORIZONS:
        rows = params_df.loc[
            pd.to_numeric(params_df["horizon"]) == horizon
        ]

        if len(rows) != 1:
            raise ValueError(
                f"H{horizon}: can dung 1 dong tham so, "
                f"nhung tim thay {len(rows)}."
            )

        raw_params = rows.iloc[0][params_col]

        try:
            params = json.loads(raw_params)
        except (json.JSONDecodeError, TypeError):
            params = ast.literal_eval(raw_params)

        best_params[horizon] = params

    print("\nDa doc bo tham so tot nhat tu Validation:")

    for horizon, params in best_params.items():
        print(f"H{horizon}: {params}")

    return best_params


# ==========================================
# 3. CHAY TUNED XGBOOST TREN TEST
# ==========================================

def run_tuned_test(data, feature_cols, config, best_params):

    df = data.copy()
    prediction_parts = []

    for horizon in HORIZONS:
        target_col = f"y_h{horizon}"

        print(f"\n========== TUNED XGBOOST H{horizon} ==========")

        windows = get_walkforward_windows(config, horizon)

        for window in windows:

            retrain_date = pd.Timestamp(window["retrain_date"])
            prediction_start = pd.Timestamp(
                window["prediction_start"]
            )
            prediction_end = pd.Timestamp(
                window["prediction_end"]
            )

            # A. Du lieu huan luyen:
            # Chi su dung nhan da biet truoc ngay retrain.

            train = available_training_rows(
                df,
                retrain_date,
                horizon
            )

            train[target_col] = pd.to_numeric(
                train[target_col], errors="coerce"
            )

            train["delta_target"] = (
                train[target_col] - train["price_t"]
            )

            train = train.loc[
                np.isfinite(train["delta_target"])
                & np.isfinite(train["price_t"])
            ].copy()

            if train.empty:
                raise ValueError(
                    f"Khong co du lieu train H{horizon} "
                    f"tai {retrain_date.date()}."
                )

            X_train = train[feature_cols]
            y_train = train["delta_target"]

            # B. Huan luyen voi tham so da chon tren Validation.

            model = XGBRegressor(
                random_state=RANDOM_SEED,
                **best_params[horizon]
            )

            model.fit(X_train, y_train)

            # C. Lay cac diem du bao trong thang.

            test = df.loc[
                (df["date"] >= prediction_start)
                & (df["date"] <= prediction_end)
            ].copy()

            valid_mask = is_valid_origin(
                test["date"], horizon
            ).to_numpy()

            test = test.loc[valid_mask].copy()

            # Loai ngay Chu nhat sao chep.

            test = test.loc[
                test["is_sunday_copy"].eq(0)
            ].copy()

            test[target_col] = pd.to_numeric(
                test[target_col], errors="coerce"
            )

            test = test.loc[
                np.isfinite(test["price_t"])
                & np.isfinite(test[target_col])
            ].copy()

            if test.empty:
                print(
                    f"Khong co du bao hop le "
                    f"trong thang {retrain_date.strftime('%Y-%m')}."
                )
                continue

            # D. Du bao Delta va quy doi ve gia.

            predicted_delta = model.predict(test[feature_cols])

            predicted_price = (
                test["price_t"].to_numpy() + predicted_delta
            )

            piece = pd.DataFrame({
                "model": "xgboost_tuned",
                "config": "c6_tuned",
                "region": test["region"].to_numpy(),
                "horizon": horizon,
                "origin_date": test["date"].dt.strftime("%Y-%m-%d"),
                "target_date": (
                    test["date"] + pd.Timedelta(days=horizon)
                ).dt.strftime("%Y-%m-%d"),
                "y_true": test[target_col].to_numpy(),
                "y_pred": predicted_price
            })

            prediction_parts.append(piece)

            print(
                f"Retrain {retrain_date.date()}: "
                f"{len(train)} dong train, "
                f"{len(piece)} dong du bao."
            )

    if not prediction_parts:
        raise ValueError("Khong tao duoc du bao Tuned nao.")

    predictions = pd.concat(
        prediction_parts,
        ignore_index=True
    )

    predictions = predictions[PREDICTION_COLUMNS]

    predictions = predictions.sort_values(
        ["region", "horizon", "origin_date"]
    ).reset_index(drop=True)

    return predictions


# ==========================================
# 4. SO SANH VOI BASELINE
# ==========================================

def compare_with_baseline(tuned_metrics):

    baseline = pd.read_csv(BASELINE_METRICS_PATH)

    baseline = baseline.loc[
        (baseline["model"] == "xgboost")
        & (baseline["config"] == "c6")
    ].copy()

    tuned = tuned_metrics.copy()

    comparison = baseline.merge(
        tuned,
        on=["region", "horizon"],
        suffixes=("_baseline", "_tuned"),
        validate="one_to_one"
    )

    comparison["mae_improvement"] = (
        comparison["mae_baseline"] - comparison["mae_tuned"]
    )

    comparison["rmse_improvement"] = (
        comparison["rmse_baseline"] - comparison["rmse_tuned"]
    )

    comparison["mape_improvement"] = (
        comparison["mape_baseline"] - comparison["mape_tuned"]
    )

    comparison["mae_improvement_pct"] = (
        comparison["mae_improvement"]
        / comparison["mae_baseline"] * 100
    )

    return comparison


# ==========================================
# 5. MAIN
# ==========================================

def main():

    data, feature_cols = load_data()
    config = load_config(CONFIG_PATH)
    best_params = load_best_params()

    predictions = run_tuned_test(
        data,
        feature_cols,
        config,
        best_params
    )

    quality = data[
        ["date", "region", "is_sunday_copy"]
    ].copy()

    metrics = evaluate_predictions(
        predictions,
        quality
    )

    comparison = compare_with_baseline(metrics)

    # Tao thu muc neu chua ton tai.

    PREDICTIONS_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    METRICS_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    # Luu cac ket qua.

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
        float_format="%.6f"
    )

    metrics.to_csv(
        METRICS_PATH,
        index=False,
        float_format="%.6f"
    )

    comparison.to_csv(
        COMPARISON_PATH,
        index=False,
        float_format="%.6f"
    )

    print("\n========== TUNED XGBOOST HOAN THANH ==========")

    print("Tong so dong du bao:", len(predictions))

    print("\nMetrics Tuned:")
    print(metrics.to_string(index=False))

    print("\nSo sanh Baseline va Tuned:")
    print(comparison.to_string(index=False))

    print("\nDa luu:")
    print(PREDICTIONS_PATH)
    print(METRICS_PATH)
    print(COMPARISON_PATH)


if __name__ == "__main__":
    main()
