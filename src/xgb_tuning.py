
from pathlib import Path
import json

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from evaluate import evaluate, evaluate_predictions, PREDICTION_COLUMNS
from split import is_valid_origin
from walkforward import load_config


# ==========================================
# 1. CAU HINH
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "features" / "data_v1.csv"
FEATURE_GROUPS_PATH = ROOT / "data" / "features" / "feature_groups.csv"
CONFIG_PATH = ROOT / "data" / "split" / "split_config.json"

RESULTS_PATH = ROOT / "results" / "tables"
PREDICTIONS_PATH = ROOT / "predictions"

HORIZONS = (7, 14, 30)
RANDOM_SEED = 42

# 9 bo tham so thu nghiem.
# Baseline default duoc giu lai de so sanh.
CANDIDATES = [
    {
        "trial": "baseline_default",
        "params": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.3,
            "subsample": 1.0,
            "colsample_bytree": 1.0
        }
    },
    *[
        {
            "trial": f"tune_{i:02d}",
            "params": {
                "n_estimators": n,
                "max_depth": depth,
                "learning_rate": lr,
                "subsample": 0.8,
                "colsample_bytree": 0.8
            }
        }
        for i, (n, depth, lr) in enumerate([
            (200, 3, 0.03),
            (200, 3, 0.10),
            (200, 5, 0.03),
            (200, 5, 0.10),
            (400, 3, 0.03),
            (400, 3, 0.10),
            (400, 5, 0.03),
            (400, 5, 0.10)
        ], start=1)
    ]
]


# ==========================================
# 2. DOC DU LIEU
# ==========================================

def load_data():
    df = pd.read_csv(DATA_PATH)
    feature_groups = pd.read_csv(FEATURE_GROUPS_PATH)
    config = load_config(CONFIG_PATH)

    df["date"] = pd.to_datetime(df["date"])

    feature_cols = feature_groups.loc[
        feature_groups["c6"] == 1, "feature"
    ].tolist()

    required = {
        "date", "region", "price_t",
        "is_sunday_copy",
        "y_h7", "y_h14", "y_h30"
    }

    missing = required - set(df.columns)
    missing_features = set(feature_cols) - set(df.columns)

    if missing:
        raise ValueError(f"Dataset thieu cot: {sorted(missing)}")

    if missing_features:
        raise ValueError(
            f"Dataset thieu features: {sorted(missing_features)}"
        )

    if len(feature_cols) != 59:
        raise ValueError(
            f"C6 phai co 59 features, hien tai co {len(feature_cols)}"
        )

    print("Da doc dataset va cau hinh thanh cong!")
    print("So dong:", len(df))
    print("So features C6:", len(feature_cols))

    return df, feature_cols, config


# ==========================================
# 3. TAO TRAIN VA VALIDATION
# ==========================================

def prepare_data(df, horizon, config):

    target_col = f"y_h{horizon}"

    train_start = pd.Timestamp(config["train"]["start"])
    train_end = pd.Timestamp(config["train"]["end"])

    val_start = pd.Timestamp(config["val"]["start"])
    val_end = pd.Timestamp(config["val"]["end"])

    # Chi lay origin thuoc Train.
    train = df.loc[
        df["date"].between(train_start, train_end)
    ].copy()

    # Purge: nhan tuong lai phai nam trong Train.
    train = train.loc[
        is_valid_origin(train["date"], horizon).to_numpy()
    ].copy()

    # Chi lay origin thuoc Validation.
    val = df.loc[
        df["date"].between(val_start, val_end)
    ].copy()

    # Purge: nhan tuong lai phai nam trong Validation.
    val = val.loc[
        is_valid_origin(val["date"], horizon).to_numpy()
    ].copy()

    # Chuyen cac cot ve dang so.
    for data in [train, val]:
        data["price_t"] = pd.to_numeric(
            data["price_t"], errors="coerce"
        )
        data[target_col] = pd.to_numeric(
            data[target_col], errors="coerce"
        )

        data["delta_target"] = (
            data[target_col] - data["price_t"]
        )

    # Train chi giu cac dong co Delta va price hop le.
    train = train.loc[
        np.isfinite(train["delta_target"])
        & np.isfinite(train["price_t"])
    ].copy()

    # Validation loai Sunday-copy de danh gia.
    val["is_sunday_copy"] = pd.to_numeric(
        val["is_sunday_copy"], errors="coerce"
    )

    val = val.loc[
        val["is_sunday_copy"].eq(0)
        & np.isfinite(val["delta_target"])
        & np.isfinite(val["price_t"])
    ].copy()

    if train.empty or val.empty:
        raise ValueError(
            f"Khong du du lieu Train/Validation cho H{horizon}"
        )

    print(
        f"H{horizon}: Train={len(train)}, "
        f"Validation={len(val)}"
    )

    return train, val, target_col


# ==========================================
# 4. CHAY HYPERPARAMETER TUNING
# ==========================================

def run_tuning(df, feature_cols, config):

    all_results = []
    predictions_by_trial = {}

    for horizon in HORIZONS:

        print("\n" + "=" * 55)
        print(f"BAT DAU TUNING H{horizon}")
        print("=" * 55)

        train, val, target_col = prepare_data(
            df, horizon, config
        )

        X_train = train[feature_cols]
        y_train = train["delta_target"]

        X_val = val[feature_cols]

        for candidate in CANDIDATES:

            trial = candidate["trial"]
            params = candidate["params"]

            print(f"\nDang chay: {trial}")

            model = XGBRegressor(
                random_state=RANDOM_SEED,
                **params
            )

            model.fit(X_train, y_train)

            # Du bao Delta.
            predicted_delta = model.predict(X_val)

            # Quy doi Delta thanh gia heo.
            predicted_price = (
                val["price_t"].to_numpy() + predicted_delta
            )

            y_true = val[target_col].to_numpy()

            # Tinh metrics tren toan bo Validation.
            metrics = evaluate(y_true, predicted_price)

            all_results.append({
                "model": "xgboost",
                "horizon": horizon,
                "trial": trial,
                "n": metrics["n"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "params": json.dumps(params)
            })

            # Luu predictions de xuat ket qua tot nhat.
            piece = pd.DataFrame({
                "model": "xgboost",
                "config": "tuned",
                "region": val["region"].to_numpy(),
                "horizon": horizon,
                "origin_date": (
                    val["date"].dt.strftime("%Y-%m-%d")
                ),
                "target_date": (
                    val["date"] + pd.Timedelta(days=horizon)
                ).dt.strftime("%Y-%m-%d"),
                "y_true": y_true,
                "y_pred": predicted_price
            })

            predictions_by_trial[(horizon, trial)] = piece

            print(
                f"MAE={metrics['mae']:.2f} | "
                f"RMSE={metrics['rmse']:.2f} | "
                f"MAPE={metrics['mape']:.4f}%"
            )

    return pd.DataFrame(all_results), predictions_by_trial


# ==========================================
# 5. CHON BO THAM SO TOT NHAT
# ==========================================

def select_best(results, predictions_by_trial):

    best_rows = []
    best_predictions = []

    for horizon in HORIZONS:

        subset = results.loc[
            results["horizon"] == horizon
        ].copy()

        # Uu tien MAE thap nhat, neu bang nhau thi RMSE thap hon.
        best = subset.sort_values(
            ["mae", "rmse"]
        ).iloc[0]

        best_rows.append(best.to_dict())

        key = (horizon, best["trial"])
        best_predictions.append(
            predictions_by_trial[key]
        )

        print("\n" + "-" * 45)
        print(f"BO THAM SO TOT NHAT H{horizon}")
        print("Trial:", best["trial"])
        print("MAE:", round(best["mae"], 2))
        print("RMSE:", round(best["rmse"], 2))
        print("MAPE:", round(best["mape"], 4), "%")
        print("Parameters:", best["params"])

    best_table = pd.DataFrame(best_rows)

    best_predictions = pd.concat(
        best_predictions,
        ignore_index=True
    )

    return best_table, best_predictions


# ==========================================
# 6. LUU KET QUA
# ==========================================

def main():

    df, feature_cols, config = load_data()

    results, predictions_by_trial = run_tuning(
        df, feature_cols, config
    )

    best_table, best_predictions = select_best(
        results, predictions_by_trial
    )

    # Metrics theo tung mien cua bo tham so tot nhat.
    quality = df[
        ["date", "region", "is_sunday_copy"]
    ].copy()

    regional_metrics = evaluate_predictions(
        best_predictions,
        quality
    )

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.mkdir(parents=True, exist_ok=True)

    results = results.sort_values(
        ["horizon", "mae"]
    )

    results.to_csv(
        RESULTS_PATH / "xgboost_tuning_metrics.csv",
        index=False,
        float_format="%.6f"
    )

    best_table.to_csv(
        RESULTS_PATH / "xgboost_tuning_best_params.csv",
        index=False
    )

    regional_metrics.to_csv(
        RESULTS_PATH / "xgboost_tuning_best_region_metrics.csv",
        index=False,
        float_format="%.6f"
    )

    best_predictions = best_predictions[
        PREDICTION_COLUMNS
    ]

    best_predictions.to_csv(
        PREDICTIONS_PATH / "xgboost_tuning_validation.csv",
        index=False,
        float_format="%.6f"
    )

    print("\n" + "=" * 55)
    print("HYPERPARAMETER TUNING HOAN THANH")
    print("=" * 55)

    print("\nBang so sanh cac bo tham so:")
    print(
        results[
            ["horizon", "trial", "n", "mae", "rmse", "mape"]
        ].to_string(index=False)
    )

    print("\nBo tham so tot nhat:")
    print(
        best_table[
            ["horizon", "trial", "mae", "rmse", "mape", "params"]
        ].to_string(index=False)
    )

    print("\nDa luu cac file:")
    print(RESULTS_PATH / "xgboost_tuning_metrics.csv")
    print(RESULTS_PATH / "xgboost_tuning_best_params.csv")
    print(RESULTS_PATH / "xgboost_tuning_best_region_metrics.csv")
    print(PREDICTIONS_PATH / "xgboost_tuning_validation.csv")


if __name__ == "__main__":
    main()

