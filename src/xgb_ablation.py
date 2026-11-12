
from pathlib import Path
import json

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from evaluate import PREDICTION_COLUMNS, evaluate_predictions
from split import is_valid_origin
from walkforward import load_config, available_training_rows, get_walkforward_windows


# ==========================================
# 1. CAU HINH
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "features" / "data_v1.csv"
FEATURE_GROUPS_PATH = ROOT / "data" / "features" / "feature_groups.csv"
ABLATION_PATH = ROOT / "data" / "features" / "ablation_config.json"
CONFIG_PATH = ROOT / "data" / "split" / "split_config.json"

PREDICTIONS_PATH = ROOT / "predictions" / "xgboost_ablation.csv"
METRICS_PATH = ROOT / "results" / "tables" / "xgboost_ablation_metrics.csv"

HORIZONS = (7, 14, 30)
MODEL_NAME = "xgboost"
RANDOM_SEED = 42


# ==========================================
# 2. DOC DU LIEU VA CAU HINH
# ==========================================

def load_data_and_configs():
    df = pd.read_csv(DATA_PATH)
    feature_groups = pd.read_csv(FEATURE_GROUPS_PATH)

    with open(ABLATION_PATH, "r", encoding="utf-8") as f:
        ablation = json.load(f)

    df["date"] = pd.to_datetime(df["date"])

    required_cols = {
        "date", "region", "price_t", "is_sunday_copy",
        "y_h7", "y_h14", "y_h30"
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset thieu cot: {sorted(missing)}")

    configs = {}

    for name, groups in ablation["configs"].items():
        selected = feature_groups.loc[
            feature_groups["group"].isin(groups), "feature"
        ].tolist()

        # Dam bao thu tu feature dung theo bang feature_groups
        selected = [
            feature for feature in feature_groups["feature"]
            if feature in selected
        ]

        expected_count = int(feature_groups[name].sum())

        if len(selected) != expected_count:
            raise ValueError(
                f"{name}: so feature khong khop "
                f"({len(selected)} != {expected_count})"
            )

        missing_features = set(selected) - set(df.columns)
        if missing_features:
            raise ValueError(
                f"{name}: dataset thieu features {sorted(missing_features)}"
            )

        configs[name] = selected

    print("Da doc dataset va cau hinh thanh cong!")
    print("So dong:", len(df))

    for name, features in configs.items():
        print(f"{name}: {len(features)} features")

    return df, configs


# ==========================================
# 3. CHAY MOT CAU HINH ABLATION
# ==========================================

def run_one_config(df, feature_cols, config_name, split_config):

    prediction_parts = []

    for horizon in HORIZONS:
        target_col = f"y_h{horizon}"

        print(f"\n--- {config_name.upper()} | H{horizon} ---")

        windows = get_walkforward_windows(
            split_config, horizon
        )

        for window in windows:
            retrain_date = pd.Timestamp(window["retrain_date"])
            prediction_start = pd.Timestamp(window["prediction_start"])
            prediction_end = pd.Timestamp(window["prediction_end"])

            # A. Du lieu huan luyen
            train = available_training_rows(
                df, retrain_date, horizon
            ).copy()

            train[target_col] = pd.to_numeric(
                train[target_col], errors="coerce"
            )

            train["price_t"] = pd.to_numeric(
                train["price_t"], errors="coerce"
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
                    f"Khong co train data: {config_name}, H{horizon}, "
                    f"{retrain_date.date()}"
                )

            X_train = train[feature_cols]
            y_train = train["delta_target"]

            # B. Huan luyen XGBoost
            model = XGBRegressor(random_state=RANDOM_SEED)
            model.fit(X_train, y_train)

            # C. Du lieu du bao
            test = df.loc[
                (df["date"] >= prediction_start)
                & (df["date"] <= prediction_end)
            ].copy()

            valid_mask = is_valid_origin(
                test["date"], horizon
            ).to_numpy()

            test = test.loc[valid_mask].copy()

            test = test.loc[
                test["is_sunday_copy"].eq(0)
            ].copy()

            test[target_col] = pd.to_numeric(
                test[target_col], errors="coerce"
            )

            test["price_t"] = pd.to_numeric(
                test["price_t"], errors="coerce"
            )

            test = test.loc[
                np.isfinite(test["price_t"])
                & np.isfinite(test[target_col])
            ].copy()

            if test.empty:
                print(
                    f"Khong co du bao hop le tai "
                    f"{retrain_date.strftime('%Y-%m')}"
                )
                continue

            # D. Du bao Delta va quy doi ve gia
            predicted_delta = model.predict(test[feature_cols])

            predicted_price = (
                test["price_t"].to_numpy() + predicted_delta
            )

            piece = pd.DataFrame({
                "model": MODEL_NAME,
                "config": config_name,
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
                f"{len(train)} train, {len(piece)} predictions"
            )

    if not prediction_parts:
        raise ValueError(f"{config_name}: khong tao duoc predictions")

    predictions = pd.concat(
        prediction_parts, ignore_index=True
    )

    return predictions[PREDICTION_COLUMNS]


# ==========================================
# 4. CHAY TAT CA C1-C6 VA LUU KET QUA
# ==========================================

def main():

    df, configs = load_data_and_configs()
    split_config = load_config(CONFIG_PATH)

    all_predictions = []

    for config_name, feature_cols in configs.items():

        print("\n" + "=" * 50)
        print(f"BAT DAU ABLATION: {config_name.upper()}")
        print("=" * 50)

        predictions = run_one_config(
            df,
            feature_cols,
            config_name,
            split_config
        )

        all_predictions.append(predictions)

    # Gop ket qua C1-C6
    final_predictions = pd.concat(
        all_predictions, ignore_index=True
    )

    final_predictions = final_predictions.sort_values(
        ["config", "region", "horizon", "origin_date"]
    ).reset_index(drop=True)

    quality = df[
        ["date", "region", "is_sunday_copy"]
    ].copy()

    metrics = evaluate_predictions(
        final_predictions,
        quality
    )

    PREDICTIONS_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    METRICS_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    final_predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
        float_format="%.6f"
    )

    metrics.to_csv(
        METRICS_PATH,
        index=False,
        float_format="%.6f"
    )

    print("\n========== ABLATION HOAN THANH ==========")
    print("Tong so predictions:", len(final_predictions))
    print("\nMetrics:")
    print(metrics.to_string(index=False))

    print("\nDa luu predictions:", PREDICTIONS_PATH)
    print("Da luu metrics:", METRICS_PATH)


if __name__ == "__main__":
    main()
