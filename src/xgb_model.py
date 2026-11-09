
from pathlib import Path

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
CONFIG_PATH = ROOT / "data" / "split" / "split_config.json"

PREDICTIONS_PATH = ROOT / "predictions" / "xgboost_c6.csv"
METRICS_PATH = ROOT / "results" / "tables" / "xgboost_c6_metrics.csv"

HORIZONS = (7, 14, 30)
MODEL_NAME = "xgboost"
CONFIG_NAME = "c6"
RANDOM_SEED = 42


# ==========================================
# 2. DOC DU LIEU
# ==========================================

def load_data():
    """Doc dataset va danh sach features thuoc C6."""

    df = pd.read_csv(DATA_PATH)
    feature_groups = pd.read_csv(FEATURE_GROUPS_PATH)

    df["date"] = pd.to_datetime(df["date"])

    feature_cols = feature_groups.loc[
        feature_groups["c6"] == 1, "feature"
    ].tolist()

    required_cols = {
        "date", "region", "price_t", "is_sunday_copy",
        "y_h7", "y_h14", "y_h30"
    }

    missing_cols = required_cols - set(df.columns)
    missing_features = set(feature_cols) - set(df.columns)

    if missing_cols:
        raise ValueError(f"Dataset thieu cot: {sorted(missing_cols)}")

    if missing_features:
        raise ValueError(
            f"Dataset thieu features: {sorted(missing_features)}"
        )

    if len(feature_cols) != 59:
        raise ValueError(
            f"So features C6 dang la {len(feature_cols)}, khong phai 59"
        )

    print("Da doc dataset thanh cong!")
    print("So dong:", len(df))
    print("So features C6:", len(feature_cols))

    return df, feature_cols


# ==========================================
# 3. HUAN LUYEN VA DU BAO XGBOOST
# ==========================================

def run_xgboost(data, feature_cols, config):
    """Du bao gia heo theo Delta voi monthly walk-forward."""

    df = data.copy()
    df["price_t"] = pd.to_numeric(df["price_t"], errors="coerce")

    prediction_parts = []

    for horizon in HORIZONS:
        target_col = f"y_h{horizon}"

        print(f"\n========== HORIZON H{horizon} ==========")

        # Moi horizon co mot bo model rieng.
        # Model duoc huan luyen lai vao dau moi thang trong Test.
        windows = get_walkforward_windows(config, horizon)

        for window in windows:
            retrain_date = pd.Timestamp(window["retrain_date"])
            prediction_start = pd.Timestamp(window["prediction_start"])
            prediction_end = pd.Timestamp(window["prediction_end"])

            # ----- A. Chuan bi du lieu huan luyen -----

            train = available_training_rows(
                df,
                retrain_date,
                horizon
            )

            train[target_col] = pd.to_numeric(
                train[target_col], errors="coerce"
            )

            # Delta = gia thuc te tuong lai - gia hien tai
            train["delta_target"] = train[target_col] - train["price_t"]

            # Chi giu cac dong co target Delta hop le.
            # Cac NaN trong features duoc giu lai vi XGBoost xu ly duoc.
            train = train.loc[
                np.isfinite(train["delta_target"])
                & np.isfinite(train["price_t"])
            ].copy()

            if train.empty:
                raise ValueError(
                    f"Khong co du lieu train cho H{horizon} "
                    f"tai ngay {retrain_date.date()}"
                )

            X_train = train[feature_cols]
            y_train = train["delta_target"]

            # ----- B. Huan luyen model -----

            model = XGBRegressor(random_state=RANDOM_SEED)
            model.fit(X_train, y_train)

            # ----- C. Chuan bi du lieu du bao trong thang -----

            test = df.loc[
                (df["date"] >= prediction_start)
                & (df["date"] <= prediction_end)
            ].copy()

            # Chi lay diem du bao co target nam trong tap Test.
            valid_mask = is_valid_origin(
                test["date"], horizon
            ).to_numpy()

            test = test.loc[valid_mask].copy()

            # Loai ngay gia bi sao chep tu Chu nhat khi danh gia.
            test = test.loc[
                test["is_sunday_copy"].eq(0)
            ].copy()

            test[target_col] = pd.to_numeric(
                test[target_col], errors="coerce"
            )

            # Can co price_t va gia thuc te de tao ban ghi danh gia.
            test = test.loc[
                np.isfinite(test["price_t"])
                & np.isfinite(test[target_col])
            ].copy()

            if test.empty:
                print(
                    f"Khong co dong du bao hop le trong "
                    f"thang {retrain_date.strftime('%Y-%m')}"
                )
                continue

            # ----- D. Du bao Delta va quy doi ve gia -----

            X_test = test[feature_cols]

            predicted_delta = model.predict(X_test)

            # Gia du bao = gia hien tai + Delta du bao
            predicted_price = test["price_t"].to_numpy() + predicted_delta

            # Tao ket qua theo dung schema chung cua project.
            # y_true va y_pred deu la GIA, khong phai Delta.
            piece = pd.DataFrame({
                "model": MODEL_NAME,
                "config": CONFIG_NAME,
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
                f"{len(piece)} dong du bao"
            )

    if not prediction_parts:
        raise ValueError("Khong tao duoc ket qua du bao nao.")

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
# 4. LUU DU BAO VA TINH METRICS
# ==========================================

def main():
    data, feature_cols = load_data()
    config = load_config(CONFIG_PATH)

    predictions = run_xgboost(data, feature_cols, config)

    # Tinh MAE, RMSE, MAPE theo tung mien va horizon.
    quality = data[["date", "region", "is_sunday_copy"]].copy()

    metrics = evaluate_predictions(
        predictions,
        quality
    )

    PREDICTIONS_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    METRICS_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

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

    print("\n========== HOAN THANH ==========")
    print("So dong du bao:", len(predictions))
    print("\nMetrics:")
    print(metrics.to_string(index=False))
    print("\nDa luu du bao tai:", PREDICTIONS_PATH)
    print("Da luu metrics tai:", METRICS_PATH)


if __name__ == "__main__":
    main()

