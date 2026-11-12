
from pathlib import Path

import pandas as pd

from evaluate import evaluate_predictions
from walkforward import load_config
from xgb_ablation import run_one_config


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "features" / "data_v1.csv"
FEATURE_GROUPS_PATH = ROOT / "data" / "features" / "feature_groups.csv"
CONFIG_PATH = ROOT / "data" / "split" / "split_config.json"

PREDICTIONS_PATH = ROOT / "predictions" / "xgboost_feed_lag.csv"
METRICS_PATH = ROOT / "results" / "tables" / "xgboost_feed_lag_metrics.csv"


def main():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    feature_groups = pd.read_csv(FEATURE_GROUPS_PATH)

    e0_features = feature_groups.loc[
        feature_groups["group"].eq("E0"), "feature"
    ].tolist()

    e2_features = feature_groups.loc[
        feature_groups["group"].eq("E2"), "feature"
    ].tolist()

    current_features = [
        f for f in e2_features
        if not any(f.endswith(f"_lag{lag}") for lag in (30, 60, 90))
    ]

    configs = {
        "current": e0_features + current_features,
    }

    for lag in (30, 60, 90):
        lag_features = [
            f for f in e2_features if f.endswith(f"_lag{lag}")
        ]

        configs[f"lag{lag}"] = (
            e0_features + current_features + lag_features
        )

    # Kiem tra cau hinh truoc khi chay
    for name, features in configs.items():
        missing = set(features) - set(df.columns)

        if missing:
            raise ValueError(
                f"{name}: dataset thieu features {sorted(missing)}"
            )

        print(f"{name}: {len(features)} features")

    if len(current_features) != 4:
        raise ValueError("So feature gia hien tai khong dung 4.")

    for lag in (30, 60, 90):
        count = sum(
            f.endswith(f"_lag{lag}") for f in e2_features
        )
        if count != 4:
            raise ValueError(f"So feature lag{lag} khong dung 4.")

    split_config = load_config(CONFIG_PATH)

    all_predictions = []

    for name, features in configs.items():
        print("\n" + "=" * 50)
        print(f"BAT DAU THI NGHIEM: {name.upper()}")
        print("=" * 50)

        predictions = run_one_config(
            df,
            features,
            name,
            split_config
        )

        all_predictions.append(predictions)

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

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

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

    print("\n========== FEED LAG EXPERIMENT HOAN THANH ==========")
    print("Tong predictions:", len(final_predictions))
    print("\nMetrics:")
    print(metrics.to_string(index=False))

    print("\nDa luu predictions:", PREDICTIONS_PATH)
    print("Da luu metrics:", METRICS_PATH)


if __name__ == "__main__":
    main()

