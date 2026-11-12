
from pathlib import Path
import json
import ast

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

from walkforward import (
    load_config,
    available_training_rows,
    get_walkforward_windows
)


# ==========================================
# 1. DUONG DAN VA CAU HINH
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "features" / "data_v1.csv"
FEATURE_GROUPS_PATH = ROOT / "data" / "features" / "feature_groups.csv"
CONFIG_PATH = ROOT / "data" / "split" / "split_config.json"

BEST_PARAMS_PATH = (
    ROOT / "results" / "tables" / "xgboost_tuning_best_params.csv"
)

OUTPUT_DIR = ROOT / "results" / "figures"

# Chi giai thich H7 va H30
HORIZONS = (7, 30)

RANDOM_SEED = 42
MAX_SAMPLES = 1000
MAX_DISPLAY = 15


# ==========================================
# 2. DOC DU LIEU
# ==========================================

def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    feature_groups = pd.read_csv(FEATURE_GROUPS_PATH)

    feature_cols = feature_groups.loc[
        feature_groups["c6"] == 1, "feature"
    ].tolist()

    if len(feature_cols) != 59:
        raise ValueError(
            f"C6 phai co 59 features, hien tai co {len(feature_cols)}"
        )

    print("Da doc dataset thanh cong!")
    print("So dong:", len(df))
    print("So features C6:", len(feature_cols))

    return df, feature_cols


# ==========================================
# 3. DOC THAM SO DA CHON TU VALIDATION
# ==========================================

def load_best_params():
    params_df = pd.read_csv(BEST_PARAMS_PATH)

    if "horizon" not in params_df.columns:
        raise ValueError("File best_params thieu cot horizon.")

    # Ho tro mot trong hai ten cot
    params_col = None

    for col in ["params", "best_params"]:
        if col in params_df.columns:
            params_col = col
            break

    if params_col is None:
        raise ValueError(
            "Khong tim thay cot params hoac best_params."
        )

    best_params = {}

    for horizon in HORIZONS:
        row = params_df.loc[
            params_df["horizon"] == horizon
        ]

        if row.empty:
            raise ValueError(
                f"Khong tim thay tham so cho H{horizon}."
            )

        raw_params = row.iloc[0][params_col]

        if isinstance(raw_params, dict):
            params = raw_params
        else:
            try:
                params = json.loads(raw_params)
            except (json.JSONDecodeError, TypeError):
                params = ast.literal_eval(raw_params)

        best_params[horizon] = params

    print("\nDa doc tham so tu Validation:")
    for horizon, params in best_params.items():
        print(f"H{horizon}: {params}")

    return best_params


# ==========================================
# 4. HUAN LUYEN MODEL TAI LAN RETRAIN CUOI
# ==========================================

def train_final_model(df, feature_cols, config, horizon, params):

    target_col = f"y_h{horizon}"

    windows = get_walkforward_windows(config, horizon)

    if not windows:
        raise ValueError(f"Khong co walk-forward window cho H{horizon}.")

    # Lay lan retrain cuoi cung: 01/10/2026
    last_window = windows[-1]

    retrain_date = pd.Timestamp(
        last_window["retrain_date"]
    )

    train = available_training_rows(
        df,
        retrain_date,
        horizon
    ).copy()

    train[target_col] = pd.to_numeric(
        train[target_col],
        errors="coerce"
    )

    train["price_t"] = pd.to_numeric(
        train["price_t"],
        errors="coerce"
    )

    # Muc tieu la Delta = gia tuong lai - gia hien tai
    train["delta_target"] = (
        train[target_col] - train["price_t"]
    )

    train = train.loc[
        np.isfinite(train["delta_target"])
        & np.isfinite(train["price_t"])
    ].copy()

    if train.empty:
        raise ValueError(f"Khong co du lieu train cho H{horizon}.")

    X_train = train[feature_cols]
    y_train = train["delta_target"]

    model = XGBRegressor(
        random_state=RANDOM_SEED,
        **params
    )

    model.fit(X_train, y_train)

    print(f"\nH{horizon}:")
    print("Ngay retrain:", retrain_date.date())
    print("So dong train:", len(train))

    return model, X_train


# ==========================================
# 5. TINH SHAP VA VE BIEU DO
# ==========================================

def create_shap_plot(model, X_train, horizon):

    # Lay toi da 1000 dong de tinh SHAP
    sample_size = min(MAX_SAMPLES, len(X_train))

    X_sample = X_train.sample(
        n=sample_size,
        random_state=RANDOM_SEED
    )

    print(
        f"Dang tinh SHAP H{horizon} "
        f"voi {sample_size} dong..."
    )

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_sample)

        # Xuat bang SHAP Feature Importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": mean_abs_shap
    })

    importance_df = importance_df.sort_values(
        by="mean_abs_shap",
        ascending=False
    )

    importance_df["rank"] = range(1, len(importance_df) + 1)

    importance_df = importance_df[
        ["rank", "feature", "mean_abs_shap"]
    ]

    importance_path = (
        OUTPUT_DIR / f"shap_xgboost_h{horizon}_importance.csv"
    )

    importance_df.to_csv(
        importance_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nTop 10 SHAP H{horizon}:")
    print(importance_df.head(10).to_string(index=False))
    print("Da luu:", importance_path)

    # Summary plot: muc do va chieu tac dong cua features
    plt.figure(figsize=(12, 8))

    shap.summary_plot(
        shap_values,
        X_sample,
        max_display=MAX_DISPLAY,
        show=False
    )

    plt.title(
        f"SHAP Summary Plot - XGBoost H{horizon}",
        fontsize=14
    )

    plt.tight_layout()

    summary_path = (
        OUTPUT_DIR / f"shap_xgboost_h{horizon}.png"
    )

    plt.savefig(
        summary_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Bar plot: xep hang muc do quan trong trung binh
    plt.figure(figsize=(12, 8))

    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        max_display=MAX_DISPLAY,
        show=False
    )

    plt.title(
        f"SHAP Feature Importance - XGBoost H{horizon}",
        fontsize=14
    )

    plt.tight_layout()

    bar_path = (
        OUTPUT_DIR / f"shap_xgboost_h{horizon}_bar.png"
    )

    plt.savefig(
        bar_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Da luu:", summary_path)
    print("Da luu:", bar_path)


# ==========================================
# 6. MAIN
# ==========================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df, feature_cols = load_data()

    config = load_config(CONFIG_PATH)

    best_params = load_best_params()

    for horizon in HORIZONS:

        print(
            f"\n========== SHAP XGBOOST H{horizon} =========="
        )

        model, X_train = train_final_model(
            df,
            feature_cols,
            config,
            horizon,
            best_params[horizon]
        )

        create_shap_plot(
            model,
            X_train,
            horizon
        )

    print("\n========== SHAP HOAN THANH ==========")
    print("Thu muc luu bieu do:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

