"""P6: sinh bang va hinh cho bao cao muc 2.2, 3.4.3, 4.2, 4.3 tu predictions/ (khong tinh lai mo hinh).

Chay tu thu muc goc repo, sau khi chay `python src/stl_pca.py`:
    python src/p6_report_assets.py
    python src/p6_report_assets.py --extra path/to/xgboost_c6.csv   # them file du bao chua merge vao predictions/

Dau ra:
  results/tables/4_2_stl_pca_bilstm_mae_horizon.csv   MAE/RMSE/MAPE trung binh +- do lech chuan 5 seed
  results/tables/4_3_model_comparison.csv             MAE/RMSE/MAPE moi mo hinh x horizon x mien (+ trung binh 3 mien)
  results/tables/4_3_dm_pvalues.csv                   DM-HLN theo cap mo hinh, horizon, mien
  results/tables/4_2_diagnostic_naive_drift.csv       Naive + do troi trung binh (doi chung)
  results/tables/4_2_stl_pca_bilstm_epochs30_sensitivity.csv  bien the epoch co dinh 30 (neu da chay --no-tune vao predictions/sensitivity)
  results/tables/4_3_dm_matrix_all_ae.csv             ma tran p-value (mien "all", ton that |e|) dang de dan vao bao cao
  results/figures/2_2_stl_decomposition.png, 3_4_3_pca_variance.png, 4_2_stl_pca_bilstm_forecast.png, 4_3_mae_by_model_horizon.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import stl_pca as sp
from dm_test import load_predictions, pairwise_dm
from evaluate import evaluate_predictions

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"
REGIONS = ["Bac", "Trung", "Nam"]
REGION_VN = {"Bac": "Bắc", "Trung": "Trung", "Nam": "Nam"}
HORIZONS = [7, 14, 30]

# Mau gan co dinh theo mo hinh (khong doi khi loc/them mo hinh). Bang mau mac dinh da validate: slot 1-5 + xam cho moc Naive.
MODEL_ORDER = ["naive", "arima", "sarimax", "xgboost", "lstm", "bilstm", "stl_pca_bilstm"]
MODEL_COLOR = {"naive": "#8a8982", "arima": "#eb6834", "sarimax": "#eda100", "xgboost": "#1baf7a",
               "lstm": "#e87ba4", "bilstm": "#008300", "stl_pca_bilstm": "#2a78d6"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b8b7b1")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def model_rank(m):
    return MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER)


def metrics_table(preds: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    m = evaluate_predictions(preds[["model", "config", "region", "horizon", "origin_date", "target_date", "y_true", "y_pred"]], quality)
    avg = m.groupby(["model", "config", "horizon"], as_index=False)[["mae", "rmse", "mape"]].mean()
    avg["region"] = "TB"
    avg["n"] = m.groupby(["model", "config", "horizon"])["n"].sum().to_numpy()
    out = pd.concat([m, avg[m.columns]], ignore_index=True)
    out["_r"] = out.model.map(model_rank)
    return out.sort_values(["horizon", "region", "_r"]).drop(columns="_r").reset_index(drop=True)


def seed_table(quality: pd.DataFrame, seed_dir: Path = ROOT / "predictions" / "seeds") -> pd.DataFrame:
    files = sorted(seed_dir.glob("stl_pca_bilstm_c6_seed*.csv"))
    if not files:
        raise SystemExit(f"Chua co {seed_dir}: chay python src/stl_pca.py truoc")
    rows = []
    for f in files:
        seed = int(f.stem.rsplit("seed", 1)[1])
        m = evaluate_predictions(pd.read_csv(f), quality)
        m["seed"] = seed
        avg = m.groupby("horizon", as_index=False)[["mae", "rmse", "mape"]].mean()
        avg["region"], avg["seed"], avg["n"] = "TB", seed, m.groupby("horizon")["n"].sum().to_numpy()
        rows += [m, avg]
    allm = pd.concat(rows, ignore_index=True)
    g = allm.groupby(["horizon", "region"])
    res = g.agg(n_seeds=("seed", "nunique"), n=("n", "first"),
                mae_mean=("mae", "mean"), mae_std=("mae", "std"), rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
                mape_mean=("mape", "mean"), mape_std=("mape", "std")).reset_index()
    return res


def drift_diagnostic(cfg: dict) -> pd.DataFrame:
    """Naive + do troi trung binh (mean Delta cua cac dong da lo nhan, walk-forward nhu mo hinh): kiem tra phan
    sai so cua STL-PCA-BiLSTM thuc su den tu hoc chuoi hay chi tu so hang troi trung binh."""
    from walkforward import available_training_rows, get_walkforward_windows
    df = sp.load_data()
    rows = []
    for h in HORIZONS:
        parts = []
        for w in get_walkforward_windows(cfg, h):
            tr = available_training_rows(df, w["retrain_date"], h)
            mu = float((tr[f"y_h{h}"] - tr["price_t"]).dropna().mean())
            te = sp.eval_rows(df, h, w["prediction_start"], cfg["test"]["end"])
            te = te[(te["date"] >= w["prediction_start"]) & (te["date"] <= w["prediction_end"])]
            parts.append(te.assign(mu=mu))
        te = pd.concat(parts)
        for region, g in te.groupby("region"):
            y, p0 = g[f"y_h{h}"].to_numpy(), g["price_t"].to_numpy()
            rows.append({"horizon": h, "region": region, "n": len(g),
                         "mae_naive": np.abs(y - p0).mean(), "mae_naive_drift": np.abs(y - p0 - g["mu"].to_numpy()).mean(),
                         "mean_mu": g["mu"].mean()})
    out = pd.DataFrame(rows)
    avg = out.groupby("horizon", as_index=False)[["n", "mae_naive", "mae_naive_drift", "mean_mu"]].agg(
        {"n": "sum", "mae_naive": "mean", "mae_naive_drift": "mean", "mean_mu": "mean"})
    avg["region"] = "TB"
    return pd.concat([out, avg[out.columns]], ignore_index=True)


def dm_matrix(res: pd.DataFrame) -> pd.DataFrame:
    s = res[(res.region == "all") & (res.loss == "ae")].copy()
    s["pair"] = s.model_a + " vs " + s.model_b
    cols = ["pair", "horizon", "n", "mean_diff", "dm_stat", "p_value", "dm_stat_bartlett", "p_value_bartlett", "better"]
    return s[cols].sort_values(["horizon", "pair"]).reset_index(drop=True)


# ------------------------------------------------------------------ hinh
def fig_stl(df_stl: pd.DataFrame):
    g = df_stl[df_stl.region == "Bac"].set_index("date")
    g = g[g.trend.notna()]
    fig, axes = plt.subplots(4, 1, figsize=(9, 7.5), sharex=True)
    series = [("price_ffill", "Giá heo hơi (đồng/kg)"), ("trend", "Trend"), ("seasonal", "Seasonal (chu kỳ 7 ngày)"),
              ("resid", "Residual")]
    for ax, (c, title) in zip(axes, series):
        ax.plot(g.index, g[c], color="#2a78d6", lw=1.3)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        style(ax)
    for x0, x1, lab in [("2025-07-02", "2026-03-31", "Train"), ("2026-04-01", "2026-06-30", "Val"), ("2026-07-01", "2026-10-30", "Test")]:
        axes[0].axvspan(pd.Timestamp(x0), pd.Timestamp(x1), color="#000000", alpha=0.04 if lab != "Val" else 0.0)
        axes[0].text(pd.Timestamp(x0) + pd.Timedelta(days=3), axes[0].get_ylim()[1], lab, va="top", fontsize=8, color=INK2)
    fig.suptitle("STL cuốn chiếu (chỉ dùng dữ liệu đến ngày t), miền Bắc", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(FIGS / "2_2_stl_decomposition.png", dpi=160)
    plt.close(fig)


def fig_pca(df_stl: pd.DataFrame, exog: list[str]):
    from sklearn.preprocessing import StandardScaler
    tr = df_stl.loc[df_stl.split == "train", exog].dropna().to_numpy(float)
    z = np.clip(StandardScaler().fit_transform(tr), -sp.CLIP, sp.CLIP)
    ev = PCA(svd_solver="full").fit(z).explained_variance_ratio_
    cum = np.cumsum(ev)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    k = np.arange(1, len(ev) + 1)
    ax.bar(k, ev * 100, color="#9db9e4", width=0.7, label="Từng thành phần")
    ax.plot(k, cum * 100, color="#2a78d6", lw=2, marker="o", ms=4, label="Tích lũy")
    for thr in (90, 95):
        ax.axhline(thr, color=INK2, lw=0.8, ls="--")
        n = int(np.argmax(cum * 100 >= thr)) + 1
        ax.text(16.4, thr - 4.5, f"{thr}% → {n} PC", ha="right", fontsize=8, color=INK2)
    ax.set_xlim(0.4, 16.6)
    ax.set_xlabel("Số thành phần chính (PC)", color=INK2)
    ax.set_ylabel("% phương sai", color=INK2)
    ax.set_title(f"PCA trên {len(exog)} biến ngoại sinh (chỉ fit Train)", loc="left", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="center right")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "3_4_3_pca_variance.png", dpi=160)
    plt.close(fig)


def fig_forecast(preds: pd.DataFrame):
    fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True)
    for i, h in enumerate(HORIZONS):
        for j, r in enumerate(REGIONS):
            ax = axes[i, j]
            d = preds[(preds.horizon == h) & (preds.region == r)]
            truth = d[d.model == "stl_pca_bilstm"].sort_values("target_date")
            ax.plot(truth.target_date, truth.y_true / 1000, color=INK, lw=1.4)
            nv = d[d.model == "naive"].sort_values("target_date")
            if len(nv):
                ax.plot(nv.target_date, nv.y_pred / 1000, color=MODEL_COLOR["naive"], lw=1.1)
            ax.plot(truth.target_date, truth.y_pred / 1000, color=MODEL_COLOR["stl_pca_bilstm"], lw=1.6)
            ax.set_title(f"{REGION_VN[r]} – horizon {h} ngày", loc="left", fontsize=9.5, color=INK)
            if j == 0:
                ax.set_ylabel("nghìn đồng/kg", color=INK2, fontsize=9)
            style(ax)
            ax.tick_params(axis="x", labelrotation=30)
            ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%Y"))
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=INK, lw=1.4, label="Giá thực tế"),
               Line2D([], [], color=MODEL_COLOR["stl_pca_bilstm"], lw=1.6, label="STL-PCA-BiLSTM (TB 5 seed)"),
               Line2D([], [], color=MODEL_COLOR["naive"], lw=1.1, label="Naive")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGS / "4_2_stl_pca_bilstm_forecast.png", dpi=160)
    plt.close(fig)


def fig_mae(metrics: pd.DataFrame, seeds: pd.DataFrame):
    avg = metrics[metrics.region == "TB"]
    models = sorted(avg.model.unique(), key=model_rank)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    w = 0.8 / len(models)
    for k, m in enumerate(models):
        v = [avg[(avg.model == m) & (avg.horizon == h)].mae.iloc[0] for h in HORIZONS]
        x = np.arange(len(HORIZONS)) + (k - (len(models) - 1) / 2) * w
        bars = ax.bar(x, v, width=w * 0.88, color=MODEL_COLOR.get(m, "#999"), label=m)
        if m == "stl_pca_bilstm":
            s = seeds[seeds.region == "TB"].set_index("horizon").loc[HORIZONS]
            ax.errorbar(x, s.mae_mean, yerr=s.mae_std, fmt="none", ecolor=INK, capsize=3, lw=1)
        pad = s.mae_std.to_numpy() if m == "stl_pca_bilstm" else np.zeros(len(v))
        for b, val, e in zip(bars, v, pad):
            ax.text(b.get_x() + b.get_width() / 2, val + e + 40, f"{val:,.0f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(range(len(HORIZONS)), [f"{h} ngày" for h in HORIZONS])
    ax.set_ylabel("MAE trung bình 3 miền (đồng/kg)", color=INK2)
    ax.set_title("MAE theo mô hình và horizon (Test 01/07–30/10/2026)", loc="left", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=len(models), loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "4_3_mae_by_model_horizon.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[], help="file du bao them (vd. xgboost_c6.csv tu nhanh P4)")
    args = ap.parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in (ROOT / "predictions").glob("*.csv") if p.is_file())
    files += [Path(p) for p in args.extra if Path(p).name not in {f.name for f in files}]
    preds = load_predictions(files)
    quality = sp.load_data()[["date", "region", "is_sunday_copy"]]
    print("Mo hinh:", ", ".join(sorted(preds.model.unique(), key=model_rank)))

    metrics = metrics_table(preds, quality)
    metrics.to_csv(TABLES / "4_3_model_comparison.csv", index=False, float_format="%.4f")
    seeds = seed_table(quality)
    seeds.to_csv(TABLES / "4_2_stl_pca_bilstm_mae_horizon.csv", index=False, float_format="%.4f")
    sens_dir = ROOT / "predictions" / "sensitivity" / "seeds"
    if sens_dir.exists():
        seed_table(quality, sens_dir).to_csv(TABLES / "4_2_stl_pca_bilstm_epochs30_sensitivity.csv", index=False, float_format="%.4f")
    drift_diagnostic(sp.load_config(sp.CONFIG_PATH)).to_csv(TABLES / "4_2_diagnostic_naive_drift.csv", index=False, float_format="%.4f")
    dm = pairwise_dm(preds)
    dm.to_csv(TABLES / "4_3_dm_pvalues.csv", index=False, float_format="%.6f")
    dm_matrix(dm).to_csv(TABLES / "4_3_dm_matrix_all_ae.csv", index=False, float_format="%.6f")

    df_stl = sp.add_stl_features(sp.load_data())
    fig_stl(df_stl)
    fig_pca(df_stl, sp.exog_columns())
    fig_forecast(preds)
    fig_mae(metrics, seeds)
    print(metrics[metrics.region == "TB"].to_string(index=False))
    print(seeds[seeds.region == "TB"].to_string(index=False))
    print("Da ghi bang vao", TABLES, "va hinh vao", FIGS)


if __name__ == "__main__":
    main()
