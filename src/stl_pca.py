"""P6: STL-PCA-BiLSTM (STL chu ky 7 ngay + PCA bien ngoai sinh + BiLSTM), du bao truc tiep.

Chay tu thu muc goc repo:
    python src/stl_pca.py --horizons 7 30 --seeds 42 --no-tune      # chay nhanh dau-cuoi (giai doan 2)
    python src/stl_pca.py                                           # tune tren Val + 5 seed + 3 horizon

Y tuong (ap dung tu Liu et al., 2025, PLOS ONE; code viet moi):
  1. STL (period = 7) tach gia thanh trend / seasonal / residual.
  2. PCA nen cac bien ngoai sinh (lich/Tet, thuc an, cung-cau, dich benh, thoi tiet/vi mo) thanh vai thanh phan chinh.
  3. BiLSTM nhan chuoi [trend, seasonal, residual, PC...] cua L ngay gan nhat, du bao
     Delta = y_h - price_t (nhu P4), roi quy doi ve gia: y_pred = price_t + Delta.

Chong ro ri (xem assert_no_leak va tests/test_p6_stl_pca.py):
  * STL cuon chieu: tai ngay t chi phan ra cac quan sat <= t (cua so toi da STL_MAX_WINDOW ngay), lay gia tri cuoi.
    STL tren ca chuoi se dung gia tuong lai cho diem t -> khong dung.
  * Scaler va PCA chi fit tren Train; Val/Test chi transform.
  * Tham so (cua so L, hidden, dropout, lr, nguong phuong sai PCA, so epoch) chon tren Val, khong nhin Test.
  * Walk-forward: dau moi thang trong Test (01/07, 01/08, 01/09, 01/10/2026) huan luyen lai tren cac dong da lo nhan
    (t + h < ngay retrain), cua so mo rong.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL
from torch import nn

from evaluate import PREDICTION_COLUMNS, evaluate_predictions
from split import is_valid_origin
from walkforward import available_training_rows, get_walkforward_windows, load_config

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "features" / "data_v1.csv"
GROUPS_PATH = ROOT / "data" / "features" / "feature_groups.csv"
CONFIG_PATH = ROOT / "data" / "split" / "split_config.json"
PRED_DIR = ROOT / "predictions"
TABLE_DIR = ROOT / "results" / "tables"

MODEL_NAME = "stl_pca_bilstm"
CONFIG_NAME = "c6"
HORIZONS = (7, 14, 30)
REGIONS = ("Bac", "Trung", "Nam")
SEEDS = (42, 43, 44, 45, 46)
TUNE_SEEDS = (42, 43)

STL_PERIOD = 7
STL_MIN_OBS = 28      # it nhat 4 tuan moi phan ra duoc
STL_MAX_WINDOW = 84   # 12 tuan gan nhat
COMP = ["trend", "seasonal", "resid"]
CLIP = 5.0            # cat gia tri da chuan hoa de bien ngoai Train khong no tung

DEFAULT_PARAMS = {"window": 14, "hidden": 32, "dropout": 0.2, "lr": 3e-3, "pca_var": 0.95, "epochs": 30}
GRID = {
    "window": [14, 28],
    "hidden": [16, 32, 64],
    "dropout": [0.1, 0.3],
    "lr": [1e-3, 3e-3],
    "pca_var": [0.90, 0.95],
}
MAX_EPOCHS = 60
BATCH = 64


# ------------------------------------------------------------------ du lieu
def load_data(data_path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["region", "date"]).reset_index(drop=True)


def exog_columns(groups_path=GROUPS_PATH, drop_lag90=True) -> list[str]:
    """Bien ngoai sinh dua vao PCA: nhom E1..E5 cua c6. E0 (gia) da duoc STL thay the.

    Cot *_lag90 thieu 12% Train (nhung ngay dau) -> bo cot (README muc 7.5: thong nhat 1 cach, khong chen gia tri).
    """
    g = pd.read_csv(groups_path)
    cols = g.loc[(g["c6"] == 1) & g["group"].isin(["E1", "E2", "E3", "E4", "E5"]), "feature"].tolist()
    if drop_lag90:
        cols = [c for c in cols if not c.endswith("_lag90")]
    return cols


# ------------------------------------------------------------------ STL cuon chieu
def causal_stl_last(values: np.ndarray, period=STL_PERIOD, max_window=STL_MAX_WINDOW) -> tuple[float, float, float]:
    """STL tren cac quan sat da cho (toi da max_window cuoi); tra ve (trend, seasonal, resid) tai diem cuoi."""
    y = np.asarray(values[-max_window:], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = STL(y, period=period, robust=True).fit()
    return float(r.trend[-1]), float(r.seasonal[-1]), float(r.resid[-1])


def rolling_stl(series: pd.Series, period=STL_PERIOD, min_obs=STL_MIN_OBS, max_window=STL_MAX_WINDOW) -> pd.DataFrame:
    """Thanh phan STL tai moi ngay t, chi dung series[:t+1]. Ngay dau (< min_obs quan sat) la NaN."""
    series = series.ffill()  # o trong giua chuoi (3 ngay Tet 20-22/02/2026) lap bang gia ngay truoc: chi dung qua khu
    s = series.dropna()
    if s.empty:
        raise ValueError("Chuoi gia rong")
    vals = s.to_numpy(dtype=float)
    out = np.full((len(series), 3), np.nan)
    start = series.index.get_loc(s.index[0])
    for k in range(min_obs - 1, len(vals)):
        out[start + k] = causal_stl_last(vals[: k + 1], period, max_window)
    return pd.DataFrame(out, index=series.index, columns=COMP)


def add_stl_features(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, g in df.groupby("region", sort=False):
        g = g.sort_values("date").copy()
        comp = rolling_stl(g.set_index("date")["price_ffill"])
        for c in COMP:
            g[c] = comp[c].to_numpy()
        parts.append(g)
    return pd.concat(parts).sort_values(["region", "date"]).reset_index(drop=True)


# ------------------------------------------------------------------ PCA (fit Train)
class ExogPCA:
    """Chuan hoa -> cat -> PCA (whiten). Chi fit tren dong Train."""

    def __init__(self, cols: list[str], var_ratio: float):
        self.cols, self.var_ratio = cols, var_ratio

    def fit(self, df: pd.DataFrame) -> "ExogPCA":
        tr = df.loc[df["split"] == "train", self.cols].dropna()  # vai dong dau thieu lag60: bo khi fit
        if len(tr) < 0.9 * (df["split"] == "train").sum():
            raise ValueError("Qua nhieu dong Train thieu bien ngoai sinh")
        self.scaler = StandardScaler().fit(tr.to_numpy(float))
        z = np.clip(self.scaler.transform(tr.to_numpy(float)), -CLIP, CLIP)
        self.pca = PCA(n_components=self.var_ratio, svd_solver="full", whiten=True).fit(z)
        return self

    @property
    def n_components(self) -> int:
        return int(self.pca.n_components_)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.cols].to_numpy(float)
        bad = np.isnan(x).any(axis=1)  # dong thieu -> PC = NaN (cua so chua dong nay bi bo)
        z = np.clip(self.scaler.transform(np.where(np.isnan(x), 0.0, x)), -CLIP, CLIP)
        out = np.clip(self.pca.transform(z), -CLIP, CLIP)
        out[bad] = np.nan
        return out


# ------------------------------------------------------------------ cua so chuoi
class Windows:
    """Chuoi dac trung L ngay cho moi (region, ngay goc). Kenh: trend_s - trend_t, seasonal_s, resid_s, PC..."""

    def __init__(self, df: pd.DataFrame, pcs: np.ndarray, window: int):
        self.window = window
        n_pc = pcs.shape[1]
        self.index: dict[tuple[str, pd.Timestamp], int] = {}
        seqs, regs, keys = [], [], []
        pc_df = pd.DataFrame(pcs, index=df.index)
        for region, g in df.groupby("region", sort=False):
            g = g.sort_values("date")
            comp = g[COMP].to_numpy(float)
            pc = pc_df.loc[g.index].to_numpy(float)
            dates = g["date"].to_numpy()
            for i in range(window - 1, len(g)):
                c = comp[i - window + 1: i + 1]
                p = pc[i - window + 1: i + 1]
                if np.isnan(c).any() or np.isnan(p).any():
                    continue
                x = np.concatenate([(c[:, [0]] - c[-1, 0]), c[:, 1:], p], axis=1)
                keys.append((region, pd.Timestamp(dates[i])))
                seqs.append(x)
                regs.append(REGIONS.index(region))
        self.x = np.asarray(seqs, dtype=np.float32)
        self.region = np.asarray(regs, dtype=np.int64)
        self.keys = keys
        self.index = {k: i for i, k in enumerate(keys)}
        self.n_in = 3 + n_pc

    def rows(self, region: pd.Series, date: pd.Series) -> np.ndarray:
        """Chi so hang cua cac (region, date); -1 neu khong co cua so day du."""
        return np.array([self.index.get((r, pd.Timestamp(d)), -1) for r, d in zip(region, date)])


class ChannelScaler:
    """Chuan hoa 3 kenh thanh phan theo do lech chuan tren Train (PC da whiten san)."""

    def fit(self, x: np.ndarray) -> "ChannelScaler":
        self.sd = x[:, :, :3].reshape(-1, 3).std(axis=0) + 1e-6
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        y = x.copy()
        y[:, :, :3] = np.clip(y[:, :, :3] / self.sd, -CLIP, CLIP)
        return y.astype(np.float32)


# ------------------------------------------------------------------ mo hinh
class BiLSTMReg(nn.Module):
    def __init__(self, n_in: int, hidden: int, dropout: float, n_regions: int = len(REGIONS)):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(2 * hidden + n_regions, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.n_regions = n_regions

    def forward(self, x, region):
        _, (h, _) = self.lstm(x)
        feat = torch.cat([h[-2], h[-1], nn.functional.one_hot(region, self.n_regions).float()], dim=1)
        return self.head(feat).squeeze(1)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_model(x, region, y, params, seed, epochs, on_epoch=None) -> BiLSTMReg:
    """Huan luyen tren (x, region, y da chuan hoa). on_epoch(model, epoch) goi sau moi epoch (dung khi tune)."""
    set_seed(seed)
    model = BiLSTMReg(x.shape[2], params["hidden"], params["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    xt, rt, yt = torch.from_numpy(x), torch.from_numpy(region), torch.from_numpy(y.astype(np.float32))
    gen = torch.Generator().manual_seed(seed)
    n = len(xt)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, BATCH):
            b = perm[i: i + BATCH]
            opt.zero_grad()
            loss_fn(model(xt[b], rt[b]), yt[b]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if on_epoch is not None:
            on_epoch(model, ep)
    return model


def predict(model: BiLSTMReg, x, region) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x), torch.from_numpy(region)).numpy()


# ------------------------------------------------------------------ pipeline dac trung
class Pipeline:
    """Gom STL + PCA(Train) + cua so + chuan hoa cho mot bo sieu tham so (window, pca_var)."""

    def __init__(self, df_stl: pd.DataFrame, exog: list[str], window: int, pca_var: float):
        self.df = df_stl
        self.pca = ExogPCA(exog, pca_var).fit(df_stl)
        self.win = Windows(df_stl, self.pca.transform(df_stl), window)
        tr_rows = self.win.rows(*self._keys(df_stl["split"].eq("train")))
        tr_rows = tr_rows[tr_rows >= 0]
        self.scaler = ChannelScaler().fit(self.win.x[tr_rows])
        self.x_all = self.scaler.transform(self.win.x)

    def _keys(self, mask):
        d = self.df.loc[mask]
        return d["region"], d["date"]

    def design(self, d: pd.DataFrame, horizon: int, with_label=True):
        """Tra ve (hang, x, region, delta_vnd, price_t) cho cac dong d co cua so day du (va nhan huu han neu with_label)."""
        d = d.copy()
        d["_row"] = self.win.rows(d["region"], d["date"])
        ok = (d["_row"] >= 0) & np.isfinite(d["price_t"])
        if with_label:
            ok &= np.isfinite(d[f"y_h{horizon}"])
        d = d.loc[ok]
        rows = d["_row"].to_numpy(dtype=int)
        delta = (d[f"y_h{horizon}"] - d["price_t"]).to_numpy(float) if with_label else None
        return d, self.x_all[rows], self.win.region[rows], delta, d["price_t"].to_numpy(float)


def eval_rows(d: pd.DataFrame, horizon: int, lo: str, hi: str) -> pd.DataFrame:
    """Dong danh gia hop le: goc trong [lo, hi], t+h <= hi, nhan huu han, khong phai Chu nhat sao chep (nhu P3/P4)."""
    d = d.loc[(d["date"] >= lo) & (d["date"] <= hi)]
    d = d.loc[(d["date"] + pd.Timedelta(days=horizon)) <= pd.Timestamp(hi)]
    return d.loc[d["is_sunday_copy"].eq(0) & np.isfinite(d[f"y_h{horizon}"]) & np.isfinite(d["price_t"])]


# ------------------------------------------------------------------ chong ro ri
def assert_no_leak(df_stl: pd.DataFrame, n_check: int = 6, seed: int = 0) -> None:
    """Kiem tra STL cuon chieu khong dung tuong lai: tinh lai tai vai ngay t chi tu du lieu <= t phai khop."""
    rng = np.random.default_rng(seed)
    for region, g in df_stl.groupby("region"):
        g = g.sort_values("date").reset_index(drop=True)
        valid = np.flatnonzero(g[COMP[0]].notna().to_numpy())
        for i in rng.choice(valid, size=min(n_check, len(valid)), replace=False):
            hist = g.loc[:i, "price_ffill"].ffill().dropna().to_numpy(float)
            got = np.array([g.loc[i, c] for c in COMP])
            want = np.array(causal_stl_last(hist))
            assert np.allclose(got, want, rtol=1e-9, atol=1e-6), f"STL ro ri tuong lai: {region} {g.loc[i, 'date']}"
    # PCA khong dung Val/Test
    exog = exog_columns()
    a = ExogPCA(exog, 0.95).fit(df_stl)
    mod = df_stl.copy()
    m = mod["split"] != "train"
    mod.loc[m, exog] = mod.loc[m, exog] * 3.0 + 1.0
    b = ExogPCA(exog, 0.95).fit(mod)
    assert np.allclose(a.pca.components_, b.pca.components_) and np.allclose(a.scaler.mean_, b.scaler.mean_), \
        "PCA/scaler phu thuoc du lieu ngoai Train"


# ------------------------------------------------------------------ tune tren Val
def val_mae_per_epoch(pipe: Pipeline, horizon: int, params: dict, seed: int, max_epochs: int, cfg: dict) -> np.ndarray:
    """MAE (VND) tren Val sau moi epoch, huan luyen chi tren Train (goc hop le trong Train)."""
    df = pipe.df
    tr = df.loc[df["split"].eq("train") & is_valid_origin(df["date"], horizon).to_numpy()]
    _, x, r, delta, _ = pipe.design(tr, horizon)
    mu, sd = delta.mean(), delta.std() + 1e-6
    v = eval_rows(df.loc[df["split"].eq("val")], horizon, cfg["val"]["start"], cfg["val"]["end"])
    vd, vx, vr, vdelta, _ = pipe.design(v, horizon)
    maes = []

    def on_epoch(model, ep):
        pred = predict(model, vx, vr) * sd + mu
        maes.append(float(np.mean(np.abs(pred - vdelta))))

    train_model(x, r, (delta - mu) / sd, params, seed, max_epochs, on_epoch)
    return np.asarray(maes)


def tune(df_stl: pd.DataFrame, exog: list[str], horizon: int, cfg: dict, grid=GRID, max_epochs=MAX_EPOCHS) -> tuple[dict, pd.DataFrame]:
    """Luoi sieu tham so tren Val. Moi to hop: trung binh MAE Val cua TUNE_SEEDS theo epoch, chon epoch tot nhat."""
    rows, pipes = [], {}
    for window, pca_var in itertools.product(grid["window"], grid["pca_var"]):
        pipes[(window, pca_var)] = Pipeline(df_stl, exog, window, pca_var)
    for window, hidden, dropout, lr, pca_var in itertools.product(
            grid["window"], grid["hidden"], grid["dropout"], grid["lr"], grid["pca_var"]):
        params = {"window": window, "hidden": hidden, "dropout": dropout, "lr": lr, "pca_var": pca_var}
        curves = np.mean([val_mae_per_epoch(pipes[(window, pca_var)], horizon, params, s, max_epochs, cfg)
                          for s in TUNE_SEEDS], axis=0)
        best = int(np.argmin(curves))
        rows.append({"horizon": horizon, **params, "n_pc": pipes[(window, pca_var)].pca.n_components,
                     "best_epoch": best + 1, "val_mae": float(curves[best])})
    res = pd.DataFrame(rows).sort_values("val_mae").reset_index(drop=True)
    top = res.iloc[0]
    best_params = {k: (int(top[k]) if k in ("window", "hidden") else float(top[k])) for k in
                   ("window", "hidden", "dropout", "lr", "pca_var")}
    best_params["epochs"] = int(top["best_epoch"])
    return best_params, res


# ------------------------------------------------------------------ walk-forward tren Test
def run_walkforward(df_stl: pd.DataFrame, exog: list[str], horizon: int, params: dict, seed: int,
                    cfg: dict, pipe: Pipeline | None = None) -> pd.DataFrame:
    """4 lan huan luyen lai (dau thang), cua so mo rong; PCA/scaler dong bang (fit Train)."""
    pipe = pipe or Pipeline(df_stl, exog, params["window"], params["pca_var"])
    parts = []
    for w in get_walkforward_windows(cfg, horizon):
        retrain = pd.Timestamp(w["retrain_date"])
        tr = available_training_rows(pipe.df, retrain, horizon)  # t + h < retrain
        _, x, r, delta, _ = pipe.design(tr, horizon)
        mu, sd = delta.mean(), delta.std() + 1e-6
        model = train_model(x, r, (delta - mu) / sd, params, seed, params["epochs"])
        te = eval_rows(pipe.df, horizon, w["prediction_start"], cfg["test"]["end"])
        te = te.loc[(te["date"] >= w["prediction_start"]) & (te["date"] <= w["prediction_end"])]
        if te.empty:  # vd. thang 10 voi h = 30: khong con goc nao co t + h <= 30/10
            continue
        te, tx, tr_reg, _, price = pipe.design(te, horizon)
        pred = price + predict(model, tx, tr_reg) * sd + mu
        parts.append(pd.DataFrame({
            "model": MODEL_NAME, "config": CONFIG_NAME, "region": te["region"].to_numpy(), "horizon": horizon,
            "origin_date": te["date"].dt.strftime("%Y-%m-%d").to_numpy(),
            "target_date": (te["date"] + pd.Timedelta(days=horizon)).dt.strftime("%Y-%m-%d").to_numpy(),
            "y_true": te[f"y_h{horizon}"].to_numpy(), "y_pred": pred}))
    return pd.concat(parts, ignore_index=True)[PREDICTION_COLUMNS]


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="STL-PCA-BiLSTM walk-forward")
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--no-tune", action="store_true", help="dung DEFAULT_PARAMS, bo qua tune tren Val")
    ap.add_argument("--output", default=str(PRED_DIR / f"{MODEL_NAME}_{CONFIG_NAME}.csv"))
    ap.add_argument("--seed-dir", default=str(PRED_DIR / "seeds"), help="thu muc file du bao tung seed")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # duong dan co dau tren Windows

    cfg = load_config(CONFIG_PATH)
    exog = exog_columns()
    print(f"Bien ngoai sinh vao PCA: {len(exog)}")
    df = add_stl_features(load_data())
    assert_no_leak(df)
    print("Assert chong ro ri: OK (STL cuon chieu, PCA/scaler chi fit Train)")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    seed_dir = Path(args.seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)
    chosen, grids = {}, []
    for h in args.horizons:
        if args.no_tune:
            chosen[h] = dict(DEFAULT_PARAMS)
        else:
            chosen[h], res = tune(df, exog, h, cfg)
            grids.append(res)
            print(f"H{h}: chon {chosen[h]} (Val MAE {res.val_mae.iloc[0]:.0f})")
    if grids:
        pd.concat(grids).to_csv(TABLE_DIR / "3_4_3_stl_pca_bilstm_val_grid.csv", index=False, float_format="%.4f")
        (TABLE_DIR / "3_4_3_stl_pca_bilstm_params.json").write_text(
            json.dumps({str(h): p for h, p in chosen.items()}, indent=2), encoding="utf-8")

    seed_frames = {s: [] for s in args.seeds}
    for h in args.horizons:
        pipe = Pipeline(df, exog, chosen[h]["window"], chosen[h]["pca_var"])
        for s in args.seeds:
            print(f"Walk-forward H{h} seed {s} ...", flush=True)
            seed_frames[s].append(run_walkforward(df, exog, h, chosen[h], s, cfg, pipe))
    per_seed = []
    for s, parts in seed_frames.items():
        d = pd.concat(parts, ignore_index=True)
        d.to_csv(seed_dir / f"{MODEL_NAME}_{CONFIG_NAME}_seed{s}.csv", index=False, float_format="%.6f")
        per_seed.append(d.assign(seed=s))
    allp = pd.concat(per_seed, ignore_index=True)
    keys = ["model", "config", "region", "horizon", "origin_date", "target_date", "y_true"]
    mean = allp.groupby(keys, as_index=False)["y_pred"].mean()[PREDICTION_COLUMNS]
    mean = mean.sort_values(["horizon", "region", "origin_date"]).reset_index(drop=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    mean.to_csv(args.output, index=False, float_format="%.6f")
    print(f"Da ghi {args.output}: {len(mean)} dong")
    print(evaluate_predictions(mean, load_data()[["date", "region", "is_sunday_copy"]]).to_string(index=False))


if __name__ == "__main__":
    main()
