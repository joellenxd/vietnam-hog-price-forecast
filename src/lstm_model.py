# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# lstm_model.py: LSTM và BiLSTM dự báo đa bước (direct) giá heo hơi Việt Nam.
# Dự báo:   y_{t+h}, h = 7, 14, 30 ngày, cho 3 miền (Bắc, Trung, Nam).
# Chiến lược: direct - mỗi (mô hình, horizon) có một mạng riêng, gộp chung 3 miền (có biến one-hot cho miền thay cho region_code).
# Mục tiêu:   học Delta = y_{t+h} - price_t rồi cộng lại: y_pred = price_t + Delta_hat.
# Đầu vào:    cửa sổ WINDOW ngày gần nhất (kết thúc tại ngày xuất phát t) của các biến trong cấu hình ablation (mặc định c6, 58 biến + 3 cột one-hot miền).
# Quy trình chung của nhóm (theo P3):
#   1) Chọn số epoch trên Validation (01/04-30/06/2026): huấn luyện chỉ với nhãn đã biết đến 31/03/2026, early stopping theo MAE trên Val. Không dùng Test.
#   2) Test 01/07-30/10/2026, walk-forward: huấn luyện lại từ đầu (epoch cố định ở bước 1) vào các mốc R = 01/07, 01/08, 01/09, 01/10/2026 trên dữ liệu mở rộng đến ngày R-1. Origin trong tháng nào dùng mô hình của mốc đầu tháng đó; trong tháng KHÔNG huấn luyện lại.
#   3) Chống rò rỉ nhãn: tại mốc R chỉ dùng origin t có t + h <= R - 1. Bộ chuẩn hóa chỉ khớp trên các ngày <= R - 1 (cột nhãn Delta cũng vậy).
#   4) Origin Chủ nhật chép lại (is_sunday_copy = 1) bị loại; ngày mục tiêu Chủ nhật vẫn giữ.
#   5) Origin hợp lệ ở Test: split = test, valid_h = 1, có nhãn y_h (đúng bộ 99/93/79 mỗi miền).
# Hướng dẫn chạy (từ thư mục gốc repo):
#     python src/lstm_model.py                 # LSTM + BiLSTM, 3 horizon, c6
#     python src/lstm_model.py --smoke         # chạy thử, nhanh (3 epoch, 1 seed)
#     python src/lstm_model.py --models lstm --horizons 7 --seeds 1
# Kết quả:
#     predictions/lstm_c6.csv, predictions/bilstm_c6.csv   (8 cột chuẩn chung)
#     results/tables/lstm_bilstm_c6_metrics.csv            (MAE, RMSE, MAPE theo miền x horizon)
#     results/tables/lstm_bilstm_c6_val_selection.csv      (epoch chọn trên Val)
# """
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1] if HERE.parent.name == "src" else Path.cwd()

# ----------------------------- hằng số của nhóm -----------------------------
TRAIN_END = pd.Timestamp("2026-03-31")
VAL_START, VAL_END = pd.Timestamp("2026-04-01"), pd.Timestamp("2026-06-30")
TEST_START, TEST_END = pd.Timestamp("2026-07-01"), pd.Timestamp("2026-10-30")
RETRAIN_DATES = [pd.Timestamp(d) for d in ("2026-07-01", "2026-08-01", "2026-09-01", "2026-10-01")]
HORIZONS = (7, 14, 30)

# ----------------------------- tham số mặc định -----------------------------
DEFAULTS = dict(
    window=14,       # số ngày trong cửa sổ đầu vào
    units=32,        # số đơn vị ẩn của LSTM (BiLSTM: mỗi chiều 32, tổng 64)
    layers=1,        # số lớp LSTM
    dropout=0.0,
    lr=1e-3,         # Adam
    batch=32,
    max_epochs=100,  # trần epoch khi chọn trên Val
    patience=10,     # early stopping trên Val
    min_epochs=5,    # sàn epoch dùng khi huấn luyện lại ở Test
    seeds=3,         # số lần khởi tạo ngẫu nhiên, lấy trung bình dự báo
    clip=10.0,       # cắt giá trị đã chuẩn hóa về [-clip, clip]
    center=False,    # True: trừ trung bình Delta khi chuẩn hóa nhãn; False: chỉ chia độ lệch chuẩn (Delta=0 là mốc Naive)
)


# ----------------------------- đọc dữ liệu -----------------------------
def find_file(name, extra=()):
    cands = [ROOT / "data" / "features" / name, ROOT / "data" / name, ROOT / name, *extra]
    for p in cands:
        if Path(p).exists():
            return Path(p)
    return None


def load_features(config):
    """Danh sách biến của cấu hình (c1..c6) từ feature_groups.csv; thiếu file thì dùng c6 mặc định."""
    fg_path = find_file("feature_groups.csv")
    if fg_path is not None:
        fg = pd.read_csv(fg_path, encoding="utf-8-sig")
        if config in fg.columns and "feature" in fg.columns:
            return fg.loc[fg[config] == 1, "feature"].tolist()
    return None


def load_panel(data_path, config):
    df = pd.read_csv(data_path, parse_dates=["date"], encoding="utf-8-sig")
    feats = load_features(config)
    if feats is None:
        if config != "c6":
            sys.exit("Không tìm thấy feature_groups.csv nên chỉ chạy được c6.")
        start, end = list(df.columns).index("price_t"), list(df.columns).index("wb_crude_usd_bbl")
        feats = [c for c in df.columns[start:end + 1] if c != "retail_idx_mean"]
    missing = [c for c in feats if c not in df.columns]
    if missing:
        sys.exit(f"Thiếu cột trong data_v1.csv: {missing}")
    num_feats = [c for c in feats if c != "region_code"]   # region_code thay bằng one-hot
    use_region = "region_code" in feats
    regions = sorted(df["region"].unique())
    P = {"regions": regions, "feats": num_feats, "use_region": use_region, "r": {}}
    for k, r in enumerate(regions):
        g = df[df["region"] == r].sort_values("date").reset_index(drop=True)
        dd = g["date"].diff().dropna().dt.days
        assert (dd == 1).all(), f"Dữ liệu miền {r} không liên tục theo ngày"
        P["r"][r] = dict(
            dates=g["date"].to_numpy(),
            X=g[num_feats].to_numpy(float),
            price_t=g["price_t"].to_numpy(float),
            sun=g["is_sunday_copy"].to_numpy(),
            split=g["split"].to_numpy(),
            y={h: g[f"y_h{h}"].to_numpy(float) for h in HORIZONS},
            valid={h: g[f"valid_h{h}"].to_numpy() for h in HORIZONS},
            onehot=np.eye(len(regions))[k],
        )
    return P


# ----------------------------- chuẩn hóa và mẫu -----------------------------
def fit_scaler(P, cutoff):
    """Trung bình/độ lệch chuẩn của biến, chỉ trên các ngày <= cutoff (mọi miền)."""
    rows = [R["X"][R["dates"] <= np.datetime64(cutoff)] for R in P["r"].values()]
    M = np.concatenate(rows, axis=0)
    mu = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0)
    mu = np.where(np.isnan(mu), 0.0, mu)
    sd = np.where((sd < 1e-8) | np.isnan(sd), 1.0, sd)
    return mu, sd


def scaled_matrix(P, R, mu, sd, clip):
    Z = (R["X"] - mu) / sd
    Z = np.clip(np.nan_to_num(Z, nan=0.0), -clip, clip)   # NaN -> 0 = trung bình của tập huấn luyện
    if P["use_region"]:
        Z = np.concatenate([Z, np.tile(R["onehot"], (len(Z), 1))], axis=1)
    return Z


def eligible(R, h, window, target_max=None, origin_min=None, origin_max=None, test_rule=False):
    """Chỉ số ngày xuất phát hợp lệ cho horizon h."""
    n = len(R["dates"])
    idx = np.arange(n)
    dates = pd.to_datetime(R["dates"])
    ok = (idx >= window - 1) & (R["sun"] == 0) & ~np.isnan(R["y"][h]) & ~np.isnan(R["price_t"])
    target = dates + pd.Timedelta(days=h)
    if target_max is not None:
        ok &= (target <= target_max)
    if origin_min is not None:
        ok &= (dates >= origin_min)
    if origin_max is not None:
        ok &= (dates <= origin_max)
    if test_rule:
        ok &= (R["split"] == "test") & (R["valid"][h] == 1)
    return idx[ok]


def make_windows(Z, idx, window):
    return np.stack([Z[i - window + 1:i + 1] for i in idx]).astype("float32") if len(idx) else np.zeros((0, window, Z.shape[1]), "float32")


def build_set(P, h, window, scal, cfg, **kw):
    """Gộp 3 miền: trả về (X, delta, price_t, idx_info)."""
    mu, sd = scal
    Xs, ds, info = [], [], []
    for r, R in P["r"].items():
        idx = eligible(R, h, window, **kw)
        if len(idx) == 0:
            continue
        Z = scaled_matrix(P, R, mu, sd, cfg["clip"])
        Xs.append(make_windows(Z, idx, window))
        ds.append(R["y"][h][idx] - R["price_t"][idx])
        info += [(r, int(i)) for i in idx]
    if not Xs:   # không có origin hợp lệ (vd. horizon 30 ở tháng 10/2026: t + 30 vượt ngày cuối Test)
        return None, None, []
    return np.concatenate(Xs, axis=0), np.concatenate(ds), info


# ----------------------------- mô hình -----------------------------
def build_model(kind, n_feat, cfg):
    import keras
    from keras import layers
    inp = keras.Input(shape=(cfg["window"], n_feat))
    x = inp
    for k in range(cfg["layers"]):
        last = k == cfg["layers"] - 1
        rnn = layers.LSTM(cfg["units"], dropout=cfg["dropout"], return_sequences=not last)
        x = layers.Bidirectional(rnn)(x) if kind == "bilstm" else rnn(x)
    out = layers.Dense(1)(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(cfg["lr"]), loss="mse", metrics=["mae"])
    return m


def set_seed(seed):
    import keras
    import tensorflow as tf
    keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def train_model(kind, Xtr, ytr, cfg, seed, epochs, val=None):
    import keras
    set_seed(seed)
    m = build_model(kind, Xtr.shape[2], cfg)
    cbs = []
    if val is not None:
        cbs.append(keras.callbacks.EarlyStopping(monitor="val_mae", patience=cfg["patience"], restore_best_weights=False))
    h = m.fit(Xtr, ytr, epochs=epochs, batch_size=cfg["batch"], shuffle=True, verbose=0,
              validation_data=val, callbacks=cbs)
    return m, h.history


# ----------------------------- bước 1: chọn epoch trên Val -----------------------------
def select_epochs(P, kind, h, cfg, log):
    mu_sd = fit_scaler(P, TRAIN_END)
    W = cfg["window"]
    Xtr, dtr, _ = build_set(P, h, W, mu_sd, cfg, target_max=TRAIN_END)
    Xv, dv, _ = build_set(P, h, W, mu_sd, cfg, origin_min=VAL_START, origin_max=VAL_END, target_max=VAL_END)
    dm = dtr.mean() if cfg["center"] else 0.0
    ds = dtr.std() if dtr.std() > 1e-8 else 1.0
    ytr, yv = ((dtr - dm) / ds).astype("float32"), ((dv - dm) / ds).astype("float32")
    best_eps, best_maes = [], []
    for s in range(cfg["seeds"]):
        _, hist = train_model(kind, Xtr, ytr, cfg, seed=s, epochs=cfg["max_epochs"], val=(Xv, yv))
        v = np.array(hist["val_mae"])
        best_eps.append(int(v.argmin()) + 1)
        best_maes.append(float(v.min() * ds))
    epochs = max(cfg["min_epochs"], int(round(np.median(best_eps))))
    row = dict(model=kind, horizon=h, n_train=len(dtr), n_val=len(dv), best_epoch_per_seed=str(best_eps),
               epochs_chosen=epochs, val_mae=float(np.mean(best_maes)), val_mae_delta0=float(np.mean(np.abs(dv))))
    log(f"  [Val] {kind:6s} h={h:2d}: n_train={len(dtr)} n_val={len(dv)} epoch={best_eps} -> {epochs} | "
        f"MAE Val {row['val_mae']:.0f} (Delta=0: {row['val_mae_delta0']:.0f})")
    return epochs, row


# ----------------------------- bước 2: walk-forward trên Test -----------------------------
def walk_forward(P, kind, h, epochs, cfg, log):
    W = cfg["window"]
    recs = []
    for k, R0 in enumerate(RETRAIN_DATES):
        cutoff = R0 - pd.Timedelta(days=1)
        next_R = RETRAIN_DATES[k + 1] if k + 1 < len(RETRAIN_DATES) else TEST_END + pd.Timedelta(days=1)
        scal = fit_scaler(P, cutoff)
        Xtr, dtr, _ = build_set(P, h, W, scal, cfg, target_max=cutoff)          # t + h <= R - 1
        Xte, dte, info = build_set(P, h, W, scal, cfg, origin_min=R0, origin_max=next_R - pd.Timedelta(days=1),
                                   test_rule=True)
        if len(info) == 0:
            log(f"  [Test] {kind:6s} h={h:2d} mốc {R0.date()}: không có origin hợp lệ, bỏ qua")
            continue
        dm = dtr.mean() if cfg["center"] else 0.0
        ds = dtr.std() if dtr.std() > 1e-8 else 1.0
        ytr = ((dtr - dm) / ds).astype("float32")
        preds = []
        for s in range(cfg["seeds"]):
            m, _ = train_model(kind, Xtr, ytr, cfg, seed=s, epochs=epochs)
            preds.append(m(Xte, training=False).numpy().ravel() * ds + dm)
        dhat = np.mean(preds, axis=0)
        for (r, i), dh in zip(info, dhat):
            R = P["r"][r]
            recs.append(dict(
                model=kind, config=cfg["config"], region=r, horizon=h,
                origin_date=pd.Timestamp(R["dates"][i]).date().isoformat(),
                target_date=(pd.Timestamp(R["dates"][i]) + pd.Timedelta(days=h)).date().isoformat(),
                y_true=R["y"][h][i], y_pred=R["price_t"][i] + dh))
        log(f"  [Test] {kind:6s} h={h:2d} mốc {R0.date()}: train n={len(dtr)}, dự báo {len(info)} origin")
    return recs


# ----------------------------- đánh giá -----------------------------
def metrics(df):
    out = []
    for (m, c, r, h), g in df.groupby(["model", "config", "region", "horizon"]):
        e = g["y_pred"] - g["y_true"]
        out.append(dict(model=m, config=c, region=r, horizon=h, n=len(g), mae=e.abs().mean(),
                        rmse=float(np.sqrt((e ** 2).mean())), mape=(e.abs() / g["y_true"].abs()).mean() * 100))
    return pd.DataFrame(out)


def compare_with_naive(pred_df, log):
    p = ROOT / "predictions" / "naive_c6.csv"
    if not p.exists():
        log("(Không có predictions/naive_c6.csv nên bỏ qua bước so khớp origin.)")
        return
    nv = pd.read_csv(p)
    keys = ["region", "horizon", "origin_date"]
    for h in sorted(pred_df["horizon"].unique()):
        a = set(map(tuple, pred_df[pred_df.horizon == h][keys].astype(str).values))
        b = set(map(tuple, nv[nv.horizon == h][keys].astype(str).values))
        log(f"So khớp origin với Naive, horizon {h}: {'KHỚP' if a == b else f'LỆCH (thiếu {len(b - a)}, thừa {len(a - b)})'} ({len(a)} dòng)")
    log("MAE theo horizon so với Naive (trung bình 3 miền):")
    mm = metrics(pred_df)
    mn = metrics(nv)
    for kind in mm["model"].unique():
        for h in sorted(mm["horizon"].unique()):
            a = mm[(mm.model == kind) & (mm.horizon == h)]["mae"].mean()
            b = mn[mn.horizon == h]["mae"].mean()
            log(f"  {kind:6s} h={h:2d}: MAE {a:8.0f} | Naive {b:8.0f} | {100 * (a / b - 1):+.1f}%")


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="đường dẫn data_v1.csv (mặc định data/features/data_v1.csv)")
    ap.add_argument("--config", default="c6", help="c1..c6 (mặc định c6)")
    ap.add_argument("--models", nargs="+", default=["lstm", "bilstm"], choices=["lstm", "bilstm"])
    ap.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    ap.add_argument("--seeds", type=int, default=DEFAULTS["seeds"])
    ap.add_argument("--window", type=int, default=DEFAULTS["window"])
    ap.add_argument("--max-epochs", type=int, default=DEFAULTS["max_epochs"])
    ap.add_argument("--smoke", action="store_true", help="chạy thử nhanh: 1 seed, 3 epoch")
    ap.add_argument("--out-dir", default=None, help="thư mục gốc để ghi predictions/ và results/ (mặc định = thư mục gốc repo)")
    a = ap.parse_args()

    cfg = dict(DEFAULTS, seeds=a.seeds, window=a.window, max_epochs=a.max_epochs, config=a.config)
    if a.smoke:
        cfg.update(seeds=1, max_epochs=3, min_epochs=2)
    out_root = Path(a.out_dir) if a.out_dir else ROOT
    data_path = Path(a.data) if a.data else find_file("data_v1.csv")
    if data_path is None or not Path(data_path).exists():
        sys.exit("Không tìm thấy data/features/data_v1.csv. Chạy lệnh từ thư mục gốc repo hoặc dùng --data.")

    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    P = load_panel(data_path, a.config)
    log(f"Dữ liệu: {data_path} | cấu hình {a.config}: {len(P['feats'])} biến"
        f"{' + one-hot miền' if P['use_region'] else ''} | cửa sổ {cfg['window']} ngày | seeds {cfg['seeds']}")

    all_preds, sel_rows = [], []
    for kind in a.models:
        for h in a.horizons:
            epochs, row = select_epochs(P, kind, h, cfg, log)
            sel_rows.append(row)
            all_preds += walk_forward(P, kind, h, epochs, cfg, log)

    pred = pd.DataFrame(all_preds)
    pred = pred.sort_values(["model", "region", "horizon", "origin_date"]).reset_index(drop=True)
    (out_root / "predictions").mkdir(parents=True, exist_ok=True)
    (out_root / "results" / "tables").mkdir(parents=True, exist_ok=True)
    for kind, g in pred.groupby("model"):
        g.to_csv(out_root / "predictions" / f"{kind}_{a.config}.csv", index=False, float_format="%.6f")
    met = metrics(pred)
    met.to_csv(out_root / "results" / "tables" / f"lstm_bilstm_{a.config}_metrics.csv", index=False, float_format="%.6f")
    pd.DataFrame(sel_rows).to_csv(out_root / "results" / "tables" / f"lstm_bilstm_{a.config}_val_selection.csv", index=False)
    log(f"Đã ghi {len(pred)} dòng dự báo, {len(met)} nhóm chỉ số vào {out_root}")
    compare_with_naive(pred, log)


if __name__ == "__main__":
    main()
