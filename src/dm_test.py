"""P6: kiem dinh Diebold-Mariano (DM) co hieu chinh Harvey-Leybourne-Newbold cho du bao nhieu buoc chong lan.

Chay tu thu muc goc repo:
    python src/dm_test.py                                  # tu tim predictions/*.csv (bo qua predictions/seeds/)
    python src/dm_test.py predictions/naive_c6.csv predictions/p3_arima_sarimax.csv --output results/tables/4_3_dm_demo.csv

Y nghia:  d_t = L(e_a,t) - L(e_b,t),  L = |e| (ae) hoac e^2 (se).
  * dm_stat < 0: mo hinh A co sai so nho hon B (A tot hon). p_value hai phia, phan phoi t(T-1).
  * Du bao h ngay (h = 7, 14, 30) tai goc t va t+1 chong lan -> sai so tu tuong quan toi bac h-1.
    Phuong sai dai han = gamma_0 + 2*sum_{k=1}^{h-1} gamma_k (tinh tren LICH NGAY: cap (t, t+k) chi tinh khi ca hai co mat,
    vi Chu nhat sao chep va nhan thieu tao khoang trong).
  * Hieu chinh Harvey et al. (1997): nhan DM voi sqrt((T + 1 - 2h + h(h-1)/T) / T), so sanh voi t(T-1).
  * Chi so sanh cac diem goc chung cua hai mo hinh (giao theo region, origin_date, horizon).
  * kernel Bartlett (cot *_bartlett) la kiem tra do ben khi h lon: DM goc co the khong tinh duoc / phong dai.
  * region = "all": lay trung binh d_t cua cac mien co mat tai moi ngay goc, roi DM tren chuoi trung binh nay.
    Ba mien cung chiu cu soc gia chung nen khong gop thang 3 chuoi nhu 3 mau doc lap.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from evaluate import HORIZONS, validate_predictions

ROOT = Path(__file__).resolve().parents[1]
LOSSES = ("ae", "se")
RESULT_COLUMNS = ["model_a", "model_b", "config", "region", "horizon", "loss", "n",
                  "mean_diff", "dm_stat", "p_value", "dm_stat_bartlett", "p_value_bartlett", "better"]


def _loss(err: np.ndarray, kind: str) -> np.ndarray:
    if kind == "ae":
        return np.abs(err)
    if kind == "se":
        return err ** 2
    raise ValueError("loss phai la 'ae' hoac 'se'")


def dm_test(d: pd.Series, h: int, hln: bool = True, kernel: str = "rect") -> dict:
    """DM tren chuoi hieu so ton that d (index = ngay goc, NaN neu thieu). Tra ve dict n, mean_diff, dm_stat, p_value.

    kernel = "rect": DM goc (Diebold-Mariano 1995), trong so 1 cho lag 1..h-1. Uoc luong phuong sai co the <= 0 hoac rat nho
             khi h lon so voi n (vd. h = 30, n ~ 80) -> khi do tra NaN / thong ke phong dai.
    kernel = "bartlett": trong so (1 - k/h) (Newey-West), luon khong am; dung lam kiem tra do ben.
    """
    if kernel not in ("rect", "bartlett"):
        raise ValueError("kernel phai la 'rect' hoac 'bartlett'")
    if h < 1:
        raise ValueError("h phai >= 1")
    d = d.sort_index()
    cal = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D"))
    x = cal.to_numpy(float)
    ok = np.isfinite(x)
    n = int(ok.sum())
    out = {"n": n, "mean_diff": float(np.nanmean(x)) if n else np.nan, "dm_stat": np.nan, "p_value": np.nan}
    if n < max(10, h + 2):
        return out
    dbar = x[ok].mean()
    xc = np.where(ok, x - dbar, 0.0)
    lrv = (xc @ xc) / n
    for k in range(1, h):
        both = ok[k:] & ok[:-k]
        w = 1.0 if kernel == "rect" else 1.0 - k / h
        lrv += 2 * w * float(np.sum(xc[k:][both] * xc[:-k][both])) / n
    if not np.isfinite(lrv) or lrv <= 0:
        return out
    stat = dbar / np.sqrt(lrv / n)
    if hln:
        corr = (n + 1 - 2 * h + h * (h - 1) / n) / n
        if corr <= 0:
            return out
        stat *= np.sqrt(corr)
    out["dm_stat"] = float(stat)
    out["p_value"] = float(2 * stats.t.sf(abs(stat), df=n - 1))
    return out


def load_predictions(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        d = validate_predictions(pd.read_csv(p))
        d["source"] = p.name
        frames.append(d)
    allp = pd.concat(frames, ignore_index=True)
    allp = allp.loc[np.isfinite(allp.y_true) & np.isfinite(allp.y_pred)]
    dup = allp.duplicated(["model", "config", "region", "horizon", "origin_date"])
    if dup.any():
        raise ValueError("Trung du bao giua cac file: " + ", ".join(sorted(set(allp.loc[dup, "source"]))))
    allp["err"] = allp.y_true - allp.y_pred
    return allp


def pairwise_dm(preds: pd.DataFrame, loss_kinds=LOSSES, hln: bool = True) -> pd.DataFrame:
    """DM cho moi cap mo hinh (A < B theo thu tu ten), moi config, horizon, region va 'all'."""
    rows = []
    preds = preds.assign(origin_date=pd.to_datetime(preds["origin_date"]))
    for (config, horizon), g in preds.groupby(["config", "horizon"]):
        models = sorted(g.model.unique())
        wide = g.pivot_table(index=["region", "origin_date"], columns="model", values="err", aggfunc="first")
        for a, b in itertools.combinations(models, 2):
            pair = wide[[a, b]].dropna()
            for kind in loss_kinds:
                diff = pd.Series(_loss(pair[a].to_numpy(), kind) - _loss(pair[b].to_numpy(), kind), index=pair.index)
                series = {r: diff.xs(r, level="region") for r in diff.index.get_level_values("region").unique()}
                series["all"] = diff.groupby(level="origin_date").mean()
                for region, s in sorted(series.items(), key=lambda kv: (kv[0] == "all", kv[0])):
                    res = dm_test(s, int(horizon), hln)
                    bart = dm_test(s, int(horizon), hln, kernel="bartlett")
                    better = "" if res["mean_diff"] == 0 or not np.isfinite(res["mean_diff"]) else (
                        a if res["mean_diff"] < 0 else b)
                    rows.append({"model_a": a, "model_b": b, "config": config, "region": region,
                                 "horizon": int(horizon), "loss": kind, **res,
                                 "dm_stat_bartlett": bart["dm_stat"], "p_value_bartlett": bart["p_value"],
                                 "better": better})
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def default_prediction_files() -> list[Path]:
    return sorted(p for p in (ROOT / "predictions").glob("*.csv") if p.is_file())


def main():
    ap = argparse.ArgumentParser(description="Diebold-Mariano (HLN) theo cap mo hinh va horizon")
    ap.add_argument("predictions", nargs="*", help="file du bao 8 cot; bo trong de dung predictions/*.csv")
    ap.add_argument("--output", default=str(ROOT / "results" / "tables" / "4_3_dm_pvalues.csv"))
    ap.add_argument("--no-hln", action="store_true", help="khong hieu chinh Harvey (DM goc)")
    args = ap.parse_args()
    files = [Path(p) for p in args.predictions] or default_prediction_files()
    if len(files) < 1:
        raise SystemExit("Khong co file du bao")
    preds = load_predictions(files)
    print("Mo hinh:", ", ".join(sorted(preds.model.unique())), "| file:", ", ".join(f.name for f in files))
    res = pairwise_dm(preds, hln=not args.no_hln)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.output, index=False, float_format="%.6f")
    show = res[(res.region == "all") & (res.loss == "ae")]
    print(show.to_string(index=False))
    print("Da luu:", args.output)


if __name__ == "__main__":
    main()
