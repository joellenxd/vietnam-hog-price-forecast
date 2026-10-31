# """split.py - Chia du lieu gia heo hoi Viet Nam (theo ngay) thanh Train / Val / Test.
# Chia theo THU TU THOI GIAN (khong xao tron). Moc ngay co dinh, gom ca hai dau.
# Nhan y_h{h} cua diem goc t la gia tai t + h ngay, nen moi tap chi dung diem goc t thoa t + h <= ngay cuoi cua tap do (loai cac diem goc sat cuoi tap, tranh ro ri nhan sang tap sau).
# Ket qua: data/split/split_config.json
# Cach chay (tu thu muc goc cua repo):
#     python src/split.py
# """
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# (ten tap, ngay dau, ngay cuoi, so ngay)
SPLITS = [
    ("train", "2025-07-02", "2026-03-31", 273),
    ("val", "2026-04-01", "2026-06-30", 91),
    ("test", "2026-07-01", "2026-10-30", 122),
]
TOTAL_DAYS = sum(s[3] for s in SPLITS)  # 486
HORIZONS = [7, 14, 30]                  # ngay
RETRAIN_DATES = ["2026-07-01", "2026-08-01", "2026-09-01", "2026-10-01"]  # dau moi thang trong Test
REGIONS = ["Bac", "Trung", "Nam"]
DEFAULT_INPUT = "data/raw/parquet/supervised_region_daily.parquet"


def assign_split(dates):
    """Tra ve Series nhan 'train'/'val'/'test' cho tung ngay."""
    dates = pd.to_datetime(pd.Series(dates))
    out = pd.Series(index=dates.index, dtype="object")
    for name, start, end, _ in SPLITS:
        out[(dates >= start) & (dates <= end)] = name
    assert out.notna().all(), "Co ngay khong thuoc tap nao"
    return out


def is_valid_origin(dates, horizon):
    """True neu diem goc t co t + horizon <= ngay cuoi cua tap chua t."""
    dates = pd.to_datetime(pd.Series(dates))
    ends = {name: pd.Timestamp(end) for name, _, end, _ in SPLITS}
    split_end = assign_split(dates).map(ends)
    return (dates + pd.Timedelta(days=horizon)) <= split_end


def main():
    ap = argparse.ArgumentParser(description="Chia Train/Val/Test theo thoi gian")
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--outdir", default="data/split")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        sys.exit(f"Khong thay file: {src}. Dat bo du lieu vao data/raw/ (xem README).")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(src)
    df["date"] = pd.to_datetime(df["date"])
    for c in ["date", "region", "price", "y_h7", "y_h14", "y_h30"]:
        assert c in df.columns, f"Thieu cot {c}"

    # --- Kiem tra dau vao ---
    assert not df.duplicated(["date", "region"]).any(), "Trung khoa (date, region)"
    assert set(df["region"]) == set(REGIONS), f"Mien khong dung: {set(df['region'])}"
    all_days = pd.date_range(SPLITS[0][1], SPLITS[-1][2], freq="D")
    assert len(all_days) == TOTAL_DAYS, "Tong so ngay cac tap khac khoang ngay"
    for r, g in df.groupby("region"):
        assert len(g) == TOTAL_DAYS and set(g["date"]) == set(all_days), f"Mien {r}: luoi ngay khong day du {TOTAL_DAYS} ngay"

    df["split"] = assign_split(df["date"])

    # --- Kiem tra ket qua ---
    cfg = {}
    prev_end = None
    for name, start, end, n in SPLITS:
        p = df[(df["split"] == name) & (df["region"] == REGIONS[0])]
        assert len(p) == n, f"{name}: {len(p)} ngay, can {n}"
        assert p["date"].min() == pd.Timestamp(start) and p["date"].max() == pd.Timestamp(end), f"{name}: sai ngay dau/cuoi"
        if prev_end is not None:
            assert pd.Timestamp(start) == prev_end + pd.Timedelta(days=1), f"Ho/chong giua cac tap truoc {name}"
        prev_end = pd.Timestamp(end)
        m = df[df["split"] == name]
        usable = {}
        for h in HORIZONS:
            ok = is_valid_origin(m["date"], h) & m[f"y_h{h}"].notna()
            usable[f"y_h{h}"] = {"3_mien": int(ok.sum()), "moi_mien_xap_xi": int(round(ok.sum() / len(REGIONS)))}
        cfg[name] = {"start": start, "end": end, "n_days": n, "usable_origins_with_label": usable}
    assert sum(c["n_days"] for c in cfg.values()) == TOTAL_DAYS, "Tong ngay ba tap sai"
    assert not (df["date"] > pd.Timestamp(SPLITS[-1][2])).any(), "Co ngay sau ngay cuoi Test"

    config = {
        **cfg,
        "total_days": TOTAL_DAYS,
        "horizons_days": HORIZONS,
        "retrain_dates_in_test": RETRAIN_DATES,
        "retrain_rule": "Dau moi thang trong Test, cua so mo rong: huan luyen lai tren moi du lieu da co nhan den thoi diem do",
        "purge_rule": "Diem goc t chi dung neu t + h <= ngay cuoi cua tap chua t (h = 7, 14, 30)",
        "note": "Ngay gom ca hai dau. Scaler, STL, PCA chi fit tren Train. Chon tham so tren Val, khong chon tren Test.",
    }
    with open(outdir / "split_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"OK: {TOTAL_DAYS} ngay = " + " + ".join(f"{n} {name}" for name, _, _, n in SPLITS))
    for name, start, end, n in SPLITS:
        print(f"    {name:5s} {start} -> {end}  ({n} ngay)  diem goc co nhan: " +
              ", ".join(f"{k}={v['moi_mien_xap_xi']}/mien" for k, v in cfg[name]['usable_origins_with_label'].items()))
    print(f"    Da ghi: {outdir / 'split_config.json'}")


if __name__ == "__main__":
    main()
