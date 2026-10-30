# """split.py - Chia clean.csv theo thoi gian thanh Train / Val / Test.
# Luu y ve ro ri du lieu: khi tao bien tre (lag), tinh tren TOAN BO chuoi theo thu tu thoi gian roi moi gan nhan split.
# """
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# (ten tap, ngay dau, ngay cuoi, so tuan) - ngay la Chu nhat dai dien tuan, gom ca hai dau
SPLITS = [
    ("train", "2016-01-03", "2021-06-20", 286),
    ("val", "2021-06-27", "2022-06-19", 52),
    ("test", "2022-06-26", "2024-05-19", 100),
]
TOTAL_WEEKS = sum(s[3] for s in SPLITS)  # 438


def assign_split(dates):
    """Tra ve Series nhan 'train'/'val'/'test' cho tung ngay."""
    dates = pd.to_datetime(pd.Series(dates))
    out = pd.Series(index=dates.index, dtype="object")
    for name, start, end, _ in SPLITS:
        out[(dates >= start) & (dates <= end)] = name
    assert out.notna().all(), "Co ngay khong thuoc tap nao"
    return out


def main():
    ap = argparse.ArgumentParser(description="Chia Train/Val/Test theo thoi gian")
    ap.add_argument("--input", default="data/clean/clean.csv")
    ap.add_argument("--outdir", default="data/split")
    ap.add_argument("--save", action="store_true", help="ghi them train.csv, val.csv, test.csv")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        sys.exit(f"Khong thay file: {src}. Chay clean.py truoc.")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src, parse_dates=["date"])

    # --- Kiem tra dau vao ---
    assert df["date"].is_monotonic_increasing and df["date"].is_unique, "Ngay phai tang dan, khong trung"
    assert (df["date"].diff().dropna().dt.days == 7).all(), "Cac tuan phai cach deu 7 ngay"
    assert len(df) == TOTAL_WEEKS, f"So dong {len(df)} != {TOTAL_WEEKS}"

    df["split"] = assign_split(df["date"])
    parts = {name: df[df["split"] == name] for name, *_ in SPLITS}

    # --- Kiem tra ket qua ---
    for name, start, end, n in SPLITS:
        p = parts[name]
        assert len(p) == n, f"{name}: {len(p)} tuan, can {n}"
        assert p["date"].iloc[0] == pd.Timestamp(start), f"{name}: ngay dau {p['date'].iloc[0].date()} != {start}"
        assert p["date"].iloc[-1] == pd.Timestamp(end), f"{name}: ngay cuoi {p['date'].iloc[-1].date()} != {end}"
    assert sum(len(p) for p in parts.values()) == len(df), "Tong ba tap khac tong so dong"
    assert parts["train"]["date"].max() < parts["val"]["date"].min(), "Train chong len Val"
    assert parts["val"]["date"].max() < parts["test"]["date"].min(), "Val chong len Test"
    assert not set(parts["train"]["date"]) & set(parts["val"]["date"]) \
        and not set(parts["val"]["date"]) & set(parts["test"]["date"]) \
        and not set(parts["train"]["date"]) & set(parts["test"]["date"]), "Ba tap co ngay trung nhau"
    # ba tap lien tiep, khong ho gap ngay nao: tuan cuoi tap truoc + 7 ngay = tuan dau tap sau
    assert parts["train"]["date"].iloc[-1] + pd.Timedelta(days=7) == parts["val"]["date"].iloc[0], "Ho giua Train va Val"
    assert parts["val"]["date"].iloc[-1] + pd.Timedelta(days=7) == parts["test"]["date"].iloc[0], "Ho giua Val va Test"

    # --- Ghi file cau hinh ---
    config = {
        name: {"start": start, "end": end, "n_weeks": n} for name, start, end, n in SPLITS
    }
    config["total_weeks"] = TOTAL_WEEKS
    config["note"] = "Chia theo thu tu thoi gian, ngay gom ca hai dau. Scaler/STL/PCA chi fit tren Train."
    with open(outdir / "split_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    if args.save:
        for name, p in parts.items():
            p.drop(columns="split").to_csv(outdir / f"{name}.csv", index=False, date_format="%Y-%m-%d")

    print(f"OK: {TOTAL_WEEKS} tuan = " + " + ".join(f"{n} {name}" for name, _, _, n in SPLITS))
    for name, start, end, n in SPLITS:
        print(f"    {name:5s} {start} -> {end}  ({n} tuan)")
    print(f"    Da ghi: {outdir / 'split_config.json'}" + (" + train/val/test.csv" if args.save else ""))


if __name__ == "__main__":
    main()
