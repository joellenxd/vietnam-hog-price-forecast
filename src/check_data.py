# """check_data.py - Kiem tra bo du lieu gia heo hoi Viet Nam da crawl.
# Lam 3 viec: doi chieu CHECKSUMS.sha256, doi chieu so dong/khoa/luoi ngay/nhan voi README, ghi bao cao data/checks/data_check.md.
# """
import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

EXPECTED_ROWS = {
    "pig_price_province_daily": 15773,
    "panel_region_daily": 1458,
    "panel_province_daily": 16524,
    "supervised_region_daily": 1458,
    "supervised_province_daily": 16524,
}
EXPECT = {"start": "2025-07-02", "end": "2026-10-30", "days": 486}
OBSERVED_DAYS = {"Bac": 464, "Trung": 464, "Nam": 462}      # quality_report.md muc 1
LABELS = {"y_h7": 1369, "y_h14": 1354, "y_h30": 1306}        # README muc 4
UNITS = [
    ("price, price_t, lag_*, roll_*", "dong/kg"),
    ("wb_maize/soybeans/soybean_meal/wheat/rice_usd_t", "USD/tan (World Bank, theo thang)"),
    ("wb_crude_usd_bbl", "USD/thung (theo thang)"),
    ("imp_*_volume_kt", "nghin tan (NSO, theo thang)"),
    ("imp_*_value_musd, imp_feed_value_musd", "trieu USD (NSO, theo thang)"),
    ("imp_*_unit_usd_t", "USD/tan (NSO, theo thang)"),
    ("temp_mean, temp_max", "do C"),
    ("rh_mean / rain_mm", "% / mm"),
    ("usd_vnd", "VND moi USD"),
    ("asf_active_localities / asf_active_share", "so dia phuong co dich / ty le tren tong so tinh"),
    ("retail_idx_*", "dong/kg (gia ban le; ngung dang sau 03/08/2026)"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--out", default="data/checks/data_check.md")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.exists():
        sys.exit(f"Khong thay thu muc {root}. Chep bo du lieu da crawl vao day (xem README).")

    lines, errors, warnings = [], [], []

    # 1. CHECKSUMS
    lines.append("## 1. Checksum (CHECKSUMS.sha256)\n")
    ck = root / "CHECKSUMS.sha256"
    if not ck.exists():
        warnings.append("Khong co CHECKSUMS.sha256")
    else:
        ok = bad = miss = 0
        for ln in ck.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            digest, name = ln.split(None, 1)
            name = name.strip().lstrip("*")
            p = root / name
            if not p.exists():
                miss += 1
                lines.append(f"- THIEU file: `{name}`")
            elif sha256(p) == digest:
                ok += 1
            else:
                bad += 1
                errors.append(f"Checksum SAI: {name}")
                lines.append(f"- SAI checksum: `{name}`")
        lines.append(f"\nKhop: {ok}; sai: {bad}; khong co file: {miss}.\n")
        if miss:
            warnings.append(f"{miss} file trong CHECKSUMS khong co trong thu muc (khong sao neu chi dung parquet)")

    # 2. So dong cac bang parquet
    lines.append("## 2. So dong cac bang\n\n| Bang | Dong thuc te | Dong theo README | Ket qua |\n|---|---|---|---|")
    tables = {}
    for name, exp in EXPECTED_ROWS.items():
        p = root / "parquet" / f"{name}.parquet"
        if not p.exists():
            lines.append(f"| {name} | khong co file | {exp} | THIEU |")
            warnings.append(f"Thieu {p}")
            continue
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        tables[name] = df
        good = len(df) == exp
        if not good:
            errors.append(f"{name}: {len(df)} dong, README ghi {exp}")
        lines.append(f"| {name} | {len(df)} | {exp} | {'OK' if good else 'LECH'} |")

    # 3. Bang chinh supervised_region_daily
    s = tables.get("supervised_region_daily")
    if s is None:
        errors.append("Khong doc duoc supervised_region_daily")
    else:
        lines.append("\n## 3. supervised_region_daily\n")
        assert_list = []
        def chk(cond, msg):
            assert_list.append((cond, msg))
            if not cond:
                errors.append(msg)
        chk(not s.duplicated(["date", "region"]).any(), "Co khoa (date, region) trung")
        chk(str(s["date"].min().date()) == EXPECT["start"] and str(s["date"].max().date()) == EXPECT["end"], "Khoang ngay khac 02/07/2025 - 30/10/2026")
        for r, g in s.groupby("region"):
            chk(len(g) == EXPECT["days"], f"Mien {r}: {len(g)} ngay, can {EXPECT['days']}")
            obs = int(g["is_observed"].sum())
            chk(obs == OBSERVED_DAYS[r], f"Mien {r}: {obs} ngay co gia, quality_report ghi {OBSERVED_DAYS[r]}")
        for c, n in LABELS.items():
            chk(int(s[c].notna().sum()) == n, f"{c}: {int(s[c].notna().sum())} nhan, README ghi {n}")
        agec = [c for c in s.columns if c.endswith("_age_days")]
        for c in agec:
            chk(not (s[c] < 0).any(), f"{c} co gia tri am (ro ri thong tin tuong lai)")
        px = s["price"].dropna()
        chk(px.between(20000, 120000).all(), "Co gia ngoai [20.000; 120.000] dong/kg")
        no_price = int((s[s["region"] == "Bac"]["is_observed"] == 0).sum())
        lines.append(f"- {sum(1 for c,_ in assert_list if c)}/{len(assert_list)} kiem tra dat.")
        lines.append(f"- Ngay khong co bao gia (mien Bac): {no_price} (Tet, le, mot so cuoi tuan; nguon khong dang bai, khong phai loi crawl).")
        lines.append(f"- Chu nhat chep lai thu Bay (`is_sunday_copy` = 1): {int(s[s['region']=='Bac']['is_sunday_copy'].sum())} ngay/mien. Loc `is_sunday_copy == 0` khi danh gia.")
        lines.append(f"- Khoang gia: {px.min():,.0f} - {px.max():,.0f} dong/kg.")
        lines.append("- Nhan `y_h*` va cot `price` bi NaN o ngay khong co bao gia, khi huan luyen loc `y_h*.notna()`.")
        lines.append("- Cot `retail_idx_mean`: " + ", ".join(
            f"{n} {s[(s['date']>=a)&(s['date']<=e)]['retail_idx_mean'].isna().mean()*100:.0f}% NaN"
            for n, a, e in [("Train", "2025-07-02", "2026-03-31"), ("Val", "2026-04-01", "2026-06-30"), ("Test", "2026-07-01", "2026-10-30")])
            + " (nguon ngung dang; khong nen dung lam feature).")

    lines.append("\n## 4. Don vi (theo data_dictionary.md cua bo crawl)\n\n| Cot | Don vi |\n|---|---|")
    lines += [f"| {a} | {b} |" for a, b in UNITS]

    head = ["# Bao cao kiem tra bo du lieu", "", f"Thu muc: `{root}`", ""]
    status = "DAT" if not errors else f"CO {len(errors)} LOI"
    head += [f"**Ket luan: {status}.**", ""]
    if errors:
        head += ["Loi:"] + [f"- {e}" for e in errors] + [""]
    if warnings:
        head += ["Luu y:"] + [f"- {w}" for w in warnings] + [""]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(head + lines) + "\n", encoding="utf-8")
    print(f"{status}. Da ghi {out}")
    for e in errors:
        print("  LOI:", e)
    for w in warnings:
        print("  Luu y:", w)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
