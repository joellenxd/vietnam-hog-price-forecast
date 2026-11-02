
from pathlib import Path

import pandas as pd

from split import assign_split, is_valid_origin


# =====================================
# 1. DUONG DAN
# =====================================

DATA_PATH = Path(
    "data/raw/parquet/supervised_region_daily.parquet"
)

OUTPUT_DIR = Path("data/features")
OUTPUT_FILE = OUTPUT_DIR / "data_v1.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================
# 2. DOC DU LIEU P1
# =====================================

df = pd.read_parquet(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

# Khong duoc trung ngay va mien
assert not df.duplicated(["date", "region"]).any(), (
    "Trung khoa date-region!"
)

print("So dong dau vao:", len(df))
print("So cot dau vao:", len(df.columns))


# =====================================
# 3. GAN TRAIN / VAL / TEST
# =====================================

df["split"] = assign_split(df["date"]).to_numpy()

assert df["split"].isin(
    ["train", "val", "test"]
).all(), "Co dong khong thuoc Train/Val/Test!"

print("\nSo dong moi tap:")
print(df["split"].value_counts())


# =====================================
# 4. TAO CO NHAN HOP LE
# =====================================

for h in [7, 14, 30]:

    label_col = f"y_h{h}"
    valid_col = f"valid_h{h}"

    assert label_col in df.columns, (
        f"Thieu nhan {label_col}"
    )

    valid_date = is_valid_origin(
        df["date"], h
    ).to_numpy()

    valid_label = df[label_col].notna().to_numpy()

    df[valid_col] = valid_date & valid_label

    print(f"\nNhan hop le h={h}:")
    print(df.groupby("split")[valid_col].sum())


# =====================================
# 5. KIEM TRA DU LIEU P2
# =====================================

assert len(df) == 1458, "Sai so dong data_v1!"

assert len(df.columns) == 74, (
    "Sai so cot data_v1!"
)

assert not df.duplicated(
    ["date", "region"]
).any(), "Trung khoa date-region!"


# =====================================
# 6. BAO VE PHIEN BAN DATA_V1
# =====================================

if OUTPUT_FILE.exists():

    print("\nDa ton tai data_v1.csv.")
    print("Dang doi chieu voi du lieu tao lai...")

    old_df = pd.read_csv(OUTPUT_FILE)

    new_df = df.copy()
    new_df["date"] = new_df["date"].dt.strftime(
        "%Y-%m-%d"
    )

    # Sap xep cot theo file hien co
    assert set(old_df.columns) == set(new_df.columns), (
        "Danh sach cot cua data_v1 da thay doi!"
    )

    new_df = new_df[old_df.columns]

    # Doi chieu du lieu voi dung sai so thuc
    pd.testing.assert_frame_equal(
        old_df,
        new_df,
        check_dtype=False,
        check_exact=False,
        rtol=1e-8,
        atol=1e-6
    )

    print("\nDAT: data_v1 khop voi du lieu tao lai.")
    print("Khong ghi de data_v1.csv.")

else:

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nDA TAO:", OUTPUT_FILE)


print("\nKICH THUOC DU LIEU:", df.shape)
print("HOAN THANH KIEM TRA DATA_V1")
