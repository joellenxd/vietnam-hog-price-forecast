
import pandas as pd
from pathlib import Path

# 1. Duong dan
BASE = Path("data/features")

DATA_PATH = BASE / "data_v1.csv"
GROUP_PATH = BASE / "feature_groups.csv"
OUTPUT_PATH = BASE / "data_dictionary_v1.md"

# 2. Doc du lieu
df = pd.read_csv(DATA_PATH)
groups = pd.read_csv(GROUP_PATH)

# 3. Tao bang anh xa feature -> group
group_map = dict(
    zip(groups["feature"], groups["group"])
)

targets = {"y_h7", "y_h14", "y_h30"}
flags = {
    "split", "valid_h7", "valid_h14", "valid_h30"
}

metadata = {
    "date", "region", "price", "price_ffill",
    "is_observed", "is_imputed", "is_sunday_copy"
}

# 4. Phan loai tung cot
rows = []

for col in df.columns:

    if col in targets:
        role = "Target"

    elif col in flags:
        role = "Split/Flag"

    elif col in metadata:
        role = "Metadata"

    elif col == "retail_idx_mean":
        role = "Excluded"

    elif col in group_map:
        role = "Feature"

    else:
        role = "Can kiem tra"

    rows.append({
        "column": col,
        "dtype": str(df[col].dtype),
        "role": role,
        "group": group_map.get(col, "-"),
        "missing": int(df[col].isna().sum())
    })

dictionary = pd.DataFrame(rows)

# 5. Tao file Markdown
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

    f.write("# Tu dien du lieu P2 - data_v1\n\n")

    f.write("## 1. Tong quan\n\n")
    f.write(f"- So dong: {len(df)}\n")
    f.write(f"- So cot: {len(df.columns)}\n")
    f.write("- Nguon: supervised_region_daily\n")
    f.write("- Split: train / val / test\n\n")

    f.write("## 2. Danh sach cot\n\n")
    f.write(
        "| Column | Data type | Role | Group | NaN |\n"
    )
    f.write(
        "|---|---|---|---|---:|\n"
    )

    for _, row in dictionary.iterrows():
        f.write(
            f"| {row['column']} | {row['dtype']} | "
            f"{row['role']} | {row['group']} | "
            f"{row['missing']} |\n"
        )

    f.write("\n## 3. Luu y\n\n")
    f.write(
        "- Khong dung y_h7, y_h14, y_h30 "
        "lam feature.\n"
    )
    f.write(
        "- retail_idx_mean khong duoc dua "
        "vao cau hinh ablation.\n"
    )
    f.write(
        "- Chi su dung nhan khi valid_h "
        "tuong ung bang True.\n"
    )
    f.write(
        "- Can kiem tra gia tri NaN va "
        "thoi diem cong bo du lieu.\n"
    )

print("DA TAO:", OUTPUT_PATH)

print("\nTHONG KE ROLE:")
print(dictionary["role"].value_counts())

print("\nCAC COT CAN KIEM TRA:")
print(
    dictionary[
        dictionary["role"] == "Can kiem tra"
    ]["column"].tolist()
)


# ==========================================
# GHEP PHAN GIAI THICH BO SUNG VAO TU DIEN
# ==========================================

from pathlib import Path

notes_path = Path("data/features/data_dictionary_notes.md")

# Bat buoc phai co file ghi chu
if not notes_path.exists():
    raise FileNotFoundError(
        f"Khong tim thay file ghi chu: {notes_path}"
    )

# Doc phan giai thich tu muc 4 den muc 8
notes = notes_path.read_text(encoding="utf-8")

# Kiem tra noi dung khong rong
if not notes.strip():
    raise ValueError("File ghi chu dang rong!")

# Ghep vao cuoi tu dien vua duoc tao
with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
    f.write("\n\n")
    f.write(notes.strip())
    f.write("\n")

print("DAT: Da ghep phan giai thich vao tu dien P2")
