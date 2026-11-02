
import pandas as pd
from pathlib import Path

# ==================================================
# 1. KHAI BAO DUONG DAN
# ==================================================

GROUPS_PATH = Path("data/raw/csv/feature_groups.csv")
DATA_PATH = Path("data/features/data_v1.csv")
OUTPUT_DIR = Path("data/features")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. DOC DU LIEU
# ==================================================

groups = pd.read_csv(GROUPS_PATH)
df = pd.read_csv(DATA_PATH)

print("=== KIEM TRA DU LIEU DAU VAO ===")
print("So nhom bien:", len(groups))
print("So dong data_v1:", len(df))
print("So cot data_v1:", len(df.columns))

# Kiem tra cau truc file cua Xuan Dung
required_columns = ["Unnamed: 0", "features"]

for col in required_columns:
    assert col in groups.columns, (
        f"Thieu cot {col} trong feature_groups.csv"
    )


# ==================================================
# 3. CHUYEN DOI BANG NHOM BIEN
# ==================================================

mapping = []

for _, row in groups.iterrows():

    group_name = row["Unnamed: 0"]

    # Vi du: E0_price -> E0
    group_id = group_name.split("_")[0]

    # Tach chuoi feature theo dau phay
    features = str(row["features"]).split(",")

    for feature in features:

        feature = feature.strip()

        # Bo qua ten cot rong
        if not feature:
            continue

        # Khong su dung retail_idx_mean theo README
        if feature == "retail_idx_mean":
            print("Loai feature: retail_idx_mean")
            continue

        mapping.append({
            "feature": feature,
            "group": group_id,
            "group_name": group_name
        })


# ==================================================
# 4. TAO DATAFRAME ANH XA
# ==================================================

feature_map = pd.DataFrame(mapping)

print("\n=== SO FEATURE THEO NHOM ===")
print(feature_map.groupby("group").size())


# ==================================================
# 5. KIEM TRA DU LIEU
# ==================================================

# Kiem tra du 6 nhom E0 - E5
expected_groups = {
    "E0", "E1", "E2", "E3", "E4", "E5"
}

actual_groups = set(feature_map["group"])

assert actual_groups == expected_groups, (
    f"Nhom bien khong dung: {actual_groups}"
)

# Kiem tra feature co ton tai trong data_v1
missing_features = (
    set(feature_map["feature"]) - set(df.columns)
)

assert not missing_features, (
    f"Feature khong ton tai: {missing_features}"
)

# Kiem tra feature bi trung lap
duplicates = feature_map[
    feature_map["feature"].duplicated()
]

assert duplicates.empty, (
    "Phat hien feature bi trung lap!"
)

# Khong dua nhan du bao vao feature
for target in ["y_h7", "y_h14", "y_h30"]:
    assert target not in feature_map["feature"].values

# Khong dua split va co hop le vao feature
excluded_cols = {
    "split",
    "valid_h7",
    "valid_h14",
    "valid_h30",
    "retail_idx_mean"
}

assert not (
    set(feature_map["feature"]) & excluded_cols
), "Feature chua cot khong duoc phep!"



# ==================================================
# 6. GAN FEATURE VAO CAC CAU HINH ABLATION C1-C6
# ==================================================

import json

CONFIG_PATH = OUTPUT_DIR / "ablation_config.json"

# Doc cau hinh ablation chinh thuc
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    ablation = json.load(f)

expected_order = [
    "E0", "E2", "E1", "E3", "E4", "E5"
]

# Kiem tra thu tu nhom
assert ablation["group_order"] == expected_order, (
    "Sai thu tu ablation theo README"
)

# Gan co 1/0 cho tung feature
for i in range(1, 7):

    config_name = f"c{i}"

    selected_groups = ablation["configs"][config_name]

    # Moi cau hinh phai cong don tu E0
    assert selected_groups == expected_order[:i], (
        f"Sai cau hinh {config_name}"
    )

    # 1: feature duoc su dung
    # 0: feature khong duoc su dung
    feature_map[config_name] = (
        feature_map["group"]
        .isin(selected_groups)
        .astype(int)
    )

# Kiem tra so feature trong tung cau hinh
expected_counts = {
    "c1": 18,
    "c2": 34,
    "c3": 47,
    "c4": 50,
    "c5": 51,
    "c6": 59
}

print("\n=== KIEM TRA ABLATION ===")

for config_name, expected_count in expected_counts.items():

    actual_count = int(feature_map[config_name].sum())

    print(f"{config_name}: {actual_count} features")

    assert actual_count == expected_count, (
        f"{config_name}: can {expected_count}, "
        f"nhung co {actual_count}"
    )

# Kiem tra so feature khong thay doi
assert len(feature_map) == 59, (
    "Tong so feature khong bang 59"
)

print("\nDAT: Anh xa feature va ablation hop le")

# ==================================================
# 7. LUU BANG ANH XA
# ==================================================

OUTPUT_FILE = OUTPUT_DIR / "feature_groups.csv"

feature_map.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ==================================================
# 8. THONG BAO KET QUA
# ==================================================

print("\n=== KET QUA P2 ===")

print("Tong so feature:", len(feature_map))

print("So feature theo nhom:")
print(feature_map.groupby("group").size())

print("\nDa tao file:")
print(OUTPUT_FILE)

print("\nDAT: Kiem tra feature_groups thanh cong!")
