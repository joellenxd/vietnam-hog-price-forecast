
import json
import pandas as pd
from pathlib import Path

# 1. Duong dan
BASE = Path("data/features")

# 2. Doc danh sach feature
features = pd.read_csv(BASE / "feature_groups.csv")

# 3. Doc cau hinh ablation
with open(
    BASE / "ablation_config.json",
    "r",
    encoding="utf-8"
) as f:
    ablation = json.load(f)

# 4. Thu tu nhom chinh thuc
expected_order = ["E0", "E2", "E1", "E3", "E4", "E5"]
expected_counts = [18, 34, 47, 50, 51, 59]

assert ablation["group_order"] == expected_order, (
    "Sai thu tu nhom ablation"
)

# 5. Kiem tra bang feature
assert features.shape == (59, 9), (
    "Sai kich thuoc feature_groups.csv"
)

assert features["feature"].notna().all(), (
    "Co ten feature bi thieu"
)

assert features["feature"].is_unique, (
    "Feature bi trung"
)

assert features["group"].isin(expected_order).all(), (
    "Co feature thuoc nhom khong hop le"
)

assert "retail_idx_mean" not in features["feature"].values, (
    "retail_idx_mean khong duoc dung de huan luyen"
)

# 6. Kiem tra tung cau hinh c1-c6
for i in range(1, 7):

    name = f"c{i}"
    groups = ablation["configs"][name]

    # Dung thu tu cong don
    assert groups == expected_order[:i], (
        f"{name}: sai thu tu cong don"
    )

    # Moi feature thuoc nhom duoc chon mang co 1
    expected = features["group"].isin(groups).astype(int)

    actual = pd.to_numeric(
        features[name],
        errors="raise"
    )

    # Chi chap nhan gia tri 0 va 1
    assert actual.notna().all(), (
        f"{name}: co gia tri thieu"
    )

    assert actual.isin([0, 1]).all(), (
        f"{name}: co gia tri khac 0/1"
    )

    # Kiem tra tung feature
    assert (actual == expected).all(), (
        f"{name}: anh xa feature sai nhom"
    )

    # Kiem tra tong so feature
    count = int(actual.sum())

    assert count == expected_counts[i - 1], (
        f"{name}: sai so feature"
    )

    print(
        f"{name}: {groups} | "
        f"{count} features - DAT"
    )

print("\nDAT: Cau hinh ablation va co c1-c6 hop le!")
