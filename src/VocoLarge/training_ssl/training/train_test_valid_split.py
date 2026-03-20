import os
import random
from pathlib import Path

# ====== CONFIG ======
root_dir = "/mnt/data/flaviu/rtnation_02_02"   # contains id1/, id2/, ...
output_dir = "../train_valid_test_split/"
seed = 42
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15
# ====================

def get_patient_ids(root_dir):
    """
    Returns list of patient folder names (ids).
    Assumes structure:
        root_dir/
            id1/
            id2/
            ...
    """
    root = Path(root_dir)
    ids = [p.name for p in root.iterdir() if p.is_dir()]
    return ids


def split_ids(ids, train_ratio, val_ratio, test_ratio, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    random.seed(seed)
    ids = sorted(ids)  # optional, for consistency before shuffling
    random.shuffle(ids)

    n = len(ids)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    return train_ids, val_ids, test_ids


def save_split(ids, path):
    with open(path, "w") as f:
        f.write(",".join(ids))   # comma-separated


def main():
    os.makedirs(output_dir, exist_ok=True)

    ids = get_patient_ids(root_dir)
    print(f"[INFO] Found {len(ids)} patients")

    train_ids, val_ids, test_ids = split_ids(
        ids, train_ratio, val_ratio, test_ratio, seed
    )

    print(f"[INFO] Train: {len(train_ids)}")
    print(f"[INFO] Val:   {len(val_ids)}")
    print(f"[INFO] Test:  {len(test_ids)}")

    save_split(train_ids, os.path.join(output_dir, "train_ids.txt"))
    save_split(val_ids,   os.path.join(output_dir, "val_ids.txt"))
    save_split(test_ids,  os.path.join(output_dir, "test_ids.txt"))

    print("[INFO] Splits saved!")


if __name__ == "__main__":
    main()