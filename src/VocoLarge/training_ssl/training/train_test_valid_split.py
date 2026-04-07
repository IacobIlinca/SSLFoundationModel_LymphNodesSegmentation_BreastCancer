import os
import random
from pathlib import Path

# ====== CONFIG ======
root_dir = "/mnt/data/flaviu/rtnation_02_02"   # contains id1/, id2/, ...
output_dir = "../train_valid_test_split/"

workshop_ids_file = "../train_valid_test_split/workshop_not_anon_id.txt"          # ids to exclude completely
segmentation_test_ids_file = "../train_valid_test_split/seg_test_ids.txt" # ids that must be in SSL test

seed = 42
train_ratio = 0.70
val_ratio = 0.20
test_ratio = 0.10
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


def read_ids_file(path):
    """
    Reads a txt file containing comma-separated ids on a single line.
    Also tolerates newlines/spaces.
    """
    with open(path, "r") as f:
        content = f.read().strip()

    if not content:
        return []

    # normalize both commas and newlines
    raw_ids = content.replace("\n", ",").split(",")
    ids = [x.strip() for x in raw_ids if x.strip()]
    return ids


def split_ids_with_forced_test(
    all_ids,
    forced_test_ids,
    train_ratio,
    val_ratio,
    test_ratio,
    seed=42,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    all_ids = sorted(set(all_ids))
    forced_test_ids = sorted(set(forced_test_ids))

    # remaining ids after reserving forced test
    remaining_ids = [x for x in all_ids if x not in forced_test_ids]

    random.seed(seed)
    random.shuffle(remaining_ids)

    n_total = len(all_ids)
    target_test_total = int(round(n_total * test_ratio))

    # how many extra test ids we still want beyond forced_test_ids
    extra_test_needed = max(0, target_test_total - len(forced_test_ids))
    extra_test_needed = min(extra_test_needed, len(remaining_ids))

    extra_test_ids = remaining_ids[:extra_test_needed]
    leftover_ids = remaining_ids[extra_test_needed:]

    # split leftover into train / val using normalized train-val proportions
    train_val_total = train_ratio + val_ratio
    if train_val_total == 0:
        train_ids = []
        val_ids = []
    else:
        train_fraction_within_leftover = train_ratio / train_val_total
        n_train = int(round(len(leftover_ids) * train_fraction_within_leftover))
        train_ids = leftover_ids[:n_train]
        val_ids = leftover_ids[n_train:]

    test_ids = sorted(forced_test_ids + extra_test_ids)
    train_ids = sorted(train_ids)
    val_ids = sorted(val_ids)

    return train_ids, val_ids, test_ids


def save_split(ids, path):
    with open(path, "w") as f:
        f.write(",".join(ids))   # comma-separated


def main():
    os.makedirs(output_dir, exist_ok=True)

    all_ids = get_patient_ids(root_dir)
    all_ids_set = set(all_ids)
    print(f"[INFO] Found {len(all_ids)} patients in root_dir")

    workshop_ids = read_ids_file(workshop_ids_file)
    seg_test_ids = read_ids_file(segmentation_test_ids_file)

    workshop_ids_set = set(workshop_ids)
    seg_test_ids_set = set(seg_test_ids)

    # warn about ids not found
    missing_workshop = sorted(workshop_ids_set - all_ids_set)
    missing_seg_test = sorted(seg_test_ids_set - all_ids_set)

    if missing_workshop:
        print(f"[WARN] {len(missing_workshop)} workshop ids not found in root_dir")
        print(f"[WARN] Missing workshop ids: {missing_workshop}")

    if missing_seg_test:
        print(f"[WARN] {len(missing_seg_test)} segmentation test ids not found in root_dir")
        print(f"[WARN] Missing segmentation test ids: {missing_seg_test}")

    # keep only ids that actually exist
    workshop_ids_set &= all_ids_set
    seg_test_ids_set &= all_ids_set

    # segmentation test ids cannot also be workshop-excluded
    overlap_excluded_forced = sorted(workshop_ids_set & seg_test_ids_set)
    if overlap_excluded_forced:
        print(f"[WARN] {len(overlap_excluded_forced)} ids are in BOTH workshop exclusion and segmentation test.")
        print("[WARN] Since workshop ids must not be included at all, they will be excluded.")
        print(f"[WARN] Overlap ids: {overlap_excluded_forced}")

    seg_test_ids_set -= workshop_ids_set

    # exclude workshop ids completely
    usable_ids = sorted(all_ids_set - workshop_ids_set)

    print(f"[INFO] Workshop ids excluded: {len(workshop_ids_set)}")
    print(f"[INFO] Remaining usable ids: {len(usable_ids)}")
    print(f"[INFO] Forced segmentation test ids: {len(seg_test_ids_set)}")

    train_ids, val_ids, test_ids = split_ids_with_forced_test(
        usable_ids,
        seg_test_ids_set,
        train_ratio,
        val_ratio,
        test_ratio,
        seed=seed,
    )

    # sanity checks
    assert len(set(train_ids) & set(val_ids)) == 0
    assert len(set(train_ids) & set(test_ids)) == 0
    assert len(set(val_ids) & set(test_ids)) == 0
    assert set(train_ids) | set(val_ids) | set(test_ids) == set(usable_ids)
    assert set(seg_test_ids_set).issubset(set(test_ids))

    print(f"[INFO] Train: {len(train_ids)}")
    print(f"[INFO] Val:   {len(val_ids)}")
    print(f"[INFO] Test:  {len(test_ids)}")

    save_split(train_ids, os.path.join(output_dir, "train_split_ids.txt"))
    save_split(val_ids,   os.path.join(output_dir, "val_split_ids.txt"))
    save_split(test_ids,  os.path.join(output_dir, "test_split_ids.txt"))

    print("[INFO] Splits saved!")


if __name__ == "__main__":
    main()