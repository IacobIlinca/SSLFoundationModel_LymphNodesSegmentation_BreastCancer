import os
from pathlib import Path
import nibabel as nib

# ====== CONFIG ======
root_dir = "/mnt/data/ilinca/structured_cases_14_16/"
output_dir = "../training_data_ids"
excluded_ids_filename = "excluded_1024_xy_ids.txt"
# ====================


def get_patient_ids(root_dir: str):
    root = Path(root_dir)
    if not root.exists():
        raise RuntimeError(f"root_dir not found: {root_dir}")

    ids = [p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    return ids


def filter_out_1024_xy(ids, root_dir: str):
    kept_ids = []
    excluded_ids = []

    for case_id in ids:
        img_path = os.path.join(root_dir, case_id, "image.nii.gz")
        if not os.path.exists(img_path):
            raise RuntimeError(f"Missing image file: {img_path}")

        shape = nib.load(img_path).shape[:3]

        if shape[0] == 1024 and shape[1] == 1024:
            excluded_ids.append(case_id)
        else:
            kept_ids.append(case_id)

    return kept_ids, excluded_ids


def save_excluded_ids(ids, path):
    with open(path, "w") as f:
        for case_id in ids:
            f.write(case_id + "\n")


def load_split(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Split file not found: {path}")

    with open(path, "r") as f:
        text = f.read().strip()

    if not text:
        return []

    return [x.strip() for x in text.split(",") if x.strip()]


def save_split(ids, path):
    with open(path, "w") as f:
        f.write(",".join(ids))


def remove_excluded_from_split(split_ids, excluded_ids):
    excluded_set = set(excluded_ids)
    kept = [case_id for case_id in split_ids if case_id not in excluded_set]
    removed = [case_id for case_id in split_ids if case_id in excluded_set]
    return kept, removed


def main():
    os.makedirs(output_dir, exist_ok=True)

    all_ids = get_patient_ids(root_dir)
    print(f"[INFO] Found {len(all_ids)} patients before filtering")

    _, excluded_ids = filter_out_1024_xy(all_ids, root_dir)
    print(f"[INFO] Found {len(excluded_ids)} patients with 1024x1024 XY images")

    excluded_ids_path = os.path.join(output_dir, excluded_ids_filename)
    save_excluded_ids(excluded_ids, excluded_ids_path)
    print(f"[INFO] Saved excluded IDs to: {excluded_ids_path}")

    train_path = os.path.join(output_dir, "train_ids.txt")
    val_path = os.path.join(output_dir, "val_ids.txt")
    test_path = os.path.join(output_dir, "test_ids.txt")

    train_ids = load_split(train_path)
    val_ids = load_split(val_path)
    test_ids = load_split(test_path)

    print(f"[INFO] Original Train: {len(train_ids)}")
    print(f"[INFO] Original Val:   {len(val_ids)}")
    print(f"[INFO] Original Test:  {len(test_ids)}")

    train_ids_new, train_removed = remove_excluded_from_split(train_ids, excluded_ids)
    val_ids_new, val_removed = remove_excluded_from_split(val_ids, excluded_ids)
    test_ids_new, test_removed = remove_excluded_from_split(test_ids, excluded_ids)

    save_split(train_ids_new, train_path)
    save_split(val_ids_new, val_path)
    save_split(test_ids_new, test_path)

    print(f"[INFO] Removed from Train: {len(train_removed)}")
    print(f"[INFO] Removed from Val:   {len(val_removed)}")
    print(f"[INFO] Removed from Test:  {len(test_removed)}")

    print(f"[INFO] Updated Train: {len(train_ids_new)}")
    print(f"[INFO] Updated Val:   {len(val_ids_new)}")
    print(f"[INFO] Updated Test:  {len(test_ids_new)}")

    print("[INFO] Split files updated!")


if __name__ == "__main__":
    main()