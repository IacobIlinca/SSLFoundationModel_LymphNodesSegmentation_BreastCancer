from pathlib import Path


def load_ids(path: Path) -> list[str]:
    content = path.read_text().strip()
    if not content:
        return []

    ids = []
    for x in content.replace("\n", ",").split(","):
        x = x.strip()
        if x:
            ids.append(x)

    seen = set()
    unique = []
    for x in ids:
        if x not in seen:
            unique.append(x)
            seen.add(x)

    return unique


# -------- CONFIG --------

new_ids_file = Path(
    "src/VocoLarge/segmentation/training_data_ids/training_level_2_ids/level_2_training.txt"
)

old_split_files = [
    Path("src/VocoLarge/segmentation/training_data_ids/train_ids.txt"),
    Path("src/VocoLarge/segmentation/training_data_ids/val_ids.txt"),
    Path("src/VocoLarge/segmentation/training_data_ids/test_ids.txt"),
]

out_file = Path(
    "src/VocoLarge/segmentation/training_data_ids/level_2_ids_not_in_old_splits.txt"
)

# -------- RUN --------

new_ids = load_ids(new_ids_file)

old_ids = set()
for file_path in old_split_files:
    old_ids.update(load_ids(file_path))

missing_from_old = [case_id for case_id in new_ids if case_id not in old_ids]

out_file.write_text(",".join(missing_from_old))

print(f"New IDs total:              {len(new_ids)}")
print(f"Old split IDs total:        {len(old_ids)}")
print(f"New IDs not in old splits:  {len(missing_from_old)}")
print(f"Saved: {out_file}")

if missing_from_old:
    print("\nIDs not in old splits:")
    for case_id in missing_from_old:
        print(case_id)