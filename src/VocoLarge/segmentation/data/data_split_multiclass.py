from pathlib import Path


def load_ids(path: Path) -> list[str]:
    """
    Supports both:
      - one ID per line
      - one comma-separated line
    """
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


def save_ids(path: Path, ids: list[str]):
    """
    Saves IDs as one comma-separated line.
    """
    path.write_text(",".join(ids))


# -------- CONFIG --------

allowed_ids_file = Path(
    "src/VocoLarge/segmentation/training_data_ids/training_multiclass_without_imn_ids/multiclass_without_imn_training.txt"
)

suffix = "multiclass_without_imn_training"

input_files = [
    Path("src/VocoLarge/segmentation/training_data_ids/train_ids.txt"),
    Path("src/VocoLarge/segmentation/training_data_ids/val_ids.txt"),
    Path("src/VocoLarge/segmentation/training_data_ids/test_ids.txt"),
]

# -------- RUN --------

allowed_ids = set(load_ids(allowed_ids_file))

for file_path in input_files:
    ids = load_ids(file_path)

    filtered_ids = [case_id for case_id in ids if case_id in allowed_ids]
    removed_ids = [case_id for case_id in ids if case_id not in allowed_ids]

    output_file = file_path.with_name(file_path.stem + f"_{suffix}.txt")
    removed_file = file_path.with_name(file_path.stem + f"_removed_not_{suffix}.txt")

    save_ids(output_file, filtered_ids)
    save_ids(removed_file, removed_ids)

    print(
        f"{file_path}: original={len(ids)}, "
        f"kept={len(filtered_ids)}, removed={len(removed_ids)}"
    )
    print(f"  saved kept:    {output_file}")
    print(f"  saved removed: {removed_file}")