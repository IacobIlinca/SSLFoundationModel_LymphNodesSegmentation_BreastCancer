def load_ids(file_path):
    """Load comma-separated IDs from a single line file."""
    with open(file_path, "r") as f:
        content = f.read().strip()
        return set(x.strip() for x in content.split(",") if x.strip())


def save_ids(file_path, ids):
    """Save IDs back as a single comma-separated line."""
    with open(file_path, "w") as f:
        f.write(",".join(sorted(ids)))


# ----------- CONFIG -----------

workshop_file = "src/VocoLarge/segmentation/training_data_ids/workshop_ids.txt"

input_files = [
    "src/VocoLarge/segmentation/training_data_ids/test_ids.txt",
    "src/VocoLarge/segmentation/training_data_ids/train_ids.txt",
    "src/VocoLarge/segmentation/training_data_ids/val_ids.txt",
]

# ----------- RUN -----------

exclude_ids = load_ids(workshop_file)

for f in input_files:
    ids = load_ids(f)

    filtered_ids = ids - exclude_ids

    output_file = f.replace(".txt", "_filtered.txt")
    save_ids(output_file, filtered_ids)

    print(f"{f}: original={len(ids)}, filtered={len(filtered_ids)}, removed={len(ids) - len(filtered_ids)}")