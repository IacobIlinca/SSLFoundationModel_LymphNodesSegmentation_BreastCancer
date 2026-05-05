import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--summary_csv",
        required=True,
        help="Path to required_multiclass_nodes_summary.csv",
    )

    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help=(
            "Labels to extract, e.g. level2 level3 level4 interpectoral imn. "
            "A case is selected only if it has masks for all given labels."
        ),
    )

    parser.add_argument(
        "--out_dir",
        default="src/VocoLarge/segmentation/training_data_ids",
        help="Directory where the output ID file should be saved.",
    )

    parser.add_argument(
        "--output_name",
        default="",
        help="Optional output filename. If empty, uses <labels>_training.txt",
    )

    parser.add_argument(
        "--comma_separated",
        action="store_true",
        help="Save IDs as one comma-separated line instead of one ID per line.",
    )

    return parser.parse_args()


def save_ids(path: Path, ids: list[str], comma_separated: bool):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        if comma_separated:
            f.write(",".join(ids))
        else:
            for case_id in ids:
                f.write(case_id + "\n")


def main():
    args = parse_args()

    summary_csv = Path(args.summary_csv)
    out_dir = Path(args.out_dir)

    labels = [label.strip().lower() for label in args.labels]

    n_cols = [f"n_{label}_masks" for label in labels]
    masks_cols = [f"{label}_masks" for label in labels]

    label_name = "_and_".join(labels)

    if args.output_name:
        output_name = args.output_name
    else:
        output_name = f"{label_name}_training.txt"

    output_path = out_dir / output_name

    selected_ids = []
    selected_rows_debug = []

    with open(summary_csv, "r", newline="") as f:
        reader = csv.DictReader(f)

        if "case_id" not in reader.fieldnames:
            raise ValueError("CSV must contain a 'case_id' column.")

        missing_cols = [col for col in n_cols if col not in reader.fieldnames]
        if missing_cols:
            raise ValueError(
                f"CSV is missing required columns: {missing_cols}. "
                f"Available columns are: {reader.fieldnames}"
            )

        for row in reader:
            case_id = row["case_id"].strip()

            label_counts = {}

            for label, n_col in zip(labels, n_cols):
                try:
                    n_masks = int(row[n_col])
                except ValueError:
                    n_masks = 0

                label_counts[label] = n_masks

            has_all_labels = all(n_masks > 0 for n_masks in label_counts.values())

            if has_all_labels:
                selected_ids.append(case_id)

                debug_row = {
                    "case_id": case_id,
                }

                for label, n_col, masks_col in zip(labels, n_cols, masks_cols):
                    debug_row[n_col] = label_counts[label]
                    debug_row[masks_col] = row.get(masks_col, "")

                selected_rows_debug.append(debug_row)

    save_ids(output_path, selected_ids, args.comma_separated)

    debug_csv = out_dir / f"{label_name}_training_selected_masks.csv"

    with open(debug_csv, "w", newline="") as f:
        fieldnames = ["case_id"]

        for n_col, masks_col in zip(n_cols, masks_cols):
            fieldnames.extend([n_col, masks_col])

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows_debug)

    print("[DONE]")
    print(f"Labels: {labels}")
    print(f"Selection rule: case must have masks for ALL labels")
    print(f"Selected cases: {len(selected_ids)}")
    print(f"Saved IDs: {output_path}")
    print(f"Saved selected-mask debug CSV: {debug_csv}")


if __name__ == "__main__":
    main()