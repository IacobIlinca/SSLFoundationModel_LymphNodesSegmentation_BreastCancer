import argparse

from src.VocoLarge.segmentation.utils.data_audit import audit_dataset


def main():
    parser = argparse.ArgumentParser("NIfTI Data Audit")
    parser.add_argument("--root_dir", type=str, required=True, default=None, help="Root directory to search.")
    parser.add_argument("--pattern", type=str, default="**/image.nii.gz", help="Glob pattern under root_dir.")
    parser.add_argument("--list_file", type=str, default=None, help="Text file with one image path per line.")
    parser.add_argument("--max_cases", type=int, default=50, help="Limit number of images to audit (None=all).")
    parser.add_argument("--use_nonzero_only", action="store_true", help="Compute percentiles on nonzero voxels only.")
    parser.add_argument("--save_json", type=str, default=None, help="Optional path to save JSON report.")

    # Optional explicit paths (supports: --paths a b c)
    parser.add_argument("--paths", nargs="*", default=None, help="Explicit list of image paths.")

    args = parser.parse_args()

    max_cases = args.max_cases
    # Allow "--max_cases -1" to mean "all"
    if max_cases is not None and max_cases < 0:
        max_cases = None

    audit_dataset(
        root_dir=args.root_dir,
        pattern=args.pattern,
        list_file=args.list_file,
        paths=args.paths,
        max_cases=max_cases,
        use_nonzero_only=args.use_nonzero_only,
        save_json=args.save_json,
    )


if __name__ == "__main__":
    main()