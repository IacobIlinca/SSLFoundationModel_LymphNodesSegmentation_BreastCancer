import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

def read_ids_file(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def find_image_path(case_dir: Path) -> str:
    """
    Prefer image.nii.gz, but keep it slightly robust.
    """
    preferred = case_dir / "image.nii.gz"
    if preferred.exists():
        return str(preferred)

    preferred = case_dir / "image.nii"
    if preferred.exists():
        return str(preferred)

    candidates = sorted(case_dir.glob("image*.nii.gz")) + sorted(case_dir.glob("image*.nii"))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No image file found in {case_dir}")

    return str(candidates[0])

def build_multiclass_files_from_ids(
    root_dir: str,
    ids: List[str],
    case_to_masks: Dict[str, Dict[str, List[str]]],
    labels: Optional[List[str]] = None,
    require_foreground: bool = True,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Builds MONAI-style data dictionaries.

    Each returned item has:
        image
        case_id
        class_masks

    If require_foreground=True, cases with no masks for any class are skipped.
    """

    root_dir = Path(root_dir)

    files: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for case_id in ids:
        case_dir = root_dir / case_id

        try:
            image_path = find_image_path(case_dir)
        except FileNotFoundError:
            skipped.append(case_id)
            continue

        class_masks = case_to_masks.get(case_id, {cls: [] for cls in labels})

        # Make sure every class exists even if missing from CSV.
        class_masks = {
            cls: list(class_masks.get(cls, []))
            for cls in labels
        }

        # Keep only mask paths that actually exist.
        cleaned_class_masks = {}
        for cls, paths in class_masks.items():
            existing = [p for p in paths if Path(p).exists()]
            cleaned_class_masks[cls] = existing

        n_masks = sum(len(v) for v in cleaned_class_masks.values())

        if require_foreground and n_masks == 0:
            skipped.append(case_id)
            continue

        files.append(
            {
                "case_id": case_id,
                "image": image_path,
                "class_masks": cleaned_class_masks,
            }
        )

    return files, skipped


def parse_pipe_list(value: str) -> List[str]:
    """
    Parses CSV cells like:
        mask_a.nii.gz|mask_b.nii.gz
    or:
        ""
    """
    if value is None:
        return []

    value = str(value).strip()
    if value == "":
        return []

    return [v.strip() for v in value.split("|") if v.strip()]


def load_multiclass_mask_csv(
    csv_path: str,
    root_dir: str,
    class_to_csv_column: Optional[Dict[str, str]] = None,
    labels: Optional[List[str]] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Returns:
        {
            case_id: {
                "level1": [absolute_mask_path, ...],
                "level2": [absolute_mask_path, ...],
                ...
            }
        }

    Missing CSV columns are allowed.
    For example, if level1_masks does not exist in the CSV, all cases get level1=[].
    """

    labels = labels
    class_to_csv_column = class_to_csv_column

    root_dir = Path(root_dir)
    out: Dict[str, Dict[str, List[str]]] = {}

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        if "case_id" not in reader.fieldnames:
            raise ValueError(f"CSV must contain a 'case_id' column: {csv_path}")

        for row in reader:
            case_id = str(row["case_id"]).strip()
            if case_id == "":
                continue

            case_dir = root_dir / case_id
            case_entry: Dict[str, List[str]] = {}

            for cls in labels:
                col = class_to_csv_column.get(cls)

                if col is None or col not in row:
                    mask_names = []
                else:
                    mask_names = parse_pipe_list(row[col])

                mask_paths = [str(case_dir / name) for name in mask_names]
                case_entry[cls] = mask_paths

            out[case_id] = case_entry

    return out