import json
import re
import csv
import argparse
from pathlib import Path


SIZE_PATTERN = re.compile(r"[0-9]+.*(mm|cm)", re.IGNORECASE)

CLASS_TERM_FILES = {
    "level1": "level1_terms.json",
    "level2": "level2_terms.json",
    "level3": "level3_terms.json",
    "level4": "level4_terms.json",
    "interpectoral": "interpectoral_terms.json",
    "imn": "imn_terms.json",
}


def normalize_text(text: str) -> str:
    """
    Normalize filenames and terms so these match equivalently:

      mask_Level-III
      mask_Level_III
      mask Level III
      mask.Level.III

    All become:
      mask-level-iii
    """
    text = str(text).lower().strip()
    text = text.replace(".nii.gz", "")
    text = text.replace(".nii", "")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text


def normalize_name(path: Path) -> str:
    return normalize_text(path.name)


def flatten_terms(data) -> list[str]:
    """
    Supports term files formatted as either:

      {
        "terms": ["level-iii", "level_iii"]
      }

    or:

      ["level-iii", "level_iii"]

    Also handles accidental nested structures like:
      [{"terms": [...]}]
    """
    out = []

    if data is None:
        return out

    if isinstance(data, str):
        term = normalize_text(data)
        return [term] if term else []

    if isinstance(data, dict):
        if "terms" in data:
            return flatten_terms(data["terms"])

        for key in ["term", "name", "value", "pattern"]:
            if key in data:
                return flatten_terms(data[key])

        raise ValueError(f"Could not extract terms from dict: {data}")

    if isinstance(data, list):
        for item in data:
            out.extend(flatten_terms(item))

        # keep order, remove duplicates
        seen = set()
        unique = []
        for term in out:
            if term and term not in seen:
                unique.append(term)
                seen.add(term)

        return unique

    term = normalize_text(data)
    return [term] if term else []


def load_terms(json_path: str | Path) -> list[str]:
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Missing term file: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    terms = flatten_terms(data)

    if not terms:
        raise ValueError(f"No terms found in: {json_path}")

    return terms


def load_required_classes_terms(terms_dir: Path) -> dict[str, list[str]]:
    required_classes = {}

    for class_name, filename in CLASS_TERM_FILES.items():
        path = terms_dir / filename
        required_classes[class_name] = load_terms(path)

    return required_classes


def contains_any(name: str, terms: list[str]) -> bool:
    return len(matched_terms(name, terms)) > 0

def matched_terms(name: str, terms: list[str]) -> list[str]:
    """
    Match normalized terms as full dash-separated tokens, not arbitrary substrings.

    Example:
      level-ii should NOT match level-iii
      level-iii should match mask-level-iii
    """
    name = normalize_text(name)
    hits = []

    for term in terms:
        term = normalize_text(term)
        if not term:
            continue

        pattern = rf"(^|-){re.escape(term)}(-|$)"
        if re.search(pattern, name):
            hits.append(term)

    return hits


def classify_mask(
    mask_path: Path,
    required_classes: dict[str, list[str]],
    exclude_terms: list[str],
) -> tuple[str | None, str, str]:
    """
    Returns:
      class_name, reason, matched_term_info

    class_name is one of:
      level2, level3, level4, interpectoral, imn

    or None if the mask should not be used.
    """
    name = normalize_name(mask_path)

    if SIZE_PATTERN.search(name):
        return None, "excluded_size_pattern", ""

    exclude_hits = matched_terms(name, exclude_terms)
    if exclude_hits:
        return None, "excluded_term", "|".join(exclude_hits)

    matched_classes = []
    matched_info = []

    for class_name, terms in required_classes.items():
        hits = matched_terms(name, terms)
        if hits:
            matched_classes.append(class_name)
            matched_info.append(f"{class_name}:{'|'.join(hits)}")

    if len(matched_classes) == 0:
        return None, "no_required_class_match", ""

    if len(matched_classes) > 1:
        return (
            None,
            "ambiguous_multiple_class_match:" + "|".join(matched_classes),
            ";".join(matched_info),
        )

    return matched_classes[0], "selected", ";".join(matched_info)


def find_case_dirs(root_dir: Path) -> list[Path]:
    return sorted([p for p in root_dir.iterdir() if p.is_dir()])


def find_masks(case_dir: Path) -> list[Path]:
    return sorted(case_dir.rglob("*.nii.gz"))


def write_ids(path: Path, ids: list[str]):
    with open(path, "w") as f:
        for case_id in ids:
            f.write(case_id + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root_dir",
        required=True,
        help="Root directory containing one subdirectory per case.",
    )

    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for CSV files and case ID lists.",
    )

    parser.add_argument(
        "--terms_dir",
        required=True,
        help=(
            "Directory containing: level2_terms.json, level3_terms.json, "
            "level4_terms.json, interpectoral_terms.json, imn_terms.json, "
            "exclude_terms.json."
        ),
    )

    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    out_dir = Path(args.out_dir)
    terms_dir = Path(args.terms_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    required_classes = load_required_classes_terms(terms_dir)
    exclude_terms = load_terms(terms_dir / "exclude_terms.json")

    print("[INFO] Loaded term files:")
    for class_name, terms in required_classes.items():
        print(f"  {class_name}: {len(terms)} terms")
    print(f"  exclude: {len(exclude_terms)} terms")

    case_dirs = find_case_dirs(root_dir)
    print(f"\n[INFO] Found {len(case_dirs)} case folders\n")

    summary_rows = []
    detail_rows = []

    multiclass_ids = []
    binary_ids = []
    review_ids = []

    for i, case_dir in enumerate(case_dirs, start=1):
        case_id = case_dir.name
        mask_paths = find_masks(case_dir)
        all_mask_names = [p.name for p in mask_paths]

        selected_by_class = {class_name: [] for class_name in required_classes}

        for mask_path in mask_paths:
            class_name, reason, matched_term_info = classify_mask(
                mask_path=mask_path,
                required_classes=required_classes,
                exclude_terms=exclude_terms,
            )

            detail_rows.append({
                "case_id": case_id,
                "mask_name": mask_path.name,
                "normalized_mask_name": normalize_name(mask_path),
                "selected_class": "" if class_name is None else class_name,
                "reason": reason,
                "matched_terms": matched_term_info,
            })

            if class_name is not None:
                selected_by_class[class_name].append(mask_path.name)

        missing_classes = [
            class_name
            for class_name, masks in selected_by_class.items()
            if len(masks) == 0
        ]

        present_classes = [
            class_name
            for class_name, masks in selected_by_class.items()
            if len(masks) > 0
        ]

        has_all_required = len(missing_classes) == 0

        if has_all_required:
            decision = "multiclass"
            multiclass_ids.append(case_id)
        elif len(present_classes) > 0:
            decision = "binary"
            binary_ids.append(case_id)
        else:
            decision = "review"
            review_ids.append(case_id)

        row = {
            "case_id": case_id,
            "decision": decision,
            "present_classes": "|".join(present_classes),
            "missing_classes": "|".join(missing_classes),
            "all_mask_names": "|".join(all_mask_names),
            "n_present_classes": len(present_classes),
            "n_missing_classes": len(missing_classes),
        }

        for class_name in required_classes:
            row[f"{class_name}_masks"] = "|".join(selected_by_class[class_name])
            row[f"n_{class_name}_masks"] = len(selected_by_class[class_name])

        summary_rows.append(row)

        print(
            f"[{i}/{len(case_dirs)}] {case_id}: "
            f"{decision} | present={present_classes} | missing={missing_classes}"
        )

    summary_csv = out_dir / "required_multiclass_nodes_summary_workshop_test.csv"
    details_csv = out_dir / "required_multiclass_nodes_details_workshop_test.csv"

    summary_fields = [
        "case_id",
        "decision",
        "present_classes",
        "missing_classes",
        "all_mask_names",
        "n_present_classes",
        "n_missing_classes",
    ]

    for class_name in required_classes:
        summary_fields.append(f"{class_name}_masks")
        summary_fields.append(f"n_{class_name}_masks")

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    detail_fields = [
        "case_id",
        "mask_name",
        "normalized_mask_name",
        "selected_class",
        "reason",
        "matched_terms",
    ]

    with open(details_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(detail_rows)

    write_ids(out_dir / "multiclass_case_ids.txt", multiclass_ids)
    write_ids(out_dir / "binary_case_ids.txt", binary_ids)
    write_ids(out_dir / "review_case_ids.txt", review_ids)

    print("\n[DONE]")
    print(f"Multiclass cases: {len(multiclass_ids)}")
    print(f"Binary cases:     {len(binary_ids)}")
    print(f"Review cases:     {len(review_ids)}")
    print()
    print(f"Saved: {summary_csv}")
    print(f"Saved: {details_csv}")
    print(f"Saved: {out_dir / 'multiclass_case_ids.txt'}")
    print(f"Saved: {out_dir / 'binary_case_ids.txt'}")
    print(f"Saved: {out_dir / 'review_case_ids.txt'}")


if __name__ == "__main__":
    main()