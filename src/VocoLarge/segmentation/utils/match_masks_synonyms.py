import json
import re
from collections import defaultdict
from typing import Dict, List, Optional


# ======== EDIT THESE PATHS ========
AUDIT_JSON = "src/VocoLarge/segmentation/data_audit_results/audit_workshop.json"
SYNONYMS_JSON = "/mnt/data/ilinca/rtnation_02_02/synonyms.json"
OUTPUT_JSON = "src/VocoLarge/segmentation/data_audit_results/grouped_masks_workshop_12_03.json"
# ==================================


def normalize_text(s: str) -> str:
    """
    Normalize text for robust case-insensitive matching.
    """
    s = s.lower().strip()

    # remove nifti extension
    s = re.sub(r"\.nii(\.gz)?$", "", s)

    # remove leading mask_
    s = re.sub(r"^mask[_\- ]*", "", s)

    # normalize separators
    s = re.sub(r"[_\-.]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s


def parse_fraction(frac: str) -> tuple[int, int]:
    """
    Parse strings like '6/20' or '1/1756'.
    Returns (count, total).
    """
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", frac)
    if not m:
        raise ValueError(f"Invalid fraction format: {frac}")
    return int(m.group(1)), int(m.group(2))


def build_synonym_lookup(synonyms: Dict[str, List[str]]) -> Dict[str, str]:
    """
    normalized synonym -> canonical group name
    """
    lookup = {}

    for canonical_name, variants in synonyms.items():
        lookup[normalize_text(canonical_name)] = canonical_name

        for v in variants:
            lookup[normalize_text(v)] = canonical_name

    return lookup


def find_canonical_group(
    mask_name: str,
    synonym_lookup: Dict[str, str],
    synonyms: Dict[str, List[str]],
) -> Optional[str]:
    """
    Match a raw mask filename to a canonical synonym group.

    Strategy:
      1. exact normalized match
      2. substring match against canonical + variants
    """
    mask_norm = normalize_text(mask_name)

    # exact match
    if mask_norm in synonym_lookup:
        return synonym_lookup[mask_norm]

    # substring match
    for canonical_name, variants in synonyms.items():
        all_terms = [canonical_name] + variants
        for term in all_terms:
            term_norm = normalize_text(term)
            if term_norm and term_norm in mask_norm:
                return canonical_name

    return None


def main():
    with open(AUDIT_JSON, "r", encoding="utf-8") as f:
        audit_data = json.load(f)

    with open(SYNONYMS_JSON, "r", encoding="utf-8") as f:
        synonyms = json.load(f)

    items = audit_data.get("mask_presence_summary_sorted", [])
    if not items:
        raise ValueError("No 'mask_presence_summary_sorted' found in audit JSON.")

    synonym_lookup = build_synonym_lookup(synonyms)

    grouped_counts = defaultdict(int)
    grouped_masks = defaultdict(list)
    unmatched = []

    total_patients = None

    for item in items:
        mask_name = item["mask_name"]
        fraction = item["fraction"]

        count, total = parse_fraction(fraction)

        if total_patients is None:
            total_patients = total
        elif total_patients != total:
            raise ValueError(
                f"Inconsistent total patient counts in fractions: "
                f"got {total}, expected {total_patients}"
            )

        canonical = find_canonical_group(mask_name, synonym_lookup, synonyms)

        if canonical is None:
            unmatched.append({
                "mask_name": mask_name,
                "fraction": fraction,
            })
            continue

        grouped_counts[canonical] += count
        grouped_masks[canonical].append({
            "mask_name": mask_name,
            "fraction": fraction,
            "count": count,
        })

    grouped_summary = []
    for canonical, count_sum in grouped_counts.items():
        grouped_summary.append({
            "group_name": canonical,
            "fraction": f"{count_sum}/{total_patients}",
            "count_sum": count_sum,
            "matched_masks": sorted(grouped_masks[canonical], key=lambda x: (-x["count"], x["mask_name"].lower())),
        })

    grouped_summary.sort(key=lambda x: (-x["count_sum"], x["group_name"].lower()))

    output = {
        "total_patients": total_patients,
        "num_original_mask_entries": len(items),
        "num_grouped_categories": len(grouped_summary),
        "num_unmatched_entries": len(unmatched),
        "grouped_mask_summary": grouped_summary,
        "unmatched_masks": sorted(unmatched, key=lambda x: x["mask_name"].lower()),
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved grouped summary to: {OUTPUT_JSON}")
    print(f"Total patients: {total_patients}")
    print(f"Original mask entries: {len(items)}")
    print(f"Grouped categories: {len(grouped_summary)}")
    print(f"Unmatched entries: {len(unmatched)}")


if __name__ == "__main__":
    main()