import json
import re
from pathlib import Path
from typing import Dict, List, Optional


# ======== EDIT THESE ========
AUDIT_JSON = "src/VocoLarge/segmentation/data_audit_results/audit_workshop.json"
SYNONYMS_JSON = "/mnt/data/ilinca/rtnation_02_02/synonyms.json"
GROUP_NAME = "ctvn_l2"   # configurable
OUTPUT_JSON = "src/VocoLarge/segmentation/data_audit_results/patients_missing_group.json"
# ============================


def normalize_text(s: str) -> str:
    """
    Normalize strings for case-insensitive, format-tolerant matching.
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


def extract_patient_id(image_path: str) -> str:
    """
    Assumes patient ID is the parent folder name of image.nii.gz
    """
    return Path(image_path).parent.name


def matches_group(filename: str, group_name: str, synonyms: Dict[str, List[str]]) -> bool:
    """
    Return True if filename matches the configured group name, using synonyms.
    """
    if group_name not in synonyms:
        raise ValueError(f"Group '{group_name}' not found in synonyms JSON.")

    fname_norm = normalize_text(filename)

    terms = [group_name] + synonyms[group_name]
    terms_norm = [normalize_text(t) for t in terms if t]

    # exact match
    if fname_norm in terms_norm:
        return True

    # substring match
    for t in terms_norm:
        if t and t in fname_norm:
            return True

    return False


def collect_mask_files(folder_contents: List[str]) -> List[str]:
    """
    Keep only mask nifti files.
    """
    out = []
    for fname in folder_contents:
        lower = fname.lower()
        if lower.startswith("mask") and (lower.endswith(".nii") or lower.endswith(".nii.gz")):
            out.append(fname)
    return sorted(out, key=lambda x: x.lower())


def main():
    with open(AUDIT_JSON, "r", encoding="utf-8") as f:
        audit_data = json.load(f)

    with open(SYNONYMS_JSON, "r", encoding="utf-8") as f:
        synonyms = json.load(f)

    images = audit_data.get("images", [])
    if not images:
        raise ValueError("No 'images' section found in audit JSON.")

    total_cases = len(images)
    missing_cases = []

    for case in images:
        image_path = case.get("path", "")
        folder_contents = case.get("folder_contents", [])

        patient_id = extract_patient_id(image_path)
        mask_files = collect_mask_files(folder_contents)

        has_group = any(matches_group(fname, GROUP_NAME, synonyms) for fname in mask_files)

        if not has_group:
            missing_cases.append({
                "patient_id": patient_id,
                "path": image_path,
                "available_masks": mask_files,
            })

    # summary
    print(f"Group checked: {GROUP_NAME}")
    print(f"Total audited patients: {total_cases}")
    print(f"Patients missing group: {len(missing_cases)}")
    print()

    for item in missing_cases:
        print(f"[MISSING {GROUP_NAME}] {item['patient_id']}")
        if item["available_masks"]:
            for m in item["available_masks"]:
                print(f"  - {m}")
        else:
            print("  - no mask files found")
        print()

    output = {
        "group_name": GROUP_NAME,
        "total_audited_patients": total_cases,
        "num_missing_group": len(missing_cases),
        "missing_group_patients": missing_cases,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved results to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()