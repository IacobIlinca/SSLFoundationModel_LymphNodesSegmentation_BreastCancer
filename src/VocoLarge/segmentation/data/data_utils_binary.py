import os
from pathlib import Path
from typing import Dict, List
import json
from pathlib import Path
from typing import List, Dict, Tuple


def read_ids_file(txt_path: str) -> List[str]:
    """
    Reads IDs from a txt file.
    Supports comma-separated format:
        id1,id2,id3
    """
    with open(txt_path, "r") as f:
        content = f.read().strip()

    if not content:
        raise RuntimeError(f"No IDs found in {txt_path}")

    if "," in content:
        ids = [x.strip() for x in content.split(",") if x.strip()]
    else:
        raise RuntimeError(f"Expected comma-separated IDs in {txt_path}")

    if len(ids) == 0:
        raise RuntimeError(f"Error while reading IDs from {txt_path}")

    return ids


def find_image_in_patient_folder(patient_dir: Path) -> str:
    """
    Finds the image file inside one patient folder.

    Adjust patterns here if needed.
    """
    patterns = [
        "image.nii.gz",
        "image.nii",
        "image.nii.tz",
    ]

    matches = []
    for pattern in patterns:
        matches.extend(patient_dir.glob(pattern))

    if len(matches) == 0:
        raise FileNotFoundError(f"No image file found in: {patient_dir}")

    if len(matches) > 1:
        raise RuntimeError(f"Multiple image files found in {patient_dir}: {matches}")

    return str(matches[0])


def find_masks_in_patient_folder(patient_dir: Path) -> List[str]:
    """
    Collect all mask files inside one patient folder.

    Does NOT filter lymph masks.
    That happens later in the MONAI transform.
    """
    mask_paths = [
        str(p)
        for p in patient_dir.iterdir()
        if p.is_file() and "mask" in p.name.lower() and (
            p.name.endswith(".nii") or p.name.endswith(".nii.gz")
        )
    ]

    if len(mask_paths) == 0:
        raise FileNotFoundError(f"No mask files found in: {patient_dir}")

    return sorted(mask_paths)


def build_segmentation_files_from_ids(root_dir: str, ids: List[str]) -> List[Dict]:
    """
    Converts patient IDs to MONAI-style dicts for segmentation:
        [
            {
                "case_id": "id1",
                "image": "/path/to/id1/image.nii.gz",
                "mask_paths": ["/path/to/id1/mask_a.nii.gz", ...],
            },
            ...
        ]
    """
    root = Path(root_dir)
    if not root.exists():
        raise RuntimeError(f"root_dir not found: {root_dir}")

    files = []

    for pid in ids:
        patient_dir = root / pid
        if not patient_dir.exists():
            raise RuntimeError(f"Missing patient folder: {patient_dir}")

        image_path = find_image_in_patient_folder(patient_dir)
        mask_paths = find_masks_in_patient_folder(patient_dir)

        files.append({
            "case_id": pid,
            "image": image_path,
            "mask_paths": mask_paths,
        })

    return files

def load_lymph_terms(json_path: str) -> List[str]:
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        terms = data
    elif isinstance(data, dict) and "terms" in data:
        terms = data["terms"]
    else:
        raise ValueError("Invalid lymph terms JSON format.")

    terms = [str(t).lower().strip() for t in terms if str(t).strip()]
    if not terms:
        raise ValueError("No lymph terms found in JSON.")

    return terms


def sample_has_lymph_mask(sample: Dict, lymph_terms: List[str]) -> bool:
    for p in sample["mask_paths"]:
        name = Path(p).name.lower()
        if any(term in name for term in lymph_terms):
            return True
    return False


def filter_positive_cases(
    samples: List[Dict],
    lymph_terms_json: str,
    log_file: str = None,
) -> Tuple[List[Dict], List[Dict]]:
    lymph_terms = load_lymph_terms(lymph_terms_json)

    kept = []
    skipped = []

    for sample in samples:
        if sample_has_lymph_mask(sample, lymph_terms):
            kept.append(sample)
        else:
            skipped.append(sample)

    if log_file is not None and len(skipped) > 0:
        with open(log_file, "a") as f:
            for s in skipped:
                available = [Path(p).name for p in s["mask_paths"]]
                f.write(
                    f"{s['case_id']} | NO_LYMPH_MATCH | masks={available}\n"
                )

    return kept, skipped