import os
from typing import Dict, List


def collect_cases_with_all_masks(root_dir: str, verbose: bool = False) -> List[Dict]:
    """
    Collect all cases and return:
        {
            "case_id": str,
            "image": path,
            "mask_paths": [list of mask paths]
        }

    Does NOT filter masks — that is done later by the transform.
    """

    if not os.path.isdir(root_dir):
        raise ValueError(f"root_dir not found: {root_dir}")

    cases = []
    ids = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])

    for case_id in ids:
        case_dir = os.path.abspath(os.path.join(root_dir, case_id))

        image_path = os.path.join(case_dir, "image.nii.gz")
        if not os.path.exists(image_path):
            continue

        # collect all masks (robust matching)
        mask_paths = [
            os.path.join(case_dir, f)
            for f in os.listdir(case_dir)
            if "mask" in f.lower() and f.endswith(".nii.gz")
        ]

        if len(mask_paths) == 0:
            continue

        if verbose:
            print(f"[INFO] Case {case_id}: found {len(mask_paths)} masks")

        cases.append({
            "case_id": case_id,
            "image": image_path,
            "mask_paths": sorted(mask_paths),
        })

    if len(cases) == 0:
        raise RuntimeError(f"No valid cases found in {root_dir}")

    return cases