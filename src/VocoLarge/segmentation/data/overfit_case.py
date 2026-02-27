import os
from typing import Dict, List, Optional, Tuple

REQUIRED_MASKS = [
    "mask_CTVn_L2.nii.gz",
    "mask_CTVn_L3.nii.gz",
    "mask_CTVn_L4.nii.gz",
]

# class index mapping (background=0)
MASK_TO_CLASS = {
    "mask_CTVn_L2.nii.gz": 1,
    "mask_CTVn_L3.nii.gz": 2,
    "mask_CTVn_L4.nii.gz": 3,
}


def find_case_with_required_masks(root_dir: str, required_masks: Optional[List[str]] = None) -> Dict:
    """
    Scans root_dir/<id>/ for a folder containing all required mask filenames.
    Returns a sample dict with image path + per-class mask paths.

    Expected structure:
      root_dir/<id>/image.nii.gz
      root_dir/<id>/mask_*.nii.gz
    """
    required_masks = required_masks or REQUIRED_MASKS

    if not os.path.isdir(root_dir):
        raise ValueError(f"root_dir not found: {root_dir}")

    ids = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

    for case_id in ids:
        case_dir = os.path.join(root_dir, case_id)
        image_path = os.path.join(case_dir, "image.nii.gz")
        if not os.path.exists(image_path):
            continue

        ok = True
        mask_paths = {}
        for m in required_masks:
            p = os.path.join(case_dir, m)
            if not os.path.exists(p):
                ok = False
                break
            mask_paths[MASK_TO_CLASS[m]] = p

        if ok:
            return {
                "case_id": case_id,
                "image": image_path,
                "masks": mask_paths,  # dict: class_index -> mask_path
            }

    raise RuntimeError(
        f"No case found in {root_dir} containing all masks: {required_masks}"
    )