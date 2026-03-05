from typing import Dict, List, Tuple

from src.VocoLarge.segmentation.data.overfit_case import find_case_with_required_masks, REQUIRED_MASKS

# Hardcode for now (or make it a CLI arg later)
ROOT_DIR = "/mnt/data/flaviu/example_pt"


def build_samples() -> Tuple[List[Dict], List[Dict]]:
    sample = find_case_with_required_masks(ROOT_DIR, REQUIRED_MASKS)

    # sample["masks"] is dict: {1: path_L2, 2: path_L3, 3: path_L4}
    # turn into stable keys mask1/mask2/mask3 for transforms
    masks = sample["masks"]
    out = {
        "case_id": sample["case_id"],
        "image": sample["image"],
        "mask1": masks[1],
        "mask2": masks[2],
        "mask3": masks[3],
    }

    # Overfit: train == val == same one case
    return [out], [out]