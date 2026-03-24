from typing import Dict, List, Tuple

from src.VocoLarge.segmentation.config import Config
from src.VocoLarge.segmentation.data.overfit_case import (
    find_case_with_required_masks,
    REQUIRED_MASKS,
)
from src.VocoLarge.segmentation.data.binary_cases import collect_cases_with_all_masks


def build_samples(cfg: Config) -> Tuple[List[Dict], List[Dict]]:
    if cfg.task_mode == "overfit_multiclass":
        sample = find_case_with_required_masks(cfg.root_dir, REQUIRED_MASKS)

        masks = sample["masks"]
        out = {
            "case_id": sample["case_id"],
            "image": sample["image"],
            "mask1": masks[1],
            "mask2": masks[2],
            "mask3": masks[3],
        }

        return [out], [out]

    elif cfg.task_mode == "train_binary":
        cases = collect_cases_with_all_masks(cfg.root_dir)

        n = len(cases)
        if n < 2:
            raise ValueError("Need at least 2 cases for train/val split.")

        split_idx = max(1, int(n * (1 - cfg.val_fraction)))
        split_idx = min(split_idx, n - 1)

        train_samples = cases[:split_idx]
        val_samples = cases[split_idx:]

        return train_samples, val_samples

    else:
        raise ValueError(
            f"Unknown task_mode '{cfg.task_mode}'. "
            f"Expected 'overfit_multiclass' or 'train_binary'."
        )