from typing import Dict, List, Optional
import numpy as np
import torch
from monai.transforms import MapTransform


class CombineBinaryMasksReportOverlapd(MapTransform):
    """
    Combine multiple binary masks into a single multiclass label map (1,H,W,D),
    while reporting tiny overlaps.

    Reports:
      - overlap_pixels: number of voxels where >=2 masks are positive
      - union_pixels: number of voxels where >=1 mask is positive
      - overlap_percent: overlap_pixels / union_pixels * 100

    Overlap resolution (when writing multiclass labels):
      - resolve="last": last mask in mask_keys wins (default; same behavior as your current code)
      - resolve="first": first mask wins (overlapping voxels keep the first assigned label)
      - resolve="priority": use class_priority dict (higher value wins by default)
          * if tie, later mask in mask_keys wins
    """

    def __init__(
        self,
        mask_keys: List[str],
        mask_key_to_class_index: Dict[str, int],
        label_key: str = "label",
        log_prefix: str = "[CombineBinaryMasks]",
        resolve: str = "last",  # "last" | "first" | "priority"
        class_priority: Optional[Dict[int, int]] = None,  # class_index -> priority score
        print_if_no_overlap: bool = False,
    ):
        super().__init__(keys=mask_keys)
        self.mask_keys = list(mask_keys)
        self.map = dict(mask_key_to_class_index)
        self.label_key = label_key
        self.log_prefix = log_prefix
        assert resolve in ("last", "first", "priority")
        self.resolve = resolve
        self.class_priority = class_priority or {}
        self.print_if_no_overlap = print_if_no_overlap

    def __call__(self, data):
        d = dict(data)
        first = d[self.mask_keys[0]]  # (1,H,W,D)
        is_torch = isinstance(first, torch.Tensor)

        # --- Pass 1: compute union + overlap statistics (order-independent) ---
        if is_torch:
            masks = [(d[mk] > 0) for mk in self.mask_keys]  # list of (1,H,W,D) bool
            # stack -> (K,1,H,W,D) -> sum over K
            stacked = torch.stack(masks, dim=0).to(dtype=torch.int16)
            count_map = stacked.sum(dim=0)  # (1,H,W,D)
            union = count_map > 0
            overlap = count_map > 1

            union_pixels = int(union.sum().item())
            overlap_pixels = int(overlap.sum().item())
        else:
            masks = [(d[mk] > 0) for mk in self.mask_keys]  # list of (1,H,W,D) bool
            stacked = np.stack(masks, axis=0).astype(np.int16)  # (K,1,H,W,D)
            count_map = stacked.sum(axis=0)  # (1,H,W,D)
            union = count_map > 0
            overlap = count_map > 1

            union_pixels = int(union.sum())
            overlap_pixels = int(overlap.sum())

        overlap_percent = (100.0 * overlap_pixels / union_pixels) if union_pixels > 0 else 0.0

        if overlap_pixels > 0 or self.print_if_no_overlap:
            print(
                f"{self.log_prefix} overlap={overlap_pixels} px "
                f"of union={union_pixels} px ({overlap_percent:.4f}%)."
            )

        # --- Pass 2: create multiclass label map + resolve overlaps ---
        if is_torch:
            label = torch.zeros_like(first, dtype=torch.long)
        else:
            label = np.zeros_like(first, dtype=np.int64)

        if self.resolve == "last":
            # later masks overwrite earlier labels on overlap
            for mk in self.mask_keys:
                cls = int(self.map[mk])
                m_bin = (d[mk] > 0)
                label[m_bin] = cls

        elif self.resolve == "first":
            # keep first assigned label; only write into background
            for mk in self.mask_keys:
                cls = int(self.map[mk])
                m_bin = (d[mk] > 0)
                write = m_bin & (label == 0)
                label[write] = cls

        else:  # priority
            # choose label with highest priority score at each voxel
            # Implementation: maintain best_score map; update if new score > old score (or tie -> last wins)
            if is_torch:
                best_score = torch.full_like(first, fill_value=-10_000, dtype=torch.int32)
            else:
                best_score = np.full_like(first, fill_value=-10_000, dtype=np.int32)

            for mk in self.mask_keys:
                cls = int(self.map[mk])
                score = int(self.class_priority.get(cls, 0))
                m_bin = (d[mk] > 0)

                if is_torch:
                    score_map = torch.full_like(best_score, score, dtype=torch.int32)
                    better = m_bin & (score_map >= best_score)  # ">=" so ties go to later masks (last wins tie)
                else:
                    score_map = np.full_like(best_score, score, dtype=np.int32)
                    better = m_bin & (score_map >= best_score)

                best_score[better] = score
                label[better] = cls

        d[self.label_key] = label
        return d