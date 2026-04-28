from typing import Dict, List, Optional
import numpy as np
import torch
from monai.data import MetaTensor
from monai.data.utils import nib
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

# ============================================================
# Custom transform: multiple masks per class -> one label
# ============================================================

class BuildMulticlassLabelFromMaskFilesd(MapTransform):
    """
    Builds one integer multiclass label map from multiple binary masks per class.

    Input:
        image: loaded image MetaTensor, channel-first [C, H, W, D]
        class_masks: dict[class_name] -> list[path]

    Output:
        label: MetaTensor [1, H, W, D]

    Overlap rule:
        last class wins.
        Classes are applied in ascending class index order, so higher class_index
        overwrites lower class_index.
    """

    def __init__(
        self,
        keys,
        class_to_index: Dict[str, int],
        class_masks_key: str = "class_masks",
        label_key: str = "label",
    ):
        super().__init__(keys)
        self.class_to_index = class_to_index
        self.class_masks_key = class_masks_key
        self.label_key = label_key

    def __call__(self, data):
        d = dict(data)

        image = d["image"]

        if not hasattr(image, "shape"):
            raise TypeError("Expected 'image' to be loaded before building label.")

        # image is channel-first: [C, H, W, D]
        spatial_shape = tuple(image.shape[1:])

        label = np.zeros(spatial_shape, dtype=np.int16)

        class_masks = d.get(self.class_masks_key, {})
        case_id = d.get("case_id", "unknown_case")

        # Deterministic order:
        # level1 -> level2 -> level3 -> level4 -> imn -> interpectoral
        # if using the default class_to_index.
        for class_name, class_index in sorted(
            self.class_to_index.items(),
            key=lambda x: x[1],
        ):
            mask_paths = class_masks.get(class_name, [])

            # if len(mask_paths) == 0:
            #     print(f"[WARN] No mask paths found for class {class_name}, case {case_id}.")

            for mask_path in mask_paths:
                mask_img = nib.load(mask_path)
                mask_arr = np.asanyarray(mask_img.dataobj)

                if mask_arr.shape != spatial_shape:
                    raise ValueError(
                        f"Shape mismatch for case {case_id}, class {class_name}: "
                        f"mask {mask_path} has shape {mask_arr.shape}, "
                        f"but image spatial shape is {spatial_shape}."
                    )

                # Last class wins.
                # If multiple masks exist for the same class, they simply union
                # into the same class index.
                label[mask_arr > 0] = class_index

        affine = getattr(image, "affine", None)
        meta = dict(getattr(image, "meta", {}))

        d[self.label_key] = MetaTensor(
            label[None, ...],
            affine=affine,
            meta=meta,
        )
        #unique_labels = np.unique(label)

        # print(
        #     f"[DEBUG] case {case_id} unique labels after label build: "
        #     f"{unique_labels.tolist()}"
        # )

        return d