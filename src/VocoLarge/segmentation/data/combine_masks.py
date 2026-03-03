from typing import Dict, List
import numpy as np
import torch
from monai.transforms import MapTransform


class CombineBinaryMasksd(MapTransform):
    """
    Combines multiple binary masks into either:
      - multiclass: single integer label map (1,H,W,D) with values 0..K
      - multilabel: multi-channel label (K,H,W,D) with {0,1}

    mask_key_to_class_index:
      keys: mask_key in sample dict (e.g., "mask1")
      values: class index in 1..K

    IMPORTANT:
      - For multilabel, output channels correspond to class indices 1..K in order.
        Channel 0 corresponds to class 1, channel K-1 corresponds to class K.
    """

    def __init__(
        self,
        mask_keys: List[str],
        mask_key_to_class_index: Dict[str, int],
        label_key: str = "label",
        label_mode: str = "multiclass",
        raise_on_overlap: bool = True,
    ):
        super().__init__(keys=mask_keys)
        self.mask_keys = list(mask_keys)
        self.map = dict(mask_key_to_class_index)
        self.label_key = label_key
        assert label_mode in ("multiclass", "multilabel")
        self.label_mode = label_mode
        self.raise_on_overlap = raise_on_overlap

        # Determine K from mapping
        self.K = max(self.map.values())

    def __call__(self, data):
        d = dict(data)
        first = d[self.mask_keys[0]]  # (1,H,W,D)

        is_torch = isinstance(first, torch.Tensor)

        if self.label_mode == "multiclass":
            # (1,H,W,D) integer map
            label = torch.zeros_like(first, dtype=torch.long) if is_torch else np.zeros_like(first, dtype=np.int64)

            for mk in self.mask_keys:
                cls = int(self.map[mk])
                m = d[mk]
                m_bin = (m > 0)

                if self.raise_on_overlap:
                    overlap = (label > 0) & m_bin
                    if (torch.any(overlap) if is_torch else np.any(overlap)):
                        raise ValueError(f"Mask overlap detected while adding {mk} as class {cls}")

                label[m_bin] = cls

            d[self.label_key] = label
            return d

        # multilabel
        # Output: (K,H,W,D) float32 with {0,1}
        if is_torch:
            _, H, W, D_ = first.shape
            label = torch.zeros((self.K, H, W, D_), dtype=torch.float32, device=first.device)
            for mk in self.mask_keys:
                cls = int(self.map[mk])          # 1..K
                ch = cls - 1                     # 0..K-1
                m = d[mk]
                label[ch] = (m > 0).float().squeeze(0)  # (H,W,D)
        else:
            _, H, W, D_ = first.shape
            label = np.zeros((self.K, H, W, D_), dtype=np.float32)
            for mk in self.mask_keys:
                cls = int(self.map[mk])
                ch = cls - 1
                m = d[mk]
                label[ch] = (m > 0).astype(np.float32).squeeze(0)

        d[self.label_key] = label
        return d