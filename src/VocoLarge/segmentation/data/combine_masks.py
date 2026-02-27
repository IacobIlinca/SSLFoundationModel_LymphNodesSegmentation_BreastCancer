from typing import Dict, List
import numpy as np
import torch
from monai.transforms import MapTransform


class CombineBinaryMasksToLabeld(MapTransform):
    """
    Combine multiple binary masks into a single integer label map.

    - Background = 0
    - Class indices are provided via mask_key_to_class_index
    - Collision rule: if masks overlap, raise ValueError
    """

    def __init__(
        self,
        mask_keys: List[str],
        mask_key_to_class_index: Dict[str, int],
        label_key: str = "label",
        raise_on_overlap: bool = True,
    ):
        # MapTransform requires keys, but we don't actually use self.keys here.
        super().__init__(keys=mask_keys)
        self.mask_keys = list(mask_keys)
        self.map = dict(mask_key_to_class_index)
        self.label_key = label_key
        self.raise_on_overlap = raise_on_overlap

    def __call__(self, data):
        d = dict(data)

        first = d[self.mask_keys[0]]  # (1,H,W,D) after EnsureChannelFirstd
        if isinstance(first, torch.Tensor):
            label = torch.zeros_like(first, dtype=torch.long)
        else:
            label = np.zeros_like(first, dtype=np.int64)

        for mk in self.mask_keys:
            cls = int(self.map[mk])
            m = d[mk]

            if isinstance(m, torch.Tensor):
                m_bin = (m > 0)
                if self.raise_on_overlap:
                    overlap = (label > 0) & m_bin
                    if torch.any(overlap):
                        raise ValueError(f"Mask overlap detected while adding key={mk} as class={cls}")
                label[m_bin] = cls
            else:
                m_bin = (m > 0)
                if self.raise_on_overlap:
                    overlap = (label > 0) & m_bin
                    if np.any(overlap):
                        raise ValueError(f"Mask overlap detected while adding key={mk} as class={cls}")
                label[m_bin] = cls

        d[self.label_key] = label
        return d