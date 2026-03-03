import numpy as np
import torch
from monai.transforms import MapTransform


class ForegroundFromMultiLabeld(MapTransform):
    """
    Creates a single-channel foreground map from a multi-channel multilabel target.

    Input:
      - label_key: label tensor with shape (K,H,W,D) or (1,K,H,W,D) depending on pipeline step
    Output:
      - fg_key: single-channel binary tensor (1,H,W,D) marking any-positive voxels

    Why:
      RandCropByPosNegLabeld is most reliable with a single-channel label_key for sampling.
    """

    def __init__(self, label_key: str = "label", fg_key: str = "fg"):
        super().__init__(keys=[label_key])
        self.label_key = label_key
        self.fg_key = fg_key

    def __call__(self, data):
        d = dict(data)
        lab = d[self.label_key]

        # Handle torch / numpy, and either (K,H,W,D) or (1,K,H,W,D)
        if isinstance(lab, torch.Tensor):
            if lab.ndim == 5:     # (B,K,H,W,D) but usually B=1 inside dict
                x = lab[0]
            else:                 # (K,H,W,D)
                x = lab
            fg = (x.sum(dim=0, keepdim=True) > 0).to(torch.float32)  # (1,H,W,D)
        else:
            if lab.ndim == 5:
                x = lab[0]
            else:
                x = lab
            fg = (np.sum(x, axis=0, keepdims=True) > 0).astype(np.float32)

        d[self.fg_key] = fg
        return d