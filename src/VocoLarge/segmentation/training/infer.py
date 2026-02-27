import torch
from monai.inferers import sliding_window_inference

from src.VocoLarge.segmentation.config import Config


@torch.no_grad()
def infer_full_volume(model, image, cfg: Config):
    """
    REQUIRED.

    Runs sliding window inference over the full validation volume.
    Keeps ROI size equal to training roi_size for consistency and to control memory.
    """
    return sliding_window_inference(
        inputs=image,
        roi_size=cfg.roi_size,
        sw_batch_size=cfg.sw_batch_size,
        predictor=model,
        overlap=cfg.val_overlap,
    )