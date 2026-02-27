import torch.nn as nn
from monai.networks.nets import SwinUNETR
from src.VocoLarge.segmentation.config import Config


def build_model(cfg: Config) -> nn.Module:
    """
    REQUIRED.

    Builds SwinUNETR.

    IMPORTANT:
      - out_channels = num_classes + 1 (includes background class 0)
      - feature_size must match the VoCo checkpoint backbone variant.
        If weight loading report shows low match %, adjust cfg.feature_size.
    """
    model = SwinUNETR(
        in_channels=1,
        out_channels=cfg.num_classes + 1,
        feature_size=cfg.feature_size,
        use_checkpoint=False,
    )
    return model