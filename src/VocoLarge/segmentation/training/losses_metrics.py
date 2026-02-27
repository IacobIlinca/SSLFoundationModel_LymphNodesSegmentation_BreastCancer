from typing import Tuple
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete

from src.VocoLarge.segmentation.config import Config


def build_loss(cfg: Config):
    """
    REQUIRED.

    For multi-class segmentation (exclusive classes):
      - Dice + CrossEntropy is a strong baseline.
      - We use softmax=True and to_onehot_y=True.
    """
    return DiceCELoss(to_onehot_y=True, softmax=True)


def build_metrics(cfg: Config) -> Tuple[DiceMetric, HausdorffDistanceMetric, AsDiscrete, AsDiscrete]:
    """
    REQUIRED.

    Dice: excludes background by default.
    HD95: percentile=95, excludes background.

    post_pred: argmax logits -> integer prediction
    post_label: integer label -> onehot label (required by metrics)
    """
    dice = DiceMetric(include_background=False, reduction="mean_batch")
    hd95 = HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="mean_batch")

    post_pred = AsDiscrete(argmax=True)
    post_label = AsDiscrete(to_onehot=cfg.num_classes + 1)

    return dice, hd95, post_pred, post_label