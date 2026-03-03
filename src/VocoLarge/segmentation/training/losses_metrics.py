from typing import Tuple
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete
import torch

from src.VocoLarge.segmentation.config import Config


def build_loss(cfg: Config):
    """
    multiclass: Dice + CE with softmax
    multilabel: Dice(sigmoid) + BCEWithLogits
    """
    if cfg.label_mode == "multiclass":
        return DiceCELoss(to_onehot_y=True, softmax=True)

    # multilabel
    dice = DiceLoss(sigmoid=True, squared_pred=True)
    bce = torch.nn.BCEWithLogitsLoss()

    def loss_fn(logits, targets):
        # targets expected float (B,K,H,W,D) in {0,1}
        return dice(logits, targets) + bce(logits, targets)

    return loss_fn


def build_metrics(cfg: Config):
    """
    multiclass:
      - pred: argmax -> int
      - label: int -> onehot
      - metrics exclude background

    multilabel:
      - pred: sigmoid+threshold -> binary per channel
      - label: already binary per channel
      - metrics computed per channel (no background channel exists)
    """
    if cfg.label_mode == "multiclass":
        dice = DiceMetric(include_background=False, reduction="mean_batch")
        hd95 = HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="mean_batch")
        post_pred = AsDiscrete(argmax=True)
        post_label = AsDiscrete(to_onehot=cfg.num_classes + 1)
        return dice, hd95, post_pred, post_label

    # multilabel: include_background=True because there's no background channel; we want all channels
    dice = DiceMetric(include_background=True, reduction="mean_batch")
    hd95 = HausdorffDistanceMetric(include_background=True, percentile=95.0, reduction="mean_batch")

    # AsDiscrete supports thresholding; we apply sigmoid in engine before this, or here if supported.
    # We'll do sigmoid in engine to keep version-safe.
    post_pred = AsDiscrete(threshold=cfg.pred_threshold)
    post_label = AsDiscrete(threshold=0.5)  # labels already 0/1
    return dice, hd95, post_pred, post_label