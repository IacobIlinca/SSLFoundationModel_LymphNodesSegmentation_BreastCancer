from typing import Tuple
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete
import torch

from src.VocoLarge.segmentation.config import Config


def build_loss(cfg: Config):
    """
    multiclass: Dice + CE with softmax
    """
    return DiceCELoss(to_onehot_y=True, softmax=True, weight=torch.FloatTensor(cfg.class_weight_for_loss).to(cfg.device))



def build_metrics(cfg: Config):
    """
    multiclass:
      - pred: argmax -> int
      - label: int -> onehot
      - metrics exclude background

    """
    dice = DiceMetric(include_background=False, reduction="mean_batch")
    hd95 = HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="mean_batch")
    post_pred = AsDiscrete(argmax=True)
    post_label = AsDiscrete()
    return dice, hd95, post_pred, post_label

