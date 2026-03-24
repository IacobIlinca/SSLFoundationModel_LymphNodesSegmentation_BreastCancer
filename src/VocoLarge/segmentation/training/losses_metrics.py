from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete
import torch

from src.VocoLarge.segmentation.config import Config


# =========================================================
# OLD MULTICLASS FUNCTIONS
# Keep untouched for the old overfit pipeline
# =========================================================
def build_loss(cfg: Config):
    """
    multiclass: Dice + CE with softmax
    """
    return DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        weight=torch.FloatTensor(cfg.class_weight_for_loss).to(cfg.device),
    )


def build_metrics(cfg: Config):
    """
    multiclass:
      - pred: argmax -> int
      - label: int
      - metrics exclude background
    """
    dice = DiceMetric(include_background=False, reduction="mean_batch")
    hd95 = HausdorffDistanceMetric(
        include_background=False,
        percentile=95.0,
        reduction="mean_batch",
    )
    post_pred = AsDiscrete(argmax=True)
    post_label = AsDiscrete()
    return dice, hd95, post_pred, post_label


# =========================================================
# NEW BINARY FUNCTIONS
# For binary training with 2-channel output:
#   channel 0 = background
#   channel 1 = lymph node
# =========================================================
def build_loss_binary(cfg):
    """
    Binary segmentation with 2-channel softmax output.
    Expects:
      - logits: (B, 2, H, W, D)
      - labels: integer {0,1}
    """
    return DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        weight=torch.FloatTensor(cfg.class_weight_for_loss).to(cfg.device),
    )


def build_metrics_binary(cfg):
    """
    Binary metrics:
      - pred: argmax -> one-hot with 2 classes
      - label: integer -> one-hot with 2 classes
      - exclude background from Dice / HD95
    """
    num_classes_total = cfg.num_classes + 1  # background + foreground

    dice = DiceMetric(
        include_background=False,
        reduction="mean_batch",
    )
    hd95 = HausdorffDistanceMetric(
        include_background=False,
        percentile=95.0,
        reduction="mean_batch",
    )

    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes_total)
    post_label = AsDiscrete(to_onehot=num_classes_total)

    return dice, hd95, post_pred, post_label