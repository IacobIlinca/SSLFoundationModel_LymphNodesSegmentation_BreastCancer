from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete, Compose, Activations
import torch
from torch import nn

from src.VocoLarge.segmentation.config import Config
from src.VocoLarge.segmentation.config_binary import ConfigBinary


# =========================================================
# OLD MULTICLASS FUNCTIONS
# Keep untouched for the old overfit pipeline
# =========================================================
def build_loss_multiclass(cfg: Config):
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
def build_loss_binary_softmax(cfg: ConfigBinary):
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

def build_loss_binary_sigmoid(cfg: ConfigBinary):
    """
    Binary segmentation with 1-channel sigmoid output.
    Expects:
      - logits: (B, 1, H, W, D)
      - labels: (B, 1, H, W, D) or (B, H, W, D) with values {0,1}
    """
    return DiceBCEBinaryLoss(
        pos_weight=cfg.class_weight_for_loss,
        dice_weight=1.0,
        bce_weight=1.0,
    ).to(cfg.device)


def build_metrics_binary_softmax(cfg):
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


def build_metrics_binary_sigmoid(cfg: ConfigBinary):
    """
    Binary metrics for 1-channel sigmoid output:
      - pred: sigmoid -> threshold
      - label: binary mask
    """
    dice = DiceMetric(
        include_background=True,
        reduction="mean_batch",
    )
    hd95 = HausdorffDistanceMetric(
        include_background=True,
        percentile=95.0,
        reduction="mean_batch",
    )

    post_pred = Compose([
        Activations(sigmoid=True),
        AsDiscrete(threshold=0.5),
    ])
    post_label = AsDiscrete(threshold=0.5)

    return dice, hd95, post_pred, post_label


class DiceBCEBinaryLoss(nn.Module):
    """
    Binary segmentation loss for 1-channel sigmoid logits.

    Expects:
      - logits: (B, 1, H, W, D)
      - labels: (B, 1, H, W, D) or (B, H, W, D), values {0,1}
    """

    def __init__(self, pos_weight=None, dice_weight=1.0, bce_weight=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

        self.dice = DiceLoss(sigmoid=True)

        if pos_weight is not None:
            pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, labels):
        if labels.ndim == logits.ndim - 1:
            labels = labels.unsqueeze(1)

        labels = labels.float()

        dice_loss = self.dice(logits, labels)
        bce_loss = self.bce(logits, labels)

        return self.dice_weight * dice_loss + self.bce_weight * bce_loss