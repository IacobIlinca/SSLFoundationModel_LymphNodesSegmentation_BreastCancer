import warnings
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.losses import DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete, Activations, Compose
from monai.utils import look_up_option, DiceCEReduction, pytorch_after
from torch.nn.modules.loss import _Loss

from src.VocoLarge.segmentation.config import Config
from src.VocoLarge.segmentation.config_binary import ConfigBinary
from src.VocoLarge.segmentation.multiclass_segmentation.config_multiclass import ConfigMulticlass


# =========================================================
# OLD MULTICLASS FUNCTIONS
# Keep untouched for the old overfit pipeline
# =========================================================
def build_loss_multiclass(cfg: ConfigMulticlass):
    """
    multiclass: Dice + CE with softmax
    """
    return DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        weight=torch.FloatTensor(cfg.class_weight_for_loss).to(cfg.device),
        lambda_dice=cfg.dice_weight,
        lambda_ce=cfg.ce_weight,
        include_background=False
    )

def build_loss_multiclass_with_components(cfg: ConfigMulticlass):
    return DiceCELossMulticlass(
        dice_weight=cfg.dice_weight,
        ce_weight=cfg.ce_weight,
        class_weights=cfg.class_weight_for_loss,
        include_background=False,
        device=cfg.device
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

def build_loss_binary_sigmoid(cfg):
    """
    Binary segmentation with 1-channel sigmoid output.
    Expects:
      - logits: (B, 1, H, W, D)
      - labels: (B, 1, H, W, D) or (B, H, W, D) with values {0,1}
    """
    return DiceBCESurfaceBinaryLoss(
        pos_weight=cfg.class_weight_for_loss,
        dice_weight=cfg.dice_weight,
        bce_weight=cfg.bce_weight,
        surface_weight=cfg.surface_weight,
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


def build_metrics_binary_sigmoid(cfg):
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


def build_metrics_multiclass_softmax(cfg):
    """
    Multiclass metrics for softmax output.

    Expects:
      logits: (B, C, H, W, D)
      label:  (B, 1, H, W, D) or (B, H, W, D), integer labels

    where:
      C = cfg.num_classes + 1
        = background + foreground classes
    """

    n_classes = cfg.num_classes + 1

    dice = DiceMetric(
        include_background=False,
        reduction="mean_batch",
        get_not_nans=True,
    )

    hd95 = HausdorffDistanceMetric(
        include_background=False,
        percentile=95.0,
        reduction="mean_batch",
        get_not_nans=True,
    )

    post_pred = Compose([
        Activations(softmax=True),
        AsDiscrete(argmax=True, to_onehot=n_classes),
    ])

    post_label = Compose([
        AsDiscrete(to_onehot=n_classes),
    ])

    return dice, hd95, post_pred, post_label


def soft_erode3d(x: torch.Tensor) -> torch.Tensor:
    """
    Soft erosion for 3D masks using directional max-pooling trick.
    x: (B, 1, H, W, D)
    """
    p1 = -F.max_pool3d(-x, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
    p2 = -F.max_pool3d(-x, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0))
    p3 = -F.max_pool3d(-x, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))
    return torch.min(torch.min(p1, p2), p3)


def soft_dilate3d(x: torch.Tensor) -> torch.Tensor:
    """
    Soft dilation for 3D masks.
    x: (B, 1, H, W, D)
    """
    p1 = F.max_pool3d(x, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
    p2 = F.max_pool3d(x, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0))
    p3 = F.max_pool3d(x, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))
    return torch.max(torch.max(p1, p2), p3)


def soft_boundary3d(x: torch.Tensor) -> torch.Tensor:
    """
    Approximate boundary map = dilation - erosion.
    Input should be probabilities or binary masks in [0,1].
    """
    return torch.clamp(soft_dilate3d(x) - soft_erode3d(x), min=0.0, max=1.0)


class SoftSurfaceLoss(nn.Module):
    """
    Differentiable surrogate of surface Dice for binary segmentation.
    Operates on probabilities and binary labels.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        probs:  (B, 1, H, W, D), after sigmoid
        labels: (B, 1, H, W, D), binary {0,1}
        """
        pred_boundary = soft_boundary3d(probs)
        gt_boundary = soft_boundary3d(labels)

        intersection = (pred_boundary * gt_boundary).sum(dim=(1, 2, 3, 4))
        denom = pred_boundary.sum(dim=(1, 2, 3, 4)) + gt_boundary.sum(dim=(1, 2, 3, 4))

        surface_dice = (2.0 * intersection + self.eps) / (denom + self.eps)
        return 1.0 - surface_dice.mean()

class DiceBCESurfaceBinaryLoss(nn.Module):
    """
    Binary segmentation loss for 1-channel sigmoid logits.

    Expects:
      - logits: (B, 1, H, W, D)
      - labels: (B, 1, H, W, D) or (B, H, W, D), values {0,1}

      total loss =
        dice_weight    * DiceLoss
      + bce_weight     * BCEWithLogitsLoss
      + surface_weight * SoftSurfaceLoss
    """

    def __init__(
        self,
        pos_weight=None,
        dice_weight=1.0,
        bce_weight=1.0,
        surface_weight=0.1,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.surface_weight = surface_weight

        self.dice = DiceLoss(sigmoid=True)
        self.surface = SoftSurfaceLoss()

        if pos_weight is not None:
            pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, labels):
        if labels.ndim == logits.ndim - 1:
            labels = labels.unsqueeze(1)

        labels = labels.float()

        dice_loss = self.dice(logits, labels)
        bce_loss = self.bce(logits, labels)

        probs = torch.sigmoid(logits)
        #surface_loss = self.surface(probs, labels)

        return ((self.dice_weight * dice_loss + self.bce_weight * bce_loss #+ self.surface_weight * surface_loss),
                 ),
                self.dice_weight * dice_loss,
                self.bce_weight * bce_loss)

class DiceCELossMulticlass(nn.Module):
    """
    Multiclass segmentation loss.

    Expects:
      logits: (B, C, H, W, D)
      labels: (B, 1, H, W, D) or (B, H, W, D)

    labels must contain integer class indices:
      0 = background
      1 = level1
      2 = level2
      3 = level3
      4 = level4
      5 = imn
      6 = interpectoral

    total loss =
        dice_weight * DiceLoss
      + ce_weight   * CrossEntropyLoss
    """

    def __init__(
        self,
        dice_weight=1.0,
        ce_weight=1.0,
        class_weights=None,
        include_background=False,
        device="cuda"
    ):
        super().__init__()

        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

        self.dice = DiceLoss(
            softmax=True,
            to_onehot_y=True,
            include_background=include_background,
        )

        if class_weights is not None:
            class_weights = torch.as_tensor(class_weights, dtype=torch.float32).to(device)

        self.register_buffer("class_weights", class_weights)

        self.ce = nn.CrossEntropyLoss(
            weight=self.class_weights,
        )

    def forward(self, logits, labels):
        """
        logits: (B, C, H, W, D)
        labels: (B, 1, H, W, D) or (B, H, W, D)
        """

        # DiceLoss with to_onehot_y=True expects labels with channel dim:
        # (B, 1, H, W, D)
        if labels.ndim == logits.ndim - 1:
            labels_dice = labels.unsqueeze(1)
        else:
            labels_dice = labels

        labels_dice = labels_dice.long()

        # CrossEntropyLoss expects:
        # logits: (B, C, H, W, D)
        # target: (B, H, W, D)
        if labels.ndim == logits.ndim:
            labels_ce = labels.squeeze(1)
        else:
            labels_ce = labels

        labels_ce = labels_ce.long()

        dice_loss = self.dice(logits, labels_dice)
        ce_loss = self.ce(logits, labels_ce)

        total = (
            self.dice_weight * dice_loss
            + self.ce_weight * ce_loss
        )

        return (
            total,
            self.dice_weight * dice_loss,
            self.ce_weight * ce_loss,
        )

class DiceCELoss(_Loss):
    """
    Compute both Dice loss and Cross Entropy Loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in ``monai.losses.DiceLoss``.
    The details of Cross Entropy Loss is shown in ``torch.nn.CrossEntropyLoss`` and ``torch.nn.BCEWithLogitsLoss()``.
    In this implementation, two deprecated parameters ``size_average`` and ``reduce``, and the parameter ``ignore_index`` are
    not supported.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        weight: torch.Tensor | None = None,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
        label_smoothing: float = 0.0,
    ) -> None:
        """
        Args:
            ``lambda_ce`` are only used for cross entropy loss.
            ``reduction`` and ``weight`` is used for both losses and other parameters are only used for dice loss.

            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss` and `BCEWithLogitsLoss`.
            softmax: if True, apply a softmax function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss` and `BCEWithLogitsLoss`.
            other_act: callable function to execute other activation layers, Defaults to ``None``. for example:
                ``other_act = torch.tanh``. only used by the `DiceLoss`, not for the `CrossEntropyLoss` and `BCEWithLogitsLoss`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``. The dice loss should
                as least reduce the spatial dimensions, which is different from cross entropy loss, thus here
                the ``none`` option cannot be used.

                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            weight: a rescaling weight given to each class for cross entropy loss for `CrossEntropyLoss`.
                or a weight of positive examples to be broadcasted with target used as `pos_weight` for `BCEWithLogitsLoss`.
                See ``torch.nn.CrossEntropyLoss()`` or ``torch.nn.BCEWithLogitsLoss()`` for more information.
                The weight is also used in `DiceLoss`.
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_ce: the trade-off weight value for cross entropy loss. The value should be no less than 0.0.
                Defaults to 1.0.
            label_smoothing: a value in [0, 1] range. If > 0, the labels are smoothed
                by the given factor to reduce overfitting.
                Defaults to 0.0.

        """
        super().__init__()
        reduction = look_up_option(reduction, DiceCEReduction).value
        dice_weight: torch.Tensor | None
        if weight is not None and not include_background:
            dice_weight = weight[1:]
        else:
            dice_weight = weight
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
            weight=dice_weight,
        )
        if pytorch_after(1, 10):
            self.cross_entropy = nn.CrossEntropyLoss(
                weight=weight, reduction=reduction, label_smoothing=label_smoothing
            )
        else:
            self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        self.binary_cross_entropy = nn.BCEWithLogitsLoss(pos_weight=weight, reduction=reduction)
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_ce < 0.0:
            raise ValueError("lambda_ce should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.old_pt_ver = not pytorch_after(1, 10)

    def ce(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute CrossEntropy loss for the input logits and target.
        Will remove the channel dim according to PyTorch CrossEntropyLoss:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?#torch.nn.CrossEntropyLoss.

        """
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch != n_target_ch and n_target_ch == 1:
            target = torch.squeeze(target, dim=1)
            target = target.long()
        elif self.old_pt_ver:
            warnings.warn(
                f"Multichannel targets are not supported in this older Pytorch version {torch.__version__}. "
                "Using argmax (as a workaround) to convert target to a single channel."
            )
            target = torch.argmax(target, dim=1)
        elif not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)

        return self.cross_entropy(input, target)  # type: ignore[no-any-return]

    def bce(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Binary CrossEntropy loss for the input logits and target in one single class.

        """
        if not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)

        return self.binary_cross_entropy(input, target)  # type: ignore[no-any-return]

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 (without one-hot encoding) nor the same as input.

        Returns:
            torch.Tensor: value of the loss.

        """
        if input.dim() != target.dim():
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} (nb dims: {len(input.shape)}) and {target.shape} (nb dims: {len(target.shape)}). "
                "if target is not one-hot encoded, please provide a tensor with shape B1H[WD]."
            )

        if target.shape[1] != 1 and target.shape[1] != input.shape[1]:
            raise ValueError(
                "number of channels for target is neither 1 (without one-hot encoding) nor the same as input, "
                f"got shape {input.shape} and {target.shape}."
            )

        dice_loss = self.dice(input, target)
        ce_loss = self.ce(input, target) if input.shape[1] != 1 else self.bce(input, target)
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss

        return total_loss, dice_loss, ce_loss