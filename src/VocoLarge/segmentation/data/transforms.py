from typing import Tuple
import torch

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    NormalizeIntensityd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandCropByPosNegLabeld,
    EnsureTyped,
    Identityd
)

from src.VocoLarge.segmentation.config import Config
from src.VocoLarge.segmentation.data.combine_masks import CombineBinaryMasksToLabeld


def get_transforms(cfg: Config) -> Tuple[Compose, Compose]:
    """
    REQUIRED.

    Returns (train_transform, val_transform).

    Key ideas:
      - We standardize orientation (RAS).
      - Optional resampling (Spacingd) is OFF by default. Turn it on later for consistent spacing.
      - For CT, we (optionally) clip HU-ish values and normalize to [0,1].
      - Training uses patch sampling with foreground bias to avoid "all background" patches.
      - Validation uses full volume (no cropping), evaluated via sliding window inference.

    Optional parts are explicitly labeled below.
    """
    mask_keys = cfg.mask_keys or []
    keys = ["image"] + mask_keys

    # ---- Required: load image+label + standardize shape ----
    base = [
        # Load NIfTI files into arrays + keep metadata (affine/spacing).
        LoadImaged(keys=keys, image_only=False),
        # Ensure channel-first: (C,H,W,D). For NIfTI, C becomes 1.
        EnsureChannelFirstd(keys=keys),
        # Standardize orientation so all cases are comparable.
        Orientationd(keys=keys, axcodes=cfg.axcodes),
    ]

    # ---- OPTIONAL but recommended later: resample to target spacing ----
    # Why optional now? You said "don't mind spacing now".
    # BUT: for best comparisons and stable training, consistent spacing helps.
    if cfg.do_resample:
        if len(mask_keys) > 0:
            base += [
                Spacingd(
                    keys=keys,
                    pixdim=cfg.target_spacing,
                    mode=("bilinear",) + ("nearest",) * len(mask_keys),
                )
            ]
        else:
            base += [Spacingd(keys=["image"], pixdim=cfg.target_spacing, mode=("bilinear",))]

    # ---- Intensity handling ----
    # CT audit indicates HU-like data with lots of air at -1000 and typical caps ~3071,
    # so clipping then z-score is a strong default for transformer backbones.
    if cfg.norm_mode == "ct_clip_zscore":
        base += [
            # Clip HU range first to remove outliers (metal/artifacts etc.)
            ScaleIntensityRanged(
                keys=["image"],
                a_min=cfg.ct_clip[0],
                a_max=cfg.ct_clip[1],
                b_min=cfg.ct_clip[0],  # keep in HU space after clipping
                b_max=cfg.ct_clip[1],
                clip=True,
            ),
            # Z-score normalize (mean=0, std=1) after clipping
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
        ]
    elif cfg.norm_mode == "ct_clip_0_1":
        base += [
            # Clip HU and map to [0,1]
            ScaleIntensityRanged(
                keys=["image"],
                a_min=cfg.ct_clip[0],
                a_max=cfg.ct_clip[1],
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
        ]
    elif cfg.norm_mode == "zscore":
        base += [
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
        ]
    else:
        raise ValueError(f"Unknown norm_mode: {cfg.norm_mode}")

    # ---- Combining binary masks into one label ----
    if len(mask_keys) > 0:
        if cfg.mask_key_to_class_index is None:
            raise ValueError("mask_key_to_class_index must be provided when mask_keys are used.")
        base += [
            CombineBinaryMasksToLabeld(
                mask_keys=mask_keys,
                mask_key_to_class_index=cfg.mask_key_to_class_index,
                label_key="label",
                raise_on_overlap=True,  # collision rule
            )
        ]
    keys = ["image", "label"]

    # ---- OPTIONAL: light augmentations (keep light for linear probe) ----
    # Augmentations are helpful but keep them mild since you are probing representations.
    aug = Identityd
    if cfg.light_aug:
        aug = [
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys=keys, factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=keys, offsets=0.1, prob=0.5),
        ]

    # ---- Required: patch sampling focused on foreground ----
    # This is critical for small/rare LN levels; otherwise many patches are empty background.
    crop = [
        RandCropByPosNegLabeld(
            keys=keys,
            label_key="label",
            spatial_size=cfg.roi_size,
            pos=2,                       # 2 parts positive
            neg=1,                       # 1 part negative
            num_samples=cfg.num_samples_per_volume,
            image_key="image",
            image_threshold=0,           # label drives pos/neg; threshold rarely matters here
        )
    ]

    # ---- Required: convert to torch tensors ----
    # NOTE: EnsureTyped with float32 will cast label to float; we convert label->long in the training step.
    # (We do this because MONAI often expects float tensors in pipelines; it's OK as long as we cast later.)
    typed = [EnsureTyped(keys=keys, dtype=torch.float32)]

    train_transform = Compose(base + crop + typed)
    val_transform = Compose(base + typed)
    return train_transform, val_transform