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
    EnsureTyped,
    RandCropByLabelClassesd,
    RandCropByPosNegLabeld, DeleteItemsd,
)

from src.VocoLarge.segmentation.config import Config
from src.VocoLarge.segmentation.config_binary import ConfigBinary
from src.VocoLarge.segmentation.data.combine_masks_multicass import CombineBinaryMasksReportOverlapd
from src.VocoLarge.segmentation.data.build_binary_lymph_mask import BuildBinaryLymphMaskd

def get_transforms_multiclass(cfg: Config) -> Tuple[Compose, Compose]:
    # --------------------------------------------------
    # 1) Decide which keys are loaded at the start
    # --------------------------------------------------
    mask_keys = cfg.mask_keys or []
    keys = ["image"] + mask_keys

    # --------------------------------------------------
    # 2) Base transforms: load + orientation + intensity
    # --------------------------------------------------
    base = [
        LoadImaged(keys=keys, image_only=False),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes=cfg.axcodes),
    ]

    if cfg.do_resample:
        base += [
            Spacingd(
                keys=keys,
                pixdim=cfg.target_spacing,
                mode=("bilinear",) + ("nearest",) * len(mask_keys),
            )
        ]

    if cfg.norm_mode == "ct_clip_zscore":
        base += [
            ScaleIntensityRanged(
                keys=["image"],
                a_min=cfg.ct_clip[0],
                a_max=cfg.ct_clip[1],
                b_min=cfg.ct_clip[0],
                b_max=cfg.ct_clip[1],
                clip=True,
            ),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
        ]
    elif cfg.norm_mode == "ct_clip_0_1":
        base += [
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

    # --------------------------------------------------
    # 3) Build label
    # --------------------------------------------------
    if len(mask_keys) == 0:
        raise ValueError(
            "cfg.mask_keys must be provided in overfit_multiclass mode."
        )
    if cfg.mask_key_to_class_index is None:
        raise ValueError(
            "mask_key_to_class_index must be provided in overfit_multiclass mode."
        )

    base += [
        CombineBinaryMasksReportOverlapd(
            mask_keys=mask_keys,
            mask_key_to_class_index=cfg.mask_key_to_class_index,
            label_key="label",
        )
    ]

    # From this point onward, both modes should have image + label
    keys = ["image", "label"]

    # --------------------------------------------------
    # 4) Optional augmentations (train only)
    # --------------------------------------------------
    aug = []
    if cfg.light_aug:
        aug = [
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        ]

    # --------------------------------------------------
    # 5) Random crop around label
    # --------------------------------------------------
    train_crop = [
        RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=cfg.roi_size,
            num_classes=cfg.num_classes + 1,
            ratios=[0.0, 1.0, 1.0, 1.0],
            num_samples=cfg.num_samples_per_volume,
        )
    ]

    val_crop = []

    # --------------------------------------------------
    # 6) Convert to tensors
    # --------------------------------------------------
    typed = [EnsureTyped(keys=keys, dtype=torch.float32)]

    train_transform = Compose(base + aug + train_crop + typed)
    val_transform = Compose(base + val_crop + typed)

    return train_transform, val_transform


def get_transforms_binary(cfg: ConfigBinary) -> Tuple[Compose, Compose]:
    # --------------------------------------------------
    # 1) Build label
    # --------------------------------------------------
    if not getattr(cfg, "lymph_terms_json", None):
        raise ValueError(
            "cfg.lymph_terms_json must be provided in train_binary mode."
        )

    base = [
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),

        BuildBinaryLymphMaskd(
            image_key="image",
            mask_paths_key="mask_paths",
            output_key="label",
            lymph_terms_json=cfg.lymph_terms_json,
            no_lymph_patients_log_file=cfg.no_lymph_patients_log_file,
        ),
        EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),

        DeleteItemsd(keys=["mask_paths", "matched_mask_paths"]),
        Orientationd(keys=["image", "label"], axcodes=cfg.axcodes),
    ]

    # From this point onward, both modes should have image + label
    keys = ["image", "label"]

    if cfg.do_resample:
        base += [
            Spacingd(
                keys=keys,
                pixdim=cfg.target_spacing,
                mode=("bilinear", "nearest"),
            )
        ]

    if cfg.norm_mode == "ct_clip_zscore":
        base += [
            ScaleIntensityRanged(
                keys=["image"],
                a_min=cfg.ct_clip[0],
                a_max=cfg.ct_clip[1],
                b_min=cfg.ct_clip[0],
                b_max=cfg.ct_clip[1],
                clip=True,
            ),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
        ]
    elif cfg.norm_mode == "ct_clip_0_1":
        base += [
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


    # --------------------------------------------------
    # 3) Optional augmentations (train only)
    # --------------------------------------------------
    aug = []
    if cfg.light_aug:
        aug = [
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        ]

    # --------------------------------------------------
    # 4) Train, val crop sampling
    # --------------------------------------------------
    train_crop = [
        RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=cfg.roi_size,
            num_classes=2,
            ratios=[0.0, 1.0],
            num_samples=cfg.num_samples_per_volume,
        )
    ]

    # Fast validation: patch-based val instead of full-volume val
    if cfg.fast_val:
        val_crop = [
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=cfg.roi_size,
                num_classes=2,
                ratios=[0.0, 1.0],
                num_samples=cfg.num_samples_per_volume,
            )
        ]
    else:
        val_crop = []

    # --------------------------------------------------
    # 5) Convert to tensors
    # --------------------------------------------------
    typed = [EnsureTyped(keys=keys, dtype=torch.float32)]

    train_transform = Compose(base + aug + train_crop + typed)
    val_transform = Compose(base + val_crop + typed)

    return train_transform, val_transform


@DeprecationWarning # please use transform for each specific case
def get_transforms(cfg: Config) -> Tuple[Compose, Compose]:
    """
    Returns (train_transform, val_transform) for either:
      - old multiclass overfit mode
      - new binary training mode

    Validation behavior:
      - if cfg.fast_val == True: validation is patch-based (fast)
      - if cfg.fast_val == False: validation is full-volume (slow, proper evaluation)
    """

    # --------------------------------------------------
    # 1) Decide which keys are loaded at the start
    # --------------------------------------------------
    if cfg.task_mode == "overfit_multiclass":
        mask_keys = cfg.mask_keys or []
        keys = ["image"] + mask_keys

    elif cfg.task_mode == "train_binary":
        mask_keys = []
        keys = ["image"]

    else:
        raise ValueError(
            f"Unknown task_mode: {cfg.task_mode}. "
            f"Expected 'overfit_multiclass' or 'train_binary'."
        )

    # --------------------------------------------------
    # 2) Base transforms: load + orientation + intensity
    # --------------------------------------------------
    base = [
        LoadImaged(keys=keys, image_only=False),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes=cfg.axcodes),
    ]

    if cfg.do_resample:
        if cfg.task_mode == "overfit_multiclass":
            base += [
                Spacingd(
                    keys=keys,
                    pixdim=cfg.target_spacing,
                    mode=("bilinear",) + ("nearest",) * len(mask_keys),
                )
            ]
        elif cfg.task_mode == "train_binary":
            # Only image is loaded here. Label is built later.
            base += [
                Spacingd(
                    keys=["image"],
                    pixdim=cfg.target_spacing,
                    mode=("bilinear",),
                )
            ]

    if cfg.norm_mode == "ct_clip_zscore":
        base += [
            ScaleIntensityRanged(
                keys=["image"],
                a_min=cfg.ct_clip[0],
                a_max=cfg.ct_clip[1],
                b_min=cfg.ct_clip[0],
                b_max=cfg.ct_clip[1],
                clip=True,
            ),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
        ]
    elif cfg.norm_mode == "ct_clip_0_1":
        base += [
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

    # --------------------------------------------------
    # 3) Build label
    # --------------------------------------------------
    if cfg.task_mode == "overfit_multiclass":
        if len(mask_keys) == 0:
            raise ValueError(
                "cfg.mask_keys must be provided in overfit_multiclass mode."
            )
        if cfg.mask_key_to_class_index is None:
            raise ValueError(
                "mask_key_to_class_index must be provided in overfit_multiclass mode."
            )

        base += [
            CombineBinaryMasksReportOverlapd(
                mask_keys=mask_keys,
                mask_key_to_class_index=cfg.mask_key_to_class_index,
                label_key="label",
            )
        ]

    elif cfg.task_mode == "train_binary":
        if not getattr(cfg, "lymph_terms_json", None):
            raise ValueError(
                "cfg.lymph_terms_json must be provided in train_binary mode."
            )

        base += [
            BuildBinaryLymphMaskd(
                mask_paths_key="mask_paths",
                output_key="label",
                lymph_terms_json=cfg.lymph_terms_json,
                no_lymph_patients_log_file=cfg.no_lymph_patients_log_file,
            ),
            EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        ]

        # Binary label is built after image loading, so resampling is not supported yet.
        if cfg.do_resample:
            raise ValueError(
                "train_binary with do_resample=True is not supported yet because the "
                "binary label is built after image loading/resampling. "
                "Set do_resample=False for now."
            )

    # From this point onward, both modes should have image + label
    keys = ["image", "label"]

    # --------------------------------------------------
    # 4) Optional augmentations (train only)
    # --------------------------------------------------
    aug = []
    if cfg.light_aug:
        aug = [
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        ]

    # --------------------------------------------------
    # 5) Train crop sampling
    # --------------------------------------------------
    if cfg.task_mode == "overfit_multiclass":
        train_crop = [
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=cfg.roi_size,
                num_classes=cfg.num_classes + 1,
                ratios=[0.0, 1.0, 1.0, 1.0],
                num_samples=cfg.num_samples_per_volume,
            )
        ]

        val_crop = []

    elif cfg.task_mode == "train_binary":
        train_crop = [
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=cfg.roi_size,
                pos=1.0,
                neg=1.0,
                num_samples=cfg.num_samples_per_volume,
                image_key="image",
                image_threshold=0,
            )
        ]

        # Fast validation: patch-based val instead of full-volume val
        if getattr(cfg, "fast_val", False):
            val_crop = [
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=cfg.roi_size,
                    pos=1.0,
                    neg=1.0,
                    num_samples=cfg.fast_val_num_samples_per_volume,
                    image_key="image",
                    image_threshold=0,
                )
            ]
        else:
            val_crop = []

    # --------------------------------------------------
    # 6) Convert to tensors
    # --------------------------------------------------
    typed = [EnsureTyped(keys=keys, dtype=torch.float32)]

    train_transform = Compose(base + aug + train_crop + typed)
    val_transform = Compose(base + val_crop + typed)

    return train_transform, val_transform