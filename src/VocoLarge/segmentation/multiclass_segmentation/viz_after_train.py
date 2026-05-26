import os

import numpy as np
import torch
import nibabel as nib
import re
from monai.data import decollate_batch
from monai.utils import set_determinism

from src.VocoLarge.segmentation.models.build import build_model
from src.VocoLarge.segmentation.models.freeze import freeze_encoder, report_trainable_by_module
from src.VocoLarge.segmentation.models.voco_loader import load_voco_encoder_weights
from src.VocoLarge.segmentation.training.losses_metrics import build_metrics_multiclass_softmax
from src.VocoLarge.segmentation.training.visuals import save_overlay_png

from tqdm.auto import tqdm

from src.VocoLarge.segmentation.data.loaders_binary import build_all_datasets_and_loaders_multiclass
from src.VocoLarge.segmentation.multiclass_segmentation.config_multiclass import ConfigMulticlass
from src.VocoLarge.segmentation.training.losses_metrics import build_loss_multiclass

from monai.inferers import sliding_window_inference

def _safe_name(x):
    x = str(x)
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", x)


def _get_affine_from_batch(batch, key, item_idx):
    """
    Tries to recover the current MONAI affine after Orientationd/Spacingd/Cropping.

    Works with either:
      - MetaTensor metadata: batch[key].meta["affine"]
      - old-style metadata dict: batch[f"{key}_meta_dict"]["affine"]
      - fallback: identity affine
    """

    # Case 1: MONAI MetaTensor
    data = batch.get(key, None)
    if hasattr(data, "meta") and data.meta is not None:
        meta = data.meta
        if "affine" in meta:
            affine = meta["affine"]

            if isinstance(affine, torch.Tensor):
                # Often shape is (B, 4, 4)
                if affine.ndim == 3:
                    return affine[item_idx].detach().cpu().numpy()
                return affine.detach().cpu().numpy()

            affine = np.asarray(affine)
            if affine.ndim == 3:
                return affine[item_idx]
            return affine

    # Case 2: dictionary metadata
    meta_key = f"{key}_meta_dict"
    if meta_key in batch:
        meta = batch[meta_key]

        for affine_key in ["affine", "original_affine"]:
            if affine_key in meta:
                affine = meta[affine_key]

                if isinstance(affine, torch.Tensor):
                    if affine.ndim == 3:
                        return affine[item_idx].detach().cpu().numpy()
                    return affine.detach().cpu().numpy()

                if isinstance(affine, (list, tuple)):
                    affine = affine[item_idx]

                affine = np.asarray(affine)
                if affine.ndim == 3:
                    return affine[item_idx]
                return affine

    return np.eye(4)


def _save_nii(array, affine, out_path, dtype=np.uint8):
    array = np.asarray(array).astype(dtype)
    nii = nib.Nifti1Image(array, affine)
    nii.set_data_dtype(dtype)
    nib.save(nii, out_path)

def make_nifti_prediction_callback(cfg: ConfigMulticlass):
    """
    Saves NIfTI files in the current transformed space.

    Because your pipeline may include:
      - Orientationd
      - Spacingd
      - SpatialPadd
      - RandCropByLabelClassesd

    these files correspond to the model input space, not necessarily the original CT space.

    Saved per patch:
      - image
      - multiclass true label
      - one true binary label per class
      - one predicted binary mask per class/channel
    """

    def cb(dice, image, label, pred, case_ids, batch=None):
        out_dir = os.path.join(cfg.save_dir, "nifti_predictions_test")
        os.makedirs(out_dir, exist_ok=True)

        # image: (B, 1, X, Y, Z)
        # label: (B, 1, X, Y, Z)
        # pred: list length B, each item usually (C, X, Y, Z)
        batch_size = image.shape[0]

        for item_idx in range(batch_size):
            case_id = _safe_name(case_ids[item_idx])

            affine = (
                _get_affine_from_batch(batch, key="image", item_idx=item_idx)
                if batch is not None
                else np.eye(4)
            )

            # Save image as seen by the model.
            # This is normalized/clipped according to your transforms.
            img_np = image[item_idx, 0].detach().cpu().numpy().astype(np.float32)

            # Save multiclass true label: 0 background, 1 level2, 2 level3, etc.
            lab_np = label[item_idx, 0].detach().cpu().numpy().astype(np.uint8)

            base_name = f"case_{case_id}_patch_{item_idx:03d}_dice_{dice:.6f}"

            image_path = os.path.join(out_dir, f"{base_name}_image.nii.gz")
            label_path = os.path.join(out_dir, f"{base_name}_label_multiclass.nii.gz")

            _save_nii(img_np, affine, image_path, dtype=np.float32)
            _save_nii(lab_np, affine, label_path, dtype=np.uint8)

            pred_oh = pred[item_idx].detach().cpu().numpy().astype(np.uint8)
            # pred_oh shape: (C, X, Y, Z)
            # Usually:
            #   channel 0 = background
            #   channel 1 = level2
            #   channel 2 = level3
            #   channel 3 = level4
            #   channel 4 = interpectoral

            for class_name in cfg.multiclass_label:
                channel_idx = cfg.class_to_index[class_name]

                if channel_idx >= pred_oh.shape[0]:
                    raise ValueError(
                        f"Class '{class_name}' uses channel {channel_idx}, "
                        f"but prediction has shape {pred_oh.shape}."
                    )

                pred_mask = pred_oh[channel_idx].astype(np.uint8)

                # Binary true label for this class.
                true_mask = (lab_np == channel_idx).astype(np.uint8)

                pred_path = os.path.join(
                    out_dir,
                    f"{base_name}_pred_{class_name}_ch{channel_idx}.nii.gz",
                )

                true_path = os.path.join(
                    out_dir,
                    f"{base_name}_true_{class_name}_ch{channel_idx}.nii.gz",
                )

                _save_nii(pred_mask, affine, pred_path, dtype=np.uint8)
                _save_nii(true_mask, affine, true_path, dtype=np.uint8)

    return cb

def make_nifti_prediction_callback_sliding_window(cfg: ConfigMulticlass):
    """
    Saves full-volume NIfTI predictions after sliding-window inference.

    This assumes that the validation loader returns the full transformed volume:
        image: (B, 1, X, Y, Z)
        label: (B, 1, X, Y, Z)
        pred:  list length B, each item (C, X, Y, Z)

    Saved per case:
      - full image
      - full multiclass label
      - full multiclass prediction
      - one binary true mask per class
      - one binary predicted mask per class
    """

    def cb(dice, image, label, pred, case_ids, batch=None):
        out_dir = os.path.join(cfg.save_dir, "nifti_predictions_val_sliding_window")
        os.makedirs(out_dir, exist_ok=True)

        batch_size = image.shape[0]

        for item_idx in range(batch_size):
            case_id = _safe_name(case_ids[item_idx])

            affine = (
                _get_affine_from_batch(batch, key="image", item_idx=item_idx)
                if batch is not None
                else np.eye(4)
            )

            img_np = image[item_idx, 0].detach().cpu().numpy().astype(np.float32)
            lab_np = label[item_idx, 0].detach().cpu().numpy().astype(np.uint8)

            pred_oh = pred[item_idx].detach().cpu().numpy().astype(np.uint8)
            pred_multiclass = np.argmax(pred_oh, axis=0).astype(np.uint8)

            base_name = f"case_{case_id}_full_volume_dice_{dice:.6f}"

            image_path = os.path.join(out_dir, f"{base_name}_image.nii.gz")
            label_path = os.path.join(out_dir, f"{base_name}_label_multiclass.nii.gz")
            pred_mc_path = os.path.join(out_dir, f"{base_name}_pred_multiclass.nii.gz")

            _save_nii(img_np, affine, image_path, dtype=np.float32)
            _save_nii(lab_np, affine, label_path, dtype=np.uint8)
            _save_nii(pred_multiclass, affine, pred_mc_path, dtype=np.uint8)

            for class_name in cfg.multiclass_label:
                channel_idx = cfg.class_to_index[class_name]

                if channel_idx >= pred_oh.shape[0]:
                    raise ValueError(
                        f"Class '{class_name}' uses channel {channel_idx}, "
                        f"but prediction has shape {pred_oh.shape}."
                    )

                pred_mask = pred_oh[channel_idx].astype(np.uint8)
                true_mask = (lab_np == channel_idx).astype(np.uint8)

                pred_path = os.path.join(
                    out_dir,
                    f"{base_name}_pred_{class_name}_ch{channel_idx}.nii.gz",
                )

                true_path = os.path.join(
                    out_dir,
                    f"{base_name}_true_{class_name}_ch{channel_idx}.nii.gz",
                )

                _save_nii(pred_mask, affine, pred_path, dtype=np.uint8)
                _save_nii(true_mask, affine, true_path, dtype=np.uint8)

    return cb

def make_visuals_callback(cfg: ConfigMulticlass):
    def cb(dice, image, label, pred, case_ids):

        img_np = image[0, 0].detach().cpu().numpy()
        case_id = case_ids[0]
        lab_np = label[0, 0].detach().cpu().numpy().astype(np.int32)

        pred_oh = pred[0].detach().cpu()          # (C, X, Y, Z)
        pred_np = pred_oh.argmax(dim=0).numpy().astype(np.int32) #(X, Y, Z)

        out_dir = os.path.join(cfg.save_dir, "visuals_test")
        os.makedirs(out_dir, exist_ok=True)

        for sidx in cfg.visuals_slices:
            if 0 <= sidx < img_np.shape[-1]:
                out_path = os.path.join(
                    out_dir,
                    f"dice_{dice:06f}_case_{case_id}_slice_{sidx:03d}.png"
                )
                save_overlay_png(img_np, lab_np, pred_np, out_path, sidx)

    return cb

def make_visuals_callback_sliding_window(cfg: ConfigMulticlass):
    """
    Saves slice overlays from the full-volume sliding-window prediction.
    """

    def cb(dice, image, label, pred, case_ids, batch=None):
        out_dir = os.path.join(cfg.save_dir, "visuals_val_sliding_window")
        os.makedirs(out_dir, exist_ok=True)

        batch_size = image.shape[0]

        for item_idx in range(batch_size):
            img_np = image[item_idx, 0].detach().cpu().numpy()
            lab_np = label[item_idx, 0].detach().cpu().numpy().astype(np.int32)

            case_id = _safe_name(case_ids[item_idx])

            pred_oh = pred[item_idx].detach().cpu()              # (C, X, Y, Z)
            pred_np = pred_oh.argmax(dim=0).numpy().astype(np.int32)

            for sidx in cfg.visuals_slices:
                if 0 <= sidx < img_np.shape[-1]:
                    out_path = os.path.join(
                        out_dir,
                        f"dice_{dice:.6f}_case_{case_id}_slice_{sidx:03d}.png"
                    )

                    save_overlay_png(
                        img_np,
                        lab_np,
                        pred_np,
                        out_path,
                        sidx,
                    )

    return cb

def run_visualize(model, cfg: ConfigMulticlass, visuals_cb=None):

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    _, val_loader, _ = build_all_datasets_and_loaders_multiclass(cfg)

    loss_fn = build_loss_multiclass(cfg)

    model.eval()

    dice_metric, hd95_metric, post_pred, post_label = build_metrics_multiclass_softmax(cfg)
    dice_metric.reset()
    hd95_metric.reset()

    loss_running = 0.0
    steps = 0

    pbar = tqdm(val_loader, desc="Val", leave=False)

    for bi, batch in enumerate(pbar):
        img = batch["image"].to(device)
        lab = batch["label"].to(device).long()
        case_ids = batch["case_id"]

        logits = model(img)


        loss, _, _, = loss_fn(logits, lab)
        loss_running += float(loss.item())
        steps += 1

        pred_list = [post_pred(x) for x in decollate_batch(logits)]
        lab_oh_list = [post_label(x) for x in decollate_batch(lab)]

        batch_dice_metric, _, _, _ = build_metrics_multiclass_softmax(cfg)
        batch_dice_metric.reset()
        batch_dice_metric(y_pred=pred_list, y=lab_oh_list)

        batch_dice_agg = batch_dice_metric.aggregate()

        if isinstance(batch_dice_agg, tuple):
            batch_dice_values, _ = batch_dice_agg
        else:
            batch_dice_values = batch_dice_agg

        current_batch_dice = torch.nanmean(batch_dice_values).item()

        # Update running validation metrics
        dice_metric(y_pred=pred_list, y=lab_oh_list)
        hd95_metric(y_pred=pred_list, y=lab_oh_list)

        dice_agg = dice_metric.aggregate()

        if isinstance(dice_agg, tuple):
            dice_values, _ = dice_agg
        else:
            dice_values = dice_agg

        running_mean_dice = torch.nanmean(dice_values).item()

        if visuals_cb is not None:
            visuals_cb(
                dice=current_batch_dice,
                image=img,
                label=lab,
                pred=pred_list,
                case_ids=case_ids,
                batch=batch,
            )

        pbar.set_postfix(
            loss=f"{loss_running / max(steps, 1):.4f}",
            dice=f"{running_mean_dice:.4f}",
            case_ids=str(case_ids),
        )

def run_visualize_sliding_window(
    model,
    cfg: ConfigMulticlass,
    visuals_cbs=None,
    roi_size=(192, 192, 64),
    sw_batch_size=1,
    overlap=0.5,
):
    """
    Full-volume validation visualization using sliding-window inference.

    Important:
    The validation dataloader must return full validation volumes.
    If your validation transform still crops around the label, this function will only
    reconstruct the cropped region, not the complete CT.
    """

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    _, val_loader, _ = build_all_datasets_and_loaders_multiclass(cfg)

    loss_fn = build_loss_multiclass(cfg)

    dice_metric, hd95_metric, post_pred, post_label = build_metrics_multiclass_softmax(cfg)
    dice_metric.reset()
    hd95_metric.reset()

    loss_running = 0.0
    steps = 0

    if visuals_cbs is None:
        visuals_cbs = []

    if callable(visuals_cbs):
        visuals_cbs = [visuals_cbs]

    pbar = tqdm(val_loader, desc="Val sliding-window", leave=False)

    with torch.inference_mode():
        for bi, batch in enumerate(pbar):
            img = batch["image"].to(device)              # (B, 1, X, Y, Z)
            lab = batch["label"].to(device).long()       # (B, 1, X, Y, Z)
            case_ids = batch["case_id"]

            def predictor(x):
                return model(x)

            logits = sliding_window_inference(
                inputs=img,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=predictor,
                overlap=overlap,
                mode="gaussian",
            )

            loss, _, _ = loss_fn(logits, lab)
            loss_running += float(loss.item())
            steps += 1

            pred_list = [post_pred(x) for x in decollate_batch(logits)]
            lab_oh_list = [post_label(x) for x in decollate_batch(lab)]

            batch_dice_metric, _, _, _ = build_metrics_multiclass_softmax(cfg)
            batch_dice_metric.reset()
            batch_dice_metric(y_pred=pred_list, y=lab_oh_list)

            batch_dice_agg = batch_dice_metric.aggregate()

            if isinstance(batch_dice_agg, tuple):
                batch_dice_values, _ = batch_dice_agg
            else:
                batch_dice_values = batch_dice_agg

            current_batch_dice = torch.nanmean(batch_dice_values).item()

            dice_metric(y_pred=pred_list, y=lab_oh_list)
            hd95_metric(y_pred=pred_list, y=lab_oh_list)

            dice_agg = dice_metric.aggregate()

            if isinstance(dice_agg, tuple):
                dice_values, _ = dice_agg
            else:
                dice_values = dice_agg

            running_mean_dice = torch.nanmean(dice_values).item()

            for visuals_cb in visuals_cbs:
                visuals_cb(
                    dice=current_batch_dice,
                    image=img,
                    label=lab,
                    pred=pred_list,
                    case_ids=case_ids,
                    batch=batch,
                )

            pbar.set_postfix(
                loss=f"{loss_running / max(steps, 1):.4f}",
                dice=f"{running_mean_dice:.4f}",
                case_ids=str(case_ids),
                shape=str(tuple(img.shape[-3:])),
            )

def main():
    cfg = ConfigMulticlass()
    cfg.val_batch_size = 1

    set_determinism(seed=cfg.seed)

    model = build_model(cfg)
    load_voco_encoder_weights(model, cfg)
    freeze_encoder(model, cfg)
    report_trainable_by_module(model)
    #
    # visuals_cb = make_nifti_prediction_callback(cfg) if cfg.save_visuals else None
    #
    # run_visualize(model, cfg, visuals_cb=visuals_cb)

    visuals_cbs = []

    if cfg.save_visuals:
        visuals_cbs.append(make_nifti_prediction_callback_sliding_window(cfg))
        visuals_cbs.append(make_visuals_callback_sliding_window(cfg))

    run_visualize_sliding_window(
        model=model,
        cfg=cfg,
        visuals_cbs=visuals_cbs,
        roi_size=(192, 192, 64),
        sw_batch_size=1,
        overlap=0.5,
    )


if __name__ == "__main__":
    main()