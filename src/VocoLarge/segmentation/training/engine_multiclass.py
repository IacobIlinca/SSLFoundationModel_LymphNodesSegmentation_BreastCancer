import json
import os

import torch
from torch.optim import Adam
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler

from monai.data import decollate_batch

from src.VocoLarge.segmentation.config_binary import ConfigBinary
from src.VocoLarge.segmentation.multiclass_segmentation.config_multiclass import ConfigMulticlass
from src.VocoLarge.segmentation.training.history import History
from src.VocoLarge.segmentation.training.infer import infer_full_volume
from src.VocoLarge.segmentation.training.losses_metrics import (
    build_loss_binary_softmax,
    build_metrics_binary_softmax, build_loss_binary_sigmoid, build_metrics_binary_sigmoid,
    build_metrics_multiclass_softmax,
)
from src.VocoLarge.segmentation.training.plots import plot_loss_curves, plot_metric_curves
from src.VocoLarge.segmentation.data.loaders_binary import build_all_datasets_and_loaders
from torch.optim.lr_scheduler import CosineAnnealingLR

def save_config(cfg):
    os.makedirs(cfg.save_dir, exist_ok=True)
    with open(os.path.join(cfg.save_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)


def train_one_epoch(model, loader, optim, loss_fn, scaler, device, cfg: ConfigMulticlass, epoch=None):
    model.train()
    running = 0.0
    dice_loss_total = 0.0
    ce_loss_total = 0.0
    steps = 0

    pbar = tqdm(loader, desc=f"Train {epoch:04d}", leave=False, dynamic_ncols=True)

    for batch in pbar:
        img = batch["image"].to(device)
        lab = batch["label"].to(device).long()
        case_ids = batch["case_id"]

        optim.zero_grad(set_to_none=True)

        with autocast(enabled=cfg.amp):
            logits = model(img)
            loss, dice_loss, ce_loss = loss_fn(logits, lab)

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        running += float(loss.item())
        dice_loss_total += float(dice_loss.item())
        ce_loss_total += float(ce_loss.item())
        steps += 1

        pbar.set_postfix(
            loss=f"{running / max(steps, 1):.4f}",
            dice_loss=f"{dice_loss_total / max(steps, 1):.4f}",
            bce_loss=f"{ce_loss_total / max(steps, 1):.4f}",
            case_ids=str(case_ids),
        )

    return running / max(steps, 1), dice_loss_total / max(steps, 1), ce_loss_total / max(steps, 1),


@torch.no_grad()
def evaluate(model, loader, device, loss_fn, cfg: ConfigMulticlass, desc="Eval", visuals_cb=None, epoch=None):
    model.eval()

    dice_metric, hd95_metric, post_pred, post_label = build_metrics_multiclass_softmax(cfg)
    dice_metric.reset()
    hd95_metric.reset()

    loss_running = 0.0
    steps = 0

    class_names = cfg.multiclass_label
    # Expected:
    # [
    #     "level1",
    #     "level2",
    #     "level3",
    #     "level4",
    #     "imn",
    #     "interpectoral",
    # ]

    pbar = tqdm(loader, desc=desc, leave=False)

    for bi, batch in enumerate(pbar):
        img = batch["image"].to(device)
        lab = batch["label"].to(device).long()
        case_ids = batch["case_id"]

        if cfg.fast_val:
            logits = model(img)
        else:
            logits = infer_full_volume(model, img, cfg)

        loss, _, _, = loss_fn(logits, lab)
        loss_running += float(loss.item())
        steps += 1

        pred_list = [post_pred(x) for x in decollate_batch(logits)]
        lab_oh_list = [post_label(x) for x in decollate_batch(lab)]

        dice_metric(y_pred=pred_list, y=lab_oh_list)
        hd95_metric(y_pred=pred_list, y=lab_oh_list)

        if visuals_cb is not None and cfg.save_visuals:
            visuals_cb(
                epoch=epoch,
                batch_index=bi,
                image=img,
                label=lab,
                pred=pred_list,
                case_ids=case_ids
            )

        dice_agg = dice_metric.aggregate()

        if isinstance(dice_agg, tuple):
            dice_values, _ = dice_agg
        else:
            dice_values = dice_agg

        current_mean_dice = torch.nanmean(dice_values).item()

        pbar.set_postfix(
            loss=f"{loss_running / max(steps, 1):.4f}",
            dice=f"{current_mean_dice:.4f}",
            case_ids=str(case_ids),
        )

    mean_loss = loss_running / max(steps, 1)

    dice_agg = dice_metric.aggregate()
    hd95_agg = hd95_metric.aggregate()

    if isinstance(dice_agg, tuple):
        dice_values, dice_not_nans = dice_agg
    else:
        dice_values = dice_agg
        dice_not_nans = None

    if isinstance(hd95_agg, tuple):
        hd95_values, hd95_not_nans = hd95_agg
    else:
        hd95_values = hd95_agg
        hd95_not_nans = None

    dice_values = dice_values.detach().cpu()
    hd95_values = hd95_values.detach().cpu()

    if dice_not_nans is not None:
        dice_not_nans = dice_not_nans.detach().cpu()

    if hd95_not_nans is not None:
        hd95_not_nans = hd95_not_nans.detach().cpu()

    mean_dice = float(torch.nanmean(dice_values).item())
    mean_hd95 = float(torch.nanmean(hd95_values).item())

    metrics = {
        "loss": float(mean_loss),
        "dice": mean_dice,
        "hd95": mean_hd95,
    }

    for i, class_name in enumerate(class_names):
        metrics[f"dice_{class_name}"] = _safe_float(dice_values[i])
        metrics[f"hd95_{class_name}"] = _safe_float(hd95_values[i])

        if dice_not_nans is not None:
            metrics[f"dice_n_{class_name}"] = _safe_float(dice_not_nans[i])

        if hd95_not_nans is not None:
            metrics[f"hd95_n_{class_name}"] = _safe_float(hd95_not_nans[i])

    dice_metric.reset()
    hd95_metric.reset()

    return metrics

def _safe_float(x):
    """
    Converts tensor/scalar to Python float.
    Keeps nan/inf as float values, but avoids crashing.
    """
    if torch.is_tensor(x):
        x = x.detach().cpu()
        if x.numel() == 1:
            return float(x.item())
    return float(x)