import os

import numpy as np
import torch
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


def make_visuals_callback(cfg: ConfigMulticlass):
    def cb(dice, image, label, pred, case_ids):
        # Only visualize if Dice is lower than given
        if dice >= 0.55:
            return

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

def run_visualize(model, cfg: ConfigMulticlass, visuals_cb=None):

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader, val_loader, _ = build_all_datasets_and_loaders_multiclass(cfg)

    loss_fn = build_loss_multiclass(cfg)

    model.eval()

    dice_metric, hd95_metric, post_pred, post_label = build_metrics_multiclass_softmax(cfg)
    dice_metric.reset()
    hd95_metric.reset()

    loss_running = 0.0
    steps = 0

    pbar = tqdm(train_loader, desc="Val", leave=False)

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
                case_ids=case_ids
            )

        pbar.set_postfix(
            loss=f"{loss_running / max(steps, 1):.4f}",
            dice=f"{running_mean_dice:.4f}",
            case_ids=str(case_ids),
        )


def main():
    cfg = ConfigMulticlass()

    set_determinism(seed=cfg.seed)

    model = build_model(cfg)
    load_voco_encoder_weights(model, cfg)
    freeze_encoder(model, cfg)
    report_trainable_by_module(model)

    visuals_cb = make_visuals_callback(cfg) if cfg.save_visuals else None

    run_visualize(model, cfg, visuals_cb=visuals_cb)


if __name__ == "__main__":
    main()