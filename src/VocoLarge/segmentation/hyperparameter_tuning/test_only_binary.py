import json
import os
import torch
from torch import GradScaler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from monai.data import decollate_batch

from src.VocoLarge.segmentation.data.loaders_binary import build_all_datasets_and_loaders, maybe_limit_loader
from src.VocoLarge.segmentation.training.enigne_binary import train_one_epoch
from src.VocoLarge.segmentation.training.history import History
from src.VocoLarge.segmentation.training.infer import infer_full_volume
from src.VocoLarge.segmentation.training.losses_metrics import (
    build_loss_binary_sigmoid,
    build_metrics_binary_sigmoid,
)
from src.VocoLarge.segmentation.training.plots import plot_loss_curves, plot_metric_curves


def save_config(cfg):
    os.makedirs(cfg.save_dir, exist_ok=True)
    with open(os.path.join(cfg.save_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)


@torch.no_grad()
def evaluate(model, loader, device, loss_fn, cfg, desc="Eval", visuals_cb=None, epoch=None):
    model.eval()

    dice_metric, hd95_metric, post_pred, post_label = build_metrics_binary_sigmoid(cfg)
    dice_metric.reset()
    hd95_metric.reset()

    loss_running = 0.0
    steps = 0

    pbar = tqdm(loader, desc=desc, leave=False)

    for bi, batch in enumerate(pbar):
        img = batch["image"].to(device)
        lab = batch["label"].to(device).long()

        if cfg.fast_val:
            logits = model(img)
        else:
            logits = infer_full_volume(model, img, cfg)

        loss = loss_fn(logits, lab)
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
            )

        current_dice = dice_metric.aggregate().mean().item()
        pbar.set_postfix(dice=f"{current_dice:.4f}")

    mean_loss = loss_running / max(steps, 1)
    mean_dice = dice_metric.aggregate().mean().item()
    mean_hd95 = hd95_metric.aggregate().mean().item()

    return mean_loss, mean_dice, mean_hd95


def run_test_only(model, cfg, visuals_cb=None):
    save_config(cfg)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader, val_loader, _ = build_all_datasets_and_loaders(cfg)
    limit_train_loader = maybe_limit_loader(train_loader, 100)

    loss_fn = build_loss_binary_sigmoid(cfg)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = AdamW(
        params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    scheduler = None
    if cfg.use_scheduler:
        if cfg.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optim,
                T_max=cfg.epochs,  # full cosine cycle over all epochs
                eta_min=cfg.min_lr,
            )
        else:
            raise ValueError(f"Unknown scheduler_type: {cfg.scheduler_type}")
    scaler = GradScaler(enabled=cfg.amp)

    history = History()
    best_dice = -1.0

    epoch_bar = tqdm(range(1, cfg.epochs + 1), desc="Epochs")

    for epoch in epoch_bar:
        tr_loss = train_one_epoch(
            model=model,
            loader=limit_train_loader,
            optim=optim,
            loss_fn=loss_fn,
            scaler=scaler,
            device=device,
            cfg=cfg,
            epoch=epoch,
        )
    #     tr_loss = 1.0
        os.makedirs(cfg.save_dir, exist_ok=True)

        if epoch == 1 or epoch % cfg.log_every == 0:
            print(f"Epoch {epoch:04d} | train loss: {tr_loss:.4f}")

            val_loss, val_dice, val_hd95 = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                loss_fn=loss_fn,
                cfg=cfg,
                desc=f"Val {epoch:04d}",
                visuals_cb=visuals_cb,
                epoch=epoch,
            )

            print(
                f"Epoch {epoch:04d} | "
                f"val loss: {val_loss:.4f} | "
                f"val Dice: {val_dice:.4f} | "
                f"val HD95: {val_hd95:.4f}"
            )

            history.add(
                epoch=epoch,
                train_loss=tr_loss,
                val_loss=val_loss,
                val_dice=val_dice,
                val_hd95=val_hd95,
            )

            hist_path = os.path.join(cfg.save_dir, "history.json")
            with open(hist_path, "w") as f:
                json.dump(history.to_dict(), f, indent=2)

            plot_loss_curves(history, os.path.join(cfg.save_dir, "plots", "loss_curves.png"))
            plot_metric_curves(history, os.path.join(cfg.save_dir, "plots", "metric_curves.png"))

            os.makedirs(cfg.save_dir, exist_ok=True)
            if val_dice > best_dice:
                best_dice = val_dice

            epoch_bar.set_postfix(
                train_loss=f"{tr_loss:.4f}",
                val_dice=f"{val_dice:.4f}",
            )
        else:
            history.add_train_loss(epoch, tr_loss)
            epoch_bar.set_postfix(train_loss=f"{tr_loss:.4f}")

        if scheduler is not None:
            scheduler.step()

        current_lr = optim.param_groups[0]["lr"]
        print(f"[LR] epoch {epoch}: {current_lr:.2e}")

    plot_loss_curves(history, os.path.join(cfg.save_dir, "plots", "loss_curves.png"))
    plot_metric_curves(history, os.path.join(cfg.save_dir, "plots", "metric_curves.png"))

    print(f"Training done. Best val Dice: {best_dice:.4f}")
