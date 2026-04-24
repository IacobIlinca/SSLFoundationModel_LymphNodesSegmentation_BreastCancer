import json
import os
import torch
from torch import GradScaler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from monai.data import decollate_batch

from src.VocoLarge.segmentation.data.loaders_binary import build_all_datasets_and_loaders, maybe_limit_loader
from src.VocoLarge.segmentation.training.enigne_binary import train_one_epoch, evaluate
from src.VocoLarge.segmentation.training.history import History
from src.VocoLarge.segmentation.training.infer import infer_full_volume
from src.VocoLarge.segmentation.training.losses_metrics import (
    build_loss_binary_sigmoid,
    build_metrics_binary_sigmoid,
)
from src.VocoLarge.segmentation.training.plots import plot_loss_curves, plot_metric_curves, plot_train_loss_components


def save_config(cfg):
    os.makedirs(cfg.save_dir, exist_ok=True)
    with open(os.path.join(cfg.save_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)


def run_test_only(model, cfg, visuals_cb=None):
    save_config(cfg)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader, val_loader, _ = build_all_datasets_and_loaders(cfg)
    train_loader = maybe_limit_loader(train_loader, 20)
    val_loader = maybe_limit_loader(val_loader, 5)

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
        tr_loss, dice_loss, bce_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optim=optim,
            loss_fn=loss_fn,
            scaler=scaler,
            device=device,
            cfg=cfg,
            epoch=epoch,
        )
    #     tr_loss = 1.0
        os.makedirs(cfg.save_dir, exist_ok=True)

        current_lr = optim.param_groups[0]["lr"]
        if epoch == 1 or epoch % cfg.log_every == 0:
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

            history.add_train(
                epoch=epoch,
                loss=tr_loss,
                dice_loss=dice_loss,
                bce_loss=bce_loss,
            )

            history.add_val(
                epoch=epoch,
                loss=val_loss,
                dice=val_dice,
                hd95=val_hd95,
            )

            hist_path = os.path.join(cfg.save_dir, "history.json")
            with open(hist_path, "w") as f:
                json.dump(history.to_dict(), f, indent=2)

            plot_loss_curves(history, os.path.join(cfg.save_dir, "plots", "loss_curves.png"))
            plot_metric_curves(history, os.path.join(cfg.save_dir, "plots", "metric_curves.png"))
            plot_train_loss_components(history, os.path.join(cfg.save_dir, "plots", "loss_components.png"))

            os.makedirs(cfg.save_dir, exist_ok=True)
            if val_dice > best_dice:
                best_dice = val_dice

            epoch_bar.set_postfix(
                train_loss=f"{tr_loss:.4f}",
                val_dice=f"{val_dice:.4f}",
                current_lr=f"{current_lr:.4f}",
            )
        else:
            history.add_train(
                epoch=epoch,
                loss=tr_loss,
                dice_loss=dice_loss,
                bce_loss=bce_loss,
            )
            epoch_bar.set_postfix(train_loss=f"{tr_loss:.4f}",current_lr=f"{current_lr:.4f}",)

        if scheduler is not None:
            scheduler.step()

    plot_loss_curves(history, os.path.join(cfg.save_dir, "plots", "loss_curves.png"))
    plot_metric_curves(history, os.path.join(cfg.save_dir, "plots", "metric_curves.png"))
    plot_train_loss_components(history, os.path.join(cfg.save_dir, "plots", "loss_components.png"))

    print(f"Training done. Best val Dice: {best_dice:.4f}")
