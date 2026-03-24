import json
import os

import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from monai.data import decollate_batch

from src.VocoLarge.segmentation.training.history import History
from src.VocoLarge.segmentation.training.infer import infer_full_volume
from src.VocoLarge.segmentation.training.losses_metrics import (
    build_loss_binary,
    build_metrics_binary,
)
from src.VocoLarge.segmentation.training.plots import plot_loss_curves, plot_metric_curves
from src.VocoLarge.segmentation.data.loaders_binary import build_all_datasets_and_loaders


def save_config(cfg):
    os.makedirs(cfg.save_dir, exist_ok=True)
    with open(os.path.join(cfg.save_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)


def train_one_epoch(model, loader, optim, loss_fn, scaler, device, cfg, epoch=None):
    model.train()
    running = 0.0
    steps = 0

    pbar = tqdm(loader, desc=f"Train {epoch:04d}" if epoch is not None else "Train", leave=False)

    for batch in pbar:
        img = batch["image"].to(device)
        lab = batch["label"].to(device).long()

        optim.zero_grad(set_to_none=True)

        with autocast(enabled=cfg.amp):
            logits = model(img)
            loss = loss_fn(logits, lab)

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        running += float(loss.item())
        steps += 1

        pbar.set_postfix(loss=f"{running / max(steps, 1):.4f}")

    return running / max(steps, 1)


@torch.no_grad()
def evaluate(model, loader, device, loss_fn, cfg, desc="Eval", visuals_cb=None, epoch=None):
    model.eval()

    dice_metric, hd95_metric, post_pred, post_label = build_metrics_binary(cfg)
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


def run_training(model, cfg, visuals_cb=None):
    """
    Binary training entry point.

    Expects:
      - cfg.task_mode == "train_binary"
      - split txt files already created
    """
    save_config(cfg)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader, val_loader, test_loader = build_all_datasets_and_loaders(cfg)

    loss_fn = build_loss_binary(cfg)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scaler = GradScaler(enabled=cfg.amp)

    history = History()
    best_dice = -1.0

    epoch_bar = tqdm(range(1, cfg.epochs + 1), desc="Epochs")

    for epoch in epoch_bar:
        tr_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optim=optim,
            loss_fn=loss_fn,
            scaler=scaler,
            device=device,
            cfg=cfg,
            epoch=epoch,
        )
        os.makedirs(cfg.save_dir, exist_ok=True)

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "epoch": epoch,
                "train_loss": tr_loss,
                "best_dice": best_dice,
            },
            os.path.join(cfg.save_dir, "pre_val_last.pt"),
        )

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
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "best_dice": best_dice},
                    os.path.join(cfg.save_dir, "best.pt"),
                )

            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "best_dice": best_dice},
                os.path.join(cfg.save_dir, "last.pt"),
            )

            epoch_bar.set_postfix(
                train_loss=f"{tr_loss:.4f}",
                val_dice=f"{val_dice:.4f}",
            )
        else:
            epoch_bar.set_postfix(train_loss=f"{tr_loss:.4f}")

    plot_loss_curves(history, os.path.join(cfg.save_dir, "plots", "loss_curves.png"))
    plot_metric_curves(history, os.path.join(cfg.save_dir, "plots", "metric_curves.png"))

    print(f"Training done. Best val Dice: {best_dice:.4f}")

    # Final test evaluation
    test_loss, test_dice, test_hd95 = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        loss_fn=loss_fn,
        cfg=cfg,
        desc="Test",
        visuals_cb=None,
        epoch=None,
    )

    print(
        f"Test results | "
        f"loss: {test_loss:.4f} | "
        f"Dice: {test_dice:.4f} | "
        f"HD95: {test_hd95:.4f}"
    )

    with open(os.path.join(cfg.save_dir, "test_metrics.json"), "w") as f:
        json.dump(
            {
                "test_loss": test_loss,
                "test_dice": test_dice,
                "test_hd95": test_hd95,
                "best_val_dice": best_dice,
            },
            f,
            indent=2,
        )