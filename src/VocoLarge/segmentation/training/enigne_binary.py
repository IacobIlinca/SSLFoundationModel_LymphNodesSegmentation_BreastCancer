import json
import os

import torch
from torch.optim import Adam
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler

from monai.data import decollate_batch

from src.VocoLarge.segmentation.config_binary import ConfigBinary
from src.VocoLarge.segmentation.training.history import History
from src.VocoLarge.segmentation.training.infer import infer_full_volume
from src.VocoLarge.segmentation.training.losses_metrics import (
    build_loss_binary_softmax,
    build_metrics_binary_softmax, build_loss_binary_sigmoid, build_metrics_binary_sigmoid,
)
from src.VocoLarge.segmentation.training.plots import plot_loss_curves, plot_metric_curves
from src.VocoLarge.segmentation.data.loaders_binary import build_all_datasets_and_loaders
from torch.optim.lr_scheduler import CosineAnnealingLR


def save_config(cfg):
    os.makedirs(cfg.save_dir, exist_ok=True)
    with open(os.path.join(cfg.save_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)


def train_one_epoch(model, loader, optim, loss_fn, scaler, device, cfg, epoch=None):
    model.train()
    running = 0.0
    dice_loss_total = 0.0
    bce_loss_total = 0.0
    steps = 0

    pbar = tqdm(loader, desc=f"Train {epoch:04d}", leave=False, dynamic_ncols=True)

    for batch in pbar:
        img = batch["image"].to(device)
        lab = batch["label"].to(device).long()
        case_ids = batch["case_id"]

        optim.zero_grad(set_to_none=True)

        with autocast(enabled=cfg.amp):
            logits = model(img)
            loss, dice_loss, bce_loss = loss_fn(logits, lab)

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        running += float(loss.item())
        dice_loss_total += float(dice_loss.item())
        bce_loss_total += float(bce_loss.item())
        steps += 1

        pbar.set_postfix(
            loss=f"{running / max(steps, 1):.4f}",
            dice_loss=f"{dice_loss_total / max(steps, 1):.4f}",
            bce_loss=f"{bce_loss_total / max(steps, 1):.4f}",
            case_ids=str(case_ids),
        )

    return running / max(steps, 1), dice_loss_total / max(steps, 1), bce_loss_total / max(steps, 1),


@torch.no_grad()
def evaluate(model, loader, device, loss_fn, cfg: ConfigBinary, desc="Eval", visuals_cb=None, epoch=None):
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

        current_dice = dice_metric.aggregate().mean().item()
        pbar.set_postfix(dice=f"{current_dice:.4f}", case_ids=str(case_ids))

    mean_loss = loss_running / max(steps, 1)
    mean_dice = dice_metric.aggregate().mean().item()
    mean_hd95 = hd95_metric.aggregate().mean().item()

    return mean_loss, mean_dice, mean_hd95


def run_training(model, cfg: ConfigBinary, visuals_cb=None):
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

    loss_fn = build_loss_binary_sigmoid(cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = Adam(
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

        if scheduler is not None:
            scheduler.step()

        current_lr = optim.param_groups[0]["lr"]
        print(f"[LR] epoch {epoch}: {current_lr:.2e}")

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