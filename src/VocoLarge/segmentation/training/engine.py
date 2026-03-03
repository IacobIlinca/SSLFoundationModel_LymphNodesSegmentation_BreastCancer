import os
import json
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from monai.data import CacheDataset, decollate_batch, list_data_collate

from src.VocoLarge.segmentation.config import Config
from src.VocoLarge.segmentation.data.transforms import get_transforms
from src.VocoLarge.segmentation.training.history import History
from src.VocoLarge.segmentation.training.infer import infer_full_volume
from src.VocoLarge.segmentation.training.losses_metrics import build_loss, build_metrics
from src.VocoLarge.segmentation.training.plots import plot_loss_curves, plot_metric_curves


def save_config(cfg: Config):
    os.makedirs(cfg.save_dir, exist_ok=True)
    with open(os.path.join(cfg.save_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)


def make_loaders(train_samples, val_samples, cfg: Config):
    """
    REQUIRED.

    Uses CacheDataset for speed. If you are iterating transforms rapidly, set cache_rate=0.
    """
    train_tf, val_tf = get_transforms(cfg)

    train_ds = CacheDataset(train_samples, transform=train_tf, cache_rate=0.2, num_workers=cfg.num_workers)
    val_ds = CacheDataset(val_samples, transform=val_tf, cache_rate=0.2, num_workers=cfg.num_workers)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=list_data_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=list_data_collate,
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, optim, loss_fn, scaler, device, cfg: Config, epoch=None):
    model.train()
    running = 0.0
    steps = 0

    pbar = tqdm(loader, desc=f"Train {epoch:04d}" if epoch is not None else "Train", leave=False)

    for batch in pbar:
        img = batch["image"].to(device)

        if cfg.label_mode == "multiclass":
            lab = batch["label"].to(device).long()
        else:
            lab = batch["label"].to(device).float()

        optim.zero_grad(set_to_none=True)
        with autocast(enabled=cfg.amp):
            logits = model(img)
            loss = loss_fn(logits, lab)

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        running += loss.item()
        steps += 1

        pbar.set_postfix(loss=f"{running/steps:.4f}")

    return running / max(steps, 1)


@torch.no_grad()
def validate(model, val_loader, device, loss_fn, epoch, cfg: Config, history, tr_loss, best_dice, visuals_cb=None):
    """
    REQUIRED.

    - sliding window inference on full volumes
    - Dice + HD95 metrics
    - optional visuals callback
    """

    model.eval()
    dice_metric, hd95_metric, post_pred, post_label = build_metrics(cfg)
    dice_metric.reset()
    hd95_metric.reset()

    val_loss_running = 0.0
    val_steps = 0

    pbar_val = tqdm(val_loader, desc=f"Val {epoch:04d}", leave=False)

    for vi, batch in enumerate(pbar_val):

        img = batch["image"].to(device)
        if cfg.label_mode == "multilabel":
            lab = batch["label"].to(device).float()
        else:
            lab = batch["label"].to(device).long()

        logits = infer_full_volume(model, img, cfg)

        # ----- val loss (optional conceptually, required for loss curve plot) -----
        # IMPORTANT: DiceCELoss expects raw logits + integer labels.
        # This is computed on the full-volume logits, so it is more expensive than patch val.
        vloss = loss_fn(logits, lab)
        val_loss_running += float(vloss.item())
        val_steps += 1

        # ----- metrics -----
        logits_for_metrics = torch.sigmoid(logits) if cfg.label_mode == "multilabel" else logits
        pred_list = [post_pred(x) for x in decollate_batch(logits_for_metrics)]
        lab_oh_list = [post_label(x) for x in decollate_batch(lab)]

        dice_metric(y_pred=pred_list, y=lab_oh_list)
        hd95_metric(y_pred=pred_list, y=lab_oh_list)

        # OPTIONAL visuals
        if visuals_cb is not None and cfg.save_visuals:
            visuals_cb(epoch=epoch, batch_index=vi, image=img, label=lab, pred=pred_list)

        pbar_val.set_postfix(
            dice=f"{dice_metric.aggregate().mean().item():.4f}" if val_steps > 0 else "0.0000"
        )

    val_dice = dice_metric.aggregate().mean().item()
    val_hd95 = hd95_metric.aggregate().mean().item()
    val_loss = val_loss_running / max(val_steps, 1)

    if epoch == 1 or epoch % cfg.log_every == 0:
        print(f"Epoch {epoch:04d} | val Dice: {val_dice:.4f} | val HD95: {val_hd95:.4f}")

    # Store curves
    history.add(epoch=epoch, train_loss=tr_loss, val_loss=val_loss, val_dice=val_dice, val_hd95=val_hd95)

    # Optionally write history to disk each log interval
    if epoch == 1 or epoch % cfg.log_every == 0:
        hist_path = os.path.join(cfg.save_dir, "history.json")
        with open(hist_path, "w") as f:
            json.dump(history.to_dict(), f, indent=2)

        # Save plots periodically
        plot_loss_curves(history, os.path.join(cfg.save_dir, "plots", "loss_curves.png"))
        plot_metric_curves(history, os.path.join(cfg.save_dir, "plots", "metric_curves.png"))

    # Checkpoints
    os.makedirs(cfg.save_dir, exist_ok=True)
    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "best_dice": best_dice},
            os.path.join(cfg.save_dir, "best.pt"),
        )

    if epoch % cfg.log_every == 0:
        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "best_dice": best_dice},
            os.path.join(cfg.save_dir, "last.pt"),
        )

    return val_dice, val_hd95


def run_training(model, train_samples, val_samples, cfg: Config, visuals_cb=None):
    """
    REQUIRED.

    Handles:
      - overfit mode filtering (debug)
      - training loop
      - checkpoint saving
    """
    save_config(cfg)

    if cfg.overfit_case_id is not None:
        train_samples = [s for s in train_samples if s.get("case_id") == cfg.overfit_case_id]
        val_samples = train_samples.copy()
        print(f"[Overfit] Using only case_id={cfg.overfit_case_id}. n={len(train_samples)}")
        if len(train_samples) != 1:
            raise ValueError("Overfit mode expects exactly 1 matching case.")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader, val_loader = make_loaders(train_samples, val_samples, cfg)

    loss_fn = build_loss(cfg)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scaler = GradScaler(enabled=cfg.amp)

    best_dice = -1.0

    history = History()

    epoch_bar = tqdm(range(1, cfg.epochs + 1), desc="Epochs")

    for epoch in epoch_bar:
        tr_loss = train_one_epoch(model, train_loader, optim, loss_fn, scaler, device, cfg)

        # Validation plots and checkpoints
        if epoch == 1 or epoch % cfg.log_every == 0:
            print(f"Epoch {epoch:04d} | train loss: {tr_loss:.4f}")
            val_dice, _ = validate(model, train_loader, device, loss_fn, epoch, cfg, history, tr_loss, best_dice, visuals_cb)


        epoch_bar.set_postfix(
            train_loss=f"{tr_loss:.4f}",
            val_dice=f"{val_dice:.4f}",
        )

    plot_loss_curves(history, os.path.join(cfg.save_dir, "plots", "loss_curves.png"))
    plot_metric_curves(history, os.path.join(cfg.save_dir, "plots", "metric_curves.png"))
    print(f"Training done. Best val Dice: {best_dice:.4f}")