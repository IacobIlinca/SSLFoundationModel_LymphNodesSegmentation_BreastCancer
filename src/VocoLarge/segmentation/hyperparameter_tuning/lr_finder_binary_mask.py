import os
import csv
import math
from copy import deepcopy

import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.amp import autocast
from torch.amp import GradScaler
from torch.optim import Adam
from src.VocoLarge.segmentation.hyperparameter_tuning.cofig_binary_test import ConfigBinaryTest
from src.VocoLarge.segmentation.models.build import build_model
from src.VocoLarge.segmentation.training.losses_metrics import build_loss_binary_sigmoid

from src.VocoLarge.segmentation.data.loaders_binary import build_all_datasets_and_loaders
from src.VocoLarge.segmentation.models.voco_loader import load_voco_encoder_weights
from src.VocoLarge.segmentation.models.freeze import freeze_encoder



def move_seg_batch_to_device(batch, device):
    """
    Expects batch to contain:
      - batch["image"]: (B, C, H, W, D)
      - batch["label"]: (B, 1, H, W, D) or (B, H, W, D)
    """
    img = batch["image"].to(device, non_blocking=True)
    lab = batch["label"].to(device, non_blocking=True)
    return img, lab


def set_optimizer_lr(optimizer, lr: float):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_optimizer_lr(optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def smooth_loss(loss, avg_loss, beta):
    if avg_loss is None:
        avg_loss = loss
    else:
        avg_loss = beta * avg_loss + (1.0 - beta) * loss
    return avg_loss


def forward_segmentation_loss(model, batch, loss_fn, device, use_amp=True):
    img, lab = move_seg_batch_to_device(batch, device)

    with autocast(device_type="cuda", enabled=(use_amp and device.type == "cuda")):
        logits = model(img)
        loss = loss_fn(logits, lab)

    return loss


def lr_range_test(
    model,
    optimizer,
    scaler,
    loader,
    loss_fn,
    device,
    start_lr=1e-7,
    end_lr=1e-2,
    num_iter=100,
    beta=0.98,
    stop_mult=4.0,
    use_amp=True,
):
    """
    Exponentially increases LR from start_lr to end_lr over num_iter steps.
    Returns lists of lrs and smoothed losses.
    """
    model.train()

    # Save initial states so the test does not permanently change the model
    model_state = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }
    opt_state = deepcopy(optimizer.state_dict())
    scaler_state = deepcopy(scaler.state_dict()) if scaler is not None else None

    set_optimizer_lr(optimizer, start_lr)
    mult = (end_lr / start_lr) ** (1 / max(1, num_iter - 1))

    lrs = []
    losses = []

    avg_loss = None
    best_loss = float("inf")
    iter_count = 0

    pbar = tqdm(total=num_iter, desc="LR Finder", leave=True)
    data_iter = iter(loader)

    while iter_count < num_iter:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        optimizer.zero_grad(set_to_none=True)

        loss = forward_segmentation_loss(
            model=model,
            batch=batch,
            loss_fn=loss_fn,
            device=device,
            use_amp=use_amp,
        )

        if scaler is not None and use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_val = float(loss.item())
        avg_loss = smooth_loss(loss_val, avg_loss, beta)
        smoothed = avg_loss / (1 - beta ** (iter_count + 1))

        lr = get_optimizer_lr(optimizer)

        lrs.append(lr)
        losses.append(smoothed)

        if smoothed < best_loss:
            best_loss = smoothed

        pbar.set_postfix(
            lr=f"{lr:.2e}",
            loss=f"{loss_val:.4f}",
            smooth=f"{smoothed:.4f}",
            best=f"{best_loss:.4f}",
        )
        pbar.update(1)

        if iter_count > 10 and smoothed > stop_mult * best_loss:
            print(f"[INFO] Stopping early: loss diverged at iter {iter_count + 1}")
            break

        lr *= mult
        set_optimizer_lr(optimizer, lr)
        iter_count += 1

    pbar.close()

    # Restore original states
    model.load_state_dict(model_state)
    optimizer.load_state_dict(opt_state)
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    return lrs, losses


def suggest_lr_from_curve(lrs, losses):
    """
    Heuristic:
    choose LR where the loss decreases fastest in log-LR space,
    and also return a conservative LR = aggressive / 10.
    """
    if len(lrs) < 5:
        return None, None

    log_lrs = [math.log10(x) for x in lrs]

    grads = []
    for i in range(1, len(losses)):
        dx = log_lrs[i] - log_lrs[i - 1]
        dy = losses[i] - losses[i - 1]
        grads.append(dy / dx)

    min_grad_idx = min(range(len(grads)), key=lambda i: grads[i])
    best_idx = min_grad_idx + 1

    suggested = lrs[best_idx]
    conservative = suggested / 10.0
    return suggested, conservative


def save_lr_finder_results(out_dir, lrs, losses):
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "lr_finder.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lr", "loss"])
        writer.writeheader()
        for lr, loss in zip(lrs, losses):
            writer.writerow({"lr": lr, "loss": loss})

    fig_path = os.path.join(out_dir, "lr_finder.png")
    plt.figure(figsize=(8, 5))
    plt.plot(lrs, losses)
    plt.xscale("log")
    plt.xlabel("Learning rate")
    plt.ylabel("Smoothed loss")
    plt.title("Segmentation LR range test")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    return csv_path, fig_path


def maybe_limit_loader(loader, max_batches=None):
    if max_batches is None:
        return loader

    class LimitedLoader:
        def __init__(self, loader, max_batches):
            self.loader = loader
            self.max_batches = max_batches

        def __iter__(self):
            for i, batch in enumerate(self.loader):
                if i >= self.max_batches:
                    break
                yield batch

        def __len__(self):
            return min(len(self.loader), self.max_batches)

    return LimitedLoader(loader, max_batches)


def main():
    cfg = ConfigBinaryTest()

    device = torch.device(
        "cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    # Build loaders
    train_loader, val_loader, _ = build_all_datasets_and_loaders(cfg)
    limited_train_loader = maybe_limit_loader(train_loader, max_batches=5)
    lr_loader = limited_train_loader

    # Optional speed limit
    # lr_loader = maybe_limit_loader(lr_loader, max_batches=20)

    model = build_model(cfg).to(device)
    load_voco_encoder_weights(model, cfg)
    freeze_encoder(model, cfg)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(
        params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    loss_fn = build_loss_binary_sigmoid(cfg)
    scaler = GradScaler("cuda", enabled=(cfg.amp and device.type == "cuda"))

    out_dir = os.path.join(cfg.save_dir, "lr_finder")
    os.makedirs(out_dir, exist_ok=True)

    lrs, losses = lr_range_test(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        loader=lr_loader,
        loss_fn=loss_fn,
        device=device,
        start_lr=1e-7,
        end_lr=1.0,
        num_iter=100,
        beta=0,
        stop_mult=4.0,
        use_amp=cfg.amp,
    )

    csv_path, fig_path = save_lr_finder_results(out_dir, lrs, losses)
    suggested, conservative = suggest_lr_from_curve(lrs, losses)

    print(f"[INFO] Saved CSV:  {csv_path}")
    print(f"[INFO] Saved plot: {fig_path}")

    if suggested is not None:
        print(f"[INFO] Suggested LR (aggressive):   {suggested:.3e}")
        print(f"[INFO] Suggested LR (conservative): {conservative:.3e}")
    else:
        print("[WARN] Could not compute a robust suggested LR.")


if __name__ == "__main__":
    main()