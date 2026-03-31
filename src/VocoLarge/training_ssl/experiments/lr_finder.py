import os
import csv
import math
from copy import deepcopy

import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from src.VocoLarge.training_ssl.pipeline.config import Config
from src.VocoLarge.training_ssl.training.datasets_and_loaders import build_all_datasets_and_loaders
from src.VocoLarge.training_ssl.training.engine import setup_model_and_optimizer
from src.VocoLarge.third_party_voco_large.utils.ops import concat_image
from src.VocoLarge.training_ssl.pipeline import to_device, forward_loss


def move_voco_batch_to_device(batch, device):
    img, labels, crops = batch
    img, crops = concat_image(img), concat_image(crops)
    img, crops, labels = to_device(img, crops, labels, device)
    return img, labels, crops


def set_optimizer_lr(optimizer, lr: float):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_optimizer_lr(optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def smooth_loss(loss, avg_loss, beta):
    if avg_loss is None:
        avg_loss = loss
    else:
        avg_loss = beta * avg_loss + (1 - beta) * loss
    return avg_loss


def lr_range_test(
    model,
    optimizer,
    scaler,
    loader,
    device,
    start_lr=1e-7,
    end_lr=1.0,
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

    # Save initial states so the test does not permanently damage the model
    model_state = deepcopy(model.state_dict())
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

        img, labels, crops = move_voco_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        loss = forward_loss(model, img, crops, labels, use_amp=(use_amp and device.type == "cuda"))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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

    # Restore states
    model.load_state_dict(model_state)
    optimizer.load_state_dict(opt_state)
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    return lrs, losses


def suggest_lr_from_curve(lrs, losses):
    """
    Simple heuristic:
    choose LR at minimum numerical gradient of loss vs log10(lr)
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
    plt.title("LR range test")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    return csv_path, fig_path


def maybe_limit_test_loader(test_loader, max_batches=None):
    if max_batches is None:
        return test_loader

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

    return LimitedLoader(test_loader, max_batches)


def main():
    args = Config()

    args.epochs = 1  # not used, but keeps intent clear

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    # Build your normal test loader
    _, _, test_loader = build_all_datasets_and_loaders(args)

    # Optional: limit batches for speed if your loader is large
    #test_loader = maybe_limit_test_loader(test_loader, max_batches=1)

    model, optimizer, scaler = setup_model_and_optimizer(args, device)

    out_dir = os.path.join(args.out_dir, "lr_finder")
    os.makedirs(out_dir, exist_ok=True)

    lrs, losses = lr_range_test(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        loader=test_loader,
        device=device,
        start_lr=1e-8,
        end_lr=1e-2,
        num_iter=50,
        beta=0.1,
        stop_mult=4.0,
        use_amp=args.amp,
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