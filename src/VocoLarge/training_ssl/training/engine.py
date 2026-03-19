import os

import torch
from torch.optim import SGD, AdamW

from src.VocoLarge.training_ssl.pipeline import build_model, load_ckpt, save_ckpt_atomic, save_diff_bundle
from src.VocoLarge.training_ssl.pipeline.config import Config
from src.VocoLarge.training_ssl.pipeline.freeze import report_trainable_by_module
from src.VocoLarge.training_ssl.pipeline.training import train_one_epoch, validate_one_epoch, \
    compute_logits_targets_for_one_image
from src.VocoLarge.training_ssl.pipeline.viz import History, plot_loss_curves
from src.VocoLarge.training_ssl.training.datasets_and_loaders import build_all_datasets_and_loaders


def build_optimizer(args: Config, model):
    params = [p for p in model.parameters() if p.requires_grad]

    if args.optimizer.lower() == "sgd":
        return SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.optimizer.lower() == "adamw":
        return AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def setup_model_and_optimizer(args: Config, device):
    model = build_model(args, device).train()

    if args.voco_ckpt_path:
        stats = load_ckpt(model, args.voco_ckpt_path, args.device, mode=args.load_mode)
        print(f"[ckpt] load_mode={args.load_mode} stats={stats}")
    else:
        print("[ckpt] no checkpoint provided; training from scratch")

    report_trainable_by_module(model)

    optimizer = build_optimizer(args, model)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    return model, optimizer, scaler

def save_checkpoint(
    save_path: str,
    model,
    optimizer,
    scaler,
    epoch: int,
):
    payload = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
    }
    save_ckpt_atomic(save_path, payload)


def run_training(args: Config):
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    train_loader, val_loader, _ = build_all_datasets_and_loaders(args)

    batch_to_viz = next(iter(val_loader))

    model, optimizer, scaler = setup_model_and_optimizer(args, device)

    history = History()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, args)
        #train_loss = 0.5
        history.add_train_loss(epoch, train_loss)

        print(f"epoch {epoch:04d}/{args.epochs} | train_loss={train_loss:.6f}")

        should_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if should_eval:
            val_loss, top1 = validate_one_epoch(model, val_loader, device, epoch)# evaluate on val loader

            print(
                f"[eval] epoch {epoch:04d} "
                f"val_loss={val_loss:.6f} "
                f"top1={top1*100:.2f}% "
            )
            history.add_val_loss(epoch, val_loss)
            history.add_top1_metric(top1)

        should_save = (epoch % args.save_every == 0) or (epoch == args.epochs)
        if should_save:

            logits, targets = compute_logits_targets_for_one_image(model, batch_to_viz)
            save_diff_bundle(logits, targets, args.out_dir, prefix=f"epoch{epoch:05d}")

            ckpt_path = os.path.join(args.out_dir, f"epoch_{epoch:04d}.pt")
            save_checkpoint(ckpt_path, model, optimizer, scaler, epoch)
            print(f"[ckpt] saved: {ckpt_path}")


        plot_loss_curves(history, args.out_dir)

    final_path = os.path.join(args.out_dir, "final.pt")
    save_checkpoint(final_path, model, optimizer, scaler, args.epochs)
    print(f"[ckpt] saved final: {final_path}")

    return model