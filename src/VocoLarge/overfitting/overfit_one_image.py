import os
import csv
import argparse
from pathlib import Path

import torch
from torch.optim import AdamW

from src.VocoLarge.pipeline import (
    build_voco_args,
    build_transforms,
    build_model,
    load_ckpt,
    save_ckpt_atomic,
    unpack_voco_output,
    to_device,
    forward_loss,
    compute_logits_targets,
    save_voco_debug_vis,
    save_diff_bundle,
    top1_match,
    mae,
    mse,
    disable_dropout,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--ckpt", default=None)

    p.add_argument("--load_mode", default="backbone", choices=["backbone", "full"])

    p.add_argument("--roi_x", type=int, default=96)
    p.add_argument("--roi_y", type=int, default=96)
    p.add_argument("--roi_z", type=int, default=64)
    p.add_argument("--feature_size", type=int, default=48)
    p.add_argument("--no_aug", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save_every", type=int, default=200)

    # NEW: force sw_batch_size for meaningful (sw_s x 9) heatmaps
    p.add_argument("--sw_batch_size", type=int, default=10)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--disable_dropout", action="store_true",
                   help="Disable dropout (not recommended for stable SSL overfit)")

    args = p.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if args.device == "cuda" else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Build VoCo args/config
    a = build_voco_args(
        roi_x=args.roi_x,
        roi_y=args.roi_y,
        roi_z=args.roi_z,
        device=args.device,
        feature_size=args.feature_size,
        amp=True,
        sw_batch_size=args.sw_batch_size,   # <- IMPORTANT CHANGE
    )

    xform = build_transforms(a, no_aug=args.no_aug)

    # Sample once: fixed targets (true overfit sanity check)
    out = xform({"image": args.image})
    img_cpu, crops_cpu, labels_cpu = unpack_voco_output(out)

    # Save initial grids
    save_voco_debug_vis(
        img=img_cpu,
        crops=crops_cpu,
        labels=labels_cpu,
        out_dir=args.out_dir,
        prefix=Path(args.image).stem + "_init",
        max_queries=8,
        slices_per_vol=6,
    )

    # Model
    model = build_model(a, device).train()

    if args.disable_dropout:
        n_drop = disable_dropout(model)
        print(f"[dropout] disabled dropout layers: {n_drop}")
    else:
        print("[dropout] keeping dropout ON (default)")

    # Load checkpoint (default backbone-only)
    if args.ckpt:
        stats = load_ckpt(model, args.ckpt, args.device, mode=args.load_mode)
        print(f"[ckpt] load_mode={args.load_mode} stats={stats}")
    else:
        print("[ckpt] no checkpoint provided; training from scratch")

    # Optimizer: weight_decay = 0 (your request), LR unchanged
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    img, crops, labels = to_device(img_cpu, crops_cpu, labels_cpu, device)

    # Metrics CSV
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["step", "loss", "top1_match", "mae", "mse"]
        )
        writer.writeheader()

        for step in range(1, args.steps + 1):
            model.train()
            opt.zero_grad(set_to_none=True)

            loss = forward_loss(model, img, crops, labels, use_amp=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_val = float(loss.item())

            if step % 50 == 0 or step == 1:
                print(f"step {step:05d}/{args.steps} | loss={loss_val:.6f}")

            # Periodic eval + save heatmaps + write metrics row
            if step % args.save_every == 0 or step == args.steps:
                logits, targets = compute_logits_targets(model, img, crops, labels)
                acc = top1_match(logits, targets)
                mae_v = mae(logits, targets)
                mse_v = mse(logits, targets)

                print(f"[eval] step {step:05d} top1={acc*100:.2f}% mae={mae_v:.6f} mse={mse_v:.6f}")

                writer.writerow({
                    "step": step,
                    "loss": loss_val,
                    "top1_match": acc,
                    "mae": mae_v,
                    "mse": mse_v,
                })
                f.flush()

                save_diff_bundle(logits, targets, out_dir=args.out_dir, prefix=f"step{step:05d}")

    # Save final checkpoint
    save_path = os.path.join(args.out_dir, "overfit_final.pt")
    save_ckpt_atomic(save_path, {"state_dict": model.state_dict(), "steps": args.steps})
    print("saved:", save_path)
    print("saved:", csv_path)
    print("done.")


if __name__ == "__main__":
    main()