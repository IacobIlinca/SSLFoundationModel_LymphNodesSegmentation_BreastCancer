import os
import csv
import argparse
from pathlib import Path

import torch
from torch.optim import AdamW

from src.VocoLarge.training_ssl.pipeline import (
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
    disable_dropout, find_case_images, NiftiListDataset, build_dataloader,
)
from src.VocoLarge.training_ssl.pipeline.config import Config
from src.VocoLarge.training_ssl.pipeline.freeze import freeze_encoder, report_trainable_by_module
from src.VocoLarge.training_ssl.pipeline.training import train_one_batch


def main():
    args = Config()

    device = torch.device("cuda" if args.device == "cuda" else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    xform = build_transforms(args, no_aug=args.no_aug)

    # Sample once: fixed targets (true overfit sanity check)
    if args.overfit_experimnet:
        image_paths = [args.overfit_image_path]
    else:
        image_paths = find_case_images(args.data_dir)
        if len(image_paths) == 0:
            raise RuntimeError(f"No NIfTI images found under: {args.data_dir}")

    ds = NiftiListDataset(image_paths, xform)
    loader = build_dataloader(
        dataset=ds,
        device_type=args.device,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
    )

    # SAMPLE for vis
    img_cpu_vis, crops_cpu_vis, labels_cpu_vis = unpack_voco_output(xform({"image": image_paths[0]}))
    # Save initial grids
    save_voco_debug_vis(
        img=img_cpu_vis,
        crops=crops_cpu_vis,
        labels=labels_cpu_vis,
        out_dir=args.out_dir,
        prefix=Path(args.overfit_image_path).stem + "_init",
        max_queries=args.max_queries_vis,
        slices_per_vol=args.slices_per_vol_vis,
    )
    img_vis, crops_vis, labels_vis = to_device(img_cpu_vis, crops_cpu_vis, labels_cpu_vis, device)

    # Model
    model = build_model(args, device).train()

    # if args.disable_dropout:
    #     n_drop = disable_dropout(model)
    #     print(f"[dropout] disabled dropout layers: {n_drop}")
    # else:
    #     print("[dropout] keeping dropout ON (default)")

    # Load checkpoint (default backbone-only)
    if args.voco_ckpt_path:
        stats = load_ckpt(model, args.voco_ckpt_path, args.device, mode=args.load_mode)
        print(f"[ckpt] load_mode={args.load_mode} stats={stats}")
    else:
        print("[ckpt] no checkpoint provided; training from scratch")

    # Freeze encoder
    freeze_encoder(model, args)
    report_trainable_by_module(model)

    # Optimizer: weight_decay = 0 (your request), LR unchanged
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Metrics CSV
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["step", "loss", "top1_match", "mae", "mse"]
        )
        writer.writeheader()

        for step in range(1, args.steps + 1):
            loss_val = 0
            batches = 0
            for batch in loader:
                loss = train_one_batch(model, opt, scaler, batch, device)

                loss_val += float(loss.item())
                batches += 1

            loss_val /= batches
            if step % 10 == 0 or step == 1:
                print(f"step {step:05d}/{args.steps} | loss={loss_val:.6f}")

            # Periodic eval + save heatmaps + write metrics row
            if step % args.save_every == 0 or step == args.steps:
                logits, targets = compute_logits_targets(model, img_vis, crops_vis, labels_vis)
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
    # save_path = os.path.join(args.out_dir, "overfit_final.pt")
    # save_ckpt_atomic(save_path, {"state_dict": model.state_dict(), "steps": args.steps})
    # print("saved:", save_path)
    # print("saved:", csv_path)
    # print("done.")


if __name__ == "__main__":
    main()