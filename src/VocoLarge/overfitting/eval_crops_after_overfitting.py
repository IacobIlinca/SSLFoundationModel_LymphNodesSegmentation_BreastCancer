import os
import argparse
from pathlib import Path

import torch

from src.VocoLarge.pipeline import (
    build_voco_args, build_transforms, build_model,
    load_ckpt,
    unpack_voco_output, to_device,
    compute_logits_targets,
    save_voco_debug_vis, save_diff_bundle, save_heatmap,
    best_crop_report,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="path to image.nii.gz")
    p.add_argument("--out_dir", required=True, help="directory to save plots")
    p.add_argument("--ckpt", required=True, help="path to .pt checkpoint (overfit_final.pt etc)")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--roi_x", type=int, default=96)
    p.add_argument("--roi_y", type=int, default=96)
    p.add_argument("--roi_z", type=int, default=64)
    p.add_argument("--feature_size", type=int, default=48)
    p.add_argument("--no_aug", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    # NEW: load only backbone or full model
    p.add_argument("--load_mode", default="full", choices=["backbone", "full"])
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if args.device == "cuda" else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    a = build_voco_args(
        roi_x=args.roi_x, roi_y=args.roi_y, roi_z=args.roi_z,
        device=args.device, feature_size=args.feature_size,
        amp=False, sw_batch_size=10,  # keep 10 queries like your old eval
    )

    xform = build_transforms(a, no_aug=args.no_aug)

    out = xform({"image": args.image})
    img_cpu, crops_cpu, labels_cpu = unpack_voco_output(out)

    # Save the same “best crop per query” grids (uses TARGETS)
    save_voco_debug_vis(
        img=img_cpu,
        crops=crops_cpu,
        labels=labels_cpu,
        out_dir=args.out_dir,
        prefix=Path(args.image).parent.name,
        max_queries=8,
        slices_per_vol=6,
    )

    model = build_model(a, device).eval()
    stats = load_ckpt(model, args.ckpt, args.device, mode=args.load_mode)
    print(f"[ckpt] load_mode={args.load_mode} stats={stats}")

    img, crops, labels = to_device(img_cpu, crops_cpu, labels_cpu, device)

    with torch.no_grad():
        logits, targets = compute_logits_targets(model, img, crops, labels)

    # Save heatmaps (targets/logits/diff/absdiff)
    save_diff_bundle(logits, targets, out_dir=args.out_dir, prefix="eval")

    # Also save standalone targets/logits if you want explicit names
    save_heatmap(targets.numpy(), "Targets (labels)", os.path.join(args.out_dir, "targets_heatmap.png"))
    save_heatmap(logits.numpy(), "Predictions (logits)", os.path.join(args.out_dir, "logits_heatmap.png"))

    rep = best_crop_report(logits, targets)
    print(f"top1 match: {rep['top1_match']*100:.2f}%")
    print("best crop per query (pred):", rep["pred_indices"].tolist())
    print("best crop per query (tgt): ", rep["tgt_indices"].tolist())


if __name__ == "__main__":
    main()