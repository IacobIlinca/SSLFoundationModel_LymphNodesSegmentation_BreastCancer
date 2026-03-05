import os
import argparse
from pathlib import Path

import torch

from src.VocoLarge.training_ssl.pipeline import (
    build_transforms, build_model,
    load_ckpt,
    unpack_voco_output, to_device,
    compute_logits_targets,
    save_voco_debug_vis, save_diff_bundle, save_heatmap,
    best_crop_report,
)
from src.VocoLarge.training_ssl.pipeline.config import Config


def main():

    args = Config()

    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if args.device == "cuda" else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    xform = build_transforms(args, no_aug=args.no_aug)

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

    model = build_model(args, device).eval()
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