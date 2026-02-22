import argparse

import torch
from monai.transforms import Compose
from monai.data.meta_tensor import MetaTensor
from pathlib import Path
from src.VocoLarge.voco_vis import save_voco_debug_vis, get_voco_logits, save_heatmap
from src.VocoLarge.third_party_voco_large.utils import data_trans, voco_trans
from src.VocoLarge.third_party_voco_large.models.voco_head import VoCoHead  # models package typically has no side effects


def build_args(roi_x, roi_y, roi_z, device):
    # mimic argparse args object the repo expects
    class A: pass
    a = A()
    a.in_channels = 1
    a.feature_size = 48
    a.dropout_path_rate = 0.0
    a.use_checkpoint = True
    a.spatial_dims = 3

    a.roi_x = roi_x
    a.roi_y = roi_y
    a.roi_z = roi_z

    # used in their transforms / augment
    a.space_x = 1.5
    a.space_y = 1.5
    a.space_z = 1.5

    # important: many repo funcs assume this exists even on CPU
    a.local_rank = 0

    # used by training scripts; keep small for CPU
    a.sw_batch_size = 10

    # amp is irrelevant here
    a.amp = False

    # convenience
    a.device = device
    return a


def load_ckpt(model, ckpt_path: str, device):
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}

    # If ckpt keys look like "swinViT.xxx" / "encoder1.xxx", they belong to Swin (backbone)
    msd = model.backbone.state_dict()

    matched = {}
    for k, v in sd.items():
        if k in msd and msd[k].shape == v.shape:
            matched[k] = v

    msd.update(matched)
    model.backbone.load_state_dict(msd, strict=True)
    print(f"[ckpt] backbone matched: {len(matched)}/{len(msd)} ({100 * len(matched) / len(msd):.1f}%)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="path to image.nii.gz")
    p.add_argument("--ckpt", default=None, help="VoCo_*_SSL_head.pt (optional)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--roi_x", type=int, default=192)
    p.add_argument("--roi_y", type=int, default=192)
    p.add_argument("--roi_z", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use_chest_trans", action="store_true",
                   help="use chest window/spacing; otherwise uses headneck-style window")
    p.add_argument("--no_aug", action="store_true",
                   help="turn off VoCoAugmentation's heavy aug (safer on CPU)")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    # build args object expected by VoCoHead + transforms
    a = build_args(args.roi_x, args.roi_y, args.roi_z, args.device)

    # choose a transform recipe from data_trans.py, but swap VoCoAugmentation args
    if args.use_chest_trans:
        trans_list = data_trans.get_chest_trans(a)
    else:
        trans_list = data_trans.get_headneck_trans(a)

    # Replace the last transform (VoCoAugmentation(args, aug=True)) with aug=False if requested
    # This avoids ops.patch_rand_drop which is CUDA-coded in this repo.
    if args.no_aug:
        for i, t in enumerate(trans_list):
            if t.__class__.__name__ == "VoCoAugmentation":
                trans_list[i] = voco_trans.VoCoAugmentation(a, aug=False)
                break

    xform = Compose(trans_list)

    # Apply to a MONAI-style dict. LoadImaged expects a filepath.
    out = xform({"image": args.image})

    # --- Handle VoCoAugmentation outputs properly ---
    if isinstance(out, (list, tuple)):
        if len(out) != 3:
            raise RuntimeError(f"Expected 3 outputs (img, crops, labels), got {len(out)}")
        img_obj, crop_obj, lab_obj = out

    elif isinstance(out, dict):
        # fallback if dict is returned (older MONAI behavior)
        img_obj = out.get("image", None)
        crop_obj = out.get("crops", None)
        lab_obj = out.get("labels", None)

    else:
        raise RuntimeError(f"Unexpected output type from transform: {type(out)}")


    if img_obj is None or crop_obj is None or lab_obj is None:
        raise RuntimeError(
            "Could not find expected outputs from transform. "
            "Please paste the printed keys and we’ll map them correctly."
        )


    # img: from img_obj, queries crop
    img = torch.stack([d["image"] for d in img_obj], dim=0)  # (sw_s, 1, 64, 64, 64)

    # crops/bases: from lab_obj, base crops
    crops = torch.stack([d["image"] for d in lab_obj], dim=0)  # (9, 1, 64, 64, 64)

    # labels/targets
    labels = torch.as_tensor(crop_obj, dtype=torch.float32).unsqueeze(0)  # (1, sw_s, 9)

    print("img", img.shape, "crops", crops.shape, "labels", labels.shape)
    save_voco_debug_vis(
        img=img,
        crops=crops,
        labels=labels,
        out_dir="debug_vis",
        prefix=Path(args.image).parent.name,  # e.g. case id folder
        max_queries=8,
        slices_per_vol=6,
    )

    # Make them MetaTensor because VoCoHead.forward expects .as_tensor()
    img = MetaTensor(img).to(args.device)
    crops = MetaTensor(crops).to(args.device)
    labels = labels.to(args.device)

    # build model
    model = VoCoHead(a).eval()
    if args.ckpt:
        print("Using checkpoint")
        load_ckpt(model, args.ckpt, args.device)
    else:
        print("Not using checkpoint")

    model = model.to(args.device)

    with torch.no_grad():
        loss = model(img, crops, labels)

        # --- NEW: logits vs targets visualizations ---
        logits, targets = get_voco_logits(model, img, crops, labels)

        save_heatmap(targets.numpy(), "Targets (labels): query vs 9 crops", "debug_vis/targets_heatmap.png")
        save_heatmap(logits.numpy(), "Predictions (logits): query vs 9 crops", "debug_vis/logits_heatmap.png")
        save_heatmap((logits - targets).numpy(), "Pred - Target", "debug_vis/diff_heatmap.png")

        # simple top-1 match metric
        top1_pred = logits.argmax(dim=1)
        top1_tgt = targets.argmax(dim=1)
        acc = (top1_pred == top1_tgt).float().mean().item()
        print("top1_prd", top1_pred, "\ntop1_tgt", top1_tgt)
        print(f"top1 match (batch item 0): {acc * 100:.2f}%")

    print(f"loss: {float(loss.item()):.6f}")


if __name__ == "__main__":
    main()
