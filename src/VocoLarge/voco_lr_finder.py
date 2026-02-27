# src/VocoLarge/lr_find_voco.py
import os
import argparse
from pathlib import Path
import math

import torch
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Compose
from monai.data.meta_tensor import MetaTensor

import matplotlib.pyplot as plt
from tqdm import tqdm

from src.VocoLarge.third_party_voco_large.models.voco_head import VoCoHead
from src.VocoLarge.third_party_voco_large.utils import voco_trans, data_trans
from src.VocoLarge.voco_eval_one_volume import load_ckpt


def build_args(roi_x, roi_y, roi_z, device):
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

    a.space_x = 1.5
    a.space_y = 1.5
    a.space_z = 1.5

    a.local_rank = 0
    a.sw_batch_size = 1
    a.amp = True
    a.device = device
    return a


class NiftiListDataset(Dataset):
    def __init__(self, image_paths, xform):
        self.image_paths = image_paths
        self.xform = xform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.xform({"image": self.image_paths[idx]})


def find_case_images(root_dir: str):
    root = Path(root_dir)
    if not root.exists():
        raise RuntimeError(f"data_dir not found: {root_dir}")

    case_dirs = [p for p in root.iterdir() if p.is_dir()]
    case_dirs.sort()

    image_paths = []
    for c in case_dirs:
        niftis = [p for p in (list(c.rglob("*.nii")) + list(c.rglob("*.nii.gz")))
                  if "mask" not in p.name.lower()]
        if len(niftis) == 0:
            print(f"[warn] no NIfTI found under case: {c}")
            continue

        preferred = [p for p in niftis if p.name.lower() in ("image.nii.gz", "image.nii", "img.nii.gz", "img.nii")]
        chosen = preferred[0] if len(preferred) else niftis[0]
        image_paths.append(str(chosen))

    return image_paths


def unpack_voco_transform_output(out):
    if isinstance(out, (list, tuple)):
        img_obj, crop_obj, lab_obj = out
    elif isinstance(out, dict):
        img_obj = out.get("image", None)
        crop_obj = out.get("crops", None)
        lab_obj = out.get("labels", None)
    else:
        raise RuntimeError(f"Unexpected output type: {type(out)}")

    if img_obj is None or crop_obj is None or lab_obj is None:
        raise RuntimeError("Transform did not return expected outputs.")

    img = torch.stack([d["image"] for d in img_obj], dim=0).squeeze(1)     # (sw_s,1,D,H,W)
    crops = torch.stack([d["image"] for d in lab_obj], dim=0).squeeze(1)   # (9,  1,D,H,W)
    labels = torch.as_tensor(crop_obj, dtype=torch.float32)                # (1,sw_s,9) usually
    return img, crops, labels


def save_lr_plot(lrs, losses, out_path):
    plt.figure(figsize=(7, 4))
    plt.plot(lrs, losses)
    plt.xscale("log")
    plt.xlabel("Learning rate (log)")
    plt.ylabel("Smoothed loss (EMA)")
    plt.title("VoCo LR range test")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def lr_range_test(model, dl, device, out_dir, lr_min=1e-7, lr_max=5e-4, steps=200):
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=lr_min, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    mult = (lr_max / lr_min) ** (1.0 / max(1, steps - 1))

    lrs, losses = [], []
    ema = None
    beta = 0.98
    best = float("inf")

    it = iter(dl)
    pbar = tqdm(range(steps), desc="LR finder", leave=True)
    for step in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)

        img, crops, labels = unpack_voco_transform_output(batch)
        img = MetaTensor(img).to(device)
        crops = MetaTensor(crops).to(device)
        labels = labels.to(device)

        lr = lr_min * (mult ** step)
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            loss = model(img, crops, labels)

        if not torch.isfinite(loss):
            print("[lr_find] loss became non-finite, stopping.")
            break

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        loss_val = float(loss.item())
        ema = loss_val if ema is None else (beta * ema + (1 - beta) * loss_val)
        ema_corr = ema / (1 - beta ** (step + 1))

        lrs.append(lr)
        losses.append(ema_corr)

        best = min(best, ema_corr)
        pbar.set_postfix({"lr": f"{lr:.2e}", "loss": f"{ema_corr:.4f}"})

        if step > 10 and ema_corr > 5 * best:
            print("[lr_find] loss diverged, early stopping.")
            break

    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, "lr_find.png")
    save_lr_plot(lrs, losses, plot_path)

    best_i = int(min(range(len(losses)), key=lambda i: losses[i]))
    best_lr = lrs[best_i]
    suggest = best_lr / 10.0  # common heuristic

    txt_path = os.path.join(out_dir, "lr_find.txt")
    with open(txt_path, "w") as f:
        f.write(f"best_lr_raw={best_lr:.6e}\n")
        f.write(f"suggest_lr={suggest:.6e}\n")
        f.write(f"steps_ran={len(lrs)}\n")

    print(f"[lr_find] saved: {plot_path}")
    print(f"[lr_find] saved: {txt_path}")
    print(f"[lr_find] best_lr_raw={best_lr:.3e} suggest_lr={suggest:.3e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", default="runs_lr_find")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--ckpt", default=None, help="optional starting ckpt")
    p.add_argument("--roi_x", type=int, default=96)
    p.add_argument("--roi_y", type=int, default=96)
    p.add_argument("--roi_z", type=int, default=64)
    p.add_argument("--no_aug", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr_min", type=float, default=1e-7)
    p.add_argument("--lr_max", type=float, default=5e-4)
    p.add_argument("--lr_steps", type=int, default=200)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    a = build_args(args.roi_x, args.roi_y, args.roi_z, args.device)

    trans_list = data_trans.get_chest_trans(a)
    if args.no_aug:
        for i, t in enumerate(trans_list):
            if t.__class__.__name__ == "VoCoAugmentation":
                trans_list[i] = voco_trans.VoCoAugmentation(a, aug=False)
                break
    xform = Compose(trans_list)

    imgs = find_case_images(args.data_dir)
    if len(imgs) == 0:
        raise RuntimeError(f"No NIfTI found under {args.data_dir}")
    print(f"Found {len(imgs)} case images")

    ds = NiftiListDataset(imgs, xform)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))

    model = VoCoHead(a).to(device).train()
    if args.ckpt:
        print("Loading checkpoint:", args.ckpt)
        load_ckpt(model, args.ckpt, args.device)

    lr_range_test(
        model=model,
        dl=dl,
        device=device,
        out_dir=args.out_dir,
        lr_min=args.lr_min,
        lr_max=args.lr_max,
        steps=args.lr_steps,
    )


if __name__ == "__main__":
    main()