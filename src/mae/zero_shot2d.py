import os
import glob
import math
import argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from transformers import ViTMAEForPreTraining, ViTImageProcessor


# ---------- patch utils ----------
def unpatchify(patches: torch.Tensor, patch_size: int, img_size: int) -> torch.Tensor:
    """
    patches: (B, L, patch_size*patch_size*3)
    returns: (B, 3, img_size, img_size)
    """
    B, L, D = patches.shape
    h = w = int(math.sqrt(L))
    assert h * w == L, "L must be a square number of patches"

    p = patch_size
    patches = patches.reshape(B, h, w, p, p, 3)
    patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()  # (B,3,h,p,w,p)
    img = patches.reshape(B, 3, h * p, w * p)
    # should already be img_size x img_size
    return img


def save_triplet_png(out_path: str, target: torch.Tensor, recon: torch.Tensor, mask: torch.Tensor,
                     patch_size: int, title: str):
    """
    target/recon: (1,3,224,224), mask: (1,L) where 1=masked
    """
    target_np = target[0, 0].detach().cpu().numpy()
    recon_np  = recon[0, 0].detach().cpu().numpy()

    L = mask.shape[1]
    h = w = int(math.sqrt(L))
    m = mask[0].reshape(h, w).detach().cpu().numpy()

    masked = target_np.copy()
    for i in range(h):
        for j in range(w):
            if m[i, j] > 0.5:
                y0, y1 = i * patch_size, (i + 1) * patch_size
                x0, x1 = j * patch_size, (j + 1) * patch_size
                masked[y0:y1, x0:x1] = 0.0

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(masked, cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("Masked input")
    ax[1].imshow(recon_np, cmap="gray", vmin=0, vmax=1)
    ax[1].set_title("Reconstruction")
    ax[2].imshow(target_np, cmap="gray", vmin=0, vmax=1)
    ax[2].set_title("Target")
    for a in ax:
        a.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice_dir", type=str, required=True,
                    help="Folder created by convert_3d_to_2d.py (contains npy/).")
    ap.add_argument("--out_dir", type=str, default="mae_zero_shot_outputs_hf")
    ap.add_argument("--model_id", type=str, default="facebook/vit-mae-base",
                    help="Hugging Face model id for ViT-MAE.")
    ap.add_argument("--mask_ratio", type=float, default=0.75)
    ap.add_argument("--max_slices", type=int, default=0)
    ap.add_argument("--save_viz", type=int, default=20)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    npy_dir = os.path.join(args.slice_dir, "npy")
    paths = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
    if not paths:
        raise FileNotFoundError(f"No .npy slices found in {npy_dir}")
    if args.max_slices and args.max_slices > 0:
        paths = paths[:args.max_slices]

    # Load MAE + processor
    processor = ViTImageProcessor.from_pretrained(args.model_id)
    model = ViTMAEForPreTraining.from_pretrained(args.model_id).to(device).eval()

    # Patch size from config
    patch_size = model.config.patch_size
    img_size = model.config.image_size  # usually 224

    losses = []
    viz_saved = 0

    for pth in tqdm(paths, desc="Slices"):
        sl01 = np.load(pth).astype(np.float32)  # [224,224] in [0,1]
        # Convert to PIL RGB expected by processor
        # Keep intensity: map [0,1] -> [0,255] uint8
        u8 = (np.clip(sl01, 0, 1) * 255.0).astype(np.uint8)
        pil = Image.fromarray(u8, mode="L").convert("RGB")

        inputs = processor(images=pil, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)  # (1,3,224,224) normalized

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, mask_ratio=args.mask_ratio)

        # outputs.loss is MAE reconstruction loss (on masked patches)
        loss = float(outputs.loss.item())
        losses.append(loss)

        # outputs.logits are predicted patches in pixel space after internal normalization
        # We'll build a reconstruction image from them.
        # Note: this reconstruction is in the model's internal "pixel-space"; for qualitative checks it’s fine.
        pred_patches = outputs.logits  # (1, L, p*p*3)
        mask = outputs.mask  # (1, L) 1=masked

        recon = unpatchify(pred_patches, patch_size=patch_size, img_size=img_size)
        # For visualization, bring recon into [0,1] roughly by min-max per image
        recon_vis = recon.clone()
        rmin = recon_vis.min()
        rmax = recon_vis.max()
        recon_vis = (recon_vis - rmin) / (rmax - rmin + 1e-6)
        recon_vis = recon_vis.clamp(0, 1)

        # Also visualize the *input* in [0,1] (from your slice)
        target = torch.from_numpy(sl01)[None, None].repeat(1, 3, 1, 1).to(device)

        if viz_saved < args.save_viz:
            base = os.path.splitext(os.path.basename(pth))[0]
            png_path = os.path.join(args.out_dir, f"{base}_loss{loss:.4f}.png")
            save_triplet_png(
                png_path,
                target=target.detach().cpu(),
                recon=recon_vis.detach().cpu(),
                mask=mask.detach().cpu(),
                patch_size=patch_size,
                title=f"{base} | loss={loss:.4f} | mask={args.mask_ratio:.2f}"
            )
            viz_saved += 1

    losses_np = np.array(losses, dtype=np.float32)
    mean_l = float(losses_np.mean()) if len(losses_np) else float("nan")
    std_l = float(losses_np.std()) if len(losses_np) else float("nan")

    with open(os.path.join(args.out_dir, "SUMMARY.txt"), "w") as f:
        f.write(f"model_id: {args.model_id}\n")
        f.write(f"mask_ratio: {args.mask_ratio}\n")
        f.write(f"num_slices: {len(losses)}\n")
        f.write(f"mean_loss: {mean_l:.6f}\n")
        f.write(f"std_loss: {std_l:.6f}\n")

    print(f"Done. mean_loss={mean_l:.6f}, std_loss={std_l:.6f}")
    print(f"Saved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
