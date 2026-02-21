import os
import glob
import math
import argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from transformers import ViTImageProcessor


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


def load_medmae_model(medmae_repo_dir: str, ckpt_path: str, device: str):
    """
    Loads the MedMAE ViT-MAE model from the lambert-x/medical_mae repo and a .pth checkpoint.
    """
    import sys
    sys.path.insert(0, medmae_repo_dir)

    import models_mae  # from medical_mae repo

    model = models_mae.mae_vit_base_patch16()  # matches ViT-Base/16 checkpoints
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # official MAE-style checkpoints store weights under "model"
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    msg = model.load_state_dict(state, strict=False)
    print("Loaded MedMAE checkpoint.")
    print("  Missing keys:", len(getattr(msg, "missing_keys", [])))
    print("  Unexpected keys:", len(getattr(msg, "unexpected_keys", [])))

    model.to(device).eval()
    return model


@torch.no_grad()
def medmae_forward(model, pixel_values: torch.Tensor, mask_ratio: float):
    """
    medical_mae model forward returns (loss, pred, mask).
      - pred: (B, L, p*p*3)
      - mask: (B, L) 0 keep, 1 mask
    """
    loss, pred, mask = model(pixel_values, mask_ratio=mask_ratio)
    return float(loss.item()), pred, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice_dir", type=str, required=True,
                    help="Folder created by convert_3d_to_2d.py (contains npy/).")
    ap.add_argument("--out_dir", type=str, default="medmae_zero_shot_outputs")
    ap.add_argument("--medmae_repo_dir", type=str, required=True,
                    help="Path to cloned medical_mae repo (must contain models_mae.py).")
    ap.add_argument("--ckpt_path", type=str, required=True,
                    help="Path to MedMAE .pth checkpoint (e.g. vit-b_CXR_0.5M_mae.pth).")
    ap.add_argument("--processor_id", type=str, default="facebook/vit-mae-base",
                    help="Use SAME processor as ImageNet script for fair preprocessing.")
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

    # Keep preprocessing identical to ImageNet MAE script
    processor = ViTImageProcessor.from_pretrained(args.processor_id)

    # Load MedMAE model + weights
    model = load_medmae_model(args.medmae_repo_dir, args.ckpt_path, device=device)

    # MedMAE ViT-B/16 defaults
    patch_size = 16
    img_size = 224

    losses = []
    viz_saved = 0

    for pth in tqdm(paths, desc="Slices"):
        sl01 = np.load(pth).astype(np.float32)  # [224,224] in [0,1]

        # Same as your ImageNet script:
        u8 = (np.clip(sl01, 0, 1) * 255.0).astype(np.uint8)
        pil = Image.fromarray(u8, mode="L").convert("RGB")

        inputs = processor(images=pil, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)  # (1,3,224,224) normalized

        loss, pred_patches, mask = medmae_forward(model, pixel_values, args.mask_ratio)
        losses.append(loss)

        recon = unpatchify(pred_patches, patch_size=patch_size, img_size=img_size)

        # For visualization: min-max to [0,1] per image (same style as your current script)
        recon_vis = recon.clone()
        rmin = recon_vis.min()
        rmax = recon_vis.max()
        recon_vis = (recon_vis - rmin) / (rmax - rmin + 1e-6)
        recon_vis = recon_vis.clamp(0, 1)

        # Visualize target in [0,1] as before (from sl01)
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
        f.write(f"model: MedMAE ViT-Base/16 (medical_mae)\n")
        f.write(f"ckpt_path: {args.ckpt_path}\n")
        f.write(f"processor_id: {args.processor_id}\n")
        f.write(f"mask_ratio: {args.mask_ratio}\n")
        f.write(f"num_slices: {len(losses)}\n")
        f.write(f"mean_loss: {mean_l:.6f}\n")
        f.write(f"std_loss: {std_l:.6f}\n")

    print(f"Done. mean_loss={mean_l:.6f}, std_loss={std_l:.6f}")
    print(f"Saved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
