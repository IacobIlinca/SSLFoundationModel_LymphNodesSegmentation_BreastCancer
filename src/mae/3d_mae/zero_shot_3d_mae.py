import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import matplotlib.pyplot as plt

# import your model builder
# from your_pkg.models.mae_3d import mae_vit_base  # adjust
# (mae_vit_base should create MaskedAutoencoderViT_3D with correct img_size/patch_size/in_chans)

def load_nii(path: str) -> np.ndarray:
    img = nib.load(path)
    arr = img.get_fdata().astype(np.float32)  # (H,W,D) typically
    return arr

def normalize_ct(x: np.ndarray, clip=(-1000, 1000)) -> np.ndarray:
    x = np.clip(x, clip[0], clip[1])
    x = (x - clip[0]) / (clip[1] - clip[0])  # to [0,1]
    return x

def center_crop_or_pad_3d(vol: torch.Tensor, out_hw_d: tuple) -> torch.Tensor:
    """
    vol: (1, 1, H, W, D)
    out_hw_d: (H0, W0, D0)
    """
    _, _, H, W, D = vol.shape
    H0, W0, D0 = out_hw_d

    # pad if needed
    pad_h = max(0, H0 - H)
    pad_w = max(0, W0 - W)
    pad_d = max(0, D0 - D)

    # F.pad uses (D_left, D_right, W_left, W_right, H_left, H_right)
    vol = F.pad(
        vol,
        (pad_d // 2, pad_d - pad_d // 2,
         pad_w // 2, pad_w - pad_w // 2,
         pad_h // 2, pad_h - pad_h // 2),
        mode="constant",
        value=0.0
    )

    # crop if needed (center crop)
    _, _, H, W, D = vol.shape
    hs = (H - H0) // 2
    ws = (W - W0) // 2
    ds = (D - D0) // 2
    vol = vol[:, :, hs:hs+H0, ws:ws+W0, ds:ds+D0]
    return vol

def psnr(mse: float, max_val: float = 1.0) -> float:
    if mse <= 0:
        return float("inf")
    return 20 * np.log10(max_val) - 10 * np.log10(mse)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nii", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--img_size", type=int, nargs=3, required=True, help="H W D used by MAE training")
    ap.add_argument("--patch_size", type=int, nargs=3, required=True, help="pH pW pD")
    ap.add_argument("--in_chans", type=int, default=1)
    ap.add_argument("--mask_ratio", type=float, default=0.75)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_png", type=str, default="mae_recon_debug.png")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- build model (adjust to your factory)
    from your_pkg.models.mae_3d import mae_vit_base  # CHANGE THIS IMPORT
    model = mae_vit_base(
        img_size=tuple(args.img_size),
        patch_size=tuple(args.patch_size),
        in_chans=args.in_chans,
        norm_pix_loss=False,
    ).to(device)

    # ---- load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)  # common patterns: {"model": state_dict} or directly state_dict
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")

    model.eval()

    # ---- load + preprocess volume to (1, C, H, W, D)
    vol = load_nii(args.nii)                 # (H,W,D)
    vol = normalize_ct(vol)                  # [0,1]
    vol_t = torch.from_numpy(vol)[None, None, ...]  # (1,1,H,W,D)
    vol_t = center_crop_or_pad_3d(vol_t, tuple(args.img_size))
    vol_t = vol_t.to(device)

    # ---- MAE forward (masked reconstruction)
    with torch.no_grad():
        loss, pred, mask = model(vol_t, mask_ratio=args.mask_ratio)
        # pred: (N, L, p^3 * C), mask: (N, L) 0 keep, 1 remove
        recon = model.unpatchify(pred)  # (1, C, H, W, D)

        # build "visible+recon" like MAE papers:
        # - visible patches come from original
        # - masked patches come from recon
        target_patches = model.patchify(vol_t)  # (1, L, p^3*C)

        # mask: 0 keep, 1 masked. expand to patch vector dim
        mask_vec = mask.unsqueeze(-1).type_as(target_patches)  # (1,L,1)

        mixed_patches = target_patches * (1.0 - mask_vec) + pred * mask_vec
        mixed = model.unpatchify(mixed_patches)  # (1,C,H,W,D)

    # ---- metrics
    mse_recon = F.mse_loss(recon, vol_t).item()
    mse_mixed = F.mse_loss(mixed, vol_t).item()
    print(f"loss(masked-only)={loss.item():.6f}")
    print(f"MSE(recon vs gt)={mse_recon:.6f}, PSNR={psnr(mse_recon):.2f} dB")
    print(f"MSE(mixed vs gt)={mse_mixed:.6f}, PSNR={psnr(mse_mixed):.2f} dB")

    # ---- visualize a few slices (axial)
    vol_np   = vol_t[0, 0].detach().cpu().numpy()
    recon_np = recon[0, 0].detach().cpu().numpy()
    mixed_np = mixed[0, 0].detach().cpu().numpy()
    err_np   = np.abs(mixed_np - vol_np)

    D = vol_np.shape[-1]
    mids = [D//4, D//2, (3*D)//4]

    fig, axes = plt.subplots(len(mids), 4, figsize=(12, 3*len(mids)))
    if len(mids) == 1:
        axes = axes[None, :]

    for r, z in enumerate(mids):
        axes[r, 0].imshow(vol_np[:, :, z], cmap="gray");  axes[r, 0].set_title(f"GT z={z}");    axes[r, 0].axis("off")
        axes[r, 1].imshow(recon_np[:, :, z], cmap="gray");axes[r, 1].set_title("Recon");       axes[r, 1].axis("off")
        axes[r, 2].imshow(mixed_np[:, :, z], cmap="gray");axes[r, 2].set_title("Recon+Visible");axes[r, 2].axis("off")
        axes[r, 3].imshow(err_np[:, :, z], cmap="gray");  axes[r, 3].set_title("|Error|");     axes[r, 3].axis("off")

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print(f"Saved: {args.out_png}")

if __name__ == "__main__":
    main()