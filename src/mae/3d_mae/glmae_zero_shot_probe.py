import argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from Finetune.models_3dvit import vit_base_patch16_3d


def load_nii_as_hwd(path: str) -> np.ndarray:
    # nibabel gives (H, W, D) for typical NIfTI
    x = nib.load(path).get_fdata().astype(np.float32)
    return x


def ct_window_to_01(x: np.ndarray, lo=-600.0, hi=200.0) -> np.ndarray:
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-8)
    return x.astype(np.float32)


def center_crop_or_pad(vol: torch.Tensor, out_shape=(96, 96, 96)) -> torch.Tensor:
    # vol: (1,1,D,H,W)
    D0, H0, W0 = out_shape
    _, _, D, H, W = vol.shape

    pad_d = max(0, D0 - D)
    pad_h = max(0, H0 - H)
    pad_w = max(0, W0 - W)

    vol = F.pad(
        vol,
        (pad_w // 2, pad_w - pad_w // 2,
         pad_h // 2, pad_h - pad_h // 2,
         pad_d // 2, pad_d - pad_d // 2),
        mode="constant",
        value=0.0,
    )

    _, _, D, H, W = vol.shape
    ds = (D - D0) // 2
    hs = (H - H0) // 2
    ws = (W - W0) // 2
    return vol[:, :, ds:ds + D0, hs:hs + H0, ws:ws + W0]


def random_aug(x: torch.Tensor) -> torch.Tensor:
    # x: (1,1,D,H,W) in [0,1]
    if torch.rand(()) < 0.5:
        x = torch.flip(x, dims=[3])  # flip H
    if torch.rand(()) < 0.5:
        x = torch.flip(x, dims=[4])  # flip W
    x = (x + 0.01 * torch.randn_like(x)).clamp(0, 1)
    return x


def patch_energy_map(patch_features: torch.Tensor, grid=(6, 6, 6)) -> np.ndarray:
    # patch_features: (1, N, C) where N = 6*6*6 = 216 for 96 with patch16
    pf = patch_features[0]  # (N, C)
    en = torch.sqrt((pf ** 2).sum(dim=1) + 1e-8)  # (N,)
    en3 = en.reshape(*grid).detach().cpu().numpy()  # (Dg, Hg, Wg)
    return en3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nii", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_png", default="glmae_probe.png")
    ap.add_argument("--crop", type=int, nargs=3, default=[96, 96, 96])  # D H W
    ap.add_argument("--mask_ratio", type=float, default=0.75)  # just for visualization
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build encoder-only model exactly as repo README suggests (img_size=96, patch=16)
    model = vit_base_patch16_3d(
        img_size=(args.crop[0], args.crop[1], args.crop[2]),
        in_chans=1,
        encoder_only=True,
    ).to(device)
    model.eval()

    # Load pretrained weights (repo’s helper handles key prefixes + pos_embed interpolation)
    missing, unexpected = model.load_pretrained_checkpoint(
        args.ckpt,
        encoder_only=True,
        interpolate_pos_embed=True,
    )
    print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")

    # Load and preprocess CT
    hwd = ct_window_to_01(load_nii_as_hwd(args.nii))
    # Convert (H,W,D) -> torch (D,H,W)
    dhw = np.transpose(hwd, (2, 0, 1))
    x = torch.from_numpy(dhw)[None, None, ...]  # (1,1,D,H,W)
    x = center_crop_or_pad(x, out_shape=tuple(args.crop)).to(device)

    # Forward (features)
    with torch.no_grad():
        feats1 = model(x, return_features=True)
        feats2 = model(random_aug(x), return_features=True)

    patch1 = feats1["patch_features"]  # (1,N,C)
    patch2 = feats2["patch_features"]  # (1,N,C)

    # Cosine similarity per patch between original vs augmented (augmentation invariance)
    p1 = F.normalize(patch1[0], dim=1)
    p2 = F.normalize(patch2[0], dim=1)
    cos = (p1 * p2).sum(dim=1).detach().cpu().numpy()  # (N,)

    # Patch self-similarity matrix (within one forward)
    sim = (p1 @ p1.T).detach().cpu().numpy()  # (N,N)

    # Patch energy map
    # For 96^3 with patch16 -> grid 6x6x6
    en3 = patch_energy_map(patch1, grid=(6, 6, 6))
    mid_g = en3.shape[0] // 2

    # Also show the CT mid-slice
    ct = x[0, 0].detach().cpu().numpy()
    mid_z = ct.shape[0] // 2

    # Visualize a “mask” (for understanding only). MAE masks are random.
    N = patch1.shape[1]
    mask = np.zeros(N, dtype=np.float32)
    k = int(N * args.mask_ratio)
    mask[np.random.choice(N, size=k, replace=False)] = 1.0
    mask3 = mask.reshape(6, 6, 6)
    mask_mid = mask3[mid_g]

    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    ax[0, 0].imshow(ct[mid_z], cmap="gray")
    ax[0, 0].set_title("CT mid slice (cropped)")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(en3[mid_g], cmap="gray")
    ax[0, 1].set_title("Patch feature energy (mid grid slice)")
    ax[0, 1].axis("off")

    ax[0, 2].imshow(mask_mid, cmap="gray")
    ax[0, 2].set_title(f"Random mask (ratio={args.mask_ratio})")
    ax[0, 2].axis("off")

    ax[1, 0].imshow(sim, cmap="gray")
    ax[1, 0].set_title("Patch self-similarity (cos)")
    ax[1, 0].axis("off")

    ax[1, 1].hist(cos, bins=40)
    ax[1, 1].set_title("Aug invariance (cos per patch)")

    ax[1, 2].plot(np.sort(cos))
    ax[1, 2].set_title("Sorted invariance cos")

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print("Saved:", args.out_png)

    print(f"Invariance cos: mean={cos.mean():.3f}, median={np.median(cos):.3f}, p10={np.percentile(cos,10):.3f}, p90={np.percentile(cos,90):.3f}")


if __name__ == "__main__":
    main()