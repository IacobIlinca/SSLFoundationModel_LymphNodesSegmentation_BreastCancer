import os
import argparse
import numpy as np
import nibabel as nib

# Optional PNG previews
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False

import torch
import torch.nn.functional as F


def hu_window_to_0_1(x: np.ndarray, hu_clip=(-1000, 1000)) -> np.ndarray:
    lo, hi = hu_clip
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo)
    return x.astype(np.float32)


def resize_to_square(slice01: np.ndarray, size: int) -> np.ndarray:
    """
    slice01: [H,W] float32 in [0,1]
    returns [size,size]
    """
    t = torch.from_numpy(slice01)[None, None]  # [1,1,H,W]
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t[0, 0].cpu().numpy().astype(np.float32)


def get_axial_slice(vol: np.ndarray, idx: int) -> np.ndarray:
    # axial assumed along last axis
    return vol[:, :, idx].astype(np.float32)


def choose_indices(n: int, num_slices: int) -> np.ndarray:
    if num_slices <= 0 or num_slices >= n:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, num_slices).astype(int)


def is_empty_slice(slice01: np.ndarray, min_nonzero_frac: float) -> bool:
    # crude: count pixels above small threshold
    return float((slice01 > 0.05).mean()) < float(min_nonzero_frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_dir", type=str, required=True,
                    help="Folder containing image.nii.gz (and masks, ignored here).")
    ap.add_argument("--ct_name", type=str, default="image.nii.gz",
                    help="CT filename inside case_dir.")
    ap.add_argument("--out_dir", type=str, default="ct_2d_slices",
                    help="Output folder for slices.")
    ap.add_argument("--plane", type=str, default="axial", choices=["axial"],
                    help="Currently exports axial slices (vol[:,:,idx]).")
    ap.add_argument("--hu_min", type=int, default=-1000)
    ap.add_argument("--hu_max", type=int, default=1000)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_slices", type=int, default=24,
                    help="Number of slices to export evenly spaced. Use 0 to export all.")
    ap.add_argument("--min_nonzero_frac", type=float, default=0.05,
                    help="Skip near-empty slices.")
    ap.add_argument("--save_png", action="store_true",
                    help="Also save PNG previews (requires matplotlib).")
    args = ap.parse_args()

    ct_path = os.path.join(args.case_dir, args.ct_name)
    if not os.path.exists(ct_path):
        raise FileNotFoundError(f"CT not found: {ct_path}")

    os.makedirs(args.out_dir, exist_ok=True)
    npy_dir = os.path.join(args.out_dir, "npy")
    os.makedirs(npy_dir, exist_ok=True)
    png_dir = os.path.join(args.out_dir, "png")
    if args.save_png:
        os.makedirs(png_dir, exist_ok=True)
        if not HAS_PLT:
            raise RuntimeError("matplotlib not available, but --save_png was set. Install matplotlib or remove --save_png.")

    vol = nib.load(ct_path).get_fdata().astype(np.float32)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D CT, got shape {vol.shape}")

    n_slices = vol.shape[2]
    idxs = choose_indices(n_slices, args.num_slices)

    manifest = []
    kept = 0

    for idx in idxs:
        sl_hu = get_axial_slice(vol, int(idx))
        sl01 = hu_window_to_0_1(sl_hu, (args.hu_min, args.hu_max))
        if is_empty_slice(sl01, args.min_nonzero_frac):
            continue

        sl01 = resize_to_square(sl01, args.img_size)  # [224,224]

        out_npy = os.path.join(npy_dir, f"slice_{int(idx):04d}.npy")
        np.save(out_npy, sl01)

        if args.save_png:
            out_png = os.path.join(png_dir, f"slice_{int(idx):04d}.png")
            plt.imsave(out_png, sl01, cmap="gray", vmin=0, vmax=1)

        manifest.append({
            "slice_index": int(idx),
            "npy_path": out_npy
        })
        kept += 1

    # Save manifest
    manifest_path = os.path.join(args.out_dir, "manifest.txt")
    with open(manifest_path, "w") as f:
        f.write(f"ct_path: {ct_path}\n")
        f.write(f"volume_shape: {vol.shape}\n")
        f.write(f"hu_clip: ({args.hu_min}, {args.hu_max})\n")
        f.write(f"img_size: {args.img_size}\n")
        f.write(f"requested_num_slices: {args.num_slices}\n")
        f.write(f"kept_slices: {kept}\n\n")
        for m in manifest:
            f.write(f"{m['slice_index']:04d}\t{m['npy_path']}\n")

    print(f"Done. Exported {kept} slices to: {args.out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
