import os
import json
import glob
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict

import numpy as np
import nibabel as nib


@dataclass
class ImageAudit:
    path: str
    shape: Tuple[int, ...]
    dtype: str
    min: float
    max: float
    p0_5: float
    p1: float
    p50: float
    p99: float
    p99_5: float
    spacing_xyz: Tuple[float, float, float]
    orientation: str  # e.g. RAS / LPS
    folder_contents: List[str]
    side: Optional[str] = None
    side_source_mask: Optional[str] = None

def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _get_spacing(img: nib.Nifti1Image) -> Tuple[float, float, float]:
    # nibabel: zooms are ordered by data axes (x,y,z) for standard NIfTI
    zooms = img.header.get_zooms()
    if len(zooms) >= 3:
        return (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    # fallback
    return (float("nan"), float("nan"), float("nan"))


def _get_orientation(img: nib.Nifti1Image) -> str:
    # Converts affine to axis codes like ('R','A','S')
    try:
        ax = nib.aff2axcodes(img.affine)
        return "".join(ax)
    except Exception:
        return "UNK"

def _list_case_folder_contents(image_path: str) -> List[str]:
    """
    Returns sorted filenames in the same folder as the image.
    Keeps only files, not subdirectories.
    """
    case_dir = os.path.dirname(image_path)

    try:
        files = []
        for fname in os.listdir(case_dir):
            fpath = os.path.join(case_dir, fname)
            if os.path.isfile(fpath):
                files.append(fname)
        return sorted(files, key=lambda x: x.lower())
    except Exception:
        return []

def _find_side_mask_for_image(image_path: str) -> Optional[str]:
    """
    Finds a mask file in the same case folder whose filename contains
    'Level-II' or 'L2' or 'Level_2' or 'mamma' or 'boost' or 'tangentiel' or 'cicatrice'(case-insensitive).
    Returns the first match found, or None if no match exists.
    """
    case_dir = os.path.dirname(image_path)

    candidates = []
    for fname in os.listdir(case_dir):
        fpath = os.path.join(case_dir, fname)
        if not os.path.isfile(fpath):
            continue

        lower = fname.lower()

        # only nifti-like files
        if not (lower.endswith(".nii") or lower.endswith(".nii.gz")):
            continue

        if "level-ii" in lower or "l2" in lower or "level_2" in lower or "mamma" in lower or "boost" in lower or "tangentiel" in lower or "cicatrice" in lower:
            candidates.append(fpath)

    if len(candidates) == 0:
        return None

    # Prefer "Level-II" over "L2" if both exist
    candidates.sort(key=lambda p: (("level-ii" not in os.path.basename(p).lower()), os.path.basename(p).lower()))
    return candidates[0]


def infer_side_from_mask(mask_path: str, min_voxels: int = 10) -> str:
    """
    Infer left/right from the mask location.

    Assumes LPS orientation:
      axis 0 = Left -> Right

    Returns one of:
      - 'left'
      - 'right'
      - 'midline'
      - 'empty'
      - 'unknown'
    """
    try:
        mask_img = nib.load(mask_path)
        mask = mask_img.get_fdata()

        coords = np.argwhere(mask > 0)
        if coords.shape[0] < min_voxels:
            return "empty"

        x_center = coords[:, 0].mean()
        x_mid = mask.shape[0] / 2.0

        # tolerance band around the center to avoid unstable assignments
        tol = mask.shape[0] * 0.05

        if abs(x_center - x_mid) <= tol:
            return "midline"
        elif x_center < x_mid:
            return "left"
        else:
            return "right"

    except Exception:
        return "unknown"



def audit_one_image(
    path: str,
    percentiles: Tuple[float, float, float, float, float] = (0.5, 1.0, 50.0, 99.0, 99.5),
    use_nonzero_only: bool = False,
) -> ImageAudit:
    """
    Loads a NIfTI image and computes basic stats.

    use_nonzero_only:
      - If True, ignores zero voxels when computing percentiles (sometimes useful if background is 0).
      - For CT HU, keep False initially.
    """
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)  # float32 to avoid huge memory from float64
    arr = data

    if use_nonzero_only:
        nz = arr[arr != 0]
        if nz.size > 0:
            arr = nz

    pvals = np.percentile(arr, list(percentiles)).astype(np.float64)

    spacing = _get_spacing(img)
    orient = _get_orientation(img)

    folder_contents = _list_case_folder_contents(path)
    side = None
    side_source_mask = None

    mask_path = _find_side_mask_for_image(path)
    if mask_path is None:
        side = "missing_level2_mask"
    else:
        side_source_mask = os.path.basename(mask_path)
        side = infer_side_from_mask(mask_path)

    return ImageAudit(
        path=path,
        shape=tuple(data.shape),
        dtype=str(data.dtype),
        min=_safe_float(np.min(arr)),
        max=_safe_float(np.max(arr)),
        p0_5=_safe_float(pvals[0]),
        p1=_safe_float(pvals[1]),
        p50=_safe_float(pvals[2]),
        p99=_safe_float(pvals[3]),
        p99_5=_safe_float(pvals[4]),
        spacing_xyz=spacing,
        orientation=orient,
        side=side,
        side_source_mask=side_source_mask,
        folder_contents=folder_contents,
    )


def _summarize_spacings(spacings: List[Tuple[float, float, float]]) -> Dict[str, Dict[str, float]]:
    xs = np.array([s[0] for s in spacings], dtype=np.float64)
    ys = np.array([s[1] for s in spacings], dtype=np.float64)
    zs = np.array([s[2] for s in spacings], dtype=np.float64)

    def stats(v: np.ndarray) -> Dict[str, float]:
        v = v[np.isfinite(v)]
        if v.size == 0:
            return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "p50": float("nan")}
        return {
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "mean": float(np.mean(v)),
            "p50": float(np.percentile(v, 50)),
        }

    return {"x": stats(xs), "y": stats(ys), "z": stats(zs)}


def _gather_paths(
    root_dir: Optional[str],
    pattern: str,
    list_file: Optional[str],
    paths: Optional[List[str]],
) -> List[str]:
    out: List[str] = []

    if paths:
        out.extend(paths)

    if list_file:
        with open(list_file, "r") as f:
            for line in f:
                p = line.strip()
                if p:
                    out.append(p)

    if root_dir:
        gpat = os.path.join(root_dir, pattern)
        out.extend(glob.glob(gpat, recursive=True))

    # unique + keep order
    seen = set()
    uniq = []
    for p in out:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            uniq.append(ap)
    return uniq


def audit_dataset(
    root_dir: Optional[str] = None,
    pattern: str = "**/image.nii.gz",
    list_file: Optional[str] = None,
    paths: Optional[List[str]] = None,
    max_cases: Optional[int] = 50,
    use_nonzero_only: bool = False,
    save_json: Optional[str] = None,
    side_counts: Dict[str, int] = {},
) -> List[ImageAudit]:
    """
    Runs audit over a set of image paths and prints a summary.

    Inputs:
      - root_dir + pattern: glob search
      - list_file: a text file with one path per line
      - paths: explicit list

    max_cases:
      - limits processing for speed (set None to process all)
    """
    img_paths = _gather_paths(root_dir, pattern, list_file, paths)
    if max_cases is not None:
        img_paths = img_paths[: int(max_cases)]

    if len(img_paths) == 0:
        raise ValueError("No image paths found. Check root_dir/pattern or list_file/paths.")

    audits: List[ImageAudit] = []
    orientations: Dict[str, int] = {}
    mask_presence_counts: Dict[str, int] = {}

    print(f"[Audit] Found {len(img_paths)} images to audit.")
    for i, p in enumerate(img_paths, start=1):
        a = audit_one_image(p, use_nonzero_only=use_nonzero_only)
        audits.append(a)
        orientations[a.orientation] = orientations.get(a.orientation, 0) + 1
        side_counts[a.side] = side_counts.get(a.side, 0) + 1
        seen_masks_this_case = set()

        for fname in a.folder_contents:
            lower = fname.lower()

            # count only masks, and only nifti masks
            if lower.startswith("mask") and (lower.endswith(".nii") or lower.endswith(".nii.gz")):
                seen_masks_this_case.add(lower)

        for mask_name in seen_masks_this_case:
            mask_presence_counts[mask_name] = mask_presence_counts.get(mask_name, 0) + 1

        print(
            f"\n[{i}/{len(img_paths)}] {a.path}\n"
            f"  shape: {a.shape} | dtype: {a.dtype}\n"
            f"  spacing(x,y,z): {a.spacing_xyz}\n"
            f"  orientation: {a.orientation}\n"
            f"  side: {a.side} | source_mask: {a.side_source_mask}\n"
            f"  min/max: {a.min:.3f} / {a.max:.3f}\n"
            f"  p0.5/p1/p50/p99/p99.5: {a.p0_5:.3f} / {a.p1:.3f} / {a.p50:.3f} / {a.p99:.3f} / {a.p99_5:.3f}"
        )
        print(f"  folder_contents ({len(a.folder_contents)} files):")
        for fname in a.folder_contents:
            print(f"    - {fname}")

    spacings = [a.spacing_xyz for a in audits]
    spacing_summary = _summarize_spacings(spacings)

    print("\n[Audit Summary]")
    print("  Orientation counts:")
    for k, v in sorted(orientations.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"    {k}: {v}")

    print("  Spacing summary (mm):")
    for axis in ["x", "y", "z"]:
        s = spacing_summary[axis]
        print(f"    {axis}: min={s['min']:.4f} max={s['max']:.4f} mean={s['mean']:.4f} p50={s['p50']:.4f}")

    print("  Side counts:")
    for k, v in sorted(side_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"    {k}: {v}")

    print("  Mask presence summary (patients containing mask / total patients):")

    sorted_masks = sorted(
        mask_presence_counts.items(),
        key=lambda kv: kv[1],
        reverse=True
    )

    max_len = max(len(m) for m in mask_presence_counts)

    for mask_name, count in sorted_masks:
        print(f"    {mask_name.ljust(max_len)} : {count}/{len(audits)}")

    if save_json:
        payload = {
            "num_images": len(audits),
            "orientation_counts": orientations,
            "side_counts": side_counts,
            "spacing_summary": spacing_summary,
            "mask_presence_summary_sorted": [
                {
                    "mask_name": mask_name,
                    "fraction": f"{count}/{len(audits)}"
                }
                for mask_name, count in sorted_masks
            ],
            "images": [asdict(a) for a in audits],
        }
        os.makedirs(os.path.dirname(save_json), exist_ok=True)
        with open(save_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[Audit] Saved JSON report to: {save_json}")

    return audits