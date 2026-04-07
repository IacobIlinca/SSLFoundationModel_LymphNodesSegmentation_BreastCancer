import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.VocoLarge.training_ssl.pipeline.data import read_ids_file, build_files_from_ids


# ===================== CONFIG =====================
root_dir = "/mnt/data/flaviu/rtnation_02_02"

train_ids_path = "../train_valid_test_split/train_split_ids.txt"
val_ids_path = "../train_valid_test_split/val_split_ids.txt"
test_ids_path = "../train_valid_test_split/test_split_ids.txt"

output_dir = "../train_valid_test_split/split_distribution_report"

compute_intensity_stats = True
num_z_slices = 8
xy_stride = 4

num_workers = 8   # try 4 first if storage is slow
# ================================================


def sample_volume_fast(dataobj, shape, num_z_slices=8, xy_stride=4) -> np.ndarray:
    sx, sy, sz = shape[:3]

    if sz <= num_z_slices:
        z_idx = np.arange(sz)
    else:
        z_idx = np.linspace(0, sz - 1, num=num_z_slices, dtype=int)

    chunks = []
    for z in z_idx:
        sl = np.asarray(dataobj[:, :, z], dtype=np.float32)
        sl = sl[::xy_stride, ::xy_stride]
        chunks.append(sl.reshape(-1))

    if not chunks:
        return np.array([], dtype=np.float32)

    return np.concatenate(chunks, axis=0)


def compute_case_stats_from_item(args: Tuple[Dict[str, str], str, bool, int, int]) -> Dict:
    item, split_name, compute_intensity, num_z_slices, xy_stride = args

    image_path = item["image"]
    patient_id = item["patient_id"]

    nii = nib.load(image_path)
    hdr = nii.header

    shape = nii.shape
    if len(shape) < 3:
        raise RuntimeError(f"Expected 3D image, got shape {shape} for {image_path}")

    sx, sy, sz = shape[:3]

    zooms = hdr.get_zooms()
    if len(zooms) < 3:
        raise RuntimeError(f"Expected 3 spacing values, got {zooms} for {image_path}")

    dx, dy, dz = zooms[:3]

    row = {
        "patient_id": patient_id,
        "split": split_name,
        "image_path": image_path,

        "shape_x": int(sx),
        "shape_y": int(sy),
        "shape_z": int(sz),

        "spacing_x": float(dx),
        "spacing_y": float(dy),
        "spacing_z": float(dz),

        "fov_x_mm": float(sx * dx),
        "fov_y_mm": float(sy * dy),
        "fov_z_mm": float(sz * dz),
    }

    if compute_intensity:
        vals = sample_volume_fast(
            dataobj=nii.dataobj,
            shape=shape,
            num_z_slices=num_z_slices,
            xy_stride=xy_stride,
        )

        if vals.size == 0:
            row.update({
                "intensity_mean": np.nan,
                "intensity_std": np.nan,
                "p01": np.nan,
                "p50": np.nan,
                "p99": np.nan,
                "frac_gt_minus900": np.nan,
            })
        else:
            row.update({
                "intensity_mean": float(np.mean(vals)),
                "intensity_std": float(np.std(vals)),
                "p01": float(np.percentile(vals, 1)),
                "p50": float(np.percentile(vals, 50)),
                "p99": float(np.percentile(vals, 99)),
                "frac_gt_minus900": float(np.mean(vals > -900)),
            })

    return row


def collect_stats_for_split_parallel(
    files: List[Dict[str, str]],
    split_name: str,
    compute_intensity: bool,
    num_z_slices: int,
    xy_stride: int,
    num_workers: int,
) -> List[Dict]:
    tasks = [
        (item, split_name, compute_intensity, num_z_slices, xy_stride)
        for item in files
    ]

    rows = []
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for i, row in enumerate(ex.map(compute_case_stats_from_item, tasks), start=1):
            rows.append(row)
            if i % 50 == 0 or i == len(tasks):
                print(f"[INFO] Processed {i}/{len(tasks)} for split={split_name}")

    return rows


def plot_hist_by_split(ax, df: pd.DataFrame, column: str, bins=30, title=None):
    split_order = ["train", "val", "test"]

    for split in split_order:
        values = df.loc[df["split"] == split, column].dropna().values
        if len(values) > 0:
            ax.hist(values, bins=bins, alpha=0.45, density=True, label=split)

    ax.set_title(title or column)
    ax.set_xlabel(column)
    ax.set_ylabel("density")
    ax.legend()


def save_hist_figure(df: pd.DataFrame, output_path: str, include_intensity: bool):
    if include_intensity:
        metrics = [
            ("spacing_z", "Slice spacing (z)"),
            ("shape_z", "Number of slices (z)"),
            ("fov_z_mm", "Physical coverage in z (mm)"),
            ("intensity_mean", "Mean intensity"),
            ("intensity_std", "Intensity std"),
            ("frac_gt_minus900", "Fraction > -900 HU"),
        ]
    else:
        metrics = [
            ("spacing_z", "Slice spacing (z)"),
            ("shape_z", "Number of slices (z)"),
            ("fov_z_mm", "Physical coverage in z (mm)"),
            ("spacing_x", "Spacing x"),
            ("spacing_y", "Spacing y"),
            ("fov_x_mm", "Physical coverage in x (mm)"),
        ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    axes = axes.ravel()

    for ax, (col, title) in zip(axes, metrics):
        plot_hist_by_split(ax, df, col, title=title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_summary_table(df: pd.DataFrame, output_csv: str):
    metrics = [c for c in df.columns if c not in ("patient_id", "split", "image_path")]
    summary = df.groupby("split")[metrics].agg(["mean", "std", "median", "min", "max"])
    summary.to_csv(output_csv)


def main():
    os.makedirs(output_dir, exist_ok=True)

    train_ids = read_ids_file(train_ids_path)
    val_ids = read_ids_file(val_ids_path)
    test_ids = read_ids_file(test_ids_path)

    print(f"[INFO] train ids: {len(train_ids)}")
    print(f"[INFO] val ids:   {len(val_ids)}")
    print(f"[INFO] test ids:  {len(test_ids)}")

    train_files = build_files_from_ids(root_dir, train_ids)
    val_files = build_files_from_ids(root_dir, val_ids)
    test_files = build_files_from_ids(root_dir, test_ids)

    print(f"[INFO] train files found: {len(train_files)}")
    print(f"[INFO] val files found:   {len(val_files)}")
    print(f"[INFO] test files found:  {len(test_files)}")
    print(f"[INFO] Using num_workers={num_workers}")

    rows = []
    rows.extend(collect_stats_for_split_parallel(
        train_files, "train",
        compute_intensity=compute_intensity_stats,
        num_z_slices=num_z_slices,
        xy_stride=xy_stride,
        num_workers=num_workers,
    ))
    rows.extend(collect_stats_for_split_parallel(
        val_files, "val",
        compute_intensity=compute_intensity_stats,
        num_z_slices=num_z_slices,
        xy_stride=xy_stride,
        num_workers=num_workers,
    ))
    rows.extend(collect_stats_for_split_parallel(
        test_files, "test",
        compute_intensity=compute_intensity_stats,
        num_z_slices=num_z_slices,
        xy_stride=xy_stride,
        num_workers=num_workers,
    ))

    df = pd.DataFrame(rows)

    per_case_csv = os.path.join(output_dir, "per_case_stats.csv")
    summary_csv = os.path.join(output_dir, "summary_stats.csv")
    hist_png = os.path.join(output_dir, "split_distributions_hist.png")

    df.to_csv(per_case_csv, index=False)
    save_summary_table(df, summary_csv)
    save_hist_figure(df, hist_png, include_intensity=compute_intensity_stats)

    print(f"[INFO] Saved per-case stats to: {per_case_csv}")
    print(f"[INFO] Saved summary stats to:  {summary_csv}")
    print(f"[INFO] Saved figure to:         {hist_png}")


if __name__ == "__main__":
    main()