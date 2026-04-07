import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
)

from src.VocoLarge.training_ssl.pipeline.config import Config
from src.VocoLarge.training_ssl.pipeline.data import read_ids_file, build_files_from_ids


# ===================== CONFIG =====================
output_dir = "../train_valid_test_split/preprocessing_distribution_report"
num_workers = 12
num_z_slices = 8
xy_stride = 4

use_spacing = True
use_crop_foreground = False

max_train_cases = None   # e.g. 150 for faster debug, or None for all
max_val_cases = None
max_test_cases = None
# ================================================


def to_numpy_image(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    elif not isinstance(x, np.ndarray):
        x = np.asarray(x)
    return x

def sample_processed_volume_fast(image_np):
    if image_np.ndim != 3:
        raise RuntimeError(f"Expected 3D image, got {image_np.shape}")

    sx, sy, sz = image_np.shape

    if sz <= num_z_slices:
        z_idx = np.arange(sz)
    else:
        z_idx = np.linspace(0, sz - 1, num=num_z_slices, dtype=int)

    chunks = []
    for z in z_idx:
        sl = image_np[:, :, z]
        sl = sl[::xy_stride, ::xy_stride]
        chunks.append(sl.reshape(-1))

    if not chunks:
        return np.array([], dtype=np.float32)

    return np.concatenate(chunks, axis=0).astype(np.float32)


def compute_processed_image_stats(
    image_np,
    split_name,
    patient_id,
):
    if image_np.ndim == 4 and image_np.shape[0] == 1:
        image_np = image_np[0]

    if image_np.ndim != 3:
        raise RuntimeError(f"Unexpected processed image shape: {image_np.shape}")

    vals = sample_processed_volume_fast(image_np=image_np)

    return {
        "patient_id": patient_id,
        "split": split_name,
        "shape_0": int(image_np.shape[0]),
        "shape_1": int(image_np.shape[1]),
        "shape_2": int(image_np.shape[2]),
        "intensity_mean": float(np.mean(vals)),
        "intensity_std": float(np.std(vals)),
        "p01": float(np.percentile(vals, 1)),
        "p50": float(np.percentile(vals, 50)),
        "p99": float(np.percentile(vals, 99)),
        "frac_zero": float(np.mean(vals == 0)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "frac_nonzero": float(np.mean(vals != 0)),
    }


def deterministic_transform():
    base_trans = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000.0,
            a_max=500.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]

    if use_spacing:
        base_trans += [
            Spacingd(
                keys=["image"],
                pixdim=(1.25, 1.25, 5.0),
                mode="bilinear",
            )
        ]

    if use_crop_foreground:
        base_trans += [
            CropForegroundd(keys=["image"], source_key="image")
        ]

    return Compose(base_trans)


def process_one_file(args: Tuple[Dict[str, str], str]) -> Dict:
    item, split_name = args

    xform = deterministic_transform()
    out = xform(item)

    image_np = to_numpy_image(out["image"])
    patient_id = out.get("patient_id", item.get("patient_id", "unknown"))

    return compute_processed_image_stats(image_np, split_name, patient_id)


def collect_stats_parallel(files: List[Dict[str, str]], split_name: str, num_workers: int) -> List[Dict]:
    rows = []
    tasks = [(item, split_name) for item in files]

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for i, row in enumerate(ex.map(process_one_file, tasks), start=1):
            rows.append(row)
            if i % 50 == 0 or i == len(tasks):
                print(f"[INFO] Processed {i}/{len(tasks)} for split={split_name}")

    return rows


def plot_hist_by_split(ax, df: pd.DataFrame, column: str, bins=30, title=None):
    split_order = ["train", "val", "test"]

    for split in split_order:
        values = df.loc[df["split"] == split, column].dropna().values
        if len(values) > 0:
            ax.hist(
                values,
                bins=bins,
                alpha=0.45,
                density=True,
                label=f"{split} (n={len(values)})",
            )

    ax.set_title(title or column)
    ax.set_xlabel(column)
    ax.set_ylabel("density")
    ax.legend()


def save_hist_figure(df: pd.DataFrame, output_path: str):
    metrics = [
        ("shape_0", "Processed shape dim 0"),
        ("shape_1", "Processed shape dim 1"),
        ("shape_2", "Processed shape dim 2"),
        ("intensity_mean", "Processed intensity mean"),
        ("intensity_std", "Processed intensity std"),
        ("frac_zero", "Fraction of zeros"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    axes = axes.ravel()

    for ax, (col, title) in zip(axes, metrics):
        plot_hist_by_split(ax, df, col, title=title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_extra_hist_figure(df: pd.DataFrame, output_path: str):
    metrics = [
        ("min", "Processed min"),
        ("max", "Processed max"),
        ("p01", "Processed p01"),
        ("p50", "Processed p50"),
        ("p99", "Processed p99"),
        ("frac_nonzero", "Fraction nonzero"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    axes = axes.ravel()

    for ax, (col, title) in zip(axes, metrics):
        if col not in df.columns:
            ax.axis("off")
            continue
        plot_hist_by_split(ax, df, col, title=title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_summary_table(df: pd.DataFrame, output_csv: str):
    metrics = [c for c in df.columns if c not in ("patient_id", "split")]
    summary = df.groupby("split")[metrics].agg(["mean", "std", "median", "min", "max"])
    summary.to_csv(output_csv)


def preprocessing_distribution(args: Config):
    os.makedirs(output_dir, exist_ok=True)

    train_ids = read_ids_file(args.train_ids_path)
    val_ids = read_ids_file(args.val_ids_path)
    test_ids = read_ids_file(args.test_ids_path)

    train_files = build_files_from_ids(args.data_dir, train_ids)
    val_files = build_files_from_ids(args.data_dir, val_ids)
    test_files = build_files_from_ids(args.data_dir, test_ids)

    if max_train_cases is not None:
        train_files = train_files[:max_train_cases]
    if max_val_cases is not None:
        val_files = val_files[:max_val_cases]
    if max_test_cases is not None:
        test_files = test_files[:max_test_cases]

    print(f"[INFO] train files used: {len(train_files)}")
    print(f"[INFO] val files used:   {len(val_files)}")
    print(f"[INFO] test files used:  {len(test_files)}")
    print(f"[INFO] num_workers:      {num_workers}")
    print(f"[INFO] use_spacing:      {use_spacing}")
    print(f"[INFO] use_crop_fg:      {use_crop_foreground}")

    rows = []
    rows.extend(collect_stats_parallel(train_files, "train", num_workers))
    rows.extend(collect_stats_parallel(val_files, "val", num_workers))
    rows.extend(collect_stats_parallel(test_files, "test", num_workers))

    df = pd.DataFrame(rows)

    per_case_csv = os.path.join(output_dir, "per_case_processed_stats.csv")
    summary_csv = os.path.join(output_dir, "summary_processed_stats.csv")
    hist_png = os.path.join(output_dir, "processed_distributions_hist.png")
    extra_png = os.path.join(output_dir, "processed_distributions_extra_hist.png")

    df.to_csv(per_case_csv, index=False)
    save_summary_table(df, summary_csv)
    save_hist_figure(df, hist_png)
    save_extra_hist_figure(df, extra_png)

    print(f"[INFO] Saved per-case stats to: {per_case_csv}")
    print(f"[INFO] Saved summary stats to:  {summary_csv}")
    print(f"[INFO] Saved figure to:         {hist_png}")
    print(f"[INFO] Saved extra figure to:   {extra_png}")


if __name__ == "__main__":
    preprocessing_distribution(Config())