import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.VocoLarge.segmentation.config_binary import ConfigBinary
from src.VocoLarge.segmentation.data.data_utils_binary import (
    read_ids_file,
    build_segmentation_files_from_ids,
)
from src.VocoLarge.segmentation.hyperparameter_tuning.cofig_binary_test import ConfigBinaryTest


def load_lymph_terms(json_path: str) -> List[str]:
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        terms = data
    elif isinstance(data, dict) and "terms" in data:
        terms = data["terms"]
    else:
        raise ValueError(
            f"Invalid lymph terms JSON format in {json_path}. "
            f"Expected list or dict with key 'terms'."
        )

    terms = [str(t).lower().strip() for t in terms if str(t).strip()]
    if not terms:
        raise ValueError(f"No lymph terms found in {json_path}")
    return terms


def is_lymph_mask(mask_path: str, lymph_terms: List[str]) -> bool:
    name = Path(mask_path).name.lower()
    return any(term in name for term in lymph_terms)


def build_binary_mask_from_paths(mask_paths: List[str], lymph_terms: List[str]) -> Tuple[np.ndarray, List[str]]:
    matched_paths = [p for p in mask_paths if is_lymph_mask(p, lymph_terms)]

    ref_img = nib.load(mask_paths[0])
    ref_shape = ref_img.shape
    binary_mask = np.zeros(ref_shape, dtype=np.uint8)

    for p in matched_paths:
        arr = nib.load(p).get_fdata()
        if arr.shape != ref_shape:
            raise ValueError(f"Shape mismatch for {p}: got {arr.shape}, expected {ref_shape}")
        binary_mask[arr > 0] = 1

    return binary_mask, matched_paths


def compute_bbox(mask: np.ndarray) -> Dict[str, Any]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return {
            "bbox_x_min": np.nan, "bbox_x_max": np.nan,
            "bbox_y_min": np.nan, "bbox_y_max": np.nan,
            "bbox_z_min": np.nan, "bbox_z_max": np.nan,
            "bbox_dx": 0, "bbox_dy": 0, "bbox_dz": 0,
            "bbox_volume": 0,
        }

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    dx = int(maxs[0] - mins[0] + 1)
    dy = int(maxs[1] - mins[1] + 1)
    dz = int(maxs[2] - mins[2] + 1)

    return {
        "bbox_x_min": int(mins[0]), "bbox_x_max": int(maxs[0]),
        "bbox_y_min": int(mins[1]), "bbox_y_max": int(maxs[1]),
        "bbox_z_min": int(mins[2]), "bbox_z_max": int(maxs[2]),
        "bbox_dx": dx, "bbox_dy": dy, "bbox_dz": dz,
        "bbox_volume": int(dx * dy * dz),
    }


def compute_center_of_mass(mask: np.ndarray) -> Dict[str, Any]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return {"com_x": np.nan, "com_y": np.nan, "com_z": np.nan}

    com = coords.mean(axis=0)
    return {"com_x": float(com[0]), "com_y": float(com[1]), "com_z": float(com[2])}


def compute_slice_presence(mask: np.ndarray) -> Dict[str, Any]:
    z_presence = (mask > 0).any(axis=(0, 1))
    return {
        "num_fg_slices": int(z_presence.sum()),
        "fg_slice_first": int(np.argmax(z_presence)) if z_presence.any() else np.nan,
        "fg_slice_last": int(len(z_presence) - 1 - np.argmax(z_presence[::-1])) if z_presence.any() else np.nan,
    }


def compute_intensity_stats(image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    fg = image[mask > 0]
    bg = image[mask == 0]

    out = {}
    if fg.size > 0:
        out["img_fg_mean"] = float(fg.mean())
        out["img_fg_std"] = float(fg.std())
        out["img_fg_p01"] = float(np.percentile(fg, 1))
        out["img_fg_p99"] = float(np.percentile(fg, 99))
    else:
        out["img_fg_mean"] = np.nan
        out["img_fg_std"] = np.nan
        out["img_fg_p01"] = np.nan
        out["img_fg_p99"] = np.nan

    if bg.size > 0:
        out["img_bg_mean"] = float(bg.mean())
        out["img_bg_std"] = float(bg.std())
        out["img_bg_p01"] = float(np.percentile(bg, 1))
        out["img_bg_p99"] = float(np.percentile(bg, 99))
    else:
        out["img_bg_mean"] = np.nan
        out["img_bg_std"] = np.nan
        out["img_bg_p01"] = np.nan
        out["img_bg_p99"] = np.nan

    return out


def compute_case_stats(sample: Dict[str, Any], lymph_terms: List[str]) -> Tuple[Dict[str, Any], np.ndarray]:
    case_id = sample["case_id"]
    image_path = sample["image"]
    mask_paths = sample["mask_paths"]

    image_nii = nib.load(image_path)
    image = image_nii.get_fdata()

    binary_mask, matched_paths = build_binary_mask_from_paths(mask_paths, lymph_terms)

    fg_voxels = int((binary_mask > 0).sum())
    total_voxels = int(binary_mask.size)
    fg_fraction = float(fg_voxels / total_voxels)

    stats = {
        "case_id": case_id,
        "image_path": image_path,
        "num_mask_paths_total": len(mask_paths),
        "num_mask_paths_matched": len(matched_paths),
        "matched_mask_names": " | ".join([Path(p).name for p in matched_paths]),
        "shape_x": int(binary_mask.shape[0]),
        "shape_y": int(binary_mask.shape[1]),
        "shape_z": int(binary_mask.shape[2]),
        "total_voxels": total_voxels,
        "fg_voxels": fg_voxels,
        "fg_fraction": fg_fraction,
        "is_empty": int(fg_voxels == 0),
    }

    stats.update(compute_bbox(binary_mask))
    stats.update(compute_center_of_mass(binary_mask))
    stats.update(compute_slice_presence(binary_mask))
    stats.update(compute_intensity_stats(image, binary_mask))

    return stats, binary_mask


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_hist(df_map: Dict[str, pd.DataFrame], column: str, out_path: str, bins=40, log_x=False):
    plt.figure(figsize=(7, 5))
    for split_name, df in df_map.items():
        vals = df[column].dropna().values
        if len(vals) == 0:
            continue
        plt.hist(vals, bins=bins, alpha=0.4, label=split_name)

    if log_x:
        plt.xscale("log")

    plt.xlabel(column)
    plt.ylabel("Count")
    plt.title(column)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_scatter(df_map: Dict[str, pd.DataFrame], xcol: str, ycol: str, out_path: str):
    plt.figure(figsize=(6, 6))
    for split_name, df in df_map.items():
        x = df[xcol].values
        y = df[ycol].values
        plt.scatter(x, y, s=12, alpha=0.5, label=split_name)

    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(f"{ycol} vs {xcol}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_bar_empty_counts(df_map: Dict[str, pd.DataFrame], out_path: str):
    splits = []
    vals = []
    for split_name, df in df_map.items():
        splits.append(split_name)
        vals.append(int(df["is_empty"].sum()))

    plt.figure(figsize=(6, 4))
    plt.bar(splits, vals)
    plt.ylabel("Empty mask count")
    plt.title("Empty masks by split")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_summary_csv(df_map: Dict[str, pd.DataFrame], out_path: str):
    rows = []
    for split_name, df in df_map.items():
        rows.append({
            "split": split_name,
            "n_cases": len(df),
            "n_empty": int(df["is_empty"].sum()),
            "fg_voxels_mean": float(df["fg_voxels"].mean()),
            "fg_voxels_median": float(df["fg_voxels"].median()),
            "fg_fraction_mean": float(df["fg_fraction"].mean()),
            "fg_fraction_median": float(df["fg_fraction"].median()),
            "num_mask_paths_matched_mean": float(df["num_mask_paths_matched"].mean()),
            "bbox_volume_mean": float(df["bbox_volume"].mean()),
            "bbox_volume_median": float(df["bbox_volume"].median()),
            "num_fg_slices_mean": float(df["num_fg_slices"].mean()),
            "num_fg_slices_median": float(df["num_fg_slices"].median()),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def accumulate_mean_axis_profiles(masks: List[np.ndarray]) -> Dict[str, np.ndarray]:
    if len(masks) == 0:
        return {"x": np.array([]), "y": np.array([]), "z": np.array([])}

    sx, sy, sz = masks[0].shape
    prof_x = np.zeros(sx, dtype=np.float64)
    prof_y = np.zeros(sy, dtype=np.float64)
    prof_z = np.zeros(sz, dtype=np.float64)

    for m in masks:
        prof_x += m.mean(axis=(1, 2))
        prof_y += m.mean(axis=(0, 2))
        prof_z += m.mean(axis=(0, 1))

    prof_x /= len(masks)
    prof_y /= len(masks)
    prof_z /= len(masks)

    return {"x": prof_x, "y": prof_y, "z": prof_z}


def save_axis_profiles(profile_map: Dict[str, Dict[str, np.ndarray]], out_dir: str):
    for axis in ["x", "y", "z"]:
        plt.figure(figsize=(7, 4))
        for split_name, profs in profile_map.items():
            prof = profs[axis]
            if prof.size == 0:
                continue
            plt.plot(prof, label=split_name)
        plt.xlabel(f"{axis} index")
        plt.ylabel("Mean foreground presence")
        plt.title(f"Mean foreground profile along {axis}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"mean_fg_profile_{axis}.png"), dpi=150)
        plt.close()


def analyze_split(split_name: str, samples: List[Dict[str, Any]], lymph_terms: List[str], out_dir: str):
    rows = []
    masks = []

    for i, sample in enumerate(samples):
        stats, binary_mask = compute_case_stats(sample, lymph_terms)
        stats["split"] = split_name
        rows.append(stats)
        masks.append(binary_mask)

        if (i + 1) % 50 == 0:
            print(f"[{split_name}] processed {i + 1}/{len(samples)}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"{split_name}_case_stats.csv"), index=False)

    profiles = accumulate_mean_axis_profiles(masks)
    return df, profiles


def main():
    cfg = ConfigBinaryTest()

    out_dir = os.path.join(cfg.save_dir, "binary_mask_distribution_analysis")
    plots_dir = os.path.join(out_dir, "plots")
    ensure_dir(out_dir)
    ensure_dir(plots_dir)

    lymph_terms = load_lymph_terms(cfg.lymph_terms_json)

    train_ids = read_ids_file(cfg.train_ids_path)
    val_ids = read_ids_file(cfg.val_ids_path)
    test_ids = read_ids_file(cfg.test_ids_path)

    train_samples = build_segmentation_files_from_ids(cfg.root_dir, train_ids)
    val_samples = build_segmentation_files_from_ids(cfg.root_dir, val_ids)
    test_samples = build_segmentation_files_from_ids(cfg.root_dir, test_ids)

    print(f"[INFO] train cases: {len(train_samples)}")
    print(f"[INFO] val cases:   {len(val_samples)}")
    print(f"[INFO] test cases:  {len(test_samples)}")

    train_df, train_profiles = analyze_split("train", train_samples, lymph_terms, out_dir)
    val_df, val_profiles = analyze_split("val", val_samples, lymph_terms, out_dir)
    test_df, test_profiles = analyze_split("test", test_samples, lymph_terms, out_dir)

    df_map = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    profile_map = {
        "train": train_profiles,
        "val": val_profiles,
        "test": test_profiles,
    }

    save_summary_csv(df_map, os.path.join(out_dir, "summary_stats.csv"))

    save_hist(df_map, "fg_voxels", os.path.join(plots_dir, "hist_fg_voxels.png"), bins=50, log_x=True)
    save_hist(df_map, "fg_fraction", os.path.join(plots_dir, "hist_fg_fraction.png"), bins=50, log_x=True)
    save_hist(df_map, "num_mask_paths_matched", os.path.join(plots_dir, "hist_num_matched_masks.png"), bins=20)
    save_hist(df_map, "bbox_volume", os.path.join(plots_dir, "hist_bbox_volume.png"), bins=50, log_x=True)
    save_hist(df_map, "bbox_dx", os.path.join(plots_dir, "hist_bbox_dx.png"), bins=40)
    save_hist(df_map, "bbox_dy", os.path.join(plots_dir, "hist_bbox_dy.png"), bins=40)
    save_hist(df_map, "bbox_dz", os.path.join(plots_dir, "hist_bbox_dz.png"), bins=40)
    save_hist(df_map, "num_fg_slices", os.path.join(plots_dir, "hist_num_fg_slices.png"), bins=40)

    save_scatter(df_map, "com_x", "com_y", os.path.join(plots_dir, "scatter_com_x_y.png"))
    save_scatter(df_map, "com_x", "com_z", os.path.join(plots_dir, "scatter_com_x_z.png"))
    save_scatter(df_map, "com_y", "com_z", os.path.join(plots_dir, "scatter_com_y_z.png"))

    save_bar_empty_counts(df_map, os.path.join(plots_dir, "empty_mask_counts.png"))
    save_axis_profiles(profile_map, plots_dir)

    print(f"[INFO] Saved analysis to: {out_dir}")


if __name__ == "__main__":
    main()