import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch


def save_heatmap(mat, title, path, xlabel="crop id (0..8)", ylabel="query id"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.figure(figsize=(8, max(3, 0.35 * mat.shape[0])))
    ax = plt.gca()
    im = ax.imshow(mat, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def _to_numpy_3d(x):
    if hasattr(x, "as_tensor"):
        x = x.as_tensor()
    if torch.is_tensor(x):
        x = x.detach().float().cpu()
        if x.ndim == 4 and x.shape[0] == 1:
            x = x[0]
        return x.numpy()
    raise TypeError(f"Expected torch tensor, got {type(x)}")


def _norm01(vol):
    vmin = float(np.min(vol))
    vmax = float(np.max(vol))
    return (vol - vmin) / (vmax - vmin + 1e-6)


def _pick_slices(depth, n=6):
    if depth <= n:
        return list(range(depth))
    lo = max(0, int(depth * 0.15))
    hi = min(depth - 1, int(depth * 0.85))
    idxs = np.linspace(lo, hi, n).round().astype(int)
    return idxs.tolist()


# Visualization mainly used in overfit experiment, but can be used elsewhere too
def save_voco_debug_vis(
    img, crops, labels,
    out_dir="debug_vis",
    prefix="case",
    max_queries=10,
    slices_per_vol=6,
):
    os.makedirs(out_dir, exist_ok=True)

    # labels -> (sw_s, 9)
    if torch.is_tensor(labels):
        lab = labels.detach().float().cpu()
        if lab.ndim == 3:
            lab = lab[0]
    else:
        lab = torch.as_tensor(labels).float()
        if lab.ndim == 3:
            lab = lab[0]

    sw_s = lab.shape[0]
    n_queries = min(sw_s, max_queries)

    img_t = img.as_tensor() if hasattr(img, "as_tensor") else img
    crops_t = crops.as_tensor() if hasattr(crops, "as_tensor") else crops
    img_t = img_t.detach().cpu()
    crops_t = crops_t.detach().cpu()

    if img_t.ndim != 5:
        raise ValueError(f"Expected img shape (sw_s,1,D,H,W), got {tuple(img_t.shape)}")
    if crops_t.ndim != 5 or crops_t.shape[0] != 9:
        raise ValueError(f"Expected crops shape (9,1,D,H,W), got {tuple(crops_t.shape)}")

    D = int(img_t.shape[2])
    slice_ids = _pick_slices(D, n=slices_per_vol)

    # 1) query grid
    fig = plt.figure(figsize=(2.4 * slices_per_vol, 2.4 * n_queries))
    for qi in range(n_queries):
        vol = _norm01(_to_numpy_3d(img_t[qi]))
        for sj, z in enumerate(slice_ids):
            ax = plt.subplot(n_queries, slices_per_vol, qi * slices_per_vol + sj + 1)
            ax.imshow(vol[z], cmap="gray")
            ax.set_axis_off()
            if sj == 0:
                ax.set_title(f"q{qi}", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_query_grid.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 2) crops grid
    fig = plt.figure(figsize=(2.4 * slices_per_vol, 2.4 * 9))
    for ci in range(9):
        vol = _norm01(_to_numpy_3d(crops_t[ci]))
        for sj, z in enumerate(slice_ids):
            ax = plt.subplot(9, slices_per_vol, ci * slices_per_vol + sj + 1)
            ax.imshow(vol[z], cmap="gray")
            ax.set_axis_off()
            if sj == 0:
                ax.set_title(f"c{ci}", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_crops_grid.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 3) labels heatmap
    save_heatmap(
        lab.numpy(),
        "VoCo labels / targets (query vs 9 crops)",
        os.path.join(out_dir, f"{prefix}_labels_heatmap.png"),
    )

    # 4) best-crop per query (middle slice)
    best = torch.argmax(lab, dim=1)
    zmid = int(D // 2)

    fig = plt.figure(figsize=(6, 2.4 * n_queries))
    for qi in range(n_queries):
        qvol = _norm01(_to_numpy_3d(img_t[qi]))
        cidx = int(best[qi].item())
        cvol = _norm01(_to_numpy_3d(crops_t[cidx]))

        ax1 = plt.subplot(n_queries, 2, qi * 2 + 1)
        ax1.imshow(qvol[zmid], cmap="gray")
        ax1.set_axis_off()
        ax1.set_title(f"q{qi} (z={zmid})", fontsize=10)

        ax2 = plt.subplot(n_queries, 2, qi * 2 + 2)
        ax2.imshow(cvol[zmid], cmap="gray")
        ax2.set_axis_off()
        ax2.set_title(f"best crop c{cidx} (label={lab[qi, cidx]:.3f})", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_bestcrop_per_query.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_diff_bundle(logits, targets, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    save_heatmap(targets.numpy(), "Targets (labels): query vs 9 crops", os.path.join(out_dir, f"{prefix}_targets.png"))
    save_heatmap(logits.numpy(), "Predictions (logits): query vs 9 crops", os.path.join(out_dir, f"{prefix}_logits.png"))
    save_heatmap((logits - targets).numpy(), "Pred - Target", os.path.join(out_dir, f"{prefix}_diff.png"))
    save_heatmap((logits - targets).abs().numpy(), "|Pred - Target|", os.path.join(out_dir, f"{prefix}_absdiff.png"))


class History:
    """
    Generic metric history.
    Stores metric_name -> list of epochs and values.

    Example keys:
        train/loss_total
        train/loss_intra
        train/grad_backbone
        val/loss_total
        val/top1
    """
    def __init__(self):
        self.data = defaultdict(lambda: {"epoch": [], "value": []})

    def add(self, metric_name: str, epoch: int, value: float):
        self.data[metric_name]["epoch"].append(int(epoch))
        self.data[metric_name]["value"].append(float(value))

    def add_many(self, epoch: int, metrics: Dict[str, float]):
        for name, value in metrics.items():
            if value is None:
                continue
            self.add(name, epoch, value)

    def get(self, metric_name: str):
        return self.data.get(metric_name, {"epoch": [], "value": []})

    def to_dict(self) -> Dict:
        return dict(self.data)

    def save_json(self, save_dir: str, filename: str = "history.json"):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, filename), "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def plot_metric(history: History, metric_name: str, save_dir: str, title: str = None, ylabel: str = None):
    series = history.get(metric_name)
    epochs = series["epoch"]
    values = series["value"]

    if len(values) == 0:
        return

    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    plt.plot(epochs, values, label=metric_name)
    plt.title(title or metric_name)
    plt.xlabel("epoch")
    plt.ylabel(ylabel or metric_name)
    plt.legend()
    plt.tight_layout()

    safe_name = metric_name.replace("/", "_")
    plt.savefig(os.path.join(save_dir, f"{safe_name}.png"), dpi=150)
    plt.close()

def plot_metric_group(history: History, metric_names: List[str], save_dir: str, filename: str, title: str, ylabel: str):
    os.makedirs(save_dir, exist_ok=True)

    plotted_any = False
    plt.figure()

    for metric_name in metric_names:
        series = history.get(metric_name)
        epochs = series["epoch"]
        values = series["value"]

        if len(values) == 0:
            continue

        plt.plot(epochs, values, label=metric_name)
        plotted_any = True

    if not plotted_any:
        plt.close()
        return

    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{filename}.png"), dpi=150)
    plt.close()

def plot_all_curves(history: History, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    # losses
    plot_metric_group(
        history,
        metric_names=[
            "train/loss_total",
            "val/loss_total",
        ],
        save_dir=save_dir,
        filename="loss_total",
        title="Total Loss",
        ylabel="loss",
    )

    plot_metric_group(
        history,
        metric_names=[
            "train/loss_intra",
            "train/loss_inter",
            "train/loss_reg",
        ],
        save_dir=save_dir,
        filename="train_loss_components",
        title="Train Loss Components",
        ylabel="loss",
    )

    # validation quality
    plot_metric_group(
        history,
        metric_names=[
            "val/top1",
        ],
        save_dir=save_dir,
        filename="val_top1",
        title="Validation Top1",
        ylabel="top1",
    )

    # gradients
    plot_metric_group(
        history,
        metric_names=[
            "train/grad_backbone",
            "train/grad_student",
            "train/grad_teacher",
        ],
        save_dir=save_dir,
        filename="grad_norms",
        title="Gradient Norms",
        ylabel="grad norm",
    )

    # collapsed gradients
    plot_metric_group(
        history,
        metric_names=[
            "train/grad_backbone_nan_tensors",
            "train/grad_student_nan_tensors",
            "train/grad_teacher_nan_tensors",
            "train/grad_backbone_inf_tensors",
            "train/grad_student_inf_tensors",
            "train/grad_teacher_nan_tensors",
        ],
        save_dir=save_dir,
        filename="grad_collapse",
        title="Gradient Collapse",
        ylabel="no. gradients",
    )

    # embedding statistics
    plot_metric_group(
        history,
        metric_names=[
            "train/emb_std",
            "train/student_std",
            "train/teacher_std",
        ],
        save_dir=save_dir,
        filename="embedding_std",
        title="Embedding Std",
        ylabel="std",
    )

    plot_metric_group(
        history,
        metric_names=[
            "train/emb_mean",
            "train/student_mean",
            "train/teacher_mean",
        ],
        save_dir=save_dir,
        filename="embedding_mean",
        title="Embedding Mean",
        ylabel="mean",
    )

    # logits / label density
    plot_metric_group(
        history,
        metric_names=[
            "train/label_positive_fraction",
            "train/logit_mean",
            "train/logit_max",
            "train/top1_top2_margin",
        ],
        save_dir=save_dir,
        filename="assignment_stats",
        title="Assignment / Similarity Stats",
        ylabel="value",
    )

    # prototype diversity
    plot_metric_group(
        history,
        metric_names=[
            "train/base_cos_offdiag_mean",
            "train/base_cos_offdiag_std",
        ],
        save_dir=save_dir,
        filename="prototype_similarity",
        title="Prototype Cosine Similarity",
        ylabel="cosine similarity",
    )


def plot_gradient_histograms(
    grad_hist_data: Dict[str, np.ndarray],
    epoch: int,
    out_dir: str,
    bins: int = 80,
    log_y: bool = True,
    fixed_x_range: Optional[Tuple[float, float]] = None,
    fallback_clip_percentile: float = 99.5,
):
    """
    Plot one histogram per monitored parameter tensor.

    If fixed_x_range is provided, ALL histograms use exactly that x-range.
    This makes plots directly comparable across layers and across epochs.

    Args:
        grad_hist_data: name -> 1D numpy array of sampled gradients
        fixed_x_range: (x_min, x_max) shared by all subplots and all epochs
        fallback_clip_percentile: only used if fixed_x_range is None
    """
    if not grad_hist_data:
        return

    # determine shared x-range
    x_min, x_max = fixed_x_range
    # if fixed_x_range is not None:
    #     x_min, x_max = fixed_x_range
    # else:
    #     all_vals = []
    #     for arr in grad_hist_data.values():
    #         if arr.size == 0:
    #             continue
    #         finite = arr[np.isfinite(arr)]
    #         if finite.size > 0:
    #             all_vals.append(finite)
    #
    #     if len(all_vals) == 0:
    #         return
    #
    #     all_vals = np.concatenate(all_vals, axis=0)
    #     abs_bound = np.percentile(np.abs(all_vals), fallback_clip_percentile)
    #     if abs_bound <= 0:
    #         abs_bound = max(np.max(np.abs(all_vals)), 1e-8)
    #     x_min, x_max = -abs_bound, abs_bound

    n = len(grad_hist_data)
    ncols = 4
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for ax, (name, arr) in zip(axes, grad_hist_data.items()):
        if arr.size == 0:
            ax.set_title(f"{name}\n(no finite grads)")
            ax.axis("off")
            continue

        finite_arr = arr[np.isfinite(arr)]
        if finite_arr.size == 0:
            ax.set_title(f"{name}\n(no finite grads)")
            ax.axis("off")
            continue

        clipped = finite_arr[(finite_arr >= x_min) & (finite_arr <= x_max)]
        if clipped.size == 0:
            clipped = finite_arr

        ax.hist(clipped, bins=bins, range=(x_min, x_max))
        ax.set_title(name)
        ax.set_xlabel("gradient value")
        ax.set_ylabel("count")
        ax.set_xlim(x_min, x_max)

        if log_y:
            ax.set_yscale("log")

    for j in range(len(grad_hist_data), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Gradient Histograms - Epoch {epoch} | fixed x-range [{x_min:.2e}, {x_max:.2e}]",
        fontsize=16
    )
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"grad_hist_epoch_{epoch:03d}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gradient_histograms_paginated(
    grad_hist_data: Dict[str, np.ndarray],
    epoch: int,
    out_dir: str,
    bins: int = 80,
    log_y: bool = True,
    fixed_x_range: Optional[Tuple[float, float]] = None,
    fallback_clip_percentile: float = 99.5,
    plots_per_page: int = 12,
    ncols: int = 3,
):
    """
    Plot gradient histograms across multiple pages so many layers stay readable.
    All pages use the same x-range.
    """
    if not grad_hist_data:
        return

    items = list(grad_hist_data.items())

    # determine shared x-range
    if fixed_x_range is not None:
        x_min, x_max = fixed_x_range
    else:
        all_vals = []
        for _, arr in items:
            if arr.size == 0:
                continue
            finite = arr[np.isfinite(arr)]
            if finite.size > 0:
                all_vals.append(finite)

        if len(all_vals) == 0:
            return

        all_vals = np.concatenate(all_vals, axis=0)
        abs_bound = np.percentile(np.abs(all_vals), fallback_clip_percentile)
        if abs_bound <= 0:
            abs_bound = max(np.max(np.abs(all_vals)), 1e-8)
        x_min, x_max = -abs_bound, abs_bound

    os.makedirs(out_dir, exist_ok=True)

    num_pages = math.ceil(len(items) / plots_per_page)

    for page_idx in range(num_pages):
        start = page_idx * plots_per_page
        end = min((page_idx + 1) * plots_per_page, len(items))
        page_items = items[start:end]

        n_this = len(page_items)
        nrows = math.ceil(n_this / ncols)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(18, 4.5 * nrows),
        )
        axes = np.array(axes).reshape(-1)

        for ax, (name, arr) in zip(axes, page_items):
            if arr.size == 0:
                ax.set_title(f"{name}\n(no finite grads)")
                ax.axis("off")
                continue

            finite_arr = arr[np.isfinite(arr)]
            if finite_arr.size == 0:
                ax.set_title(f"{name}\n(no finite grads)")
                ax.axis("off")
                continue

            clipped = finite_arr[(finite_arr >= x_min) & (finite_arr <= x_max)]
            if clipped.size == 0:
                clipped = finite_arr

            ax.hist(clipped, bins=bins, range=(x_min, x_max))
            ax.set_title(name, fontsize=10)
            ax.set_xlabel("gradient value", fontsize=10)
            ax.set_ylabel("count", fontsize=10)
            ax.set_xlim(x_min, x_max)

            ax.tick_params(axis="x", labelsize=9, rotation=45)
            ax.tick_params(axis="y", labelsize=9)

            if log_y:
                ax.set_yscale("log")

        for j in range(len(page_items), len(axes)):
            axes[j].axis("off")

        fig.suptitle(
            f"Gradient Histograms - Epoch {epoch} - Page {page_idx + 1}/{num_pages} | "
            f"fixed x-range [{x_min:.2e}, {x_max:.2e}]",
            fontsize=16
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.subplots_adjust(hspace=0.5, wspace=0.25)

        path = os.path.join(
            out_dir,
            f"grad_hist_epoch_{epoch:03d}_page_{page_idx + 1:02d}.png"
        )
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)