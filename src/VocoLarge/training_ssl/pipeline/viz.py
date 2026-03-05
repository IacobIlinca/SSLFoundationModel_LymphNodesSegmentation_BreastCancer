import os
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


def save_voco_debug_vis(
    img, crops, labels,
    out_dir="debug_vis",
    prefix="case",
    max_queries=8,
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