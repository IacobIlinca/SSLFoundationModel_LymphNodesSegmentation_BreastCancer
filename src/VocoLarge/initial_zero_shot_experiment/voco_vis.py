import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.VocoLarge.third_party_voco_large.models.voco_head import online_assign


def save_heatmap(mat, title, path, xlabel="crop id (0..8)", ylabel="query id"):
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


@torch.no_grad()
def get_voco_logits(model, img, crops, labels):
    """
    Reproduce the intra-volume logits used by VoCoHead.forward:
      logits = online_assign(x_student, bases_teacher)
    Returns:
      logits: (sw_size, 9) for batch item 0
      targets: (sw_size, 9) labels[0]
    """
    model.eval()

    # Ensure tensors
    img = img.as_tensor() if hasattr(img, "as_tensor") else img
    crops = crops.as_tensor() if hasattr(crops, "as_tensor") else crops

    device = next(model.parameters()).device
    img = img.to(device)
    crops = crops.to(device)
    labels = labels.to(device)

    batch_size = labels.size(0)
    total_size = img.size(0)
    sw_size = total_size // batch_size
    bases_num = crops.size(0) // batch_size  # should be 9

    # forward backbone on concatenated inputs
    inputs = torch.cat([img, crops], dim=0)
    embeddings = model.backbone(inputs)

    # student uses dropout-augmented embeddings
    aug_embeddings = torch.nn.Dropout1d(0.2)(embeddings)
    student = model.student(aug_embeddings)

    # teacher uses raw embeddings
    teacher = model.teacher(embeddings)

    x_student, bases_student = student[:total_size], student[total_size:]
    x_teacher, bases_teacher = teacher[:total_size], teacher[total_size:]

    # take batch item 0
    i = 0
    x_stu = x_student[i * sw_size:(i + 1) * sw_size]                  # (sw_size, dim)
    bases_tea = bases_teacher[i * bases_num:(i + 1) * bases_num]       # (9, dim)

    # predicted similarities, relu’ed cosine sim
    logits = online_assign(x_stu, bases_tea)                           # (sw_size, 9)

    targets = labels[i]                                                # (sw_size, 9)
    return logits.detach().cpu(), targets.detach().cpu()

def _to_numpy_3d(x):
    """
    Accepts (1, D, H, W) or (D, H, W) torch tensor / MetaTensor and returns (D, H, W) float32 numpy.
    """
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
    """
    Pick n slice indices across the volume (avoid extreme edges).
    """
    if depth <= n:
        return list(range(depth))
    lo = max(0, int(depth * 0.15))
    hi = min(depth - 1, int(depth * 0.85))
    idxs = np.linspace(lo, hi, n).round().astype(int)
    return idxs.tolist()


def save_voco_debug_vis(
    img,            # (sw_s, 1, D, H, W)
    crops,          # (9,    1, D, H, W)
    labels,         # (1, sw_s, 9) or (sw_s, 9)
    out_dir="debug_vis",
    prefix="case",
    max_queries=8,
    slices_per_vol=6,
):
    """
    Saves:
      - {prefix}_query_grid.png
      - {prefix}_crops_grid.png
      - {prefix}_labels_heatmap.png
      - {prefix}_bestcrop_per_query.png
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---- normalize labels shape ----
    if torch.is_tensor(labels):
        lab = labels.detach().float().cpu()
        if lab.ndim == 3:
            lab = lab[0]
    else:
        lab = torch.as_tensor(labels).float()
        if lab.ndim == 3:
            lab = lab[0]
    # lab: (sw_s, 9)
    sw_s = lab.shape[0]
    n_queries = min(sw_s, max_queries)

    # ---- convert query vols ----
    # img: (sw_s, 1, D, H, W)
    if hasattr(img, "as_tensor"):
        img_t = img.as_tensor()
    else:
        img_t = img
    img_t = img_t.detach().cpu()
    if img_t.ndim != 5:
        raise ValueError(f"Expected img shape (sw_s,1,D,H,W), got {tuple(img_t.shape)}")

    # ---- convert crops vols ----
    if hasattr(crops, "as_tensor"):
        crops_t = crops.as_tensor()
    else:
        crops_t = crops
    crops_t = crops_t.detach().cpu()
    if crops_t.ndim != 5 or crops_t.shape[0] != 9:
        raise ValueError(f"Expected crops shape (9,1,D,H,W), got {tuple(crops_t.shape)}")

    D = int(img_t.shape[2])
    slice_ids = _pick_slices(D, n=slices_per_vol)

    # ============================================================
    # 1) QUERY GRID: show n_queries x slices_per_vol axial slices
    # ============================================================
    fig = plt.figure(figsize=(2.4 * slices_per_vol, 2.4 * n_queries))
    for qi in range(n_queries):
        vol = _to_numpy_3d(img_t[qi])  # (D,H,W)
        vol = _norm01(vol)
        for sj, z in enumerate(slice_ids):
            ax = plt.subplot(n_queries, slices_per_vol, qi * slices_per_vol + sj + 1)
            ax.imshow(vol[z], cmap="gray")
            ax.set_axis_off()
            if sj == 0:
                ax.set_title(f"q{qi}", fontsize=10)
    plt.tight_layout()
    p = os.path.join(out_dir, f"{prefix}_query_grid.png")
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ============================================================
    # 2) CROPS GRID: 9 crops x slices_per_vol axial slices
    # ============================================================
    fig = plt.figure(figsize=(2.4 * slices_per_vol, 2.4 * 9))
    for ci in range(9):
        vol = _to_numpy_3d(crops_t[ci])  # (D,H,W)
        vol = _norm01(vol)
        for sj, z in enumerate(slice_ids):
            ax = plt.subplot(9, slices_per_vol, ci * slices_per_vol + sj + 1)
            ax.imshow(vol[z], cmap="gray")
            ax.set_axis_off()
            if sj == 0:
                ax.set_title(f"c{ci}", fontsize=10)
    plt.tight_layout()
    p = os.path.join(out_dir, f"{prefix}_crops_grid.png")
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ============================================================
    # 3) LABELS HEATMAP: (sw_s x 9)
    # ============================================================
    fig = plt.figure(figsize=(8, max(3, 0.35 * sw_s)))
    ax = plt.gca()
    im = ax.imshow(lab.numpy(), aspect="auto")
    ax.set_xlabel("crop id (0..8)")
    ax.set_ylabel("query patch id")
    ax.set_title("VoCo labels / targets (query vs 9 crops)")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()
    p = os.path.join(out_dir, f"{prefix}_labels_heatmap.png")
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ============================================================
    # 4) BEST-CROP PER QUERY: show query slice + best crop slice
    # ============================================================
    best = torch.argmax(lab, dim=1)  # (sw_s,)
    # pick a single representative slice index (middle of the volume)
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
    p = os.path.join(out_dir, f"{prefix}_bestcrop_per_query.png")
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"[vis] saved debug images to: {os.path.abspath(out_dir)}")

