import os
import numpy as np
import matplotlib.pyplot as plt


def save_overlay_png(image_1ch: np.ndarray, label_int: np.ndarray, pred_int: np.ndarray,
                     out_path: str, slice_idx: int) -> None:
    """
    OPTIONAL.

    Saves a quick overlay PNG for debugging/figures.
    image_1ch: (H,W,D)
    label_int/pred_int: (H,W,D) integer masks
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    img = image_1ch[:, :, slice_idx]
    gt = label_int[:, :, slice_idx]
    pr = pred_int[:, :, slice_idx]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("GT")
    plt.imshow(img, cmap="gray")
    plt.imshow(gt, alpha=0.35)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Pred")
    plt.imshow(img, cmap="gray")
    plt.imshow(pr, alpha=0.35)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()