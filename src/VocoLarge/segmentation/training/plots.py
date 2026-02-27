import os
from typing import Optional
import matplotlib.pyplot as plt

from src.VocoLarge.segmentation.training.history import History


def plot_loss_curves(history: History, save_path: str, title: str = "Loss Curves") -> None:
    """
    REQUIRED for your request.

    Saves a single plot with train loss and val loss across epochs.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure()
    plt.title(title)
    plt.plot(history.epoch, history.train_loss, label="train_loss")
    plt.plot(history.epoch, history.val_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_metric_curves(history: History, save_path: str, title: str = "Validation Metrics") -> None:
    """
    REQUIRED (updated).

    Saves one plot with:
      - Left Y-axis  -> Dice
      - Right Y-axis -> HD95

    This avoids scale distortion between Dice (~0-1) and HD95 (often 0-20+).
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax1 = plt.subplots()

    # ----- Left axis: Dice -----
    ax1.set_title(title)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Dice", color="tab:blue")
    ax1.plot(history.epoch, history.val_dice, color="tab:blue", label="val_dice")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(0.0, 1.0)  # Dice range

    # ----- Right axis: HD95 -----
    ax2 = ax1.twinx()
    ax2.set_ylabel("HD95", color="tab:red")
    ax2.plot(history.epoch, history.val_hd95, color="tab:red", label="val_hd95")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Optional: auto-scale HD95 nicely
    if len(history.val_hd95) > 0:
        max_hd = max(history.val_hd95)
        ax2.set_ylim(0.0, max(1.0, max_hd * 1.1))

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)