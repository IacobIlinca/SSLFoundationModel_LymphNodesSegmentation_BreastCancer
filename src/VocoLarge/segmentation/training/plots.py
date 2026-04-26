import os
import matplotlib.pyplot as plt

from src.VocoLarge.segmentation.training.history import History


def plot_loss_curves(history: History, save_path: str, title: str = "Loss Curves") -> None:
    """
    Saves a single plot with train loss and val loss across epochs.

    Expected keys:
        history.epochs["train"]
        history.epochs["val"]
        history.metrics["train_loss"]
        history.metrics["val_loss"]
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_epochs = history.epochs.get("train", [])
    val_epochs = history.epochs.get("val", [])

    train_loss = history.metrics.get("train_loss", [])
    val_loss = history.metrics.get("val_loss", [])

    plt.figure()
    plt.title(title)

    if len(train_loss) > 0:
        plt.plot(train_epochs[:len(train_loss)], train_loss, label="train_loss")

    if len(val_loss) > 0:
        plt.plot(val_epochs[:len(val_loss)], val_loss, label="val_loss")

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_metric_curves(history: History, save_path: str, title: str = "Validation Metrics") -> None:
    """
    Saves one plot with:
      - Left Y-axis  -> Dice
      - Right Y-axis -> HD95

    Expected keys:
        history.epochs["val"]
        history.metrics["val_dice"]
        history.metrics["val_hd95"]
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    val_epochs = history.epochs.get("val", [])
    val_dice = history.metrics.get("val_dice", [])
    val_hd95 = history.metrics.get("val_hd95", [])

    fig, ax1 = plt.subplots()

    ax1.set_title(title)
    ax1.set_xlabel("epoch")

    # ----- Left axis: Dice -----
    if len(val_dice) > 0:
        ax1.set_ylabel("Dice", color="tab:blue")
        ax1.plot(
            val_epochs[:len(val_dice)],
            val_dice,
            color="tab:blue",
            label="val_dice",
        )
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.set_ylim(0.0, 1.0)

    # ----- Right axis: HD95 -----
    ax2 = ax1.twinx()

    if len(val_hd95) > 0:
        ax2.set_ylabel("HD95", color="tab:red")
        ax2.plot(
            val_epochs[:len(val_hd95)],
            val_hd95,
            color="tab:red",
            label="val_hd95",
        )
        ax2.tick_params(axis="y", labelcolor="tab:red")

        max_hd = max(val_hd95)
        ax2.set_ylim(0.0, max(1.0, max_hd * 1.1))

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_train_loss_components(
    history: History,
    save_path: str,
    title: str = "Training Loss Components",
) -> None:
    """
    Saves a plot with:
      - total train loss
      - Dice loss
      - BCE loss

    Expected keys:
        history.epochs["train"]
        history.metrics["train_loss"]
        history.metrics["train_dice_loss"]
        history.metrics["train_bce_loss"]
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_epochs = history.epochs.get("train", [])

    train_loss = history.metrics.get("train_loss", [])
    dice_loss = history.metrics.get("train_dice_loss", [])
    bce_loss = history.metrics.get("train_bce_loss", [])
    sf_loss = history.metrics.get("train_surface_loss", [])

    plt.figure()
    plt.title(title)

    if len(train_loss) > 0:
        plt.plot(
            train_epochs[:len(train_loss)],
            train_loss,
            label="train_loss",
        )

    if len(dice_loss) > 0:
        plt.plot(
            train_epochs[:len(dice_loss)],
            dice_loss,
            label="train_dice_loss",
        )

    if len(bce_loss) > 0:
        plt.plot(
            train_epochs[:len(bce_loss)],
            bce_loss,
            label="train_bce_loss",
        )
    if len(sf_loss) > 0:
        plt.plot(
            train_epochs[:len(sf_loss)],
            sf_loss,
            label="train_surface_loss",
        )

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()