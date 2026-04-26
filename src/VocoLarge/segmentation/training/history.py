from dataclasses import dataclass, field
from typing import Dict, List, Any
import json


@dataclass
class History:
    """
    Generic epoch-wise history tracker.

    Stores any scalar metric dynamically, for example:
      - train_loss
      - val_loss
      - val_dice
      - val_hd95
      - dice_loss
      - bce_loss
      - top1_acc
      - top2_acc
      - mean_iou

    Metrics are stored as:
        history.metrics["train_loss"] = [0.8, 0.6, 0.4]
        history.metrics["val_dice"]   = [0.2, 0.35, 0.48]

    Epochs are stored per split:
        history.epochs["train"] = [1, 2, 3]
        history.epochs["val"]   = [1, 2, 3]
    """

    epochs: Dict[str, List[int]] = field(default_factory=dict)
    metrics: Dict[str, List[float]] = field(default_factory=dict)

    def add_metric(self, name: str, value: float):
        """
        Add one scalar value to a metric list.
        """
        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append(float(value))

    def add_epoch(self, split: str, epoch: int):
        """
        Add an epoch index for a split, e.g. train/val/test.
        """
        if split not in self.epochs:
            self.epochs[split] = []

        self.epochs[split].append(int(epoch))

    def add(self, epoch: int, split: str, **metrics: float):
        """
        Add several metrics for one epoch and one split.

        Example:
            history.add(
                epoch=1,
                split="train",
                loss=0.72,
                dice_loss=0.45,
                bce_loss=0.27,
            )

            history.add(
                epoch=1,
                split="val",
                loss=0.68,
                dice=0.41,
                hd95=35.2,
            )

        This stores:
            train_loss
            train_dice_loss
            train_bce_loss
            val_loss
            val_dice
            val_hd95
        """
        self.add_epoch(split, epoch)

        for metric_name, value in metrics.items():
            full_name = f"{split}_{metric_name}"
            self.add_metric(full_name, value)

    def add_train(self, epoch: int, **metrics: float):
        """
        Convenience method for adding training metrics.

        Example:
            history.add_train(
                epoch=1,
                loss=0.7,
                dice_loss=0.4,
                bce_loss=0.3,
            )
        """
        self.add(epoch=epoch, split="train", **metrics)

    def add_val(self, epoch: int, **metrics: float):
        """
        Convenience method for adding validation metrics.

        Example:
            history.add_val(
                epoch=1,
                loss=0.6,
                dice=0.45,
                hd95=32.1,
            )
        """
        self.add(epoch=epoch, split="val", **metrics)

    def get_metric(self, name: str) -> List[float]:
        """
        Return a metric list by name.
        """
        return self.metrics.get(name, [])

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert history to a serializable dictionary.
        """
        return {
            "epochs": self.epochs,
            "metrics": self.metrics,
        }

    def save_json(self, path: str):
        """
        Save history to a JSON file.
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "History":
        """
        Load history from a JSON file.
        """
        with open(path, "r") as f:
            data = json.load(f)

        history = cls()
        history.epochs = data.get("epochs", {})
        history.metrics = data.get("metrics", {})
        return history