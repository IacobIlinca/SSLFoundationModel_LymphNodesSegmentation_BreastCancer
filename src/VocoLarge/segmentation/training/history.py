from dataclasses import dataclass, asdict
from typing import List, Dict


@dataclass
class History:
    """
    Stores epoch-wise scalars for plotting and later analysis.

    REQUIRED fields for your request:
      - train_loss, val_loss, val_dice, val_hd95

    Notes:
      - val_loss is computed on the same val loader using full-volume sliding window inference.
        It's optional conceptually, but REQUIRED if you want train+val loss on the same plot.
    """
    epoch: List[int]
    train_loss: List[float]
    val_loss: List[float]
    val_dice: List[float]
    val_hd95: List[float]

    def __init__(self):
        self.epoch = []
        self.train_loss = []
        self.val_loss = []
        self.val_dice = []
        self.val_hd95 = []

    def add(self, epoch: int, train_loss: float, val_loss: float, val_dice: float, val_hd95: float):
        self.epoch.append(int(epoch))
        self.train_loss.append(float(train_loss))
        self.val_loss.append(float(val_loss))
        self.val_dice.append(float(val_dice))
        self.val_hd95.append(float(val_hd95))

    def to_dict(self) -> Dict:
        return asdict(self)