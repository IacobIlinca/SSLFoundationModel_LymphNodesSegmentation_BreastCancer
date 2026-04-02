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

    train_epoch: List[int]
    train_loss: List[float]
    val_epoch: List[int]
    val_loss: List[float]
    val_dice: List[float]
    val_hd95: List[float]

    def __init__(self):
      self.train_epoch = []
      self.val_epoch = []
      self.train_loss = []
      self.val_loss = []
      self.val_dice = []
      self.val_hd95 = []

    def add(self, epoch: int, train_loss: float, val_loss: float, val_dice: float, val_hd95: float):
      self.train_epoch.append(int(epoch))
      self.val_epoch.append(int(epoch))
      self.train_loss.append(float(train_loss))
      self.val_loss.append(float(val_loss))
      self.val_dice.append(float(val_dice))
      self.val_hd95.append(float(val_hd95))

    def add_train_loss(self, epoch: int, train_loss: float):
      self.train_epoch.append(int(epoch))
      self.train_loss.append(float(train_loss))

    def add_val_loss(self, epoch: int, val_loss: float):
      self.val_epoch.append(int(epoch))
      self.val_loss.append(float(val_loss))

    def add_val_dice(self, val_dice: float):
      self.val_dice.append(float(val_dice))

    def add_val_hd95(self, val_hd95: float):
      self.val_hd95.append(float(val_hd95))


    def to_dict(self) -> Dict:
      return asdict(self)