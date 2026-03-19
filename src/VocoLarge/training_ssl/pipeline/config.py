from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Literal

LoadMode = Literal["backbone", "full"]
DeviceStr = Literal["cpu", "cuda"]
FreezeScope = Literal["swin_plus_conv", "swin"]


@dataclass
class Config:
    """
    Central config for VoCo SSL training / debugging.

    """

    # --------------------
    # Data / I/O
    # --------------------
    # Single-image debug / overfit
    overfit_experimnet: bool = False
    overfit_image_path: Optional[str] = "/mnt/data/flaviu/example_pt/30692BF6DB8F95/image.nii.gz"

    # Used for overfit experiment
    # Used
    data_dir: Optional[str] = "/mnt/data/flaviu/rtnation_02_02/"

    out_dir: str = "/processing/flaviu/ssl_training/10_epochs_lr_5e3"

    # --------------------
    # Reproducibility / runtime
    # --------------------
    device: DeviceStr = "cuda"
    amp: bool = True
    local_rank: int = 0  # reserved for DDP later

    # --------------------
    # Dataloader (used by build_dataloader)
    # --------------------
    batch_size: int = 4 # used in overfit experiment
    train_batch_size: int = 4 # used in actual training
    val_batch_size: int = 4
    test_batch_size: int = 4
    shuffle: bool = True
    num_workers: int = 0
    train_ids_path: str = "../training_data/train_ids.txt"
    val_ids_path: str = "../training_data/val_ids.txt"
    test_ids_path: str = "../training_data/test_ids.txt"
    cache_dir: str = "/mnt/data/flaviu/ssl_training/cache_dir"

    # --------------------
    # Transforms / augmentation
    # --------------------
    # If True, VoCoAugmentation(aug=False)
    no_aug: bool = False

    # Chest transform geometry (used by data_trans.get_chest_trans(voco_args))
    roi_x: int = 192
    roi_y: int = 192
    roi_z: int = 64

    # --------------------
    # Model (VoCoHead / Swin backbone knobs)
    # --------------------
    in_channels: int = 1
    feature_size: int = 48
    dropout_path_rate: float = 0.0
    use_checkpoint: bool = True
    spatial_dims: int = 3

    # Critical for heatmaps / logits shape: number of queries (sw_s)
    sw_batch_size: int = 1

    # --------------------
    # Checkpoint loading (ckpt.py)
    # --------------------
    voco_ckpt_path: Optional[str] = "/processing/flaviu/pretrained/VoCo_B_SSL_head.pt"
    load_mode: LoadMode = "backbone"   # "backbone" or "full"
    freeze_scope: FreezeScope = "swin_plus_conv"

    # --------------------
    # Training loop
    # --------------------
    epochs: int = 10
    lr: float = 5e-3
    optimizer: str = "sgd"
    weight_decay: float = 1e-4
    momentum: float = 0.9
    save_every: int = 5
    eval_every: int = 2

    # --------------------
    # Visualization / debug output (viz.py usage)
    # Mostly used for overfit experiment, but can be used elsewhere too
    # --------------------
    save_visuals: bool = True
    max_queries_vis: int = 10
    slices_per_vol_vis: int = 3

    # --------------------
    # Convenience properties
    # --------------------
    @property
    def roi_size(self) -> Tuple[int, int, int]:
        return (self.roi_x, self.roi_y, self.roi_z)

    def to_dict(self):
        return asdict(self)