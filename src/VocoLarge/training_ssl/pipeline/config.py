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

    # Folder-of-cases SSL (used by find_case_images + NiftiListDataset)
    # (keep it even if you don't use it yet—this config is meant to scale to the full pipeline)
    data_dir: Optional[str] = "/mnt/data/flaviu/example_pt/"

    out_dir: str = "/processing/flaviu/overfitting/runs_overfit_case6_freeze_encoder"

    # --------------------
    # Reproducibility / runtime
    # --------------------
    device: DeviceStr = "cuda"
    amp: bool = True
    local_rank: int = 0  # reserved for DDP later

    # --------------------
    # Dataloader (used by build_dataloader)
    # NOTE: for VoCoAugmentation outputs you typically keep batch_size=1 unless you custom-collate
    # --------------------
    batch_size: int = 4
    shuffle: bool = False
    num_workers: int = 0

    # --------------------
    # Transforms / augmentation
    # --------------------
    # If True, VoCoAugmentation(aug=False)
    no_aug: bool = True

    # Chest transform geometry (used by data_trans.get_chest_trans(voco_args))
    # ROI for crops/queries
    roi_x: int = 96
    roi_y: int = 96
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
    # Training loop (overfit_one_image.py)
    # --------------------
    steps: int = 28
    lr: float = 5e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    save_every: int = 4

    # --------------------
    # Visualization / debug output (viz.py usage)
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