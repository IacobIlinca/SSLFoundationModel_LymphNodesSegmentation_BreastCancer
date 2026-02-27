from dataclasses import dataclass, asdict
from typing import Optional, Tuple


@dataclass
class Config:
    """
    Central config for the linear-probe segmentation experiment.

    REQUIRED fields:
      - num_classes, roi_size, voco_ckpt_path, save_dir

    OPTIONAL fields:
      - do_resample/target_spacing (recommended later for consistency)
      - overfit_case_id (debug only)
      - save_visuals (debug/figures)
    """

    # ---- Task ----
    num_classes: int = 3  # K (excluding background). Model out_channels = K+1.
    mask_keys = ["mask1", "mask2", "mask3"]
    mask_key_to_class_index = {"mask1": 1, "mask2": 2, "mask3": 3}

    # ---- Patch training / inference ----
    roi_size: Tuple[int, int, int] = (192, 192, 48)  # must fit Z=~60
    num_samples_per_volume: int = 1                  # patches per volume
    batch_size: int = 1                               # volumes per batch (patching happens inside transform)
    val_overlap: float = 0.5
    sw_batch_size: int = 2

    # ---- Preprocessing ----
    axcodes: str = "RAS"  # standardize orientation
    do_resample: bool = False
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 2.0)
    light_aug: bool = False

    # --- Intensity normalization ---
    # Options:
    #   "ct_clip_zscore": clip to ct_clip then z-score normalize
    #   "ct_clip_0_1":    clip+scale to [0,1] (old behavior)
    #   "zscore":         z-score without clipping (not recommended for your data)
    norm_mode: str = "ct_clip_zscore"
    ct_clip: Tuple[float, float] = (-1000.0, 500.0)

    # ---- Training ----
    seed: int = 0
    device: str = "cuda"
    epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 1e-5
    amp: bool = True
    num_workers: int = 8
    log_every: int = 10

    # ---- Linear probing specifics ----
    # We load VoCo into Swin encoder and freeze the encoder.
    voco_ckpt_path: str = "/processing/flaviu/pretrained/VoCo_B_SSL_head.pt"
    feature_size: int = 48  # IMPORTANT: must match VoCo variant (48/96/192 etc). Change if load report says mismatch.
    freeze_scope: str = "swin"  # "swin" or "swin_plus_conv" (stricter)

    # Weight loading safety gate
    strict_load: bool = True
    strict_load_threshold: float = 0.95  # require encoder tensors matched, else crash

    # ---- Debug / convenience ----
    save_dir: str = "/processing/flaviu/runs_swinunetr_overfit_probe"
    overfit_case_id: Optional[str] = None  # set to one case_id to overfit; DEBUG ONLY
    save_visuals: bool = True              # OPTIONAL
    visuals_case_index: int = 0            # OPTIONAL
    visuals_slices: Tuple[int, int, int] = (20, 30, 40)  # OPTIONAL

    def to_dict(self):
        return asdict(self)