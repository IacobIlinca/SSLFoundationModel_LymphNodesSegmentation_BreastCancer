from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple


@dataclass
class ConfigBinary:
    """
    Config for binary lymph-node segmentation training.

    This config is meant to work with the shared:
      - build_samples(cfg)
      - get_transforms(cfg)

    by setting:
      task_mode = "train_binary"
    """

    # ---- Task / data ----
    task_mode: str = "train_binary"
    root_dir: str = "/mnt/data/ilinca/structured_cases_14_16/"
    val_fraction: float = 0.2
    lymph_terms_json: str = "src/VocoLarge/segmentation/masks/lymph_terms.json"

    # ---- Segmentation task ----
    # Number of foreground classes, excluding background.
    # Binary task = lymph node vs background => 1 foreground class.
    num_classes: int = 1

    # These are not used in binary mode, but kept as None for compatibility
    # with shared code that may check for their existence.
    mask_keys: Optional[List[str]] = None
    mask_key_to_class_index: Optional[dict] = None

    # ---- Patch training / inference ----
    roi_size: Tuple[int, int, int] = (192, 192, 64)
    num_samples_per_volume: int = 1
    batch_size: int = 1
    val_overlap: float = 0.5
    sw_batch_size: int = 2

    # ---- Preprocessing ----
    axcodes: str = "RAS"
    do_resample: bool = False
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 2.0)
    light_aug: bool = False

    # ---- Intensity normalization ----
    # Options:
    #   "ct_clip_zscore"
    #   "ct_clip_0_1"
    #   "zscore"
    norm_mode: str = "ct_clip_zscore"
    ct_clip: Tuple[float, float] = (-1000.0, 500.0)

    # ---- Training ----
    seed: int = 0
    device: str = "cuda"
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-5
    amp: bool = True
    num_workers: int = 8
    log_every: int = 1

    # For softmax-style binary setup with background + foreground
    class_weight_for_loss: List[float] = field(default_factory=lambda: [0.1, 1.0])

    # ---- Inference / metrics ----
    pred_threshold: float = 0.5

    # ---- Linear probing specifics ----
    voco_ckpt_path: str = "/processing/flaviu/pretrained/VoCo_B_SSL_head.pt"
    feature_size: int = 48
    freeze_scope: str = "swin_plus_conv"

    # ---- Weight loading safety ----
    strict_load: bool = True
    strict_load_threshold: float = 0.95

    # ---- Debug / convenience ----
    save_dir: str = "/processing/flaviu/binary_segmentation_runs/run_02"
    overfit_case_id: Optional[str] = None
    save_visuals: bool = True
    visuals_case_index: int = 0
    visuals_slices: Tuple[int, int, int] = (20, 30, 40)

    # ---- Split data -----
    train_ids_path: str = "../training_data/train_ids.txt"
    val_ids_path: str = "../training_data/val_ids.txt"
    test_ids_path: str = "../training_data/test_ids.txt"
    cache_dir: str = "/processing/flaviu/binary_segmentation_cache"
    shuffle: bool = True

    # log file for patient with no lymph masks available
    no_lymph_patients_log_file: str = "src/VocoLarge/segmentation/training_data_ids/binary_missing_lymph_cases.txt"

    # ---- Validation mode ----
    fast_val: bool = True
    fast_val_num_samples_per_volume: int = 1
    compute_hd95: bool = False

    def to_dict(self):
        return asdict(self)