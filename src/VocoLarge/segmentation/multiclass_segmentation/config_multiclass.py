from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple


@dataclass
class ConfigMulticlass:
    """
    Config for binary lymph-node segmentation training.

    This config is meant to work with the shared:
      - build_samples(cfg)
      - get_transforms(cfg)

    by setting:
      task_mode = "train_binary"
    """

    # ---- Task / data ----
    root_dir: str = "/mnt/data/ilinca/structured_cases_14_16/"
    val_fraction: float = 0.2
    multiclass_masks_csv_path: str = "../masks/binary_mask_selection_audit/required_multiclass_nodes_summary2.csv"
    multiclass_label = [
        #"level1",
        "level2",
        #"level3",
        #"level4",
        #"imn",
        #"interpectoral",
    ]
    class_to_index = {
        #"level1": 1,
        "level2": 1,
        #"level3": 3,
        #"level4": 4,
        #"imn": 5,
        #"interpectoral": 6,
    }
    class_to_csv_column = {
        #"level1": "level1_masks",
        "level2": "level2_masks",
        #"level3": "level3_masks",
        #"level4": "level4_masks",
        #"imn": "imn_masks",
        #"interpectoral": "interpectoral_masks",
    }
    class_crop_ratios = [0.0, 1.0] # background + 6 classes

    # 1 class for background will be added when the model is computed
    # 1 output channel with sigmoid activation
    num_classes: int = 1

    # ---- Patch training / inference ----
    roi_size: Tuple[int, int, int] = (192, 192, 32)
    num_samples_per_volume: int = 1
    batch_size: int = 4 #train
    val_batch_size: int = 4
    test_batch_size: int = 1

    # ---- Preprocessing ----
    axcodes: str = "RAS"
    do_resample: bool = False
    target_spacing: Tuple[float, float, float] = (1.25, 1.25, 5.0)
    light_aug: bool = False

    # ---- Intensity normalization ----
    # Options:
    #   "ct_clip_zscore"
    #   "ct_clip_0_1"
    #   "zscore"
    norm_mode: str = "ct_clip_0_1"
    ct_clip: Tuple[float, float] = (-1000.0, 500.0)

    # ---- Training ----
    seed: int = 0
    device: str = "cuda"
    epochs: int = 100
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 1e-5
    amp: bool = True
    num_workers: int = 2
    log_every: int = 3

    # For loss function
    class_weight_for_loss = [1.0, 1.0]
    ce_weight: float = 1.0
    dice_weight: float = 1.0

    # ---- Linear probing specifics ----
    voco_ckpt_path: str = "/processing/flaviu/pretrained/SwinUnet_from_VoCo_B.pt"
    feature_size: int = 48
    freeze_scope: str = "all_beside_last_decoder"

    # ---- Debug / convenience ----
    save_dir: str = "/processing/flaviu/multiclass_segmentation_runs/run_11_segment_only_l2_full_run"
    save_visuals: bool = True
    visuals_case_indices: tuple[int, int, int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    visuals_slices: Tuple[int, int, int] = (10, 25, 40, 55, 70, 95)

    # ---- Split data -----
    train_ids_path: str = "../training_data_ids/train_ids_level1_optional.txt"
    val_ids_path: str = "../training_data_ids/val_ids_level1_optional.txt"
    test_ids_path: str = "../training_data_ids/test_ids_level1_optional.txt"
    cache_dir: str = "/processing/flaviu/multiclass_segmentation_runs/cache"
    shuffle: bool = True

    # ---- Validation mode ----
    fast_val: bool = True
    fast_val_num_samples_per_volume: int = 1

    # ---- LR scheduler ----
    use_scheduler: bool = True
    scheduler_type: str = "cosine"

    min_lr: float = 1e-6  # final LR at the end of training

    def to_dict(self):
        return asdict(self)