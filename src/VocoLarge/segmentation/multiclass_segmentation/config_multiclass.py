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
    root_dir: str = "/mnt/data/ilinca/Workshop_data_0505206/workshop_all_data"
    val_fraction: float = 0.2
    multiclass_masks_csv_path: str = "/home/flaviu/Local/SSLFoundationModel_LymphNodesSegmentation_BreastCancer/src/VocoLarge/segmentation/masks/workshop_data_all/required_multiclass_nodes_summary_workshop.csv" #!!!!!
    multiclass_label = [
        #"level1",
        "level2",
        "level3",
        "level4",
        #"imn",
        "interpectoral",
    ]
    class_to_index = {
        #"level1": 1,
        "level2": 1,
        "level3": 2,
        "level4": 3,
        # "imn": 1,
        "interpectoral": 4,
    }
    class_to_csv_column = {
        #"level1": "level1_masks",
        "level2": "level2_masks",
        "level3": "level3_masks",
        "level4": "level4_masks",
        #"imn": "imn_masks",
        "interpectoral": "interpectoral_masks",
    }
    class_crop_ratios = [0.0, 1.0, 1.0, 1.0, 1.0] # background + 6 classes

    # 1 class for background will be added when the model is computed
    # 1 output channel with sigmoid activation
    num_classes: int = 4

    # ---- Patch training / inference ----
    roi_size: Tuple[int, int, int] = (512, 512, 64)
    num_samples_per_volume: int = 1
    batch_size: int = 4 #train
    val_batch_size: int = 2
    test_batch_size: int = 1

    # ---- Preprocessing ----
    axcodes: str = "RAS"
    do_resample: bool = False
    target_spacing: Tuple[float, float, float] = (1.25, 1.25, 3.0)
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
    lr: float = 1e-4 #!!!!!!
    momentum: float = 0.9
    weight_decay: float = 1e-5
    amp: bool = True
    num_workers: int = 2
    log_every: int = 5

    # For loss function
    class_weight_for_loss = [1.0, 1.0, 1.0, 1.0, 1.0]
    ce_weight: float = 1.0
    dice_weight: float = 1.0

    # ---- Linear probing specifics ----
    voco_ckpt_path: str = "/processing/flaviu/pretrained/VoCo_B_SSL_head_domain_pretrained.pt" #!!!!!!
    feature_size: int = 48
    freeze_scope: str = "swin_plus_conv" #!!!!!!

    # ---- Debug / convenience ----
    save_dir: str = "/processing/flaviu/multiclass_segmentation_runs/workshop/run_15_mixed_crops_from_ssl_domain"
    save_visuals: bool = True
    visuals_case_indices: tuple[int, int, int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    visuals_slices: Tuple[int, int, int] = (10, 20, 30, 40, 50, 60)

    # ---- Split data -----
    train_ids_path: str = "/home/flaviu/Local/SSLFoundationModel_LymphNodesSegmentation_BreastCancer/src/VocoLarge/segmentation/training_data_ids/training_workshop_data_ids/train_ids_workshop.txt"
    val_ids_path: str = "/home/flaviu/Local/SSLFoundationModel_LymphNodesSegmentation_BreastCancer/src/VocoLarge/segmentation/training_data_ids/training_workshop_data_ids/val_ids_workshop.txt"
    test_ids_path: str = "/home/flaviu/Local/SSLFoundationModel_LymphNodesSegmentation_BreastCancer/src/VocoLarge/segmentation/training_data_ids/training_workshop_data_ids/val_ids_workshop.txt"
    cache_dir: str = "/mnt/data/flaviu/multiclass_segmentation_runs/cache/workshop_multiclass_wo_l1_imn"
    shuffle: bool = True

    # ---- Validation mode ----
    fast_val: bool = True
    fast_val_num_samples_per_volume: int = 1

    # ---- LR scheduler ----
    use_scheduler: bool = True
    scheduler_type: str = "cosine"

    min_lr: float = 1e-7  # final LR at the end of training

    def to_dict(self):
        return asdict(self)