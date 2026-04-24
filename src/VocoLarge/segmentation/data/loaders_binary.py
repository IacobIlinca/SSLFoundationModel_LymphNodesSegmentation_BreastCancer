import os

from monai.data import PersistentDataset, list_data_collate
from torch.utils.data import DataLoader

from src.VocoLarge.segmentation.config_binary import ConfigBinary
from src.VocoLarge.segmentation.data.transforms import get_transforms_binary
from src.VocoLarge.segmentation.data.data_utils_binary import (
    read_ids_file,
    build_segmentation_files_from_ids,
)
from src.VocoLarge.segmentation.data.data_utils_binary import filter_positive_cases


def build_persistent_dataset(files, transform, cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    return PersistentDataset(
        data=files,
        transform=transform,
        cache_dir=cache_dir,
    )


def build_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device_type: str,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device_type == "cuda"),
        collate_fn=list_data_collate,
    )


def build_all_datasets_and_loaders(cfg: ConfigBinary):
    """
    Builds train / val / test datasets and loaders from split txt files.
    Uses PersistentDataset, aligned as closely as possible with the SSL pipeline.
    """

    train_ids = read_ids_file(cfg.train_ids_path)
    val_ids = read_ids_file(cfg.val_ids_path)
    test_ids = read_ids_file(cfg.test_ids_path)

    print(f"[INFO] train ids: {len(train_ids)}")
    print(f"[INFO] val ids:   {len(val_ids)}")
    print(f"[INFO] test ids:  {len(test_ids)}")

    train_files = build_segmentation_files_from_ids(cfg.root_dir, train_ids)
    val_files = build_segmentation_files_from_ids(cfg.root_dir, val_ids)
    test_files = build_segmentation_files_from_ids(cfg.root_dir, test_ids)

    train_files, skipped_train = filter_positive_cases(
        train_files,
        lymph_terms_json=cfg.lymph_terms_json,
        log_file=cfg.no_lymph_patients_log_file,
    )

    val_files, skipped_val = filter_positive_cases(
        val_files,
        lymph_terms_json=cfg.lymph_terms_json,
        log_file=cfg.no_lymph_patients_log_file,
    )

    test_files, skipped_test = filter_positive_cases(
        test_files,
        lymph_terms_json=cfg.lymph_terms_json,
        log_file=cfg.no_lymph_patients_log_file,
    )

    print(f"[INFO] train files kept: {len(train_files)} | skipped: {len(skipped_train)}")
    print(f"[INFO] val files kept:   {len(val_files)} | skipped: {len(skipped_val)}")
    print(f"[INFO] test files kept:  {len(test_files)} | skipped: {len(skipped_test)}")

    if len(train_files) == 0:
        raise RuntimeError("No training files found.")
    if len(val_files) == 0:
        raise RuntimeError("Validation set is empty.")
    if len(test_files) == 0:
        raise RuntimeError("Test set is empty.")

    train_transform, val_transform = get_transforms_binary(cfg)
    test_transform = val_transform

    train_ds = build_persistent_dataset(
        files=train_files,
        transform=train_transform,
        cache_dir=os.path.join(cfg.cache_dir, "train"),
    )
    val_ds = build_persistent_dataset(
        files=val_files,
        transform=val_transform,
        cache_dir=os.path.join(cfg.cache_dir, "val"),
    )
    test_ds = build_persistent_dataset(
        files=test_files,
        transform=test_transform,
        cache_dir=os.path.join(cfg.cache_dir, "test"),
    )

    train_loader = build_dataloader(
        dataset=train_ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        device_type=cfg.device,
    )
    val_loader = build_dataloader(
        dataset=val_ds,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=0,
        device_type=cfg.device,
    )
    test_loader = build_dataloader(
        dataset=test_ds,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        device_type=cfg.device,
    )

    return train_loader, val_loader, test_loader



def maybe_limit_loader(loader, max_batches=None):
    if max_batches is None:
        return loader

    class LimitedLoader:
        def __init__(self, loader, max_batches):
            self.loader = loader
            self.max_batches = max_batches

        def __iter__(self):
            for i, batch in enumerate(self.loader):
                if i >= self.max_batches:
                    break
                yield batch

        def __len__(self):
            return min(len(self.loader), self.max_batches)

    return LimitedLoader(loader, max_batches)