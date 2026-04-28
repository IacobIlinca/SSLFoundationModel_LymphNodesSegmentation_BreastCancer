import os

from monai.data import PersistentDataset, list_data_collate
from torch.utils.data import DataLoader

from src.VocoLarge.segmentation.config_binary import ConfigBinary
from src.VocoLarge.segmentation.data.data_utils_multiclass import load_multiclass_mask_csv, \
    build_multiclass_files_from_ids
from src.VocoLarge.segmentation.data.transforms import get_transforms_binary, get_transforms_multiclass
from src.VocoLarge.segmentation.data.data_utils_binary import (
    read_ids_file,
    build_segmentation_files_from_ids,
)
from src.VocoLarge.segmentation.data.data_utils_binary import filter_positive_cases
from src.VocoLarge.segmentation.multiclass_segmentation.config_multiclass import ConfigMulticlass


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
        num_workers=cfg.num_workers,
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


def build_all_datasets_and_loaders_multiclass(cfg: ConfigMulticlass):
    """
    Multiclass version.

    Required cfg fields:
        root_dir
        multiclass_masks_csv_path
        train_ids_path
        val_ids_path
        test_ids_path
        cache_dir

        num_classes = 6
        class_to_index optional
        batch_size
        val_batch_size
        test_batch_size
        shuffle
        num_workers
        device

    This does NOT use binary-specific functions like filter_positive_cases.
    """

    train_ids = read_ids_file(cfg.train_ids_path)
    val_ids = read_ids_file(cfg.val_ids_path)
    test_ids = read_ids_file(cfg.test_ids_path)

    print(f"[INFO] train ids: {len(train_ids)}")
    print(f"[INFO] val ids:   {len(val_ids)}")
    print(f"[INFO] test ids:  {len(test_ids)}")

    class_to_index = cfg.class_to_index
    labels = list(class_to_index.keys())

    case_to_masks = load_multiclass_mask_csv(
        csv_path=cfg.multiclass_masks_csv_path,
        root_dir=cfg.root_dir,
        class_to_csv_column=cfg.class_to_csv_column,
        labels=labels,
    )

    print(f"[INFO] cases in multiclass CSV: {len(case_to_masks)}")

    train_files, skipped_train = build_multiclass_files_from_ids(
        root_dir=cfg.root_dir,
        ids=train_ids,
        case_to_masks=case_to_masks,
        labels=labels,
        require_foreground=True,
    )

    val_files, skipped_val = build_multiclass_files_from_ids(
        root_dir=cfg.root_dir,
        ids=val_ids,
        case_to_masks=case_to_masks,
        labels=labels,
        require_foreground=True,
    )

    test_files, skipped_test = build_multiclass_files_from_ids(
        root_dir=cfg.root_dir,
        ids=test_ids,
        case_to_masks=case_to_masks,
        labels=labels,
        require_foreground=True,
    )

    print(f"[INFO] train files kept: {len(train_files)} | skipped: {len(skipped_train)}")
    print(f"[INFO] val files kept:   {len(val_files)} | skipped: {len(skipped_val)}")
    print(f"[INFO] test files kept:  {len(test_files)} | skipped: {len(skipped_test)}")

    if len(skipped_train) > 0:
        print(f"[INFO] first skipped train ids: {skipped_train[:10]}")
    if len(skipped_val) > 0:
        print(f"[INFO] first skipped val ids:   {skipped_val[:10]}")
    if len(skipped_test) > 0:
        print(f"[INFO] first skipped test ids:  {skipped_test[:10]}")

    if len(train_files) == 0:
        raise RuntimeError("No multiclass training files found.")
    if len(val_files) == 0:
        raise RuntimeError("Multiclass validation set is empty.")
    if len(test_files) == 0:
        raise RuntimeError("Multiclass test set is empty.")

    train_transform, val_transform = get_transforms_multiclass(cfg)
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
        num_workers=cfg.num_workers,
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