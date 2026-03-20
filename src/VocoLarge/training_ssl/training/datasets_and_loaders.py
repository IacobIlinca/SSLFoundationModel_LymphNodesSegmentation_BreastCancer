import os

from src.VocoLarge.training_ssl.pipeline import build_transforms
from src.VocoLarge.training_ssl.pipeline.config import Config
from src.VocoLarge.training_ssl.pipeline.data import read_ids_file, build_files_from_ids, build_persistent_dataset, \
    build_dataloader


def build_all_datasets_and_loaders(args: Config):
    """
    Builds train / val / test datasets and loaders from split txt files.
    Assumes you already have split files.
    """
    train_ids = read_ids_file(args.train_ids_path)
    val_ids = read_ids_file(args.val_ids_path)
    test_ids = read_ids_file(args.test_ids_path)

    print(f"[INFO] train ids: {len(train_ids)}")
    print(f"[INFO] val ids:   {len(val_ids)}")
    print(f"[INFO] test ids:  {len(test_ids)}")

    train_files = build_files_from_ids(args.data_dir, train_ids)
    val_files = build_files_from_ids(args.data_dir, val_ids)
    test_files = build_files_from_ids(args.data_dir, test_ids)

    print(f"[INFO] train files found: {len(train_files)}")
    print(f"[INFO] val files found:   {len(val_files)}")
    print(f"[INFO] test files found:  {len(test_files)}")

    if len(train_files) == 0:
        raise RuntimeError("No training files found.")
    if len(val_files) == 0:
        raise RuntimeError("Validation set is empty.")
    if len(test_files) == 0:
        raise RuntimeError("Test set is empty.")

    train_transform = build_transforms(args, False)
    val_transform = build_transforms(args, True)
    test_transform = build_transforms(args, True)

    train_ds = build_persistent_dataset(
        files=train_files,
        transform=train_transform,
        cache_dir=os.path.join(args.cache_dir, "train"),
    )
    val_ds = build_persistent_dataset(
        files=val_files,
        transform=val_transform,
        cache_dir=os.path.join(args.cache_dir, "val"),
    )
    test_ds = build_persistent_dataset(
        files=test_files,
        transform=test_transform,
        cache_dir=os.path.join(args.cache_dir, "test"),
    )

    train_loader = build_dataloader(
        dataset=train_ds,
        batch_size=args.train_batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        device_type=args.device,
    )
    val_loader = build_dataloader(
        dataset=val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device_type=args.device,
    )
    test_loader = build_dataloader(
        dataset=test_ds,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device_type=args.device,
    )

    return train_loader, val_loader, test_loader