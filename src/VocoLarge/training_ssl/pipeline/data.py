import os
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import torch
from monai.data import PersistentDataset
from torch.utils.data import Dataset, DataLoader, default_collate


# Used in overfit experiment
def find_case_images(root_dir: str) -> List[str]:
    root = Path(root_dir)
    if not root.exists():
        raise RuntimeError(f"data_dir not found: {root_dir}")

    case_dirs = [p for p in root.iterdir() if p.is_dir()]
    case_dirs.sort()

    image_paths = []
    for c in case_dirs:
        niftis = [p for p in (list(c.rglob("*.nii")) + list(c.rglob("*.nii.gz")))
                  if "mask" not in p.name.lower()]
        if len(niftis) == 0:
            continue

        preferred = [p for p in niftis if p.name.lower() in ("image.nii.gz", "image.nii", "img.nii.gz", "img.nii")]
        chosen = preferred[0] if len(preferred) else niftis[0]
        image_paths.append(str(chosen))

    return image_paths

# Dataset used in overfit experiment
class NiftiListDataset(Dataset):
    """
    Each item is a MONAI dict run through your transform pipeline.
    The output can be variable-structured (VoCoAugmentation returns tuples/lists),
    hence batch_size should remain 1 unless you implement a custom collate.
    """
    def __init__(self, image_paths, xform):
        self.image_paths = list(image_paths)
        self.xform = xform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.xform({"image": self.image_paths[idx]})

def inspect_obj(name, obj, indent=0):
    prefix = " " * indent

    if torch.is_tensor(obj):
        print(f"{prefix}{name}: TENSOR shape={tuple(obj.shape)} dtype={obj.dtype}")
    elif isinstance(obj, np.ndarray):
        print(f"{prefix}{name}: NUMPY shape={obj.shape} dtype={obj.dtype}")
    elif isinstance(obj, dict):
        print(f"{prefix}{name}: DICT")
        for k, v in obj.items():
            inspect_obj(f"{k}", v, indent + 2)
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{name}: {type(obj).__name__.upper()} len={len(obj)}")
        for i, v in enumerate(obj):
            inspect_obj(f"[{i}]", v, indent + 2)
    else:
        print(f"{prefix}{name}: {type(obj)} value={obj}")

def debug_collate(batch):
    print("\n========== NEW BATCH ==========")

    for bi, sample in enumerate(batch):
        print(f"\n--- sample {bi} ---")
        inspect_obj("sample", sample)

    return default_collate(batch)

def build_dataloader(
    dataset: Dataset,
    device_type: str,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device_type == "cuda"),
        #collate_fn=debug_collate,
    )

def read_ids_file(txt_path: str) -> List[str]:
    """
    Reads IDs from a txt file.
    Supports either:
      - comma-separated: id1,id2,id3
    """
    with open(txt_path, "r") as f:
        content = f.read().strip()

    if not content:
        raise RuntimeError(f"No IDs found in {txt_path}")

    if "," in content:
        ids = [x.strip() for x in content.split(",") if x.strip()]
    else:
        raise RuntimeError(f"No comma separated file {txt_path}")

    if len(ids) == 0:
        raise RuntimeError(f"Error while reading ids from {txt_path}")

    return ids


def find_image_in_patient_folder(patient_dir: Path) -> str:
    """
    Finds the image file inside one patient folder.

    Adjust the patterns here if your naming differs.
    """
    patterns = [
        "image.nii.gz",
        "image.nii",
        "image.nii.tz",
    ]

    matches = []
    for pattern in patterns:
        matches.extend(patient_dir.glob(pattern))

    if len(matches) == 0:
        raise FileNotFoundError(f"No image file found in: {patient_dir}")

    if len(matches) > 1:
        raise RuntimeError(f"Multiple image files found in {patient_dir}")

    return str(matches[0])


def build_files_from_ids(root_dir: str, ids: List[str]) -> List[Dict[str, str]]:
    """
    Converts patient IDs to MONAI-style dicts:
        [{"image": "/path/to/id1/image.nii.gz"}, ...]
    """
    root = Path(root_dir)
    files = []

    for pid in ids:
        patient_dir = root / pid
        if not patient_dir.exists():
            raise RuntimeError (f"Missing patient folder: {patient_dir}")

        image_path = find_image_in_patient_folder(patient_dir)
        files.append({
            "image": image_path,
            "patient_id": pid,
        })

    return files


def build_persistent_dataset(
    files: List[Dict[str, str]],
    transform,
    cache_dir: str,
):
    os.makedirs(cache_dir, exist_ok=True)
    return PersistentDataset(
        data=files,
        transform=transform,
        cache_dir=cache_dir,
    )