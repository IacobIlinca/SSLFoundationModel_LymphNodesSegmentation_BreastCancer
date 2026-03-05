from pathlib import Path
from typing import List, Optional
from torch.utils.data import Dataset, DataLoader


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
    )