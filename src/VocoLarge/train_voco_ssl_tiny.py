import os
import glob
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from monai.transforms import Compose
from monai.data.meta_tensor import MetaTensor

from src.VocoLarge.third_party_voco_large.models.voco_head import VoCoHead
from src.VocoLarge.third_party_voco_large.utils import voco_trans, data_trans


def build_args(roi_x, roi_y, roi_z, device):
    class A: pass
    a = A()
    a.in_channels = 1
    a.feature_size = 48
    a.dropout_path_rate = 0.0
    a.use_checkpoint = True
    a.spatial_dims = 3

    a.roi_x = roi_x
    a.roi_y = roi_y
    a.roi_z = roi_z

    a.space_x = 1.5
    a.space_y = 1.5
    a.space_z = 1.5

    a.local_rank = 0
    a.sw_batch_size = 10
    a.amp = True
    a.device = device
    return a


class NiftiListDataset(Dataset):
    def __init__(self, image_paths, xform):
        self.image_paths = image_paths
        self.xform = xform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # MONAI dict-style
        return self.xform({"image": self.image_paths[idx]})


def unpack_voco_transform_output(out):
    """
    Matches your eval script assumptions:
      out is (img_obj, crop_obj, lab_obj) OR dict.
    Returns:
      img:   torch.Tensor (sw_s, 1, D, H, W)
      crops: torch.Tensor (9,    1, D, H, W)
      labels: torch.Tensor (1, sw_s, 9)
    """
    if isinstance(out, (list, tuple)):
        img_obj, crop_obj, lab_obj = out
    elif isinstance(out, dict):
        img_obj = out.get("image", None)
        crop_obj = out.get("crops", None)
        lab_obj = out.get("labels", None)
    else:
        raise RuntimeError(f"Unexpected output type: {type(out)}")

    if img_obj is None or crop_obj is None or lab_obj is None:
        raise RuntimeError("Transform did not return expected outputs.")

    img = torch.stack([d["image"] for d in img_obj], dim=0)     # (sw_s, 1, D, H, W)
    crops = torch.stack([d["image"] for d in lab_obj], dim=0)   # (9,   1, D, H, W)
    labels = torch.as_tensor(crop_obj, dtype=torch.float32).unsqueeze(0)  # (1, sw_s, 9)
    return img, crops, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="folder with .nii or .nii.gz images")
    p.add_argument("--out_dir", default="runs_voco_tiny")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--roi_x", type=int, default=192)
    p.add_argument("--roi_y", type=int, default=192)
    p.add_argument("--roi_z", type=int, default=64)
    p.add_argument("--use_chest_trans", action="store_true")
    p.add_argument("--no_aug", action="store_true", help="disable heavy aug (for debugging)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.device == "cuda" else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # build repo-style args object
    a = build_args(args.roi_x, args.roi_y, args.roi_z, args.device)

    # transforms
    trans_list = data_trans.get_chest_trans(a) if args.use_chest_trans else data_trans.get_headneck_trans(a)
    if args.no_aug:
        for i, t in enumerate(trans_list):
            if t.__class__.__name__ == "VoCoAugmentation":
                trans_list[i] = voco_trans.VoCoAugmentation(a, aug=False)
                break
    xform = Compose(trans_list)

    # data
    def find_case_images(root_dir: str):
        root = Path(root_dir)
        if not root.exists():
            raise RuntimeError(f"data_dir not found: {root_dir}")

        case_dirs = [p for p in root.iterdir() if p.is_dir()]
        case_dirs.sort()

        image_paths = []
        for c in case_dirs:
            # search recursively for nifti
            niftis = [p for p in (list(c.rglob("*.nii")) + list(c.rglob("*.nii.gz")))
                      if "mask" not in p.name.lower()]
            if len(niftis) == 0:
                print(f"[warn] no NIfTI found under case: {c}")
                continue

            # prefer common filenames if present
            preferred = [p for p in niftis if p.name.lower() in ("image.nii.gz", "image.nii", "img.nii.gz", "img.nii")]
            chosen = preferred[0] if len(preferred) else niftis[0]

            image_paths.append(str(chosen))

        return image_paths

    imgs = find_case_images(args.data_dir)
    if len(imgs) == 0:
        raise RuntimeError(f"No NIfTI files found under any case folders in {args.data_dir}")

    print(f"Found {len(imgs)} case images")
    print("Example:", imgs[0])

    ds = NiftiListDataset(imgs, xform)

    # IMPORTANT: each __getitem__ returns variable-structure; keep batch_size=1
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))

    # model
    model = VoCoHead(a).to(device).train()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # tiny training loop
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for batch in dl:
            # batch is length-1 list-ish because batch_size=1 with weird object
            out = batch[0] if isinstance(batch, (list, tuple)) else batch
            img, crops, labels = unpack_voco_transform_output(out)

            # MetaTensor like your eval
            img = MetaTensor(img).to(device)
            crops = MetaTensor(crops).to(device)
            labels = labels.to(device)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                loss = model(img, crops, labels)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            epoch_loss += float(loss.item())
            global_step += 1

        epoch_loss /= max(1, len(dl))
        print(f"epoch {epoch:03d}/{args.epochs}  loss={epoch_loss:.6f}")

        # checkpoint occasionally
        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.out_dir, f"voco_tiny_epoch{epoch:03d}.pt")
            torch.save({"state_dict": model.state_dict(), "epoch": epoch}, ckpt_path)
            print(f"saved: {ckpt_path}")

    print("done.")


if __name__ == "__main__":
    main()