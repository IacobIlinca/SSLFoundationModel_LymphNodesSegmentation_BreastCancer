import os
import torch
from typing import Dict, Any


def _clean_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Handles:
      - {"state_dict": ...}
      - DDP 'module.' prefix
    """
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}

    return sd


def _match_and_load(target_module, sd: Dict[str, torch.Tensor], name: str):
    """
    Loads only matching keys into target_module and reports stats.
    """
    msd = target_module.state_dict()

    matched = {}
    skipped_shape = []
    missing = []

    for k, v in sd.items():
        if k in msd:
            if msd[k].shape == v.shape:
                matched[k] = v
            else:
                skipped_shape.append(k)

    for k in msd.keys():
        if k not in sd:
            missing.append(k)

    msd.update(matched)
    target_module.load_state_dict(msd, strict=True)

    matched_n = len(matched)
    total_target = len(msd)
    total_ckpt = len(sd)
    pct = 100.0 * matched_n / max(1, total_target)

    print(f"[INFO] Loading checkpoint into '{name}'")
    print(f"[INFO]   tensors in checkpoint : {total_ckpt}")
    print(f"[INFO]   tensors in model      : {total_target}")
    print(f"[INFO]   matched tensors      : {matched_n}/{total_target} ({pct:.1f}%)")
    print(f"[INFO]   shape mismatches     : {len(skipped_shape)}")
    print(f"[INFO]   missing in ckpt      : {len(missing)}")

    return {
        "matched": matched_n,
        "total_model": total_target,
        "total_ckpt": total_ckpt,
        "pct": pct,
        "shape_mismatch": len(skipped_shape),
        "missing": len(missing),
    }


def load_ckpt(
    model,
    ckpt_path: str,
    device: str,
    mode: str = "full",  # "full" | "backbone"
) -> Dict[str, Any]:
    """
    mode:
        - "full"      → load backbone + student + teacher
        - "backbone"  → load only backbone weights
    """

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[INFO] Loading checkpoint: {ckpt_path}")

    sd = torch.load(ckpt_path, map_location=device)
    sd = _clean_state_dict(sd)

    stats = {}

    if mode == "backbone":
        stats["backbone"] = _match_and_load(model.backbone, sd, "backbone")

    elif mode == "full":
        stats["full"] = _match_and_load(model, sd, "full model")

    else:
        raise ValueError(f"Unknown load mode: {mode}")

    return stats


def save_ckpt_atomic(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)