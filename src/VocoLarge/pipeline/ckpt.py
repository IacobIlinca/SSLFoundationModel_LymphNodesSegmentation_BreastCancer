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
    Loads only matching keys into target_module.
    """
    msd = target_module.state_dict()
    matched = {}

    for k, v in sd.items():
        if k in msd and msd[k].shape == v.shape:
            matched[k] = v

    msd.update(matched)
    target_module.load_state_dict(msd, strict=True)

    pct = 100.0 * len(matched) / max(1, len(msd))
    print(f"[ckpt] {name} matched: {len(matched)}/{len(msd)} ({pct:.1f}%)")

    return {
        "matched": len(matched),
        "total": len(msd),
        "pct": pct,
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

    sd = torch.load(ckpt_path, map_location=device)
    sd = _clean_state_dict(sd)

    stats = {}

    if mode == "backbone":
        stats["backbone"] = _match_and_load(model.backbone, sd, "backbone")

    elif mode == "full":
        # Full model state_dict (backbone + student + teacher)
        stats["full"] = _match_and_load(model, sd, "full model")

    else:
        raise ValueError(f"Unknown load mode: {mode}")

    return stats


def save_ckpt_atomic(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)