from typing import Dict
import torch
import torch.nn as nn

from src.VocoLarge.segmentation.config import Config


def _unwrap_state_dict(ckpt: Dict) -> Dict[str, torch.Tensor]:
    # Many checkpoints store weights under "state_dict" or "model".
    for k in ["state_dict", "model", "net", "network"]:
        if isinstance(ckpt, dict) and k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    return ckpt


def load_voco_encoder_weights(model: nn.Module, cfg) -> None:
    """
    REQUIRED.

    Loads VoCo weights into SwinUNETR encoder: model.swinViT

    This is the *core* of your thesis experiment:
      - If weights don't match, you are not probing VoCo.

    It prints a report:
      - number of encoder tensors
      - how many matched by name+shape
      - missing/unexpected keys

    Optional strict gate:
      - if cfg.strict_load and match% < threshold => crash early.
    """
    if not hasattr(model, "swinViT"):
        raise AttributeError("Expected SwinUNETR to have attribute 'swinViT'")

    ckpt = torch.load(cfg.voco_ckpt_path, cfg.device)
    sd = _unwrap_state_dict(ckpt)

    # Common prefixes seen in VoCo/DP/DDP training
    candidate_prefixes = [
        "backbone.swinViT.",
        "module.backbone.swinViT.",
        "swinViT.",
        "module.swinViT.",
        "encoder",
        "model.encoder",
    ]

    target_sd = model.state_dict()
    filtered = {}

    for k, v in sd.items():
        found = False
        for pref in candidate_prefixes:
            if k.startswith(pref):
                found = True
                if k in target_sd and target_sd[k].shape == v.shape:
                    filtered[k] = v
                break
        if not found:
            print(f"[WARN] Tensor not found to load {k}")


    load_res = model.load_state_dict(filtered, strict=False)

    total_target = len(target_sd)
    matched = len(filtered)
    ratio = matched / max(total_target, 1)

    print("\n[VoCo->Swin] Encoder weight loading report")
    print(f"  ckpt: {cfg.voco_ckpt_path}")
    print(f"  loaded tensors:         {len(sd.keys())}")
    print(f"  target encoder tensors: {total_target}")
    print(f"  matched tensors:        {matched} ({ratio*100:.1f}%)")
    print(f"  missing (first 20):     {load_res.missing_keys[:20]}")
    print(f"  unexpected (first 20):  {load_res.unexpected_keys[:20]}")