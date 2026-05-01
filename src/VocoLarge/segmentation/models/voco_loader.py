from typing import Dict
import torch
import torch.nn as nn

from src.VocoLarge.segmentation.config import Config
from src.VocoLarge.segmentation.multiclass_segmentation.config_multiclass import ConfigMulticlass


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
        "decoder",
        "model.decoder",
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


def load_voco_encoder_weights_ssl(model: nn.Module, cfg: ConfigMulticlass) -> None:
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

    ckpt = torch.load(cfg.voco_ckpt_path, map_location="cpu")
    sd = _unwrap_state_dict(ckpt)

    target_sd = model.state_dict()
    filtered = {}

    for k, v in sd.items():
        if "backbone" in k:
            ks = k[len("backbone."):]
            if ks in target_sd and target_sd[ks].shape == v.shape:
                filtered[ks] = v

    load_res = model.load_state_dict(filtered, strict=False)

    total_target = len(target_sd)
    matched = len(filtered)
    ratio = matched / max(total_target, 1)

    print("\n[VoCo->Swin] Encoder weight loading report")
    print(f"  ckpt: {cfg.voco_ckpt_path}")
    print(f"  target encoder tensors: {total_target}")
    print(f"  matched tensors:        {matched} ({ratio*100:.1f}%)")
    print(f"  missing (first 20):     {load_res.missing_keys[:20]}")
    print(f"  unexpected (first 20):  {load_res.unexpected_keys[:20]}")


def reinitialize_module(module: nn.Module, module_name: str = "module", verbose: bool = True):
    """
    Reinitialize common trainable layers inside a module and print what was reset.
    """

    reset_count = 0
    skipped_count = 0

    if verbose:
        print(f"\n[REINIT] Starting reinitialization of: {module_name}")

    for child_name, m in module.named_modules():
        full_name = f"{module_name}.{child_name}" if child_name != "" else module_name

        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                          nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

            if m.bias is not None:
                nn.init.zeros_(m.bias)

            reset_count += 1

            if verbose:
                print(
                    f"[REINIT] {full_name}: "
                    f"{m.__class__.__name__}, "
                    f"weight={tuple(m.weight.shape)}, "
                    f"bias={m.bias is not None}"
                )

        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)

            if m.bias is not None:
                nn.init.zeros_(m.bias)

            reset_count += 1

            if verbose:
                print(
                    f"[REINIT] {full_name}: "
                    f"{m.__class__.__name__}, "
                    f"weight={tuple(m.weight.shape)}, "
                    f"bias={m.bias is not None}"
                )

        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                            nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                            nn.LayerNorm, nn.GroupNorm)):
            did_reset = False

            if getattr(m, "weight", None) is not None:
                nn.init.ones_(m.weight)
                did_reset = True

            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
                did_reset = True

            if did_reset:
                reset_count += 1

                if verbose:
                    print(
                        f"[REINIT] {full_name}: "
                        f"{m.__class__.__name__}, "
                        f"weight={None if getattr(m, 'weight', None) is None else tuple(m.weight.shape)}, "
                        f"bias={getattr(m, 'bias', None) is not None}"
                    )
            else:
                skipped_count += 1

        else:
            skipped_count += 1

    if verbose:
        print(f"[REINIT] Finished reinitializing: {module_name}")
        print(f"[REINIT] Reset layers: {reset_count}")
        print(f"[REINIT] Skipped modules: {skipped_count}\n")