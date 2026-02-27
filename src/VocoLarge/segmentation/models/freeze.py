import torch.nn as nn

from src.VocoLarge.segmentation.config import Config


def freeze_encoder(model: nn.Module, cfg: Config) -> None:
    """
    REQUIRED for linear probing.

    freeze_scope:
      - "swin": freeze only transformer encoder (recommended)
      - "swin_plus_conv": stricter freeze of additional encoder-like blocks (optional)

    Always prints % trainable params so you can verify freezing worked.
    """
    if cfg.freeze_scope == "swin":
        for p in model.swinViT.parameters():
            p.requires_grad = False

    elif cfg.freeze_scope == "swin_plus_conv":
        for p in model.swinViT.parameters():
            p.requires_grad = False
        # Heuristic: freeze parameters with "encoder"/"enc" in name but not decoder/out
        for name, p in model.named_parameters():
            n = name.lower()
            if ("enc" in n or "encoder" in n) and ("decoder" not in n) and ("out" not in n):
                p.requires_grad = False
    else:
        raise ValueError(f"Unknown freeze_scope: {cfg.freeze_scope}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Freeze] total params: {total:,} | trainable: {trainable:,} ({trainable/total*100:.2f}%)")