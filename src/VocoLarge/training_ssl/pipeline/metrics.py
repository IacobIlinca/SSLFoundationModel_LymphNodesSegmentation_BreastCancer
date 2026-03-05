import torch

def top1_match(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    logits/targets: (sw_s, 9)
    """
    top1_pred = logits.argmax(dim=1)
    top1_tgt = targets.argmax(dim=1)
    return float((top1_pred == top1_tgt).float().mean().item())


def best_crop_indices_from_targets(targets: torch.Tensor) -> torch.Tensor:
    """
    targets: (sw_s, 9)
    returns: (sw_s,) long indices, best crop per query according to targets
    """
    return targets.argmax(dim=1)


def best_crop_indices_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: (sw_s, 9)
    returns: (sw_s,) long indices, best crop per query according to model predictions
    """
    return logits.argmax(dim=1)


def best_crop_report(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    Returns a small report you can print/log.
    """
    pred = best_crop_indices_from_logits(logits)
    tgt = best_crop_indices_from_targets(targets)
    acc = float((pred == tgt).float().mean().item())
    return {
        "top1_match": acc,
        "pred_indices": pred.cpu(),
        "tgt_indices": tgt.cpu(),
    }


def mae(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float((logits - targets).abs().mean().item())


def mse(logits: torch.Tensor, targets: torch.Tensor) -> float:
    d = (logits - targets)
    return float((d * d).mean().item())