import torch
from src.VocoLarge.third_party_voco_large.models.voco_head import VoCoHead


def build_model(voco_args, device: torch.device) -> VoCoHead:
    model = VoCoHead(voco_args).to(device)
    return model

def set_dropout_p(module: torch.nn.Module, p: float) -> int:
    """
    Set Dropout/Dropout1d/Dropout2d/Dropout3d probability for all submodules.
    Returns number of dropout layers updated.
    """
    n = 0
    for m in module.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            m.p = float(p)
            n += 1
    return n


def disable_dropout(module: torch.nn.Module) -> int:
    """
    Convenience: set all dropout probabilities to 0.
    """
    return set_dropout_p(module, 0.0)