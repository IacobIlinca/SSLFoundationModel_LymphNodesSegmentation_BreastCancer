import torch
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from src.VocoLarge.third_party_voco_large.models.voco_head import VoCoHead
from src.VocoLarge.training_ssl.pipeline.ckpt import load_ckpt, save_ckpt_atomic
from src.VocoLarge.training_ssl.pipeline.config import Config
from src.VocoLarge.training_ssl.pipeline.freeze import report_trainable_by_module


def build_model(voco_args, device: torch.device) -> VoCoHead:
    model = VoCoHead(voco_args).to(device)
    return model


def build_scheduler(args: Config, optimizer):
    if args.scheduler is None or args.scheduler.lower() == "none":
        return None

    name = args.scheduler.lower()

    if name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr_min,
        )

    raise ValueError(f"Unsupported scheduler: {args.scheduler}")

def build_optimizer(args: Config, model):
    params = [p for p in model.parameters() if p.requires_grad]

    if args.optimizer.lower() == "sgd":
        return SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.optimizer.lower() == "adamw":
        return AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def setup_model_and_optimizer(args: Config, device):
    model = build_model(args, device).train()

    if args.voco_ckpt_path:
        stats = load_ckpt(model, args.voco_ckpt_path, args.device, mode=args.load_mode)
        print(f"[ckpt] load_mode={args.load_mode} stats={stats}")
    else:
        print("[ckpt] no checkpoint provided; training from scratch")

    report_trainable_by_module(model)

    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    return model, optimizer, scaler, scheduler

def save_checkpoint(
    save_path: str,
    model,
    optimizer,
    scaler,
    scheduler,
    epoch: int,
):
    payload = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
    }
    save_ckpt_atomic(save_path, payload)


# Used for overfit experiment
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

# Used for overfit experiment
def disable_dropout(module: torch.nn.Module) -> int:
    """
    Convenience: set all dropout probabilities to 0.
    """
    return set_dropout_p(module, 0.0)