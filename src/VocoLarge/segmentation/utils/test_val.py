import torch

from src.VocoLarge.segmentation.config_binary import ConfigBinary
from src.VocoLarge.segmentation.models.build import build_model
from src.VocoLarge.segmentation.models.voco_loader import load_voco_encoder_weights
from src.VocoLarge.segmentation.models.freeze import freeze_encoder
from src.VocoLarge.segmentation.data.loaders_binary import build_all_datasets_and_loaders
from src.VocoLarge.segmentation.training.enigne_binary import evaluate
from src.VocoLarge.segmentation.training.losses_metrics import build_loss_binary


def main():
    cfg = ConfigBinary()

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    load_voco_encoder_weights(model, cfg)
    freeze_encoder(model, cfg)

    loss_fn = build_loss_binary(cfg)

    _, val_loader, _ = build_all_datasets_and_loaders(cfg)

    val_loss, val_dice, val_hd95 = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        loss_fn=loss_fn,
        cfg=cfg,
        desc="Val debug",
        visuals_cb=None,
        epoch=0,
    )

    print(f"[RESULT] val_loss={val_loss:.4f}, val_dice={val_dice:.4f}, val_hd95={val_hd95}")

if __name__ == "__main__":
    main()