import os

from monai.utils import set_determinism

from src.VocoLarge.segmentation.config import Config
from src.VocoLarge.segmentation.data.samples import build_samples
from src.VocoLarge.segmentation.models.build import build_model
from src.VocoLarge.segmentation.models.voco_loader import load_voco_encoder_weights
from src.VocoLarge.segmentation.models.freeze import freeze_encoder
from src.VocoLarge.segmentation.training.engine import run_training

# OPTIONAL
from src.VocoLarge.segmentation.training.visuals import save_overlay_png
import numpy as np


def make_visuals_callback(cfg: Config):
    """
    OPTIONAL.

    Returns a function that saves overlays for a fixed val case index and slice indices.
    """
    def cb(epoch, batch_index, image, label, pred):
        # Only visualize a single chosen validation case index
        if batch_index != cfg.visuals_case_index:
            return
        if epoch is None:
            return
        if epoch != 1 and epoch % cfg.log_every != 0:
            return

        # image: (1,1,H,W,D)
        img_np = image[0, 0].detach().cpu().numpy()  # (H,W,D)
        lab_np = label[0, 0].detach().cpu().numpy().astype(np.int32)

        # pred[0]: (H,W,D) integer after AsDiscrete(argmax=True)
        pred_np = pred[0].detach().cpu().numpy().astype(np.int32)

        out_dir = os.path.join(cfg.save_dir, "visuals")
        for sidx in cfg.visuals_slices:
            if 0 <= sidx < img_np.shape[-1]:
                out_path = os.path.join(out_dir, f"epoch_{epoch:04d}_slice_{sidx:03d}.png")
                save_overlay_png(img_np, lab_np, pred_np, out_path, sidx)

    return cb


def main():
    cfg = Config(
        overfit_case_id="30692BF6DB8F95",
    )

    set_determinism(seed=cfg.seed)

    # Build samples (TODO: parsing later)
    train_samples, val_samples = build_samples()

    # Model
    model = build_model(cfg)

    # Load VoCo encoder weights + report
    load_voco_encoder_weights(model, cfg)

    # Freeze encoder
    freeze_encoder(model, cfg)

    # Optional visuals callback
    visuals_cb = make_visuals_callback(cfg) if cfg.save_visuals else None

    # Train
    run_training(model, train_samples, val_samples, cfg, visuals_cb=visuals_cb)


if __name__ == "__main__":
    main()