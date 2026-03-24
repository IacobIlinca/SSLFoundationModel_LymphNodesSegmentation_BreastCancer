import os
import numpy as np

from monai.utils import set_determinism

from src.VocoLarge.segmentation.config_binary import ConfigBinary
from src.VocoLarge.segmentation.models.build import build_model
from src.VocoLarge.segmentation.models.voco_loader import load_voco_encoder_weights
from src.VocoLarge.segmentation.models.freeze import freeze_encoder, report_trainable_by_module
from src.VocoLarge.segmentation.training.enigne_binary import run_training

# OPTIONAL
from src.VocoLarge.segmentation.training.visuals import save_overlay_png


def make_visuals_callback(cfg: ConfigBinary):
    """
    Saves overlay images for one validation case at specific slices.
    Works for binary segmentation as well.
    """
    def cb(epoch, batch_index, image, label, pred):
        if batch_index != cfg.visuals_case_index:
            return
        if epoch is None:
            return
        if epoch != 1 and epoch % cfg.log_every != 0:
            return

        # image: (1,1,H,W,D)
        img_np = image[0, 0].detach().cpu().numpy()

        # label: (1,1,H,W,D)
        lab_np = label[0, 0].detach().cpu().numpy().astype(np.int32)

        # pred[0]: (H,W,D)
        pred_np = pred[0][1].detach().cpu().numpy().astype(np.int32)

        out_dir = os.path.join(cfg.save_dir, "visuals")
        os.makedirs(out_dir, exist_ok=True)

        for sidx in cfg.visuals_slices:
            if 0 <= sidx < img_np.shape[-1]:
                out_path = os.path.join(
                    out_dir,
                    f"epoch_{epoch:04d}_slice_{sidx:03d}.png"
                )
                save_overlay_png(img_np, lab_np, pred_np, out_path, sidx)

    return cb


def main():
    cfg = ConfigBinary()

    set_determinism(seed=cfg.seed)

    # ---- Model ----
    model = build_model(cfg)

    # Load pretrained encoder weights
    load_voco_encoder_weights(model, cfg)

    # Freeze encoder for linear probe
    freeze_encoder(model, cfg)
    report_trainable_by_module(model)

    # ---- Visuals ----
    visuals_cb = make_visuals_callback(cfg) if cfg.save_visuals else None

    # ---- Training ----
    run_training(model, cfg, visuals_cb=visuals_cb)


if __name__ == "__main__":
    main()