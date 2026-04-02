import os
import numpy as np

from monai.utils import set_determinism

from src.VocoLarge.segmentation.hyperparameter_tuning.cofig_binary_test import ConfigBinaryTest
from src.VocoLarge.segmentation.models.build import build_model
from src.VocoLarge.segmentation.models.voco_loader import load_voco_encoder_weights
from src.VocoLarge.segmentation.models.freeze import freeze_encoder, report_trainable_by_module
from src.VocoLarge.segmentation.hyperparameter_tuning.test_only_binary import run_test_only
from src.VocoLarge.segmentation.training.visuals import save_overlay_png


def make_visuals_callback(cfg: ConfigBinaryTest):
    def cb(epoch, batch_index, image, label, pred):
        if batch_index != cfg.visuals_case_index:
            return

        img_np = image[0, 0].detach().cpu().numpy()
        lab_np = label[0, 0].detach().cpu().numpy().astype(np.int32)

        # adapt this if your sigmoid metric output shape differs
        pred_np = pred[0][0].detach().cpu().numpy().astype(np.int32)

        out_dir = os.path.join(cfg.save_dir, "visuals_test")
        os.makedirs(out_dir, exist_ok=True)

        for sidx in cfg.visuals_slices:
            if 0 <= sidx < img_np.shape[-1]:
                out_path = os.path.join(
                    out_dir,
                    f"epoch_{epoch}_test_slice_{sidx:03d}.png"
                )
                save_overlay_png(img_np, lab_np, pred_np, out_path, sidx)

    return cb


def main():
    cfg = ConfigBinaryTest()

    cfg.fast_val = True
    cfg.save_visuals = True

    # tweak loss weights here
    # cfg.class_weight_for_loss = [0.05, 1.0]
    # or try:
    # cfg.class_weight_for_loss = [0.01, 1.0]
    # cfg.class_weight_for_loss = [0.1, 2.0]

    set_determinism(seed=cfg.seed)

    model = build_model(cfg)
    load_voco_encoder_weights(model, cfg)
    freeze_encoder(model, cfg)
    report_trainable_by_module(model)

    visuals_cb = make_visuals_callback(cfg) if cfg.save_visuals else None

    run_test_only(model, cfg, visuals_cb=visuals_cb)


if __name__ == "__main__":
    main()