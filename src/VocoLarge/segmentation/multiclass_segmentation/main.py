import os

import numpy as np
from monai.utils import set_determinism

from src.VocoLarge.segmentation.models.build import build_model
from src.VocoLarge.segmentation.models.freeze import freeze_encoder, report_trainable_by_module
from src.VocoLarge.segmentation.models.voco_loader import load_voco_encoder_weights
from src.VocoLarge.segmentation.multiclass_segmentation.config_multiclass import ConfigMulticlass
from src.VocoLarge.segmentation.multiclass_segmentation.train_loop import run_training
from src.VocoLarge.segmentation.training.visuals import save_overlay_png


def make_visuals_callback(cfg: ConfigMulticlass):
    def cb(epoch, batch_index, image, label, pred, case_ids):
        if batch_index not in cfg.visuals_case_indices:
            return

        img_np = image[0, 0].detach().cpu().numpy()
        case_id = case_ids[0]
        lab_np = label[0, 0].detach().cpu().numpy().astype(np.int32)

        # adapt this if your sigmoid metric output shape differs
        pred_np = pred[0][0].detach().cpu().numpy().astype(np.int32)

        out_dir = os.path.join(cfg.save_dir, "visuals_test")
        os.makedirs(out_dir, exist_ok=True)

        for sidx in cfg.visuals_slices:
            if 0 <= sidx < img_np.shape[-1]:
                out_path = os.path.join(
                    out_dir,
                    f"epoch_{epoch:04d}_case_{case_id}_slice_{sidx:03d}.png"
                )
                save_overlay_png(img_np, lab_np, pred_np, out_path, sidx)

    return cb


def main():
    cfg = ConfigMulticlass()

    set_determinism(seed=cfg.seed)

    model = build_model(cfg)
    load_voco_encoder_weights(model, cfg)
    freeze_encoder(model, cfg)
    report_trainable_by_module(model)

    visuals_cb = make_visuals_callback(cfg) if cfg.save_visuals else None

    run_training(model, cfg, visuals_cb=visuals_cb)


if __name__ == "__main__":
    main()