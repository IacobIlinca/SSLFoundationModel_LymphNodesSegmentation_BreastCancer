import os

import torch
from tqdm.auto import tqdm

from src.VocoLarge.third_party_voco_large.utils.ops import concat_image
from src.VocoLarge.training_ssl.pipeline import to_device, forward_loss, save_diff_bundle
from src.VocoLarge.training_ssl.pipeline.config import Config
from src.VocoLarge.training_ssl.pipeline.model import setup_model_and_optimizer
from src.VocoLarge.training_ssl.pipeline.train_and_valid_steps import compute_logits_targets_for_one_image
from src.VocoLarge.training_ssl.training.datasets_and_loaders import build_all_datasets_and_loaders


def run_validate():
    args = Config()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    _, val_loader, _ = build_all_datasets_and_loaders(args)

    model, _, _, _ = setup_model_and_optimizer(args, device)

    model.eval()

    total_loss = 0.0
    total_top1 = 0.0
    n_batches = 0

    desc = f"Validate after training"
    pbar = tqdm(val_loader, desc=desc, leave=False)

    for batch in pbar:
        img, labels, crops = batch
        img, crops = concat_image(img), concat_image(crops)
        img, crops, labels = to_device(img, crops, labels, device)

        loss, details = forward_loss(model, img, crops, labels, args.amp, False)
        total_loss += float(loss.item())
        total_top1 += details["top1"]

        n_batches += 1
        avg_loss = total_loss / n_batches
        avg_top1 = total_top1 / n_batches

        pbar.set_postfix(
            batch_loss=f"{loss.item():.4f}",
            avg_loss=f"{avg_loss:.4f}",
            avg_top1=f"{avg_top1 * 100:.2f}%",
        )

        # if details["top1"] < 1:
        #     logits, targets = compute_logits_targets_for_one_image(model, batch)
        #     save_diff_bundle(logits, targets, args.out_dir, prefix=f"batch{n_batches:05d}")

    if n_batches == 0:
        return 0.0, 0.0

    avg_loss = total_loss / n_batches
    avg_top1 = total_top1 / n_batches
    return avg_loss, avg_top1

if __name__ == "__main__":
    run_validate()