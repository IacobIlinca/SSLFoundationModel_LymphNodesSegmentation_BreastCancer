import os

import torch

from src.VocoLarge.training_ssl.experiments.lr_finder import maybe_limit_test_loader
from src.VocoLarge.training_ssl.pipeline.model import setup_model_and_optimizer, save_checkpoint
from src.VocoLarge.training_ssl.pipeline.config import Config, save_config
from src.VocoLarge.training_ssl.pipeline.train_and_valid_steps import train_one_epoch, validate_one_epoch, \
    compute_logits_targets_for_one_image
from src.VocoLarge.training_ssl.pipeline.viz import History, plot_loss_curves, save_diff_bundle
from src.VocoLarge.training_ssl.training.datasets_and_loaders import build_all_datasets_and_loaders


def run_test():
    args = Config()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    _, val_loader, test_loader = build_all_datasets_and_loaders(args)
    #test_loader = maybe_limit_test_loader(test_loader, max_batches=100)

    batch_to_viz = next(iter(test_loader))

    model, optimizer, scaler = setup_model_and_optimizer(args, device)

    history = History()
    save_config(args)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, test_loader, optimizer, scaler, device, epoch, args)
        #train_loss = 0.5
        history.add_train_loss(epoch, train_loss)

        print(f"epoch {epoch:04d}/{args.epochs} | train_loss={train_loss:.6f}")

        should_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if should_eval:
            val_loss, top1 = validate_one_epoch(model, val_loader, device, epoch)  # evaluate on val loader

            print(
                f"[eval] epoch {epoch:04d} "
                f"val_loss={val_loss:.6f} "
                f"top1={top1 * 100:.2f}% "
            )
            history.add_val_loss(epoch, val_loss)
            history.add_top1_metric(top1)

        should_save = (epoch % args.save_every == 0) or (epoch == args.epochs)
        if should_save:

            logits, targets = compute_logits_targets_for_one_image(model, batch_to_viz)
            save_diff_bundle(logits, targets, args.out_dir, prefix=f"epoch{epoch:05d}")

            ckpt_path = os.path.join(args.out_dir, f"epoch_{epoch:04d}.pt")
            save_checkpoint(ckpt_path, model, optimizer, scaler, epoch)
            print(f"[ckpt] saved: {ckpt_path}")


        plot_loss_curves(history, args.out_dir)

    final_path = os.path.join(args.out_dir, "final.pt")
    save_checkpoint(final_path, model, optimizer, scaler, args.epochs)
    print(f"[ckpt] saved final: {final_path}")

    return model

if __name__ == "__main__":
    run_test()