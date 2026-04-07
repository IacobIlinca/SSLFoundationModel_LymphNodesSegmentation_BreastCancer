from collections import defaultdict

import numpy as np
import torch
from tqdm.auto import tqdm

from src.VocoLarge.third_party_voco_large.models.voco_head import online_assign, ce_loss, regularization_loss
from src.VocoLarge.third_party_voco_large.utils.ops import concat_image
from src.VocoLarge.training_ssl.pipeline import forward_loss, to_device
from src.VocoLarge.training_ssl.pipeline.config import Config
from src.VocoLarge.training_ssl.pipeline.grad_monitor import module_grad_stats, pick_monitor_layers, \
    summarize_gradient_health, collect_selected_grad_histograms


def create_train_stats(loss, grad_stats, details):
    stats = {"loss": float(loss.item()), **grad_stats}

    if details is not None:
        logits = details["logits"]
        labs = details["labels"]
        base_cos = details["base_cos"]

        if logits is not None:
            top2 = torch.topk(logits, k=min(2, logits.shape[1]), dim=1).values
            margin = (top2[:, 0] - top2[:, 1]).mean().item() if top2.shape[1] == 2 else float("nan")
            stats.update({
                "loss_intra": float(details["loss_intra"].item()),
                "loss_inter": float(details["loss_inter"].item()),
                "loss_reg": float(details["loss_reg"].item()),
                "emb_mean": float(details["emb_mean"].item()),
                "student_mean": float(details["student_mean"].item()),
                "teacher_mean": float(details["teacher_mean"].item()),
                "emb_std": float(details["emb_std"].item()),
                "student_std": float(details["student_std"].item()),
                "teacher_std": float(details["teacher_std"].item()),
                "label_positive_fraction": float(labs.float().mean().item()),
                "logit_mean": float(logits.mean().item()),
                "logit_std": float(logits.std().item()),
                "logit_max": float(logits.max().item()),
                "top1_top2_margin": float(margin),
            })

        if base_cos is not None:
            # base_cos: [B, N, N]
            n = base_cos.shape[-1]
            mask = ~torch.eye(n, dtype=torch.bool, device=base_cos.device)  # [N, N]
            offdiag = base_cos[:, mask].float()  # [B, N*(N-1)]
            stats["base_cos_offdiag_mean"] = float(offdiag.mean().item())
            stats["base_cos_offdiag_std"] = float(offdiag.std().item())

    return stats

def train_one_batch(model, opt, scaler, batch, device, epoch, selected_grad_layers,  args: Config):
    model.train()

    img, labels, crops = batch
    img, crops = concat_image(img), concat_image(crops)
    img, crops, labels = to_device(img, crops, labels, device)
    model.train()
    opt.zero_grad(set_to_none=True)

    loss, details = forward_loss(model, img, crops, labels, args.amp, True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    grad_stats = module_grad_stats(model, epoch)
    grad_hists = collect_selected_grad_histograms(
        model,
        selected_names=selected_grad_layers,
        max_elements_per_tensor=20000,
    )
    scaler.step(opt)
    scaler.update()

    stats = create_train_stats(loss, grad_stats, details)

    return loss, stats, grad_hists

def train_one_epoch(model, loader, optimizer, scaler, device, epoch, args: Config) -> float:
    model.train()

    total_loss = 0.0
    n_batches = 0
    sums = defaultdict(float)

    desc = f"Train Epoch {epoch}"
    pbar = tqdm(loader, desc=desc, leave=False)

    selected_grad_layers = pick_monitor_layers(model)
    print("Selected gradient monitor layers:", selected_grad_layers)
    epoch_grad_hist_store = {}

    for batch in pbar:
        loss, stats, grad_hists = train_one_batch(model, optimizer, scaler, batch, device, epoch, selected_grad_layers, args)

        total_loss += float(loss.item())
        for k, v in stats.items():
            sums[k] += float(v)
        n_batches += 1

        avg_loss = total_loss / n_batches
        pbar.set_postfix(
            batch_loss=f"{loss.item():.4f}",
            avg_loss=f"{avg_loss:.4f}",
        )

        for name, arr in grad_hists.items():
            if name not in epoch_grad_hist_store:
                epoch_grad_hist_store[name] = []
            if arr.size > 0:
                epoch_grad_hist_store[name].append(arr)

    if n_batches == 0:
        return 0.0

    epoch_grad_hist_store = {
        name: np.concatenate(chunks, axis=0)
        for name, chunks in epoch_grad_hist_store.items()
        if len(chunks) > 0
    }

    means = {f"train/{k if k != 'loss' else 'loss_total'}": v / n_batches for k, v in sums.items()}
    mean_loss = total_loss / n_batches
    return mean_loss, means, epoch_grad_hist_store


@torch.no_grad()
def validate_one_epoch(model, val_loader, device, epoch, args: Config):
    model.eval()

    total_loss = 0.0
    total_top1 = 0.0
    n_batches = 0

    desc = f"Val Epoch {epoch}"
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

    if n_batches == 0:
        return 0.0, 0.0

    avg_loss = total_loss / n_batches
    avg_top1 = total_top1 / n_batches
    return avg_loss, avg_top1


@torch.no_grad()
def compute_logits_targets_for_one_image(model, batch):
    """
    Reproduces the query-vs-9 logits used by VoCoHead.
    Returns logits, targets for batch item 0: (sw_s, 9), (sw_s, 9)
    """
    model.eval()
    device = next(model.parameters()).device

    img, labels, crops = batch
    img, crops = concat_image(img), concat_image(crops)
    img, crops, labels = to_device(img, crops, labels, device)

    img_t = img.as_tensor() if hasattr(img, "as_tensor") else img
    crops_t = crops.as_tensor() if hasattr(crops, "as_tensor") else crops

    img_t = img_t.to(device)
    crops_t = crops_t.to(device)
    labels = labels.to(device)

    batch_size = labels.size(0)
    total_size = img_t.size(0)
    sw_size = total_size // batch_size
    bases_num = crops_t.size(0) // batch_size  # expected 9

    inputs = torch.cat([img_t, crops_t], dim=0)
    embeddings = model.backbone(inputs)

    # aug_embeddings = torch.nn.Dropout1d(0.2)(embeddings)
    aug_embeddings = embeddings
    student = model.student(aug_embeddings)
    teacher = model.teacher(embeddings)

    x_student, bases_student = student[:total_size], student[total_size:]
    x_teacher, bases_teacher = teacher[:total_size], teacher[total_size:]

    i = 0
    x_stu = x_student[i * sw_size:(i + 1) * sw_size]
    bases_tea = bases_teacher[i * bases_num:(i + 1) * bases_num]

    logits = online_assign(x_stu, bases_tea)   # (sw_s, 9)
    targets = labels[i]                        # (sw_s, 9)

    return logits.detach().cpu(), targets.detach().cpu()