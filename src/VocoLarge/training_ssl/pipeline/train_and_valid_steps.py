import torch
from tqdm.auto import tqdm

from src.VocoLarge.third_party_voco_large.models.voco_head import online_assign, ce_loss, regularization_loss
from src.VocoLarge.third_party_voco_large.utils.ops import concat_image
from src.VocoLarge.training_ssl.pipeline import forward_loss, to_device, top1_match
from src.VocoLarge.training_ssl.pipeline.config import Config


def train_one_batch(model, opt, scaler, batch, device, args: Config):
    model.train()

    img, labels, crops = batch
    img, crops = concat_image(img), concat_image(crops)
    img, crops, labels = to_device(img, crops, labels, device)
    model.train()
    opt.zero_grad(set_to_none=True)

    loss = forward_loss(model, img, crops, labels, args.amp)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()

    return loss

def train_one_epoch(model, loader, optimizer, scaler, device, epoch, args: Config) -> float:
    model.train()

    total_loss = 0.0
    n_batches = 0

    desc = f"Train Epoch {epoch}"
    pbar = tqdm(loader, desc=desc, leave=False)

    for batch in pbar:
        loss = train_one_batch(model, optimizer, scaler, batch, device, args)

        total_loss += float(loss.item())
        n_batches += 1

        avg_loss = total_loss / n_batches
        pbar.set_postfix(
            batch_loss=f"{loss.item():.4f}",
            avg_loss=f"{avg_loss:.4f}",
        )

    if n_batches == 0:
        return 0.0

    return total_loss / n_batches


@torch.no_grad()
def validate_one_epoch(model, val_loader, device, epoch):
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

        loss, top1 = validate_one_batch(model, img, crops, labels)
        total_loss += float(loss.item())
        total_top1 += top1

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
def validate_one_batch(model, img, crops, labels):
    model.eval()

    batch_size = labels.size()[0]
    total_size = img.size()[0]
    sw_size = total_size // batch_size

    # loss accumulate
    intra, inter, total_b_loss, top1 = 0.0, 0.0, 0.0, 0.0

    img, crops = img.as_tensor(), crops.as_tensor()
    inputs = torch.cat([img, crops], dim=0)

    # here we do norm on all instances
    embeddings = model.backbone(inputs)

    # feature augmentation
    # aug_embeddings = nn.Dropout1d(0.2)(embeddings)
    aug_embeddings = embeddings
    student = model.student(aug_embeddings)
    teacher = model.teacher(embeddings)

    x_student, bases_student = student[:total_size], student[total_size:]
    x_teacher, bases_teacher = teacher[:total_size], teacher[total_size:]

    for i in range(batch_size):
        label = labels[i]
        bases_num = 9

        x_stu, bases_stu = x_student[i * sw_size:(i + 1) * sw_size], bases_student[i * bases_num:(i + 1) * bases_num]
        x_tea, bases_tea = x_teacher[i * sw_size:(i + 1) * sw_size], bases_teacher[i * bases_num:(i + 1) * bases_num]
        logits = online_assign(x_stu, bases_tea)

        top1_metric = top1_match(logits.detach().cpu(), label.detach().cpu())
        top1 += top1_metric

        # if i == 0:
        #     print('labels and logits:', label[0].data, logits[0].data)

        intra_loss = ce_loss(label, logits)
        intra += intra_loss

        # teacher bases for inter volume contrast
        # j: different case
        j = (i + 1) % batch_size
        inter_bases_stu = bases_student[j * bases_num:(j + 1) * bases_num]
        inter_bases_tea = bases_teacher[j * bases_num:(j + 1) * bases_num]

        inter_loss = model.inter_volume(x_stu, x_tea, inter_bases_stu, inter_bases_tea)
        inter += inter_loss

        b_loss = regularization_loss(bases_stu)
        total_b_loss += b_loss

    intra = intra / batch_size
    inter = inter / batch_size
    total_b_loss = total_b_loss / batch_size
    top1 = top1 / batch_size

    loss = intra + inter + total_b_loss
    return loss, top1

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