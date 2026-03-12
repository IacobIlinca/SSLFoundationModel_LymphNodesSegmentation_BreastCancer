import torch
from monai.data.meta_tensor import MetaTensor

from src.VocoLarge.third_party_voco_large.models.voco_head import online_assign


def unpack_voco_output(out):
    """
    Accepts output from your MONAI/VoCo transform pipeline.
    Returns:
      img:    (sw_s, 1, D, H, W) torch.Tensor
      crops:  (9,    1, D, H, W) torch.Tensor
      labels: (1, sw_s, 9) torch.Tensor OR (sw_s,9) depending on transform
    """
    if isinstance(out, (list, tuple)):
        img_obj, lab_obj, crops_obj = out
    elif isinstance(out, dict):
        img_obj = out.get("image", None)
        crops_obj = out.get("crops", None)
        lab_obj = out.get("labels", None)
    else:
        raise RuntimeError(f"Unexpected output type: {type(out)}")

    if img_obj is None or crops_obj is None or lab_obj is None:
        raise RuntimeError("Transform did not return expected outputs (image/crops/labels).")

    img = torch.stack([d["image"] for d in img_obj], dim=0)
    crops = torch.stack([d["image"] for d in crops_obj], dim=0)

    if img.ndim == 6 and crops.ndim == 6:
        img = img.squeeze(1)
        crops = crops.squeeze(1)

    labels = torch.as_tensor(lab_obj, dtype=torch.float32)
    if labels.ndim == 2:
        labels = labels.unsqueeze(0)
    if labels.ndim != 3:
        raise ValueError(f"Unexpected labels shape: {tuple(labels.shape)}")

    return img, crops, labels


def to_device(img, crops, labels, device: torch.device):
    """
    VoCoHead.forward expects MetaTensor-like with .as_tensor()
    """
    img = MetaTensor(img).to(device)
    crops = MetaTensor(crops).to(device)
    labels = labels.to(device)
    return img, crops, labels


def forward_loss(model, img, crops, labels, use_amp: bool):
    """
    One forward that returns scalar loss.
    """
    if use_amp and next(model.parameters()).is_cuda:
        with torch.cuda.amp.autocast(True):
            loss = model(img, crops, labels)
    else:
        loss = model(img, crops, labels)
    return loss


@torch.no_grad()
def compute_logits_targets(model, img, crops, labels):
    """
    Reproduces the query-vs-9 logits used by VoCoHead.
    Returns logits, targets for batch item 0: (sw_s, 9), (sw_s, 9)
    """
    model.eval()

    img_t = img.as_tensor() if hasattr(img, "as_tensor") else img
    crops_t = crops.as_tensor() if hasattr(crops, "as_tensor") else crops

    device = next(model.parameters()).device
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