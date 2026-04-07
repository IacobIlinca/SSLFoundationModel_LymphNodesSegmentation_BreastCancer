import math
from typing import Dict, List

import numpy as np
import torch

def grad_norm(parameters, epoch):
    total = 0.0
    count = 0
    nan_tensors = 0
    inf_tensors = 0

    for name, p in parameters:
        if p.grad is not None:
            g = p.grad.detach()
            if torch.isnan(g).any():
                nan_tensors += 1
                print("[WARN] Epoch: ", epoch, " --- Nan grad in:", name, "shape:", tuple(p.grad.shape))
                continue
            if torch.isinf(g).any():
                inf_tensors += 1
                continue
            total += g.norm(2).item() ** 2
            count += 1

    return math.sqrt(total), count, nan_tensors, inf_tensors

def param_norm(parameters):
    total = 0.0
    count = 0
    for p in parameters:
        d = p.detach()
        total += d.norm(2).item() ** 2
        count += 1
    return math.sqrt(total), count

def module_grad_stats(model, epoch):
    bb_grad, bb_count, bb_nan, bb_inf = grad_norm(model.backbone.named_parameters(), epoch)
    stu_grad, stu_count, stu_nan, stu_inf = grad_norm(model.student.named_parameters(), epoch)
    tea_grad, tea_count, tea_nan, tea_inf = grad_norm(model.teacher.named_parameters(), epoch)
    return {
        "grad_backbone": bb_grad,
        "grad_student": stu_grad,
        "grad_teacher": tea_grad,
        "grad_backbone_nan_tensors": bb_nan,
        "grad_backbone_inf_tensors": bb_inf,
        "grad_student_nan_tensors": stu_nan,
        "grad_student_inf_tensors": stu_inf,
        "grad_teacher_nan_tensors": tea_nan,
        "grad_teacher_inf_tensors": tea_inf,
    }


def get_named_gradient_samples(
    named_parameters,
    max_elements_per_tensor: int = 20000,
) -> Dict[str, np.ndarray]:
    """
    Collect flattened gradient samples for selected parameters.

    Returns:
        dict: param_name -> 1D numpy array of finite gradient values
    """
    out = {}

    for name, p in named_parameters:
        if p.grad is None:
            continue

        g = p.grad.detach().float().view(-1)

        # keep only finite values for histogram plotting
        finite_mask = torch.isfinite(g)
        g = g[finite_mask]

        if g.numel() == 0:
            out[name] = np.array([], dtype=np.float32)
            continue

        if g.numel() > max_elements_per_tensor:
            idx = torch.randperm(g.numel(), device=g.device)[:max_elements_per_tensor]
            g = g[idx]

        out[name] = g.cpu().numpy()

    return out


def summarize_gradient_health(named_parameters) -> Dict[str, int]:
    """
    Count how many parameter tensors have finite / nan / inf gradients.
    """
    stats = {
        "num_with_grad": 0,
        "num_nan_tensors": 0,
        "num_inf_tensors": 0,
        "num_all_zero_tensors": 0,
    }

    for _, p in named_parameters:
        if p.grad is None:
            continue

        stats["num_with_grad"] += 1
        g = p.grad.detach()

        if torch.isnan(g).any():
            stats["num_nan_tensors"] += 1
        if torch.isinf(g).any():
            stats["num_inf_tensors"] += 1
        if torch.count_nonzero(g) == 0:
            stats["num_all_zero_tensors"] += 1

    return stats


def pick_monitor_layers(model) -> Dict[str, List[str]]:
    """
    Choose a few representative parameter names from backbone/student/teacher.
    Edit this if you want more specific layers.
    """
    def pick(names: List[str]) -> List[str]:
        if len(names) == 0:
            return []
        if len(names) <= 3:
            return names
        return names[::3]

    bb_names = [n for n, p in model.backbone.named_parameters() if p.requires_grad]
    #stu_names = [n for n, p in model.student.named_parameters() if p.requires_grad]
    #tea_names = [n for n, p in model.teacher.named_parameters() if p.requires_grad]

    return {
        "backbone": pick(bb_names),
        #"student": pick(stu_names),
        #"teacher": pick(tea_names),
    }


def collect_selected_grad_histograms(
    model,
    selected_names: Dict[str, List[str]],
    max_elements_per_tensor: int = 20000,
) -> Dict[str, np.ndarray]:
    """
    Collect gradient samples only for selected parameter names.

    Output keys are prefixed with module name, e.g.
    backbone.encoder.layer1.weight
    """
    out = {}

    for module_name in ["backbone", "student", "teacher"]:
        module = getattr(model, module_name)
        wanted = set(selected_names.get(module_name, []))

        named_params = []
        for name, p in module.named_parameters():
            if name in wanted:
                named_params.append((f"{module_name}.{name}", p))

        out.update(
            get_named_gradient_samples(
                named_params,
                max_elements_per_tensor=max_elements_per_tensor,
            )
        )

    return out