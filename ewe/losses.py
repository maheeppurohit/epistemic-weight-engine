"""
EWE Loss Functions
==================
Noise-robust loss functions for use with EWEGate.
"""

import torch
import torch.nn.functional as F


def gce_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    q: float = 0.7,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Generalised Cross-Entropy loss (Zhang & Sabuncu, NeurIPS 2018).

    Interpolates between cross-entropy (q→0) and MAE (q=1).
    Naturally robust to label noise.

    Args:
        outputs:   Model logits, shape (N, C).
        targets:   Target class indices, shape (N,).
        q:         Interpolation parameter in (0, 1]. Default 0.7.
        reduction: 'mean', 'sum', or 'none'. Default 'mean'.

    Returns:
        Scalar loss (reduction='mean'/'sum') or per-sample loss (='none').
    """
    probs = F.softmax(outputs, dim=1)
    probs_y = probs[range(len(targets)), targets].clamp(min=1e-7)
    loss = (1.0 - probs_y ** q) / q
    stab = 1e-4 * F.cross_entropy(outputs, targets)

    if reduction == "none":
        return loss + stab
    elif reduction == "sum":
        return loss.sum() + stab
    return loss.mean() + stab


def label_smoothing_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """
    Label Smoothing cross-entropy loss (Szegedy et al., CVPR 2016).

    Args:
        outputs:     Model logits, shape (N, C).
        targets:     Target class indices, shape (N,).
        num_classes: Number of output classes C.
        smoothing:   Smoothing coefficient. Default 0.1.

    Returns:
        Scalar mean loss.
    """
    confidence = 1.0 - smoothing
    smooth_val = smoothing / (num_classes - 1)
    one_hot = torch.zeros_like(outputs).scatter_(
        1, targets.unsqueeze(1), 1
    )
    smooth_one_hot = one_hot * confidence + (1 - one_hot) * smooth_val
    log_prob = F.log_softmax(outputs, dim=1)
    return -(smooth_one_hot * log_prob).sum(dim=1).mean()
