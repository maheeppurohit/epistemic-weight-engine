"""
Loss functions compatible with EWE.
Can be used standalone or combined with EWEGate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCELoss(nn.Module):
    """
    Generalised Cross-Entropy Loss.
    Zhang & Sabuncu, NeurIPS 2018.

    Noise-robust loss that interpolates between
    cross-entropy (q→0) and MAE (q=1).

    Args:
        q (float): Robustness parameter. Default 0.7.
                   Higher q = more robust to label noise.
        num_classes (int): Number of classes. Default 10.

    Example:
        >>> criterion = GCELoss(q=0.7)
        >>> loss = criterion(logits, labels)
    """

    def __init__(self, q: float = 0.7, num_classes: int = 10):
        super().__init__()
        self.q = q
        self.num_classes = num_classes

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        reduction: str = "mean"
    ) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        probs_y = probs[torch.arange(len(labels)), labels].clamp(min=1e-7)
        loss = (1 - probs_y ** self.q) / self.q

        # Stabilisation term prevents gradient collapse
        stab = 1e-4 * F.cross_entropy(logits, labels)

        if reduction == "none":
            return loss + stab
        return loss.mean() + stab

    def __repr__(self) -> str:
        return f"GCELoss(q={self.q})"


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss.
    Szegedy et al., CVPR 2016.

    Reduces overconfidence by softening hard targets.

    Args:
        smoothing (float): Smoothing factor. Default 0.1.
        num_classes (int): Number of classes. Default 10.

    Example:
        >>> criterion = LabelSmoothingLoss(smoothing=0.1)
        >>> loss = criterion(logits, labels)
    """

    def __init__(self, smoothing: float = 0.1, num_classes: int = 10):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        conf = 1.0 - self.smoothing
        smooth_val = self.smoothing / (self.num_classes - 1)
        one_hot = torch.zeros_like(logits).scatter_(
            1, labels.unsqueeze(1), 1
        )
        smooth_oh = one_hot * conf + (1 - one_hot) * smooth_val
        return -(smooth_oh * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

    def __repr__(self) -> str:
        return f"LabelSmoothingLoss(smoothing={self.smoothing})"
