"""
EWE Trainer
===========
High-level training wrapper with EWEGate integration.
"""

from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from .gate import EWEGate, adaptive_k
from .losses import gce_loss


class EWETrainer:
    """
    Training wrapper that integrates EWEGate with any PyTorch model.

    Supports:
    - Standard EWE (cross-entropy loss + EWE gate)
    - EWE+GCE (GCE loss + EWE gate)
    - Warmup period before gate activation

    Args:
        model:       PyTorch model to train.
        optimizer:   PyTorch optimizer.
        num_classes: Number of output classes.
        k:           Gate threshold. None = adaptive. Default None.
        use_gce:     Use GCE loss instead of CE. Default False.
        warmup:      Epochs of standard training before gate. Default 10.
        gce_q:       GCE interpolation parameter. Default 0.7.
        device:      Device to train on. Default 'cuda' if available.

    Example:
        >>> trainer = EWETrainer(model, optimizer, num_classes=10)
        >>> for epoch in range(100):
        ...     for x, y in dataloader:
        ...         trainer.step(x, y, epoch)
        >>> print(f"Final acceptance rate: {trainer.gate.acceptance_rate:.1%}")
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        num_classes: int = 10,
        k: Optional[float] = None,
        use_gce: bool = False,
        warmup: int = 10,
        gce_q: float = 0.7,
        device: Optional[str] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.use_gce = use_gce
        self.warmup = warmup
        self.gce_q = gce_q

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.gate = EWEGate(num_classes=num_classes, k=k)
        self._current_epoch = 0

    # ------------------------------------------------------------------
    def step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        epoch: int,
    ) -> Optional[float]:
        """
        Run one training step on a mini-batch.

        Args:
            x:     Input batch, shape (N, ...).
            y:     Target labels, shape (N,).
            epoch: Current epoch number (1-indexed).

        Returns:
            Loss value as float, or None if all samples rejected.
        """
        x = x.to(self.device)
        y = y.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(x)

        in_warmup = epoch <= self.warmup

        if in_warmup:
            loss = F.cross_entropy(outputs, y)
            loss.backward()
            self.optimizer.step()
            return loss.item()

        if self.use_gce:
            losses = gce_loss(outputs, y, q=self.gce_q, reduction="none")
        else:
            losses = F.cross_entropy(outputs, y, reduction="none")

        mask = self.gate(losses.detach(), outputs.detach())

        if mask.sum() == 0:
            return None

        loss = losses[mask].mean()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    # ------------------------------------------------------------------
    @property
    def acceptance_rate(self) -> float:
        """Current gate acceptance rate."""
        return self.gate.acceptance_rate

    def __repr__(self) -> str:
        return (
            f"EWETrainer(num_classes={self.num_classes}, "
            f"use_gce={self.use_gce}, warmup={self.warmup}, "
            f"gate={self.gate})"
        )
