"""
Epistemic Weight Engine (EWE) — Core Gate Implementation
Author: Maheep Purohit, Independent Researcher, Bikaner, India
Paper: https://doi.org/10.5281/zenodo.18940011
ACM TIST Submission: TIST-2026-03-0289
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class EWEGate:
    """
    Epistemic Weight Engine — Pre-update gating mechanism.

    Evaluates each training sample across three independent signals
    before deciding whether it should trigger a parameter update.

    Three Layers:
        I(x) — Impact Assessment: gradient magnitude proxy
        R(x) — Reality Alignment: evidence vs approval signal
        P(x) — Paradigm Shift: loss deviation from moving baseline

    Composite: W(x) = alpha*I + beta*R + gamma*P
    Gate:       W(x) >= mu_W - k * sigma_W  (adaptive threshold)

    Args:
        alpha (float): Weight for Impact module. Default 0.45.
        beta (float): Weight for Reality Alignment module. Default 0.40.
        gamma (float): Weight for Paradigm Shift module. Default 0.15.
        k (float): Adaptive threshold sensitivity. Default 0.25.
                   Higher k = stricter gate = fewer updates.
        tau (float): Smoothing constant for I(x). Default 0.5.
        lam (float): Approval penalty in R(x). Default 0.5.
        eps (float): Stability constant for P(x). Default 0.1.
        ema_decay (float): EMA decay for loss baseline. Default 0.99.

    Example:
        >>> gate = EWEGate()
        >>> for x, y in dataloader:
        ...     optimizer.zero_grad()
        ...     outputs = model(x)
        ...     losses = criterion(outputs, y)
        ...     if gate.should_update(losses, outputs):
        ...         losses.mean().backward()
        ...         optimizer.step()
    """

    def __init__(
        self,
        alpha: float = 0.45,
        beta: float = 0.40,
        gamma: float = 0.15,
        k: float = 0.25,
        tau: float = 0.5,
        lam: float = 0.5,
        eps: float = 0.1,
        ema_decay: float = 0.99,
    ):
        # Validate weights sum to 1
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, \
            f"alpha + beta + gamma must equal 1.0, got {alpha + beta + gamma}"

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k = k
        self.tau = tau
        self.lam = lam
        self.eps = eps
        self.ema_decay = ema_decay

        # State
        self._loss_ema: Optional[float] = None
        self._total: int = 0
        self._accepted: int = 0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of samples accepted by the gate so far."""
        return self._accepted / self._total if self._total > 0 else 0.0

    @property
    def suppression_rate(self) -> float:
        """Fraction of samples routed to passive memory."""
        return 1.0 - self.acceptance_rate

    def reset_stats(self) -> None:
        """Reset acceptance statistics."""
        self._total = 0
        self._accepted = 0

    def compute_scores(
        self,
        losses: torch.Tensor,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute I(x), R(x), P(x) and W(x) for a batch.

        Args:
            losses: Per-sample losses [B] — detached from graph.
            logits: Model output logits [B, C] — detached from graph.

        Returns:
            Tuple of (W, I, R, P) tensors, each shape [B].
        """
        # ── Layer 1: Impact I(x) ──────────────────────────────
        # Normalised loss magnitude — proxy for gradient impact
        I = torch.clamp(losses / (self.tau + losses), 0.0, 1.0)

        # ── Layer 2: Reality Alignment R(x) ──────────────────
        # A(x): approval signal = max softmax confidence
        probs = F.softmax(logits, dim=1)
        A = probs.max(dim=1).values

        # sim(x): evidence consistency = inverse normalised loss
        # Decoupled from A(x) to prevent signal collapse
        loss_norm = losses / (losses.max() + 1e-8)
        sim_x = 1.0 - loss_norm
        R = torch.clamp(sim_x - self.lam * A, min=0.0)

        # ── Layer 3: Paradigm Shift P(x) ─────────────────────
        mean_loss = losses.mean().item()
        if self._loss_ema is None:
            self._loss_ema = mean_loss
        else:
            self._loss_ema = (
                self.ema_decay * self._loss_ema
                + (1 - self.ema_decay) * mean_loss
            )
        P = torch.clamp(
            (losses - self._loss_ema) / (self._loss_ema + self.eps),
            min=0.0, max=1.0
        )

        # ── Composite W(x) ────────────────────────────────────
        W = self.alpha * I + self.beta * R + self.gamma * P

        return W, I, R, P

    def get_mask(
        self,
        losses: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get boolean mask of samples that should trigger parameter updates.

        Args:
            losses: Per-sample losses [B] — detached.
            logits: Model output logits [B, C] — detached.

        Returns:
            Boolean tensor [B]. True = sample accepted for update.
        """
        W, I, R, P = self.compute_scores(losses, logits)

        # Adaptive threshold: accept samples above mean - k*std
        theta = W.mean() - self.k * W.std()
        mask = W >= theta

        self._total += losses.shape[0]
        self._accepted += int(mask.sum().item())

        return mask

    def should_update(
        self,
        losses: torch.Tensor,
        logits: torch.Tensor,
    ) -> bool:
        """
        Batch-level gate decision.
        Returns True if at least one sample in the batch should update.

        Args:
            losses: Per-sample losses [B] — detached.
            logits: Model output logits [B, C] — detached.

        Returns:
            True if any sample passes the gate.
        """
        mask = self.get_mask(losses, logits)
        return bool(mask.any().item())

    def filter_losses(
        self,
        losses: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Filter losses to only include accepted samples.
        Returns mean loss over accepted samples for backpropagation.

        Args:
            losses: Per-sample losses [B] — with gradient.
            logits: Model output logits [B, C] — detached.

        Returns:
            Scalar loss for accepted samples, or None if none accepted.
        """
        mask = self.get_mask(losses.detach(), logits.detach())
        if mask.sum() == 0:
            return None
        return losses[mask].mean()

    def __repr__(self) -> str:
        return (
            f"EWEGate("
            f"alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}, "
            f"k={self.k}, lam={self.lam}, "
            f"acceptance_rate={self.acceptance_rate:.1%})"
        )
