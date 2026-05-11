"""
Epistemic Weight Engine (EWE) — Core Gate Module
=================================================
Pre-update gating mechanism for noise-robust neural network training.

Paper: "Epistemic Weight Engine (EWE): Adaptive Pre-Update Gating
        for Noise-Robust Learning Under Real-World Label Noise"
       Submitted to IEEE TNNLS, 2026.

Preprint: https://doi.org/10.5281/zenodo.18940011
GitHub:   https://github.com/maheeppurohit/epistemic-weight-engine

Usage
-----
    from ewe import EWEGate

    gate = EWEGate(num_classes=10)          # adaptive k
    gate = EWEGate(num_classes=10, k=0.25)  # manual k

    # In training loop:
    outputs = model(x)
    losses  = F.cross_entropy(outputs, y, reduction='none')
    mask    = gate(losses.detach(), outputs.detach())
    if mask.sum() > 0:
        losses[mask].mean().backward()
        optimizer.step()
"""

import math
import torch
import torch.nn.functional as F


def adaptive_k(num_classes: int, base_k: float = 0.25) -> float:
    """
    Compute adaptive gate threshold parameter from class count.

    Formula: k* = base_k / log(num_classes + 1)

    Motivated by information theory: the entropy of a uniform
    distribution over C classes scales as log(C), so the gate
    threshold should loosen proportionally as class complexity grows.

    Args:
        num_classes: Number of output classes (C).
        base_k:      Universal base parameter. Default 0.25.

    Returns:
        Adaptive threshold k* in (0, base_k].

    Examples:
        >>> adaptive_k(10)   # CIFAR-10
        0.1043
        >>> adaptive_k(100)  # CIFAR-100
        0.0543
        >>> adaptive_k(2)    # Binary classification
        0.2276
    """
    return base_k / math.log(num_classes + 1)


class EWEGate:
    """
    Epistemic Weight Engine gate.

    Evaluates each training sample through three modules before
    deciding whether to allow a parameter update:

    1. Impact Assessment  I(x) — gradient magnitude proxy
    2. Reality Alignment  R(x) — label-evidence consistency
    3. Paradigm Shift     P(x) — informational novelty

    Composite score: W(x) = alpha*I(x) + beta*R(x) + gamma*P(x)
    Gate decision:   G(x) = 1 if W(x) >= mu_W - k * sigma_W

    The threshold k is set automatically via adaptive_k(num_classes)
    or can be provided manually.

    Args:
        num_classes: Number of output classes. Used for adaptive k.
        k:           Gate threshold parameter. If None, uses
                     adaptive_k(num_classes). Default None.
        alpha:       Weight for Impact Assessment. Default 0.45.
        beta:        Weight for Reality Alignment. Default 0.40.
        gamma:       Weight for Paradigm Shift. Default 0.15.
        tau:         Stability constant for I(x). Default 0.5.
        lam:         Approval penalty in R(x). Default 0.40.
        eps:         Stability constant for P(x). Default 0.1.
        ema_decay:   EMA decay rate for loss tracking. Default 0.99.

    Example:
        >>> import torch
        >>> import torch.nn.functional as F
        >>> gate = EWEGate(num_classes=10)
        >>> outputs = model(x)
        >>> losses = F.cross_entropy(outputs, y, reduction='none')
        >>> mask = gate(losses.detach(), outputs.detach())
        >>> if mask.sum() > 0:
        ...     losses[mask].mean().backward()
        ...     optimizer.step()
    """

    def __init__(
        self,
        num_classes: int = 10,
        k: float = None,
        alpha: float = 0.45,
        beta: float = 0.40,
        gamma: float = 0.15,
        tau: float = 0.5,
        lam: float = 0.40,
        eps: float = 0.1,
        ema_decay: float = 0.99,
    ):
        self.num_classes = num_classes
        self.k = k if k is not None else adaptive_k(num_classes)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.lam = lam
        self.eps = eps
        self.ema_decay = ema_decay

        self._loss_ema: float = None
        self._total: int = 0
        self._accepted: int = 0

        self._adaptive = k is None

    # ------------------------------------------------------------------
    @property
    def acceptance_rate(self) -> float:
        """Fraction of samples accepted so far."""
        return self._accepted / self._total if self._total > 0 else 0.0

    @property
    def is_adaptive(self) -> bool:
        """True if k was set automatically via adaptive_k."""
        return self._adaptive

    def reset_stats(self):
        """Reset acceptance rate counters."""
        self._total = 0
        self._accepted = 0
        self._loss_ema = None

    # ------------------------------------------------------------------
    def _impact(self, losses: torch.Tensor) -> torch.Tensor:
        """
        Impact Assessment I(x) = L(x) / (tau + L(x)).
        Measures gradient significance. Output in [0, 1].
        """
        return torch.clamp(losses / (self.tau + losses), 0.0, 1.0)

    def _reality(
        self,
        losses: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reality Alignment R(x) = max(0, sim(x) - lambda * A(x)).
        Penalises updates where model confidence conflicts with
        label-evidence consistency (approval bias signature).
        """
        probs = F.softmax(logits, dim=1)
        A = probs.max(dim=1).values                          # approval proxy
        sim = 1.0 - losses / (losses.max() + 1e-8)          # consistency
        return torch.clamp(sim - self.lam * A, min=0.0)

    def _paradigm(self, losses: torch.Tensor) -> torch.Tensor:
        """
        Paradigm Shift P(x) = max(0, (L(x) - L_EMA) / (L_EMA + eps)).
        Identifies samples carrying genuinely novel information.
        """
        mean_loss = losses.mean().item()
        if self._loss_ema is None:
            self._loss_ema = mean_loss
        else:
            self._loss_ema = (
                self.ema_decay * self._loss_ema
                + (1.0 - self.ema_decay) * mean_loss
            )
        return torch.clamp(
            (losses - self._loss_ema) / (self._loss_ema + self.eps),
            min=0.0,
        )

    # ------------------------------------------------------------------
    def __call__(
        self,
        losses: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gate mask for a mini-batch.

        Args:
            losses: Per-sample losses, shape (N,). Should be detached.
            logits: Model output logits, shape (N, C). Should be detached.

        Returns:
            Boolean mask of shape (N,). True = accepted (update allowed).
        """
        I = self._impact(losses)
        R = self._reality(losses, logits)
        P = self._paradigm(losses)

        W = self.alpha * I + self.beta * R + self.gamma * P
        theta = W.mean() - self.k * W.std()
        mask = W >= theta

        self._total += losses.shape[0]
        self._accepted += int(mask.sum().item())

        return mask

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        mode = f"adaptive k={self.k:.4f}" if self._adaptive \
               else f"manual k={self.k:.4f}"
        return (
            f"EWEGate(num_classes={self.num_classes}, {mode}, "
            f"alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}, "
            f"acceptance_rate={self.acceptance_rate:.1%})"
        )
