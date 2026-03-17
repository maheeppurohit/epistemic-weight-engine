"""
EWETrainer — High-level trainer that wraps any PyTorch model with EWE.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Callable
from .gate import EWEGate


class EWETrainer:
    """
    High-level trainer that wraps any PyTorch model with EWE gating.

    Handles the training loop automatically. Just pass your model,
    optimizer, and dataloader.

    Args:
        model (nn.Module): Your PyTorch model.
        optimizer: PyTorch optimizer.
        criterion: Loss function. Must support reduction='none'.
        gate (EWEGate): EWE gate instance. Creates default if None.
        device (str): Device to train on. Auto-detects if None.

    Example:
        >>> model = ResNet18()
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>> criterion = nn.CrossEntropyLoss(reduction='none')
        >>> trainer = EWETrainer(model, optimizer, criterion)
        >>> for epoch in range(50):
        ...     loss, acc, rate = trainer.train_epoch(dataloader)
        ...     print(f"Epoch {epoch} | Loss: {loss:.3f} | Accept: {rate:.1%}")
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        criterion: nn.Module,
        gate: Optional[EWEGate] = None,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.gate = gate if gate is not None else EWEGate()
        self.device = device

    def train_epoch(
        self,
        dataloader: DataLoader,
    ):
        """
        Train for one epoch with EWE gating.

        Args:
            dataloader: DataLoader. Items should be (images, labels)
                        or (images, labels, indices).

        Returns:
            Tuple of (mean_loss, accuracy, acceptance_rate).
        """
        self.model.train()
        total_loss = correct = total = 0

        for batch in dataloader:
            # Support dataloaders that return 2 or 3 items
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch

            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(x)

            # Compute per-sample losses
            try:
                losses = self.criterion(outputs, y)
            except Exception:
                # Fallback if criterion doesn't support reduction='none'
                base = nn.CrossEntropyLoss(reduction='none')
                losses = base(outputs, y)

            # EWE gate decision
            filtered_loss = self.gate.filter_losses(
                losses, outputs.detach()
            )

            if filtered_loss is not None:
                filtered_loss.backward()
                self.optimizer.step()

            total_loss += losses.mean().item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)

        mean_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        accept_rate = self.gate.acceptance_rate

        return mean_loss, accuracy, accept_rate

    def evaluate(self, dataloader: DataLoader) -> float:
        """
        Evaluate model accuracy on a dataloader.

        Args:
            dataloader: DataLoader with (images, labels) pairs.

        Returns:
            Accuracy as percentage.
        """
        self.model.eval()
        correct = total = 0

        with torch.no_grad():
            for batch in dataloader:
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)
                _, predicted = self.model(x).max(1)
                correct += predicted.eq(y).sum().item()
                total += y.size(0)

        return 100.0 * correct / total

    def __repr__(self) -> str:
        return (
            f"EWETrainer(\n"
            f"  model={self.model.__class__.__name__},\n"
            f"  device={self.device},\n"
            f"  gate={self.gate}\n"
            f")"
        )
