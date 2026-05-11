"""
ewe-gate: Epistemic Weight Engine for noise-robust learning.

Pre-update gating mechanism that addresses label noise and
approval bias in neural network training.

    pip install ewe-gate

Quick start:
    from ewe import EWEGate

    gate = EWEGate(num_classes=10)   # adaptive threshold
    mask = gate(losses, logits)       # True = accepted
"""

from .gate import EWEGate, adaptive_k
from .losses import gce_loss, label_smoothing_loss
from .trainer import EWETrainer

__version__ = "0.2.0"
__author__ = "Maheep Purohit"
__email__ = "purohitmaheep@gmail.com"
__license__ = "MIT"

__all__ = [
    "EWEGate",
    "adaptive_k",
    "gce_loss",
    "label_smoothing_loss",
    "EWETrainer",
]
