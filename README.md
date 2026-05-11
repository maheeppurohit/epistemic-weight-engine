# Epistemic Weight Engine (EWE)

[![PyPI version](https://badge.fury.io/py/ewe-gate.svg)](https://badge.fury.io/py/ewe-gate)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/paper-Zenodo-green.svg)](https://doi.org/10.5281/zenodo.18940011)

**Pre-update gating mechanism for noise-robust neural network training.**

EWE intercepts the gradient-to-parameter pathway and applies a binary accept/reject decision before any weight modification occurs — addressing both the **Uniform Weighting Problem** (all samples treated equally regardless of label reliability) and the **Approval Bias Problem** (models trained on human feedback optimise for rater approval rather than accuracy).

> **Submitted to IEEE Transactions on Neural Networks and Learning Systems (TNNLS), May 2026.**  
> Preprint: [doi.org/10.5281/zenodo.18940011](https://doi.org/10.5281/zenodo.18940011)

---

## Installation

```bash
pip install ewe-gate
```

---

## Quick Start

```python
import torch
import torch.nn.functional as F
from ewe import EWEGate

# Adaptive threshold — no manual tuning needed
gate = EWEGate(num_classes=10)

# In your training loop
for x, y in dataloader:
    optimizer.zero_grad()
    outputs = model(x)
    losses  = F.cross_entropy(outputs, y, reduction='none')

    # Gate decides which samples are allowed to update weights
    mask = gate(losses.detach(), outputs.detach())

    if mask.sum() > 0:
        losses[mask].mean().backward()
        optimizer.step()

print(f"Acceptance rate: {gate.acceptance_rate:.1%}")
```

---

## How It Works

EWE evaluates each training sample through three modules:

| Module | Formula | Purpose |
|--------|---------|---------|
| Impact Assessment I(x) | L(x) / (τ + L(x)) | Gradient significance |
| Reality Alignment R(x) | max(0, sim(x) − λ·A(x)) | Label-evidence consistency |
| Paradigm Shift P(x) | max(0, (L(x) − L_EMA) / (L_EMA + ε)) | Informational novelty |

**Composite score:** W(x) = 0.45·I(x) + 0.40·R(x) + 0.15·P(x)

**Gate decision:** G(x) = 1 if W(x) ≥ μ_W − k·σ_W

**Adaptive threshold:** k* = 0.25 / log(C + 1) where C = number of classes

---

## Results

### CIFAR-10N (ResNet-18, 5 seeds, 40.2% real human annotation noise)

| Method | Accuracy | Std | vs. Standard | Networks |
|--------|----------|-----|-------------|---------|
| Standard CE | 72.37% | ±0.16% | — | 1 |
| GCE | 81.74% | ±0.20% | +9.37% | 1 |
| Co-teaching | 85.46% | ±0.06% | +13.09% | 2 |
| DivideMix | 92.42% | ±0.12% | +20.05% | 2 |
| **Adaptive EWE+GCE** | **91.95%** | **±0.06%** | **+19.58%** | **1** |

Adaptive EWE+GCE achieves within **0.47%** of DivideMix using **one network** at **half the computational cost**.

### CIFAR-100N (ResNet-34, 5 seeds, 40.2% real human annotation noise)

| Method | Accuracy | Std | vs. Standard |
|--------|----------|-----|-------------|
| Standard CE | 54.66% | ±0.70% | — |
| GCE | 58.40% | ±0.42% | +3.74% |
| Co-teaching | 64.16% | ±0.60% | +9.50% |
| DivideMix | 63.01% | ±0.37% | +8.35% |
| **Adaptive EWE+GCE** | **58.90%** | **±0.37%** | **+4.24%** |

### Synthetic Noise Levels (CIFAR-10, ResNet-18, 3 seeds)

| Method | 20% | 40% | 60% | 80% |
|--------|-----|-----|-----|-----|
| Standard CE | 81.74% | 63.15% | 41.81% | 43.57% |
| GCE | 92.21% | 87.85% | 75.33% | 42.73% |
| **Adaptive EWE** | 84.20% | **88.38%** | **83.26%** | 25.23% |

EWE achieves best single-network results at **40% and 60% noise** — the range typical of real-world annotation.

### Approval Bias (DistilBERT, IMDb, 3 seeds)

| Condition | Accuracy |
|-----------|---------|
| Clean labels | 84.97% |
| 30% approval bias | 80.43% |
| **Bias damage** | **−4.53%** |

---

## Advanced Usage

### EWE combined with GCE loss

```python
from ewe import EWEGate
from ewe.losses import gce_loss

gate = EWEGate(num_classes=10)

for x, y in dataloader:
    optimizer.zero_grad()
    outputs = model(x)
    losses  = gce_loss(outputs, y, q=0.7, reduction='none')
    mask    = gate(losses.detach(), outputs.detach())
    if mask.sum() > 0:
        losses[mask].mean().backward()
        optimizer.step()
```

### Using EWETrainer (high-level wrapper)

```python
from ewe import EWETrainer

trainer = EWETrainer(
    model=model,
    optimizer=optimizer,
    num_classes=10,
    use_gce=True,     # EWE+GCE
    warmup=10,        # 10 epoch warmup
)

for epoch in range(1, 101):
    for x, y in dataloader:
        trainer.step(x, y, epoch)

print(trainer)
```

### Manual threshold (override adaptive)

```python
# Manual k — use when you want explicit control
gate = EWEGate(num_classes=10, k=0.25)

# Adaptive k — recommended for most cases
gate = EWEGate(num_classes=10)         # k = 0.25/log(11) = 0.104
gate = EWEGate(num_classes=100)        # k = 0.25/log(101) = 0.054
```

### Check adaptive k for your dataset

```python
from ewe import adaptive_k

print(adaptive_k(10))    # 0.1043 — CIFAR-10
print(adaptive_k(100))   # 0.0543 — CIFAR-100
print(adaptive_k(1000))  # 0.0362 — ImageNet
print(adaptive_k(2))     # 0.2276 — Binary classification
```

---

## Experiment Code

All experiments from the paper are available in the `experiments/` folder:

| File | Description |
|------|-------------|
| `train.py` | CIFAR-10N manual EWE (3 seeds) |
| `train_adaptive_ewe.py` | Adaptive EWE — CIFAR-10N + CIFAR-100N |
| `train_cifar100n.py` | CIFAR-100N manual EWE (5 seeds) |
| `train_dividemix.py` | DivideMix baseline |
| `train_synthetic_noise.py` | Synthetic noise levels 20/40/60/80% |
| `train_lm_approval_bias.py` | Language model approval bias |
| `train_ablation.py` | Neural network ablation study |

### Run CIFAR-10N experiment

```bash
# Download CIFAR-10N labels
# https://github.com/UCSC-REAL/cifar-10-100n

python experiments/train_adaptive_ewe.py
```

---

## Theoretical Guarantee

**Proposition 1 (Gradient Corruption Bound):** Under label noise rate ε ∈ [0, 0.5), there exists δ > 0 such that:

```
E[‖∇_EWE − ∇_clean‖] / E[‖∇_std − ∇_clean‖] ≤ 1 − δ
```

EWE strictly reduces expected gradient corruption for any noise rate below 50%. The reduction is greatest at 20–60% noise — the range most common in real-world annotation.

---

## Citation

```bibtex
@article{purohit2026ewe,
  title   = {Epistemic Weight Engine ({EWE}): Adaptive Pre-Update Gating
             for Noise-Robust Learning Under Real-World Label Noise},
  author  = {Purohit, Maheep},
  journal = {arXiv preprint},
  year    = {2026},
  doi     = {10.5281/zenodo.18940011},
  url     = {https://doi.org/10.5281/zenodo.18940011}
}
```

*Citation will be updated upon IEEE TNNLS publication.*

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Author

**Maheep Purohit**  
Independent Researcher, Bikaner, Rajasthan, India  
[purohitmaheep@gmail.com](mailto:purohitmaheep@gmail.com)  
ORCID: [0009-0003-4739-6786](https://orcid.org/0009-0003-4739-6786)
