# Epistemic Weight Engine (EWE)

> A modular pre-update gating mechanism for signal-reliability-weighted learning in AI systems.

[![PyPI](https://img.shields.io/pypi/v/ewe-gate)](https://pypi.org/project/ewe-gate/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-ACM%20TIST-blue)](https://doi.org/10.5281/zenodo.18940011)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://pypi.org/project/ewe-gate/)

**Author:** Maheep Purohit — Independent Researcher, Bikaner, Rajasthan, India  
**Paper:** Submitted to ACM Transactions on Intelligent Systems and Technology (TIST-2026-03-0289)  
**Preprint DOI:** [10.5281/zenodo.18940011](https://doi.org/10.5281/zenodo.18940011)  
**Contact:** purohitmaheep@gmail.com

---

## Install

```bash
pip install ewe-gate
```

---

## What is EWE?

Current AI systems treat every training sample equally — regardless of whether the information is reliable, novel, or approval-biased. This creates two structural problems:

1. **Uniform Weighting Problem** — gradient-based learning applies equal parameter-update eligibility to all samples regardless of signal reliability.
2. **Approval Bias Problem** — systems trained with human feedback learn to please raters rather than learn what is accurate.

The **Epistemic Weight Engine** is a modular pre-update gate that sits between incoming training data and the parameter update step. Before any sample is allowed to change the model's knowledge, EWE asks three questions:

| Layer | Module | Question |
|---|---|---|
| Layer 1 | Impact Assessment I(x) | Does this sample actually matter? |
| Layer 2 | Reality Alignment R(x) | Is this label evidence-consistent or just approval-consistent? |
| Layer 3 | Paradigm Shift P(x) | Is this genuinely new information? |

Only samples that pass all three layers trigger a parameter update. Everything else is routed to passive memory.

---

## Quick Start

### Simplest usage — just the gate

```python
import torch
import torch.nn as nn
from ewe import EWEGate

model = YourModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss(reduction='none')
gate = EWEGate()

for epoch in range(50):
    for x, y in dataloader:
        optimizer.zero_grad()
        outputs = model(x)
        losses = criterion(outputs, y)

        # EWE filters which samples update parameters
        filtered_loss = gate.filter_losses(losses, outputs.detach())
        if filtered_loss is not None:
            filtered_loss.backward()
            optimizer.step()

    print(f"Epoch {epoch} | Accept rate: {gate.acceptance_rate:.1%}")
```

### High-level trainer

```python
from ewe import EWETrainer
import torch.nn as nn

trainer = EWETrainer(
    model=model,
    optimizer=optimizer,
    criterion=nn.CrossEntropyLoss(reduction='none'),
)

for epoch in range(50):
    loss, acc, rate = trainer.train_epoch(train_loader)
    val_acc = trainer.evaluate(val_loader)
    print(f"Epoch {epoch} | Acc: {acc:.1f}% | Accept: {rate:.1%}")
```

### Best results — EWE + GCE combined

```python
from ewe import EWEGate, GCELoss

gate = EWEGate()
criterion = GCELoss(q=0.7)

for x, y in dataloader:
    optimizer.zero_grad()
    outputs = model(x)
    losses = criterion(outputs, y, reduction='none')
    filtered_loss = gate.filter_losses(losses, outputs.detach())
    if filtered_loss is not None:
        filtered_loss.backward()
        optimizer.step()
```

---

## Key Results

### CIFAR-10N — ResNet-18 — Real Human Annotation Noise ~40%

| Method | Accuracy | vs Standard | Gate Level |
|---|---|---|---|
| Standard Training | 72.37% | baseline | None |
| Label Smoothing | 72.72% | +0.35% | Loss |
| GCE | 81.74% | +9.37% | Loss |
| Co-teaching | 85.46% | +13.09% | Sample |
| **EWE (ours)** | **79.36%** | **+6.99%** | **Pre-update** |
| **EWE+GCE (ours)** | **84.33%** | **+11.96%** | **Pre-update + Loss** |

**EWE+GCE outperforms GCE alone by +2.59% using a single model** — confirming the pre-update gate provides independent additive value at a distinct architectural level.

### UCI Benchmark Datasets — Logistic Regression — N=20 seeds

- EWE advantage scales monotonically with label noise severity
- Reality Alignment module confirmed as most critical component
- Robust to hyperparameter choice across full tested range

---

## Configuration

```python
gate = EWEGate(
    alpha=0.45,      # Weight for Impact module I(x)
    beta=0.40,       # Weight for Reality Alignment R(x) — most important
    gamma=0.15,      # Weight for Paradigm Shift P(x)
    k=0.25,          # Gate sensitivity — higher = stricter
    lam=0.5,         # Approval penalty strength
    ema_decay=0.99,  # Loss baseline decay rate
)

# Monitor the gate
print(f"Acceptance rate: {gate.acceptance_rate:.1%}")
print(f"Suppression rate: {gate.suppression_rate:.1%}")

# Reset stats between experiments
gate.reset_stats()
```

**Tuning k:**
- `k=0.10` → ~70% acceptance (loose)
- `k=0.25` → ~60% acceptance (default)
- `k=0.50` → ~50% acceptance (strict)

---

## The Core Equations

```
I(x) = L(x) / (τ + L(x))                           # Impact Assessment

R(x) = max(0, sim(x) − λ·A(x))                     # Reality Alignment
       sim(x) = 1 − L(x)/max(L)                     # Inverse loss formulation
       A(x)   = max softmax probability              # Approval signal

P(x) = max(0, (L(x) − L_ema) / (L_ema + ε))        # Paradigm Shift

W(x) = α·I(x) + β·R(x) + γ·P(x)                   # Composite score
       α=0.45, β=0.40, γ=0.15

Gate: W(x) ≥ μ_W − k·σ_W                           # Adaptive threshold
```

---

## Repository Structure

```
epistemic-weight-engine/
│
├── README.md                     # This file
├── setup.py                      # PyPI packaging
├── pyproject.toml                # Build configuration
├── requirements.txt              # Dependencies
│
├── ewe/                          # Installable library
│   ├── __init__.py
│   ├── gate.py                   # Core EWEGate class
│   ├── losses.py                 # GCELoss, LabelSmoothingLoss
│   └── trainer.py                # EWETrainer wrapper
│
├── train.py                      # CIFAR-10N main experiment
├── ewe_gce_experiment.py         # EWE+GCE combined experiment
│
├── ewe_results.json              # Main experiment results
├── ewe_combined_results.json     # EWE+GCE results
└── ewe_cifar10n_results.png      # Results figure
```

---

## Run the Experiments

### Requirements

```bash
pip install torch torchvision numpy matplotlib seaborn pandas
```

### Download CIFAR-10N labels

Download `CIFAR-10_human.pt` from the [CIFAR-10N repository](https://github.com/UCSC-REAL/cifar-10-100n) and place it in the same folder as `train.py`.

### Run main experiment

```bash
python train.py
```

Trains 5 methods on CIFAR-10N across 3 seeds. Takes 3 to 4 hours on RTX 3050.

### Run EWE+GCE experiment

```bash
python ewe_gce_experiment.py
```

Requires `ewe_results.json` from the main experiment.

---

## Citation

If you use EWE in your research please cite:

```bibtex
@article{purohit2026ewe,
  title={Epistemic Weight Engine (EWE): A Framework for Signal-Reliability-Weighted
         Learning in Artificial Neural Systems, with Multi-Dataset Experimental Evaluation},
  author={Purohit, Maheep},
  journal={ACM Transactions on Intelligent Systems and Technology},
  year={2026},
  note={Manuscript ID: TIST-2026-03-0289},
  doi={10.5281/zenodo.18940011}
}
```

---

## Links

- **PyPI:** https://pypi.org/project/ewe-gate/
- **Preprint:** https://doi.org/10.5281/zenodo.18940011
- **Contact:** purohitmaheep@gmail.com

---

## Author

**Maheep Purohit**  
Independent Researcher, Bikaner, Rajasthan, India  
Patent Applicant: Adaptive Intelligent Pipeline Integrity System (filed 2025, India Patent Office)

*This research was conducted entirely independently without institutional affiliation, laboratory access, external funding, or academic supervision.*
