# epistemic-weight-engine
Epistemic Weight Engine (EWE) — A pre-update gating mechanism for signal-reliability-weighted learning in AI systems. ACM TIST 2026.

# Epistemic Weight Engine (EWE)

> A modular pre-update gating mechanism for signal-reliability-weighted learning in artificial neural systems.

**Author:** Maheep Purohit — Independent Researcher, Bikaner, Rajasthan, India  
**Paper:** Submitted to ACM Transactions on Intelligent Systems and Technology (TIST-2026-03-0289)  
**Preprint DOI:** [10.5281/zenodo.18940011](https://doi.org/10.5281/zenodo.18940011)  
**Contact:** purohitmaheep@gmail.com

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

## Key Results

### UCI Benchmark Datasets (Logistic Regression, N=20 seeds)
- EWE advantage scales monotonically with label noise severity
- Reality Alignment module confirmed as most critical component
- Robust to hyperparameter choice across full tested range

### CIFAR-10N (ResNet-18, Real Human Annotation Noise ~40%, N=3 seeds)

| Method | Accuracy | vs Standard |
|---|---|---|
| Standard Training | 72.37% | baseline |
| Label Smoothing | 72.72% | +0.35% |
| EWE (ours) | **79.36%** | **+6.99%** |
| GCE | 81.74% | +9.37% |
| Co-teaching | 85.46% | +13.09% |
| **EWE+GCE (ours)** | **84.33%** | **+11.96%** |

**EWE+GCE outperforms GCE alone by +2.59% using a single model**, confirming that the pre-update gate provides independent additive value at a distinct architectural level from loss-function modifications.

---

## The Core Equations

```
I(x) = L(x) / (τ + L(x))                          # Impact Assessment

R(x) = max(0, sim(x) − λ·A(x))                    # Reality Alignment
       where sim(x) = 1 − L(x)/max(L)              # Neural network formulation

P(x) = max(0, (L(x) − L_ema) / (L_ema + ε))       # Paradigm Shift

W(x) = α·I(x) + β·R(x) + γ·P(x)                  # Composite score
       α=0.45, β=0.40, γ=0.15

Gate: W(x) ≥ θ  where θ = μ_W − k·σ_W (adaptive)  # Restructuring condition
```

---

## Repository Structure

```
epistemic-weight-engine/
│
├── README.md                    # This file
├── train.py                     # Main CIFAR-10N experiment
├── ewe_gce_experiment.py        # EWE+GCE combined experiment
├── requirements.txt             # Dependencies
└── results/
    ├── ewe_results.json         # Main experiment results
    ├── ewe_combined_results.json # EWE+GCE results
    └── ewe_cifar10n_results.png  # Results figure
```

---

## How to Run

### Requirements
```bash
pip install torch torchvision numpy matplotlib seaborn pandas
```

Python 3.11 recommended. NVIDIA GPU recommended (RTX 3050 or better).

### Download CIFAR-10N labels
Download `CIFAR-10_human.pt` from the [CIFAR-10N repository](https://github.com/UCSC-REAL/cifar-10-100n) and place it in the same folder as `train.py`.

### Run main experiment
```bash
python train.py
```

This trains 5 methods (Standard, EWE, GCE, Label Smoothing, Co-teaching) on CIFAR-10N across 3 seeds. Takes approximately 3 to 4 hours on an RTX 3050.

### Run EWE+GCE experiment
```bash
python ewe_gce_experiment.py
```

Runs the combined EWE+GCE method. Requires `ewe_results.json` from the main experiment.

---

## Results

After running, the following files are saved:

- `ewe_results.json` — all method results across seeds
- `ewe_combined_results.json` — EWE+GCE combined results
- `ewe_cifar10n_results.png` — publication quality figure
- `ewe_checkpoint.json` — per-seed checkpoint

---

## Citation

If you use this code or build on this work please cite:

```
Purohit, M. (2026). Epistemic Weight Engine (EWE): A Framework for 
Signal-Reliability-Weighted Learning in Artificial Neural Systems, 
with Multi-Dataset Experimental Evaluation. 
Submitted to ACM Transactions on Intelligent Systems and Technology.
DOI: 10.5281/zenodo.18940011
```

---

## Related Work

- Zhang & Sabuncu (2018) — Generalised Cross-Entropy Loss (GCE)
- Han et al. (2018) — Co-teaching
- Bengio et al. (2009) — Curriculum Learning
- Kirkpatrick et al. (2017) — Elastic Weight Consolidation

---

## Author

**Maheep Purohit**  
Independent Researcher, Bikaner, Rajasthan, India  
Patent Applicant: Adaptive Intelligent Pipeline Integrity System (filed 2025, India Patent Office)  
purohitmaheep@gmail.com

*This research was conducted entirely independently without institutional affiliation, laboratory access, external funding, or academic supervision.*
