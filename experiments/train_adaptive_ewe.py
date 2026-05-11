"""
Adaptive EWE Experiment
Author: Maheep Purohit, Independent Researcher, Bikaner, India
Paper: https://doi.org/10.5281/zenodo.18940011

KEY CONTRIBUTION OF THIS EXPERIMENT:
Instead of manually choosing k per dataset, EWE automatically
calibrates using: k = base_k / log(num_classes + 1)

CIFAR-10N:  k = 0.25 / log(11) = 0.104  (automatic)
CIFAR-100N: k = 0.25 / log(101) = 0.054 (automatic)

No manual tuning. One formula. Both datasets.

Runs EWE and EWE+GCE on BOTH datasets in one script.
Previous results for other methods are loaded from JSON files.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from datetime import datetime

# ── Device ────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ── Config ────────────────────────────────────────────────────
SEEDS      = [42, 123, 456, 789, 1024]
EPOCHS     = 100
WARMUP     = 10
BATCH_SIZE = 128
LR         = 0.1
BASE_K     = 0.25   # universal base parameter

# Adaptive formula
def adaptive_k(num_classes):
    k = BASE_K / np.log(num_classes + 1)
    print(f'Adaptive k for {num_classes} classes: {k:.4f}')
    return k

# ── Transforms ────────────────────────────────────────────────
def get_transforms(dataset_name):
    if dataset_name == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1994, 0.2010)
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std  = (0.2675, 0.2565, 0.2761)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, test_tf


# ── Dataset ───────────────────────────────────────────────────
class NoisyDataset(Dataset):
    def __init__(self, root, transform, noise_file,
                 dataset_name='cifar10', train=True):
        if dataset_name == 'cifar10':
            self.dataset = torchvision.datasets.CIFAR10(
                root=root, train=train,
                download=True, transform=transform)
        else:
            self.dataset = torchvision.datasets.CIFAR100(
                root=root, train=train,
                download=True, transform=transform)

        self.train = train

        if train and os.path.exists(noise_file):
            noise_data = torch.load(noise_file, weights_only=False)
            # CIFAR-10N uses 'aggre_label', CIFAR-100N uses 'noisy_label'
            if 'noisy_label' in noise_data:
                label_key = 'noisy_label'
            elif 'aggre_label' in noise_data:
                label_key = 'aggre_label'
            else:
                label_key = list(noise_data.keys())[1]
            self.noisy_labels = np.array(noise_data[label_key])
            noise_rate = (
                self.noisy_labels !=
                np.array(noise_data['clean_label'])
            ).mean()
            print(f'{dataset_name} noise rate: {noise_rate:.1%} '
                  f'(using key: {label_key})')
        else:
            self.noisy_labels = np.array(self.dataset.targets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        label  = int(self.noisy_labels[idx]) if self.train \
                 else int(self.dataset.targets[idx])
        return img, label, idx


# ── Models ────────────────────────────────────────────────────
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNetCIFAR(nn.Module):
    def __init__(self, num_classes=10, depth=18):
        super().__init__()
        # depth 18 → [2,2,2,2], depth 34 → [3,4,6,3]
        cfg = [2,2,2,2] if depth == 18 else [3,4,6,3]
        self.conv1  = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.layer1 = self._make(64,  64,  cfg[0], 1)
        self.layer2 = self._make(64,  128, cfg[1], 2)
        self.layer3 = self._make(128, 256, cfg[2], 2)
        self.layer4 = self._make(256, 512, cfg[3], 2)
        self.fc     = nn.Linear(512, num_classes)

    def _make(self, in_p, p, n, s):
        layers = [BasicBlock(in_p, p, s)]
        for _ in range(n-1):
            layers.append(BasicBlock(p, p, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        return self.fc(x.view(x.size(0), -1))


# ── Loss Functions ────────────────────────────────────────────
def ce_loss(out, y, reduction='mean'):
    return F.cross_entropy(out, y, reduction=reduction)

def gce_loss(out, y, q=0.7, reduction='mean'):
    p   = F.softmax(out, 1)
    py  = p[range(len(y)), y].clamp(1e-7)
    loss = (1 - py**q) / q
    stab = 1e-4 * F.cross_entropy(out, y)
    return (loss.mean() + stab) if reduction == 'mean' \
           else (loss + stab)


# ── Adaptive EWE Gate ─────────────────────────────────────────
class AdaptiveEWEGate:
    """
    EWE gate with automatic k calibration.
    k = BASE_K / log(num_classes + 1)
    No manual tuning required.
    """
    def __init__(self, num_classes, alpha=0.45, beta=0.40,
                 gamma=0.15, tau=0.5, lam=0.40,
                 eps=0.1, ema_decay=0.99):
        self.k         = adaptive_k(num_classes)
        self.alpha     = alpha
        self.beta      = beta
        self.gamma     = gamma
        self.tau       = tau
        self.lam       = lam
        self.eps       = eps
        self.ema_decay = ema_decay
        self._loss_ema = None
        self._total    = 0
        self._accepted = 0

    @property
    def acceptance_rate(self):
        return self._accepted / self._total \
               if self._total > 0 else 0.0

    def gate(self, losses, logits):
        # Impact Assessment
        I = torch.clamp(losses / (self.tau + losses), 0, 1)

        # Reality Alignment
        probs = F.softmax(logits, 1)
        A     = probs.max(1).values
        sim_x = 1.0 - losses / (losses.max() + 1e-8)
        R     = torch.clamp(sim_x - self.lam * A, min=0)

        # Paradigm Shift
        mean_loss = losses.mean().item()
        if self._loss_ema is None:
            self._loss_ema = mean_loss
        else:
            self._loss_ema = (self.ema_decay * self._loss_ema +
                              (1 - self.ema_decay) * mean_loss)
        P = torch.clamp(
            (losses - self._loss_ema) /
            (self._loss_ema + self.eps), 0, 1
        )

        # Composite score with ADAPTIVE threshold
        W     = self.alpha*I + self.beta*R + self.gamma*P
        theta = W.mean() - self.k * W.std()
        mask  = W >= theta

        self._total    += losses.shape[0]
        self._accepted += int(mask.sum().item())
        return mask


# ── Evaluation ────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            _, pred = model(x).max(1)
            correct += pred.eq(y).sum().item()
            total   += y.size(0)
    return 100.0 * correct / total


# ── Training ──────────────────────────────────────────────────
def train_epoch(model, loader, optimizer,
                method, gate, epoch):
    model.train()
    in_warmup = epoch <= WARMUP

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)

        if in_warmup or gate is None:
            ce_loss(out, y).backward()
        elif method == 'adaptive-ewe':
            losses = ce_loss(out, y, 'none')
            mask   = gate.gate(losses.detach(), out.detach())
            if mask.sum() > 0:
                losses[mask].mean().backward()
            else:
                return
        elif method == 'adaptive-ewe-gce':
            losses = gce_loss(out, y, reduction='none')
            mask   = gate.gate(losses.detach(), out.detach())
            if mask.sum() > 0:
                losses[mask].mean().backward()
            else:
                return

        optimizer.step()


# ── Run One Seed ──────────────────────────────────────────────
def run_seed(method, train_loader, test_loader,
             num_classes, depth, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ResNetCIFAR(num_classes, depth).to(device)
    opt   = torch.optim.SGD(model.parameters(), lr=LR,
                             momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, EPOCHS)

    gate = AdaptiveEWEGate(num_classes) \
           if 'ewe' in method else None

    for ep in range(1, EPOCHS+1):
        train_epoch(model, train_loader, opt,
                    method, gate, ep)
        sched.step()

        if ep % 10 == 0:
            acc   = evaluate(model, test_loader)
            rate  = gate.acceptance_rate if gate else 1.0
            phase = 'WARMUP' if ep <= WARMUP else 'EWE'
            print(f'  Ep {ep:3d} [{phase}] | '
                  f'Acc: {acc:.2f}% | Accept: {rate:.1%}')

    return evaluate(model, test_loader)


# ── Run One Dataset ───────────────────────────────────────────
def run_dataset(dataset_name, noise_file, num_classes, depth):
    print(f'\n{"="*60}')
    print(f'Dataset: {dataset_name.upper()} | '
          f'Classes: {num_classes} | ResNet-{depth}')
    print(f'Adaptive k = {BASE_K:.2f} / '
          f'log({num_classes}+1) = '
          f'{adaptive_k(num_classes):.4f}')
    print('='*60)

    train_tf, test_tf = get_transforms(dataset_name)
    train_set = NoisyDataset('./data', train_tf, noise_file,
                              dataset_name, train=True)
    test_set  = NoisyDataset('./data', test_tf,  noise_file,
                              dataset_name, train=False)
    train_loader = DataLoader(train_set, BATCH_SIZE,
                               shuffle=True, num_workers=2,
                               pin_memory=True)
    test_loader  = DataLoader(test_set, BATCH_SIZE,
                               shuffle=False, num_workers=2,
                               pin_memory=True)

    methods = ['adaptive-ewe', 'adaptive-ewe-gce']
    results = {m: [] for m in methods}

    for method in methods:
        print(f'\n── {method.upper()} ──')
        for seed in SEEDS:
            seed_key = f'{dataset_name}_{method}_{seed}'
            # Skip if already done in checkpoint
            if seed_key in checkpoint:
                acc = checkpoint[seed_key]
                print(f'  Seed {seed} → {acc:.2f}% [RESUMED]')
                results[method].append(acc)
                continue

            print(f'  Seed {seed}')
            acc = run_seed(method, train_loader, test_loader,
                           num_classes, depth, seed)
            results[method].append(acc)
            print(f'  Final: {acc:.2f}%')

            # Save checkpoint immediately after each seed
            checkpoint[seed_key] = acc
            with open('ewe_adaptive_checkpoint.json', 'w') as f:
                json.dump(checkpoint, f, indent=2)
            print(f'  [Checkpoint saved]')

        mean = np.mean(results[method])
        std  = np.std(results[method])
        print(f'  {method}: {mean:.2f}% ± {std:.2f}%')

    return results


# ── Main ──────────────────────────────────────────────────────
def main():
    print('ADAPTIVE EWE EXPERIMENT')
    print(f'Formula: k = {BASE_K} / log(num_classes + 1)')
    print(f'Seeds: {SEEDS} | Warmup: {WARMUP} | '
          f'Epochs: {EPOCHS}')

    # Load checkpoint if exists
    global checkpoint
    if os.path.exists('ewe_adaptive_checkpoint.json'):
        with open('ewe_adaptive_checkpoint.json', 'r') as f:
            checkpoint = json.load(f)
        print(f'Resuming from checkpoint — '
              f'{len(checkpoint)} seeds already done')
    else:
        # Load known results from previous run
        checkpoint = {
            'cifar10n_adaptive-ewe_42':   90.13,
            'cifar10n_adaptive-ewe_123':  89.89,
            'cifar10n_adaptive-ewe_456':  90.18,
            'cifar10n_adaptive-ewe_789':  90.18,
            'cifar10n_adaptive-ewe_1024': 90.01,
            'cifar10n_adaptive-ewe-gce_42':  91.64,
            'cifar10n_adaptive-ewe-gce_123': 92.02,
        }
        print('Fresh start — known results pre-loaded')

    # Previous results from your experiments
    previous = {
        'cifar10n': {
            'standard':        (72.37, 0.16),
            'label-smoothing': (72.72, 0.17),
            'gce':             (81.74, 0.20),
            'co-teaching':     (85.46, 0.06),
            'ewe':             (79.36, 0.18),
            'ewe-gce':         (84.33, 0.16),
        },
        'cifar100n': {
            'standard':        (54.66, 0.70),
            'label-smoothing': (56.03, 0.62),
            'gce':             (58.40, 0.42),
            'co-teaching':     (64.16, 0.60),
            'ewe':             (53.59, 0.83),
            'ewe-gce':         (58.90, 0.97),
        }
    }

    all_results = {}

    # CIFAR-10N — ResNet-18
    r10 = run_dataset(
        'cifar10',
        './CIFAR-10_human.pt',
        num_classes=10,
        depth=18
    )
    all_results['cifar10n'] = r10

    # CIFAR-100N — ResNet-34
    r100 = run_dataset(
        'cifar100',
        './CIFAR-100_human.pt',
        num_classes=100,
        depth=34
    )
    all_results['cifar100n'] = r100

    # ── Final Summary ─────────────────────────────────────────
    print('\n' + '='*60)
    print('ADAPTIVE EWE — COMPLETE RESULTS')
    print('='*60)

    for ds in ['cifar10n', 'cifar100n']:
        base = previous[ds]['standard'][0]
        gce  = previous[ds]['gce'][0]
        print(f'\n{ds.upper()}:')
        print(f'  {"standard":25s}: {base:.2f}%  (baseline)')
        print(f'  {"gce":25s}: {gce:.2f}%')

        for method, vals in all_results[ds].items():
            mean  = np.mean(vals)
            std   = np.std(vals)
            delta = mean - base
            vgce  = mean - gce
            sign  = '+' if delta >= 0 else ''
            sgce  = '+' if vgce  >= 0 else ''
            print(f'  {method:25s}: {mean:.2f}% ± {std:.2f}%  '
                  f'({sign}{delta:.2f}% vs std, '
                  f'{sgce}{vgce:.2f}% vs gce)')

    # Save
    output = {
        'experiment':   'Adaptive EWE',
        'formula':      f'k = {BASE_K} / log(C+1)',
        'seeds':        SEEDS,
        'warmup':       WARMUP,
        'epochs':       EPOCHS,
        'timestamp':    datetime.now().isoformat(),
        'previous':     previous,
        'adaptive_results': {
            ds: {
                m: {
                    'per_seed':    v,
                    'mean':        float(np.mean(v)),
                    'std':         float(np.std(v)),
                    'vs_standard': float(
                        np.mean(v) -
                        previous[ds]['standard'][0]),
                    'vs_gce':      float(
                        np.mean(v) -
                        previous[ds]['gce'][0]),
                    'adaptive_k':  float(
                        adaptive_k(
                            10 if ds=='cifar10n' else 100))
                }
                for m, v in all_results[ds].items()
            }
            for ds in all_results
        }
    }

    with open('ewe_adaptive_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print('\nResults saved to ewe_adaptive_results.json')
    print('\nKEY CLAIM VERIFIED IF:')
    print('  adaptive-ewe-gce > gce on BOTH datasets')
    print('  with NO manual k selection')


if __name__ == '__main__':
    main()
