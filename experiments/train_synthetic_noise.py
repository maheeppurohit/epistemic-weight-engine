"""
Synthetic Noise Levels Experiment
Author: Maheep Purohit, Independent Researcher, Bikaner, India
Paper: https://doi.org/10.5281/zenodo.18940011

Tests Adaptive EWE across synthetic noise levels:
20%, 40%, 60%, 80% symmetric label noise on CIFAR-10

Shows EWE is robust across ALL noise levels
not just the specific 40.2% in CIFAR-10N.

Methods compared at each noise level:
- Standard Training
- GCE
- Co-teaching
- Adaptive EWE
- Adaptive EWE+GCE
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
SEEDS        = [42, 123, 456]   # 3 seeds for speed
EPOCHS       = 100
WARMUP       = 10
BATCH_SIZE   = 128
LR           = 0.1
NUM_CLASSES  = 10
BASE_K       = 0.25
NOISE_LEVELS = [0.2, 0.4, 0.6, 0.8]
CHECKPOINT   = 'synthetic_noise_checkpoint.json'

def adaptive_k(num_classes):
    return BASE_K / np.log(num_classes + 1)

# ── Transforms ────────────────────────────────────────────────
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)),
])

# ── Dataset with Synthetic Noise ──────────────────────────────
class SyntheticNoiseCIFAR10(Dataset):
    def __init__(self, root, transform,
                 noise_rate=0.4, train=True, seed=42):
        self.dataset = torchvision.datasets.CIFAR10(
            root=root, train=train,
            download=True, transform=transform)
        self.train = train

        if train:
            np.random.seed(seed)
            labels = np.array(self.dataset.targets)
            n      = len(labels)
            n_noisy = int(noise_rate * n)
            noisy_idx = np.random.choice(
                n, n_noisy, replace=False)

            # Symmetric noise — replace with random label
            for idx in noisy_idx:
                current = labels[idx]
                choices = list(range(NUM_CLASSES))
                choices.remove(current)
                labels[idx] = np.random.choice(choices)

            self.noisy_labels = labels
            actual_noise = (
                self.noisy_labels !=
                np.array(self.dataset.targets)
            ).mean()
            print(f'Synthetic noise: {noise_rate:.0%} '
                  f'(actual: {actual_noise:.1%})')
        else:
            self.noisy_labels = np.array(
                self.dataset.targets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        label  = int(self.noisy_labels[idx])
        return img, label, idx


# ── Model ─────────────────────────────────────────────────────
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, 3,
            stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1  = nn.Conv2d(
            3, 64, 3, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.layer1 = self._make(64,  64,  2, 1)
        self.layer2 = self._make(64,  128, 2, 2)
        self.layer3 = self._make(128, 256, 2, 2)
        self.layer4 = self._make(256, 512, 2, 2)
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
    p    = F.softmax(out, 1)
    py   = p[range(len(y)), y].clamp(1e-7)
    loss = (1 - py**q) / q
    stab = 1e-4 * F.cross_entropy(out, y)
    return (loss.mean() + stab) if reduction == 'mean' \
           else (loss + stab)


# ── Adaptive EWE Gate ─────────────────────────────────────────
class AdaptiveEWEGate:
    def __init__(self, num_classes=10):
        self.k         = adaptive_k(num_classes)
        self.alpha     = 0.45
        self.beta      = 0.40
        self.gamma     = 0.15
        self.tau       = 0.5
        self.lam       = 0.40
        self.eps       = 0.1
        self.ema_decay = 0.99
        self._loss_ema = None
        self._total    = 0
        self._accepted = 0

    @property
    def acceptance_rate(self):
        return self._accepted / self._total \
               if self._total > 0 else 0.0

    def gate(self, losses, logits):
        I     = torch.clamp(
            losses / (self.tau + losses), 0, 1)
        probs = F.softmax(logits, 1)
        A     = probs.max(1).values
        sim_x = 1.0 - losses / (losses.max() + 1e-8)
        R     = torch.clamp(
            sim_x - self.lam * A, min=0)
        mean_loss = losses.mean().item()
        if self._loss_ema is None:
            self._loss_ema = mean_loss
        else:
            self._loss_ema = (
                self.ema_decay * self._loss_ema +
                (1 - self.ema_decay) * mean_loss)
        P = torch.clamp(
            (losses - self._loss_ema) /
            (self._loss_ema + self.eps), 0, 1)
        W     = self.alpha*I + self.beta*R + self.gamma*P
        theta = W.mean() - self.k * W.std()
        mask  = W >= theta
        self._total    += losses.shape[0]
        self._accepted += int(mask.sum().item())
        return mask


# ── Co-teaching ───────────────────────────────────────────────
def coteaching_select(loss1, loss2, keep_rate=0.7):
    n_keep = max(1, int(len(loss1) * keep_rate))
    idx1   = torch.argsort(loss1)[:n_keep]
    idx2   = torch.argsort(loss2)[:n_keep]
    return idx1, idx2


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


# ── Run One Seed ──────────────────────────────────────────────
def run_seed(method, train_loader, test_loader, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if method == 'co-teaching':
        m1 = ResNet18CIFAR(NUM_CLASSES).to(device)
        m2 = ResNet18CIFAR(NUM_CLASSES).to(device)
        o1 = torch.optim.SGD(
            m1.parameters(), lr=LR,
            momentum=0.9, weight_decay=5e-4)
        o2 = torch.optim.SGD(
            m2.parameters(), lr=LR,
            momentum=0.9, weight_decay=5e-4)
        s1 = torch.optim.lr_scheduler\
            .CosineAnnealingLR(o1, EPOCHS)
        s2 = torch.optim.lr_scheduler\
            .CosineAnnealingLR(o2, EPOCHS)
        keep = 0.7
        for ep in range(1, EPOCHS+1):
            m1.train(); m2.train()
            for x, y, _ in train_loader:
                x, y = x.to(device), y.to(device)
                l1 = ce_loss(m1(x), y, 'none')
                l2 = ce_loss(m2(x), y, 'none')
                i1, i2 = coteaching_select(
                    l1.detach(), l2.detach(), keep)
                o1.zero_grad()
                ce_loss(m1(x)[i2],
                        y[i2]).backward()
                o1.step()
                o2.zero_grad()
                ce_loss(m2(x)[i1],
                        y[i1]).backward()
                o2.step()
            s1.step(); s2.step()
        return evaluate(m1, test_loader)

    model = ResNet18CIFAR(NUM_CLASSES).to(device)
    opt   = torch.optim.SGD(
        model.parameters(), lr=LR,
        momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler\
        .CosineAnnealingLR(opt, EPOCHS)
    gate  = AdaptiveEWEGate(NUM_CLASSES) \
            if 'ewe' in method else None

    for ep in range(1, EPOCHS+1):
        model.train()
        in_warmup = ep <= WARMUP

        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)

            if in_warmup or method == 'standard':
                ce_loss(out, y).backward()
            elif method == 'gce':
                gce_loss(out, y).backward()
            elif method == 'adaptive-ewe':
                losses = ce_loss(out, y, 'none')
                mask   = gate.gate(
                    losses.detach(), out.detach())
                if mask.sum() > 0:
                    losses[mask].mean().backward()
                else:
                    continue
            elif method == 'adaptive-ewe-gce':
                losses = gce_loss(
                    out, y, reduction='none')
                mask   = gate.gate(
                    losses.detach(), out.detach())
                if mask.sum() > 0:
                    losses[mask].mean().backward()
                else:
                    continue
            opt.step()
        sched.step()

    return evaluate(model, test_loader)


# ── Checkpoint ────────────────────────────────────────────────
def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            d = json.load(f)
            print(f'Resuming — {len(d)} done')
            return d
    return {}

def save_checkpoint(d):
    with open(CHECKPOINT, 'w') as f:
        json.dump(d, f, indent=2)


# ── Main ──────────────────────────────────────────────────────
def main():
    print('SYNTHETIC NOISE EXPERIMENT')
    print(f'Noise levels: {NOISE_LEVELS}')
    print(f'Seeds: {SEEDS} | Epochs: {EPOCHS}')
    print(f'Adaptive k: {adaptive_k(NUM_CLASSES):.4f}')
    print('='*60)

    checkpoint = load_checkpoint()
    methods    = ['standard', 'gce', 'co-teaching',
                  'adaptive-ewe', 'adaptive-ewe-gce']
    all_results = {}

    # Download dataset once
    torchvision.datasets.CIFAR10(
        './data', train=True,
        download=True,
        transform=transform_train)

    for noise in NOISE_LEVELS:
        noise_key = f'noise_{int(noise*100)}'
        print(f'\n{"="*60}')
        print(f'Noise Level: {noise:.0%}')
        print('='*60)

        all_results[noise_key] = {}

        test_set = SyntheticNoiseCIFAR10(
            './data', transform_test,
            noise_rate=noise, train=False)
        test_loader = DataLoader(
            test_set, BATCH_SIZE, shuffle=False,
            num_workers=2, pin_memory=True)

        for method in methods:
            print(f'\n── {method.upper()} ──')
            results = []

            for seed in SEEDS:
                key = f'{noise_key}_{method}_{seed}'

                if key in checkpoint:
                    acc = checkpoint[key]
                    print(f'  Seed {seed} → '
                          f'{acc:.2f}% [RESUMED]')
                    results.append(acc)
                    continue

                # Fresh dataset with this seed's noise
                train_set = SyntheticNoiseCIFAR10(
                    './data', transform_train,
                    noise_rate=noise,
                    train=True, seed=seed)
                train_loader = DataLoader(
                    train_set, BATCH_SIZE,
                    shuffle=True, num_workers=2,
                    pin_memory=True)

                print(f'  Seed {seed} — running...')
                acc = run_seed(
                    method, train_loader,
                    test_loader, seed)
                results.append(acc)
                print(f'  Seed {seed} → '
                      f'Final: {acc:.2f}%')

                checkpoint[key] = acc
                save_checkpoint(checkpoint)
                print(f'  [Checkpoint saved]')

            mean = np.mean(results)
            std  = np.std(results)
            all_results[noise_key][method] = {
                'per_seed': results,
                'mean': float(mean),
                'std':  float(std)
            }
            print(f'  {method}: '
                  f'{mean:.2f}% ± {std:.2f}%')

    # ── Summary Table ─────────────────────────────────────────
    print('\n' + '='*60)
    print('SYNTHETIC NOISE — COMPLETE RESULTS')
    print('='*60)
    print(f'\n{"Method":<22}', end='')
    for noise in NOISE_LEVELS:
        print(f'  {int(noise*100)}%  ', end='')
    print()
    print('-'*60)

    for method in methods:
        print(f'{method:<22}', end='')
        for noise in NOISE_LEVELS:
            key  = f'noise_{int(noise*100)}'
            mean = all_results[key][method]['mean']
            print(f'  {mean:.1f}', end='')
        print()

    # Save
    output = {
        'experiment':   'Synthetic Noise Levels',
        'noise_levels': NOISE_LEVELS,
        'seeds':        SEEDS,
        'epochs':       EPOCHS,
        'timestamp':    datetime.now().isoformat(),
        'results':      all_results
    }
    with open('synthetic_noise_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print('\nResults saved to synthetic_noise_results.json')


if __name__ == '__main__':
    main()
