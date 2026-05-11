"""
CIFAR-100N Experiment — EWE Tuned for CIFAR-100N
Author: Maheep Purohit, Independent Researcher, Bikaner, India
Paper: https://doi.org/10.5281/zenodo.18940011

CHANGES FROM v1:
- k = 0.10 (was 0.25) — looser gate for 100 classes
- lam = 0.30 (was 0.50) — less aggressive approval penalty
- WARMUP = 15 epochs — standard training before EWE activates
- Only runs EWE and EWE+GCE (other methods already done)

Previous results to combine with:
- Standard:        54.66% ± 0.70%
- Label Smoothing: 56.03% ± 0.62%
- GCE:             58.40% ± 0.42%
- Co-teaching:     64.16% ± 0.60%
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
SEEDS       = [42, 123, 456, 789, 1024]
EPOCHS      = 100
WARMUP      = 15       # NEW — train standard first 15 epochs
BATCH_SIZE  = 128
NUM_CLASSES = 100
LR          = 0.1
NOISE_FILE  = './CIFAR-100_human.pt'

# EWE parameters tuned for CIFAR-100N
EWE_K    = 0.10   # was 0.25 — looser gate
EWE_LAM  = 0.30   # was 0.50 — less aggressive

print(f'EWE params: k={EWE_K}, lam={EWE_LAM}, warmup={WARMUP}')

# ── Transforms ────────────────────────────────────────────────
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])

# ── Dataset ───────────────────────────────────────────────────
class CIFAR100N(Dataset):
    def __init__(self, root, transform, noise_file, train=True):
        self.dataset = torchvision.datasets.CIFAR100(
            root=root, train=train, download=True, transform=transform
        )
        self.train = train

        if train and os.path.exists(noise_file):
            noise_data = torch.load(noise_file, weights_only=False)
            self.noisy_labels = np.array(noise_data['noisy_label'])
            self.clean_labels = np.array(noise_data['clean_label'])
            noise_rate = (self.noisy_labels != self.clean_labels).mean()
            print(f'CIFAR-100N noise rate: {noise_rate:.1%}')
        else:
            self.noisy_labels = np.array(self.dataset.targets)
            self.clean_labels = np.array(self.dataset.targets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        label  = int(self.noisy_labels[idx]) if self.train \
                 else int(self.dataset.targets[idx])
        return img, label, idx


# ── ResNet-34 for CIFAR ───────────────────────────────────────
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
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


class ResNet34CIFAR(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1  = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64,  64,  3, 1)
        self.layer2 = self._make_layer(64,  128, 4, 2)
        self.layer3 = self._make_layer(128, 256, 6, 2)
        self.layer4 = self._make_layer(256, 512, 3, 2)
        self.fc     = nn.Linear(512, num_classes)

    def _make_layer(self, in_planes, planes, num_blocks, stride):
        layers = [BasicBlock(in_planes, planes, stride)]
        for _ in range(num_blocks - 1):
            layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        return self.fc(x.view(x.size(0), -1))


# ── Loss Functions ────────────────────────────────────────────
def ce_loss(outputs, labels, reduction='mean'):
    return F.cross_entropy(outputs, labels, reduction=reduction)

def gce_loss(outputs, labels, q=0.7, reduction='mean'):
    probs   = F.softmax(outputs, 1)
    probs_y = probs[range(len(labels)), labels].clamp(1e-7)
    loss    = (1 - probs_y ** q) / q
    stab    = 1e-4 * F.cross_entropy(outputs, labels)
    if reduction == 'none':
        return loss + stab
    return loss.mean() + stab


# ── EWE Gate — Tuned for CIFAR-100N ──────────────────────────
class EWEGate:
    def __init__(self, alpha=0.45, beta=0.40, gamma=0.15,
                 k=EWE_K, tau=0.5, lam=EWE_LAM,
                 eps=0.1, ema_decay=0.99):
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.k     = k
        self.tau   = tau
        self.lam   = lam
        self.eps   = eps
        self.ema_decay  = ema_decay
        self._loss_ema  = None
        self._total     = 0
        self._accepted  = 0

    @property
    def acceptance_rate(self):
        return self._accepted / self._total if self._total > 0 else 0.0

    def gate(self, losses, logits):
        I     = torch.clamp(losses / (self.tau + losses), 0, 1)
        probs = F.softmax(logits, 1)
        A     = probs.max(1).values
        sim_x = 1.0 - losses / (losses.max() + 1e-8)
        R     = torch.clamp(sim_x - self.lam * A, min=0)
        mean_loss = losses.mean().item()
        if self._loss_ema is None:
            self._loss_ema = mean_loss
        else:
            self._loss_ema = (self.ema_decay * self._loss_ema +
                              (1 - self.ema_decay) * mean_loss)
        P    = torch.clamp(
            (losses - self._loss_ema) / (self._loss_ema + self.eps),
            0, 1
        )
        W     = self.alpha * I + self.beta * R + self.gamma * P
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


# ── Training Functions ────────────────────────────────────────
def train_standard_epoch(model, loader, optimizer):
    model.train()
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        ce_loss(model(x), y).backward()
        optimizer.step()

def train_ewe_epoch(model, loader, optimizer, gate):
    model.train()
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        losses  = ce_loss(outputs, y, reduction='none')
        mask    = gate.gate(losses.detach(), outputs.detach())
        if mask.sum() > 0:
            losses[mask].mean().backward()
            optimizer.step()

def train_ewe_gce_epoch(model, loader, optimizer, gate):
    model.train()
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        losses  = gce_loss(outputs, y, reduction='none')
        mask    = gate.gate(losses.detach(), outputs.detach())
        if mask.sum() > 0:
            losses[mask].mean().backward()
            optimizer.step()


# ── Run One Method ────────────────────────────────────────────
def run_method(method, train_loader, test_loader, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ResNet34CIFAR(NUM_CLASSES).to(device)
    opt   = torch.optim.SGD(model.parameters(), lr=LR,
                             momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    gate  = EWEGate()

    for ep in range(EPOCHS):
        ep1 = ep + 1

        if ep1 <= WARMUP:
            # Warmup phase — standard training
            train_standard_epoch(model, train_loader, opt)
        else:
            # EWE phase
            if method == 'ewe':
                train_ewe_epoch(model, train_loader, opt, gate)
            elif method == 'ewe-gce':
                train_ewe_gce_epoch(model, train_loader, opt, gate)

        sched.step()

        if ep1 % 10 == 0:
            acc  = evaluate(model, test_loader)
            rate = gate.acceptance_rate
            phase = 'WARMUP' if ep1 <= WARMUP else 'EWE'
            print(f'  Ep {ep1:3d} [{phase}] | '
                  f'Acc: {acc:.2f}% | Accept: {rate:.1%}')

    return evaluate(model, test_loader)


# ── Main ──────────────────────────────────────────────────────
def main():
    print('=' * 60)
    print('EWE CIFAR-100N Experiment v2 — Tuned Parameters')
    print(f'k={EWE_K}, lam={EWE_LAM}, warmup={WARMUP} epochs')
    print(f'Seeds: {SEEDS} | Epochs: {EPOCHS} | Device: {device}')
    print('=' * 60)

    train_set    = CIFAR100N('./data', transform_train,
                              NOISE_FILE, train=True)
    test_set     = CIFAR100N('./data', transform_test,
                              NOISE_FILE, train=False)
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    methods = ['ewe', 'ewe-gce']
    results = {m: [] for m in methods}

    for method in methods:
        print(f'\n── {method.upper()} (tuned) ──')
        for seed in SEEDS:
            print(f'  Seed {seed}')
            acc = run_method(method, train_loader, test_loader, seed)
            results[method].append(acc)
            print(f'  Final: {acc:.2f}%')

        mean = np.mean(results[method])
        std  = np.std(results[method])
        print(f'  {method}: {mean:.2f}% ± {std:.2f}%')

    # Summary with previous results
    print('\n' + '=' * 60)
    print('CIFAR-100N COMPLETE RESULTS')
    print('=' * 60)

    previous = {
        'standard':        (54.66, 0.70),
        'label-smoothing': (56.03, 0.62),
        'gce':             (58.40, 0.42),
        'co-teaching':     (64.16, 0.60),
    }

    baseline = 54.66
    print(f'{"standard":20s}: 54.66% ± 0.70%  (+0.00%)')
    print(f'{"label-smoothing":20s}: 56.03% ± 0.62%  (+1.37%)')
    print(f'{"gce":20s}: 58.40% ± 0.42%  (+3.73%)')
    print(f'{"co-teaching":20s}: 64.16% ± 0.60%  (+9.50%)')

    for method in methods:
        mean  = np.mean(results[method])
        std   = np.std(results[method])
        delta = mean - baseline
        sign  = '+' if delta >= 0 else ''
        print(f'{method + " (tuned)":20s}: {mean:.2f}% ± '
              f'{std:.2f}%  ({sign}{delta:.2f}%)')

    # Save
    output = {
        'dataset':      'CIFAR-100N',
        'architecture': 'ResNet-34',
        'ewe_params':   {'k': EWE_K, 'lam': EWE_LAM,
                         'warmup': WARMUP},
        'seeds':        SEEDS,
        'epochs':       EPOCHS,
        'timestamp':    datetime.now().isoformat(),
        'previous_results': {
            k: {'mean': v[0], 'std': v[1]}
            for k, v in previous.items()
        },
        'new_results': {
            m: {
                'per_seed': results[m],
                'mean':     float(np.mean(results[m])),
                'std':      float(np.std(results[m])),
                'vs_standard': float(
                    np.mean(results[m]) - baseline)
            } for m in methods
        }
    }

    with open('ewe_cifar100n_v2_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print('\nResults saved to ewe_cifar100n_v2_results.json')


if __name__ == '__main__':
    main()
