"""
DivideMix Baseline Experiment
Author: Maheep Purohit, Independent Researcher, Bikaner, India
Paper: https://doi.org/10.5281/zenodo.18940011

Implements DivideMix (Li et al. 2020) as a baseline comparison.
DivideMix uses GMM to divide data into clean/noisy sets,
then trains with MixUp on clean samples.

Reference: Li et al. "DivideMix: Learning with Noisy Labels as
Semi-Supervised Learning" ICLR 2020.

Runs on:
- CIFAR-10N (real human noise ~40.2%)
- CIFAR-100N (real human noise ~40.2%)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import json
import os
from datetime import datetime
from sklearn.mixture import GaussianMixture

# ── Device ────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ── Config ────────────────────────────────────────────────────
SEEDS      = [42, 123, 456, 789, 1024]
EPOCHS     = 100
WARMUP     = 10
BATCH_SIZE = 128
LR         = 0.02
ALPHA      = 4.0    # MixUp parameter
LAMBDA_U   = 25     # Unsupervised loss weight
CHECKPOINT = 'dividemix_checkpoint.json'

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
            noise_data = torch.load(
                noise_file, weights_only=False)
            if 'noisy_label' in noise_data:
                label_key = 'noisy_label'
            elif 'aggre_label' in noise_data:
                label_key = 'aggre_label'
            else:
                label_key = list(noise_data.keys())[1]
            self.noisy_labels = np.array(
                noise_data[label_key])
            noise_rate = (
                self.noisy_labels !=
                np.array(noise_data['clean_label'])
            ).mean()
            print(f'{dataset_name} noise rate: '
                  f'{noise_rate:.1%}')
        else:
            self.noisy_labels = np.array(
                self.dataset.targets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        label  = int(self.noisy_labels[idx]) if self.train \
                 else int(self.dataset.targets[idx])
        return img, label, idx


# ── Model ─────────────────────────────────────────────────────
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3,
                               stride=stride,
                               padding=1, bias=False)
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
        cfg = [2,2,2,2] if depth == 18 else [3,4,6,3]
        self.conv1  = nn.Conv2d(3, 64, 3,
                                padding=1, bias=False)
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


# ── GMM Division ──────────────────────────────────────────────
def gmm_divide(model, loader, num_classes):
    """Use GMM to divide samples into clean and noisy sets."""
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            out  = model(x)
            loss = F.cross_entropy(
                out, y, reduction='none')
            losses.append(loss.cpu())
    losses = torch.cat(losses).numpy()

    # Normalize losses
    losses = (losses - losses.min()) / \
             (losses.max() - losses.min() + 1e-8)
    losses = losses.reshape(-1, 1)

    # Fit GMM with 2 components
    gmm = GaussianMixture(n_components=2,
                          max_iter=10, tol=1e-2,
                          reg_covar=5e-4)
    gmm.fit(losses)

    # Clean set = component with lower mean loss
    prob   = gmm.predict_proba(losses)
    clean_idx = gmm.means_.argmin()
    clean_prob = prob[:, clean_idx]
    return clean_prob


# ── MixUp ────────────────────────────────────────────────────
def mixup(x1, y1, x2, y2, alpha, num_classes):
    lam  = np.random.beta(alpha, alpha)
    lam  = max(lam, 1 - lam)
    x    = lam * x1 + (1 - lam) * x2
    y1oh = F.one_hot(y1, num_classes).float()
    y2oh = F.one_hot(y2, num_classes).float()
    y    = lam * y1oh + (1 - lam) * y2oh
    return x, y


# ── DivideMix Training ────────────────────────────────────────
def train_dividemix_epoch(model1, model2, loader,
                          opt1, num_classes, epoch):
    model1.train()
    model2.eval()

    for x, y, idx in loader:
        x, y = x.to(device), y.to(device)
        n    = x.size(0)
        if n < 2:
            continue

        # Split batch into two halves for MixUp
        half = n // 2
        x1, y1 = x[:half], y[:half]
        x2, y2 = x[half:half*2], y[half:half*2]

        # MixUp
        xm, ym = mixup(x1, y1, x2, y2,
                       ALPHA, num_classes)

        # Forward
        opt1.zero_grad()
        out = model1(xm)
        log_soft = F.log_softmax(out, dim=1)
        loss = -(ym * log_soft).sum(1).mean()
        loss.backward()
        opt1.step()


# ── Run One Seed ──────────────────────────────────────────────
def run_seed(train_loader, test_loader,
             num_classes, depth, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Two networks as per DivideMix
    model1 = ResNetCIFAR(num_classes, depth).to(device)
    model2 = ResNetCIFAR(num_classes, depth).to(device)

    opt1 = torch.optim.SGD(
        model1.parameters(), lr=LR,
        momentum=0.9, weight_decay=5e-4)
    opt2 = torch.optim.SGD(
        model2.parameters(), lr=LR,
        momentum=0.9, weight_decay=5e-4)

    sched1 = torch.optim.lr_scheduler\
        .CosineAnnealingLR(opt1, EPOCHS)
    sched2 = torch.optim.lr_scheduler\
        .CosineAnnealingLR(opt2, EPOCHS)

    for ep in range(1, EPOCHS + 1):
        # Warmup — standard training
        if ep <= WARMUP:
            model1.train()
            for x, y, _ in train_loader:
                x, y = x.to(device), y.to(device)
                opt1.zero_grad()
                F.cross_entropy(model1(x), y).backward()
                opt1.step()
            model2.train()
            for x, y, _ in train_loader:
                x, y = x.to(device), y.to(device)
                opt2.zero_grad()
                F.cross_entropy(model2(x), y).backward()
                opt2.step()
        else:
            # DivideMix training
            train_dividemix_epoch(
                model1, model2, train_loader,
                opt1, num_classes, ep)
            train_dividemix_epoch(
                model2, model1, train_loader,
                opt2, num_classes, ep)

        sched1.step()
        sched2.step()

        if ep % 10 == 0:
            acc = evaluate(model1, test_loader)
            print(f'  Ep {ep:3d} | Acc: {acc:.2f}%')

    return evaluate(model1, test_loader)


# ── Load / Save Checkpoint ────────────────────────────────────
def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            data = json.load(f)
            print(f'Resuming — '
                  f'{len(data)} seeds done')
            return data
    return {}

def save_checkpoint(data):
    with open(CHECKPOINT, 'w') as f:
        json.dump(data, f, indent=2)


# ── Main ──────────────────────────────────────────────────────
def main():
    print('DIVIDEMIX BASELINE EXPERIMENT')
    print(f'Seeds: {SEEDS} | Epochs: {EPOCHS}')
    print(f'Device: {device}')
    print('='*60)

    checkpoint = load_checkpoint()

    datasets = [
        {
            'name':        'cifar10n',
            'torch_name':  'cifar10',
            'noise_file':  './CIFAR-10_human.pt',
            'num_classes': 10,
            'depth':       18,
        },
        {
            'name':        'cifar100n',
            'torch_name':  'cifar100',
            'noise_file':  './CIFAR-100_human.pt',
            'num_classes': 100,
            'depth':       34,
        },
    ]

    all_results = {}

    for ds in datasets:
        dsname = ds['name']
        print(f'\n{"="*60}')
        print(f'Dataset: {dsname.upper()} | '
              f'Classes: {ds["num_classes"]}')
        print('='*60)

        train_tf, test_tf = get_transforms(
            ds['torch_name'])
        train_set = NoisyDataset(
            './data', train_tf, ds['noise_file'],
            ds['torch_name'], train=True)
        test_set  = NoisyDataset(
            './data', test_tf, ds['noise_file'],
            ds['torch_name'], train=False)
        train_loader = DataLoader(
            train_set, BATCH_SIZE, shuffle=True,
            num_workers=2, pin_memory=True)
        test_loader  = DataLoader(
            test_set, BATCH_SIZE, shuffle=False,
            num_workers=2, pin_memory=True)

        results = []

        for seed in SEEDS:
            key = f'{dsname}_{seed}'

            if key in checkpoint:
                acc = checkpoint[key]
                print(f'  Seed {seed} → '
                      f'{acc:.2f}% [RESUMED]')
                results.append(acc)
                continue

            print(f'  Seed {seed} — running...')
            acc = run_seed(
                train_loader, test_loader,
                ds['num_classes'], ds['depth'], seed)
            results.append(acc)
            print(f'  Seed {seed} → Final: {acc:.2f}%')

            checkpoint[key] = acc
            save_checkpoint(checkpoint)
            print(f'  [Checkpoint saved]')

        mean = np.mean(results)
        std  = np.std(results)
        all_results[dsname] = {
            'per_seed': results,
            'mean': float(mean),
            'std':  float(std)
        }
        print(f'\n  DivideMix {dsname}: '
              f'{mean:.2f}% ± {std:.2f}%')

    # ── Summary ───────────────────────────────────────────────
    print('\n' + '='*60)
    print('DIVIDEMIX RESULTS SUMMARY')
    print('='*60)

    previous = {
        'cifar10n': {
            'standard':       72.37,
            'gce':            81.74,
            'co-teaching':    85.46,
            'adaptive-ewe':   90.26,
            'adaptive-ewegce':91.95,
        },
        'cifar100n': {
            'standard':       54.66,
            'gce':            58.40,
            'co-teaching':    64.16,
            'adaptive-ewe':   54.24,
            'adaptive-ewegce':58.90,
        }
    }

    for dsname, res in all_results.items():
        base = previous[dsname]['standard']
        mean = res['mean']
        std  = res['std']
        print(f'\n{dsname.upper()}:')
        print(f'  DivideMix: {mean:.2f}% ± {std:.2f}%  '
              f'(vs standard: {mean-base:+.2f}%)')
        for method, val in previous[dsname].items():
            diff = mean - val
            sign = '+' if diff >= 0 else ''
            print(f'  vs {method:20s}: {sign}{diff:.2f}%')

    # Save final
    output = {
        'experiment': 'DivideMix Baseline',
        'timestamp':  datetime.now().isoformat(),
        'seeds':      SEEDS,
        'epochs':     EPOCHS,
        'results':    all_results,
        'previous':   previous
    }
    with open('dividemix_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print('\nResults saved to dividemix_results.json')


if __name__ == '__main__':
    main()
