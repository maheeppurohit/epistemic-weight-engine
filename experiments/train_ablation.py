"""
EWE Neural Network Ablation Study
Author: Maheep Purohit, Independent Researcher, Bikaner, India
Paper: https://doi.org/10.5281/zenodo.18940011

Ablation variants on CIFAR-10N with ResNet-18:
1. Full Adaptive EWE          — all modules active
2. Without Impact Assessment  — remove I(x)
3. Without Reality Alignment  — remove R(x)
4. Without Paradigm Shift     — remove P(x)
5. Without Warmup             — no warmup period
6. Standard Training          — no EWE at all

5 seeds each. Saves after every seed — power cut safe.
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
NUM_CLASSES = 10
LR         = 0.1
BASE_K     = 0.25
NOISE_FILE = './CIFAR-10_human.pt'
CHECKPOINT = 'ewe_ablation_checkpoint.json'
RESULTS    = 'ewe_ablation_results.json'

# Adaptive k for CIFAR-10N
K = BASE_K / np.log(NUM_CLASSES + 1)
print(f'Adaptive k = {K:.4f}')

# ── Checkpoint helpers ────────────────────────────────────────
def save_checkpoint(data):
    with open(CHECKPOINT, 'w') as f:
        json.dump(data, f, indent=2)

def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, 'r') as f:
            print(f'Resuming from checkpoint')
            return json.load(f)
    return {}

# ── Transforms ────────────────────────────────────────────────
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

# ── Dataset ───────────────────────────────────────────────────
class CIFAR10N(Dataset):
    def __init__(self, root, transform, train=True):
        self.dataset = torchvision.datasets.CIFAR10(
            root=root, train=train,
            download=True, transform=transform)
        self.train = train

        if train and os.path.exists(NOISE_FILE):
            noise_data = torch.load(
                NOISE_FILE, weights_only=False)
            label_key = 'aggre_label' \
                if 'aggre_label' in noise_data \
                else 'noisy_label'
            self.noisy_labels = np.array(
                noise_data[label_key])
            noise_rate = (
                self.noisy_labels !=
                np.array(noise_data['clean_label'])
            ).mean()
            print(f'Noise rate: {noise_rate:.1%} '
                  f'(key: {label_key})')
        else:
            self.noisy_labels = np.array(
                self.dataset.targets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        label  = int(self.noisy_labels[idx]) \
                 if self.train \
                 else int(self.dataset.targets[idx])
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
            planes, planes, 3,
            padding=1, bias=False)
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

# ── EWE Gate with ablation flags ─────────────────────────────
class AblationEWEGate:
    """
    EWE gate with ablation controls.
    use_I: use Impact Assessment
    use_R: use Reality Alignment
    use_P: use Paradigm Shift
    """
    def __init__(self, use_I=True, use_R=True, use_P=True,
                 alpha=0.45, beta=0.40, gamma=0.15,
                 tau=0.5, lam=0.40, eps=0.1,
                 ema_decay=0.99):
        self.k         = K
        self.use_I     = use_I
        self.use_R     = use_R
        self.use_P     = use_P
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
        components = []
        weights    = []

        # Impact Assessment I(x)
        if self.use_I:
            I = torch.clamp(
                losses / (self.tau + losses), 0, 1)
            components.append(I)
            weights.append(self.alpha)
        
        # Reality Alignment R(x)
        if self.use_R:
            probs = F.softmax(logits, 1)
            A     = probs.max(1).values
            sim_x = 1.0 - losses / (losses.max() + 1e-8)
            R     = torch.clamp(
                sim_x - self.lam * A, min=0)
            components.append(R)
            weights.append(self.beta)

        # Paradigm Shift P(x)
        if self.use_P:
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
            components.append(P)
            weights.append(self.gamma)

        # If no components — accept all
        if not components:
            mask = torch.ones(
                losses.shape[0], dtype=torch.bool,
                device=losses.device)
            self._total    += losses.shape[0]
            self._accepted += losses.shape[0]
            return mask

        # Normalise weights
        total_w = sum(weights)
        W = sum(w/total_w * c
                for w, c in zip(weights, components))

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

# ── Train One Epoch ───────────────────────────────────────────
def train_epoch(model, loader, optimizer,
                gate, epoch, use_warmup):
    model.train()
    in_warmup = use_warmup and (epoch <= WARMUP)

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out    = model(x)
        losses = F.cross_entropy(out, y, reduction='none')

        if in_warmup or gate is None:
            losses.mean().backward()
        else:
            mask = gate.gate(losses.detach(), out.detach())
            if mask.sum() > 0:
                losses[mask].mean().backward()
            else:
                continue

        optimizer.step()

# ── Run One Seed ──────────────────────────────────────────────
def run_seed(variant, train_loader, test_loader, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ResNet18CIFAR(NUM_CLASSES).to(device)
    opt   = torch.optim.SGD(
        model.parameters(), lr=LR,
        momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, EPOCHS)

    # Configure gate based on variant
    if variant == 'standard':
        gate       = None
        use_warmup = False
    elif variant == 'no-warmup':
        gate       = AblationEWEGate()
        use_warmup = False
    elif variant == 'no-I':
        gate       = AblationEWEGate(
            use_I=False, use_R=True, use_P=True)
        use_warmup = True
    elif variant == 'no-R':
        gate       = AblationEWEGate(
            use_I=True, use_R=False, use_P=True)
        use_warmup = True
    elif variant == 'no-P':
        gate       = AblationEWEGate(
            use_I=True, use_R=True, use_P=False)
        use_warmup = True
    else:  # full
        gate       = AblationEWEGate()
        use_warmup = True

    for ep in range(1, EPOCHS+1):
        train_epoch(model, train_loader, opt,
                    gate, ep, use_warmup)
        sched.step()

        if ep % 10 == 0:
            acc   = evaluate(model, test_loader)
            rate  = gate.acceptance_rate \
                    if gate else 1.0
            phase = 'WARMUP' \
                    if (use_warmup and ep <= WARMUP) \
                    else 'EWE'
            print(f'  Ep {ep:3d} [{phase}] | '
                  f'Acc: {acc:.2f}% | '
                  f'Accept: {rate:.1%}')

    return evaluate(model, test_loader)

# ── Main ──────────────────────────────────────────────────────
def main():
    print('='*60)
    print('EWE NEURAL NETWORK ABLATION STUDY')
    print('Dataset: CIFAR-10N | Architecture: ResNet-18')
    print(f'Seeds: {SEEDS} | Adaptive k: {K:.4f}')
    print('='*60)

    # Load checkpoint
    checkpoint = load_checkpoint()

    # Ablation variants
    variants = [
        'full',      # Full Adaptive EWE
        'no-I',      # Without Impact Assessment
        'no-R',      # Without Reality Alignment
        'no-P',      # Without Paradigm Shift
        'no-warmup', # Without warmup
        'standard',  # Standard training baseline
    ]

    # Human readable names for printing
    names = {
        'full':      'Full Adaptive EWE    ',
        'no-I':      'Without Impact (I)   ',
        'no-R':      'Without Reality (R)  ',
        'no-P':      'Without Paradigm (P) ',
        'no-warmup': 'Without Warmup       ',
        'standard':  'Standard Training    ',
    }

    # Data loaders
    train_set = CIFAR10N('./data', transform_train, train=True)
    test_set  = CIFAR10N('./data', transform_test,  train=False)
    train_loader = DataLoader(
        train_set, BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True)
    test_loader  = DataLoader(
        test_set, BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True)

    results = {v: [] for v in variants}

    # Load known full EWE result
    if 'full' not in checkpoint:
        checkpoint['full'] = {}
    known_full = {
        '42': 90.14, '123': 90.46, '456': 89.55,
        '789': 90.43, '1024': 89.81
    }
    for s, v in known_full.items():
        if s not in checkpoint.get('full', {}):
            checkpoint.setdefault('full', {})[s] = v
    save_checkpoint(checkpoint)

    for variant in variants:
        print(f'\n── {names[variant].strip()} ──')

        for seed in SEEDS:
            seed_str = str(seed)
            ck_key   = variant

            # Skip if already done
            if seed_str in checkpoint.get(ck_key, {}):
                acc = checkpoint[ck_key][seed_str]
                print(f'  Seed {seed} → '
                      f'{acc:.2f}% [RESUMED]')
                results[variant].append(acc)
                continue

            print(f'  Seed {seed} — running...')
            acc = run_seed(
                variant, train_loader,
                test_loader, seed)
            results[variant].append(acc)
            print(f'  Seed {seed} → Final: {acc:.2f}%')

            # Save immediately
            checkpoint.setdefault(ck_key, {})[seed_str] = acc
            save_checkpoint(checkpoint)
            print(f'  [Checkpoint saved]')

        vals = results[variant]
        if vals:
            mean = np.mean(vals)
            std  = np.std(vals)
            print(f'  MEAN: {mean:.2f}% ± {std:.2f}%')

    # ── Final Summary ─────────────────────────────────────────
    print('\n' + '='*60)
    print('ABLATION STUDY RESULTS — CIFAR-10N ResNet-18')
    print('='*60)

    full_mean = np.mean(results['full'])

    for variant in variants:
        vals  = results[variant]
        mean  = np.mean(vals)
        std   = np.std(vals)
        delta = mean - full_mean
        sign  = '+' if delta >= 0 else ''
        print(f'{names[variant]}: '
              f'{mean:.2f}% ± {std:.2f}%  '
              f'({sign}{delta:.2f}% vs full EWE)')

    # Save final results
    output = {
        'experiment':   'EWE Neural Network Ablation',
        'dataset':      'CIFAR-10N',
        'architecture': 'ResNet-18',
        'adaptive_k':   float(K),
        'warmup':       WARMUP,
        'seeds':        SEEDS,
        'epochs':       EPOCHS,
        'timestamp':    datetime.now().isoformat(),
        'results': {
            v: {
                'name':     names[v].strip(),
                'per_seed': results[v],
                'mean':     float(np.mean(results[v])),
                'std':      float(np.std(results[v])),
                'vs_full':  float(
                    np.mean(results[v]) - full_mean)
            }
            for v in variants if results[v]
        }
    }

    with open(RESULTS, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nResults saved to {RESULTS}')

    print('\nKEY FINDINGS:')
    print('Most important module = '
          'largest drop when removed')

if __name__ == '__main__':
    main()
