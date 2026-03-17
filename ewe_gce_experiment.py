# ============================================================
# EWE + GCE Combined Experiment
# Author: Maheep Purohit, Independent Researcher, Bikaner India
# DOI: 10.5281/zenodo.18940012
# ============================================================
# This script runs ONE additional method: EWE + GCE combined
# Uses same setup as main experiment for fair comparison
# Previous results loaded from ewe_results.json
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import os, json

def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s)

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')

# ── Load previous results ─────────────────────────────────────
assert os.path.exists('ewe_results.json'), 'Missing ewe_results.json'
with open('ewe_results.json', 'r') as f:
    prev = json.load(f)

print('\nPrevious results loaded:')
for mn, vals in prev['summary'].items():
    print(f'  {mn:<15}: {vals["mean"]:.2f} +/- {vals["std"]:.2f}%')

# ── Data ──────────────────────────────────────────────────────
noise_file = 'CIFAR-10_human.pt'
assert os.path.exists(noise_file), 'Missing CIFAR-10_human.pt'
noise_data   = torch.load(noise_file, map_location='cpu', weights_only=False)
noisy_labels = np.array(noise_data['worse_label'])
cifar_train  = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
clean_labels = np.array(cifar_train.targets)
noise_rate   = float((noisy_labels != clean_labels).mean())
print(f'\nNoise rate: {noise_rate*100:.1f}%')

T_tr = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
])
T_te = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
])

class NoisyDataset(Dataset):
    def __init__(self, labels, transform):
        self.ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        self.labels = labels; self.transform = transform
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        img, _ = self.ds[i]
        return self.transform(img), int(self.labels[i]), i

tr_ds = NoisyDataset(noisy_labels, T_tr)
te_ds = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=T_te)
tr_ld = DataLoader(tr_ds, 128, shuffle=True,  num_workers=0)
te_ld = DataLoader(te_ds, 256, shuffle=False, num_workers=0)

# ── Model ─────────────────────────────────────────────────────
def get_model():
    m = torchvision.models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(3,64,3,1,1,bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512,10)
    return m.to(device)

def acc(model, loader):
    model.eval(); c=t=0
    with torch.no_grad():
        for b in loader:
            x,y = b[0].to(device), b[1].to(device)
            c += model(x).max(1)[1].eq(y).sum().item(); t += y.size(0)
    return 100.*c/t

# ── EWE gate ──────────────────────────────────────────────────
class EWE:
    def __init__(self, alpha=0.45, beta=0.40, gamma=0.15,
                 k=0.25, tau=0.5, lam=0.5, eps=0.1, ema=0.99):
        self.alpha=alpha; self.beta=beta; self.gamma=gamma
        self.k=k; self.tau=tau; self.lam=lam; self.eps=eps; self.ema=ema
        self.loss_ema=None; self.total=0; self.accepted=0

    @property
    def rate(self): return self.accepted/self.total if self.total>0 else 0.

    def gate(self, losses, logits):
        I     = torch.clamp(losses/(self.tau+losses), 0., 1.)
        probs = F.softmax(logits, dim=1)
        A     = probs.max(1).values
        sim   = 1. - losses/(losses.max()+1e-8)
        R     = torch.clamp(sim - self.lam*A, min=0.)
        ml    = losses.mean().item()
        if self.loss_ema is None: self.loss_ema=ml
        else: self.loss_ema=self.ema*self.loss_ema+(1-self.ema)*ml
        P     = torch.clamp((losses-self.loss_ema)/(self.loss_ema+self.eps),0.,1.)
        W     = self.alpha*I + self.beta*R + self.gamma*P
        theta = W.mean() - self.k*W.std()
        mask  = W >= theta
        self.total += losses.shape[0]; self.accepted += int(mask.sum())
        return mask

# ── EWE + GCE training function ───────────────────────────────
def train_ewe_gce(model, loader, optimizer, ewe, q=0.7):
    """
    EWE gate applied with GCE loss instead of CrossEntropy.
    GCE provides noise-robust loss values.
    EWE gate then filters which of those samples update parameters.
    Two complementary mechanisms working at different levels.
    """
    model.train()
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)

        # Per-sample GCE loss
        probs   = F.softmax(outputs, dim=1)
        probs_y = probs[torch.arange(len(labels)), labels].clamp(min=1e-7)
        losses  = (1 - probs_y**q) / q  # per-sample GCE [B]

        # EWE gate decision using GCE losses as signal
        mask = ewe.gate(losses.detach(), outputs.detach())

        # Only gate-accepted samples update parameters
        if mask.sum() > 0:
            losses[mask].mean().backward()
            optimizer.step()

def make_opt(m): return optim.SGD(m.parameters(),lr=0.1,momentum=0.9,weight_decay=5e-4)
def make_sch(o,e): return optim.lr_scheduler.CosineAnnealingLR(o,T_max=e)

# ── Run EWE+GCE experiment ────────────────────────────────────
NE = 50; NS = 3
ewe_gce_results = []
ewe_gce_rates   = []

print('\n' + '='*60)
print('EWE + GCE Combined Experiment')
print(f'Epochs:{NE} | Seeds:{NS} | Device:{device}')
print('Hypothesis: EWE gate + GCE loss > both individually')
print('='*60)

for seed in range(NS):
    print(f'\n{"="*60}\nSEED {seed+1}/{NS}\n{"="*60}')
    set_seed(seed*42+1)

    m   = get_model()
    o   = make_opt(m)
    s   = make_sch(o, NE)
    ewe = EWE()

    for ep in range(NE):
        train_ewe_gce(m, tr_ld, o, ewe); s.step()
        if (ep+1)%10==0:
            a = acc(m, te_ld)
            print(f'  Ep{ep+1:3d}/{NE} | Acc:{a:.2f}% | Accept:{ewe.rate:.1%}')

    final_acc = acc(m, te_ld)
    ewe_gce_results.append(final_acc)
    ewe_gce_rates.append(ewe.rate)
    print(f'  Final: {final_acc:.2f}% | Accept: {ewe.rate:.1%}')

# ── Results ───────────────────────────────────────────────────
mean_egce = float(np.mean(ewe_gce_results))
std_egce  = float(np.std(ewe_gce_results))

print('\n' + '='*60)
print('COMPLETE RESULTS — All Methods')
print('='*60)

all_summary = dict(prev['summary'])
all_summary['EWE+GCE'] = {'mean': mean_egce, 'std': std_egce}
all_results = dict(prev['results'])
all_results['EWE+GCE'] = ewe_gce_results

methods_display = ['Standard','EWE','GCE','LabelSmoothing','CoTeaching','EWE+GCE']
for mn in methods_display:
    s_data = all_summary[mn]
    marker = ' <-- YOUR METHOD' if mn == 'EWE' else ''
    marker = ' <-- COMBINED (NEW)' if mn == 'EWE+GCE' else marker
    print(f'{mn:<15}: {s_data["mean"]:.2f} +/- {s_data["std"]:.2f}%{marker}')

print(f'\nEWE+GCE vs Standard : {mean_egce - all_summary["Standard"]["mean"]:+.2f}%')
print(f'EWE+GCE vs EWE      : {mean_egce - all_summary["EWE"]["mean"]:+.2f}%')
print(f'EWE+GCE vs GCE      : {mean_egce - all_summary["GCE"]["mean"]:+.2f}%')
print(f'EWE+GCE accept rate : {float(np.mean(ewe_gce_rates)):.1%}')

# Save updated results
with open('ewe_combined_results.json', 'w') as f:
    json.dump({
        'results': all_results,
        'summary': all_summary,
        'ewe_gce_rates': ewe_gce_rates
    }, f, indent=2)
print('\nSaved: ewe_combined_results.json')

# ── Figure ────────────────────────────────────────────────────
print('Generating updated figure...')
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.patch.set_facecolor('white')

ml     = ['Standard', 'EWE\n(ours)', 'GCE', 'Label\nSmooth', 'Co-teach', 'EWE+GCE\n(ours)']
means  = [all_summary[m]['mean'] for m in methods_display]
stds   = [all_summary[m]['std']  for m in methods_display]
colors = ['#888888','#1E3A5F','#1565A0','#1B6B3A','#7A5C10','#8B0000']

ax = axes[0]
bars = ax.bar(ml, means, yerr=stds, color=colors, capsize=5, width=0.6)
bars[1].set_edgecolor('#FF6B00'); bars[1].set_linewidth(2.5)
bars[5].set_edgecolor('#FF6B00'); bars[5].set_linewidth(2.5)
ax.set_ylabel('Test Accuracy (%)')
ax.set_title('CIFAR-10N — All Methods\n(Real Human Label Noise ~40%)', fontweight='bold')
ax.set_ylim(min(means)-3, max(means)+3)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
for bar, mean, std in zip(bars, means, stds):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+std+0.3,
            f'{mean:.1f}%', ha='center', fontsize=8, fontweight='bold')

ax2 = axes[1]
for i, (mn, col) in enumerate(zip(methods_display, colors)):
    vals = all_results[mn]
    ax2.scatter([i]*len(vals), vals, color=col, s=80, zorder=3)
    ax2.plot([i-.2,i+.2], [np.mean(vals)]*2, color=col, lw=2.5, zorder=4)
ax2.set_xticks(range(len(methods_display)))
ax2.set_xticklabels(ml, fontsize=8)
ax2.set_ylabel('Test Accuracy (%)')
ax2.set_title('Per-Seed Results\n(each dot = one seed)', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

plt.suptitle('Epistemic Weight Engine — Complete Method Comparison',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('ewe_combined_results.png', dpi=200,
            bbox_inches='tight', facecolor='white')
print('Saved: ewe_combined_results.png')

# ── Paper table ───────────────────────────────────────────────
print('\n' + '='*60)
print('FINAL TABLE FOR PAPER')
print('='*60)
lmap = ['Standard Training','EWE (ours)','GCE',
        'Label Smoothing','Co-teaching','EWE+GCE (ours)']
print(f'{"Method":<22} {"Acc(%)":>8} {"+-Std":>6}')
print('-'*40)
for mn, label in zip(methods_display, lmap):
    mk = ' <--' if mn in ['EWE','EWE+GCE'] else ''
    print(f'{label:<22} {all_summary[mn]["mean"]:>8.2f} '
          f'{all_summary[mn]["std"]:>6.2f}{mk}')
print('-'*40)
print(f'CIFAR-10N | ~40% noise | ResNet-18 | {NS} seeds | {NE} epochs')
