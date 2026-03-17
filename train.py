# ============================================================
# Epistemic Weight Engine (EWE) — CIFAR-10N Experiment
# Author: Maheep Purohit, Independent Researcher, Bikaner India
# DOI: 10.5281/zenodo.18940012
# Version: 5.0 — Adaptive threshold, simulation-verified
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

# ── Data ──────────────────────────────────────────────────────
noise_file = 'CIFAR-10_human.pt'
assert os.path.exists(noise_file), 'Missing CIFAR-10_human.pt'
noise_data   = torch.load(noise_file, map_location='cpu', weights_only=False)
noisy_labels = np.array(noise_data['worse_label'])
cifar_train  = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
clean_labels = np.array(cifar_train.targets)
noise_rate   = float((noisy_labels != clean_labels).mean())
print(f'Noise rate: {noise_rate*100:.1f}%')

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
print(f'Train:{len(tr_ds)} Test:{len(te_ds)}')

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

# ── EWE v5 — Adaptive Threshold ───────────────────────────────
class EWE:
    """
    EWE v5.0 — Key fix: ADAPTIVE threshold.

    Problem with fixed theta: W distribution shifts dramatically
    during training (mean W drops from ~0.61 early to ~0.45 late).
    Any fixed threshold either accepts everything early or nothing late.

    Solution: threshold = W_mean - k * W_std (computed per batch)
    This keeps acceptance rate stable across all training stages
    at approximately (1 - percentile(k)) regardless of W distribution.

    k=0.25 targets ~60% acceptance rate throughout training.
    k=0.50 targets ~50% acceptance rate throughout training.

    Three decoupled signals:
    I(x) = L(x)/(tau+L(x))           — gradient impact proxy
    R(x) = max(0, sim(x) - lam*A(x)) — reality alignment
           sim(x) = 1 - L(x)/max(L)  — inverse loss (decoupled from A)
           A(x)   = max softmax       — approval signal
    P(x) = max(0,(L(x)-L_ema)/(L_ema+eps)) — paradigm shift

    W(x) = alpha*I + beta*R + gamma*P
    Gate: W(x) >= W_mean - k*W_std   (adaptive)
    """
    def __init__(self, alpha=0.45, beta=0.40, gamma=0.15,
                 k=0.25, tau=0.5, lam=0.5, eps=0.1, ema=0.99):
        self.alpha=alpha; self.beta=beta; self.gamma=gamma
        self.k=k; self.tau=tau; self.lam=lam; self.eps=eps; self.ema=ema
        self.loss_ema=None; self.total=0; self.accepted=0

    @property
    def rate(self): return self.accepted/self.total if self.total>0 else 0.

    def gate(self, losses, logits):
        # I(x)
        I = torch.clamp(losses/(self.tau+losses), 0., 1.)
        # R(x) — decoupled
        probs = F.softmax(logits, dim=1)
        A     = probs.max(1).values
        sim   = 1. - losses/(losses.max()+1e-8)
        R     = torch.clamp(sim - self.lam*A, min=0.)
        # P(x)
        ml = losses.mean().item()
        if self.loss_ema is None: self.loss_ema=ml
        else: self.loss_ema=self.ema*self.loss_ema+(1-self.ema)*ml
        P = torch.clamp((losses-self.loss_ema)/(self.loss_ema+self.eps),0.,1.)
        # Adaptive threshold
        W     = self.alpha*I + self.beta*R + self.gamma*P
        theta = W.mean() - self.k*W.std()
        mask  = W >= theta
        self.total += losses.shape[0]; self.accepted += int(mask.sum())
        return mask

# ── Loss functions ────────────────────────────────────────────
class GCE(nn.Module):
    def __init__(self, q=0.7): super().__init__(); self.q=q
    def forward(self, logits, labels):
        p = F.softmax(logits,1)[torch.arange(len(labels)),labels].clamp(1e-7)
        return ((1-p**self.q)/self.q).mean() + 1e-4*F.cross_entropy(logits,labels)

class LabelSmooth(nn.Module):
    def __init__(self, s=0.1, nc=10): super().__init__(); self.s=s; self.nc=nc
    def forward(self, logits, labels):
        oh = torch.zeros_like(logits).scatter_(1,labels.unsqueeze(1),1)
        sm = oh*(1-self.s) + (1-oh)*self.s/(self.nc-1)
        return -(sm*F.log_softmax(logits,1)).sum(1).mean()

# ── Training ──────────────────────────────────────────────────
def train_std(m, ld, opt, crit):
    m.train()
    for x,y,_ in ld:
        x,y=x.to(device),y.to(device)
        opt.zero_grad(); crit(m(x),y).backward(); opt.step()

def train_ewe(m, ld, opt, ewe):
    m.train(); ce=nn.CrossEntropyLoss(reduction='none')
    for x,y,_ in ld:
        x,y=x.to(device),y.to(device)
        opt.zero_grad(); out=m(x); L=ce(out,y)
        mask=ewe.gate(L.detach(), out.detach())
        if mask.sum()>0: L[mask].mean().backward(); opt.step()

def train_cot(m1,m2,ld,o1,o2,ep,ne,nr=0.4):
    m1.train(); m2.train(); ce=nn.CrossEntropyLoss(reduction='none')
    rem=1-min(nr,nr*min(ep/(ne*0.5),1.))
    for x,y,_ in ld:
        x,y=x.to(device),y.to(device); nR=max(1,int(rem*x.size(0)))
        with torch.no_grad():
            i1=ce(m1(x),y).argsort()[:nR]; i2=ce(m2(x),y).argsort()[:nR]
        o1.zero_grad(); ce(m1(x[i2]),y[i2]).mean().backward(); o1.step()
        o2.zero_grad(); ce(m2(x[i1]),y[i1]).mean().backward(); o2.step()

def mk_opt(m): return optim.SGD(m.parameters(),lr=0.1,momentum=0.9,weight_decay=5e-4)
def mk_sch(o,e): return optim.lr_scheduler.CosineAnnealingLR(o,T_max=e)

# ── Experiment ────────────────────────────────────────────────
NE=50; NS=3
methods=['Standard','EWE','GCE','LabelSmoothing','CoTeaching']
results={m:[] for m in methods}; ewe_rates=[]

print('='*60)
print('EWE CIFAR-10N v5.0 — Adaptive Threshold')
print(f'Epochs:{NE} Seeds:{NS} Device:{device}')
print('Expected EWE acceptance: ~60% stable throughout training')
print('='*60)

for seed in range(NS):
    print(f'\n{"="*60}\nSEED {seed+1}/{NS}\n{"="*60}')
    set_seed(seed*42+1); sr={}

    # Standard
    print('\n[1/5] Standard...')
    m=get_model(); o=mk_opt(m); s=mk_sch(o,NE)
    for ep in range(NE):
        train_std(m,tr_ld,o,nn.CrossEntropyLoss()); s.step()
        if (ep+1)%10==0: print(f'  Ep{ep+1} Acc:{acc(m,te_ld):.2f}%')
    sr['Standard']=acc(m,te_ld); print(f'  Final:{sr["Standard"]:.2f}%')

    # EWE
    print('\n[2/5] EWE...')
    m=get_model(); o=mk_opt(m); s=mk_sch(o,NE); ewe=EWE()
    for ep in range(NE):
        train_ewe(m,tr_ld,o,ewe); s.step()
        if (ep+1)%10==0:
            print(f'  Ep{ep+1} Acc:{acc(m,te_ld):.2f}% Accept:{ewe.rate:.1%}')
    sr['EWE']=acc(m,te_ld); ewe_rates.append(ewe.rate)
    print(f'  Final:{sr["EWE"]:.2f}% Accept:{ewe.rate:.1%}')

    # GCE
    print('\n[3/5] GCE...')
    m=get_model(); o=mk_opt(m); s=mk_sch(o,NE)
    for ep in range(NE):
        train_std(m,tr_ld,o,GCE()); s.step()
        if (ep+1)%10==0: print(f'  Ep{ep+1} Acc:{acc(m,te_ld):.2f}%')
    sr['GCE']=acc(m,te_ld); print(f'  Final:{sr["GCE"]:.2f}%')

    # Label Smoothing
    print('\n[4/5] LabelSmoothing...')
    m=get_model(); o=mk_opt(m); s=mk_sch(o,NE)
    for ep in range(NE):
        train_std(m,tr_ld,o,LabelSmooth()); s.step()
        if (ep+1)%10==0: print(f'  Ep{ep+1} Acc:{acc(m,te_ld):.2f}%')
    sr['LabelSmoothing']=acc(m,te_ld); print(f'  Final:{sr["LabelSmoothing"]:.2f}%')

    # Co-teaching
    print('\n[5/5] Co-teaching...')
    m1=get_model(); m2=get_model()
    o1=mk_opt(m1); o2=mk_opt(m2)
    s1=mk_sch(o1,NE); s2=mk_sch(o2,NE)
    for ep in range(NE):
        train_cot(m1,m2,tr_ld,o1,o2,ep,NE); s1.step(); s2.step()
        if (ep+1)%10==0: print(f'  Ep{ep+1} Acc:{acc(m1,te_ld):.2f}%')
    sr['CoTeaching']=acc(m1,te_ld); print(f'  Final:{sr["CoTeaching"]:.2f}%')

    for mn in methods: results[mn].append(sr[mn])
    print(f'\nSeed {seed+1}:')
    for mn in methods: print(f'  {mn:<15}: {sr[mn]:.2f}%')
    with open('ewe_checkpoint.json','w') as f:
        json.dump({'seed':seed+1,'results':results,'rates':ewe_rates},f,indent=2)
    print('  Checkpoint saved.')

# ── Final results ─────────────────────────────────────────────
print('\n'+'='*60)
print('FINAL RESULTS — CIFAR-10N (~40% real human noise)')
print('='*60)
summary={}
for mn in methods:
    mean=float(np.mean(results[mn])); std=float(np.std(results[mn]))
    summary[mn]={'mean':mean,'std':std}
    mk=' <-- YOUR METHOD' if mn=='EWE' else ''
    print(f'{mn:<15}: {mean:.2f} +/- {std:.2f}%{mk}')

ewe_m=summary['EWE']['mean']; std_m=summary['Standard']['mean']; gce_m=summary['GCE']['mean']
print(f'\nEWE vs Standard: {ewe_m-std_m:+.2f}%')
print(f'EWE vs GCE     : {ewe_m-gce_m:+.2f}%')
print(f'EWE accept rate: {float(np.mean(ewe_rates)):.1%}')
print(f'EWE suppress   : {1-float(np.mean(ewe_rates)):.1%} to passive memory')

with open('ewe_results.json','w') as f:
    json.dump({'results':results,'summary':summary,'rates':ewe_rates},f,indent=2)
print('Saved: ewe_results.json')

# ── Figure ────────────────────────────────────────────────────
fig,axes=plt.subplots(1,2,figsize=(14,5)); fig.patch.set_facecolor('white')
ml=['Standard','EWE\n(ours)','GCE','Label\nSmooth','Co-teach']
means=[summary[m]['mean'] for m in methods]
stds =[summary[m]['std']  for m in methods]
cols =['#888888','#1E3A5F','#1565A0','#1B6B3A','#7A5C10']
ax=axes[0]
bars=ax.bar(ml,means,yerr=stds,color=cols,capsize=5,width=0.6)
bars[1].set_edgecolor('#FF6B00'); bars[1].set_linewidth(2.5)
ax.set_ylabel('Test Accuracy (%)'); ax.grid(axis='y',alpha=0.3)
ax.set_title('CIFAR-10N Method Comparison\n(Real Human Noise ~40%)',fontweight='bold')
ax.set_ylim(min(means)-3,max(means)+3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
for bar,m,s in zip(bars,means,stds):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+s+0.3,
            f'{m:.1f}%',ha='center',fontsize=9,fontweight='bold')
ax2=axes[1]
for i,(mn,c) in enumerate(zip(methods,cols)):
    v=results[mn]; ax2.scatter([i]*len(v),v,color=c,s=80,zorder=3)
    ax2.plot([i-.2,i+.2],[np.mean(v)]*2,color=c,lw=2.5,zorder=4)
ax2.set_xticks(range(len(methods))); ax2.set_xticklabels(ml,fontsize=9)
ax2.set_ylabel('Test Accuracy (%)')
ax2.set_title('Per-Seed Results\n(each dot = one seed)',fontweight='bold')
ax2.grid(axis='y',alpha=0.3)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
plt.suptitle('Epistemic Weight Engine — CIFAR-10N Results',fontsize=13,fontweight='bold',y=1.02)
plt.tight_layout()
plt.savefig('ewe_cifar10n_results.png',dpi=200,bbox_inches='tight',facecolor='white')
print('Saved: ewe_cifar10n_results.png')

# ── Paper table ───────────────────────────────────────────────
print('\n'+'='*60+'\nTABLE FOR PAPER\n'+'='*60)
lmap=['Standard Training','EWE (ours)','GCE','Label Smoothing','Co-teaching']
print(f'{"Method":<20}{"Acc(%)":>9}{"+-Std":>7}')
print('-'*38)
for mn,lb in zip(methods,lmap):
    mk=' <--' if mn=='EWE' else ''
    print(f'{lb:<20}{summary[mn]["mean"]:>9.2f}{summary[mn]["std"]:>7.2f}{mk}')
print('-'*38)
print(f'CIFAR-10N | ~40% noise | ResNet-18 | {NS} seeds | {NE} epochs')
print(f'EWE: {float(np.mean(ewe_rates))*100:.1f}% accepted | {(1-float(np.mean(ewe_rates)))*100:.1f}% suppressed')