"""
Language Model Experiment — Approval Bias Validation
Author: Maheep Purohit, Independent Researcher, Bikaner, India
Paper: https://doi.org/10.5281/zenodo.18940011

Fixed for transformers 5.8.0
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from datetime import datetime

print('Starting Language Model Experiment...')
print(f'PyTorch: {torch.__version__}')

import transformers
print(f'Transformers: {transformers.__version__}')

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_scheduler
)
from torch.optim import AdamW
import tarfile

# ── Device ────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available()
                      else 'cpu')
print(f'Device: {device}')

# ── Config ────────────────────────────────────────────────────
SEEDS       = [42, 123, 456]
EPOCHS      = 3
BATCH_SIZE  = 16
LR          = 2e-5
MAX_LEN     = 128
BIAS_RATE   = 0.30
BASE_K      = 0.25
NUM_CLASSES = 2
CHECKPOINT  = 'lm_experiment_checkpoint.json'
TRAIN_SIZE  = 3000
TEST_SIZE   = 1000

def adaptive_k(num_classes):
    return BASE_K / np.log(num_classes + 1)

K = adaptive_k(NUM_CLASSES)
print(f'Adaptive k: {K:.4f}')


# ── Dataset ───────────────────────────────────────────────────
class IMDbDataset(Dataset):
    def __init__(self, encodings, labels):
        self.input_ids      = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']
        self.labels         = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids':      torch.tensor(
                self.input_ids[idx],
                dtype=torch.long),
            'attention_mask': torch.tensor(
                self.attention_mask[idx],
                dtype=torch.long),
            'label':          torch.tensor(
                self.labels[idx],
                dtype=torch.long),
            'idx':            idx
        }


def inject_approval_bias(labels, bias_rate, seed):
    np.random.seed(seed)
    labels  = list(labels)
    neg_idx = [i for i,l in enumerate(labels) if l==0]
    n_flip  = int(len(neg_idx) * bias_rate)
    flip_idx = np.random.choice(
        neg_idx, n_flip, replace=False)
    for i in flip_idx:
        labels[i] = 1
    print(f'Bias injected: {n_flip} flipped '
          f'({n_flip/len(labels):.1%})')
    return labels


# ── Adaptive EWE Gate ─────────────────────────────────────────
class AdaptiveEWEGate:
    def __init__(self):
        self.k         = K
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


# ── Evaluation ────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            y    = batch['label'].to(device)
            out  = model(
                input_ids=ids,
                attention_mask=mask).logits
            pred = out.argmax(1)
            correct += pred.eq(y).sum().item()
            total   += y.size(0)
    return 100.0 * correct / total


# ── Training ──────────────────────────────────────────────────
def train_epoch(model, loader, optimizer,
                scheduler, method, gate=None):
    model.train()
    for batch in loader:
        ids  = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        y    = batch['label'].to(device)
        optimizer.zero_grad()
        out    = model(
            input_ids=ids,
            attention_mask=mask).logits
        losses = F.cross_entropy(
            out, y, reduction='none')
        if method == 'standard':
            losses.mean().backward()
        elif method == 'ewe':
            eg = gate.gate(
                losses.detach(), out.detach())
            if eg.sum() > 0:
                losses[eg].mean().backward()
            else:
                continue
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()


# ── Run One Seed ──────────────────────────────────────────────
def run_seed(method, train_enc, train_labels,
             test_enc, test_labels, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_set = IMDbDataset(train_enc, train_labels)
    test_set  = IMDbDataset(test_enc,  test_labels)
    train_loader = DataLoader(
        train_set, BATCH_SIZE,
        shuffle=True, num_workers=0)
    test_loader  = DataLoader(
        test_set, BATCH_SIZE,
        shuffle=False, num_workers=0)

    model = DistilBertForSequenceClassification\
        .from_pretrained(
            'distilbert-base-uncased',
            num_labels=2).to(device)

    optimizer   = AdamW(
        model.parameters(), lr=LR,
        weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_scheduler(
        'linear', optimizer=optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps)

    gate = AdaptiveEWEGate() \
           if method == 'ewe' else None

    for ep in range(1, EPOCHS+1):
        train_epoch(model, train_loader,
                    optimizer, scheduler,
                    method, gate)
        acc  = evaluate(model, test_loader)
        rate = gate.acceptance_rate \
               if gate else 1.0
        print(f'  Ep {ep}/{EPOCHS} | '
              f'Acc: {acc:.2f}% | '
              f'Accept: {rate:.1%}')

    return evaluate(model, test_loader)


# ── Checkpoint ────────────────────────────────────────────────
def load_ckpt():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            d = json.load(f)
            print(f'Resuming — {len(d)} seeds done')
            return d
    return {}

def save_ckpt(d):
    with open(CHECKPOINT, 'w') as f:
        json.dump(d, f, indent=2)


# ── Main ──────────────────────────────────────────────────────
def main():
    print('\nLANGUAGE MODEL APPROVAL BIAS EXPERIMENT')
    print(f'Model: DistilBERT')
    print(f'Dataset: IMDb')
    print(f'Bias rate: {BIAS_RATE:.0%}')
    print(f'Seeds: {SEEDS} | Epochs: {EPOCHS}')
    print('='*60)

    checkpoint = load_ckpt()

    print('Loading tokenizer...')
    tokenizer = DistilBertTokenizerFast\
        .from_pretrained('distilbert-base-uncased')
    print('Tokenizer loaded')

    # Load IMDb from local tar.gz file
    print('Extracting IMDb...')
    if not os.path.exists('aclImdb'):
        with tarfile.open(
                'aclImdb_v1.tar.gz', 'r:gz') as t:
            t.extractall('.')
    print('IMDb extracted')

    def read_split(split, size):
        texts = []; labels = []
        for label, sentiment in [(1,'pos'),(0,'neg')]:
            folder = f'aclImdb/{split}/{sentiment}'
            files  = sorted(
                os.listdir(folder))[:size//2]
            for fname in files:
                fpath = os.path.join(folder, fname)
                with open(fpath, 'r',
                          encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(label)
        idx = list(range(len(texts)))
        np.random.seed(42)
        np.random.shuffle(idx)
        return ([texts[i] for i in idx],
                [labels[i] for i in idx])

    print('Loading splits...')
    train_texts, train_labels = read_split(
        'train', TRAIN_SIZE)
    test_texts,  test_labels  = read_split(
        'test',  TEST_SIZE)
    print(f'Train: {len(train_texts)} | '
          f'Test: {len(test_texts)}')

    print('Tokenizing test set...')
    test_enc = tokenizer(
        test_texts,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True)
    print('Test set tokenized')

    all_results = {
        'standard-clean':  [],
        'standard-biased': [],
        'ewe-biased':      [],
    }

    for seed in SEEDS:
        print(f'\n{"="*60}')
        print(f'Seed {seed}')
        print('='*60)

        biased_labels = inject_approval_bias(
            train_labels, BIAS_RATE, seed)

        print('Tokenizing train set...')
        train_enc = tokenizer(
            train_texts,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True)
        print('Train set tokenized')

        # Method 1 — Standard clean
        key = f'standard-clean_{seed}'
        if key in checkpoint:
            acc = checkpoint[key]
            print(f'Standard (clean): '
                  f'{acc:.2f}% [RESUMED]')
        else:
            print('\nMethod: Standard (clean)...')
            acc = run_seed(
                'standard',
                train_enc, train_labels,
                test_enc, test_labels, seed)
            checkpoint[key] = acc
            save_ckpt(checkpoint)
            print(f'Standard (clean): {acc:.2f}% ✅')
        all_results['standard-clean'].append(acc)

        # Method 2 — Standard biased
        key = f'standard-biased_{seed}'
        if key in checkpoint:
            acc = checkpoint[key]
            print(f'Standard (biased): '
                  f'{acc:.2f}% [RESUMED]')
        else:
            print('\nMethod: Standard (biased)...')
            acc = run_seed(
                'standard',
                train_enc, biased_labels,
                test_enc, test_labels, seed)
            checkpoint[key] = acc
            save_ckpt(checkpoint)
            print(f'Standard (biased): {acc:.2f}% ✅')
        all_results['standard-biased'].append(acc)

        # Method 3 — EWE biased
        key = f'ewe-biased_{seed}'
        if key in checkpoint:
            acc = checkpoint[key]
            print(f'EWE (biased): '
                  f'{acc:.2f}% [RESUMED]')
        else:
            print('\nMethod: EWE (biased)...')
            acc = run_seed(
                'ewe',
                train_enc, biased_labels,
                test_enc, test_labels, seed)
            checkpoint[key] = acc
            save_ckpt(checkpoint)
            print(f'EWE (biased): {acc:.2f}% ✅')
        all_results['ewe-biased'].append(acc)

    # Summary
    print('\n' + '='*60)
    print('APPROVAL BIAS — FINAL RESULTS')
    print('='*60)

    cm = np.mean(all_results['standard-clean'])
    bm = np.mean(all_results['standard-biased'])
    em = np.mean(all_results['ewe-biased'])
    cs = np.std(all_results['standard-clean'])
    bs = np.std(all_results['standard-biased'])
    es = np.std(all_results['ewe-biased'])

    drop = cm - bm
    rec  = em - bm
    pct  = rec/drop*100 if drop > 0 else 0

    print(f'Standard (clean):  {cm:.2f}% ± {cs:.2f}%')
    print(f'Standard (biased): {bm:.2f}% ± {bs:.2f}%'
          f'  (drop: -{drop:.2f}%)')
    print(f'EWE (biased):      {em:.2f}% ± {es:.2f}%'
          f'  (recovery: +{rec:.2f}%)')
    print(f'EWE recovers {pct:.1f}% of approval bias damage')

    output = {
        'experiment':  'Approval Bias LM',
        'model':       'DistilBERT',
        'dataset':     'IMDb',
        'bias_rate':   BIAS_RATE,
        'timestamp':   datetime.now().isoformat(),
        'results': {
            'standard_clean': {
                'mean': float(cm), 'std': float(cs),
                'per_seed': all_results['standard-clean']
            },
            'standard_biased': {
                'mean': float(bm), 'std': float(bs),
                'per_seed': all_results['standard-biased']
            },
            'ewe_biased': {
                'mean': float(em), 'std': float(es),
                'per_seed': all_results['ewe-biased']
            }
        },
        'key_findings': {
            'bias_damage':  float(drop),
            'ewe_recovery': float(rec),
            'recovery_pct': float(pct)
        }
    }

    with open('lm_approval_bias_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print('Saved to lm_approval_bias_results.json')


if __name__ == '__main__':
    main()
