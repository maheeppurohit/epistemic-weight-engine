"""
Language Model Experiment — EWE Rerun with Tuned Gate
Author: Maheep Purohit, Independent Researcher, Bikaner, India

Reruns only EWE with looser gate k=0.05
Previous results loaded from checkpoint.

Fix: k=0.05 instead of 0.2276
     Accept rate ~80% instead of ~50%
     Better for small datasets with 2 classes
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_scheduler
)
import numpy as np
import json
import os
import tarfile
from datetime import datetime

print('EWE LM Rerun — Tuned Gate')
device = torch.device('cuda' if torch.cuda.is_available()
                      else 'cpu')
print(f'Device: {device}')

# ── Config ────────────────────────────────────────────────────
SEEDS       = [42, 123, 456]
EPOCHS      = 5        # more epochs for stability
BATCH_SIZE  = 16
LR          = 2e-5
MAX_LEN     = 128
BIAS_RATE   = 0.30
TRAIN_SIZE  = 3000
TEST_SIZE   = 1000
CHECKPOINT  = 'lm_tuned_checkpoint.json'

# Tuned gate for binary LM task
K     = 0.05   # much looser than 0.2276
ALPHA = 0.45
BETA  = 0.40
GAMMA = 0.15
TAU   = 0.5
LAM   = 0.20   # less aggressive reality alignment
EPS   = 0.1

print(f'EWE gate: k={K}, lam={LAM}, epochs={EPOCHS}')

# Previous known results from checkpoint
KNOWN = {
    'standard-clean_42':   85.10,
    'standard-clean_123':  84.50,
    'standard-clean_456':  85.30,
    'standard-biased_42':  79.80,
    'standard-biased_123': 80.40,
    'standard-biased_456': 81.10,
}


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
                self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(
                self.attention_mask[idx], dtype=torch.long),
            'label':          torch.tensor(
                self.labels[idx], dtype=torch.long),
        }


def inject_approval_bias(labels, bias_rate, seed):
    np.random.seed(seed)
    labels   = list(labels)
    neg_idx  = [i for i,l in enumerate(labels) if l==0]
    n_flip   = int(len(neg_idx) * bias_rate)
    flip_idx = np.random.choice(
        neg_idx, n_flip, replace=False)
    for i in flip_idx:
        labels[i] = 1
    print(f'Bias: {n_flip} flipped '
          f'({n_flip/len(labels):.1%})')
    return labels


# ── Tuned EWE Gate ────────────────────────────────────────────
class TunedEWEGate:
    def __init__(self):
        self.k         = K
        self._loss_ema = None
        self._total    = 0
        self._accepted = 0

    @property
    def acceptance_rate(self):
        return self._accepted / self._total \
               if self._total > 0 else 0.0

    def gate(self, losses, logits):
        I     = torch.clamp(
            losses / (TAU + losses), 0, 1)
        probs = F.softmax(logits, 1)
        A     = probs.max(1).values
        sim_x = 1.0 - losses / (losses.max() + 1e-8)
        R     = torch.clamp(
            sim_x - LAM * A, min=0)
        mean_loss = losses.mean().item()
        if self._loss_ema is None:
            self._loss_ema = mean_loss
        else:
            self._loss_ema = (
                0.99 * self._loss_ema +
                0.01 * mean_loss)
        P = torch.clamp(
            (losses - self._loss_ema) /
            (self._loss_ema + EPS), 0, 1)
        W     = ALPHA*I + BETA*R + GAMMA*P
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
                scheduler, gate):
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
        eg = gate.gate(losses.detach(), out.detach())
        if eg.sum() > 0:
            losses[eg].mean().backward()
        else:
            continue
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()


# ── Run One Seed ──────────────────────────────────────────────
def run_ewe_seed(train_enc, train_labels,
                 test_enc, test_labels, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_set    = IMDbDataset(train_enc, train_labels)
    test_set     = IMDbDataset(test_enc,  test_labels)
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
    gate = TunedEWEGate()

    for ep in range(1, EPOCHS+1):
        train_epoch(model, train_loader,
                    optimizer, scheduler, gate)
        acc  = evaluate(model, test_loader)
        rate = gate.acceptance_rate
        print(f'  Ep {ep}/{EPOCHS} | '
              f'Acc: {acc:.2f}% | '
              f'Accept: {rate:.1%}')

    return evaluate(model, test_loader)


# ── Checkpoint ────────────────────────────────────────────────
def load_ckpt():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            return json.load(f)
    return dict(KNOWN)

def save_ckpt(d):
    with open(CHECKPOINT, 'w') as f:
        json.dump(d, f, indent=2)


# ── Main ──────────────────────────────────────────────────────
def main():
    print('\nEWE LM RERUN — TUNED GATE')
    print('='*60)

    checkpoint = load_ckpt()

    # Load data
    print('Extracting IMDb...')
    if not os.path.exists('aclImdb'):
        with tarfile.open(
                'aclImdb_v1.tar.gz', 'r:gz') as t:
            t.extractall('.')

    def read_split(split, size):
        texts = []; labels = []
        for label, sent in [(1,'pos'),(0,'neg')]:
            folder = f'aclImdb/{split}/{sent}'
            files  = sorted(
                os.listdir(folder))[:size//2]
            for fname in files:
                with open(
                        os.path.join(folder, fname),
                        'r', encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(label)
        idx = list(range(len(texts)))
        np.random.seed(42)
        np.random.shuffle(idx)
        return ([texts[i] for i in idx],
                [labels[i] for i in idx])

    train_texts, train_labels = read_split(
        'train', TRAIN_SIZE)
    test_texts,  test_labels  = read_split(
        'test',  TEST_SIZE)
    print(f'Train: {len(train_texts)} | '
          f'Test: {len(test_texts)}')

    tokenizer = DistilBertTokenizerFast\
        .from_pretrained('distilbert-base-uncased')

    print('Tokenizing test set...')
    test_enc = tokenizer(
        test_texts, max_length=MAX_LEN,
        padding='max_length', truncation=True)

    ewe_results = []

    for seed in SEEDS:
        print(f'\n{"="*60}')
        print(f'Seed {seed}')
        print('='*60)

        key = f'ewe-tuned_{seed}'
        if key in checkpoint:
            acc = checkpoint[key]
            print(f'EWE tuned: {acc:.2f}% [RESUMED]')
            ewe_results.append(acc)
            continue

        biased_labels = inject_approval_bias(
            train_labels, BIAS_RATE, seed)

        print('Tokenizing train set...')
        train_enc = tokenizer(
            train_texts, max_length=MAX_LEN,
            padding='max_length', truncation=True)

        print(f'Running EWE (tuned k={K})...')
        acc = run_ewe_seed(
            train_enc, biased_labels,
            test_enc, list(test_labels), seed)

        checkpoint[key] = acc
        save_ckpt(checkpoint)
        ewe_results.append(acc)
        print(f'EWE tuned Seed {seed}: {acc:.2f}% ✅')

    # Final summary
    print('\n' + '='*60)
    print('COMPLETE APPROVAL BIAS RESULTS')
    print('='*60)

    clean  = [checkpoint[f'standard-clean_{s}']
              for s in SEEDS]
    biased = [checkpoint[f'standard-biased_{s}']
              for s in SEEDS]

    cm = np.mean(clean);  cs = np.std(clean)
    bm = np.mean(biased); bs = np.std(biased)
    em = np.mean(ewe_results)
    es = np.std(ewe_results)

    drop = cm - bm
    rec  = em - bm
    pct  = rec/drop*100 if drop > 0 else 0

    print(f'\nStandard (clean):    '
          f'{cm:.2f}% ± {cs:.2f}%  ← upper bound')
    print(f'Standard (biased):   '
          f'{bm:.2f}% ± {bs:.2f}%  '
          f'← bias drops by -{drop:.2f}%')
    print(f'EWE tuned (biased):  '
          f'{em:.2f}% ± {es:.2f}%  '
          f'← recovers +{rec:.2f}%')
    print(f'\nEWE recovers {pct:.1f}% '
          f'of approval bias damage')

    output = {
        'experiment':  'Approval Bias LM Tuned',
        'model':       'DistilBERT',
        'dataset':     'IMDb',
        'bias_rate':   BIAS_RATE,
        'ewe_k':       K,
        'ewe_lam':     LAM,
        'epochs':      EPOCHS,
        'timestamp':   datetime.now().isoformat(),
        'results': {
            'standard_clean': {
                'mean': float(cm), 'std': float(cs),
                'per_seed': clean
            },
            'standard_biased': {
                'mean': float(bm), 'std': float(bs),
                'per_seed': biased
            },
            'ewe_tuned': {
                'mean': float(em), 'std': float(es),
                'per_seed': ewe_results
            }
        },
        'key_findings': {
            'bias_damage':  float(drop),
            'ewe_recovery': float(rec),
            'recovery_pct': float(pct)
        }
    }

    with open('lm_tuned_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print('Saved to lm_tuned_results.json')


if __name__ == '__main__':
    main()
