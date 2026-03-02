# Current Task: CLIP Encoder Optimisation — Two-Stage Parametric Sweep

**Status:** ACTIVE
**Prerequisite reading:** `@CLAUDE.md`, `@context/upstream_diagnostic_analysis.md`
**Depends on:** Upstream Diagnostics (DONE — encoder identified as bottleneck), Notebook cleanup (DONE — 114 cells after diagnostics were added)

---

## Context

The upstream diagnostics (Phase D) identified the CLIP Brain Encoder as the gating bottleneck:
- **Diagnostic A**: Seen-only 1654-way accuracy = 1.90% (weak encoder)
- **Diagnostic B2**: WGAN-GP internal CV = 68.38% (generator is fine)
- **Diagnostic B1**: Synth→Real transfer = 1.73% (generator fails because encoder produces diffuse embeddings)
- **Diagnostic C**: Prototypes well-separated (not the bottleneck)

The current encoder is critically undertrained (20 epochs = ~1040 gradient steps) with a flat learning rate, fixed temperature, and shallow architecture. We need a systematic sweep to find the optimal training recipe and architecture.

**Primary metric**: Seen-only 1654-way LogReg top-1 accuracy on CLIP embeddings.
**Current baseline**: 1.90%.

---

## What to Implement

Create `helper_files/add_clip_optimisation.py` that appends **12 new cells** (cells 114–125) to the notebook.

---

### Cell 1 (Cell 114): Markdown Header

```markdown
---

# CLIP Encoder Optimisation — Two-Stage Parametric Sweep

**Goal:** Systematically find the optimal CLIP encoder configuration to maximise embedding quality.

Phase D diagnostics showed the encoder is the gating bottleneck (1.90% seen-only accuracy).
Root causes: undertrained (20 epochs), no LR schedule, fixed τ=0.07, shallow architecture.

**Stage 1** (dim=64 held constant): Sweep training dynamics — epochs, LR schedule, temperature.
**Stage 2** (best Stage 1 recipe): Sweep architecture — embed_dim, LayerNorm, depth, batch size.
**Final**: Re-run full pipeline (CLIP → WGAN-GP → Synthesis → GZSL eval) with best config.
```

---

### Cell 2 (Cell 115): Experiment Infrastructure

This is the most important cell. It defines all reusable infrastructure for the sweep.

```python
# =============================================================================
# CLIP ENCODER OPTIMISATION — EXPERIMENT INFRASTRUCTURE
# =============================================================================
# Unified experiment runner for the two-stage parametric sweep.
# Each experiment: build models → train → evaluate → log results.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.preprocessing import StandardScaler
import math

SEED = 42

# ---- Device ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ---- Load & prepare data (ONCE, reused across all experiments) ----
# These variables should already exist from earlier cells, but reload to be safe
from sklearn.model_selection import train_test_split

# Raw data (from earlier cells — brain_seen, text_seen, label_seen should be in scope)
X_seen = brain_seen.numpy() if isinstance(brain_seen, torch.Tensor) else brain_seen
y_seen = label_seen.numpy().flatten() if isinstance(label_seen, torch.Tensor) else label_seen.flatten()
T_seen = text_seen.numpy() if isinstance(text_seen, torch.Tensor) else text_seen

X_unseen_raw = brain_unseen.numpy() if isinstance(brain_unseen, torch.Tensor) else brain_unseen
T_unseen_raw = text_unseen.numpy() if isinstance(text_unseen, torch.Tensor) else text_unseen
y_unseen_raw = label_unseen.numpy().flatten() if isinstance(label_unseen, torch.Tensor) else label_unseen.flatten()

# 80/20 stratified split (same as original)
np.random.seed(SEED)
X_train, X_test, y_train, y_test, T_train, T_test = train_test_split(
    X_seen, y_seen, T_seen, test_size=0.2, random_state=SEED, stratify=y_seen
)

# StandardScaler on brain features (fit on train only)
raw_scaler = StandardScaler()
X_train_scaled = raw_scaler.fit_transform(X_train)
X_test_scaled = raw_scaler.transform(X_test)
X_unseen_scaled = raw_scaler.transform(X_unseen_raw)

print(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}, Unseen: {X_unseen_scaled.shape}")
print(f"Seen classes: {len(np.unique(y_train))}, Unseen classes: {len(np.unique(y_unseen_raw))}")

# ---- Architecture Variants ----

class BrainEncoderBaseline(nn.Module):
    """Original: 561 → 1024 → 512 → embed_dim (ReLU + Dropout, L2-norm)."""
    def __init__(self, input_dim=561, embed_dim=64, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.normalize(x, p=2, dim=-1)


class BrainEncoderLayerNorm(nn.Module):
    """Baseline + LayerNorm after each hidden layer."""
    def __init__(self, input_dim=561, embed_dim=64, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.ln1 = nn.LayerNorm(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.ln2 = nn.LayerNorm(512)
        self.fc3 = nn.Linear(512, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.normalize(x, p=2, dim=-1)


class BrainEncoderDeep(nn.Module):
    """4-layer: 561 → 1024 → 1024 → 512 → embed_dim + LayerNorm."""
    def __init__(self, input_dim=561, embed_dim=64, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.ln1 = nn.LayerNorm(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.ln2 = nn.LayerNorm(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.ln3 = nn.LayerNorm(512)
        self.fc4 = nn.Linear(512, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return F.normalize(x, p=2, dim=-1)


class BrainEncoderResidual(nn.Module):
    """Residual: 561 → 1024 → 1024 (+ skip) → 512 → embed_dim + LayerNorm."""
    def __init__(self, input_dim=561, embed_dim=64, dropout=0.1):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, 1024)
        self.ln_in = nn.LayerNorm(1024)
        self.fc_res = nn.Linear(1024, 1024)
        self.ln_res = nn.LayerNorm(1024)
        self.fc_down = nn.Linear(1024, 512)
        self.ln_down = nn.LayerNorm(512)
        self.fc_out = nn.Linear(512, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.ln_in(self.fc_in(x)))
        x = self.dropout(x)
        residual = x
        x = F.relu(self.ln_res(self.fc_res(x)))
        x = self.dropout(x)
        x = x + residual  # Skip connection
        x = F.relu(self.ln_down(self.fc_down(x)))
        x = self.dropout(x)
        x = self.fc_out(x)
        return F.normalize(x, p=2, dim=-1)


class TextProjectorVar(nn.Module):
    """Text projector with variable output dim: 512 → 512 → embed_dim."""
    def __init__(self, input_dim=512, hidden_dim=512, embed_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=-1)


ARCH_REGISTRY = {
    'baseline': BrainEncoderBaseline,
    'layernorm': BrainEncoderLayerNorm,
    'deep': BrainEncoderDeep,
    'residual': BrainEncoderResidual,
}


# ---- Contrastive Loss ----

def clip_loss_fixed(brain_embeds, text_embeds, tau):
    """Symmetric InfoNCE with fixed temperature."""
    logits = torch.matmul(brain_embeds, text_embeds.T) / tau
    targets = torch.arange(brain_embeds.size(0), device=brain_embeds.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2


def clip_loss_learnable(brain_embeds, text_embeds, log_tau):
    """Symmetric InfoNCE with learnable log-temperature."""
    tau = torch.clamp(torch.exp(-log_tau), min=0.01, max=1.0)
    logits = torch.matmul(brain_embeds, text_embeds.T) / tau
    targets = torch.arange(brain_embeds.size(0), device=brain_embeds.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2


# ---- LR Schedule Helpers ----

def get_cosine_warmup_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warmup for warmup_steps, then cosine annealing to 0."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---- Evaluation ----

def evaluate_encoder(brain_encoder, text_projector, device):
    """
    Evaluate encoder quality via seen-only LogReg classification.
    Returns dict with top-1, top-5, top-10 accuracies.
    """
    brain_encoder.eval()
    text_projector.eval()

    with torch.no_grad():
        E_tr = brain_encoder(torch.FloatTensor(X_train_scaled).to(device)).cpu().numpy()
        E_te = brain_encoder(torch.FloatTensor(X_test_scaled).to(device)).cpu().numpy()

    # Scale embeddings
    emb_scaler = StandardScaler()
    E_tr_s = emb_scaler.fit_transform(E_tr)
    E_te_s = emb_scaler.transform(E_te)

    n_classes = len(np.unique(y_train))
    random_baseline = 1.0 / n_classes

    clf = LogisticRegression(max_iter=2000, solver='lbfgs', C=1.0, random_state=SEED, n_jobs=-1)
    clf.fit(E_tr_s, y_train)

    y_pred = clf.predict(E_te_s)
    top1 = accuracy_score(y_test, y_pred)

    y_proba = clf.predict_proba(E_te_s)
    top5 = top_k_accuracy_score(y_test, y_proba, k=5, labels=clf.classes_)
    top10 = top_k_accuracy_score(y_test, y_proba, k=10, labels=clf.classes_)

    return {
        'top1': top1,
        'top5': top5,
        'top10': top10,
        'random_baseline': random_baseline,
        'signal_ratio': top1 / random_baseline,
    }


# ---- Main Experiment Runner ----

def run_clip_experiment(config, verbose=True):
    """
    Run a single CLIP encoder training experiment.

    Config keys:
        name (str): experiment identifier
        epochs (int): training epochs
        lr (float): peak learning rate
        schedule (str): 'flat' or 'cosine_warmup'
        tau (float): temperature value (used if tau_mode='fixed')
        tau_mode (str): 'fixed' or 'learnable'
        embed_dim (int): embedding dimension
        arch (str): one of 'baseline', 'layernorm', 'deep', 'residual'
        batch_size (int): batch size
        dropout (float): dropout rate (default 0.1)
        weight_decay (float): AdamW weight decay (default 1e-4)

    Returns:
        dict with config, metrics, loss_curve, training_time
    """
    name = config['name']
    epochs = config['epochs']
    lr = config['lr']
    schedule = config.get('schedule', 'flat')
    tau_val = config.get('tau', 0.07)
    tau_mode = config.get('tau_mode', 'fixed')
    embed_dim = config.get('embed_dim', 64)
    arch = config.get('arch', 'baseline')
    batch_size = config.get('batch_size', 256)
    dropout = config.get('dropout', 0.1)
    weight_decay = config.get('weight_decay', 1e-4)

    if verbose:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {name}")
        print(f"{'='*60}")
        print(f"  arch={arch}, embed_dim={embed_dim}, epochs={epochs}")
        print(f"  lr={lr}, schedule={schedule}, tau={tau_val} ({tau_mode})")
        print(f"  batch_size={batch_size}, dropout={dropout}")

    # Seed everything
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Build models
    EncoderClass = ARCH_REGISTRY[arch]
    brain_enc = EncoderClass(input_dim=561, embed_dim=embed_dim, dropout=dropout).to(device)
    text_proj = TextProjectorVar(input_dim=512, hidden_dim=512, embed_dim=embed_dim).to(device)

    n_params = sum(p.numel() for p in brain_enc.parameters()) + sum(p.numel() for p in text_proj.parameters())
    if verbose:
        print(f"  Total parameters: {n_params:,}")

    # Learnable temperature
    log_tau = None
    if tau_mode == 'learnable':
        log_tau = nn.Parameter(torch.tensor(math.log(1.0 / tau_val), device=device))

    # Optimizer
    params = list(brain_enc.parameters()) + list(text_proj.parameters())
    if log_tau is not None:
        params.append(log_tau)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # DataLoader
    train_ds = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(T_train),
        torch.LongTensor(y_train)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    total_steps = epochs * len(train_loader)

    # LR scheduler
    scheduler = None
    if schedule == 'cosine_warmup':
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_cosine_warmup_scheduler(optimizer, warmup_steps, total_steps)

    # Training loop
    loss_curve = []
    start_time = time.time()

    for epoch in range(epochs):
        brain_enc.train()
        text_proj.train()
        epoch_loss = 0.0

        for brain_batch, text_batch, _ in train_loader:
            brain_batch = brain_batch.to(device)
            text_batch = text_batch.to(device)

            brain_embeds = brain_enc(brain_batch)
            text_embeds = text_proj(text_batch)

            if tau_mode == 'learnable':
                loss = clip_loss_learnable(brain_embeds, text_embeds, log_tau)
            else:
                loss = clip_loss_fixed(brain_embeds, text_embeds, tau_val)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_curve.append(avg_loss)

        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            lr_now = optimizer.param_groups[0]['lr']
            tau_now = torch.clamp(torch.exp(-log_tau), 0.01, 1.0).item() if log_tau is not None else tau_val
            print(f"  Epoch {epoch+1:4d}/{epochs}: loss={avg_loss:.4f}, lr={lr_now:.6f}, tau={tau_now:.4f}")

    elapsed = time.time() - start_time

    # Evaluate
    metrics = evaluate_encoder(brain_enc, text_proj, device)
    final_tau = torch.clamp(torch.exp(-log_tau), 0.01, 1.0).item() if log_tau is not None else tau_val

    result = {
        'name': name,
        'config': config,
        'top1': metrics['top1'],
        'top5': metrics['top5'],
        'top10': metrics['top10'],
        'signal_ratio': metrics['signal_ratio'],
        'final_loss': loss_curve[-1],
        'final_tau': final_tau,
        'loss_curve': loss_curve,
        'training_time': elapsed,
        'n_params': n_params,
        'brain_encoder': brain_enc,
        'text_projector': text_proj,
    }

    if verbose:
        print(f"\n  RESULTS: top-1={metrics['top1']*100:.2f}%, top-5={metrics['top5']*100:.2f}%, "
              f"top-10={metrics['top10']*100:.2f}%, signal={metrics['signal_ratio']:.1f}x")
        print(f"  Time: {elapsed:.1f}s, Final loss: {loss_curve[-1]:.4f}, Final tau: {final_tau:.4f}")

    return result


# Global results accumulator
experiment_results = []

print("\nExperiment infrastructure ready.")
print(f"Baseline reference: top-1 = 1.90%, signal = 31.5x (20 epochs, flat lr, fixed tau=0.07, dim=64)")
```

---

### Cell 3 (Cell 116): Stage 1 Round 1 — Epoch Scaling

```python
# =============================================================================
# STAGE 1 ROUND 1: EPOCH SCALING (dim=64 held constant)
# =============================================================================
# Most impactful variable. Tests whether encoder was simply undertrained.
# Baseline: 20 epochs → 1.90% top-1
# =============================================================================

round1_configs = [
    {'name': 'R1-A: 50ep',  'epochs': 50,  'lr': 1e-4, 'schedule': 'flat', 'tau': 0.07, 'tau_mode': 'fixed', 'embed_dim': 64, 'arch': 'baseline', 'batch_size': 256},
    {'name': 'R1-B: 100ep', 'epochs': 100, 'lr': 1e-4, 'schedule': 'flat', 'tau': 0.07, 'tau_mode': 'fixed', 'embed_dim': 64, 'arch': 'baseline', 'batch_size': 256},
    {'name': 'R1-C: 200ep', 'epochs': 200, 'lr': 1e-4, 'schedule': 'flat', 'tau': 0.07, 'tau_mode': 'fixed', 'embed_dim': 64, 'arch': 'baseline', 'batch_size': 256},
]

print("STAGE 1 ROUND 1: Epoch Scaling")
print("=" * 60)

for cfg in round1_configs:
    result = run_clip_experiment(cfg)
    experiment_results.append(result)

# Select best epoch count
r1_results = [r for r in experiment_results if r['name'].startswith('R1')]
best_r1 = max(r1_results, key=lambda r: r['top1'])
BEST_EPOCHS = best_r1['config']['epochs']

print(f"\n{'='*60}")
print(f"ROUND 1 WINNER: {best_r1['name']} — top-1 = {best_r1['top1']*100:.2f}%")
print(f"Selected E* = {BEST_EPOCHS} epochs")

# Check for diminishing returns
for r in r1_results:
    print(f"  {r['name']}: top-1={r['top1']*100:.2f}%, top-5={r['top5']*100:.2f}%, loss={r['final_loss']:.4f}, time={r['training_time']:.0f}s")
```

---

### Cell 4 (Cell 117): Stage 1 Round 2 — LR Schedule + Value

```python
# =============================================================================
# STAGE 1 ROUND 2: LR SCHEDULE + LR VALUE
# =============================================================================
# Using E* from Round 1, test cosine annealing with warmup at 3 learning rates.
# =============================================================================

round2_configs = [
    {'name': f'R2-D: cos_warmup lr=1e-4',  'epochs': BEST_EPOCHS, 'lr': 1e-4, 'schedule': 'cosine_warmup', 'tau': 0.07, 'tau_mode': 'fixed', 'embed_dim': 64, 'arch': 'baseline', 'batch_size': 256},
    {'name': f'R2-E: cos_warmup lr=3e-4',  'epochs': BEST_EPOCHS, 'lr': 3e-4, 'schedule': 'cosine_warmup', 'tau': 0.07, 'tau_mode': 'fixed', 'embed_dim': 64, 'arch': 'baseline', 'batch_size': 256},
    {'name': f'R2-F: cos_warmup lr=1e-3',  'epochs': BEST_EPOCHS, 'lr': 1e-3, 'schedule': 'cosine_warmup', 'tau': 0.07, 'tau_mode': 'fixed', 'embed_dim': 64, 'arch': 'baseline', 'batch_size': 256},
]

print(f"STAGE 1 ROUND 2: LR Schedule + Value (using E*={BEST_EPOCHS} epochs)")
print("=" * 60)

for cfg in round2_configs:
    result = run_clip_experiment(cfg)
    experiment_results.append(result)

# Compare Round 2 against Round 1 winner
r2_results = [r for r in experiment_results if r['name'].startswith('R2')]
all_r1r2 = [best_r1] + r2_results
best_r2 = max(all_r1r2, key=lambda r: r['top1'])
BEST_LR = best_r2['config']['lr']
BEST_SCHEDULE = best_r2['config']['schedule']

print(f"\n{'='*60}")
print(f"ROUND 2 WINNER: {best_r2['name']} — top-1 = {best_r2['top1']*100:.2f}%")
print(f"Selected S* = (lr={BEST_LR}, schedule={BEST_SCHEDULE})")

for r in all_r1r2:
    print(f"  {r['name']}: top-1={r['top1']*100:.2f}%, top-5={r['top5']*100:.2f}%, loss={r['final_loss']:.4f}")
```

---

### Cell 5 (Cell 118): Stage 1 Round 3 — Temperature

```python
# =============================================================================
# STAGE 1 ROUND 3: TEMPERATURE
# =============================================================================
# Using E* + S*, test learnable temperature and a softer fixed value.
# =============================================================================

round3_configs = [
    {'name': f'R3-G: learnable_tau',  'epochs': BEST_EPOCHS, 'lr': BEST_LR, 'schedule': BEST_SCHEDULE, 'tau': 0.07, 'tau_mode': 'learnable', 'embed_dim': 64, 'arch': 'baseline', 'batch_size': 256},
    {'name': f'R3-H: fixed_tau=0.15', 'epochs': BEST_EPOCHS, 'lr': BEST_LR, 'schedule': BEST_SCHEDULE, 'tau': 0.15, 'tau_mode': 'fixed',     'embed_dim': 64, 'arch': 'baseline', 'batch_size': 256},
]

print(f"STAGE 1 ROUND 3: Temperature (using E*={BEST_EPOCHS}, S*=(lr={BEST_LR}, sched={BEST_SCHEDULE}))")
print("=" * 60)

for cfg in round3_configs:
    result = run_clip_experiment(cfg)
    experiment_results.append(result)

# Compare Round 3 against Round 2 winner
r3_results = [r for r in experiment_results if r['name'].startswith('R3')]
all_r1r2r3 = [best_r2] + r3_results
best_r3 = max(all_r1r2r3, key=lambda r: r['top1'])
BEST_TAU = best_r3['config']['tau']
BEST_TAU_MODE = best_r3['config']['tau_mode']

print(f"\n{'='*60}")
print(f"ROUND 3 WINNER: {best_r3['name']} — top-1 = {best_r3['top1']*100:.2f}%")
print(f"Selected T* = (tau={BEST_TAU}, mode={BEST_TAU_MODE})")
if BEST_TAU_MODE == 'learnable':
    print(f"  Final learned tau = {best_r3['final_tau']:.4f}")

for r in all_r1r2r3:
    print(f"  {r['name']}: top-1={r['top1']*100:.2f}%, final_tau={r['final_tau']:.4f}")
```

---

### Cell 6 (Cell 119): Stage 1 Results Summary

```python
# =============================================================================
# STAGE 1 RESULTS SUMMARY
# =============================================================================

# Collect all Stage 1 results
stage1_results = [r for r in experiment_results if r['name'].startswith('R1') or r['name'].startswith('R2') or r['name'].startswith('R3')]

# Summary table
print("=" * 80)
print("STAGE 1 RESULTS — TRAINING DYNAMICS SWEEP (dim=64)")
print("=" * 80)
print(f"{'Name':<30} {'Top-1':>7} {'Top-5':>7} {'Top-10':>7} {'Signal':>8} {'Loss':>7} {'Time':>6} {'Tau':>6}")
print("-" * 80)
for r in stage1_results:
    print(f"{r['name']:<30} {r['top1']*100:>6.2f}% {r['top5']*100:>6.2f}% {r['top10']*100:>6.2f}% {r['signal_ratio']:>7.1f}x {r['final_loss']:>7.4f} {r['training_time']:>5.0f}s {r['final_tau']:>6.4f}")
print("-" * 80)
print(f"Baseline (20ep, flat, tau=0.07): top-1 = 1.90%")

# Best Stage 1 config
BEST_STAGE1 = {
    'epochs': BEST_EPOCHS,
    'lr': BEST_LR,
    'schedule': BEST_SCHEDULE,
    'tau': BEST_TAU,
    'tau_mode': BEST_TAU_MODE,
}
print(f"\nBEST STAGE 1 CONFIG: {BEST_STAGE1}")
print(f"Best top-1: {best_r3['top1']*100:.2f}% (improvement: {(best_r3['top1'] - 0.019) / 0.019 * 100:.0f}% over baseline)")

# Loss curves
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
for r in stage1_results:
    ax.plot(r['loss_curve'], label=f"{r['name']} ({r['top1']*100:.1f}%)", alpha=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('Contrastive Loss')
ax.set_title('Stage 1: Training Loss Curves')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/clip_opt_stage1_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figures/clip_opt_stage1_comparison.png")
```

---

### Cell 7 (Cell 120): Stage 2 Round 4 — embed_dim

```python
# =============================================================================
# STAGE 2 ROUND 4: EMBEDDING DIMENSION
# =============================================================================
# Using BEST_STAGE1 recipe, test larger embedding dimensions.
# =============================================================================

round4_configs = [
    {'name': 'R4-I: dim=128', 'epochs': BEST_EPOCHS, 'lr': BEST_LR, 'schedule': BEST_SCHEDULE, 'tau': BEST_TAU, 'tau_mode': BEST_TAU_MODE, 'embed_dim': 128, 'arch': 'baseline', 'batch_size': 256},
    {'name': 'R4-J: dim=256', 'epochs': BEST_EPOCHS, 'lr': BEST_LR, 'schedule': BEST_SCHEDULE, 'tau': BEST_TAU, 'tau_mode': BEST_TAU_MODE, 'embed_dim': 256, 'arch': 'baseline', 'batch_size': 256},
]

print(f"STAGE 2 ROUND 4: Embedding Dimension (using BEST_STAGE1)")
print("=" * 60)

for cfg in round4_configs:
    result = run_clip_experiment(cfg)
    experiment_results.append(result)

# Compare with Stage 1 best (dim=64)
r4_results = [r for r in experiment_results if r['name'].startswith('R4')]
all_dim = [best_r3] + r4_results  # best_r3 is Stage 1 best at dim=64
best_r4 = max(all_dim, key=lambda r: r['top1'])
BEST_DIM = best_r4['config']['embed_dim']

print(f"\n{'='*60}")
print(f"ROUND 4 WINNER: {best_r4['name']} — top-1 = {best_r4['top1']*100:.2f}%")
print(f"Selected D* = {BEST_DIM}")

for r in all_dim:
    dim = r['config']['embed_dim']
    print(f"  dim={dim}: top-1={r['top1']*100:.2f}%, top-5={r['top5']*100:.2f}%, params={r['n_params']:,}")
```

---

### Cell 8 (Cell 121): Stage 2 Round 5 — Architecture

```python
# =============================================================================
# STAGE 2 ROUND 5: ARCHITECTURE VARIANTS
# =============================================================================
# Using BEST_STAGE1 + D*, test enhanced architectures.
# =============================================================================

round5_configs = [
    {'name': 'R5-K: layernorm',  'epochs': BEST_EPOCHS, 'lr': BEST_LR, 'schedule': BEST_SCHEDULE, 'tau': BEST_TAU, 'tau_mode': BEST_TAU_MODE, 'embed_dim': BEST_DIM, 'arch': 'layernorm', 'batch_size': 256},
    {'name': 'R5-L: deep',       'epochs': BEST_EPOCHS, 'lr': BEST_LR, 'schedule': BEST_SCHEDULE, 'tau': BEST_TAU, 'tau_mode': BEST_TAU_MODE, 'embed_dim': BEST_DIM, 'arch': 'deep',      'batch_size': 256},
    {'name': 'R5-M: residual',   'epochs': BEST_EPOCHS, 'lr': BEST_LR, 'schedule': BEST_SCHEDULE, 'tau': BEST_TAU, 'tau_mode': BEST_TAU_MODE, 'embed_dim': BEST_DIM, 'arch': 'residual',  'batch_size': 256},
]

print(f"STAGE 2 ROUND 5: Architecture (using BEST_STAGE1 + D*={BEST_DIM})")
print("=" * 60)

for cfg in round5_configs:
    result = run_clip_experiment(cfg)
    experiment_results.append(result)

# Compare with Round 4 winner (baseline arch at D*)
r5_results = [r for r in experiment_results if r['name'].startswith('R5')]
all_arch = [best_r4] + r5_results
best_r5 = max(all_arch, key=lambda r: r['top1'])
BEST_ARCH = best_r5['config']['arch']

print(f"\n{'='*60}")
print(f"ROUND 5 WINNER: {best_r5['name']} — top-1 = {best_r5['top1']*100:.2f}%")
print(f"Selected A* = {BEST_ARCH}")

for r in all_arch:
    print(f"  {r['name']}: top-1={r['top1']*100:.2f}%, params={r['n_params']:,}")
```

---

### Cell 9 (Cell 122): Stage 2 Round 6 — Batch Size

```python
# =============================================================================
# STAGE 2 ROUND 6: BATCH SIZE
# =============================================================================
# Larger batch = more negatives per InfoNCE step.
# Using BEST_STAGE1 + D* + A*.
# =============================================================================

round6_configs = [
    {'name': 'R6-N: batch=512', 'epochs': BEST_EPOCHS, 'lr': BEST_LR, 'schedule': BEST_SCHEDULE, 'tau': BEST_TAU, 'tau_mode': BEST_TAU_MODE, 'embed_dim': BEST_DIM, 'arch': BEST_ARCH, 'batch_size': 512},
]

print(f"STAGE 2 ROUND 6: Batch Size (using BEST_STAGE1 + D*={BEST_DIM} + A*={BEST_ARCH})")
print("=" * 60)

for cfg in round6_configs:
    result = run_clip_experiment(cfg)
    experiment_results.append(result)

# Compare with Round 5 winner (batch=256)
r6_results = [r for r in experiment_results if r['name'].startswith('R6')]
all_bs = [best_r5] + r6_results
best_r6 = max(all_bs, key=lambda r: r['top1'])
BEST_BS = best_r6['config']['batch_size']

print(f"\n{'='*60}")
print(f"ROUND 6 WINNER: {best_r6['name']} — top-1 = {best_r6['top1']*100:.2f}%")
print(f"Selected BS* = {BEST_BS}")
```

---

### Cell 10 (Cell 123): Stage 2 Results Summary

```python
# =============================================================================
# STAGE 2 RESULTS SUMMARY + BEST OVERALL CONFIG
# =============================================================================

stage2_results = [r for r in experiment_results if r['name'].startswith('R4') or r['name'].startswith('R5') or r['name'].startswith('R6')]

print("=" * 80)
print("STAGE 2 RESULTS — ARCHITECTURE + DIMENSIONALITY SWEEP")
print("=" * 80)
print(f"{'Name':<30} {'Top-1':>7} {'Top-5':>7} {'Top-10':>7} {'Signal':>8} {'Params':>10} {'Time':>6}")
print("-" * 80)
# Include Stage 1 best as reference
ref = best_r3
print(f"{'[Stage1 best, dim=64]':<30} {ref['top1']*100:>6.2f}% {ref['top5']*100:>6.2f}% {ref['top10']*100:>6.2f}% {ref['signal_ratio']:>7.1f}x {ref['n_params']:>10,} {ref['training_time']:>5.0f}s")
print("-" * 80)
for r in stage2_results:
    print(f"{r['name']:<30} {r['top1']*100:>6.2f}% {r['top5']*100:>6.2f}% {r['top10']*100:>6.2f}% {r['signal_ratio']:>7.1f}x {r['n_params']:>10,} {r['training_time']:>5.0f}s")

# Best overall
BEST_OVERALL = {
    'epochs': BEST_EPOCHS,
    'lr': BEST_LR,
    'schedule': BEST_SCHEDULE,
    'tau': BEST_TAU,
    'tau_mode': BEST_TAU_MODE,
    'embed_dim': BEST_DIM,
    'arch': BEST_ARCH,
    'batch_size': BEST_BS,
    'dropout': 0.1,
    'weight_decay': 1e-4,
}

best_overall_result = best_r6  # The final round winner

print(f"\n{'='*80}")
print(f"BEST OVERALL CONFIG:")
for k, v in BEST_OVERALL.items():
    print(f"  {k}: {v}")
print(f"\nBest top-1: {best_overall_result['top1']*100:.2f}%")
print(f"Improvement over baseline (1.90%): {(best_overall_result['top1'] - 0.019) / 0.019 * 100:.0f}%")

# Save Stage 2 figure
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# All experiments bar chart
all_results = experiment_results
names = [r['name'] for r in all_results]
top1s = [r['top1'] * 100 for r in all_results]
colors = ['steelblue' if r['name'].startswith('R1') or r['name'].startswith('R2') or r['name'].startswith('R3')
          else 'coral' for r in all_results]

axes[0].barh(range(len(names)), top1s, color=colors, alpha=0.8)
axes[0].set_yticks(range(len(names)))
axes[0].set_yticklabels(names, fontsize=7)
axes[0].set_xlabel('Top-1 Accuracy (%)')
axes[0].set_title('All Experiments: Top-1 Accuracy')
axes[0].axvline(x=1.90, color='red', linestyle='--', alpha=0.5, label='Baseline (1.90%)')
axes[0].legend()

# Loss curves of top 5
top5_results = sorted(all_results, key=lambda r: r['top1'], reverse=True)[:5]
for r in top5_results:
    axes[1].plot(r['loss_curve'], label=f"{r['name']} ({r['top1']*100:.1f}%)", alpha=0.8)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Contrastive Loss')
axes[1].set_title('Top 5 Configs: Loss Curves')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/clip_opt_stage2_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figures/clip_opt_stage2_comparison.png")
```

---

### Cell 11 (Cell 124): Full Pipeline Re-run with Best Config

```python
# =============================================================================
# FULL PIPELINE RE-RUN WITH BEST CONFIG
# =============================================================================
# 1. Retrain CLIP with BEST_OVERALL
# 2. Extract embeddings & prototypes
# 3. Retrain WGAN-GP
# 4. Generate synthetics
# 5. GZSL evaluation with balanced counts (Phase 1 fix)
# =============================================================================

print("=" * 70)
print("FULL PIPELINE RE-RUN WITH OPTIMISED CLIP ENCODER")
print("=" * 70)
print(f"Config: {BEST_OVERALL}")
print()

# ---- Step 1: Retrain CLIP ----
print("Step 1: Retraining CLIP encoder...")
BEST_OVERALL['name'] = 'FINAL_BEST'
final_result = run_clip_experiment(BEST_OVERALL)
best_brain_encoder = final_result['brain_encoder']
best_text_projector = final_result['text_projector']
print(f"  Encoder top-1: {final_result['top1']*100:.2f}%")

# ---- Step 2: Extract embeddings ----
print("\nStep 2: Extracting embeddings...")
best_brain_encoder.eval()
best_text_projector.eval()

with torch.no_grad():
    E_train_seen_new = best_brain_encoder(torch.FloatTensor(X_train_scaled).to(device)).cpu().numpy()
    E_test_seen_new = best_brain_encoder(torch.FloatTensor(X_test_scaled).to(device)).cpu().numpy()
    E_unseen_new = best_brain_encoder(torch.FloatTensor(X_unseen_scaled).to(device)).cpu().numpy()
    T_train_embeds = best_text_projector(torch.FloatTensor(T_train).to(device)).cpu().numpy()
    T_test_embeds = best_text_projector(torch.FloatTensor(T_test).to(device)).cpu().numpy()
    T_unseen_embeds = best_text_projector(torch.FloatTensor(T_unseen_raw).to(device)).cpu().numpy()

print(f"  E_train_seen: {E_train_seen_new.shape}")
print(f"  E_test_seen:  {E_test_seen_new.shape}")
print(f"  E_unseen:     {E_unseen_new.shape}")

# ---- Step 3: Compute prototypes ----
print("\nStep 3: Computing prototypes...")

def compute_prototypes(embeddings, labels):
    unique_labels = np.unique(labels)
    prototypes = {}
    for c in unique_labels:
        mask = labels == c
        proto = embeddings[mask].mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        prototypes[c] = proto
    return prototypes

S_seen_proto_new = compute_prototypes(T_train_embeds, y_train)
S_unseen_proto_new = compute_prototypes(T_unseen_embeds, y_unseen_raw)

seen_classes = sorted(S_seen_proto_new.keys())
unseen_classes = sorted(S_unseen_proto_new.keys())
S_seen_array_new = np.array([S_seen_proto_new[c] for c in seen_classes])
S_unseen_array_new = np.array([S_unseen_proto_new[c] for c in unseen_classes])

print(f"  Seen prototypes: {S_seen_array_new.shape}")
print(f"  Unseen prototypes: {S_unseen_array_new.shape}")

# ---- Step 4: Save to cached_arrays ----
print("\nStep 4: Saving cached arrays...")
import os
os.makedirs('cached_arrays', exist_ok=True)
np.save('cached_arrays/E_train_seen.npy', E_train_seen_new)
np.save('cached_arrays/E_test_seen.npy', E_test_seen_new)
np.save('cached_arrays/E_unseen.npy', E_unseen_new)
np.save('cached_arrays/y_train_seen.npy', y_train)
np.save('cached_arrays/y_test_seen.npy', y_test)
np.save('cached_arrays/y_unseen.npy', y_unseen_raw)
np.save('cached_arrays/S_seen_prototypes.npy', S_seen_array_new)
np.save('cached_arrays/S_unseen_prototypes.npy', S_unseen_array_new)
np.save('cached_arrays/seen_classes.npy', np.array(seen_classes))
np.save('cached_arrays/unseen_classes.npy', np.array(unseen_classes))
print("  Cached arrays overwritten.")

# ---- Step 5: Retrain WGAN-GP ----
print("\nStep 5: Retraining WGAN-GP...")

WGAN_CONFIG = {
    'z_dim': 100,
    'embed_dim': BEST_DIM,
    'lr': 1e-4,
    'betas': (0.0, 0.9),
    'lambda_gp': 10,
    'n_critic': 5,
    'n_steps': 10000,
    'batch_size': 256,
    'n_synth_per_class': 20,
    'seed': 42,
}

# Generator and Critic (adapted for variable embed_dim)
class GeneratorOpt(nn.Module):
    def __init__(self, z_dim=100, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + embed_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, embed_dim),
        )
    def forward(self, z, s_c):
        x = torch.cat([z, s_c], dim=-1)
        return F.normalize(self.net(x), p=2, dim=-1)

class CriticOpt(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * embed_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )
    def forward(self, e, s_c):
        return self.net(torch.cat([e, s_c], dim=-1))

def compute_gp(critic, real, fake, s_c, device):
    eps = torch.rand(real.size(0), 1, device=device)
    e_bar = eps * real + (1 - eps) * fake
    e_bar.requires_grad_(True)
    d = critic(e_bar, s_c)
    grads = torch.autograd.grad(d, e_bar, torch.ones_like(d), create_graph=True, retain_graph=True)[0]
    return ((grads.view(grads.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()

np.random.seed(WGAN_CONFIG['seed'])
torch.manual_seed(WGAN_CONFIG['seed'])

gen = GeneratorOpt(WGAN_CONFIG['z_dim'], WGAN_CONFIG['embed_dim']).to(device)
crit = CriticOpt(WGAN_CONFIG['embed_dim']).to(device)

g_opt = torch.optim.Adam(gen.parameters(), lr=WGAN_CONFIG['lr'], betas=WGAN_CONFIG['betas'])
c_opt = torch.optim.Adam(crit.parameters(), lr=WGAN_CONFIG['lr'], betas=WGAN_CONFIG['betas'])

# Prepare WGAN data
def get_proto_for_labels(labels, proto_dict):
    return np.array([proto_dict[int(l)] for l in labels])

E_train_tensor = torch.FloatTensor(E_train_seen_new)
S_train_conds = torch.FloatTensor(get_proto_for_labels(y_train, S_seen_proto_new))
wgan_ds = TensorDataset(E_train_tensor, S_train_conds)
wgan_loader = DataLoader(wgan_ds, batch_size=WGAN_CONFIG['batch_size'], shuffle=True, drop_last=True)

data_iter = iter(wgan_loader)
for g_step in range(1, WGAN_CONFIG['n_steps'] + 1):
    # Critic
    for _ in range(WGAN_CONFIG['n_critic']):
        try:
            e_real, s_c = next(data_iter)
        except StopIteration:
            data_iter = iter(wgan_loader)
            e_real, s_c = next(data_iter)
        e_real, s_c = e_real.to(device), s_c.to(device)
        z = torch.randn(e_real.size(0), WGAN_CONFIG['z_dim'], device=device)
        e_fake = gen(z, s_c)
        gp = compute_gp(crit, e_real, e_fake.detach(), s_c, device)
        c_loss = -crit(e_real, s_c).mean() + crit(e_fake.detach(), s_c).mean() + WGAN_CONFIG['lambda_gp'] * gp
        c_opt.zero_grad()
        c_loss.backward()
        c_opt.step()
    # Generator
    try:
        _, s_c = next(data_iter)
    except StopIteration:
        data_iter = iter(wgan_loader)
        _, s_c = next(data_iter)
    s_c = s_c.to(device)
    z = torch.randn(s_c.size(0), WGAN_CONFIG['z_dim'], device=device)
    g_loss = -crit(gen(z, s_c), s_c).mean()
    g_opt.zero_grad()
    g_loss.backward()
    g_opt.step()
    if g_step % 2000 == 0:
        print(f"  WGAN step {g_step}/10000: G={g_loss.item():.4f}, C={c_loss.item():.4f}")

print("  WGAN-GP training complete.")

# ---- Step 6: Generate synthetics ----
print("\nStep 6: Generating synthetic unseen embeddings...")
gen.eval()
synth_embs, synth_labs = [], []
with torch.no_grad():
    for c in unseen_classes:
        s_c = torch.FloatTensor(S_unseen_proto_new[c]).unsqueeze(0).repeat(WGAN_CONFIG['n_synth_per_class'], 1).to(device)
        z = torch.randn(WGAN_CONFIG['n_synth_per_class'], WGAN_CONFIG['z_dim'], device=device)
        synth_embs.append(gen(z, s_c).cpu().numpy())
        synth_labs.extend([c] * WGAN_CONFIG['n_synth_per_class'])

E_synth_new = np.vstack(synth_embs)
y_synth_new = np.array(synth_labs)
np.save('cached_arrays/E_synth_unseen.npy', E_synth_new)
np.save('cached_arrays/y_synth_unseen.npy', y_synth_new)
print(f"  E_synth_unseen: {E_synth_new.shape}")

# ---- Step 7: GZSL evaluation with balanced counts ----
print("\nStep 7: GZSL evaluation (balanced counts)...")

# Remap unseen labels
LABEL_OFFSET = int(np.max(y_train))
y_unseen_remap = y_unseen_raw + LABEL_OFFSET
y_synth_remap = y_synth_new + LABEL_OFFSET

# Downsample unseen to match seen per-class count
seen_classes_arr, seen_counts = np.unique(y_train, return_counts=True)
target_per_class = int(np.median(seen_counts))

ds_idx = []
for c in np.unique(y_synth_remap):
    c_idx = np.where(y_synth_remap == c)[0]
    if len(c_idx) > target_per_class:
        c_idx = np.random.choice(c_idx, target_per_class, replace=False)
    ds_idx.extend(c_idx)
E_synth_ds = E_synth_new[ds_idx]
y_synth_ds = y_synth_remap[ds_idx]

# Train GZSL classifier
X_gzsl = np.vstack([E_train_seen_new, E_synth_ds])
y_gzsl = np.concatenate([y_train, y_synth_ds])

from sklearn.linear_model import LogisticRegression
clf_gzsl = LogisticRegression(max_iter=2000, solver='lbfgs', C=1.0, random_state=SEED, n_jobs=-1)
clf_gzsl.fit(X_gzsl, y_gzsl)

# Evaluate
seen_labels_set = set(np.unique(y_train))
unseen_labels_set = set(np.unique(y_unseen_remap))

y_pred_seen = clf_gzsl.predict(E_test_seen_new)
y_pred_unseen = clf_gzsl.predict(E_unseen_new)

acc_s = accuracy_score(y_test, y_pred_seen)
acc_u = accuracy_score(y_unseen_remap, y_pred_unseen)
h_mean = 2 * acc_s * acc_u / (acc_s + acc_u) if (acc_s + acc_u) > 0 else 0

# Routing rate
routed_to_unseen = sum(1 for p in y_pred_seen if p in unseen_labels_set)
routing_rate = routed_to_unseen / len(y_pred_seen)

# Diagnostic B re-run (synth→real transfer)
emb_sc = StandardScaler()
E_synth_s = emb_sc.fit_transform(E_synth_new)
E_unseen_s = emb_sc.transform(E_unseen_new)
clf_transfer = LogisticRegression(max_iter=2000, solver='lbfgs', C=1.0, random_state=SEED, n_jobs=-1)
clf_transfer.fit(E_synth_s, y_synth_new)
transfer_acc = accuracy_score(y_unseen_raw, clf_transfer.predict(E_unseen_s))

print(f"\n{'='*60}")
print(f"OPTIMISED PIPELINE RESULTS")
print(f"{'='*60}")
print(f"  Encoder top-1 (seen-only): {final_result['top1']*100:.2f}%")
print(f"  AccS (GZSL):               {acc_s*100:.2f}%")
print(f"  AccU (GZSL):               {acc_u*100:.2f}%")
print(f"  H-mean:                    {h_mean*100:.2f}%")
print(f"  Routing rate:              {routing_rate*100:.2f}%")
print(f"  Synth→Real transfer:       {transfer_acc*100:.2f}%")
```

---

### Cell 12 (Cell 125): Final Comparison

```python
# =============================================================================
# FINAL COMPARISON: OPTIMISED vs BASELINE
# =============================================================================

print("=" * 80)
print("FINAL COMPARISON")
print("=" * 80)
print()
print(f"{'Metric':<40} {'Baseline':>12} {'Phase1 Bal':>12} {'Optimised':>12}")
print("-" * 80)
print(f"{'Encoder top-1 (seen-only)':<40} {'1.90%':>12} {'1.90%':>12} {final_result['top1']*100:>11.2f}%")
print(f"{'AccS (GZSL)':<40} {'0.06%':>12} {'2.90%':>12} {acc_s*100:>11.2f}%")
print(f"{'AccU (GZSL)':<40} {'2.13%':>12} {'0.22%':>12} {acc_u*100:>11.2f}%")
print(f"{'H-mean':<40} {'0.12%':>12} {'0.42%':>12} {h_mean*100:>11.2f}%")
print(f"{'Routing rate':<40} {'99.70%':>12} {'9.16%':>12} {routing_rate*100:>11.2f}%")
print(f"{'Synth→Real transfer (200-way)':<40} {'1.73%':>12} {'1.73%':>12} {transfer_acc*100:>11.2f}%")
print("-" * 80)

# Summary figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Encoder quality comparison
metrics_names = ['Encoder\ntop-1', 'AccS', 'AccU', 'H-mean']
baseline_vals = [1.90, 0.06, 2.13, 0.12]
phase1_vals = [1.90, 2.90, 0.22, 0.42]
opt_vals = [final_result['top1']*100, acc_s*100, acc_u*100, h_mean*100]

x = np.arange(len(metrics_names))
w = 0.25
axes[0].bar(x - w, baseline_vals, w, label='Baseline', alpha=0.8, color='lightcoral')
axes[0].bar(x, phase1_vals, w, label='Phase 1 Balanced', alpha=0.8, color='steelblue')
axes[0].bar(x + w, opt_vals, w, label='Optimised', alpha=0.8, color='forestgreen')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics_names)
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title('GZSL Pipeline: Before vs After Optimisation')
axes[0].legend()

# Right: Loss curve of final best
axes[1].plot(final_result['loss_curve'], color='forestgreen', alpha=0.8)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Contrastive Loss')
axes[1].set_title(f'Final Best Config Loss Curve')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/clip_opt_final_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figures/clip_opt_final_comparison.png")

print("\n" + "=" * 80)
print("CLIP ENCODER OPTIMISATION COMPLETE")
print(f"Best config: {BEST_OVERALL}")
print("=" * 80)
```

---

## Implementation Notes

### Variable Dependencies

Cell 115 (infrastructure) re-loads and re-splits data from the raw variables that exist in notebook scope from earlier cells (`brain_seen`, `text_seen`, `label_seen`, `brain_unseen`, `text_unseen`, `label_unseen`). All subsequent cells depend on cell 115.

The sweep cells (116–122) depend on variables set by previous rounds (`BEST_EPOCHS`, `BEST_LR`, etc.). They MUST be run sequentially.

Cell 124 (pipeline re-run) redefines Generator and Critic classes locally (as `GeneratorOpt`, `CriticOpt`) to avoid name conflicts with earlier notebook cells and to support variable `embed_dim`.

### Helper Script

Create `helper_files/add_clip_optimisation.py` following the same pattern as existing helpers. It should:
1. Read `COMP2261_ArizMLCW_with_baseline.ipynb`
2. Append the 12 cells above
3. Write back
4. Print confirmation with cell count

### Figures

Save to `figures/` at 150 dpi:
- `clip_opt_stage1_comparison.png`
- `clip_opt_stage2_comparison.png`
- `clip_opt_final_comparison.png`

---

## What to Report When Done

1. Confirm helper script created and run successfully
2. New cell count (should be **126** = 114 + 12)
3. List cell indices and first lines of the 12 new cells
4. **Do NOT run the cells** — they execute in Colab. Just confirm injection.

## What NOT to Touch

- All existing cells 0–113
- Cached `.npy` files (overwritten at runtime)
- The `figures/` directory
- The evaluation harness or Phase 1 cells
