# Phase F: Regularisation, Supervised Contrastive Learning & Structured Architecture

**Status**: ACTIVE
**Implementer**: Antigravity Agent
**Deliverable**: `helper_files/add_phase_f_cells.py` → injects 11 cells (114–124) into the notebook
**Prerequisite reading**: `@CLAUDE.md`, `@context/clip_optimisation_analysis.md`

---

## Context

Phase E optimised CLIP hyperparameters and hit a ceiling at 2.87% top-1 (1654-way). The dominant signal is **overfitting/shortcut learning** — training loss → 0 while test accuracy degrades with longer training. The model memorises trial-specific noise patterns instead of learning class-invariant features.

Phase F attacks three root causes via sequential elimination:
- **F1**: Data augmentation (combat overfitting — the #1 signal)
- **F2**: Supervised contrastive loss (exploit 10 trials/class structure)
- **F3**: Structured encoder (exploit 17ch × 33t spatial-temporal structure)

## Notebook Context

- **Current notebook**: 114 cells (cells 0–113)
- **After injection**: 125 cells (cells 114–124 added)
- **Existing objects available** (from earlier cells):
  - `CLIP_CONFIG` (cell 29): embed_dim=128, tau=0.15, lr=1e-3, cosine_warmup, 50 epochs
  - `BrainEncoder`, `TextProjector`, `clip_loss` (cell 30): production models
  - `train_loader` (cell 31): DataLoader with shuffle
  - `X_train_tensor`, `T_train_tensor`, `Y_train_tensor` (cell 31): training tensors
  - `X_test_tensor`, `T_test_tensor`, `Y_test_tensor` (cell 31): test tensors
  - `X_unseen_tensor`, `T_unseen_tensor`, `Y_unseen_tensor` (cell 31): unseen tensors
  - `device` (cell 29): torch device
  - `Generator`, `Critic`, `gradient_penalty` (cell 40): WGAN-GP models
  - `WGAN_CONFIG` (cell 39): WGAN config with embed_dim=128
  - `label_train_seen`, `label_test_seen` (cell 31): numpy label arrays

---

## Cell 114 — Markdown Header

```markdown
---

# Phase F: Regularisation, Contrastive Learning & Structured Architecture

**Goal:** Improve CLIP encoder internal representation quality by attacking the three root causes of shortcut learning:

1. **F1 — Data Augmentation**: Inject noise and transformations to combat overfitting (dominant signal from Phase E)
2. **F2 — Supervised Contrastive Loss**: Exploit multi-trial class structure (10 trials/class) for stronger learning signal
3. **F3 — Structured Encoder**: Replace flat MLP with EEGNet to exploit spatial-temporal structure (17ch × 33t)

**Method**: Three-stage sequential elimination sweep. Each stage builds on the previous winner.

**Baseline**: Phase E best — 2.87% top-1 (MLP+LayerNorm, dim=128, lr=1e-3, cosine warmup, τ=0.15, 50 epochs)
```

---

## Cell 115 — Experiment Infrastructure

```python
# =============================================================================
# PHASE F: EXPERIMENT INFRASTRUCTURE
# =============================================================================
# Augmentation, SupCon loss, class-balanced sampling, EEGNet, ShallowConvNet,
# unified experiment runner, evaluation.

import math
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Sampler
from sklearn.linear_model import LogisticRegression
import numpy as np

# ---- Constants ----
N_CHANNELS = 17
N_TIMES = 33
N_SEEN_CLASSES = 1654

# =============================================================================
# 1. DATA AUGMENTATION
# =============================================================================

class EEGAugmentation:
    """
    On-the-fly EEG data augmentation operating on (17, 33) spatial-temporal structure.
    Accepts (B, 561) tensors, reshapes internally, applies augmentations, returns (B, 561).
    """
    def __init__(self, config):
        self.gaussian_std = config.get('gaussian_std', 0.0)
        self.time_shift = config.get('time_shift', 0)
        self.channel_drop_prob = config.get('channel_drop_prob', 0.0)
        self.time_mask_width = config.get('time_mask_width', 0)
        self.mixup_alpha = config.get('mixup_alpha', 0.0)

    def __call__(self, brain_batch, text_batch=None):
        """
        Apply augmentations to a batch of EEG data.
        Args:
            brain_batch: (B, 561) tensor
            text_batch: (B, D) tensor (needed for mixup to keep pairs aligned)
        Returns:
            augmented brain_batch (B, 561), optionally augmented text_batch
        """
        B = brain_batch.shape[0]
        device = brain_batch.device
        x = brain_batch.view(B, N_CHANNELS, N_TIMES)

        # Gaussian noise
        if self.gaussian_std > 0:
            noise = torch.randn_like(x) * self.gaussian_std
            x = x + noise

        # Time shift (circular)
        if self.time_shift > 0:
            shift = random.randint(-self.time_shift, self.time_shift)
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=-1)

        # Channel dropout
        if self.channel_drop_prob > 0:
            mask = (torch.rand(B, N_CHANNELS, 1, device=device) > self.channel_drop_prob).float()
            x = x * mask

        # Time masking
        if self.time_mask_width > 0:
            start = random.randint(0, N_TIMES - self.time_mask_width)
            x[:, :, start:start + self.time_mask_width] = 0

        brain_out = x.view(B, -1)

        # Mixup (applied after other augmentations, mixes with rolled batch)
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            lam = max(lam, 1 - lam)  # ensure lam >= 0.5 (original dominates)
            brain_out = lam * brain_out + (1 - lam) * brain_out.roll(1, 0)
            if text_batch is not None:
                text_batch = lam * text_batch + (1 - lam) * text_batch.roll(1, 0)

        if text_batch is not None:
            return brain_out, text_batch
        return brain_out


# =============================================================================
# 2. LOSS FUNCTIONS
# =============================================================================

def supcon_loss(features, labels, tau=0.15):
    """
    Supervised Contrastive Loss (Khosla et al., 2020).
    All same-class pairs are positives, all different-class pairs are negatives.
    Self-pairs are excluded.

    Args:
        features: (B, d) L2-normalised embeddings
        labels: (B,) integer class labels
        tau: temperature
    Returns:
        Scalar loss
    """
    device = features.device
    B = features.shape[0]

    # Cosine similarity / tau
    sim = torch.matmul(features, features.T) / tau  # (B, B)

    # Positive mask: same class, exclude self
    labels_col = labels.unsqueeze(1)  # (B, 1)
    pos_mask = (labels_col == labels_col.T).float()  # (B, B)
    pos_mask.fill_diagonal_(0)

    # Numerical stability
    logits_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - logits_max.detach()

    # Denominator: sum over all j != i
    exp_sim = torch.exp(sim)
    self_mask = 1.0 - torch.eye(B, device=device)
    denom = (exp_sim * self_mask).sum(dim=1, keepdim=True)  # (B, 1)

    # Log probability of each pair
    log_prob = sim - torch.log(denom + 1e-8)  # (B, B)

    # Mean log-prob over positives for each anchor
    n_pos = pos_mask.sum(dim=1)  # (B,)
    mean_log_prob = (pos_mask * log_prob).sum(dim=1) / (n_pos + 1e-8)

    # Average over anchors with at least one positive
    valid = n_pos > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return -mean_log_prob[valid].mean()


def combined_loss(brain_embeds, text_embeds, labels, tau=0.15, lambda_sup=0.5):
    """
    Combined loss: lambda * SupCon(EEG<->EEG) + (1-lambda) * InfoNCE(EEG<->Text).
    """
    l_sup = supcon_loss(brain_embeds, labels, tau)
    l_nce = clip_loss(brain_embeds, text_embeds, tau)
    return lambda_sup * l_sup + (1.0 - lambda_sup) * l_nce


# =============================================================================
# 3. CLASS-BALANCED BATCH SAMPLER
# =============================================================================

class ClassBalancedBatchSampler(Sampler):
    """
    Yields batches of K classes x M instances = K*M samples.
    Ensures every sample has M-1 same-class partners for SupCon.
    """
    def __init__(self, labels, K=64, M=4, n_batches=52):
        self.labels = np.array(labels)
        self.K = K
        self.M = M
        self.n_batches = n_batches

        self.class_to_indices = {}
        for idx, lab in enumerate(self.labels):
            self.class_to_indices.setdefault(int(lab), []).append(idx)
        self.classes = list(self.class_to_indices.keys())

    def __iter__(self):
        for _ in range(self.n_batches):
            selected = random.sample(self.classes, min(self.K, len(self.classes)))
            batch = []
            for cls in selected:
                indices = self.class_to_indices[cls]
                if len(indices) >= self.M:
                    batch.extend(random.sample(indices, self.M))
                else:
                    batch.extend(random.choices(indices, k=self.M))
            yield batch

    def __len__(self):
        return self.n_batches


# =============================================================================
# 4. ENCODER ARCHITECTURES
# =============================================================================

class EEGNetEncoder(nn.Module):
    """
    EEGNet (Lawhern et al., 2018) adapted for contrastive learning.
    Input: (B, 561) flattened EEG -> reshape to (B, 1, 17, 33) internally.
    Output: (B, embed_dim) L2-normalised.

    Args:
        F1: number of temporal filters
        D: depth multiplier for spatial filters
        F2: number of pointwise filters
        kern_t: temporal kernel size
        kern_s: separable temporal kernel size
        embed_dim: output embedding dimension
        dropout: dropout rate
    """
    def __init__(self, F1=8, D=2, F2=16, kern_t=9, kern_s=5,
                 embed_dim=128, dropout=0.25, **kwargs):
        super().__init__()

        # Block 1: Temporal convolution
        self.conv_temporal = nn.Conv2d(1, F1, (1, kern_t), padding=(0, kern_t // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Block 2: Depthwise spatial convolution
        self.conv_spatial = nn.Conv2d(F1, F1 * D, (N_CHANNELS, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 3: Separable convolution
        self.conv_sep_depth = nn.Conv2d(F1 * D, F1 * D, (1, kern_s),
                                        padding=(0, kern_s // 2), groups=F1 * D, bias=False)
        self.conv_sep_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 2))
        self.drop2 = nn.Dropout(dropout)

        # Compute flattened size
        t_after_pool1 = N_TIMES // 4  # 33 // 4 = 8
        t_after_pool2 = t_after_pool1 // 2  # 8 // 2 = 4
        flat_size = F2 * t_after_pool2

        # Head
        self.fc = nn.Linear(flat_size, embed_dim)

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, 1, N_CHANNELS, N_TIMES)  # (B, 1, 17, 33)

        # Block 1
        x = self.bn1(self.conv_temporal(x))  # (B, F1, 17, 33)

        # Block 2
        x = self.conv_spatial(x)  # (B, F1*D, 1, 33)
        x = F.elu(self.bn2(x))
        x = self.drop1(self.pool1(x))  # (B, F1*D, 1, 8)

        # Block 3
        x = self.conv_sep_depth(x)  # (B, F1*D, 1, 8)
        x = self.conv_sep_point(x)  # (B, F2, 1, 8)
        x = F.elu(self.bn3(x))
        x = self.drop2(self.pool2(x))  # (B, F2, 1, 4)

        # Head
        x = x.flatten(1)  # (B, F2*4)
        x = self.fc(x)  # (B, embed_dim)
        return F.normalize(x, p=2, dim=-1)


class ShallowConvEncoder(nn.Module):
    """
    ShallowConvNet (Schirrmeister et al., 2017) adapted for contrastive learning.
    Uses square-log activation designed for EEG band power extraction.
    Input: (B, 561) -> reshape to (B, 1, 17, 33).
    Output: (B, embed_dim) L2-normalised.
    """
    def __init__(self, n_filters=40, kern_t=13, embed_dim=128, dropout=0.5, **kwargs):
        super().__init__()

        self.conv_temporal = nn.Conv2d(1, n_filters, (1, kern_t),
                                       padding=(0, kern_t // 2), bias=False)
        self.conv_spatial = nn.Conv2d(n_filters, n_filters, (N_CHANNELS, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters)
        self.pool = nn.AvgPool2d((1, 6), stride=(1, 3))
        self.drop = nn.Dropout(dropout)

        # Compute flattened size: time stays 33, spatial reduces to 1, pool: (33-6)//3 + 1 = 10
        t_after_pool = (N_TIMES - 6) // 3 + 1  # 10
        flat_size = n_filters * t_after_pool

        self.fc = nn.Linear(flat_size, embed_dim)

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, 1, N_CHANNELS, N_TIMES)

        x = self.conv_temporal(x)  # (B, 40, 17, 33)
        x = self.conv_spatial(x)   # (B, 40, 1, 33)
        x = self.bn(x)

        # Square-log activation (designed for EEG band power)
        x = x ** 2
        x = self.pool(x)  # (B, 40, 1, 10)
        x = torch.log(torch.clamp(x, min=1e-6))

        x = self.drop(x)
        x = x.flatten(1)  # (B, 400)
        x = self.fc(x)    # (B, embed_dim)
        return F.normalize(x, p=2, dim=-1)


# Architecture registry
ENCODER_REGISTRY = {
    'mlp_ln': lambda embed_dim=128, dropout=0.1, **kw: BrainEncoder(
        input_dim=561, embed_dim=embed_dim, dropout=dropout),
    'eegnet_small': lambda embed_dim=128, dropout=0.25, **kw: EEGNetEncoder(
        F1=8, D=2, F2=16, kern_t=9, kern_s=5, embed_dim=embed_dim, dropout=dropout),
    'eegnet_large': lambda embed_dim=128, dropout=0.25, **kw: EEGNetEncoder(
        F1=16, D=2, F2=32, kern_t=9, kern_s=5, embed_dim=embed_dim, dropout=dropout),
    'shallow_conv': lambda embed_dim=128, dropout=0.5, **kw: ShallowConvEncoder(
        n_filters=40, kern_t=13, embed_dim=embed_dim, dropout=dropout),
}


# =============================================================================
# 5. EVALUATION
# =============================================================================

def evaluate_encoder(encoder, text_proj, X_train, T_train, Y_train,
                     X_test, T_test, Y_test, device):
    """
    Evaluate encoder quality via seen-only LogReg classification.
    Returns dict with top-1, top-5, top-10, signal ratio.
    """
    encoder.eval()
    text_proj.eval()
    with torch.no_grad():
        E_train = encoder(X_train.to(device)).cpu().numpy()
        E_test = encoder(X_test.to(device)).cpu().numpy()

    clf = LogisticRegression(max_iter=1000, solver='saga', n_jobs=-1, random_state=42)
    clf.fit(E_train, Y_train)

    proba = clf.predict_proba(E_test)
    n_classes = proba.shape[1]

    results = {}
    for k in [1, 5, 10]:
        top_k_preds = np.argsort(proba, axis=1)[:, -k:]
        correct = np.array([Y_test[i] in top_k_preds[i] for i in range(len(Y_test))])
        results[f'top_{k}'] = correct.mean()

    results['signal'] = results['top_1'] / (1.0 / n_classes)
    results['n_classes'] = n_classes
    return results


# =============================================================================
# 6. UNIFIED EXPERIMENT RUNNER
# =============================================================================

# Global results accumulator
phase_f_results = []

def run_phase_f_experiment(config):
    """
    Run a single Phase F experiment.

    Config keys:
        name (str): experiment name
        encoder (str): key in ENCODER_REGISTRY
        embed_dim (int): embedding dimension (default 128)
        epochs (int): training epochs (default 50)
        lr (float): peak learning rate (default 1e-3)
        tau (float): contrastive temperature (default 0.15)
        schedule (str): 'cosine_warmup' or 'flat'
        loss_type (str): 'infonce', 'supcon', 'combined'
        lambda_sup (float): SupCon weight for combined loss (default 0.5)
        batch_mode (str): 'shuffle' or 'class_balanced'
        K (int): classes per batch for class_balanced (default 64)
        M (int): instances per class for class_balanced (default 4)
        augmentation (dict or None): augmentation config for EEGAugmentation
        dropout (float): dropout rate
        weight_decay (float): AdamW weight decay
    """
    name = config['name']
    embed_dim = config.get('embed_dim', 128)
    epochs = config.get('epochs', 50)
    lr = config.get('lr', 1e-3)
    tau = config.get('tau', 0.15)
    schedule = config.get('schedule', 'cosine_warmup')
    loss_type = config.get('loss_type', 'infonce')
    lambda_sup = config.get('lambda_sup', 0.5)
    batch_mode = config.get('batch_mode', 'shuffle')
    K = config.get('K', 64)
    M = config.get('M', 4)
    aug_config = config.get('augmentation', None)
    dropout = config.get('dropout', 0.1)
    weight_decay = config.get('weight_decay', 1e-4)
    encoder_name = config.get('encoder', 'mlp_ln')

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*60}")
    print(f"  encoder={encoder_name}, embed_dim={embed_dim}, epochs={epochs}")
    print(f"  lr={lr}, schedule={schedule}, tau={tau}")
    print(f"  loss={loss_type}, batch={batch_mode}", end="")
    if batch_mode == 'class_balanced':
        print(f" (K={K}, M={M})", end="")
    print(f"\n  augmentation={aug_config}")

    # Seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Build encoder + text projector
    enc = ENCODER_REGISTRY[encoder_name](embed_dim=embed_dim, dropout=dropout).to(device)
    tp = TextProjector(input_dim=512, embed_dim=embed_dim).to(device)
    n_params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in tp.parameters())
    print(f"  Total parameters: {n_params:,}")

    # Build augmentation
    augment = EEGAugmentation(aug_config) if aug_config else None

    # Build data loader
    train_ds = TensorDataset(X_train_tensor, T_train_tensor, Y_train_tensor)
    if batch_mode == 'class_balanced':
        batch_sampler = ClassBalancedBatchSampler(
            label_train_seen, K=K, M=M, n_batches=len(train_loader)
        )
        loader = DataLoader(train_ds, batch_sampler=batch_sampler)
    else:
        loader = DataLoader(train_ds, batch_size=CLIP_CONFIG['batch_size'], shuffle=True)

    # Optimizer
    params = list(enc.parameters()) + list(tp.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Scheduler
    total_steps = epochs * len(loader)
    warmup_steps = int(0.1 * total_steps) if schedule == 'cosine_warmup' else 0

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        if schedule == 'cosine_warmup':
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return 1.0

    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training
    t0 = time.time()
    losses = []

    for epoch in range(epochs):
        enc.train()
        tp.train()
        epoch_loss = 0.0

        for brain_batch, text_batch, label_batch in loader:
            brain_batch = brain_batch.to(device)
            text_batch = text_batch.to(device)
            label_batch = label_batch.to(device)

            # Augmentation
            if augment is not None:
                brain_batch, text_batch = augment(brain_batch, text_batch)

            # Forward
            brain_embeds = enc(brain_batch)
            text_embeds = tp(text_batch)

            # Loss
            if loss_type == 'infonce':
                loss = clip_loss(brain_embeds, text_embeds, tau)
            elif loss_type == 'supcon':
                loss = supcon_loss(brain_embeds, label_batch, tau)
            elif loss_type == 'combined':
                loss = combined_loss(brain_embeds, text_embeds, label_batch, tau, lambda_sup)
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sched.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        current_lr = sched.get_last_lr()[0]

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: loss={avg_loss:.4f}, lr={current_lr:.6f}")

    elapsed = time.time() - t0

    # Evaluate
    res = evaluate_encoder(
        enc, tp,
        X_train_tensor, T_train_tensor, label_train_seen,
        X_test_tensor, T_test_tensor, label_test_seen,
        device
    )

    print(f"\n  RESULTS: top-1={res['top_1']*100:.2f}%, top-5={res['top_5']*100:.2f}%, "
          f"top-10={res['top_10']*100:.2f}%, signal={res['signal']:.1f}x")
    print(f"  Time: {elapsed:.1f}s, Final loss: {losses[-1]:.4f}")

    result = {
        'name': name,
        'config': config,
        'top_1': res['top_1'],
        'top_5': res['top_5'],
        'top_10': res['top_10'],
        'signal': res['signal'],
        'final_loss': losses[-1],
        'time': elapsed,
        'losses': losses,
        'n_params': n_params,
        'encoder_obj': enc,
        'text_proj_obj': tp,
    }
    phase_f_results.append(result)
    return result


# Print readiness
print("Phase F experiment infrastructure ready.")
print(f"  Encoders: {list(ENCODER_REGISTRY.keys())}")
print(f"  Loss types: infonce, supcon, combined")
print(f"  Augmentations: gaussian, time_shift, channel_drop, time_mask, mixup")
print(f"  Baseline reference: top-1 = 2.87% (Phase E best)")
```

---

## Cell 116 — Stage F1 Round 1: Individual Augmentations

```python
# =============================================================================
# STAGE F1 ROUND 1: INDIVIDUAL AUGMENTATIONS
# =============================================================================
# Test each augmentation alone against the no-augmentation baseline.
# All use current best config: MLP+LN, dim=128, InfoNCE, standard batches.

print("STAGE F1 ROUND 1: Individual Augmentations")
print("=" * 60)

# Baseline (no augmentation) — control
run_phase_f_experiment({
    'name': 'F1-ctrl: no_aug',
    'encoder': 'mlp_ln',
    'loss_type': 'infonce',
    'batch_mode': 'shuffle',
    'augmentation': None,
})

# F1-A: Gaussian noise
run_phase_f_experiment({
    'name': 'F1-A: gaussian',
    'encoder': 'mlp_ln',
    'loss_type': 'infonce',
    'batch_mode': 'shuffle',
    'augmentation': {'gaussian_std': 0.3},
})

# F1-B: Time shift
run_phase_f_experiment({
    'name': 'F1-B: time_shift',
    'encoder': 'mlp_ln',
    'loss_type': 'infonce',
    'batch_mode': 'shuffle',
    'augmentation': {'time_shift': 2},
})

# F1-C: Channel dropout
run_phase_f_experiment({
    'name': 'F1-C: channel_drop',
    'encoder': 'mlp_ln',
    'loss_type': 'infonce',
    'batch_mode': 'shuffle',
    'augmentation': {'channel_drop_prob': 0.15},
})

# F1-D: Time masking
run_phase_f_experiment({
    'name': 'F1-D: time_mask',
    'encoder': 'mlp_ln',
    'loss_type': 'infonce',
    'batch_mode': 'shuffle',
    'augmentation': {'time_mask_width': 4},
})

# F1-E: Mixup
run_phase_f_experiment({
    'name': 'F1-E: mixup',
    'encoder': 'mlp_ln',
    'loss_type': 'infonce',
    'batch_mode': 'shuffle',
    'augmentation': {'mixup_alpha': 0.3},
})

# Summary
print(f"\n{'='*60}")
print("ROUND 1 RESULTS:")
r1_results = [r for r in phase_f_results if r['name'].startswith('F1-')]
r1_results_sorted = sorted(r1_results, key=lambda r: r['top_1'], reverse=True)
for r in r1_results_sorted:
    print(f"  {r['name']:<30s} top-1={r['top_1']*100:.2f}%  top-5={r['top_5']*100:.2f}%  signal={r['signal']:.1f}x")

# Select top 3
top3_augs = r1_results_sorted[:3]
print(f"\nTop 3 augmentations for Round 2: {[r['name'] for r in top3_augs]}")
```

---

## Cell 117 — Stage F1 Round 2: Combined Augmentations

```python
# =============================================================================
# STAGE F1 ROUND 2: COMBINED AUGMENTATIONS
# =============================================================================
# Combine the top augmentations from Round 1.

print("STAGE F1 ROUND 2: Combined Augmentations")
print("=" * 60)

# Extract the augmentation configs from top 3
aug_configs_r1 = []
for r in top3_augs:
    if r['config']['augmentation'] is not None:
        aug_configs_r1.append(r['config']['augmentation'])

# Combine top 2
combo_top2 = {}
for cfg in aug_configs_r1[:2]:
    combo_top2.update(cfg)

run_phase_f_experiment({
    'name': 'F1-F: combo_top2',
    'encoder': 'mlp_ln',
    'loss_type': 'infonce',
    'batch_mode': 'shuffle',
    'augmentation': combo_top2,
})

# Combine top 3
combo_top3 = {}
for cfg in aug_configs_r1[:3]:
    combo_top3.update(cfg)

run_phase_f_experiment({
    'name': 'F1-G: combo_top3',
    'encoder': 'mlp_ln',
    'loss_type': 'infonce',
    'batch_mode': 'shuffle',
    'augmentation': combo_top3,
})

# All augmentations with reduced intensity
combo_all = {
    'gaussian_std': 0.15,
    'time_shift': 1,
    'channel_drop_prob': 0.1,
    'time_mask_width': 3,
    'mixup_alpha': 0.2,
}

run_phase_f_experiment({
    'name': 'F1-H: combo_all_mild',
    'encoder': 'mlp_ln',
    'loss_type': 'infonce',
    'batch_mode': 'shuffle',
    'augmentation': combo_all,
})

# Summary
print(f"\n{'='*60}")
print("ROUND 2 RESULTS:")
all_f1 = [r for r in phase_f_results if r['name'].startswith('F1-')]
all_f1_sorted = sorted(all_f1, key=lambda r: r['top_1'], reverse=True)
BEST_AUG = all_f1_sorted[0]
print(f"\nBEST AUGMENTATION: {BEST_AUG['name']} — top-1 = {BEST_AUG['top_1']*100:.2f}%")
BEST_AUG_CONFIG = BEST_AUG['config']['augmentation']
print(f"AUG* config: {BEST_AUG_CONFIG}")
```

---

## Cell 118 — Stage F1 Results Summary

```python
# =============================================================================
# STAGE F1 RESULTS — AUGMENTATION SWEEP
# =============================================================================

print("=" * 80)
print("STAGE F1 RESULTS — DATA AUGMENTATION SWEEP")
print("=" * 80)

all_f1 = [r for r in phase_f_results if r['name'].startswith('F1-')]
print(f"{'Name':<35s} {'Top-1':>7s} {'Top-5':>7s} {'Top-10':>7s} {'Signal':>8s} {'Loss':>8s} {'Time':>6s}")
print("-" * 80)
for r in sorted(all_f1, key=lambda x: x['top_1'], reverse=True):
    print(f"{r['name']:<35s} {r['top_1']*100:>6.2f}% {r['top_5']*100:>6.2f}% "
          f"{r['top_10']*100:>6.2f}% {r['signal']:>7.1f}x {r['final_loss']:>7.4f} {r['time']:>5.0f}s")
print("-" * 80)
print(f"\nBASELINE (Phase E, no aug): top-1 = 2.87%")
print(f"BEST AUG: {BEST_AUG['name']} — top-1 = {BEST_AUG['top_1']*100:.2f}%")

# Save figure
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

names = [r['name'].replace('F1-', '') for r in sorted(all_f1, key=lambda x: x['top_1'])]
accs = [r['top_1']*100 for r in sorted(all_f1, key=lambda x: x['top_1'])]
colors = ['#4CAF50' if r['top_1'] == BEST_AUG['top_1'] else '#2196F3'
          for r in sorted(all_f1, key=lambda x: x['top_1'])]
ax1.barh(names, accs, color=colors)
ax1.axvline(x=2.87, color='red', linestyle='--', label='Phase E baseline (2.87%)')
ax1.set_xlabel('Top-1 Accuracy (%)')
ax1.set_title('F1: Augmentation Sweep')
ax1.legend()

top5 = sorted(all_f1, key=lambda x: x['top_1'], reverse=True)[:5]
for r in top5:
    ax2.plot(r['losses'], label=f"{r['name'].replace('F1-', '')} ({r['top_1']*100:.1f}%)")
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Top 5 Loss Curves')
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig('figures/phase_f_stage1_augmentation.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figures/phase_f_stage1_augmentation.png")
```

---

## Cell 119 — Stage F2: Loss Function Variants

```python
# =============================================================================
# STAGE F2: LOSS FUNCTION VARIANTS (using AUG*)
# =============================================================================
# Test SupCon and combined losses with class-balanced batching.

print("STAGE F2: Loss Function Variants (using AUG*)")
print("=" * 60)
print(f"  AUG* = {BEST_AUG_CONFIG}")

# F2-I: SupCon only (EEG<->EEG)
run_phase_f_experiment({
    'name': 'F2-I: supcon_only',
    'encoder': 'mlp_ln',
    'loss_type': 'supcon',
    'batch_mode': 'class_balanced',
    'K': 64, 'M': 4,
    'augmentation': BEST_AUG_CONFIG,
})

# F2-J: Combined (0.5 SupCon + 0.5 InfoNCE)
run_phase_f_experiment({
    'name': 'F2-J: combined_0.5',
    'encoder': 'mlp_ln',
    'loss_type': 'combined',
    'lambda_sup': 0.5,
    'batch_mode': 'class_balanced',
    'K': 64, 'M': 4,
    'augmentation': BEST_AUG_CONFIG,
})

# F2-K: Combined (0.3 SupCon + 0.7 InfoNCE)
run_phase_f_experiment({
    'name': 'F2-K: combined_0.3',
    'encoder': 'mlp_ln',
    'loss_type': 'combined',
    'lambda_sup': 0.3,
    'batch_mode': 'class_balanced',
    'K': 64, 'M': 4,
    'augmentation': BEST_AUG_CONFIG,
})

# Compare with AUG* baseline (InfoNCE, standard batch)
all_f2 = [r for r in phase_f_results if r['name'].startswith('F2-')]
all_f2.append(BEST_AUG)  # include F1 winner for comparison
all_f2_sorted = sorted(all_f2, key=lambda r: r['top_1'], reverse=True)
BEST_LOSS = all_f2_sorted[0]

print(f"\n{'='*60}")
print("STAGE F2 RESULTS:")
for r in all_f2_sorted:
    print(f"  {r['name']:<30s} top-1={r['top_1']*100:.2f}%  top-5={r['top_5']*100:.2f}%  signal={r['signal']:.1f}x")
print(f"\nBEST LOSS: {BEST_LOSS['name']} — top-1 = {BEST_LOSS['top_1']*100:.2f}%")
BEST_LOSS_CONFIG = BEST_LOSS['config']
```

---

## Cell 120 — Stage F2 Results Summary

```python
# =============================================================================
# STAGE F2 RESULTS — LOSS FUNCTION SWEEP
# =============================================================================

print("=" * 80)
print("STAGE F2 RESULTS — LOSS FUNCTION SWEEP")
print("=" * 80)

f2_display = [r for r in phase_f_results if r['name'].startswith('F2-')]
f2_display.append(BEST_AUG)
print(f"{'Name':<35s} {'Top-1':>7s} {'Top-5':>7s} {'Top-10':>7s} {'Signal':>8s} {'Loss':>8s} {'Time':>6s}")
print("-" * 80)
for r in sorted(f2_display, key=lambda x: x['top_1'], reverse=True):
    print(f"{r['name']:<35s} {r['top_1']*100:>6.2f}% {r['top_5']*100:>6.2f}% "
          f"{r['top_10']*100:>6.2f}% {r['signal']:>7.1f}x {r['final_loss']:>7.4f} {r['time']:>5.0f}s")
print("-" * 80)
print(f"BEST LOSS: {BEST_LOSS['name']} — top-1 = {BEST_LOSS['top_1']*100:.2f}%")
```

---

## Cell 121 — Stage F3: Architecture Variants

```python
# =============================================================================
# STAGE F3: ARCHITECTURE VARIANTS (using AUG* + LOSS*)
# =============================================================================
# Test EEGNet and ShallowConvNet against MLP+LayerNorm baseline.

print("STAGE F3: Architecture Variants (using AUG* + LOSS*)")
print("=" * 60)

# Extract loss settings from best
best_loss_type = BEST_LOSS_CONFIG.get('loss_type', 'infonce')
best_lambda = BEST_LOSS_CONFIG.get('lambda_sup', 0.5)
best_batch_mode = BEST_LOSS_CONFIG.get('batch_mode', 'shuffle')
best_K = BEST_LOSS_CONFIG.get('K', 64)
best_M = BEST_LOSS_CONFIG.get('M', 4)
best_aug = BEST_LOSS_CONFIG.get('augmentation', BEST_AUG_CONFIG)

base_config = {
    'loss_type': best_loss_type,
    'lambda_sup': best_lambda,
    'batch_mode': best_batch_mode,
    'K': best_K, 'M': best_M,
    'augmentation': best_aug,
}

# F3-L: MLP+LN (control)
run_phase_f_experiment({
    **base_config,
    'name': 'F3-L: mlp_ln',
    'encoder': 'mlp_ln',
    'dropout': 0.1,
})

# F3-M: EEGNet small
run_phase_f_experiment({
    **base_config,
    'name': 'F3-M: eegnet_small',
    'encoder': 'eegnet_small',
    'dropout': 0.25,
})

# F3-N: EEGNet large
run_phase_f_experiment({
    **base_config,
    'name': 'F3-N: eegnet_large',
    'encoder': 'eegnet_large',
    'dropout': 0.25,
})

# F3-O: ShallowConvNet
run_phase_f_experiment({
    **base_config,
    'name': 'F3-O: shallow_conv',
    'encoder': 'shallow_conv',
    'dropout': 0.5,
})

# Select best
all_f3 = [r for r in phase_f_results if r['name'].startswith('F3-')]
all_f3_sorted = sorted(all_f3, key=lambda r: r['top_1'], reverse=True)
BEST_ARCH = all_f3_sorted[0]

print(f"\n{'='*60}")
print("STAGE F3 RESULTS:")
for r in all_f3_sorted:
    print(f"  {r['name']:<30s} top-1={r['top_1']*100:.2f}%  top-5={r['top_5']*100:.2f}%  "
          f"signal={r['signal']:.1f}x  params={r['n_params']:,}")
print(f"\nBEST ARCHITECTURE: {BEST_ARCH['name']} — top-1 = {BEST_ARCH['top_1']*100:.2f}%")
```

---

## Cell 122 — Stage F3 Results Summary

```python
# =============================================================================
# STAGE F3 RESULTS — ARCHITECTURE SWEEP
# =============================================================================

print("=" * 80)
print("STAGE F3 RESULTS — ARCHITECTURE SWEEP")
print("=" * 80)

all_f3 = [r for r in phase_f_results if r['name'].startswith('F3-')]
print(f"{'Name':<35s} {'Top-1':>7s} {'Top-5':>7s} {'Top-10':>7s} {'Signal':>8s} {'Params':>10s} {'Time':>6s}")
print("-" * 80)
for r in sorted(all_f3, key=lambda x: x['top_1'], reverse=True):
    print(f"{r['name']:<35s} {r['top_1']*100:>6.2f}% {r['top_5']*100:>6.2f}% "
          f"{r['top_10']*100:>6.2f}% {r['signal']:>7.1f}x {r['n_params']:>10,} {r['time']:>5.0f}s")
print("-" * 80)
print(f"\nBASELINE (Phase E): top-1 = 2.87%")
print(f"BEST F: {BEST_ARCH['name']} — top-1 = {BEST_ARCH['top_1']*100:.2f}%")

BEST_F = BEST_ARCH['config']
print(f"\n{'='*60}")
print("BEST OVERALL PHASE F CONFIG:")
for k, v in BEST_F.items():
    if k != 'name':
        print(f"  {k}: {v}")

# Save figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

all_results = sorted(phase_f_results, key=lambda x: x['top_1'])
names = [r['name'] for r in all_results]
accs = [r['top_1']*100 for r in all_results]
stage_colors = []
for r in all_results:
    if r['name'].startswith('F1-'): stage_colors.append('#2196F3')
    elif r['name'].startswith('F2-'): stage_colors.append('#FF9800')
    elif r['name'].startswith('F3-'): stage_colors.append('#4CAF50')
    else: stage_colors.append('#9E9E9E')
ax1.barh(names, accs, color=stage_colors)
ax1.axvline(x=2.87, color='red', linestyle='--', label='Phase E baseline')
ax1.set_xlabel('Top-1 Accuracy (%)')
ax1.set_title('All Phase F Experiments')
ax1.legend()

top5 = sorted(phase_f_results, key=lambda x: x['top_1'], reverse=True)[:5]
for r in top5:
    ax2.plot(r['losses'], label=f"{r['name']} ({r['top_1']*100:.1f}%)")
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Top 5 Configs: Loss Curves')
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig('figures/phase_f_stage3_architecture.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figures/phase_f_stage3_architecture.png")
```

---

## Cell 123 — Full Pipeline Re-Run

```python
# =============================================================================
# FULL PIPELINE RE-RUN WITH BEST PHASE F CONFIG
# =============================================================================

print("=" * 70)
print("FULL PIPELINE RE-RUN WITH BEST PHASE F CONFIG")
print("=" * 70)
print(f"Config: {BEST_F}")

# Step 1: Use the already-trained encoder from the best experiment
best_enc = BEST_ARCH['encoder_obj']
best_tp = BEST_ARCH['text_proj_obj']
best_embed_dim = BEST_F.get('embed_dim', 128)

print(f"\nStep 1: Using best encoder from {BEST_ARCH['name']}")
print(f"  Encoder top-1: {BEST_ARCH['top_1']*100:.2f}%")

# Step 2: Extract embeddings
print("\nStep 2: Extracting embeddings...")
best_enc.eval()
best_tp.eval()
with torch.no_grad():
    E_train_seen_f = best_enc(X_train_tensor.to(device)).cpu().numpy()
    E_test_seen_f = best_enc(X_test_tensor.to(device)).cpu().numpy()
    E_unseen_f = best_enc(X_unseen_tensor.to(device)).cpu().numpy()
print(f"  E_train_seen: {E_train_seen_f.shape}")
print(f"  E_test_seen:  {E_test_seen_f.shape}")
print(f"  E_unseen:     {E_unseen_f.shape}")

# Step 3: Compute prototypes
print("\nStep 3: Computing prototypes...")
with torch.no_grad():
    T_train_proj = best_tp(T_train_tensor.to(device)).cpu().numpy()
    T_unseen_proj = best_tp(T_unseen_tensor.to(device)).cpu().numpy()

seen_classes_sorted = np.sort(np.unique(label_train_seen))
S_seen_f = np.zeros((len(seen_classes_sorted), best_embed_dim))
for i, c in enumerate(seen_classes_sorted):
    mask = label_train_seen == c
    S_seen_f[i] = T_train_proj[mask].mean(axis=0)
    S_seen_f[i] /= np.linalg.norm(S_seen_f[i])

unseen_labels_np = Y_unseen_tensor.numpy()
unseen_classes_sorted = np.sort(np.unique(unseen_labels_np))
S_unseen_f = np.zeros((len(unseen_classes_sorted), best_embed_dim))
for i, c in enumerate(unseen_classes_sorted):
    mask = unseen_labels_np == c
    S_unseen_f[i] = T_unseen_proj[mask].mean(axis=0)
    S_unseen_f[i] /= np.linalg.norm(S_unseen_f[i])

print(f"  Seen prototypes: {S_seen_f.shape}")
print(f"  Unseen prototypes: {S_unseen_f.shape}")

# Step 4: Save cached arrays
print("\nStep 4: Saving cached arrays...")
np.save('cached_arrays/E_train_seen.npy', E_train_seen_f)
np.save('cached_arrays/E_test_seen.npy', E_test_seen_f)
np.save('cached_arrays/E_unseen.npy', E_unseen_f)
np.save('cached_arrays/S_seen_prototypes.npy', S_seen_f)
np.save('cached_arrays/S_unseen_prototypes.npy', S_unseen_f)
print("  Cached arrays overwritten.")

# Step 5: Retrain WGAN-GP
print("\nStep 5: Retraining WGAN-GP...")
torch.manual_seed(42)

wgan_embed_dim = best_embed_dim
gen_f = Generator(z_dim=WGAN_CONFIG['z_dim'], embed_dim=wgan_embed_dim).to(device)
crit_f = Critic(embed_dim=wgan_embed_dim).to(device)

opt_g = torch.optim.Adam(gen_f.parameters(), lr=WGAN_CONFIG['lr'], betas=WGAN_CONFIG['betas'])
opt_c = torch.optim.Adam(crit_f.parameters(), lr=WGAN_CONFIG['lr'], betas=WGAN_CONFIG['betas'])

E_train_tensor_f = torch.FloatTensor(E_train_seen_f).to(device)
S_train_cond_f = torch.FloatTensor(T_train_proj).to(device)
S_train_cond_f = F.normalize(S_train_cond_f, p=2, dim=-1)

n_samples = E_train_tensor_f.shape[0]
n_critic = WGAN_CONFIG['n_critic']
n_steps = WGAN_CONFIG['n_steps']
lam_gp = WGAN_CONFIG['lambda_gp']
z_dim = WGAN_CONFIG['z_dim']
bs = WGAN_CONFIG['batch_size']

for step in range(1, n_steps + 1):
    for _ in range(n_critic):
        idx = torch.randint(0, n_samples, (bs,))
        real_e = E_train_tensor_f[idx]
        cond = S_train_cond_f[idx]
        z = torch.randn(bs, z_dim, device=device)
        fake_e = gen_f(z, cond)

        c_real = crit_f(real_e, cond).mean()
        c_fake = crit_f(fake_e.detach(), cond).mean()
        gp = gradient_penalty(crit_f, real_e, fake_e.detach(), cond, device, lam_gp)

        c_loss = c_fake - c_real + gp
        opt_c.zero_grad()
        c_loss.backward()
        opt_c.step()

    z = torch.randn(bs, z_dim, device=device)
    idx = torch.randint(0, n_samples, (bs,))
    cond = S_train_cond_f[idx]
    fake_e = gen_f(z, cond)
    g_loss = -crit_f(fake_e, cond).mean()
    opt_g.zero_grad()
    g_loss.backward()
    opt_g.step()

    if step % 2000 == 0:
        print(f"  WGAN step {step}/{n_steps}: G={g_loss.item():.4f}, C={c_loss.item():.4f}")

print("  WGAN-GP training complete.")

# Step 6: Generate synthetic unseen embeddings
print("\nStep 6: Generating synthetic unseen embeddings...")
gen_f.eval()
n_synth = WGAN_CONFIG['n_synth_per_class']
E_synth_list = []
y_synth_list = []
with torch.no_grad():
    for i, c in enumerate(unseen_classes_sorted):
        cond = torch.FloatTensor(S_unseen_f[i]).unsqueeze(0).repeat(n_synth, 1).to(device)
        z = torch.randn(n_synth, z_dim, device=device)
        synth = gen_f(z, cond).cpu().numpy()
        E_synth_list.append(synth)
        y_synth_list.extend([c] * n_synth)

E_synth_unseen_f = np.vstack(E_synth_list)
y_synth_unseen_f = np.array(y_synth_list)
print(f"  E_synth_unseen: {E_synth_unseen_f.shape}")

# Step 7: GZSL evaluation (balanced counts)
print("\nStep 7: GZSL evaluation (balanced counts)...")
from collections import Counter
seen_counts = Counter(label_train_seen)
mean_seen_per_class = np.mean(list(seen_counts.values()))
target_per_unseen = int(round(mean_seen_per_class))

E_synth_bal = []
y_synth_bal = []
for c in unseen_classes_sorted:
    mask = y_synth_unseen_f == c
    indices = np.where(mask)[0]
    if len(indices) > target_per_unseen:
        chosen = np.random.choice(indices, target_per_unseen, replace=False)
    else:
        chosen = indices
    E_synth_bal.append(E_synth_unseen_f[chosen])
    y_synth_bal.extend([c] * len(chosen))

E_synth_bal = np.vstack(E_synth_bal)
y_synth_bal = np.array(y_synth_bal)

max_seen = int(seen_classes_sorted.max())
y_synth_bal_offset = y_synth_bal + max_seen
y_unseen_offset = unseen_labels_np + max_seen

X_gzsl = np.vstack([E_train_seen_f, E_synth_bal])
y_gzsl = np.concatenate([label_train_seen, y_synth_bal_offset])

clf_f = LogisticRegression(max_iter=1000, solver='saga', n_jobs=-1, random_state=42)
clf_f.fit(X_gzsl, y_gzsl)

pred_seen = clf_f.predict(E_test_seen_f)
acc_s = (pred_seen == label_test_seen).mean()

pred_unseen = clf_f.predict(E_unseen_f)
acc_u = (pred_unseen == y_unseen_offset).mean()

h_mean = 2 * acc_s * acc_u / (acc_s + acc_u + 1e-8)

unseen_label_set = set(y_synth_bal_offset)
all_preds = np.concatenate([pred_seen, pred_unseen])
routing_rate = np.mean([p in unseen_label_set for p in all_preds])

clf_synth = LogisticRegression(max_iter=1000, solver='saga', n_jobs=-1, random_state=42)
clf_synth.fit(E_synth_unseen_f, y_synth_unseen_f)
transfer_acc = clf_synth.score(E_unseen_f, unseen_labels_np)

print(f"\n{'='*60}")
print("OPTIMISED PIPELINE RESULTS (Phase F)")
print(f"{'='*60}")
print(f"  Encoder top-1 (seen-only): {BEST_ARCH['top_1']*100:.2f}%")
print(f"  AccS (GZSL):               {acc_s*100:.2f}%")
print(f"  AccU (GZSL):               {acc_u*100:.2f}%")
print(f"  H-mean:                    {h_mean*100:.2f}%")
print(f"  Routing rate:              {routing_rate*100:.2f}%")
print(f"  Synth->Real transfer:      {transfer_acc*100:.2f}%")

pipeline_f_results = {
    'encoder_top1': BEST_ARCH['top_1'],
    'acc_s': acc_s, 'acc_u': acc_u, 'h_mean': h_mean,
    'routing_rate': routing_rate, 'transfer': transfer_acc,
}
```

---

## Cell 124 — Final Comparison

```python
# =============================================================================
# FINAL COMPARISON — PHASE F vs PHASE E vs BASELINE
# =============================================================================

print("=" * 80)
print("FINAL COMPARISON")
print("=" * 80)

baseline = {'enc': 1.90, 'acc_s': 0.06, 'acc_u': 2.13, 'h': 0.12, 'rr': 99.70, 'xfer': 1.73}
phase1 =   {'enc': 1.90, 'acc_s': 2.90, 'acc_u': 0.22, 'h': 0.42, 'rr': 9.16,  'xfer': 1.73}
phase_e =  {'enc': 2.87, 'acc_s': 2.96, 'acc_u': 0.40, 'h': 0.70, 'rr': 11.31, 'xfer': 2.51}
phase_f =  {
    'enc': BEST_ARCH['top_1']*100,
    'acc_s': acc_s*100, 'acc_u': acc_u*100, 'h': h_mean*100,
    'rr': routing_rate*100, 'xfer': transfer_acc*100,
}

print(f"\n{'Metric':<45s} {'Baseline':>10s} {'Phase1':>10s} {'Phase E':>10s} {'Phase F':>10s}")
print("-" * 85)
for label, key in [
    ('Encoder top-1 (seen-only)', 'enc'),
    ('AccS (GZSL)', 'acc_s'),
    ('AccU (GZSL)', 'acc_u'),
    ('H-mean', 'h'),
    ('Routing rate', 'rr'),
    ('Synth->Real transfer (200-way)', 'xfer'),
]:
    print(f"{label:<45s} {baseline[key]:>9.2f}% {phase1[key]:>9.2f}% "
          f"{phase_e[key]:>9.2f}% {phase_f[key]:>9.2f}%")
print("-" * 85)

# Save figure
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

phases = ['Baseline', 'Phase 1', 'Phase E', 'Phase F']
colors = ['#EF5350', '#42A5F5', '#66BB6A', '#AB47BC']

enc_vals = [baseline['enc'], phase1['enc'], phase_e['enc'], phase_f['enc']]
axes[0].bar(phases, enc_vals, color=colors)
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title('Encoder Top-1')

h_vals = [baseline['h'], phase1['h'], phase_e['h'], phase_f['h']]
axes[1].bar(phases, h_vals, color=colors)
axes[1].set_ylabel('H-mean (%)')
axes[1].set_title('GZSL H-mean')

for r in phase_f_results:
    c = '#2196F3' if r['name'].startswith('F1') else '#FF9800' if r['name'].startswith('F2') else '#4CAF50'
    axes[2].scatter(r['n_params'], r['top_1']*100, color=c, s=60, zorder=5)
    axes[2].annotate(r['name'].split(': ')[1] if ': ' in r['name'] else r['name'],
                     (r['n_params'], r['top_1']*100), fontsize=6, ha='left', va='bottom')
axes[2].set_xlabel('Parameters')
axes[2].set_ylabel('Top-1 (%)')
axes[2].set_title('Accuracy vs Model Size')
axes[2].set_xscale('log')

plt.tight_layout()
plt.savefig('figures/phase_f_final_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figures/phase_f_final_comparison.png")

print(f"\n{'='*60}")
print("PHASE F COMPLETE")
print(f"Best config: {BEST_F}")
print(f"{'='*60}")
```

---

## Implementation Notes for Agent

1. **Create** `helper_files/add_phase_f_cells.py` that reads the notebook and appends these 11 cells (114–124)
2. **Cell 115 is the big one** — ~400 lines. All other cells are 30-80 lines.
3. **Do NOT modify existing cells** (0–113). Only append new cells.
4. **All objects from earlier cells are available**: `BrainEncoder`, `TextProjector`, `clip_loss`, `CLIP_CONFIG`, `device`, `train_loader`, all tensors, `Generator`, `Critic`, `gradient_penalty`, `WGAN_CONFIG`, `label_train_seen`, `label_test_seen`.
5. **The helper script pattern**: see `helper_files/add_clip_optimisation.py` or `helper_files/integrate_clip_optimisation.py` for reference.
6. Verify the notebook goes from 114 → 125 cells after running the script.
