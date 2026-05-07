"""
Integrate Phase E CLIP optimisation results into the production pipeline.

Changes:
  - Cell 28: Update markdown (architecture description)
  - Cell 29: Update CLIP_CONFIG (embed_dim=128, tau=0.15, lr=1e-3, epochs=50, cosine_warmup)
  - Cell 30: Add LayerNorm to BrainEncoder
  - Cell 32: Add cosine warmup scheduler to training loop
  - Cell 39: Update WGAN_CONFIG embed_dim to 128
  - Cell 108: Update markdown (R^64 → R^128)
  - Cell 111: Update comments (d=64 → d=128)
  - Cells 114–125: DELETE (Phase E sweep scaffolding)

Result: Notebook goes from 126 → 114 cells with optimised config baked in.
"""

import json
import sys
import os
import shutil
from datetime import datetime

NOTEBOOK_PATH = 'COMP2261_ArizMLCW_with_baseline.ipynb'

def make_source(text):
    """Convert a multi-line string into notebook source format (list of lines)."""
    lines = text.split('\n')
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + '\n')
        else:
            result.append(line)
    return result


def main():
    # Load notebook
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    cells = nb['cells']
    print(f"Notebook loaded: {len(cells)} cells")
    assert len(cells) == 126, f"Expected 126 cells, got {len(cells)}"

    # =========================================================================
    # CELL 28: Update markdown description
    # =========================================================================
    cells[28]['source'] = make_source("""---

# Brain-Text CLIP Encoder

This section implements a **CLIP-style contrastive encoder** that aligns EEG embeddings and text embeddings in a shared semantic space.

**Architecture:**
- Brain Encoder `f_b`: 561 → 1024 → 512 → d (d=128, with LayerNorm)
- Text Projector `g`: 512 → 512 → d
- Both outputs are L2-normalised

**Training:**
- Loss: Symmetric InfoNCE / CLIP contrastive loss
- Schedule: Cosine warmup (10% warmup, cosine decay to 0)
- Temperature: τ=0.15 (fixed)

**Training data:** Seen EEG trials only (no leakage from unseen)""")
    print("  Cell 28 (markdown): Updated architecture description")

    # =========================================================================
    # CELL 29: Update CLIP_CONFIG
    # =========================================================================
    cells[29]['source'] = make_source("""# =============================================================================
# CLIP CONFIGURATION
# =============================================================================
# All hyperparameters in one place for reproducibility
# Updated with Phase E optimisation results (best config from parametric sweep)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import random
import math

# Configuration dictionary — Phase E optimised
CLIP_CONFIG = {
    'embed_dim': 128,          # Output embedding dimension d (was 64)
    'tau': 0.15,               # Temperature for contrastive loss (was 0.07)
    'epochs': 50,              # Training epochs (was 20)
    'batch_size': 256,         # Batch size
    'lr': 1e-3,                # Learning rate (was 1e-4)
    'weight_decay': 1e-4,      # AdamW weight decay
    'dropout': 0.1,            # Dropout rate
    'schedule': 'cosine_warmup',  # LR schedule (was flat)
    'seed': 42,                # Random seed
}

# Set seeds for reproducibility
SEED = CLIP_CONFIG['seed']
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 60)
print("CLIP CONFIGURATION (Phase E Optimised)")
print("=" * 60)
for k, v in CLIP_CONFIG.items():
    print(f"  {k}: {v}")
print(f"  device: {device}")
print("=" * 60)""")
    print("  Cell 29 (CLIP_CONFIG): Updated to optimised values")

    # =========================================================================
    # CELL 30: Add LayerNorm to BrainEncoder
    # =========================================================================
    cells[30]['source'] = make_source("""# =============================================================================
# CLIP MODEL DEFINITIONS
# =============================================================================

class BrainEncoder(nn.Module):
    \"\"\"EEG feature encoder: 561 -> d with LayerNorm + L2 normalisation.\"\"\"

    def __init__(self, input_dim=561, hidden_dims=[1024, 512], embed_dim=64, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.ln1 = nn.LayerNorm(hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.ln2 = nn.LayerNorm(hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


class TextProjector(nn.Module):
    \"\"\"Text embedding projector: 512 -> d with L2 normalisation.\"\"\"

    def __init__(self, input_dim=512, hidden_dim=512, embed_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


def clip_loss(brain_embeds, text_embeds, tau=0.07):
    \"\"\"
    Symmetric CLIP/InfoNCE contrastive loss.

    Args:
        brain_embeds: (B, d) L2-normalised brain embeddings
        text_embeds: (B, d) L2-normalised text embeddings
        tau: temperature parameter

    Returns:
        Scalar loss: (L_e2t + L_t2e) / 2
    \"\"\"
    logits = torch.matmul(brain_embeds, text_embeds.T) / tau  # (B, B)
    batch_size = brain_embeds.size(0)
    targets = torch.arange(batch_size, device=brain_embeds.device)
    loss_e2t = F.cross_entropy(logits, targets)
    loss_t2e = F.cross_entropy(logits.T, targets)
    return (loss_e2t + loss_t2e) / 2


# Instantiate models
brain_encoder = BrainEncoder(
    input_dim=561,
    embed_dim=CLIP_CONFIG['embed_dim'],
    dropout=CLIP_CONFIG['dropout']
).to(device)

text_projector = TextProjector(
    input_dim=512,
    embed_dim=CLIP_CONFIG['embed_dim']
).to(device)

print(f"Brain Encoder parameters: {sum(p.numel() for p in brain_encoder.parameters()):,}")
print(f"Text Projector parameters: {sum(p.numel() for p in text_projector.parameters()):,}")""")
    print("  Cell 30 (models): BrainEncoder updated with LayerNorm")

    # =========================================================================
    # CELL 32: Add cosine warmup scheduler to training loop
    # =========================================================================
    cells[32]['source'] = make_source("""# =============================================================================
# CLIP TRAINING
# =============================================================================

# Optimizer
clip_params = list(brain_encoder.parameters()) + list(text_projector.parameters())
clip_optimizer = torch.optim.AdamW(
    clip_params,
    lr=CLIP_CONFIG['lr'],
    weight_decay=CLIP_CONFIG['weight_decay']
)

# Cosine warmup LR scheduler
total_steps = CLIP_CONFIG['epochs'] * len(train_loader)
warmup_steps = int(0.1 * total_steps)

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(clip_optimizer, lr_lambda)

# Training loop
clip_losses = []
print("Training CLIP encoder...")
print(f"Epochs: {CLIP_CONFIG['epochs']}, Batches per epoch: {len(train_loader)}")
print(f"Schedule: cosine warmup ({warmup_steps} warmup steps / {total_steps} total)")

for epoch in range(CLIP_CONFIG['epochs']):
    brain_encoder.train()
    text_projector.train()
    epoch_loss = 0.0

    for batch_idx, (brain_batch, text_batch, _) in enumerate(train_loader):
        brain_batch = brain_batch.to(device)
        text_batch = text_batch.to(device)

        # Forward pass
        brain_embeds = brain_encoder(brain_batch)
        text_embeds = text_projector(text_batch)

        # Compute loss
        loss = clip_loss(brain_embeds, text_embeds, tau=CLIP_CONFIG['tau'])

        # Backward pass
        clip_optimizer.zero_grad()
        loss.backward()
        clip_optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    clip_losses.append(avg_loss)
    current_lr = scheduler.get_last_lr()[0]

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:3d}/{CLIP_CONFIG['epochs']}: Loss = {avg_loss:.4f}, LR = {current_lr:.6f}")

print(f"\\nCLIP training complete!")
print(f"Final loss: {clip_losses[-1]:.4f}")""")
    print("  Cell 32 (training): Added cosine warmup scheduler")

    # =========================================================================
    # CELL 39: Update WGAN_CONFIG embed_dim
    # =========================================================================
    old_39 = ''.join(cells[39]['source'])
    new_39 = old_39.replace("'embed_dim': 64,", "'embed_dim': 128,")
    new_39 = new_39.replace("# Must match CLIP embed_dim", "# Must match CLIP embed_dim (updated from 64)")
    cells[39]['source'] = make_source(new_39)
    print("  Cell 39 (WGAN_CONFIG): embed_dim 64 → 128")

    # =========================================================================
    # CELL 108: Update markdown (R^64 → R^128)
    # =========================================================================
    old_108 = ''.join(cells[108]['source'])
    new_108 = old_108.replace('R^64', 'R^128')
    cells[108]['source'] = make_source(new_108)
    print("  Cell 108 (markdown): R^64 → R^128")

    # =========================================================================
    # CELL 111: Update comments (d=64 → d=128)
    # =========================================================================
    old_111 = ''.join(cells[111]['source'])
    new_111 = old_111.replace(
        'separable in R^64?',
        'separable in R^128?'
    ).replace(
        'So for d=64, std ≈ 0.125.',
        'So for d=128, std ≈ 0.088.'
    )
    cells[111]['source'] = make_source(new_111)
    print("  Cell 111 (Diag C): Updated dimension comments")

    # =========================================================================
    # DELETE CELLS 114–125 (Phase E sweep scaffolding)
    # =========================================================================
    deleted_titles = []
    for i in range(114, 126):
        src = ''.join(cells[i]['source'])[:80]
        deleted_titles.append(f"    Cell {i}: {src.strip()[:60]}...")

    del cells[114:126]
    print(f"  Cells 114–125: DELETED ({len(deleted_titles)} cells)")
    for t in deleted_titles:
        print(t)

    # =========================================================================
    # SAVE
    # =========================================================================
    # Backup first
    backup_path = NOTEBOOK_PATH + f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    shutil.copy2(NOTEBOOK_PATH, backup_path)
    print(f"\n  Backup saved: {backup_path}")

    nb['cells'] = cells
    with open(NOTEBOOK_PATH, 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"\n{'='*60}")
    print(f"INTEGRATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Cells: {len(cells)} (was 126)")
    print(f"  CLIP config: embed_dim=128, tau=0.15, lr=1e-3, epochs=50, cosine_warmup")
    print(f"  BrainEncoder: +LayerNorm after fc1, fc2")
    print(f"  Training: +cosine warmup scheduler")
    print(f"  WGAN: embed_dim=128")
    print(f"  Removed: 12 Phase E sweep cells")


if __name__ == '__main__':
    main()
