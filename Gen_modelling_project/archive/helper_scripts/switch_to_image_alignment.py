#!/usr/bin/env python3
"""
Switch pipeline from EEG-to-text to EEG-to-image (CORnet-S) alignment.

Modifies GZSL_EEG_Pipeline_v2.ipynb in-place (with backup):
1. Removes PCA truncation (cell 7) — uses full 1000-D CORnet-S features
2. Updates section header (cell 28) — Brain-Image Encoder
3. Replaces CLIP_CONFIG with ENCODER_CONFIG (cell 29)
4. Replaces TextProjector with ImageProjector (cell 30)
5. Replaces text data prep with image data prep (cell 31)
6. Updates training loop for image alignment (cell 32)
7. Updates loss curve plot (cell 33)
8. Updates embedding extraction for image projector (cell 34)
9. Updates prototype computation to use image embeddings (cell 35)
10. Updates caching (cell 36)
11. Updates t-SNE labels (cell 37)
12. Updates WGAN-GP references (cells 38, 47, 48)
13. Updates classifier references (cells 49, 61)
14. Deletes visual investigation cells (67-75)

Run from project root:
    python helper_files/switch_to_image_alignment.py
"""

import json
import copy
import os
import shutil
from datetime import datetime

NOTEBOOK = 'GZSL_EEG_Pipeline_v2.ipynb'


def make_source(text):
    """Convert multi-line string into notebook source format (list of lines)."""
    lines = text.split('\n')
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + '\n')
        else:
            result.append(line)
    return result


def replace_in_cell(cell, old, new):
    """Replace a string in a cell's source. Raises if not found."""
    src = ''.join(cell['source'])
    if old not in src:
        raise RuntimeError(f"Could not find anchor string in cell: {repr(old[:80])}")
    src = src.replace(old, new)
    cell['source'] = make_source(src)


def replace_cell_source(cell, new_source_text):
    """Replace entire cell source with new text."""
    cell['source'] = make_source(new_source_text)


# =============================================================================
# CELL MODIFICATION FUNCTIONS
# =============================================================================

def modify_cell_7(cell):
    """Remove PCA truncation — use full 1000-D CORnet-S features."""
    # Remove the two truncation lines
    replace_in_cell(cell, 'image_seen = image_seen[:,0:100]\n', '')
    replace_in_cell(cell, 'image_unseen = image_unseen[:, 0:100]\n', '')


def modify_cell_28(cell):
    """Update section header: Brain-Text → Brain-Image."""
    replace_cell_source(cell, """\
---

# Brain-Image Contrastive Encoder (CORnet-S)

This section implements a **contrastive encoder** that aligns EEG embeddings and CORnet-S visual feature embeddings in a shared semantic space.

**Paradigm shift**: Previous pipeline aligned EEG→text (CLIP text embeddings). Visual investigation revealed that CORnet-S image features have dramatically better class separation (inter-class cosine sim = -0.001 vs text's 0.668) and 2x higher effective rank (335-D vs 169-D at 95% variance). Switching to image alignment is the highest-impact intervention available.

**Architecture:**
- Brain Encoder `f_b`: 561 → 1024 → 512 → d (d=128, with LayerNorm)
- Image Projector `g_v`: 1000 → 512 → d (with LayerNorm)
- Both outputs are L2-normalised

**Training:**
- Loss: Symmetric InfoNCE contrastive loss
- Schedule: Cosine warmup (10% warmup, cosine decay to 0)
- Temperature: τ=0.15 (fixed)

**Training data:** Seen EEG trials only (no leakage from unseen)""")


def modify_cell_29(cell):
    """Replace CLIP_CONFIG with ENCODER_CONFIG."""
    replace_cell_source(cell, """\
# =============================================================================
# ENCODER CONFIGURATION
# =============================================================================
# All hyperparameters in one place for reproducibility
# Phase E optimised values retained; alignment target switched to image (CORnet-S)

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

# Configuration dictionary — image alignment
ENCODER_CONFIG = {
    'embed_dim': 128,              # Output embedding dimension d
    'image_input_dim': 1000,       # CORnet-S full PCA dimensionality
    'tau': 0.15,                   # Temperature for contrastive loss
    'epochs': 50,                  # Training epochs
    'batch_size': 256,             # Batch size
    'lr': 1e-3,                    # Learning rate
    'weight_decay': 1e-4,          # AdamW weight decay
    'dropout': 0.1,                # Dropout rate
    'schedule': 'cosine_warmup',   # LR schedule
    'warmup_ratio': 0.1,           # Warmup fraction of total steps
    'seed': 42,                    # Random seed
    'alignment_target': 'image (CORnet-S)',  # Alignment modality
}

# Set seeds for reproducibility
SEED = ENCODER_CONFIG['seed']
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 60)
print("ENCODER CONFIGURATION (Image Alignment — CORnet-S)")
print("=" * 60)
for k, v in ENCODER_CONFIG.items():
    print(f"  {k}: {v}")
print(f"  device: {device}")
print("=" * 60)""")


def modify_cell_30(cell):
    """Replace TextProjector with ImageProjector, rename clip_loss → contrastive_loss."""
    replace_cell_source(cell, """\
# =============================================================================
# MODEL DEFINITIONS
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


class ImageProjector(nn.Module):
    \"\"\"CORnet-S visual feature projector: 1000 -> d with LayerNorm + L2 normalisation.

    Design rationale:
    - 2-layer MLP (mirrors original TextProjector depth). CORnet-S PCA features
      are clean/structured — no need for BrainEncoder's 3-layer depth.
    - LayerNorm added because variance is concentrated in early PCA components
      (first 100 dims = 32.4% variance). LN stabilises gradient flow.
    - Dropout at 0.1 (same as BrainEncoder).
    - L2-normalised output (stays on unit sphere, matches BrainEncoder).
    \"\"\"

    def __init__(self, input_dim=1000, hidden_dim=512, embed_dim=128, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


def contrastive_loss(brain_embeds, image_embeds, tau=0.07):
    \"\"\"
    Symmetric InfoNCE contrastive loss.

    Args:
        brain_embeds: (B, d) L2-normalised brain embeddings
        image_embeds: (B, d) L2-normalised image embeddings
        tau: temperature parameter

    Returns:
        Scalar loss: (L_b2i + L_i2b) / 2
    \"\"\"
    logits = torch.matmul(brain_embeds, image_embeds.T) / tau  # (B, B)
    batch_size = brain_embeds.size(0)
    targets = torch.arange(batch_size, device=brain_embeds.device)
    loss_b2i = F.cross_entropy(logits, targets)
    loss_i2b = F.cross_entropy(logits.T, targets)
    return (loss_b2i + loss_i2b) / 2


# Instantiate models
brain_encoder = BrainEncoder(
    input_dim=561,
    embed_dim=ENCODER_CONFIG['embed_dim'],
    dropout=ENCODER_CONFIG['dropout']
).to(device)

image_projector = ImageProjector(
    input_dim=ENCODER_CONFIG['image_input_dim'],
    hidden_dim=512,
    embed_dim=ENCODER_CONFIG['embed_dim'],
    dropout=ENCODER_CONFIG['dropout']
).to(device)

print(f"Brain Encoder parameters: {sum(p.numel() for p in brain_encoder.parameters()):,}")
print(f"Image Projector parameters: {sum(p.numel() for p in image_projector.parameters()):,}")""")


def modify_cell_31(cell):
    """Replace text data prep with image data prep."""
    replace_cell_source(cell, """\
# =============================================================================
# ENCODER DATA PREPARATION
# =============================================================================
# Use only seen train split for encoder training (no leakage!)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Recreate the split indices (same seed as baseline)
indices_seen = np.arange(len(brain_seen))
train_idx, test_idx = train_test_split(
    indices_seen, test_size=0.2, random_state=42, stratify=label_seen.numpy().flatten()
)

# Get brain and image features for train/test
brain_train_seen = brain_seen[train_idx].numpy()
brain_test_seen_np = brain_seen[test_idx].numpy()
image_train_seen = image_seen[train_idx].numpy()
image_test_seen = image_seen[test_idx].numpy()
label_train_seen = label_seen[train_idx].numpy().flatten()
label_test_seen = label_seen[test_idx].numpy().flatten()

# Standardise brain features (fit on train only)
# NOTE: Image features are NOT standardised — they are already PCA-whitened
# (mean ~0, std ~0.06) and scaling is handled by the *50.0 factor in cell 7
clip_scaler = StandardScaler()
brain_train_seen_scaled = clip_scaler.fit_transform(brain_train_seen)
brain_test_seen_scaled = clip_scaler.transform(brain_test_seen_np)
brain_unseen_scaled = clip_scaler.transform(brain_unseen.numpy())

# Convert to tensors — brain
X_train_tensor = torch.FloatTensor(brain_train_seen_scaled)
X_test_tensor = torch.FloatTensor(brain_test_seen_scaled)
X_unseen_tensor = torch.FloatTensor(brain_unseen_scaled)

# Convert to tensors — image (CORnet-S, 1000-D)
I_train_tensor = torch.FloatTensor(image_train_seen)
I_test_tensor = torch.FloatTensor(image_test_seen)
I_unseen_tensor = torch.FloatTensor(image_unseen.numpy())

# Labels
Y_train_tensor = torch.LongTensor(label_train_seen)
Y_test_tensor = torch.LongTensor(label_test_seen)
Y_unseen_tensor = torch.LongTensor(label_unseen.numpy().flatten())

# Create DataLoader for training (brain + image pairs)
train_dataset = TensorDataset(X_train_tensor, I_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=ENCODER_CONFIG['batch_size'], shuffle=True)

print(f"Encoder Training data: {len(X_train_tensor)} samples")
print(f"  Brain features: {X_train_tensor.shape[1]}-D (StandardScaled)")
print(f"  Image features: {I_train_tensor.shape[1]}-D (CORnet-S, raw)")
print(f"Encoder Test data (seen): {len(X_test_tensor)} samples")
print(f"Unseen data (test only): {len(X_unseen_tensor)} samples")""")


def modify_cell_32(cell):
    """Update training loop: image_projector, contrastive_loss, ENCODER_CONFIG."""
    replace_cell_source(cell, """\
# =============================================================================
# CONTRASTIVE ENCODER TRAINING
# =============================================================================

# Optimizer — brain encoder + image projector (jointly trained)
encoder_params = list(brain_encoder.parameters()) + list(image_projector.parameters())
encoder_optimizer = torch.optim.AdamW(
    encoder_params,
    lr=ENCODER_CONFIG['lr'],
    weight_decay=ENCODER_CONFIG['weight_decay']
)

# Cosine warmup LR scheduler
total_steps = ENCODER_CONFIG['epochs'] * len(train_loader)
warmup_steps = int(ENCODER_CONFIG.get('warmup_ratio', 0.1) * total_steps)

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(encoder_optimizer, lr_lambda)

# Training loop
encoder_losses = []
print("Training Brain-Image contrastive encoder...")
print(f"Epochs: {ENCODER_CONFIG['epochs']}, Batches per epoch: {len(train_loader)}")
print(f"Schedule: cosine warmup ({warmup_steps} warmup steps / {total_steps} total)")
print(f"Alignment target: {ENCODER_CONFIG['alignment_target']}")

for epoch in range(ENCODER_CONFIG['epochs']):
    brain_encoder.train()
    image_projector.train()
    epoch_loss = 0.0

    for batch_idx, (brain_batch, image_batch, _) in enumerate(train_loader):
        brain_batch = brain_batch.to(device)
        image_batch = image_batch.to(device)

        # Forward pass
        brain_embeds = brain_encoder(brain_batch)
        image_embeds = image_projector(image_batch)

        # Compute loss
        loss = contrastive_loss(brain_embeds, image_embeds, tau=ENCODER_CONFIG['tau'])

        # Backward pass
        encoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    encoder_losses.append(avg_loss)
    current_lr = scheduler.get_last_lr()[0]

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:3d}/{ENCODER_CONFIG['epochs']}: Loss = {avg_loss:.4f}, LR = {current_lr:.6f}")

print(f"\\nBrain-Image encoder training complete!")
print(f"Final loss: {encoder_losses[-1]:.4f}")""")


def modify_cell_33(cell):
    """Update loss curve: encoder_losses, titles."""
    replace_cell_source(cell, """\
# =============================================================================
# ENCODER LOSS CURVE
# =============================================================================

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(encoder_losses)+1), encoder_losses, 'b-', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Contrastive Loss', fontsize=12)
plt.title('Brain-Image Encoder Training Loss (CORnet-S)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/encoder_loss_curve.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Saved: figures/encoder_loss_curve.png")""")


def modify_cell_34(cell):
    """Update embedding extraction: add image embeddings via image_projector."""
    replace_cell_source(cell, """\
# =============================================================================
# COMPUTE AND CACHE EMBEDDINGS
# =============================================================================

brain_encoder.eval()
image_projector.eval()

with torch.no_grad():
    # Brain embeddings (for classifier and WGAN-GP)
    E_train_seen = brain_encoder(X_train_tensor.to(device)).cpu().numpy()
    E_test_seen = brain_encoder(X_test_tensor.to(device)).cpu().numpy()
    E_unseen = brain_encoder(X_unseen_tensor.to(device)).cpu().numpy()

    # Image embeddings (for prototype computation)
    V_train_embeds = image_projector(I_train_tensor.to(device)).cpu().numpy()
    V_test_embeds = image_projector(I_test_tensor.to(device)).cpu().numpy()
    V_unseen_embeds = image_projector(I_unseen_tensor.to(device)).cpu().numpy()

print(f"Brain embeddings:")
print(f"  E_train_seen: {E_train_seen.shape}")
print(f"  E_test_seen: {E_test_seen.shape}")
print(f"  E_unseen: {E_unseen.shape}")
print(f"Image embeddings:")
print(f"  V_train_embeds: {V_train_embeds.shape}")
print(f"  V_unseen_embeds: {V_unseen_embeds.shape}")""")


def modify_cell_35(cell):
    """Update prototype computation: use image embeddings instead of text."""
    replace_in_cell(cell,
        '# COMPUTE SEMANTIC PROTOTYPES (CLASS CENTROIDS IN TEXT SPACE)',
        '# COMPUTE SEMANTIC PROTOTYPES (CLASS CENTROIDS IN IMAGE SPACE)')

    replace_in_cell(cell,
        '# Compute prototypes for seen classes (from training text embeddings)\n'
        'S_seen_prototypes = compute_prototypes(T_train_embeds, label_train_seen)',
        '# Compute prototypes for seen classes (from training image embeddings)\n'
        'S_seen_prototypes = compute_prototypes(V_train_embeds, label_train_seen)')

    replace_in_cell(cell,
        '# Compute prototypes for unseen classes (from unseen text embeddings)\n'
        'S_unseen_prototypes = compute_prototypes(T_unseen_embeds, Y_unseen_tensor.numpy())',
        '# Compute prototypes for unseen classes (from unseen image embeddings)\n'
        'S_unseen_prototypes = compute_prototypes(V_unseen_embeds, Y_unseen_tensor.numpy())')


def modify_cell_36(cell):
    """Update caching to also save image embeddings."""
    replace_cell_source(cell, """\
# =============================================================================
# CACHE EMBEDDINGS AND PROTOTYPES TO DISK
# =============================================================================

os.makedirs('cached_arrays', exist_ok=True)

# Save brain embeddings
np.save('cached_arrays/E_train_seen.npy', E_train_seen)
np.save('cached_arrays/E_test_seen.npy', E_test_seen)
np.save('cached_arrays/E_unseen.npy', E_unseen)

# Save image embeddings
np.save('cached_arrays/V_train_embeds.npy', V_train_embeds)
np.save('cached_arrays/V_unseen_embeds.npy', V_unseen_embeds)

# Save labels
np.save('cached_arrays/y_train_seen.npy', label_train_seen)
np.save('cached_arrays/y_test_seen.npy', label_test_seen)
np.save('cached_arrays/y_unseen.npy', Y_unseen_tensor.numpy())

# Save prototypes
np.save('cached_arrays/S_seen_prototypes.npy', S_seen_array)
np.save('cached_arrays/S_unseen_prototypes.npy', S_unseen_array)
np.save('cached_arrays/seen_classes.npy', np.array(seen_classes))
np.save('cached_arrays/unseen_classes.npy', np.array(unseen_classes))

print("Cached arrays saved to cached_arrays/:")
for f in sorted(os.listdir('cached_arrays')):
    print(f"  - {f}")""")


def modify_cell_37(cell):
    """Update t-SNE title: Text → Image prototypes."""
    replace_in_cell(cell,
        "plt.title('CLIP Embedding Space: EEG Embeddings and Text Prototypes', fontsize=14)",
        "plt.title('Embedding Space: EEG Embeddings and Image Prototypes (CORnet-S)', fontsize=14)")


def modify_cell_38(cell):
    """Update WGAN-GP markdown: text prototypes → image prototypes."""
    replace_in_cell(cell,
        'conditioned on text prototypes `s_c`',
        'conditioned on image prototypes `s_c`')
    replace_in_cell(cell,
        'from CLIP encoder',
        'from Brain-Image encoder')


def modify_cell_47(cell):
    """Update real vs synth t-SNE title."""
    replace_in_cell(cell,
        "plt.title('Real Seen vs Synthetic Unseen Embeddings in CLIP Space', fontsize=14)",
        "plt.title('Real Seen vs Synthetic Unseen Embeddings in Brain-Image Space', fontsize=14)")


def modify_cell_48(cell):
    """Update implementation summary references."""
    replace_in_cell(cell, 'CLIP + cWGAN-GP IMPLEMENTATION SUMMARY', 'BRAIN-IMAGE ENCODER + cWGAN-GP IMPLEMENTATION SUMMARY')
    replace_in_cell(cell, '### CLIP ENCODER ###', '### BRAIN-IMAGE ENCODER ###')
    replace_in_cell(cell, "CLIP_CONFIG['embed_dim']", "ENCODER_CONFIG['embed_dim']")
    replace_in_cell(cell, "CLIP_CONFIG['epochs']", "ENCODER_CONFIG['epochs']")
    replace_in_cell(cell, "clip_losses[-1]", "encoder_losses[-1]")
    replace_in_cell(cell, 'figures/clip_loss_curve.png', 'figures/encoder_loss_curve.png')


def modify_cell_49(cell):
    """Update GZSL classifier markdown: CLIP → Brain-Image."""
    replace_in_cell(cell,
        '`E_train_seen` (from CLIP encoder)',
        '`E_train_seen` (from Brain-Image encoder)')


def modify_cell_61(cell):
    """Update final summary: Brain-Text → Brain-Image."""
    replace_in_cell(cell,
        'Brain-Text CLIP encoder: Maps EEG',
        'Brain-Image encoder (CORnet-S): Maps EEG')
    replace_in_cell(cell,
        'from text prototypes',
        'from image prototypes')
    replace_in_cell(cell,
        'Using CLIP to create a shared EEG-text semantic space',
        'Using contrastive learning to create a shared EEG-image semantic space')


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"{'=' * 60}")
    print(f"SWITCH TO IMAGE ALIGNMENT (CORnet-S)")
    print(f"{'=' * 60}")
    print(f"Notebook: {NOTEBOOK}")
    print()

    # 1. Load notebook
    with open(NOTEBOOK, 'r') as f:
        nb = json.load(f)

    n_cells_before = len(nb['cells'])
    print(f"Loaded {n_cells_before} cells.")

    # 2. Backup
    backup_name = NOTEBOOK + f'.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    shutil.copy2(NOTEBOOK, backup_name)
    print(f"Backup saved: {backup_name}")

    # 3. Apply cell modifications
    modifications = [
        (7, modify_cell_7, "Remove PCA truncation (full 1000-D)"),
        (28, modify_cell_28, "Update section header → Brain-Image"),
        (29, modify_cell_29, "CLIP_CONFIG → ENCODER_CONFIG"),
        (30, modify_cell_30, "TextProjector → ImageProjector"),
        (31, modify_cell_31, "Text data prep → Image data prep"),
        (32, modify_cell_32, "Training loop → image alignment"),
        (33, modify_cell_33, "Loss curve → encoder_losses"),
        (34, modify_cell_34, "Embeddings → add image embeddings"),
        (35, modify_cell_35, "Prototypes → image-based"),
        (36, modify_cell_36, "Caching → include image embeddings"),
        (37, modify_cell_37, "t-SNE → Image Prototypes label"),
        (38, modify_cell_38, "WGAN markdown → image prototypes"),
        (47, modify_cell_47, "WGAN t-SNE → Brain-Image Space"),
        (48, modify_cell_48, "Summary → Brain-Image references"),
        (49, modify_cell_49, "Classifier markdown → Brain-Image"),
        (61, modify_cell_61, "Final summary → Brain-Image"),
    ]

    print(f"\nApplying {len(modifications)} cell modifications...")
    for cell_idx, mod_func, description in modifications:
        try:
            mod_func(nb['cells'][cell_idx])
            print(f"  ✓ Cell {cell_idx:2d}: {description}")
        except Exception as e:
            print(f"  ✗ Cell {cell_idx:2d}: {description} — FAILED: {e}")
            raise

    # 4. Delete visual investigation cells (67-75)
    # These are the last 9 cells (indices 67 through 75 inclusive)
    cells_to_delete = list(range(67, 76))
    print(f"\nDeleting {len(cells_to_delete)} visual investigation cells ({cells_to_delete[0]}-{cells_to_delete[-1]})...")
    # Delete from the end to preserve indices
    for idx in reversed(cells_to_delete):
        del nb['cells'][idx]

    n_cells_after = len(nb['cells'])
    print(f"  Cells: {n_cells_before} → {n_cells_after} (deleted {n_cells_before - n_cells_after})")

    # 5. Clear all outputs and execution counts
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None

    # 6. Write modified notebook
    with open(NOTEBOOK, 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"\n{'=' * 60}")
    print(f"DONE — Image alignment switch complete")
    print(f"{'=' * 60}")
    print(f"  Notebook: {NOTEBOOK}")
    print(f"  Cells: {n_cells_after}")
    print(f"  Backup: {backup_name}")

    # 7. Verification summary
    print(f"\n--- Cell Structure ---")
    sections = [
        ("Setup", 0, 6),
        ("Data Loading (full 1000-D, label fix)", 7, 9),
        ("Data Exploration", 10, 16),
        ("Baseline [A]", 17, 22),
        ("GZSL Baseline", 23, 27),
        ("Brain-Image Encoder (CORnet-S)", 28, 37),
        ("cWGAN-GP", 38, 48),
        ("GZSL Classifier [A+B] (with sample balance)", 49, 61),
        ("Evaluation Harness", 62, 66),
    ]
    for name, start, end in sections:
        print(f"  Cells {start:2d}-{end:2d}: {name}")

    # 8. Sanity checks
    print(f"\n--- Sanity Checks ---")
    full_source = ''.join(''.join(c['source']) for c in nb['cells'])

    checks = [
        ("ImageProjector defined", "class ImageProjector" in full_source),
        ("contrastive_loss defined", "def contrastive_loss" in full_source),
        ("ENCODER_CONFIG defined", "ENCODER_CONFIG = {" in full_source),
        ("No TextProjector in code", "class TextProjector" not in full_source),
        ("No clip_loss function", "def clip_loss" not in full_source),
        ("No CLIP_CONFIG", "CLIP_CONFIG = {" not in full_source),
        ("No PCA truncation", "[:,0:100]" not in full_source and "[:, 0:100]" not in full_source),
        ("I_train_tensor present", "I_train_tensor" in full_source),
        ("V_train_embeds present", "V_train_embeds" in full_source),
        ("encoder_losses present", "encoder_losses" in full_source),
        ("Cell count is 67", n_cells_after == 67),
    ]

    all_passed = True
    for check_name, result in checks:
        status = "✓" if result else "✗"
        print(f"  {status} {check_name}")
        if not result:
            all_passed = False

    if all_passed:
        print(f"\n  All sanity checks passed!")
    else:
        print(f"\n  WARNING: Some sanity checks failed!")


if __name__ == '__main__':
    main()
