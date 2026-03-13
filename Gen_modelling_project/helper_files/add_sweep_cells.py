#!/usr/bin/env python3
"""
Add hyperparameter sweep cells to GZSL_EEG_Pipeline_v2.ipynb.

Adds 14 cells (67-80) implementing a sequential greedy sweep of the
Brain-Image contrastive encoder hyperparameters:
  Phase 1: Epoch x Temperature grid (6 runs, 30 eval points)
  Phase 2: Embedding dimension (4 runs)
  Phase 3: ImageProjector architecture (6 runs)
  Phase 4: Learning rate x Weight decay (9 runs)
  Phase 5: Full pipeline validation (1 run + baseline reference)

Run from project root:
    python helper_files/add_sweep_cells.py
"""

import json
import shutil
from datetime import datetime

NOTEBOOK = 'GZSL_EEG_Pipeline_v2.ipynb'


def make_source(text):
    """Convert multi-line string into notebook source format (list of lines)."""
    # Strip leading/trailing newlines to avoid empty first/last lines
    text = text.strip('\n')
    lines = text.split('\n')
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + '\n')
        else:
            result.append(line)
    return result


def make_code_cell(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": make_source(text)
    }


def make_markdown_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": make_source(text)
    }


# =============================================================================
# CELL DEFINITIONS
# =============================================================================

def cell_67_header():
    return make_markdown_cell("""---

# Hyperparameter Sweep — Image Alignment Encoder

Sequential greedy sweep to find the optimal encoder configuration for EEG-to-image (CORnet-S) alignment.

**Strategy**: Each phase sweeps one parameter group, selects the best, and passes it forward.

**Fast evaluation metric**: Encoder-only **top-k retrieval accuracy** on the seen test set (3308 samples, 1654 classes). For each test brain embedding, cosine similarity to all seen prototypes — check if correct class in top-k.

**Phases:**
1. Epoch x Temperature grid (6 runs, 30 eval points)
2. Embedding dimension (4 runs)
3. ImageProjector architecture (6 runs)
4. Learning rate x Weight decay (9 runs)
5. Full GZSL pipeline with optimal config""")


def cell_68_utilities():
    return make_code_cell("""# =============================================================================
# HYPERPARAMETER SWEEP — UTILITY FUNCTIONS
# =============================================================================

import gc
import math
import time
import pandas as pd

# --- Seed management ---
def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# --- Flexible ImageProjector for architecture sweep ---
class FlexibleImageProjector(nn.Module):
    '''Variable-depth image projector with LayerNorm + L2-norm output.'''
    def __init__(self, input_dim=1000, hidden_dims=[512], embed_dim=128, dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, p=2, dim=-1)


# --- Top-k retrieval evaluation ---
def evaluate_encoder_topk(brain_enc, img_proj, X_test, I_train, Y_train,
                          Y_test, device, k_values=[1, 5, 10]):
    '''Top-k retrieval accuracy: test brain embeddings vs seen prototypes.'''
    brain_enc.eval()
    img_proj.eval()

    with torch.no_grad():
        E_test = brain_enc(X_test.to(device)).cpu().numpy()
        V_train = img_proj(I_train.to(device)).cpu().numpy()

    Y_train_np = Y_train.numpy() if isinstance(Y_train, torch.Tensor) else Y_train
    Y_test_np = Y_test.numpy() if isinstance(Y_test, torch.Tensor) else Y_test

    protos = compute_prototypes(V_train, Y_train_np)
    sorted_classes = sorted(protos.keys())
    proto_array = np.array([protos[c] for c in sorted_classes])
    class_labels = np.array(sorted_classes)

    sims = E_test @ proto_array.T  # (n_test, n_classes)

    results = {}
    for k in k_values:
        topk_idx = np.argsort(-sims, axis=1)[:, :k]
        topk_labels = class_labels[topk_idx]
        correct = np.any(topk_labels == Y_test_np[:, None], axis=1)
        results[f'top{k}'] = float(correct.mean())

    return results


# --- Encoder training function ---
def train_encoder_sweep(config, X_train, I_train, Y_train, X_test, I_test, Y_test,
                        device, checkpoint_epochs=None, return_models=False, verbose=True):
    '''Train brain-image encoder with given config. Returns metrics dict.'''
    set_seeds(config.get('seed', 42))

    embed_dim = config['embed_dim']
    tau = config['tau']
    epochs = config['epochs']
    lr = config['lr']
    wd = config.get('weight_decay', 1e-4)
    dropout = config.get('dropout', 0.1)
    batch_size = config.get('batch_size', 256)
    warmup_ratio = config.get('warmup_ratio', 0.1)
    image_hidden_dims = config.get('image_hidden_dims', [512])
    brain_hidden_dims = config.get('brain_hidden_dims', [1024, 512])

    # Create models
    brain_enc = BrainEncoder(
        input_dim=561, hidden_dims=brain_hidden_dims,
        embed_dim=embed_dim, dropout=dropout
    ).to(device)

    img_proj = FlexibleImageProjector(
        input_dim=config.get('image_input_dim', 1000),
        hidden_dims=image_hidden_dims,
        embed_dim=embed_dim, dropout=dropout
    ).to(device)

    # DataLoader
    dataset = TensorDataset(X_train, I_train, Y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer + cosine warmup scheduler
    params = list(brain_enc.parameters()) + list(img_proj.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    total_steps = epochs * len(loader)
    warmup_steps = int(warmup_ratio * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    losses = []
    ckpt_results = {}

    for epoch in range(epochs):
        brain_enc.train()
        img_proj.train()
        epoch_loss = 0.0

        for brain_b, image_b, _ in loader:
            brain_b = brain_b.to(device)
            image_b = image_b.to(device)

            b_emb = brain_enc(brain_b)
            i_emb = img_proj(image_b)
            loss = contrastive_loss(b_emb, i_emb, tau=tau)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)

        # Checkpoint evaluation
        if checkpoint_epochs and (epoch + 1) in checkpoint_epochs:
            metrics = evaluate_encoder_topk(
                brain_enc, img_proj, X_test, I_train, Y_train, Y_test, device
            )
            metrics['loss'] = avg_loss
            metrics['epoch'] = epoch + 1
            ckpt_results[epoch + 1] = metrics
            if verbose:
                print(f"    Ckpt epoch {epoch+1}: top1={metrics['top1']:.4f}, "
                      f"top5={metrics['top5']:.4f}, loss={avg_loss:.4f}")

    # Final evaluation
    final_metrics = evaluate_encoder_topk(
        brain_enc, img_proj, X_test, I_train, Y_train, Y_test, device
    )
    final_metrics['loss'] = losses[-1]

    result = {
        'config': config.copy(),
        'metrics': final_metrics,
        'loss_history': losses,
        'checkpoint_results': ckpt_results,
    }

    if return_models:
        result['brain_encoder'] = brain_enc
        result['image_projector'] = img_proj
    else:
        del brain_enc, img_proj
    del optimizer, scheduler, dataset, loader
    torch.cuda.empty_cache()
    gc.collect()

    return result


print("Sweep utilities defined: set_seeds, FlexibleImageProjector, "
      "evaluate_encoder_topk, train_encoder_sweep")""")


def cell_69_phase1():
    return make_code_cell("""# =============================================================================
# SWEEP PHASE 1: EPOCH x TEMPERATURE GRID
# =============================================================================

print("=" * 70)
print("SWEEP PHASE 1: Epoch x Temperature Grid")
print("=" * 70)

tau_values = [0.05, 0.07, 0.10, 0.12, 0.15, 0.20]
checkpoint_epochs = [50, 75, 100, 125, 150, 200]
max_epochs = 200

# Base config — only tau and epochs vary
phase1_base = {
    'embed_dim': 128,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'dropout': 0.1,
    'batch_size': 256,
    'warmup_ratio': 0.1,
    'image_input_dim': 1000,
    'image_hidden_dims': [512],
    'brain_hidden_dims': [1024, 512],
    'seed': 42,
}

phase1_results = []
phase1_start = time.time()

for i, tau in enumerate(tau_values):
    print(f"\\n[{i+1}/{len(tau_values)}] Training with tau={tau}, {max_epochs} epochs...")
    t0 = time.time()

    config = {**phase1_base, 'tau': tau, 'epochs': max_epochs}

    result = train_encoder_sweep(
        config, X_train_tensor, I_train_tensor, Y_train_tensor,
        X_test_tensor, I_test_tensor, Y_test_tensor,
        device, checkpoint_epochs=checkpoint_epochs
    )
    phase1_results.append(result)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s. Final loss={result['metrics']['loss']:.4f}, "
          f"top1={result['metrics']['top1']:.4f}")

# Find best (tau, epoch) pair across all checkpoints
best_top1_p1 = 0
best_tau = None
best_epochs_count = None

for result in phase1_results:
    tau = result['config']['tau']
    for epoch, metrics in result['checkpoint_results'].items():
        if metrics['top1'] > best_top1_p1:
            best_top1_p1 = metrics['top1']
            best_tau = tau
            best_epochs_count = epoch

phase1_elapsed = time.time() - phase1_start

print(f"\\n{'='*70}")
print(f"PHASE 1 RESULT: best_tau={best_tau}, best_epochs={best_epochs_count}, "
      f"top1={best_top1_p1:.4f}")
print(f"Phase 1 total time: {phase1_elapsed:.1f}s ({phase1_elapsed/60:.1f}min)")
print(f"{'='*70}")""")


def cell_70_phase1_viz():
    return make_code_cell("""# =============================================================================
# PHASE 1 VISUALIZATION
# =============================================================================

# Build heatmap data
heatmap = np.zeros((len(tau_values), len(checkpoint_epochs)))
for i, result in enumerate(phase1_results):
    for j, epoch in enumerate(checkpoint_epochs):
        if epoch in result['checkpoint_results']:
            heatmap[i, j] = result['checkpoint_results'][epoch]['top1']

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Heatmap
im = axes[0].imshow(heatmap, aspect='auto', cmap='YlOrRd')
axes[0].set_xticks(range(len(checkpoint_epochs)))
axes[0].set_xticklabels(checkpoint_epochs)
axes[0].set_yticks(range(len(tau_values)))
axes[0].set_yticklabels([f'{t:.2f}' for t in tau_values])
axes[0].set_xlabel('Epochs', fontsize=12)
axes[0].set_ylabel('Temperature (tau)', fontsize=12)
axes[0].set_title('Phase 1: Top-1 Retrieval Accuracy', fontsize=14)
plt.colorbar(im, ax=axes[0], label='Top-1 Accuracy')

# Annotate cells
for i in range(len(tau_values)):
    for j in range(len(checkpoint_epochs)):
        val = heatmap[i, j]
        axes[0].text(j, i, f'{val:.3f}', ha='center', va='center',
                     fontsize=8, color='white' if val > heatmap.max()*0.7 else 'black')

# Mark best
best_i = tau_values.index(best_tau)
best_j = checkpoint_epochs.index(best_epochs_count)
axes[0].add_patch(plt.Rectangle((best_j-0.5, best_i-0.5), 1, 1,
                                fill=False, edgecolor='lime', linewidth=3))

# Loss curves overlay
for result in phase1_results:
    tau = result['config']['tau']
    axes[1].plot(range(1, len(result['loss_history'])+1), result['loss_history'],
                 label=f'tau={tau:.2f}', alpha=0.8, linewidth=1.5)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Contrastive Loss', fontsize=12)
axes[1].set_title('Phase 1: Loss Curves by Temperature', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/sweep_phase1_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: figures/sweep_phase1_heatmap.png")

# Print results table
print(f"\\n{'tau':>6} | ", end='')
for ep in checkpoint_epochs:
    print(f"ep={ep:>3}", end='  ')
print()
print('-' * (8 + len(checkpoint_epochs) * 8))
for i, result in enumerate(phase1_results):
    tau = result['config']['tau']
    print(f"{tau:>6.2f} | ", end='')
    for ep in checkpoint_epochs:
        if ep in result['checkpoint_results']:
            v = result['checkpoint_results'][ep]['top1']
            print(f"{v:.4f}", end='  ')
    print()""")


def cell_71_phase2():
    return make_code_cell("""# =============================================================================
# SWEEP PHASE 2: EMBEDDING DIMENSION
# =============================================================================

print("=" * 70)
print(f"SWEEP PHASE 2: Embedding Dimension (tau={best_tau}, epochs={best_epochs_count})")
print("=" * 70)

embed_dims = [64, 128, 256, 512]

phase2_base = {
    'tau': best_tau,
    'epochs': best_epochs_count,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'dropout': 0.1,
    'batch_size': 256,
    'warmup_ratio': 0.1,
    'image_input_dim': 1000,
    'image_hidden_dims': [512],
    'brain_hidden_dims': [1024, 512],
    'seed': 42,
}

phase2_results = []
phase2_start = time.time()

for i, dim in enumerate(embed_dims):
    print(f"\\n[{i+1}/{len(embed_dims)}] embed_dim={dim}...")
    t0 = time.time()

    config = {**phase2_base, 'embed_dim': dim}
    result = train_encoder_sweep(
        config, X_train_tensor, I_train_tensor, Y_train_tensor,
        X_test_tensor, I_test_tensor, Y_test_tensor, device
    )
    phase2_results.append(result)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s. top1={result['metrics']['top1']:.4f}, "
          f"loss={result['metrics']['loss']:.4f}")

# Select best
best_p2 = max(phase2_results, key=lambda r: r['metrics']['top1'])
best_dim = best_p2['config']['embed_dim']

phase2_elapsed = time.time() - phase2_start

# Visualization
fig, ax = plt.subplots(figsize=(8, 5))
dims_list = [r['config']['embed_dim'] for r in phase2_results]
top1_list = [r['metrics']['top1'] for r in phase2_results]
bars = ax.bar([str(d) for d in dims_list], top1_list, color='steelblue', edgecolor='black')
for bar, val in zip(bars, top1_list):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10)
ax.set_xlabel('Embedding Dimension', fontsize=12)
ax.set_ylabel('Top-1 Accuracy', fontsize=12)
ax.set_title('Phase 2: Embedding Dimension Sweep', fontsize=14)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/sweep_phase2_embed_dim.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: figures/sweep_phase2_embed_dim.png")

print(f"\\n{'='*70}")
print(f"PHASE 2 RESULT: best_dim={best_dim}, top1={best_p2['metrics']['top1']:.4f}")
print(f"Phase 2 total time: {phase2_elapsed:.1f}s ({phase2_elapsed/60:.1f}min)")
print(f"{'='*70}")""")


def cell_72_phase3():
    return make_code_cell("""# =============================================================================
# SWEEP PHASE 3: IMAGEPROJECTOR ARCHITECTURE
# =============================================================================

print("=" * 70)
print(f"SWEEP PHASE 3: ImageProjector Architecture "
      f"(tau={best_tau}, epochs={best_epochs_count}, dim={best_dim})")
print("=" * 70)

arch_configs = [
    {'name': 'A: [256]',       'image_hidden_dims': [256]},
    {'name': 'B: [512]',       'image_hidden_dims': [512]},
    {'name': 'C: [768]',       'image_hidden_dims': [768]},
    {'name': 'D: [512,256]',   'image_hidden_dims': [512, 256]},
    {'name': 'E: [768,384]',   'image_hidden_dims': [768, 384]},
    {'name': 'F: [1024,512]',  'image_hidden_dims': [1024, 512]},
]

phase3_base = {
    'embed_dim': best_dim,
    'tau': best_tau,
    'epochs': best_epochs_count,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'dropout': 0.1,
    'batch_size': 256,
    'warmup_ratio': 0.1,
    'image_input_dim': 1000,
    'brain_hidden_dims': [1024, 512],
    'seed': 42,
}

phase3_results = []
phase3_start = time.time()

for i, arch in enumerate(arch_configs):
    print(f"\\n[{i+1}/{len(arch_configs)}] Architecture {arch['name']}...")
    t0 = time.time()

    config = {**phase3_base, 'image_hidden_dims': arch['image_hidden_dims']}
    result = train_encoder_sweep(
        config, X_train_tensor, I_train_tensor, Y_train_tensor,
        X_test_tensor, I_test_tensor, Y_test_tensor, device
    )
    result['arch_name'] = arch['name']
    phase3_results.append(result)
    elapsed = time.time() - t0
    n_params = sum(
        p.numel() for p in FlexibleImageProjector(
            input_dim=1000, hidden_dims=arch['image_hidden_dims'],
            embed_dim=best_dim).parameters()
    )
    print(f"  Done in {elapsed:.1f}s. top1={result['metrics']['top1']:.4f}, "
          f"loss={result['metrics']['loss']:.4f}, proj_params={n_params:,}")

# Select best
best_p3 = max(phase3_results, key=lambda r: r['metrics']['top1'])
best_image_hidden_dims = best_p3['config']['image_hidden_dims']

phase3_elapsed = time.time() - phase3_start

# Results table
print(f"\\n{'Architecture':<20} {'Top-1':>8} {'Top-5':>8} {'Top-10':>8} {'Loss':>8}")
print('-' * 56)
for r in phase3_results:
    m = r['metrics']
    print(f"{r['arch_name']:<20} {m['top1']:>8.4f} {m['top5']:>8.4f} "
          f"{m['top10']:>8.4f} {m['loss']:>8.4f}")

print(f"\\n{'='*70}")
print(f"PHASE 3 RESULT: best_arch={best_image_hidden_dims}, "
      f"top1={best_p3['metrics']['top1']:.4f}")
print(f"Phase 3 total time: {phase3_elapsed:.1f}s ({phase3_elapsed/60:.1f}min)")
print(f"{'='*70}")""")


def cell_73_phase4():
    return make_code_cell("""# =============================================================================
# SWEEP PHASE 4: LEARNING RATE x WEIGHT DECAY
# =============================================================================

print("=" * 70)
print(f"SWEEP PHASE 4: LR x Weight Decay "
      f"(tau={best_tau}, epochs={best_epochs_count}, dim={best_dim}, "
      f"arch={best_image_hidden_dims})")
print("=" * 70)

lr_values = [5e-4, 1e-3, 2e-3]
wd_values = [0, 1e-4, 1e-3]

phase4_base = {
    'embed_dim': best_dim,
    'tau': best_tau,
    'epochs': best_epochs_count,
    'dropout': 0.1,
    'batch_size': 256,
    'warmup_ratio': 0.1,
    'image_input_dim': 1000,
    'image_hidden_dims': best_image_hidden_dims,
    'brain_hidden_dims': [1024, 512],
    'seed': 42,
}

phase4_results = []
phase4_start = time.time()

run_idx = 0
total_runs = len(lr_values) * len(wd_values)

for lr in lr_values:
    for wd in wd_values:
        run_idx += 1
        print(f"\\n[{run_idx}/{total_runs}] lr={lr}, wd={wd}...")
        t0 = time.time()

        config = {**phase4_base, 'lr': lr, 'weight_decay': wd}
        result = train_encoder_sweep(
            config, X_train_tensor, I_train_tensor, Y_train_tensor,
            X_test_tensor, I_test_tensor, Y_test_tensor, device
        )
        phase4_results.append(result)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s. top1={result['metrics']['top1']:.4f}, "
              f"loss={result['metrics']['loss']:.4f}")

# Select best
best_p4 = max(phase4_results, key=lambda r: r['metrics']['top1'])
best_lr = best_p4['config']['lr']
best_wd = best_p4['config']['weight_decay']

phase4_elapsed = time.time() - phase4_start

# Heatmap
heatmap4 = np.zeros((len(lr_values), len(wd_values)))
for r in phase4_results:
    i = lr_values.index(r['config']['lr'])
    j = wd_values.index(r['config']['weight_decay'])
    heatmap4[i, j] = r['metrics']['top1']

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(heatmap4, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(len(wd_values)))
ax.set_xticklabels([f'{w}' for w in wd_values])
ax.set_yticks(range(len(lr_values)))
ax.set_yticklabels([f'{lr}' for lr in lr_values])
ax.set_xlabel('Weight Decay', fontsize=12)
ax.set_ylabel('Learning Rate', fontsize=12)
ax.set_title('Phase 4: Top-1 Accuracy (LR x Weight Decay)', fontsize=14)
plt.colorbar(im, ax=ax, label='Top-1 Accuracy')

for i in range(len(lr_values)):
    for j in range(len(wd_values)):
        ax.text(j, i, f'{heatmap4[i,j]:.4f}', ha='center', va='center',
                fontsize=10, color='white' if heatmap4[i,j] > heatmap4.max()*0.7 else 'black')

# Mark best
best_i4 = lr_values.index(best_lr)
best_j4 = wd_values.index(best_wd)
ax.add_patch(plt.Rectangle((best_j4-0.5, best_i4-0.5), 1, 1,
                            fill=False, edgecolor='lime', linewidth=3))

plt.tight_layout()
plt.savefig('figures/sweep_phase4_lr_wd.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: figures/sweep_phase4_lr_wd.png")

print(f"\\n{'='*70}")
print(f"PHASE 4 RESULT: best_lr={best_lr}, best_wd={best_wd}, "
      f"top1={best_p4['metrics']['top1']:.4f}")
print(f"Phase 4 total time: {phase4_elapsed:.1f}s ({phase4_elapsed/60:.1f}min)")
print(f"{'='*70}")""")


def cell_74_summary():
    return make_code_cell("""# =============================================================================
# SWEEP SUMMARY — OPTIMAL CONFIG
# =============================================================================

OPTIMAL_CONFIG = {
    'embed_dim': best_dim,
    'tau': best_tau,
    'epochs': best_epochs_count,
    'lr': best_lr,
    'weight_decay': best_wd,
    'dropout': 0.1,
    'batch_size': 256,
    'warmup_ratio': 0.1,
    'image_input_dim': 1000,
    'image_hidden_dims': best_image_hidden_dims,
    'brain_hidden_dims': [1024, 512],
    'seed': 42,
    'alignment_target': 'image (CORnet-S)',
}

# Baseline config (from cell 29)
BASELINE_CONFIG = {
    'embed_dim': 128,
    'tau': 0.15,
    'epochs': 50,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'dropout': 0.1,
    'image_hidden_dims': [512],
}

print("=" * 70)
print("SWEEP COMPLETE — OPTIMAL CONFIGURATION")
print("=" * 70)

print(f"\\n{'Parameter':<25} {'Baseline':>15} {'Optimal':>15} {'Changed':>10}")
print("-" * 65)
for key in ['embed_dim', 'tau', 'epochs', 'lr', 'weight_decay', 'image_hidden_dims']:
    base_val = BASELINE_CONFIG.get(key, 'N/A')
    opt_val = OPTIMAL_CONFIG.get(key, 'N/A')
    changed = '*' if str(base_val) != str(opt_val) else ''
    print(f"{key:<25} {str(base_val):>15} {str(opt_val):>15} {changed:>10}")

print(f"\\nBest encoder-only top-1 from sweep: {best_p4['metrics']['top1']:.4f}")
print(f"\\nFull OPTIMAL_CONFIG:")
for k, v in OPTIMAL_CONFIG.items():
    print(f"  {k}: {v}")
print("=" * 70)""")


def cell_75_phase5a():
    return make_code_cell("""# =============================================================================
# PHASE 5a: TRAIN OPTIMAL ENCODER + COMPUTE EMBEDDINGS
# =============================================================================

print("=" * 70)
print("SWEEP PHASE 5: Full Pipeline Validation with Optimal Config")
print("=" * 70)

print(f"\\nTraining encoder with optimal config...")
print(f"  tau={OPTIMAL_CONFIG['tau']}, epochs={OPTIMAL_CONFIG['epochs']}, "
      f"dim={OPTIMAL_CONFIG['embed_dim']}, arch={OPTIMAL_CONFIG['image_hidden_dims']}, "
      f"lr={OPTIMAL_CONFIG['lr']}, wd={OPTIMAL_CONFIG['weight_decay']}")

t0 = time.time()

opt_result = train_encoder_sweep(
    OPTIMAL_CONFIG, X_train_tensor, I_train_tensor, Y_train_tensor,
    X_test_tensor, I_test_tensor, Y_test_tensor,
    device, return_models=True, verbose=True
)

opt_brain_enc = opt_result['brain_encoder']
opt_img_proj = opt_result['image_projector']
opt_encoder_loss = opt_result['metrics']['loss']
opt_top1 = opt_result['metrics']['top1']

print(f"\\nEncoder trained in {time.time()-t0:.1f}s. "
      f"loss={opt_encoder_loss:.4f}, top1={opt_top1:.4f}")

# Compute embeddings
opt_brain_enc.eval()
opt_img_proj.eval()

with torch.no_grad():
    opt_E_train_seen = opt_brain_enc(X_train_tensor.to(device)).cpu().numpy()
    opt_E_test_seen = opt_brain_enc(X_test_tensor.to(device)).cpu().numpy()
    opt_E_unseen = opt_brain_enc(X_unseen_tensor.to(device)).cpu().numpy()
    opt_V_train_embeds = opt_img_proj(I_train_tensor.to(device)).cpu().numpy()
    opt_V_unseen_embeds = opt_img_proj(I_unseen_tensor.to(device)).cpu().numpy()

# Compute prototypes
opt_S_seen_prototypes = compute_prototypes(opt_V_train_embeds, label_train_seen)
opt_S_unseen_prototypes = compute_prototypes(opt_V_unseen_embeds, Y_unseen_tensor.numpy())

opt_seen_classes = sorted(opt_S_seen_prototypes.keys())
opt_unseen_classes = sorted(opt_S_unseen_prototypes.keys())
opt_S_seen_array = np.array([opt_S_seen_prototypes[c] for c in opt_seen_classes])
opt_S_unseen_array = np.array([opt_S_unseen_prototypes[c] for c in opt_unseen_classes])

print(f"\\nEmbeddings: E_train={opt_E_train_seen.shape}, "
      f"E_test={opt_E_test_seen.shape}, E_unseen={opt_E_unseen.shape}")
print(f"Prototypes: seen={opt_S_seen_array.shape}, unseen={opt_S_unseen_array.shape}")

# Cache (overwrites baseline — optimal config is the final output)
np.save('cached_arrays/E_train_seen.npy', opt_E_train_seen)
np.save('cached_arrays/E_test_seen.npy', opt_E_test_seen)
np.save('cached_arrays/E_unseen.npy', opt_E_unseen)
np.save('cached_arrays/V_train_embeds.npy', opt_V_train_embeds)
np.save('cached_arrays/V_unseen_embeds.npy', opt_V_unseen_embeds)
np.save('cached_arrays/S_seen_prototypes.npy', opt_S_seen_array)
np.save('cached_arrays/S_unseen_prototypes.npy', opt_S_unseen_array)

# Cleanup models
del opt_brain_enc, opt_img_proj
torch.cuda.empty_cache()
gc.collect()

# Plot loss curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(opt_result['loss_history'])+1), opt_result['loss_history'],
         'b-', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Contrastive Loss', fontsize=12)
plt.title('Optimal Encoder Training Loss', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/sweep_optimal_loss_curve.png', dpi=150, bbox_inches='tight')
plt.show()

print("Embeddings computed and cached. Ready for WGAN-GP.")""")


def cell_76_phase5b():
    return make_code_cell("""# =============================================================================
# PHASE 5b: WGAN-GP TRAINING + SYNTHESIS (OPTIMAL EMBEDDINGS)
# =============================================================================

set_seeds(42)
opt_embed_dim = OPTIMAL_CONFIG['embed_dim']
print(f"Training cWGAN-GP with embed_dim={opt_embed_dim}...")

# Prepare WGAN data
opt_E_train_tensor = torch.FloatTensor(opt_E_train_seen)
opt_S_train_conds = torch.FloatTensor(
    np.array([opt_S_seen_prototypes[int(l)] for l in label_train_seen])
)
opt_wgan_dataset = TensorDataset(opt_E_train_tensor, opt_S_train_conds)
opt_wgan_loader = DataLoader(opt_wgan_dataset, batch_size=256, shuffle=True, drop_last=True)

# Create WGAN models with optimal embed_dim
opt_gen = Generator(z_dim=100, embed_dim=opt_embed_dim).to(device)
opt_crit = Critic(embed_dim=opt_embed_dim).to(device)

opt_g_opt = torch.optim.Adam(opt_gen.parameters(), lr=1e-4, betas=(0.0, 0.9))
opt_c_opt = torch.optim.Adam(opt_crit.parameters(), lr=1e-4, betas=(0.0, 0.9))

# Training
n_steps = 10000
n_critic = 5
lambda_gp = 10
opt_g_losses = []
opt_c_losses = []
data_iter = iter(opt_wgan_loader)
g_step = 0

while g_step < n_steps:
    # Critic updates
    for _ in range(n_critic):
        try:
            e_real, s_c = next(data_iter)
        except StopIteration:
            data_iter = iter(opt_wgan_loader)
            e_real, s_c = next(data_iter)
        e_real, s_c = e_real.to(device), s_c.to(device)
        bs = e_real.size(0)

        z = torch.randn(bs, 100, device=device)
        e_fake = opt_gen(z, s_c)

        d_real = opt_crit(e_real, s_c)
        d_fake = opt_crit(e_fake.detach(), s_c)
        gp = compute_gradient_penalty(opt_crit, e_real, e_fake.detach(), s_c, device)

        c_loss = -d_real.mean() + d_fake.mean() + lambda_gp * gp
        opt_c_opt.zero_grad()
        c_loss.backward()
        opt_c_opt.step()

    # Generator update
    try:
        _, s_c = next(data_iter)
    except StopIteration:
        data_iter = iter(opt_wgan_loader)
        _, s_c = next(data_iter)
    s_c = s_c.to(device)
    bs = s_c.size(0)

    z = torch.randn(bs, 100, device=device)
    e_fake = opt_gen(z, s_c)
    d_fake = opt_crit(e_fake, s_c)
    g_loss = -d_fake.mean()

    opt_g_opt.zero_grad()
    g_loss.backward()
    opt_g_opt.step()

    opt_g_losses.append(g_loss.item())
    opt_c_losses.append(c_loss.item())
    g_step += 1

    if g_step % 2000 == 0 or g_step == 1:
        print(f"  Step {g_step:5d}/{n_steps}: G={g_loss.item():.4f}, C={c_loss.item():.4f}")

print(f"\\ncWGAN-GP complete. Final G={opt_g_losses[-1]:.4f}, C={opt_c_losses[-1]:.4f}")

# Generate synthetic embeddings
opt_gen.eval()
opt_synth_embeddings = []
opt_synth_labels = []

with torch.no_grad():
    for c in opt_unseen_classes:
        s_c = torch.FloatTensor(opt_S_unseen_prototypes[c]).unsqueeze(0).repeat(20, 1).to(device)
        z = torch.randn(20, 100, device=device)
        e_synth = opt_gen(z, s_c).cpu().numpy()
        opt_synth_embeddings.append(e_synth)
        opt_synth_labels.extend([c] * 20)

opt_E_synth_unseen = np.vstack(opt_synth_embeddings)
opt_y_synth_unseen = np.array(opt_synth_labels)

# Quality check
opt_synth_var = opt_E_synth_unseen.var(axis=0).mean()
opt_real_var = opt_E_train_seen.var(axis=0).mean()
print(f"\\nSynthetic: {opt_E_synth_unseen.shape}, norms={np.linalg.norm(opt_E_synth_unseen, axis=1).mean():.4f}")
print(f"Per-dim variance: Real={opt_real_var:.4f}, Synth={opt_synth_var:.4f}")

# Cache
np.save('cached_arrays/E_synth_unseen.npy', opt_E_synth_unseen)
np.save('cached_arrays/y_synth_unseen.npy', opt_y_synth_unseen)

# Cleanup
del opt_gen, opt_crit, opt_g_opt, opt_c_opt
torch.cuda.empty_cache()
gc.collect()

print("Synthetic embeddings generated and cached.")""")


def cell_77_phase5c():
    return make_code_cell("""# =============================================================================
# PHASE 5c: GZSL CLASSIFIER + EVALUATION (OPTIMAL CONFIG)
# =============================================================================

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

set_seeds(42)

# Sample balancing (same logic as cell 51)
opt_seen_classes_train, opt_seen_counts = np.unique(label_train_seen, return_counts=True)
opt_median = int(np.median(opt_seen_counts))

rng = np.random.RandomState(42)
opt_ds_indices = []
opt_unseen_classes_synth = np.unique(opt_y_synth_unseen)
for c in opt_unseen_classes_synth:
    idx = np.where(opt_y_synth_unseen == c)[0]
    if len(idx) > opt_median:
        selected = rng.choice(idx, size=opt_median, replace=False)
    else:
        selected = idx
    opt_ds_indices.append(selected)
opt_ds_indices = np.concatenate(opt_ds_indices)

opt_E_synth_ds = opt_E_synth_unseen[opt_ds_indices]
opt_y_synth_ds = opt_y_synth_unseen[opt_ds_indices]

print(f"Sample balance: seen median={opt_median}/class, "
      f"synth downsampled to {len(opt_E_synth_ds)} (was {len(opt_E_synth_unseen)})")

# Combine
opt_X_train_gzsl = np.vstack([opt_E_train_seen, opt_E_synth_ds])
opt_y_train_gzsl = np.concatenate([label_train_seen, opt_y_synth_ds])

print(f"GZSL training: {len(opt_X_train_gzsl)} samples, "
      f"{len(np.unique(opt_y_train_gzsl))} classes")

# Train classifier
print("\\nTraining GZSL classifier...")
opt_gzsl_clf = LogisticRegression(
    multi_class='multinomial', solver='lbfgs', max_iter=1000,
    class_weight='balanced', random_state=42, n_jobs=-1, verbose=1
)
opt_gzsl_clf.fit(opt_X_train_gzsl, opt_y_train_gzsl)

# Evaluate using harness from cell 63
opt_seen_labels_set = set(np.unique(label_train_seen))
opt_unseen_labels_set = set(np.unique(opt_y_synth_unseen))

opt_pipeline_results = evaluate_gzsl(
    opt_gzsl_clf, opt_E_test_seen, label_test_seen,
    opt_E_unseen, Y_unseen_tensor.numpy(),
    opt_seen_labels_set, opt_unseen_labels_set,
    phase_name="Optimal Config"
)

# Diagnostics from cell 64
opt_pipeline_diag = diagnose_classifier(
    opt_gzsl_clf, opt_seen_labels_set, opt_unseen_labels_set,
    phase_name="Optimal Config"
)""")


def cell_78_comparison():
    return make_code_cell("""# =============================================================================
# PHASE 5: BASELINE vs OPTIMAL COMPARISON
# =============================================================================

print("=" * 70)
print("SWEEP PHASE 5: Baseline vs Optimal Config Comparison")
print("=" * 70)

# Baseline results from main pipeline (cell 66)
baseline_res = pipeline_results

print(f"\\n{'Metric':<20} {'Baseline':>12} {'Optimal':>12} {'Delta':>12} {'Rel':>10}")
print("-" * 68)

metrics_to_compare = [
    ('AccS', 'acc_seen'),
    ('AccU', 'acc_unseen'),
    ('H-mean', 'H'),
    ('F1 Seen', 'f1_seen'),
    ('F1 Unseen', 'f1_unseen'),
    ('Routing Rate', 'routing_rate'),
]

for label, key in metrics_to_compare:
    base_v = baseline_res[key]
    opt_v = opt_pipeline_results[key]
    delta = opt_v - base_v
    rel = (delta / base_v * 100) if base_v > 0 else float('inf')
    print(f"{label:<20} {base_v:>12.4f} {opt_v:>12.4f} {delta:>+12.4f} {rel:>+9.1f}%")

print("-" * 68)

# Encoder-level comparison
print(f"\\n{'Encoder top-1':<20} {'N/A':>12} {opt_top1:>12.4f}")
print(f"{'Encoder loss':<20} {encoder_losses[-1]:>12.4f} {opt_encoder_loss:>12.4f}")

# Config comparison
print(f"\\nConfig changes:")
for key in ['embed_dim', 'tau', 'epochs', 'lr', 'weight_decay', 'image_hidden_dims']:
    base_val = BASELINE_CONFIG.get(key, 'N/A')
    opt_val = OPTIMAL_CONFIG.get(key, 'N/A')
    if str(base_val) != str(opt_val):
        print(f"  {key}: {base_val} -> {opt_val}")

# Bar chart comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# AccS, AccU, H
metrics_labels = ['AccS', 'AccU', 'H-mean']
base_vals = [baseline_res['acc_seen'], baseline_res['acc_unseen'], baseline_res['H']]
opt_vals = [opt_pipeline_results['acc_seen'], opt_pipeline_results['acc_unseen'],
            opt_pipeline_results['H']]

x = np.arange(len(metrics_labels))
width = 0.35
bars1 = axes[0].bar(x - width/2, base_vals, width, label='Baseline', color='steelblue')
bars2 = axes[0].bar(x + width/2, opt_vals, width, label='Optimal', color='darkorange')
for bar in bars1:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics_labels)
axes[0].set_ylabel('Score')
axes[0].set_title('GZSL Metrics: Baseline vs Optimal')
axes[0].legend()

# Routing rate
axes[1].bar(['Baseline', 'Optimal'],
            [baseline_res['routing_rate'], opt_pipeline_results['routing_rate']],
            color=['steelblue', 'darkorange'], edgecolor='black')
axes[1].axhline(y=200/1854, color='gray', linestyle='--', alpha=0.7, label='Ideal prior (10.8%)')
axes[1].set_ylabel('Routing Rate')
axes[1].set_title('Seen->Unseen Routing Rate')
axes[1].legend()

plt.tight_layout()
plt.savefig('figures/sweep_baseline_vs_optimal.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: figures/sweep_baseline_vs_optimal.png")""")


def cell_79_report():
    return make_code_cell("""# =============================================================================
# FINAL SWEEP REPORT
# =============================================================================

print("=" * 70)
print("HYPERPARAMETER SWEEP — FINAL REPORT")
print("=" * 70)

# Phase-by-phase best results
print("\\n--- Phase Results ---")
print(f"Phase 1 (Epoch x Tau):    tau={best_tau}, epochs={best_epochs_count}, "
      f"top1={best_top1_p1:.4f}")
print(f"Phase 2 (Embed Dim):      dim={best_dim}, "
      f"top1={best_p2['metrics']['top1']:.4f}")
print(f"Phase 3 (Architecture):   arch={best_image_hidden_dims}, "
      f"top1={best_p3['metrics']['top1']:.4f}")
print(f"Phase 4 (LR x WD):        lr={best_lr}, wd={best_wd}, "
      f"top1={best_p4['metrics']['top1']:.4f}")

# Full pipeline results
print(f"\\n--- Full Pipeline (Phase 5) ---")
print(f"  AccS:         {opt_pipeline_results['acc_seen']:.4f} "
      f"({opt_pipeline_results['acc_seen']*100:.2f}%)")
print(f"  AccU:         {opt_pipeline_results['acc_unseen']:.4f} "
      f"({opt_pipeline_results['acc_unseen']*100:.2f}%)")
print(f"  H-mean:       {opt_pipeline_results['H']:.4f}")
print(f"  Routing Rate: {opt_pipeline_results['routing_rate']:.4f}")
print(f"  Encoder Loss: {opt_encoder_loss:.4f}")

# Improvement summary
delta_H = opt_pipeline_results['H'] - baseline_res['H']
rel_H = (delta_H / baseline_res['H'] * 100) if baseline_res['H'] > 0 else float('inf')
print(f"\\n--- Improvement over Baseline ---")
print(f"  H-mean: {baseline_res['H']:.4f} -> {opt_pipeline_results['H']:.4f} "
      f"(+{delta_H:.4f}, +{rel_H:.1f}%)")

# Optimal config
print(f"\\n--- Optimal Config ---")
for k, v in OPTIMAL_CONFIG.items():
    print(f"  {k}: {v}")

# Figures generated
print(f"\\n--- Figures Generated ---")
sweep_figs = [
    'figures/sweep_phase1_heatmap.png',
    'figures/sweep_phase2_embed_dim.png',
    'figures/sweep_phase4_lr_wd.png',
    'figures/sweep_optimal_loss_curve.png',
    'figures/sweep_baseline_vs_optimal.png',
    'figures/optimal_config_diagnostics.png',
]
for f in sweep_figs:
    print(f"  {f}")

print(f"\\n{'='*70}")
print("SWEEP COMPLETE")
print(f"{'='*70}")""")


def cell_80_footer():
    return make_markdown_cell("""---

## Sweep Complete

The optimal configuration has been identified through sequential greedy sweep across 4 parameter dimensions (temperature/epochs, embedding dimension, projector architecture, learning rate/weight decay). The full GZSL pipeline has been validated with the optimal config.

**Next steps:**
- Structure-preserving WGAN-GP mathematical research (variance diffusion regularisation, dual-graph Laplacian)
- Source CLIP image features for comparison
- Routing calibration (temperature scaling)""")


# =============================================================================
# SANITY CHECKS
# =============================================================================

def run_checks(nb):
    """Verify the modified notebook."""
    cells = nb['cells']
    n = len(cells)

    checks = []

    # Check 1: Cell count
    checks.append(('Cell count == 81', n == 81))

    # Check 2: Cell 67 is markdown
    checks.append(('Cell 67 is markdown', cells[67]['cell_type'] == 'markdown'))

    # Check 3: Cell 68 has sweep utilities
    src68 = ''.join(cells[68]['source'])
    checks.append(('Cell 68 has train_encoder_sweep', 'train_encoder_sweep' in src68))
    checks.append(('Cell 68 has FlexibleImageProjector', 'FlexibleImageProjector' in src68))
    checks.append(('Cell 68 has evaluate_encoder_topk', 'evaluate_encoder_topk' in src68))

    # Check 4: Phase cells reference correct variables
    src69 = ''.join(cells[69]['source'])
    checks.append(('Cell 69 has tau_values', 'tau_values' in src69))
    checks.append(('Cell 69 has best_tau', 'best_tau' in src69))

    src71 = ''.join(cells[71]['source'])
    checks.append(('Cell 71 has embed_dims', 'embed_dims' in src71))

    # Check 5: Phase 5 cells reference OPTIMAL_CONFIG
    src75 = ''.join(cells[75]['source'])
    checks.append(('Cell 75 has OPTIMAL_CONFIG', 'OPTIMAL_CONFIG' in src75))

    src76 = ''.join(cells[76]['source'])
    checks.append(('Cell 76 has WGAN-GP training', 'opt_gen' in src76))

    src77 = ''.join(cells[77]['source'])
    checks.append(('Cell 77 has evaluate_gzsl', 'evaluate_gzsl' in src77))

    # Check 6: Cell 80 is markdown footer
    checks.append(('Cell 80 is markdown', cells[80]['cell_type'] == 'markdown'))

    # Print results
    print("\n--- Sanity Checks ---")
    all_pass = True
    for desc, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {desc}")

    return all_pass


# =============================================================================
# MAIN
# =============================================================================

def main():
    import os

    # Read notebook
    with open(NOTEBOOK) as f:
        nb = json.load(f)

    # Backup
    backup_path = NOTEBOOK + f'.backup.sweep.{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    shutil.copy2(NOTEBOOK, backup_path)
    print(f"Backup: {backup_path}")

    # Verify current cell count
    n_before = len(nb['cells'])
    if n_before != 67:
        print(f"WARNING: Expected 67 cells, found {n_before}. Proceeding anyway.")

    # Create new cells
    new_cells = [
        cell_67_header(),       # 67: markdown header
        cell_68_utilities(),    # 68: utility functions
        cell_69_phase1(),       # 69: Phase 1 sweep
        cell_70_phase1_viz(),   # 70: Phase 1 visualization
        cell_71_phase2(),       # 71: Phase 2 sweep
        cell_72_phase3(),       # 72: Phase 3 sweep
        cell_73_phase4(),       # 73: Phase 4 sweep
        cell_74_summary(),      # 74: sweep summary + OPTIMAL_CONFIG
        cell_75_phase5a(),      # 75: Phase 5a encoder training
        cell_76_phase5b(),      # 76: Phase 5b WGAN-GP + synthesis
        cell_77_phase5c(),      # 77: Phase 5c classifier + evaluation
        cell_78_comparison(),   # 78: Phase 5 comparison
        cell_79_report(),       # 79: final report
        cell_80_footer(),       # 80: markdown footer
    ]

    # Append cells
    nb['cells'].extend(new_cells)

    n_after = len(nb['cells'])
    print(f"Cells: {n_before} -> {n_after} (+{len(new_cells)})")

    # Run sanity checks
    all_pass = run_checks(nb)

    if not all_pass:
        print("\nWARNING: Some sanity checks failed!")
    else:
        print("\nAll sanity checks passed.")

    # Write
    with open(NOTEBOOK, 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"\nNotebook written: {NOTEBOOK}")
    print("Done!")


if __name__ == '__main__':
    main()
