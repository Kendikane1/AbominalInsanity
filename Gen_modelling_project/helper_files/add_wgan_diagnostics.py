#!/usr/bin/env python3
"""
Add WGAN-GP synthesis diagnostic cells to GZSL_EEG_Pipeline_v2.ipynb.

Adds 4 cells (67-70) after the existing 67 cells (0-66):
  Cell 67: Markdown header — diagnostic section overview
  Cell 68: Diagnostic 1 — Per-dimension variance profile (variance decay detection)
  Cell 69: Diagnostic 2 — Class-conditional variance (within-class diversity)
  Cell 70: Diagnostic 3 — Prototype neighbourhood preservation (structural fidelity)

All diagnostics reference variables already in memory after running cells 0-66:
  E_train_seen, E_synth_unseen, y_train_seen, y_synth_unseen (from cell 50)
  S_seen_array, S_unseen_array, seen_classes, unseen_classes (from cell 35)

Run from project root:
  python helper_files/add_wgan_diagnostics.py
"""

import json
import os
import shutil
from datetime import datetime

NOTEBOOK_PATH = "GZSL_EEG_Pipeline_v2.ipynb"

def make_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.split("\n") if isinstance(source, str) else source
    }

def make_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.split("\n") if isinstance(source, str) else source
    }

def fix_source_lines(cell):
    """Ensure each line (except last) ends with \\n."""
    lines = cell["source"]
    fixed = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1 and not line.endswith("\n"):
            fixed.append(line + "\n")
        else:
            fixed.append(line)
    cell["source"] = fixed
    return cell

# =============================================================================
# CELL DEFINITIONS
# =============================================================================

CELL_67_MARKDOWN = """---

# WGAN-GP Synthesis Diagnostics

Three targeted diagnostics to characterise the cWGAN-GP's synthesis quality, guided by pathologies identified in Long et al. (TPAMI 2018):

1. **Per-dimension variance profile** — Does the GAN preserve per-dimension information content, or does it compress/inflate specific dimensions? (Variance decay pathology)
2. **Class-conditional variance** — Does the GAN preserve within-class diversity, or do synthetic samples collapse toward prototypes? (Mode collapse / structural fidelity)
3. **Prototype neighbourhood preservation** — Does the GAN preserve inter-class distance structure, or does it distort the geometry of the embedding space? (Structural discrepancy pathology)

These diagnostics operate on the embeddings already computed by the pipeline above. No retraining is needed."""

CELL_68_DIAG1 = '''# =============================================================================
# DIAGNOSTIC 1: Per-Dimension Variance Profile
# =============================================================================
# Checks for variance decay — whether the GAN preserves per-dimension
# information content from real to synthetic embeddings.
# Reference: Long et al. (TPAMI 2018) Fig. 6 — variance decay pathology
#
# Key question: Does the GAN uniformly reproduce variance across all embedding
# dimensions, or does it compress high-variance dims and inflate low-variance ones?

import numpy as np
import matplotlib.pyplot as plt

d = E_train_seen.shape[1]

# ---- Global per-dimension variance (across all samples) ----
var_real = np.var(E_train_seen, axis=0)       # (d,)
var_synth = np.var(E_synth_unseen, axis=0)    # (d,)

# Sort by real variance (descending) — canonical ordering
sort_idx = np.argsort(var_real)[::-1]
var_real_sorted = var_real[sort_idx]
var_synth_sorted = var_synth[sort_idx]

# Cumulative variance (information concentration)
cumvar_real = np.cumsum(var_real_sorted) / var_real_sorted.sum()
cumvar_synth = np.cumsum(var_synth_sorted) / var_synth_sorted.sum()

# Per-dimension variance ratio
var_ratio = var_synth_sorted / (var_real_sorted + 1e-10)

# ---- Cosine-to-prototype distribution ----
# How far are real/synthetic samples from their class prototypes?
cos_to_proto_real = []
for c in seen_classes:
    mask = y_train_seen == c
    class_embeds = E_train_seen[mask]
    proto = S_seen_array[seen_classes.index(c) if isinstance(seen_classes, list) else np.searchsorted(seen_classes, c)]
    sims = class_embeds @ proto  # cosine sim (both L2-normed)
    cos_to_proto_real.extend(sims)
cos_to_proto_real = np.array(cos_to_proto_real)

cos_to_proto_synth = []
for c in unseen_classes:
    mask = y_synth_unseen == c
    class_embeds = E_synth_unseen[mask]
    proto = S_unseen_array[unseen_classes.index(c) if isinstance(unseen_classes, list) else np.searchsorted(unseen_classes, c)]
    sims = class_embeds @ proto
    cos_to_proto_synth.extend(sims)
cos_to_proto_synth = np.array(cos_to_proto_synth)

# ---- Summary statistics ----
print("=" * 70)
print("DIAGNOSTIC 1: Per-Dimension Variance Profile")
print("=" * 70)
print(f"  Embedding dimension: {d}")
print(f"  Real samples: {E_train_seen.shape[0]}, Synthetic samples: {E_synth_unseen.shape[0]}")
print(f"\\n  Total variance — Real: {var_real.sum():.6f}, Synthetic: {var_synth.sum():.6f}")
print(f"  Global variance ratio (synth/real): {var_synth.sum() / var_real.sum():.4f}")
print(f"\\n  Per-dim variance ratio stats:")
print(f"    Mean: {var_ratio.mean():.4f}, Std: {var_ratio.std():.4f}")
print(f"    Min:  {var_ratio.min():.4f}, Max: {var_ratio.max():.4f}")
print(f"    Dims where synth < 50% of real:  {(var_ratio < 0.5).sum()}/{d}")
print(f"    Dims where synth > 150% of real: {(var_ratio > 1.5).sum()}/{d}")
print(f"    Dims within 20% of real:         {((var_ratio > 0.8) & (var_ratio < 1.2)).sum()}/{d}")

# Coefficient of variation of variance profile
cv_real = np.std(var_real) / (np.mean(var_real) + 1e-10)
cv_synth = np.std(var_synth) / (np.mean(var_synth) + 1e-10)
print(f"\\n  Variance profile CV — Real: {cv_real:.4f}, Synthetic: {cv_synth:.4f}")
print(f"  (Higher CV = more concentrated information across dimensions)")

# Dims needed for 90% variance
dims_90_real = np.searchsorted(cumvar_real, 0.9) + 1
dims_90_synth = np.searchsorted(cumvar_synth, 0.9) + 1
print(f"\\n  Dims for 90% variance — Real: {dims_90_real}/{d}, Synthetic: {dims_90_synth}/{d}")

# Cosine-to-prototype stats
print(f"\\n  Cosine sim to own prototype:")
print(f"    Real (seen):       mean={cos_to_proto_real.mean():.4f}, std={cos_to_proto_real.std():.4f}")
print(f"    Synthetic (unseen): mean={cos_to_proto_synth.mean():.4f}, std={cos_to_proto_synth.std():.4f}")

# ---- Variance decay severity score ----
# Weighted by real variance magnitude: how much info is lost in top dims?
weights = var_real_sorted / var_real_sorted.sum()
weighted_ratio_dev = np.sum(weights * np.abs(var_ratio - 1.0))
print(f"\\n  Variance decay severity (weighted MAE from 1.0): {weighted_ratio_dev:.4f}")
print(f"  (0 = perfect match, higher = more distortion in important dims)")
print("=" * 70)

# ---- Plot ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Sorted variance profiles
axes[0, 0].plot(range(d), var_real_sorted, 'b-', alpha=0.7, linewidth=1.5, label='Real (seen)')
axes[0, 0].plot(range(d), var_synth_sorted, 'r-', alpha=0.7, linewidth=1.5, label='Synthetic (unseen)')
axes[0, 0].set_xlabel('Dimension (sorted by real variance)')
axes[0, 0].set_ylabel('Variance')
axes[0, 0].set_title('Per-Dimension Variance Profile')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Panel 2: Variance ratio per dimension
colors = ['green' if 0.8 <= r <= 1.2 else 'orange' if 0.5 <= r <= 1.5 else 'red' for r in var_ratio]
axes[0, 1].bar(range(d), var_ratio, color=colors, alpha=0.6, width=1.0)
axes[0, 1].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Perfect match')
axes[0, 1].axhline(y=0.5, color='r', linestyle=':', alpha=0.3)
axes[0, 1].axhline(y=1.5, color='r', linestyle=':', alpha=0.3)
axes[0, 1].set_xlabel('Dimension (sorted by real variance)')
axes[0, 1].set_ylabel('Variance ratio (synth/real)')
axes[0, 1].set_title('Per-Dimension Variance Ratio')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Panel 3: Cumulative variance
axes[1, 0].plot(range(d), cumvar_real, 'b-', alpha=0.7, linewidth=1.5, label='Real (seen)')
axes[1, 0].plot(range(d), cumvar_synth, 'r-', alpha=0.7, linewidth=1.5, label='Synthetic (unseen)')
axes[1, 0].axhline(y=0.9, color='k', linestyle=':', alpha=0.3, label='90% threshold')
axes[1, 0].set_xlabel('Dimension (sorted by real variance)')
axes[1, 0].set_ylabel('Cumulative variance fraction')
axes[1, 0].set_title('Information Concentration')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Panel 4: Cosine-to-prototype distribution
axes[1, 1].hist(cos_to_proto_real, bins=50, alpha=0.5, color='blue', density=True,
                label=f'Real→proto (μ={cos_to_proto_real.mean():.3f})')
axes[1, 1].hist(cos_to_proto_synth, bins=50, alpha=0.5, color='red', density=True,
                label=f'Synth→proto (μ={cos_to_proto_synth.mean():.3f})')
axes[1, 1].set_xlabel('Cosine similarity to class prototype')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('Sample-to-Prototype Distance Distribution')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/diag1_variance_profile.png', dpi=150, bbox_inches='tight')
plt.show()
print("\\nSaved: figures/diag1_variance_profile.png")'''

CELL_69_DIAG2 = '''# =============================================================================
# DIAGNOSTIC 2: Class-Conditional Variance
# =============================================================================
# Checks whether within-class diversity is preserved by the GAN.
# If the GAN collapses within-class variance (even while matching global variance),
# synthetic embeddings cluster too tightly around prototypes and the classifier
# gets overconfident on unseen classes — exacerbating seen-unseen routing bias.
#
# Key question: Does the GAN generate realistic intra-class spread, or do
# synthetic samples collapse toward their conditioning prototype?

import numpy as np
import matplotlib.pyplot as plt

d = E_train_seen.shape[1]

# ---- Real seen: within-class per-dim variance ----
real_class_variances = []
real_class_sizes = []
for c in seen_classes:
    mask = y_train_seen == c
    class_embeds = E_train_seen[mask]
    if len(class_embeds) > 1:
        real_class_variances.append(np.var(class_embeds, axis=0))
        real_class_sizes.append(len(class_embeds))
real_class_variances = np.array(real_class_variances)  # (n_seen, d)
mean_real_class_var = real_class_variances.mean(axis=0)  # (d,)

# ---- Synthetic unseen: within-class per-dim variance ----
synth_class_variances = []
synth_class_sizes = []
for c in unseen_classes:
    mask = y_synth_unseen == c
    class_embeds = E_synth_unseen[mask]
    if len(class_embeds) > 1:
        synth_class_variances.append(np.var(class_embeds, axis=0))
        synth_class_sizes.append(len(class_embeds))
synth_class_variances = np.array(synth_class_variances)  # (n_unseen, d)
mean_synth_class_var = synth_class_variances.mean(axis=0)  # (d,)

# ---- Per-class total variance (trace of covariance) ----
real_total_var = real_class_variances.sum(axis=1)      # (n_seen,)
synth_total_var = synth_class_variances.sum(axis=1)    # (n_unseen,)

# ---- Variance decomposition (ANOVA-style) ----
# Total = Within-class + Between-class
# Real seen
global_mean_real = E_train_seen.mean(axis=0)
total_var_real = np.var(E_train_seen, axis=0).sum()
within_var_real = mean_real_class_var.sum()
between_var_real = total_var_real - within_var_real

# Synthetic unseen
global_mean_synth = E_synth_unseen.mean(axis=0)
total_var_synth = np.var(E_synth_unseen, axis=0).sum()
within_var_synth = mean_synth_class_var.sum()
between_var_synth = total_var_synth - within_var_synth

# ---- Also check real UNSEEN (E_unseen with y_unseen) for ground truth ----
real_unseen_class_variances = []
for c in unseen_classes:
    mask = y_unseen == c
    class_embeds = E_unseen[mask]
    if len(class_embeds) > 1:
        real_unseen_class_variances.append(np.var(class_embeds, axis=0))
real_unseen_class_variances = np.array(real_unseen_class_variances)
mean_real_unseen_class_var = real_unseen_class_variances.mean(axis=0)
real_unseen_total_var = real_unseen_class_variances.sum(axis=1)

# ---- Summary ----
print("=" * 70)
print("DIAGNOSTIC 2: Class-Conditional Variance")
print("=" * 70)
print(f"  Real seen classes: {len(real_class_variances)} (median {np.median(real_class_sizes):.0f} samples/class)")
print(f"  Synthetic unseen classes: {len(synth_class_variances)} ({synth_class_sizes[0]} samples/class)")
print(f"  Real unseen classes: {len(real_unseen_class_variances)} (ground truth reference)")

print(f"\\n  Mean within-class variance (averaged over dims):")
print(f"    Real seen:       {mean_real_class_var.mean():.6f}")
print(f"    Synthetic unseen: {mean_synth_class_var.mean():.6f}")
print(f"    Real unseen:     {mean_real_unseen_class_var.mean():.6f}  ← ground truth")
print(f"    Ratio synth/real_seen:   {mean_synth_class_var.mean() / (mean_real_class_var.mean() + 1e-10):.4f}")
print(f"    Ratio synth/real_unseen: {mean_synth_class_var.mean() / (mean_real_unseen_class_var.mean() + 1e-10):.4f}")

print(f"\\n  Per-class total variance (trace):")
print(f"    Real seen:       mean={real_total_var.mean():.6f}, std={real_total_var.std():.6f}")
print(f"    Synthetic unseen: mean={synth_total_var.mean():.6f}, std={synth_total_var.std():.6f}")
print(f"    Real unseen:     mean={real_unseen_total_var.mean():.6f}, std={real_unseen_total_var.std():.6f}")

print(f"\\n  Variance decomposition (Total = Within + Between):")
print(f"    Real seen:       Total={total_var_real:.6f}, Within={within_var_real:.6f} ({100*within_var_real/total_var_real:.1f}%), Between={between_var_real:.6f} ({100*between_var_real/total_var_real:.1f}%)")
print(f"    Synthetic unseen: Total={total_var_synth:.6f}, Within={within_var_synth:.6f} ({100*within_var_synth/total_var_synth:.1f}%), Between={between_var_synth:.6f} ({100*between_var_synth/total_var_synth:.1f}%)")

# Within/Between ratio — higher = less discriminable
wb_ratio_real = within_var_real / (between_var_real + 1e-10)
wb_ratio_synth = within_var_synth / (between_var_synth + 1e-10)
print(f"\\n  Within/Between ratio (lower = more discriminable):")
print(f"    Real seen:       {wb_ratio_real:.4f}")
print(f"    Synthetic unseen: {wb_ratio_synth:.4f}")

if wb_ratio_synth < wb_ratio_real * 0.5:
    print("\\n  ⚠ WARNING: Synthetic within-class variance is disproportionately low.")
    print("    The GAN may be collapsing toward prototypes (mode collapse lite).")
elif wb_ratio_synth > wb_ratio_real * 2.0:
    print("\\n  ⚠ WARNING: Synthetic within-class variance is disproportionately high.")
    print("    The GAN may be generating noisy/unfocused embeddings.")
else:
    print("\\n  ✓ Within/Between ratio is in reasonable range.")
print("=" * 70)

# ---- Plot ----
sort_idx = np.argsort(mean_real_class_var)[::-1]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Mean within-class variance per dimension
axes[0, 0].plot(range(d), mean_real_class_var[sort_idx], 'b-', alpha=0.7, linewidth=1.5, label='Real seen')
axes[0, 0].plot(range(d), mean_synth_class_var[sort_idx], 'r-', alpha=0.7, linewidth=1.5, label='Synthetic unseen')
axes[0, 0].plot(range(d), mean_real_unseen_class_var[sort_idx], 'g--', alpha=0.5, linewidth=1.0, label='Real unseen (truth)')
axes[0, 0].set_xlabel('Dimension (sorted by real within-class var)')
axes[0, 0].set_ylabel('Mean within-class variance')
axes[0, 0].set_title('Within-Class Variance Profile')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Panel 2: Distribution of per-class total variance
axes[0, 1].hist(real_total_var, bins=30, alpha=0.4, color='blue', density=True,
                label=f'Real seen (μ={real_total_var.mean():.4f})')
axes[0, 1].hist(synth_total_var, bins=30, alpha=0.4, color='red', density=True,
                label=f'Synth unseen (μ={synth_total_var.mean():.4f})')
axes[0, 1].hist(real_unseen_total_var, bins=30, alpha=0.4, color='green', density=True,
                label=f'Real unseen (μ={real_unseen_total_var.mean():.4f})')
axes[0, 1].set_xlabel('Per-class total variance (trace)')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('Distribution of Per-Class Variance')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Panel 3: Within-class variance ratio per dimension (synth/real)
class_var_ratio = mean_synth_class_var / (mean_real_class_var + 1e-10)
class_var_ratio_unseen = mean_synth_class_var / (mean_real_unseen_class_var + 1e-10)
axes[1, 0].bar(range(d), class_var_ratio[sort_idx], color='purple', alpha=0.5, width=1.0, label='Synth/Real seen')
axes[1, 0].axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
axes[1, 0].set_xlabel('Dimension (sorted)')
axes[1, 0].set_ylabel('Within-class variance ratio')
axes[1, 0].set_title('Within-Class Variance Ratio (Synth / Real Seen)')
axes[1, 0].grid(True, alpha=0.3)

# Panel 4: Scatter — per-class total variance (synth vs real unseen ground truth)
# Each point = one unseen class. x = real unseen variance, y = synthetic variance
axes[1, 1].scatter(real_unseen_total_var, synth_total_var, alpha=0.5, s=20, color='purple')
lim = max(real_unseen_total_var.max(), synth_total_var.max()) * 1.1
axes[1, 1].plot([0, lim], [0, lim], 'k--', alpha=0.3, label='y=x')
axes[1, 1].set_xlabel('Real unseen per-class variance')
axes[1, 1].set_ylabel('Synthetic per-class variance')
axes[1, 1].set_title('Per-Class Variance: Synth vs Real Unseen')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/diag2_class_conditional_variance.png', dpi=150, bbox_inches='tight')
plt.show()
print("\\nSaved: figures/diag2_class_conditional_variance.png")'''

CELL_70_DIAG3 = '''# =============================================================================
# DIAGNOSTIC 3: Prototype Neighbourhood Preservation
# =============================================================================
# Checks whether inter-class distances in prototype space are preserved
# in synthetic embedding space. If the GAN distorts neighbourhood structure,
# the classifier will have wrong decision boundaries for unseen classes.
#
# Three tests:
#   1. Pairwise distance correlation (Spearman ρ) — ordinal structure
#   2. k-NN preservation rate — local neighbourhood fidelity
#   3. Seen reference — same metrics on real data (upper bound)
#
# Key question: When the GAN generates embeddings conditioned on prototype s_c,
# does the distance between synthetic clusters reflect the distance between
# conditioning prototypes?

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr

# ---- Unseen: prototype distances vs synthetic centroid distances ----
proto_dists_unseen = squareform(pdist(S_unseen_array, metric='cosine'))  # (200, 200)

# Synthetic centroids (mean of synthetic embeddings per class, then L2-norm)
synth_centroids = []
for c in unseen_classes:
    mask = y_synth_unseen == c
    centroid = E_synth_unseen[mask].mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
    synth_centroids.append(centroid)
synth_centroids = np.array(synth_centroids)
synth_dists = squareform(pdist(synth_centroids, metric='cosine'))  # (200, 200)

# Also compute real unseen brain embedding centroids (ground truth)
real_unseen_centroids = []
for c in unseen_classes:
    mask = y_unseen == c
    centroid = E_unseen[mask].mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
    real_unseen_centroids.append(centroid)
real_unseen_centroids = np.array(real_unseen_centroids)
real_unseen_dists = squareform(pdist(real_unseen_centroids, metric='cosine'))

# Flatten upper triangles
triu_idx = np.triu_indices(len(unseen_classes), k=1)
proto_flat = proto_dists_unseen[triu_idx]
synth_flat = synth_dists[triu_idx]
real_unseen_flat = real_unseen_dists[triu_idx]

# Correlations: synth vs proto
sp_synth_proto, sp_synth_proto_p = spearmanr(proto_flat, synth_flat)
pr_synth_proto, pr_synth_proto_p = pearsonr(proto_flat, synth_flat)

# Correlations: real unseen vs proto (ground truth upper bound)
sp_real_proto, _ = spearmanr(proto_flat, real_unseen_flat)
pr_real_proto, _ = pearsonr(proto_flat, real_unseen_flat)

# Correlations: synth vs real unseen (direct quality measure)
sp_synth_real, _ = spearmanr(real_unseen_flat, synth_flat)
pr_synth_real, _ = pearsonr(real_unseen_flat, synth_flat)

# ---- Seen reference: real centroids vs prototypes ----
real_seen_centroids = []
for c in seen_classes:
    mask = y_train_seen == c
    centroid = E_train_seen[mask].mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
    real_seen_centroids.append(centroid)
real_seen_centroids = np.array(real_seen_centroids)

# Subsample 200 seen classes for fair comparison
rng = np.random.RandomState(42)
n_sub = min(200, len(seen_classes))
sub_idx = rng.choice(len(seen_classes), size=n_sub, replace=False)
proto_dists_seen_sub = squareform(pdist(S_seen_array[sub_idx], metric='cosine'))
real_dists_seen_sub = squareform(pdist(real_seen_centroids[sub_idx], metric='cosine'))
triu_seen = np.triu_indices(n_sub, k=1)
proto_flat_seen = proto_dists_seen_sub[triu_seen]
real_flat_seen = real_dists_seen_sub[triu_seen]
sp_seen_ref, _ = spearmanr(proto_flat_seen, real_flat_seen)
pr_seen_ref, _ = pearsonr(proto_flat_seen, real_flat_seen)

# ---- k-NN preservation ----
k_values = [5, 10, 20, 50]
knn_synth = {}
knn_real_unseen = {}
knn_seen_ref = {}

for k in k_values:
    # Synth vs proto (unseen)
    matches_s = 0
    # Real unseen vs proto
    matches_r = 0
    for i in range(len(unseen_classes)):
        proto_nn = set(np.argsort(proto_dists_unseen[i])[1:k+1])
        synth_nn = set(np.argsort(synth_dists[i])[1:k+1])
        real_nn = set(np.argsort(real_unseen_dists[i])[1:k+1])
        matches_s += len(proto_nn & synth_nn)
        matches_r += len(proto_nn & real_nn)
    knn_synth[k] = matches_s / (len(unseen_classes) * k)
    knn_real_unseen[k] = matches_r / (len(unseen_classes) * k)

    # Seen reference
    matches_seen = 0
    for i in range(n_sub):
        proto_nn = set(np.argsort(proto_dists_seen_sub[i])[1:k+1])
        real_nn = set(np.argsort(real_dists_seen_sub[i])[1:k+1])
        matches_seen += len(proto_nn & real_nn)
    knn_seen_ref[k] = matches_seen / (n_sub * k)

# ---- Summary ----
print("=" * 70)
print("DIAGNOSTIC 3: Prototype Neighbourhood Preservation")
print("=" * 70)

print(f"\\n  Pairwise distance correlations ({len(proto_flat)} pairs):")
print(f"  {'':40s} {'Spearman ρ':>12s} {'Pearson r':>12s}")
print(f"  {'─' * 64}")
print(f"  {'Synth centroids vs Image prototypes':40s} {sp_synth_proto:12.4f} {pr_synth_proto:12.4f}")
print(f"  {'Real unseen centroids vs Image protos':40s} {sp_real_proto:12.4f} {pr_real_proto:12.4f}  ← ground truth")
print(f"  {'Synth centroids vs Real unseen centroids':40s} {sp_synth_real:12.4f} {pr_synth_real:12.4f}  ← direct quality")
print(f"  {'Seen: Real centroids vs Image protos':40s} {sp_seen_ref:12.4f} {pr_seen_ref:12.4f}  ← seen reference")

print(f"\\n  Neighbourhood preservation gap:")
print(f"    Δ Spearman (synth − real_unseen): {sp_synth_proto - sp_real_proto:.4f}")

print(f"\\n  k-NN preservation rate:")
print(f"  {'k':>4s}  {'Synth↔Proto':>14s} {'RealUnseen↔Proto':>18s} {'Seen ref':>10s}")
print(f"  {'─' * 50}")
for k in k_values:
    print(f"  {k:4d}  {knn_synth[k]:14.4f} {knn_real_unseen[k]:18.4f} {knn_seen_ref[k]:10.4f}")

# Random baseline for k-NN preservation
for k in k_values:
    random_baseline = k / (len(unseen_classes) - 1)
    print(f"  Random baseline for k={k}: {random_baseline:.4f}")

print("=" * 70)

# ---- Plot ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Scatter — proto dist vs synth dist (unseen)
axes[0, 0].scatter(proto_flat, synth_flat, alpha=0.02, s=1, color='red', rasterized=True)
lims = [0, max(proto_flat.max(), synth_flat.max()) * 1.05]
axes[0, 0].plot(lims, lims, 'k--', alpha=0.3)
axes[0, 0].set_xlabel('Prototype cosine distance')
axes[0, 0].set_ylabel('Synthetic centroid cosine distance')
axes[0, 0].set_title(f'Unseen: Proto vs Synth (ρ={sp_synth_proto:.3f})')
axes[0, 0].grid(True, alpha=0.3)

# Panel 2: Scatter — proto dist vs real unseen dist (ground truth)
axes[0, 1].scatter(proto_flat, real_unseen_flat, alpha=0.02, s=1, color='green', rasterized=True)
lims = [0, max(proto_flat.max(), real_unseen_flat.max()) * 1.05]
axes[0, 1].plot(lims, lims, 'k--', alpha=0.3)
axes[0, 1].set_xlabel('Prototype cosine distance')
axes[0, 1].set_ylabel('Real unseen centroid cosine distance')
axes[0, 1].set_title(f'Unseen: Proto vs Real (ρ={sp_real_proto:.3f})')
axes[0, 1].grid(True, alpha=0.3)

# Panel 3: Scatter — seen reference
axes[1, 0].scatter(proto_flat_seen, real_flat_seen, alpha=0.02, s=1, color='blue', rasterized=True)
lims = [0, max(proto_flat_seen.max(), real_flat_seen.max()) * 1.05]
axes[1, 0].plot(lims, lims, 'k--', alpha=0.3)
axes[1, 0].set_xlabel('Prototype cosine distance')
axes[1, 0].set_ylabel('Real seen centroid cosine distance')
axes[1, 0].set_title(f'Seen: Proto vs Real (ρ={sp_seen_ref:.3f})')
axes[1, 0].grid(True, alpha=0.3)

# Panel 4: k-NN preservation grouped bar chart
x = np.arange(len(k_values))
width = 0.25
bars1 = axes[1, 1].bar(x - width, [knn_synth[k] for k in k_values], width,
                         alpha=0.7, color='red', label='Synth↔Proto')
bars2 = axes[1, 1].bar(x, [knn_real_unseen[k] for k in k_values], width,
                         alpha=0.7, color='green', label='RealUnseen↔Proto')
bars3 = axes[1, 1].bar(x + width, [knn_seen_ref[k] for k in k_values], width,
                         alpha=0.7, color='blue', label='Seen ref')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels([f'k={k}' for k in k_values])
axes[1, 1].set_ylabel('k-NN preservation rate')
axes[1, 1].set_title('Neighbourhood Preservation')
axes[1, 1].set_ylim(0, 1)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/diag3_neighbourhood_preservation.png', dpi=150, bbox_inches='tight')
plt.show()
print("\\nSaved: figures/diag3_neighbourhood_preservation.png")'''

# =============================================================================
# ASSEMBLE AND INJECT
# =============================================================================

def main():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"ERROR: {NOTEBOOK_PATH} not found. Run from project root.")
        return

    # Backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{NOTEBOOK_PATH}.backup.wgan_diag.{timestamp}"
    shutil.copy2(NOTEBOOK_PATH, backup_path)
    print(f"Backup: {backup_path}")

    with open(NOTEBOOK_PATH, "r") as f:
        nb = json.load(f)

    n_before = len(nb["cells"])
    print(f"Cells before: {n_before}")

    if n_before != 67:
        print(f"WARNING: Expected 67 cells, found {n_before}. Proceeding anyway.")

    # Build new cells
    new_cells = [
        fix_source_lines(make_markdown_cell(CELL_67_MARKDOWN)),
        fix_source_lines(make_code_cell(CELL_68_DIAG1)),
        fix_source_lines(make_code_cell(CELL_69_DIAG2)),
        fix_source_lines(make_code_cell(CELL_70_DIAG3)),
    ]

    nb["cells"].extend(new_cells)

    n_after = len(nb["cells"])
    print(f"Cells after: {n_after}")
    print(f"Added {n_after - n_before} cells ({n_before}→{n_after - 1})")

    with open(NOTEBOOK_PATH, "w") as f:
        json.dump(nb, f, indent=1)

    print(f"\nDone! Notebook updated: {NOTEBOOK_PATH}")
    print("New cells:")
    print("  Cell 67: [markdown] WGAN-GP Synthesis Diagnostics header")
    print("  Cell 68: [code] Diagnostic 1 — Per-dimension variance profile")
    print("  Cell 69: [code] Diagnostic 2 — Class-conditional variance")
    print("  Cell 70: [code] Diagnostic 3 — Prototype neighbourhood preservation")

if __name__ == "__main__":
    main()
