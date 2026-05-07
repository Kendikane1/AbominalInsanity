#!/usr/bin/env python3
"""
Create a clean GZSL pipeline notebook from the original.

Extracts essential cells, fixes label collision at source, removes
experimental/diagnostic cruft, and adds visual feature investigation section.

Run from project root:
    python helper_files/create_clean_notebook.py
"""

import json
import copy
import os
from datetime import datetime

OLD_NOTEBOOK = 'COMP2261_ArizMLCW_with_baseline.ipynb'
NEW_NOTEBOOK = 'GZSL_EEG_Pipeline_v2.ipynb'


def make_source(text):
    """Convert multi-line string into notebook source format."""
    lines = text.split('\n')
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + '\n')
        else:
            result.append(line)
    return result


def make_code_cell(source_text):
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': make_source(source_text)
    }


def make_markdown_cell(source_text):
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': make_source(source_text)
    }


# ---------------------------------------------------------------------------
# Cell modifications
# ---------------------------------------------------------------------------

def modify_cell_0(cell):
    """Update notebook title to indicate v2."""
    src = ''.join(cell['source'])
    src = src.replace(
        '# **Multimodal Machine Learning with Brain, Image, and Text Data**',
        '# **Multimodal Machine Learning with Brain, Image, and Text Data** (v2 — Clean Pipeline)'
    )
    cell['source'] = make_source(src)


def modify_cell_7(cell):
    """Add label offset at data loading time to fix collision at source."""
    src = ''.join(cell['source'])

    # Insert label offset code after the last loadmat/reshape/slice block
    # and before the torch.from_numpy conversions.
    # Anchor: 'image_unseen = image_unseen[:, 0:100]' is the last data-prep line
    # before the torch conversions.
    insertion = """
# --- LABEL COLLISION FIX ---
# Raw unseen labels [1..200] collide with seen labels [1..1654].
# Offset so unseen occupy disjoint range [1655..1854].
_label_offset = int(label_seen.max())
label_unseen = label_unseen + _label_offset
print(f"Label fix: unseen labels offset by {_label_offset} -> [{label_unseen.min()}, {label_unseen.max()}]")
"""

    anchor = 'image_unseen = image_unseen[:, 0:100]'
    if anchor in src:
        src = src.replace(anchor, anchor + '\n' + insertion)
    else:
        raise RuntimeError(f"Could not find anchor line in cell 7: '{anchor}'")

    cell['source'] = make_source(src)


def modify_cell_14(cell):
    """Fix np.bincount for offset labels — use np.unique instead."""
    src = ''.join(cell['source'])

    src = src.replace(
        'seen_class_counts = np.bincount(label_seen_np)[1:]',
        '_, seen_class_counts = np.unique(label_seen_np, return_counts=True)'
    )
    src = src.replace(
        'unseen_class_counts = np.bincount(label_unseen_np)[1:]',
        '_, unseen_class_counts = np.unique(label_unseen_np, return_counts=True)'
    )

    cell['source'] = make_source(src)


def modify_cell_74(cell):
    """Fix embed_dim reference from 64D to 128D."""
    src = ''.join(cell['source'])
    src = src.replace('64D', '128D')
    cell['source'] = make_source(src)


# ---------------------------------------------------------------------------
# New cells: Eval harness bridge + Visual Feature Investigation
# ---------------------------------------------------------------------------

HARNESS_EVAL_CODE = """\
# =============================================================================
# EVALUATE GZSL CLASSIFIER WITH HARNESS
# =============================================================================

# Build label sets from cached arrays
seen_labels_set = set(np.unique(y_train_seen))
unseen_labels_set = set(np.unique(y_synth_unseen))

print(f"Seen labels: {len(seen_labels_set)} classes (range {min(seen_labels_set)}-{max(seen_labels_set)})")
print(f"Unseen labels: {len(unseen_labels_set)} classes (range {min(unseen_labels_set)}-{max(unseen_labels_set)})")
print(f"Overlap: {seen_labels_set & unseen_labels_set}")

# Evaluate with harness
pipeline_results = evaluate_gzsl(
    gzsl_clf, E_test_seen, y_test_seen, E_unseen, y_unseen,
    seen_labels_set, unseen_labels_set, phase_name="Pipeline v2"
)
all_phase_results['Pipeline v2'] = pipeline_results

# Diagnostics
pipeline_diag = diagnose_classifier(
    gzsl_clf, seen_labels_set, unseen_labels_set, phase_name="Pipeline v2"
)"""

VIS_HEADER_MD = """\
---

# Visual Feature Investigation

**Motivation**: The Phase F analysis identified that aligning EEG to IMAGE embeddings (rather than text) is the highest-impact intervention. NICE-EEG (ICLR 2024) achieves 15.6% top-1 on 200-way ZSL using EEG-to-CLIP-image alignment. Our current pipeline aligns EEG to text and hits a ceiling at ~3% top-1.

**Goal**: Investigate the available visual features in the dataset and assess alignment options:
1. What are the CORnet-S features? Shape, dimensionality, PCA components?
2. How do visual features compare to text features as alignment targets?
3. Can we obtain CLIP image embeddings for the stimuli?
4. What is the effective dimensionality of each modality?"""

VIS_INSPECT_CODE = """\
# =============================================================================
# VISUAL FEATURE INSPECTION
# =============================================================================

import scipy.io as sio
import numpy as np
import os

# Load full visual features (not sliced to 100 PCA components)
vis_seen_raw = sio.loadmat(os.path.join(image_dir_seen, 'feat_pca_train.mat'))
vis_unseen_raw = sio.loadmat(os.path.join(image_dir_unseen, 'feat_pca_test.mat'))

print("=" * 60)
print("VISUAL FEATURE INSPECTION")
print("=" * 60)

# Show all keys in the .mat files
print("\\nSeen visual feature keys:", [k for k in vis_seen_raw.keys() if not k.startswith('_')])
print("Unseen visual feature keys:", [k for k in vis_unseen_raw.keys() if not k.startswith('_')])

# Get data arrays
vis_seen_data = vis_seen_raw['data'].astype('double')
vis_unseen_data = vis_unseen_raw['data'].astype('double')

print(f"\\nFull visual feature shapes:")
print(f"  vis_seen (train): {vis_seen_data.shape}")
print(f"  vis_unseen (test): {vis_unseen_data.shape}")

print(f"\\nCurrently using first 100 PCA components out of {vis_seen_data.shape[1]} total")
print(f"Information retained: 100/{vis_seen_data.shape[1]} dimensions = {100/vis_seen_data.shape[1]*100:.1f}%")

# Basic statistics
print(f"\\nVisual Feature Statistics (full dimensions):")
print(f"  Seen  - mean: {vis_seen_data.mean():.4f}, std: {vis_seen_data.std():.4f}, "
      f"norm: {np.linalg.norm(vis_seen_data, axis=1).mean():.2f}")
print(f"  Unseen - mean: {vis_unseen_data.mean():.4f}, std: {vis_unseen_data.std():.4f}, "
      f"norm: {np.linalg.norm(vis_unseen_data, axis=1).mean():.2f}")"""

VIS_PCA_SPECTRUM_CODE = """\
# =============================================================================
# VISUAL FEATURE PCA VARIANCE SPECTRUM
# =============================================================================

import matplotlib.pyplot as plt

# Per-dimension variance (features are already PCA, so column variance = component variance)
var_per_dim = vis_seen_data.var(axis=0)
cumvar = np.cumsum(var_per_dim)
total_var = cumvar[-1]
pct_cumvar = cumvar / total_var * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Per-dimension variance
axes[0].plot(var_per_dim, 'b-', linewidth=1)
axes[0].set_xlabel('PCA Component', fontsize=12)
axes[0].set_ylabel('Variance', fontsize=12)
axes[0].set_title('Per-Component Variance (CORnet-S PCA)', fontsize=13)
axes[0].axvline(x=100, color='red', linestyle='--', alpha=0.7, label='Current cutoff (100)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Cumulative variance
axes[1].plot(pct_cumvar, 'g-', linewidth=2)
axes[1].set_xlabel('Number of PCA Components', fontsize=12)
axes[1].set_ylabel('Cumulative Variance (%)', fontsize=12)
axes[1].set_title('Cumulative Variance Explained', fontsize=13)
axes[1].axvline(x=100, color='red', linestyle='--', alpha=0.7,
               label=f'100 dims: {pct_cumvar[min(99, len(pct_cumvar)-1)]:.1f}%')
axes[1].axhline(y=95, color='orange', linestyle=':', alpha=0.7, label='95% threshold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/visual_feature_pca_spectrum.png', dpi=150, bbox_inches='tight')
plt.show()

# Dimension counts for key thresholds
for thresh in [90, 95, 99]:
    n_dims = np.searchsorted(pct_cumvar, thresh) + 1
    print(f"  {thresh}% variance explained at: {n_dims} dimensions")"""

VIS_COMPARE_CODE = """\
# =============================================================================
# COMPARE VISUAL vs TEXT AS ALIGNMENT TARGETS
# =============================================================================

from collections import defaultdict
from itertools import combinations

def compute_class_prototypes_raw(features, labels):
    \"\"\"Compute mean feature per class.\"\"\"
    protos = defaultdict(list)
    for feat, lbl in zip(features, labels):
        protos[int(lbl)].append(feat)
    return {k: np.mean(v, axis=0) for k, v in protos.items()}

def avg_pairwise_cosine(protos_dict, n_sample=500):
    \"\"\"Average cosine similarity between random pairs of prototypes.\"\"\"
    keys = list(protos_dict.keys())
    pairs = list(combinations(keys, 2))
    if len(pairs) > n_sample:
        np.random.seed(42)
        idx = np.random.choice(len(pairs), n_sample, replace=False)
        pairs = [pairs[i] for i in idx]
    sims = []
    for k1, k2 in pairs:
        p1, p2 = protos_dict[k1], protos_dict[k2]
        cos = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2) + 1e-10)
        sims.append(cos)
    return np.array(sims)

# Compute class prototypes for visual and text features (seen classes)
y_seen_np = label_seen.numpy().flatten()
vis_protos = compute_class_prototypes_raw(vis_seen_data, y_seen_np)
text_protos = compute_class_prototypes_raw(text_seen.numpy(), y_seen_np)

vis_sims = avg_pairwise_cosine(vis_protos)
text_sims = avg_pairwise_cosine(text_protos)

print("=" * 60)
print("ALIGNMENT TARGET COMPARISON: Visual vs Text")
print("=" * 60)
print(f"\\nInter-class cosine similarity (lower = better class separation):")
print(f"  Visual (CORnet-S): mean={vis_sims.mean():.4f}, std={vis_sims.std():.4f}")
print(f"  Text (CLIP):       mean={text_sims.mean():.4f}, std={text_sims.std():.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(vis_sims, bins=50, alpha=0.6, label='Visual (CORnet-S)', color='coral')
ax.hist(text_sims, bins=50, alpha=0.6, label='Text (CLIP)', color='steelblue')
ax.set_xlabel('Cosine Similarity Between Class Prototypes', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Inter-Class Separation: Visual vs Text Prototypes', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/visual_vs_text_separation.png', dpi=150, bbox_inches='tight')
plt.show()"""

VIS_CROSSMODAL_CODE = """\
# =============================================================================
# CROSS-MODAL CORRELATION: EEG-Image vs EEG-Text
# =============================================================================
# Per-class Pearson correlation between EEG and visual/text prototypes.
# Higher correlation = easier alignment learning.

brain_seen_np = brain_seen.numpy()
brain_protos = compute_class_prototypes_raw(brain_seen_np, y_seen_np)

classes = sorted(set(brain_protos.keys()) & set(vis_protos.keys()) & set(text_protos.keys()))

eeg_vis_corrs = []
eeg_text_corrs = []

for c in classes:
    bp = brain_protos[c]
    vp = vis_protos[c]
    tp = text_protos[c]

    # Dimension-matched correlation (truncate to shorter)
    min_dim_v = min(len(bp), len(vp))
    min_dim_t = min(len(bp), len(tp))

    r_vis = np.corrcoef(bp[:min_dim_v], vp[:min_dim_v])[0, 1]
    r_text = np.corrcoef(bp[:min_dim_t], tp[:min_dim_t])[0, 1]

    if not np.isnan(r_vis):
        eeg_vis_corrs.append(r_vis)
    if not np.isnan(r_text):
        eeg_text_corrs.append(r_text)

eeg_vis_corrs = np.array(eeg_vis_corrs)
eeg_text_corrs = np.array(eeg_text_corrs)

print("=" * 60)
print("CROSS-MODAL CORRELATION (per-class prototypes)")
print("=" * 60)
print(f"  EEG-Visual:  mean r = {eeg_vis_corrs.mean():.4f}, std = {eeg_vis_corrs.std():.4f}")
print(f"  EEG-Text:    mean r = {eeg_text_corrs.mean():.4f}, std = {eeg_text_corrs.std():.4f}")
print(f"\\nNote: These are raw feature-space correlations (before any learned projection).")
print(f"Higher absolute correlation suggests an easier alignment target.")"""

VIS_CLIP_CHECK_CODE = """\
# =============================================================================
# CLIP IMAGE FEATURE AVAILABILITY CHECK
# =============================================================================
# Scan the dataset directories for available visual feature models.

print("=" * 60)
print("CLIP IMAGE FEATURE AVAILABILITY CHECK")
print("=" * 60)

vis_base = os.path.join(data_dir_root, 'visual_feature')
print("\\nVisual feature directory structure:")
for split in ['ThingsTrain', 'ThingsTest']:
    split_dir = os.path.join(vis_base, split)
    if os.path.exists(split_dir):
        for framework in sorted(os.listdir(split_dir)):
            framework_dir = os.path.join(split_dir, framework)
            if os.path.isdir(framework_dir):
                for model in sorted(os.listdir(framework_dir)):
                    model_dir = os.path.join(framework_dir, model)
                    if os.path.isdir(model_dir):
                        n_subs = len([d for d in os.listdir(model_dir) if d.startswith('sub-')])
                        print(f"  {split}/{framework}/{model}/ ({n_subs} subjects)")

# Specifically check for CLIP image features
clip_candidates = [
    os.path.join(vis_base, 'ThingsTrain', 'pytorch', 'clip'),
    os.path.join(vis_base, 'ThingsTrain', 'pytorch', 'CLIP'),
    os.path.join(vis_base, 'ThingsTrain', 'clip'),
    os.path.join(vis_base, 'ThingsTrain', 'CLIP'),
    os.path.join(vis_base, 'ThingsTrain', 'openai_clip'),
]
print("\\nSearching for CLIP image features:")
found_clip = False
for p in clip_candidates:
    if os.path.exists(p):
        print(f"  FOUND: {p}")
        found_clip = True
    else:
        print(f"  not found: {os.path.basename(p)}")

if not found_clip:
    print("\\n--- ASSESSMENT ---")
    print("CLIP image embeddings are NOT pre-computed in this dataset.")
    print("CORnet-S features are biologically-inspired CNN features, not CLIP.")
    print("Options for image alignment:")
    print("  1. Use CORnet-S features directly as visual alignment target")
    print("  2. Compute CLIP image embeddings from original THINGS stimuli (requires images)")
    print("  3. Download pre-computed CLIP features from NICE-EEG or BraVL repos")"""

VIS_DIM_ANALYSIS_CODE = """\
# =============================================================================
# DIMENSIONALITY ANALYSIS: Effective Rank via SVD
# =============================================================================

from numpy.linalg import svd

print("=" * 60)
print("EFFECTIVE DIMENSIONALITY ANALYSIS")
print("=" * 60)

vis_dim = vis_seen_data.shape[1]
text_dim = text_seen.shape[1]

print(f"\\nRaw feature dimensions:")
print(f"  EEG (brain):            {brain_seen.shape[1]}-D")
print(f"  Text (CLIP):            {text_dim}-D")
print(f"  Visual (CORnet-S full): {vis_dim}-D")
print(f"  Visual (current slice): 100-D")

# SVD-based effective rank
n_svd_samples = min(2000, vis_seen_data.shape[0])
np.random.seed(42)
svd_idx = np.random.choice(vis_seen_data.shape[0], n_svd_samples, replace=False)

_, S_vis, _ = svd(vis_seen_data[svd_idx], full_matrices=False)
s_cumvar_vis = np.cumsum(S_vis**2) / np.sum(S_vis**2) * 100

_, S_text, _ = svd(text_seen.numpy()[svd_idx], full_matrices=False)
s_cumvar_text = np.cumsum(S_text**2) / np.sum(S_text**2) * 100

_, S_brain, _ = svd(brain_seen.numpy()[svd_idx], full_matrices=False)
s_cumvar_brain = np.cumsum(S_brain**2) / np.sum(S_brain**2) * 100

for name, cumvar in [("Visual (CORnet-S)", s_cumvar_vis),
                     ("Text (CLIP)", s_cumvar_text),
                     ("EEG (Brain)", s_cumvar_brain)]:
    eff_95 = np.searchsorted(cumvar, 95) + 1
    eff_99 = np.searchsorted(cumvar, 99) + 1
    print(f"\\n  {name}:")
    print(f"    95% variance at {eff_95} dimensions")
    print(f"    99% variance at {eff_99} dimensions")

print(f"\\n--- IMPLICATIONS ---")
print(f"Higher effective rank = richer supervision signal for alignment.")
print(f"If visual features have higher effective rank than text, they provide")
print(f"more information per sample for the encoder to learn from.")"""

VIS_SUMMARY_MD = """\
---

## Visual Feature Investigation: Summary

**Key Findings** (fill in after running cells above):
1. CORnet-S PCA features: full dimensionality = ? (check cell output)
2. Currently using 100/N components — explained variance = ?%
3. Visual vs text inter-class separation: which has lower cosine similarity?
4. EEG-Visual vs EEG-Text correlation: which modality is EEG more correlated with?
5. CLIP image features: available in dataset? If not, what are the options?

**Next Steps:**
- **Option A**: Replace text alignment with CORnet-S visual alignment (minimal code change — swap TextProjector input from 512-D to N-D)
- **Option B**: Compute CLIP image embeddings from THINGS stimuli on Colab (requires image access + CLIP model)
- **Option C**: Multi-modal alignment: EEG -> (image + text) jointly"""

VIS_SUMMARY_CODE = """\
# =============================================================================
# VISUAL FEATURE INVESTIGATION SUMMARY
# =============================================================================

print("=" * 60)
print("VISUAL FEATURE INVESTIGATION SUMMARY")
print("=" * 60)

print(f"\\n1. CORnet-S features: {vis_dim}-D total, using {100} PCA components")
print(f"   100 dims explain {pct_cumvar[min(99, len(pct_cumvar)-1)]:.1f}% variance")

print(f"\\n2. Text features: {text_dim}-D")

print(f"\\n3. Inter-class separation (mean cosine sim, lower = better):")
print(f"   Visual: {vis_sims.mean():.4f}")
print(f"   Text:   {text_sims.mean():.4f}")

print(f"\\n4. Cross-modal correlation with EEG (higher = easier alignment):")
print(f"   EEG-Visual: r = {eeg_vis_corrs.mean():.4f}")
print(f"   EEG-Text:   r = {eeg_text_corrs.mean():.4f}")

clip_status = "FOUND" if found_clip else "NOT available locally"
print(f"\\n5. CLIP image features: {clip_status}")

print("\\n" + "=" * 60)"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"{'=' * 60}")
    print(f"CREATE CLEAN NOTEBOOK")
    print(f"{'=' * 60}")
    print(f"Source: {OLD_NOTEBOOK}")
    print(f"Target: {NEW_NOTEBOOK}")
    print()

    # 1. Load old notebook
    with open(OLD_NOTEBOOK, 'r') as f:
        old_nb = json.load(f)
    old_cells = old_nb['cells']
    print(f"Loaded {len(old_cells)} cells from source notebook.")

    # 2. Define cell mapping: (old_index, modification_func_or_None)
    cell_map = [
        # Section 0: Setup (old 0-6)
        (0, modify_cell_0),
        (1, None), (2, None), (3, None), (4, None), (5, None), (6, None),
        # Section 1: Data Loading (old 7-9)
        (7, modify_cell_7),
        (8, None),
        (9, None),
        # Section 2: Data Exploration (old 10-16)
        (10, None), (11, None), (12, None), (13, None),
        (14, modify_cell_14),
        (15, None), (16, None),
        # Section 3: Baseline [A] (old 17-22)
        (17, None), (18, None), (19, None), (20, None), (21, None), (22, None),
        # Section 4: GZSL Baseline (old 23-27)
        (23, None), (24, None), (25, None), (26, None), (27, None),
        # Section 5: CLIP Encoder (old 28-37)
        (28, None), (29, None), (30, None), (31, None), (32, None),
        (33, None), (34, None), (35, None), (36, None), (37, None),
        # Section 6: cWGAN-GP (old 38-48)
        (38, None), (39, None), (40, None), (41, None), (42, None),
        (43, None), (44, None), (45, None), (46, None), (47, None), (48, None),
        # Section 7: GZSL Classifier (old 62-74)
        (62, None), (63, None), (64, None), (65, None), (66, None),
        (67, None), (68, None), (69, None), (70, None), (71, None),
        (72, None), (73, None), (74, modify_cell_74),
        # Section 8: Eval Harness (old 97-100)
        (97, None), (98, None), (99, None), (100, None),
    ]

    # 3. Extract and modify cells
    new_cells = []
    for old_idx, mod_func in cell_map:
        if old_idx >= len(old_cells):
            raise RuntimeError(f"Old cell index {old_idx} out of range (notebook has {len(old_cells)} cells)")
        cell = copy.deepcopy(old_cells[old_idx])
        # Clear outputs and execution count for a clean notebook
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
        if mod_func is not None:
            mod_func(cell)
        new_cells.append(cell)

    n_ported = len(new_cells)
    print(f"Ported {n_ported} cells from source.")

    # 4. Add new cells

    # Harness eval bridge cell
    new_cells.append(make_code_cell(HARNESS_EVAL_CODE))

    # Section 9: Visual Feature Investigation
    new_cells.append(make_markdown_cell(VIS_HEADER_MD))
    new_cells.append(make_code_cell(VIS_INSPECT_CODE))
    new_cells.append(make_code_cell(VIS_PCA_SPECTRUM_CODE))
    new_cells.append(make_code_cell(VIS_COMPARE_CODE))
    new_cells.append(make_code_cell(VIS_CROSSMODAL_CODE))
    new_cells.append(make_code_cell(VIS_CLIP_CHECK_CODE))
    new_cells.append(make_code_cell(VIS_DIM_ANALYSIS_CODE))
    new_cells.append(make_markdown_cell(VIS_SUMMARY_MD))
    new_cells.append(make_code_cell(VIS_SUMMARY_CODE))

    n_new = len(new_cells) - n_ported
    print(f"Added {n_new} new cells.")

    # 5. Assemble new notebook
    new_nb = {
        'nbformat': old_nb['nbformat'],
        'nbformat_minor': old_nb['nbformat_minor'],
        'metadata': copy.deepcopy(old_nb['metadata']),
        'cells': new_cells,
    }

    # 6. Write
    with open(NEW_NOTEBOOK, 'w') as f:
        json.dump(new_nb, f, indent=1)

    total = len(new_cells)
    removed = len(old_cells) - n_ported
    print(f"\n{'=' * 60}")
    print(f"DONE")
    print(f"{'=' * 60}")
    print(f"  New notebook: {NEW_NOTEBOOK}")
    print(f"  Total cells:  {total} (ported {n_ported} + added {n_new})")
    print(f"  Removed:      {removed} cells from original")
    print(f"  Old notebook:  {OLD_NOTEBOOK} (unchanged)")

    # 7. Verification summary
    print(f"\n--- Cell Structure ---")
    sections = [
        ("Setup", 0, 6),
        ("Data Loading (with label fix)", 7, 9),
        ("Data Exploration", 10, 16),
        ("Baseline [A]", 17, 22),
        ("GZSL Baseline", 23, 27),
        ("CLIP Encoder", 28, 37),
        ("cWGAN-GP", 38, 48),
        ("GZSL Classifier [A+B]", 49, 61),
        ("Eval Harness", 62, 66),
        ("Visual Feature Investigation", 67, total - 1),
    ]
    for name, start, end in sections:
        print(f"  Cells {start:2d}-{end:2d}: {name}")


if __name__ == '__main__':
    main()
