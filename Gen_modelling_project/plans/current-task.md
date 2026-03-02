# Current Task: Upstream Bottleneck Diagnostics

**Status:** ACTIVE
**Prerequisite reading:** `@CLAUDE.md`, `@context/inverse-bias-fix-session-debrief.md`, `@context/phase1_analysis.md`
**Depends on:** Notebook cleanup (DONE — notebook is now 108 cells), Phase 0+1 (DONE — eval harness + sample balancing in cells 97–107)

---

## Context

Phase 1 (sample count balancing) solved the inverse bias — routing rate collapsed from 99.7% to ~9%, weight norms converged. But it revealed a deeper problem: **discriminative power is near-floor** (H ≈ 0.4%, AccS ≈ 2.9%, AccU ≈ 0.2%).

The original 5-phase plan (Phases 2–5) assumed the representations were decent but miscalibrated. Phase 1 proved calibration was the easy part. The hard part is that the CLIP embeddings and/or WGAN-GP synthetics don't carry enough class-discriminative signal.

Before investing in Phases 2–5, we need to **identify WHERE the bottleneck is**. There are three candidate failure points:

| ID | Hypothesis | Question |
|----|-----------|----------|
| A | CLIP encoder is weak | Do 64-dim CLIP embeddings of seen EEG carry enough signal for 1654-way classification? |
| B | WGAN-GP generator is weak | Do synthetic unseen embeddings carry class identity, or are they mode-collapsed junk? |
| C | Text prototypes are too similar | Do 200 unseen prototypes in R^64 have enough spread for a generator to produce discriminable embeddings? |

---

## What to Implement

Create `helper_files/add_upstream_diagnostics.py` that appends **6 new cells** to the notebook (after the current cell 107).

### Cell 1: Markdown Header

```markdown
---

# Upstream Bottleneck Diagnostics

**Goal:** Identify where the representation bottleneck lies — CLIP encoder, WGAN-GP generator, or text prototypes.

Phase 1 fixed calibration (routing rate ~9%). Discriminative power is near-floor (H ≈ 0.4%).
Three diagnostics isolate the bottleneck:
- **Diag A**: Seen-only classification — is the CLIP encoder learning useful features?
- **Diag B**: Unseen-only classification — is the generator producing class-discriminable synthetics?
- **Diag C**: Prototype spread — are the text prototypes separable in R^64?
```

### Cell 2: Diagnostic A — Seen-Only 1654-Way Classification

**Purpose:** Measure how much class-discriminative signal the CLIP brain encoder captures for seen classes, in isolation from the GZSL machinery.

```python
# =============================================================================
# DIAGNOSTIC A: SEEN-ONLY 1654-WAY CLASSIFICATION
# =============================================================================
# Question: Does f_b (CLIP brain encoder) produce embeddings that carry
# enough class-discriminative signal for seen classes?
#
# Method: Train LogReg on E_train_seen, test on E_test_seen.
# This is a pure encoder quality test — no generator, no unseen classes.
# =============================================================================

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)

# Load cached embeddings
E_train_seen = np.load('cached_arrays/E_train_seen.npy')
E_test_seen = np.load('cached_arrays/E_test_seen.npy')
y_train_seen = np.load('cached_arrays/y_train_seen.npy')
y_test_seen = np.load('cached_arrays/y_test_seen.npy')

# Standardise (fit on train only)
scaler = StandardScaler()
E_train_s = scaler.fit_transform(E_train_seen)
E_test_s = scaler.transform(E_test_seen)

n_classes_seen = len(np.unique(y_train_seen))
random_baseline = 1.0 / n_classes_seen

print(f"Seen-only classification: {n_classes_seen} classes")
print(f"Train: {E_train_s.shape}, Test: {E_test_s.shape}")
print(f"Random baseline: {random_baseline:.4f} ({random_baseline*100:.2f}%)")
print()

# Train LogReg
clf_seen = LogisticRegression(
    max_iter=2000, solver='lbfgs', C=1.0, random_state=SEED,
    multi_class='multinomial', n_jobs=-1
)
clf_seen.fit(E_train_s, y_train_seen)

# Evaluate
y_pred = clf_seen.predict(E_test_s)
acc_top1 = accuracy_score(y_test_seen, y_pred)

# Top-5 accuracy
y_proba = clf_seen.predict_proba(E_test_s)
acc_top5 = top_k_accuracy_score(y_test_seen, y_proba, k=5, labels=clf_seen.classes_)

# Top-10 accuracy
acc_top10 = top_k_accuracy_score(y_test_seen, y_proba, k=10, labels=clf_seen.classes_)

# Signal-above-random ratio
signal_ratio = acc_top1 / random_baseline

print("=" * 60)
print("DIAGNOSTIC A RESULTS: Seen-Only Classification")
print("=" * 60)
print(f"  Top-1 Accuracy:  {acc_top1:.4f} ({acc_top1*100:.2f}%)")
print(f"  Top-5 Accuracy:  {acc_top5:.4f} ({acc_top5*100:.2f}%)")
print(f"  Top-10 Accuracy: {acc_top10:.4f} ({acc_top10*100:.2f}%)")
print(f"  Random baseline: {random_baseline:.4f} ({random_baseline*100:.2f}%)")
print(f"  Signal ratio:    {signal_ratio:.1f}x above random")
print()

# Interpretation
if acc_top1 > 0.05:
    print(">>> CLIP encoder captures MODERATE signal. Bottleneck likely downstream.")
elif acc_top1 > 0.01:
    print(">>> CLIP encoder captures WEAK signal. Encoder improvement may help.")
else:
    print(">>> CLIP encoder captures MINIMAL signal. Encoder is a primary bottleneck.")
```

### Cell 3: Diagnostic B — Unseen-Only 200-Way Classification (Synthetic → Real)

**Purpose:** Test if the WGAN-GP generator produces embeddings that carry class identity. Train a classifier on synthetic unseen, test on real unseen EEG embeddings.

```python
# =============================================================================
# DIAGNOSTIC B: UNSEEN-ONLY 200-WAY CLASSIFICATION
# =============================================================================
# Question: Does G(z, s_c) produce embeddings that carry class identity?
#
# Method 1 (B1): Train LogReg on E_synth_unseen, test on E_unseen (real).
#   This is the transfer test — can synthetics classify real data?
#
# Method 2 (B2): 5-fold cross-validation on E_synth_unseen only.
#   This tests internal discriminability — are synthetics at least
#   distinct from each other across classes?
# =============================================================================

from sklearn.model_selection import cross_val_score

E_unseen = np.load('cached_arrays/E_unseen.npy')
E_synth_unseen = np.load('cached_arrays/E_synth_unseen.npy')
y_unseen = np.load('cached_arrays/y_unseen.npy')
y_synth_unseen = np.load('cached_arrays/y_synth_unseen.npy')

n_classes_unseen = len(np.unique(y_unseen))
random_baseline_u = 1.0 / n_classes_unseen

print(f"Unseen-only classification: {n_classes_unseen} classes")
print(f"Synthetic train: {E_synth_unseen.shape}")
print(f"Real unseen test: {E_unseen.shape}")
print(f"Random baseline: {random_baseline_u:.4f} ({random_baseline_u*100:.2f}%)")
print()

# --- B1: Train on synthetic, test on real ---
scaler_u = StandardScaler()
E_synth_s = scaler_u.fit_transform(E_synth_unseen)
E_unseen_s = scaler_u.transform(E_unseen)

clf_unseen = LogisticRegression(
    max_iter=2000, solver='lbfgs', C=1.0, random_state=SEED,
    multi_class='multinomial', n_jobs=-1
)
clf_unseen.fit(E_synth_s, y_synth_unseen)

y_pred_u = clf_unseen.predict(E_unseen_s)
acc_b1 = accuracy_score(y_unseen, y_pred_u)

y_proba_u = clf_unseen.predict_proba(E_unseen_s)
acc_b1_top5 = top_k_accuracy_score(y_unseen, y_proba_u, k=5, labels=clf_unseen.classes_)

signal_ratio_b1 = acc_b1 / random_baseline_u

# --- B2: Cross-val on synthetics only ---
clf_cv = LogisticRegression(
    max_iter=2000, solver='lbfgs', C=1.0, random_state=SEED,
    multi_class='multinomial', n_jobs=-1
)
cv_scores = cross_val_score(clf_cv, E_synth_s, y_synth_unseen, cv=5, scoring='accuracy')
acc_b2_mean = cv_scores.mean()
acc_b2_std = cv_scores.std()
signal_ratio_b2 = acc_b2_mean / random_baseline_u

print("=" * 60)
print("DIAGNOSTIC B RESULTS: Unseen-Only Classification")
print("=" * 60)
print()
print("B1: Synthetic → Real Transfer")
print(f"  Top-1 Accuracy:  {acc_b1:.4f} ({acc_b1*100:.2f}%)")
print(f"  Top-5 Accuracy:  {acc_b1_top5:.4f} ({acc_b1_top5*100:.2f}%)")
print(f"  Signal ratio:    {signal_ratio_b1:.1f}x above random")
print()
print("B2: Synthetic Internal Discriminability (5-fold CV)")
print(f"  Accuracy:        {acc_b2_mean:.4f} ± {acc_b2_std:.4f}")
print(f"  Signal ratio:    {signal_ratio_b2:.1f}x above random")
print()

# Interpretation
if acc_b2_mean > 0.10 and acc_b1 < 0.02:
    print(">>> Synthetics are internally discriminable but don't transfer to real data.")
    print("    Generator captures some structure but doesn't match real EEG distribution.")
elif acc_b2_mean < 0.02:
    print(">>> Synthetics are NOT internally discriminable. Generator is mode-collapsed.")
    print("    Generator is a primary bottleneck.")
elif acc_b1 > 0.02:
    print(">>> Synthetics transfer reasonably to real data. Generator is not the bottleneck.")
else:
    print(">>> Mixed signal. Check prototype analysis (Diagnostic C) for root cause.")
```

### Cell 4: Diagnostic C — Prototype Spread Analysis

**Purpose:** Measure whether the 200 unseen text prototypes have enough angular separation in R^64 for a generator to produce discriminable embeddings.

```python
# =============================================================================
# DIAGNOSTIC C: PROTOTYPE SPREAD ANALYSIS
# =============================================================================
# Question: Are the 200 unseen text prototypes separable in R^64?
#
# If prototypes cluster tightly (high pairwise cosine), then even a
# perfect generator cannot produce discriminable embeddings because
# G(z, s_c) ≈ G(z, s_{c'}) when s_c ≈ s_{c'}.
#
# Reference: For random unit vectors in R^d, E[cos(u,v)] = 0,
# Var[cos(u,v)] ≈ 1/d. So for d=64, std ≈ 0.125.
# =============================================================================

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

S_unseen = np.load('cached_arrays/S_unseen_prototypes.npy')
S_seen = np.load('cached_arrays/S_seen_prototypes.npy')

print(f"Unseen prototypes: {S_unseen.shape}")
print(f"Seen prototypes: {S_seen.shape}")
print()

# --- Unseen pairwise cosine similarity ---
cos_sim_unseen = cosine_similarity(S_unseen)
n_u = cos_sim_unseen.shape[0]
# Extract upper triangle (off-diagonal)
mask = np.triu(np.ones((n_u, n_u), dtype=bool), k=1)
off_diag_unseen = cos_sim_unseen[mask]

# --- Seen pairwise cosine similarity (for comparison) ---
cos_sim_seen = cosine_similarity(S_seen)
n_s = cos_sim_seen.shape[0]
mask_s = np.triu(np.ones((n_s, n_s), dtype=bool), k=1)
off_diag_seen = cos_sim_seen[mask_s]

# --- Statistics ---
print("=" * 60)
print("DIAGNOSTIC C RESULTS: Prototype Spread")
print("=" * 60)
print()
print(f"{'Metric':<25} {'Unseen (200)':>15} {'Seen (1654)':>15}")
print("-" * 55)
print(f"{'Mean cosine':<25} {off_diag_unseen.mean():>15.4f} {off_diag_seen.mean():>15.4f}")
print(f"{'Median cosine':<25} {np.median(off_diag_unseen):>15.4f} {np.median(off_diag_seen):>15.4f}")
print(f"{'Std cosine':<25} {off_diag_unseen.std():>15.4f} {off_diag_seen.std():>15.4f}")
print(f"{'Min cosine':<25} {off_diag_unseen.min():>15.4f} {off_diag_seen.min():>15.4f}")
print(f"{'Max cosine':<25} {off_diag_unseen.max():>15.4f} {off_diag_seen.max():>15.4f}")
print(f"{'% pairs > 0.9':<25} {(off_diag_unseen > 0.9).mean()*100:>14.1f}% {(off_diag_seen > 0.9).mean()*100:>14.1f}%")
print(f"{'% pairs > 0.95':<25} {(off_diag_unseen > 0.95).mean()*100:>14.1f}% {(off_diag_seen > 0.95).mean()*100:>14.1f}%")
print()
print(f"Random unit vectors in R^64: E[cos] ≈ 0, std ≈ {1/np.sqrt(64):.3f}")
print()

# --- Nearest-neighbour analysis ---
# For each prototype, what's the cosine to its nearest neighbour?
np.fill_diagonal(cos_sim_unseen, -1)  # exclude self
nn_cos_unseen = cos_sim_unseen.max(axis=1)
np.fill_diagonal(cos_sim_seen, -1)
nn_cos_seen = cos_sim_seen.max(axis=1)

print(f"Nearest-neighbour cosine (unseen): mean={nn_cos_unseen.mean():.4f}, min={nn_cos_unseen.min():.4f}, max={nn_cos_unseen.max():.4f}")
print(f"Nearest-neighbour cosine (seen):   mean={nn_cos_seen.mean():.4f}, min={nn_cos_seen.min():.4f}, max={nn_cos_seen.max():.4f}")
print()

# Interpretation
if off_diag_unseen.mean() > 0.8:
    print(">>> CRITICAL: Unseen prototypes are highly clustered (mean cos > 0.8).")
    print("    No generator can produce discriminable embeddings from near-identical conditions.")
    print("    Prototypes are a primary bottleneck.")
elif off_diag_unseen.mean() > 0.5:
    print(">>> WARNING: Unseen prototypes have moderate overlap (mean cos > 0.5).")
    print("    Generator needs to be very expressive to overcome prototype similarity.")
elif off_diag_unseen.mean() > 0.2:
    print(">>> Unseen prototypes have mild overlap. Spread is suboptimal but workable.")
else:
    print(">>> Prototypes are well-separated. Bottleneck is NOT in the semantic space.")
```

### Cell 5: Diagnostic C Visualization

```python
# =============================================================================
# DIAGNOSTIC C: VISUALISATION
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Histogram of pairwise cosine similarities
axes[0].hist(off_diag_unseen, bins=50, alpha=0.7, label='Unseen (200)', density=True, color='coral')
axes[0].hist(off_diag_seen, bins=50, alpha=0.5, label='Seen (1654)', density=True, color='steelblue')
axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Random expectation')
axes[0].set_xlabel('Pairwise Cosine Similarity')
axes[0].set_ylabel('Density')
axes[0].set_title('Prototype Pairwise Cosine Distribution')
axes[0].legend()

# 2. Nearest-neighbour cosine distribution
axes[1].hist(nn_cos_unseen, bins=30, alpha=0.7, label='Unseen', color='coral')
axes[1].hist(nn_cos_seen, bins=30, alpha=0.5, label='Seen', color='steelblue')
axes[1].set_xlabel('Nearest-Neighbour Cosine')
axes[1].set_ylabel('Count')
axes[1].set_title('Nearest-Neighbour Cosine Distribution')
axes[1].legend()

# 3. Cosine similarity heatmap (unseen only, sorted)
# Reset diagonal for heatmap
np.fill_diagonal(cos_sim_unseen, 1.0)
import seaborn as sns
sns.heatmap(cos_sim_unseen, cmap='RdYlBu_r', vmin=-0.2, vmax=1.0,
            square=True, ax=axes[2], cbar_kws={'shrink': 0.8})
axes[2].set_title(f'Unseen Prototype Similarity ({n_u}×{n_u})')
axes[2].set_xlabel('Class index')
axes[2].set_ylabel('Class index')

plt.tight_layout()
plt.savefig('figures/upstream_diag_c_prototypes.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figures/upstream_diag_c_prototypes.png")
```

### Cell 6: Summary & Bottleneck Verdict

```python
# =============================================================================
# UPSTREAM DIAGNOSTICS: SUMMARY & BOTTLENECK VERDICT
# =============================================================================

print("=" * 70)
print("UPSTREAM BOTTLENECK DIAGNOSTICS — SUMMARY")
print("=" * 70)
print()

# Collect results
results = {
    'A_seen_top1': acc_top1,
    'A_seen_top5': acc_top5,
    'A_seen_signal': signal_ratio,
    'B1_transfer_top1': acc_b1,
    'B1_transfer_signal': signal_ratio_b1,
    'B2_internal_acc': acc_b2_mean,
    'B2_internal_signal': signal_ratio_b2,
    'C_mean_cos_unseen': off_diag_unseen.mean(),
    'C_mean_cos_seen': off_diag_seen.mean(),
    'C_nn_cos_unseen': nn_cos_unseen.mean(),
}

print(f"{'Diagnostic':<45} {'Value':>10} {'Signal':>10}")
print("-" * 65)
print(f"{'A: Seen top-1 acc (1654-way)':<45} {acc_top1*100:>9.2f}% {signal_ratio:>9.1f}x")
print(f"{'A: Seen top-5 acc':<45} {acc_top5*100:>9.2f}%")
print(f"{'B1: Synth→Real transfer acc (200-way)':<45} {acc_b1*100:>9.2f}% {signal_ratio_b1:>9.1f}x")
print(f"{'B2: Synth internal CV acc (200-way)':<45} {acc_b2_mean*100:>9.2f}% {signal_ratio_b2:>9.1f}x")
print(f"{'C: Unseen prototype mean cosine':<45} {off_diag_unseen.mean():>10.4f}")
print(f"{'C: Seen prototype mean cosine (ref)':<45} {off_diag_seen.mean():>10.4f}")
print(f"{'C: Unseen NN cosine mean':<45} {nn_cos_unseen.mean():>10.4f}")
print()

# --- Bottleneck verdict ---
print("BOTTLENECK ANALYSIS:")
print("-" * 65)

bottlenecks = []

# Encoder assessment
if acc_top1 < 0.01:
    bottlenecks.append(("CLIP Encoder", "CRITICAL", f"top-1 = {acc_top1*100:.2f}%, barely above random"))
elif acc_top1 < 0.05:
    bottlenecks.append(("CLIP Encoder", "WEAK", f"top-1 = {acc_top1*100:.2f}%, limited signal"))
else:
    bottlenecks.append(("CLIP Encoder", "OK", f"top-1 = {acc_top1*100:.2f}%, reasonable signal"))

# Generator assessment
if acc_b2_mean < 0.02:
    bottlenecks.append(("WGAN-GP Generator", "CRITICAL", f"internal CV = {acc_b2_mean*100:.2f}%, mode-collapsed"))
elif acc_b1 < 0.01:
    bottlenecks.append(("WGAN-GP Generator", "WEAK", f"transfer = {acc_b1*100:.2f}%, poor real-data match"))
else:
    bottlenecks.append(("WGAN-GP Generator", "OK", f"transfer = {acc_b1*100:.2f}%"))

# Prototype assessment
if off_diag_unseen.mean() > 0.8:
    bottlenecks.append(("Text Prototypes", "CRITICAL", f"mean cos = {off_diag_unseen.mean():.3f}, highly clustered"))
elif off_diag_unseen.mean() > 0.5:
    bottlenecks.append(("Text Prototypes", "CONCERNING", f"mean cos = {off_diag_unseen.mean():.3f}, moderate overlap"))
else:
    bottlenecks.append(("Text Prototypes", "OK", f"mean cos = {off_diag_unseen.mean():.3f}, adequate spread"))

for component, severity, detail in bottlenecks:
    print(f"  [{severity:>10}] {component}: {detail}")

print()
print("NEXT STEPS (to be decided by orchestrator based on these results):")
print("  - If Encoder CRITICAL → increase embed_dim, train longer, curriculum learning")
print("  - If Generator CRITICAL → covariance-aware WGAN, more training steps, architecture changes")
print("  - If Prototypes CRITICAL → better text features, larger semantic space, alternative alignment")
print("  - If multiple CRITICAL → address upstream first (prototypes → encoder → generator)")
```

---

## Implementation Notes

### Variable Dependencies

The cells assume these variables exist from earlier notebook execution:

| Variable | Source Cell | Description |
|----------|-----------|-------------|
| `E_train_seen` | Cell 82 (or cached) | CLIP embeddings of seen training data |
| `E_test_seen` | Cell 82 (or cached) | CLIP embeddings of seen test data |
| `E_unseen` | Cell 82 (or cached) | CLIP embeddings of unseen test data |
| `E_synth_unseen` | Cell 82 (or cached) | WGAN-GP synthetic unseen embeddings |
| `y_train_seen` | Cell 82 (or cached) | Seen training labels |
| `y_test_seen` | Cell 82 (or cached) | Seen test labels |
| `y_unseen` | Cell 82 (or cached) | Unseen labels (original, 1-200) |
| `y_synth_unseen` | Cell 82 (or cached) | Synthetic unseen labels (original, 1-200) |
| `S_unseen_prototypes` | Cached | Text prototypes for 200 unseen classes |
| `S_seen_prototypes` | Cached | Text prototypes for 1654 seen classes |

Each diagnostic cell **reloads from `cached_arrays/`** so cells are self-contained and can be re-run independently. Do NOT rely on in-memory variables from prior cells.

### Cached Array Filenames

The filenames are established in earlier cells. Verify they match:
- `cached_arrays/E_train_seen.npy`
- `cached_arrays/E_test_seen.npy`
- `cached_arrays/E_unseen.npy`
- `cached_arrays/E_synth_unseen.npy`
- `cached_arrays/y_train_seen.npy`
- `cached_arrays/y_test_seen.npy`
- `cached_arrays/y_unseen.npy`
- `cached_arrays/y_synth_unseen.npy`
- `cached_arrays/S_unseen_prototypes.npy`
- `cached_arrays/S_seen_prototypes.npy`

### Helper Script

Create `helper_files/add_upstream_diagnostics.py` following the same pattern as the existing helper scripts (e.g., `add_eval_harness.py`). The script should:

1. Read the notebook JSON from `COMP2261_ArizMLCW_with_baseline.ipynb`
2. Append the 6 cells above
3. Write back to the same file
4. Print confirmation with cell count

### Figures

Save to `figures/upstream_diag_c_prototypes.png` at 150 dpi.

---

## What to Report When Done

1. Confirm the helper script was created and run successfully
2. New cell count (should be 108 + 6 = **114**)
3. List cell indices and first lines of the 6 new cells
4. **Do NOT run the cells** — they will be run in Colab where the cached arrays exist. Just confirm the cells were injected correctly.

---

## What NOT to Touch

- All existing cells 0–107 — leave them exactly as they are
- Cached `.npy` files
- The `figures/` directory (new figure will be added when cells run)
- The evaluation harness or Phase 1 cells
