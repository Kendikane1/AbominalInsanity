"""
=============================================================================
GZSL EEG Classification - Baseline Model Implementation
=============================================================================
COMP2261 Machine Learning Coursework

This file contains all the code cells to be added to the notebook:
  COMP2261_ArizMLCW.ipynb

INSTRUCTIONS:
1. Open the notebook in Colab/Jupyter
2. Run all existing cells first (to load data)
3. Add new cells and paste the code sections below
4. Run the new cells in order

Each section below corresponds to a separate cell in the notebook.
=============================================================================
"""

# =============================================================================
# CELL 1: MARKDOWN - Pipeline Introduction
# =============================================================================
"""
---

# GZSL Pipeline: Baseline Model and Advanced Methods

The following sections implement:
1. **Baseline Model [A]**: Logistic Regression on raw EEG features
2. **Brain-Text CLIP Encoder** (future)
3. **cWGAN-GP for Embedding Synthesis** (future)
4. **GZSL Classifier [A+B]** (future)

---
"""

# =============================================================================
# CELL 2: MARKDOWN - Data Structure Summary Header
# =============================================================================
"""
## Data Structure Summary

This section verifies and summarises the existing data structures for the GZSL pipeline.
"""

# =============================================================================
# CELL 3: CODE - Data Structure Summary
# =============================================================================
# Copy everything below this line into a code cell:

import pandas as pd
import numpy as np

# Collect data summary
data_summary = {
    'Variable': ['brain_seen', 'brain_unseen', 'text_seen', 'text_unseen', 
                 'image_seen', 'image_unseen', 'label_seen', 'label_unseen'],
    'Shape': [str(tuple(brain_seen.shape)), str(tuple(brain_unseen.shape)),
              str(tuple(text_seen.shape)), str(tuple(text_unseen.shape)),
              str(tuple(image_seen.shape)), str(tuple(image_unseen.shape)),
              str(tuple(label_seen.shape)), str(tuple(label_unseen.shape))],
    'Dtype': [str(brain_seen.dtype), str(brain_unseen.dtype),
              str(text_seen.dtype), str(text_unseen.dtype),
              str(image_seen.dtype), str(image_unseen.dtype),
              str(label_seen.dtype), str(label_unseen.dtype)]
}

df_summary = pd.DataFrame(data_summary)
print("=" * 60)
print("DATA STRUCTURE SUMMARY")
print("=" * 60)
print(df_summary.to_string(index=False))
print()

# Label statistics
unique_seen = torch.unique(label_seen)
unique_unseen = torch.unique(label_unseen)

print(f"Seen classes: {len(unique_seen)} unique labels (range: {unique_seen.min().item()} to {unique_seen.max().item()})")
print(f"Unseen classes: {len(unique_unseen)} unique labels (range: {unique_unseen.min().item()} to {unique_unseen.max().item()})")
print()
print(f"Samples per seen class (approx): {brain_seen.shape[0] / len(unique_seen):.1f}")
print(f"Samples per unseen class (approx): {brain_unseen.shape[0] / len(unique_unseen):.1f}")


# =============================================================================
# CELL 4: MARKDOWN - Class Distribution Header
# =============================================================================
"""
## Data Exploration: Class Distributions

Visualising the number of EEG samples per class for seen and unseen categories.
"""

# =============================================================================
# CELL 5: CODE - Class Distribution Histogram
# =============================================================================
# Copy everything below this line into a code cell:

import matplotlib.pyplot as plt
import os

# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

# Count samples per class for seen and unseen
label_seen_np = label_seen.numpy().flatten()
label_unseen_np = label_unseen.numpy().flatten()

seen_class_counts = np.bincount(label_seen_np)[1:]  # Skip 0 since labels are 1-indexed
unseen_class_counts = np.bincount(label_unseen_np)[1:]

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Seen classes histogram
axes[0].hist(seen_class_counts, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Samples per Class', fontsize=12)
axes[0].set_ylabel('Number of Classes', fontsize=12)
axes[0].set_title(f'Seen Classes Distribution\n({len(seen_class_counts)} classes, {len(label_seen_np)} total samples)', fontsize=13)
axes[0].axvline(np.mean(seen_class_counts), color='red', linestyle='--', label=f'Mean: {np.mean(seen_class_counts):.1f}')
axes[0].legend()

# Unseen classes histogram
axes[1].hist(unseen_class_counts, bins=30, color='darkorange', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Samples per Class', fontsize=12)
axes[1].set_ylabel('Number of Classes', fontsize=12)
axes[1].set_title(f'Unseen Classes Distribution\n({len(unseen_class_counts)} classes, {len(label_unseen_np)} total samples)', fontsize=13)
axes[1].axvline(np.mean(unseen_class_counts), color='red', linestyle='--', label=f'Mean: {np.mean(unseen_class_counts):.1f}')
axes[1].legend()

plt.tight_layout()
plt.savefig('figures/class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nFigure saved to: figures/class_distribution.png")


# =============================================================================
# CELL 6: MARKDOWN - EEG Norm Header
# =============================================================================
"""
## Data Exploration: EEG Feature Norms

Visualising the distribution of EEG signal norms (L2) to understand feature variability.
"""

# =============================================================================
# CELL 7: CODE - EEG Feature Norm Distribution
# =============================================================================
# Copy everything below this line into a code cell:

# Compute L2 norms for each sample
brain_seen_np = brain_seen.numpy()
brain_unseen_np = brain_unseen.numpy()

norms_seen = np.linalg.norm(brain_seen_np, axis=1)
norms_unseen = np.linalg.norm(brain_unseen_np, axis=1)

# Create figure
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(norms_seen, bins=50, alpha=0.6, label=f'Seen (n={len(norms_seen)})', color='steelblue', edgecolor='black')
ax.hist(norms_unseen, bins=50, alpha=0.6, label=f'Unseen (n={len(norms_unseen)})', color='darkorange', edgecolor='black')

ax.set_xlabel('L2 Norm of EEG Features', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Distribution of EEG Feature Norms (Seen vs Unseen)', fontsize=13)
ax.legend()

# Add statistics
stats_text = f'Seen: μ={np.mean(norms_seen):.2f}, σ={np.std(norms_seen):.2f}\n'
stats_text += f'Unseen: μ={np.mean(norms_unseen):.2f}, σ={np.std(norms_unseen):.2f}'
ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('figures/eeg_norm_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nFigure saved to: figures/eeg_norm_distribution.png")


# =============================================================================
# CELL 8: MARKDOWN - Baseline Model Header
# =============================================================================
"""
---

## Baseline Model A: Logistic Regression on Raw EEG

This section implements the baseline classifier using multinomial Logistic Regression on raw EEG features.

**Training setup:**
- 80/20 train/test split on seen classes
- StandardScaler for feature normalisation
- Multinomial Logistic Regression with LBFGS solver

**Metrics:**
- Accuracy
- Macro F1-score
"""

# =============================================================================
# CELL 9: CODE - Data Preparation
# =============================================================================
# Copy everything below this line into a code cell:

import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

# -----------------------------------------------------------------------------
# 1. Set random seeds for reproducibility
# -----------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Random seed set to: {SEED}")

# -----------------------------------------------------------------------------
# 2. Prepare data: Convert to numpy and create train/test split
# -----------------------------------------------------------------------------
X_seen = brain_seen.numpy()  # Already flattened to (n_samples, 561)
y_seen = label_seen.numpy().flatten()  # Flatten to 1D

# 80/20 train/test split, stratified by class
X_train, X_test, y_train, y_test = train_test_split(
    X_seen, y_seen, 
    test_size=0.2, 
    random_state=SEED, 
    stratify=y_seen
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Feature dimension: {X_train.shape[1]}")

# -----------------------------------------------------------------------------
# 3. Standardise features (Z-score normalisation)
# -----------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nFeatures standardised (mean=0, std=1)")


# =============================================================================
# CELL 10: CODE - Train Baseline Model
# =============================================================================
# Copy everything below this line into a code cell:
# Note: Training may take a few minutes due to the large number of classes (1654)

print("Training Logistic Regression classifier...")
print(f"Number of classes: {len(np.unique(y_train))}")
print("This may take a few minutes...\n")

# Train multinomial logistic regression
baseline_clf = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    random_state=SEED,
    n_jobs=-1,  # Use all CPU cores
    verbose=1
)

baseline_clf.fit(X_train_scaled, y_train)
print("\nTraining complete!")


# =============================================================================
# CELL 11: CODE - Evaluate Baseline Model
# =============================================================================
# Copy everything below this line into a code cell:

# Predictions on test set
y_pred = baseline_clf.predict(X_test_scaled)

# Compute metrics
baseline_accuracy = accuracy_score(y_test, y_pred)
baseline_macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

# Store results in dictionary (for later comparison with GZSL model)
baseline_results = {
    'model': 'Baseline A (Logistic Regression on Raw EEG)',
    'accuracy': baseline_accuracy,
    'macro_f1': baseline_macro_f1,
    'n_train': X_train.shape[0],
    'n_test': X_test.shape[0],
    'n_classes': len(np.unique(y_train)),
    'n_features': X_train.shape[1]
}

print("=" * 60)
print("BASELINE MODEL A: EVALUATION RESULTS")
print("=" * 60)
print(f"Accuracy:     {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
print(f"Macro F1:     {baseline_macro_f1:.4f}")
print("=" * 60)


# =============================================================================
# CELL 12: MARKDOWN - Summary Header
# =============================================================================
"""
---

## Summary: Baseline Model and Data Exploration

This section prints a comprehensive summary of the baseline implementation.
"""

# =============================================================================
# CELL 13: CODE - Final Summary
# =============================================================================
# Copy everything below this line into a code cell:

print("="*70)
print("SUMMARY: BASELINE MODEL [A] IMPLEMENTATION")
print("="*70)

print("\n### 1. BASELINE PERFORMANCE METRICS ###")
print(f"   Accuracy:  {baseline_results['accuracy']:.4f} ({baseline_results['accuracy']*100:.2f}%)")
print(f"   Macro F1:  {baseline_results['macro_f1']:.4f}")

print("\n### 2. KEY ARRAY SHAPES ###")
print(f"   brain_seen:    {tuple(brain_seen.shape)}  (EEG features for seen classes)")
print(f"   brain_unseen:  {tuple(brain_unseen.shape)}  (EEG features for unseen classes)")
print(f"   text_seen:     {tuple(text_seen.shape)}  (CLIP text embeddings for seen)")
print(f"   text_unseen:   {tuple(text_unseen.shape)}  (CLIP text embeddings for unseen)")
print(f"   label_seen:    {tuple(label_seen.shape)}  (class labels for seen)")
print(f"   label_unseen:  {tuple(label_unseen.shape)}  (class labels for unseen)")

print("\n### 3. ASSUMPTIONS MADE ###")
print("   1. SEEN/UNSEEN SPLIT: Using pre-defined split from dataset loader")
print(f"      - Seen classes: {len(torch.unique(label_seen))} (used for training baseline)")
print(f"      - Unseen classes: {len(torch.unique(label_unseen))} (reserved for GZSL evaluation)")
print("   2. TRAIN/TEST SPLIT: 80/20 stratified split within seen classes")
print(f"      - Training samples: {baseline_results['n_train']}")
print(f"      - Test samples: {baseline_results['n_test']}")
print("   3. FEATURE PREPROCESSING: StandardScaler (Z-score normalisation)")
print("   4. LABELS: 1-indexed (labels start from 1, not 0)")
print("   5. RANDOM SEED: 42 (for reproducibility)")

print("\n### 4. FIGURES GENERATED ###")
print("   - figures/class_distribution.png")
print("   - figures/eeg_norm_distribution.png")

print("\n" + "="*70)
print("NEXT STEPS: Implement Brain-Text CLIP Encoder")
print("="*70)
