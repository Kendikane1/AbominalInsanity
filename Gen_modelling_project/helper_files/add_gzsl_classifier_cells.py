#!/usr/bin/env python3
"""
Script to add GZSL Classifier [A+B] cells to the notebook.

The GZSL classifier trains on:
- Real seen embeddings (E_train_seen)
- Synthetic unseen embeddings (E_synth_unseen)

Then evaluates on:
- Real seen test embeddings (E_test_seen) -> Acc_seen
- Real unseen embeddings (E_unseen) -> Acc_unseen
- Harmonic mean H
"""

import json
import os

NOTEBOOK_PATH = "COMP2261_ArizMLCW_with_baseline.ipynb"
OUTPUT_PATH = "COMP2261_ArizMLCW_with_baseline.ipynb"

NEW_CELLS = [
    # =========================================================================
    # GZSL CLASSIFIER SECTION
    # =========================================================================
    
    # Cell 1: Markdown Header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "# GZSL Classifier [A+B]: Real Seen + Synthetic Unseen\n",
            "\n",
            "This section trains the final **Generalised Zero-Shot Learning classifier** using:\n",
            "\n",
            "**Training data:**\n",
            "- Real seen embeddings: `E_train_seen` (from CLIP encoder)\n",
            "- Synthetic unseen embeddings: `E_synth_unseen` (from cWGAN-GP)\n",
            "\n",
            "**Evaluation:**\n",
            "- Seen test set: `E_test_seen` → `Acc_seen`\n",
            "- Unseen real data: `E_unseen` → `Acc_unseen`\n",
            "- Harmonic mean: `H = 2 * Acc_seen * Acc_unseen / (Acc_seen + Acc_unseen)`\n",
            "\n",
            "**Comparison with Baseline [A]:** The baseline was trained on raw EEG and could only predict seen classes."
        ]
    },
    
    # Cell 2: Load Data and Prepare
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# GZSL CLASSIFIER: DATA PREPARATION\n",
            "# =============================================================================\n",
            "\n",
            "import numpy as np\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.metrics import accuracy_score, f1_score, confusion_matrix\n",
            "import matplotlib.pyplot as plt\n",
            "import random\n",
            "\n",
            "# Set seed for reproducibility\n",
            "SEED = 42\n",
            "np.random.seed(SEED)\n",
            "random.seed(SEED)\n",
            "\n",
            "# Load cached embeddings\n",
            "E_train_seen = np.load('cached_arrays/E_train_seen.npy')\n",
            "E_test_seen = np.load('cached_arrays/E_test_seen.npy')\n",
            "E_unseen = np.load('cached_arrays/E_unseen.npy')\n",
            "E_synth_unseen = np.load('cached_arrays/E_synth_unseen.npy')\n",
            "\n",
            "# Load labels\n",
            "y_train_seen = np.load('cached_arrays/y_train_seen.npy')\n",
            "y_test_seen = np.load('cached_arrays/y_test_seen.npy')\n",
            "y_unseen = np.load('cached_arrays/y_unseen.npy')\n",
            "y_synth_unseen = np.load('cached_arrays/y_synth_unseen.npy')\n",
            "\n",
            "print(\"Loaded embeddings:\")\n",
            "print(f\"  E_train_seen:   {E_train_seen.shape}\")\n",
            "print(f\"  E_test_seen:    {E_test_seen.shape}\")\n",
            "print(f\"  E_unseen:       {E_unseen.shape}\")\n",
            "print(f\"  E_synth_unseen: {E_synth_unseen.shape}\")"
        ]
    },
    
    # Cell 3: Combine Training Data
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# COMBINE REAL SEEN + SYNTHETIC UNSEEN FOR TRAINING\n",
            "# =============================================================================\n",
            "\n",
            "# Combine training data\n",
            "X_train_gzsl = np.vstack([E_train_seen, E_synth_unseen])\n",
            "y_train_gzsl = np.concatenate([y_train_seen, y_synth_unseen])\n",
            "\n",
            "print(\"GZSL Training Data:\")\n",
            "print(f\"  Real seen:      {len(E_train_seen)} samples, {len(np.unique(y_train_seen))} classes\")\n",
            "print(f\"  Synthetic unseen: {len(E_synth_unseen)} samples, {len(np.unique(y_synth_unseen))} classes\")\n",
            "print(f\"  Combined:       {len(X_train_gzsl)} samples, {len(np.unique(y_train_gzsl))} classes\")\n",
            "\n",
            "# Verify no label overlap between seen and unseen\n",
            "seen_labels = set(np.unique(y_train_seen))\n",
            "unseen_labels = set(np.unique(y_synth_unseen))\n",
            "overlap = seen_labels.intersection(unseen_labels)\n",
            "print(f\"\\nLabel overlap check: {len(overlap)} overlapping labels\")\n",
            "if len(overlap) == 0:\n",
            "    print(\"  ✓ No overlap — seen and unseen label spaces are disjoint.\")"
        ]
    },
    
    # Cell 4: Train GZSL Classifier
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# TRAIN GZSL LOGISTIC REGRESSION CLASSIFIER\n",
            "# =============================================================================\n",
            "\n",
            "print(\"Training GZSL classifier (Logistic Regression)...\")\n",
            "print(f\"  Total classes: {len(np.unique(y_train_gzsl))}\")\n",
            "print(f\"  Total samples: {len(X_train_gzsl)}\")\n",
            "\n",
            "gzsl_clf = LogisticRegression(\n",
            "    multi_class='multinomial',\n",
            "    solver='lbfgs',\n",
            "    max_iter=1000,\n",
            "    random_state=SEED,\n",
            "    n_jobs=-1,\n",
            "    verbose=1\n",
            ")\n",
            "\n",
            "gzsl_clf.fit(X_train_gzsl, y_train_gzsl)\n",
            "print(\"\\nTraining complete!\")"
        ]
    },
    
    # Cell 5: Evaluate on Seen Classes
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# EVALUATE ON SEEN CLASSES (E_test_seen)\n",
            "# =============================================================================\n",
            "\n",
            "y_pred_seen = gzsl_clf.predict(E_test_seen)\n",
            "\n",
            "acc_seen_gzsl = accuracy_score(y_test_seen, y_pred_seen)\n",
            "f1_seen_gzsl = f1_score(y_test_seen, y_pred_seen, average='macro', zero_division=0)\n",
            "\n",
            "print(\"=\" * 60)\n",
            "print(\"GZSL EVALUATION: SEEN CLASSES\")\n",
            "print(\"=\" * 60)\n",
            "print(f\"Accuracy (seen):  {acc_seen_gzsl:.4f} ({acc_seen_gzsl*100:.2f}%)\")\n",
            "print(f\"Macro F1 (seen):  {f1_seen_gzsl:.4f}\")"
        ]
    },
    
    # Cell 6: Evaluate on Unseen Classes
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# EVALUATE ON UNSEEN CLASSES (E_unseen — real EEG, never seen during training)\n",
            "# =============================================================================\n",
            "\n",
            "y_pred_unseen = gzsl_clf.predict(E_unseen)\n",
            "\n",
            "acc_unseen_gzsl = accuracy_score(y_unseen, y_pred_unseen)\n",
            "f1_unseen_gzsl = f1_score(y_unseen, y_pred_unseen, average='macro', zero_division=0)\n",
            "\n",
            "print(\"=\" * 60)\n",
            "print(\"GZSL EVALUATION: UNSEEN CLASSES\")\n",
            "print(\"=\" * 60)\n",
            "print(f\"Accuracy (unseen): {acc_unseen_gzsl:.4f} ({acc_unseen_gzsl*100:.2f}%)\")\n",
            "print(f\"Macro F1 (unseen): {f1_unseen_gzsl:.4f}\")"
        ]
    },
    
    # Cell 7: Compute Harmonic Mean
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# COMPUTE HARMONIC MEAN (H)\n",
            "# =============================================================================\n",
            "\n",
            "# H = 2 * Acc_seen * Acc_unseen / (Acc_seen + Acc_unseen)\n",
            "if acc_seen_gzsl + acc_unseen_gzsl > 0:\n",
            "    H_gzsl = 2 * acc_seen_gzsl * acc_unseen_gzsl / (acc_seen_gzsl + acc_unseen_gzsl)\n",
            "else:\n",
            "    H_gzsl = 0.0\n",
            "\n",
            "print(\"=\" * 60)\n",
            "print(\"GZSL HARMONIC MEAN\")\n",
            "print(\"=\" * 60)\n",
            "print(f\"H = 2 * {acc_seen_gzsl:.4f} * {acc_unseen_gzsl:.4f} / ({acc_seen_gzsl:.4f} + {acc_unseen_gzsl:.4f})\")\n",
            "print(f\"H = {H_gzsl:.4f}\")"
        ]
    },
    
    # Cell 8: Store GZSL Results
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# STORE GZSL RESULTS\n",
            "# =============================================================================\n",
            "\n",
            "gzsl_results = {\n",
            "    'model': 'GZSL [A+B] (LR on CLIP embeddings + cWGAN-GP synthetic)',\n",
            "    'acc_seen': acc_seen_gzsl,\n",
            "    'acc_unseen': acc_unseen_gzsl,\n",
            "    'H': H_gzsl,\n",
            "    'macro_f1_seen': f1_seen_gzsl,\n",
            "    'macro_f1_unseen': f1_unseen_gzsl,\n",
            "    'n_train_seen': len(E_train_seen),\n",
            "    'n_train_synth': len(E_synth_unseen),\n",
            "    'n_test_seen': len(E_test_seen),\n",
            "    'n_test_unseen': len(E_unseen),\n",
            "    'n_classes_seen': len(np.unique(y_train_seen)),\n",
            "    'n_classes_unseen': len(np.unique(y_synth_unseen))\n",
            "}\n",
            "\n",
            "print(\"GZSL results stored in 'gzsl_results' dictionary.\")"
        ]
    },
    
    # Cell 9: Markdown - Comparison Header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## Comparison: Baseline [A] vs GZSL [A+B]\n",
            "\n",
            "Comparing the baseline logistic regression (raw EEG, seen classes only) with the GZSL classifier (CLIP embeddings + synthetic unseen)."
        ]
    },
    
    # Cell 10: Comparison Table
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# COMPARISON: BASELINE [A] vs GZSL [A+B]\n",
            "# =============================================================================\n",
            "\n",
            "# Retrieve baseline results (computed earlier in notebook)\n",
            "# baseline_gzsl_results should exist from the GZSL baseline evaluation\n",
            "\n",
            "print(\"=\"*70)\n",
            "print(\"COMPARISON: BASELINE [A] vs GZSL [A+B]\")\n",
            "print(\"=\"*70)\n",
            "print(\"\")\n",
            "print(f\"{'Metric':<20} {'Baseline [A]':>15} {'GZSL [A+B]':>15} {'Improvement':>15}\")\n",
            "print(\"-\"*70)\n",
            "\n",
            "# Seen accuracy\n",
            "base_seen = baseline_gzsl_results['acc_seen']\n",
            "gzsl_seen = gzsl_results['acc_seen']\n",
            "delta_seen = gzsl_seen - base_seen\n",
            "print(f\"{'Acc (seen)':<20} {base_seen:>15.4f} {gzsl_seen:>15.4f} {delta_seen:>+15.4f}\")\n",
            "\n",
            "# Unseen accuracy\n",
            "base_unseen = baseline_gzsl_results['acc_unseen']\n",
            "gzsl_unseen = gzsl_results['acc_unseen']\n",
            "delta_unseen = gzsl_unseen - base_unseen\n",
            "print(f\"{'Acc (unseen)':<20} {base_unseen:>15.4f} {gzsl_unseen:>15.4f} {delta_unseen:>+15.4f}\")\n",
            "\n",
            "# Harmonic mean\n",
            "base_H = baseline_gzsl_results['H']\n",
            "gzsl_H = gzsl_results['H']\n",
            "delta_H = gzsl_H - base_H\n",
            "print(f\"{'Harmonic Mean (H)':<20} {base_H:>15.4f} {gzsl_H:>15.4f} {delta_H:>+15.4f}\")\n",
            "\n",
            "# Macro F1 seen\n",
            "base_f1_seen = baseline_gzsl_results['macro_f1_seen']\n",
            "gzsl_f1_seen = gzsl_results['macro_f1_seen']\n",
            "print(f\"{'F1 (seen)':<20} {base_f1_seen:>15.4f} {gzsl_f1_seen:>15.4f} {gzsl_f1_seen - base_f1_seen:>+15.4f}\")\n",
            "\n",
            "# Macro F1 unseen\n",
            "base_f1_unseen = baseline_gzsl_results['macro_f1_unseen']\n",
            "gzsl_f1_unseen = gzsl_results['macro_f1_unseen']\n",
            "print(f\"{'F1 (unseen)':<20} {base_f1_unseen:>15.4f} {gzsl_f1_unseen:>15.4f} {gzsl_f1_unseen - base_f1_unseen:>+15.4f}\")\n",
            "\n",
            "print(\"-\"*70)"
        ]
    },
    
    # Cell 11: Bar Chart Comparison
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# VISUALIZATION: BASELINE vs GZSL COMPARISON\n",
            "# =============================================================================\n",
            "\n",
            "metrics = ['Acc (seen)', 'Acc (unseen)', 'Harmonic Mean']\n",
            "baseline_vals = [baseline_gzsl_results['acc_seen'], baseline_gzsl_results['acc_unseen'], baseline_gzsl_results['H']]\n",
            "gzsl_vals = [gzsl_results['acc_seen'], gzsl_results['acc_unseen'], gzsl_results['H']]\n",
            "\n",
            "x = np.arange(len(metrics))\n",
            "width = 0.35\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(10, 6))\n",
            "bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline [A]', color='steelblue')\n",
            "bars2 = ax.bar(x + width/2, gzsl_vals, width, label='GZSL [A+B]', color='darkorange')\n",
            "\n",
            "# Add value labels on bars\n",
            "for bar in bars1:\n",
            "    height = bar.get_height()\n",
            "    ax.annotate(f'{height:.3f}',\n",
            "                xy=(bar.get_x() + bar.get_width() / 2, height),\n",
            "                xytext=(0, 3), textcoords=\"offset points\",\n",
            "                ha='center', va='bottom', fontsize=10)\n",
            "for bar in bars2:\n",
            "    height = bar.get_height()\n",
            "    ax.annotate(f'{height:.3f}',\n",
            "                xy=(bar.get_x() + bar.get_width() / 2, height),\n",
            "                xytext=(0, 3), textcoords=\"offset points\",\n",
            "                ha='center', va='bottom', fontsize=10)\n",
            "\n",
            "ax.set_ylabel('Score', fontsize=12)\n",
            "ax.set_title('GZSL Performance: Baseline [A] vs Customised Model [A+B]', fontsize=14)\n",
            "ax.set_xticks(x)\n",
            "ax.set_xticklabels(metrics, fontsize=12)\n",
            "ax.legend(fontsize=11)\n",
            "ax.set_ylim(0, max(max(baseline_vals), max(gzsl_vals)) * 1.2)\n",
            "ax.grid(axis='y', alpha=0.3)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('figures/gzsl_comparison.png', dpi=150, bbox_inches='tight')\n",
            "plt.show()\n",
            "\n",
            "print(f\"Saved: figures/gzsl_comparison.png\")"
        ]
    },
    
    # Cell 12: Confusion Matrix Analysis
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# PREDICTION DISTRIBUTION ANALYSIS\n",
            "# =============================================================================\n",
            "\n",
            "# Check if GZSL classifier predicts unseen labels (which baseline couldn't)\n",
            "pred_unseen_labels = set(np.unique(y_pred_unseen))\n",
            "true_unseen_labels = set(np.unique(y_unseen))\n",
            "seen_label_set = set(np.unique(y_train_seen))\n",
            "\n",
            "# How many predictions are in the unseen label space?\n",
            "n_pred_in_unseen = sum(1 for p in y_pred_unseen if p in true_unseen_labels)\n",
            "n_pred_in_seen = sum(1 for p in y_pred_unseen if p in seen_label_set)\n",
            "\n",
            "print(\"=\" * 60)\n",
            "print(\"PREDICTION DISTRIBUTION ON UNSEEN DATA\")\n",
            "print(\"=\" * 60)\n",
            "print(f\"Total predictions on unseen data: {len(y_pred_unseen)}\")\n",
            "print(f\"  Predictions in unseen label space: {n_pred_in_unseen} ({100*n_pred_in_unseen/len(y_pred_unseen):.1f}%)\")\n",
            "print(f\"  Predictions in seen label space:   {n_pred_in_seen} ({100*n_pred_in_seen/len(y_pred_unseen):.1f}%)\")\n",
            "print(\"\")\n",
            "print(\"KEY OBSERVATION:\")\n",
            "if n_pred_in_unseen > 0:\n",
            "    print(f\"  ✓ GZSL classifier CAN predict unseen classes (unlike baseline).\")\n",
            "    print(f\"    This demonstrates successful zero-shot transfer via CLIP + cWGAN-GP.\")\n",
            "else:\n",
            "    print(f\"  ⚠ Classifier still only predicts seen labels — check training.\")"
        ]
    },
    
    # Cell 13: Final Summary
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# FINAL GZSL IMPLEMENTATION SUMMARY\n",
            "# =============================================================================\n",
            "\n",
            "print(\"=\"*70)\n",
            "print(\"FINAL SUMMARY: GZSL CLASSIFIER [A+B]\")\n",
            "print(\"=\"*70)\n",
            "\n",
            "print(\"\\n### MODEL PIPELINE ###\")\n",
            "print(\"  1. Brain-Text CLIP encoder: Maps EEG → 64D semantic space\")\n",
            "print(\"  2. cWGAN-GP: Generates synthetic unseen embeddings from text prototypes\")\n",
            "print(\"  3. GZSL Classifier: Logistic Regression on real seen + synthetic unseen\")\n",
            "\n",
            "print(\"\\n### GZSL RESULTS ###\")\n",
            "print(f\"  Acc (seen):        {gzsl_results['acc_seen']:.4f} ({gzsl_results['acc_seen']*100:.2f}%)\")\n",
            "print(f\"  Acc (unseen):      {gzsl_results['acc_unseen']:.4f} ({gzsl_results['acc_unseen']*100:.2f}%)\")\n",
            "print(f\"  Harmonic Mean (H): {gzsl_results['H']:.4f}\")\n",
            "print(f\"  Macro F1 (seen):   {gzsl_results['macro_f1_seen']:.4f}\")\n",
            "print(f\"  Macro F1 (unseen): {gzsl_results['macro_f1_unseen']:.4f}\")\n",
            "\n",
            "print(\"\\n### IMPROVEMENT OVER BASELINE ###\")\n",
            "print(f\"  Δ Acc (seen):   {gzsl_results['acc_seen'] - baseline_gzsl_results['acc_seen']:+.4f}\")\n",
            "print(f\"  Δ Acc (unseen): {gzsl_results['acc_unseen'] - baseline_gzsl_results['acc_unseen']:+.4f}\")\n",
            "print(f\"  Δ H:            {gzsl_results['H'] - baseline_gzsl_results['H']:+.4f}\")\n",
            "\n",
            "print(\"\\n### KEY INSIGHT ###\")\n",
            "print(\"  The baseline achieves ~0% on unseen classes because it only knows seen labels.\")\n",
            "print(\"  GZSL [A+B] achieves non-zero unseen accuracy by:\")\n",
            "print(\"    - Using CLIP to create a shared EEG-text semantic space\")\n",
            "print(\"    - Using cWGAN-GP to synthesize training data for unseen classes\")\n",
            "print(\"    - Training a classifier that sees both seen and (synthetic) unseen classes\")\n",
            "\n",
            "print(\"\\n### FIGURES GENERATED ###\")\n",
            "print(\"  - figures/gzsl_comparison.png\")\n",
            "\n",
            "print(\"\\n\" + \"=\"*70)\n",
            "print(\"GZSL IMPLEMENTATION COMPLETE\")\n",
            "print(\"=\"*70)"
        ]
    }
]


def main():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Error: Notebook not found at {NOTEBOOK_PATH}")
        return
    
    print(f"Loading notebook: {NOTEBOOK_PATH}")
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    original_count = len(notebook['cells'])
    notebook['cells'].extend(NEW_CELLS)
    
    print(f"Adding {len(NEW_CELLS)} new cells...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"\nSuccess! Updated: {OUTPUT_PATH}")
    print(f"  - Original cells: {original_count}")
    print(f"  - New cells added: {len(NEW_CELLS)}")
    print(f"  - Total cells: {len(notebook['cells'])}")


if __name__ == "__main__":
    main()
