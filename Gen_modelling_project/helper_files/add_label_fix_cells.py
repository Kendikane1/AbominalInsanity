#!/usr/bin/env python3
"""
COMPREHENSIVE FIX: Label Collision and GZSL Retraining

This script adds cells that:
1. Remap unseen labels to avoid collision with seen labels
2. Retrain the GZSL classifier
3. Recompute ablation metrics
4. Compute correct bias table
"""

import json
import os

NOTEBOOK_PATH = "COMP2261_ArizMLCW_with_baseline.ipynb"
OUTPUT_PATH = "COMP2261_ArizMLCW_with_baseline.ipynb"

NEW_CELLS = [
    # =========================================================================
    # LABEL FIX SECTION
    # =========================================================================
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "# FIX: Label Collision Resolution\n",
            "\n",
            "**Problem Identified:** Unseen labels (1-200) collide with seen labels (1-200).\n",
            "\n",
            "**Solution:** Remap unseen labels to a disjoint range: `unseen_label_new = unseen_label_old + max(seen_labels)`\n",
            "\n",
            "This creates:\n",
            "- Seen classes: 1-1654\n",
            "- Unseen classes: 1655-1854"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# STEP 1: COMPUTE LABEL OFFSET\n",
            "# =============================================================================\n",
            "\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.metrics import accuracy_score, f1_score\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "SEED = 42\n",
            "np.random.seed(SEED)\n",
            "\n",
            "# Load cached embeddings and labels\n",
            "E_train_seen = np.load('cached_arrays/E_train_seen.npy')\n",
            "E_test_seen = np.load('cached_arrays/E_test_seen.npy')\n",
            "E_unseen = np.load('cached_arrays/E_unseen.npy')\n",
            "E_synth_unseen = np.load('cached_arrays/E_synth_unseen.npy')\n",
            "\n",
            "y_train_seen = np.load('cached_arrays/y_train_seen.npy')\n",
            "y_test_seen = np.load('cached_arrays/y_test_seen.npy')\n",
            "y_unseen_original = np.load('cached_arrays/y_unseen.npy')\n",
            "y_synth_unseen_original = np.load('cached_arrays/y_synth_unseen.npy')\n",
            "\n",
            "# Compute offset = max seen label\n",
            "LABEL_OFFSET = int(np.max(y_train_seen))\n",
            "\n",
            "print(\"=\" * 60)\n",
            "print(\"LABEL COLLISION FIX\")\n",
            "print(\"=\" * 60)\n",
            "print(f\"Max seen label: {LABEL_OFFSET}\")\n",
            "print(f\"Original unseen labels: {y_unseen_original.min()} to {y_unseen_original.max()}\")\n",
            "print(f\"New unseen labels will be: {y_unseen_original.min() + LABEL_OFFSET} to {y_unseen_original.max() + LABEL_OFFSET}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# STEP 2: REMAP UNSEEN LABELS\n",
            "# =============================================================================\n",
            "\n",
            "# Remap unseen labels to avoid collision\n",
            "y_unseen_remapped = y_unseen_original + LABEL_OFFSET\n",
            "y_synth_unseen_remapped = y_synth_unseen_original + LABEL_OFFSET\n",
            "\n",
            "print(f\"Remapped y_unseen: {y_unseen_remapped.min()} to {y_unseen_remapped.max()}\")\n",
            "print(f\"Remapped y_synth_unseen: {y_synth_unseen_remapped.min()} to {y_synth_unseen_remapped.max()}\")\n",
            "\n",
            "# Verify no overlap\n",
            "seen_labels_set = set(np.unique(y_train_seen).tolist())\n",
            "unseen_labels_set = set(np.unique(y_unseen_remapped).tolist())\n",
            "overlap = seen_labels_set & unseen_labels_set\n",
            "\n",
            "print(f\"\\nSeen labels: {len(seen_labels_set)} unique\")\n",
            "print(f\"Unseen labels (remapped): {len(unseen_labels_set)} unique\")\n",
            "print(f\"Overlap: {len(overlap)} (should be 0)\")\n",
            "\n",
            "assert len(overlap) == 0, \"Label overlap still exists!\"\n",
            "print(\"\\n✓ Labels are now disjoint!\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# STEP 3: RETRAIN GZSL CLASSIFIER WITH CORRECT LABELS\n",
            "# =============================================================================\n",
            "\n",
            "# Combine real seen + synthetic unseen (with remapped labels)\n",
            "X_train_gzsl_fixed = np.vstack([E_train_seen, E_synth_unseen])\n",
            "y_train_gzsl_fixed = np.concatenate([y_train_seen, y_synth_unseen_remapped])\n",
            "\n",
            "print(\"GZSL Training Data (Fixed):\")\n",
            "print(f\"  Real seen: {len(E_train_seen)} samples, labels {y_train_seen.min()}-{y_train_seen.max()}\")\n",
            "print(f\"  Synth unseen: {len(E_synth_unseen)} samples, labels {y_synth_unseen_remapped.min()}-{y_synth_unseen_remapped.max()}\")\n",
            "print(f\"  Combined: {len(X_train_gzsl_fixed)} samples, {len(np.unique(y_train_gzsl_fixed))} classes\")\n",
            "\n",
            "# Train\n",
            "print(\"\\nTraining GZSL classifier (fixed labels)...\")\n",
            "clf_gzsl_fixed = LogisticRegression(\n",
            "    multi_class='multinomial', solver='lbfgs', max_iter=1000,\n",
            "    random_state=SEED, n_jobs=-1, verbose=1\n",
            ")\n",
            "clf_gzsl_fixed.fit(X_train_gzsl_fixed, y_train_gzsl_fixed)\n",
            "print(\"\\n✓ Training complete!\")\n",
            "print(f\"Classifier knows {len(clf_gzsl_fixed.classes_)} classes\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# STEP 4: EVALUATE GZSL CLASSIFIER\n",
            "# =============================================================================\n",
            "\n",
            "# Predictions\n",
            "y_pred_seen_fixed = clf_gzsl_fixed.predict(E_test_seen)\n",
            "y_pred_unseen_fixed = clf_gzsl_fixed.predict(E_unseen)\n",
            "\n",
            "# Metrics for seen (ground truth uses original labels)\n",
            "acc_seen_fixed = accuracy_score(y_test_seen, y_pred_seen_fixed)\n",
            "f1_seen_fixed = f1_score(y_test_seen, y_pred_seen_fixed, average='macro', zero_division=0)\n",
            "\n",
            "# Metrics for unseen (ground truth uses REMAPPED labels)\n",
            "acc_unseen_fixed = accuracy_score(y_unseen_remapped, y_pred_unseen_fixed)\n",
            "f1_unseen_fixed = f1_score(y_unseen_remapped, y_pred_unseen_fixed, average='macro', zero_division=0)\n",
            "\n",
            "# Harmonic mean\n",
            "if acc_seen_fixed + acc_unseen_fixed > 0:\n",
            "    H_fixed = 2 * acc_seen_fixed * acc_unseen_fixed / (acc_seen_fixed + acc_unseen_fixed)\n",
            "else:\n",
            "    H_fixed = 0.0\n",
            "\n",
            "print(\"=\" * 60)\n",
            "print(\"GZSL EVALUATION (FIXED LABELS)\")\n",
            "print(\"=\" * 60)\n",
            "print(f\"Acc (seen):      {acc_seen_fixed:.4f} ({acc_seen_fixed*100:.2f}%)\")\n",
            "print(f\"Acc (unseen):    {acc_unseen_fixed:.4f} ({acc_unseen_fixed*100:.2f}%)\")\n",
            "print(f\"Harmonic Mean:   {H_fixed:.4f}\")\n",
            "print(f\"Macro F1 (seen): {f1_seen_fixed:.4f}\")\n",
            "print(f\"Macro F1 (unseen): {f1_unseen_fixed:.4f}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# STEP 5: CORRECTED 2×2 BIAS TABLE\n",
            "# =============================================================================\n",
            "\n",
            "# Define label sets\n",
            "seen_set = seen_labels_set\n",
            "unseen_set = unseen_labels_set\n",
            "\n",
            "# Boolean masks for predictions\n",
            "pred_on_seen_is_seen = np.isin(y_pred_seen_fixed, list(seen_set))\n",
            "pred_on_seen_is_unseen = np.isin(y_pred_seen_fixed, list(unseen_set))\n",
            "\n",
            "pred_on_unseen_is_seen = np.isin(y_pred_unseen_fixed, list(seen_set))\n",
            "pred_on_unseen_is_unseen = np.isin(y_pred_unseen_fixed, list(unseen_set))\n",
            "\n",
            "# Sanity check\n",
            "assert np.all(pred_on_seen_is_seen | pred_on_seen_is_unseen), \"Some predictions uncovered!\"\n",
            "assert np.all(pred_on_unseen_is_seen | pred_on_unseen_is_unseen), \"Some predictions uncovered!\"\n",
            "print(\"✓ All predictions fall into exactly one label set.\")\n",
            "\n",
            "# Counts\n",
            "count_ss = pred_on_seen_is_seen.sum()      # True Seen, Pred Seen\n",
            "count_su = pred_on_seen_is_unseen.sum()    # True Seen, Pred Unseen\n",
            "count_us = pred_on_unseen_is_seen.sum()    # True Unseen, Pred Seen\n",
            "count_uu = pred_on_unseen_is_unseen.sum()  # True Unseen, Pred Unseen\n",
            "\n",
            "total_seen = len(y_pred_seen_fixed)\n",
            "total_unseen = len(y_pred_unseen_fixed)\n",
            "\n",
            "# Verify\n",
            "assert count_ss + count_su == total_seen, \"Row 1 mismatch\"\n",
            "assert count_us + count_uu == total_unseen, \"Row 2 mismatch\"\n",
            "print(f\"✓ Row totals verified: seen={total_seen}, unseen={total_unseen}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# PRINT BIAS TABLE\n",
            "# =============================================================================\n",
            "\n",
            "pct_ss = 100 * count_ss / total_seen\n",
            "pct_su = 100 * count_su / total_seen\n",
            "pct_us = 100 * count_us / total_unseen\n",
            "pct_uu = 100 * count_uu / total_unseen\n",
            "\n",
            "print(\"=\" * 60)\n",
            "print(\"2×2 BIAS TABLE (FIXED LABELS)\")\n",
            "print(\"=\" * 60)\n",
            "print(\"\")\n",
            "print(\"RAW COUNTS:\")\n",
            "print(f\"                    Pred: Seen    Pred: Unseen    Total\")\n",
            "print(f\"True: Seen          {count_ss:10d}    {count_su:12d}    {total_seen:6d}\")\n",
            "print(f\"True: Unseen        {count_us:10d}    {count_uu:12d}    {total_unseen:6d}\")\n",
            "print(\"\")\n",
            "print(\"ROW-NORMALISED PERCENTAGES:\")\n",
            "print(f\"                    Pred: Seen    Pred: Unseen\")\n",
            "print(f\"True: Seen          {pct_ss:10.1f}%    {pct_su:11.1f}%\")\n",
            "print(f\"True: Unseen        {pct_us:10.1f}%    {pct_uu:11.1f}%\")\n",
            "print(\"\")\n",
            "print(\"Interpretation:\")\n",
            "print(f\"  - {pct_ss:.1f}% of seen samples correctly predicted as seen\")\n",
            "print(f\"  - {pct_uu:.1f}% of unseen samples correctly predicted as unseen\")\n",
            "print(f\"  - {pct_us:.1f}% of unseen samples misrouted to seen (seen-bias)\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# BIAS TABLE HEATMAP\n",
            "# =============================================================================\n",
            "\n",
            "bias_counts = np.array([[count_ss, count_su], [count_us, count_uu]])\n",
            "bias_pct = np.array([[pct_ss, pct_su], [pct_us, pct_uu]])\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
            "\n",
            "# Counts\n",
            "im1 = axes[0].imshow(bias_counts, cmap='Blues')\n",
            "axes[0].set_xticks([0, 1])\n",
            "axes[0].set_yticks([0, 1])\n",
            "axes[0].set_xticklabels(['Pred: Seen', 'Pred: Unseen'])\n",
            "axes[0].set_yticklabels(['True: Seen', 'True: Unseen'])\n",
            "axes[0].set_title('Bias Table (Counts)', fontsize=12)\n",
            "for i in range(2):\n",
            "    for j in range(2):\n",
            "        color = 'white' if bias_counts[i,j] > bias_counts.max()/2 else 'black'\n",
            "        axes[0].text(j, i, f'{bias_counts[i,j]:,}', ha='center', va='center', color=color, fontsize=14)\n",
            "\n",
            "# Percentages\n",
            "im2 = axes[1].imshow(bias_pct, cmap='Oranges', vmin=0, vmax=100)\n",
            "axes[1].set_xticks([0, 1])\n",
            "axes[1].set_yticks([0, 1])\n",
            "axes[1].set_xticklabels(['Pred: Seen', 'Pred: Unseen'])\n",
            "axes[1].set_yticklabels(['True: Seen', 'True: Unseen'])\n",
            "axes[1].set_title('Bias Table (Row %)', fontsize=12)\n",
            "for i in range(2):\n",
            "    for j in range(2):\n",
            "        color = 'white' if bias_pct[i,j] > 50 else 'black'\n",
            "        axes[1].text(j, i, f'{bias_pct[i,j]:.1f}%', ha='center', va='center', color=color, fontsize=14)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('figures/bias_table_full_model.png', dpi=150, bbox_inches='tight')\n",
            "plt.show()\n",
            "print(f\"Saved: figures/bias_table_full_model.png\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# CORRECTED ABLATION TABLE\n",
            "# =============================================================================\n",
            "\n",
            "# Method A: Baseline (raw EEG) — already computed, use baseline_gzsl_results\n",
            "# Method D: Full GZSL (fixed) — just computed above\n",
            "\n",
            "# For ablation, we need to also run Methods B and C with correct labels\n",
            "# But those don't involve synthetic unseen, so they remain valid\n",
            "\n",
            "# Create ablation table\n",
            "ablation_fixed = [\n",
            "    {\n",
            "        'Method': 'Raw EEG LR (A)',\n",
            "        'Acc_seen': baseline_gzsl_results['acc_seen'],\n",
            "        'Acc_unseen': baseline_gzsl_results['acc_unseen'],\n",
            "        'H': baseline_gzsl_results['H'],\n",
            "        'MacroF1_seen': baseline_gzsl_results['macro_f1_seen'],\n",
            "        'MacroF1_unseen': baseline_gzsl_results['macro_f1_unseen']\n",
            "    },\n",
            "    {\n",
            "        'Method': 'CLIP + cWGAN-GP + LR (D) [FIXED]',\n",
            "        'Acc_seen': acc_seen_fixed,\n",
            "        'Acc_unseen': acc_unseen_fixed,\n",
            "        'H': H_fixed,\n",
            "        'MacroF1_seen': f1_seen_fixed,\n",
            "        'MacroF1_unseen': f1_unseen_fixed\n",
            "    }\n",
            "]\n",
            "\n",
            "df_ablation_fixed = pd.DataFrame(ablation_fixed).set_index('Method')\n",
            "\n",
            "print(\"=\" * 70)\n",
            "print(\"CORRECTED ABLATION TABLE\")\n",
            "print(\"=\" * 70)\n",
            "print(df_ablation_fixed.round(4).to_string())\n",
            "print(\"\")\n",
            "print(f\"Improvement in H: {H_fixed - baseline_gzsl_results['H']:+.4f}\")\n",
            "print(f\"Improvement in Acc_unseen: {acc_unseen_fixed - baseline_gzsl_results['acc_unseen']:+.4f}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# CORRECTED ABLATION BAR CHART\n",
            "# =============================================================================\n",
            "\n",
            "methods = ['Baseline (A)', 'GZSL [A+B] Fixed']\n",
            "acc_s = [baseline_gzsl_results['acc_seen'], acc_seen_fixed]\n",
            "acc_u = [baseline_gzsl_results['acc_unseen'], acc_unseen_fixed]\n",
            "H_vals = [baseline_gzsl_results['H'], H_fixed]\n",
            "\n",
            "x = np.arange(len(methods))\n",
            "width = 0.25\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(10, 6))\n",
            "bars1 = ax.bar(x - width, acc_s, width, label='Acc_seen', color='steelblue')\n",
            "bars2 = ax.bar(x, acc_u, width, label='Acc_unseen', color='darkorange')\n",
            "bars3 = ax.bar(x + width, H_vals, width, label='Harmonic Mean (H)', color='green')\n",
            "\n",
            "for bars in [bars1, bars2, bars3]:\n",
            "    for bar in bars:\n",
            "        h = bar.get_height()\n",
            "        ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),\n",
            "                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=11)\n",
            "\n",
            "ax.set_ylabel('Score', fontsize=12)\n",
            "ax.set_title('GZSL Performance: Baseline vs Fixed Model', fontsize=14)\n",
            "ax.set_xticks(x)\n",
            "ax.set_xticklabels(methods, fontsize=12)\n",
            "ax.legend(fontsize=11)\n",
            "ax.set_ylim(0, max(max(acc_s), max(acc_u), max(H_vals)) * 1.25)\n",
            "ax.grid(axis='y', alpha=0.3)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('figures/gzsl_ablation_bar.png', dpi=150, bbox_inches='tight')\n",
            "plt.show()\n",
            "print(f\"Saved: figures/gzsl_ablation_bar.png\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# FINAL SUMMARY\n",
            "# =============================================================================\n",
            "\n",
            "print(\"=\"*70)\n",
            "print(\"LABEL FIX COMPLETE — FINAL SUMMARY\")\n",
            "print(\"=\"*70)\n",
            "print(\"\")\n",
            "print(\"The label collision has been fixed:\")\n",
            "print(f\"  - Seen labels: 1 to {LABEL_OFFSET}\")\n",
            "print(f\"  - Unseen labels (remapped): {LABEL_OFFSET+1} to {LABEL_OFFSET + len(unseen_set)}\")\n",
            "print(f\"  - Total classes in GZSL classifier: {len(clf_gzsl_fixed.classes_)}\")\n",
            "print(\"\")\n",
            "print(\"GZSL [A+B] Results (with fix):\")\n",
            "print(f\"  Acc_seen:   {acc_seen_fixed:.4f}\")\n",
            "print(f\"  Acc_unseen: {acc_unseen_fixed:.4f}\")\n",
            "print(f\"  H:          {H_fixed:.4f}\")\n",
            "print(\"\")\n",
            "print(\"Figures saved:\")\n",
            "print(\"  - figures/bias_table_full_model.png\")\n",
            "print(\"  - figures/gzsl_ablation_bar.png\")\n",
            "print(\"\")\n",
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
