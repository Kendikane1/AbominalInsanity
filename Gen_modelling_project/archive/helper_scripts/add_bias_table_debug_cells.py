#!/usr/bin/env python3
"""
Script to add debug + fixed bias table cells.
"""

import json
import os

NOTEBOOK_PATH = "COMP2261_ArizMLCW_with_baseline.ipynb"
OUTPUT_PATH = "COMP2261_ArizMLCW_with_baseline.ipynb"

NEW_CELLS = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## Debug + Fixed 2×2 Bias Table\n",
            "\n",
            "First debug the label sets, then compute bias table using classifier's actual label space."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# DEBUG: IDENTIFY LABEL SET MISMATCH\n",
            "# =============================================================================\n",
            "\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "# Get label sets from different sources\n",
            "clf_d_classes = set(clf_d.classes_.tolist())\n",
            "seen_classes_set = set(seen_classes.tolist())\n",
            "unseen_classes_set = set(unseen_classes.tolist())\n",
            "\n",
            "# Also check y_train_seen and y_synth_unseen\n",
            "train_seen_labels = set(np.unique(y_train_seen).tolist())\n",
            "synth_unseen_labels = set(np.unique(y_synth_unseen).tolist())\n",
            "\n",
            "print(\"=\" * 60)\n",
            "print(\"LABEL SET DEBUGGING\")\n",
            "print(\"=\" * 60)\n",
            "print(f\"clf_d knows {len(clf_d_classes)} classes\")\n",
            "print(f\"seen_classes (cached): {len(seen_classes_set)} classes\")\n",
            "print(f\"unseen_classes (cached): {len(unseen_classes_set)} classes\")\n",
            "print(f\"y_train_seen unique: {len(train_seen_labels)} classes\")\n",
            "print(f\"y_synth_unseen unique: {len(synth_unseen_labels)} classes\")\n",
            "print(\"\")\n",
            "\n",
            "# Check overlaps\n",
            "cached_union = seen_classes_set | unseen_classes_set\n",
            "training_union = train_seen_labels | synth_unseen_labels\n",
            "\n",
            "print(f\"cached seen ∪ unseen: {len(cached_union)} classes\")\n",
            "print(f\"training labels union: {len(training_union)} classes\")\n",
            "print(\"\")\n",
            "\n",
            "# Find mismatches\n",
            "in_clf_not_cached = clf_d_classes - cached_union\n",
            "in_cached_not_clf = cached_union - clf_d_classes\n",
            "\n",
            "print(f\"Labels in clf_d but NOT in cached union: {len(in_clf_not_cached)}\")\n",
            "print(f\"Labels in cached union but NOT in clf_d: {len(in_cached_not_clf)}\")\n",
            "\n",
            "if len(in_clf_not_cached) > 0:\n",
            "    print(f\"  Sample of missing from cached: {list(in_clf_not_cached)[:5]}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# FIXED BIAS TABLE: USE CLASSIFIER'S ACTUAL LABEL SPACE\n",
            "# =============================================================================\n",
            "# Instead of using cached seen_classes/unseen_classes, use the actual labels\n",
            "# from the training data: y_train_seen and y_synth_unseen\n",
            "\n",
            "# Define sets from actual training labels (what clf_d was trained on)\n",
            "actual_seen_set = set(np.unique(y_train_seen).tolist())\n",
            "actual_unseen_set = set(np.unique(y_synth_unseen).tolist())\n",
            "\n",
            "print(f\"Using actual training label sets:\")\n",
            "print(f\"  Seen classes (from y_train_seen): {len(actual_seen_set)}\")\n",
            "print(f\"  Unseen classes (from y_synth_unseen): {len(actual_unseen_set)}\")\n",
            "print(f\"  Overlap: {len(actual_seen_set & actual_unseen_set)} (should be 0)\")\n",
            "\n",
            "# Get predictions\n",
            "y_pred_on_seen = clf_d.predict(E_test_seen)\n",
            "y_pred_on_unseen = clf_d.predict(E_unseen)\n",
            "\n",
            "# Ground truth\n",
            "y_true_seen = y_test_seen\n",
            "y_true_unseen = y_unseen\n",
            "\n",
            "print(f\"\\nPrediction counts: on_seen={len(y_pred_on_seen)}, on_unseen={len(y_pred_on_unseen)}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# COMPUTE BOOLEAN MASKS WITH ACTUAL LABEL SETS\n",
            "# =============================================================================\n",
            "\n",
            "# For predictions on seen test data\n",
            "pred_seen_is_seen = np.isin(y_pred_on_seen, list(actual_seen_set))\n",
            "pred_seen_is_unseen = np.isin(y_pred_on_seen, list(actual_unseen_set))\n",
            "\n",
            "# For predictions on unseen test data  \n",
            "pred_unseen_is_seen = np.isin(y_pred_on_unseen, list(actual_seen_set))\n",
            "pred_unseen_is_unseen = np.isin(y_pred_on_unseen, list(actual_unseen_set))\n",
            "\n",
            "# Check coverage\n",
            "seen_coverage = pred_seen_is_seen | pred_seen_is_unseen\n",
            "unseen_coverage = pred_unseen_is_seen | pred_unseen_is_unseen\n",
            "\n",
            "print(f\"Predictions on seen data covered: {seen_coverage.sum()}/{len(seen_coverage)} ({100*seen_coverage.mean():.1f}%)\")\n",
            "print(f\"Predictions on unseen data covered: {unseen_coverage.sum()}/{len(unseen_coverage)} ({100*unseen_coverage.mean():.1f}%)\")\n",
            "\n",
            "# If not 100%, show what's missing\n",
            "if not np.all(seen_coverage):\n",
            "    uncovered = y_pred_on_seen[~seen_coverage]\n",
            "    print(f\"  Uncovered predictions on seen: {len(uncovered)}, sample: {np.unique(uncovered)[:5]}\")\n",
            "if not np.all(unseen_coverage):\n",
            "    uncovered = y_pred_on_unseen[~unseen_coverage]\n",
            "    print(f\"  Uncovered predictions on unseen: {len(uncovered)}, sample: {np.unique(uncovered)[:5]}\")\n",
            "\n",
            "# For bias table, we proceed with what IS covered\n",
            "print(\"\\n✓ Proceeding with bias table computation...\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# BUILD 2×2 BIAS TABLE\n",
            "# =============================================================================\n",
            "\n",
            "# Raw counts\n",
            "count_true_seen_pred_seen = pred_seen_is_seen.sum()\n",
            "count_true_seen_pred_unseen = pred_seen_is_unseen.sum()\n",
            "count_true_unseen_pred_seen = pred_unseen_is_seen.sum()\n",
            "count_true_unseen_pred_unseen = pred_unseen_is_unseen.sum()\n",
            "\n",
            "# Row totals (these should equal dataset sizes if all predictions are covered)\n",
            "total_seen_test = len(y_true_seen)\n",
            "total_unseen_test = len(y_true_unseen)\n",
            "\n",
            "row1_sum = count_true_seen_pred_seen + count_true_seen_pred_unseen\n",
            "row2_sum = count_true_unseen_pred_seen + count_true_unseen_pred_unseen\n",
            "\n",
            "print(\"=\" * 60)\n",
            "print(\"2×2 BIAS TABLE: RAW COUNTS\")\n",
            "print(\"=\" * 60)\n",
            "print(f\"                    Pred: Seen    Pred: Unseen    Row Total\")\n",
            "print(f\"True: Seen          {count_true_seen_pred_seen:10d}    {count_true_seen_pred_unseen:12d}    {row1_sum:9d}\")\n",
            "print(f\"True: Unseen        {count_true_unseen_pred_seen:10d}    {count_true_unseen_pred_unseen:12d}    {row2_sum:9d}\")\n",
            "print(f\"\")\n",
            "print(f\"Expected totals: seen_test={total_seen_test}, unseen_test={total_unseen_test}\")\n",
            "\n",
            "# Row-normalised percentages  \n",
            "pct_true_seen_pred_seen = 100 * count_true_seen_pred_seen / total_seen_test\n",
            "pct_true_seen_pred_unseen = 100 * count_true_seen_pred_unseen / total_seen_test\n",
            "pct_true_unseen_pred_seen = 100 * count_true_unseen_pred_seen / total_unseen_test\n",
            "pct_true_unseen_pred_unseen = 100 * count_true_unseen_pred_unseen / total_unseen_test\n",
            "\n",
            "print(\"\")\n",
            "print(\"=\" * 60)\n",
            "print(\"2×2 BIAS TABLE: ROW-NORMALISED PERCENTAGES\")\n",
            "print(\"=\" * 60)\n",
            "print(f\"                    Pred: Seen    Pred: Unseen\")\n",
            "print(f\"True: Seen          {pct_true_seen_pred_seen:10.1f}%    {pct_true_seen_pred_unseen:11.1f}%\")\n",
            "print(f\"True: Unseen        {pct_true_unseen_pred_seen:10.1f}%    {pct_true_unseen_pred_unseen:11.1f}%\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# HEATMAP VISUALIZATION\n",
            "# =============================================================================\n",
            "\n",
            "# Create DataFrames\n",
            "bias_table_counts = pd.DataFrame(\n",
            "    [[count_true_seen_pred_seen, count_true_seen_pred_unseen],\n",
            "     [count_true_unseen_pred_seen, count_true_unseen_pred_unseen]],\n",
            "    index=['True: Seen', 'True: Unseen'],\n",
            "    columns=['Pred: Seen', 'Pred: Unseen']\n",
            ")\n",
            "\n",
            "bias_table_pct = pd.DataFrame(\n",
            "    [[pct_true_seen_pred_seen, pct_true_seen_pred_unseen],\n",
            "     [pct_true_unseen_pred_seen, pct_true_unseen_pred_unseen]],\n",
            "    index=['True: Seen', 'True: Unseen'],\n",
            "    columns=['Pred: Seen', 'Pred: Unseen']\n",
            ")\n",
            "\n",
            "# Plot\n",
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
            "\n",
            "# Counts heatmap\n",
            "im1 = axes[0].imshow(bias_table_counts.values, cmap='Blues')\n",
            "axes[0].set_xticks([0, 1])\n",
            "axes[0].set_yticks([0, 1])\n",
            "axes[0].set_xticklabels(['Pred: Seen', 'Pred: Unseen'])\n",
            "axes[0].set_yticklabels(['True: Seen', 'True: Unseen'])\n",
            "axes[0].set_title('Bias Table (Counts)', fontsize=12)\n",
            "\n",
            "for i in range(2):\n",
            "    for j in range(2):\n",
            "        val = bias_table_counts.values[i, j]\n",
            "        color = 'white' if val > bias_table_counts.values.max() / 2 else 'black'\n",
            "        axes[0].text(j, i, f'{val:,}', ha='center', va='center', color=color, fontsize=14)\n",
            "\n",
            "# Percentage heatmap\n",
            "im2 = axes[1].imshow(bias_table_pct.values, cmap='Oranges', vmin=0, vmax=100)\n",
            "axes[1].set_xticks([0, 1])\n",
            "axes[1].set_yticks([0, 1])\n",
            "axes[1].set_xticklabels(['Pred: Seen', 'Pred: Unseen'])\n",
            "axes[1].set_yticklabels(['True: Seen', 'True: Unseen'])\n",
            "axes[1].set_title('Bias Table (Row %)', fontsize=12)\n",
            "\n",
            "for i in range(2):\n",
            "    for j in range(2):\n",
            "        val = bias_table_pct.values[i, j]\n",
            "        color = 'white' if val > 50 else 'black'\n",
            "        axes[1].text(j, i, f'{val:.1f}%', ha='center', va='center', color=color, fontsize=14)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('figures/bias_table_full_model.png', dpi=150, bbox_inches='tight')\n",
            "plt.show()\n",
            "\n",
            "print(f\"\\nSaved: figures/bias_table_full_model.png\")\n",
            "print(\"\")\n",
            "print(\"Interpretation:\")\n",
            "print(f\"  - {pct_true_seen_pred_seen:.1f}% of seen samples → predicted as seen\")\n",
            "print(f\"  - {pct_true_unseen_pred_unseen:.1f}% of unseen samples → predicted as unseen\")\n",
            "print(f\"  - {pct_true_unseen_pred_seen:.1f}% of unseen samples → misrouted to seen (seen-class bias)\")"
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
