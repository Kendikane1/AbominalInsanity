#!/usr/bin/env python3
"""
Script to add corrected 2×2 bias table cell.
"""

import json
import os

NOTEBOOK_PATH = "COMP2261_ArizMLCW_with_baseline.ipynb"
OUTPUT_PATH = "COMP2261_ArizMLCW_with_baseline.ipynb"

NEW_CELLS = [
    # Cell 1: Corrected Bias Table
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## Corrected 2×2 Bias Table (Full Model D)\n",
            "\n",
            "Recomputing the true-group vs predicted-group bias table with proper counts."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# CORRECTED 2×2 BIAS TABLE FOR FULL MODEL (D)\n",
            "# =============================================================================\n",
            "# Recompute from scratch using proper boolean masks.\n",
            "\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "# -----------------------------------------------------------------------------\n",
            "# 1. Get ground truth labels\n",
            "# -----------------------------------------------------------------------------\n",
            "y_true_seen = y_test_seen  # from cached arrays (3308 samples)\n",
            "y_true_unseen = y_unseen   # from cached arrays (16000 samples)\n",
            "\n",
            "print(f\"Ground truth sizes: seen test = {len(y_true_seen)}, unseen test = {len(y_true_unseen)}\")\n",
            "\n",
            "# -----------------------------------------------------------------------------\n",
            "# 2. Get Model D predictions (clf_d was trained on seen+synth unseen)\n",
            "# -----------------------------------------------------------------------------\n",
            "# clf_d should already exist from the ablation study\n",
            "y_pred_on_seen = clf_d.predict(E_test_seen)\n",
            "y_pred_on_unseen = clf_d.predict(E_unseen)\n",
            "\n",
            "print(f\"Prediction sizes: on seen = {len(y_pred_on_seen)}, on unseen = {len(y_pred_on_unseen)}\")\n",
            "\n",
            "# -----------------------------------------------------------------------------\n",
            "# 3. Define seen/unseen label sets\n",
            "# -----------------------------------------------------------------------------\n",
            "seen_set = set(seen_classes.tolist())\n",
            "unseen_set = set(unseen_classes.tolist())\n",
            "\n",
            "print(f\"Label sets: |seen| = {len(seen_set)}, |unseen| = {len(unseen_set)}\")\n",
            "print(f\"Overlap check: {len(seen_set & unseen_set)} overlapping labels (should be 0)\")\n",
            "\n",
            "# -----------------------------------------------------------------------------\n",
            "# 4. Compute boolean masks for predictions\n",
            "# -----------------------------------------------------------------------------\n",
            "# For predictions on seen test data\n",
            "pred_seen_is_seen = np.isin(y_pred_on_seen, list(seen_set))\n",
            "pred_seen_is_unseen = np.isin(y_pred_on_seen, list(unseen_set))\n",
            "\n",
            "# For predictions on unseen test data\n",
            "pred_unseen_is_seen = np.isin(y_pred_on_unseen, list(seen_set))\n",
            "pred_unseen_is_unseen = np.isin(y_pred_on_unseen, list(unseen_set))\n",
            "\n",
            "# -----------------------------------------------------------------------------\n",
            "# 5. Sanity check: every prediction must be in exactly one set\n",
            "# -----------------------------------------------------------------------------\n",
            "assert np.all(pred_seen_is_seen ^ pred_seen_is_unseen), \"Some predictions on seen data are not in seen or unseen set!\"\n",
            "assert np.all(pred_unseen_is_seen ^ pred_unseen_is_unseen), \"Some predictions on unseen data are not in seen or unseen set!\"\n",
            "print(\"\\n✓ Sanity check passed: all predictions fall into exactly one label set.\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# BUILD 2×2 BIAS TABLE WITH CORRECT COUNTS\n",
            "# =============================================================================\n",
            "\n",
            "# Compute raw counts\n",
            "# Rows: True group (Seen, Unseen)\n",
            "# Cols: Predicted group (Seen, Unseen)\n",
            "\n",
            "count_true_seen_pred_seen = pred_seen_is_seen.sum()\n",
            "count_true_seen_pred_unseen = pred_seen_is_unseen.sum()\n",
            "count_true_unseen_pred_seen = pred_unseen_is_seen.sum()\n",
            "count_true_unseen_pred_unseen = pred_unseen_is_unseen.sum()\n",
            "\n",
            "# Row totals\n",
            "total_seen_test = len(y_true_seen)\n",
            "total_unseen_test = len(y_true_unseen)\n",
            "\n",
            "# Verify row totals\n",
            "row1_sum = count_true_seen_pred_seen + count_true_seen_pred_unseen\n",
            "row2_sum = count_true_unseen_pred_seen + count_true_unseen_pred_unseen\n",
            "\n",
            "print(\"=\" * 60)\n",
            "print(\"2×2 BIAS TABLE: RAW COUNTS\")\n",
            "print(\"=\" * 60)\n",
            "print(f\"                    Pred: Seen    Pred: Unseen    Row Total\")\n",
            "print(f\"True: Seen          {count_true_seen_pred_seen:10d}    {count_true_seen_pred_unseen:12d}    {row1_sum:9d}\")\n",
            "print(f\"True: Unseen        {count_true_unseen_pred_seen:10d}    {count_true_unseen_pred_unseen:12d}    {row2_sum:9d}\")\n",
            "print(\"\")\n",
            "print(f\"Expected row totals: Seen test = {total_seen_test}, Unseen test = {total_unseen_test}\")\n",
            "assert row1_sum == total_seen_test, f\"Row 1 sum mismatch: {row1_sum} != {total_seen_test}\"\n",
            "assert row2_sum == total_unseen_test, f\"Row 2 sum mismatch: {row2_sum} != {total_unseen_test}\"\n",
            "print(\"✓ Row totals verified!\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# ROW-NORMALISED PERCENTAGES\n",
            "# =============================================================================\n",
            "\n",
            "pct_true_seen_pred_seen = 100 * count_true_seen_pred_seen / total_seen_test\n",
            "pct_true_seen_pred_unseen = 100 * count_true_seen_pred_unseen / total_seen_test\n",
            "pct_true_unseen_pred_seen = 100 * count_true_unseen_pred_seen / total_unseen_test\n",
            "pct_true_unseen_pred_unseen = 100 * count_true_unseen_pred_unseen / total_unseen_test\n",
            "\n",
            "print(\"=\" * 60)\n",
            "print(\"2×2 BIAS TABLE: ROW-NORMALISED PERCENTAGES\")\n",
            "print(\"=\" * 60)\n",
            "print(f\"                    Pred: Seen    Pred: Unseen\")\n",
            "print(f\"True: Seen          {pct_true_seen_pred_seen:10.1f}%    {pct_true_seen_pred_unseen:11.1f}%\")\n",
            "print(f\"True: Unseen        {pct_true_unseen_pred_seen:10.1f}%    {pct_true_unseen_pred_unseen:11.1f}%\")\n",
            "print(\"\")\n",
            "print(\"Interpretation:\")\n",
            "print(f\"  - {pct_true_seen_pred_seen:.1f}% of seen samples are correctly routed to seen labels\")\n",
            "print(f\"  - {pct_true_unseen_pred_unseen:.1f}% of unseen samples are correctly routed to unseen labels\")\n",
            "print(f\"  - {pct_true_unseen_pred_seen:.1f}% of unseen samples are misrouted to seen labels (seen-class bias)\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# CREATE PANDAS DATAFRAME VERSION\n",
            "# =============================================================================\n",
            "\n",
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
            "print(\"\\nBias Table (Counts):\")\n",
            "print(bias_table_counts.to_string())\n",
            "print(\"\\nBias Table (Percentages):\")\n",
            "print(bias_table_pct.round(1).to_string())"
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
            "# Add count annotations\n",
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
            "# Add percentage annotations\n",
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
            "print(f\"\\nSaved: figures/bias_table_full_model.png\")"
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
