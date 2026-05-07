#!/usr/bin/env python3
"""
Script to add GZSL evaluation cells to the baseline notebook.

Run this script from the project directory:
    python add_gzsl_eval_cells.py
"""

import json
import os

NOTEBOOK_PATH = "COMP2261_ArizMLCW_with_baseline.ipynb"
OUTPUT_PATH = "COMP2261_ArizMLCW_with_baseline.ipynb"  # Overwrite same file

NEW_CELLS = [
    # Cell 1: Markdown - GZSL Evaluation Header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## GZSL-Style Baseline Evaluation\n",
            "\n",
            "This section evaluates the baseline classifier under a **Generalised Zero-Shot Learning (GZSL)** setting:\n",
            "\n",
            "1. **Seen-class evaluation**: Already computed above on `X_test_seen`\n",
            "2. **Unseen-class evaluation**: Apply the same classifier to `brain_unseen` (EEG from unseen classes)\n",
            "3. **Harmonic mean (H)**: Balance between seen and unseen accuracy\n",
            "\n",
            "> **Expected result**: The baseline classifier will achieve ~0% accuracy on unseen classes because it was trained only on seen-class labels. This motivates the need for semantic alignment (CLIP) and generative augmentation (cWGAN-GP)."
        ]
    },
    # Cell 2: Code - GZSL Evaluation
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# GZSL-STYLE BASELINE EVALUATION\n",
            "# =============================================================================\n",
            "# Evaluate the baseline classifier on BOTH seen and unseen classes.\n",
            "# The classifier was trained ONLY on seen classes, so it cannot predict unseen labels.\n",
            "\n",
            "# -----------------------------------------------------------------------------\n",
            "# 1. Store seen-class results (from previous evaluation)\n",
            "# -----------------------------------------------------------------------------\n",
            "baseline_seen_results = {\n",
            "    'acc_seen': baseline_accuracy,\n",
            "    'macro_f1_seen': baseline_macro_f1\n",
            "}\n",
            "\n",
            "print(\"Seen-class evaluation (from training split):\")\n",
            "print(f\"  Accuracy:  {baseline_seen_results['acc_seen']:.4f}\")\n",
            "print(f\"  Macro F1:  {baseline_seen_results['macro_f1_seen']:.4f}\")\n",
            "\n",
            "# -----------------------------------------------------------------------------\n",
            "# 2. Prepare unseen EEG data using the SAME scaler\n",
            "# -----------------------------------------------------------------------------\n",
            "X_unseen = brain_unseen.numpy()\n",
            "y_unseen = label_unseen.numpy().flatten()\n",
            "\n",
            "# Apply the same standardisation as training data\n",
            "X_unseen_scaled = scaler.transform(X_unseen)\n",
            "\n",
            "print(f\"\\nUnseen-class data prepared:\")\n",
            "print(f\"  Samples: {X_unseen.shape[0]}\")\n",
            "print(f\"  Features: {X_unseen.shape[1]}\")\n",
            "print(f\"  Unique labels: {len(np.unique(y_unseen))}\")"
        ]
    },
    # Cell 3: Code - Predict on unseen and compute GZSL metrics
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# PREDICT ON UNSEEN CLASSES\n",
            "# =============================================================================\n",
            "\n",
            "# Predict using the baseline classifier (trained only on seen classes)\n",
            "y_pred_unseen = baseline_clf.predict(X_unseen_scaled)\n",
            "\n",
            "# -----------------------------------------------------------------------------\n",
            "# Compute unseen-class metrics\n",
            "# -----------------------------------------------------------------------------\n",
            "acc_unseen = accuracy_score(y_unseen, y_pred_unseen)\n",
            "macro_f1_unseen = f1_score(y_unseen, y_pred_unseen, average='macro', zero_division=0)\n",
            "\n",
            "# -----------------------------------------------------------------------------\n",
            "# Compute Harmonic Mean (H)\n",
            "# -----------------------------------------------------------------------------\n",
            "# H = 2 * Acc_seen * Acc_unseen / (Acc_seen + Acc_unseen)\n",
            "# Handle division by zero if both are 0\n",
            "acc_seen = baseline_seen_results['acc_seen']\n",
            "if acc_seen + acc_unseen > 0:\n",
            "    H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)\n",
            "else:\n",
            "    H = 0.0\n",
            "\n",
            "# -----------------------------------------------------------------------------\n",
            "# Store GZSL results\n",
            "# -----------------------------------------------------------------------------\n",
            "baseline_gzsl_results = {\n",
            "    'acc_seen': acc_seen,\n",
            "    'acc_unseen': acc_unseen,\n",
            "    'H': H,\n",
            "    'macro_f1_seen': baseline_seen_results['macro_f1_seen'],\n",
            "    'macro_f1_unseen': macro_f1_unseen\n",
            "}\n",
            "\n",
            "print(\"GZSL Evaluation Results:\")\n",
            "print(f\"  Acc (seen):    {acc_seen:.4f} ({acc_seen*100:.2f}%)\")\n",
            "print(f\"  Acc (unseen):  {acc_unseen:.4f} ({acc_unseen*100:.2f}%)\")\n",
            "print(f\"  Harmonic Mean: {H:.4f}\")"
        ]
    },
    # Cell 4: Code - Verify predictions are confined to seen labels
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# VERIFY: PREDICTIONS CONFINED TO SEEN-CLASS LABELS\n",
            "# =============================================================================\n",
            "# The classifier can ONLY predict labels it was trained on (seen classes).\n",
            "# Unseen-class labels are never in the prediction set.\n",
            "\n",
            "# Get the set of labels the classifier knows\n",
            "seen_labels_set = set(baseline_clf.classes_)\n",
            "unseen_labels_set = set(np.unique(y_unseen))\n",
            "predicted_labels_set = set(np.unique(y_pred_unseen))\n",
            "\n",
            "print(\"Label Set Analysis:\")\n",
            "print(f\"  Seen class labels (classifier knows):  {len(seen_labels_set)} classes\")\n",
            "print(f\"  Unseen class labels (ground truth):    {len(unseen_labels_set)} classes\")\n",
            "print(f\"  Predicted labels on unseen data:       {len(predicted_labels_set)} unique values\")\n",
            "\n",
            "# Check overlap\n",
            "overlap = predicted_labels_set.intersection(unseen_labels_set)\n",
            "print(f\"\\n  Overlap (correctly predictable):        {len(overlap)} classes\")\n",
            "\n",
            "# Confirm predictions are confined to seen labels\n",
            "predictions_in_seen = predicted_labels_set.issubset(seen_labels_set)\n",
            "print(f\"\\n  All predictions within seen labels?    {predictions_in_seen}\")\n",
            "\n",
            "if predictions_in_seen and len(overlap) == 0:\n",
            "    print(\"\\n  ✓ CONFIRMED: Classifier never predicts unseen-class labels.\")\n",
            "    print(\"    This is expected — the baseline has no mechanism for zero-shot transfer.\")"
        ]
    },
    # Cell 5: Code - Final GZSL Summary
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# GZSL BASELINE SUMMARY\n",
            "# =============================================================================\n",
            "\n",
            "print(\"=\"*70)\n",
            "print(\"GZSL BASELINE EVALUATION SUMMARY\")\n",
            "print(\"=\"*70)\n",
            "\n",
            "print(\"\\n### baseline_seen_results ###\")\n",
            "for k, v in baseline_seen_results.items():\n",
            "    print(f\"   {k}: {v:.4f}\")\n",
            "\n",
            "print(\"\\n### baseline_gzsl_results ###\")\n",
            "for k, v in baseline_gzsl_results.items():\n",
            "    print(f\"   {k}: {v:.4f}\")\n",
            "\n",
            "print(\"\\n\" + \"-\"*70)\n",
            "print(\"KEY OBSERVATION:\")\n",
            "print(\"-\"*70)\n",
            "print(\"The baseline classifier achieves ~0% accuracy on unseen classes because:\")\n",
            "print(\"  1. It was trained ONLY on seen-class labels\")\n",
            "print(\"  2. It has no semantic knowledge to transfer to new categories\")\n",
            "print(\"  3. All predictions are confined to the seen-class label set\")\n",
            "print(\"\\nThis motivates the need for:\")\n",
            "print(\"  • Brain-Text CLIP encoder (semantic alignment)\")\n",
            "print(\"  • cWGAN-GP (synthetic unseen-class embeddings)\")\n",
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
    print(f"\nRefresh the notebook in Jupyter and run the new cells.")


if __name__ == "__main__":
    main()
