#!/usr/bin/env python3
"""
Script to add baseline model cells to the COMP2261_ArizMLCW notebook.

Run this script from the project directory:
    python add_baseline_cells.py

This will create a new notebook with the baseline cells appended:
    COMP2261_ArizMLCW_with_baseline.ipynb
"""

import json
import os

# Path to the original notebook
NOTEBOOK_PATH = "COMP2261_ArizMLCW (1).ipynb"
OUTPUT_PATH = "COMP2261_ArizMLCW_with_baseline.ipynb"

# New cells to add
NEW_CELLS = [
    # Cell 1: Markdown - Pipeline Introduction
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "# GZSL Pipeline: Baseline Model and Advanced Methods\n",
            "\n",
            "The following sections implement:\n",
            "1. **Baseline Model [A]**: Logistic Regression on raw EEG features\n",
            "2. **Brain-Text CLIP Encoder** (future)\n",
            "3. **cWGAN-GP for Embedding Synthesis** (future)\n",
            "4. **GZSL Classifier [A+B]** (future)\n",
            "\n",
            "---"
        ]
    },
    # Cell 2: Markdown - Data Structure Summary Header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Data Structure Summary\n",
            "\n",
            "This section verifies and summarises the existing data structures for the GZSL pipeline."
        ]
    },
    # Cell 3: Code - Data Structure Summary
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# DATA STRUCTURE SUMMARY\n",
            "# =============================================================================\n",
            "\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "\n",
            "# Collect data summary\n",
            "data_summary = {\n",
            "    'Variable': ['brain_seen', 'brain_unseen', 'text_seen', 'text_unseen', \n",
            "                 'image_seen', 'image_unseen', 'label_seen', 'label_unseen'],\n",
            "    'Shape': [str(tuple(brain_seen.shape)), str(tuple(brain_unseen.shape)),\n",
            "              str(tuple(text_seen.shape)), str(tuple(text_unseen.shape)),\n",
            "              str(tuple(image_seen.shape)), str(tuple(image_unseen.shape)),\n",
            "              str(tuple(label_seen.shape)), str(tuple(label_unseen.shape))],\n",
            "    'Dtype': [str(brain_seen.dtype), str(brain_unseen.dtype),\n",
            "              str(text_seen.dtype), str(text_unseen.dtype),\n",
            "              str(image_seen.dtype), str(image_unseen.dtype),\n",
            "              str(label_seen.dtype), str(label_unseen.dtype)]\n",
            "}\n",
            "\n",
            "df_summary = pd.DataFrame(data_summary)\n",
            "print(\"=\" * 60)\n",
            "print(\"DATA STRUCTURE SUMMARY\")\n",
            "print(\"=\" * 60)\n",
            "print(df_summary.to_string(index=False))\n",
            "print()\n",
            "\n",
            "# Label statistics\n",
            "unique_seen = torch.unique(label_seen)\n",
            "unique_unseen = torch.unique(label_unseen)\n",
            "\n",
            "print(f\"Seen classes: {len(unique_seen)} unique labels (range: {unique_seen.min().item()} to {unique_seen.max().item()})\")\n",
            "print(f\"Unseen classes: {len(unique_unseen)} unique labels (range: {unique_unseen.min().item()} to {unique_unseen.max().item()})\")\n",
            "print()\n",
            "print(f\"Samples per seen class (approx): {brain_seen.shape[0] / len(unique_seen):.1f}\")\n",
            "print(f\"Samples per unseen class (approx): {brain_unseen.shape[0] / len(unique_unseen):.1f}\")"
        ]
    },
    # Cell 4: Markdown - Class Distribution Header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Data Exploration: Class Distributions\n",
            "\n",
            "Visualising the number of EEG samples per class for seen and unseen categories."
        ]
    },
    # Cell 5: Code - Class Distribution Histogram
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# CLASS DISTRIBUTION HISTOGRAM\n",
            "# =============================================================================\n",
            "\n",
            "import matplotlib.pyplot as plt\n",
            "import os\n",
            "\n",
            "os.makedirs('figures', exist_ok=True)\n",
            "\n",
            "label_seen_np = label_seen.numpy().flatten()\n",
            "label_unseen_np = label_unseen.numpy().flatten()\n",
            "\n",
            "seen_class_counts = np.bincount(label_seen_np)[1:]\n",
            "unseen_class_counts = np.bincount(label_unseen_np)[1:]\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "axes[0].hist(seen_class_counts, bins=30, color='steelblue', edgecolor='black', alpha=0.7)\n",
            "axes[0].set_xlabel('Samples per Class', fontsize=12)\n",
            "axes[0].set_ylabel('Number of Classes', fontsize=12)\n",
            "axes[0].set_title(f'Seen Classes Distribution\\n({len(seen_class_counts)} classes, {len(label_seen_np)} total samples)', fontsize=13)\n",
            "axes[0].axvline(np.mean(seen_class_counts), color='red', linestyle='--', label=f'Mean: {np.mean(seen_class_counts):.1f}')\n",
            "axes[0].legend()\n",
            "\n",
            "axes[1].hist(unseen_class_counts, bins=30, color='darkorange', edgecolor='black', alpha=0.7)\n",
            "axes[1].set_xlabel('Samples per Class', fontsize=12)\n",
            "axes[1].set_ylabel('Number of Classes', fontsize=12)\n",
            "axes[1].set_title(f'Unseen Classes Distribution\\n({len(unseen_class_counts)} classes, {len(label_unseen_np)} total samples)', fontsize=13)\n",
            "axes[1].axvline(np.mean(unseen_class_counts), color='red', linestyle='--', label=f'Mean: {np.mean(unseen_class_counts):.1f}')\n",
            "axes[1].legend()\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('figures/class_distribution.png', dpi=150, bbox_inches='tight')\n",
            "plt.show()\n",
            "\n",
            "print(f\"\\nFigure saved to: figures/class_distribution.png\")"
        ]
    },
    # Cell 6: Markdown - EEG Norm Header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Data Exploration: EEG Feature Norms\n",
            "\n",
            "Visualising the distribution of EEG signal norms (L2) to understand feature variability."
        ]
    },
    # Cell 7: Code - EEG Feature Norm Distribution
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# EEG FEATURE NORM DISTRIBUTION\n",
            "# =============================================================================\n",
            "\n",
            "brain_seen_np = brain_seen.numpy()\n",
            "brain_unseen_np = brain_unseen.numpy()\n",
            "\n",
            "norms_seen = np.linalg.norm(brain_seen_np, axis=1)\n",
            "norms_unseen = np.linalg.norm(brain_unseen_np, axis=1)\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(10, 5))\n",
            "\n",
            "ax.hist(norms_seen, bins=50, alpha=0.6, label=f'Seen (n={len(norms_seen)})', color='steelblue', edgecolor='black')\n",
            "ax.hist(norms_unseen, bins=50, alpha=0.6, label=f'Unseen (n={len(norms_unseen)})', color='darkorange', edgecolor='black')\n",
            "\n",
            "ax.set_xlabel('L2 Norm of EEG Features', fontsize=12)\n",
            "ax.set_ylabel('Count', fontsize=12)\n",
            "ax.set_title('Distribution of EEG Feature Norms (Seen vs Unseen)', fontsize=13)\n",
            "ax.legend()\n",
            "\n",
            "stats_text = f'Seen: μ={np.mean(norms_seen):.2f}, σ={np.std(norms_seen):.2f}\\n'\n",
            "stats_text += f'Unseen: μ={np.mean(norms_unseen):.2f}, σ={np.std(norms_unseen):.2f}'\n",
            "ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=10,\n",
            "        verticalalignment='top', horizontalalignment='right',\n",
            "        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('figures/eeg_norm_distribution.png', dpi=150, bbox_inches='tight')\n",
            "plt.show()\n",
            "\n",
            "print(f\"\\nFigure saved to: figures/eeg_norm_distribution.png\")"
        ]
    },
    # Cell 8: Markdown - Baseline Model Header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## Baseline Model A: Logistic Regression on Raw EEG\n",
            "\n",
            "This section implements the baseline classifier using multinomial Logistic Regression on raw EEG features.\n",
            "\n",
            "**Training setup:**\n",
            "- 80/20 train/test split on seen classes\n",
            "- StandardScaler for feature normalisation\n",
            "- Multinomial Logistic Regression with LBFGS solver\n",
            "\n",
            "**Metrics:**\n",
            "- Accuracy\n",
            "- Macro F1-score"
        ]
    },
    # Cell 9: Code - Data Preparation
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# BASELINE MODEL A: DATA PREPARATION\n",
            "# =============================================================================\n",
            "\n",
            "import random\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.metrics import accuracy_score, f1_score\n",
            "\n",
            "# Set random seeds\n",
            "SEED = 42\n",
            "np.random.seed(SEED)\n",
            "torch.manual_seed(SEED)\n",
            "random.seed(SEED)\n",
            "if torch.cuda.is_available():\n",
            "    torch.cuda.manual_seed_all(SEED)\n",
            "\n",
            "print(f\"Random seed set to: {SEED}\")\n",
            "\n",
            "# Prepare data\n",
            "X_seen = brain_seen.numpy()\n",
            "y_seen = label_seen.numpy().flatten()\n",
            "\n",
            "# 80/20 train/test split\n",
            "X_train, X_test, y_train, y_test = train_test_split(\n",
            "    X_seen, y_seen, test_size=0.2, random_state=SEED, stratify=y_seen\n",
            ")\n",
            "\n",
            "print(f\"\\nTrain set: {X_train.shape[0]} samples\")\n",
            "print(f\"Test set: {X_test.shape[0]} samples\")\n",
            "print(f\"Feature dimension: {X_train.shape[1]}\")\n",
            "\n",
            "# Standardise features\n",
            "scaler = StandardScaler()\n",
            "X_train_scaled = scaler.fit_transform(X_train)\n",
            "X_test_scaled = scaler.transform(X_test)\n",
            "\n",
            "print(f\"\\nFeatures standardised (mean=0, std=1)\")"
        ]
    },
    # Cell 10: Code - Train Baseline Model
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# TRAIN BASELINE LOGISTIC REGRESSION\n",
            "# =============================================================================\n",
            "# Note: Training may take a few minutes due to the large number of classes\n",
            "\n",
            "print(\"Training Logistic Regression classifier...\")\n",
            "print(f\"Number of classes: {len(np.unique(y_train))}\")\n",
            "print(\"This may take a few minutes...\\n\")\n",
            "\n",
            "baseline_clf = LogisticRegression(\n",
            "    multi_class='multinomial',\n",
            "    solver='lbfgs',\n",
            "    max_iter=1000,\n",
            "    random_state=SEED,\n",
            "    n_jobs=-1,\n",
            "    verbose=1\n",
            ")\n",
            "\n",
            "baseline_clf.fit(X_train_scaled, y_train)\n",
            "print(\"\\nTraining complete!\")"
        ]
    },
    # Cell 11: Code - Evaluate Baseline Model
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# EVALUATE BASELINE MODEL\n",
            "# =============================================================================\n",
            "\n",
            "y_pred = baseline_clf.predict(X_test_scaled)\n",
            "\n",
            "baseline_accuracy = accuracy_score(y_test, y_pred)\n",
            "baseline_macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)\n",
            "\n",
            "# Store results\n",
            "baseline_results = {\n",
            "    'model': 'Baseline A (Logistic Regression on Raw EEG)',\n",
            "    'accuracy': baseline_accuracy,\n",
            "    'macro_f1': baseline_macro_f1,\n",
            "    'n_train': X_train.shape[0],\n",
            "    'n_test': X_test.shape[0],\n",
            "    'n_classes': len(np.unique(y_train)),\n",
            "    'n_features': X_train.shape[1]\n",
            "}\n",
            "\n",
            "print(\"=\" * 60)\n",
            "print(\"BASELINE MODEL A: EVALUATION RESULTS\")\n",
            "print(\"=\" * 60)\n",
            "print(f\"Accuracy:     {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)\")\n",
            "print(f\"Macro F1:     {baseline_macro_f1:.4f}\")\n",
            "print(\"=\" * 60)"
        ]
    },
    # Cell 12: Markdown - Summary Header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## Summary: Baseline Model and Data Exploration\n",
            "\n",
            "This section prints a comprehensive summary of the baseline implementation."
        ]
    },
    # Cell 13: Code - Final Summary
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
            "print(\"SUMMARY: BASELINE MODEL [A] IMPLEMENTATION\")\n",
            "print(\"=\"*70)\n",
            "\n",
            "print(\"\\n### 1. BASELINE PERFORMANCE METRICS ###\")\n",
            "print(f\"   Accuracy:  {baseline_results['accuracy']:.4f} ({baseline_results['accuracy']*100:.2f}%)\")\n",
            "print(f\"   Macro F1:  {baseline_results['macro_f1']:.4f}\")\n",
            "\n",
            "print(\"\\n### 2. KEY ARRAY SHAPES ###\")\n",
            "print(f\"   brain_seen:    {tuple(brain_seen.shape)}  (EEG features for seen classes)\")\n",
            "print(f\"   brain_unseen:  {tuple(brain_unseen.shape)}  (EEG features for unseen classes)\")\n",
            "print(f\"   text_seen:     {tuple(text_seen.shape)}  (CLIP text embeddings for seen)\")\n",
            "print(f\"   text_unseen:   {tuple(text_unseen.shape)}  (CLIP text embeddings for unseen)\")\n",
            "print(f\"   label_seen:    {tuple(label_seen.shape)}  (class labels for seen)\")\n",
            "print(f\"   label_unseen:  {tuple(label_unseen.shape)}  (class labels for unseen)\")\n",
            "\n",
            "print(\"\\n### 3. ASSUMPTIONS MADE ###\")\n",
            "print(\"   1. SEEN/UNSEEN SPLIT: Using pre-defined split from dataset loader\")\n",
            "print(f\"      - Seen classes: {len(torch.unique(label_seen))} (used for training baseline)\")\n",
            "print(f\"      - Unseen classes: {len(torch.unique(label_unseen))} (reserved for GZSL evaluation)\")\n",
            "print(\"   2. TRAIN/TEST SPLIT: 80/20 stratified split within seen classes\")\n",
            "print(f\"      - Training samples: {baseline_results['n_train']}\")\n",
            "print(f\"      - Test samples: {baseline_results['n_test']}\")\n",
            "print(\"   3. FEATURE PREPROCESSING: StandardScaler (Z-score normalisation)\")\n",
            "print(\"   4. LABELS: 1-indexed (labels start from 1, not 0)\")\n",
            "print(\"   5. RANDOM SEED: 42 (for reproducibility)\")\n",
            "\n",
            "print(\"\\n### 4. FIGURES GENERATED ###\")\n",
            "print(\"   - figures/class_distribution.png\")\n",
            "print(\"   - figures/eeg_norm_distribution.png\")\n",
            "\n",
            "print(\"\\n\" + \"=\"*70)\n",
            "print(\"NEXT STEPS: Implement Brain-Text CLIP Encoder\")\n",
            "print(\"=\"*70)"
        ]
    }
]


def main():
    # Check if notebook exists
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Error: Notebook not found at {NOTEBOOK_PATH}")
        return
    
    # Load the notebook
    print(f"Loading notebook: {NOTEBOOK_PATH}")
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Append new cells
    original_cell_count = len(notebook['cells'])
    notebook['cells'].extend(NEW_CELLS)
    new_cell_count = len(notebook['cells'])
    
    # Save the modified notebook
    print(f"Adding {len(NEW_CELLS)} new cells...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"\nSuccess! Created: {OUTPUT_PATH}")
    print(f"  - Original cells: {original_cell_count}")
    print(f"  - New cells added: {len(NEW_CELLS)}")
    print(f"  - Total cells: {new_cell_count}")
    print(f"\nOpen {OUTPUT_PATH} in Jupyter/Colab to run the baseline model.")


if __name__ == "__main__":
    main()
