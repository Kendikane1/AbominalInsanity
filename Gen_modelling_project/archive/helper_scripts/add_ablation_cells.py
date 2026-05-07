#!/usr/bin/env python3
"""
Script to add ablation study cells to the notebook.

Implements:
- Method A: Baseline LR (raw EEG, seen-only)
- Method B: CLIP Prototype Retrieval
- Method C: CLIP + LR (seen-only)
- Method D: Full CLIP + cWGAN-GP + LR (GZSL)

Outputs:
- Ablation summary table (pandas DataFrame)
- 2×2 bias table for full model
- figures/gzsl_ablation_bar.png
"""

import json
import os

NOTEBOOK_PATH = "COMP2261_ArizMLCW_with_baseline.ipynb"
OUTPUT_PATH = "COMP2261_ArizMLCW_with_baseline.ipynb"

NEW_CELLS = [
    # =========================================================================
    # ABLATION STUDY SECTION
    # =========================================================================
    
    # Cell 1: Markdown Header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "# Ablation Study: Attributing GZSL Performance Gains\n",
            "\n",
            "This section compares four methods to isolate the contribution of each component:\n",
            "\n",
            "| Method | Description | Training Data |\n",
            "|--------|-------------|---------------|\n",
            "| **A** | Raw EEG + LR (baseline) | Seen raw EEG only |\n",
            "| **B** | CLIP Prototype Retrieval | No training (cosine similarity) |\n",
            "| **C** | CLIP Embedding + LR | Seen CLIP embeddings only |\n",
            "| **D** | Full CLIP + cWGAN-GP + LR | Seen real + Unseen synthetic |"
        ]
    },
    
    # Cell 2: Load All Data
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# ABLATION STUDY: LOAD ALL REQUIRED DATA\n",
            "# =============================================================================\n",
            "\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "from sklearn.metrics import accuracy_score, f1_score\n",
            "from sklearn.model_selection import train_test_split\n",
            "import matplotlib.pyplot as plt\n",
            "import random\n",
            "\n",
            "SEED = 42\n",
            "np.random.seed(SEED)\n",
            "random.seed(SEED)\n",
            "\n",
            "# Load CLIP embeddings\n",
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
            "# Load prototypes and class lists\n",
            "S_seen_array = np.load('cached_arrays/S_seen_prototypes.npy')\n",
            "S_unseen_array = np.load('cached_arrays/S_unseen_prototypes.npy')\n",
            "seen_classes = np.load('cached_arrays/seen_classes.npy')\n",
            "unseen_classes = np.load('cached_arrays/unseen_classes.npy')\n",
            "\n",
            "# Build label-to-index mappings\n",
            "seen_label_to_idx = {c: i for i, c in enumerate(seen_classes)}\n",
            "unseen_label_to_idx = {c: i for i, c in enumerate(unseen_classes)}\n",
            "\n",
            "# Also need raw EEG for baseline\n",
            "# Recreate train/test split from raw data\n",
            "brain_seen_np = brain_seen.numpy()\n",
            "label_seen_np = label_seen.numpy().flatten()\n",
            "\n",
            "indices_seen = np.arange(len(brain_seen_np))\n",
            "train_idx, test_idx = train_test_split(\n",
            "    indices_seen, test_size=0.2, random_state=SEED, stratify=label_seen_np\n",
            ")\n",
            "\n",
            "X_train_raw = brain_seen_np[train_idx]\n",
            "X_test_raw = brain_seen_np[test_idx]\n",
            "y_train_raw = label_seen_np[train_idx]\n",
            "y_test_raw = label_seen_np[test_idx]\n",
            "X_unseen_raw = brain_unseen.numpy()\n",
            "y_unseen_raw = label_unseen.numpy().flatten()\n",
            "\n",
            "print(\"Data loaded for ablation study.\")\n",
            "print(f\"  Seen classes: {len(seen_classes)}, Unseen classes: {len(unseen_classes)}\")"
        ]
    },
    
    # Cell 3: Helper Functions
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# HELPER FUNCTIONS FOR ABLATION\n",
            "# =============================================================================\n",
            "\n",
            "def compute_metrics(y_true, y_pred):\n",
            "    \"\"\"Compute accuracy and macro F1.\"\"\"\n",
            "    acc = accuracy_score(y_true, y_pred)\n",
            "    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)\n",
            "    return acc, f1\n",
            "\n",
            "def compute_harmonic_mean(acc_seen, acc_unseen):\n",
            "    \"\"\"Compute H = 2 * Acc_S * Acc_U / (Acc_S + Acc_U).\"\"\"\n",
            "    if acc_seen + acc_unseen > 0:\n",
            "        return 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)\n",
            "    return 0.0\n",
            "\n",
            "def prototype_retrieval(embeddings, prototypes, class_list):\n",
            "    \"\"\"Predict class by argmax cosine similarity to prototypes.\"\"\"\n",
            "    sims = np.dot(embeddings, prototypes.T)  # (N, n_classes)\n",
            "    pred_indices = np.argmax(sims, axis=1)\n",
            "    pred_labels = class_list[pred_indices]\n",
            "    return pred_labels\n",
            "\n",
            "# Store all ablation results\n",
            "ablation_results = []"
        ]
    },
    
    # Cell 4: Method A - Baseline Raw EEG LR
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# METHOD A: BASELINE LR (RAW EEG, SEEN-ONLY)\n",
            "# =============================================================================\n",
            "print(\"=\"*60)\n",
            "print(\"METHOD A: Raw EEG + LR (Baseline)\")\n",
            "print(\"=\"*60)\n",
            "\n",
            "# Standardise\n",
            "scaler_a = StandardScaler()\n",
            "X_train_a = scaler_a.fit_transform(X_train_raw)\n",
            "X_test_a = scaler_a.transform(X_test_raw)\n",
            "X_unseen_a = scaler_a.transform(X_unseen_raw)\n",
            "\n",
            "# Train\n",
            "clf_a = LogisticRegression(\n",
            "    multi_class='multinomial', solver='lbfgs', max_iter=1000, \n",
            "    random_state=SEED, n_jobs=-1\n",
            ")\n",
            "clf_a.fit(X_train_a, y_train_raw)\n",
            "\n",
            "# Evaluate\n",
            "y_pred_seen_a = clf_a.predict(X_test_a)\n",
            "y_pred_unseen_a = clf_a.predict(X_unseen_a)\n",
            "\n",
            "acc_seen_a, f1_seen_a = compute_metrics(y_test_raw, y_pred_seen_a)\n",
            "acc_unseen_a, f1_unseen_a = compute_metrics(y_unseen_raw, y_pred_unseen_a)\n",
            "H_a = compute_harmonic_mean(acc_seen_a, acc_unseen_a)\n",
            "\n",
            "print(f\"  Acc_seen:  {acc_seen_a:.4f}, Acc_unseen: {acc_unseen_a:.4f}, H: {H_a:.4f}\")\n",
            "\n",
            "ablation_results.append({\n",
            "    'Method': 'Raw EEG LR (A)',\n",
            "    'Acc_seen': acc_seen_a,\n",
            "    'Acc_unseen': acc_unseen_a,\n",
            "    'H': H_a,\n",
            "    'MacroF1_seen': f1_seen_a,\n",
            "    'MacroF1_unseen': f1_unseen_a\n",
            "})"
        ]
    },
    
    # Cell 5: Method B - CLIP Prototype Retrieval
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# METHOD B: CLIP PROTOTYPE RETRIEVAL (NO CLASSIFIER)\n",
            "# =============================================================================\n",
            "print(\"=\"*60)\n",
            "print(\"METHOD B: CLIP Prototype Retrieval\")\n",
            "print(\"=\"*60)\n",
            "\n",
            "# Split-wise retrieval (recommended variant)\n",
            "# Seen test: predict among seen prototypes only\n",
            "y_pred_seen_b = prototype_retrieval(E_test_seen, S_seen_array, seen_classes)\n",
            "\n",
            "# Unseen test: predict among unseen prototypes only\n",
            "y_pred_unseen_b = prototype_retrieval(E_unseen, S_unseen_array, unseen_classes)\n",
            "\n",
            "acc_seen_b, f1_seen_b = compute_metrics(y_test_seen, y_pred_seen_b)\n",
            "acc_unseen_b, f1_unseen_b = compute_metrics(y_unseen, y_pred_unseen_b)\n",
            "H_b = compute_harmonic_mean(acc_seen_b, acc_unseen_b)\n",
            "\n",
            "print(f\"  Acc_seen:  {acc_seen_b:.4f}, Acc_unseen: {acc_unseen_b:.4f}, H: {H_b:.4f}\")\n",
            "\n",
            "ablation_results.append({\n",
            "    'Method': 'CLIP Prototype Retrieval (B)',\n",
            "    'Acc_seen': acc_seen_b,\n",
            "    'Acc_unseen': acc_unseen_b,\n",
            "    'H': H_b,\n",
            "    'MacroF1_seen': f1_seen_b,\n",
            "    'MacroF1_unseen': f1_unseen_b\n",
            "})"
        ]
    },
    
    # Cell 6: Method C - CLIP Embedding + LR (seen-only)
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# METHOD C: CLIP EMBEDDING + LR (SEEN-ONLY CLASSIFIER)\n",
            "# =============================================================================\n",
            "print(\"=\"*60)\n",
            "print(\"METHOD C: CLIP Embedding + LR (seen-only)\")\n",
            "print(\"=\"*60)\n",
            "\n",
            "# Train LR on CLIP embeddings (seen only)\n",
            "clf_c = LogisticRegression(\n",
            "    multi_class='multinomial', solver='lbfgs', max_iter=1000,\n",
            "    random_state=SEED, n_jobs=-1\n",
            ")\n",
            "clf_c.fit(E_train_seen, y_train_seen)\n",
            "\n",
            "# Evaluate\n",
            "y_pred_seen_c = clf_c.predict(E_test_seen)\n",
            "y_pred_unseen_c = clf_c.predict(E_unseen)\n",
            "\n",
            "acc_seen_c, f1_seen_c = compute_metrics(y_test_seen, y_pred_seen_c)\n",
            "acc_unseen_c, f1_unseen_c = compute_metrics(y_unseen, y_pred_unseen_c)\n",
            "H_c = compute_harmonic_mean(acc_seen_c, acc_unseen_c)\n",
            "\n",
            "print(f\"  Acc_seen:  {acc_seen_c:.4f}, Acc_unseen: {acc_unseen_c:.4f}, H: {H_c:.4f}\")\n",
            "\n",
            "ablation_results.append({\n",
            "    'Method': 'CLIP Embedding + LR (C)',\n",
            "    'Acc_seen': acc_seen_c,\n",
            "    'Acc_unseen': acc_unseen_c,\n",
            "    'H': H_c,\n",
            "    'MacroF1_seen': f1_seen_c,\n",
            "    'MacroF1_unseen': f1_unseen_c\n",
            "})"
        ]
    },
    
    # Cell 7: Method D - Full GZSL
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# METHOD D: FULL CLIP + cWGAN-GP + LR (GZSL)\n",
            "# =============================================================================\n",
            "print(\"=\"*60)\n",
            "print(\"METHOD D: CLIP + cWGAN-GP + LR (Full A+B)\")\n",
            "print(\"=\"*60)\n",
            "\n",
            "# Combine real seen + synthetic unseen\n",
            "X_train_d = np.vstack([E_train_seen, E_synth_unseen])\n",
            "y_train_d = np.concatenate([y_train_seen, y_synth_unseen])\n",
            "\n",
            "# Train LR\n",
            "clf_d = LogisticRegression(\n",
            "    multi_class='multinomial', solver='lbfgs', max_iter=1000,\n",
            "    random_state=SEED, n_jobs=-1\n",
            ")\n",
            "clf_d.fit(X_train_d, y_train_d)\n",
            "\n",
            "# Evaluate\n",
            "y_pred_seen_d = clf_d.predict(E_test_seen)\n",
            "y_pred_unseen_d = clf_d.predict(E_unseen)\n",
            "\n",
            "acc_seen_d, f1_seen_d = compute_metrics(y_test_seen, y_pred_seen_d)\n",
            "acc_unseen_d, f1_unseen_d = compute_metrics(y_unseen, y_pred_unseen_d)\n",
            "H_d = compute_harmonic_mean(acc_seen_d, acc_unseen_d)\n",
            "\n",
            "print(f\"  Acc_seen:  {acc_seen_d:.4f}, Acc_unseen: {acc_unseen_d:.4f}, H: {H_d:.4f}\")\n",
            "\n",
            "ablation_results.append({\n",
            "    'Method': 'CLIP + cWGAN-GP + LR (D)',\n",
            "    'Acc_seen': acc_seen_d,\n",
            "    'Acc_unseen': acc_unseen_d,\n",
            "    'H': H_d,\n",
            "    'MacroF1_seen': f1_seen_d,\n",
            "    'MacroF1_unseen': f1_unseen_d\n",
            "})"
        ]
    },
    
    # Cell 8: Ablation Summary Table
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# ABLATION SUMMARY TABLE\n",
            "# =============================================================================\n",
            "\n",
            "df_ablation = pd.DataFrame(ablation_results)\n",
            "df_ablation = df_ablation.set_index('Method')\n",
            "\n",
            "# Format for display\n",
            "print(\"=\"*80)\n",
            "print(\"ABLATION STUDY: SUMMARY TABLE\")\n",
            "print(\"=\"*80)\n",
            "print(df_ablation.to_string(float_format='%.4f'))\n",
            "print(\"=\"*80)"
        ]
    },
    
    # Cell 9: 2x2 Bias Table for Full Model
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# 2×2 BIAS TABLE: TRUE GROUP vs PREDICTED GROUP (FULL MODEL D)\n",
            "# =============================================================================\n",
            "\n",
            "seen_label_set = set(seen_classes)\n",
            "unseen_label_set = set(unseen_classes)\n",
            "\n",
            "def count_group_predictions(y_true, y_pred, true_is_seen):\n",
            "    \"\"\"Count predictions falling into seen vs unseen label space.\"\"\"\n",
            "    n_pred_seen = sum(1 for p in y_pred if p in seen_label_set)\n",
            "    n_pred_unseen = sum(1 for p in y_pred if p in unseen_label_set)\n",
            "    n_total = len(y_pred)\n",
            "    return n_pred_seen, n_pred_unseen, n_total\n",
            "\n",
            "# Seen test samples\n",
            "n_seen_pred_s, n_seen_pred_u, n_seen_total = count_group_predictions(\n",
            "    y_test_seen, y_pred_seen_d, True\n",
            ")\n",
            "\n",
            "# Unseen test samples\n",
            "n_unseen_pred_s, n_unseen_pred_u, n_unseen_total = count_group_predictions(\n",
            "    y_unseen, y_pred_unseen_d, False\n",
            ")\n",
            "\n",
            "# Create bias table\n",
            "bias_table = pd.DataFrame({\n",
            "    'Pred: Seen': [\n",
            "        f\"{n_seen_pred_s} ({100*n_seen_pred_s/n_seen_total:.1f}%)\",\n",
            "        f\"{n_unseen_pred_s} ({100*n_unseen_pred_s/n_unseen_total:.1f}%)\"\n",
            "    ],\n",
            "    'Pred: Unseen': [\n",
            "        f\"{n_seen_pred_u} ({100*n_seen_pred_u/n_seen_total:.1f}%)\",\n",
            "        f\"{n_unseen_pred_u} ({100*n_unseen_pred_u/n_unseen_total:.1f}%)\"\n",
            "    ]\n",
            "}, index=['True: Seen', 'True: Unseen'])\n",
            "\n",
            "print(\"=\"*60)\n",
            "print(\"2×2 BIAS TABLE: TRUE GROUP vs PREDICTED GROUP (Full Model D)\")\n",
            "print(\"=\"*60)\n",
            "print(bias_table.to_string())\n",
            "print(\"=\"*60)\n",
            "print(\"\")\n",
            "print(\"Interpretation:\")\n",
            "print(f\"  - Seen samples predicted as seen:     {100*n_seen_pred_s/n_seen_total:.1f}%\")\n",
            "print(f\"  - Unseen samples predicted as unseen: {100*n_unseen_pred_u/n_unseen_total:.1f}%\")"
        ]
    },
    
    # Cell 10: Ablation Bar Chart
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# ABLATION BAR CHART\n",
            "# =============================================================================\n",
            "\n",
            "methods = ['Raw EEG (A)', 'CLIP Proto (B)', 'CLIP+LR (C)', 'Full GZSL (D)']\n",
            "acc_seen_vals = [acc_seen_a, acc_seen_b, acc_seen_c, acc_seen_d]\n",
            "acc_unseen_vals = [acc_unseen_a, acc_unseen_b, acc_unseen_c, acc_unseen_d]\n",
            "H_vals = [H_a, H_b, H_c, H_d]\n",
            "\n",
            "x = np.arange(len(methods))\n",
            "width = 0.25\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(12, 6))\n",
            "\n",
            "bars1 = ax.bar(x - width, acc_seen_vals, width, label='Acc_seen', color='steelblue')\n",
            "bars2 = ax.bar(x, acc_unseen_vals, width, label='Acc_unseen', color='darkorange')\n",
            "bars3 = ax.bar(x + width, H_vals, width, label='Harmonic Mean (H)', color='green')\n",
            "\n",
            "# Add value labels\n",
            "for bars in [bars1, bars2, bars3]:\n",
            "    for bar in bars:\n",
            "        height = bar.get_height()\n",
            "        ax.annotate(f'{height:.3f}',\n",
            "                    xy=(bar.get_x() + bar.get_width() / 2, height),\n",
            "                    xytext=(0, 3), textcoords=\"offset points\",\n",
            "                    ha='center', va='bottom', fontsize=9)\n",
            "\n",
            "ax.set_ylabel('Score', fontsize=12)\n",
            "ax.set_title('Ablation Study: GZSL Performance by Method', fontsize=14)\n",
            "ax.set_xticks(x)\n",
            "ax.set_xticklabels(methods, fontsize=11)\n",
            "ax.legend(fontsize=11)\n",
            "ax.set_ylim(0, max(max(acc_seen_vals), max(acc_unseen_vals), max(H_vals)) * 1.25)\n",
            "ax.grid(axis='y', alpha=0.3)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('figures/gzsl_ablation_bar.png', dpi=150, bbox_inches='tight')\n",
            "plt.show()\n",
            "\n",
            "print(f\"Saved: figures/gzsl_ablation_bar.png\")"
        ]
    },
    
    # Cell 11: Final Ablation Summary
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# ABLATION STUDY: KEY FINDINGS\n",
            "# =============================================================================\n",
            "\n",
            "print(\"=\"*70)\n",
            "print(\"ABLATION STUDY: KEY FINDINGS\")\n",
            "print(\"=\"*70)\n",
            "\n",
            "print(\"\\n### COMPONENT CONTRIBUTIONS ###\")\n",
            "print(\"\")\n",
            "\n",
            "# CLIP representation gain\n",
            "clip_gain_seen = acc_seen_c - acc_seen_a\n",
            "print(f\"1. CLIP Representation (C vs A):\")\n",
            "print(f\"   Δ Acc_seen = {clip_gain_seen:+.4f}\")\n",
            "print(f\"   → CLIP embeddings {'improve' if clip_gain_seen > 0 else 'do not improve'} seen-class performance.\")\n",
            "\n",
            "# cWGAN-GP generation gain\n",
            "gan_gain_unseen = acc_unseen_d - acc_unseen_c\n",
            "print(f\"\")\n",
            "print(f\"2. cWGAN-GP Synthesis (D vs C):\")\n",
            "print(f\"   Δ Acc_unseen = {gan_gain_unseen:+.4f}\")\n",
            "print(f\"   → Synthetic embeddings {'enable' if gan_gain_unseen > 0 else 'do not enable'} zero-shot transfer.\")\n",
            "\n",
            "# Overall improvement\n",
            "overall_H_gain = H_d - H_a\n",
            "print(f\"\")\n",
            "print(f\"3. Overall Improvement (D vs A):\")\n",
            "print(f\"   Δ H = {overall_H_gain:+.4f}\")\n",
            "print(f\"   → Full model {'achieves' if overall_H_gain > 0 else 'does not achieve'} better GZSL trade-off.\")\n",
            "\n",
            "print(\"\")\n",
            "print(\"### ABLATION TABLE (PANDAS) ###\")\n",
            "print(\"\")\n",
            "print(df_ablation.round(4).to_string())\n",
            "\n",
            "print(\"\\n\" + \"=\"*70)\n",
            "print(\"ABLATION COMPLETE — awaiting confirmation before further analysis.\")\n",
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
