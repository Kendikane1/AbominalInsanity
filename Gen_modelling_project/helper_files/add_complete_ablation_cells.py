#!/usr/bin/env python3
"""
Add complete ablation study cells with all 4 methods (A, B, C, D).
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
            "## Complete Ablation Study (Methods A-D)\n",
            "\n",
            "Computing all four ablation methods with corrected label handling."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# COMPLETE ABLATION: ALL METHODS A-D\n",
            "# =============================================================================\n",
            "\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "\n",
            "# We need raw EEG data for Method A\n",
            "# Recreate train/test split from raw data\n",
            "brain_seen_np = brain_seen.numpy()\n",
            "label_seen_np = label_seen.numpy().flatten()\n",
            "brain_unseen_np = brain_unseen.numpy()\n",
            "label_unseen_np = label_unseen.numpy().flatten()\n",
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
            "X_unseen_raw = brain_unseen_np\n",
            "y_unseen_raw = label_unseen_np\n",
            "\n",
            "# Remap unseen raw labels for consistent comparison\n",
            "y_unseen_raw_remapped = y_unseen_raw + LABEL_OFFSET\n",
            "\n",
            "print(\"Raw EEG data prepared for ablation.\")\n",
            "print(f\"  Train: {len(X_train_raw)}, Test: {len(X_test_raw)}, Unseen: {len(X_unseen_raw)}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# METHOD A: RAW EEG + LR (BASELINE)\n",
            "# =============================================================================\n",
            "print(\"=\"*60)\n",
            "print(\"METHOD A: Raw EEG + LR (Baseline)\")\n",
            "print(\"=\"*60)\n",
            "\n",
            "scaler_a = StandardScaler()\n",
            "X_train_a = scaler_a.fit_transform(X_train_raw)\n",
            "X_test_a = scaler_a.transform(X_test_raw)\n",
            "X_unseen_a = scaler_a.transform(X_unseen_raw)\n",
            "\n",
            "clf_a = LogisticRegression(\n",
            "    multi_class='multinomial', solver='lbfgs', max_iter=1000,\n",
            "    random_state=SEED, n_jobs=-1\n",
            ")\n",
            "clf_a.fit(X_train_a, y_train_raw)\n",
            "\n",
            "y_pred_seen_a = clf_a.predict(X_test_a)\n",
            "y_pred_unseen_a = clf_a.predict(X_unseen_a)\n",
            "\n",
            "# Note: Baseline can only predict seen labels, so unseen accuracy will be ~0\n",
            "acc_seen_a = accuracy_score(y_test_raw, y_pred_seen_a)\n",
            "f1_seen_a = f1_score(y_test_raw, y_pred_seen_a, average='macro', zero_division=0)\n",
            "\n",
            "# For unseen, compare against remapped labels (but baseline only predicts seen)\n",
            "acc_unseen_a = accuracy_score(y_unseen_raw_remapped, y_pred_unseen_a)\n",
            "f1_unseen_a = f1_score(y_unseen_raw_remapped, y_pred_unseen_a, average='macro', zero_division=0)\n",
            "\n",
            "H_a = 2 * acc_seen_a * acc_unseen_a / (acc_seen_a + acc_unseen_a) if (acc_seen_a + acc_unseen_a) > 0 else 0\n",
            "\n",
            "print(f\"Acc_seen: {acc_seen_a:.4f}, Acc_unseen: {acc_unseen_a:.4f}, H: {H_a:.4f}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# METHOD B: CLIP PROTOTYPE RETRIEVAL\n",
            "# =============================================================================\n",
            "print(\"=\"*60)\n",
            "print(\"METHOD B: CLIP Prototype Retrieval\")\n",
            "print(\"=\"*60)\n",
            "\n",
            "# Load prototypes\n",
            "S_seen_array = np.load('cached_arrays/S_seen_prototypes.npy')\n",
            "S_unseen_array = np.load('cached_arrays/S_unseen_prototypes.npy')\n",
            "seen_classes_cached = np.load('cached_arrays/seen_classes.npy')\n",
            "unseen_classes_cached = np.load('cached_arrays/unseen_classes.npy')\n",
            "\n",
            "# Remap unseen class ids to match our label convention\n",
            "unseen_classes_remapped = unseen_classes_cached + LABEL_OFFSET\n",
            "\n",
            "def prototype_retrieval(embeddings, prototypes, class_list):\n",
            "    \"\"\"Predict class by argmax cosine similarity.\"\"\"\n",
            "    sims = np.dot(embeddings, prototypes.T)\n",
            "    pred_indices = np.argmax(sims, axis=1)\n",
            "    return class_list[pred_indices]\n",
            "\n",
            "# Split-wise retrieval\n",
            "y_pred_seen_b = prototype_retrieval(E_test_seen, S_seen_array, seen_classes_cached)\n",
            "y_pred_unseen_b = prototype_retrieval(E_unseen, S_unseen_array, unseen_classes_remapped)\n",
            "\n",
            "acc_seen_b = accuracy_score(y_test_seen, y_pred_seen_b)\n",
            "f1_seen_b = f1_score(y_test_seen, y_pred_seen_b, average='macro', zero_division=0)\n",
            "acc_unseen_b = accuracy_score(y_unseen_remapped, y_pred_unseen_b)\n",
            "f1_unseen_b = f1_score(y_unseen_remapped, y_pred_unseen_b, average='macro', zero_division=0)\n",
            "\n",
            "H_b = 2 * acc_seen_b * acc_unseen_b / (acc_seen_b + acc_unseen_b) if (acc_seen_b + acc_unseen_b) > 0 else 0\n",
            "\n",
            "print(f\"Acc_seen: {acc_seen_b:.4f}, Acc_unseen: {acc_unseen_b:.4f}, H: {H_b:.4f}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# METHOD C: CLIP EMBEDDING + LR (SEEN-ONLY)\n",
            "# =============================================================================\n",
            "print(\"=\"*60)\n",
            "print(\"METHOD C: CLIP Embedding + LR (seen-only)\")\n",
            "print(\"=\"*60)\n",
            "\n",
            "clf_c = LogisticRegression(\n",
            "    multi_class='multinomial', solver='lbfgs', max_iter=1000,\n",
            "    random_state=SEED, n_jobs=-1\n",
            ")\n",
            "clf_c.fit(E_train_seen, y_train_seen)\n",
            "\n",
            "y_pred_seen_c = clf_c.predict(E_test_seen)\n",
            "y_pred_unseen_c = clf_c.predict(E_unseen)\n",
            "\n",
            "acc_seen_c = accuracy_score(y_test_seen, y_pred_seen_c)\n",
            "f1_seen_c = f1_score(y_test_seen, y_pred_seen_c, average='macro', zero_division=0)\n",
            "\n",
            "# Unseen accuracy (model only knows seen labels, so expect ~0)\n",
            "acc_unseen_c = accuracy_score(y_unseen_remapped, y_pred_unseen_c)\n",
            "f1_unseen_c = f1_score(y_unseen_remapped, y_pred_unseen_c, average='macro', zero_division=0)\n",
            "\n",
            "H_c = 2 * acc_seen_c * acc_unseen_c / (acc_seen_c + acc_unseen_c) if (acc_seen_c + acc_unseen_c) > 0 else 0\n",
            "\n",
            "print(f\"Acc_seen: {acc_seen_c:.4f}, Acc_unseen: {acc_unseen_c:.4f}, H: {H_c:.4f}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# METHOD D: FULL GZSL (CLIP + cWGAN-GP + LR) — Already computed above\n",
            "# =============================================================================\n",
            "print(\"=\"*60)\n",
            "print(\"METHOD D: CLIP + cWGAN-GP + LR (Full GZSL)\")\n",
            "print(\"=\"*60)\n",
            "\n",
            "# Use the already-computed results from clf_gzsl_fixed\n",
            "acc_seen_d = acc_seen_fixed\n",
            "acc_unseen_d = acc_unseen_fixed\n",
            "H_d = H_fixed\n",
            "f1_seen_d = f1_seen_fixed\n",
            "f1_unseen_d = f1_unseen_fixed\n",
            "\n",
            "print(f\"Acc_seen: {acc_seen_d:.4f}, Acc_unseen: {acc_unseen_d:.4f}, H: {H_d:.4f}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# COMPLETE ABLATION TABLE (A-D)\n",
            "# =============================================================================\n",
            "\n",
            "ablation_all = [\n",
            "    {'Method': 'A: Raw EEG + LR', 'Acc_seen': acc_seen_a, 'Acc_unseen': acc_unseen_a, \n",
            "     'H': H_a, 'MacroF1_seen': f1_seen_a, 'MacroF1_unseen': f1_unseen_a},\n",
            "    {'Method': 'B: CLIP Prototype', 'Acc_seen': acc_seen_b, 'Acc_unseen': acc_unseen_b,\n",
            "     'H': H_b, 'MacroF1_seen': f1_seen_b, 'MacroF1_unseen': f1_unseen_b},\n",
            "    {'Method': 'C: CLIP + LR (seen)', 'Acc_seen': acc_seen_c, 'Acc_unseen': acc_unseen_c,\n",
            "     'H': H_c, 'MacroF1_seen': f1_seen_c, 'MacroF1_unseen': f1_unseen_c},\n",
            "    {'Method': 'D: CLIP + GAN + LR', 'Acc_seen': acc_seen_d, 'Acc_unseen': acc_unseen_d,\n",
            "     'H': H_d, 'MacroF1_seen': f1_seen_d, 'MacroF1_unseen': f1_unseen_d},\n",
            "]\n",
            "\n",
            "df_ablation_all = pd.DataFrame(ablation_all).set_index('Method')\n",
            "\n",
            "print(\"=\"*80)\n",
            "print(\"COMPLETE ABLATION TABLE (METHODS A-D)\")\n",
            "print(\"=\"*80)\n",
            "print(df_ablation_all.round(4).to_string())\n",
            "print(\"=\"*80)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# COMPLETE ABLATION BAR CHART (A-D)\n",
            "# =============================================================================\n",
            "\n",
            "methods = ['A: Raw EEG', 'B: CLIP Proto', 'C: CLIP+LR', 'D: Full GZSL']\n",
            "acc_seen_all = [acc_seen_a, acc_seen_b, acc_seen_c, acc_seen_d]\n",
            "acc_unseen_all = [acc_unseen_a, acc_unseen_b, acc_unseen_c, acc_unseen_d]\n",
            "H_all = [H_a, H_b, H_c, H_d]\n",
            "\n",
            "x = np.arange(len(methods))\n",
            "width = 0.25\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(12, 6))\n",
            "bars1 = ax.bar(x - width, acc_seen_all, width, label='Acc_seen', color='steelblue')\n",
            "bars2 = ax.bar(x, acc_unseen_all, width, label='Acc_unseen', color='darkorange')\n",
            "bars3 = ax.bar(x + width, H_all, width, label='Harmonic Mean (H)', color='green')\n",
            "\n",
            "# Add value labels\n",
            "for bars in [bars1, bars2, bars3]:\n",
            "    for bar in bars:\n",
            "        h = bar.get_height()\n",
            "        ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),\n",
            "                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)\n",
            "\n",
            "ax.set_ylabel('Score', fontsize=12)\n",
            "ax.set_title('Ablation Study: GZSL Performance (Methods A-D)', fontsize=14)\n",
            "ax.set_xticks(x)\n",
            "ax.set_xticklabels(methods, fontsize=11)\n",
            "ax.legend(fontsize=11)\n",
            "ax.set_ylim(0, max(max(acc_seen_all), max(acc_unseen_all), max(H_all)) * 1.3)\n",
            "ax.grid(axis='y', alpha=0.3)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('figures/gzsl_ablation_bar.png', dpi=150, bbox_inches='tight')\n",
            "plt.show()\n",
            "print(f\"\\nSaved: figures/gzsl_ablation_bar.png\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# COMPONENT CONTRIBUTION ANALYSIS\n",
            "# =============================================================================\n",
            "\n",
            "print(\"=\"*70)\n",
            "print(\"COMPONENT CONTRIBUTION ANALYSIS\")\n",
            "print(\"=\"*70)\n",
            "print(\"\")\n",
            "\n",
            "# CLIP representation gain (C vs A)\n",
            "print(\"1. CLIP Representation (Method C vs A):\")\n",
            "print(f\"   Δ Acc_seen = {acc_seen_c - acc_seen_a:+.4f}\")\n",
            "print(f\"   → CLIP embeddings {'improve' if acc_seen_c > acc_seen_a else 'do not improve'} seen-class discrimination.\")\n",
            "print(\"\")\n",
            "\n",
            "# Prototype retrieval (B)\n",
            "print(\"2. Zero-shot with Prototypes (Method B):\")\n",
            "print(f\"   Acc_unseen (proto retrieval) = {acc_unseen_b:.4f}\")\n",
            "print(f\"   → CLIP alignment enables {acc_unseen_b*100:.1f}% unseen accuracy via nearest prototype.\")\n",
            "print(\"\")\n",
            "\n",
            "# GAN synthesis gain (D vs C)\n",
            "print(\"3. cWGAN-GP Synthesis (Method D vs C):\")\n",
            "print(f\"   Δ Acc_unseen = {acc_unseen_d - acc_unseen_c:+.4f}\")\n",
            "print(f\"   → Synthetic embeddings {'enable' if acc_unseen_d > acc_unseen_c else 'do not enable'} zero-shot transfer via classifier.\")\n",
            "print(\"\")\n",
            "\n",
            "# Overall improvement (D vs A)\n",
            "print(\"4. Overall Improvement (Method D vs A):\")\n",
            "print(f\"   Δ H = {H_d - H_a:+.4f}\")\n",
            "print(f\"   → Full model {'achieves' if H_d > H_a else 'does not achieve'} better GZSL performance.\")\n",
            "\n",
            "print(\"\\n\" + \"=\"*70)"
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
