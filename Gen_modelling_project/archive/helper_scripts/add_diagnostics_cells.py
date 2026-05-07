#!/usr/bin/env python3
"""
Script to add diagnostic cells for CLIP + cWGAN-GP outputs.
"""

import json
import os

NOTEBOOK_PATH = "COMP2261_ArizMLCW_with_baseline.ipynb"
OUTPUT_PATH = "COMP2261_ArizMLCW_with_baseline.ipynb"

NEW_CELLS = [
    # =========================================================================
    # DIAGNOSTICS SECTION
    # =========================================================================
    
    # Cell 1: Markdown Header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "# Quantitative Diagnostics: CLIP + cWGAN-GP\n",
            "\n",
            "Before training the GZSL classifier, we run two diagnostic checks:\n",
            "\n",
            "1. **Prototype Alignment**: Verify embeddings are closer to their own class prototype than to others\n",
            "2. **Class-Conditional Diversity**: Check synthetic embeddings haven't collapsed (mode collapse)\n",
            "\n",
            "These diagnostics use **only cached arrays** — no retraining of CLIP or WGAN."
        ]
    },
    
    # Cell 2: Load Cached Arrays
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# LOAD CACHED ARRAYS FOR DIAGNOSTICS\n",
            "# =============================================================================\n",
            "\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import os\n",
            "\n",
            "# Load embeddings\n",
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
            "# Load prototypes\n",
            "S_seen_array = np.load('cached_arrays/S_seen_prototypes.npy')\n",
            "S_unseen_array = np.load('cached_arrays/S_unseen_prototypes.npy')\n",
            "seen_classes = np.load('cached_arrays/seen_classes.npy')\n",
            "unseen_classes = np.load('cached_arrays/unseen_classes.npy')\n",
            "\n",
            "# Build label-to-index mappings (labels are 1-indexed, arrays are 0-indexed)\n",
            "seen_label_to_idx = {c: i for i, c in enumerate(seen_classes)}\n",
            "unseen_label_to_idx = {c: i for i, c in enumerate(unseen_classes)}\n",
            "\n",
            "print(\"Loaded arrays:\")\n",
            "print(f\"  E_test_seen: {E_test_seen.shape}\")\n",
            "print(f\"  E_synth_unseen: {E_synth_unseen.shape}\")\n",
            "print(f\"  S_seen_array: {S_seen_array.shape} ({len(seen_classes)} classes)\")\n",
            "print(f\"  S_unseen_array: {S_unseen_array.shape} ({len(unseen_classes)} classes)\")"
        ]
    },
    
    # Cell 3: Markdown - Check 1 Header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Check 1: Prototype Alignment\n",
            "\n",
            "Verify that embeddings are **closer to their own class prototype** than to random other prototypes.\n",
            "\n",
            "Metrics:\n",
            "- **sim_pos**: cosine similarity between embedding and its own class prototype\n",
            "- **sim_neg**: cosine similarity between embedding and random other prototypes\n",
            "- **margin**: sim_pos - sim_neg (should be positive)"
        ]
    },
    
    # Cell 4: Code - Prototype Alignment for Seen
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# CHECK 1A: PROTOTYPE ALIGNMENT — SEEN (REAL) EMBEDDINGS\n",
            "# =============================================================================\n",
            "\n",
            "np.random.seed(42)\n",
            "n_neg_samples = 10  # Number of negative prototypes to sample per embedding\n",
            "\n",
            "sim_pos_seen = []\n",
            "sim_neg_seen = []\n",
            "\n",
            "for i in range(len(E_test_seen)):\n",
            "    e = E_test_seen[i]\n",
            "    label = y_test_seen[i]\n",
            "    \n",
            "    # Get the correct prototype index\n",
            "    if label not in seen_label_to_idx:\n",
            "        continue\n",
            "    pos_idx = seen_label_to_idx[label]\n",
            "    \n",
            "    # Positive similarity (dot product = cosine since L2-normalised)\n",
            "    s_pos = np.dot(e, S_seen_array[pos_idx])\n",
            "    sim_pos_seen.append(s_pos)\n",
            "    \n",
            "    # Negative similarities (sample random other classes)\n",
            "    other_indices = [j for j in range(len(seen_classes)) if j != pos_idx]\n",
            "    neg_indices = np.random.choice(other_indices, min(n_neg_samples, len(other_indices)), replace=False)\n",
            "    for neg_idx in neg_indices:\n",
            "        s_neg = np.dot(e, S_seen_array[neg_idx])\n",
            "        sim_neg_seen.append(s_neg)\n",
            "\n",
            "sim_pos_seen = np.array(sim_pos_seen)\n",
            "sim_neg_seen = np.array(sim_neg_seen)\n",
            "\n",
            "print(\"=\" * 60)\n",
            "print(\"CHECK 1A: PROTOTYPE ALIGNMENT — SEEN (REAL)\")\n",
            "print(\"=\" * 60)\n",
            "print(f\"Positive similarities (e vs own proto):\")\n",
            "print(f\"  Mean: {sim_pos_seen.mean():.4f} ± {sim_pos_seen.std():.4f}\")\n",
            "print(f\"Negative similarities (e vs random other proto):\")\n",
            "print(f\"  Mean: {sim_neg_seen.mean():.4f} ± {sim_neg_seen.std():.4f}\")\n",
            "print(f\"Mean Margin (pos - neg): {sim_pos_seen.mean() - sim_neg_seen.mean():.4f}\")"
        ]
    },
    
    # Cell 5: Code - Plot Seen Alignment
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# PLOT: SEEN PROTOTYPE ALIGNMENT HISTOGRAM\n",
            "# =============================================================================\n",
            "\n",
            "plt.figure(figsize=(10, 5))\n",
            "plt.hist(sim_pos_seen, bins=50, alpha=0.7, label=f'Positive (own proto), μ={sim_pos_seen.mean():.3f}', color='green', density=True)\n",
            "plt.hist(sim_neg_seen, bins=50, alpha=0.7, label=f'Negative (other proto), μ={sim_neg_seen.mean():.3f}', color='red', density=True)\n",
            "plt.xlabel('Cosine Similarity', fontsize=12)\n",
            "plt.ylabel('Density', fontsize=12)\n",
            "plt.title('Check 1A: Seen Embeddings — Prototype Alignment', fontsize=14)\n",
            "plt.legend()\n",
            "plt.axvline(sim_pos_seen.mean(), color='darkgreen', linestyle='--', linewidth=2)\n",
            "plt.axvline(sim_neg_seen.mean(), color='darkred', linestyle='--', linewidth=2)\n",
            "plt.tight_layout()\n",
            "plt.savefig('figures/prototype_alignment_seen.png', dpi=150, bbox_inches='tight')\n",
            "plt.show()\n",
            "\n",
            "print(f\"Saved: figures/prototype_alignment_seen.png\")"
        ]
    },
    
    # Cell 6: Code - Prototype Alignment for Synthetic Unseen
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# CHECK 1B: PROTOTYPE ALIGNMENT — UNSEEN (SYNTHETIC) EMBEDDINGS\n",
            "# =============================================================================\n",
            "\n",
            "np.random.seed(42)\n",
            "\n",
            "sim_pos_synth = []\n",
            "sim_neg_synth = []\n",
            "\n",
            "for i in range(len(E_synth_unseen)):\n",
            "    e_hat = E_synth_unseen[i]\n",
            "    label = y_synth_unseen[i]\n",
            "    \n",
            "    if label not in unseen_label_to_idx:\n",
            "        continue\n",
            "    pos_idx = unseen_label_to_idx[label]\n",
            "    \n",
            "    # Positive similarity\n",
            "    s_pos = np.dot(e_hat, S_unseen_array[pos_idx])\n",
            "    sim_pos_synth.append(s_pos)\n",
            "    \n",
            "    # Negative similarities\n",
            "    other_indices = [j for j in range(len(unseen_classes)) if j != pos_idx]\n",
            "    neg_indices = np.random.choice(other_indices, min(n_neg_samples, len(other_indices)), replace=False)\n",
            "    for neg_idx in neg_indices:\n",
            "        s_neg = np.dot(e_hat, S_unseen_array[neg_idx])\n",
            "        sim_neg_synth.append(s_neg)\n",
            "\n",
            "sim_pos_synth = np.array(sim_pos_synth)\n",
            "sim_neg_synth = np.array(sim_neg_synth)\n",
            "\n",
            "print(\"=\" * 60)\n",
            "print(\"CHECK 1B: PROTOTYPE ALIGNMENT — UNSEEN (SYNTHETIC)\")\n",
            "print(\"=\" * 60)\n",
            "print(f\"Positive similarities (e_hat vs own proto):\")\n",
            "print(f\"  Mean: {sim_pos_synth.mean():.4f} ± {sim_pos_synth.std():.4f}\")\n",
            "print(f\"Negative similarities (e_hat vs random other proto):\")\n",
            "print(f\"  Mean: {sim_neg_synth.mean():.4f} ± {sim_neg_synth.std():.4f}\")\n",
            "print(f\"Mean Margin (pos - neg): {sim_pos_synth.mean() - sim_neg_synth.mean():.4f}\")"
        ]
    },
    
    # Cell 7: Code - Plot Synthetic Alignment
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# PLOT: SYNTHETIC UNSEEN PROTOTYPE ALIGNMENT HISTOGRAM\n",
            "# =============================================================================\n",
            "\n",
            "plt.figure(figsize=(10, 5))\n",
            "plt.hist(sim_pos_synth, bins=50, alpha=0.7, label=f'Positive (own proto), μ={sim_pos_synth.mean():.3f}', color='green', density=True)\n",
            "plt.hist(sim_neg_synth, bins=50, alpha=0.7, label=f'Negative (other proto), μ={sim_neg_synth.mean():.3f}', color='red', density=True)\n",
            "plt.xlabel('Cosine Similarity', fontsize=12)\n",
            "plt.ylabel('Density', fontsize=12)\n",
            "plt.title('Check 1B: Synthetic Unseen Embeddings — Prototype Alignment', fontsize=14)\n",
            "plt.legend()\n",
            "plt.axvline(sim_pos_synth.mean(), color='darkgreen', linestyle='--', linewidth=2)\n",
            "plt.axvline(sim_neg_synth.mean(), color='darkred', linestyle='--', linewidth=2)\n",
            "plt.tight_layout()\n",
            "plt.savefig('figures/prototype_alignment_unseen_synth.png', dpi=150, bbox_inches='tight')\n",
            "plt.show()\n",
            "\n",
            "print(f\"Saved: figures/prototype_alignment_unseen_synth.png\")"
        ]
    },
    
    # Cell 8: Code - Top-1 Prototype Retrieval Accuracy
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# CHECK 1C: TOP-1 PROTOTYPE RETRIEVAL ACCURACY\n",
            "# =============================================================================\n",
            "\n",
            "def top1_prototype_accuracy(embeddings, labels, prototypes, label_to_idx, class_list):\n",
            "    \"\"\"Compute top-1 accuracy by predicting class = argmax cosine(e, proto).\"\"\"\n",
            "    correct = 0\n",
            "    total = 0\n",
            "    \n",
            "    # Compute all similarities at once\n",
            "    similarities = np.dot(embeddings, prototypes.T)  # (N, n_classes)\n",
            "    pred_indices = np.argmax(similarities, axis=1)  # (N,)\n",
            "    pred_labels = class_list[pred_indices]\n",
            "    \n",
            "    accuracy = (pred_labels == labels).mean()\n",
            "    return accuracy\n",
            "\n",
            "# Seen embeddings retrieval accuracy\n",
            "acc_seen_retrieval = top1_prototype_accuracy(\n",
            "    E_test_seen, y_test_seen, S_seen_array, seen_label_to_idx, seen_classes\n",
            ")\n",
            "\n",
            "# Synthetic unseen retrieval accuracy\n",
            "acc_synth_retrieval = top1_prototype_accuracy(\n",
            "    E_synth_unseen, y_synth_unseen, S_unseen_array, unseen_label_to_idx, unseen_classes\n",
            ")\n",
            "\n",
            "print(\"=\" * 60)\n",
            "print(\"CHECK 1C: TOP-1 PROTOTYPE RETRIEVAL ACCURACY\")\n",
            "print(\"=\" * 60)\n",
            "print(f\"Seen (real) test embeddings:     {acc_seen_retrieval:.4f} ({acc_seen_retrieval*100:.2f}%)\")\n",
            "print(f\"Unseen (synthetic) embeddings:   {acc_synth_retrieval:.4f} ({acc_synth_retrieval*100:.2f}%)\")"
        ]
    },
    
    # Cell 9: Markdown - Check 2 Header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Check 2: Class-Conditional Diversity\n",
            "\n",
            "Check that synthetic embeddings have **sufficient diversity within each class** (no mode collapse).\n",
            "\n",
            "For each unseen class:\n",
            "- Average pairwise cosine similarity within the class\n",
            "- Average pairwise cosine distance = 1 - similarity\n",
            "- Per-dimension variance"
        ]
    },
    
    # Cell 10: Code - Diversity Check
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# CHECK 2: CLASS-CONDITIONAL DIVERSITY (MODE COLLAPSE CHECK)\n",
            "# =============================================================================\n",
            "\n",
            "class_diversity_stats = []\n",
            "\n",
            "for c in unseen_classes:\n",
            "    mask = y_synth_unseen == c\n",
            "    E_class = E_synth_unseen[mask]\n",
            "    \n",
            "    if len(E_class) < 2:\n",
            "        continue\n",
            "    \n",
            "    # Pairwise cosine similarities (upper triangle only)\n",
            "    sim_matrix = np.dot(E_class, E_class.T)\n",
            "    n = len(E_class)\n",
            "    upper_tri_idx = np.triu_indices(n, k=1)\n",
            "    pairwise_sims = sim_matrix[upper_tri_idx]\n",
            "    \n",
            "    avg_sim = pairwise_sims.mean()\n",
            "    avg_dist = 1 - avg_sim\n",
            "    \n",
            "    # Per-dimension variance\n",
            "    per_dim_var = E_class.var(axis=0).mean()\n",
            "    \n",
            "    class_diversity_stats.append({\n",
            "        'class': c,\n",
            "        'avg_sim': avg_sim,\n",
            "        'avg_dist': avg_dist,\n",
            "        'per_dim_var': per_dim_var,\n",
            "        'n_samples': len(E_class)\n",
            "    })\n",
            "\n",
            "# Convert to arrays for analysis\n",
            "avg_sims = np.array([s['avg_sim'] for s in class_diversity_stats])\n",
            "avg_dists = np.array([s['avg_dist'] for s in class_diversity_stats])\n",
            "per_dim_vars = np.array([s['per_dim_var'] for s in class_diversity_stats])\n",
            "\n",
            "print(\"=\" * 60)\n",
            "print(\"CHECK 2: CLASS-CONDITIONAL DIVERSITY\")\n",
            "print(\"=\" * 60)\n",
            "print(f\"Across {len(class_diversity_stats)} unseen classes:\")\n",
            "print(f\"  Avg within-class cosine similarity: {avg_sims.mean():.4f} ± {avg_sims.std():.4f}\")\n",
            "print(f\"  Avg within-class cosine distance:   {avg_dists.mean():.4f} ± {avg_dists.std():.4f}\")\n",
            "print(f\"  Avg per-dimension variance:         {per_dim_vars.mean():.6f} ± {per_dim_vars.std():.6f}\")"
        ]
    },
    
    # Cell 11: Code - Identify Low/High Diversity Classes
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# IDENTIFY LOW AND HIGH DIVERSITY CLASSES\n",
            "# =============================================================================\n",
            "\n",
            "# Sort by diversity (distance) — low distance = low diversity = potential collapse\n",
            "sorted_stats = sorted(class_diversity_stats, key=lambda x: x['avg_dist'])\n",
            "\n",
            "print(\"\\n5 LOWEST DIVERSITY CLASSES (potential mode collapse):\")\n",
            "print(\"-\" * 50)\n",
            "for s in sorted_stats[:5]:\n",
            "    print(f\"  Class {s['class']:4d}: avg_dist={s['avg_dist']:.4f}, per_dim_var={s['per_dim_var']:.6f}\")\n",
            "\n",
            "print(\"\\n5 HIGHEST DIVERSITY CLASSES:\")\n",
            "print(\"-\" * 50)\n",
            "for s in sorted_stats[-5:]:\n",
            "    print(f\"  Class {s['class']:4d}: avg_dist={s['avg_dist']:.4f}, per_dim_var={s['per_dim_var']:.6f}\")"
        ]
    },
    
    # Cell 12: Code - Plot Diversity Distribution
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# PLOT: DIVERSITY DISTRIBUTION ACROSS CLASSES\n",
            "# =============================================================================\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "# Histogram of average cosine distance\n",
            "axes[0].hist(avg_dists, bins=30, color='teal', edgecolor='black', alpha=0.7)\n",
            "axes[0].axvline(avg_dists.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_dists.mean():.4f}')\n",
            "axes[0].set_xlabel('Avg Within-Class Cosine Distance', fontsize=12)\n",
            "axes[0].set_ylabel('Number of Classes', fontsize=12)\n",
            "axes[0].set_title('Synthetic Embedding Diversity by Class', fontsize=14)\n",
            "axes[0].legend()\n",
            "\n",
            "# Sorted bar plot\n",
            "sorted_dists = np.sort(avg_dists)\n",
            "axes[1].bar(range(len(sorted_dists)), sorted_dists, color='teal', alpha=0.7)\n",
            "axes[1].axhline(avg_dists.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_dists.mean():.4f}')\n",
            "axes[1].set_xlabel('Class (sorted by diversity)', fontsize=12)\n",
            "axes[1].set_ylabel('Avg Within-Class Cosine Distance', fontsize=12)\n",
            "axes[1].set_title('Per-Class Diversity (Sorted)', fontsize=14)\n",
            "axes[1].legend()\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('figures/synth_diversity_by_class.png', dpi=150, bbox_inches='tight')\n",
            "plt.show()\n",
            "\n",
            "print(f\"Saved: figures/synth_diversity_by_class.png\")"
        ]
    },
    
    # Cell 13: Code - Final Diagnostics Summary
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# DIAGNOSTICS SUMMARY\n",
            "# =============================================================================\n",
            "\n",
            "print(\"=\"*70)\n",
            "print(\"DIAGNOSTICS SUMMARY: CLIP + cWGAN-GP\")\n",
            "print(\"=\"*70)\n",
            "\n",
            "print(\"\\n### CHECK 1: PROTOTYPE ALIGNMENT ###\")\n",
            "print(\"\")\n",
            "print(\"  (A) Seen (real) embeddings:\")\n",
            "print(f\"      Positive sim: {sim_pos_seen.mean():.4f} ± {sim_pos_seen.std():.4f}\")\n",
            "print(f\"      Negative sim: {sim_neg_seen.mean():.4f} ± {sim_neg_seen.std():.4f}\")\n",
            "print(f\"      Margin:       {sim_pos_seen.mean() - sim_neg_seen.mean():.4f}\")\n",
            "print(\"\")\n",
            "print(\"  (B) Unseen (synthetic) embeddings:\")\n",
            "print(f\"      Positive sim: {sim_pos_synth.mean():.4f} ± {sim_pos_synth.std():.4f}\")\n",
            "print(f\"      Negative sim: {sim_neg_synth.mean():.4f} ± {sim_neg_synth.std():.4f}\")\n",
            "print(f\"      Margin:       {sim_pos_synth.mean() - sim_neg_synth.mean():.4f}\")\n",
            "print(\"\")\n",
            "print(\"  (C) Top-1 Prototype Retrieval Accuracy:\")\n",
            "print(f\"      Seen (real):       {acc_seen_retrieval:.4f} ({acc_seen_retrieval*100:.2f}%)\")\n",
            "print(f\"      Unseen (synth):    {acc_synth_retrieval:.4f} ({acc_synth_retrieval*100:.2f}%)\")\n",
            "\n",
            "print(\"\\n### CHECK 2: CLASS-CONDITIONAL DIVERSITY ###\")\n",
            "print(\"\")\n",
            "print(f\"  Avg within-class cosine distance: {avg_dists.mean():.4f} ± {avg_dists.std():.4f}\")\n",
            "print(f\"  Avg per-dimension variance:       {per_dim_vars.mean():.6f} ± {per_dim_vars.std():.6f}\")\n",
            "print(f\"  Min class diversity (distance):   {avg_dists.min():.4f}\")\n",
            "print(f\"  Max class diversity (distance):   {avg_dists.max():.4f}\")\n",
            "\n",
            "print(\"\\n### FIGURES SAVED ###\")\n",
            "print(\"  - figures/prototype_alignment_seen.png\")\n",
            "print(\"  - figures/prototype_alignment_unseen_synth.png\")\n",
            "print(\"  - figures/synth_diversity_by_class.png\")\n",
            "\n",
            "# Interpretation\n",
            "print(\"\\n\" + \"=\"*70)\n",
            "print(\"INTERPRETATION\")\n",
            "print(\"=\"*70)\n",
            "margin_seen = sim_pos_seen.mean() - sim_neg_seen.mean()\n",
            "margin_synth = sim_pos_synth.mean() - sim_neg_synth.mean()\n",
            "\n",
            "if margin_seen > 0.1:\n",
            "    print(\"✓ CLIP alignment (seen): Good margin between positive and negative.\")\n",
            "else:\n",
            "    print(\"⚠ CLIP alignment (seen): Low margin — encoder may need more training.\")\n",
            "\n",
            "if margin_synth > 0.05:\n",
            "    print(\"✓ GAN conditioning (synth): Synthetic embeddings align with prototypes.\")\n",
            "else:\n",
            "    print(\"⚠ GAN conditioning (synth): Weak alignment — check conditioning quality.\")\n",
            "\n",
            "if avg_dists.mean() > 0.05:\n",
            "    print(\"✓ Diversity check: No severe mode collapse detected.\")\n",
            "else:\n",
            "    print(\"⚠ Diversity check: Low diversity — possible mode collapse.\")\n",
            "\n",
            "print(\"\\n\" + \"=\"*70)\n",
            "print(\"NEXT: Await confirmation before training GZSL classifier [A+B]\")\n",
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
