#!/usr/bin/env python3
"""
Script to add Phase 1 (Sample Count Balancing) cells to the notebook.

Run from project root:
    python helper_files/add_phase1_sample_balance.py
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
            "# Phase 1: Sample Count Balancing\n",
            "\n",
            "**Target:** Mechanism M2 (sample imbalance: ~20/unseen vs ~8/seen)\n",
            "**Strategy:** Downsample synthetic unseen to match seen per-class count, then test upsampling seen as secondary experiment.\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# PHASE 1 CONFIGURATION\n",
            "# =============================================================================\n",
            "\n",
            "PHASE1_CONFIG = {\n",
            "    'seed': 42,\n",
            "    'strategy': 'downsample_unseen',  # Primary: reduce unseen to match seen\n",
            "    'target_per_class': None,         # None = auto-detect from seen data\n",
            "}\n",
            "\n",
            "np.random.seed(PHASE1_CONFIG['seed'])\n",
            "\n",
            "print(\"Phase 1 Configuration:\")\n",
            "for k, v in PHASE1_CONFIG.items():\n",
            "    print(f\"  {k}: {v}\")\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# PHASE 1A: DOWNSAMPLE UNSEEN TO MATCH SEEN\n",
            "# =============================================================================\n",
            "\n",
            "# Compute per-class sample counts for seen\n",
            "seen_classes_train, seen_counts = np.unique(y_train_seen, return_counts=True)\n",
            "median_seen_per_class = int(np.median(seen_counts))\n",
            "mean_seen_per_class = seen_counts.mean()\n",
            "\n",
            "print(f\"Seen training: {len(seen_classes_train)} classes\")\n",
            "print(f\"  Per-class: mean={mean_seen_per_class:.1f}, median={median_seen_per_class}, \"\n",
            "      f\"min={seen_counts.min()}, max={seen_counts.max()}\")\n",
            "\n",
            "# Compute per-class sample counts for unseen synth\n",
            "unseen_classes_synth, unseen_counts = np.unique(y_synth_unseen_remapped, return_counts=True)\n",
            "print(f\"\\nUnseen synthetic: {len(unseen_classes_synth)} classes\")\n",
            "print(f\"  Per-class: mean={unseen_counts.mean():.1f}, median={int(np.median(unseen_counts))}, \"\n",
            "      f\"min={unseen_counts.min()}, max={unseen_counts.max()}\")\n",
            "\n",
            "# Target: match median seen per-class count\n",
            "target_n = PHASE1_CONFIG['target_per_class'] or median_seen_per_class\n",
            "print(f\"\\nTarget per-class count: {target_n}\")\n",
            "\n",
            "# Downsample unseen\n",
            "rng = np.random.RandomState(PHASE1_CONFIG['seed'])\n",
            "ds_indices = []\n",
            "for c in unseen_classes_synth:\n",
            "    class_idx = np.where(y_synth_unseen_remapped == c)[0]\n",
            "    if len(class_idx) > target_n:\n",
            "        selected = rng.choice(class_idx, size=target_n, replace=False)\n",
            "    else:\n",
            "        selected = class_idx  # Keep all if already <= target\n",
            "    ds_indices.append(selected)\n",
            "ds_indices = np.concatenate(ds_indices)\n",
            "\n",
            "E_synth_ds = E_synth_unseen[ds_indices]\n",
            "y_synth_ds = y_synth_unseen_remapped[ds_indices]\n",
            "\n",
            "print(f\"\\nDownsampled unseen: {len(E_synth_ds)} total \"\n",
            "      f\"(was {len(E_synth_unseen)}, reduction: {1 - len(E_synth_ds)/len(E_synth_unseen):.1%})\")\n",
            "\n",
            "# Verify balance\n",
            "us_classes, us_counts = np.unique(y_synth_ds, return_counts=True)\n",
            "print(f\"  Per-class after downsample: mean={us_counts.mean():.1f}, all={target_n}? {np.all(us_counts == target_n)}\")\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# PHASE 1A: TRAIN LOGREG ON BALANCED DATA\n",
            "# =============================================================================\n",
            "\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "\n",
            "# Combine: real seen train + downsampled synthetic unseen\n",
            "E_train_p1 = np.vstack([E_train_seen, E_synth_ds])\n",
            "y_train_p1 = np.concatenate([y_train_seen, y_synth_ds])\n",
            "\n",
            "print(f\"Phase 1A training set: {E_train_p1.shape[0]} samples \"\n",
            "      f\"({len(E_train_seen)} seen + {len(E_synth_ds)} unseen)\")\n",
            "\n",
            "# Train LogReg (same hyperparameters as Method D for fair comparison)\n",
            "clf_p1a = LogisticRegression(\n",
            "    max_iter=5000,\n",
            "    solver='lbfgs',\n",
            "    multi_class='multinomial',\n",
            "    random_state=42,\n",
            "    C=1.0,\n",
            ")\n",
            "clf_p1a.fit(E_train_p1, y_train_p1)\n",
            "print(\"LogReg fitted.\")\n",
            "\n",
            "# Evaluate\n",
            "p1a_results = evaluate_gzsl(\n",
            "    clf_p1a, E_test_seen, y_test_seen, E_unseen, y_unseen_remapped,\n",
            "    seen_labels_set, unseen_labels_set, phase_name=\"Phase 1A (downsample unseen)\"\n",
            ")\n",
            "all_phase_results['P1A: Downsample Unseen'] = p1a_results\n",
            "\n",
            "# Diagnostics\n",
            "p1a_diag = diagnose_classifier(\n",
            "    clf_p1a, seen_labels_set, unseen_labels_set, phase_name=\"Phase 1A (downsample unseen)\"\n",
            ")\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# PHASE 1B: UPSAMPLE SEEN (SECONDARY EXPERIMENT)\n",
            "# =============================================================================\n",
            "# Generate additional seen-class embeddings using the EXISTING generator\n",
            "# with SEEN prototypes. This raises seen per-class count to ~20.\n",
            "# Note: these synthetic seen embeddings will have the same low-variance\n",
            "# property as synthetic unseen — this is diagnostically informative.\n",
            "\n",
            "import torch\n",
            "\n",
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
            "generator.eval()\n",
            "\n",
            "z_dim = 100  # from WGAN config\n",
            "target_total = 20  # target per-class\n",
            "\n",
            "rng = np.random.RandomState(42)\n",
            "synth_seen_list = []\n",
            "synth_seen_labels = []\n",
            "\n",
            "with torch.no_grad():\n",
            "    for i, c in enumerate(seen_classes_train):\n",
            "        # How many real samples does this class have?\n",
            "        n_real = seen_counts[i]\n",
            "        n_to_generate = max(0, target_total - n_real)\n",
            "\n",
            "        if n_to_generate > 0:\n",
            "            # Get prototype for this class\n",
            "            proto_idx = i  # assuming prototypes are ordered same as seen_classes_train\n",
            "            s_c = torch.tensor(S_seen_array[proto_idx], dtype=torch.float32).unsqueeze(0).to(device)\n",
            "            s_c = s_c.expand(n_to_generate, -1)\n",
            "\n",
            "            z = torch.randn(n_to_generate, z_dim, device=device)\n",
            "            fake = generator(z, s_c).cpu().numpy()\n",
            "            synth_seen_list.append(fake)\n",
            "            synth_seen_labels.append(np.full(n_to_generate, c))\n",
            "\n",
            "if synth_seen_list:\n",
            "    E_synth_seen = np.vstack(synth_seen_list)\n",
            "    y_synth_seen = np.concatenate(synth_seen_labels)\n",
            "    print(f\"Generated {len(E_synth_seen)} synthetic seen embeddings \"\n",
            "          f\"across {len(synth_seen_list)} classes needing augmentation\")\n",
            "else:\n",
            "    E_synth_seen = np.empty((0, E_train_seen.shape[1]))\n",
            "    y_synth_seen = np.array([], dtype=y_train_seen.dtype)\n",
            "    print(\"No seen classes needed augmentation.\")\n",
            "\n",
            "# Combine: real seen + synth seen + full synthetic unseen (20/class)\n",
            "E_train_p1b = np.vstack([E_train_seen, E_synth_seen, E_synth_unseen])\n",
            "y_train_p1b = np.concatenate([y_train_seen, y_synth_seen, y_synth_unseen_remapped])\n",
            "\n",
            "print(f\"\\nPhase 1B training set: {E_train_p1b.shape[0]} samples\")\n",
            "print(f\"  Real seen: {len(E_train_seen)}, Synth seen: {len(E_synth_seen)}, \"\n",
            "      f\"Synth unseen: {len(E_synth_unseen)}\")\n",
            "\n",
            "# Train LogReg\n",
            "clf_p1b = LogisticRegression(\n",
            "    max_iter=5000,\n",
            "    solver='lbfgs',\n",
            "    multi_class='multinomial',\n",
            "    random_state=42,\n",
            "    C=1.0,\n",
            ")\n",
            "clf_p1b.fit(E_train_p1b, y_train_p1b)\n",
            "print(\"LogReg fitted.\")\n",
            "\n",
            "# Evaluate\n",
            "p1b_results = evaluate_gzsl(\n",
            "    clf_p1b, E_test_seen, y_test_seen, E_unseen, y_unseen_remapped,\n",
            "    seen_labels_set, unseen_labels_set, phase_name=\"Phase 1B (upsample seen)\"\n",
            ")\n",
            "all_phase_results['P1B: Upsample Seen'] = p1b_results\n",
            "\n",
            "# Diagnostics\n",
            "p1b_diag = diagnose_classifier(\n",
            "    clf_p1b, seen_labels_set, unseen_labels_set, phase_name=\"Phase 1B (upsample seen)\"\n",
            ")\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# PHASE 1 COMPARISON\n",
            "# =============================================================================\n",
            "\n",
            "compare_phases(all_phase_results)\n",
            "\n",
            "print(\"\\nInterpretation:\")\n",
            "print(\"- If P1A improves AccS but barely touches AccU → M2 contributes but isn't dominant\")\n",
            "print(\"- If P1B (upsample seen) does NOT help → the issue is variance, not count\")\n",
            "print(\"- If P1B matches P1A → variance mismatch is symmetric and M2 is the main driver\")\n",
            "print(\"- Routing rate is the key metric: how far does it move from 99.6%?\")\n"
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

    print(f"\\nSuccess! Updated: {OUTPUT_PATH}")
    print(f"  - Original cells: {original_count}")
    print(f"  - New cells added: {len(NEW_CELLS)}")
    print(f"  - Total cells: {len(notebook['cells'])}")
    print(f"\\nRefresh the notebook in Jupyter and run the new cells.")

if __name__ == "__main__":
    main()
