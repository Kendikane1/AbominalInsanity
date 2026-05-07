#!/usr/bin/env python3
"""
Assemble main_pipeline.ipynb from the archived v2 notebook.
Reads archive/notebooks/GZSL_EEG_Pipeline_v2.ipynb, bakes in optimal config
(embed_dim=64, tau=0.05, epochs=75, lr=2e-3), adds WandB auth cell, and writes
main_pipeline.ipynb at project root.

Run once from project root after reorganise.py, then this script is archived too.
"""
import json
import copy
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "archive" / "notebooks" / "GZSL_EEG_Pipeline_v2.ipynb"
DST = ROOT / "main_pipeline.ipynb"


def make_code_cell(lines):
    if isinstance(lines, str):
        raw = lines.split("\n")
        source = [l + "\n" for l in raw[:-1]] + ([raw[-1]] if raw[-1] else [])
    else:
        source = lines
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def make_markdown_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [text] if isinstance(text, str) else text,
    }


# ─── Replacement for cell 29: optimal encoder config ───────────────────────

OPTIMAL_ENCODER_CONFIG = """\
# =============================================================================
# ENCODER CONFIGURATION — OPTIMAL CONFIG
# =============================================================================
# Validated via 25-config sweep + dim sweep. Do not change without re-running
# the full pipeline evaluation — proxy metrics (encoder top-k) do not predict
# downstream H-mean.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import random
import math

ENCODER_CONFIG = {
    'embed_dim': 64,               # Sweep-optimal: lower d → easier WGAN-GP generation
    'image_input_dim': 1000,       # CORnet-S full PCA (no truncation)
    'tau': 0.05,                   # Sweep-optimal: hard negatives work with orthogonal image prototypes
    'epochs': 75,                  # Sweep-optimal: overfits beyond ~75
    'batch_size': 256,
    'lr': 2e-3,                    # Sweep-optimal
    'weight_decay': 1e-4,
    'dropout': 0.1,
    'schedule': 'cosine_warmup',
    'warmup_ratio': 0.1,
    'seed': 42,
    'alignment_target': 'image (CORnet-S)',
}

SEED = ENCODER_CONFIG['seed']
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 60)
print("ENCODER CONFIGURATION (Optimal — Image Alignment / CORnet-S)")
print("=" * 60)
for k, v in ENCODER_CONFIG.items():
    print(f"  {k}: {v}")
print(f"  device: {device}")
print("=" * 60)
"""

# ─── Replacement for cell 39: optimal WGAN config ──────────────────────────

OPTIMAL_WGAN_CONFIG = """\
# =============================================================================
# cWGAN-GP CONFIGURATION — OPTIMAL CONFIG
# =============================================================================

WGAN_CONFIG = {
    'z_dim': 100,
    'embed_dim': 64,           # Must match ENCODER_CONFIG['embed_dim']
    'lr': 1e-4,
    'betas': (0.0, 0.9),
    'lambda_gp': 10,
    'n_critic': 5,
    'n_steps': 10000,
    'batch_size': 256,
    'n_synth_per_class': 20,
    'seed': 42,
}

print("=" * 60)
print("cWGAN-GP CONFIGURATION (Optimal)")
print("=" * 60)
for k, v in WGAN_CONFIG.items():
    print(f"  {k}: {v}")
print("=" * 60)
"""

# ─── WandB auth cells to insert after cell 4 (installs/imports) ────────────

WANDB_MARKDOWN = """\
---
## WandB Experiment Tracking

Authenticate once per Colab session. API key is in `.env` locally; in Colab use the interactive prompt or store in Colab Secrets as `WANDB_API_KEY`.\
"""

WANDB_CODE = """\
# === WandB Authentication ===
!pip install -q wandb
import wandb, os, sys

# Add shared/ to Python path so wandb_utils is importable from experiment notebooks
# Adjust this path to match your Colab Drive mount location
REPO_ROOT = '/content/drive/MyDrive/Gen_modelling_project'
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

WANDB_PROJECT = "gzsl-eeg-bravl"
WANDB_ENTITY  = ""  # fill in your W&B username once

# Authenticate (paste API key when prompted, or set WANDB_API_KEY env var)
wandb.login()
print(f"WandB ready. Project: {WANDB_PROJECT}")
"""


def main():
    if not SRC.exists():
        raise FileNotFoundError(
            f"Source notebook not found: {SRC}\n"
            "Run reorganise.py first to move the v2 notebook to archive/notebooks/."
        )

    print(f"Loading source: {SRC}")
    with open(SRC) as f:
        nb = json.load(f)

    src_cells = nb["cells"]
    print(f"Source notebook: {len(src_cells)} cells")

    # ── Select cells 0–66 (setup through eval harness, no experiments) ──
    selected = [copy.deepcopy(c) for c in src_cells[:67]]

    # ── Patch cell 29: encoder config → optimal values ──
    selected[29] = make_code_cell(OPTIMAL_ENCODER_CONFIG)
    print("  Patched cell 29: optimal encoder config (d=64, tau=0.05, epochs=75, lr=2e-3)")

    # ── Patch cell 39: WGAN config → optimal values ──
    selected[39] = make_code_cell(OPTIMAL_WGAN_CONFIG)
    print("  Patched cell 39: optimal WGAN config (embed_dim=64)")

    # ── Clear all outputs (clean notebook) ──
    for c in selected:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None

    # ── Insert WandB auth cells after cell 4 (first code cell: installs) ──
    wandb_md = make_markdown_cell(WANDB_MARKDOWN)
    wandb_code = make_code_cell(WANDB_CODE)
    selected = selected[:5] + [wandb_md, wandb_code] + selected[5:]
    print("  Inserted WandB auth cells after cell 4 (installs/imports)")

    # ── Assemble notebook ──
    new_nb = {
        "nbformat": nb["nbformat"],
        "nbformat_minor": nb.get("nbformat_minor", 4),
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "cells": selected,
    }

    with open(DST, "w") as f:
        json.dump(new_nb, f, indent=1)

    print(f"\nWritten: {DST}")
    print(f"  Total cells: {len(selected)}")
    print(f"  Optimal config baked in throughout (d=64, tau=0.05, epochs=75, lr=2e-3)")
    print(f"  WandB auth cell included at top")
    print(f"\nNext: copy to experiments/04_film_conditioning/notebook.ipynb")
    print(f"  cp main_pipeline.ipynb experiments/04_film_conditioning/notebook.ipynb")


if __name__ == "__main__":
    main()
