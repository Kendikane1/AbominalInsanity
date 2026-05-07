#!/usr/bin/env python3
"""
Build all experiment notebooks from canonical sources.
Run once from project root, then archived.

Produces:
  experiments/00_text_alignment/notebook.ipynb   — COMP2261 v2 cleaned up
  experiments/01_wgan_diagnostics/notebook.ipynb — main_pipeline + archive cells 67-70
  experiments/02_eta_sweep/notebook.ipynb        — main_pipeline + alias + archive 71-72,74-76
  experiments/03_variance_regularisation/notebook.ipynb — main_pipeline + alias + archive 77-83
"""

import json
import copy
import shutil
from pathlib import Path

ROOT = Path(__file__).parent

# ── Source paths ──────────────────────────────────────────────────────────────
MAIN        = ROOT / "main_pipeline.ipynb"
ARCHIVE     = ROOT / "archive/notebooks/GZSL_EEG_Pipeline_v2.ipynb"
COMP_V2     = ROOT / "COMP2261_ArizMLCW_with_baseline.ipynb"          # 131 cells — has Phase E
COMP_V1     = ROOT / "COMP2261_ArizMLCW_with_baseline (1).ipynb"      # 125 cells — has Phase F
GZSL_V1     = ROOT / "GZSL_EEG_Pipeline_v2 (1).ipynb"
GZSL_V2     = ROOT / "GZSL_EEG_Pipeline_v2 (2).ipynb"
GZSL_V3     = ROOT / "GZSL_EEG_Pipeline_v2 (3).ipynb"
GZSL_V4     = ROOT / "GZSL_EEG_Pipeline_v2 (4).ipynb"
GZSL_ROOT   = ROOT / "GZSL_EEG_Pipeline_v2.ipynb"

LEGACY_NOTEBOOKS = [COMP_V2, COMP_V1, GZSL_V1, GZSL_V2, GZSL_V3, GZSL_V4, GZSL_ROOT]


# ── Notebook I/O helpers ──────────────────────────────────────────────────────

def load_nb(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def make_code_cell(source: str) -> dict:
    lines = source.split("\n")
    src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


def make_markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}


def clean_cells(cells: list) -> list:
    """Clear outputs and execution counts; deep-copy each cell."""
    result = []
    for c in cells:
        c = copy.deepcopy(c)
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None
        result.append(c)
    return result


def write_nb(dst: Path, cells: list, template_nb: dict):
    nb = {
        "nbformat": template_nb["nbformat"],
        "nbformat_minor": template_nb.get("nbformat_minor", 4),
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": clean_cells(cells),
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"  Written {dst.relative_to(ROOT)}  ({len(cells)} cells)")


# ── Shared WandB auth cells ───────────────────────────────────────────────────

WANDB_MD = """\
---
## WandB Experiment Tracking

Authenticate once per Colab session. API key in `.env` locally; in Colab use interactive prompt or Colab Secrets.\
"""

WANDB_CODE = """\
# === WandB Authentication ===
!pip install -q wandb
import wandb, os, sys

REPO_ROOT = '/content/drive/MyDrive/Gen_modelling_project'
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

WANDB_PROJECT = "gzsl-eeg-bravl"
WANDB_ENTITY  = ""  # fill in your W&B username once

wandb.login()
print(f"WandB ready. Project: {WANDB_PROJECT}")
"""

wandb_md_cell   = make_markdown_cell(WANDB_MD)
wandb_code_cell = make_code_cell(WANDB_CODE)


# ── Alias cell for experiments 02 and 03 ────────────────────────────────────
# Archive experiment cells reference OPT_ variables from the per-cell retrain.
# In experiment notebooks the main pipeline already uses optimal config, so we alias.

ALIAS_CODE = """\
# === Experiment aliases: main pipeline → OPT_ names used by archived cells ===
# The archived experiment cells were written against a re-trained "OPT_" set of
# variables. This notebook already uses the optimal config throughout, so we
# create aliases rather than re-running the full training.
OPT_ENCODER_CONFIG = ENCODER_CONFIG
OPT_WGAN_CONFIG    = WGAN_CONFIG

generator_opt     = generator
critic_opt        = critic
brain_encoder_opt = brain_encoder
image_proj_opt    = image_projector

E_train_opt  = E_train_seen
E_test_opt   = E_test_seen
E_unseen_opt = E_unseen

y_train_opt  = y_train_seen
y_test_opt   = y_test_seen
y_unseen_opt = y_unseen

S_seen_opt         = S_seen_prototypes
S_unseen_opt       = S_unseen_prototypes
seen_classes_opt   = seen_classes
unseen_classes_opt = unseen_classes
S_seen_arr_opt     = S_seen_array
S_unseen_arr_opt   = S_unseen_array

# median_seen_per_class is defined in the sample-balancing cell above
median_per_class_opt = median_seen_per_class

seen_labels_set_opt   = seen_labels_set
unseen_labels_set_opt = unseen_labels_set

d = ENCODER_CONFIG['embed_dim']
print(f"Aliases set. d={d}, median_per_class_opt={median_per_class_opt}")
"""

alias_cell = make_code_cell(ALIAS_CODE)


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 00 — Text Alignment (Brain-TEXT CLIP, original coursework era)
# Source: COMP2261_ArizMLCW_with_baseline.ipynb (v2, 131 cells — has Phase E)
#         + Phase F cells from (1).ipynb (v1, 125 cells — has Phase F)
# ─────────────────────────────────────────────────────────────────────────────

def build_exp00():
    print("\n=== Building Experiment 00: Text Alignment ===")

    comp_v2_nb = load_nb(COMP_V2)
    comp_v1_nb = load_nb(COMP_V1)
    comp_v2_cells = comp_v2_nb["cells"]
    comp_v1_cells = comp_v1_nb["cells"]

    # v2 cells 0-113: shared base (baseline + CLIP + WGAN + ablation + Phase D)
    # v2 cells 114-125: Phase E — CLIP encoder optimisation (2-stage parametric sweep)
    # v1 cells 114-124: Phase F — Augmentation / SupCon / architecture sweep (dead end)
    # Skip empty/minimal cells at end of both notebooks

    base_cells = comp_v2_cells[:114]      # cells 0-113 (identical in v1 and v2)
    phase_e_cells = comp_v2_cells[114:126] # cells 114-125 (Phase E, 12 cells)
    phase_f_cells = comp_v1_cells[114:125] # cells 114-124 (Phase F, 11 cells)

    # Insert a divider markdown before Phase F
    phase_f_header = make_markdown_cell(
        "---\n"
        "# Phase F: Augmentation, Contrastive Learning & Architecture Sweep\n\n"
        "**Status**: Complete — Dead End\n\n"
        "Phase E revealed the encoder top-1 ceiling (~3%). Phase F explored whether\n"
        "data augmentation (channel_drop, temporal_shift), SupCon loss, or structured\n"
        "encoders (EEGNet, ShallowConvNet) could push past this ceiling.\n\n"
        "**Conclusion**: all approaches failed; the 561-D EEG→text alignment with MLP\n"
        "on 17ch×33t resolves to ~3% top-1 regardless of training regime.\n"
        "Root cause: insufficient class-discriminative information in the EEG features\n"
        "at this resolution. Led directly to the paradigm shift to EEG→image alignment."
    )

    all_cells = list(base_cells) + list(phase_e_cells) + [phase_f_header] + list(phase_f_cells)

    # Insert WandB auth cells after cell 4 (first code cell: pip installs)
    final_cells = all_cells[:5] + [wandb_md_cell, wandb_code_cell] + all_cells[5:]

    # Create experiment folder and README
    exp00_dir = ROOT / "experiments/00_text_alignment"
    exp00_dir.mkdir(parents=True, exist_ok=True)
    (exp00_dir / "results").mkdir(exist_ok=True)
    (exp00_dir / "figures").mkdir(exist_ok=True)
    (exp00_dir / "context").mkdir(exist_ok=True)

    readme = """\
# Experiment 00: Text Alignment Baseline (Brain-TEXT CLIP)

**Status**: Complete — Superseded by Image Alignment
**Era**: Original coursework (COMP2261) — pre-paradigm-shift
**Modality**: EEG → Text (CLIP 512-D sentence embeddings)

## Summary

The original GZSL pipeline using Brain-TEXT CLIP alignment. A contrastive encoder
maps 561-D EEG features to a shared space with 512-D text embeddings (CLIP). A
conditional WGAN-GP then synthesises unseen-class embeddings from text prototypes.

This experiment documents the full exploration arc of the text-alignment paradigm,
including ablation studies, inverse bias diagnosis, Phase E encoder optimisation,
and Phase F architecture sweep.

## Architecture (Text Alignment Era)

- **BrainEncoder**: 561 → 1024 → 512 → d, LayerNorm + L2-norm
- **TextProjector**: 512 (CLIP) → 512 → d, LayerNorm + L2-norm
- **cWGAN-GP**: same topology as image-alignment pipeline

## Phases in this notebook

| Phase | Section | Key Result |
|-------|---------|-----------|
| Baseline [A] | Cells 17–22 | LogReg on raw EEG: ~1.5% seen accuracy |
| CLIP encoder | Cells 28–37 | Initial config: embed_dim=64, τ=0.07, 20 epochs |
| WGAN-GP | Cells 38–48 | cWGAN-GP on text prototypes |
| Ablation [A-D] | Cells 75–101 | Label collision fix; routing catastrophe isolated |
| Phase 1 | Cells 102–107 | Sample balancing: downsampled unseen → routing fixed |
| Phase D | Cells 108–113 | Upstream diagnostics: encoder is bottleneck (~1.9% top-1) |
| Phase E | Cells 114–125 | 2-stage sweep: τ=0.15, 50ep, lr=1e-3 → **H=0.70%** (+67%) |
| Phase F | Cells 126–136 | Augmentation + SupCon + EEGNet/ShallowConvNet → **dead end** |

## Key Findings

- **Best H-mean**: 0.70% (Phase E optimised config) — confirmed EEG→text is the ceiling
- **Phase E**: embed_dim=128, LayerNorm, lr=1e-3, cosine warmup, τ=0.15, 50 epochs
- **Phase F conclusion**: ~3% top-1 is the hard ceiling for EEG→text with any MLP variant
  on 17ch×33t data. All augmentation/loss/architecture experiments failed to improve.
- **Root cause of ceiling**: insufficient class-discriminative information in EEG features
  at this temporal resolution; text embeddings also have high intra-class cosine similarity
  (avg ρ=0.668), making separation hard.

## Why superseded

Switched to EEG→image (CORnet-S 1000-D PCA) which produced:
- H=4.53% vs 0.70% (6.5× improvement)
- AccU: 5.73% vs 0.32% (18× improvement)
- Contrastive loss converged for the first time (0.93 vs ~4.2 for text)
- Analysis: `context/research_history/CORnet_S_paradigm_analysis.md`

## Reference

- Archive notebooks: `archive/notebooks/COMP2261_ArizMLCW_with_baseline.ipynb` (v2/Phase E)
  and `archive/notebooks/COMP2261_ArizMLCW_with_baseline (1).ipynb` (v1/Phase F standalone)
"""
    (exp00_dir / "README.md").write_text(readme.strip() + "\n")
    print(f"  Written experiments/00_text_alignment/README.md")

    dst = exp00_dir / "notebook.ipynb"
    write_nb(dst, final_cells, comp_v2_nb)


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 01 — WGAN-GP Synthesis Diagnostics
# Source: main_pipeline.ipynb (69 cells) + archive cells 67–70
# ─────────────────────────────────────────────────────────────────────────────

def build_exp01():
    print("\n=== Building Experiment 01: WGAN Diagnostics ===")

    main_nb   = load_nb(MAIN)
    archive_nb = load_nb(ARCHIVE)
    main_cells  = main_nb["cells"]
    arc_cells   = archive_nb["cells"]

    # Cells 67-70 from archive: markdown header + 3 diagnostic code cells
    diag_cells = [copy.deepcopy(c) for c in arc_cells[67:71]]

    final_cells = list(main_cells) + diag_cells

    dst = ROOT / "experiments/01_wgan_diagnostics/notebook.ipynb"
    write_nb(dst, final_cells, main_nb)


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 02 — Eta Sweep (Noise-Augmented Prototype Conditioning)
# Source: main_pipeline.ipynb + alias + archive cells 71–72, skip 73, 74–76
# ─────────────────────────────────────────────────────────────────────────────

def build_exp02():
    print("\n=== Building Experiment 02: Eta Sweep ===")

    main_nb   = load_nb(MAIN)
    archive_nb = load_nb(ARCHIVE)
    main_cells  = main_nb["cells"]
    arc_cells   = archive_nb["cells"]

    # arc_cells[71]: markdown section header
    # arc_cells[72]: utility functions + ETA_VALUES + OPT_ configs
    # arc_cells[73]: RETRAIN — SKIP (main pipeline already uses optimal config)
    # arc_cells[74]: eta sweep run loop
    # arc_cells[75]: analysis + figures
    # arc_cells[76]: cache best-eta results

    exp_cells = (
        [copy.deepcopy(c) for c in arc_cells[71:73]]  # header + functions
        + [alias_cell]                                  # variable aliases
        + [copy.deepcopy(c) for c in arc_cells[74:77]] # run + analysis + cache
    )

    final_cells = list(main_cells) + exp_cells

    dst = ROOT / "experiments/02_eta_sweep/notebook.ipynb"
    write_nb(dst, final_cells, main_nb)


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 03 — Variance Regularisation (L_var Training-Time Loss)
# Source: main_pipeline.ipynb + alias + archive cells 77–83
# ─────────────────────────────────────────────────────────────────────────────

def build_exp03():
    print("\n=== Building Experiment 03: Variance Regularisation ===")

    main_nb   = load_nb(MAIN)
    archive_nb = load_nb(ARCHIVE)
    main_cells  = main_nb["cells"]
    arc_cells   = archive_nb["cells"]

    # arc_cells[77]: markdown section header
    # arc_cells[78]: compute_L_var, train_wgan_with_lvar, calibrate_alpha
    # arc_cells[79]: var_target computation + 5 correctness checks
    # arc_cells[80]: alpha calibration (gradient-norm matching)
    # arc_cells[81]: alpha sweep (fresh WGAN-GP per alpha)
    # arc_cells[82]: analysis figures (semilog sweep + training dynamics)
    # arc_cells[83]: cache best-alpha results

    exp_cells = (
        [copy.deepcopy(arc_cells[77])]   # header
        + [alias_cell]                    # aliases
        + [copy.deepcopy(c) for c in arc_cells[78:84]]  # functions through cache
    )

    final_cells = list(main_cells) + exp_cells

    dst = ROOT / "experiments/03_variance_regularisation/notebook.ipynb"
    write_nb(dst, final_cells, main_nb)


# ─────────────────────────────────────────────────────────────────────────────
# ARCHIVE legacy notebooks
# ─────────────────────────────────────────────────────────────────────────────

def archive_legacy():
    print("\n=== Archiving legacy notebooks ===")
    dst_dir = ROOT / "archive/notebooks/legacy"
    dst_dir.mkdir(parents=True, exist_ok=True)

    for src in LEGACY_NOTEBOOKS:
        if src.exists():
            dst = dst_dir / src.name
            shutil.move(str(src), str(dst))
            print(f"  mv  {src.name}  →  archive/notebooks/legacy/")
        else:
            print(f"  SKIP (not found): {src.name}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Building experiment notebooks...\n")

    build_exp00()
    build_exp01()
    build_exp02()
    build_exp03()
    archive_legacy()

    # Self-archive
    print("\n=== Self-archiving build script ===")
    self_path = ROOT / "build_experiment_notebooks.py"
    dst = ROOT / "archive/misc/build_experiment_notebooks.py"
    shutil.move(str(self_path), str(dst))
    print(f"  mv  build_experiment_notebooks.py  →  archive/misc/")

    print("\n=== Done ===")
    print("Experiment notebooks written:")
    for p in sorted((ROOT / "experiments").rglob("notebook.ipynb")):
        nb = json.loads(p.read_text())
        print(f"  {p.relative_to(ROOT)}  ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
