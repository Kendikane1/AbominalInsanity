# Session Debrief: Notebook Cleanup v2 + Visual Investigation Setup

**Date**: 2026-03-03
**Session**: Notebook cleanup, Phase F anomaly diagnosis, visual feature investigation prep

---

## 1. Task Summary

This session had three objectives:
1. **Tier 3 cleanup**: Create a clean notebook (`GZSL_EEG_Pipeline_v2.ipynb`) by removing experimental debt (ablation, diagnostics, Phase 1/D/F experiments), fixing the label collision at its source, and debugging the Phase F control anomaly.
2. **Visual feature investigation**: Add cells to explore the dataset's CORnet-S visual features and assess image alignment options (Tier 1 intervention).
3. **Session kickstart**: Onboard to the project after context reset, absorb all Phase F results and analyses.

---

## 2. What Changed

### Files Created
| File | Description |
|------|-------------|
| `GZSL_EEG_Pipeline_v2.ipynb` | **New active notebook** — 76 cells (down from 125). Clean pipeline + visual investigation section. |
| `helper_files/create_clean_notebook.py` | Script that reads old notebook JSON and constructs the clean v2. Applies modifications, creates new cells. |
| `context/notebook-cleanup-v2-debrief.md` | This debrief file. |

### Files Modified
| File | Description |
|------|-------------|
| `CLAUDE.md` | Updated: key files (v2 is now active notebook), hard constraints (label fix at source), cell structure (76 cells), completed phases (E+F), next phase (image alignment). |
| `MEMORY.md` | Updated: Phase F marked COMPLETE/DEAD END, notebook cleanup v2 complete, added research direction on structure-preserving WGAN-GP, added dataset visual feature notes. |

### Files Unchanged
| File | Description |
|------|-------------|
| `COMP2261_ArizMLCW_with_baseline.ipynb` | Legacy notebook (125 cells) — archived, untouched. |

---

## 3. Key Decisions

### D1: Label Collision Fixed at Source (Cell 7)
- **Problem**: Unseen labels [1..200] collided with seen labels [1..1654]. The old fix (cells 81-88) applied a post-hoc offset, which mutated global label tensors and caused the Phase F control anomaly.
- **Solution**: Insert `label_unseen += max(label_seen)` in cell 7 (data loading), before torch conversion. Unseen labels now [1655..1854] from the start. All downstream cells (prototypes, WGAN synthesis, GZSL classifier) automatically use correct labels.
- **Side fix**: `np.bincount` in cell 14 replaced with `np.unique(..., return_counts=True)` since bincount creates sparse arrays with offset labels.

### D2: Phase F Control Anomaly — Root Cause
- **Diagnosis**: Cells 82-83 in the old notebook mutated `label_train_seen` and `label_test_seen` in the global namespace. When Phase F's `evaluate_encoder()` (cell 115) used these corrupted labels, LogReg trained on wrong label space → 0.03% (random chance) instead of 2.87%.
- **Resolution**: By fixing labels at source in v2, no downstream mutation occurs. The anomaly is eliminated by design.

### D3: What Was Removed (49 cells)
- Cells 49-61: Quantitative diagnostics (prototype alignment, diversity checks) — diagnostic only, not pipeline
- Cells 75-96: Ablation study + label collision fix — user deems ablation unnecessary, label fix moved to source
- Cells 102-107: Phase 1 sample balance experiments — completed, archived
- Cells 108-113: Upstream diagnostics — completed, archived
- Cells 114-124: Phase F experiments — dead end, archived

### D4: User Wants to Keep WGAN-GP
- The user explicitly stated they do NOT want to pivot away from WGAN-GP to UVDS/linear methods.
- Instead, they want to mathematically derive structure-preserving loss function modifications that incorporate Long et al.'s insights INTO the GAN framework.
- This is a pen-and-paper mathematical research exercise, scheduled for AFTER the visual investigation results are analysed.
- Potential novelty: the ZSL literature abandoned GANs — making them work with structure preservation is an unexplored direction.

---

## 4. Current State

### Active Notebook: `GZSL_EEG_Pipeline_v2.ipynb` (76 cells)

```
Cells  0- 6: Setup (installs, descriptions)
Cells  7- 9: Data Loading (label collision fixed at source)
Cells 10-16: Data Exploration (structure summary, class distribution, EEG norms)
Cells 17-22: Baseline [A] — LogReg on raw EEG
Cells 23-27: GZSL Baseline Evaluation
Cells 28-37: CLIP Encoder (config, models, training, embeddings, prototypes, caching)
Cells 38-48: cWGAN-GP (config, models, training, synthesis, caching)
Cells 49-61: GZSL Classifier [A+B] (train, evaluate, comparison, summary)
Cells 62-66: Evaluation Harness (evaluate_gzsl, diagnostics, phase comparison)
Cells 67-75: Visual Feature Investigation (NEW — CORnet-S inspection, PCA spectrum,
             cross-modal comparison, CLIP availability check, dimensionality analysis)
```

### Pipeline Performance (Phase E best, to be re-verified on v2)
- Encoder top-1: 2.87% (1654-way seen-only)
- GZSL: AccS=2.96%, AccU=0.40%, H-mean=0.70%
- Routing rate: 11.31%

### Visual Investigation Results
- **The user has run the visual investigation cells (67-75) on Colab and compiled results into an .md file.**
- **The next Claude session should read this results file** — it will contain CORnet-S dimensionality, PCA variance spectrum, visual vs text inter-class separation, EEG-image vs EEG-text correlation, and CLIP image feature availability.

---

## 5. Open Items

### O1: Visual Investigation Results (READY — compiled by user)
The user ran cells 67-75 on Colab. Results are compiled in an md file for the next session to read and analyse. This is the **immediate next action**.

### O2: Image Alignment Paradigm Shift (Tier 1)
Based on the visual investigation results, the next session should design the switch from EEG→text to EEG→image alignment. Key questions:
- Are CORnet-S features sufficient, or do we need CLIP image embeddings?
- What dimensionality should we use (full PCA vs truncated)?
- How does the ImageProjector architecture change (input dim)?
- Can we use text prototypes for unseen classes at test time if we train with image alignment? (Yes — CLIP's shared space enables this)

### O3: Structure-Preserving WGAN-GP (Tier 2 — later)
Mathematical research into modifying the WGAN-GP loss function:
- Variance diffusion regularisation as a generator loss term
- Dual-graph Laplacian structure preservation
- Pen-and-paper derivation of gradient dynamics
- This comes AFTER image alignment is working

### O4: Full Pipeline Re-verification on v2
The v2 notebook hasn't been run end-to-end yet (the user ran cells 67-75 for investigation, but the full pipeline cells 7-66 should also be verified to produce consistent results with the label fix).

---

## 6. Next Steps (for the next Claude session)

1. **Read the visual investigation results** — the user compiled Colab outputs into an md file. Read it, analyse the findings.

2. **Analyse CORnet-S vs text comparison** — which modality provides better class separation? What's the effective rank? Is EEG more correlated with visual or text features?

3. **Design the image alignment switch** — modify the CLIP pipeline to align EEG→image instead of EEG→text. This means:
   - New `ImageProjector` (input dim = CORnet-S full dim or CLIP 512-D)
   - Modify training loop to use `image_seen` instead of `text_seen`
   - For ZSL: still use text/image prototypes for unseen classes at test time
   - Inject new cells via helper script

4. **Keep WGAN-GP** — do NOT replace with UVDS. The user wants to research mathematical improvements to the GAN loss function later.

5. **Run `/session-kickstart`** to onboard — read `CLAUDE.md`, `MEMORY.md`, this debrief, and the visual investigation results file.

---

## Key Context Files for Next Session

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Project spec, architecture, constraints, current state |
| `MEMORY.md` (auto-loaded) | Persistent memory across sessions |
| `context/notebook-cleanup-v2-debrief.md` | This debrief |
| `context/Claude_phaseF_analysis.md` | Phase F analysis + literature review + path forward |
| `context/phasef_analysis.md` | Agent's Phase F analysis |
| `context/phasef_results.md` | Raw Phase F experiment outputs |
| The user's visual investigation results .md file | **READ THIS FIRST** — contains Colab outputs from cells 67-75 |
