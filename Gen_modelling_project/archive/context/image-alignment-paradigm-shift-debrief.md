# Session Debrief: CORnet-S Image Alignment Paradigm Shift

**Date**: 2026-03-03 to 2026-03-04
**Session scope**: Sample balance fix, visual investigation analysis, paradigm shift design + implementation, first run analysis
**Result**: H-mean improved 7.8x (0.58% → 4.53%), AccU improved 18x (0.32% → 5.73%)

---

## 1. Task Summary

This session had four sequential objectives:

### Objective 1: Fix the v2 Pipeline Routing Catastrophe
The user noticed that the v2 notebook (created in the previous session's cleanup) had a **97.9% routing rate** — nearly all seen test samples were being predicted as unseen. The root cause: the previous cleanup agent archived the Phase 1 sample balance experiments (cells 102-107 of the old notebook) as "completed" but **forgot to integrate the actual fix** into the v2 pipeline.

The fix involved two changes to the v2 notebook:
- **Cell 51**: Added downsampling of synthetic unseen embeddings to match seen per-class median (~8/class)
- **Cell 52**: Added `class_weight='balanced'` to LogisticRegression

**Result**: Routing rate dropped from 97.9% to 10.8%. AccS recovered from 0.33% to 3.45%. Weight norm ratio from 1.76x to 1.05x. Pipeline verified correct.

### Objective 2: Deep Analysis of Visual Investigation Results
The user had compiled Colab outputs from the visual investigation cells (67-75) into `context/Visual_investigation_results.md`. This session provided a comprehensive mathematical analysis of:
- Inter-class cosine similarity: Visual -0.001 (near-orthogonal) vs Text 0.668 (tightly clustered)
- Effective rank at 95% variance: Visual 335-D vs Text 169-D vs EEG 116-D
- PCA variance spectrum: first 100 components = only 32.4% variance (severe information loss)
- Cross-modal correlations: both near-zero pre-projection (r ≈ 0)
- CLIP image availability: NOT in dataset, only CORnet-S

### Objective 3: Design and Implement the Paradigm Shift
The user directed: "go with CORnet-S first", "use full 1000-D features", "extreme detail and attention to logic and theory". Plan mode was used to design the switch from EEG-to-text to EEG-to-image (CORnet-S) alignment.

**Key decisions made by the user:**
- Use CORnet-S first (not CLIP image features) — it's already in the dataset
- Use full 1000-D PCA features (no truncation from 1000 to 100)
- Remove visual investigation cells (67-75) since investigation is complete
- Keep WGAN-GP (do not replace with UVDS/linear methods)

The plan was approved and implemented via a helper script (`helper_files/switch_to_image_alignment.py`) that modified 16 cells and deleted 9 cells, taking the notebook from 76 to 67 cells.

### Objective 4: Analyse First Run Results
The user ran the modified notebook on Colab and compiled results into `context/CORnet_S_paradigm.md`. A comprehensive analysis was performed comparing image alignment results against the text alignment baseline.

---

## 2. What Changed

### Files Created

| File | Description |
|------|-------------|
| `helper_files/switch_to_image_alignment.py` | Helper script that modifies the v2 notebook for image alignment. Applies 16 cell modifications (string replacements and full source replacements) and deletes 9 visual investigation cells. Includes backup, sanity checks (11 automated checks), and verification summary. |
| `context/CORnet_S_paradigm_analysis.md` | Comprehensive analysis report of the first image alignment run. Contains metric-by-metric breakdown, visual analysis of t-SNE and diagnostics figures, theoretical interpretation, SOTA comparison, and next steps recommendations. |
| `context/image-alignment-paradigm-shift-debrief.md` | This debrief file. |

### Files Modified

| File | Description |
|------|-------------|
| `GZSL_EEG_Pipeline_v2.ipynb` | **Active notebook** — 76 → 67 cells. PCA truncation removed (full 1000-D CORnet-S), TextProjector → ImageProjector, CLIP_CONFIG → ENCODER_CONFIG, text data prep → image data prep, prototypes from image embeddings, visual investigation cells deleted. Sample balance fix (cells 51-52) also applied earlier in session. |
| `plans/current-task.md` | Replaced Phase F task with image alignment paradigm shift documentation. Contains full context, architecture details, cell map, verification checklist, expected results, risks. Note: the changes were already implemented by the helper script, so this serves as a reference document, not an implementation handoff. |
| `CLAUDE.md` | Updated: architecture diagram (text → image), key files (67 cells), hyperparameters (ENCODER_CONFIG), cell structure, current phase (image alignment applied), expected outputs. |
| `MEMORY.md` | Updated: project status (image alignment applied + first run results), architecture facts (ImageProjector specs), key findings (paradigm shift validated). |

### Files Created by User (on Colab, then copied locally)

| File | Description |
|------|-------------|
| `context/CORnet_S_paradigm.md` | Raw Colab cell outputs from the image alignment run — encoder config, training logs, embedding shapes, WGAN-GP training, synthesis stats, GZSL classifier results, evaluation harness output, diagnostics. |
| `figures/Paradigm_shift_run/clip_embedding_tsne.png` | t-SNE of EEG embeddings + image prototypes (seen green stars, unseen red triangles dispersed across space) |
| `figures/Paradigm_shift_run/real_vs_synth_tsne.png` | t-SNE of real seen vs synthetic unseen embeddings (good overlap, no separation artifacts) |
| `figures/Paradigm_shift_run/pipeline_v2_diagnostics.png` | Weight norms and intercepts histogram (overlapping distributions, ratio 1.11x, healthy calibration) |

### Files Unchanged

| File | Description |
|------|-------------|
| `COMP2261_ArizMLCW_with_baseline.ipynb` | Legacy notebook (125 cells) — archived, untouched |
| `context/Visual_investigation_results.md` | Visual investigation outputs — archived, investigation complete |
| `context/Claude_phaseF_analysis.md` | Phase F analysis — archived, still relevant for context |
| `context/notebook-cleanup-v2-debrief.md` | Previous session debrief — archived |
| `GZSL_EEG_Pipeline_v2.ipynb.backup.*` | Backup created by helper script before modifications |

---

## 3. Key Decisions

### D1: Sample Balance Fix — "Belt and Suspenders" Approach
**Problem**: v2 notebook had 97.9% routing catastrophe because cleanup agent dropped Phase 1 sample balance fix.
**Decision**: User chose BOTH downsampling + class_weight='balanced' (not one or the other).
**Cell 51**: Downsamples synthetic unseen from 20/class to match seen per-class median (~8/class).
**Cell 52**: LogReg with `class_weight='balanced'`.
**Result**: Routing rate 97.9% → 10.8%, AccS 0.33% → 3.45%.

### D2: CORnet-S First, CLIP Image Later
**Decision**: User explicitly chose to use CORnet-S features (already in dataset) before considering external CLIP image embeddings.
**Rationale**: Establishes image alignment infrastructure, avoids external dependencies, tests the paradigm with available data.

### D3: Full 1000-D Features, No PCA Truncation
**Decision**: Remove the `image_seen[:,0:100]` truncation that was discarding 67.6% of variance.
**Rationale**: User wants "experiments on every ounce of data we have to achieve the best we possibly can."

### D4: ImageProjector Architecture — 2-Layer MLP with LayerNorm
**Decision**: 1000→512→128 with LayerNorm + Dropout + L2-norm.
**Rationale**:
- 2-layer (mirrors TextProjector depth, CORnet-S PCA features are clean)
- LayerNorm (variance concentrated in early PCA components — LN stabilises gradients)
- Dropout 0.1 (regularisation against 512K params with 13,232 samples)
- L2-norm output (unit sphere, matches BrainEncoder)
- ~579K parameters

### D5: Keep Same Hyperparameters Initially
**Decision**: ENCODER_CONFIG uses identical hyperparameters to Phase E text config (tau=0.15, lr=1e-3, epochs=50, etc.) except for image_input_dim=1000.
**Rationale**: Phase E config was validated to work well. Start with known-good config, then tune.

### D6: Prototype Variable Names Kept Identical
**Decision**: `S_seen_prototypes`, `S_unseen_prototypes`, `S_seen_array`, `S_unseen_array` — same names as text pipeline.
**Rationale**: This means the entire WGAN-GP (cells 39-46) and classifier (cells 50-61) pipelines inherit image prototypes automatically without any code changes. Minimises modification scope and risk.

### D7: Visual Investigation Cells Removed
**Decision**: Delete cells 67-75 (9 cells) since investigation is complete and results are archived.
**Rationale**: Clean notebook, investigation served its purpose, results preserved in `context/Visual_investigation_results.md`.

---

## 4. Current State

### Active Notebook: `GZSL_EEG_Pipeline_v2.ipynb` (67 cells)

```
Cells  0- 6: Setup (installs, descriptions)
Cells  7- 9: Data Loading (full 1000-D CORnet-S, label collision fix at source)
Cells 10-16: Data Exploration (structure summary, class distribution, EEG norms)
Cells 17-22: Baseline [A] — LogReg on raw EEG
Cells 23-27: GZSL Baseline Evaluation
Cells 28-37: Brain-Image Encoder (ENCODER_CONFIG, ImageProjector, training, embeddings, image prototypes, caching)
Cells 38-48: cWGAN-GP (config, models, data prep, training, synthesis, quality check, t-SNE, summary)
Cells 49-61: GZSL Classifier [A+B] (sample balance, LogReg, evaluation, comparison, summary)
Cells 62-66: Evaluation Harness (evaluate_gzsl, diagnostics, phase comparison)
```

### First Run Results (CORnet-S Image Alignment)

| Metric | Text Alignment (baseline) | Image Alignment | Change |
|--------|---------------------------|-----------------|--------|
| AccS | 3.45% | **3.75%** | +0.30% abs (+8.7% rel) |
| AccU | 0.32% | **5.73%** | +5.41% abs (**18x**) |
| H-mean | 0.58% | **4.53%** | +3.95% abs (**7.8x**) |
| Encoder loss | ~4.2 | **0.93** | 4.5x lower |
| Routing rate | 10.8% | 20.3% | +9.5% |
| Weight norm ratio | 1.05x | 1.11x | Slight increase (healthy) |
| WGAN G_loss final | N/A | 0.60 | Stable convergence |
| Synth quality (cosine to proto) | N/A | 0.5612 | Moderate conditioning |
| Synth per-dim variance | N/A | 0.0078 (= real) | No mode collapse |

### Key Architecture (Image Alignment)

- **BrainEncoder**: 561→1024→512→128, LayerNorm+ReLU+Dropout, L2-norm (~1.12M params) — **UNCHANGED**
- **ImageProjector**: 1000→512→128, LayerNorm+ReLU+Dropout, L2-norm (~579K params) — **NEW**
- **contrastive_loss**: Symmetric InfoNCE (renamed from clip_loss) — same function, new name
- **ENCODER_CONFIG**: Same as CLIP_CONFIG + `image_input_dim: 1000`, `alignment_target: 'image (CORnet-S)'`
- **WGAN-GP**: Unchanged architecture, unchanged config, but now receives image prototypes instead of text prototypes
- **Classifier**: Unchanged (sample balance + class_weight='balanced')

### Hard Constraints (Still Active)

- **Notebook-only implementation**: all code in `GZSL_EEG_Pipeline_v2.ipynb`
- **No data leakage**: unseen EEG is test-only
- **Label collision fix at source**: cell 7, unseen labels [1655..1854]
- **Sample balancing**: cell 51 downsampling + cell 52 class_weight='balanced'
- **Seeds and reproducibility**: SEED=42 everywhere
- **Keep WGAN-GP**: user does NOT want to replace with UVDS/linear methods
- **User wants mathematical depth**: full notation, rigorous theory, no hand-waving

---

## 5. Open Items

### O1: Encoder-Only Top-1 Not Measured
The GZSL AccS (3.75%) is measured on 1854-way classification. The encoder-only top-1 on 1654-way seen classification was not measured in this run. It was 2.87% under text alignment. Should be measured in the next run for fair comparison.

### O2: Routing Rate Slightly High (20.3%)
The optimal prior routing rate is ~10.8% (200/1854 classes). Our 20.3% means the classifier over-predicts unseen by ~2x. This is a trade-off for the AccU gain, but could be calibrated with temperature scaling or by reducing n_synth_per_class.

### O3: Hyperparameters Un-Tuned
All hyperparameters carried over from Phase E text alignment config. The optimal values for image alignment likely differ, particularly:
- **tau**: With better prototype separation, harder negatives (lower tau) may now be informative. Try {0.08, 0.10, 0.12, 0.15, 0.20}.
- **epochs**: Loss was still decreasing at 50. Try {50, 75, 100}.
- **embed_dim**: 1000-D input compressed to 128 is aggressive. Try 256.
- **ImageProjector architecture**: 2-layer vs 3-layer, hidden_dim sweep.

### O4: CLIP Image Features Not Yet Sourced
NICE-EEG achieves 15.6% top-1 with CLIP image features. Our CORnet-S gives 5.73% AccU. Sourcing CLIP image embeddings for the THINGS stimuli (from NICE-EEG or BraVL repos) is a future high-impact intervention.

### O5: Structure-Preserving WGAN-GP (Research Thread)
User's research direction: mathematically derive loss function modifications for the WGAN-GP:
- Variance diffusion regularisation (Long et al.'s orthogonal rotation) as a generator loss term
- Dual-graph Laplacian structure preservation in brain and semantic spaces
- Pen-and-paper derivation first, then implementation
- This is the user's unique contribution — the ZSL literature abandoned GANs, making them work with structure preservation is unexplored

---

## 6. Next Steps (Recommended Priority Order)

### Priority 1: Hyperparameter Tuning on Image Alignment
**Why now**: Quick wins available. The paradigm works; optimise within it.
**What to sweep**:
- Epochs: {50, 75, 100} — loss was still decreasing at 50
- Temperature: {0.08, 0.10, 0.12, 0.15, 0.20} — optimal may differ with better prototypes
- embed_dim: {128, 256} — more capacity for 1000-D features
- ImageProjector hidden_dim: {256, 512} — regularisation vs capacity trade-off
**Expected impact**: Could push H-mean from 4.53% to 6-8%.

### Priority 2: Structure-Preserving WGAN-GP (Mathematical Research)
**Why now**: The image alignment paradigm gives a solid foundation. The WGAN-GP is now the second bottleneck (after EEG quality). Improving synthetic embedding quality could further boost AccU.
**What to do**:
- Mathematical derivation of variance diffusion regularisation as a generator loss term
- Dual-graph Laplacian structure preservation formulation
- Gradient dynamics analysis (pen-and-paper)
- Then implementation and testing
**This is the user's main research interest and potential novel contribution.**

### Priority 3: Source CLIP Image Features
**Why**: If CORnet-S (biologically-inspired, not contrastively trained) gives 5.73% AccU, CLIP image features (contrastively trained, shares latent space with text) could push significantly higher.
**How**: Download pre-computed CLIP image embeddings from NICE-EEG or BraVL repos for the THINGS stimuli.
**Expected impact**: Potentially 10-15% AccU (closer to NICE-EEG's 15.6% in unseen-only setting).

### Priority 4: Routing Calibration
**Why**: 20.3% routing rate is 2x the optimal prior. Fixing this would improve AccS with minimal AccU cost.
**How**: Post-hoc temperature scaling on classifier logits, or reduce n_synth_per_class from 20 to 10-15.

---

## 7. Key Context Files for Next Session

| File | Purpose | Priority |
|------|---------|----------|
| `CLAUDE.md` | Project spec, architecture, constraints, current state | **READ FIRST** |
| `MEMORY.md` (auto-loaded) | Persistent memory with all project history and results | **AUTO-LOADED** |
| `context/CORnet_S_paradigm_analysis.md` | Detailed analysis of image alignment results (metric-by-metric, visual analysis, theory, SOTA comparison) | **READ FOR CONTEXT** |
| `context/CORnet_S_paradigm.md` | Raw Colab outputs from the image alignment run | Reference |
| `context/image-alignment-paradigm-shift-debrief.md` | This debrief | **READ FOR HANDOFF** |
| `context/Visual_investigation_results.md` | Visual investigation Colab outputs (archived) | Background |
| `context/Claude_phaseF_analysis.md` | Phase F analysis + literature review (why text alignment failed) | Background |
| `context/notebook-cleanup-v2-debrief.md` | Previous session debrief (v2 cleanup) | Background |
| `plans/current-task.md` | Image alignment task document (already implemented, serves as reference) | Reference |
| `helper_files/switch_to_image_alignment.py` | Script that applied the paradigm shift | Reference |

### Project Configuration

- **Orchestration**: Claude Code (orchestrator) + Antigravity Agents (implementer)
- **Execution**: Google Colab Pro (H100/A100 GPUs, v5e-1 TPU)
- **Compute budget**: 100 credits, user prioritises quality over cost
- **Notebook**: `GZSL_EEG_Pipeline_v2.ipynb` (67 cells, image alignment)
- **User preferences**: Mathematical depth, rigorous theory, full notation, no hand-waving. Wants to keep WGAN-GP and research structure-preserving modifications.

---

## 8. Session Timeline

1. **Session kickstart**: Read CLAUDE.md, MEMORY.md, context files, visual investigation results
2. **Routing catastrophe diagnosis**: Compared v2 and legacy notebooks, found missing sample balance fix
3. **Sample balance fix applied**: Cells 51-52 modified, user ran on Colab, confirmed fix worked (routing 97.9% → 10.8%)
4. **Visual investigation deep analysis**: Mathematical interpretation of CORnet-S vs CLIP text comparison
5. **Plan mode entered**: Designed paradigm shift with 3 explore agents + 1 plan agent
6. **Plan approved**: User approved with emphasis on detailed handoff document
7. **Helper script written and executed**: 16 cell modifications + 9 cell deletions, all 11 sanity checks passed
8. **Handoff document written**: plans/current-task.md (comprehensive reference)
9. **CLAUDE.md and MEMORY.md updated**: Reflect new architecture and state
10. **User ran on Colab**: Results compiled in context/CORnet_S_paradigm.md
11. **Results analysis**: Comprehensive analysis written to context/CORnet_S_paradigm_analysis.md
12. **This debrief**: Handoff document for next session
