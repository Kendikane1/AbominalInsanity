# Session Debrief: Inverse Bias Fix — Phase 0+1 Results & Trajectory Pivot

**Project**: GZSL EEG Decoding (Gen_modelling_project)
**Date**: 2026-03-01
**Session scope**: Plan delivery for Phase 0+1, results analysis, notebook cleanup task, research trajectory reassessment

---

## 1. Task Summary

This session had three objectives:

1. **Deliver Phase 0+1 implementation plan** to the Antigravity agent (eval harness + sample count balancing)
2. **Analyse Phase 1 results** returned by the agent and determine next steps
3. **Notebook cleanup** — produce a task to remove obsolete cells before continuing

The broader context: the GZSL pipeline's classifier (Method D) routes 99.6% of seen-class test inputs to unseen labels (inverse bias). A 5-phase fix strategy was designed targeting three mechanisms: variance mismatch (M1), sample imbalance (M2), and max-over-classes amplification (M3).

---

## 2. What Changed

### Files Created

| File | Purpose |
|------|---------|
| `plans/current-task.md` | Updated twice: first with Phase 0+1 implementation spec, then with notebook cleanup task |
| `plans/sub-plans/inverse-bias-fix-master.md` | Master plan tracking all 5 phases with progress log |
| `/Users/ariz/Desktop/DU/CSlvl2/SE/Target/Context/inverse-bias-fix-session-debrief.md` | This debrief |

### Files Modified

| File | Change |
|------|--------|
| `ANTIGRAVITY_AGENT.md` | Updated status from "stand by" to "first task is live" with pointers to current-task.md and master plan |

### Files Created by Agent (prior to this session)

| File | Content |
|------|---------|
| `helper_files/add_eval_harness.py` | Evaluation harness cells (evaluate_gzsl, diagnose_classifier, compare_phases) |
| `helper_files/add_phase1_sample_balance.py` | Phase 1 cells (downsample unseen, upsample seen, comparison) |
| `phase1_analysis.md` | Agent's analysis of Phase 1 experimental results |

### Notebook State

`COMP2261_ArizMLCW_with_baseline.ipynb` currently has **129 cells**. The agent added Phase 0+1 cells (118–128). A cleanup task is now queued to remove 21 obsolete cells (broken-label ablation, debug artifacts, report text), bringing it to **108 cells**.

---

## 3. Key Decisions

### Decision 1: M2 was the dominant mechanism

Phase 1 results were definitive:

| Metric | Baseline (Method D) | P1A (Downsample) | P1B (Upsample) |
|--------|---------------------|-------------------|-----------------|
| Routing Rate | 99.70% | 9.16% | 8.80% |
| AccS | 0.0006 | 0.0290 | 0.0239 |
| AccU | 0.0213 | 0.0022 | 0.0026 |
| H | 0.0012 | 0.0042 | 0.0046 |

Sample count imbalance (M2) was responsible for nearly all the inverse bias. Simply balancing per-class counts (either by downsampling unseen or upsampling seen) collapsed the routing rate from 99.7% to ~9%. Weight norms converged to ~1.04x ratio (from severely skewed). Calibration is solved.

### Decision 2: Original 2% AccU was an artifact

The baseline's 2.13% unseen accuracy was an illusion — if 99.7% of all predictions go to unseen labels, you catch some correct unseen predictions by volume. With calibration fixed, true AccU is ~0.2% (only 4x above random chance of 1/1854 = 0.054%).

### Decision 3: Trajectory pivot — from calibration fixes to upstream bottleneck identification

The original 5-phase plan (Phases 2-5: covariance-aware generator, cosine classifier, noise injection, post-hoc calibration) was designed assuming the representations were decent but miscalibrated. Phase 1 showed **calibration is now solved, but fundamental discriminative power is near-floor**. This changes the value calculus:

- Phase 2 (cov generator): **Low value** — M1 was secondary to M2
- Phase 3 (cosine classifier): **Moderate** — won't fix weak representations
- Phase 4 (noise injection): **Low** — same reasoning
- Phase 5 (post-hoc calibration): **Near-zero** — already fixed

**New trajectory**: Before investing in Phases 2-5, we need to identify WHERE the representation bottleneck is:

**(A) Is the CLIP encoder weak?** — 64-dim for 1854 classes may be too compressed, or contrastive training on noisy EEG didn't learn enough structure. Diagnostic: seen-only 1654-way classification accuracy.

**(B) Is the WGAN-GP producing useless synthetics?** — Generator may be mode-collapsed, producing near-identical embeddings across unseen classes. Diagnostic: unseen-only 200-way classification on synthetics.

**(C) Are the text prototypes too similar?** — If 200 unseen prototypes cluster tightly in 64-dim space, no generator can make them discriminable. Diagnostic: pairwise cosine similarity among unseen prototypes.

### Decision 4: Notebook cleanup before new experiments

21 cells identified for removal:
- Cells 81–91: Original ablation with broken labels (superseded by corrected ablation)
- Cells 92–97: Debug intermediates (label collision discovery process)
- Cells 106–108, 117: Report-oriented text and duplicate charts

This takes the notebook from 129 to 108 cells with clean flow: data loading → baseline → CLIP → WGAN → diagnostics → label fix → corrected ablation → Phase 0+1.

---

## 4. Current State

### Pipeline Architecture (unchanged)
```
Raw EEG (561-D) → CLIP Brain Encoder → R^64 (L2-normalized)
Text (512-D)    → CLIP Text Projector → R^64 (L2-normalized)
                                              │
                        ┌─────────────────────┤
                        ▼                     ▼
                cWGAN-GP trains on     Text prototypes
                seen embeddings        for unseen classes
                        │                     │
                        ▼                     ▼
                Generator G(z,s_c) → Synthetic unseen embeddings
                        │
                        ▼
                LogReg on real seen + synthetic unseen
                        │
                        ▼
                GZSL: AccS=2.9%, AccU=0.2%, H=0.4% (with balanced counts)
```

### What we know now
- Calibration is fixed (routing rate ~9%, weight norms balanced)
- Discriminative power is near-floor (H ≈ 0.4%)
- Seen embeddings carry ~54x-above-random signal
- Unseen embeddings carry ~4x-above-random signal (barely)
- The bottleneck is upstream (representation quality), not downstream (classifier calibration)

### Active task in queue
Notebook cleanup (`plans/current-task.md`) — awaiting agent execution.

---

## 5. Open Items

1. **Notebook cleanup** — Agent needs to run `cleanup_obsolete_cells.py` to remove 21 cells
2. **Upstream diagnostic task** — Needs to be written after cleanup completes. Three measurements:
   - Seen-only accuracy (CLIP encoder quality)
   - Unseen-only accuracy (generator quality)
   - Prototype spread (semantic space quality)
3. **Phases 2-5 reassessment** — Depending on diagnostic results, some phases may be deprioritized or replaced with upstream improvements (e.g., retraining CLIP with larger embed_dim, training WGAN longer, using better text features)
4. **Master plan update** — `plans/sub-plans/inverse-bias-fix-master.md` needs updating once diagnostics determine the new direction

---

## 6. Next Steps (in order)

1. **Agent executes notebook cleanup** (current task in queue)
2. **Orchestrator writes upstream diagnostic task** → `plans/current-task.md`
   - Seen-only 1654-way LogReg accuracy on CLIP embeddings
   - Unseen-only 200-way LogReg accuracy on synthetic embeddings
   - Pairwise cosine similarity matrix of 200 unseen text prototypes
   - Compare with theoretical baselines (random, nearest-prototype)
3. **Agent runs diagnostics, reports back**
4. **Orchestrator interprets results** and decides:
   - If CLIP encoder is weak → improve contrastive training (more epochs, larger d, curriculum)
   - If generator is weak → invest in Phase 2 (covariance-aware WGAN) or alternative generators
   - If prototypes are too similar → improve text features or use different semantic space
   - If all are weak → consider paradigm change (different alignment strategy, different architecture)
5. **Iterate on the identified bottleneck** before returning to Phases 3-5

---

## Orchestrator–Agent Protocol Notes

- Plans delivered via `plans/current-task.md`
- Agent reads via `@plans/current-task.md` mention
- Results returned via user relay + analysis files (e.g., `phase1_analysis.md`)
- Master plan tracked at `plans/sub-plans/inverse-bias-fix-master.md`
- Agent operating manual at `ANTIGRAVITY_AGENT.md`
