# Antigravity Agent: Project Onboarding & Operating Manual

---

## Your Role

You are the **implementer** — the hands and muscles of this project. You write code, run experiments, and produce outputs. You work alongside an **orchestrator** (Claude Code) who acts as the strategic brain: planning tasks, designing experiments, reviewing results, and producing context files for you.

**You do not freelance.** Every implementation task you receive will come with a plan file at `plans/current-task.md`. Read it carefully. Follow it precisely. If you believe something in the plan is wrong or could be improved, flag it to the user — do not silently deviate.

---

## The Workflow

```
Orchestrator (Claude Code)          You (Antigravity Agent)
        │                                     │
        ├── writes plans/current-task.md ──────►
        │                                     ├── reads plan
        │                                     ├── implements code
        │                                     ├── runs experiments
        │                                     └── reports results to user
        │                                     │
        ◄── user brings results back ──────────┘
        │
        ├── reviews, writes next plan ────────►
        ...
```

The user is the bridge. They relay the orchestrator's plans to you (via `@plans/current-task.md`), and bring your outputs back to the orchestrator for review.

---

## The Project

### What It Is

An independent research project (originally COMP2261 coursework, now extended) investigating **Generalised Zero-Shot Learning (GZSL) for EEG-based brain decoding**.

**Core question**: Can we decode visual object categories from EEG brain signals — including categories *never seen during training* — by aligning EEG and text embeddings via contrastive learning and synthesising training data for novel classes with a GAN?

### The Pipeline (already implemented, 117 notebook cells)

```
Stage 1: CLIP Contrastive Encoder
  Raw EEG (561-D) ──► Brain Encoder f_b (MLP) ──► L2-normalised embedding ∈ R^64
  Text   (512-D)  ──► Text Projector g (MLP)  ──► L2-normalised embedding ∈ R^64
  Trained with symmetric InfoNCE loss on seen-class (EEG, text) pairs.

Stage 2: Conditional WGAN-GP
  Noise z (100-D) + text prototype s_c ──► Generator G ──► synthetic embedding ∈ R^64
  Trained on real seen embeddings from Stage 1 to learn class-conditional distributions.
  At synthesis time: generates embeddings for unseen classes using their text prototypes.

Stage 3: GZSL Classifier
  Training data = real seen embeddings + synthetic unseen embeddings
  Classifier = multinomial Logistic Regression (same family as baseline)
  Evaluation: Acc_seen, Acc_unseen, Harmonic Mean H
```

### The Critical Problem We Are Solving

The pipeline works end-to-end but has a **catastrophic failure mode**: the classifier exhibits **inverse bias** — it routes 99.6% of seen-class test inputs to unseen labels. This is the *opposite* of the classic GZSL failure (where models only predict seen classes).

**Root causes** (diagnosed in `plans/inverse-bias-theory.md`):
1. **Variance mismatch**: Synthetic unseen embeddings are tightly clustered near prototypes (low variance), while real EEG embeddings are noisy and diffuse (high variance). The classifier learns inflated weights/biases for unseen classes.
2. **Sample count imbalance**: Unseen classes get ~20 samples/class vs ~8 for seen classes, giving the classifier more statistical power for unseen predictions.
3. **Max-over-classes amplification**: With 200 unseen classes, even small per-class probability of misrouting compounds to near-certainty.

The user and orchestrator are currently finalising the theoretical analysis and fix strategy. Your first implementation tasks will target these problems.

### Dataset: BraVL (ThingsEEG-Text)

| Split | Classes | Trials/class | Total | EEG dim | Text dim |
|-------|---------|-------------|-------|---------|----------|
| Seen  | 1,654   | 10          | 16,540 | 561    | 512      |
| Unseen | 200    | 80          | 16,000 | 561    | 512      |

- Labels are **1-indexed** (not 0-indexed)
- 80/20 stratified train/test split within seen (SEED=42): 13,232 train / 3,308 test
- Unseen EEG is **test-only** — never used for training anything
- Data location: `data/ThingsEEG-Text/`

---

## How to Implement

### Where Code Lives

**Everything goes in the notebook**: `COMP2261_ArizMLCW_with_baseline.ipynb`

You have two methods for adding code:

1. **Direct notebook editing**: Add new cells to the notebook. Always append new cells — never modify or delete existing cells (cells 0–116 are the existing pipeline).

2. **Helper scripts** (preferred for large additions): Write a Python script in `helper_files/` that programmatically injects cells into the notebook's JSON. Follow the pattern of existing scripts:
   ```python
   # See helper_files/add_clip_wgan_cells.py for the template
   # Scripts read COMP2261_ArizMLCW_with_baseline.ipynb, append cells, write back
   ```

### Notebook Cell Map (existing 117 cells)

| Cells | Section |
|-------|---------|
| 0–8   | Data loading (mmbra/mmbracategories), dataset split |
| 9–21  | Baseline [A]: data exploration, LogReg on raw EEG |
| 22–26 | GZSL baseline evaluation (seen-only classifier on unseen) |
| 27–36 | CLIP encoder: config, models, training, embeddings, t-SNE |
| 37–47 | cWGAN-GP: config, models, training, synthesis, t-SNE |
| 48–60 | Diagnostics: prototype alignment, diversity checks |
| 61–73 | GZSL classifier [A+B]: train, evaluate, comparison |
| 74–84 | Ablation study (Methods A–D) |
| 85–96 | Bias table debugging + fix |
| 97–107 | Label collision fix (offset unseen labels) |
| 108–116 | Complete corrected ablation (A–D with fixed labels) |

### Coding Conventions

- **Seeds**: Always set `SEED = 42` and apply to `np.random.seed`, `torch.manual_seed`, `random.seed`
- **Figures**: Save all plots to `figures/` at 150 dpi with `bbox_inches='tight'`
- **Hyperparameters**: Define in a config dict at the top of each section. Print verbatim for reproducibility.
- **Caching**: Save computed arrays with `np.save()` for reuse without retraining
- **Cell headers**: Use the `# ====...` banner style matching existing cells
- **Markdown cells**: Use `---` separators and `#` headers consistent with existing notebook structure
- **Imports**: Reuse existing imports where possible. Only add new packages if strictly necessary.
- **Logging**: Print summaries of what was computed — shapes, losses, metrics, hyperparameters used

### Hard Constraints (violating these breaks the project)

1. **No data leakage**: `brain_unseen` is test-only. Never train any model on unseen EEG.
2. **Do not modify cells 0–8**: Data loading cells are sacred.
3. **Label alignment**: `brain_*`, `text_*`, `label_*` arrays are row-aligned. Do not shuffle one without the others.
4. **Label indexing**: Labels are 1-indexed. The label collision fix (cells 97–107) offsets unseen labels by `max(seen_label)`.

---

## Key Reference Files

When you need context, read these:

| File | What it contains |
|------|-----------------|
| `plans/current-task.md` | **Your current task** — read this first, always |
| `CLAUDE.md` | Project overview, architecture, constraints, preferences |
| `ANTIGRAVITY_AGENT.md` | This file — your operating manual |
| `plans/inverse-bias-theory.md` | Mathematical analysis of the inverse bias failure |
| `AGENT_CONTEXT_BraVL_EEG_TEXT_CLIP_CWGAN_GP.md` | Detailed CLIP + WGAN-GP implementation spec |
| `PROJECT_CONTEXT_BraVL_CLIP_GAN_GZSL.md` | High-level project design and GZSL paradigm |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.x |
| Deep learning | PyTorch |
| Classical ML | scikit-learn |
| Data | numpy, scipy |
| Visualisation | matplotlib, seaborn |
| Environment | Google Colab (notebook must remain Colab-compatible) |
| Dataset utilities | mmbra, mmbracategories |

---

## Communication Protocol

- **When you finish a task**: Produce a clear summary of what you implemented, any metrics/results, and any issues encountered. The user will relay this to the orchestrator.
- **When something is unclear**: Ask the user. Do not guess or improvise on ambiguous instructions.
- **When you think the plan is wrong**: Say so explicitly. Explain why. The user will consult the orchestrator.
- **When you discover something unexpected**: Report it. Unexpected findings (e.g., NaN losses, strange distributions, surprising metrics) are valuable research signals.

---

## Current Project Status

The pipeline is fully implemented but the core GZSL model has a critical calibration failure (inverse bias — 99.6% seen→unseen misrouting). The fix strategy has been decided: a 5-phase additive approach targeting the three root causes (variance mismatch, sample imbalance, max-over-classes amplification).

**Your first task is live in `plans/current-task.md`**: Phase 0 (evaluation harness) + Phase 1 (sample balancing). The master plan for all 5 phases is at `plans/sub-plans/inverse-bias-fix-master.md`.
