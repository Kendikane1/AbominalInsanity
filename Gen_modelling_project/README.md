# GZSL EEG Decoding — Independent Research

Generalised Zero-Shot Learning (GZSL) for EEG-based visual category decoding using the BraVL / ThingsEEG-Text dataset. The pipeline maps EEG brain signals to a shared embedding space with CORnet-S image features, trains a conditional WGAN-GP to synthesise embeddings for unseen classes, and evaluates a GZSL classifier under the seen+unseen split.

## Research Goal

Improve GZSL H-mean beyond the current 4.77% ceiling by fixing the WGAN-GP synthesis pathology (structural overcoupling, ρ_sp=0.857). Active direction: FiLM/projection-based conditioning to decouple prototype identity from variance behaviour.

## Pipeline Overview

```
Raw EEG (561-D)  ──► BrainEncoder (561→64)  ──┐
                                               ├──► Contrastive training (InfoNCE, τ=0.05)
CORnet-S (1000-D) ──► ImageProjector (1000→64) ──┘
                                               │
                              ┌────────────────┤
                              ▼                ▼
                     cWGAN-GP trains       Image prototypes
                     on seen embeddings    (all classes)
                              │
                              ▼
                  G(z, s_c) → synthetic unseen embeddings
                              │
                              ▼
               LogReg on real seen + synthetic unseen
                              │
                              ▼
               H-mean = 2·AccS·AccU / (AccS + AccU)
```

## Optimal Config (baked into `main_pipeline.ipynb`)

| Component | Parameter | Value |
|-----------|-----------|-------|
| Encoder | embed_dim | 64 |
| Encoder | tau | 0.05 |
| Encoder | lr | 2e-3 |
| Encoder | epochs | 75 |
| Encoder | schedule | cosine warmup (10%) |
| WGAN-GP | z_dim | 100 |
| WGAN-GP | lr | 1e-4 |
| WGAN-GP | betas | (0.0, 0.9) |
| WGAN-GP | n_steps | 10000 |
| WGAN-GP | n_synth_per_class | 20 |

**Best results to date**: H=4.77%, AccS=4.11%, AccU=5.69% (eta sweep baseline, optimal config)

## Directory Structure

```
Gen_modelling_project/
├── main_pipeline.ipynb          # CANONICAL: clean, optimal config, full pipeline
├── README.md
├── CLAUDE.md
├── .env                         # WandB API key (gitignored)
├── .gitignore
│
├── experiments/
│   ├── README.md                # Experiment index + status table
│   ├── 01_wgan_diagnostics/     # [Complete] characterised synthesis pathology
│   ├── 02_eta_sweep/            # [Failed] post-hoc perturbation
│   ├── 03_variance_regularisation/  # [Failed] L_var training-time loss
│   └── 04_film_conditioning/    # [Active] FiLM/projection conditioning
│       └── notebook.ipynb       # copy of main_pipeline.ipynb + modifications
│
├── context/
│   ├── math/                    # Math curriculum, system prompts, project context
│   └── research_history/        # Analysis docs: CORnet-S paradigm, sweeps, WGAN-GP
│
├── shared/
│   └── wandb_utils.py           # WandB helper functions (init_run, log_training_step, etc.)
│
├── figures/                     # Main pipeline figures (loss curves, t-SNE)
├── data/ThingsEEG-Text/         # Dataset (unchanged)
└── archive/                     # Frozen legacy artefacts — do not edit
    ├── notebooks/               # v2 monolith + legacy notebook + backups
    ├── helper_scripts/          # All add_*.py injection scripts
    ├── context/                 # Session debriefs + old plans
    └── misc/                    # Misc scripts, PDF report, historical figures
```

## How to Run

1. Upload `main_pipeline.ipynb` to Google Colab Pro
2. Mount Drive and set paths in cell 7 (data loading)
3. Fill in `WANDB_ENTITY` in the WandB auth cell
4. Run all cells top-to-bottom (H100/A100 recommended; ~15 min total)

## Starting a New Experiment

```bash
# Experiment folder already exists for 04 — for future ones:
mkdir -p experiments/05_new_idea/{context,results,figures}
cp main_pipeline.ipynb experiments/05_new_idea/notebook.ipynb
# Edit notebook.ipynb in Colab, save results/figures to experiment folder
```

## WandB

Project: `gzsl-eeg-bravl`. Each training run calls `wandb.init()` with full config dict. API key stored in `.env` (gitignored). See `shared/wandb_utils.py` for helper functions.
