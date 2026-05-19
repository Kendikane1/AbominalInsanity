# Debrief: Experiments 05 & 06 — High-Capacity FiLM and Residual FiLM

**Date**: 2026-05-19  
**Session scope**: Exp05 build + analysis → Exp06 (50K) build + analysis → Exp06 (10K) analysis  
**Status**: Exp05 and Exp06 complete. Next experiment (Exp07) not yet started.

---

## 1. Task Summary

The objective was to address the FiLM variance deficit identified in Experiments 04a–c: despite FiLM correctly decoupling the noise path from prototype conditioning (ρ_sp fixed: 0.857→0.639), H-mean did not improve over the concat baseline (4.77%) and VarR_unseen worsened (0.847 vs 0.872).

The hypothesis was that the Exp04 generator lacked capacity (hidden_dim=256) and training budget (10K steps). Two sequential experiments were run:

- **Exp05**: Scale up (hidden_dim=512, film_hidden=256, 50K steps, L_var α=1.0) — **FAILED** due to γ collapse
- **Exp06**: Surgical fix for γ collapse (residual FiLM γ=1+Δγ, L_γ floor, film_hidden=128) — run at both **50K** and **10K** steps

The session revealed two independent failure modes: γ collapse (Exp05) and routing pathology from generator over-convergence (Exp05 and Exp06 at 50K, and partially at 10K with the larger generator).

---

## 2. What Changed

### New Files Created

| File | Description |
|---|---|
| `experiments/05_high_capacity_film/README.md` | Hypothesis, architecture spec, changes vs Exp04, WGAN_CONFIG, key metrics table, WandB tracking list, checkpoint naming |
| `experiments/05_high_capacity_film/notebook.ipynb` | 77-cell notebook derived from `notebook_film_lvar.ipynb`. Single training run, no sweep. Key changes in cells 41/42/44/45/75. |
| `experiments/05_high_capacity_film/results/analysis.md` | Full Jacobian-level analysis of γ collapse. Explains L_var vanishing gradient, critic blindness, β-dominant shortcut. Prescribes Exp06. |
| `experiments/06_residual_film/README.md` | Three-fix architecture spec (residual γ, L_γ floor, film_hidden=128), zero-bias init explanation, param counts, WandB tracking |
| `experiments/06_residual_film/notebook.ipynb` | 77-cell notebook derived from Exp05. Residual FiLM `_film`, `_init_film_identity`, `get_film_stats`, `compute_lgamma` implementations. |
| `experiments/06_residual_film/results/analysis.md` | Two-part document: (1) 50K-step run analysis, (2) Section 9 addendum for 10K-step run |
| `archive/misc/build_exp05_notebook.py` | Build script used to create Exp05 notebook (archived immediately after use) |
| `archive/misc/build_exp06_notebook.py` | Build script used to create Exp06 notebook (archived immediately after use) |

### Files Modified

| File | Change |
|---|---|
| `experiments/README.md` | Added Exp05 row (Failed, 4.10%) and Exp06 row (Active) |
| `CLAUDE.md` | Directory structure updated (Exp05→Failed, Exp06→Active); Key Files section updated; Active Experiment section replaced |
| `memory/MEMORY.md` | Project status updated for Exp05 failure, Exp06 implementation, and Exp06 10K results |

---

## 3. Key Decisions and Architecture

### Exp05: FiLMGenerator(512, 256) — What Was Built

```
z(100) → Linear(100,512) → LeakyReLU → h1
          ↓
FiLM 1: s_c(64) → Linear(64,256) → ReLU → Linear(256,1024) → (γ1[512], β1[512])
         h1' = γ1 ⊙ h1 + β1      ← direct γ learning
          ↓
h1' → Linear(512,512) → LeakyReLU → h2
          ↓
FiLM 2: same structure → (γ2[512], β2[512])
         h2' = γ2 ⊙ h2 + β2
          ↓
h2' → Linear(512,64) → L2_norm → ê

Init: bias[:512] = 1.0, bias[512:] = 0.0 (γ=1, β=0 at t=0)
Params: 906,816
```

**WGAN_CONFIG (Exp05)**:
```python
{'hidden_dim': 512, 'film_hidden': 256, 'n_steps': 50000,
 'alpha': 1.0, 'experiment': 'film_highcap'}
```

### Exp06: FiLMGenerator(512, 128) — Residual Parameterisation

```
z(100) → Linear(100,512) → LeakyReLU → h1
          ↓
FiLM 1: s_c(64) → Linear(64,128) → ReLU → Linear(128,1024) → (Δγ1[512], β1[512])
         h1' = (1 + Δγ1) ⊙ h1 + β1    ← residual: γ = 1 + Δγ
          ↓
... (same structure, layer 2)

Init: ALL last-layer weights AND biases zeroed → Δγ=0 → γ=1, β=0 at t=0
Params: 628,032
```

**WGAN_CONFIG (Exp06)**:
```python
{'hidden_dim': 512, 'film_hidden': 128, 'n_steps': 50000,  # also run at 10000
 'alpha': 1.0, 'gamma_min': 0.5, 'lambda_gamma': 0.1,
 'experiment': 'film_residual'}
```

**Three-term generator loss**:
```
L_G = L_wass + 1.0 · L_var + 0.1 · L_gamma
L_gamma = Σ_{layers} E_{s_c}[ ReLU(0.5 − (1+Δγ))² ]
```

**Key implementation details in `notebook.ipynb`**:
- `_init_film_identity`: zeros all weights AND biases of last Linear in each FiLM MLP (contrast with Exp05 which set `bias[:hidden_dim] = 1.0`)
- `_film(self, h, mlp, s_c)`: `params.chunk(2) → delta_gamma, beta; return (1.0 + delta_gamma) * h + beta`
- `get_film_stats(self, s_c)`: returns delta_gamma1_mean, gamma1_mean, gamma1_min/max, beta1_norm, same for layer 2
- `compute_lgamma(gen, s_c)`: iterates over `(gen.film1_mlp, gen.film2_mlp)`, accumulates floor penalty

**WandB tracking added in Exp06 (not in Exp05)**:
- `train/L_gamma`, `train/delta_gamma1_mean`, `train/gamma1_mean`, `train/gamma1_min`, `train/gamma1_max`
- Same for layer 2
- Cell 50: gamma histogram with `axvline(x=0.5, color='red', label='γ_min')`, `axvline(x=1.0, color='green', label='γ=1 identity')`, `axvline(x=0.0, color='black', label='γ=0 collapse')`

---

## 4. Results Summary

### Full Experiment Comparison Table

| Metric | Baseline | Exp04a | Exp04c α=1.0 | Exp05 50K | Exp06 50K | Exp06 10K |
|---|---|---|---|---|---|---|
| **H-mean** | **4.77%** | **4.69%** | 4.67% | 4.10% | 4.05% | **4.60%** |
| AccS | 4.11% | 4.17% | 3.75% | 3.33% | 3.36% | 3.69% |
| AccU | 5.69% | 5.09% | 6.19% | 5.36% | 5.11% | **6.10%** |
| routing | ~20% | ~20% | ~20% | 54.81% | 48.15% | 32.89% |
| VarR_seen | 0.872 | 0.925 | 0.931 | 0.698 | 0.776 | 0.929 |
| VarR_unseen | 0.872 | 0.847 | 0.847 | 0.593 | 0.676 | 0.840 |
| ρ_sp | 0.857 | 0.639 | 0.678 | 0.524 | 0.553 | 0.703 |
| γ1_mean | N/A | ~1.0 | ~1.0 | 0.160 | 0.510 | 0.597 |
| γ1_min | N/A | — | — | — | −0.252 | +0.074 |
| c_loss | — | — | — | −0.025 | −0.022 | **−0.012** |

### Failure Mode 1: γ Collapse (Exp05)

The Exp05 generator (film_hidden=256) learned to suppress γ1 from its initialisation of 1.0 to 0.16 over 50K steps. The mechanism:

```
γ1 → 0  ⟹  h1' ≈ β1(s_c)     (noise path suppressed)
         ⟹  D2 becomes prototype-specific (FiLM's structural guarantee nullified)
         ⟹  Var[G(z,s_c)] → 0
         ⟹  ∇L_var → 0         (L_var cannot recover — gradient vanishes at degenerate equilibrium)
```

With film_hidden=256, the β1 MLP had sufficient capacity to encode all 1654 class centroids alone, making γ redundant. 50K steps allowed full convergence to this local minimum. VarR_unseen = 0.593, routing = 54.81%, H = 4.10%.

### Failure Mode 2: Routing Pathology from Generator Over-Convergence

Identified across both Exp05 and Exp06. The causal chain:

```
50K steps (or hidden_dim=512 at 10K steps) → c_loss → 0 (generator fully converged)
→ tight synthetic clusters per class
→ LogReg builds over-confident unseen decision boundaries
→ routing spikes (48–55% at 50K; 32.89% at 10K with large generator)
→ AccS suppressed (seen EEG overlaps unseen boundaries)
→ H-mean degrades despite VarR improvement
```

**The routing–c_loss law** (confirmed across 5 data points):

| Experiment | hidden_dim | n_steps | c_loss_final | routing | H-mean |
|---|---|---|---|---|---|
| Exp04a | 256 | 10,000 | ~−1 to −2 | ~20% | 4.69% |
| Exp04c | 256 | 10,000 | ~−1 to −2 | ~20% | 4.67% |
| Exp06 10K | 512 | 10,000 | −0.012 | 32.89% | 4.60% |
| Exp05 50K | 512 | 50,000 | −0.025 | 54.81% | 4.10% |
| Exp06 50K | 512 | 50,000 | −0.022 | 48.15% | 4.05% |

Routing tracks c_loss, not n_steps or architecture directly. The hidden_dim=512 generator (628K params) converges approximately 2.5× faster than hidden_dim=256 (257K params).

### Exp06 10K: Partial γ Fix, Routing Still Elevated

- γ1_min = +0.074 (all positive — floor held, no sign inversion)
- Δγ_mean = +0.403 → effective γ ≈ 1.40 (amplification, not suppression)
- But c_loss = −0.012 (near-converged at 10K) → routing = 32.89%
- ρ_sp = 0.703 (regression from Exp04a's 0.639)
- H = 4.60% — not an improvement over Exp04a (4.69%), within stochastic noise band (±0.13pp)

---

## 5. Current State

**Project best**: H = 4.77% (concat baseline, eta=0, optimal config, 10K steps)

**Research arc status**:
- Generator architectural improvements: **saturating**. Six WGAN-GP variants, none exceeding 4.77%.
- AccU ceiling: 5.69–6.19% across all experiments. H-mean range: 4.05–4.77%.
- Encoder bottleneck hypothesis: **strengthening**. 30% encoder gain at Phase E produced 0% H-mean gain. Generator improvements also not breaking through.

**Residual FiLM state** (`experiments/06_residual_film/notebook.ipynb`):
- 77 cells, fully functional
- Cells 41–44: WGAN_CONFIG with gamma_min/lambda_gamma, FiLMGenerator with residual _film, compute_lgamma, training loop
- Cell 50: FiLM diagnostic figures with γ reference lines
- Cell 75: harvest cell with gamma1_min, delta_gamma1_mean, gamma2_min

**Analysis document** (`experiments/06_residual_film/results/analysis.md`):
- Sections 1–8: 50K run analysis
- Section 9: 10K run addendum (routing–c_loss law, amplification regime, ρ_sp regression, three-option path forward)

---

## 6. Open Items

1. **Exp05 README status** — still says "Active". Should be updated to "Failed".
2. **Exp06 README parameter count** — states 690,688 but correct count is 628,032 (minor documentation error; training used correct architecture).
3. **ACE handover for Exp06** — not yet generated. User requested handovers after Exp04 and Exp05; Exp06 handover pending.
4. **experiments/README.md Exp06 status** — still "Active". Should be updated to "Complete (Failed)" once the user decides the next experiment.
5. **CLAUDE.md active experiment section** — still points to Exp06 as active. Needs update once Exp07 begins.

---

## 7. Next Steps

Three options, assessed in `experiments/06_residual_film/results/analysis.md` section 9.7:

### Option A — Exp07: Convergence-Equivalent Step Count for hidden_dim=512 (~4K steps) [Recommended]

Run the Exp06 architecture with n_steps ≈ 3,500–4,000, targeting c_loss in the −1 to −2 range at training end. This achieves the same non-convergence state as Exp04 at 10K steps.

```python
WGAN_CONFIG = {
    'hidden_dim': 512, 'film_hidden': 128,
    'n_steps': 4000,        # target: c_loss ≈ −1 to −2 at end
    'alpha': 1.0, 'gamma_min': 0.5, 'lambda_gamma': 0.1,
    'experiment': 'film_residual_4k',
}
```

Monitor WandB: if c_loss → 0 before step 4K, reduce further. Key test: does residual FiLM + hidden_dim=512 beat Exp04a's 4.69% at equivalent convergence? If yes → genuine architectural benefit. If H-mean stays at ~4.7% → confirms encoder bottleneck.

### Option B — Encoder Architecture Shift

The WGAN-GP generator improvements have saturated. Next meaningful lever is EEG representation quality: richer temporal models (EEGNet, LSTM per channel), larger EEG training data, stronger contrastive objectives. Larger architectural change requiring new experimental design.

### Option C — Residual FiLM at hidden_dim=256 (~10K steps)

Test whether the residual parameterisation itself (independent of hidden_dim increase) provides benefit over Exp04a. Smaller scope than Option A; closes the residual-vs-direct γ question definitively.

---

## 8. Important Technical Notes for Handoff

- **Label convention**: seen labels [1..1654], unseen labels [1655..1854]. Collision fix at source (cell 7). Never remap downstream.
- **StandardScaler**: fit on train_seen only, applied to test_seen and unseen.
- **Sample balance**: synthetic unseen downsampled to seen per-class median (~8/class). LogReg with `class_weight='balanced'`.
- **WGAN-GP stochasticity**: same config can produce ±0.13pp H-mean variance across runs (WGAN-GP inherent).
- **c_loss diagnostic**: monitor WandB `train/L_D` — if it approaches 0 before training ends, routing will spike. Target: c_loss in range −1 to −5 throughout training for healthy routing.
- **γ diagnostics**: `train/gamma1_mean` and `train/gamma1_min` are the primary collapse indicators. If gamma1_mean < 0.5 or gamma1_min < 0, the floor is not holding.
- **Notebook lineage**: `main_pipeline.ipynb` (69 cells, canonical) → Exp04 notebooks → Exp05 notebook → Exp06 notebook. Each derived by build script from the previous.
- **WandB project**: `gzsl-eeg-bravl`. All experiment runs named by experiment tag (e.g., `06_residual_film`).
