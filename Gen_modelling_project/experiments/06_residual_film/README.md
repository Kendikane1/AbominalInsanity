# Experiment 06: Residual FiLM

**Status**: Active
**Notebook**: `notebook.ipynb`

## Hypothesis

Exp05 established a new failure mode: **γ collapse**. With film_hidden=256 and 50K training
steps, the FiLM generator learned to suppress γ1 (to 0.16) and encode all class structure in β1
alone, producing near-deterministic outputs (VarR_unseen=0.593, H=4.10%). The collapse is
self-reinforcing: γ1→0 → Var→0 → ∇L_var→0, so L_var cannot recover.

Two surgical fixes address the root cause without changing the core FiLM architecture:

**Fix 1 — Residual FiLM parameterisation (primary)**

Replace direct γ learning with a residual deviation Δγ:

```
h' = (1 + Δγ(s_c)) ⊙ h + β(s_c)
```

γ = 1 + Δγ is now expressed as a deviation from identity. To reach γ=0.16 requires Δγ=−0.84.
The training dynamics around Δγ=0 (γ=1) are stable: small Δγ perturbations barely affect the
Wasserstein loss (since γ≈1 is the natural solution that preserves noise), while the gradient from
L_var and L_γ constantly pulls Δγ back toward zero. This is the standard approach used in
conditional normalisation literature (Perez et al. 2018, FiLM) for exactly this reason.

**Fix 2 — γ floor regularisation (secondary)**

```
L_γ = λ_γ · E_{s_c}[ ReLU(γ_min − γ(s_c))² ]   summed over both FiLM layers
```

where γ_min=0.5, λ_γ=0.1. This provides an explicit restoring gradient whenever γ < 0.5.
Critically, this gradient is **active precisely at the degenerate state** — unlike L_var which has
∇L_var=0 when variance=0. L_γ is the complement of L_var's failure mode.

**Fix 3 — Reduce film_hidden: 256 → 128**

Limiting FiLM MLP capacity reduces β1's ability to encode all 1654 class centroids without γ,
making the β-dominant shortcut less effective. Combined with fixes 1 & 2, this is belt-and-
suspenders against the Exp05 failure mode.

## Architecture

```
FiLMGenerator(hidden_dim=512, film_hidden=128) — Residual parameterisation:

  z(100) → Linear(100,512) → LeakyReLU → h1
                  ↓
  FiLM layer 1:  s_c(64) → Linear(64,128) → ReLU → Linear(128,1024) → (Δγ1[512], β1[512])
                  h1' = (1 + Δγ1) ⊙ h1 + β1       ← γ = 1 + Δγ (residual)
                  ↓
  h1' → Linear(512,512) → LeakyReLU → h2
                  ↓
  FiLM layer 2:  s_c(64) → Linear(64,128) → ReLU → Linear(128,1024) → (Δγ2[512], β2[512])
                  h2' = (1 + Δγ2) ⊙ h2 + β2
                  ↓
  h2' → Linear(512,64) → L2_norm → ê

Critic: [e(64); s_c(64)] → 256 → 256 → 1  (unchanged)

Jacobian: J_G = W3 · D2(z,s_c) · diag(γ2) · W2 · D1(z) · diag(γ1) · W1
  D1 class-agnostic ✓ (preserved as long as γ1 > 0).
```

### Initialisation

All FiLM MLP last-layer weights and biases initialised to zero:
- Δγ = 0 → γ = 1 (identity scaling at t=0)
- β = 0 (zero shift at t=0)

No explicit bias setting to 1 needed (contrast with Exp04's `bias[:hidden_dim] = 1.0`).

### Parameter counts

| Component | Params | Note |
|---|---|---|
| FiLMGenerator(512, 128) | 690,688 | between Exp04 (257K) and Exp05 (907K) |
| Critic(256) | 99,073 | unchanged |

FiLM MLP: Linear(64,128)=8,320 + Linear(128,1024)=132,096 = 140,416 per layer (×2 = 280,832 total for both layers).

## Changes vs Exp05

| Dimension | Exp05 | **Exp06** | Rationale |
|---|---|---|---|
| film_hidden | 256 | **128** | Limit β capacity — Exp05's β1 encoded all 1654 class centroids |
| FiLM param. | direct γ | **γ = 1 + Δγ** | Stable fixed point; collapse requires Δγ→−0.84 |
| L_γ | none | **λ_γ=0.1, γ_min=0.5** | Restoring gradient active at degenerate state |
| L_G | L_w + L_v | **L_w + L_v + L_γ** | Three-term loss |
| hidden_dim | 512 | 512 | Keep — generator main path capacity |
| n_steps | 50,000 | 50,000 | Keep — training budget correct |
| alpha | 1.0 | 1.0 | Keep — L_var at sweep optimum |

## Training Configuration

```python
WGAN_CONFIG = {
    'z_dim': 100, 'embed_dim': 64,
    'hidden_dim': 512,
    'film_hidden': 128,       # reduced from 256
    'lr': 1e-4, 'betas': (0.0, 0.9),
    'lambda_gp': 10, 'n_critic': 5,
    'n_steps': 50000, 'batch_size': 256,
    'n_synth_per_class': 20,
    'alpha': 1.0,              # L_var weight
    'gamma_min': 0.5,          # floor regularisation threshold
    'lambda_gamma': 0.1,       # floor regularisation weight
    'seed': 42,
    'experiment': 'film_residual',
}
```

## Key Metrics

| Metric | Baseline | Exp04a | Exp05 | **Exp06 target** |
|---|---|---|---|---|
| H-mean | 4.77% | 4.69% | 4.10% | **> 4.77%** |
| VarR_unseen | 0.872 | 0.847 | 0.593 | **> 0.950** |
| VarR_gap | — | 0.078 | 0.105 | **< 0.030** |
| ρ_sp | 0.857 | 0.639 | 0.524 | **< 0.668** |
| γ1_mean | N/A | ~1.0 | 0.160 | **> 0.5** |

## WandB Tracking

Single run. Logs every 500 steps:
`train/L_wasserstein`, `train/L_var`, `train/L_gamma`, `train/L_G`, `train/L_D`, `train/GP`,
`train/delta_gamma1_mean`, `train/gamma1_mean`, `train/gamma1_min`, `train/gamma1_max`,
`train/beta1_norm`, `train/delta_gamma2_mean`, `train/gamma2_mean`, `train/gamma2_min`,
`train/gamma2_max`, `train/beta2_norm`

Every 1000 steps: `train/VarR_seen`  
Every 5000 steps: `train/grad_cos_sim` (L_wass vs L_var conflict)  
Final eval: `eval/VarR_seen`, `eval/VarR_unseen`, `eval/VarR_gap`, `eval/rho_sp`,
`eval/kNN10`, `eval/H_mean`, `eval/AccS`, `eval/AccU`

## Checkpoints

`results/generator_residual_step{N}.pt` every 10K steps + `results/generator_residual_final.pt`.
