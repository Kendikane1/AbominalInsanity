# Experiment 05: High-Capacity FiLM

**Status**: Active
**Notebook**: `notebook.ipynb`

## Hypothesis

Exp04 established that FiLM architecture correctly decouples the noise pathway (D1 class-agnostic)
from prototype conditioning, fixing ρ_sp (0.857→0.639). Despite this structural fix, H-mean did
not improve (4.59–4.69% vs 4.77% baseline), and VarR_unseen worsened (0.847 vs 0.872).

Three compounding deficits were identified from the Exp04 analysis:

1. **Insufficient training budget**: 10K steps with 3× more generator parameters (256K→907K) is
   inadequate for convergence. The ratio of steps-to-params dropped 3×.
2. **Generator capacity too low for structured variance**: hidden_dim=256 with FiLM has the same
   hidden width as the concat baseline, but the FiLM path adds conditioning overhead. The generator
   may not have capacity to simultaneously satisfy the Wasserstein objective and L_var.
3. **L_var at α=1.0 was the Exp04 sweep optimum** but was trained for only 10K steps. At 50K
   steps, the generator has time to internalise the variance objective rather than fighting it.

This experiment combines all three corrections in a single architecture:

```
FiLMGenerator(hidden_dim=512, film_hidden=256) + L_var(α=1.0) + n_steps=50,000
```

## Architecture

```
Generator (hidden_dim=512, film_hidden=256):
  z(100) → Linear(100,512) → LeakyReLU → h1
                              ↓
  FiLM layer 1:  s_c(64) → Linear(64,256) → ReLU → Linear(256,1024) → (γ1[512], β1[512])
                              h1' = γ1 ⊙ h1 + β1
                              ↓
  h1' → Linear(512,512) → LeakyReLU → h2
                              ↓
  FiLM layer 2:  s_c(64) → Linear(64,256) → ReLU → Linear(256,1024) → (γ2[512], β2[512])
                              h2' = γ2 ⊙ h2 + β2
                              ↓
  h2' → Linear(512,64) → L2_norm → ê

Critic (unchanged from Exp04, hidden_dim=256):
  [e(64); s_c(64)] → Linear(128,256) → LeakyReLU → Linear(256,256) → LeakyReLU → Linear(256,1)

Jacobian: J_G = W3 · D2(z, s_c) · diag(γ2) · W2 · D1(z) · diag(γ1) · W1
  D1 = diag(1[fc1(z) > 0])  — depends only on z (class-agnostic)
  D2 = diag(1[fc2(γ1⊙h1+β1) > 0])  — depends on both z and s_c via modulation
```

### Parameter counts
| Component | Params | vs Exp04 |
|---|---|---|
| FiLMGenerator(512, 256) | 906,816 | 3.53× |
| Critic(256) | 99,073 | unchanged |

### Initialisation
- γ = 1, β = 0 (identity modulation at t=0): last linear of each FiLM MLP zero-initialised,
  bias set to `[1.0]*512 + [0.0]*512`.

## Changes vs Exp04 Variants

| Dimension | Exp04a (baseline) | Exp04c (lvar) | **Exp05 (this)** |
|---|---|---|---|
| hidden_dim | 256 | 256 | **512** |
| film_hidden | 128 | 128 | **256** |
| n_steps | 10,000 | 10,000 | **50,000** |
| L_var α | — | 1.0 | **1.0** |
| Generator params | 256K | 256K | **907K** |

## Training Configuration

```python
WGAN_CONFIG = {
    'z_dim': 100, 'embed_dim': 64,
    'hidden_dim': 512, 'film_hidden': 256,
    'lr': 1e-4, 'betas': (0.0, 0.9),
    'lambda_gp': 10, 'n_critic': 5,
    'n_steps': 50000, 'batch_size': 256,
    'n_synth_per_class': 20, 'alpha': 1.0,
    'experiment': 'film_highcap',
}
```

## Key Metrics

| Metric | Baseline (Exp01) | Exp04a best | Exp04c (α=1.0) | **Exp05 target** |
|---|---|---|---|---|
| H-mean | 4.77% | 4.69% | 4.62% | **> 4.77%** |
| VarR_unseen | 0.872 | 0.847 | 0.873 | **> 0.950** |
| VarR transfer gap | ? | 0.138 | 0.109 | **< 0.030** |
| ρ_sp | 0.857 | 0.639 | 0.594 | **< 0.668** |

## WandB Tracking

Single run per execution. Logs:
- Every 500 steps: `train/L_wasserstein`, `train/L_var`, `train/L_G`, `train/L_D`, `train/GP`,
  `train/gamma1_mean`, `train/gamma2_mean`, `train/beta1_norm`, `train/beta2_norm`
- Every 1000 steps: `train/VarR_seen`
- Every 5000 steps: `train/grad_cos_sim` (gradient conflict diagnostic)
- Post-training: `eval/VarR_seen_post_training`
- Final eval: `eval/VarR_seen`, `eval/VarR_unseen`, `eval/VarR_gap`, `eval/rho_sp`,
  `eval/kNN10`, `eval/H_mean`, `eval/AccS`, `eval/AccU`

## Checkpoints

Saved every 10K steps to `results/generator_highcap_step{N}.pt` + final model to
`results/generator_highcap_final.pt`.
