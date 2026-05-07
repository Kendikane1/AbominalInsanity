# Experiment 04: FiLM Conditioning

**Status**: Active
**Notebooks**:
- `notebook.ipynb` — FiLM baseline (pure architecture fix, no auxiliary loss)
- `notebook_film_ln.ipynb` — FiLM + LayerNorm after each modulation step
- `notebook_film_lvar.ipynb` — FiLM + L_var auxiliary loss (alpha sweep: 0.0→10.0)

## Hypothesis

FiLM (Feature-wise Linear Modulation) replaces concatenation conditioning in the cWGAN-GP Generator. Concatenation conditioning was identified as the root cause of Exp 03's transfer failure: `G(z, s_c) = MLP([z; s_c])` entangles the noise pathway with the prototype, so learned variance behaviour is prototype-specific and does not transfer to unseen classes.

FiLM separates these:
- Noise path (class-agnostic): `z → h1 → h2 → ê`
- Prototype conditioning: `s_c → (γ_i, β_i)` per layer

Jacobian under FiLM: `J_G = W3 · D2(z, s_c) · diag(γ2) · W2 · D1(z) · diag(γ1) · W1`

`D1` depends only on `z` (not `s_c`). Variance is `Var[G(z, s_c)] ≈ A(s_c) · B · A(s_c)^T` where `B` is class-agnostic and `A(s_c) = diag(γ2) · W2 · D1 · diag(γ1)` is a smooth, continuous function of `s_c` via learned MLPs. Unseen prototypes can interpolate smoothly across `A(s_c)` — the mechanism that should close the VarR transfer gap.

## Architecture Change

**Exp 01–03** (concatenation, failed):
```
G(z, s_c): [z; s_c] → 256 → 256 → d
```

**Exp 04** (FiLM):
```
G(z, s_c): z → h1 = LeakyReLU(W1·z)
                 h1' = γ1(s_c) ⊙ h1 + β1(s_c)     ← FiLM layer 1
                 h2 = LeakyReLU(W2·h1')
                 h2' = γ2(s_c) ⊙ h2 + β2(s_c)     ← FiLM layer 2
                 ê  = L2_norm(W3·h2')

FiLM MLP: s_c (64-D) → 128 → 512 → [γ_i | β_i]  (256-D each)
Init: γ = 1, β = 0 (identity modulation at t=0)
```

## Variant Rationale

| Notebook | Key Change | Addresses |
|---|---|---|
| `notebook.ipynb` | Pure FiLM | Root cause: concatenation conditioning |
| `notebook_film_ln.ipynb` | FiLM + LayerNorm post-modulation | β magnitude problem: large β creates s_c-dependent D2 |
| `notebook_film_lvar.ipynb` | FiLM + L_var sweep | Combines architecture fix with explicit variance regularisation |

## Key Metrics

| Metric | Baseline (Exp 01) | Exp 03 best | FiLM target |
|---|---|---|---|
| H-mean | 4.77% | 4.58% | > 4.77% |
| VarR_unseen | 0.872 | 0.875 | > 0.950 |
| VarR transfer gap | ? | 0.098 | < 0.030 |
| ρ_sp | 0.857 | 0.880 | < 0.800 |

## WandB Tracking

Each run logs: `train/VarR_seen` (every 1000 steps), `train/gamma1_mean`, `train/gamma2_mean`, `train/beta1_norm`, `train/beta2_norm`, `eval/VarR_seen`, `eval/VarR_unseen`, `eval/VarR_gap`, `eval/rho_sp`, `eval/kNN10`, `eval/H_mean`, `eval/AccS`, `eval/AccU`.

L_var variant additionally logs: `train/L_var`, `train/grad_cos_sim` (gradient conflict diagnostic).
