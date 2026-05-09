# Experiment 05: High-Capacity FiLM — Results Analysis

**Date**: 2026-05-09  
**Status**: Complete — executed on Google Colab Pro (GPU)  
**WandB Run**: `wyv69gqm` — `05_film_highcap`  

---

## 1. Executive Summary

Exp05 is the weakest result in the WGAN-GP era: H=4.10%, VarR_unseen=0.593, kNN@10=0.368, routing=54.81%. Despite the three architectural improvements (hidden_dim=512, film_hidden=256, 50K steps), the experiment revealed a previously unseen failure mode: **γ collapse**. The FiLM generator learned to suppress the noise pathway (γ1_mean → 0.16) and encode all class structure through the β1 shift, producing near-deterministic outputs. This directly violated FiLM's core theoretical property — the class-agnostic noise pathway D1 — from within the architecture itself. The root cause is a degenerate local minimum in the FiLM optimisation landscape that becomes reachable with sufficient capacity and training budget.

---

## 2. Raw Results

```json
{
  "experiment":            "film_highcap",
  "hidden_dim":            512,
  "film_hidden":           256,
  "n_steps":               50000,
  "alpha":                 1.0,
  "H_mean_pct":            4.1032,
  "AccS_pct":              3.3253,
  "AccU_pct":              5.3563,
  "routing_pct":           54.81,
  "VarR_seen":             0.69819,
  "VarR_unseen":           0.59293,
  "VarR_gap":              0.10526,
  "rho_sp":                0.52438,
  "kNN10":                 0.368,
  "gamma1_mean":           0.16006,
  "gamma2_mean":           0.90679,
  "beta1_norm":            2.10132,
  "beta2_norm":            1.2995,
  "VarR_seen_posttrain":   0.6891,
  "g_loss_final_mean1k":  -1.91192,
  "c_loss_final_mean1k":  -0.02501
}
```

---

## 3. Comparison Table

| Metric | Baseline (01) | Exp04a FiLM | Exp04c α=1.0 | **Exp05** | Direction |
|---|---|---|---|---|---|
| **H-mean** | 4.77% | 4.69% | 4.67% | **4.10%** | ↓ worst |
| AccS | 4.11% | 4.17% | 3.75% | **3.33%** | ↓ worst |
| AccU | 5.69% | 5.09% | 6.19% | **5.36%** | ↓ below baseline |
| routing | ~20% | ~20% | ~20% | **54.81%** | ↑↑ catastrophic |
| VarR_seen | 0.872 | 0.925 | 0.931 | **0.698** | ↓ worst |
| VarR_unseen | 0.872 | 0.847 | 0.847 | **0.593** | ↓ worst |
| VarR_gap | — | 0.078 | 0.084 | **0.105** | ↔ similar |
| ρ_sp | 0.857 | 0.639 | 0.678 | **0.524** | ↓ (good: further below real 0.668) |
| kNN@10 | 0.611 | 0.463 | 0.521 | **0.368** | ↓ worst |
| γ1_mean | N/A | ~1.0 | ~1.0 | **0.160** | ↓ collapsed |
| β1_norm | N/A | small | small | **2.101** | ↑ dominant |

---

## 4. Root Cause: γ Collapse

### 4.1 Mechanism

The FiLM generator applies modulation at each hidden layer:

```
h1' = γ1(s_c) ⊙ h1 + β1(s_c)
h2  = LeakyReLU(W2 · h1')
h2' = γ2(s_c) ⊙ h2 + β2(s_c)
ê   = L2_norm(W3 · h2')
```

FiLM's theoretical guarantee for zero-shot variance transfer rests on the Jacobian:

```
J_G = W3 · D2(z, s_c) · diag(γ2) · W2 · D1(z) · diag(γ1) · W1
```

where `D1 = diag(𝟙[W1·z > 0])` depends only on z (class-agnostic). With γ1 ≈ 0.16:

```
h1' = 0.16·h1 + β1(s_c)   ≈   β1(s_c)
```

The noise contribution h1 = LeakyReLU(W1·z) is attenuated to 16% of its magnitude. h2 then becomes:

```
h2 = LeakyReLU(W2·h1') ≈ LeakyReLU(W2·β1(s_c))
```

`D2 = diag(𝟙[h1' > 0])` — the activation pattern of the second layer — is now almost entirely determined by s_c via β1(s_c), not by z. FiLM's key property (class-agnostic D1) is preserved in principle, but D2 has become prototype-specific again. The Jacobian degenerates to:

```
J_G ≈ W3 · D2(s_c) · diag(γ2) · W2 · 0.16·D1(z)·diag(γ1)·W1
```

The noise z enters through a 0.16× attenuated D1, while D2 is prototype-driven. Within-class variance:

```
Var[G(z, s_c)] ≈ (0.16)² · W3·D2(s_c)·diag(γ2)·W2 · Cov[D1(z)·h1] · [same]ᵀ
```

The (0.16)² = 0.026 factor explains VarR_seen = 0.698 (down from ~0.93 in Exp04a). The generator produces outputs that cluster tightly around a prototype-dependent centroid f(β1(s_c)), with only 2.6% of the noise-driven variance reaching the output.

### 4.2 Why β1 Became Dominant

The FiLM MLP1 (s_c(64) → Linear(64,256) → ReLU → Linear(256,1024) → [γ1|β1]) has 279,808 parameters — sufficient capacity to encode the centroid of each of the 1654 seen classes in β1 alone. Once the network discovered that β1(s_c) can place synthetic embeddings near real class centroids, the Wasserstein loss reinforced this: the critic saw accurate centroid placement and assigned high D(fake) scores, providing a stable gradient that encouraged further specialisation of β1.

At this point, γ1 becomes redundant for the Wasserstein objective. The optimiser has no incentive to maintain large γ1, and it drifts toward zero as a gradient pressure to reduce the scale of unnecessary activations.

### 4.3 Why L_var Failed to Prevent Collapse

`L_var = -mean_c Var_within[G(z, s_c)]`. At the degenerate equilibrium:

```
G(z, s_c) ≈ f(β1(s_c))   [nearly deterministic]
Var_within[G(z, s_c)] ≈ 0
L_var ≈ 0,   ∇L_var ≈ 0
```

L_var provides **zero gradient at the degenerate fixed point**. Once γ1 becomes small, within-class variance collapses, and L_var has no signal to recover from. The collapse is self-reinforcing: small γ1 → low variance → zero L_var gradient → γ1 stays small.

This is the fundamental limitation of L_var as a variance regulariser: it requires non-zero variance to provide non-zero gradient. A variance-killing local minimum is invisible to L_var. This contrasts with a variance-constraining loss (like a lower-bound penalty on variance), which would remain active precisely when variance is too low.

### 4.4 Why More Capacity Made It Worse

In Exp04 (film_hidden=128), the FiLM MLP has 74,368 parameters per layer. With film_hidden=256, it has 279,808 — 3.76× more capacity. The larger MLP can encode the 1654-class centroid structure more accurately in β1, making the β-dominant solution more attractive. Exp04's 10K-step run may have been too short to fully converge to this degenerate minimum; Exp05's 50K steps allowed complete convergence.

**More capacity + more training time allowed the network to find and settle into a deeper β-dominant local minimum.**

---

## 5. Secondary Effects

### 5.1 Routing Rate: 54.81% (Catastrophic)

Near-deterministic generation produces 20 near-identical synthetic embeddings per unseen class. After balancing to ~8 per class, the LogReg classifier sees 200 tight point-clouds as its only unseen-class training signal. Tight decision boundaries form around each point-cloud.

Real unseen test EEG (which has genuine within-class spread) partially overlaps these tight regions. Because the synthetic centroids are well-positioned by the high-capacity β1(s_c) function, 54.81% of unseen test samples fall within an unseen decision boundary — but the specific class assignment is wrong (AccU = 5.36%, vs random chance of ~0.5% for 200 classes).

High routing damages AccS (3.33%, down from 4.11% baseline) because some seen-class test embeddings also land near unseen centroids and are misclassified. The routing–H-mean trade-off collapses: routing up 2.7× but H-mean is the worst observed.

### 5.2 Critic Near-Saturation: c_loss = -0.025

`L_D = E[D(fake)] - E[D(real)] + λ·GP ≈ -0.025`. E[D(fake)] ≈ E[D(real)] — the critic assigns nearly equal scores to real and fake. The critic is fooled by near-deterministic outputs that accurately replicate real class means.

This exposes a fundamental limitation of the WGAN-GP discriminator for GZSL synthesis: the critic receives individual samples `[ê, s_c]` and has no mechanism to measure within-class diversity. A generator that produces a single point-mass per class satisfies the critic perfectly if that point-mass is inside the real distribution. The critic is blind to variance deficit.

### 5.3 ρ_sp = 0.524 (Project Minimum)

Structural decoupling continues to improve as generator capacity grows. ρ_sp = 0.524 is 0.144 below the real-data reference (0.668) and 0.115 below Exp04a (0.639). The β1(s_c) function, despite encoding class centroids well, does not preserve inter-class prototype distances — it encodes a distorted centroid geometry. This is expected: β1 is a learned projection from prototype space, not a distance-preserving embedding. FiLM's architectural decoupling is working correctly; only the variance collapse is the problem.

### 5.4 VarR_gap = 0.105 (Unchanged from Exp04)

Despite dramatically lower absolute VarR values, the seen-to-unseen transfer gap (0.105) is comparable to Exp04c (0.084). This confirms that FiLM's zero-shot transfer property holds: the variance collapse is uniform across seen and unseen classes. The problem is not that variance fails to transfer — it's that there is no variance to transfer.

---

## 6. Failure Mode Taxonomy

| Failure | Experiment | Mechanism | VarR_seen | VarR_unseen |
|---|---|---|---|---|
| Prototype entanglement | Baseline (concat) | J_G entangles z and s_c; seen-specific tricks | ~0.872 | 0.872 |
| Seen-specific variance | Exp03 (L_var+concat) | Generator learns variance for seen prototypes only | 0.973 | 0.875 |
| **γ collapse (β-dominance)** | **Exp05** | **FiLM MLP suppresses noise path; z irrelevant** | **0.698** | **0.593** |

All three failure modes produce low VarR_unseen. The mechanisms are distinct:
- Exp03: variance behaviour is seen-specific (doesn't transfer)
- Exp05: no variance behaviour exists (collapse to near-deterministic)

---

## 7. Path Forward: Exp06

The fix is surgical: **prevent γ from collapsing**. Two mechanisms address the root cause directly.

### 7.1 Residual FiLM Parameterisation (Primary Fix)

Replace direct γ learning with residual Δγ:

```python
def _film(self, h, mlp, s_c):
    params = mlp(s_c)
    delta_gamma, beta = params.chunk(2, dim=-1)
    return (1.0 + delta_gamma) * h + beta   # γ = 1 + Δγ
```

**Initialisation**: `bias = 0` throughout (Δγ starts at 0, β starts at 0). γ starts at exactly 1.

**Why this helps**: To achieve γ ≈ 0.16, the network must learn Δγ ≈ -0.84. In the direct-γ parameterisation, γ can drift from 1.0 to 0.16 via small gradient steps with no restoring force. In the residual parameterisation, γ = 1 + Δγ means that γ is always expressed as a deviation from identity. The optimisation landscape around Δγ=0 (γ=1) is a stable fixed point: small perturbations of Δγ have linear effect on the output h', while the Wasserstein loss provides strong signal around γ≈1. Pushing Δγ to -0.84 requires a large cumulative gradient drive that competes against the stable noise-path gradient from z.

This is the standard approach in conditional normalisation literature (Perez et al. 2018, Park et al. 2019 SPADE). The residual form makes identity-modulation the natural solution rather than the initial condition.

### 7.2 γ Floor Regularisation (Secondary Fix)

Add an explicit penalty to L_G when γ drops below γ_min:

```python
def compute_lgamma(generator):
    """Penalise gamma < gamma_min across both FiLM layers."""
    total = 0.0
    for mlp in (generator.film1_mlp, generator.film2_mlp):
        params = mlp(s_c_sample)   # s_c_sample: batch of prototypes
        gamma, _ = params.chunk(2, dim=-1)
        total += F.relu(gamma_min - gamma.abs()).pow(2).mean()
    return total

# In training loop:
l_gamma = compute_lgamma(generator)
g_loss  = l_wass + alpha * l_var + lambda_gamma * l_gamma
```

With `gamma_min = 0.5`, `lambda_gamma = 0.1`. The penalty activates whenever |γ| < 0.5 and provides a direct gradient signal `∂L/∂γ = -2·λ_γ·(γ_min - |γ|)` pushing γ back above the floor. This is an active restoring gradient — it works precisely when variance is near zero, unlike L_var which vanishes at the degenerate state.

### 7.3 Reduce film_hidden: 256 → 128 (Capacity Constraint)

With film_hidden=128, the FiLM MLP has 74,368 parameters (vs 279,808 with film_hidden=256). This limits β1's capacity to encode the full class centroid structure, making the β-dominant solution less effective and preserving the generator's dependence on γ·h. Keep hidden_dim=512 (main path capacity) and n_steps=50K (training budget).

### 7.4 Exp06 Configuration

```python
WGAN_CONFIG = {
    'hidden_dim':    512,     # keep — generator main path capacity
    'film_hidden':   128,     # reduce — limit β encoding capacity
    'n_steps':       50000,   # keep — 50K budget is correct
    'alpha':         1.0,     # keep — L_var at sweep optimum
    'gamma_min':     0.5,     # new — floor regularisation threshold
    'lambda_gamma':  0.1,     # new — floor regularisation weight
    'film_residual': True,    # new — γ = 1 + Δγ parameterisation
}

# L_G = L_wasserstein + 1.0·L_var + 0.1·L_gamma_floor
```

Key monitoring additions:
- `train/delta_gamma1_mean` and `train/delta_gamma2_mean` — track Δγ evolution
- `train/L_gamma` — floor regularisation magnitude
- `train/gamma1_min` and `train/gamma1_max` — per-batch γ distribution (not just mean)

---

## 8. Artefacts

| File | Description |
|---|---|
| `results/analysis.md` | This document |
| WandB run `wyv69gqm` | Full training curves (γ1, β1_norm, VarR_seen, L_G, L_D, grad_cos_sim) |
| `results/generator_highcap_step*.pt` | Intermediate checkpoints (10K, 20K, 30K, 40K steps) |
| `results/generator_highcap_final.pt` | Final model (50K steps) |
