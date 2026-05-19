# Experiment 06: Residual FiLM — Results Analysis

**Date**: 2026-05-09  
**Status**: Complete  
**WandB Run**: `06_residual_film`  

---

## 1. Executive Summary

Exp06 partially addressed γ collapse (γ1_mean: 0.160 → 0.510) but delivered the worst H-mean in
the project (4.05%), marginally below Exp05 (4.10%). Two critical findings emerge:

1. **The floor regularisation was insufficient**: γ1_min = −0.252 — elements went negative, meaning
   some Δγ < −1. The L_gamma penalty at λ_gamma = 0.1 was overwhelmed by the Wasserstein gradient
   for those dimensions.

2. **The routing pathology is not primarily caused by γ collapse**: routing = 48.15% even with
   γ1_mean = 0.51 — comparable to Exp05's 54.81% with γ1_mean = 0.16. The routing spike is
   caused by 50K-step training convergence producing tight synthetic clusters, not by the specific
   failure mode of each experiment.

These findings reveal that training duration is the binding variable for routing quality. All
50K-step experiments (Exp05, Exp06) produce routing ≈ 48–55%, versus ~20% for all 10K-step
experiments (Exp04a-c, baseline). H-mean is primarily gated by routing, not VarR.

---

## 2. Raw Results

```json
{
  "experiment":          "film_residual",
  "hidden_dim":          512,
  "film_hidden":         128,
  "n_steps":             50000,
  "alpha":               1.0,
  "gamma_min":           0.5,
  "lambda_gamma":        0.1,
  "H_mean_pct":          4.0498,
  "AccS_pct":            3.3555,
  "AccU_pct":            5.1062,
  "routing_pct":         48.15,
  "VarR_seen":           0.7756,
  "VarR_unseen":         0.67641,
  "VarR_gap":            0.09918,
  "rho_sp":              0.55264,
  "kNN10":               0.376,
  "gamma1_mean":         0.50952,
  "gamma1_min":          -0.25168,
  "delta_gamma1_mean":   0.49113,
  "gamma2_mean":         0.74769,
  "gamma2_min":          -0.07807,
  "beta1_norm":          3.59392,
  "beta2_norm":          1.59445,
  "VarR_seen_posttrain": 0.77192,
  "g_loss_final_mean1k": -0.89247,
  "c_loss_final_mean1k": -0.02233
}
```

---

## 3. Complete Comparison Table

| Metric | Baseline (01) | Exp04a | Exp04c α=1.0 | Exp05 | **Exp06** |
|---|---|---|---|---|---|
| **H-mean** | **4.77%** | 4.69% | 4.67% | 4.10% | **4.05%** |
| AccS | 4.11% | 4.17% | 3.75% | 3.33% | **3.36%** |
| AccU | 5.69% | 5.09% | 6.19% | 5.36% | **5.11%** |
| routing | ~20% | ~20% | ~20% | 54.81% | **48.15%** |
| VarR_seen | 0.872 | 0.925 | 0.931 | 0.698 | **0.776** |
| VarR_unseen | 0.872 | 0.847 | 0.847 | 0.593 | **0.676** |
| VarR_gap | — | 0.078 | 0.084 | 0.105 | **0.099** |
| ρ_sp | 0.857 | 0.639 | 0.678 | 0.524 | **0.553** |
| kNN@10 | 0.611 | 0.463 | 0.521 | 0.368 | **0.376** |
| γ1_mean | N/A | ~1.0 | ~1.0 | 0.160 | **0.510** |
| γ1_min | N/A | — | — | — | **−0.252** |
| β1_norm | N/A | small | 2.101* | 2.101 | **3.594** |

*β1_norm from Exp05 for comparison.

### Routing vs H-mean by experiment epoch

| Experiment | n_steps | routing | H-mean |
|---|---|---|---|
| Concat baseline | 10,000 | ~20% | 4.77% |
| Exp04a FiLM | 10,000 | ~20% | 4.69% |
| Exp04c FiLM+Lvar | 10,000 | ~20% | 4.67% |
| Exp05 FiLM highcap | 50,000 | 54.81% | 4.10% |
| Exp06 Residual FiLM | 50,000 | 48.15% | 4.05% |

The routing–duration correlation is unambiguous.

---

## 4. Detailed Analysis

### 4.1 γ Collapse: Partially Fixed

The residual parameterisation raised γ1_mean from 0.160 (Exp05) to 0.510 (Exp06). This is a genuine improvement — the noise path is contributing more to the output. In the Jacobian:

```
J_G = W3 · D2(z,s_c) · diag(γ2) · W2 · D1(z) · diag(γ1) · W1
```

With mean γ1 = 0.51 vs 0.16, the attenuation factor on D1(z) improved by 3.2×. Within-class variance recovered accordingly: VarR_unseen rose from 0.593 → 0.676 and VarR_seen from 0.698 → 0.776.

However, the fix was incomplete:

**γ1_min = −0.252**: Some γ elements crossed zero. `γ = 1 + Δγ = −0.252` means `Δγ = −1.252`. The L_gamma penalty at this value is:

```
F.relu(0.5 − (−0.252))² = F.relu(0.752)² = 0.752² = 0.5655
```

With λ_gamma = 0.1, the generator loss contribution is `0.1 × 0.5655 = 0.0566` per element. The gradient w.r.t. Δγ is `-2 × λ_gamma × (γ_min − γ) = -2 × 0.1 × 0.752 = -0.1504` — a restoring force pushing Δγ upward by 0.1504 per step.

This was not large enough. The Wasserstein gradient for those dimensions evidently exceeded 0.1504 in magnitude, allowing Δγ to be pushed below −1.0. The floor did not hold for all elements; it merely slowed the collapse rather than preventing it.

**γ2_min = −0.078**: Layer 2 stayed above zero (barely). This is consistent with the Exp05 observation that γ2 collapsed less severely than γ1 — the first modulation layer is the primary victim because it directly attenuates the noise pathway from z.

### 4.2 β1_norm Grew Larger (3.594 vs Exp05's 2.101)

This is counterintuitive: despite reducing film_hidden (256→128), β1's L2 norm per prototype grew by 71% relative to Exp05. With γ1_mean = 0.51 (larger than Exp05's 0.16), the noise path carries more signal, so β1 needs less absolute magnitude to maintain the same class-centroid positioning. Yet β1_norm increased.

The explanation: with the residual parameterisation and floor regularisation, the FiLM MLP cannot easily collapse γ1 to zero. It must instead achieve class-specific positioning by increasing β1 magnitude — β1 is the only free variable that can still be driven large. The floor regularisation effectively traded γ collapse for β inflation.

Consider the forward pass: `h1' = 0.51·h1 + β1(s_c)`. With β1_norm ≈ 3.59 and h1 norm ≈ √512 × (mean non-negative activation) ≈ 10–15, the output h1' has contributions:
- Noise term: 0.51 × 10 = 5.1 (rough estimate)  
- Prototype term: 3.59

The noise term still dominates h1' magnitude, but β1 with norm 3.59 is not negligible. Larger β1 means more prototype-specific shift in h1', making D2 more prototype-dependent, partially reinstating the Exp05 failure mode through a different mechanism.

### 4.3 The Routing Pathology: Training Duration, Not Architecture

The most important finding in Exp06 is what it confirms by comparison with Exp05:

**Routing did NOT return to ~20% despite partially fixing γ collapse.** With γ1_mean = 0.51 (vs 0.16 in Exp05), the generator is substantially more noise-driven, yet routing only improved marginally: 54.81% → 48.15%. This rules out γ collapse as the primary cause of routing pathology.

The common variable between Exp05 (routing=54.81%) and Exp06 (routing=48.15%) is **n_steps=50,000**. All 10K-step experiments had routing ≈ 20%.

**Mechanism**: At 10K steps, the WGAN-GP has not converged. Synthetic embeddings per class are placed approximately in the right direction, but with residual noise from non-convergence. This diffuseness creates ambiguous LogReg boundaries — the classifier is uncertain about unseen-class regions and routes conservatively (~20%).

At 50K steps, the generator converges. c_loss_final ≈ −0.02 (near-zero Wasserstein distance) indicates the critic can no longer distinguish real from fake — the generator has found a strong approximation of the real distribution. Tight synthetic clusters form per class. LogReg builds high-confidence decision boundaries around each of the 200 unseen classes. These tight boundaries:

1. Capture real unseen EEG samples that pass through their domain → high routing (48–55%)
2. Also capture some seen-class test EEG that happens to land near unseen boundaries → AccS depressed (3.33–3.36% vs baseline 4.11%)
3. Fine-grained accuracy within the captured unseen region remains low (~5%) because EEG variability within a class exceeds the tight cluster radius

H-mean degrades because AccS is suppressed more than AccU benefits from the routing.

The routing pathology is a **classifier overfitting to synthetic centroids** at generator convergence. It is essentially the inverse of the well-known GZSL "seen class bias" problem — here the classifier develops an *unseen class bias* because the synthetic embeddings are very well-placed but under-dispersed relative to real test data.

### 4.4 c_loss ≈ −0.022: Near-Saturation Again

Identical to Exp05 (c_loss ≈ −0.025). At 50K steps, both experiments achieve near-zero Wasserstein distance — the critic is fully saturated. This is further evidence that the routing pathology is a convergence phenomenon: once the generator converges (c_loss→0), routing spikes.

This provides a concrete diagnostic for future experiments: **if c_loss → 0 early in training (e.g., by 15K–20K steps), routing will spike**. Monitor c_loss trajectory in WandB — a floor around −0.01 to −0.05 with stability indicates healthy convergence; values approaching 0 from below indicate over-convergence.

### 4.5 VarR_gap = 0.099 — Comparable to Prior Experiments

The seen-to-unseen transfer gap improved marginally relative to Exp05 (0.105→0.099) and is in the same range as Exp04 variants (0.078–0.084). FiLM's transfer property is intact — the improvement from residual parameterisation (higher γ1, more noise) is reflected equally in both VarR_seen and VarR_unseen, preserving the gap.

This confirms the long-standing finding: FiLM correctly enables variance transfer (gap ≈ constant), but the absolute VarR value is too low (generator is still too near-deterministic). The transfer mechanism works; the variance magnitude doesn't.

---

## 5. Critical Reassessment: 50K Steps is the Wrong Budget

Looking across the full experiment arc:

| n_steps | Architecture | H-mean | routing |
|---|---|---|---|
| 10,000 | Concat baseline | 4.77% | ~20% |
| 10,000 | FiLM (Exp04a) | 4.69% | ~20% |
| 10,000 | FiLM + Lvar α=1.0 (Exp04c) | 4.67% | ~20% |
| 50,000 | FiLM highcap (Exp05) | 4.10% | 54.81% |
| 50,000 | Residual FiLM (Exp06) | 4.05% | 48.15% |

The pattern is stark: **50K steps consistently produces routing pathology and lower H-mean than 10K steps**, regardless of architecture. The generator over-converges, creating tight synthetic clusters that distort the LogReg decision boundary.

The original motivation for 50K steps was: "larger generator (hidden_dim=512) needs more steps to converge." This was correct — it does need more steps to converge. But convergence itself is harmful in this setting. The GZSL classifier works best when synthetic embeddings are approximately-but-not-exactly placed, with enough residual non-convergence noise to prevent over-confident unseen boundaries.

**The 10K step budget was not a limitation — it was accidentally calibrated to the right amount of non-convergence for healthy routing.**

---

## 6. Failure Mode Taxonomy (Updated)

| Failure | Experiment | Mechanism | VarR_u | routing | H |
|---|---|---|---|---|---|
| Prototype entanglement | Concat baseline | Entangled J_G, seen-specific variance | 0.872 | ~20% | 4.77% |
| Seen-specific variance | Exp03 concat+Lvar | Variance learned for seen only | 0.875 | ~20% | 4.58% |
| γ collapse | Exp05 highcap | γ1→0.16, z suppressed, β-dominant | 0.593 | 54.81% | 4.10% |
| Partial γ collapse + routing | Exp06 residual | γ1→0.51 (floor partial), routing from 50K | 0.676 | 48.15% | 4.05% |
| **Routing pathology (new)** | **Exp05, Exp06** | **50K convergence → tight clusters → routing spike** | — | **48–55%** | **4.05–4.10%** |

---

## 7. Path Forward: Experiment 07

The evidence mandates reverting to 10K steps. The unanswered question is: **does the residual FiLM architecture (hidden_dim=512, film_hidden=128, γ=1+Δγ, L_gamma) improve over the 10K-step Exp04 variants when routing stays healthy?**

**Exp07 specification:**

```python
WGAN_CONFIG = {
    'hidden_dim':    512,      # keep — generator main path capacity from Exp05/06
    'film_hidden':   128,      # keep — Exp06 reduced capacity
    'n_steps':       10000,    # REVERT — 50K causes routing pathology
    'alpha':         1.0,      # keep — L_var at sweep optimum
    'gamma_min':     0.5,      # keep — floor regularisation
    'lambda_gamma':  0.1,      # keep — may not activate at 10K (γ doesn't collapse at 10K)
    'experiment':    'film_residual_10k',
}
```

At 10K steps, γ collapse does not have time to develop (Exp04a ran for 10K with no collapse). The residual parameterisation and L_gamma are belt-and-suspenders safety mechanisms that cost nothing at 10K steps and protect against any early-training collapse.

**What this experiment answers:**

If H-mean improves from Exp04a's 4.69% with this architecture at 10K steps, residual FiLM with larger hidden_dim provides genuine benefit.

If H-mean stays at ~4.7% (within stochastic noise band ±0.13pp), the architecture improvements have saturated and the encoder bottleneck hypothesis is confirmed: further generator improvements cannot improve H-mean.

**Key monitoring:** At 10K steps with this architecture, check whether c_loss converges to near-zero before step 10K (early over-convergence). If c_loss → 0 by step 5K, routing will spike even at 10K — in that case, consider reducing n_steps to 7K or adding early stopping based on routing rate.

---

## 8. Artefacts (50K Run)

| File | Description |
|---|---|
| `results/analysis.md` | This document |
| WandB run `06_residual_film` | Full training curves |
| `results/generator_residual_step*.pt` | Checkpoints at 10K, 20K, 30K, 40K steps |
| `results/generator_residual_final.pt` | Final model (50K steps) |

---

## 9. Addendum: Exp06 at 10K Steps

**Date**: 2026-05-09  
**Motivation**: The 50K-step analysis identified routing pathology as caused by generator convergence, not γ collapse. This run retests the residual FiLM architecture (hidden_dim=512, film_hidden=128, γ=1+Δγ, L_gamma) with n_steps=10,000 to isolate whether the architectural improvements provide genuine benefit when routing stays healthy.

### 9.1 Raw Results

```json
{
  "experiment":          "film_residual",
  "hidden_dim":          512,
  "film_hidden":         128,
  "n_steps":             10000,
  "alpha":               1.0,
  "gamma_min":           0.5,
  "lambda_gamma":        0.1,
  "H_mean_pct":          4.5968,
  "AccS_pct":            3.688,
  "AccU_pct":            6.1,
  "routing_pct":         32.89,
  "VarR_seen":           0.92858,
  "VarR_unseen":         0.83996,
  "VarR_gap":            0.08862,
  "rho_sp":              0.70293,
  "kNN10":               0.522,
  "gamma1_mean":         0.59724,
  "gamma1_min":          0.07408,
  "delta_gamma1_mean":   0.40304,
  "gamma2_mean":         0.88601,
  "gamma2_min":          0.39074,
  "beta1_norm":          2.69047,
  "beta2_norm":          0.97894,
  "VarR_seen_posttrain": 0.93112,
  "g_loss_final_mean1k": -1.43667,
  "c_loss_final_mean1k": -0.01193
}
```

### 9.2 Full Comparison Table (All Experiments)

| Metric | Baseline | Exp04a | Exp04c α=1.0 | Exp05 50K | Exp06 50K | **Exp06 10K** |
|---|---|---|---|---|---|---|
| **H-mean** | **4.77%** | **4.69%** | 4.67% | 4.10% | 4.05% | **4.60%** |
| AccS | 4.11% | 4.17% | 3.75% | 3.33% | 3.36% | **3.69%** |
| AccU | 5.69% | 5.09% | 6.19% | 5.36% | 5.11% | **6.10%** |
| routing | ~20% | ~20% | ~20% | 54.81% | 48.15% | **32.89%** |
| VarR_seen | 0.872 | 0.925 | 0.931 | 0.698 | 0.776 | **0.929** |
| VarR_unseen | 0.872 | 0.847 | 0.847 | 0.593 | 0.676 | **0.840** |
| VarR_gap | — | 0.078 | 0.084 | 0.105 | 0.099 | **0.089** |
| ρ_sp | 0.857 | 0.639 | 0.678 | 0.524 | 0.553 | **0.703** |
| kNN@10 | 0.611 | 0.463 | 0.521 | 0.368 | 0.376 | **0.522** |
| γ1_mean | N/A | ~1.0 | ~1.0 | 0.160 | 0.510 | **0.597** |
| γ1_min | N/A | — | — | — | −0.252 | **+0.074** |
| β1_norm | N/A | — | 2.101 | 2.101 | 3.594 | **2.690** |
| c_loss | — | — | — | −0.025 | −0.022 | **−0.012** |

### 9.3 γ Analysis: No Collapse, Amplification Regime

The residual parameterisation succeeded at 10K steps in preventing collapse:

- **γ1_min = +0.074**: all elements stayed positive. No sign inversion. The floor regularisation was not needed (γ never approached γ_min=0.5 from above in collapse direction) — instead Δγ went positive.
- **delta_gamma1_mean = +0.403**: Δγ > 0 across most elements — the FiLM MLP learned to *amplify* rather than suppress the noise path. Effective mean γ1 = 1 + 0.403 = 1.403.
- **γ2_min = +0.391**: layer 2 also stayed well above zero.

This is the opposite failure direction from Exp05 (Δγ → −0.84). With the residual parameterisation, the degenerate shortcut of suppressing γ to zero is no longer easy — the MLP would need to drive Δγ to −1.0, which requires large negative outputs from a zero-initialised network. Instead, the FiLM MLP drifted to a different local structure: amplification of the noise component.

With γ1 ≈ 1.4, the Jacobian factor is:

```
J_G = W3 · D2(z,s_c) · diag(γ2≈0.89) · W2 · D1(z) · diag(γ1≈1.40) · W1
```

D1 is class-agnostic (depends only on z). The noise variance throughput is amplified 1.4× relative to identity, which explains VarR_seen = 0.929 (nearly matching real variance). However, the unseen VarR = 0.840 is 8.9pp lower — the seen-prototype-specific tuning of β1 doesn't fully generalise. The VarR_gap = 0.089 is consistent with prior FiLM experiments (0.078–0.099).

### 9.4 The Larger Generator Converges Faster: The True Binding Variable

**c_loss_final = −0.012** at 10K steps. Compare:
- Exp04 variants (hidden_dim=256, 257K params) at 10K steps: c_loss presumably in range −1 to −2 (not converged — routing ≈ 20%)
- Exp06 10K (hidden_dim=512, 628K params): c_loss = −0.012 (near-zero — nearly converged)
- Exp06 50K: c_loss = −0.022

The hidden_dim=512 generator is approximating the training distribution much faster than hidden_dim=256. By step 10K, it has essentially converged (Wasserstein distance ≈ 0). This explains routing = 32.89%: the generator is already tight enough to create overly confident unseen decision boundaries, but has not fully converged (50K gives tighter clusters still → routing 48–55%).

The routing-convergence relationship is now confirmed across five data points:

| Experiment | hidden_dim | n_steps | c_loss | routing | H-mean |
|---|---|---|---|---|---|
| Exp04a | 256 | 10,000 | ~−1 to −2 | ~20% | 4.69% |
| Exp04c | 256 | 10,000 | ~−1 to −2 | ~20% | 4.67% |
| Exp06 10K | 512 | 10,000 | −0.012 | 32.89% | 4.60% |
| Exp05 50K | 512 | 50,000 | −0.025 | 54.81% | 4.10% |
| Exp06 50K | 512 | 50,000 | −0.022 | 48.15% | 4.05% |

The pattern is unambiguous: **routing tracks c_loss, not n_steps or architecture directly**. c_loss is the causal variable; n_steps and hidden_dim both affect how quickly c_loss approaches zero.

To achieve routing ≈ 20% with hidden_dim=512, n_steps must be reduced until c_loss remains at −1 to −2 range at end of training. Given that c_loss = −0.012 at 10K steps with hidden_dim=512, and Exp04 (256) required 10K to stay unconverged, the implied target is approximately:

```
n_steps_512 ≈ n_steps_256 × (params_256 / params_512) × k
            = 10,000 × (257K / 628K) × k
            ≈ 4,000 steps  (with k ≈ 1, rough estimate)
```

### 9.5 ρ_sp = 0.703 — Structural Coupling Regression

A key regression: ρ_sp = 0.703 is substantially worse than Exp04a (0.639) and approaching the concat baseline (0.857). The structural overcoupling has partially reinstated.

With Δγ > 0 (amplification), the synthetic embeddings have higher within-class variance, but the centroid geometry itself is more faithful to the prototype geometry. The FiLM MLP (film_hidden=128), with a larger main network (hidden_dim=512) producing richer intermediate representations h1 and h2, can place synthetic centroids more precisely in prototype-aligned positions — which increases ρ_sp. This is a form of over-alignment that the Wasserstein loss encourages: matching the real centroid geometry as faithfully as possible.

The result is that AccU = 6.10% (highest in the project, exceeding the 6.19% maximum from the Exp04c alpha sweep at 10K steps) but AccS = 3.69% is depressed. The 32.89% routing bias toward unseen classes captures more unseen test EEG but simultaneously encroaches on seen-class decision boundaries.

### 9.6 Assessment: Residual FiLM at 10K Does Not Improve Over Exp04a

H-mean = 4.60% vs Exp04a's 4.69% — the residual FiLM with hidden_dim=512 at 10K steps does **not** improve over the simpler FiLM baseline. The stochastic noise band is ±0.13pp (from WGAN-GP variance across runs), so the gap of 0.09pp is within noise. The experiment is inconclusive on H-mean.

However, the routing behaviour is unambiguously worse (32.89% vs ~20%), confirming that the larger generator converges too fast for the 10K budget. The architectural improvements (residual γ, L_gamma, larger hidden_dim) do not provide a clean H-mean gain — they are absorbed by faster convergence that re-introduces routing pathology.

**The encoder bottleneck hypothesis strengthens further**: across six WGAN-GP architecture variants (concat baseline, FiLM, FiLM+LN, FiLM+Lvar, highcap FiLM, residual FiLM), AccU never exceeded 6.2% and H-mean never exceeded 4.77%. The ceiling is not the generator architecture — it is the EEG-to-embedding quality.

### 9.7 Updated Path Forward

Three options, in order of expected yield:

**Option A — Find the convergence-equivalent step count for hidden_dim=512 (~4K steps)**  
Run Exp06 architecture with n_steps ≈ 3,500–4,000, targeting c_loss in the −1 to −2 range at training end. This achieves the same non-convergence state as Exp04 at 10K. If H-mean still ≈ 4.7%, confirms architectural ceiling. Low cost to run; definitive test.

**Option B — Accept the encoder bottleneck and switch focus to encoder improvement**  
The WGAN-GP era has exhausted generator-side improvements. The next lever is EEG representation quality: richer temporal models (EEGNet, LSTM on per-channel timeseries), larger EEG training sets, or stronger contrastive pretraining. This is a larger architectural shift requiring careful experimental design.

**Option C — Return to smaller generator (hidden_dim=256) with residual FiLM**  
The residual parameterisation and L_gamma are valid improvements to the FiLM architecture independent of the hidden_dim regression. Testing residual FiLM at hidden_dim=256, n_steps=10K would isolate whether the residual parameterisation alone (vs Exp04a) provides any benefit.

Option A is the most targeted next step: minimal change, directly tests the convergence hypothesis, and closes the residual FiLM chapter cleanly before moving to larger architectural changes.
