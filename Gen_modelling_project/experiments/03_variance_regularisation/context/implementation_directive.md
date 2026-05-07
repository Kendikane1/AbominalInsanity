# Implementation Directive — Variance Regularisation for WGAN-GP Generator

*Context file for Claude Code agents. Written by the project's mathematics tutor after completing a full derivation sequence with the researcher. This document contains the mathematical specification, implementation directives, and critical warnings for adding variance regularisation to the conditional WGAN-GP generator in a Generalised Zero-Shot Learning (GZSL) EEG decoding pipeline.*

*Date: 2026-04-11*

---

## 1. What This Document Is

This is a directive from the project's mathematical analysis, intended to guide implementation of a **single, well-defined modification**: adding a variance regularisation loss term to the generator objective of an existing conditional WGAN-GP. The modification is motivated by a diagnosed generator collapse pathology (synthetic embeddings are too tightly clustered around conditioning prototypes) and a failed post-hoc perturbation experiment that confirmed the fix must occur at training time.

**Read the full project context in `project_math_context.md` before proceeding.** This document assumes familiarity with the pipeline architecture, dataset structure, and diagnostic results described there.

---

## 2. The Problem Being Solved

### 2.1 Diagnosed Pathology

The generator produces synthetic embeddings for unseen classes that are **too tightly coupled to the conditioning prototype**. Evidence:

| Diagnostic | Value | Interpretation |
|---|---|---|
| VarR (variance ratio) | 0.872 | Synthetic per-dimension variance is 12.8% below real unseen variance |
| ρ(synth centroids, prototypes) | 0.857 | Synthetic centroids are highly correlated with prototype geometry |
| ρ(real centroids, prototypes) | 0.668 | Real brain centroids are much less correlated — more noisy |

The synthetic ellipsoid (per-class data cloud) is shrunken compared to the real one. The logistic regression classifier trains on tight clusters, then encounters more spread-out real brain embeddings at test time, causing misclassification.

### 2.2 Why Post-Hoc Perturbation Failed

Perturbing the conditioning prototype at synthesis time with isotropic noise:

```
s̃_c = L2_normalize(s_c + η · ξ),   ξ ~ N(0, I_64)
```

was swept across η ∈ {0.00, 0.02, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25}. **H-mean strictly decreased with increasing η.** The reason:

- Output perturbation δê ≈ J_G · (η · ξ) lies in the **column space of J_G** (the generator Jacobian)
- The columns of J_G encode the generator's learned sensitivity directions, which have no alignment with real brain variability directions
- At η=0.10, VarR=1.002 (correct variance *amount*) but H=2.56% (much worse) — variance in the wrong directions is worse than too little variance in the right directions

**Conclusion:** The fix must modify the generator's weights during training, reshaping J_G itself, not perturbing inputs through a fixed J_G.

### 2.3 Why Training-Time Modification Works

Adding a loss term L_var to L_G changes ∂L_G/∂θ_G. Gradient descent adjusts W₁, W₂, W₃, which changes the Jacobian:

```
J_G = (I − êêᵀ)/||u||₂ · W₃ · D₂ · W₂ · D₁ · W₁⁽ˢ⁾
```

Changed weights → changed J_G → changed column space → the generator's output distribution is reshaped to accommodate the variance target. This operates on the cage itself, not within it.

---

## 3. Mathematical Specification of the Modification

### 3.1 The Existing Generator Loss (DO NOT MODIFY)

```python
# Existing generator loss — Wasserstein term
L_wasserstein = -torch.mean(critic(G(z, s_c), s_c))
```

This must remain intact. The variance regularisation is **added** to it, not substituted.

### 3.2 The New Loss Term — Full Definition

**Precomputation (before training loop, computed once):**

For each seen class c ∈ {1, ..., 1654}, compute per-dimension variance of the real brain embeddings in the shared embedding space (after passing through the trained encoder f_b):

```
For each class c:
    e_real_c = { f_b(x_i) : x_i belongs to class c, from TRAINING set only }
    μ_c = mean(e_real_c, dim=0)                        # R^64
    var_c = mean((e_real_c - μ_c)² , dim=0)            # R^64, per-dimension variance

Target variance (averaged across all seen classes):
    var_target = mean(var_c for all seen c, dim=0)      # R^64
```

**CRITICAL:** Use the **trained, frozen encoder** to compute these. The encoder must be the same one used during WGAN-GP training. Do not recompute during training — `var_target` is a fixed tensor.

**CRITICAL:** Use only the **training split** of seen data (13,232 samples), NOT the test split (3,308 samples). The test split must remain untouched for evaluation.

**Per-batch computation (inside training loop, generator update step):**

```
For each class c in the current mini-batch:
    Generate K samples: ê_c = {G(z_k, s_c) for k=1..K}    # K vectors on S^63
    
    μ̂_c = mean(ê_c, dim=0)                                 # R^64
    
    var_synth_c = mean((ê_c - μ̂_c)², dim=0)                # R^64

L_var = mean over classes c [ sum over 64 dims ( (var_synth_c - var_target)² ) ]
```

Or equivalently in PyTorch-like pseudocode:

```python
def compute_L_var(fake_embeddings, class_labels, var_target):
    """
    fake_embeddings: (batch_size, 64) — generated embeddings on S^63
    class_labels:    (batch_size,)    — class index for each sample
    var_target:      (64,)           — precomputed target variance vector
    
    Returns: scalar loss
    """
    unique_classes = torch.unique(class_labels)
    loss = 0.0
    for c in unique_classes:
        mask = (class_labels == c)
        samples_c = fake_embeddings[mask]          # (K, 64)
        mu_c = samples_c.mean(dim=0)               # (64,)
        var_c = ((samples_c - mu_c) ** 2).mean(dim=0)  # (64,)
        loss += ((var_c - var_target) ** 2).sum()
    loss /= len(unique_classes)
    return loss
```

### 3.3 The Modified Generator Objective

```python
L_G = L_wasserstein + alpha * L_var
```

where `alpha` is a hyperparameter controlling the balance. See Section 5 for tuning guidance.

### 3.4 Gradient of L_var (For Verification)

The analytical gradient of L_var with respect to a single generated sample ê_{c,k} is:

```
∂L_var/∂ê_{c,k} = (4/K) · Δ_c ⊙ (ê_{c,k} − μ̂_c)

where:
    Δ_c = var_synth_c - var_target    (R^64, per-dimension variance deficit)
    ⊙ = element-wise multiplication
    K = number of samples for class c in the batch
```

**Geometric interpretation:** When variance is below target (Δ_c < 0, our current situation), this is a **repulsion force from the class centroid**. Each sample is pushed away from the mean, proportional to how far it already is from the mean and how large the variance deficit is. When variance exceeds the target, the force reverses to attraction.

**You should not need to implement this gradient manually** — PyTorch autograd will compute it. But you can use it to verify correctness: after implementing L_var, compute the gradient via autograd and compare against this analytical formula numerically on a small test case. They should match to floating-point precision.

---

## 4. Implementation Directives

### 4.1 Where to Modify

The modification touches **only** the generator training step. Do not modify:

- The critic architecture or loss
- The gradient penalty computation
- The encoder (f_b) or image projector (g_v)
- The classifier
- The data loading or preprocessing
- The synthesis procedure (how fake embeddings are generated for classifier training after WGAN-GP training is complete)

### 4.2 Training Loop Structure

The existing training loop alternates n_critic=5 critic updates per 1 generator update. The generator update is the only place that changes:

```python
# EXISTING (do not modify):
for step in range(num_steps):
    # --- Critic update (5 times) ---
    for _ in range(n_critic):
        z = torch.randn(batch_size, 100)
        s_c = sample_prototypes(batch_size)  # conditioning prototypes
        
        fake = G(z, s_c)
        real = sample_real_embeddings(batch_size)
        
        L_D = critic(fake.detach(), s_c).mean() - critic(real, s_c).mean() + lambda_gp * GP
        
        optimizer_D.zero_grad()
        L_D.backward()
        optimizer_D.step()
    
    # --- Generator update (1 time) --- THIS IS WHERE L_var IS ADDED
    z = torch.randn(batch_size, 100)
    s_c, class_labels = sample_prototypes_with_labels(batch_size)
    
    fake = G(z, s_c)
    
    L_wasserstein = -critic(fake, s_c).mean()
    L_var = compute_L_var(fake, class_labels, var_target)  # NEW
    
    L_G = L_wasserstein + alpha * L_var                     # MODIFIED
    
    optimizer_G.zero_grad()
    L_G.backward()
    optimizer_G.step()
```

### 4.3 Batching Considerations

The current training may sample each batch as random (embedding, prototype) pairs without grouping by class. **L_var requires multiple samples per class within the same batch** to compute within-class variance. If the current batching does not guarantee this, it must be modified.

**Recommended approach:** structured batching where each batch contains exactly K samples for each of C randomly selected classes, giving batch_size = K × C. For example, K=10 samples × 16 classes = batch_size of 160.

**Minimum viable K:** The within-class variance estimated from K samples has noise proportional to 1/√K. With K < 4, the variance estimate is very unreliable and L_var becomes noisy. **K ≥ 8 is recommended.** K = 10–20 is safer.

**If the existing batching is purely random:** the simplest fix is to group the batch by class after sampling and only compute L_var for classes with ≥ 4 samples in the batch. But structured batching is more principled.

### 4.4 Gradient Flow Requirements

L_var must be **differentiable with respect to the generator's outputs**, and those outputs must retain their computation graph back to θ_G. This means:

- Do NOT call `.detach()` on `fake` before passing to `compute_L_var`
- Do NOT use `fake.data` or `fake.clone()` in a way that breaks the graph
- The `mean()` and element-wise operations in `compute_L_var` are all autograd-compatible — no special handling needed
- The L2 normalisation at the generator's output must be part of the computation graph (it almost certainly already is)

### 4.5 var_target Computation — Detailed Steps

This is a precomputation step before the WGAN-GP training loop begins:

```python
# After encoder f_b is trained and frozen:

all_class_variances = []

for c in seen_classes:                          # c ∈ {1, ..., 1654}
    eeg_features_c = get_training_eeg(class=c)  # (n_c, 561), n_c ≈ 8
    with torch.no_grad():
        embeddings_c = f_b(eeg_features_c)      # (n_c, 64), on S^63
    
    mu_c = embeddings_c.mean(dim=0)             # (64,)
    var_c = ((embeddings_c - mu_c) ** 2).mean(dim=0)  # (64,)
    all_class_variances.append(var_c)

var_target = torch.stack(all_class_variances).mean(dim=0)  # (64,)
```

**CRITICAL WARNINGS:**

1. Use the **training split only** (13,232 samples, ~8 per class). Not the test split.
2. The encoder must be in **eval mode** with `torch.no_grad()`. Its weights are frozen.
3. `var_target` should be stored as a fixed tensor (register as a buffer or save/load). Do not recompute during training.
4. Some classes may have very few samples (as few as 6–8). The per-class variance estimates will be noisy. Averaging across 1654 classes mitigates this substantially.

---

## 5. Hyperparameter Tuning

### 5.1 The Critical Hyperparameter: α

`alpha` controls the relative weight of L_var against L_wasserstein. The right value depends on the relative gradient magnitudes of the two terms.

**Initial calibration procedure:**

1. Before modifying the loss, run one generator update step with the existing loss only
2. Record ||∂L_wasserstein/∂θ_G||₂ (the L2 norm of the Wasserstein gradient)
3. Compute L_var for the same batch (without adding it to the loss)
4. Compute ||∂L_var/∂θ_G||₂
5. Set initial α = ||∂L_wasserstein/∂θ_G||₂ / ||∂L_var/∂θ_G||₂

This puts both gradient contributions on roughly equal footing. Then sweep around this value.

**Suggested sweep range:** Once the calibration gives α₀, sweep α ∈ {0.01·α₀, 0.1·α₀, 0.5·α₀, α₀, 2·α₀, 5·α₀, 10·α₀}. At least 7 values spanning two orders of magnitude.

### 5.2 Other Hyperparameters (Do Not Change Without Justification)

These are the existing WGAN-GP hyperparameters that should remain at their validated values unless a strong reason emerges:

```
λ (gradient penalty) = 10
n_critic = 5
lr = 1×10⁻⁴
β = (0.0, 0.9)   [Adam betas]
steps = 10,000
Generator architecture: 164 → 256 → 256 → 64, LeakyReLU, L2 normalisation
Critic architecture: 128 → 256 → 256 → 1, LeakyReLU
```

If the variance regularisation destabilises training at all α values, then consider reducing the learning rate as a secondary intervention before changing λ or n_critic.

### 5.3 K (Samples Per Class Per Batch)

If structured batching is implemented, K is the number of generator samples per class per batch. Higher K gives more stable variance estimates but increases memory and computation.

Recommended starting value: K = 10 or K = 20 (matching the synthesis-time K = 20).

---

## 6. Monitoring and Diagnostics

### 6.1 Quantities to Log Every N Steps

Log the following at regular intervals (every 100–500 steps):

```
1. L_wasserstein                    — should remain stable or slowly decrease
2. L_var                            — should decrease from initial value toward zero
3. L_G (total)                      — the combined loss
4. VarR (sampled)                   — generate a batch of samples, compute VarR against var_target
                                      Expected: starts ~0.87, should increase toward 1.0
5. Critic loss L_D                  — should remain stable; instability here means the 
                                      regularisation is destabilising the minimax game
6. Gradient norms                   — ||∂L_wasserstein/∂θ_G|| and ||∂L_var/∂θ_G|| separately
                                      Monitor for one term dominating the other
```

### 6.2 Success Criteria

The modification is working correctly if:

- L_var decreases over training
- VarR moves from ~0.87 toward 1.0 (but does NOT overshoot to >1.3 or oscillate wildly)
- L_wasserstein remains in the same range as the unmodified training (±50%)
- The critic loss L_D does not diverge or collapse

### 6.3 Failure Modes to Watch For

**Mode 1: Variance oscillation.** VarR oscillates rapidly between <0.5 and >1.5. Cause: α too large, or K too small (noisy variance estimates). Fix: reduce α by 2–5×, increase K.

**Mode 2: Wasserstein collapse.** L_wasserstein spikes upward (becomes less negative or positive) and does not recover. The regularisation is overpowering the Wasserstein objective. Fix: reduce α by 10×.

**Mode 3: No effect.** L_var barely changes over training, VarR stays at ~0.87. Cause: α too small, gradient from L_var is negligible. Fix: increase α by 10×.

**Mode 4: Critic divergence.** L_D becomes increasingly negative or unstable. The modified generator is producing samples the critic has difficulty evaluating. Fix: increase n_critic to 10, or reduce α. The critic may need more updates to keep up with the changed generator.

### 6.4 Full Pipeline Evaluation

After training the modified WGAN-GP, run the full evaluation pipeline:

1. Generate synthetic unseen embeddings (20 per class, 200 classes)
2. Downsample to ~8 per class to match seen density
3. Train logistic regression on real seen + synthetic unseen
4. Evaluate AccS, AccU, H on test sets
5. Compute diagnostic metrics: VarR, ρ(synth centroids, prototypes), ρ(synth centroids, real unseen centroids)

**Compare against baseline:** H=4.77%, AccS=4.11%, AccU=5.69%, VarR=0.872, ρ_sp=0.857, ρ_sr=0.588.

**What to expect if the modification works:**

- VarR closer to 1.0 (variance better matched)
- ρ_sp slightly decreased (synthetic centroids less coupled to prototypes — some of the prototype structure is traded for within-class diversity)
- ρ_sr increased (synthetic centroids better match real unseen centroids)
- AccU improved (unseen classification better because classifier was trained on more realistic spread)
- AccS may decrease slightly (seen accuracy may trade off with unseen)
- H improved (the key metric — harmonic mean should increase)

---

## 7. Correctness Verification Checklist

Before running full training, verify each component:

- [ ] `var_target` has shape (64,) and all entries are positive
- [ ] `var_target` is computed from training split only, using the frozen encoder in eval mode
- [ ] `compute_L_var` returns a scalar with `requires_grad=True` (connected to generator graph)
- [ ] Gradient verification: compare autograd gradient ∂L_var/∂ê against the analytical formula (4/K) · Δ ⊙ (ê − μ̂) on a small test case — should match to ~1e-5
- [ ] L_var = 0 when synthetic variance exactly matches target (test with a synthetic input)
- [ ] The critic loss and gradient penalty computation are completely unchanged
- [ ] Generated embeddings remain on S^63 after modification (verify ||ê||₂ = 1 for random samples)
- [ ] The label space is consistent — class labels used in batching for L_var match the prototype indexing

---

## 8. Notes for the Planning Agent

### 8.1 Scope Discipline

This modification is surgically targeted. The temptation to "improve" other parts of the pipeline simultaneously (e.g., changing the encoder, trying a different classifier, modifying the critic) must be resisted. Change one thing at a time and measure. The variance regularisation is the single intervention being tested.

### 8.2 Statistical Estimation Constraint

The use of diagonal-only covariance matching (64 numbers per class instead of the full 64×64 = 4096 entries) is a deliberate choice driven by sample scarcity (~8 real samples per seen class). With 8 data points in 64 dimensions, the full covariance matrix would be rank-deficient (rank ≤ 8) and dominated by estimation noise. Matching the full matrix would force the generator to reproduce noise, making synthetic data worse. **Do not attempt to match the full covariance matrix.** This is not a computational limitation — it is a statistical one.

### 8.3 The Synthesis Step Is Unchanged

After training the modified WGAN-GP, the synthesis procedure (generating 20 embeddings per unseen class using random noise and the class prototype) remains exactly the same. The modification only changes *what the generator learns*, not *how we use it at inference*.

### 8.4 Compatibility With Future Loss Terms

The architecture of this modification (adding a term to L_G) is designed to be composable. If variance regularisation alone is insufficient, additional terms (diversity loss L_div, structural preservation L_struct) can be added to L_G with their own weights. Each term adds a gradient contribution that gets summed during backprop. The monitoring infrastructure (per-term loss logging, per-term gradient norms) should be built with this extensibility in mind from the start.

---

## 9. Summary of Key Mathematical Results

For quick reference during implementation:

**Generator Jacobian (L3.1):**
```
∂ê/∂s_c = (I − êêᵀ)/||u||₂ · W₃ · D₂ · W₂ · D₁ · W₁⁽ˢ⁾ ∈ R^{64×64}
```

**Variance regularisation loss (L3.2):**
```
L_var = (1/|C_batch|) Σ_c Σ_j ( (Σ̂_c)_jj − (Σ̄^real)_jj )²

where (Σ̂_c)_jj = (1/K) Σ_k (ê_{c,k,j} − μ̂_{c,j})²
```

**Gradient of L_var per sample (L3.2):**
```
∂L_var/∂ê_{c,k} = (4/K) · (diag(Σ̂_c) − diag(Σ̄^real)) ⊙ (ê_{c,k} − μ̂_c)
```

**Geometric meaning:** centroid-repulsion force when variance is below target, centroid-attraction when above.

**Modified generator objective:**
```
L_G = -E[D(G(z, s_c), s_c)] + α · L_var
```

**Why this works and perturbation didn't:** Perturbation operates within the fixed column space of J_G. Regularisation reshapes J_G by modifying the weight matrices through gradient descent. One works within the cage; the other rebuilds it.
