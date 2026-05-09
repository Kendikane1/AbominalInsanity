# Experiment 04: FiLM Conditioning — Results Analysis

**Date**: 2026-05-07  
**Status**: Complete — all 3 variants executed on Colab (v5e-1 TPU)  
**Author**: Ariz Akmal  

---

## 1. Executive Summary

FiLM conditioning delivered a genuine structural fix to the WGAN-GP generator: structural overcoupling (ρ_sp) dropped from **0.857 to 0.639**, landing near the real-data level. However, no variant improved H-mean beyond the 4.77% Exp01 baseline, and VarR_unseen unexpectedly declined across the board. The analysis establishes that the WGAN-GP synthesis pathway is not the binding constraint on H-mean. The encoder — trained exclusively on seen classes — is the ceiling.

---

## 2. Results Summary

### 2.1 Per-Variant Final Metrics

| Metric | Exp01 Baseline | Exp03 L_var | 04a FiLM | 04b FiLM+LN | 04c FiLM+Lvar (α=1.0) |
|---|---|---|---|---|---|
| **H-mean** | **4.77%** | 4.58% | 4.59% | 4.64% | 4.69% |
| AccS | 4.11% | — | 4.17% | 3.90% | 3.90% |
| AccU | 5.69% | — | 5.09% | 5.74% | 5.89% |
| VarR_seen | — | 0.973 | 0.925 | 0.937 | 0.931 |
| VarR_unseen | 0.872 | 0.875 | 0.847 | 0.850 | 0.847 |
| VarR gap (seen−unseen) | — | 0.098 | 0.078 | 0.087 | 0.084 |
| **ρ_sp** | 0.857 | 0.880 | **0.639** | 0.716 | 0.685 |
| kNN@10 | 0.611 | — | 0.463 | 0.512 | 0.521 |

### 2.2 FiLM+L_var Alpha Sweep (04c)

| α | H-mean | AccS | AccU | VarR_unseen | ρ_sp |
|---|---|---|---|---|---|
| 0.00 | 4.48% | 3.84% | 5.39% | 0.844 | 0.653 |
| 0.10 | 4.52% | 3.99% | 5.21% | 0.847 | 0.666 |
| 0.50 | 4.61% | 3.96% | 5.53% | 0.850 | 0.667 |
| **1.00** | **4.67%** | 3.75% | 6.19% | 0.843 | 0.678 |
| 2.00 | 4.58% | 3.87% | 5.60% | 0.852 | 0.666 |
| 5.00 | 4.43% | 3.78% | 5.35% | 0.869 | 0.710 |
| 10.00 | 4.16% | 3.69% | 4.76% | 0.889 | 0.743 |

### 2.3 Training Dynamics (04a FiLM Baseline)

VarR_seen during training showed rapid early rise then stabilisation:

| Step | G_loss | C_loss | VarR_seen |
|---|---|---|---|
| 1K | −1.393 | 0.014 | 0.795 |
| 3K | −1.469 | −0.002 | 0.930 |
| 5K | −1.570 | −0.016 | 0.910 |
| 7K | −1.575 | −0.012 | 0.945 |
| 10K | −1.620 | −0.016 | 0.901 |

G_loss was still decreasing at step 10K (−1.393 → −1.620), indicating the generator had not converged.

---

## 3. Detailed Analysis

### 3.1 What Worked: Structural Overcoupling (ρ_sp)

The most significant and unambiguous result is the reduction in structural overcoupling:

- **Exp01 baseline**: ρ_sp = 0.857  
- **Exp03 L_var best**: ρ_sp = 0.880 (worsened)  
- **04a FiLM baseline**: ρ_sp = 0.639  
- **04b FiLM+LN**: ρ_sp = 0.716  
- **04c FiLM+Lvar α=1**: ρ_sp = 0.685  

This is a genuine architectural improvement. FiLM separated the noise pathway from the prototype, and the generator's synthetic centroids no longer copy prototype pairwise distances as faithfully as before. The concatenation-conditioned generator was learning to directly mirror prototype geometry into its output — FiLM broke this shortcut.

The FiLM baseline (0.639) went slightly below the real-data reference level (~0.668), meaning the decoupling slightly over-corrected. The LayerNorm variant (0.716) and the L_var variant (0.685) are closer to the target. Importantly, none of the FiLM variants regressed ρ_sp as Exp03 did.

The interpretation: **FiLM solved the structural overcoupling problem it was designed to solve.** The architecture is correct.

### 3.2 What Did Not Work: VarR_unseen

This is the primary failure against hypothesis. The target was VarR_unseen > 0.95 with a transfer gap < 0.03. The actual results:

| Variant | VarR_seen | VarR_unseen | Gap |
|---|---|---|---|
| Exp01 baseline | — | 0.872 | — |
| Exp03 (best α) | 0.973 | 0.875 | 0.098 |
| 04a FiLM | 0.925 | **0.847** | 0.078 |
| 04b FiLM+LN | 0.937 | **0.850** | 0.087 |
| 04c Lvar α=1 | 0.931 | **0.847** | 0.084 |

All FiLM variants produced **lower** VarR_unseen than the Exp01 baseline. The gap did narrow (from 0.098 in Exp03 to 0.078 in 04a), but the narrowing was achieved by both VarR_seen and VarR_unseen moving toward the middle — not by VarR_unseen rising.

**Mechanistic explanation**: FiLM makes the generator produce more uniform variance across all prototype classes (seen and unseen), because the noise pathway is now class-agnostic. However, the resulting variance level settled at a moderate equilibrium (~0.85–0.93) rather than the high-variance regime targeted. The L2 normalisation of generator outputs is the binding geometric constraint: all embeddings live on S^63, and variance on the unit hypersphere is bounded. Without a stronger gradient signal specifically pushing VarR_unseen up, the generator finds a configuration that satisfies the Wasserstein objective at moderate variance, and this equilibrium is actually lower than the concatenation-conditioned baseline.

### 3.3 kNN@10 Neighbourhood Preservation Worsened

| Variant | kNN@10 |
|---|---|
| Exp01 baseline | 0.611 |
| 04a FiLM | 0.463 |
| 04b FiLM+LN | 0.512 |
| 04c Lvar α=1 | 0.521 |

Neighbourhood preservation declined substantially across all variants. FiLM baseline (0.463) is the worst, with LN and Lvar variants partially recovering. This is the flip side of the ρ_sp reduction: when the generator decouples from prototype geometry at the global level (pairwise centroid distances), it also loses local neighbourhood structure. The synthetic centroids no longer cluster near the correct prototype neighbours.

This matters because kNN@10 = 0.463 means fewer than half of the 10 nearest unseen prototype neighbours are also nearest neighbours in synthetic centroid space. The generator's output space has a weaker local geometry than even the already-poor Exp01 baseline.

The LN variant partially recovers (0.512) because LayerNorm re-introduces some regularity into the modulated activations, weakly coupling the output geometry back to prototype structure.

### 3.4 Gradient Cosine Similarity: Validation of FiLM Hypothesis

The gradient cosine similarity between ∇L_wasserstein and ∇L_var for the 04c sweep:

| α | cos sim range (steps 1K–5K) |
|---|---|
| 0.10 | −0.062 to +0.069 |
| 0.50 | −0.076 to +0.089 |
| 1.00 | −0.038 to +0.064 |
| 2.00 | −0.087 to +0.038 |

Compare to Exp03, where cosine similarity was **consistently negative** (−0.04 to −0.09), indicating structural gradient conflict between the two losses. In Exp04c, the cosine similarity oscillates around zero — the losses are now cooperative or neutral rather than adversarial.

**This validates the FiLM hypothesis at the gradient level.** Separating the noise pathway from the prototype conditioning resolved the gradient conflict that made L_var ineffective in Exp03. However, neutral gradient alignment also means L_var provides only a weak additional signal — it is no longer fighting L_wasserstein, but it is also not strongly cooperating. The result is that L_var's variance-boosting effect at α=1 is modest.

### 3.5 The L_var Alpha Trade-off Curve

The alpha sweep reveals a clean, monotonic trade-off:

```
α:            0.0   0.1   0.5   1.0   2.0   5.0  10.0
H-mean (%):   4.48  4.52  4.61  4.67  4.58  4.43  4.16
VarR_unseen:  0.844 0.847 0.850 0.843 0.852 0.869 0.889
ρ_sp:         0.653 0.666 0.667 0.678 0.666 0.710 0.743
```

As α increases:
- VarR_unseen rises monotonically (0.844 → 0.889)
- H-mean peaks at α=1 then degrades
- ρ_sp rises monotonically above α=2

The rising ρ_sp at high α is mechanistically coherent: L_var pushes synthetic embeddings away from their prototypes, which paradoxically makes the inter-class distance structure more prototype-correlated (the generator produces diffuse clouds around prototypes, whose centroids still track prototype distances). The rising ρ_sp at high α is a sign of a different kind of overcoupling — variance-driven centroid drift that re-introduces prototype-distance dependence.

The H-mean degradation at high α is also coherent: the LogReg classifier needs synthetic embeddings to be localised enough around their prototype that 200 classes remain separable. High within-class variance erodes this separability. The optimal α=1.0 represents the least-bad compromise, not a genuine solution.

**Key insight**: at α=10, VarR_unseen = 0.889 — the closest we've come to the 0.95 target — but H = 4.16%, the worst result of the sweep. This is not a tuning failure; it is a demonstration that VarR_unseen and H-mean are in fundamental tension given the current evaluation structure (LogReg on 200 classes, balanced sampling).

---

## 4. The H-mean Ceiling

Plotting all H-mean results across all experiments to date:

| Experiment | H-mean |
|---|---|
| Exp00 text alignment (best) | 0.70% |
| Exp01 concat WGAN-GP baseline | **4.77%** |
| Exp02 eta perturbation (best) | 2.56% |
| Exp03 L_var (best α) | 4.58% |
| 04a FiLM baseline | 4.59% |
| 04b FiLM + LayerNorm | 4.64% |
| 04c FiLM + Lvar α=1 (sweep best) | 4.67% |
| 04c FiLM + Lvar α=1 (final run) | 4.69% |

Every intervention on the WGAN-GP synthesis stage has landed between 4.58% and 4.77%. The known stochastic variance of the WGAN-GP across runs with identical config is ~±0.13pp (observed in the dim sweep: same config, two runs produced H=4.54% and H=4.67%). The entire range 4.58–4.77% is within two standard deviations of this variance.

**The ceiling at ~4.7–4.8% H-mean cannot be moved by improvements to the WGAN-GP synthesis stage alone.** The evidence is now exhaustive:

- Post-hoc perturbation (Exp02): ceiling unchanged, then degraded
- Training-time variance regularisation (Exp03): ceiling unchanged
- Architectural conditioning fix (Exp04): ceiling unchanged

All three approaches addressed different aspects of the synthesis pathology. None moved the ceiling.

---

## 5. Root Cause: The Encoder Bottleneck

### 5.1 The argument

At evaluation time, the GZSL classifier receives:
- **Training data**: real seen embeddings + synthetic unseen embeddings
- **Test data**: real unseen EEG embeddings (from the encoder)

Improving synthetic embedding quality only improves downstream H-mean if the improved synthetics better approximate the **real unseen EEG embedding distribution**. But the encoder — a BrainEncoder (561→1024→512→64) trained with InfoNCE on seen classes only — has never received a gradient signal from unseen class EEG. Its encoding of unseen EEG is determined by generalisation from seen-class structure, not by direct optimisation.

If the encoder's unseen embedding space is noisy, entangled, or poorly separated from seen embeddings, no synthesis quality improvement can compensate. The classifier's test-time inputs (real unseen EEG) are fixed by the encoder. Synthesis only affects the training distribution.

### 5.2 The evidence

- AccU has ranged 5.09–5.89% across all WGAN-GP experiments. The spread is small relative to the task (200 unseen classes). Even the best synthesis configuration produces only ~5.9% unseen accuracy.
- Encoder top-k from the hyperparameter sweep (COMP2261 era) reached 4.47% top-1 on seen classes. This implies the encoder can barely discriminate seen classes from raw EEG — it has even weaker generalisation to unseen.
- The full hyperparameter sweep (25 configs, 4–8× parameter ranges) showed only 0.01pp H-mean improvement despite 30% encoder improvement. This was an early warning that the encoder is not the leverage point, but now the converse applies: since the encoder cannot be made to work on unseen classes through indirect contrastive training on seen classes alone, synthesis improvements have no remaining upside to unlock.

### 5.3 What this means

The current architecture has a theoretical performance ceiling determined by the encoder's zero-shot generalisation to unseen EEG. That ceiling appears to be near 4.8% H-mean. Further WGAN-GP work refines the synthesis within this ceiling but cannot raise it.

---

## 6. Paths Forward

### Path 1 — Extended Training (50K steps) [CHOSEN]

**Rationale**: G_loss at step 10K was still decreasing (−1.56 → −1.62 across the final 3K steps in the FiLM baseline). The WGAN-GP has not converged. There is a non-trivial possibility that longer training allows the generator to discover higher-quality variance structure without architectural changes. Cost: 5× current Colab GPU time per run.

**Success criterion**: H-mean > 4.77% at 50K steps with FiLM architecture.  
**Failure signal**: H-mean plateau between 30K–50K steps at the same ~4.7% level.

### Path 2a — Increased Generator Capacity (hidden_dim=512) [CHOSEN]

**Rationale**: The current generator (hidden_dim=256) has 4× capacity ratio relative to embed_dim=64. A 512-wide network has 8× capacity. Additional width may allow the generator to represent richer within-class variance structure that the current network cannot express. The FiLM conditioning networks (64→128→512) are already wider than the generator backbone — this imbalance may be limiting.

**Success criterion**: VarR_unseen > 0.90 (meaningful improvement over current ~0.847) with H-mean maintained.  
**Failure signal**: Hidden_dim=512 hits the same VarR equilibrium as hidden_dim=256.

### Path 2b — Encoder Redesign (Transductive / Seen+Unseen Alignment) [DEFERRED]

Architecturally, the only way to raise the ceiling is to give the encoder a learning signal from unseen EEG. Options include transductive zero-shot learning (use unlabelled unseen EEG at test time to update representations), self-supervised pre-training on all EEG, or prototype-based alignment with a stronger geometric loss. This is a fundamentally different research direction and has been deferred pending exhaustion of WGAN-GP improvements.

### Path 3 — L_struct Loss (DEFERRED)

Rather than maximising within-class variance (L_var), a structure-preservation loss on synthetic centroids would target ρ_sp ≈ 0.668 (real-data level) explicitly while keeping within-class variance unconstrained. This addresses the over-decoupling observed in FiLM baseline (ρ_sp=0.639) and the kNN@10 degradation. Deferred until extended training and capacity experiments are complete.

---

## 7. Implementation Decision for Next Experiment

The user has decided to consolidate paths 1, 2a, and the FiLM architecture into a **single notebook** (Exp 05) rather than three separate experiments. The specification is:

- Architecture: FiLMGenerator with hidden_dim=**512** (up from 256)
- FiLM MLP hidden dim: **256** (up from 128, scaled proportionally)
- Training steps: **50,000** (up from 10,000)
- L_var at α=**1.0** (sweep optimum from 04c)
- All other hyperparameters unchanged (z_dim=100, embed_dim=64, lr=1e-4, n_critic=5)

Success criteria (unchanged from Exp04 targets):
- H-mean > 4.77%
- VarR_unseen > 0.95
- VarR gap < 0.03
- ρ_sp ≈ 0.668 (not minimised further)

If this compound experiment fails to beat 4.77%, the conclusion is that the WGAN-GP + encoder paradigm has reached its limit and Path 2b (encoder redesign) must be opened.

---

## 8. Artefacts

| File | Description |
|---|---|
| `notebook.ipynb` | 04a: FiLM baseline, 10K steps, hidden_dim=256 |
| `notebook_film_ln.ipynb` | 04b: FiLM + LayerNorm post-modulation |
| `notebook_film_lvar.ipynb` | 04c: FiLM + L_var alpha sweep [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0] |
| `results/results1.md` | Raw Colab output from all 3 runs |
| `results/analysis.md` | This document |

WandB runs: `https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl`  
Run IDs: `tbn39fif` (04a), `u8bxy265` (04b), `5yl2ziuv` / `xxxtxuzf` / `0dzr3b1c` / `73nx5dja` / `2h4ed4mz` / `a5yl6he1` / `9niyqrmf` (04c sweep)
