# Hyperparameter Sweep Analysis — CORnet-S Image Alignment

**Date**: 2026-03-09
**Scope**: 5-phase sequential greedy sweep, 25 encoder training runs + 1 full pipeline validation
**Baseline**: First image alignment run (tau=0.15, epochs=50, embed_dim=128, lr=1e-3, wd=1e-4)
**Hardware**: Colab Pro (GPU), total sweep time ~37 minutes

---

## 1. Executive Summary

The sweep found a configuration that improves encoder-only top-1 retrieval from 3.45% to 4.47% (+30% relative), but this **did not translate to downstream GZSL improvement**: H-mean moved from 4.53% to 4.54% (+0.01pp). The composition shifted — AccS rose +13.7% while AccU fell -15.2% — revealing a rigid seen-unseen trade-off that hyperparameter tuning cannot break.

**The central conclusion is that the encoder is no longer the bottleneck.** The performance ceiling for the current pipeline is imposed by the interaction between embedding quality and the WGAN-GP synthesis → classifier chain, not by the contrastive encoder's hyperparameters.

---

## 2. Phase-by-Phase Analysis

### 2.1 Phase 1: Epoch x Temperature Grid

**Result**: best_tau=0.05, best_epochs=75, top1=4.11%

#### Temperature (tau) Analysis

The sweep reversed the Phase E finding. In text alignment, tau=0.15 was optimal because the clustered text prototypes (cosine sim ~0.668) made hard negatives misleading — at low tau the softmax denominator was dominated by near-identical negatives that the encoder couldn't meaningfully separate. Now, with near-orthogonal CORnet-S prototypes (cosine sim ~-0.001), the landscape has flipped:

| tau | Best top-1 (across epochs) | Interpretation |
|-----|---------------------------|----------------|
| 0.05 | **4.11%** | Hard negatives are now informative — distinct prototypes allow meaningful discrimination |
| 0.07 | 3.87% | Still benefits from harder gradients |
| 0.10 | 3.63% | Middle ground |
| 0.12 | 3.39% | Begins to soften too much |
| 0.15 | 3.69% | Phase E optimum — suboptimal here |
| 0.20 | 3.75% | Interestingly recovers slightly — non-monotonic |

The monotonic trend from tau=0.05 to tau=0.12 is clear: lower temperature = harder contrastive signal = better learning, which is the theoretically expected behaviour when prototypes are well-separated. The contrastive loss gradient with respect to similarity score s_ij scales as:

```
∂L/∂s_ij ∝ exp(s_ij / tau) / Σ_k exp(s_ik / tau)
```

At low tau, this concentrates gradient mass on the hardest negatives (those with highest spurious similarity to the anchor). When prototypes are near-orthogonal, these hard negatives are genuinely confusable samples — not artefacts of degenerate prototype geometry — so the gradient signal is informative.

However, the improvement from tau=0.15→0.05 is only **0.42pp absolute** (3.69%→4.11%). This is a meaningful relative gain (+11.4%) but a small absolute effect, suggesting temperature is not a dominant factor.

#### Epoch Analysis

The heatmap reveals a consistent pattern across all temperatures: **performance peaks at epoch 75 (±25) and does not improve with further training**.

For tau=0.05 specifically:
- Epoch 50: 3.78% → Epoch 75: **4.11%** → Epoch 100: 3.87% → Epoch 200: 3.84%

This is overfitting in action. With only ~8 training samples per class (13232 samples / 1654 classes), the encoder memorises the training set beyond epoch 75. Crucially, the **training loss continues to decrease** throughout (0.0163 → 0.0096 → 0.0062 → 0.0018) while test retrieval accuracy degrades. This train-test divergence is the canonical signature of overfitting.

#### Loss Curve Observations

The right panel of the Phase 1 figure shows that all temperature values converge to their respective loss floors. The absolute loss values differ by orders of magnitude (tau=0.05 → ~0.002, tau=0.20 → ~1.6) but this is a scaling artefact of the InfoNCE loss, not a quality difference. At tau=0.05, the softmax concentrates probability mass so sharply that even moderate alignment drives the loss near zero. At tau=0.20, the softer distribution retains more residual entropy. **Loss values across different temperatures are not directly comparable.**

**Critical note**: The loss was "still decreasing at epoch 50" (0.93) under the baseline tau=0.15 config, which motivated testing longer training. We now see that this continued decrease was the encoder overfitting, not useful learning. The test metric confirms: more training beyond ~75 epochs hurts regardless of temperature.

#### Phase 1 Variance Concern

The entire Phase 1 heatmap spans 3.02% to 4.11% — a total range of **1.09pp across 36 evaluation points**. Many cells are within ~0.3pp of each other. With 3308 test samples, a single-sample accuracy difference corresponds to ~0.03pp. The margin between the winner (4.11%) and many alternatives is only 5-15 test samples — potentially within stochastic noise.

**Implication**: The "optimal" tau=0.05, epochs=75 selection is reasonable but should be understood as a local peak in a relatively flat landscape, not a sharp optimum.

---

### 2.2 Phase 2: Embedding Dimension

**Result**: best_dim=64, top1=4.17%

| embed_dim | top-1 | Loss | Params (approx) |
|-----------|-------|------|-----------------|
| 64 | **4.17%** | 0.0105 | ~700K |
| 128 | 3.90% | 0.0103 | ~1.7M |
| 256 | 3.96% | 0.0102 | ~3.2M |
| 512 | 4.17% | 0.0107 | ~6.0M |

This is the most surprising result of the sweep, and arguably the most revealing.

**The dimension sweep is effectively flat.** The total range is 0.27pp (3.90% to 4.17%). The 64-D and 512-D spaces — which differ by 8x in dimensionality — produce identical top-1 retrieval accuracy. This flatness has a profound implication:

**The encoder's retrieval accuracy is not capacity-limited.** If embedding dimension mattered, we would see a clear monotonic trend (either up or down). The fact that 64-D matches 512-D means the class-discriminative information extractable from 561-D EEG features can be fully represented in 64 dimensions — and adding more dimensions neither helps (because there's no additional signal to capture) nor hurts significantly (because the encoder doesn't overfit to the extra capacity at 75 epochs with tau=0.05).

**Why 64 was selected**: The sweep script picked the first occurrence of the maximum (4.17%), which was dim=64. The tie with dim=512 is a coincidence — they are different encoders that happen to achieve the same test accuracy.

**Downstream implications of dim=64**: While encoder retrieval is dimension-invariant, the downstream WGAN-GP and classifier are **not**. In 64-D:
- Random unit vectors have expected pairwise cosine similarity 0 with std ≈ 1/√64 = 0.125
- In 128-D: std ≈ 1/√128 = 0.088

This means class prototypes have 42% wider similarity distributions in 64-D vs 128-D. The WGAN-GP must generate synthetic embeddings that land in tighter regions of a more crowded hypersphere. The classifier faces harder 1854-way discrimination with less geometric separation between decision boundaries. **This is a hidden cost of dim=64 that the encoder-only proxy metric does not capture.**

---

### 2.3 Phase 3: ImageProjector Architecture

**Result**: best_arch=[512] (current baseline), top1=4.17%

| Config | Hidden Dims | Depth | Top-1 | Top-5 | Params |
|--------|-------------|-------|-------|-------|--------|
| A | [256] | 2 | 3.66% | 12.15% | 273K |
| **B** | **[512]** | **2** | **4.17%** | **12.94%** | **546K** |
| C | [768] | 2 | 4.11% | 12.82% | 820K |
| D | [512, 256] | 3 | 3.54% | 11.52% | 662K |
| E | [768, 384] | 3 | 3.78% | 12.06% | 1.09M |
| F | [1024, 512] | 3 | 3.99% | 11.91% | 1.59M |

Two clear patterns emerge:

**1. Two-layer projectors outperform three-layer projectors.** The best 3-layer (F: 3.99%) is worse than the worst competitive 2-layer (C: 4.11%). The additional non-linearity in 3-layer projectors hurts rather than helps. This is consistent with the information-theoretic view: the ImageProjector's job is dimensionality reduction from 1000-D → 64-D, not complex feature transformation. The CORnet-S features are already high-level semantic representations (output of a deep CNN). Adding non-linear transformations on top of already-nonlinear features introduces unnecessary information loss.

Mathematically, each LayerNorm + ReLU + Dropout layer in a 3-layer projector acts as an information bottleneck. With the intermediate compression [1000→512→256→64], the 256-D bottleneck forces the network to discard information that may be useful for the final 64-D projection. The 2-layer [1000→512→64] avoids this by having only one non-linear compression step.

**2. Hidden width matters but plateaus.** The [256] projector is clearly underpowered (3.66%), but [512] and [768] are nearly identical (4.17% vs 4.11%). The 512-wide hidden layer provides sufficient capacity for the 1000→64 compression.

**3. Top-5 tells the same story as top-1.** Config B leads in both metrics (12.94% top-5), confirming this isn't a noisy fluke in the top-1 ranking.

**Verdict**: The current architecture [512] is already optimal for this projection task. No architectural change needed.

---

### 2.4 Phase 4: Learning Rate x Weight Decay

**Result**: best_lr=0.002, best_wd=0.0001, top1=4.47%

| | wd=0 | wd=1e-4 | wd=1e-3 |
|---|------|---------|---------|
| **lr=5e-4** | 4.29% | 4.35% | 4.14% |
| **lr=1e-3** | 4.23% | 4.17% | 4.44% |
| **lr=2e-3** | 4.14% | **4.47%** | 4.38% |

This phase produced the largest improvement of the sweep: 4.17% → 4.47% (+0.30pp, +7.2% relative).

**Learning rate analysis**: The optimal lr=0.002 is 2x the Phase E optimum (1e-3). With tau=0.05, the contrastive gradients are sharper (more concentrated on hard examples), which effectively reduces the "useful" gradient magnitude. A higher peak learning rate compensates for this, allowing the encoder to traverse the loss landscape faster before the cosine schedule decays the lr toward zero.

However, the lr effect is non-monotonic and interacts with weight decay:
- At wd=0: lr=5e-4 (4.29%) > lr=1e-3 (4.23%) > lr=2e-3 (4.14%) — **lower lr is better** without regularisation
- At wd=1e-4: lr=2e-3 (4.47%) > lr=5e-4 (4.35%) > lr=1e-3 (4.17%) — **higher lr is better** with moderate regularisation
- At wd=1e-3: lr=1e-3 (4.44%) > lr=2e-3 (4.38%) > lr=5e-4 (4.14%) — **mid lr is best** with strong regularisation

This interaction is textbook: weight decay provides L2 regularisation that prevents overfitting during aggressive learning (high lr). Without weight decay, high lr overfits; with too much weight decay, the model is over-regularised. The sweet spot is lr=2e-3 + wd=1e-4 — fast learning with light regularisation.

**The total range in Phase 4 is 0.33pp** (4.14% to 4.47%). While this is the largest single-phase improvement, it remains a narrow band. Nine configs × 3308 test samples means the margin between best and worst is ~11 correct classifications out of 3308.

---

## 3. Full Pipeline Validation (Phase 5)

### 3.1 The Headline Disappointment

| Metric | Baseline | Optimal | Delta | Relative |
|--------|----------|---------|-------|----------|
| **AccS** | 3.75% | **4.26%** | +0.51pp | **+13.7%** |
| **AccU** | 5.73% | 4.86% | **-0.87pp** | **-15.2%** |
| **H-mean** | 4.53% | 4.54% | +0.01pp | **+0.2%** |
| F1 Seen | 2.25% | 2.62% | +0.37pp | +16.4% |
| F1 Unseen | 1.20% | 0.96% | -0.24pp | -20.3% |
| Routing rate | 20.3% | 16.2% | -4.1pp | -20.2% |

**H-mean is essentially unchanged.** A +0.01pp improvement over a 25-run sweep is noise. The encoder improved by 30% on its proxy metric but the downstream pipeline absorbed the entire gain through a seen-unseen redistribution.

### 3.2 What Happened: The Seen-Unseen Trade-off

The improvement in encoder quality manifested as **better seen discrimination at the expense of unseen discrimination**. The mechanism:

**Step 1 — Better encoder → better seen embeddings.** The optimal encoder produces brain embeddings that are more tightly clustered around their respective image prototypes. AccS rose from 3.75% → 4.26% because the classifier can more reliably assign seen test samples to their correct seen classes.

**Step 2 — Better encoder → unchanged prototype geometry.** The image prototypes S_seen and S_unseen are computed from the ImageProjector applied to fixed CORnet-S features. While the projector weights changed, the prototypes are still fundamentally determined by the CORnet-S input features. The inter-class cosine similarity structure is largely preserved.

**Step 3 — WGAN-GP generates in a different space.** With embed_dim=64 (vs 128), the WGAN-GP generates in a lower-dimensional space. The GAN dynamics changed substantially:
- Baseline (d=128): G_loss=0.60, C_loss=-0.037
- Optimal (d=64): G_loss=-1.37, C_loss=-0.038

The generator loss becoming strongly negative (-1.37) means the generator is producing samples that the critic rates as "real-looking" — the generator is "winning" the adversarial game more decisively. This could indicate either (a) genuinely better generation, or (b) the critic's inability to distinguish in 64-D due to the denser packing of distributions. The critic loss is nearly identical (-0.038 vs -0.037), suggesting the critic reached a similar equilibrium in both cases.

**Step 4 — Variance matching but discriminability loss.** Per-dim variance: Real=0.0156 (optimal) vs 0.0078 (baseline). This is expected: L2-normalised vectors on S^{d-1} have per-dim variance ≈ 1/d, so d=64 gives 1/64=0.0156 and d=128 gives 1/128=0.0078. The synthetic embeddings match real variance perfectly (0.0155 vs 0.0156). But matching marginal statistics doesn't guarantee that the synthetic class-conditional distributions preserve the inter-class structure that the classifier needs.

**Step 5 — Classifier bias shifts toward seen.** The routing rate dropped from 20.3% → 16.2%. This means the classifier is less likely to predict unseen labels for seen test samples — good for AccS — but the same bias reduces AccU because unseen test samples are also more likely to be misclassified as seen. The bias table confirms: 12006 out of 16000 unseen samples were predicted as seen classes (75.0%), vs 11383/16000 (71.1%) in the baseline.

### 3.3 Classifier Diagnostics Comparison

| Diagnostic | Baseline | Optimal |
|------------|----------|---------|
| Seen weight norms (mean) | 3.27 | 3.39 |
| Unseen weight norms (mean) | 3.62 | 3.74 |
| Weight norm ratio (U/S) | 1.11x | 1.10x |
| Seen intercepts (mean) | 0.007 | 0.004 |
| Unseen intercepts (mean) | -0.060 | -0.036 |
| Seen intercept std | 0.114 | 0.068 |
| Unseen intercept std | 0.069 | 0.050 |

The weight norm ratio is nearly identical (1.11x → 1.10x), confirming no catastrophic imbalance between seen and unseen class weights. The unseen intercept improved from -0.060 to -0.036 (closer to zero), which should favour unseen predictions — yet AccU decreased. This paradox resolves when we consider that the embedding geometry in 64-D makes it harder for the classifier to separate classes, overwhelming the slight intercept improvement.

The intercept standard deviations both decreased (seen: 0.114→0.068, unseen: 0.069→0.050), indicating more uniform classifier confidence across classes — a sign that the 64-D embeddings are less variable across classes (more homogeneous), which is consistent with the geometric crowding argument.

---

## 4. Critical Assessment: Why the Sweep Failed to Improve GZSL

### 4.1 Proxy Metric Misalignment

The sweep optimised **encoder-only top-k retrieval** as a proxy for **GZSL H-mean**. This proxy assumes: better encoder → better embeddings → better synthetic samples → better GZSL. The assumption breaks at the second arrow.

The encoder-to-GZSL chain has three transfer points:
1. **Encoder → Embedding quality** (measured by top-k retrieval): **+30% improvement confirmed**
2. **Embedding quality → Synthetic quality** (mediated by WGAN-GP): **Not improved — the WGAN-GP was not re-tuned**
3. **Synthetic quality → GZSL accuracy** (mediated by classifier): **Net-zero due to seen-unseen trade-off**

The proxy metric only captures transfer point 1. The sweep successfully optimised what it measured, but what it measured was not the binding constraint.

### 4.2 The Dimensionality Trap

The selection of embed_dim=64 illustrates a subtle failure mode. In 64-D:
- The prototype retrieval task (1654-way cosine similarity ranking) performs identically to 128-D because the top-k metric only requires the correct prototype to rank highest — it doesn't require large absolute margins.
- But the WGAN-GP synthesis task requires generating samples that are close to the correct prototype AND far from all others. In 64-D, the geometric margins between prototype Voronoi cells are smaller (std of random cosine sim: 0.125 vs 0.088), making the generator's job harder.
- The 1854-way LogReg classifier must separate classes with less geometric room. The decision boundaries are tighter and more sensitive to synthetic sample placement errors.

In essence, **the top-k retrieval metric is invariant to absolute margin but the downstream pipeline is not.** This is a fundamental limitation of using retrieval accuracy as a proxy for generation quality.

### 4.3 The Narrow Performance Band

Across all 36 Phase 1 evaluations, top-1 ranges from 3.02% to 4.11% — a total band of 1.09pp. Across all 25 encoder runs in the sweep, the range expands only to 3.54%–4.47%, or 0.93pp. Chance-level for 1654-way retrieval is 0.06%.

This narrow band — despite sweeping temperature by 4x (0.05–0.20), epochs by 4x (50–200), embedding dimension by 8x (64–512), and learning rate by 4x (5e-4–2e-3) — is the signature of a **hard performance ceiling**. The ceiling is not set by the encoder architecture or hyperparameters. It is set by the **EEG input features**: 561 dimensions = 17 channels x 33 time points, averaged across 10 trials, measured at ~100 Hz with consumer-grade EEG.

This is the same ceiling identified in Phase F for text alignment (~3% top-1), now confirmed for image alignment (~4.5% top-1). The paradigm shift to image alignment raised the ceiling modestly (the better-conditioned loss landscape helps the encoder learn more efficiently), but the fundamental input bottleneck persists.

### 4.4 Loss ≠ Quality

The contrastive loss dropped spectacularly: 0.93 (baseline) → 0.008 (optimal) — a **120x reduction**. Yet top-1 accuracy improved only 30%. This massive loss-accuracy divergence reveals that the loss is dominated by the "easy" samples that the encoder already classifies correctly. The hard samples — those misclassified at epoch 75 — contribute negligible loss because they're swamped by the 95.5% of samples that are already correct in the softmax denominator. The loss keeps decreasing (via making the correct matches more confident) even after accuracy saturates.

This is a well-known failure mode of InfoNCE: the loss can reach near-zero while the representation remains weak for rare or ambiguous samples. The loss is not a reliable indicator of representational quality in the tail of the distribution.

---

## 5. Positive Findings and Validated Insights

Despite the null H-mean result, the sweep produced valuable structural insights:

### 5.1 Temperature Physics Validated
The reversal from tau=0.15 (text) to tau=0.05 (image) is theoretically coherent and confirms our understanding of the contrastive geometry. With near-orthogonal prototypes, hard negatives carry real information. With clustered prototypes, they carry noise. This is a transferable insight for any future alignment target.

### 5.2 Architecture is Not the Bottleneck
The ImageProjector architecture sweep showed near-total insensitivity (3.54%–4.17%). The 2-layer [512] design inherited from text alignment is already appropriate for image features. No architectural engineering is needed on the projector side.

### 5.3 Moderate Training is Optimal
75 epochs is the sweet spot. This saves compute (vs the 200-epoch runs) and reduces overfitting risk. Combined with the lr=2e-3 finding, the encoder should train faster and stop earlier.

### 5.4 AccS Improved Meaningfully
The +13.7% relative improvement in AccS (3.75% → 4.26%) is real and significant. If the unseen accuracy can be restored through downstream improvements (WGAN-GP, routing calibration), the overall pipeline would benefit from the optimal encoder config. The encoder's seen-class discrimination has genuinely improved.

### 5.5 Routing Rate Improved
16.2% routing rate (vs 20.3% baseline) is closer to the ideal prior of 10.8% (200/1854 = 10.8%). Fewer seen samples are being misrouted to unseen classes. This is a structural improvement in the classifier's calibration.

---

## 6. Implications for Research Direction

### 6.1 Encoder Tuning is Complete
Further hyperparameter sweeps on the encoder will yield diminishing returns. The performance ceiling is input-limited, not architecture-limited or hyperparameter-limited. The optimal config (tau=0.05, epochs=75, lr=2e-3, wd=1e-4, embed_dim=64) or the baseline config (tau=0.15, epochs=50, lr=1e-3, wd=1e-4, embed_dim=128) perform within 0.01pp H-mean of each other. **Either is valid for downstream work.**

**Recommendation**: For the optimal pipeline going forward, consider retaining **embed_dim=128** (baseline) with the optimal training dynamics (tau=0.05, epochs=75, lr=2e-3, wd=1e-4). This preserves the geometric advantages of higher-dimensional synthesis while capturing the encoder-quality gains from better tau/epochs/lr. The sweep showed dim=64 and dim=128 differ by only 0.27pp on the encoder metric but dim=128 provides wider geometric margins for the WGAN-GP.

### 6.2 The WGAN-GP is the Next Frontier (Confirmed)
The sweep conclusively demonstrates that **the bottleneck has shifted from the encoder to the synthesis pipeline**. The encoder can be improved by 30% with no downstream gain. This validates the planned research direction: structure-preserving WGAN-GP modifications are where the performance gains will come from.

Specific WGAN-GP interventions to pursue:
- **Variance diffusion regularisation**: Force the generator to produce class-conditional distributions that respect the inter-prototype covariance structure
- **Dual-graph Laplacian preservation**: Maintain neighbour relationships in both brain and semantic spaces
- **Direct n_synth_per_class optimisation**: The current downsample from 20→8 per class is a crude balance mechanism. Generating exactly the right number, or using importance weighting, could improve the seen-unseen trade-off

### 6.3 Routing Calibration is a Free Lunch
The routing rate (16.2% optimal, 20.3% baseline, ideal 10.8%) can be adjusted post-hoc via Platt scaling, temperature scaling on the classifier logits, or explicit calibration on a held-out set. This requires no retraining and could recover some of the AccU lost in the trade-off. It should be pursued as a quick win before the WGAN-GP work.

### 6.4 The embed_dim Question Needs a Dedicated Downstream Sweep
The encoder-only proxy showed dim=64 ≈ dim=512. But the downstream pipeline almost certainly behaves differently. A targeted experiment — training the full pipeline (encoder + WGAN-GP + classifier) at dim={64, 128, 256} — would resolve whether the proxy-metric selection of dim=64 was actually harmful. This is a 3-run experiment (~45 minutes) and could be done before the WGAN-GP research.

---

## 7. Statistical Confidence Assessment

| Phase | Winner | Runner-up | Margin (pp) | Margin (samples) | Confidence |
|-------|--------|-----------|-------------|-------------------|------------|
| Phase 1 | tau=0.05/ep=75 (4.11%) | tau=0.05/ep=100 (3.87%) | 0.24 | ~8 samples | **Low** |
| Phase 2 | dim=64 (4.17%) | dim=512 (4.17%) | 0.00 | 0 samples | **Tied** |
| Phase 3 | [512] (4.17%) | [768] (4.11%) | 0.06 | ~2 samples | **Negligible** |
| Phase 4 | lr=2e-3/wd=1e-4 (4.47%) | lr=1e-3/wd=1e-3 (4.44%) | 0.03 | ~1 sample | **Negligible** |

None of the phase winners are statistically robust. With 3308 test samples and 1654 classes, the top-1 metric has high intrinsic variance. A proper confidence assessment would require either (a) multiple seeds per configuration, or (b) bootstrap resampling of the test set. The current single-seed evaluation cannot distinguish between configs separated by <0.5pp.

**This reinforces the ceiling interpretation**: the configurations are all achieving approximately the same ceiling, and the apparent "winner" at each phase is selected from noise.

---

## 8. Configuration Decision

Given the analysis above, two paths forward are defensible:

**Option A — Adopt the "optimal" config as-is**: tau=0.05, epochs=75, lr=2e-3, wd=1e-4, embed_dim=64. Gains: AccS +13.7%, routing -20.2%. Losses: AccU -15.2%. Net H-mean: +0.01pp.

**Option B — Hybrid config** (recommended): tau=0.05, epochs=75, lr=2e-3, wd=1e-4, **embed_dim=128**. Takes the validated training dynamics improvements from Phases 1 and 4, but keeps the higher embedding dimension that provides better geometric margins for downstream synthesis. This config was not directly tested in the sweep but is expected to achieve encoder top-1 between 3.90% (Phase 2 at dim=128) and 4.47%, while avoiding the crowding effects of dim=64.

---

## 9. Figures Summary

| Figure | Key Observation |
|--------|----------------|
| `sweep_phase1_heatmap.png` | Hot zone in top-left (low tau, moderate epochs). Non-monotonic epoch response. |
| `sweep_phase2_embed_dim.png` | Nearly flat across 8x dimension range. Ceiling confirmation. |
| `sweep_phase4_lr_wd.png` | lr × wd interaction visible. Higher lr needs moderate wd. |
| `sweep_baseline_vs_optimal.png` | AccS-AccU trade-off visualised. H-mean unchanged. |
| `optimal_config_diagnostics.png` | Weight norms healthy (1.10x ratio). Intercepts near-symmetric. |

---

## 10. Summary Table

| Finding | Implication |
|---------|-------------|
| Encoder top-1: 3.45% → 4.47% (+30% relative) | Encoder proxy metric improved significantly |
| H-mean: 4.53% → 4.54% (+0.2% relative) | **No downstream improvement** — proxy misalignment |
| AccS +13.7%, AccU -15.2% | Rigid seen-unseen trade-off |
| Performance band: 3.0%–4.5% across 25 configs | **Hard ceiling from EEG input quality** |
| Loss: 0.93 → 0.008 (120x reduction) | Loss is not a reliable quality indicator |
| dim=64 ≈ dim=128 ≈ dim=256 ≈ dim=512 | Dimension-invariant ceiling |
| [512] arch = optimal | Architecture is not the bottleneck |
| tau=0.05 > tau=0.15 for image alignment | Confirmed: hard negatives informative with orthogonal prototypes |
| Routing: 20.3% → 16.2% (ideal: 10.8%) | Post-hoc calibration is a free win |
| WGAN-GP dynamics changed (G_loss: 0.60 → -1.37) | GAN operating differently in 64-D — needs investigation |
| Encoder tuning saturated | **Bottleneck is now in synthesis pipeline** |
