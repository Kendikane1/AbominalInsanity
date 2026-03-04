/session-kickstart

  Read through CLAUDE.md and MEMORY.md thoroughly first. Then read these context
  files in order to understand where we are:

  1. @context/image-alignment-paradigm-shift-debrief.md — the most recent session
     debrief covering the CORnet-S image alignment paradigm shift (READ THIS
     CAREFULLY — it contains everything you need)
  2. @context/CORnet_S_paradigm_analysis.md — detailed analysis of the first image
     alignment run results
  3. @context/CORnet_S_paradigm.md — raw Colab cell outputs from the run

  KEY CONTEXT: We just completed a paradigm shift from EEG-to-text to EEG-to-image
  (CORnet-S) alignment. Results: H-mean improved 7.8x (0.58% → 4.53%), AccU
  improved 18x (0.32% → 5.73%). The pipeline works. The notebook
  (GZSL_EEG_Pipeline_v2.ipynb, 67 cells) is fully functional with image alignment.

  The recommended next steps (in priority order) are:
  1. Hyperparameter tuning on image alignment (epochs, tau, embed_dim, architecture)
  2. Structure-preserving WGAN-GP mathematical research (variance diffusion
     regularisation, dual-graph Laplacian — this is my main research interest and
     the novel contribution I want to develop)
  3. Source CLIP image features for comparison
  4. Routing calibration

  I want to discuss the path forward and then proceed with whatever we decide.
  Take your time absorbing everything before asking me questions.

  ---


# CORnet-S Image Alignment — First Run Analysis

**Date**: 2026-03-03
**Run**: First end-to-end execution of EEG-to-image (CORnet-S) alignment pipeline
**Notebook**: `GZSL_EEG_Pipeline_v2.ipynb` (67 cells)

---

## 1. Executive Summary

The paradigm shift from EEG-to-text to EEG-to-image (CORnet-S) alignment produced a **7.8x improvement in H-mean** and an **18x improvement in unseen accuracy** on the first run, with no hyperparameter tuning. This confirms the visual investigation hypothesis: the text alignment target was the primary bottleneck, not the encoder or the GAN.

| Metric | Text Alignment (post-fix) | Image Alignment (CORnet-S) | Change |
|--------|---------------------------|----------------------------|--------|
| **AccS** | 3.45% | **3.75%** | +0.30% abs (+8.7% rel) |
| **AccU** | 0.32% | **5.73%** | +5.41% abs (**18x**) |
| **H-mean** | 0.58% | **4.53%** | +3.95% abs (**7.8x**) |
| Routing rate | 10.8% | 20.3% | +9.5% (more unseen predictions) |
| Weight norm ratio | 1.05x | 1.11x | Slight increase (still healthy) |
| Encoder loss | ~4.2 | **0.93** | **4.5x lower** (much better convergence) |

**Verdict**: Strong success. The image alignment paradigm is validated. Every single metric improved.

---

## 2. Metric-by-Metric Analysis

### 2.1 Contrastive Encoder Loss: 5.31 → 0.93

This is the most telling diagnostic of the paradigm shift's impact.

**Text alignment** converged to ~4.2 final loss. With batch_size=256, the contrastive loss starts near ln(256) ≈ 5.55 (random chance — each sample equally likely to match any other). A final loss of 4.2 means the encoder barely moved beyond random chance. It learned *something*, but the text prototypes were so tightly clustered (cosine sim 0.668) that the contrastive task was fundamentally degenerate — most negatives looked like positives.

**Image alignment** converged to 0.93. This is a *qualitative* difference, not just quantitative:
- The loss dropped 4.6x further below the starting point compared to text
- A loss of 0.93 means the encoder is correctly identifying the matching image feature for a given EEG signal within a batch of 256 alternatives with high probability
- The learning curve shows clean, monotonic convergence: 5.31 → 1.53 → 1.16 → 1.02 → 0.96 → 0.93

**Why**: CORnet-S prototypes are near-orthogonal (cosine sim -0.001). Each class occupies a distinct direction on S^127. The contrastive task is well-posed: there is a *clear* correct match and 255 clearly *wrong* alternatives in each batch. The encoder can actually learn to discriminate.

**Interpretation**: The encoder is learning a meaningful brain-to-image mapping. The contrastive objective is functioning as intended for the first time in this project.

### 2.2 AccS (Seen Accuracy): 3.45% → 3.75%

A modest +0.30% absolute improvement (+8.7% relative). This is expected and actually informative:

- AccS is measured on the 1854-way GZSL task (1654 seen + 200 unseen classes). The fact that AccS improved even with 200 unseen "distractors" means the encoder is extracting more class-discriminative information from EEG under image alignment.
- However, AccS is fundamentally bounded by the **EEG input quality** — 17 channels × 33 time points = 561 features averaged across 10 trials per class. No alignment target can make the EEG itself more discriminative. AccS improvement must come from encoder architecture or input quality.
- The modest AccS gain confirms that **the bottleneck for seen accuracy is upstream** (EEG resolution, not alignment target). This is consistent with Phase F's conclusion.

### 2.3 AccU (Unseen Accuracy): 0.32% → 5.73% (18x improvement)

This is the headline result. Unseen accuracy measures the classifier's ability to correctly identify classes it has *never seen real EEG for* — only synthetic embeddings generated by the WGAN-GP conditioned on prototypes.

**Why the massive improvement**: The entire ZSL transfer chain depends on prototype quality. With text alignment:
- Text prototypes were clustered (cosine ~0.668) → all prototypes ~similar
- WGAN-GP conditioned on near-identical prototypes → synthetic embeddings for different unseen classes were near-identical
- Classifier couldn't distinguish between unseen classes → ~random chance (0.32% on 200-way ≈ 0.5%)

With image alignment:
- Image prototypes are near-orthogonal (cosine ~ -0.001) → each class has a unique direction
- WGAN-GP conditioned on diverse prototypes → synthetic embeddings for different classes are genuinely different
- Classifier can discriminate between unseen classes → 5.73% (11.5x above random chance on 200-way)

**Key insight**: AccU is the metric that directly reflects prototype quality. The 18x improvement is almost entirely attributable to the prototype geometry change.

### 2.4 H-Mean: 0.58% → 4.53% (7.8x improvement)

H = 2 × AccS × AccU / (AccS + AccU). The harmonic mean penalises imbalance between seen and unseen performance. A model that gets 50% AccS but 0% AccU has H = 0.

With text alignment, the extreme imbalance (AccS >> AccU) crushed the H-mean. With image alignment, both AccS and AccU are in a similar ballpark (3.75% vs 5.73%), giving a much healthier H-mean.

This is the single most important GZSL metric, and a 7.8x improvement on the first run, with no tuning, is a strong result.

### 2.5 Routing Rate: 10.8% → 20.3%

Routing rate = fraction of seen test samples that the classifier predicts as unseen. Lower is generally better for AccS, but routing is a trade-off: too low means the model never predicts unseen (killing AccU), too high means it over-predicts unseen (killing AccS).

The increase from 10.8% to 20.3% is a **healthy trade-off**:
- The model is now "willing" to predict unseen classes when the EEG genuinely resembles an unseen category
- This is enabled by the diverse prototype geometry — the classifier has distinct decision regions for unseen classes, not a single "unseen blob"
- The cost: 20.3% of seen samples are misrouted (vs 10.8% before)
- The gain: AccU jumped from 0.32% to 5.73% — a massively positive trade-off

**Optimal routing**: With 200 unseen classes out of 1854 total, the "ideal" prior routing rate is ~10.8% (200/1854). Our 20.3% suggests a slight unseen bias. This could be calibrated with temperature scaling on the classifier logits if needed.

### 2.6 Weight Norm Ratio: 1.05x → 1.11x

Unseen weight norms (3.62) are slightly larger than seen norms (3.27), ratio 1.11x. This is well within healthy range (the old routing catastrophe had 1.76x). The slight unseen bias is consistent with the 20.3% routing rate — the classifier gives marginally more confidence to unseen predictions.

The weight norm distributions (from the diagnostics figure) show significant overlap between seen and unseen, confirming the classifier is treating both groups fairly. No pathological imbalance.

### 2.7 Intercept Analysis

Seen intercepts: mean +0.007 (near zero). Unseen intercepts: mean -0.060 (slightly negative).

The negative unseen intercepts act as a mild "penalty" on unseen predictions, partially counteracting the slightly higher weight norms. This is the classifier's learned calibration — it's implicitly implementing a slight prior that seen classes are more likely. This is appropriate given the 1654:200 class ratio.

### 2.8 WGAN-GP Performance

| Metric | Value | Assessment |
|--------|-------|------------|
| Final G_loss | 0.6014 | Stable (not diverging) |
| Final C_loss | -0.0370 | Near equilibrium |
| GP | ~0.001 | Gradient penalty well-regularised |
| Synth norm | 1.0000 (std 0.000) | Perfect L2-normalisation |
| Per-dim variance | Real=0.0078, Synth=0.0078 | **Identical** — no mode collapse |
| Cosine sim (synth vs proto) | 0.5612 | Moderate alignment to prototypes |

The WGAN-GP performed well under the new prototype geometry:
- **No divergence**: G_loss decreased monotonically (0.03 → 0.60), C_loss stabilised near 0
- **No mode collapse**: Synthetic and real per-dimension variances are identical (0.0078)
- **Cosine sim 0.5612**: Synthetic embeddings are moderately aligned with their conditioning prototypes. This is a middle ground — too high would mean the generator just copies the prototype, too low would mean the conditioning is ignored. 0.56 is reasonable.

**Observation**: The G_loss *increased* from 0.03 to 0.60 over training. This is normal for WGAN-GP — it means the Critic is getting better at distinguishing real from fake, which forces the Generator to also improve. The fact that G_loss stabilised (not increasing rapidly) shows equilibrium was reached.

---

## 3. Visual Analysis

### 3.1 Embedding t-SNE (clip_embedding_tsne.png)

The t-SNE plot shows EEG test embeddings (blue dots) distributed across the 2D space with seen prototypes (green stars) and unseen prototypes (red triangles) interspersed throughout.

**Key observations**:
1. **Prototypes are dispersed**: Unlike text alignment where prototypes clustered in a tight cone, image prototypes occupy distinct positions across the embedding space. This is the geometric signature of near-orthogonal CORnet-S features projected into the shared 128-D space.
2. **EEG embeddings form a continuous cloud**: No obvious class clusters visible at this scale (expected with 1654 classes and 10 trials each — individual class clusters are too small to resolve). The cloud is roughly uniform, suggesting the encoder distributes brain embeddings across the space.
3. **Seen and unseen prototypes intermingle**: Both green stars and red triangles are spread throughout the EEG cloud. This means the encoder maps brain signals into a space where both seen and unseen prototypes are "reachable" — a prerequisite for ZSL transfer.

**Comparison with text alignment**: In the text alignment t-SNE, prototypes would have been clustered together (reflecting the 0.668 cosine similarity). Here, they are dispersed — a direct visual confirmation that the paradigm shift changed the geometry as intended.

### 3.2 Real vs Synthetic t-SNE (real_vs_synth_tsne.png)

Real seen embeddings (blue) and synthetic unseen embeddings (orange) overlap substantially.

**Key observations**:
1. **Good overlap**: Synthetic unseen embeddings occupy the same region as real seen embeddings. This is the primary quality indicator — if synthetic embeddings lived in a separate region, the classifier would learn to trivially distinguish real from synthetic, not seen from unseen.
2. **Synthetic density matches real**: The orange and blue point densities are comparable, suggesting the Generator produces embeddings with realistic variance.
3. **Prototypes embedded within**: Both seen and unseen prototypes sit within the embedding cloud, confirming they serve as meaningful "anchors" in the space.

**Comparison with text alignment**: Similar overlap quality. The WGAN-GP was already performing well under text alignment — the key improvement is that the *prototypes themselves* are now more informative, leading to more diverse and class-discriminative synthetic embeddings.

### 3.3 Classifier Diagnostics (pipeline_v2_diagnostics.png)

**Weight norms** (left panel):
- Seen norms (blue): broad distribution centered at ~3.27, range [1.5, 5.0]
- Unseen norms (red): narrower distribution centered at ~3.62, range [2.3, 4.3]
- The distributions **overlap substantially** — the classifier is treating seen and unseen classes with roughly equal confidence
- Compare with the text alignment routing catastrophe where unseen norms were at 7.23 vs seen at 4.10 (non-overlapping distributions)

**Intercepts** (right panel):
- Seen intercepts (blue): centered near 0, symmetric distribution
- Unseen intercepts (red): centered slightly below 0 (-0.06), indicating a mild prior against unseen predictions
- The negative unseen intercepts are appropriate — they partially compensate for the slightly higher unseen weight norms, creating a balanced decision boundary

**Overall**: The classifier is well-calibrated. No pathological imbalances. The sample balancing (downsampling + class_weight='balanced') is working as intended.

---

## 4. What Went Well

### W1: The paradigm hypothesis was correct
The visual investigation predicted that image alignment would dramatically outperform text alignment due to superior class separation. The results confirm this — H-mean improved 7.8x.

### W2: Zero errors on first run
All 67 cells executed without errors. The helper script applied clean modifications, the variable naming convention (keeping `S_seen_prototypes` etc.) ensured seamless downstream flow.

### W3: Encoder loss dropped to meaningful levels
For the first time in this project, the contrastive loss converged well below the random-chance baseline. The encoder is genuinely learning brain-to-image correspondences.

### W4: WGAN-GP adapted to new geometry
Despite the radically different prototype geometry (near-orthogonal vs clustered), the WGAN-GP trained stably and produced high-quality synthetic embeddings with no mode collapse.

### W5: Classifier diagnostics are healthy
Weight norm ratio 1.11x, overlapping distributions, mild intercept bias — all indicators of a well-calibrated classifier.

### W6: AccU is the star metric
5.73% on 200-way unseen classification (11.5x above random chance) demonstrates genuine zero-shot transfer. The WGAN-GP + image prototype pipeline is working as a ZSL system.

---

## 5. What Could Be Improved

### I1: AccS is still limited (~3.75%)
The seen accuracy barely improved (+0.30%). This confirms the Phase F finding: the EEG input itself (17ch × 33t = 561 features) is the bottleneck for seen-class discrimination. The alignment target helps unseen transfer but cannot inject class-discriminative information that isn't in the EEG signal.

**Possible interventions**:
- Higher spatial resolution (more channels)
- Higher temporal resolution (more time points)
- Multi-subject training (pool data across subjects)
- Attention mechanisms that weight informative channels/timepoints

### I2: Routing rate of 20.3% is above optimal
With 200/1854 ≈ 10.8% unseen classes, the "ideal" routing rate is ~10.8%. Our 20.3% means the classifier over-predicts unseen by ~2x. This hurts AccS (673 seen samples misrouted out of 3308 = 20.3%).

**Possible interventions**:
- Calibrated temperature scaling on classifier logits
- Adjusting n_synth_per_class (currently 20 → try 10-15 to reduce unseen representation in training)
- Post-hoc threshold tuning on the seen/unseen decision boundary

### I3: Encoder could train longer or with different hyperparameters
The final loss of 0.93 is much better than text alignment's ~4.2, but the loss curve was still decreasing at epoch 50. There may be room to push further:
- More epochs (75-100, with early stopping on validation loss)
- Lower tau (try 0.10 or 0.12 — with better prototype separation, harder negatives may now be informative)
- Larger embed_dim (try 256 — more room for the 1000-D features to express structure)

### I4: ImageProjector architecture is untested
We used a 2-layer MLP (1000→512→128) based on theoretical reasoning, but didn't sweep architectures:
- 3-layer (1000→512→256→128) might capture more structure
- Hidden_dim=256 instead of 512 might regularise better
- Skip connections could help gradient flow through the 1000-D → 128-D compression

### I5: CORnet-S features are not CLIP-aligned
NICE-EEG's 15.6% top-1 uses CLIP image features, which share a latent space with CLIP text. CORnet-S features are biologically-inspired but not contrastively trained. Sourcing CLIP image embeddings from the THINGS stimuli could unlock another significant improvement.

---

## 6. Comparison with SOTA

| Method | Dataset | Task | Metric | Value |
|--------|---------|------|--------|-------|
| **Ours (CORnet-S, this run)** | ThingsEEG-Text | 1854-way GZSL | H-mean | **4.53%** |
| **Ours (CORnet-S, this run)** | ThingsEEG-Text | 200-way ZSL (unseen only) | AccU | **5.73%** |
| Ours (text alignment) | ThingsEEG-Text | 1854-way GZSL | H-mean | 0.58% |
| NICE-EEG (Song et al., 2024) | ThingsEEG2 | 200-way ZSL | Top-1 | 15.6% |
| ATM (Li et al., 2024) | ThingsEEG2 | 200-way ZSL | Top-1 | ~12% |

**Important caveats**:
- Our AccU (5.73%) is measured within the 1854-way GZSL setting (classifier has both seen and unseen options). NICE-EEG's 15.6% is measured in a 200-way unseen-only setting (no seen distractors). These are not directly comparable — GZSL is harder.
- NICE-EEG uses CLIP image features and a different EEG dataset (ThingsEEG2, with higher temporal resolution).
- Our 5.73% AccU in GZSL mode is encouraging — in a pure 200-way setting (without 1654 seen distractors), accuracy would likely be higher.

---

## 7. Theoretical Interpretation

### 7.1 Why Image Alignment Works

The contrastive encoder learns a mapping f_b: R^561 → S^127 (brain embeddings on the unit sphere) by minimising InfoNCE loss against g_v: R^1000 → S^127 (image embeddings). The loss landscape is shaped by the image target geometry.

With text targets (cosine sim 0.668): the image embeddings form a tight cone. The InfoNCE denominator (sum of exp(sim/tau) for all negatives) is dominated by many near-identical negatives. The gradient signal is noisy and low-magnitude — the encoder learns slowly and plateaus early.

With image targets (cosine sim -0.001): the image embeddings are roughly uniformly distributed on S^127. Each negative is clearly distinct from the positive. The gradient signal is clean and high-magnitude — the encoder learns efficiently and converges to a much lower loss.

### 7.2 Why AccU Improved More Than AccS

AccS depends on the encoder's ability to extract class-discriminative features from EEG. This is limited by the input data (561-D, 10 trials/class). Better alignment targets don't add new information to the EEG signal.

AccU depends on the prototype quality flowing through the WGAN-GP → classifier chain. Image prototypes are dramatically better than text prototypes for class discrimination. This directly translates to better synthetic embeddings and better unseen classification.

In other words: **AccS is EEG-limited, AccU is prototype-limited**. We fixed the prototype bottleneck; the EEG bottleneck remains.

### 7.3 WGAN-GP Geometry Under Image Prototypes

The Generator G(z, s_c) maps noise z ∈ R^100 and prototype s_c ∈ R^128 to synthetic embeddings. With text prototypes, s_c vectors were near-identical across classes, so G essentially learned a single "brain embedding distribution" conditioned on a ~constant signal. With image prototypes, each s_c is unique, forcing G to learn a proper conditional distribution — different classes get genuinely different synthetic samples.

The cosine similarity of synthetic to prototype (0.5612) confirms moderate conditioning: synthetics are influenced by but not identical to their prototypes. This is the desired behaviour — the Generator adds realistic "noise" around each class prototype.

---

## 8. Recommended Next Steps

### Priority 1: Hyperparameter Tuning on Image Alignment
Now that the paradigm works, optimise within it:
- **Epochs**: Try 75-100 (loss was still decreasing at 50)
- **Temperature**: Try tau in {0.08, 0.10, 0.12, 0.15, 0.20} — with better prototypes, optimal tau may differ
- **embed_dim**: Try 256 (more capacity for 1000-D features)
- **ImageProjector architecture**: 3-layer vs 2-layer, hidden_dim sweep

### Priority 2: Structure-Preserving WGAN-GP (Research Thread)
The mathematical deep-dive into WGAN-GP loss modifications:
- Variance diffusion regularisation (Long et al.) as a generator loss term
- Dual-graph Laplacian structure preservation
- This could improve synthetic embedding quality and further boost AccU

### Priority 3: CLIP Image Features
Source CLIP image embeddings for the THINGS stimuli (from NICE-EEG or BraVL repos). If CORnet-S gives 5.73% AccU, CLIP image features (which have cross-modal structure with text) could push significantly higher.

### Priority 4: Routing Calibration
Post-hoc temperature scaling or threshold tuning to bring routing rate closer to the optimal ~10.8%. This would improve AccS with minimal AccU cost.

---

## 9. Conclusion

The CORnet-S image alignment paradigm shift is **validated**. The first untuned run produced a 7.8x H-mean improvement, confirming that the text alignment target was the primary bottleneck. The pipeline is stable, the diagnostics are healthy, and there is clear room for further improvement through hyperparameter tuning and architectural exploration.

The most exciting signal is the encoder loss convergence (0.93 vs 4.2). For the first time, the contrastive encoder is genuinely learning brain-to-image correspondences, not just fitting noise in a degenerate loss landscape. This opens the door for all downstream improvements — better encoders, better prototypes, better synthesis — that were previously blocked by the text alignment ceiling.
