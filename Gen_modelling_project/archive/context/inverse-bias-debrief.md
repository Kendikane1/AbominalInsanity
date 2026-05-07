# Inverse Bias Phenomenon — Debrief & Implementation Directive

**Project**: GZSL EEG Decoding — Hybrid Paradigm  
**Date**: March 2026  
**Scope**: Theoretical diagnosis of the 99.6% seen→unseen misrouting failure in Method D, followed by a prioritised fix strategy targeting internal representation equivalence across the pipeline.

---

## Part I: Diagnosis Summary

### The Failure Mode

The full GZSL classifier (Method D) routes 99.6% of true-seen test trials to unseen labels — an *inverse* of the classic GZSL bias where unseen classes get starved. Three compounding mechanisms were identified.

### Mechanism 1 — Variance Mismatch

Real seen embeddings (from noisy EEG through a brain encoder) have high angular spread on $\mathbb{S}^{d-1}$: $\operatorname{tr}(\Sigma_s)$ is large. Synthetic unseen embeddings (from a clean generator MLP) cluster tightly around their conditioning prototypes: $\operatorname{tr}(\tilde{\Sigma}_u) \ll \operatorname{tr}(\Sigma_s)$.

This asymmetry causes the logistic classifier's optimizer to learn inflated weight norms $\|w_u\| \gg \|w_s\|$ and biases $\beta_u > \beta_s$ for unseen classes, because they present a geometrically trivial classification problem (high Fisher discriminant ratio $J_u \gg J_s$).

### Mechanism 2 — Sample Count Imbalance

Unseen classes receive ~20 synthetic samples per class versus ~8 real samples per seen class. Logistic regression's MLE produces higher-magnitude parameters for classes with more data and cleaner signal, further inflating unseen-class confidence.

### Mechanism 3 — Max-Over-Classes Amplification

Prediction requires $\arg\max$ over all 1,854 classes. For a seen test input to be correctly classified, its true seen logit must exceed *all* 200 unseen logits. Even with per-class leakage probability as low as 0.02, the compound probability of at least one unseen logit exceeding the correct seen logit is $1 - 0.98^{200} \approx 0.982$. The maximum of many moderate random variables overwhelms a single moderate value.

These three mechanisms compound multiplicatively, producing near-total misrouting.

---

## Part II: Fix Landscape

Six candidate interventions were analysed against the three mechanisms:

| Fix | Description | Targets | Intervention Level |
|-----|-------------|---------|-------------------|
| 1 | Inject calibrated noise into synthetic embeddings | M1, M3 | Data |
| 2 | Balance per-class sample counts (seen vs unseen) | M2 | Data |
| 3A | Cosine classifier (normalise weight vectors) | M1, M3 | Architecture |
| 3B | Split temperatures $\tau_S, \tau_U$ | M1, M3 | Decision (post-hoc) |
| 4 | Post-hoc calibration bias $\gamma$ (AUSUC-style) | M1, M3 | Decision (post-hoc) |
| 5 | Covariance-aware generator (moment matching loss) | M1 at source | Generative model |
| 6 | Prototype retrieval (no learned classifier) | All (sidestep) | Paradigm change |

---

## Part III: Agreed Directive

### Guiding Principles

1. **Internal representation equivalence** is the primary objective — the generator, encoder, and classifier should operate on statistically equivalent distributions, not rely on post-hoc corrections to mask distributional mismatch.
2. **Hybrid paradigm performance** is the metric — we are optimising the full GZSL pipeline (Method D), not falling back to prototype retrieval.
3. Fix 6 (prototype retrieval) is excluded as it is redundant to our focus on the hybrid paradigm.

### Implementation Plan

#### Phase 1 — Baseline Hygiene (Immediate)

**Fix 2: Balance per-class sample counts.**

- Equalise training samples per class across seen and unseen splits.
- Options: downsample synthetic unseen to ~8 per class, or generate synthetic seen embeddings to raise seen per-class counts to 20, or a combination.
- This removes M2 and ensures any remaining asymmetry is due to data geometry alone, not unequal statistical power.
- Effort: trivial. Run as a controlled experiment to quantify M2's isolated contribution.

#### Phase 2 — Core Intervention (Primary Research Effort)

**Fix 5: Covariance-aware generator.**

- Augment the WGAN-GP generator loss with a second-order moment matching term:

$$\mathcal{L}_G = \mathcal{L}_{\text{WGAN}} + \lambda \sum_{s \in \mathcal{S}} \Big\| \hat{\Sigma}_G^{(s)} - \hat{\Sigma}_{\text{real}}^{(s)} \Big\|_F$$

- This eliminates M1 at the generative stage. If $G$ produces embeddings with real-like covariance, the downstream classifier never encounters the distributional mismatch.
- **Known challenges**:
  - Full covariance matching requires estimating $64 \times 64 = 4{,}096$ entries from small per-class batches (~8 samples). Consider low-rank approximations or matching only the top-$k$ eigenvalues of $\Sigma$.
  - Trace matching ($\operatorname{tr}(\tilde{\Sigma}) \approx \operatorname{tr}(\Sigma)$) is a weak but tractable first step — equivalent to matching total variance without directional structure.
  - The $\lambda$ hyperparameter controls the trade-off between adversarial realism and moment matching. Sweep over $\lambda$ using harmonic mean accuracy on a validation set.
- **Ablation plan**: Compare trace-only matching vs top-$k$ eigenvalue matching vs full Frobenius matching to determine how much second-order structure matters.

#### Phase 3 — Architectural Safeguard

**Fix 3A: Cosine classifier.**

- Replace the standard logistic classifier with a cosine classifier: normalise both $w_c$ and $e$, use a learnable temperature $\tau$.
- This structurally prevents the optimizer from encoding distributional differences into weight norms ($\|w_u\| \gg \|w_s\|$), providing a guarantee that complements Fix 5's statistical approach.
- Even if Fix 5 imperfectly matches covariances, the cosine classifier prevents residual mismatch from manifesting as asymmetric confidence.
- Effort: moderate (architectural change to classifier, retraining required).

#### Phase 4 — Fallback & Diagnostics

**Fix 1: Noise injection into synthetic embeddings.**

- If Fix 5 proves difficult to train or converges slowly, noise injection provides an approximate version of the same effect at the data level with no generator retraining.
- Estimate $\hat{\Sigma}_{\text{real}}$ from pooled seen training embeddings. Add $\eta \sim \mathcal{N}(0, \alpha \hat{\Sigma}_{\text{real}})$ to synthetic embeddings before classifier training, tuning $\alpha$ on a validation set.
- **Diagnostic value**: If noise injection alone closes most of the seen-accuracy gap, it confirms the covariance mismatch is the dominant factor, validating the investment in Fix 5.

#### Phase 5 — Final Polish (If Needed)

**Fix 4: Post-hoc calibration $\gamma$.**

- After Phases 1–3, if a small seen-unseen routing bias remains, apply an additive logit correction $\gamma \cdot \mathbb{1}[c \in \mathcal{U}]$ tuned on a held-out set.
- On top of already-calibrated representations, this should require only a small $\gamma$ — qualitatively different from the large correction needed to compensate for a fundamentally broken distribution.
- If $\gamma$ is near zero, the upstream fixes have fully resolved the calibration problem.

### Evaluation Protocol

- **Primary metric**: Harmonic mean of seen and unseen accuracy, $H = 2 \cdot \frac{\text{Acc}_S \cdot \text{Acc}_U}{\text{Acc}_S + \text{Acc}_U}$.
- **Diagnostic metrics**: Per-class logit magnitude distributions ($\|w_c\|$, $\beta_c$) for seen vs unseen, Fisher discriminant ratios $J_c$, and the empirical routing rate (fraction of seen-class test inputs predicted as unseen).
- **Ablation structure**: Each phase is additive. Report results after each phase to isolate the contribution of each fix.

### Expected Outcome

If the diagnosis is correct, Fix 5 + Fix 2 + Fix 3A should produce a classifier whose internal representations are distributionally equivalent across seen and unseen classes. The seen-accuracy collapse should be largely resolved, with Fix 4 providing at most a small residual correction. The harmonic mean should improve substantially over the current Method D baseline.

---

## Part IV: Key Theoretical Insight

The inverse bias phenomenon is fundamentally a **domain gap within the training set** — synthetic and real embeddings share a representation space but occupy it differently. Classic GZSL methods assume the generator bridges the seen-unseen gap, but if the generator produces embeddings that are *too clean*, it creates a new gap between synthetic and real data that the classifier exploits in the wrong direction. The fix is not to make the generator worse, but to make it *realistically noisy* — matching not just the location but the shape of real embedding distributions.

---

*This document serves as the handoff between theoretical analysis and implementation. Bring this to Claude Code for implementation planning.*
