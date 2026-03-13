---                                                                                                                                
  The Three Pathologies (UVDS) vs Our cWGAN-GP                                                                                       
                                                                                       
  Long et al. identify three fundamental problems with synthesised data for ZSL. What's important is that these are not specific to  
  their linear projection method — they are structural properties of the synthesis task itself. Let me walk through each one against
  our setup.                                                                                                                         
                                                                                                                                     
  ---                                                                                                                                
  Pathology 1: Visual-Semantic Structural Discrepancy                                                                                
                                                                                                                                     
  What the paper says: The data structures of visual and semantic spaces are fundamentally different. Two points close in semantic
  space can be far apart in visual space (their "HasTail" example — Zebra and Pig share the attribute but look nothing alike). A
  direct mapping that minimises reconstruction error will learn the principal components between the two spaces, which captures the
  global correlation but destroys local neighbourhood structure.

  How this maps to us: Our pipeline has an analogous structural gap, but it manifests differently because we have a GAN rather than a
   linear projection.

  Our WGAN-GP learns p(e | s_c) — the conditional distribution of brain embeddings given a class prototype s_c. It trains exclusively
   on seen classes. The generator sees (prototype, brain_embedding) pairs for 1654 seen classes and learns how brain embeddings
  distribute around their prototypes.

  The structural discrepancy enters at inference: when we condition on unseen prototypes, we're asking the generator to extrapolate.
  If the relationship between prototype geometry and embedding geometry is non-stationary — i.e., the way embeddings cluster around
  prototypes differs between seen and unseen class regions of the prototype space — then the generator will produce synthetic unseen
  embeddings that inherit the seen-class structural bias.

  Concretely: the generator learns "given a prototype at this location on S^63, brain embeddings tend to scatter in these directions
  with these magnitudes." If unseen prototypes occupy a different region of S^63 where the scattering pattern is different, the
  generator doesn't know this. It applies the seen-class mapping.

  This is exactly the over-fitting to seen classes phenomenon that Long et al. describe — their Eq. (2) learns P from seen data only,
   and ours learns G(z, s_c) from seen data only. Same fundamental problem, different computational substrate.

  Status in our pipeline: Almost certainly present. We have direct evidence: AccU=5.81% while the encoder top-1 on seen classes is
  4.23%. The synthesis chain loses discriminative power specifically for unseen classes. The routing table shows 75% of unseen
  samples are predicted as seen — the synthetic unseen embeddings look too much like seen embeddings.

  ---
  Pathology 2: Variance Decay

  What the paper says: When projecting from low-dimensional attributes (M-D) to high-dimensional visual features (D-D, where D >> M),
   the projection matrix P concentrates information in a few principal directions. Result: the synthesised data has high variance in
  a small number of dimensions and near-zero variance in the rest (their Fig. 6 shows this strikingly). The redundant low-variance
  dimensions carry no discriminative information, degrading classifier performance.

  Mathematically, if the projection P ∈ R^{M×D} has rank at most M, then the synthesised data X̂ = AP lies in an M-dimensional
  subspace of R^D. The variances along the remaining D-M directions are exactly zero. Even with regularisation, the effective rank of
   the synthesised data is severely limited.

  How this maps to us: Our dimension ratio is different — we synthesise in 64-D conditioned on 64-D prototypes, not 85-D → 4096-D. So
   the extreme rank-deficiency that Long et al. face doesn't apply in the same form.

  However, a subtler version of variance decay can still occur in a GAN. Here's why:

  The generator G: R^{100+64} → R^64 maps (z, s_c) to an embedding. The input is 164-D, the hidden layers are 256-wide, and the
  output is 64-D followed by L2-normalisation. The critic D only evaluates whether the output "looks real" — it doesn't explicitly
  require that all 64 output dimensions carry balanced information.

  If the critic can be fooled by getting a few dimensions right (because those dimensions carry most of the discriminative signal in
  the real data), the generator will learn to concentrate its effort on those dimensions. The remaining dimensions will have variance
   that matches the marginal statistics (because L2-normalisation forces ||e||=1) but the class-conditional variance structure may be
   degenerate.

  The key distinction from Long et al.: their variance decay is about marginal per-dimension variance. Ours would be about
  class-conditional per-dimension variance. Our marginal variances match perfectly (0.0156 real vs 0.0155 synthetic). But within each
   class, the synthetic samples may not spread across all dimensions the way real samples do.

  Status in our pipeline: Unknown — we haven't diagnosed this. We have the marginal variance match (confirmed), but we need to check:
  1. Per-dimension variance profile (sorted descending) — real vs synthetic
  2. Class-conditional variance profiles — for individual classes, how does the per-dim variance of synthetic samples compare to real
   samples?
  3. Effective rank of the synthetic data within each class

  This is a critical diagnostic we should run before implementing any fix.

  ---
  Pathology 3: Over-fitting / Bias Toward Seen Classes

  What the paper says: The trained model achieves high performance on seen classes but "dramatically degrades on the synthesised
  unseen data" (their Table 3 shows 20-32% accuracy drops from seen to unseen). This is because the projection P overfits to the
  seen-class principal directions.

  How this maps to us: This is the most directly confirmed pathology in our pipeline. Our results:
  - AccS = 3.90% (seen classes, real embeddings)
  - AccU = 5.81% (unseen classes, synthetic embeddings)

  While AccU is actually higher than AccS in our case (because there are far fewer unseen classes to confuse — 200 vs 1654), the
  per-class accuracy tells a different story. With 200 unseen classes, random chance is 0.5%, so AccU=5.81% is 11.6x above chance.
  With 1654 seen classes, random chance is 0.06%, so AccS=3.90% is 65x above chance. The seen classifier is far more discriminative
  relative to its difficulty level.

  More importantly, the bias table shows the asymmetry: 75% of unseen samples get predicted as seen. The synthetic embeddings don't
  create a sufficiently distinct region in embedding space for unseen classes.

  Status: Confirmed. This is our primary failure mode.

  ---
  The UVDS Solutions and Their GAN Analogues

  Now here's where it gets interesting. Long et al. propose two regularisations and a latent space. Let me map each to what we could
  implement in our GAN:

  Solution 1: Dual-Graph Regularisation (GR)

  Their Eq. (5):

  Ω₁(V) = ½ Σᵢⱼ ||vᵢ - vⱼ||² wᵢⱼ = Tr(V^T L V)

  where L = D - W is the graph Laplacian, and W is the average of k-NN graphs built in visual and semantic spaces.

  Principle: If two points are neighbours in either the visual or semantic space, they should be neighbours in the latent embedding
  space. This preserves local structure from both modalities.

  GAN analogue: We could add a neighbourhood-consistency loss to the generator. For a batch of synthetic embeddings {ê₁, ..., ê_B}
  conditioned on prototypes {s₁, ..., s_B}:

  L_graph = Σᵢⱼ ||êᵢ - êⱼ||² · w(sᵢ, sⱼ)

  where w(sᵢ, sⱼ) encodes how close prototypes i and j are (e.g., k-NN indicator, or cosine similarity kernel). This penalises the
  generator for producing synthetic embeddings that are far apart when their conditioning prototypes are nearby.

  This directly addresses Pathology 1 (structural discrepancy) by forcing the generator to respect the prototype neighbourhood
  structure in its outputs.

  Solution 2: Diffusion Regularisation (DR)

  Their approach: find an orthogonal rotation Q (QQ^T = I) that maximises the ℓ₂,₁ norm of the data matrix (Eq. 8):

  max_Q ||Q^T V^T||₂,₁ = Σ_d √(Σ_n x²_nd / N)  = Σ_d σ'_d

  Since the orthogonal constraint preserves total variance (Σ σ_d = Σ σ'_d = const), maximising the sum of standard deviations is
  equivalent to minimising the variance of the standard deviations — i.e., making all dimensions equally informative.

  GAN analogue: We can't apply an orthogonal rotation post-hoc to GAN outputs (the classifier needs raw embeddings, not rotated
  ones). But we can incorporate the same principle as a variance-balancing regularisation on the generator.

  For a batch of synthetic embeddings E_synth ∈ R^{B×d}, compute per-dimension standard deviations σ₁, ..., σ_d. Then penalise their
  imbalance:

  L_diffusion = Var(σ₁, ..., σ_d) / Mean(σ₁, ..., σ_d)²

  This is the squared coefficient of variation of the per-dim standard deviations. It's zero when all dimensions have equal variance
  (perfect diffusion) and large when variance is concentrated.

  Alternatively, we could directly maximise the ℓ₂,₁ norm (sum of per-dim stds), which is their exact objective — just implemented as
   a generator loss term rather than an orthogonal rotation.

  Solution 3: Latent Embedding Space V

  Their V is an auxiliary space that simultaneously satisfies reconstruction of visual features AND preservation of semantic
  structure.

  Our analogue: We already have this — our 64-D shared embedding space IS the latent space. The contrastive encoder creates it. But
  our encoder's loss (InfoNCE) only optimises for alignment (brain-image matching), not for structure preservation. The dual-graph
  regularisation could be added to the encoder's contrastive loss to make the shared space more structure-preserving from the start.

  ---
  What I Propose We Do Next

  Before implementing anything, we need diagnostics. The paper's Fig. 6 is the most important diagnostic — it tells you whether
  variance decay is present. I propose:

  Diagnostic 1: Per-dimension variance profile. Extract real seen embeddings and synthetic unseen embeddings from the current
  pipeline. Sort per-dim variances descending. Plot both on the same axes (like their Fig. 6). This tells us if our WGAN-GP has
  variance decay.

  Diagnostic 2: Class-conditional variance. For a subset of seen classes (where we have both real and can-generate synthetic),
  compare the within-class per-dim variance between real and synthetic. This tells us if the GAN preserves intra-class structure.

  Diagnostic 3: Prototype neighbourhood preservation. Compute pairwise cosine similarities between all seen prototypes. Then compute
  pairwise cosine similarities between the corresponding class-mean synthetic embeddings. Correlate the two. This tells us if the GAN
   preserves inter-class structure (the graph regularisation target).

  These three diagnostics will tell us which of the three pathologies are actually active in our pipeline, and therefore which
  regularisation terms will have the highest impact. No point implementing diffusion regularisation if we don't have variance decay.

  What do you think? Shall we run these diagnostics first, or do you want to dig deeper into the mathematical derivations before
  touching code?

  AFTER RUNNING DIAGNOSTICS :

  These results are genuinely revealing. Let me walk you through each diagnostic from the ground up, building intuition before giving
   the punchline.

  ---
  Diagnostic 1: Per-Dimension Variance Profile

  What this measures and why:

  Think of your 128-D embedding space as 128 independent "channels" of information. Each dimension carries some amount of signal —
  measured by its variance across samples. If a dimension has high variance, samples spread out along it, meaning it's
  discriminative. If variance is near zero, that dimension is dead weight.

  The UVDS paper (Long et al.) found that linear projections suffer from variance decay: the projection systematically crushes the
  high-variance dimensions while leaving low-variance ones intact. This destroys information because the dimensions that matter most
  for classification are precisely the high-variance ones.

  What we found — the surprise:

  The GAN is doing an excellent job here. Almost suspiciously excellent.

  - Global variance ratio: 0.9934 — the synthetic embeddings have 99.3% of the total variance of real embeddings. Essentially
  identical.
  - Per-dimension: 127 out of 128 dimensions are within 20% of the real variance. Zero dimensions below 50% or above 150%. The worst
  dimension is at 78.6% of real — barely a deviation.
  - Information concentration is identical: both need 111/128 dims for 90% cumulative variance. The "shape" of how information is
  distributed across dimensions is preserved.
  - Variance decay severity: 0.053 — almost negligible.
  - Cosine-to-prototype: Real samples sit at mean 0.589 from their prototype, synthetic at 0.584. The GAN generates embeddings at
  essentially the same "orbital radius" around prototypes.

  Look at the top-left panel: the blue (real) and red (synthetic) curves are nearly superimposed. The variance ratio plot (top-right)
   is a noisy band centered perfectly on 1.0.

  What this means:

  Pathology 2 (variance decay) from the UVDS paper is NOT present in our GAN. The WGAN-GP critic implicitly enforces marginal
  distribution matching — the generator can't "cheat" by collapsing variance because the critic would immediately detect that the
  fake distribution is narrower than the real one. This is a fundamental advantage of adversarial training over the linear
  projections in UVDS. The Wasserstein distance is sensitive to distributional shape, not just means.

  So we can cross variance decay off our list. The UVDS-style diffusion regularisation (L_diffusion) we proposed is not needed — the
  problem it solves doesn't exist here.

  ---
  Diagnostic 2: Class-Conditional Variance

  What this measures and why:

  Diagnostic 1 looked at global variance — all samples pooled together. But that can be misleading. Imagine two classes whose
  centroids are far apart. Even if every sample within each class is identical (zero within-class variance), the global variance
  would be high because the centroids differ.

  What matters for the classifier is the within-class variance — how spread out are samples within each class? If the GAN matches
  global variance perfectly but achieves it by making all classes look alike (collapsing within-class structure), the classifier will
   fail.

  This diagnostic decomposes total variance into:
  - Within-class: average spread of samples around their class centroid
  - Between-class: spread of class centroids around the global mean

  And critically, it compares three populations:
  1. Real seen — what the GAN trained on
  2. Synthetic unseen — what the GAN produced
  3. Real unseen — the ground truth (what the GAN should be producing, but never saw)

  What we found — the first real signal:

  ┌─────────────────────────────┬───────────┬──────────────────┬─────────────────────┐
  │           Metric            │ Real seen │ Synthetic unseen │ Real unseen (truth) │
  ├─────────────────────────────┼───────────┼──────────────────┼─────────────────────┤
  │ Mean within-class var       │ 0.00498   │ 0.00480          │ 0.00547             │
  ├─────────────────────────────┼───────────┼──────────────────┼─────────────────────┤
  │ Per-class total var (trace) │ 0.638     │ 0.614            │ 0.700               │
  ├─────────────────────────────┼───────────┼──────────────────┼─────────────────────┤
  │ Within/Total ratio          │ 63.8%     │ 61.8%            │ —                   │
  └─────────────────────────────┴───────────┴──────────────────┴─────────────────────┘

  The GAN closely matches the seen distribution (synth/real_seen ratio = 0.96). But look at the ground truth comparison:
  synth/real_unseen ratio = 0.877. The synthetic unseen embeddings have 12.3% less within-class variance than real unseen embeddings
  actually have.

  Now look at the bottom-right scatter plot in the figure. Each dot is one unseen class. The x-axis is that class's real variance,
  y-axis is synthetic variance. Almost every point falls below the y=x line — meaning for nearly every unseen class, the GAN
  under-generates within-class spread.

  Also look at the top-right histogram: the green distribution (real unseen, mean=0.700) is shifted right of the red (synthetic,
  mean=0.614). The GAN learned to mimic the seen distribution's variance (blue, mean=0.638) and applied it to unseen classes — but
  unseen classes actually have more within-class variance than seen classes.

  Why does this happen?

  The GAN trains exclusively on seen-class embeddings. It learns the statistics of seen classes: "given a prototype, generate samples
   with ~0.638 per-class variance." When we condition it on unseen prototypes, it applies the same learned variance. But unseen
  classes (80 trials each, from brain_unseen) have more variance (0.700) than seen classes (8 trials each, from the 80/20 split of
  brain_seen). The GAN has no way to know this — it extrapolates seen statistics.

  This is a mild form of Pathology 1 (structural discrepancy): the GAN inherits the statistical properties of seen classes and
  applies them uniformly to unseen classes, even though the two populations differ.

  But — is this actually a problem? The within/between ratio (1.62 vs 1.76) suggests synthetic embeddings are actually slightly more
  discriminable than real ones (lower ratio = more separation). And the total effect is modest (12% variance deficit). This is a real
   signal but probably not the dominant bottleneck.

  ---
  Diagnostic 3: Prototype Neighbourhood Preservation

  What this measures and why — this is the critical one:

  Imagine 200 unseen classes as 200 points in prototype space (the image embedding space). Some classes are "neighbours" (similar
  images → nearby prototypes), others are distant. When the classifier sees a new unseen EEG embedding, it needs to decide which
  class it belongs to. It does this by checking which class's learned decision boundary the embedding falls into.

  For this to work, the geometry of synthetic embeddings must reflect the geometry of prototypes. If prototypes A and B are close
  (similar visual concepts), synthetic embeddings for A and B should also be close. If the GAN distorts this geometry — placing class
   A's synthetics near class C instead of class B — the classifier learns wrong boundaries.

  This diagnostic measures:
  - Spearman ρ: Do the rankings of pairwise distances match? (If classes 5 and 12 are the 3rd-closest pair in prototype space, are
  they also close in synthetic space?)
  - k-NN preservation: For each class, are its k nearest neighbours the same in both spaces?

  And we compare against two baselines:
  - Seen reference (real centroids vs prototypes): the upper bound — how well does the actual encoder preserve prototype geometry?
  This is the best we could possibly expect.
  - Real unseen (real unseen brain centroids vs prototypes): the ground truth — how well does the brain naturally reflect prototype
  structure for unseen classes?

  What we found — the headline result:

  ┌───────────────────────────────────────────┬────────────┬───────────┐
  │                Comparison                 │ Spearman ρ │ Pearson r │
  ├───────────────────────────────────────────┼────────────┼───────────┤
  │ Synth centroids vs Image prototypes       │ 0.893      │ 0.900     │
  ├───────────────────────────────────────────┼────────────┼───────────┤
  │ Real unseen centroids vs Image prototypes │ 0.668      │ 0.679     │
  ├───────────────────────────────────────────┼────────────┼───────────┤
  │ Synth vs Real unseen centroids            │ 0.612      │ 0.624     │
  ├───────────────────────────────────────────┼────────────┼───────────┤
  │ Seen reference (real vs proto)            │ 0.991      │ 0.992     │
  └───────────────────────────────────────────┴────────────┴───────────┘

  Read this carefully. The synthetic embeddings preserve prototype geometry better than the real unseen brain embeddings do (ρ=0.893
  vs ρ=0.668). And the direct comparison — synth vs real unseen — is only ρ=0.612.

  Now the k-NN preservation:

  ┌─────┬─────────────┬──────────────────┬──────────┐
  │  k  │ Synth↔Proto │ RealUnseen↔Proto │ Seen ref │
  ├─────┼─────────────┼──────────────────┼──────────┤
  │ 5   │ 57.7%       │ 33.6%            │ 88.4%    │
  ├─────┼─────────────┼──────────────────┼──────────┤
  │ 10  │ 64.1%       │ 39.7%            │ 89.3%    │
  ├─────┼─────────────┼──────────────────┼──────────┤
  │ 20  │ 69.8%       │ 47.7%            │ 91.6%    │
  └─────┴─────────────┴──────────────────┴──────────┘

  Look at the scatter plots. Top-left (synth vs proto): a tight linear relationship — the GAN faithfully reproduces prototype
  distances. Top-right (real unseen vs proto): a diffuse cloud — real brain signals only weakly reflect prototype structure.
  Bottom-left (seen reference): near-perfect diagonal — the encoder beautifully preserves structure for seen classes.

  What this means — and this is the key insight:

  The GAN is doing too good a job preserving prototype structure. It generates synthetic embeddings that are essentially "projected
  prototypes with noise added" — the inter-class distances in synthetic space are 89.3% correlated with prototype distances. But real
   brain signals for unseen classes only correlate at 67.9% with prototype distances.

  This is Pathology 1 (structural discrepancy) made concrete. The synthetic distribution lives in a different geometric regime than
  the real unseen distribution:

  - Synthetic unseen: tightly governed by prototype geometry (ρ=0.893 to prototypes)
  - Real unseen: loosely governed by prototype geometry (ρ=0.668 to prototypes)
  - Direct overlap: only ρ=0.612

  The classifier trains on synthetic data that's "too clean" — too faithful to prototype structure. When it encounters real unseen
  brain embeddings, which are noisier and less structure-preserving, the learned decision boundaries are wrong.

  And the seen reference (ρ=0.991) tells us why the seen accuracy is higher: for seen classes, the encoder does preserve prototype
  structure beautifully, and the classifier trains on real seen data that has this structure. There's no distribution mismatch for
  seen classes.

  ---
  The Diagnosis Summary

  ┌────────────────────┬────────────────────┬────────────────────────────────────────────────────────────────────────────────────┐
  │     Pathology      │       Status       │                                      Severity                                      │
  ├────────────────────┼────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ 1. Structural      │ CONFIRMED          │ High — synth-to-proto ρ=0.893 vs real-to-proto ρ=0.668. The GAN generates          │
  │ discrepancy        │                    │ embeddings in a geometrically different regime than real brain signals.            │
  ├────────────────────┼────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ 2. Variance decay  │ ABSENT             │ None — per-dim variance matched to 99.3%, 127/128 dims within 20%.                 │
  ├────────────────────┼────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ 3. Seen-class bias │ CONFIRMED (from    │ Moderate — within-class variance calibrated to seen (0.638) not unseen (0.700)     │
  │                    │ prior work)        │ statistics.                                                                        │
  └────────────────────┴────────────────────┴────────────────────────────────────────────────────────────────────────────────────┘

  The dominant pathology is clear: the GAN's synthetic embeddings are too faithful to prototype geometry. Real brain signals are
  noisy and only partially encode visual semantics (ρ≈0.67 to prototypes). But the GAN, conditioned directly on prototypes, generates
   samples that are ρ≈0.89 correlated with that prototype structure. The classifier learns decision boundaries tuned for this "clean"
   geometry, then fails when confronted with the "messy" reality of actual brain signals.

  This completely reframes what the neighbourhood-consistency loss should do. Our original proposal was to improve structure
  preservation — make synthetic embeddings more faithful to prototype geometry. But the data says the opposite: we need to inject
  realistic noise that degrades the prototype-geometry correlation from 0.89 down toward 0.67, to match what real brain signals
  actually look like.

  This is a precise, actionable target. Want to continue with the mathematical formulation of what this "realistic degradation" loss
  should look like?

The Problem, Stated Precisely

  We have a generator G(z, s_c) that takes noise z ∈ ℝ¹⁰⁰ and a prototype s_c ∈ S^63 (the unit sphere in ℝ⁶⁴) and outputs a synthetic
   embedding ê ∈ S^63.

  The diagnostics revealed:

  The generator's conditional distribution p_G(ê | s_c) is too tightly coupled to s_c.

  What does "too tightly coupled" mean mathematically? Let's define it. Consider two unseen classes i, j with prototypes s_i, s_j.
  Define:

  - d_proto(i,j) = ‖s_i − s_j‖_cos (cosine distance between prototypes)
  - d_synth(i,j) = ‖μ̂_i − μ̂_j‖_cos (cosine distance between synthetic centroids, where μ̂_i = 𝔼[G(z, s_i)])
  - d_real(i,j) = ‖μ_i − μ_j‖_cos (cosine distance between real unseen brain centroids)

  The diagnostics measured:

  - ρ(d_proto, d_synth) = 0.893 — the GAN almost perfectly mirrors prototype geometry
  - ρ(d_proto, d_real) = 0.668 — real brain signals only loosely mirror prototype geometry
  - ρ(d_synth, d_real) = 0.612 — the two distributions are geometrically misaligned

  The classifier trains on synthetic data where inter-class geometry is governed by d_synth, but at test time encounters data
  governed by d_real. This is a domain shift in the geometric structure of the embedding space.

  ---
  What Causes This? The Generator's Architecture

  Let's trace exactly how the generator processes its inputs. G(z, s_c) concatenates [z; s_c] ∈ ℝ¹⁶⁴ and passes it through:

  h₁ = LeakyReLU(W₁[z; s_c] + b₁)     ∈ ℝ²⁵⁶
  h₂ = LeakyReLU(W₂h₁ + b₂)            ∈ ℝ²⁵⁶
  ê_raw = W₃h₂ + b₃                      ∈ ℝ⁶⁴
  ê = ê_raw / ‖ê_raw‖₂                   ∈ S⁶³

  Now decompose the first layer's operation. W₁ ∈ ℝ²⁵⁶ˣ¹⁶⁴ can be split into two blocks:

  W₁[z; s_c] = W₁ᶻz + W₁ˢs_c

  where W₁ᶻ ∈ ℝ²⁵⁶ˣ¹⁰⁰ acts on noise and W₁ˢ ∈ ℝ²⁵⁶ˣ⁶⁴ acts on the prototype. The first hidden activation is:

  h₁ = LeakyReLU(W₁ᶻz + W₁ˢs_c + b₁)

  Here's the key insight. After training, the network learns to treat s_c as a strong positional signal — it tells the generator
  where in embedding space to place the output. The noise z provides variation around that location. But the WGAN-GP critic only
  enforces that the marginal distribution p_G(ê | s_c) matches p_real(e | c) for each seen class c — it doesn't constrain the
  inter-class geometry.

  Why does the output mirror prototype geometry so faithfully? Because the generator is a smooth, continuous function of s_c. If s_i
  and s_j are close on the unit sphere, then W₁ˢs_i ≈ W₁ˢs_j, which means h₁ is similar for both, which propagates through the
  network to make G(z, s_i) ≈ G(z, s_j) for the same z. The generator is approximately Lipschitz-continuous in s_c — small changes in
   prototype → small changes in output. This is an inherent property of MLPs with bounded weights.

  So the correlation ρ(d_proto, d_synth) ≈ 0.89 is a structural consequence of the generator's architecture. It can't help but
  preserve prototype geometry.

  But real brain signals don't work this way. The EEG encoder maps noisy 561-D brain signals through a separate MLP (BrainEncoder),
  and the resulting geometry is only ρ ≈ 0.67 correlated with prototype geometry. The brain introduces its own distortions — some
  visually similar objects produce dissimilar brain responses, and vice versa.

  ---
  What Should the Loss Function Do?

  We need a loss term that breaks the generator's excessive fidelity to prototype geometry. Specifically, we want to transform:

  ρ(d_proto, d_synth) : 0.893 → ~0.67

  while keeping individual class distributions healthy (variance, norms, etc.).

  There are two conceptual approaches:

  Approach A: Explicit Geometric Degradation

  Directly penalise the generator when its inter-class distances correlate too strongly with prototype distances.

  Approach B: Distribution Matching in Geometry Space

  Train the generator to produce embeddings whose geometric statistics match those of the real seen embeddings (which have the
  "right" amount of geometric noise).

  Approach B is more principled because it doesn't require knowing the target ρ a priori — it learns it from data. Let's develop
  both, starting with A for intuition, then moving to B.

  ---
  Approach A: Geometric Decorrelation Loss

  The Setup

  During WGAN-GP training, at each generator step we have a batch of real embeddings {e_i} with conditions {s_cᵢ}. We generate fake
  embeddings {ê_i} = G(z_i, s_cᵢ). Within this batch, there are typically samples from multiple classes.

  Consider all pairs of samples (i, j) in the batch that belong to different classes (c_i ≠ c_j). For each pair, compute:

  - Prototype distance: δ_ij^proto = 1 − s_cᵢᵀs_cⱼ (cosine distance, since prototypes are unit-normed)
  - Synthetic distance: δ_ij^synth = 1 − êᵢᵀêⱼ

  If the generator perfectly mirrors prototype geometry, then δ^synth ≈ f(δ^proto) for some monotonic function f. We want to penalise
   this.

  The Naive Penalty (and Why It Fails)

  You might try:

  L_decorr = |Corr(δ^proto, δ^synth)|²

  This penalises correlation directly. But this is terrible for two reasons:

  1. Correlation is a batch-level statistic — it's computed over all pairs in the batch. With batch_size=256, there are ~32K pairs.
  The gradient ∂L_decorr/∂θ_G requires backpropagating through a correlation computation over 32K pairs, which is expensive and has
  high variance.
  2. It doesn't tell the generator how to decorrelate. Should it push nearby classes apart? Pull distant classes together? Randomly
  scramble everything? The gradient provides no useful direction.

  A Better Formulation: Pairwise Distance Noise

  Instead of penalising correlation, we inject structured noise into the generator's perception of inter-class distances. Define a
  perturbation loss for each pair:

  L_pair(i,j) = (δ_ij^synth − δ_ij^proto)²

  This forces synthetic distances to match prototype distances — which is what the current generator already does. But now replace
  the target with a noisy version of prototype distances:

  δ_ij^target = δ_ij^proto + ε_ij

  where ε_ij is calibrated noise. The loss becomes:

  L_pair(i,j) = (δ_ij^synth − δ_ij^proto − ε_ij)²

  But this is ad hoc — we're injecting arbitrary noise. We want the noise to be realistic, matching how real brain signals distort
  prototype geometry.

  ---
  Approach B: Pairwise Distance Distribution Matching (the principled version)

  This is the one I'd actually implement. Here's the idea:

  Step 1: Characterise the "Real" Geometry

  From the seen training data, we can measure exactly how the encoder distorts prototype geometry. For each pair of seen classes (a,
  b):

  - d_proto(a,b) = cosine distance between prototypes s_a, s_b
  - d_real(a,b) = cosine distance between real brain centroids μ_a, μ_b

  The relationship d_real vs d_proto is a scatter cloud, not a line. We can characterise it by its conditional distribution:

  p(d_real | d_proto) — "given two prototypes at distance δ, what's the distribution of distances between their real brain
  centroids?"

  In practice, we bin d_proto into intervals and compute the mean and variance of d_real within each bin. Let:

  - μ_dist(δ) = 𝔼[d_real | d_proto = δ] — the expected real distance given prototype distance δ
  - σ_dist(δ) = Std[d_real | d_proto = δ] — the noise in that mapping

  For seen classes, from the diagnostics we know ρ(d_proto, d_real) ≈ 0.991. This means μ_dist(δ) ≈ α·δ + β (nearly linear) with very
   low σ_dist. But this is for seen classes — the encoder was trained on them.

  For unseen classes, ρ drops to 0.668, meaning σ_dist is much larger. The brain introduces substantial geometric noise for classes
  the encoder wasn't trained on.

  Step 2: The Loss Function

  We want the generator's pairwise distances to match the real unseen distribution, not the prototype distribution. But we can't use
  real unseen data (that's test data). So we use seen data as a proxy and add a controlled degradation factor.

  Here's the concrete formulation. During training, for each generator step, sample a minibatch of pairs {(ê_i, s_cᵢ), (ê_j, s_cⱼ)}
  where c_i ≠ c_j. Compute:

  $$L_{\text{geo}} = \frac{1}{|P|} \sum_{(i,j) \in P} \left( d(\hat{e}_i, \hat{e}j) - \tilde{d}{ij} \right)^2$$

  where P is the set of cross-class pairs in the minibatch, d(·,·) is cosine distance, and d̃_ij is the target distance — sampled from
   the empirical distribution of real brain centroid distances for seen classes with similar prototype distances.

  In practice, d̃_ij is computed by:

  1. Look up δ = d_proto(c_i, c_j) — the prototype distance for this pair
  2. Find all seen-class pairs (a,b) with similar prototype distance: |d_proto(a,b) − δ| < ε
  3. Sample one of those pairs' real distances d_real(a,b) as the target

  This is elegant but expensive. A simpler approximation:

  Step 3: The Practical Loss — Noise-Augmented Prototype Conditioning

  Instead of modifying the loss function, modify the conditioning input. Before passing s_c to the generator, perturb it:

  s̃_c = normalise(s_c + η · ξ), where ξ ~ 𝒩(0, I_d)

  where normalise(v) = v/‖v‖₂ projects back onto the unit sphere, and η controls the perturbation magnitude.

  Visualise this: s_c is a point on the 63-dimensional unit sphere. Adding Gaussian noise and renormalising moves it to a nearby
  point on the sphere. The generator now sees a "blurred" version of the prototype, which naturally degrades its ability to preserve
  precise inter-class distances.

  The beauty of this approach is:

  1. No new loss term — we modify the input, not the objective
  2. Differentiable — the generator still trains normally via WGAN-GP
  3. Calibratable — η directly controls the degree of geometric degradation
  4. Geometrically interpretable — the noise magnitude η maps to an angular perturbation on S^63

  The Geometry of Spherical Perturbation

  Let me make this precise. If s_c ∈ S^(d-1) and ξ ~ 𝒩(0, I_d), then the perturbed direction is:

  s̃_c = (s_c + ηξ) / ‖s_c + ηξ‖

  The angular deviation θ between s_c and s̃_c satisfies:

  cos θ = s_cᵀs̃_c

  For small η, we can expand:

  cos θ ≈ 1 / √(1 + η²‖ξ_⊥‖²)

  where ξ_⊥ is the component of ξ orthogonal to s_c (which has dimension d−1 = 63 and ‖ξ_⊥‖² ~ χ²₆₃ with mean 63).

  So:

  𝔼[cos θ] ≈ 1 / √(1 + 63η²)

  For η = 0.1: 𝔼[cos θ] ≈ 0.62, meaning average angular deviation ≈ 52°
  For η = 0.05: 𝔼[cos θ] ≈ 0.84, meaning average angular deviation ≈ 33°
  For η = 0.02: 𝔼[cos θ] ≈ 0.95, meaning average angular deviation ≈ 18°

  We can calibrate η by matching the observed prototype-to-centroid correlation. Currently:
  - Synth ρ to proto: 0.893 (too high)
  - Real ρ to proto: 0.668 (target)
  - Gap: 0.225

  A moderate η (somewhere around 0.05–0.1) should degrade the correlation by the right amount. The exact value can be determined
  empirically — or better, derived from the seen-class statistics.

  Deriving η from Data

  For seen classes, we know the encoder maps brain signals to embeddings with ρ(d_real, d_proto) = 0.991. This means the encoder
  introduces very little geometric noise for seen classes. But for unseen classes, ρ drops to 0.668 — the encoder was never trained
  on these classes, so its mapping is noisier.

  The noise η should simulate this "unseen-class degradation." We can estimate it as:

  1. For each seen class c, compute the angular deviation between the prototype s_c and the real brain centroid μ_c: θ_c =
  arccos(s_cᵀμ̂_c) where μ̂_c = μ_c/‖μ_c‖
  2. The distribution of θ_c over seen classes gives us the "seen noise level"
  3. The unseen noise level is higher (because the encoder wasn't trained on unseen classes)
  4. From the cosine-to-prototype stats: real seen mean cos(θ) = 0.589, synthetic unseen mean cos(θ) = 0.584
  5. At the centroid level, the seen reference ρ=0.991 implies very tight centroids; the unseen ρ=0.668 implies loose centroids

  The factor by which ρ degrades from seen (0.991) to unseen (0.668) is 0.668/0.991 = 0.674. We want our perturbation to achieve a
  similar degradation from the current synth ρ=0.893: target ≈ 0.893 × 0.674 ≈ 0.602 (which is close to the actual real-unseen ρ of
  0.668 — good sanity check).

  ---
  How to Visualise This

  Think of the 200 unseen prototypes as 200 stars in a night sky (projected onto the surface of a sphere). The current generator,
  given a star's position s_c, places synthetic embeddings in a tight cloud around that exact star. The inter-star distances in
  synthetic space are almost identical to the real star map — ρ = 0.893.

  But real brain signals are like viewing those stars through a turbulent atmosphere. Each star's apparent position wobbles —
  sometimes a lot. The average apparent position of each star is close to its true position, but the wobble means that the perceived
  distance between two stars is noisy. Two stars that are close might look far apart on a given night, and vice versa. That's why ρ
  drops to 0.668.

  The noise-augmented conditioning is like adding the same atmospheric turbulence to the generator's "star map." Instead of giving it
   the precise star position s_c, we give it a wobbled version s̃_c. The generator then places its synthetic cloud around the wobbled
  position. Over many samples, the synthetic inter-class distances will have the same noise profile as real brain signals.

  ---
  The Two-Part Strategy

  Based on all of this, I'd propose:

  1. Noise-augmented prototype conditioning (primary mechanism) — perturb s_c before feeding to the generator, calibrated by η
  derived from seen-class statistics
  2. Within-class variance scaling (secondary correction) — the GAN under-generates within-class variance by ~12%. A simple
  multiplicative correction during synthesis: multiply synthetic embeddings' deviation from their centroid by √(0.700/0.614) ≈ 1.068,
   then renormalise

  Strategy 1 addresses the dominant pathology (structural discrepancy). Strategy 2 addresses the mild variance deficit from
  Diagnostic 2. Together, they should align the synthetic distribution closer to what real unseen brain embeddings actually look
  like.

  Shall we proceed to implementation, or do you want to dig deeper into any of the mathematical steps — for instance, the exact
  calibration of η, or the gradient flow through the noise-augmented conditioning?