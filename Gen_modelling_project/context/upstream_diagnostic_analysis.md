# Upstream Bottleneck Analysis
**Date**: March 2026
**Based on**: Results from `context/Upstream_diagnostic_results.md` and `figures/upstream_diag_c_prototypes.png`

## 1. Summary Verdict
The **primary, gating bottleneck** in the GZSL pipeline is the **CLIP Brain Encoder**. 

While both the encoder and generator currently demonstrate poor downstream task performance (AccU ~ 0.2%), the generator is actually performing its mathematical duty correctly. The generator is failing on real data *because the target distribution provided by the CLIP encoder is highly diffuse, noisy, and overlapping*. Until the CLIP encoder can produce class-discriminable embeddings from raw EEG, no downstream generative model or classifier will succeed.

---

## 2. Detailed Diagnostic Breakdown

### Diagnostic C: Text Prototypes (The Semantic Space)
* **Finding**: The unseen text prototypes exhibit a mean pairwise cosine similarity of **0.0135** and a standard deviation of **0.1569**. The nearest-neighbor cosine mean is **0.4848**. Visually, the heatmap shows no large highly-correlated blocks, and the pairwise distribution tightly centers around 0, matching the theoretical expectation for random unit vectors in $R^{64}$ ($E[\cos] \approx 0$).
* **Conclusion**: **The prototypes are perfectly healthy.** The text projection space provides an excellent, nearly orthogonal scaffolding for 200 unseen classes. The bottleneck is absolutely not in the linguistic/semantic domain.

### Diagnostic A: CLIP Brain Encoder (The Representation Space)
* **Finding**: A linear classifier trained purely on real, seen EEG embeddings ($E_{train\_seen}$) achieves only **1.90% Top-1 Accuracy** (against a random baseline of 0.06%) across 1654 classes. 
* **Conclusion**: **Critical Weakness.** While the signal ratio is 31.5x above random, an absolute accuracy of 1.90% proves that the CLIP encoder is failing to map raw 561-D EEG trials into tight, easily separable class clusters. The embeddings of different classes heavily overlap in the 64-D space. Contrastive learning aims to pull samples of the same class together and push different classes apart; this collapse indicates the InfoNCE loss has not successfully driven this separation.

### Diagnostic B: cWGAN-GP (The Generative Space)
* **Finding (Internal CV - B2)**: The generator achieves **68.38%** accuracy when evaluated strictly on its own synthetic outputs.
* **Finding (Transfer - B1)**: When trained on synthetics and tested on true real unseen embeddings, accuracy plummets to **1.73%**.
* **Conclusion**: The GAN is **not the root cause**, but a victim of the encoder. The 68% internal accuracy proves the GAN successfully learns to condition on the text sequence ($s_c$) and generate distinct points in $R^{64}$. It is effectively creating separated, distinct "blobs" for each class. However, because the true real unseen EEG data (encoded by CLIP) is a massive, overlapping cloud (as proven by Diag A), the tight synthetic blobs fail to align with the true decision boundaries, resulting in the 1.73% transfer accuracy. 

---

## 3. Scientific Reasoning & Next Steps

The GAN is learning essentially mode-collapsed, low-variance representations for each class prototype because the real embeddings it was trained on lack a strong, structured manifold. To fix the pipeline, we must drastically improve the **CLIP Brain Encoder**'s ability to extract class-invariant neural signatures from the EEG data.

### Recommendations for Encoder Improvement (Phase 2 Focus)
1. **Extended Training Duration (Epochs)**: 20 epochs is generally insufficient for contrastive learning on highly noisy physiological data. We should increase this substantially (e.g., 100-200 epochs).
2. **Learning Rate Scheduling**: InfoNCE loss optimization is notoriously unstable. Implementing a **Cosine Annealing with Linear Warmup** scheduler will help the model escape early local minima and settle into better representations.
3. **Dimensionality Increase**: $R^{64}$ might be a restrictive bottleneck for 1654 orthogonal classes. Expanding `embed_dim` to **128 or 256** could provide the necessary degrees of freedom for the representations to un-tangle.
4. **Learnable Temperature ($\tau$)**: If currently fixed at 0.07, making the contrastive temperature parameter learnable allows the model to dynamically scale the penalty for hard negatives, which is crucial when class boundaries overlap heavily.
5. **Model Capacity**: The MLP depth or hidden layer width of $f_b$ might need an increase if it lacks the non-linear capacity to project raw EEG features into the shared semantic space.

**Verdict**: We must halt downstream GAN/calibration fixes and initiate a **CLIP Encoder Optimization Phase**.
