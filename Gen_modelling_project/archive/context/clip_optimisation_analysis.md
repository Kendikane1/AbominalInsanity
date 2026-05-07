# CLIP Encoder Optimisation Analysis
**Date**: March 2026
**Based on**: Results from `context/CLIP_optimisation_results.md` and Phase 2 Figures

## 1. Summary Verdict
The systematic two-stage parametric sweep successfully identified a vastly superior training recipe and architecture for the CLIP Brain Encoder, increasing the relative Top-1 accuracy by **51%** (from 1.90% to 2.87%).

However, in absolute terms, **a 2.87% Top-1 accuracy across 1654 classes remains critically insufficient for any meaningful downstream generative task.** The features extracted by the MLP encoder from the raw EEG data are still fundamentally overlapping and unstructured. The contrastive loss successfully optimizes, but the underlying neural network architecture (a flat MLP) lacks the inductive biases necessary to extract highly discriminable, class-invariant semantic signals from noisy, continuous time-series EEG data.

---

## 2. Detailed Findings

### Stage 1: Training Dynamics (Dim = 64)
The baseline model (20 epochs, flat LR $1e{-4}$, fixed $\tau=0.07$) achieved 1.90%.
* **Epochs**: Simply training longer (R1-C, 200 epochs) actually *hurt* performance (1.42%), indicating the model was overfitting to the noise in the EEG data rather than learning generalizable semantics.
* **Learning Rate Schedule**: Adding a Cosine Annealing scheduler with linear warmup (R2-F) and raising the peak learning rate to $1e{-3}$ allowed the model to escape local minima, boosting accuracy to 2.12%.
* **Temperature ($\tau$)**: Softening the contrastive temperature penalty from 0.07 to 0.15 (R3-H) yielded the best Stage 1 result (2.21%). A softer temperature reduces the penalty applied to "hard negatives", which is crucial here because EEG data is incredibly noisy and classes naturally overlap heavily in the raw feature space.

### Stage 2: Architecture & Dimensionality
* **Dimensionality**: Increasing `embed_dim` from 64 to 128 (R4-I) provided more representational capacity to separate classes in the latent space, boosting accuracy to 2.36%. Pushing it further to 256 (R4-J) caused a collapse (1.96%), likely due to the curse of dimensionality or overfitting given the limited dataset size.
* **Architecture**: Adding `LayerNorm` (R5-K) was the single biggest architectural win, jumping performance to **2.87%**. Layer normalization prevents internal covariate shift, smoothing the very rough loss landscape inherent to InfoNCE optimization on physiological data. The `deep` and `residual` variants performed worse, suggesting the model lacks the appropriate inductive biases (e.g., temporal convolution) rather than simply needing more depth.
* **Batch Size**: Increasing batch size to 512 (R6-N) hurt performance (2.48%). While larger batches provide more negative samples for InfoNCE, they might have smoothed the gradient too much, preventing the model from traversing the noisy loss landscape effectively.

### Final Pipeline Breakdown
With the optimized encoder (LayerNorm, dim=128, LR=$1e{-3}$ with cosine warmup, $\tau=0.15$), the full GZSL pipeline was re-run:
* **Encoder Top-1**: Improved from 1.90% $\rightarrow$ 2.87%
* **Synth $\rightarrow$ Real Transfer**: Improved from 1.73% $\rightarrow$ 2.51%
* **AccS**: 2.96%
* **AccU**: 0.40%
* **H-mean**: Improved from 0.12% $\rightarrow$ **0.70%**

---

## 3. Scientific Conclusion & Why It is Still Failing

The optimization sweep did exactly what it was supposed to do: it found the absolute mathematical ceiling of the current approach. 

The fact that an extensively optimized architecture and training schedule tops out at $\approx 3\%$ accuracy points to a fundamental limitation in feature extraction. **The MLP architecture is destroying the temporal structure of the EEG.** 

By treating the 561-dimensional EEG trial array as a flat, un-ordered vector of features, the MLP throws away the sequence, phase, and frequency dynamics that encode cognitive representations. InfoNCE is trying to pull matching elements together, but the MLP is incapable of separating the massive subject-to-subject and trial-to-trial variance from the subtle class-to-class semantic variance.

### Next Steps Recommendation
To achieve the high (>20%) Top-1 accuracy required for the GAN to synthesize distinct distributions, Phase 3 must abandon the flat MLP Brain Encoder.

We must implement an architecture designed for physiological time-series data:
1. **1D-CNN (Temporal Convolutional Network)**: To extract local temporal features and frequency-band power envelopes.
2. **LSTM / GRU**: To capture the sequential evolution of the brain wave over the duration of the trial.
3. **Transformer Encoder**: To use self-attention across the time steps of the EEG trial.

Until the raw EEG representations are structurally disentangled by a temporal-aware encoder, downstream GZSL performance will remain negligible.
