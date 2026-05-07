# Phase F: Augmentation + SupCon + Architecture Analysis
**Date**: March 2026
**Based on**: Results from `context/phasef_results.md` and Phase F Figures

## 1. Executive Summary: The Regularisation Paradox
The results of Phase F reveal a catastrophic, yet highly informative, breakdown in the representation pipeline. In attempting to cure the "shortcut learning" identified in Phase E by applying heavy regularisation (Data Augmentation, Contrastive Loss, and inductive biases via CNNs), the **Encoder Top-1 accuracy collapsed by 1 order of magnitude**, dropping from **2.87% (Phase E)** to **0.24% (Phase F)**. 

Paradoxically, despite the encoder strictly losing classification power on the *seen* classes, both the **GZSL H-mean** and the **Routing Rate** more than doubled (H-mean from 0.70% -> 1.66%, Routing from 11.31% -> 46.41%). 

I'll break down scientifically why these results occurred and why this is a massive red flag that we need to address.

## 2. Stage-by-Stage Breakdown

### Stage F1: Data Augmentation
- **Control**: The `no_aug` config yielded 0.03% top-1 natively in this context, demonstrating that the setup was extremely sensitive.
- **Winner**: `F1-C: channel_drop` with a 15% dropout probability won with a meager 0.24% top-1 accuracy. Gaussian noise (0.21%) and temporal shift (0.18%) were slightly behind.
- **Combinations Failed**: Combining top augmentations (e.g., `F1-F: combo_top2`) yielded worse results (0.15%), implying that the signal-to-noise ratio in the EEG data is so fragile that stacking augmentations obliterates the underlying class-invariant structure.

### Stage F2: Supervised Contrastive Loss
- **Pure SupCon Failed**: `F2-I: supcon_only` yielded a dismal 0.06% top-1. Supervised contrastive learning heavily relies on having distinct "positive" pairs. In EEG, if trials from the same class look indistinguishable from noise, the loss forces the network to pull pure noise closer together, collapsing the representation space.
- **Combined Losses**: Reintroducing InfoNCE alongside SupCon (`F2-J` and `F2-K`) restored accuracy back to the ~0.2% range, strictly because the InfoNCE text-anchor projection is the only thing providing structural gradients. 

### Stage F3: Architecture Sweeps
- **MLP > CNNs**: The deep `mlp_ln` out-performed every single spatiotemporal architecture (ShallowConvNet: 0.12%, EEGNet-Small: 0.09%, EEGNet-Large: 0.06%). 
- **Why?**: Architectures like EEGNet introduce strong inductive biases built for event-related potentials (ERPs) or specific spectral bands. Our current brain-to-text semantic mapping task seems wildly decoupled from the features EEGNet naturally extracts. An MLP forces fewer constraints, allowing the network to lazily map whatever latent noise it finds to the embedding space.

## 3. The Final Comparison: Explaining the Paradox
When we look at the final outputs (Cell 10):
* **Phase E**: AccS=2.96%, AccU=0.40%, H=0.70%, Routing=11.31%
* **Phase F**: AccS=2.45%, AccU=1.25%, H=1.66%, Routing=46.41%

Why did `AccU` and `H-mean` go *up* when the internal encoder representations completely cratered (2.87% -> 0.24%)?

**The "Blob" Effect**
1. By regularising the encoder into oblivion via Channel Dropout + SupCon, we forced the embeddings into a tightly packed, featureless "blob" in the latent space. The network could no longer overfit to individual seen classes.
2. Because the seen prototypes and synthetic unseen prototypes are all packed closely together (due to the collapsed representation space), the classifier boundaries are extremely close.
3. The WGAN-GP generates synthetic unseen samples that are virtually indistinguishable from the seen samples. 
4. Therefore, when evaluating $X_{test}$, the classifier essentially acts like a random router. It guesses randomly between seen and unseen classes, which pushes the **Routing Rate** massive high (from 11% to 46.41% — nearly a 50/50 flip).
5. By guessing "Unseen" more often, Unseen Accuracy (`AccU`) naturally increases due to chance. The harmonic mean (`H-mean`) mathematically penalises extreme imbalances. By forcing the predictions closer to random noise, we balanced `AccS` and `AccU` artificially, raising the `H-mean` without actually learning meaningful semantic separation.

## 4. Conclusion and Next Steps
The Phase F experiments proved that the current bottleneck is **not** a lack of regularisation or standard overfitting, but rather a **fundamental lack of semantic signal in the embeddings**. 

We cannot regularise our way to better GZSL. The EEG representations simply do not contain the necessary structure to distinguish between 1654 distinct image concepts using our current pipeline. 

We must pivot away from local hyperparameter and architecture sweeps and address the core issue: the raw semantic mapping between the EEG input (17 channels, 33 time steps) and the CLIP textual text space is either fundamentally misaligned or mathematically underdetermined in its current formulation. We likely need to investigate exactly what the text projections look like or re-prioritise phase synchronization/bandpower feature extraction before throwing it into contrastive loops.
