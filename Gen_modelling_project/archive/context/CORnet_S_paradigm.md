Output of cells (most) starting from CLIP cells, look through carefully

cell 1:
============================================================
ENCODER CONFIGURATION (Image Alignment — CORnet-S)
============================================================
  embed_dim: 128
  image_input_dim: 1000
  tau: 0.15
  epochs: 50
  batch_size: 256
  lr: 0.001
  weight_decay: 0.0001
  dropout: 0.1
  schedule: cosine_warmup
  warmup_ratio: 0.1
  seed: 42
  alignment_target: image (CORnet-S)
  device: cpu
============================================================

cell 2:
Encoder Training data: 13232 samples
  Brain features: 561-D (StandardScaled)
  Image features: 1000-D (CORnet-S, raw)
Encoder Test data (seen): 3308 samples
Unseen data (test only): 16000 samples

cell 3:
Training Brain-Image contrastive encoder...
Epochs: 50, Batches per epoch: 52
Schedule: cosine warmup (260 warmup steps / 2600 total)
Alignment target: image (CORnet-S)
  Epoch   1/50: Loss = 5.3107, LR = 0.000200
  Epoch  10/50: Loss = 1.5348, LR = 0.000970
  Epoch  20/50: Loss = 1.1615, LR = 0.000750
  Epoch  30/50: Loss = 1.0235, LR = 0.000413
  Epoch  40/50: Loss = 0.9561, LR = 0.000117
  Epoch  50/50: Loss = 0.9342, LR = 0.000000

Brain-Image encoder training complete!
Final loss: 0.9342

cell 4:
Brain embeddings:
  E_train_seen: (13232, 128)
  E_test_seen: (3308, 128)
  E_unseen: (16000, 128)
Image embeddings:
  V_train_embeds: (13232, 128)
  V_unseen_embeds: (16000, 128)

cell 5:
Computing t-SNE (this may take a moment)...
/usr/local/lib/python3.12/dist-packages/sklearn/manifold/_t_sne.py:1164: FutureWarning: 'n_iter' was renamed to 'max_iter' in version 1.5 and will be removed in 1.7.
  warnings.warn(

Saved: figures/clip_embedding_tsne.png

cell 6:
============================================================
cWGAN-GP CONFIGURATION
============================================================
  z_dim: 100
  embed_dim: 128
  lr: 0.0001
  betas: (0.0, 0.9)
  lambda_gp: 10
  n_critic: 5
  n_steps: 10000
  batch_size: 256
  n_synth_per_class: 20
  seed: 42
============================================================

cell 7:
Training cWGAN-GP for 10000 generator steps...
Critic updates per G step: 5
  Step     1/10000: G_loss=0.0321, C_loss=8.0184, GP=0.8034
  Step  1000/10000: G_loss=-0.1065, C_loss=-0.0368, GP=0.0009
  Step  2000/10000: G_loss=0.6925, C_loss=-0.0502, GP=0.0013
  Step  3000/10000: G_loss=0.9207, C_loss=-0.0524, GP=0.0014
  Step  4000/10000: G_loss=0.8767, C_loss=-0.0431, GP=0.0011
  Step  5000/10000: G_loss=0.7974, C_loss=-0.0393, GP=0.0014
  Step  6000/10000: G_loss=0.7923, C_loss=-0.0346, GP=0.0012
  Step  7000/10000: G_loss=0.7436, C_loss=-0.0343, GP=0.0015
  Step  8000/10000: G_loss=0.6997, C_loss=-0.0523, GP=0.0013
  Step  9000/10000: G_loss=0.6415, C_loss=-0.0415, GP=0.0011
  Step 10000/10000: G_loss=0.6014, C_loss=-0.0370, GP=0.0012

cWGAN-GP training complete!
Final G_loss: 0.6014
Final C_loss: -0.0370

cell 8:
Generating 20 synthetic embeddings per unseen class...
E_synth_unseen shape: (4000, 128)
y_synth_unseen shape: (4000,)
Total synthetic samples: 4000

cell 9:
============================================================
SYNTHETIC EMBEDDING QUALITY CHECK
============================================================
Synthetic embedding norms: mean=1.0000, std=0.000000
Mean per-dim variance: Real=0.0078, Synthetic=0.0078
Cosine sim (synth vs proto): mean=0.5612, std=0.1788

✓ Synthetic embeddings have reasonable variance (no mode collapse).

cell 10:
Computing t-SNE for real vs synthetic embeddings...
/usr/local/lib/python3.12/dist-packages/sklearn/manifold/_t_sne.py:1164: FutureWarning: 'n_iter' was renamed to 'max_iter' in version 1.5 and will be removed in 1.7.
  warnings.warn(

Saved: figures/real_vs_synth_tsne.png

cell 11:
======================================================================
BRAIN-IMAGE ENCODER + cWGAN-GP IMPLEMENTATION SUMMARY
======================================================================

### BRAIN-IMAGE ENCODER ###
  Embedding dimension: 128
  Training epochs: 50
  Final contrastive loss: 0.9342
  Embeddings: E_train_seen (13232, 128), E_test_seen (3308, 128), E_unseen (16000, 128)
  Prototypes: S_seen (1654, 128), S_unseen (200, 128)

### cWGAN-GP ###
  Generator steps: 10000
  Final G_loss: 0.6014
  Final C_loss: -0.0370
  Synthetic per class: 20
  Total synthetic samples: 4000

### CACHED ARRAYS ###
  - cached_arrays/E_synth_unseen.npy
  - cached_arrays/E_test_seen.npy
  - cached_arrays/E_train_seen.npy
  - cached_arrays/E_unseen.npy
  - cached_arrays/S_seen_prototypes.npy
  - cached_arrays/S_unseen_prototypes.npy
  - cached_arrays/V_train_embeds.npy
  - cached_arrays/V_unseen_embeds.npy
  - cached_arrays/seen_classes.npy
  - cached_arrays/unseen_classes.npy
  - cached_arrays/y_synth_unseen.npy
  - cached_arrays/y_test_seen.npy
  - cached_arrays/y_train_seen.npy
  - cached_arrays/y_unseen.npy

### FIGURES GENERATED ###
  - figures/encoder_loss_curve.png
  - figures/clip_embedding_tsne.png
  - figures/wgan_losses.png
  - figures/real_vs_synth_tsne.png

======================================================================
NEXT STEPS: Train GZSL Classifier [A+B] on real + synthetic embeddings
======================================================================

cell 12:
Seen training: 1654 classes, median=8/class, mean=8.0/class
Unseen synthetic (before): 200 classes, 20/class, 4000 total
Unseen synthetic (after):  1600 total (8/class, was 20/class)

GZSL Training Data (balanced):
  Real seen:        13232 samples, 1654 classes
  Synth unseen (ds): 1600 samples, 200 classes
  Combined:         14832 samples, 1854 classes

Label overlap check: 0 overlapping labels
  No overlap -- seen and unseen label spaces are disjoint.

cell 13:
======================================================================
COMPARISON: BASELINE [A] vs GZSL [A+B]
======================================================================

Metric                  Baseline [A]      GZSL [A+B]     Improvement
----------------------------------------------------------------------
Acc (seen)                    0.0181          0.0375         +0.0193
Acc (unseen)                  0.0000          0.0573         +0.0573
Harmonic Mean (H)             0.0000          0.0453         +0.0453
F1 (seen)                     0.0162          0.0225         +0.0063
F1 (unseen)                   0.0000          0.0120         +0.0120
----------------------------------------------------------------------

cell 14:
============================================================
PREDICTION DISTRIBUTION ON UNSEEN DATA
============================================================
Total predictions on unseen data: 16000
  Predictions in unseen label space: 4617 (28.9%)
  Predictions in seen label space:   11383 (71.1%)

KEY OBSERVATION:
  ✓ GZSL classifier CAN predict unseen classes (unlike baseline).
    This demonstrates successful zero-shot transfer via CLIP + cWGAN-GP.

cell 15:
======================================================================
FINAL SUMMARY: GZSL CLASSIFIER [A+B]
======================================================================

### MODEL PIPELINE ###
  1. Brain-Image encoder (CORnet-S): Maps EEG → 128D semantic space
  2. cWGAN-GP: Generates synthetic unseen embeddings from image prototypes
  3. GZSL Classifier: Logistic Regression on real seen + synthetic unseen

### GZSL RESULTS ###
  Acc (seen):        0.0375 (3.75%)
  Acc (unseen):      0.0573 (5.73%)
  Harmonic Mean (H): 0.0453
  Macro F1 (seen):   0.0225
  Macro F1 (unseen): 0.0120

### IMPROVEMENT OVER BASELINE ###
  Δ Acc (seen):   +0.0193
  Δ Acc (unseen): +0.0573
  Δ H:            +0.0453

### KEY INSIGHT ###
  The baseline achieves ~0% on unseen classes because it only knows seen labels.
  GZSL [A+B] achieves non-zero unseen accuracy by:
    - Using contrastive learning to create a shared EEG-image semantic space
    - Using cWGAN-GP to synthesize training data for unseen classes
    - Training a classifier that sees both seen and (synthetic) unseen classes

### FIGURES GENERATED ###
  - figures/gzsl_comparison.png

======================================================================
GZSL IMPLEMENTATION COMPLETE
======================================================================

cell 16:
Seen labels: 1654 classes (range 1-1654)
Unseen labels: 200 classes (range 1655-1854)
Overlap: set()

============================================================
GZSL EVALUATION — Pipeline v2
============================================================
  Acc Seen:      0.0375
  Acc Unseen:    0.0573
  Harmonic Mean: 0.0453
  F1 Seen:       0.0225
  F1 Unseen:     0.0120
  Routing Rate (seen→unseen): 0.2034 (673/3308)

  Bias Table:
                         Pred Seen    Pred Unseen
    True Seen                2635         673  (20.3% misrouted)
    True Unseen             11383        4617
============================================================

============================================================
CLASSIFIER DIAGNOSTICS — Pipeline v2
============================================================
  Weight norms ||w_c||:
    Seen   — mean: 3.2707, std: 0.5712, min: 1.5075, max: 5.0221
    Unseen — mean: 3.6212, std: 0.3239, min: 2.3453, max: 4.3079
    Ratio (unseen/seen mean): 1.11x

  Intercepts (biases) β_c:
    Seen   — mean: 0.0073, std: 0.1135
    Unseen — mean: -0.0601, std: 0.0691
============================================================

Saved: figures/pipeline_v2_diagnostics.png

