cell 1
Phase F experiment infrastructure ready.
  Encoders: ['mlp_ln', 'eegnet_small', 'eegnet_large', 'shallow_conv']
  Loss types: infonce, supcon, combined
  Augmentations: gaussian, time_shift, channel_drop, time_mask, mixup
  Baseline reference: top-1 = 2.87% (Phase E best)

cell 2
STAGE F1 ROUND 1: Individual Augmentations
============================================================

============================================================
EXPERIMENT: F1-ctrl: no_aug
============================================================
  encoder=mlp_ln, embed_dim=128, epochs=50
  lr=0.001, schedule=cosine_warmup, tau=0.15
  loss=infonce, batch=shuffle
  augmentation=None
  Total parameters: 1,497,344
  Epoch   1/50: loss=5.4503, lr=0.000200
  Epoch  10/50: loss=2.5012, lr=0.000970
  Epoch  20/50: loss=1.5036, lr=0.000750
  Epoch  30/50: loss=1.1513, lr=0.000413
  Epoch  40/50: loss=1.0130, lr=0.000117
  Epoch  50/50: loss=0.9798, lr=0.000000

  RESULTS: top-1=0.03%, top-5=0.73%, top-10=1.09%, signal=0.5x
  Time: 5.5s, Final loss: 0.9798

============================================================
EXPERIMENT: F1-A: gaussian
============================================================
  encoder=mlp_ln, embed_dim=128, epochs=50
  lr=0.001, schedule=cosine_warmup, tau=0.15
  loss=infonce, batch=shuffle
  augmentation={'gaussian_std': 0.3}
  Total parameters: 1,497,344
  Epoch   1/50: loss=5.4621, lr=0.000200
  Epoch  10/50: loss=2.7865, lr=0.000970
  Epoch  20/50: loss=1.7947, lr=0.000750
  Epoch  30/50: loss=1.3927, lr=0.000413
  Epoch  40/50: loss=1.2076, lr=0.000117
  Epoch  50/50: loss=1.1651, lr=0.000000

  RESULTS: top-1=0.21%, top-5=0.70%, top-10=1.27%, signal=3.5x
  Time: 5.6s, Final loss: 1.1651

============================================================
EXPERIMENT: F1-B: time_shift
============================================================
  encoder=mlp_ln, embed_dim=128, epochs=50
  lr=0.001, schedule=cosine_warmup, tau=0.15
  loss=infonce, batch=shuffle
  augmentation={'time_shift': 2}
  Total parameters: 1,497,344
  Epoch   1/50: loss=5.4780, lr=0.000200
  Epoch  10/50: loss=3.3181, lr=0.000970
  Epoch  20/50: loss=2.2728, lr=0.000750
  Epoch  30/50: loss=1.7841, lr=0.000413
  Epoch  40/50: loss=1.5046, lr=0.000117
  Epoch  50/50: loss=1.4259, lr=0.000000

  RESULTS: top-1=0.18%, top-5=0.42%, top-10=0.94%, signal=3.0x
  Time: 5.8s, Final loss: 1.4259

============================================================
EXPERIMENT: F1-C: channel_drop
============================================================
  encoder=mlp_ln, embed_dim=128, epochs=50
  lr=0.001, schedule=cosine_warmup, tau=0.15
  loss=infonce, batch=shuffle
  augmentation={'channel_drop_prob': 0.15}
  Total parameters: 1,497,344
  Epoch   1/50: loss=5.4734, lr=0.000200
  Epoch  10/50: loss=3.0513, lr=0.000970
  Epoch  20/50: loss=2.0665, lr=0.000750
  Epoch  30/50: loss=1.6092, lr=0.000413
  Epoch  40/50: loss=1.3966, lr=0.000117
  Epoch  50/50: loss=1.3418, lr=0.000000

  RESULTS: top-1=0.24%, top-5=0.70%, top-10=1.21%, signal=4.0x
  Time: 5.8s, Final loss: 1.3418

============================================================
EXPERIMENT: F1-D: time_mask
============================================================
  encoder=mlp_ln, embed_dim=128, epochs=50
  lr=0.001, schedule=cosine_warmup, tau=0.15
  loss=infonce, batch=shuffle
  augmentation={'time_mask_width': 4}
  Total parameters: 1,497,344
  Epoch   1/50: loss=5.4677, lr=0.000200
  Epoch  10/50: loss=3.0422, lr=0.000970
  Epoch  20/50: loss=2.0678, lr=0.000750
  Epoch  30/50: loss=1.6136, lr=0.000413
  Epoch  40/50: loss=1.3899, lr=0.000117
  Epoch  50/50: loss=1.3454, lr=0.000000

  RESULTS: top-1=0.15%, top-5=0.76%, top-10=1.09%, signal=2.5x
  Time: 5.6s, Final loss: 1.3454

============================================================
EXPERIMENT: F1-E: mixup
============================================================
  encoder=mlp_ln, embed_dim=128, epochs=50
  lr=0.001, schedule=cosine_warmup, tau=0.15
  loss=infonce, batch=shuffle
  augmentation={'mixup_alpha': 0.3}
  Total parameters: 1,497,344
  Epoch   1/50: loss=5.4551, lr=0.000200
  Epoch  10/50: loss=2.9205, lr=0.000970
  Epoch  20/50: loss=2.0354, lr=0.000750
  Epoch  30/50: loss=1.6646, lr=0.000413
  Epoch  40/50: loss=1.4994, lr=0.000117
  Epoch  50/50: loss=1.3705, lr=0.000000

  RESULTS: top-1=0.12%, top-5=0.63%, top-10=1.06%, signal=2.0x
  Time: 5.7s, Final loss: 1.3705

============================================================
ROUND 1 RESULTS:
  F1-C: channel_drop             top-1=0.24%  top-5=0.70%  signal=4.0x
  F1-A: gaussian                 top-1=0.21%  top-5=0.70%  signal=3.5x
  F1-B: time_shift               top-1=0.18%  top-5=0.42%  signal=3.0x
  F1-D: time_mask                top-1=0.15%  top-5=0.76%  signal=2.5x
  F1-E: mixup                    top-1=0.12%  top-5=0.63%  signal=2.0x
  F1-ctrl: no_aug                top-1=0.03%  top-5=0.73%  signal=0.5x

Top 3 augmentations for Round 2: ['F1-C: channel_drop', 'F1-A: gaussian', 'F1-B: time_shift']

cell 3
STAGE F1 ROUND 2: Combined Augmentations
============================================================

============================================================
EXPERIMENT: F1-F: combo_top2
============================================================
  encoder=mlp_ln, embed_dim=128, epochs=50
  lr=0.001, schedule=cosine_warmup, tau=0.15
  loss=infonce, batch=shuffle
  augmentation={'channel_drop_prob': 0.15, 'gaussian_std': 0.3}
  Total parameters: 1,497,344
  Epoch   1/50: loss=5.4797, lr=0.000200
  Epoch  10/50: loss=3.2216, lr=0.000970
  Epoch  20/50: loss=2.2838, lr=0.000750
  Epoch  30/50: loss=1.8026, lr=0.000413
  Epoch  40/50: loss=1.5666, lr=0.000117
  Epoch  50/50: loss=1.4978, lr=0.000000

  RESULTS: top-1=0.15%, top-5=0.76%, top-10=1.12%, signal=2.5x
  Time: 5.7s, Final loss: 1.4978

============================================================
EXPERIMENT: F1-G: combo_top3
============================================================
  encoder=mlp_ln, embed_dim=128, epochs=50
  lr=0.001, schedule=cosine_warmup, tau=0.15
  loss=infonce, batch=shuffle
  augmentation={'channel_drop_prob': 0.15, 'gaussian_std': 0.3, 'time_shift': 2}
  Total parameters: 1,497,344
  Epoch   1/50: loss=5.5050, lr=0.000200
  Epoch  10/50: loss=3.7688, lr=0.000970
  Epoch  20/50: loss=2.8133, lr=0.000750
  Epoch  30/50: loss=2.3167, lr=0.000413
  Epoch  40/50: loss=2.0087, lr=0.000117
  Epoch  50/50: loss=1.9157, lr=0.000000

  RESULTS: top-1=0.15%, top-5=0.39%, top-10=0.88%, signal=2.5x
  Time: 6.1s, Final loss: 1.9157

============================================================
EXPERIMENT: F1-H: combo_all_mild
============================================================
  encoder=mlp_ln, embed_dim=128, epochs=50
  lr=0.001, schedule=cosine_warmup, tau=0.15
  loss=infonce, batch=shuffle
  augmentation={'gaussian_std': 0.15, 'time_shift': 1, 'channel_drop_prob': 0.1, 'time_mask_width': 3, 'mixup_alpha': 0.2}
  Total parameters: 1,497,344
  Epoch   1/50: loss=5.4914, lr=0.000200
  Epoch  10/50: loss=3.7576, lr=0.000970
  Epoch  20/50: loss=3.0312, lr=0.000750
  Epoch  30/50: loss=2.7606, lr=0.000413
  Epoch  40/50: loss=2.4376, lr=0.000117
  Epoch  50/50: loss=2.2257, lr=0.000000

  RESULTS: top-1=0.15%, top-5=0.45%, top-10=0.97%, signal=2.5x
  Time: 6.0s, Final loss: 2.2257

============================================================
ROUND 2 RESULTS:

BEST AUGMENTATION: F1-C: channel_drop — top-1 = 0.24%
AUG* config: {'channel_drop_prob': 0.15}

cell 4
================================================================================
STAGE F1 RESULTS — DATA AUGMENTATION SWEEP
================================================================================
Name                                  Top-1   Top-5  Top-10   Signal     Loss   Time
--------------------------------------------------------------------------------
F1-C: channel_drop                    0.24%   0.70%   1.21%     4.0x  1.3418     6s
F1-A: gaussian                        0.21%   0.70%   1.27%     3.5x  1.1651     6s
F1-B: time_shift                      0.18%   0.42%   0.94%     3.0x  1.4259     6s
F1-D: time_mask                       0.15%   0.76%   1.09%     2.5x  1.3454     6s
F1-F: combo_top2                      0.15%   0.76%   1.12%     2.5x  1.4978     6s
F1-G: combo_top3                      0.15%   0.39%   0.88%     2.5x  1.9157     6s
F1-H: combo_all_mild                  0.15%   0.45%   0.97%     2.5x  2.2257     6s
F1-E: mixup                           0.12%   0.63%   1.06%     2.0x  1.3705     6s
F1-ctrl: no_aug                       0.03%   0.73%   1.09%     0.5x  0.9798     5s
--------------------------------------------------------------------------------

BASELINE (Phase E, no aug): top-1 = 2.87%
BEST AUG: F1-C: channel_drop — top-1 = 0.24%
Saved: figures/phase_f_stage1_augmentation.png

cell 5
STAGE F2: Loss Function Variants (using AUG*)
============================================================
  AUG* = {'channel_drop_prob': 0.15}

============================================================
EXPERIMENT: F2-I: supcon_only
============================================================
  encoder=mlp_ln, embed_dim=128, epochs=50
  lr=0.001, schedule=cosine_warmup, tau=0.15
  loss=supcon, batch=class_balanced (K=64, M=4)
  augmentation={'channel_drop_prob': 0.15}
  Total parameters: 1,497,344
  Epoch   1/50: loss=5.4516, lr=0.000200
  Epoch  10/50: loss=4.5925, lr=0.000970
  Epoch  20/50: loss=4.0071, lr=0.000750
  Epoch  30/50: loss=3.4609, lr=0.000413
  Epoch  40/50: loss=3.0655, lr=0.000117
  Epoch  50/50: loss=2.9857, lr=0.000000

  RESULTS: top-1=0.06%, top-5=0.42%, top-10=0.79%, signal=1.0x
  Time: 6.4s, Final loss: 2.9857

============================================================
EXPERIMENT: F2-J: combined_0.5
============================================================
  encoder=mlp_ln, embed_dim=128, epochs=50
  lr=0.001, schedule=cosine_warmup, tau=0.15
  loss=combined, batch=class_balanced (K=64, M=4)
  augmentation={'channel_drop_prob': 0.15}
  Total parameters: 1,497,344
  Epoch   1/50: loss=5.4666, lr=0.000200
  Epoch  10/50: loss=3.9114, lr=0.000970
  Epoch  20/50: loss=3.2012, lr=0.000750
  Epoch  30/50: loss=2.7665, lr=0.000413
  Epoch  40/50: loss=2.5053, lr=0.000117
  Epoch  50/50: loss=2.4519, lr=0.000000

  RESULTS: top-1=0.21%, top-5=0.70%, top-10=1.21%, signal=3.5x
  Time: 7.2s, Final loss: 2.4519

============================================================
EXPERIMENT: F2-K: combined_0.3
============================================================
  encoder=mlp_ln, embed_dim=128, epochs=50
  lr=0.001, schedule=cosine_warmup, tau=0.15
  loss=combined, batch=class_balanced (K=64, M=4)
  augmentation={'channel_drop_prob': 0.15}
  Total parameters: 1,497,344
  Epoch   1/50: loss=5.4626, lr=0.000200
  Epoch  10/50: loss=3.6151, lr=0.000970
  Epoch  20/50: loss=2.8448, lr=0.000750
  Epoch  30/50: loss=2.4384, lr=0.000413
  Epoch  40/50: loss=2.2063, lr=0.000117
  Epoch  50/50: loss=2.1608, lr=0.000000

  RESULTS: top-1=0.18%, top-5=0.45%, top-10=1.21%, signal=3.0x
  Time: 7.3s, Final loss: 2.1608

============================================================
STAGE F2 RESULTS:
  F1-C: channel_drop             top-1=0.24%  top-5=0.70%  signal=4.0x
  F2-J: combined_0.5             top-1=0.21%  top-5=0.70%  signal=3.5x
  F2-K: combined_0.3             top-1=0.18%  top-5=0.45%  signal=3.0x
  F2-I: supcon_only              top-1=0.06%  top-5=0.42%  signal=1.0x

BEST LOSS: F1-C: channel_drop — top-1 = 0.24%

cell 6
================================================================================
STAGE F2 RESULTS — LOSS FUNCTION SWEEP
================================================================================
Name                                  Top-1   Top-5  Top-10   Signal     Loss   Time
--------------------------------------------------------------------------------
F1-C: channel_drop                    0.24%   0.70%   1.21%     4.0x  1.3418     6s
F2-J: combined_0.5                    0.21%   0.70%   1.21%     3.5x  2.4519     7s
F2-K: combined_0.3                    0.18%   0.45%   1.21%     3.0x  2.1608     7s
F2-I: supcon_only                     0.06%   0.42%   0.79%     1.0x  2.9857     6s
--------------------------------------------------------------------------------
BEST LOSS: F1-C: channel_drop — top-1 = 0.24%

cell 7
STAGE F3: Architecture Variants (using AUG* + LOSS*)
============================================================

============================================================
EXPERIMENT: F3-L: mlp_ln
============================================================
  encoder=mlp_ln, embed_dim=128, epochs=50
  lr=0.001, schedule=cosine_warmup, tau=0.15
  loss=infonce, batch=shuffle
  augmentation={'channel_drop_prob': 0.15}
  Total parameters: 1,497,344
  Epoch   1/50: loss=5.4734, lr=0.000200
  Epoch  10/50: loss=3.0513, lr=0.000970
  Epoch  20/50: loss=2.0665, lr=0.000750
  Epoch  30/50: loss=1.6092, lr=0.000413
  Epoch  40/50: loss=1.3966, lr=0.000117
  Epoch  50/50: loss=1.3418, lr=0.000000

  RESULTS: top-1=0.24%, top-5=0.70%, top-10=1.21%, signal=4.0x
  Time: 5.5s, Final loss: 1.3418

============================================================
EXPERIMENT: F3-M: eegnet_small
============================================================
  encoder=eegnet_small, embed_dim=128, epochs=50
  lr=0.001, schedule=cosine_warmup, tau=0.15
  loss=infonce, batch=shuffle
  augmentation={'channel_drop_prob': 0.15}
  Total parameters: 337,400
  Epoch   1/50: loss=5.5474, lr=0.000200
  Epoch  10/50: loss=4.2599, lr=0.000970
  Epoch  20/50: loss=3.2779, lr=0.000750
  Epoch  30/50: loss=2.7463, lr=0.000413
  Epoch  40/50: loss=2.4617, lr=0.000117
  Epoch  50/50: loss=2.3870, lr=0.000000

  RESULTS: top-1=0.09%, top-5=0.39%, top-10=1.00%, signal=1.5x
  Time: 7.0s, Final loss: 2.3870

============================================================
EXPERIMENT: F3-N: eegnet_large
============================================================
  encoder=eegnet_large, embed_dim=128, epochs=50
  lr=0.001, schedule=cosine_warmup, tau=0.15
  loss=infonce, batch=shuffle
  augmentation={'channel_drop_prob': 0.15}
  Total parameters: 346,864
  Epoch   1/50: loss=5.5520, lr=0.000200
  Epoch  10/50: loss=4.0491, lr=0.000970
  Epoch  20/50: loss=3.0422, lr=0.000750
  Epoch  30/50: loss=2.5044, lr=0.000413
  Epoch  40/50: loss=2.2331, lr=0.000117
  Epoch  50/50: loss=2.1610, lr=0.000000

  RESULTS: top-1=0.06%, top-5=0.39%, top-10=0.82%, signal=1.0x
  Time: 6.4s, Final loss: 2.1610

============================================================
EXPERIMENT: F3-O: shallow_conv
============================================================
  encoder=shallow_conv, embed_dim=128, epochs=50
  lr=0.001, schedule=cosine_warmup, tau=0.15
  loss=infonce, batch=shuffle
  augmentation={'channel_drop_prob': 0.15}
  Total parameters: 407,448
  Epoch   1/50: loss=5.6175, lr=0.000200
  Epoch  10/50: loss=4.4510, lr=0.000970
  Epoch  20/50: loss=3.3860, lr=0.000750
  Epoch  30/50: loss=2.7874, lr=0.000413
  Epoch  40/50: loss=2.4933, lr=0.000117
  Epoch  50/50: loss=2.4074, lr=0.000000

  RESULTS: top-1=0.12%, top-5=0.45%, top-10=0.94%, signal=2.0x
  Time: 6.2s, Final loss: 2.4074

============================================================
STAGE F3 RESULTS:
  F3-L: mlp_ln                   top-1=0.24%  top-5=0.70%  signal=4.0x  params=1,497,344
  F3-O: shallow_conv             top-1=0.12%  top-5=0.45%  signal=2.0x  params=407,448
  F3-M: eegnet_small             top-1=0.09%  top-5=0.39%  signal=1.5x  params=337,400
  F3-N: eegnet_large             top-1=0.06%  top-5=0.39%  signal=1.0x  params=346,864

BEST ARCHITECTURE: F3-L: mlp_ln — top-1 = 0.24%

cell 8
================================================================================
STAGE F3 RESULTS — ARCHITECTURE SWEEP
================================================================================
Name                                  Top-1   Top-5  Top-10   Signal     Params   Time
--------------------------------------------------------------------------------
F3-L: mlp_ln                          0.24%   0.70%   1.21%     4.0x  1,497,344     6s
F3-O: shallow_conv                    0.12%   0.45%   0.94%     2.0x    407,448     6s
F3-M: eegnet_small                    0.09%   0.39%   1.00%     1.5x    337,400     7s
F3-N: eegnet_large                    0.06%   0.39%   0.82%     1.0x    346,864     6s
--------------------------------------------------------------------------------

BASELINE (Phase E): top-1 = 2.87%
BEST F: F3-L: mlp_ln — top-1 = 0.24%

============================================================
BEST OVERALL PHASE F CONFIG:
  loss_type: infonce
  lambda_sup: 0.5
  batch_mode: shuffle
  K: 64
  M: 4
  augmentation: {'channel_drop_prob': 0.15}
  encoder: mlp_ln
  dropout: 0.1

Saved: figures/phase_f_stage3_architecture.png

cell 9
======================================================================
FULL PIPELINE RE-RUN WITH BEST PHASE F CONFIG
======================================================================
Config: {'loss_type': 'infonce', 'lambda_sup': 0.5, 'batch_mode': 'shuffle', 'K': 64, 'M': 4, 'augmentation': {'channel_drop_prob': 0.15}, 'name': 'F3-L: mlp_ln', 'encoder': 'mlp_ln', 'dropout': 0.1}

Step 1: Using best encoder from F3-L: mlp_ln
  Encoder top-1: 0.24%

Step 2: Extracting embeddings...
  E_train_seen: (13232, 128)
  E_test_seen:  (3308, 128)
  E_unseen:     (16000, 128)

Step 3: Computing prototypes...
  Seen prototypes: (1654, 128)
  Unseen prototypes: (200, 128)

Step 4: Saving cached arrays...
  Cached arrays overwritten.

Step 5: Retraining WGAN-GP...
  WGAN step 2000/10000: G=0.1082, C=-0.0300
  WGAN step 4000/10000: G=0.1046, C=-0.0246
  WGAN step 6000/10000: G=-0.2827, C=-0.0206
  WGAN step 8000/10000: G=-0.6143, C=-0.0255
  WGAN step 10000/10000: G=-0.8564, C=-0.0240
  WGAN-GP training complete.

Step 6: Generating synthetic unseen embeddings...
  E_synth_unseen: (4000, 128)

Step 7: GZSL evaluation (balanced counts)...

============================================================
OPTIMISED PIPELINE RESULTS (Phase F)
============================================================
  Encoder top-1 (seen-only): 0.24%
  AccS (GZSL):               2.45%
  AccU (GZSL):               1.25%
  H-mean:                    1.66%
  Routing rate:              46.41%
  Synth->Real transfer:      2.41%

cell 10
================================================================================
FINAL COMPARISON
================================================================================

Metric                                          Baseline     Phase1    Phase E    Phase F
-------------------------------------------------------------------------------------
Encoder top-1 (seen-only)                          1.90%      1.90%      2.87%      0.24%
AccS (GZSL)                                        0.06%      2.90%      2.96%      2.45%
AccU (GZSL)                                        2.13%      0.22%      0.40%      1.25%
H-mean                                             0.12%      0.42%      0.70%      1.66%
Routing rate                                      99.70%      9.16%     11.31%     46.41%
Synth->Real transfer (200-way)                     1.73%      1.73%      2.51%      2.41%
-------------------------------------------------------------------------------------

Saved: figures/phase_f_final_comparison.png

============================================================
PHASE F COMPLETE
Best config: {'loss_type': 'infonce', 'lambda_sup': 0.5, 'batch_mode': 'shuffle', 'K': 64, 'M': 4, 'augmentation': {'channel_drop_prob': 0.15}, 'name': 'F3-L: mlp_ln', 'encoder': 'mlp_ln', 'dropout': 0.1}
============================================================

