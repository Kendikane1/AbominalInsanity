cell 1

Seen-only classification: 1654 classes
Train: (13232, 64), Test: (3308, 64)
Random baseline: 0.0006 (0.06%)

/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:1247: FutureWarning: 'multi_class' was deprecated in version 1.5 and will be removed in 1.7. From then on, it will always use 'multinomial'. Leave it to its default value to avoid this warning.
  warnings.warn(
============================================================
DIAGNOSTIC A RESULTS: Seen-Only Classification
============================================================
  Top-1 Accuracy:  0.0190 (1.90%)
  Top-5 Accuracy:  0.0653 (6.53%)
  Top-10 Accuracy: 0.0992 (9.92%)
  Random baseline: 0.0006 (0.06%)
  Signal ratio:    31.5x above random

>>> CLIP encoder captures WEAK signal. Encoder improvement may help.

cell 2

Unseen-only classification: 200 classes
Synthetic train: (4000, 64)
Real unseen test: (16000, 64)
Random baseline: 0.0050 (0.50%)

/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:1247: FutureWarning: 'multi_class' was deprecated in version 1.5 and will be removed in 1.7. From then on, it will always use 'multinomial'. Leave it to its default value to avoid this warning.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:1247: FutureWarning: 'multi_class' was deprecated in version 1.5 and will be removed in 1.7. From then on, it will always use 'multinomial'. Leave it to its default value to avoid this warning.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:1247: FutureWarning: 'multi_class' was deprecated in version 1.5 and will be removed in 1.7. From then on, it will always use 'multinomial'. Leave it to its default value to avoid this warning.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:1247: FutureWarning: 'multi_class' was deprecated in version 1.5 and will be removed in 1.7. From then on, it will always use 'multinomial'. Leave it to its default value to avoid this warning.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:1247: FutureWarning: 'multi_class' was deprecated in version 1.5 and will be removed in 1.7. From then on, it will always use 'multinomial'. Leave it to its default value to avoid this warning.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:1247: FutureWarning: 'multi_class' was deprecated in version 1.5 and will be removed in 1.7. From then on, it will always use 'multinomial'. Leave it to its default value to avoid this warning.
  warnings.warn(
============================================================
DIAGNOSTIC B RESULTS: Unseen-Only Classification
============================================================

B1: Synthetic → Real Transfer
  Top-1 Accuracy:  0.0173 (1.73%)
  Top-5 Accuracy:  0.0798 (7.98%)
  Signal ratio:    3.5x above random

B2: Synthetic Internal Discriminability (5-fold CV)
  Accuracy:        0.6838 ± 0.0171
  Signal ratio:    136.8x above random

>>> Synthetics are internally discriminable but don't transfer to real data.
    Generator captures some structure but doesn't match real EEG distribution.

cell 3

Unseen prototypes: (200, 64)
Seen prototypes: (1654, 64)

============================================================
DIAGNOSTIC C RESULTS: Prototype Spread
============================================================


Metric                       Unseen (200)     Seen (1654)
-------------------------------------------------------
Mean cosine                        0.0135          0.0055
Median cosine                      0.0082         -0.0048
Std cosine                         0.1569          0.1750
Min cosine                        -0.5012         -0.6534
Max cosine                         0.9579          0.9769
% pairs > 0.9                        0.0%            0.0%
% pairs > 0.95                       0.0%            0.0%

Random unit vectors in R^64: E[cos] ≈ 0, std ≈ 0.125

Nearest-neighbour cosine (unseen): mean=0.4848, min=0.2929, max=0.9579
Nearest-neighbour cosine (seen):   mean=0.6930, min=0.3793, max=0.9769

>>> Prototypes are well-separated. Bottleneck is NOT in the semantic space.

cell 4
figure

cell 5
======================================================================
UPSTREAM BOTTLENECK DIAGNOSTICS — SUMMARY
======================================================================

Diagnostic                                         Value     Signal
-----------------------------------------------------------------
A: Seen top-1 acc (1654-way)                       1.90%      31.5x
A: Seen top-5 acc                                  6.53%
B1: Synth→Real transfer acc (200-way)              1.73%       3.5x
B2: Synth internal CV acc (200-way)               68.38%     136.8x
C: Unseen prototype mean cosine                   0.0135
C: Seen prototype mean cosine (ref)               0.0055
C: Unseen NN cosine mean                          0.4848

BOTTLENECK ANALYSIS:
-----------------------------------------------------------------
  [      WEAK] CLIP Encoder: top-1 = 1.90%, limited signal
  [        OK] WGAN-GP Generator: transfer = 1.73%
  [        OK] Text Prototypes: mean cos = 0.014, adequate spread

NEXT STEPS (to be decided by orchestrator based on these results):
  - If Encoder CRITICAL → increase embed_dim, train longer, curriculum learning
  - If Generator CRITICAL → covariance-aware WGAN, more training steps, architecture changes
  - If Prototypes CRITICAL → better text features, larger semantic space, alternative alignment
  - If multiple CRITICAL → address upstream first (prototypes → encoder → generator)