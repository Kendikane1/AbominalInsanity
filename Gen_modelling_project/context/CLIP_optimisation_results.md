cell 1
Device: cpu
Train: (13232, 561), Test: (3308, 561), Unseen: (16000, 561)
Seen classes: 1654, Unseen classes: 200

Experiment infrastructure ready.
Baseline reference: top-1 = 1.90%, signal = 31.5x (20 epochs, flat lr, fixed tau=0.07, dim=64)

cell 2
STAGE 1 ROUND 1: Epoch Scaling
============================================================

============================================================
EXPERIMENT: R1-A: 50ep
============================================================
  arch=baseline, embed_dim=64, epochs=50
  lr=0.0001, schedule=flat, tau=0.07 (fixed)
  batch_size=256, dropout=0.1
  Total parameters: 1,428,608
  Epoch   10/50: loss=2.1910, lr=0.000100, tau=0.0700
  Epoch   20/50: loss=0.9543, lr=0.000100, tau=0.0700
  Epoch   30/50: loss=0.5360, lr=0.000100, tau=0.0700
  Epoch   40/50: loss=0.3584, lr=0.000100, tau=0.0700
  Epoch   50/50: loss=0.2577, lr=0.000100, tau=0.0700

  RESULTS: top-1=1.93%, top-5=5.74%, top-10=9.28%, signal=32.0x
  Time: 32.3s, Final loss: 0.2577, Final tau: 0.0700

============================================================
EXPERIMENT: R1-B: 100ep
============================================================
  arch=baseline, embed_dim=64, epochs=100
  lr=0.0001, schedule=flat, tau=0.07 (fixed)
  batch_size=256, dropout=0.1
  Total parameters: 1,428,608
  Epoch   20/100: loss=0.9543, lr=0.000100, tau=0.0700
  Epoch   40/100: loss=0.3584, lr=0.000100, tau=0.0700
  Epoch   60/100: loss=0.2017, lr=0.000100, tau=0.0700
  Epoch   80/100: loss=0.1357, lr=0.000100, tau=0.0700
  Epoch  100/100: loss=0.0992, lr=0.000100, tau=0.0700

  RESULTS: top-1=1.78%, top-5=5.41%, top-10=8.98%, signal=29.5x
  Time: 64.4s, Final loss: 0.0992, Final tau: 0.0700

============================================================
EXPERIMENT: R1-C: 200ep
============================================================
  arch=baseline, embed_dim=64, epochs=200
  lr=0.0001, schedule=flat, tau=0.07 (fixed)
  batch_size=256, dropout=0.1
  Total parameters: 1,428,608
  Epoch   40/200: loss=0.3584, lr=0.000100, tau=0.0700
  Epoch   80/200: loss=0.1357, lr=0.000100, tau=0.0700
  Epoch  120/200: loss=0.0771, lr=0.000100, tau=0.0700
  Epoch  160/200: loss=0.0527, lr=0.000100, tau=0.0700
  Epoch  200/200: loss=0.0384, lr=0.000100, tau=0.0700

  RESULTS: top-1=1.42%, top-5=4.50%, top-10=7.62%, signal=23.5x
  Time: 130.2s, Final loss: 0.0384, Final tau: 0.0700

============================================================
ROUND 1 WINNER: R1-A: 50ep — top-1 = 1.93%
Selected E* = 50 epochs
  R1-A: 50ep: top-1=1.93%, top-5=5.74%, loss=0.2577, time=32s
  R1-B: 100ep: top-1=1.78%, top-5=5.41%, loss=0.0992, time=64s
  R1-C: 200ep: top-1=1.42%, top-5=4.50%, loss=0.0384, time=130s

cell 3
STAGE 1 ROUND 2: LR Schedule + Value (using E*=50 epochs)
============================================================

============================================================
EXPERIMENT: R2-D: cos_warmup lr=1e-4
============================================================
  arch=baseline, embed_dim=64, epochs=50
  lr=0.0001, schedule=cosine_warmup, tau=0.07 (fixed)
  batch_size=256, dropout=0.1
  Total parameters: 1,428,608
  Epoch   10/50: loss=2.8680, lr=0.000097, tau=0.0700
  Epoch   20/50: loss=1.1873, lr=0.000075, tau=0.0700
  Epoch   30/50: loss=0.7284, lr=0.000041, tau=0.0700
  Epoch   40/50: loss=0.5840, lr=0.000012, tau=0.0700
  Epoch   50/50: loss=0.5496, lr=0.000000, tau=0.0700

  RESULTS: top-1=1.60%, top-5=6.20%, top-10=9.92%, signal=26.5x
  Time: 31.7s, Final loss: 0.5496, Final tau: 0.0700

============================================================
EXPERIMENT: R2-E: cos_warmup lr=3e-4
============================================================
  arch=baseline, embed_dim=64, epochs=50
  lr=0.0003, schedule=cosine_warmup, tau=0.07 (fixed)
  batch_size=256, dropout=0.1
  Total parameters: 1,428,608
  Epoch   10/50: loss=1.5082, lr=0.000291, tau=0.0700
  Epoch   20/50: loss=0.3640, lr=0.000225, tau=0.0700
  Epoch   30/50: loss=0.2018, lr=0.000124, tau=0.0700
  Epoch   40/50: loss=0.1535, lr=0.000035, tau=0.0700
  Epoch   50/50: loss=0.1404, lr=0.000000, tau=0.0700

  RESULTS: top-1=1.81%, top-5=6.35%, top-10=9.64%, signal=30.0x
  Time: 31.9s, Final loss: 0.1404, Final tau: 0.0700

============================================================
EXPERIMENT: R2-F: cos_warmup lr=1e-3
============================================================
  arch=baseline, embed_dim=64, epochs=50
  lr=0.001, schedule=cosine_warmup, tau=0.07 (fixed)
  batch_size=256, dropout=0.1
  Total parameters: 1,428,608
  Epoch   10/50: loss=1.1518, lr=0.000970, tau=0.0700
  Epoch   20/50: loss=0.1749, lr=0.000750, tau=0.0700
  Epoch   30/50: loss=0.0944, lr=0.000413, tau=0.0700
  Epoch   40/50: loss=0.0707, lr=0.000117, tau=0.0700
  Epoch   50/50: loss=0.0647, lr=0.000000, tau=0.0700

  RESULTS: top-1=2.12%, top-5=7.22%, top-10=11.55%, signal=35.0x
  Time: 31.3s, Final loss: 0.0647, Final tau: 0.0700

============================================================
ROUND 2 WINNER: R2-F: cos_warmup lr=1e-3 — top-1 = 2.12%
Selected S* = (lr=0.001, schedule=cosine_warmup)
  R1-A: 50ep: top-1=1.93%, top-5=5.74%, loss=0.2577
  R2-D: cos_warmup lr=1e-4: top-1=1.60%, top-5=6.20%, loss=0.5496
  R2-E: cos_warmup lr=3e-4: top-1=1.81%, top-5=6.35%, loss=0.1404
  R2-F: cos_warmup lr=1e-3: top-1=2.12%, top-5=7.22%, loss=0.0647

cell 4
STAGE 1 ROUND 3: Temperature (using E*=50, S*=(lr=0.001, sched=cosine_warmup))
============================================================

============================================================
EXPERIMENT: R3-G: learnable_tau
============================================================
  arch=baseline, embed_dim=64, epochs=50
  lr=0.001, schedule=cosine_warmup, tau=0.07 (learnable)
  batch_size=256, dropout=0.1
  Total parameters: 1,428,608
  Epoch   10/50: loss=1.0415, lr=0.000970, tau=0.0490
  Epoch   20/50: loss=0.0862, lr=0.000750, tau=0.0358
  Epoch   30/50: loss=0.0357, lr=0.000413, tau=0.0331
  Epoch   40/50: loss=0.0187, lr=0.000117, tau=0.0324
  Epoch   50/50: loss=0.0159, lr=0.000000, tau=0.0322

  RESULTS: top-1=2.18%, top-5=8.80%, top-10=13.45%, signal=36.0x
  Time: 31.3s, Final loss: 0.0159, Final tau: 0.0322

============================================================
EXPERIMENT: R3-H: fixed_tau=0.15
============================================================
  arch=baseline, embed_dim=64, epochs=50
  lr=0.001, schedule=cosine_warmup, tau=0.15 (fixed)
  batch_size=256, dropout=0.1
  Total parameters: 1,428,608
  Epoch   10/50: loss=2.3883, lr=0.000970, tau=0.1500
  Epoch   20/50: loss=1.4048, lr=0.000750, tau=0.1500
  Epoch   30/50: loss=1.1087, lr=0.000413, tau=0.1500
  Epoch   40/50: loss=0.9854, lr=0.000117, tau=0.1500
  Epoch   50/50: loss=0.9550, lr=0.000000, tau=0.1500

  RESULTS: top-1=2.21%, top-5=8.40%, top-10=12.94%, signal=36.5x
  Time: 31.1s, Final loss: 0.9550, Final tau: 0.1500

============================================================
ROUND 3 WINNER: R3-H: fixed_tau=0.15 — top-1 = 2.21%
Selected T* = (tau=0.15, mode=fixed)
  R2-F: cos_warmup lr=1e-3: top-1=2.12%, final_tau=0.0700
  R3-G: learnable_tau: top-1=2.18%, final_tau=0.0322
  R3-H: fixed_tau=0.15: top-1=2.21%, final_tau=0.1500

cell 5
================================================================================
STAGE 1 RESULTS — TRAINING DYNAMICS SWEEP (dim=64)
================================================================================
Name                             Top-1   Top-5  Top-10   Signal    Loss   Time    Tau
--------------------------------------------------------------------------------
R1-A: 50ep                       1.93%   5.74%   9.28%    32.0x  0.2577    32s 0.0700
R1-B: 100ep                      1.78%   5.41%   8.98%    29.5x  0.0992    64s 0.0700
R1-C: 200ep                      1.42%   4.50%   7.62%    23.5x  0.0384   130s 0.0700
R2-D: cos_warmup lr=1e-4         1.60%   6.20%   9.92%    26.5x  0.5496    32s 0.0700
R2-E: cos_warmup lr=3e-4         1.81%   6.35%   9.64%    30.0x  0.1404    32s 0.0700
R2-F: cos_warmup lr=1e-3         2.12%   7.22%  11.55%    35.0x  0.0647    31s 0.0700
R3-G: learnable_tau              2.18%   8.80%  13.45%    36.0x  0.0159    31s 0.0322
R3-H: fixed_tau=0.15             2.21%   8.40%  12.94%    36.5x  0.9550    31s 0.1500
--------------------------------------------------------------------------------
Baseline (20ep, flat, tau=0.07): top-1 = 1.90%

BEST STAGE 1 CONFIG: {'epochs': 50, 'lr': 0.001, 'schedule': 'cosine_warmup', 'tau': 0.15, 'tau_mode': 'fixed'}
Best top-1: 2.21% (improvement: 16% over baseline)

figure is at figures folder

cell 6
STAGE 2 ROUND 4: Embedding Dimension (using BEST_STAGE1)
============================================================

============================================================
EXPERIMENT: R4-I: dim=128
============================================================
  arch=baseline, embed_dim=128, epochs=50
  lr=0.001, schedule=cosine_warmup, tau=0.15 (fixed)
  batch_size=256, dropout=0.1
  Total parameters: 1,494,272
  Epoch   10/50: loss=2.4248, lr=0.000970, tau=0.1500
  Epoch   20/50: loss=1.4315, lr=0.000750, tau=0.1500
  Epoch   30/50: loss=1.1269, lr=0.000413, tau=0.1500
  Epoch   40/50: loss=1.0082, lr=0.000117, tau=0.1500
  Epoch   50/50: loss=0.9753, lr=0.000000, tau=0.1500

  RESULTS: top-1=2.36%, top-5=8.07%, top-10=12.91%, signal=39.0x
  Time: 32.8s, Final loss: 0.9753, Final tau: 0.1500

============================================================
EXPERIMENT: R4-J: dim=256
============================================================
  arch=baseline, embed_dim=256, epochs=50
  lr=0.001, schedule=cosine_warmup, tau=0.15 (fixed)
  batch_size=256, dropout=0.1
  Total parameters: 1,625,600
  Epoch   10/50: loss=2.4429, lr=0.000970, tau=0.1500
  Epoch   20/50: loss=1.4264, lr=0.000750, tau=0.1500
  Epoch   30/50: loss=1.1311, lr=0.000413, tau=0.1500
  Epoch   40/50: loss=1.0156, lr=0.000117, tau=0.1500
  Epoch   50/50: loss=0.9837, lr=0.000000, tau=0.1500

  RESULTS: top-1=1.96%, top-5=7.38%, top-10=11.28%, signal=32.5x
  Time: 33.3s, Final loss: 0.9837, Final tau: 0.1500

============================================================
ROUND 4 WINNER: R4-I: dim=128 — top-1 = 2.36%
Selected D* = 128
  dim=64: top-1=2.21%, top-5=8.40%, params=1,428,608
  dim=128: top-1=2.36%, top-5=8.07%, params=1,494,272
  dim=256: top-1=1.96%, top-5=7.38%, params=1,625,600

cell 7
STAGE 2 ROUND 5: Architecture (using BEST_STAGE1 + D*=128)
============================================================

============================================================
EXPERIMENT: R5-K: layernorm
============================================================
  arch=layernorm, embed_dim=128, epochs=50
  lr=0.001, schedule=cosine_warmup, tau=0.15 (fixed)
  batch_size=256, dropout=0.1
  Total parameters: 1,497,344
  Epoch   10/50: loss=2.4973, lr=0.000970, tau=0.1500
  Epoch   20/50: loss=1.4887, lr=0.000750, tau=0.1500
  Epoch   30/50: loss=1.1445, lr=0.000413, tau=0.1500
  Epoch   40/50: loss=1.0116, lr=0.000117, tau=0.1500
  Epoch   50/50: loss=0.9742, lr=0.000000, tau=0.1500

  RESULTS: top-1=2.87%, top-5=9.01%, top-10=13.42%, signal=47.5x
  Time: 33.2s, Final loss: 0.9742, Final tau: 0.1500

============================================================
EXPERIMENT: R5-L: deep
============================================================
  arch=deep, embed_dim=128, epochs=50
  lr=0.001, schedule=cosine_warmup, tau=0.15 (fixed)
  batch_size=256, dropout=0.1
  Total parameters: 2,548,992
  Epoch   10/50: loss=2.4878, lr=0.000970, tau=0.1500
  Epoch   20/50: loss=1.4508, lr=0.000750, tau=0.1500
  Epoch   30/50: loss=1.1407, lr=0.000413, tau=0.1500
  Epoch   40/50: loss=1.0134, lr=0.000117, tau=0.1500
  Epoch   50/50: loss=0.9813, lr=0.000000, tau=0.1500

  RESULTS: top-1=2.30%, top-5=8.16%, top-10=12.97%, signal=38.0x
  Time: 47.1s, Final loss: 0.9813, Final tau: 0.1500

============================================================
EXPERIMENT: R5-M: residual
============================================================
  arch=residual, embed_dim=128, epochs=50
  lr=0.001, schedule=cosine_warmup, tau=0.15 (fixed)
  batch_size=256, dropout=0.1
  Total parameters: 2,548,992
  Epoch   10/50: loss=2.4948, lr=0.000970, tau=0.1500
  Epoch   20/50: loss=1.4523, lr=0.000750, tau=0.1500
  Epoch   30/50: loss=1.1302, lr=0.000413, tau=0.1500
  Epoch   40/50: loss=1.0020, lr=0.000117, tau=0.1500
  Epoch   50/50: loss=0.9664, lr=0.000000, tau=0.1500

  RESULTS: top-1=2.69%, top-5=8.62%, top-10=13.66%, signal=44.5x
  Time: 45.7s, Final loss: 0.9664, Final tau: 0.1500

============================================================
ROUND 5 WINNER: R5-K: layernorm — top-1 = 2.87%
Selected A* = layernorm
  R4-I: dim=128: top-1=2.36%, params=1,494,272
  R5-K: layernorm: top-1=2.87%, params=1,497,344
  R5-L: deep: top-1=2.30%, params=2,548,992
  R5-M: residual: top-1=2.69%, params=2,548,992

cell 8
STAGE 2 ROUND 6: Batch Size (using BEST_STAGE1 + D*=128 + A*=layernorm)
============================================================

============================================================
EXPERIMENT: R6-N: batch=512
============================================================
  arch=layernorm, embed_dim=128, epochs=50
  lr=0.001, schedule=cosine_warmup, tau=0.15 (fixed)
  batch_size=512, dropout=0.1
  Total parameters: 1,497,344
  Epoch   10/50: loss=3.2066, lr=0.000970, tau=0.1500
  Epoch   20/50: loss=2.1829, lr=0.000750, tau=0.1500
  Epoch   30/50: loss=1.7863, lr=0.000413, tau=0.1500
  Epoch   40/50: loss=1.6240, lr=0.000117, tau=0.1500
  Epoch   50/50: loss=1.5812, lr=0.000000, tau=0.1500

  RESULTS: top-1=2.48%, top-5=8.16%, top-10=13.00%, signal=41.0x
  Time: 27.8s, Final loss: 1.5812, Final tau: 0.1500

============================================================
ROUND 6 WINNER: R5-K: layernorm — top-1 = 2.87%
Selected BS* = 256

cell 9
================================================================================
STAGE 2 RESULTS — ARCHITECTURE + DIMENSIONALITY SWEEP
================================================================================
Name                             Top-1   Top-5  Top-10   Signal     Params   Time
--------------------------------------------------------------------------------
[Stage1 best, dim=64]            2.21%   8.40%  12.94%    36.5x  1,428,608    31s
--------------------------------------------------------------------------------
R4-I: dim=128                    2.36%   8.07%  12.91%    39.0x  1,494,272    33s
R4-J: dim=256                    1.96%   7.38%  11.28%    32.5x  1,625,600    33s
R5-K: layernorm                  2.87%   9.01%  13.42%    47.5x  1,497,344    33s
R5-L: deep                       2.30%   8.16%  12.97%    38.0x  2,548,992    47s
R5-M: residual                   2.69%   8.62%  13.66%    44.5x  2,548,992    46s
R6-N: batch=512                  2.48%   8.16%  13.00%    41.0x  1,497,344    28s

================================================================================
BEST OVERALL CONFIG:
  epochs: 50
  lr: 0.001
  schedule: cosine_warmup
  tau: 0.15
  tau_mode: fixed
  embed_dim: 128
  arch: layernorm
  batch_size: 256
  dropout: 0.1
  weight_decay: 0.0001

Best top-1: 2.87%
Improvement over baseline (1.90%): 51%

figure at figure folder

cell 10
======================================================================
FULL PIPELINE RE-RUN WITH OPTIMISED CLIP ENCODER
======================================================================
Config: {'epochs': 50, 'lr': 0.001, 'schedule': 'cosine_warmup', 'tau': 0.15, 'tau_mode': 'fixed', 'embed_dim': 128, 'arch': 'layernorm', 'batch_size': 256, 'dropout': 0.1, 'weight_decay': 0.0001}

Step 1: Retraining CLIP encoder...

============================================================
EXPERIMENT: FINAL_BEST
============================================================
  arch=layernorm, embed_dim=128, epochs=50
  lr=0.001, schedule=cosine_warmup, tau=0.15 (fixed)
  batch_size=256, dropout=0.1
  Total parameters: 1,497,344
  Epoch   10/50: loss=2.4973, lr=0.000970, tau=0.1500
  Epoch   20/50: loss=1.4887, lr=0.000750, tau=0.1500
  Epoch   30/50: loss=1.1445, lr=0.000413, tau=0.1500
  Epoch   40/50: loss=1.0116, lr=0.000117, tau=0.1500
  Epoch   50/50: loss=0.9742, lr=0.000000, tau=0.1500

  RESULTS: top-1=2.87%, top-5=9.01%, top-10=13.42%, signal=47.5x
  Time: 32.8s, Final loss: 0.9742, Final tau: 0.1500
  Encoder top-1: 2.87%

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
  WGAN step 2000/10000: G=-0.5135, C=-0.0405
  WGAN step 4000/10000: G=-0.3832, C=-0.0345
  WGAN step 6000/10000: G=-0.5203, C=-0.0356
  WGAN step 8000/10000: G=-0.5352, C=-0.0366
  WGAN step 10000/10000: G=-0.4848, C=-0.0369
  WGAN-GP training complete.

Step 6: Generating synthetic unseen embeddings...
  E_synth_unseen: (4000, 128)

Step 7: GZSL evaluation (balanced counts)...

============================================================
OPTIMISED PIPELINE RESULTS
============================================================
  Encoder top-1 (seen-only): 2.87%
  AccS (GZSL):               2.96%
  AccU (GZSL):               0.40%
  H-mean:                    0.70%
  Routing rate:              11.31%
  Synth→Real transfer:       2.51%

cell 11
================================================================================
FINAL COMPARISON
================================================================================

Metric                                       Baseline   Phase1 Bal    Optimised
--------------------------------------------------------------------------------
Encoder top-1 (seen-only)                       1.90%        1.90%        2.87%
AccS (GZSL)                                     0.06%        2.90%        2.96%
AccU (GZSL)                                     2.13%        0.22%        0.40%
H-mean                                          0.12%        0.42%        0.70%
Routing rate                                   99.70%        9.16%       11.31%
Synth→Real transfer (200-way)                   1.73%        1.73%        2.51%
--------------------------------------------------------------------------------
Saved: figures/clip_opt_final_comparison.png

================================================================================
CLIP ENCODER OPTIMISATION COMPLETE
Best config: {'epochs': 50, 'lr': 0.001, 'schedule': 'cosine_warmup', 'tau': 0.15, 'tau_mode': 'fixed', 'embed_dim': 128, 'arch': 'layernorm', 'batch_size': 256, 'dropout': 0.1, 'weight_decay': 0.0001, 'name': 'FINAL_BEST'}