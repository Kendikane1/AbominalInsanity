Hyperparameter Sweep for CORnet-S Image Alignment output notebook cells :

cell 1:
======================================================================
SWEEP PHASE 1: Epoch x Temperature Grid
======================================================================

[1/6] Training with tau=0.05, 200 epochs...
    Ckpt epoch 50: top1=0.0378, top5=0.1206, loss=0.0163
    Ckpt epoch 75: top1=0.0411, top5=0.1218, loss=0.0096
    Ckpt epoch 100: top1=0.0387, top5=0.1255, loss=0.0062
    Ckpt epoch 125: top1=0.0393, top5=0.1227, loss=0.0044
    Ckpt epoch 150: top1=0.0372, top5=0.1206, loss=0.0027
    Ckpt epoch 200: top1=0.0384, top5=0.1200, loss=0.0018
  Done in 169.3s. Final loss=0.0018, top1=0.0384

[2/6] Training with tau=0.07, 200 epochs...
    Ckpt epoch 50: top1=0.0363, top5=0.1224, loss=0.0461
    Ckpt epoch 75: top1=0.0387, top5=0.1258, loss=0.0291
    Ckpt epoch 100: top1=0.0345, top5=0.1215, loss=0.0214
    Ckpt epoch 125: top1=0.0363, top5=0.1197, loss=0.0162
    Ckpt epoch 150: top1=0.0369, top5=0.1197, loss=0.0132
    Ckpt epoch 200: top1=0.0378, top5=0.1167, loss=0.0107
  Done in 169.8s. Final loss=0.0107, top1=0.0378

[3/6] Training with tau=0.1, 200 epochs...
    Ckpt epoch 50: top1=0.0302, top5=0.1176, loss=0.2224
    Ckpt epoch 75: top1=0.0363, top5=0.1122, loss=0.1791
    Ckpt epoch 100: top1=0.0363, top5=0.1094, loss=0.1561
    Ckpt epoch 125: top1=0.0357, top5=0.1131, loss=0.1403
    Ckpt epoch 150: top1=0.0323, top5=0.1137, loss=0.1280
    Ckpt epoch 200: top1=0.0333, top5=0.1155, loss=0.1192
  Done in 168.4s. Final loss=0.1192, top1=0.0333

[4/6] Training with tau=0.12, 200 epochs...
    Ckpt epoch 50: top1=0.0339, top5=0.1091, loss=0.4782
    Ckpt epoch 75: top1=0.0333, top5=0.1082, loss=0.4162
    Ckpt epoch 100: top1=0.0326, top5=0.1073, loss=0.3780
    Ckpt epoch 125: top1=0.0314, top5=0.1073, loss=0.3504
    Ckpt epoch 150: top1=0.0320, top5=0.1046, loss=0.3327
    Ckpt epoch 200: top1=0.0333, top5=0.1076, loss=0.3184
  Done in 166.5s. Final loss=0.3184, top1=0.0333

[5/6] Training with tau=0.15, 200 epochs...
    Ckpt epoch 50: top1=0.0345, top5=0.1103, loss=0.9800
    Ckpt epoch 75: top1=0.0369, top5=0.1061, loss=0.9153
    Ckpt epoch 100: top1=0.0354, top5=0.1103, loss=0.8783
    Ckpt epoch 125: top1=0.0314, top5=0.1058, loss=0.8481
    Ckpt epoch 150: top1=0.0351, top5=0.1034, loss=0.8261
    Ckpt epoch 200: top1=0.0333, top5=0.1058, loss=0.8090
  Done in 168.0s. Final loss=0.8090, top1=0.0333

[6/6] Training with tau=0.2, 200 epochs...
    Ckpt epoch 50: top1=0.0308, top5=0.1049, loss=1.7779
    Ckpt epoch 75: top1=0.0366, top5=0.1070, loss=1.7238
    Ckpt epoch 100: top1=0.0339, top5=0.1037, loss=1.6903
    Ckpt epoch 125: top1=0.0375, top5=0.1040, loss=1.6636
    Ckpt epoch 150: top1=0.0339, top5=0.1037, loss=1.6444
    Ckpt epoch 200: top1=0.0348, top5=0.1049, loss=1.6273
  Done in 169.3s. Final loss=1.6273, top1=0.0348

======================================================================
PHASE 1 RESULT: best_tau=0.05, best_epochs=75, top1=0.0411
Phase 1 total time: 1011.1s (16.9min)
======================================================================

cell 2: 
Saved: figures/sweep_phase1_heatmap.png

   tau | ep= 50  ep= 75  ep=100  ep=125  ep=150  ep=200  
--------------------------------------------------------
  0.05 | 0.0378  0.0411  0.0387  0.0393  0.0372  0.0384  
  0.07 | 0.0363  0.0387  0.0345  0.0363  0.0369  0.0378  
  0.10 | 0.0302  0.0363  0.0363  0.0357  0.0323  0.0333  
  0.12 | 0.0339  0.0333  0.0326  0.0314  0.0320  0.0333  
  0.15 | 0.0345  0.0369  0.0354  0.0314  0.0351  0.0333  
  0.20 | 0.0308  0.0366  0.0339  0.0375  0.0339  0.0348  

cell 3:
======================================================================
SWEEP PHASE 2: Embedding Dimension (tau=0.05, epochs=75)
======================================================================

[1/4] embed_dim=64...
  Done in 60.4s. top1=0.0417, loss=0.0105

[2/4] embed_dim=128...
  Done in 62.6s. top1=0.0390, loss=0.0103

[3/4] embed_dim=256...
  Done in 63.6s. top1=0.0396, loss=0.0102

[4/4] embed_dim=512...
  Done in 68.6s. top1=0.0417, loss=0.0107

Saved: figures/sweep_phase2_embed_dim.png

======================================================================
PHASE 2 RESULT: best_dim=64, top1=0.0417
Phase 2 total time: 255.3s (4.3min)
======================================================================

cell 4:
======================================================================
SWEEP PHASE 3: ImageProjector Architecture (tau=0.05, epochs=75, dim=64)
======================================================================

[1/6] Architecture A: [256]...
  Done in 55.9s. top1=0.0366, loss=0.0166, proj_params=273,216

[2/6] Architecture B: [512]...
  Done in 60.1s. top1=0.0417, loss=0.0105, proj_params=546,368

[3/6] Architecture C: [768]...
  Done in 65.8s. top1=0.0411, loss=0.0085, proj_params=819,520

[4/6] Architecture D: [512,256]...
  Done in 66.7s. top1=0.0354, loss=0.0165, proj_params=661,824

[5/6] Architecture E: [768,384]...
  Done in 75.2s. top1=0.0378, loss=0.0108, proj_params=1,091,008

[6/6] Architecture F: [1024,512]...
  Done in 83.6s. top1=0.0399, loss=0.0081, proj_params=1,585,728

Architecture            Top-1    Top-5   Top-10     Loss
--------------------------------------------------------
A: [256]               0.0366   0.1215   0.1814   0.0166
B: [512]               0.0417   0.1294   0.1929   0.0105
C: [768]               0.0411   0.1282   0.1920   0.0085
D: [512,256]           0.0354   0.1152   0.1765   0.0165
E: [768,384]           0.0378   0.1206   0.1781   0.0108
F: [1024,512]          0.0399   0.1191   0.1841   0.0081

======================================================================
PHASE 3 RESULT: best_arch=[512], top1=0.0417
Phase 3 total time: 407.3s (6.8min)
======================================================================

cell 5:
======================================================================
SWEEP PHASE 4: LR x Weight Decay (tau=0.05, epochs=75, dim=64, arch=[512])
======================================================================

[1/9] lr=0.0005, wd=0...
  Done in 60.8s. top1=0.0429, loss=0.0179

[2/9] lr=0.0005, wd=0.0001...
  Done in 60.7s. top1=0.0435, loss=0.0179

[3/9] lr=0.0005, wd=0.001...
  Done in 60.3s. top1=0.0414, loss=0.0179

[4/9] lr=0.001, wd=0...
  Done in 59.9s. top1=0.0423, loss=0.0105

[5/9] lr=0.001, wd=0.0001...
  Done in 59.8s. top1=0.0417, loss=0.0105

[6/9] lr=0.001, wd=0.001...
  Done in 60.3s. top1=0.0444, loss=0.0106

[7/9] lr=0.002, wd=0...
  Done in 60.2s. top1=0.0414, loss=0.0077

[8/9] lr=0.002, wd=0.0001...
  Done in 60.3s. top1=0.0447, loss=0.0077

[9/9] lr=0.002, wd=0.001...
  Done in 60.8s. top1=0.0438, loss=0.0075

Saved: figures/sweep_phase4_lr_wd.png

======================================================================
PHASE 4 RESULT: best_lr=0.002, best_wd=0.0001, top1=0.0447
Phase 4 total time: 543.1s (9.1min)
======================================================================

cell 6:
======================================================================
SWEEP COMPLETE — OPTIMAL CONFIGURATION
======================================================================

Parameter                        Baseline         Optimal    Changed
-----------------------------------------------------------------
embed_dim                             128              64          *
tau                                  0.15            0.05          *
epochs                                 50              75          *
lr                                  0.001           0.002          *
weight_decay                       0.0001          0.0001           
image_hidden_dims                   [512]           [512]           

Best encoder-only top-1 from sweep: 0.0447

Full OPTIMAL_CONFIG:
  embed_dim: 64
  tau: 0.05
  epochs: 75
  lr: 0.002
  weight_decay: 0.0001
  dropout: 0.1
  batch_size: 256
  warmup_ratio: 0.1
  image_input_dim: 1000
  image_hidden_dims: [512]
  brain_hidden_dims: [1024, 512]
  seed: 42
  alignment_target: image (CORnet-S)
======================================================================

cell 7:
======================================================================
SWEEP PHASE 5: Full Pipeline Validation with Optimal Config
======================================================================

Training encoder with optimal config...
  tau=0.05, epochs=75, dim=64, arch=[512], lr=0.002, wd=0.0001

Encoder trained in 60.1s. loss=0.0077, top1=0.0447

Embeddings: E_train=(13232, 64), E_test=(3308, 64), E_unseen=(16000, 64)
Prototypes: seen=(1654, 64), unseen=(200, 64)

Embeddings computed and cached. Ready for WGAN-GP.

cell 8:
Training cWGAN-GP with embed_dim=64...
  Step     1/10000: G=0.0631, C=8.2205
  Step  2000/10000: G=-1.3000, C=-0.0227
  Step  4000/10000: G=-1.4423, C=-0.0260
  Step  6000/10000: G=-1.3287, C=-0.0204
  Step  8000/10000: G=-1.3355, C=-0.0342
  Step 10000/10000: G=-1.3713, C=-0.0380

cWGAN-GP complete. Final G=-1.3713, C=-0.0380

Synthetic: (4000, 64), norms=1.0000
Per-dim variance: Real=0.0156, Synth=0.0155
Synthetic embeddings generated and cached.

cell 9:
Sample balance: seen median=8/class, synth downsampled to 1600 (was 4000)
GZSL training: 14832 samples, 1854 classes

Training GZSL classifier...
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:1247: FutureWarning: 'multi_class' was deprecated in version 1.5 and will be removed in 1.7. From then on, it will always use 'multinomial'. Leave it to its default value to avoid this warning.
  warnings.warn(
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 24 concurrent workers.

============================================================
GZSL EVALUATION — Optimal Config
============================================================
  Acc Seen:      0.0426
  Acc Unseen:    0.0486
  Harmonic Mean: 0.0454
  F1 Seen:       0.0262
  F1 Unseen:     0.0096
  Routing Rate (seen→unseen): 0.1623 (537/3308)

  Bias Table:
                         Pred Seen    Pred Unseen
    True Seen                2771         537  (16.2% misrouted)
    True Unseen             12006        3994
============================================================

============================================================
CLASSIFIER DIAGNOSTICS — Optimal Config
============================================================
  Weight norms ||w_c||:
    Seen   — mean: 3.3908, std: 0.5040, min: 2.0610, max: 5.1291
    Unseen — mean: 3.7423, std: 0.3062, min: 2.7682, max: 4.4575
    Ratio (unseen/seen mean): 1.10x

  Intercepts (biases) β_c:
    Seen   — mean: 0.0044, std: 0.0679
    Unseen — mean: -0.0363, std: 0.0500
============================================================

Saved: figures/optimal_config_diagnostics.png

cell 10:
======================================================================
SWEEP PHASE 5: Baseline vs Optimal Config Comparison
======================================================================

Metric                   Baseline      Optimal        Delta        Rel
--------------------------------------------------------------------
AccS                       0.0375       0.0426      +0.0051     +13.7%
AccU                       0.0573       0.0486      -0.0087     -15.2%
H-mean                     0.0453       0.0454      +0.0001      +0.2%
F1 Seen                    0.0225       0.0262      +0.0037     +16.4%
F1 Unseen                  0.0120       0.0096      -0.0024     -20.3%
Routing Rate               0.2034       0.1623      -0.0411     -20.2%
--------------------------------------------------------------------

Encoder top-1                 N/A       0.0447
Encoder loss               0.9342       0.0077

Config changes:
  embed_dim: 128 -> 64
  tau: 0.15 -> 0.05
  epochs: 50 -> 75
  lr: 0.001 -> 0.002

Saved: figures/sweep_baseline_vs_optimal.png

cell 11:
======================================================================
HYPERPARAMETER SWEEP — FINAL REPORT
======================================================================

--- Phase Results ---
Phase 1 (Epoch x Tau):    tau=0.05, epochs=75, top1=0.0411
Phase 2 (Embed Dim):      dim=64, top1=0.0417
Phase 3 (Architecture):   arch=[512], top1=0.0417
Phase 4 (LR x WD):        lr=0.002, wd=0.0001, top1=0.0447

--- Full Pipeline (Phase 5) ---
  AccS:         0.0426 (4.26%)
  AccU:         0.0486 (4.86%)
  H-mean:       0.0454
  Routing Rate: 0.1623
  Encoder Loss: 0.0077

--- Improvement over Baseline ---
  H-mean: 0.0453 -> 0.0454 (+0.0001, +0.2%)

--- Optimal Config ---
  embed_dim: 64
  tau: 0.05
  epochs: 75
  lr: 0.002
  weight_decay: 0.0001
  dropout: 0.1
  batch_size: 256
  warmup_ratio: 0.1
  image_input_dim: 1000
  image_hidden_dims: [512]
  brain_hidden_dims: [1024, 512]
  seed: 42
  alignment_target: image (CORnet-S)

--- Figures Generated ---
  figures/sweep_phase1_heatmap.png
  figures/sweep_phase2_embed_dim.png
  figures/sweep_phase4_lr_wd.png
  figures/sweep_optimal_loss_curve.png
  figures/sweep_baseline_vs_optimal.png
  figures/optimal_config_diagnostics.png

======================================================================
SWEEP COMPLETE
======================================================================























