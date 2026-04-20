cell 1:
============================================================
VARIANCE REGULARISATION CONFIG
============================================================
  K_gen: 20
  C_batch: 16
  monitor_interval: 500
  alpha_sweep_multipliers: [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

  Generator batch size: 320
  Critic batch size: 256 (unchanged)
============================================================
Functions defined: compute_L_var, train_wgan_with_lvar, calibrate_alpha

cell 2:
Computing var_target from training seen embeddings...

var_target statistics:
  Shape: torch.Size([64])
  All positive: True
  Mean:  0.010926
  Std:   0.001865
  Min:   0.005930
  Max:   0.015431
  Sum:   0.699247 (L2-norm budget: should be < 1.0)
  Classes: 1654
  Samples/class: min=8, max=8, median=8

============================================================
CORRECTNESS VERIFICATION
============================================================
  [PASS] Check 1: var_target shape (64,) and all positive
  [PASS] Check 2: compute_L_var returns scalar (dim=0) with requires_grad=True
  [PASS] Check 3: L_var=0 when variance matches target (L_var=9.90e-17)
  [PASS] Check 4: Gradient verification (max |autograd - analytical| = 5.96e-08, threshold=1e-4)
  [PASS] Check 5: Generator outputs on S^63 (max |norm - 1| = 1.19e-07)
============================================================
ALL CHECKS PASSED. Ready to proceed with alpha calibration.
============================================================


cell 3:
Running alpha calibration...

============================================================
ALPHA CALIBRATION RESULTS
============================================================
  ||grad L_wasserstein||: 0.181351
  ||grad L_var||:         0.009135
  alpha_0 (norm ratio):   19.852067
  L_wasserstein:          0.0669
  L_var:                  0.001272
  cos(grad_w, grad_v):    -0.1497

Alpha sweep values (7 points, 2 OOM range):
   0.01 x alpha_0 = 0.198521
   0.10 x alpha_0 = 1.985207
   0.50 x alpha_0 = 9.926033
   1.00 x alpha_0 = 19.852067 <-- alpha_0
   2.00 x alpha_0 = 39.704133
   5.00 x alpha_0 = 99.260334
  10.00 x alpha_0 = 198.520667
============================================================


cell 4:
Running alpha calibration...

============================================================
ALPHA CALIBRATION RESULTS
============================================================
  ||grad L_wasserstein||: 0.181351
  ||grad L_var||:         0.009135
  alpha_0 (norm ratio):   19.852067
  L_wasserstein:          0.0669
  L_var:                  0.001272
  cos(grad_w, grad_v):    -0.1497

Alpha sweep values (7 points, 2 OOM range):
   0.01 x alpha_0 = 0.198521
   0.10 x alpha_0 = 1.985207
   0.50 x alpha_0 = 9.926033
   1.00 x alpha_0 = 19.852067 <-- alpha_0
   2.00 x alpha_0 = 39.704133
   5.00 x alpha_0 = 99.260334
  10.00 x alpha_0 = 198.520667
============================================================

cell 5:
Saved: figures/var_reg_alpha_sweep.png

Saved: figures/var_reg_training_dynamics.png

============================================================
COMPARISON WITH BASELINE
============================================================
              Metric   Baseline Best alpha      Delta
-------------------------------------------------------
          H-mean (%)       4.77       4.58      -0.19
            AccS (%)       4.11       3.81      -0.30
            AccU (%)       5.69       5.76      +0.07
         Routing (%)       20.0       17.8       -2.2
                VarR      0.872      0.875     +0.003
              rho_sp     0.8572     0.8798    +0.0226
              rho_sr     0.5875     0.5847    -0.0028
              kNN@10      0.611      0.606     -0.005
               alpha        --- 198.520667
          alpha_mult        ---      10.00
-------------------------------------------------------

rho_sp did NOT decrease — L_var addressed variance but not overcoupling.
Next intervention if needed: L_struct (inter-class structure loss).

cell 6:
Best alpha: 198.520667 (10.00x alpha_0)
H-mean: 4.58%

Retraining best-alpha WGAN-GP for caching...
  Step     1/10000: L_w=0.0661, L_var=0.001266, L_D=8.2193, VarR=0.710
  Step  2500/10000: L_w=0.2236, L_var=0.000771, L_D=-0.0255, VarR=0.928
  Step  5000/10000: L_w=0.2074, L_var=0.000720, L_D=-0.0310, VarR=0.964
  Step  7500/10000: L_w=0.0876, L_var=0.000796, L_D=-0.0572, VarR=0.969
  Step 10000/10000: L_w=0.1678, L_var=0.000778, L_D=-0.0694, VarR=0.973
  Training complete in 456s. L_w=0.1678, L_var=0.000778, VarR=0.973

Generating synthetic unseen embeddings...
E_synth_vareg: (4000, 64)
y_synth_vareg: (4000,)

Cached: E_synth_vareg.npy, y_synth_vareg.npy
Cached: generator_vareg_best.pt
Cached: var_reg_sweep_results.json

============================================================
VARIANCE REGULARISATION EXPERIMENT COMPLETE
============================================================
  Best alpha: 198.520667 (10.00x alpha_0)
  H-mean:     4.58% (baseline: 4.77%, delta: -0.19pp)
  VarR:       0.875 (baseline: 0.872, delta: +0.003)
  rho_sp:     0.8798 (baseline: 0.8572, delta: +0.0226)
============================================================
