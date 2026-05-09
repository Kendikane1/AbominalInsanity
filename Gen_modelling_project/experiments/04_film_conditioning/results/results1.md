for notebook.ipynb : 

wandb: WARNING Using a boolean value for 'reinit' is deprecated. Use 'return_previous' or 'finish_previous' instead.
Tracking run with wandb version 0.26.1
Run data is saved locally in /content/wandb/run-20260507_202812-tbn39fif
Syncing run 04_film_baseline to Weights & Biases (docs)
View project at https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl
View run at https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl/runs/tbn39fif
Training FiLM cWGAN-GP for 10000 generator steps...
Critic updates per G step: 5
  Step     1/10000: G=0.0173, C=8.4834, GP=0.8474
  Step   500/10000: G=-1.5538, C=0.0007, GP=0.0038
  Step  1000/10000: G=-1.3929, C=0.0142, GP=0.0034, VarR_seen=0.795
  Step  1500/10000: G=-1.3364, C=0.0016, GP=0.0026
  Step  2000/10000: G=-1.3045, C=-0.0085, GP=0.0018, VarR_seen=0.935
  Step  2500/10000: G=-1.3847, C=-0.0049, GP=0.0015
  Step  3000/10000: G=-1.4685, C=-0.0022, GP=0.0016, VarR_seen=0.930
  Step  3500/10000: G=-1.4914, C=-0.0121, GP=0.0016
  Step  4000/10000: G=-1.4883, C=-0.0120, GP=0.0013, VarR_seen=0.912
  Step  4500/10000: G=-1.5468, C=-0.0090, GP=0.0017
  Step  5000/10000: G=-1.5703, C=-0.0160, GP=0.0016, VarR_seen=0.910
  Step  5500/10000: G=-1.5633, C=-0.0146, GP=0.0019
  Step  6000/10000: G=-1.5467, C=-0.0011, GP=0.0018, VarR_seen=0.913
  Step  6500/10000: G=-1.5708, C=-0.0174, GP=0.0017
  Step  7000/10000: G=-1.5749, C=-0.0117, GP=0.0017, VarR_seen=0.945
  Step  7500/10000: G=-1.5777, C=-0.0138, GP=0.0016
  Step  8000/10000: G=-1.5636, C=-0.0146, GP=0.0017, VarR_seen=0.912
  Step  8500/10000: G=-1.5728, C=-0.0125, GP=0.0019
  Step  9000/10000: G=-1.5657, C=-0.0203, GP=0.0017, VarR_seen=0.902
  Step  9500/10000: G=-1.5786, C=-0.0057, GP=0.0018
  Step 10000/10000: G=-1.6199, C=-0.0157, GP=0.0018, VarR_seen=0.901

FiLM cWGAN-GP training complete!

============================================================
VARIANCE RATIO ANALYSIS
============================================================
  VarR (seen  classes, training): 0.9247
  VarR (unseen classes, eval):    0.8468
  Transfer gap (seen - unseen):   0.0779

  Reference baselines:
    Exp 01 (concat WGAN-GP):  VarR_unseen = 0.872
    Exp 03 (L_var training):  VarR_seen=0.973, VarR_unseen=0.875, gap=0.098
    FiLM target:              VarR_unseen > 0.95, gap < 0.03
FiLMGenerator(
  (fc1): Linear(in_features=100, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=256, bias=True)
  (fc3): Linear(in_features=256, out_features=64, bias=True)
  (film1_mlp): Sequential(
    (0): Linear(in_features=64, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=512, bias=True)
  )
  (film2_mlp): Sequential(
    (0): Linear(in_features=64, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=512, bias=True)
  )
)

============================================================
STRUCTURAL OVERCOUPLING DIAGNOSTICS
============================================================
  rho_sp (centroid vs prototype):   0.6387  (p=0.00e+00)
  kNN@10 neighbourhood preservation: 0.4630

  Reference: Exp 01 baseline: rho_sp=0.857, kNN@10=0.611
             Exp 03 L_var:    rho_sp=0.880 (worse, overcoupling increased)
             Real seen data:  rho_sp~0.668
  FiLM target: rho_sp < 0.800

  ============================================================
GZSL HARMONIC MEAN
============================================================
H = 2 * 0.0417 * 0.0509 / (0.0417 + 0.0509)
H = 0.0459

for ln.ipynb :

wandb: WARNING Using a boolean value for 'reinit' is deprecated. Use 'return_previous' or 'finish_previous' instead.
Tracking run with wandb version 0.26.1
Run data is saved locally in /content/wandb/run-20260507_202225-u8bxy265
Syncing run 04_film_layernorm to Weights & Biases (docs)
View project at https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl
View run at https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl/runs/u8bxy265
Training FiLM cWGAN-GP for 10000 generator steps...
Critic updates per G step: 5
  Step     1/10000: G=0.0261, C=8.4864, GP=0.8486
  Step   500/10000: G=-1.6017, C=-0.0685, GP=0.0038
  Step  1000/10000: G=-1.1295, C=0.0257, GP=0.0038, VarR_seen=0.807
  Step  1500/10000: G=-0.4962, C=0.0185, GP=0.0033
  Step  2000/10000: G=-0.3075, C=0.0023, GP=0.0017, VarR_seen=0.930
  Step  2500/10000: G=-0.4079, C=-0.0063, GP=0.0017
  Step  3000/10000: G=-0.3941, C=-0.0123, GP=0.0014, VarR_seen=0.923
  Step  3500/10000: G=-0.4613, C=-0.0271, GP=0.0017
  Step  4000/10000: G=-0.5309, C=-0.0256, GP=0.0014, VarR_seen=0.922
  Step  4500/10000: G=-0.5756, C=-0.0356, GP=0.0014
  Step  5000/10000: G=-0.6007, C=-0.0122, GP=0.0030, VarR_seen=0.913
  Step  5500/10000: G=-0.5553, C=-0.0298, GP=0.0014
  Step  6000/10000: G=-0.6029, C=-0.0307, GP=0.0014, VarR_seen=0.947
  Step  6500/10000: G=-0.5625, C=-0.0235, GP=0.0021
  Step  7000/10000: G=-0.6484, C=-0.0427, GP=0.0016, VarR_seen=0.954
  Step  7500/10000: G=-0.7060, C=-0.0488, GP=0.0018
  Step  8000/10000: G=-0.7318, C=-0.0364, GP=0.0022, VarR_seen=0.911
  Step  8500/10000: G=-0.7296, C=-0.0362, GP=0.0018
  Step  9000/10000: G=-0.7181, C=-0.0419, GP=0.0022, VarR_seen=0.920
  Step  9500/10000: G=-0.7258, C=-0.0463, GP=0.0021
  Step 10000/10000: G=-0.5855, C=-0.0395, GP=0.0022, VarR_seen=0.916

FiLM cWGAN-GP training complete!

============================================================
VARIANCE RATIO ANALYSIS
============================================================
  VarR (seen  classes, training): 0.9367
  VarR (unseen classes, eval):    0.8495
  Transfer gap (seen - unseen):   0.0873

  Reference baselines:
    Exp 01 (concat WGAN-GP):  VarR_unseen = 0.872
    Exp 03 (L_var training):  VarR_seen=0.973, VarR_unseen=0.875, gap=0.098
    FiLM target:              VarR_unseen > 0.95, gap < 0.03
FiLMGeneratorLN(
  (fc1): Linear(in_features=100, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=256, bias=True)
  (fc3): Linear(in_features=256, out_features=64, bias=True)
  (film1_mlp): Sequential(
    (0): Linear(in_features=64, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=512, bias=True)
  )
  (film2_mlp): Sequential(
    (0): Linear(in_features=64, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=512, bias=True)
  )
  (ln1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
  (ln2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
)

============================================================
STRUCTURAL OVERCOUPLING DIAGNOSTICS
============================================================
  rho_sp (centroid vs prototype):   0.7155  (p=0.00e+00)
  kNN@10 neighbourhood preservation: 0.5115

  Reference: Exp 01 baseline: rho_sp=0.857, kNN@10=0.611
             Exp 03 L_var:    rho_sp=0.880 (worse, overcoupling increased)
             Real seen data:  rho_sp~0.668
  FiLM target: rho_sp < 0.800

============================================================
GZSL HARMONIC MEAN
============================================================
H = 2 * 0.0390 * 0.0574 / (0.0390 + 0.0574)
H = 0.0464

for lvar.ipynb :


============================================================
Alpha = 0.00  (L_G = L_wass + 0.00 * L_var)
============================================================
wandb: WARNING Using a boolean value for 'reinit' is deprecated. Use 'return_previous' or 'finish_previous' instead.
Tracking run with wandb version 0.26.1
Run data is saved locally in /content/wandb/run-20260507_202120-5yl2ziuv
Syncing run 04_film_lvar_alpha0.00 to Weights & Biases (docs)
View project at https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl
View run at https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl/runs/5yl2ziuv
  a=0.00 step     1: G=-0.0405, Lvar=0.0000
  a=0.00 step  1000: G=-1.3833, Lvar=0.0000, VarR_seen=0.856
  a=0.00 step  2000: G=-1.3952, Lvar=0.0000, VarR_seen=0.934
  a=0.00 step  3000: G=-1.4103, Lvar=0.0000, VarR_seen=0.904
  a=0.00 step  4000: G=-1.4972, Lvar=0.0000, VarR_seen=0.913
  a=0.00 step  5000: G=-1.5711, Lvar=0.0000, VarR_seen=0.945
  a=0.00 step  6000: G=-1.6620, Lvar=0.0000, VarR_seen=0.946
  a=0.00 step  7000: G=-1.6678, Lvar=0.0000, VarR_seen=0.898
  a=0.00 step  8000: G=-1.7283, Lvar=0.0000, VarR_seen=0.915
  a=0.00 step  9000: G=-1.7876, Lvar=0.0000, VarR_seen=0.907
  a=0.00 step 10000: G=-1.8431, Lvar=0.0000, VarR_seen=0.920
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:1247: FutureWarning: 'multi_class' was deprecated in version 1.5 and will be removed in 1.7. From then on, it will always use 'multinomial'. Leave it to its default value to avoid this warning.
  warnings.warn(


Run history:

eval/AccS	▁
eval/AccU	▁
eval/H_mean	▁
eval/VarR	▁
eval/rho_sp	▁
eval/routing_rate	▁
train/GP	█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
train/L_D	█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
train/L_G	█▂▃▃▃▃▃▃▂▂▂▂▂▂▂▁▁▁▁▁▁
train/L_var	▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
+7	...

Run summary:

eval/AccS	0.03839
eval/AccU	0.05387
eval/H_mean	0.04483
eval/VarR	0.84444
eval/rho_sp	0.65332
eval/routing_rate	0.31744
train/GP	0.00165
train/L_D	-0.02765
train/L_G	-1.84308
train/L_var	0
+7	...

View run 04_film_lvar_alpha0.00 at: https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl/runs/5yl2ziuv
View project at: https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl
Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
Find logs at: ./wandb/run-20260507_202120-5yl2ziuv/logs
  alpha=0.00: H=0.0448, AccS=0.0384, AccU=0.0539, VarR_unseen=0.8444, rho_sp=0.6533

============================================================
Alpha = 0.10  (L_G = L_wass + 0.10 * L_var)
============================================================
Tracking run with wandb version 0.26.1
Run data is saved locally in /content/wandb/run-20260507_202809-xxxtxuzf
Syncing run 04_film_lvar_alpha0.10 to Weights & Biases (docs)
View project at https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl
View run at https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl/runs/xxxtxuzf
  a=0.10 step     1: G=-0.0414, Lvar=-0.0087
  a=0.10 step  1000: G=-1.4124, Lvar=-0.0113, VarR_seen=0.862, cos=0.069
  a=0.10 step  2000: G=-1.3485, Lvar=-0.0105, VarR_seen=0.942, cos=0.023
  a=0.10 step  3000: G=-1.4449, Lvar=-0.0114, VarR_seen=0.915, cos=0.064
  a=0.10 step  4000: G=-1.5295, Lvar=-0.0110, VarR_seen=0.914, cos=-0.062
  a=0.10 step  5000: G=-1.5983, Lvar=-0.0113, VarR_seen=0.942, cos=-0.006
  a=0.10 step  6000: G=-1.6875, Lvar=-0.0115, VarR_seen=0.930
  a=0.10 step  7000: G=-1.7423, Lvar=-0.0118, VarR_seen=0.901
  a=0.10 step  8000: G=-1.7214, Lvar=-0.0113, VarR_seen=0.907
  a=0.10 step  9000: G=-1.7863, Lvar=-0.0104, VarR_seen=0.912
  a=0.10 step 10000: G=-1.8817, Lvar=-0.0110, VarR_seen=0.914
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:1247: FutureWarning: 'multi_class' was deprecated in version 1.5 and will be removed in 1.7. From then on, it will always use 'multinomial'. Leave it to its default value to avoid this warning.
  warnings.warn(


Run history:

eval/AccS	▁
eval/AccU	▁
eval/H_mean	▁
eval/VarR	▁
eval/rho_sp	▁
eval/routing_rate	▁
train/GP	█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
train/L_D	█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
train/L_G	█▂▃▃▃▃▃▃▂▂▂▂▂▂▂▂▂▁▁▁▁
train/L_var	█▇▃▅▄▃▂▂▃▁▃▅▂▃▂▅▃▂▅▁▃
+8	...

Run summary:

eval/AccS	0.0399
eval/AccU	0.05212
eval/H_mean	0.0452
eval/VarR	0.84718
eval/rho_sp	0.66566
eval/routing_rate	0.3065
train/GP	0.00192
train/L_D	-0.01984
train/L_G	-1.88174
train/L_var	-0.01095
+8	...

View run 04_film_lvar_alpha0.10 at: https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl/runs/xxxtxuzf
View project at: https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl
Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
Find logs at: ./wandb/run-20260507_202809-xxxtxuzf/logs
  alpha=0.10: H=0.0452, AccS=0.0399, AccU=0.0521, VarR_unseen=0.8472, rho_sp=0.6657

============================================================
Alpha = 0.50  (L_G = L_wass + 0.50 * L_var)
============================================================
Tracking run with wandb version 0.26.1
Run data is saved locally in /content/wandb/run-20260507_203523-0dzr3b1c
Syncing run 04_film_lvar_alpha0.50 to Weights & Biases (docs)
View project at https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl
View run at https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl/runs/0dzr3b1c
  a=0.50 step     1: G=-0.0449, Lvar=-0.0087
  a=0.50 step  1000: G=-1.4059, Lvar=-0.0104, VarR_seen=0.837, cos=0.089
  a=0.50 step  2000: G=-1.3560, Lvar=-0.0109, VarR_seen=0.935, cos=0.039
  a=0.50 step  3000: G=-1.4711, Lvar=-0.0112, VarR_seen=0.912, cos=-0.002
  a=0.50 step  4000: G=-1.5856, Lvar=-0.0111, VarR_seen=0.930, cos=-0.076
  a=0.50 step  5000: G=-1.6681, Lvar=-0.0108, VarR_seen=0.947, cos=-0.002
  a=0.50 step  6000: G=-1.7209, Lvar=-0.0108, VarR_seen=0.939
  a=0.50 step  7000: G=-1.8181, Lvar=-0.0105, VarR_seen=0.908
  a=0.50 step  8000: G=-1.8872, Lvar=-0.0110, VarR_seen=0.909
  a=0.50 step  9000: G=-2.0009, Lvar=-0.0108, VarR_seen=0.915
  a=0.50 step 10000: G=-2.0875, Lvar=-0.0117, VarR_seen=0.929
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:1247: FutureWarning: 'multi_class' was deprecated in version 1.5 and will be removed in 1.7. From then on, it will always use 'multinomial'. Leave it to its default value to avoid this warning.
  warnings.warn(


Run history:

eval/AccS	▁
eval/AccU	▁
eval/H_mean	▁
eval/VarR	▁
eval/rho_sp	▁
eval/routing_rate	▁
train/GP	█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
train/L_D	█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
train/L_G	█▃▃▄▄▃▃▃▃▃▂▂▂▂▂▂▂▁▁▁▁
train/L_var	█▆▄▄▄▃▃▁▃▂▄▃▄▁▄▅▃▃▄▂▂
+8	...

Run summary:

eval/AccS	0.0396
eval/AccU	0.05525
eval/H_mean	0.04613
eval/VarR	0.85017
eval/rho_sp	0.66738
eval/routing_rate	0.31519
train/GP	0.00163
train/L_D	-0.02257
train/L_G	-2.08751
train/L_var	-0.0117
+8	...

View run 04_film_lvar_alpha0.50 at: https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl/runs/0dzr3b1c
View project at: https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl
Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
Find logs at: ./wandb/run-20260507_203523-0dzr3b1c/logs
  alpha=0.50: H=0.0461, AccS=0.0396, AccU=0.0553, VarR_unseen=0.8502, rho_sp=0.6674

============================================================
Alpha = 1.00  (L_G = L_wass + 1.00 * L_var)
============================================================
Tracking run with wandb version 0.26.1
Run data is saved locally in /content/wandb/run-20260507_204239-73nx5dja
Syncing run 04_film_lvar_alpha1.00 to Weights & Biases (docs)
View project at https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl
View run at https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl/runs/73nx5dja
  a=1.00 step     1: G=-0.0492, Lvar=-0.0087
  a=1.00 step  1000: G=-1.4240, Lvar=-0.0108, VarR_seen=0.844, cos=0.064
  a=1.00 step  2000: G=-1.3947, Lvar=-0.0114, VarR_seen=0.937, cos=0.006
  a=1.00 step  3000: G=-1.5252, Lvar=-0.0109, VarR_seen=0.916, cos=-0.001
  a=1.00 step  4000: G=-1.6235, Lvar=-0.0116, VarR_seen=0.939, cos=-0.030
  a=1.00 step  5000: G=-1.6692, Lvar=-0.0111, VarR_seen=0.949, cos=-0.038
  a=1.00 step  6000: G=-1.7006, Lvar=-0.0104, VarR_seen=0.939
  a=1.00 step  7000: G=-1.7890, Lvar=-0.0106, VarR_seen=0.912
  a=1.00 step  8000: G=-1.8145, Lvar=-0.0115, VarR_seen=0.914
  a=1.00 step  9000: G=-1.9075, Lvar=-0.0112, VarR_seen=0.930
  a=1.00 step 10000: G=-2.0138, Lvar=-0.0113, VarR_seen=0.933
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:1247: FutureWarning: 'multi_class' was deprecated in version 1.5 and will be removed in 1.7. From then on, it will always use 'multinomial'. Leave it to its default value to avoid this warning.
  warnings.warn(


Run history:

eval/AccS	▁
eval/AccU	▁
eval/H_mean	▁
eval/VarR	▁
eval/rho_sp	▁
eval/routing_rate	▁
train/GP	█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
train/L_D	█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
train/L_G	█▂▃▃▃▃▃▃▂▂▂▂▂▂▂▂▂▂▁▁▁
train/L_var	█▆▄▃▃▃▄▁▂▂▃▅▅▄▄▆▃▃▃▃▃
+8	...

Run summary:

eval/AccS	0.03748
eval/AccU	0.06194
eval/H_mean	0.0467
eval/VarR	0.84335
eval/rho_sp	0.67798
eval/routing_rate	0.3245
train/GP	0.00164
train/L_D	-0.01927
train/L_G	-2.01378
train/L_var	-0.01125
+8	...

View run 04_film_lvar_alpha1.00 at: https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl/runs/73nx5dja
View project at: https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl
Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
Find logs at: ./wandb/run-20260507_204239-73nx5dja/logs
  alpha=1.00: H=0.0467, AccS=0.0375, AccU=0.0619, VarR_unseen=0.8433, rho_sp=0.6780

============================================================
Alpha = 2.00  (L_G = L_wass + 2.00 * L_var)
============================================================
Tracking run with wandb version 0.26.1
Run data is saved locally in /content/wandb/run-20260507_204956-2h4ed4mz
Syncing run 04_film_lvar_alpha2.00 to Weights & Biases (docs)
View project at https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl
View run at https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl/runs/2h4ed4mz
  a=2.00 step     1: G=-0.0579, Lvar=-0.0087
  a=2.00 step  1000: G=-1.4154, Lvar=-0.0107, VarR_seen=0.854, cos=0.022
  a=2.00 step  2000: G=-1.4195, Lvar=-0.0111, VarR_seen=0.943, cos=0.038
  a=2.00 step  3000: G=-1.5397, Lvar=-0.0106, VarR_seen=0.919, cos=0.002
  a=2.00 step  4000: G=-1.6346, Lvar=-0.0105, VarR_seen=0.942, cos=-0.067
  a=2.00 step  5000: G=-1.7238, Lvar=-0.0116, VarR_seen=0.949, cos=-0.087
  a=2.00 step  6000: G=-1.7907, Lvar=-0.0109, VarR_seen=0.939
  a=2.00 step  7000: G=-1.8690, Lvar=-0.0113, VarR_seen=0.924
  a=2.00 step  8000: G=-1.9059, Lvar=-0.0116, VarR_seen=0.927
  a=2.00 step  9000: G=-2.0053, Lvar=-0.0105, VarR_seen=0.932
  a=2.00 step 10000: G=-2.0447, Lvar=-0.0117, VarR_seen=0.944
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:1247: FutureWarning: 'multi_class' was deprecated in version 1.5 and will be removed in 1.7. From then on, it will always use 'multinomial'. Leave it to its default value to avoid this warning.
  warnings.warn(


Run history:

eval/AccS	▁
eval/AccU	▁
eval/H_mean	▁
eval/VarR	▁
eval/rho_sp	▁
eval/routing_rate	▁
train/GP	█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
train/L_D	█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
train/L_G	█▃▃▃▃▃▃▃▂▂▂▂▂▂▂▁▁▁▁▁▁
train/L_var	█▆▄▃▃▃▄▁▄▂▂▃▃▄▃▆▂▃▄▃▂
+8	...

Run summary:

eval/AccS	0.03869
eval/AccU	0.056
eval/H_mean	0.04577
eval/VarR	0.85221
eval/rho_sp	0.66638
eval/routing_rate	0.31269
train/GP	0.00169
train/L_D	-0.02734
train/L_G	-2.04472
train/L_var	-0.01173
+8	...

View run 04_film_lvar_alpha2.00 at: https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl/runs/2h4ed4mz
View project at: https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl
Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
Find logs at: ./wandb/run-20260507_204956-2h4ed4mz/logs
  alpha=2.00: H=0.0458, AccS=0.0387, AccU=0.0560, VarR_unseen=0.8522, rho_sp=0.6664

============================================================
Alpha = 5.00  (L_G = L_wass + 5.00 * L_var)
============================================================
Tracking run with wandb version 0.26.1
Run data is saved locally in /content/wandb/run-20260507_205709-a5yl6he1
Syncing run 04_film_lvar_alpha5.00 to Weights & Biases (docs)
View project at https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl
View run at https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl/runs/a5yl6he1
  a=5.00 step     1: G=-0.0841, Lvar=-0.0087
  a=5.00 step  1000: G=-1.4498, Lvar=-0.0105, VarR_seen=0.859, cos=0.049
  a=5.00 step  2000: G=-1.3999, Lvar=-0.0111, VarR_seen=0.945, cos=0.022
  a=5.00 step  3000: G=-1.4257, Lvar=-0.0115, VarR_seen=0.925, cos=0.010
  a=5.00 step  4000: G=-1.4643, Lvar=-0.0104, VarR_seen=0.941, cos=-0.090
  a=5.00 step  5000: G=-1.5192, Lvar=-0.0122, VarR_seen=0.953, cos=-0.089
  a=5.00 step  6000: G=-1.5416, Lvar=-0.0115, VarR_seen=0.955
  a=5.00 step  7000: G=-1.6456, Lvar=-0.0106, VarR_seen=0.930
  a=5.00 step  8000: G=-1.6822, Lvar=-0.0114, VarR_seen=0.942
  a=5.00 step  9000: G=-1.7751, Lvar=-0.0107, VarR_seen=0.949
  a=5.00 step 10000: G=-1.8549, Lvar=-0.0102, VarR_seen=0.940
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:1247: FutureWarning: 'multi_class' was deprecated in version 1.5 and will be removed in 1.7. From then on, it will always use 'multinomial'. Leave it to its default value to avoid this warning.
  warnings.warn(


Run history:

eval/AccS	▁
eval/AccU	▁
eval/H_mean	▁
eval/VarR	▁
eval/rho_sp	▁
eval/routing_rate	▁
train/GP	█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
train/L_D	█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
train/L_G	█▂▃▃▃▃▃▃▃▂▂▂▂▂▂▂▂▂▁▁▁
train/L_var	█▅▄▃▃▄▂▂▅▃▁▃▂▂▄▆▃▄▄▁▅
+8	...

Run summary:

eval/AccS	0.03779
eval/AccU	0.0535
eval/H_mean	0.04429
eval/VarR	0.86945
eval/rho_sp	0.71005
eval/routing_rate	0.28937
train/GP	0.00189
train/L_D	-0.01976
train/L_G	-1.85493
train/L_var	-0.01018
+8	...

View run 04_film_lvar_alpha5.00 at: https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl/runs/a5yl6he1
View project at: https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl
Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
Find logs at: ./wandb/run-20260507_205709-a5yl6he1/logs
  alpha=5.00: H=0.0443, AccS=0.0378, AccU=0.0535, VarR_unseen=0.8694, rho_sp=0.7100

============================================================
Alpha = 10.00  (L_G = L_wass + 10.00 * L_var)
============================================================
Tracking run with wandb version 0.26.1
Run data is saved locally in /content/wandb/run-20260507_210422-9niyqrmf
Syncing run 04_film_lvar_alpha10.00 to Weights & Biases (docs)
View project at https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl
View run at https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl/runs/9niyqrmf
  a=10.00 step     1: G=-0.1277, Lvar=-0.0087
  a=10.00 step  1000: G=-1.5909, Lvar=-0.0106, VarR_seen=0.875, cos=0.047
  a=10.00 step  2000: G=-1.5662, Lvar=-0.0109, VarR_seen=0.955, cos=0.010
  a=10.00 step  3000: G=-1.6197, Lvar=-0.0112, VarR_seen=0.935, cos=-0.006
  a=10.00 step  4000: G=-1.6300, Lvar=-0.0110, VarR_seen=0.957, cos=-0.047
  a=10.00 step  5000: G=-1.6798, Lvar=-0.0118, VarR_seen=0.950, cos=-0.184
  a=10.00 step  6000: G=-1.6935, Lvar=-0.0118, VarR_seen=0.966
  a=10.00 step  7000: G=-1.7183, Lvar=-0.0112, VarR_seen=0.951
  a=10.00 step  8000: G=-1.7481, Lvar=-0.0118, VarR_seen=0.962
  a=10.00 step  9000: G=-1.7768, Lvar=-0.0114, VarR_seen=0.952
  a=10.00 step 10000: G=-1.8085, Lvar=-0.0102, VarR_seen=0.949
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:1247: FutureWarning: 'multi_class' was deprecated in version 1.5 and will be removed in 1.7. From then on, it will always use 'multinomial'. Leave it to its default value to avoid this warning.
  warnings.warn(


Run history:

eval/AccS	▁
eval/AccU	▁
eval/H_mean	▁
eval/VarR	▁
eval/rho_sp	▁
eval/routing_rate	▁
train/GP	█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
train/L_D	█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
train/L_G	█▂▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁
train/L_var	█▃▄▄▃▂▃▁▃▂▂▂▁▃▃▅▂▂▂▂▅
+8	...

Run summary:

eval/AccS	0.03688
eval/AccU	0.04763
eval/H_mean	0.04157
eval/VarR	0.88877
eval/rho_sp	0.7433
eval/routing_rate	0.25538
train/GP	0.00218
train/L_D	-0.02325
train/L_G	-1.80851
train/L_var	-0.01016
+8	...

View run 04_film_lvar_alpha10.00 at: https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl/runs/9niyqrmf
View project at: https://wandb.ai/arizakml-durham-university/gzsl-eeg-bravl
Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
Find logs at: ./wandb/run-20260507_210422-9niyqrmf/logs
  alpha=10.00: H=0.0416, AccS=0.0369, AccU=0.0476, VarR_unseen=0.8888, rho_sp=0.7433

============================================================
ALPHA SWEEP COMPLETE
============================================================
  alpha=0.00: H=0.0448, VarR_u=0.8444, rho_sp=0.6533
  alpha=0.10: H=0.0452, VarR_u=0.8472, rho_sp=0.6657
  alpha=0.50: H=0.0461, VarR_u=0.8502, rho_sp=0.6674
  alpha=1.00: H=0.0467, VarR_u=0.8433, rho_sp=0.6780 <-- best
  alpha=2.00: H=0.0458, VarR_u=0.8522, rho_sp=0.6664
  alpha=5.00: H=0.0443, VarR_u=0.8694, rho_sp=0.7100
  alpha=10.00: H=0.0416, VarR_u=0.8888, rho_sp=0.7433

Best alpha = 1.00 (H=0.0467)
Best generator loaded into `generator` — downstream cells run on best-alpha model.

============================================================
VARIANCE RATIO ANALYSIS
============================================================
  VarR (seen  classes, training): 0.9307
  VarR (unseen classes, eval):    0.8469
  Transfer gap (seen - unseen):   0.0838

  Reference baselines:
    Exp 01 (concat WGAN-GP):  VarR_unseen = 0.872
    Exp 03 (L_var training):  VarR_seen=0.973, VarR_unseen=0.875, gap=0.098
    FiLM target:              VarR_unseen > 0.95, gap < 0.03
FiLMGenerator(
  (fc1): Linear(in_features=100, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=256, bias=True)
  (fc3): Linear(in_features=256, out_features=64, bias=True)
  (film1_mlp): Sequential(
    (0): Linear(in_features=64, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=512, bias=True)
  )
  (film2_mlp): Sequential(
    (0): Linear(in_features=64, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=512, bias=True)
  )
)

============================================================
STRUCTURAL OVERCOUPLING DIAGNOSTICS
============================================================
  rho_sp (centroid vs prototype):   0.6851  (p=0.00e+00)
  kNN@10 neighbourhood preservation: 0.5210

  Reference: Exp 01 baseline: rho_sp=0.857, kNN@10=0.611
             Exp 03 L_var:    rho_sp=0.880 (worse, overcoupling increased)
             Real seen data:  rho_sp~0.668
  FiLM target: rho_sp < 0.800

============================================================
GZSL HARMONIC MEAN
============================================================
H = 2 * 0.0390 * 0.0589 / (0.0390 + 0.0589)
H = 0.0469