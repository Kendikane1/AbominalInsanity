# Experiment 02: Eta Sweep (Post-Hoc Prototype Perturbation)

**Status**: Complete — Failed
**Reference**: `archive/notebooks/GZSL_EEG_Pipeline_v2.ipynb` cells 71–76

## Hypothesis

Perturbing prototypes at synthesis time with calibrated Gaussian noise (η·ξ, re-normalised to unit sphere) would increase within-class variance of synthetic embeddings and improve H-mean.

## Results

| η | H-mean | VarR | ρ_sp |
|---|--------|------|------|
| 0.00 | **4.77%** | 0.872 | 0.857 |
| 0.05 | 4.41% | 0.927 | 0.857 |
| 0.10 | 2.56% | 1.002 | 0.854 |
| 0.20 | 0.62% | — | — |
| 0.25 | 0.29% | — | — |

H-mean strictly decreases. At η=0.10, VarR=1.002 (perfect) but H=2.56% — correct variance amount in wrong directions.

## Root Cause

Output perturbation δê ≈ J_G·(η·ξ) lies in the column space of J_G, which does not align with real brain variability directions. Isotropic noise in prototype space maps through a fixed Jacobian into arbitrary (mostly unhelpful) output directions.

**Conclusion**: Post-hoc perturbation is provably ineffective. Fix must be at training time (reshaping J_G itself via weight updates).
