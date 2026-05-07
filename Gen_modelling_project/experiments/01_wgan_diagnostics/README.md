# Experiment 01: WGAN-GP Synthesis Diagnostics

**Status**: Complete
**Reference**: `archive/notebooks/GZSL_EEG_Pipeline_v2.ipynb` cells 67–70

## Findings

Three targeted diagnostics on the baseline cWGAN-GP (optimal config, η=0):

| Metric | Value | Interpretation |
|--------|-------|---------------|
| VarR | 0.872 | Mild within-class variance deficit (synth vs real) |
| ρ_sp | 0.857 | **Primary pathology**: structural overcoupling (real unseen: 0.668) |
| kNN@10 | 0.611 | Moderate neighbourhood preservation |

**Conclusion**: The generator over-couples synthetic embeddings to their conditioning prototypes, producing centroids tightly clustered around prototype positions. Within-class variance is mildly insufficient. Both pathologies need addressing for better seen→unseen transfer.
