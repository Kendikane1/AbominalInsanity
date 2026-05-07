# Experiment 04: FiLM / Projection-Based Conditioning

**Status**: Active
**Notebook**: `notebook.ipynb` (copy of `main_pipeline.ipynb` + Generator modification)

## Hypothesis

FiLM (Feature-wise Linear Modulation) or projection-based conditioning replaces concatenation conditioning in the cWGAN-GP Generator. By computing per-layer scale (γ) and shift (β) from the prototype s_c via small MLPs, the generator learns a more compositional mapping where prototype identity and variance behaviour are decoupled. This should improve seen→unseen transfer of variance statistics.

## Architecture Change

**Current** (concatenation):
```
G(z, s_c): [z; s_c] → 256 → 256 → d
```

**Proposed** (FiLM):
```
G(z, s_c): z → 256 --FiLM(s_c)--> 256 --FiLM(s_c)--> d
FiLM: γ, β = MLP(s_c); h_out = γ ⊙ h_in + β
```

## Key Metrics to Track

- H-mean (primary)
- VarR on **unseen** classes (gap vs training VarR)
- ρ_sp (inter-class structure)
- Gradient cosine similarity between L_wasserstein and any auxiliary losses
