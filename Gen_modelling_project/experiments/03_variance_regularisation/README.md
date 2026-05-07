# Experiment 03: Variance Regularisation (L_var)

**Status**: Complete — Failed
**Reference**: `archive/notebooks/GZSL_EEG_Pipeline_v2.ipynb` cells 77–83
**Spec**: `context/implementation_directive.md`
**Full analysis**: `results/experiment_log.md`

## Hypothesis

Adding L_var = mean_c[Σ_d (var_synth_{c,d} − var_target_d)²] to the generator loss reshapes J_G to produce higher within-class variance on unseen classes, improving H-mean.

## Results

| α | H-mean | AccS | AccU | VarR | ρ_sp |
|---|--------|------|------|------|------|
| baseline | 4.77% | 4.11% | 5.69% | 0.872 | 0.857 |
| best (10α₀) | **4.58%** | 3.96% | 5.49% | 0.875 | 0.880 |

- Training VarR (seen): reached 0.973 by end of training
- Evaluation VarR (unseen): only 0.875 — **0.098 gap**
- ρ_sp increased (worse overcoupling) across all α values
- Gradient cosine similarity consistently negative (−0.04 to −0.09)

## Root Cause

Concatenation conditioning gives the generator sufficient capacity to learn prototype-specific variance for seen classes. This behaviour does not transfer to unseen prototypes — the zero-shot transfer gap applies to the generator itself. L_var teaches seen-prototype-specific tricks that are useless at evaluation time.

**Conclusion**: Concatenation conditioning is the architectural bottleneck. Next intervention: FiLM/projection-based conditioning to decouple prototype identity from variance behaviour.
