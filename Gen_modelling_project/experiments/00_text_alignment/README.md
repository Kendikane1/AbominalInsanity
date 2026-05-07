# Experiment 00: Text Alignment Baseline (Brain-TEXT CLIP)

**Status**: Complete — Superseded by Image Alignment
**Era**: Original coursework (COMP2261) — pre-paradigm-shift
**Modality**: EEG → Text (CLIP 512-D sentence embeddings)

## Summary

The original GZSL pipeline using Brain-TEXT CLIP alignment. A contrastive encoder
maps 561-D EEG features to a shared space with 512-D text embeddings (CLIP). A
conditional WGAN-GP then synthesises unseen-class embeddings from text prototypes.

This experiment documents the full exploration arc of the text-alignment paradigm,
including ablation studies, inverse bias diagnosis, Phase E encoder optimisation,
and Phase F architecture sweep.

## Architecture (Text Alignment Era)

- **BrainEncoder**: 561 → 1024 → 512 → d, LayerNorm + L2-norm
- **TextProjector**: 512 (CLIP) → 512 → d, LayerNorm + L2-norm
- **cWGAN-GP**: same topology as image-alignment pipeline

## Phases in this notebook

| Phase | Section | Key Result |
|-------|---------|-----------|
| Baseline [A] | Cells 17–22 | LogReg on raw EEG: ~1.5% seen accuracy |
| CLIP encoder | Cells 28–37 | Initial config: embed_dim=64, τ=0.07, 20 epochs |
| WGAN-GP | Cells 38–48 | cWGAN-GP on text prototypes |
| Ablation [A-D] | Cells 75–101 | Label collision fix; routing catastrophe isolated |
| Phase 1 | Cells 102–107 | Sample balancing: downsampled unseen → routing fixed |
| Phase D | Cells 108–113 | Upstream diagnostics: encoder is bottleneck (~1.9% top-1) |
| Phase E | Cells 114–125 | 2-stage sweep: τ=0.15, 50ep, lr=1e-3 → **H=0.70%** (+67%) |
| Phase F | Cells 126–136 | Augmentation + SupCon + EEGNet/ShallowConvNet → **dead end** |

## Key Findings

- **Best H-mean**: 0.70% (Phase E optimised config) — confirmed EEG→text is the ceiling
- **Phase E**: embed_dim=128, LayerNorm, lr=1e-3, cosine warmup, τ=0.15, 50 epochs
- **Phase F conclusion**: ~3% top-1 is the hard ceiling for EEG→text with any MLP variant
  on 17ch×33t data. All augmentation/loss/architecture experiments failed to improve.
- **Root cause of ceiling**: insufficient class-discriminative information in EEG features
  at this temporal resolution; text embeddings also have high intra-class cosine similarity
  (avg ρ=0.668), making separation hard.

## Why superseded

Switched to EEG→image (CORnet-S 1000-D PCA) which produced:
- H=4.53% vs 0.70% (6.5× improvement)
- AccU: 5.73% vs 0.32% (18× improvement)
- Contrastive loss converged for the first time (0.93 vs ~4.2 for text)
- Analysis: `context/research_history/CORnet_S_paradigm_analysis.md`

## Reference

- Archive notebooks: `archive/notebooks/COMP2261_ArizMLCW_with_baseline.ipynb` (v2/Phase E)
  and `archive/notebooks/COMP2261_ArizMLCW_with_baseline (1).ipynb` (v1/Phase F standalone)
