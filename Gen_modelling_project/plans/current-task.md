# Current Task

**Status:** AWAITING USER DIRECTION

## Context

The GZSL EEG decoding pipeline is fully implemented (117 notebook cells):
- Baseline [A]: Logistic Regression on raw EEG
- CLIP encoder + cWGAN-GP for embedding synthesis
- GZSL classifier [A+B] on real seen + synthetic unseen embeddings
- Ablation study (Methods A–D) with corrected label handling
- Bias table diagnostics

The label collision bug (unseen labels 1-200 overlapping seen labels 1-200) has been identified and fixed with an offset.

## Waiting For

User to specify the next task. Possible directions:
1. Improve model performance (hyperparameter tuning, architecture changes)
2. Add new analysis or visualisations
3. Clean up / refactor notebook cells
4. Report writing support
5. New research direction or experiment
