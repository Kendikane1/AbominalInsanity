# Experiments

| # | Name | Hypothesis | Status | H-mean |
|---|------|-----------|--------|--------|
| 00 | Text alignment baseline | Brain-TEXT CLIP can classify unseen EEG categories | Superseded | 0.70% |
| 01 | WGAN diagnostics | Characterise synthesis quality | Complete | — |
| 02 | Eta sweep | Post-hoc prototype perturbation fixes VarR | Failed | 4.77→2.56 |
| 03 | Variance regularisation | L_var at training time improves VarR | Failed | 4.77→4.58 |
| 04 | FiLM conditioning | Projection conditioning improves seen→unseen transfer | Active | — |

Each experiment subfolder contains:
- `README.md` — hypothesis, changes, findings, conclusion
- `notebook.ipynb` — standalone (exp 00) or main_pipeline.ipynb + experiment cells appended (exps 01–04)
- `context/` — spec documents, implementation directives, debriefs
- `results/` — `.npy`, `.json`, `.md` result files
- `figures/` — experiment figures at 150 dpi

## Notebook lineage

**Experiment 00** is a standalone notebook (Brain-TEXT CLIP, different architecture from main_pipeline.ipynb).

**Experiments 01–04** all extend `main_pipeline.ipynb` (69 cells). Experiments 02 and 03 include an alias
cell immediately after the section header that maps `main_pipeline` variable names to the `OPT_` names
used in the archived experiment cells (e.g. `generator_opt = generator`, `E_train_opt = E_train_seen`).

## Research arc

```
Exp 00: Text alignment (COMP2261 era) — H=0.70%
  → Image alignment paradigm shift (×6.5 H-mean improvement)
    → Exp 01: Diagnosed WGAN pathology (ρ_sp=0.857 overcoupling, VarR=0.872)
      → Exp 02: Post-hoc noise perturbation — FAILED (correct variance, wrong directions)
        → Exp 03: L_var training-time loss — FAILED (doesn't generalise seen→unseen)
          → Exp 04: FiLM conditioning (architecture fix) — ACTIVE
```
