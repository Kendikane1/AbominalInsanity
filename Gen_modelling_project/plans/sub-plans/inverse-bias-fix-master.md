# Inverse Bias Fix — 5-Phase Master Plan

**Status**: Active — implementing phase-by-phase
**Owner**: Orchestrator (Claude Code)
**Implementer**: Antigravity Agent

## Overview

Fix the inverse bias in Method D (99.6% seen→unseen misrouting) via 5 additive phases targeting internal representation equivalence.

### Root Causes

| ID | Mechanism | Description |
|----|-----------|-------------|
| M1 | Variance mismatch | tr(Σ̃_u) << tr(Σ_s) → inflated unseen weights |
| M2 | Sample imbalance | ~20 synth/class vs ~8 real/class |
| M3 | Max-over-classes | 200 unseen classes compound per-class leakage |

### Phase Summary

| Phase | Name | Targets | Effort | File |
|-------|------|---------|--------|------|
| 0 | Evaluation Harness | Infrastructure | Low | `add_eval_harness.py` |
| 1 | Sample Balance | M2 | Low | `add_phase1_sample_balance.py` |
| 2 | Covariance-Aware Generator | M1 (at source) | High | `add_phase2_cov_generator.py` |
| 3 | Cosine Classifier | M1, M3 | Moderate | `add_phase3_cosine_clf.py` |
| 4 | Noise Injection | M1, M3 | Low | `add_phase4_noise_injection.py` |
| 5 | Post-Hoc Calibration | Residual | Low | `add_phase5_posthoc_calib.py` |

### Dependency Graph

```
Phase 0 (eval harness) ──► Phase 1 (sample balance) ──► Phase 2 (cov generator) ──► Phase 3 (cosine clf) ──► Phase 5 (post-hoc)
                                                    └──► Phase 4 (noise injection, parallel with Phase 2)
```

### Success Criteria

- Routing rate drops from 99.6% toward ~50%
- H-mean improves substantially over Method D
- Weight norm distributions for seen vs unseen converge (Phase 3 guarantees)
- Post-hoc γ* near zero (upstream fixes sufficient)

### Detailed Phase Specs

Full implementation details for each phase are provided in `plans/current-task.md` as each phase becomes active. The complete plan with all code sketches is preserved in the orchestrator's conversation history.

---

## Progress Log

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| 0 | ACTIVE | 2026-03-01 | First task issued |
| 1 | ACTIVE | 2026-03-01 | Bundled with Phase 0 |
| 2 | PENDING | — | — |
| 3 | PENDING | — | — |
| 4 | PENDING | — | — |
| 5 | PENDING | — | — |
