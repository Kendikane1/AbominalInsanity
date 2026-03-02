# Inverse Bias Fix — Master Plan

**Status**: Active — trajectory pivoted to upstream diagnostics
**Owner**: Orchestrator (Claude Code)
**Implementer**: Antigravity Agent

## Overview

Fix the inverse bias in Method D (99.6% seen→unseen misrouting) via systematic investigation. Phase 1 solved calibration; the bottleneck is now upstream representation quality.

### Root Causes

| ID | Mechanism | Description | Status |
|----|-----------|-------------|--------|
| M1 | Variance mismatch | tr(Σ̃_u) << tr(Σ_s) → inflated unseen weights | Secondary — subordinate to M2 |
| M2 | Sample imbalance | ~20 synth/class vs ~8 real/class | **SOLVED** (Phase 1) |
| M3 | Max-over-classes | 200 unseen classes compound per-class leakage | Mitigated by M2 fix |

### Phase Summary

| Phase | Name | Targets | Effort | Status |
|-------|------|---------|--------|--------|
| 0 | Evaluation Harness | Infrastructure | Low | **COMPLETE** |
| 1 | Sample Balance | M2 | Low | **COMPLETE** |
| D | **Upstream Diagnostics** | Bottleneck ID | Low | **ACTIVE** |
| 2 | Covariance-Aware Generator | M1 (at source) | High | PENDING — value depends on Diag results |
| 3 | Cosine Classifier | M1, M3 | Moderate | PENDING — value depends on Diag results |
| 4 | Noise Injection | M1, M3 | Low | PENDING — value depends on Diag results |
| 5 | Post-Hoc Calibration | Residual | Low | **DEPRIORITISED** — calibration solved by Phase 1 |

### Trajectory Pivot (2026-03-01)

Phase 1 results were definitive: sample count imbalance (M2) was the dominant mechanism. Balancing counts collapsed routing rate from 99.7% → ~9%, but exposed near-floor discriminative power (H ≈ 0.4%). The original 2% AccU was an artifact of predicting unseen for everything.

**New priority**: Identify the representation bottleneck before investing in Phases 2–5.

Three candidate failure points:
- **(A) CLIP encoder**: 64-dim embeddings may be too compressed for 1654 classes
- **(B) WGAN-GP generator**: May be mode-collapsed, producing indiscriminable synthetics
- **(C) Text prototypes**: 200 unseen prototypes may cluster too tightly in R^64

### Decision Framework (post-diagnostics)

| Diagnostic Result | Action |
|-------------------|--------|
| Encoder CRITICAL | Increase embed_dim, train longer, curriculum learning |
| Generator CRITICAL | Invest in Phase 2 (cov-aware WGAN) or alternative generators |
| Prototypes CRITICAL | Better text features, larger semantic space, alternative alignment |
| Multiple CRITICAL | Address upstream first: prototypes → encoder → generator |

---

## Progress Log

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| 0 | **COMPLETE** | 2026-03-01 | Eval harness deployed (cells 97–101) |
| 1 | **COMPLETE** | 2026-03-01 | M2 confirmed as dominant. Routing rate 99.7% → ~9%. H ≈ 0.4% |
| Cleanup | **COMPLETE** | 2026-03-02 | Notebook 129 → 108 cells, obsolete sections removed |
| D | **ACTIVE** | 2026-03-02 | Upstream diagnostics task issued. 3 measurements: seen-only acc, unseen-only acc, prototype spread |
| 2 | PENDING | — | Value TBD by diagnostics |
| 3 | PENDING | — | Value TBD by diagnostics |
| 4 | PENDING | — | Value TBD by diagnostics |
| 5 | DEPRIORITISED | 2026-03-01 | Calibration already solved |
