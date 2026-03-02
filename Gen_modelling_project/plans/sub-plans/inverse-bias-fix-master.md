# Inverse Bias Fix — Master Plan

**Status**: Active — CLIP Encoder Optimisation phase
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
| D | Upstream Diagnostics | Bottleneck ID | Low | **COMPLETE** |
| E | **CLIP Encoder Optimisation** | Encoder quality | High | **ACTIVE** |
| 2 | Covariance-Aware Generator | M1 (at source) | High | PENDING — reassess after Phase E |
| 3 | Cosine Classifier | M1, M3 | Moderate | PENDING — reassess after Phase E |
| 4 | Noise Injection | M1, M3 | Low | PENDING — reassess after Phase E |
| 5 | Post-Hoc Calibration | Residual | Low | **DEPRIORITISED** — calibration solved by Phase 1 |

### Trajectory Pivot (2026-03-01)

Phase 1 results were definitive: sample count imbalance (M2) was the dominant mechanism. Balancing counts collapsed routing rate from 99.7% → ~9%, but exposed near-floor discriminative power (H ≈ 0.4%). The original 2% AccU was an artifact of predicting unseen for everything.

**New priority**: Identify the representation bottleneck before investing in Phases 2–5.

Three candidate failure points:
- **(A) CLIP encoder**: 64-dim embeddings may be too compressed for 1654 classes
- **(B) WGAN-GP generator**: May be mode-collapsed, producing indiscriminable synthetics
- **(C) Text prototypes**: 200 unseen prototypes may cluster too tightly in R^64

### Diagnostic Verdict (2026-03-02)

| Component | Status | Evidence |
|-----------|--------|----------|
| CLIP Encoder | **WEAK** (gating bottleneck) | 1.90% seen-only top-1 (31.5x above random but near-floor) |
| WGAN-GP Generator | OK (victim, not cause) | 68.38% internal CV, 1.73% transfer — fails because encoder is diffuse |
| Text Prototypes | OK (well-separated) | Mean cosine ≈ 0.014, adequate angular spread |

**Decision**: Halt downstream fixes. Invest in CLIP Encoder Optimisation (Phase E) — two-stage parametric sweep over training dynamics and architecture.

---

## Progress Log

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| 0 | **COMPLETE** | 2026-03-01 | Eval harness deployed (cells 97–101) |
| 1 | **COMPLETE** | 2026-03-01 | M2 confirmed as dominant. Routing rate 99.7% → ~9%. H ≈ 0.4% |
| Cleanup | **COMPLETE** | 2026-03-02 | Notebook 129 → 108 cells, obsolete sections removed |
| D | **COMPLETE** | 2026-03-02 | Encoder = bottleneck (1.90%), generator OK (68% internal), prototypes OK |
| Diag cells | **COMPLETE** | 2026-03-02 | 6 diagnostic cells added (108→114 cells) |
| E | **ACTIVE** | 2026-03-02 | CLIP encoder optimisation: 2-stage sweep (14 runs) + full pipeline re-run. 12 cells to add (114→126) |
| 2 | PENDING | — | Reassess after Phase E |
| 3 | PENDING | — | Reassess after Phase E |
| 4 | PENDING | — | Reassess after Phase E |
| 5 | DEPRIORITISED | 2026-03-01 | Calibration already solved |
