# Inverse Bias Fix — Master Plan

**Status**: Active — Phase F (Augmentation + SupCon + Architecture)
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
| E | CLIP Encoder Optimisation | Encoder quality | High | **COMPLETE** |
| F | **Augmentation + SupCon + Architecture** | Generalization + signal + inductive bias | High | **ACTIVE** |
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
| E | **COMPLETE** | 2026-03-02 | Encoder top-1: 1.90%→2.87% (+51%), H-mean: 0.42%→0.70% (+67%). Best: dim=128, LN, lr=1e-3, cosine, τ=0.15. Config integrated, sweep cells removed (126→114) |
| F | **ACTIVE** | 2026-03-02 | Task spec complete. Three-stage sweep: F1 augmentation (9 runs), F2 SupCon loss (3 runs), F3 EEGNet/ShallowConvNet (4 runs). Targets: 5-10% encoder top-1 → H-mean 2-5%. Cells 114-124 to be injected. |
| 2 | PENDING | — | Reassess after Phase F |
| 3 | PENDING | — | Reassess after Phase F |
| 4 | PENDING | — | Reassess after Phase F |
| 5 | DEPRIORITISED | 2026-03-01 | Calibration already solved |
