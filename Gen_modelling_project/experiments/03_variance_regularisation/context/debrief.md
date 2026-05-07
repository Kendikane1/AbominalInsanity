# Debrief — Variance Regularisation Analysis & Pre-Implementation Review

*Date: 2026-04-11*
*Session scope: Analysed eta sweep results, set up knowledge-teacher skill, critically reviewed the variance regularisation implementation directive*

---

## 1. Task Summary

This session had three objectives:
1. **Analyse the noise-augmented prototype conditioning (eta sweep) results** from Colab
2. **Create a "knowledge teacher" system** for the user to study the project's mathematical foundations in Claude Desktop, then generalise it as a reusable Claude Code skill
3. **Critically review the variance regularisation implementation directive** (`context/variance_regularisation_implementation_directive.md`) written by the math tutor, before proceeding to implementation

---

## 2. What Changed — Files Created/Modified

### New files created this session

| File | Purpose |
|------|---------|
| `context/math_teacher_system_prompt.md` | System prompt for Claude.ai Project — paste as Custom Instructions |
| `context/project_math_context.md` | Knowledge file for the Claude.ai Project — full pipeline math reference |
| `context/math_curriculum.md` | 20-lesson learning roadmap (4 levels: prerequisites → research frontier) |
| `~/.claude/skills/knowledge-teacher/SKILL.md` | Global Claude Code skill — generates teacher artifacts for any future project (`/knowledge-teacher [domain]`) |
| `context/variance_regularisation_implementation_directive.md` | Written by the math tutor (uploaded by user) — full spec for the L_var modification |
| `context/variance_regularisation_debrief.md` | This file |

### Files NOT modified this session
- `GZSL_EEG_Pipeline_v2.ipynb` — no new cells added (implementation not started yet)
- No helper scripts written yet

---

## 3. Eta Sweep Results — Complete Analysis

### Results Table (from cell 74, run on Colab)

```
   eta        H     AccS     AccU   Route   rho_sp   rho_sr   kNN10    VarR
  0.00    4.77%    4.11%    5.69%   20.0%   0.8572   0.5875  0.6110  0.872 ← BEST
  0.02    4.56%    4.14%    5.06%   19.5%   0.8558   0.5930  0.5915  0.880
  0.05    4.08%    4.35%    3.84%   15.0%   0.8440   0.5864  0.5740  0.917
  0.07    3.50%    4.38%    2.92%   11.9%   0.8303   0.5783  0.5600  0.950
  0.10    2.56%    4.50%    1.79%    7.7%   0.8012   0.5608  0.5235  1.002
  0.15    1.16%    4.56%    0.66%    3.4%   0.7339   0.5185  0.4510  1.072
  0.20    0.49%    4.66%    0.26%    1.7%   0.6531   0.4666  0.3755  1.118
  0.25    0.29%    4.69%    0.15%    1.2%   0.5702   0.4121  0.3100  1.147
```

### Key findings

1. **Monotonic degradation**: H-mean drops from 4.77% → 0.29%. No sweet spot exists.
2. **AccU collapses** (5.69% → 0.15%) while **AccS improves** (4.11% → 4.69%) — the routing catastrophe returns.
3. **ρ(synth, real_unseen) DECREASES** with η — perturbation pushes synthetic data AWAY from real unseen, not toward it.
4. **At η=0.10, VarR=1.002** (perfect variance match) but **H=2.56%** — right amount of variance in the wrong directions is worse than too little in the right directions.
5. **Conclusion**: Synthesis-only (post-hoc) perturbation cannot help. The generator's Jacobian columns don't align with real brain variability directions. Changes must be made at TRAINING TIME by modifying the generator's loss.

### Why it failed (mathematical explanation)

Output perturbation `δê ≈ J_G · (η · ξ)` lies in the column space of J_G. The columns encode the generator's learned sensitivity directions, not real brain variability directions. Isotropic noise in prototype space maps to geometrically arbitrary perturbations in output space. The WGAN-GP's tight prototype coupling is a feature (maximally informative synthetic data) — any degradation of this structure strictly hurts.

---

## 4. Critical Assessment of the Variance Regularisation Directive

The directive (`context/variance_regularisation_implementation_directive.md`) was written by the math tutor after completing the full curriculum with the user. It specifies adding L_var to the generator loss.

### What is mathematically correct and implementation-ready

- **Gradient formula verified**: `∂L_var/∂ê_{c,k} = (4/K) · Δ_c ⊙ (ê_{c,k} − μ̂_c)` — verified by hand, cross-terms cancel via the zero-sum property of deviations from the mean
- **Diagonal-only covariance matching** — statistically necessary with ~8 samples/class in 64-D (full covariance would be rank-deficient)
- **var_target precomputation**: frozen encoder, training split only, averaged across 1654 seen classes → 64-D target vector
- **Pseudocode in Section 3.2**: correct, autograd-compatible, no graph-breaking operations
- **Alpha calibration via gradient-norm matching**: good starting heuristic, with 7-value sweep spanning 2 orders of magnitude as safety net
- **Monitoring quantities (Section 6.1)**: comprehensive — L_wasserstein, L_var, VarR, L_D, per-term gradient norms
- **Correctness checklist (Section 7)**: covers all real failure modes
- **Composability (Section 8.4)**: per-term gradient norm logging enables future loss terms without infrastructure rebuild

### Concerns raised during review

**Concern 1 — Primary vs secondary pathology mismatch**
The dominant diagnostic finding was structural overcoupling (ρ_sp=0.857 vs 0.668), not variance deficit (VarR=0.872, described as "mild"). L_var directly addresses the variance deficit but only INDIRECTLY addresses ρ_sp. If the generator satisfies L_var by uniformly inflating clusters while keeping centroids locked to prototypes, ρ_sp won't budge and H-mean won't improve.

*Mitigation*: L_var may reduce ρ_sp indirectly by making the generator more z-responsive (noise has more influence → conditioning has less relative influence). Monitor ρ_sp during the alpha sweep. If it doesn't decrease, L_struct (inter-class structure loss) is the next intervention.

**Concern 2 — L2 normalisation creates a total variance budget**
On S^{63}: `Σ_d Var(ê_d) = 1 - ||μ||² ≤ 1`. Pushing variance up in some dimensions necessarily pulls it down in others. The diagonal-only loss treats dimensions independently, but they're coupled by the sphere constraint.

*Mitigation*: var_target was computed from L2-normalised vectors, so it already respects the budget. The normalisation Jacobian `J_{L2} = (I - êêᵀ)/||u||₂` redistributes gradients in the backward pass. Monitor for dimension-specific oscillation — if some dimensions overshoot while others undershoot, the budget constraint is binding.

**Concern 3 — Absolute vs relative variance loss**
The directive uses absolute L2: `Σ_d (var[d] - target[d])²`. If variance scales differ across dimensions, high-variance dimensions dominate the loss. A relative alternative: `Σ_d (var[d]/target[d] - 1)²` treats all dimensions equally.

*Mitigation*: On S^{63} with d=64, per-dimension variances tend to be uniform. Start with absolute (simpler). If the generator matches high-variance dimensions while ignoring low-variance ones, switch to relative.

**Concern 4 — Structured batching for the generator step**
The current WGAN-GP probably samples random (embedding, prototype) pairs → ~0.15 samples per class per batch with 1654 classes. L_var needs K≥8 per class for stable estimates. The directive recommends structured batching.

*Implementation detail*: Structure the GENERATOR step only (K=20 samples × C=12-16 classes = 240-320 per batch). Keep the CRITIC step with random batching (it needs diverse real examples). This asymmetry is fine — they're separate forward passes.

**Concern 5 — Training steps may need extending**
With a dual objective, convergence may take longer. The directive keeps steps=10,000 without change.

*Mitigation*: Log L_var at step 10,000. If still decreasing meaningfully, extend to 15,000-20,000.

**Concern 6 — Alpha gradient alignment**
Equal gradient norms don't guarantee equal influence if gradients point in opposite directions. If `cos(∂L_wasserstein/∂θ, ∂L_var/∂θ) < 0`, the objectives oppose each other.

*Mitigation*: Log the cosine similarity between the two gradient vectors during training. If consistently negative, α needs to be small.

**Concern 7 — Noise responsiveness**
L_var forces higher variance, but doesn't guarantee the generator achieves it through the noise input z. The generator could satisfy L_var via the L2 normalisation denominator without diversifying z-dependent output.

*Mitigation*: Log `||∂G/∂z||_F` (Frobenius norm of the noise Jacobian) during training. If this doesn't increase, the generator isn't becoming more z-responsive.

---

## 5. Current State of the Project

### Pipeline state
- **Notebook**: `GZSL_EEG_Pipeline_v2.ipynb` — 77 cells
  - Cells 0-66: Full pipeline (encoder → WGAN-GP → classifier) using OLD config (dim=128)
  - Cells 67-70: WGAN-GP synthesis diagnostics (dim=128 baseline)
  - Cells 71-76: Eta sweep (optimal config: dim=64, tau=0.05, epochs=75) — COMPLETE, η=0 is best
- **Best result**: H=4.77%, AccS=4.11%, AccU=5.69% (eta sweep baseline, optimal encoder config)
- **Optimal encoder config**: embed_dim=64, tau=0.05, epochs=75, lr=2e-3, wd=1e-4, ImageProjector hidden=512
- **Optimal WGAN-GP config**: z_dim=100, embed_dim=64, lr=1e-4, betas=(0.0, 0.9), lambda_gp=10, n_critic=5, steps=10000, n_synth=20

### Research state
- Encoder optimisation: EXHAUSTED (30% encoder gain → 0% H-mean gain)
- Post-hoc perturbation: EXHAUSTED (monotonic degradation, no sweet spot)
- **Next**: Training-time variance regularisation (L_var added to L_G)
- **After that (if needed)**: L_struct (inter-class structure preservation), L_div (diversity loss)

### Knowledge infrastructure
- Math teacher set up as Claude.ai Project (3 files in `context/`)
- User completed full curriculum (L0-L3)
- Implementation directive written by math tutor: `context/variance_regularisation_implementation_directive.md`
- Knowledge-teacher skill created globally: `~/.claude/skills/knowledge-teacher/SKILL.md`

---

## 6. Open Items

- [ ] **Variance regularisation implementation** — NOT STARTED. The directive is reviewed and approved. Implementation plan needs to be created.
- [ ] **Structured batching design** — the current WGAN-GP training loop needs to be modified for the generator step to use structured batching (K samples per class). Exact mechanism to be determined during planning.
- [ ] **Alpha calibration procedure** — needs to be implemented as a pre-sweep step
- [ ] **Extended monitoring** — gradient cosine similarity, noise Jacobian norm — these go beyond the directive's recommendations and need to be scoped during planning

---

## 7. Next Steps — Implementation Plan Outline

When the next session begins, the implementation should proceed as:

### Step 1: Read the directive thoroughly
Read `context/variance_regularisation_implementation_directive.md` in full. This is the mathematical specification.

### Step 2: Read current notebook cells
Read cells 30 (encoder/projector), 40-42 (WGAN-GP models and training), 72-73 (optimal config and retrain from eta sweep) to understand the current code structure. These are the cells that the new implementation must interface with.

### Step 3: Plan the helper script
Design a helper script (`helper_files/add_variance_regularisation.py`) that injects cells into the notebook. The cells should:

1. **Config cell**: Define var_reg hyperparameters (alpha, K, C, etc.) alongside existing optimal config
2. **var_target precomputation cell**: Compute the 64-D target variance vector from training seen embeddings
3. **Modified WGAN-GP training cell**: The core modification — adds L_var to the generator step with structured batching
4. **Alpha calibration cell**: Run the gradient-norm matching procedure to determine α₀
5. **Alpha sweep cell**: Sweep α ∈ {0.01α₀, 0.1α₀, 0.5α₀, α₀, 2α₀, 5α₀, 10α₀} with full pipeline evaluation
6. **Diagnostics cell**: Training curves (L_wasserstein, L_var, VarR over steps), per-term gradient norms
7. **Results cell**: Final comparison table with baseline

### Step 4: Critical implementation details

**Structured batching for generator step**:
- Sample C classes uniformly from seen classes
- For each class: generate K=20 samples with the same prototype
- Generator batch = K×C = 240-320
- Critic step keeps random batching (unchanged)

**var_target computation**:
- Use frozen encoder from cell 73 (optimal config retrain)
- Process all 13,232 training seen samples through encoder
- Group by class, compute per-dimension variance per class, average across 1654 classes → (64,)

**L_var integration**:
- Compute AFTER generating fake samples, BEFORE backward()
- Add to L_wasserstein: `L_G = L_wasserstein + alpha * L_var`
- Must NOT detach fake embeddings before computing L_var

**class_labels tracking**:
- Current training loop may not track which class each sample belongs to
- Structured batching provides this naturally: labels = repeat_interleave(sampled_classes, K)

**Monitoring (per 100-500 steps)**:
- L_wasserstein, L_var, L_G (total)
- VarR (sample a batch, compute variance ratio against var_target)
- ||∂L_wasserstein/∂θ_G||, ||∂L_var/∂θ_G||
- Critic loss L_D
- Optional: cosine similarity between Wasserstein and var gradients

### Step 5: Run on Colab
- Upload updated notebook
- Run alpha calibration → determine α₀
- Run alpha sweep (7 configs × ~15 min each ≈ 1.5-2 hours total)
- Relay results back for analysis

### Estimated compute
- Alpha calibration: ~1 min (single forward/backward pass)
- Alpha sweep: 7 configs × 10,000 steps × (encoder train 75 epochs + WGAN-GP train + synthesis + classifier) ≈ 7 × 15 min ≈ 1.5-2 hours on GPU
- Can be reduced if encoder and WGAN-GP are trained once and only alpha is varied (similar to eta sweep strategy) — but L_var changes TRAINING, so each alpha needs a full WGAN-GP retrain

---

## 8. Key Context for Session Handoff

### Files the next session MUST read before implementing:
1. `context/variance_regularisation_implementation_directive.md` — the mathematical specification (418 lines)
2. `CLAUDE.md` — project state, cell map, hard constraints
3. The current WGAN-GP training code in the notebook (cells 40-42 for old config, cell 73 for optimal config retrain)

### Variables that exist after cell 73 runs (from eta sweep):
- `brain_encoder_opt`, `image_projector_opt` — frozen encoder models (optimal config)
- `generator_opt` — trained generator (optimal config, unmodified loss)
- `E_train_opt`, `E_test_opt`, `E_unseen_opt` — encoder embeddings
- `S_seen_opt`, `S_unseen_opt` — prototype dictionaries
- `y_train_seen`, `y_test_seen`, `y_unseen` — label arrays
- `X_train_tensor`, `I_train_tensor` — data tensors (independent of embed_dim)
- `seen_classes`, `unseen_classes` — class label arrays
- `seen_labels_set`, `unseen_labels_set` — label sets

### Baseline to beat
H=4.77%, AccS=4.11%, AccU=5.69%, VarR=0.872, ρ_sp=0.857, ρ_sr=0.588

### Hard constraints (unchanged)
- Notebook-only implementation

- No data leakage (brain_unseen is test-only)
- Sample balancing in GZSL classifier (downsample to ~8/class, class_weight='balanced')
- Seeds (SEED=42), figures to `figures/` at 150 dpi
