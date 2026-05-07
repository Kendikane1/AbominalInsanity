#!/usr/bin/env python3
"""One-time project reorganisation script. Run once from project root, then archived."""
import shutil
from pathlib import Path

ROOT = Path(__file__).parent


def mkdirs():
    dirs = [
        "experiments/01_wgan_diagnostics/results",
        "experiments/01_wgan_diagnostics/figures",
        "experiments/02_eta_sweep/results",
        "experiments/02_eta_sweep/figures",
        "experiments/03_variance_regularisation/context",
        "experiments/03_variance_regularisation/results",
        "experiments/03_variance_regularisation/figures",
        "experiments/04_film_conditioning/context",
        "experiments/04_film_conditioning/results",
        "experiments/04_film_conditioning/figures",
        "context/math",
        "context/research_history",
        "archive/notebooks/backups",
        "archive/helper_scripts",
        "archive/context/plans",
        "archive/misc/figures",
    ]
    for d in dirs:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    print("  Directories created.")


def mv(src_rel, dst_rel, rename=None):
    src = ROOT / src_rel
    if not src.exists():
        print(f"  SKIP (not found): {src_rel}")
        return
    dst_dir = ROOT / dst_rel
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / (rename or src.name)
    shutil.move(str(src), str(dst))
    print(f"  mv  {src_rel}  →  {dst_rel}/{rename or src.name}")


def write_readme(rel_path, content):
    p = ROOT / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content.strip() + "\n")
    print(f"  write {rel_path}")


def main():
    print("\n=== 1. Create directories ===")
    mkdirs()

    print("\n=== 2. Experiment 01 — WGAN diagnostics figures ===")
    for fname in [
        "diag1_variance_profile.png",
        "diag2_class_conditional_variance.png",
        "diag3_neighbourhood_preservation.png",
    ]:
        mv(f"figures/{fname}", "experiments/01_wgan_diagnostics/figures")

    print("\n=== 3. Experiment 02 — Eta sweep figures ===")
    mv("figures/eta_sweep_results.png", "experiments/02_eta_sweep/figures")

    print("\n=== 4. Experiment 03 — Variance regularisation ===")
    mv(
        "context/variance_regularisation_implementation_directive.md",
        "experiments/03_variance_regularisation/context",
        "implementation_directive.md",
    )
    mv(
        "context/variance_regularisation_debrief.md",
        "experiments/03_variance_regularisation/context",
        "debrief.md",
    )
    mv(
        "var_reg_sweep_results.json",
        "experiments/03_variance_regularisation/results",
        "sweep_results.json",
    )
    mv(
        "context/variance_regularisation_experiment.md",
        "experiments/03_variance_regularisation/results",
        "experiment_log.md",
    )
    for fname in ["E_synth_vareg.npy", "y_synth_vareg.npy", "generator_vareg_best.pt"]:
        mv(fname, "experiments/03_variance_regularisation/results")
    mv(
        "figures/var_reg_alpha_sweep.png",
        "experiments/03_variance_regularisation/figures",
        "alpha_sweep.png",
    )
    mv(
        "figures/var_reg_training_dynamics.png",
        "experiments/03_variance_regularisation/figures",
        "training_dynamics.png",
    )

    print("\n=== 5. context/math ===")
    for fname in ["math_teacher_system_prompt.md", "project_math_context.md", "math_curriculum.md"]:
        mv(f"context/{fname}", "context/math")

    print("\n=== 6. context/research_history ===")
    for fname in [
        "CORnet_S_paradigm_analysis.md",
        "hyperparameter_sweep_analysis.md",
        "WGAN-GP_research.md",
    ]:
        mv(f"context/{fname}", "context/research_history")

    print("\n=== 7. archive/misc — misc root files ===")
    for fname in ["baseline_implementation.py", "build_adder.py", "pgbh35.pdf"]:
        mv(fname, "archive/misc")
    mv("context/ANTIGRAVITY_AGENT.md", "archive/misc")

    print("\n=== 8. archive/notebooks — main notebooks ===")
    mv("GZSL_EEG_Pipeline_v2.ipynb", "archive/notebooks")
    mv("COMP2261_ArizMLCW_with_baseline.ipynb", "archive/notebooks")

    print("\n=== 9. archive/notebooks/backups — backup files ===")
    for f in sorted(ROOT.glob("*.backup*")):
        mv(f.name, "archive/notebooks/backups")
    for f in sorted(ROOT.glob("COMP2261_ArizMLCW_with_baseline.ipynb.backup*")):
        mv(f.name, "archive/notebooks/backups")

    print("\n=== 10. archive/helper_scripts ===")
    helper_dir = ROOT / "helper_files"
    if helper_dir.exists():
        for f in sorted(helper_dir.iterdir()):
            if f.is_file():
                mv(f"helper_files/{f.name}", "archive/helper_scripts")
        shutil.rmtree(helper_dir, ignore_errors=True)
        print("  rmdir helper_files/")

    print("\n=== 11. archive/context/plans ===")
    plans_dir = ROOT / "plans"
    if plans_dir.exists():
        for f in sorted(plans_dir.rglob("*")):
            if f.is_file():
                rel = f.relative_to(ROOT / "plans")
                dst = ROOT / "archive/context/plans" / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(dst))
                print(f"  mv  plans/{rel}  →  archive/context/plans/{rel}")
        shutil.rmtree(plans_dir, ignore_errors=True)
        print("  rmdir plans/")

    print("\n=== 12. archive/context — remaining context files ===")
    ctx_dir = ROOT / "context"
    if ctx_dir.exists():
        for f in sorted(ctx_dir.iterdir()):
            if f.is_file():
                mv(f"context/{f.name}", "archive/context")

    print("\n=== 13. archive/misc/figures — historical figures ===")
    keep_figures = {
        "encoder_loss_curve.png",
        "Paradigm_shift_run",
        "visual_feature_pca_spectrum.png",
        "visual_vs_text_separation.png",
    }
    figs_dir = ROOT / "figures"
    if figs_dir.exists():
        for f in sorted(figs_dir.iterdir()):
            if f.name in keep_figures or f.name.startswith("."):
                continue
            if f.is_file():
                mv(f"figures/{f.name}", "archive/misc/figures")
            elif f.is_dir():
                dst = ROOT / "archive/misc/figures" / f.name
                shutil.move(str(f), str(dst))
                print(f"  mv  figures/{f.name}/  →  archive/misc/figures/{f.name}/")

    print("\n=== 14. Delete .DS_Store files ===")
    for ds in sorted(ROOT.rglob(".DS_Store")):
        ds.unlink()
        print(f"  rm  {ds.relative_to(ROOT)}")

    print("\n=== 15. Write README files ===")
    write_readme(
        "experiments/README.md",
        """# Experiments

| # | Name | Hypothesis | Status | H-mean |
|---|------|-----------|--------|--------|
| 01 | WGAN diagnostics | Characterise synthesis quality | Complete | — |
| 02 | Eta sweep | Post-hoc prototype perturbation fixes VarR | Failed | 4.77→2.56 |
| 03 | Variance regularisation | L_var at training time improves VarR | Failed | 4.77→4.58 |
| 04 | FiLM conditioning | Projection conditioning improves seen→unseen transfer | Active | — |

Each experiment subfolder contains: `README.md`, `notebook.ipynb` (copy of main_pipeline.ipynb + modifications), `context/`, `results/`, `figures/`.
""",
    )

    write_readme(
        "experiments/01_wgan_diagnostics/README.md",
        """# Experiment 01: WGAN-GP Synthesis Diagnostics

**Status**: Complete
**Reference**: `archive/notebooks/GZSL_EEG_Pipeline_v2.ipynb` cells 67–70

## Findings

Three targeted diagnostics on the baseline cWGAN-GP (optimal config, η=0):

| Metric | Value | Interpretation |
|--------|-------|---------------|
| VarR | 0.872 | Mild within-class variance deficit (synth vs real) |
| ρ_sp | 0.857 | **Primary pathology**: structural overcoupling (real unseen: 0.668) |
| kNN@10 | 0.611 | Moderate neighbourhood preservation |

**Conclusion**: The generator over-couples synthetic embeddings to their conditioning prototypes, producing centroids tightly clustered around prototype positions. Within-class variance is mildly insufficient. Both pathologies need addressing for better seen→unseen transfer.
""",
    )

    write_readme(
        "experiments/02_eta_sweep/README.md",
        """# Experiment 02: Eta Sweep (Post-Hoc Prototype Perturbation)

**Status**: Complete — Failed
**Reference**: `archive/notebooks/GZSL_EEG_Pipeline_v2.ipynb` cells 71–76

## Hypothesis

Perturbing prototypes at synthesis time with calibrated Gaussian noise (η·ξ, re-normalised to unit sphere) would increase within-class variance of synthetic embeddings and improve H-mean.

## Results

| η | H-mean | VarR | ρ_sp |
|---|--------|------|------|
| 0.00 | **4.77%** | 0.872 | 0.857 |
| 0.05 | 4.41% | 0.927 | 0.857 |
| 0.10 | 2.56% | 1.002 | 0.854 |
| 0.20 | 0.62% | — | — |
| 0.25 | 0.29% | — | — |

H-mean strictly decreases. At η=0.10, VarR=1.002 (perfect) but H=2.56% — correct variance amount in wrong directions.

## Root Cause

Output perturbation δê ≈ J_G·(η·ξ) lies in the column space of J_G, which does not align with real brain variability directions. Isotropic noise in prototype space maps through a fixed Jacobian into arbitrary (mostly unhelpful) output directions.

**Conclusion**: Post-hoc perturbation is provably ineffective. Fix must be at training time (reshaping J_G itself via weight updates).
""",
    )

    write_readme(
        "experiments/03_variance_regularisation/README.md",
        """# Experiment 03: Variance Regularisation (L_var)

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
""",
    )

    write_readme(
        "experiments/04_film_conditioning/README.md",
        """# Experiment 04: FiLM / Projection-Based Conditioning

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
""",
    )

    print("\n=== 16. Self-archive this script ===")
    mv("reorganise.py", "archive/misc")

    print("\n=== Reorganisation complete! ===\n")


if __name__ == "__main__":
    main()
