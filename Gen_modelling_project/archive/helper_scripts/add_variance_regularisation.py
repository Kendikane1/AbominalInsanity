#!/usr/bin/env python3
"""
Add variance regularisation (L_var) cells to GZSL_EEG_Pipeline_v2.ipynb.

Adds 7 cells (77-83) after the existing 77 cells (0-76):
  Cell 77: Markdown — section header
  Cell 78: Config + function definitions (compute_L_var, train_wgan_with_lvar, calibrate_alpha)
  Cell 79: Compute var_target + correctness verification (5 checks)
  Cell 80: Alpha calibration (gradient-norm matching -> alpha_0)
  Cell 81: Alpha sweep (7 values, full pipeline eval per alpha)
  Cell 82: Analysis + figures (4-panel sweep, 3-panel dynamics)
  Cell 83: Cache best-alpha results

Prerequisites (must run before new cells):
  - Cells 0-41: Data loading, model class defs, data tensors
  - Cell 72: Utility functions (evaluate_eta, compute_sweep_diagnostics, etc.)
  - Cell 73: Encoder training + embeddings + prototypes (_opt suffixed variables)

Reuses class definitions from cells 30 (BrainEncoder, ImageProjector, contrastive_loss),
35 (compute_prototypes), 40 (Generator, Critic, compute_gradient_penalty),
41 (get_prototype_for_labels). Data tensors from cells 7-31.

Mathematical spec: context/variance_regularisation_implementation_directive.md
Critical assessment: context/variance_regularisation_debrief.md

Run from project root:
  python helper_files/add_variance_regularisation.py
"""

import json
import os
import shutil
from datetime import datetime

NOTEBOOK_PATH = "GZSL_EEG_Pipeline_v2.ipynb"


def make_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.split("\n") if isinstance(source, str) else source
    }


def make_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.split("\n") if isinstance(source, str) else source
    }


def fix_source_lines(cell):
    """Ensure each line (except last) ends with \\n."""
    lines = cell["source"]
    fixed = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1 and not line.endswith("\n"):
            fixed.append(line + "\n")
        else:
            fixed.append(line)
    cell["source"] = fixed
    return cell


# =============================================================================
# CELL DEFINITIONS
# =============================================================================

CELL_77_MARKDOWN = """---

# Variance Regularisation (L_var): Training-Time WGAN-GP Modification

**Diagnosis**: WGAN-GP synthesis diagnostics (cells 67-70) revealed two pathologies:
1. **Structural overcoupling**: synthetic centroids track prototypes too faithfully
   (rho_sp=0.857 vs real 0.668)
2. **Variance deficit**: synthetic per-dimension variance is 12.8% below real
   (VarR=0.872)

**Failed intervention**: Noise-augmented prototype conditioning (eta sweep, cells 71-76)
showed monotonic degradation — H-mean dropped from 4.77% to 0.29%. Post-hoc perturbation
adds variance in the wrong directions (column space of J_G, not brain variability).

**This intervention**: Modify the generator's training objective to include a variance
regularisation term that penalises per-class variance deficit:
```
L_G = L_wasserstein + alpha * L_var
L_var = mean_c [ sum_d ( (var_synth_{c,d} - var_target_d)^2 ) ]
```

This reshapes the generator's Jacobian J_G through weight updates, changing *what* the
generator learns rather than perturbing inputs through a fixed J_G.

**Strategy**: Calibrate alpha via gradient-norm matching, then sweep 7 values spanning
2 orders of magnitude. Each alpha retrains the WGAN-GP (encoder is frozen and reused).

**Mathematical spec**: `context/variance_regularisation_implementation_directive.md`
**Critical assessment**: `context/variance_regularisation_debrief.md`"""

CELL_78_CONFIG_FUNCTIONS = r'''# =============================================================================
# VARIANCE REGULARISATION: CONFIG + FUNCTION DEFINITIONS
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
import gc

SEED = 42

# ---- Variance regularisation config ----
VAR_REG_CONFIG = {
    'K_gen': 20,           # samples per class in generator structured batch
    'C_batch': 16,         # classes per generator batch (gen batch = K*C = 320)
    'monitor_interval': 500,   # steps between detailed monitoring
    'alpha_sweep_multipliers': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
}

print("=" * 60)
print("VARIANCE REGULARISATION CONFIG")
print("=" * 60)
for k, v in VAR_REG_CONFIG.items():
    print(f"  {k}: {v}")
print(f"\n  Generator batch size: {VAR_REG_CONFIG['K_gen'] * VAR_REG_CONFIG['C_batch']}")
print(f"  Critic batch size: {OPT_WGAN_CONFIG['batch_size']} (unchanged)")
print("=" * 60)


# ---- L_var loss function ----
def compute_L_var(fake_embeddings, class_labels, var_target):
    """
    Variance regularisation loss.

    L_var = (1/|C_batch|) * sum_c [ sum_d ( (var_synth_{c,d} - var_target_d)^2 ) ]

    Args:
        fake_embeddings: (batch_size, d) generated embeddings on S^{d-1}, WITH grad
        class_labels:    (batch_size,) integer class label for each sample
        var_target:      (d,) precomputed target variance vector (detached)

    Returns:
        Scalar loss with requires_grad=True (connected to generator graph)

    Note: uses absolute L2 distance. If high-variance dimensions dominate while
    low-variance ones are ignored, consider relative alternative:
        sum_d ( (var_synth_d / var_target_d - 1)^2 )
    """
    unique_classes = torch.unique(class_labels)
    loss = torch.tensor(0.0, device=fake_embeddings.device)
    for c in unique_classes:
        mask = (class_labels == c)
        samples_c = fake_embeddings[mask]          # (K, d)
        mu_c = samples_c.mean(dim=0)               # (d,)
        var_c = ((samples_c - mu_c) ** 2).mean(dim=0)  # (d,)
        loss = loss + ((var_c - var_target) ** 2).sum()
    loss = loss / len(unique_classes)
    return loss


# ---- Modified WGAN-GP training function ----
def train_wgan_with_lvar(alpha, var_target, E_train, y_train, S_seen_protos,
                          seen_classes, wgan_config, var_reg_config, device):
    """
    Train WGAN-GP with variance regularisation: L_G = L_wasserstein + alpha * L_var.

    Critic step: UNCHANGED (random batching from DataLoader).
    Generator step: MODIFIED (structured batching for L_var).

    Args:
        alpha: weight for L_var term
        var_target: (d,) target variance vector (tensor on device)
        E_train: (N, d) real training embeddings (numpy)
        y_train: (N,) training labels (numpy)
        S_seen_protos: dict {class_id: prototype_vector} (numpy)
        seen_classes: sorted list of seen class IDs
        wgan_config: OPT_WGAN_CONFIG dict
        var_reg_config: VAR_REG_CONFIG dict
        device: torch device

    Returns:
        generator: trained Generator model
        history: dict with per-step and per-monitor-interval logs
    """
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    d = wgan_config['embed_dim']
    z_dim = wgan_config['z_dim']
    n_steps = wgan_config['n_steps']
    n_critic = wgan_config['n_critic']
    K = var_reg_config['K_gen']
    C = var_reg_config['C_batch']
    monitor_interval = var_reg_config['monitor_interval']

    seen_classes_arr = np.array(seen_classes)

    # ---- Instantiate fresh models ----
    gen = Generator(z_dim=z_dim, embed_dim=d).to(device)
    crit = Critic(embed_dim=d).to(device)

    g_optim = torch.optim.Adam(gen.parameters(), lr=wgan_config['lr'],
                                betas=wgan_config['betas'])
    c_optim = torch.optim.Adam(crit.parameters(), lr=wgan_config['lr'],
                                betas=wgan_config['betas'])

    # ---- Critic DataLoader (random batching, unchanged) ----
    E_train_t = torch.FloatTensor(E_train)
    S_train_cond = torch.FloatTensor(get_prototype_for_labels(y_train, S_seen_protos))
    wgan_ds = TensorDataset(E_train_t, S_train_cond)
    wgan_dl = DataLoader(wgan_ds, batch_size=wgan_config['batch_size'],
                          shuffle=True, drop_last=True)
    data_iter = iter(wgan_dl)

    # ---- Prototype lookup for structured batching ----
    proto_array = {c: S_seen_protos[c] for c in seen_classes}

    # ---- History ----
    hist = {
        'L_wasserstein': [], 'L_var': [], 'L_G': [], 'L_D': [],
        'monitor_steps': [],
        'grad_norm_w': [], 'grad_norm_v': [], 'grad_cos_sim': [],
        'VarR': [],
    }

    rng = np.random.RandomState(SEED)
    t0 = time.time()
    g_step = 0

    while g_step < n_steps:
        # -----------------------------------------------------------------
        # Critic update (n_critic times) — UNCHANGED
        # -----------------------------------------------------------------
        for _ in range(n_critic):
            try:
                e_real, s_c = next(data_iter)
            except StopIteration:
                data_iter = iter(wgan_dl)
                e_real, s_c = next(data_iter)
            e_real, s_c = e_real.to(device), s_c.to(device)
            bs = e_real.size(0)

            z = torch.randn(bs, z_dim, device=device)
            e_fake = gen(z, s_c)
            d_real = crit(e_real, s_c)
            d_fake = crit(e_fake.detach(), s_c)
            gp = compute_gradient_penalty(crit, e_real, e_fake.detach(), s_c, device)
            c_loss = -d_real.mean() + d_fake.mean() + wgan_config['lambda_gp'] * gp

            c_optim.zero_grad()
            c_loss.backward()
            c_optim.step()

        # -----------------------------------------------------------------
        # Generator update — MODIFIED (structured batching + L_var)
        # -----------------------------------------------------------------
        # Sample C classes, K samples each
        batch_classes = rng.choice(seen_classes_arr, size=C, replace=False)
        s_c_list = []
        labels_list = []
        for c_id in batch_classes:
            proto = proto_array[c_id]
            s_c_list.append(np.tile(proto, (K, 1)))
            labels_list.append(np.full(K, c_id, dtype=np.int64))
        s_c_gen = torch.FloatTensor(np.vstack(s_c_list)).to(device)
        class_labels = torch.LongTensor(np.concatenate(labels_list)).to(device)

        z = torch.randn(C * K, z_dim, device=device)
        e_fake = gen(z, s_c_gen)

        # Wasserstein loss
        L_w = -crit(e_fake, s_c_gen).mean()

        # Variance regularisation loss
        L_v = compute_L_var(e_fake, class_labels, var_target)

        # Combined generator loss
        L_G_total = L_w + alpha * L_v

        # -----------------------------------------------------------------
        # Monitoring (before backward, using retain_graph)
        # -----------------------------------------------------------------
        do_monitor = (g_step % monitor_interval == 0) or (g_step == n_steps - 1)

        if do_monitor:
            # Separate gradient norms (Concern 6: gradient alignment)
            grad_w = torch.autograd.grad(L_w, gen.parameters(),
                                          retain_graph=True, create_graph=False)
            grad_v = torch.autograd.grad(L_v, gen.parameters(),
                                          retain_graph=True, create_graph=False)

            norm_w = torch.sqrt(sum(g.pow(2).sum() for g in grad_w)).item()
            norm_v = torch.sqrt(sum(g.pow(2).sum() for g in grad_v)).item()

            # Cosine similarity
            flat_w = torch.cat([g.flatten() for g in grad_w])
            flat_v = torch.cat([g.flatten() for g in grad_v])
            cos_sim = F.cosine_similarity(flat_w.unsqueeze(0),
                                           flat_v.unsqueeze(0)).item()

            # VarR: sample classes, generate, measure variance
            gen.eval()
            with torch.no_grad():
                mon_classes = rng.choice(seen_classes_arr,
                                         size=min(50, len(seen_classes_arr)),
                                         replace=False)
                mon_vars = []
                for mc in mon_classes:
                    s_m = torch.FloatTensor(proto_array[mc]).unsqueeze(0).repeat(K, 1).to(device)
                    z_m = torch.randn(K, z_dim, device=device)
                    e_m = gen(z_m, s_m)
                    var_m = ((e_m - e_m.mean(dim=0)) ** 2).mean(dim=0)
                    mon_vars.append(var_m)
                mean_synth_var = torch.stack(mon_vars).mean(dim=0)
                var_ratio = mean_synth_var.sum().item() / var_target.sum().item()
            gen.train()

            hist['monitor_steps'].append(g_step)
            hist['grad_norm_w'].append(norm_w)
            hist['grad_norm_v'].append(norm_v)
            hist['grad_cos_sim'].append(cos_sim)
            hist['VarR'].append(var_ratio)

        # -----------------------------------------------------------------
        # Backward + step
        # -----------------------------------------------------------------
        g_optim.zero_grad()
        L_G_total.backward()
        g_optim.step()

        # Log
        hist['L_wasserstein'].append(L_w.item())
        hist['L_var'].append(L_v.item())
        hist['L_G'].append(L_G_total.item())
        hist['L_D'].append(c_loss.item())

        g_step += 1

        if g_step % 2500 == 0 or g_step == 1:
            vr_str = f", VarR={hist['VarR'][-1]:.3f}" if hist['VarR'] else ""
            print(f"  Step {g_step:5d}/{n_steps}: "
                  f"L_w={L_w.item():.4f}, L_var={L_v.item():.6f}, "
                  f"L_D={c_loss.item():.4f}{vr_str}")

    elapsed = time.time() - t0

    # ---- End-of-training check (Concern 5: may need extending) ----
    L_var_arr = np.array(hist['L_var'])
    if len(L_var_arr) > 4000:
        mid_avg = L_var_arr[4000:6000].mean()
        end_avg = L_var_arr[-2000:].mean()
        if mid_avg > 0 and end_avg < 0.8 * mid_avg:
            print(f"  WARNING: L_var still decreasing significantly "
                  f"(mid={mid_avg:.6f} -> end={end_avg:.6f}, "
                  f"-{100*(1-end_avg/mid_avg):.0f}%). "
                  f"Consider extending training beyond {n_steps} steps.")

    print(f"  Training complete in {elapsed:.0f}s. "
          f"L_w={hist['L_wasserstein'][-1]:.4f}, "
          f"L_var={hist['L_var'][-1]:.6f}, "
          f"VarR={hist['VarR'][-1]:.3f}")

    return gen, hist


# ---- Alpha calibration function ----
def calibrate_alpha(var_target, S_seen_protos, seen_classes, wgan_config,
                     var_reg_config, device):
    """
    Determine alpha_0 by matching gradient norms of L_wasserstein and L_var.

    Instantiates a fresh Generator + Critic, runs one structured generator step,
    measures ||grad L_w|| and ||grad L_var|| separately.

    Returns:
        alpha_0: float, ||grad L_w|| / ||grad L_var||
        info: dict with norms, losses, cosine similarity
    """
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    d = wgan_config['embed_dim']
    z_dim = wgan_config['z_dim']
    K = var_reg_config['K_gen']
    C = var_reg_config['C_batch']

    gen = Generator(z_dim=z_dim, embed_dim=d).to(device)
    crit = Critic(embed_dim=d).to(device)

    seen_classes_arr = np.array(seen_classes)
    rng = np.random.RandomState(SEED)

    # Structured batch
    batch_classes = rng.choice(seen_classes_arr, size=C, replace=False)
    s_c_list, labels_list = [], []
    for c_id in batch_classes:
        proto = S_seen_protos[c_id]
        s_c_list.append(np.tile(proto, (K, 1)))
        labels_list.append(np.full(K, c_id, dtype=np.int64))
    s_c = torch.FloatTensor(np.vstack(s_c_list)).to(device)
    class_labels = torch.LongTensor(np.concatenate(labels_list)).to(device)

    z = torch.randn(C * K, z_dim, device=device)
    e_fake = gen(z, s_c)

    # Wasserstein term
    L_w = -crit(e_fake, s_c).mean()
    grad_w = torch.autograd.grad(L_w, gen.parameters(), retain_graph=True)
    norm_w = torch.sqrt(sum(g.pow(2).sum() for g in grad_w)).item()

    # L_var term
    L_v = compute_L_var(e_fake, class_labels, var_target)
    grad_v = torch.autograd.grad(L_v, gen.parameters())
    norm_v = torch.sqrt(sum(g.pow(2).sum() for g in grad_v)).item()

    # Cosine similarity
    flat_w = torch.cat([g.flatten() for g in grad_w])
    flat_v = torch.cat([g.flatten() for g in grad_v])
    cos_sim = F.cosine_similarity(flat_w.unsqueeze(0), flat_v.unsqueeze(0)).item()

    alpha_0 = norm_w / (norm_v + 1e-10)

    info = {
        'norm_w': norm_w, 'norm_v': norm_v,
        'L_w': L_w.item(), 'L_v': L_v.item(),
        'cos_sim': cos_sim,
    }

    del gen, crit
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

    return alpha_0, info


print("Functions defined: compute_L_var, train_wgan_with_lvar, calibrate_alpha")'''

CELL_79_VAR_TARGET = r'''# =============================================================================
# COMPUTE var_target + CORRECTNESS VERIFICATION
# =============================================================================
# var_target: average per-class per-dimension variance of SEEN training embeddings.
# Computed from E_train_opt (L2-normalised encoder output on S^63, from cell 73).
# This is a fixed 64-D vector used as the regularisation target.

d = OPT_ENCODER_CONFIG['embed_dim']

# ---- Step 1: Compute var_target ----
print("Computing var_target from training seen embeddings...")
var_target_list = []
class_sample_counts = []
for c in seen_classes_opt:
    mask = (y_train_opt == c)
    emb_c = E_train_opt[mask]  # (n_c, 64)
    n_c = emb_c.shape[0]
    class_sample_counts.append(n_c)
    mu_c = emb_c.mean(axis=0)                        # (64,)
    var_c = ((emb_c - mu_c) ** 2).mean(axis=0)       # (64,)
    var_target_list.append(var_c)

var_target_np = np.stack(var_target_list).mean(axis=0)   # (64,)
var_target = torch.FloatTensor(var_target_np).to(device)

print(f"\nvar_target statistics:")
print(f"  Shape: {var_target.shape}")
print(f"  All positive: {(var_target > 0).all().item()}")
print(f"  Mean:  {var_target.mean().item():.6f}")
print(f"  Std:   {var_target.std().item():.6f}")
print(f"  Min:   {var_target.min().item():.6f}")
print(f"  Max:   {var_target.max().item():.6f}")
print(f"  Sum:   {var_target.sum().item():.6f} (L2-norm budget: should be < 1.0)")
print(f"  Classes: {len(var_target_list)}")
print(f"  Samples/class: min={min(class_sample_counts)}, "
      f"max={max(class_sample_counts)}, "
      f"median={int(np.median(class_sample_counts))}")

# ---- Step 2: Correctness verification (directive Section 7 checklist) ----
print("\n" + "=" * 60)
print("CORRECTNESS VERIFICATION")
print("=" * 60)

all_pass = True

# Check 1: var_target shape and positivity
check1 = var_target.shape == (d,) and (var_target > 0).all().item()
print(f"  [{'PASS' if check1 else 'FAIL'}] Check 1: var_target shape ({d},) and all positive")
all_pass &= check1

# Check 2: compute_L_var returns scalar with requires_grad
torch.manual_seed(SEED)
test_gen = Generator(z_dim=OPT_WGAN_CONFIG['z_dim'], embed_dim=d).to(device)
test_z = torch.randn(32, OPT_WGAN_CONFIG['z_dim'], device=device)
test_s = torch.randn(32, d, device=device)
test_s = F.normalize(test_s, p=2, dim=-1)
test_labels = torch.LongTensor([0]*16 + [1]*16).to(device)
test_fake = test_gen(test_z, test_s)
test_loss = compute_L_var(test_fake, test_labels, var_target)
check2 = test_loss.dim() == 0 and test_loss.requires_grad
print(f"  [{'PASS' if check2 else 'FAIL'}] Check 2: compute_L_var returns scalar "
      f"(dim={test_loss.dim()}) with requires_grad={test_loss.requires_grad}")
all_pass &= check2

# Check 3: L_var = 0 when variance matches target exactly
# Craft input: for each class, samples whose variance equals var_target
torch.manual_seed(SEED)
n_test_classes = 3
K_test = 20
crafted_embeddings = []
crafted_labels = []
for ci in range(n_test_classes):
    # Create samples centred at origin with variance = var_target
    # x_k = sqrt(var_target) * u_k where u_k are unit-variance samples centred at 0
    # Use: x_k,d = mu_d + sqrt(var_target_d) * (standard normal centred)
    raw = torch.randn(K_test, d, device=device)
    raw = raw - raw.mean(dim=0)  # centre exactly
    # Scale to match target variance: var(raw) per dim, then rescale
    raw_var = ((raw ** 2).mean(dim=0)).sqrt()
    raw_var = raw_var.clamp(min=1e-8)
    target_std = var_target.sqrt()
    scaled = raw * (target_std / raw_var)
    # Verify: scaled has var = var_target per dim (up to numerical precision)
    crafted_embeddings.append(scaled)
    crafted_labels.extend([ci] * K_test)
crafted_emb = torch.cat(crafted_embeddings, dim=0)
crafted_lab = torch.LongTensor(crafted_labels).to(device)
# Need requires_grad for the function to work in autograd context
crafted_emb.requires_grad_(True)
L_var_zero = compute_L_var(crafted_emb, crafted_lab, var_target)
check3 = L_var_zero.item() < 1e-8
print(f"  [{'PASS' if check3 else 'FAIL'}] Check 3: L_var=0 when variance matches target "
      f"(L_var={L_var_zero.item():.2e})")
all_pass &= check3

# Check 4: Gradient verification — autograd vs analytical formula
# Analytical: dL_var/de_{c,k} = (4/K) * Delta_c * (e_{c,k} - mu_hat_c)
# where Delta_c = var_synth_c - var_target
torch.manual_seed(42)
K_gv = 10
C_gv = 3
gv_embeddings = torch.randn(K_gv * C_gv, d, device=device, requires_grad=True)
gv_labels = torch.LongTensor(sum([[ci]*K_gv for ci in range(C_gv)], [])).to(device)
gv_loss = compute_L_var(gv_embeddings, gv_labels, var_target)
gv_loss.backward()
autograd_grad = gv_embeddings.grad.clone()

# Analytical gradient
with torch.no_grad():
    analytical_grad = torch.zeros_like(gv_embeddings)
    for ci in range(C_gv):
        mask = (gv_labels == ci)
        samples_c = gv_embeddings[mask]  # (K_gv, d)
        mu_c = samples_c.mean(dim=0)
        var_c = ((samples_c - mu_c) ** 2).mean(dim=0)
        delta_c = var_c - var_target
        # dL_var/de_{c,k} = (4/K) * delta * (e - mu)
        # But also divided by C_gv (the mean over classes)
        for k_idx, global_idx in enumerate(torch.where(mask)[0]):
            analytical_grad[global_idx] = (4.0 / K_gv) * delta_c * (samples_c[k_idx] - mu_c) / C_gv

max_diff = (autograd_grad - analytical_grad).abs().max().item()
check4 = max_diff < 1e-4
print(f"  [{'PASS' if check4 else 'FAIL'}] Check 4: Gradient verification "
      f"(max |autograd - analytical| = {max_diff:.2e}, threshold=1e-4)")
all_pass &= check4

# Check 5: Generator outputs on S^{d-1} (unit sphere)
torch.manual_seed(SEED)
check_gen = Generator(z_dim=OPT_WGAN_CONFIG['z_dim'], embed_dim=d).to(device)
check_z = torch.randn(100, OPT_WGAN_CONFIG['z_dim'], device=device)
check_s = F.normalize(torch.randn(100, d, device=device), p=2, dim=-1)
with torch.no_grad():
    check_out = check_gen(check_z, check_s)
    norms = check_out.norm(p=2, dim=-1)
check5 = (norms - 1.0).abs().max().item() < 1e-5
print(f"  [{'PASS' if check5 else 'FAIL'}] Check 5: Generator outputs on S^{d-1} "
      f"(max |norm - 1| = {(norms - 1.0).abs().max().item():.2e})")
all_pass &= check5

# Cleanup
del test_gen, check_gen, test_fake, crafted_emb, gv_embeddings
torch.cuda.empty_cache() if torch.cuda.is_available() else None
gc.collect()

print("=" * 60)
if all_pass:
    print("ALL CHECKS PASSED. Ready to proceed with alpha calibration.")
else:
    print("SOME CHECKS FAILED. Investigate before proceeding.")
print("=" * 60)'''

CELL_80_CALIBRATION = r'''# =============================================================================
# ALPHA CALIBRATION (GRADIENT-NORM MATCHING)
# =============================================================================
# Determine alpha_0 such that ||grad L_wasserstein|| ~ ||alpha * grad L_var||
# i.e. alpha_0 = ||grad L_w|| / ||grad L_var||
# Then sweep alpha in {0.01*alpha_0, ..., 10*alpha_0}

print("Running alpha calibration...")
alpha_0, cal_info = calibrate_alpha(
    var_target=var_target,
    S_seen_protos=S_seen_opt,
    seen_classes=seen_classes_opt,
    wgan_config=OPT_WGAN_CONFIG,
    var_reg_config=VAR_REG_CONFIG,
    device=device,
)

print(f"\n{'=' * 60}")
print("ALPHA CALIBRATION RESULTS")
print(f"{'=' * 60}")
print(f"  ||grad L_wasserstein||: {cal_info['norm_w']:.6f}")
print(f"  ||grad L_var||:         {cal_info['norm_v']:.6f}")
print(f"  alpha_0 (norm ratio):   {alpha_0:.6f}")
print(f"  L_wasserstein:          {cal_info['L_w']:.4f}")
print(f"  L_var:                  {cal_info['L_v']:.6f}")
print(f"  cos(grad_w, grad_v):    {cal_info['cos_sim']:.4f}")

# Sanity check
if alpha_0 < 1e-6 or alpha_0 > 1e6:
    print(f"\n  WARNING: alpha_0 = {alpha_0:.2e} is extreme. "
          f"Gradient norms differ by >6 OOM. Inspect L_var scale.")

# Compute sweep values
alpha_sweep_values = [m * alpha_0 for m in VAR_REG_CONFIG['alpha_sweep_multipliers']]
print(f"\nAlpha sweep values (7 points, 2 OOM range):")
for m, a in zip(VAR_REG_CONFIG['alpha_sweep_multipliers'], alpha_sweep_values):
    label = " <-- alpha_0" if m == 1.0 else ""
    print(f"  {m:5.2f} x alpha_0 = {a:.6f}{label}")
print(f"{'=' * 60}")'''

CELL_81_ALPHA_SWEEP = r'''# =============================================================================
# ALPHA SWEEP: TRAIN MODIFIED WGAN-GP FOR EACH ALPHA
# =============================================================================
# Each alpha: fresh WGAN-GP training (10,000 steps) -> synthesise -> evaluate.
# Encoder is reused from cell 73 (frozen).

print("=" * 60)
print(f"ALPHA SWEEP: {len(alpha_sweep_values)} values")
print("=" * 60)
print(f"  WGAN-GP steps per alpha: {OPT_WGAN_CONFIG['n_steps']}")
print(f"  Structured batch: K={VAR_REG_CONFIG['K_gen']} x C={VAR_REG_CONFIG['C_batch']} "
      f"= {VAR_REG_CONFIG['K_gen'] * VAR_REG_CONFIG['C_batch']}")
print(f"  Baseline to beat: H=4.77%, AccS=4.11%, AccU=5.69%, VarR=0.872, rho_sp=0.857")
print()

sweep_results = []
sweep_histories = []
t_sweep_start = time.time()

for i, alpha_val in enumerate(alpha_sweep_values):
    mult = VAR_REG_CONFIG['alpha_sweep_multipliers'][i]
    print(f"\n{'─' * 60}")
    print(f"[{i+1}/{len(alpha_sweep_values)}] alpha = {alpha_val:.6f} ({mult:.2f} x alpha_0)")
    print(f"{'─' * 60}")

    # Train modified WGAN-GP
    gen_a, hist_a = train_wgan_with_lvar(
        alpha=alpha_val,
        var_target=var_target,
        E_train=E_train_opt,
        y_train=y_train_opt,
        S_seen_protos=S_seen_opt,
        seen_classes=seen_classes_opt,
        wgan_config=OPT_WGAN_CONFIG,
        var_reg_config=VAR_REG_CONFIG,
        device=device,
    )

    # Evaluate full pipeline (reuse evaluate_eta with eta=0)
    result = evaluate_eta(
        eta=0.0,
        generator=gen_a,
        E_train=E_train_opt, y_train=y_train_opt,
        E_test=E_test_opt, y_test=y_test_opt,
        E_unseen=E_unseen_opt, y_unseen=y_unseen_opt,
        S_unseen_protos=S_unseen_opt, S_unseen_array=S_unseen_arr_opt,
        unseen_classes=unseen_classes_opt,
        seen_labels_set=seen_labels_set_opt,
        unseen_labels_set=unseen_labels_set_opt,
        median_per_class=median_per_class_opt,
        device=device,
    )

    result['alpha'] = alpha_val
    result['alpha_mult'] = mult
    result['final_L_var'] = hist_a['L_var'][-1]
    result['final_L_w'] = hist_a['L_wasserstein'][-1]
    result['final_VarR'] = hist_a['VarR'][-1] if hist_a['VarR'] else None
    result['final_cos_sim'] = hist_a['grad_cos_sim'][-1] if hist_a['grad_cos_sim'] else None

    sweep_results.append(result)
    sweep_histories.append(hist_a)

    # Progressive results table
    print(f"\n  Results: H={result['H']*100:.2f}%, "
          f"AccS={result['acc_seen']*100:.2f}%, AccU={result['acc_unseen']*100:.2f}%, "
          f"Route={result['routing_rate']*100:.1f}%, "
          f"VarR={result['var_ratio']:.3f}, rho_sp={result['rho_synth_proto']:.3f}")

    # Cleanup
    del gen_a
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

t_sweep = time.time() - t_sweep_start
print(f"\n{'=' * 60}")
print(f"SWEEP COMPLETE ({t_sweep:.0f}s total)")
print(f"{'=' * 60}")

# Summary table
print(f"\n{'alpha':>10s} {'mult':>6s} {'H%':>7s} {'AccS%':>7s} {'AccU%':>7s} "
      f"{'Route%':>7s} {'VarR':>6s} {'rho_sp':>7s} {'rho_sr':>7s} {'kNN10':>6s}")
print("-" * 85)
for r in sweep_results:
    print(f"{r['alpha']:10.6f} {r['alpha_mult']:6.2f} "
          f"{r['H']*100:7.2f} {r['acc_seen']*100:7.2f} {r['acc_unseen']*100:7.2f} "
          f"{r['routing_rate']*100:7.1f} {r['var_ratio']:6.3f} "
          f"{r['rho_synth_proto']:7.4f} {r['rho_synth_real']:7.4f} {r['knn_10']:6.3f}")
print("-" * 85)
print(f"{'BASELINE':>10s} {'---':>6s} "
      f"{'4.77':>7s} {'4.11':>7s} {'5.69':>7s} "
      f"{'20.0':>7s} {'0.872':>6s} {'0.8572':>7s} {'0.5875':>7s} {'0.611':>6s}")

# Identify best
best_idx = max(range(len(sweep_results)), key=lambda i: sweep_results[i]['H'])
best = sweep_results[best_idx]
print(f"\nBest alpha: {best['alpha']:.6f} ({best['alpha_mult']:.2f}x alpha_0)")
print(f"  H={best['H']*100:.2f}% (baseline: 4.77%, delta: {(best['H']-0.0477)*100:+.2f}pp)")'''

CELL_82_ANALYSIS = r'''# =============================================================================
# ANALYSIS + FIGURES
# =============================================================================

import os
os.makedirs('figures', exist_ok=True)

alphas = [r['alpha'] for r in sweep_results]
mults = [r['alpha_mult'] for r in sweep_results]
Hs = [r['H'] * 100 for r in sweep_results]
AccSs = [r['acc_seen'] * 100 for r in sweep_results]
AccUs = [r['acc_unseen'] * 100 for r in sweep_results]
VarRs = [r['var_ratio'] for r in sweep_results]
rho_sps = [r['rho_synth_proto'] for r in sweep_results]
rho_srs = [r['rho_synth_real'] for r in sweep_results]

# ---- Figure 1: Alpha sweep results (4-panel) ----
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Variance Regularisation: Alpha Sweep Results', fontsize=14, fontweight='bold')

# Panel A: H-mean vs alpha
ax = axes[0, 0]
ax.semilogx(alphas, Hs, 'bo-', markersize=6, linewidth=1.5)
ax.axhline(y=4.77, color='r', linestyle='--', alpha=0.7, label='Baseline H=4.77%')
ax.set_xlabel('alpha')
ax.set_ylabel('H-mean (%)')
ax.set_title('A. Harmonic Mean vs Alpha')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel B: AccS and AccU vs alpha
ax = axes[0, 1]
ax.semilogx(alphas, AccSs, 's-', color='tab:blue', markersize=5, label='AccS', linewidth=1.5)
ax.semilogx(alphas, AccUs, 'D-', color='tab:orange', markersize=5, label='AccU', linewidth=1.5)
ax.axhline(y=4.11, color='tab:blue', linestyle='--', alpha=0.4, label='Baseline AccS=4.11%')
ax.axhline(y=5.69, color='tab:orange', linestyle='--', alpha=0.4, label='Baseline AccU=5.69%')
ax.set_xlabel('alpha')
ax.set_ylabel('Accuracy (%)')
ax.set_title('B. Seen/Unseen Accuracy vs Alpha')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel C: VarR vs alpha
ax = axes[1, 0]
ax.semilogx(alphas, VarRs, 'g^-', markersize=6, linewidth=1.5)
ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Target VarR=1.0')
ax.axhline(y=0.872, color='r', linestyle='--', alpha=0.5, label='Baseline VarR=0.872')
ax.set_xlabel('alpha')
ax.set_ylabel('VarR (synth/real variance ratio)')
ax.set_title('C. Variance Ratio vs Alpha')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel D: rho_sp vs alpha
ax = axes[1, 1]
ax.semilogx(alphas, rho_sps, 'rv-', markersize=6, linewidth=1.5, label='rho(synth, proto)')
ax.semilogx(alphas, rho_srs, 'mp-', markersize=5, linewidth=1.5, label='rho(synth, real)')
ax.axhline(y=0.8572, color='r', linestyle='--', alpha=0.4, label='Baseline rho_sp=0.857')
ax.axhline(y=0.668, color='gray', linestyle='--', alpha=0.4, label='Real rho_rp=0.668')
ax.set_xlabel('alpha')
ax.set_ylabel('Spearman rho')
ax.set_title('D. Structural Coupling vs Alpha')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/var_reg_alpha_sweep.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figures/var_reg_alpha_sweep.png")

# ---- Figure 2: Training dynamics for best alpha (3-panel) ----
best_hist = sweep_histories[best_idx]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle(f'Training Dynamics (best alpha = {sweep_results[best_idx]["alpha"]:.6f})',
             fontsize=13, fontweight='bold')

# Panel A: L_wasserstein and L_var over steps
ax1 = axes[0]
steps = np.arange(len(best_hist['L_wasserstein']))
color1 = 'tab:blue'
ax1.plot(steps, best_hist['L_wasserstein'], color=color1, alpha=0.4, linewidth=0.5)
# Smoothed
window = 200
if len(best_hist['L_wasserstein']) > window:
    L_w_smooth = np.convolve(best_hist['L_wasserstein'],
                              np.ones(window)/window, mode='valid')
    ax1.plot(np.arange(len(L_w_smooth)) + window//2, L_w_smooth,
             color=color1, linewidth=1.5, label='L_wasserstein (smoothed)')
ax1.set_xlabel('Generator Step')
ax1.set_ylabel('L_wasserstein', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.plot(steps, best_hist['L_var'], color=color2, alpha=0.4, linewidth=0.5)
if len(best_hist['L_var']) > window:
    L_v_smooth = np.convolve(best_hist['L_var'],
                              np.ones(window)/window, mode='valid')
    ax2.plot(np.arange(len(L_v_smooth)) + window//2, L_v_smooth,
             color=color2, linewidth=1.5, label='L_var (smoothed)')
ax2.set_ylabel('L_var', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
ax1.set_title('A. Loss Components')
ax1.grid(True, alpha=0.3)

# Panel B: VarR over steps
ax = axes[1]
mon_steps = best_hist['monitor_steps']
ax.plot(mon_steps, best_hist['VarR'], 'go-', markersize=4, linewidth=1.5)
ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Target')
ax.axhline(y=0.872, color='r', linestyle='--', alpha=0.5, label='Baseline')
ax.set_xlabel('Generator Step')
ax.set_ylabel('VarR')
ax.set_title('B. Variance Ratio During Training')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel C: Gradient norms + cosine similarity
ax = axes[2]
ax.plot(mon_steps, best_hist['grad_norm_w'], 's-', color='tab:blue',
        markersize=3, linewidth=1, label='||grad L_w||')
ax.plot(mon_steps, best_hist['grad_norm_v'], 'D-', color='tab:red',
        markersize=3, linewidth=1, label='||grad L_var|| (x alpha)')
ax.set_xlabel('Generator Step')
ax.set_ylabel('Gradient Norm')
ax.legend(loc='upper left', fontsize=8)

ax3 = ax.twinx()
ax3.plot(mon_steps, best_hist['grad_cos_sim'], 'k-', alpha=0.5, linewidth=1,
         label='cos(grad_w, grad_v)')
ax3.set_ylabel('Cosine Similarity', color='gray')
ax3.tick_params(axis='y', labelcolor='gray')
ax3.set_ylim(-1.1, 1.1)
ax3.legend(loc='upper right', fontsize=8)
ax.set_title('C. Gradient Norms + Alignment')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/var_reg_training_dynamics.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figures/var_reg_training_dynamics.png")

# ---- Summary table ----
print(f"\n{'=' * 60}")
print("COMPARISON WITH BASELINE")
print(f"{'=' * 60}")
print(f"{'Metric':>20s} {'Baseline':>10s} {'Best alpha':>10s} {'Delta':>10s}")
print("-" * 55)
print(f"{'H-mean (%)':>20s} {'4.77':>10s} {best['H']*100:>10.2f} "
      f"{(best['H']-0.0477)*100:>+10.2f}")
print(f"{'AccS (%)':>20s} {'4.11':>10s} {best['acc_seen']*100:>10.2f} "
      f"{(best['acc_seen']-0.0411)*100:>+10.2f}")
print(f"{'AccU (%)':>20s} {'5.69':>10s} {best['acc_unseen']*100:>10.2f} "
      f"{(best['acc_unseen']-0.0569)*100:>+10.2f}")
print(f"{'Routing (%)':>20s} {'20.0':>10s} {best['routing_rate']*100:>10.1f} "
      f"{(best['routing_rate']-0.200)*100:>+10.1f}")
print(f"{'VarR':>20s} {'0.872':>10s} {best['var_ratio']:>10.3f} "
      f"{best['var_ratio']-0.872:>+10.3f}")
print(f"{'rho_sp':>20s} {'0.8572':>10s} {best['rho_synth_proto']:>10.4f} "
      f"{best['rho_synth_proto']-0.8572:>+10.4f}")
print(f"{'rho_sr':>20s} {'0.5875':>10s} {best['rho_synth_real']:>10.4f} "
      f"{best['rho_synth_real']-0.5875:>+10.4f}")
print(f"{'kNN@10':>20s} {'0.611':>10s} {best['knn_10']:>10.3f} "
      f"{best['knn_10']-0.611:>+10.3f}")
print(f"{'alpha':>20s} {'---':>10s} {best['alpha']:>10.6f}")
print(f"{'alpha_mult':>20s} {'---':>10s} {best['alpha_mult']:>10.2f}")
print("-" * 55)

# Concern 1 check: did rho_sp decrease?
if best['rho_synth_proto'] < 0.857:
    print("\nrho_sp DECREASED — L_var indirectly reduced structural overcoupling.")
else:
    print("\nrho_sp did NOT decrease — L_var addressed variance but not overcoupling.")
    print("Next intervention if needed: L_struct (inter-class structure loss).")'''

CELL_83_CACHE = r'''# =============================================================================
# CACHE BEST-ALPHA RESULTS
# =============================================================================

import os
os.makedirs('figures', exist_ok=True)

best_alpha = sweep_results[best_idx]['alpha']
best_mult = sweep_results[best_idx]['alpha_mult']

print(f"Best alpha: {best_alpha:.6f} ({best_mult:.2f}x alpha_0)")
print(f"H-mean: {best['H']*100:.2f}%")
print()

# Retrain best config for caching (or reuse if we stored the generator)
# Since generators were deleted during sweep, retrain the best one
print("Retraining best-alpha WGAN-GP for caching...")
gen_best, hist_best = train_wgan_with_lvar(
    alpha=best_alpha,
    var_target=var_target,
    E_train=E_train_opt,
    y_train=y_train_opt,
    S_seen_protos=S_seen_opt,
    seen_classes=seen_classes_opt,
    wgan_config=OPT_WGAN_CONFIG,
    var_reg_config=VAR_REG_CONFIG,
    device=device,
)

# Generate synthetic embeddings
print("\nGenerating synthetic unseen embeddings...")
E_synth_vareg, y_synth_vareg = synthesise_with_eta(
    generator=gen_best,
    unseen_classes=unseen_classes_opt,
    S_unseen_protos=S_unseen_opt,
    n_synth=OPT_WGAN_CONFIG['n_synth_per_class'],
    z_dim=OPT_WGAN_CONFIG['z_dim'],
    eta=0.0,
    device=device,
)
print(f"E_synth_vareg: {E_synth_vareg.shape}")
print(f"y_synth_vareg: {y_synth_vareg.shape}")

# Cache arrays
np.save('E_synth_vareg.npy', E_synth_vareg)
np.save('y_synth_vareg.npy', y_synth_vareg)
print("\nCached: E_synth_vareg.npy, y_synth_vareg.npy")

# Cache generator state dict
torch.save(gen_best.state_dict(), 'generator_vareg_best.pt')
print("Cached: generator_vareg_best.pt")

# Cache sweep results
import json as json_lib
sweep_results_serialisable = []
for r in sweep_results:
    sr = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
          for k, v in r.items()}
    sweep_results_serialisable.append(sr)

with open('var_reg_sweep_results.json', 'w') as f:
    json_lib.dump({
        'alpha_0': alpha_0,
        'calibration': cal_info,
        'sweep_results': sweep_results_serialisable,
        'best_idx': best_idx,
        'baseline': {
            'H': 0.0477, 'AccS': 0.0411, 'AccU': 0.0569,
            'VarR': 0.872, 'rho_sp': 0.8572, 'rho_sr': 0.5875,
        }
    }, f, indent=2)
print("Cached: var_reg_sweep_results.json")

print(f"\n{'=' * 60}")
print("VARIANCE REGULARISATION EXPERIMENT COMPLETE")
print(f"{'=' * 60}")
print(f"  Best alpha: {best_alpha:.6f} ({best_mult:.2f}x alpha_0)")
print(f"  H-mean:     {best['H']*100:.2f}% (baseline: 4.77%, "
      f"delta: {(best['H']-0.0477)*100:+.2f}pp)")
print(f"  VarR:       {best['var_ratio']:.3f} (baseline: 0.872, "
      f"delta: {best['var_ratio']-0.872:+.3f})")
print(f"  rho_sp:     {best['rho_synth_proto']:.4f} (baseline: 0.8572, "
      f"delta: {best['rho_synth_proto']-0.8572:+.4f})")
print(f"{'=' * 60}")'''


# =============================================================================
# MAIN: READ NOTEBOOK, BACKUP, APPEND CELLS, WRITE
# =============================================================================

def main():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"ERROR: {NOTEBOOK_PATH} not found. Run from project root.")
        return

    # Read notebook
    with open(NOTEBOOK_PATH) as f:
        nb = json.load(f)

    n_before = len(nb['cells'])
    print(f"Read {NOTEBOOK_PATH}: {n_before} cells")

    if n_before != 77:
        print(f"WARNING: Expected 77 cells, found {n_before}. "
              f"New cells will be appended at index {n_before}.")

    # Create timestamped backup
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{NOTEBOOK_PATH}.backup.var_reg.{ts}"
    shutil.copy2(NOTEBOOK_PATH, backup)
    print(f"Backup: {backup}")

    # Build cells
    new_cells = [
        fix_source_lines(make_markdown_cell(CELL_77_MARKDOWN)),
        fix_source_lines(make_code_cell(CELL_78_CONFIG_FUNCTIONS)),
        fix_source_lines(make_code_cell(CELL_79_VAR_TARGET)),
        fix_source_lines(make_code_cell(CELL_80_CALIBRATION)),
        fix_source_lines(make_code_cell(CELL_81_ALPHA_SWEEP)),
        fix_source_lines(make_code_cell(CELL_82_ANALYSIS)),
        fix_source_lines(make_code_cell(CELL_83_CACHE)),
    ]

    # Append
    nb['cells'].extend(new_cells)
    n_after = len(nb['cells'])
    print(f"Appended {len(new_cells)} cells: {n_before}-{n_after-1}")

    # Write
    with open(NOTEBOOK_PATH, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote {NOTEBOOK_PATH}: {n_after} cells")

    # Summary
    print(f"\nNew cells:")
    print(f"  [{n_before}] markdown — Variance Regularisation section header")
    print(f"  [{n_before+1}] code    — Config + function definitions")
    print(f"  [{n_before+2}] code    — var_target + correctness verification")
    print(f"  [{n_before+3}] code    — Alpha calibration")
    print(f"  [{n_before+4}] code    — Alpha sweep (7 values)")
    print(f"  [{n_before+5}] code    — Analysis + figures")
    print(f"  [{n_before+6}] code    — Cache best-alpha results")


if __name__ == "__main__":
    main()
