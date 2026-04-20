#!/usr/bin/env python3
"""
Add noise-augmented prototype conditioning sweep cells to GZSL_EEG_Pipeline_v2.ipynb.

Adds 6 cells (71-76) after the existing 71 cells (0-70):
  Cell 71: Markdown — section header
  Cell 72: Config + utility functions (perturb, synthesise, evaluate)
  Cell 73: Train encoder + WGAN-GP with optimal config
  Cell 74: eta sweep on synthesis only
  Cell 75: Analysis + figures
  Cell 76: Cache best-eta results

Reuses class definitions from cells 30, 35, 40, 41, 63.
Data tensors from cell 31 (embed_dim-independent).

Run from project root:
  python helper_files/add_perturbation_sweep.py
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

CELL_71_MARKDOWN = """---

# Noise-Augmented Prototype Conditioning: eta Sweep

**Motivation**: Diagnostic 3 revealed that synthetic embeddings preserve prototype geometry
too faithfully (Spearman rho=0.893) compared to real unseen brain embeddings (rho=0.668).
The classifier trains on geometrically "too clean" synthetic data, causing a structural
mismatch at test time.

**Mechanism**: Perturb unseen prototypes during synthesis with calibrated Gaussian noise:
```
s_tilde = normalize(s_c + eta * xi),  xi ~ N(0, I_d)
```
Each synthetic sample gets an independent perturbation, simultaneously:
1. Degrading inter-class geometric fidelity (rho toward 0.668)
2. Increasing within-class variance (each sample conditioned on a slightly different point)

**Strategy**: Train encoder + WGAN-GP once with optimal config (embed_dim=64, tau=0.05).
Then sweep eta on synthesis only (~30s per eta value). The generator's learned smooth
mapping means perturbed inputs produce well-behaved outputs."""

CELL_72_CONFIG_UTILS = r'''# =============================================================================
# OPTIMAL CONFIG + PERTURBATION UTILITIES
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import math
import gc

# ---- Optimal config (validated by sweep + dim sweep) ----
OPT_ENCODER_CONFIG = {
    'embed_dim': 64,
    'image_input_dim': 1000,
    'tau': 0.05,
    'epochs': 75,
    'batch_size': 256,
    'lr': 2e-3,
    'weight_decay': 1e-4,
    'dropout': 0.1,
    'warmup_ratio': 0.1,
    'seed': 42,
}

OPT_WGAN_CONFIG = {
    'z_dim': 100,
    'embed_dim': 64,
    'lr': 1e-4,
    'betas': (0.0, 0.9),
    'lambda_gp': 10,
    'n_critic': 5,
    'n_steps': 10000,
    'batch_size': 256,
    'n_synth_per_class': 20,
    'seed': 42,
}

# Dense sweep near small eta + wider coverage
ETA_VALUES = [0.0, 0.02, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25]

SEED = 42

print("=" * 60)
print("NOISE-AUGMENTED PROTOTYPE CONDITIONING")
print("=" * 60)
print(f"  Encoder: embed_dim={OPT_ENCODER_CONFIG['embed_dim']}, tau={OPT_ENCODER_CONFIG['tau']}, "
      f"epochs={OPT_ENCODER_CONFIG['epochs']}, lr={OPT_ENCODER_CONFIG['lr']}")
print(f"  WGAN: z_dim={OPT_WGAN_CONFIG['z_dim']}, steps={OPT_WGAN_CONFIG['n_steps']}, "
      f"n_synth={OPT_WGAN_CONFIG['n_synth_per_class']}")
print(f"  eta sweep: {ETA_VALUES}")
d = OPT_ENCODER_CONFIG['embed_dim']
print(f"\n  Expected angular deviations (d={d}):")
for eta in ETA_VALUES:
    if eta == 0:
        print(f"    eta={eta:.2f}: no perturbation (baseline)")
    else:
        cos_theta = 1.0 / math.sqrt(1 + eta**2 * (d - 1))
        angle = math.degrees(math.acos(cos_theta))
        print(f"    eta={eta:.2f}: E[cos theta]={cos_theta:.3f}, angle~{angle:.1f} deg")
print("=" * 60)


# ---- Perturbation function ----
def perturb_prototype(s_c, eta, device):
    """
    Apply Gaussian perturbation to prototype and renormalize to unit sphere.
    s_c: (n, d) tensor of prototypes
    eta: perturbation magnitude (0 = no perturbation)
    Returns: (n, d) tensor on unit sphere
    """
    if eta == 0 or eta is None:
        return s_c
    xi = torch.randn_like(s_c, device=device)
    s_perturbed = s_c + eta * xi
    return F.normalize(s_perturbed, p=2, dim=-1)


# ---- Synthesis with perturbation ----
def synthesise_with_eta(generator, unseen_classes, S_unseen_protos, n_synth, z_dim, eta, device):
    """
    Generate synthetic unseen embeddings with noise-augmented prototype conditioning.
    Each of n_synth samples per class gets an independent perturbation xi_k.
    When eta=0, reproduces original synthesis exactly.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    generator.eval()
    synth_embeddings = []
    synth_labels = []

    with torch.no_grad():
        for c in unseen_classes:
            s_c = torch.FloatTensor(S_unseen_protos[c]).unsqueeze(0).repeat(n_synth, 1).to(device)
            # Apply per-sample perturbation
            s_c_perturbed = perturb_prototype(s_c, eta, device)
            z = torch.randn(n_synth, z_dim, device=device)
            e_synth = generator(z, s_c_perturbed).cpu().numpy()
            synth_embeddings.append(e_synth)
            synth_labels.extend([c] * n_synth)

    E_synth = np.vstack(synth_embeddings)
    y_synth = np.array(synth_labels)
    return E_synth, y_synth


# ---- Diagnostic metrics ----
def compute_sweep_diagnostics(E_synth, y_synth, E_unseen, y_unseen,
                               S_unseen_array, unseen_classes):
    """
    Compute diagnostic metrics for one eta value.
    Returns dict with rho_synth_proto, rho_synth_real, knn_10, var_ratio.
    """
    n_classes = len(unseen_classes)

    # Synth centroids
    synth_centroids = []
    synth_class_vars = []
    for c in unseen_classes:
        mask = y_synth == c
        embeds = E_synth[mask]
        centroid = embeds.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        synth_centroids.append(centroid)
        if len(embeds) > 1:
            synth_class_vars.append(np.var(embeds, axis=0).sum())
    synth_centroids = np.array(synth_centroids)

    # Real unseen centroids
    real_centroids = []
    real_class_vars = []
    for c in unseen_classes:
        mask = y_unseen == c
        embeds = E_unseen[mask]
        centroid = embeds.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        real_centroids.append(centroid)
        if len(embeds) > 1:
            real_class_vars.append(np.var(embeds, axis=0).sum())
    real_centroids = np.array(real_centroids)

    # Pairwise distances
    proto_dists = squareform(pdist(S_unseen_array, metric='cosine'))
    synth_dists = squareform(pdist(synth_centroids, metric='cosine'))
    real_dists = squareform(pdist(real_centroids, metric='cosine'))

    triu = np.triu_indices(n_classes, k=1)
    proto_flat = proto_dists[triu]
    synth_flat = synth_dists[triu]
    real_flat = real_dists[triu]

    rho_synth_proto, _ = spearmanr(proto_flat, synth_flat)
    rho_synth_real, _ = spearmanr(real_flat, synth_flat)
    rho_real_proto, _ = spearmanr(proto_flat, real_flat)

    # k-NN preservation at k=10
    k = 10
    matches = 0
    for i in range(n_classes):
        proto_nn = set(np.argsort(proto_dists[i])[1:k+1])
        synth_nn = set(np.argsort(synth_dists[i])[1:k+1])
        matches += len(proto_nn & synth_nn)
    knn_10 = matches / (n_classes * k)

    # Within-class variance ratio (synth / real unseen)
    mean_synth_var = np.mean(synth_class_vars) if synth_class_vars else 0
    mean_real_var = np.mean(real_class_vars) if real_class_vars else 1
    var_ratio = mean_synth_var / (mean_real_var + 1e-10)

    return {
        'rho_synth_proto': rho_synth_proto,
        'rho_synth_real': rho_synth_real,
        'rho_real_proto': rho_real_proto,
        'knn_10': knn_10,
        'var_ratio': var_ratio,
        'mean_synth_var': mean_synth_var,
        'mean_real_var': mean_real_var,
    }


# ---- Full evaluation for one eta ----
def evaluate_eta(eta, generator, E_train, y_train, E_test, y_test,
                 E_unseen, y_unseen, S_unseen_protos, S_unseen_array,
                 unseen_classes, seen_labels_set, unseen_labels_set,
                 median_per_class, device):
    """
    Full pipeline for one eta: synthesise -> balance -> classify -> evaluate -> diagnose.
    """
    z_dim = OPT_WGAN_CONFIG['z_dim']
    n_synth = OPT_WGAN_CONFIG['n_synth_per_class']

    # 1. Synthesise with perturbation
    E_synth, y_synth = synthesise_with_eta(
        generator, unseen_classes, S_unseen_protos, n_synth, z_dim, eta, device)

    # 2. Sample balance: downsample to median_per_class
    rng = np.random.RandomState(SEED)
    ds_indices = []
    for c in unseen_classes:
        idx = np.where(y_synth == c)[0]
        if len(idx) > median_per_class:
            selected = rng.choice(idx, size=median_per_class, replace=False)
        else:
            selected = idx
        ds_indices.append(selected)
    ds_indices = np.concatenate(ds_indices)
    E_synth_ds = E_synth[ds_indices]
    y_synth_ds = y_synth[ds_indices]

    # 3. Combine + train classifier
    X_train_gzsl = np.vstack([E_train, E_synth_ds])
    y_train_gzsl = np.concatenate([y_train, y_synth_ds])

    clf = LogisticRegression(
        multi_class='multinomial', solver='lbfgs', max_iter=1000,
        class_weight='balanced', random_state=SEED, n_jobs=-1)
    clf.fit(X_train_gzsl, y_train_gzsl)

    # 4. GZSL evaluation
    pred_seen = clf.predict(E_test)
    pred_unseen = clf.predict(E_unseen)
    acc_seen = accuracy_score(y_test, pred_seen)
    acc_unseen = accuracy_score(y_unseen, pred_unseen)
    H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen) if (acc_seen + acc_unseen) > 0 else 0.0

    seen_as_unseen = np.isin(pred_seen, list(unseen_labels_set)).sum()
    routing_rate = seen_as_unseen / len(pred_seen)

    # 5. Diagnostic metrics
    diag = compute_sweep_diagnostics(
        E_synth, y_synth, E_unseen, y_unseen, S_unseen_array, unseen_classes)

    return {
        'eta': eta,
        'acc_seen': acc_seen,
        'acc_unseen': acc_unseen,
        'H': H,
        'routing_rate': routing_rate,
        'n_synth_ds': len(E_synth_ds),
        **diag,
    }


print("Utility functions defined: perturb_prototype, synthesise_with_eta, "
      "compute_sweep_diagnostics, evaluate_eta")'''

CELL_73_TRAIN = r'''# =============================================================================
# TRAIN ENCODER + WGAN-GP WITH OPTIMAL CONFIG
# =============================================================================
# Single training run. The trained generator is then reused for all eta values.

import time
t_start = time.time()

# ---- Set seeds ----
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

d = OPT_ENCODER_CONFIG['embed_dim']

# ---- Instantiate models ----
brain_encoder_opt = BrainEncoder(
    input_dim=561, embed_dim=d, dropout=OPT_ENCODER_CONFIG['dropout']
).to(device)

image_proj_opt = ImageProjector(
    input_dim=OPT_ENCODER_CONFIG['image_input_dim'],
    hidden_dim=512, embed_dim=d, dropout=OPT_ENCODER_CONFIG['dropout']
).to(device)

print(f"Encoder: BrainEncoder({sum(p.numel() for p in brain_encoder_opt.parameters()):,} params) "
      f"+ ImageProjector({sum(p.numel() for p in image_proj_opt.parameters()):,} params)")

# ---- Encoder training ----
optimizer_enc = torch.optim.AdamW(
    list(brain_encoder_opt.parameters()) + list(image_proj_opt.parameters()),
    lr=OPT_ENCODER_CONFIG['lr'], weight_decay=OPT_ENCODER_CONFIG['weight_decay']
)

total_steps = OPT_ENCODER_CONFIG['epochs'] * len(train_loader)
warmup_steps = int(total_steps * OPT_ENCODER_CONFIG['warmup_ratio'])

def lr_lambda(step):
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler_enc = torch.optim.lr_scheduler.LambdaLR(optimizer_enc, lr_lambda)

print(f"\nTraining contrastive encoder for {OPT_ENCODER_CONFIG['epochs']} epochs "
      f"(tau={OPT_ENCODER_CONFIG['tau']}, lr={OPT_ENCODER_CONFIG['lr']}, d={d})...")

enc_losses = []
for epoch in range(OPT_ENCODER_CONFIG['epochs']):
    brain_encoder_opt.train()
    image_proj_opt.train()
    epoch_loss = 0
    n_batches = 0
    for X_batch, I_batch, Y_batch in train_loader:
        X_batch = X_batch.to(device)
        I_batch = I_batch.to(device)
        brain_emb = brain_encoder_opt(X_batch)
        image_emb = image_proj_opt(I_batch)
        loss = contrastive_loss(brain_emb, image_emb, tau=OPT_ENCODER_CONFIG['tau'])
        optimizer_enc.zero_grad()
        loss.backward()
        optimizer_enc.step()
        scheduler_enc.step()
        epoch_loss += loss.item()
        n_batches += 1
    avg_loss = epoch_loss / n_batches
    enc_losses.append(avg_loss)
    if (epoch + 1) % 25 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:3d}/{OPT_ENCODER_CONFIG['epochs']}: loss={avg_loss:.4f}")

print(f"Encoder training done. Final loss: {enc_losses[-1]:.4f}")

# ---- Compute embeddings ----
brain_encoder_opt.eval()
image_proj_opt.eval()
with torch.no_grad():
    E_train_opt = brain_encoder_opt(X_train_tensor.to(device)).cpu().numpy()
    E_test_opt = brain_encoder_opt(X_test_tensor.to(device)).cpu().numpy()
    E_unseen_opt = brain_encoder_opt(X_unseen_tensor.to(device)).cpu().numpy()
    V_train_opt = image_proj_opt(I_train_tensor.to(device)).cpu().numpy()
    V_unseen_opt = image_proj_opt(I_unseen_tensor.to(device)).cpu().numpy()

y_train_opt = label_train_seen.copy()
y_test_opt = label_test_seen.copy()
y_unseen_opt = Y_unseen_tensor.numpy().flatten()

print(f"Embeddings: E_train={E_train_opt.shape}, E_test={E_test_opt.shape}, E_unseen={E_unseen_opt.shape}")

# ---- Compute prototypes ----
S_seen_opt = compute_prototypes(V_train_opt, y_train_opt)
S_unseen_opt = compute_prototypes(V_unseen_opt, y_unseen_opt)
seen_classes_opt = sorted(S_seen_opt.keys())
unseen_classes_opt = sorted(S_unseen_opt.keys())
S_seen_arr_opt = np.array([S_seen_opt[c] for c in seen_classes_opt])
S_unseen_arr_opt = np.array([S_unseen_opt[c] for c in unseen_classes_opt])
print(f"Prototypes: {len(seen_classes_opt)} seen, {len(unseen_classes_opt)} unseen")

# ---- Compute seen per-class median (for sample balancing) ----
_, seen_counts_opt = np.unique(y_train_opt, return_counts=True)
median_per_class_opt = int(np.median(seen_counts_opt))
print(f"Seen per-class median: {median_per_class_opt}")

# Label sets for GZSL evaluation
seen_labels_set_opt = set(seen_classes_opt)
unseen_labels_set_opt = set(unseen_classes_opt)

# ---- Train WGAN-GP ----
print(f"\nTraining cWGAN-GP ({OPT_WGAN_CONFIG['n_steps']} gen steps, embed_dim={d})...")

generator_opt = Generator(z_dim=OPT_WGAN_CONFIG['z_dim'], embed_dim=d).to(device)
critic_opt = Critic(embed_dim=d).to(device)

g_opt = torch.optim.Adam(generator_opt.parameters(), lr=OPT_WGAN_CONFIG['lr'],
                          betas=OPT_WGAN_CONFIG['betas'])
c_opt_adam = torch.optim.Adam(critic_opt.parameters(), lr=OPT_WGAN_CONFIG['lr'],
                               betas=OPT_WGAN_CONFIG['betas'])

E_train_t = torch.FloatTensor(E_train_opt)
S_train_cond = torch.FloatTensor(get_prototype_for_labels(y_train_opt, S_seen_opt))
wgan_ds = TensorDataset(E_train_t, S_train_cond)
wgan_dl = DataLoader(wgan_ds, batch_size=OPT_WGAN_CONFIG['batch_size'], shuffle=True, drop_last=True)

data_iter_opt = iter(wgan_dl)
g_step = 0
g_losses_opt, c_losses_opt = [], []

while g_step < OPT_WGAN_CONFIG['n_steps']:
    # Critic
    for _ in range(OPT_WGAN_CONFIG['n_critic']):
        try:
            e_real, s_c = next(data_iter_opt)
        except StopIteration:
            data_iter_opt = iter(wgan_dl)
            e_real, s_c = next(data_iter_opt)
        e_real, s_c = e_real.to(device), s_c.to(device)
        bs = e_real.size(0)
        z = torch.randn(bs, OPT_WGAN_CONFIG['z_dim'], device=device)
        e_fake = generator_opt(z, s_c)
        d_real = critic_opt(e_real, s_c)
        d_fake = critic_opt(e_fake.detach(), s_c)
        gp = compute_gradient_penalty(critic_opt, e_real, e_fake.detach(), s_c, device)
        c_loss = -d_real.mean() + d_fake.mean() + OPT_WGAN_CONFIG['lambda_gp'] * gp
        c_opt_adam.zero_grad()
        c_loss.backward()
        c_opt_adam.step()

    # Generator
    try:
        e_real, s_c = next(data_iter_opt)
    except StopIteration:
        data_iter_opt = iter(wgan_dl)
        e_real, s_c = next(data_iter_opt)
    s_c = s_c.to(device)
    bs = s_c.size(0)
    z = torch.randn(bs, OPT_WGAN_CONFIG['z_dim'], device=device)
    e_fake = generator_opt(z, s_c)
    d_fake = critic_opt(e_fake, s_c)
    g_loss = -d_fake.mean()
    g_opt.zero_grad()
    g_loss.backward()
    g_opt.step()

    g_losses_opt.append(g_loss.item())
    c_losses_opt.append(c_loss.item())
    g_step += 1
    if g_step % 2500 == 0 or g_step == 1:
        print(f"  Step {g_step:5d}/{OPT_WGAN_CONFIG['n_steps']}: G={g_loss.item():.4f}, C={c_loss.item():.4f}")

t_train = time.time() - t_start
print(f"\nTraining complete in {t_train:.0f}s. G_loss={g_losses_opt[-1]:.4f}, C_loss={c_losses_opt[-1]:.4f}")
print(f"Generator params: {sum(p.numel() for p in generator_opt.parameters()):,}")'''

CELL_74_SWEEP = r'''# =============================================================================
# ETA SWEEP: SYNTHESIS-ONLY PERTURBATION
# =============================================================================
# Train once (cell 73), sweep eta on synthesis only.
# Each eta: synthesise -> balance -> classify -> evaluate GZSL -> diagnostics.

import time

print(f"Running eta sweep: {ETA_VALUES}")
print(f"{'='*90}")
print(f"{'eta':>6s}  {'AccS':>7s}  {'AccU':>7s}  {'H':>7s}  {'Route':>7s}  "
      f"{'rho_sp':>7s}  {'rho_sr':>7s}  {'kNN@10':>7s}  {'VarR':>7s}")
print(f"{'─'*90}")

sweep_results = []

for eta in ETA_VALUES:
    t0 = time.time()
    result = evaluate_eta(
        eta=eta,
        generator=generator_opt,
        E_train=E_train_opt, y_train=y_train_opt,
        E_test=E_test_opt, y_test=y_test_opt,
        E_unseen=E_unseen_opt, y_unseen=y_unseen_opt,
        S_unseen_protos=S_unseen_opt,
        S_unseen_array=S_unseen_arr_opt,
        unseen_classes=unseen_classes_opt,
        seen_labels_set=seen_labels_set_opt,
        unseen_labels_set=unseen_labels_set_opt,
        median_per_class=median_per_class_opt,
        device=device,
    )
    dt = time.time() - t0
    sweep_results.append(result)

    print(f"{eta:6.2f}  {result['acc_seen']*100:6.2f}%  {result['acc_unseen']*100:6.2f}%  "
          f"{result['H']*100:6.2f}%  {result['routing_rate']*100:5.1f}%  "
          f"{result['rho_synth_proto']:7.4f}  {result['rho_synth_real']:7.4f}  "
          f"{result['knn_10']:7.4f}  {result['var_ratio']:6.3f}  ({dt:.1f}s)")

print(f"{'='*90}")

# Find best eta by H-mean
best_result = max(sweep_results, key=lambda r: r['H'])
baseline_result = sweep_results[0]  # eta=0

print(f"\nBest eta: {best_result['eta']:.2f}")
print(f"  H-mean: {best_result['H']*100:.2f}% (baseline: {baseline_result['H']*100:.2f}%, "
      f"delta: {(best_result['H'] - baseline_result['H'])*100:+.2f}pp)")
print(f"  AccS:   {best_result['acc_seen']*100:.2f}% (baseline: {baseline_result['acc_seen']*100:.2f}%)")
print(f"  AccU:   {best_result['acc_unseen']*100:.2f}% (baseline: {baseline_result['acc_unseen']*100:.2f}%)")
print(f"  rho(synth,proto): {best_result['rho_synth_proto']:.4f} (baseline: {baseline_result['rho_synth_proto']:.4f})")
print(f"  rho(synth,real):  {best_result['rho_synth_real']:.4f} (baseline: {baseline_result['rho_synth_real']:.4f})")
print(f"  kNN@10: {best_result['knn_10']:.4f} (baseline: {baseline_result['knn_10']:.4f})")
print(f"  VarRatio: {best_result['var_ratio']:.4f} (baseline: {baseline_result['var_ratio']:.4f})")

best_eta = best_result['eta']'''

CELL_75_ANALYSIS = r'''# =============================================================================
# ANALYSIS + FIGURES
# =============================================================================

import matplotlib.pyplot as plt
import os
os.makedirs('figures', exist_ok=True)

etas = [r['eta'] for r in sweep_results]
Hs = [r['H'] * 100 for r in sweep_results]
AccSs = [r['acc_seen'] * 100 for r in sweep_results]
AccUs = [r['acc_unseen'] * 100 for r in sweep_results]
rho_sps = [r['rho_synth_proto'] for r in sweep_results]
rho_srs = [r['rho_synth_real'] for r in sweep_results]
knn10s = [r['knn_10'] for r in sweep_results]
var_ratios = [r['var_ratio'] for r in sweep_results]
routing = [r['routing_rate'] * 100 for r in sweep_results]

# Reference: real unseen rho to proto (from diagnostic 3)
rho_real_proto = sweep_results[0]['rho_real_proto']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: H-mean, AccS, AccU vs eta
ax = axes[0, 0]
ax.plot(etas, Hs, 'ko-', linewidth=2, markersize=8, label='H-mean', zorder=3)
ax.plot(etas, AccSs, 'b^--', linewidth=1, markersize=6, alpha=0.7, label='AccS')
ax.plot(etas, AccUs, 'rs--', linewidth=1, markersize=6, alpha=0.7, label='AccU')
best_idx = Hs.index(max(Hs))
ax.scatter([etas[best_idx]], [Hs[best_idx]], s=200, c='gold', edgecolors='k',
           zorder=4, label=f'Best (eta={etas[best_idx]:.2f})')
ax.set_xlabel('eta (perturbation magnitude)')
ax.set_ylabel('Accuracy (%)')
ax.set_title('GZSL Performance vs eta')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: rho vs eta
ax = axes[0, 1]
ax.plot(etas, rho_sps, 'ro-', linewidth=2, markersize=7, label='rho(synth, proto)')
ax.plot(etas, rho_srs, 'gs-', linewidth=2, markersize=7, label='rho(synth, real)')
ax.axhline(y=rho_real_proto, color='green', linestyle=':', alpha=0.5,
           label=f'rho(real, proto)={rho_real_proto:.3f} (target)')
ax.set_xlabel('eta')
ax.set_ylabel('Spearman rho')
ax.set_title('Geometric Coupling vs eta')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3: kNN preservation + routing rate
ax = axes[1, 0]
ax.plot(etas, knn10s, 'p-', color='purple', linewidth=2, markersize=7, label='kNN@10 preservation')
ax2 = ax.twinx()
ax2.plot(etas, routing, 'c--', linewidth=1.5, markersize=5, alpha=0.7, label='Routing rate (%)')
ax.set_xlabel('eta')
ax.set_ylabel('kNN@10 preservation', color='purple')
ax2.set_ylabel('Routing rate (%)', color='cyan')
ax.set_title('Neighbourhood Preservation & Routing')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='center left')
ax.grid(True, alpha=0.3)

# Panel 4: Within-class variance ratio
ax = axes[1, 1]
ax.bar(range(len(etas)), var_ratios, color=['steelblue' if e != best_eta else 'gold' for e in etas],
       alpha=0.7, edgecolor='k')
ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Perfect match (1.0)')
ax.set_xticks(range(len(etas)))
ax.set_xticklabels([f'{e:.2f}' for e in etas], rotation=45)
ax.set_xlabel('eta')
ax.set_ylabel('Within-class var ratio (synth/real)')
ax.set_title('Within-Class Variance Ratio')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('Noise-Augmented Prototype Conditioning: eta Sweep', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/eta_sweep_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figures/eta_sweep_results.png")

# ---- Print comprehensive summary ----
print("\n" + "=" * 70)
print("SWEEP SUMMARY")
print("=" * 70)
print(f"\n{'eta':>6s}  {'H':>7s}  {'AccS':>7s}  {'AccU':>7s}  {'Route':>6s}  "
      f"{'rho_sp':>7s}  {'rho_sr':>7s}  {'kNN10':>6s}  {'VarR':>6s}")
print(f"{'─'*70}")
for r in sweep_results:
    marker = " <-- BEST" if r['eta'] == best_eta else ""
    print(f"{r['eta']:6.2f}  {r['H']*100:6.2f}%  {r['acc_seen']*100:6.2f}%  "
          f"{r['acc_unseen']*100:6.2f}%  {r['routing_rate']*100:5.1f}%  "
          f"{r['rho_synth_proto']:7.4f}  {r['rho_synth_real']:7.4f}  "
          f"{r['knn_10']:6.4f}  {r['var_ratio']:5.3f}{marker}")
print(f"{'='*70}")

delta_H = (best_result['H'] - baseline_result['H']) * 100
if delta_H > 0.1:
    print(f"\nVerdict: Perturbation at eta={best_eta:.2f} improved H-mean by {delta_H:+.2f}pp")
elif delta_H < -0.1:
    print(f"\nVerdict: Perturbation degraded H-mean. Best eta=0 (no perturbation).")
else:
    print(f"\nVerdict: Perturbation had negligible effect on H-mean ({delta_H:+.2f}pp).")
    print("  The structural mismatch may not be the dominant bottleneck, or")
    print("  synthesis-only perturbation may be insufficient — consider training-time perturbation.")'''

CELL_76_CACHE = r'''# =============================================================================
# CACHE BEST-ETA RESULTS
# =============================================================================

os.makedirs('cached_arrays', exist_ok=True)

# Regenerate best-eta synthetics for caching
E_synth_best, y_synth_best = synthesise_with_eta(
    generator_opt, unseen_classes_opt, S_unseen_opt,
    OPT_WGAN_CONFIG['n_synth_per_class'], OPT_WGAN_CONFIG['z_dim'],
    best_eta, device)

# Cache
np.save('cached_arrays/E_synth_unseen_eta.npy', E_synth_best)
np.save('cached_arrays/y_synth_unseen_eta.npy', y_synth_best)
np.save('cached_arrays/E_train_opt.npy', E_train_opt)
np.save('cached_arrays/E_test_opt.npy', E_test_opt)
np.save('cached_arrays/E_unseen_opt.npy', E_unseen_opt)

print(f"Cached best-eta (eta={best_eta:.2f}) arrays:")
print(f"  E_synth_unseen_eta: {E_synth_best.shape}")
print(f"  y_synth_unseen_eta: {y_synth_best.shape}")
print(f"  E_train_opt: {E_train_opt.shape}")
print(f"  E_test_opt: {E_test_opt.shape}")
print(f"  E_unseen_opt: {E_unseen_opt.shape}")

print(f"\nFinal config:")
print(f"  Encoder: embed_dim={OPT_ENCODER_CONFIG['embed_dim']}, tau={OPT_ENCODER_CONFIG['tau']}, "
      f"epochs={OPT_ENCODER_CONFIG['epochs']}, lr={OPT_ENCODER_CONFIG['lr']}")
print(f"  WGAN-GP: z_dim={OPT_WGAN_CONFIG['z_dim']}, steps={OPT_WGAN_CONFIG['n_steps']}")
print(f"  Perturbation: eta={best_eta:.2f}")
print(f"  Best H-mean: {best_result['H']*100:.2f}% "
      f"(AccS={best_result['acc_seen']*100:.2f}%, AccU={best_result['acc_unseen']*100:.2f}%)")'''

# =============================================================================
# ASSEMBLE AND INJECT
# =============================================================================

def main():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"ERROR: {NOTEBOOK_PATH} not found. Run from project root.")
        return

    # Backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{NOTEBOOK_PATH}.backup.eta_sweep.{timestamp}"
    shutil.copy2(NOTEBOOK_PATH, backup_path)
    print(f"Backup: {backup_path}")

    with open(NOTEBOOK_PATH, "r") as f:
        nb = json.load(f)

    n_before = len(nb["cells"])
    print(f"Cells before: {n_before}")

    if n_before != 71:
        print(f"WARNING: Expected 71 cells, found {n_before}. Proceeding anyway.")

    new_cells = [
        fix_source_lines(make_markdown_cell(CELL_71_MARKDOWN)),
        fix_source_lines(make_code_cell(CELL_72_CONFIG_UTILS)),
        fix_source_lines(make_code_cell(CELL_73_TRAIN)),
        fix_source_lines(make_code_cell(CELL_74_SWEEP)),
        fix_source_lines(make_code_cell(CELL_75_ANALYSIS)),
        fix_source_lines(make_code_cell(CELL_76_CACHE)),
    ]

    nb["cells"].extend(new_cells)

    n_after = len(nb["cells"])
    print(f"Cells after: {n_after}")
    print(f"Added {n_after - n_before} cells ({n_before}→{n_after - 1})")

    with open(NOTEBOOK_PATH, "w") as f:
        json.dump(nb, f, indent=1)

    print(f"\nDone! Notebook updated: {NOTEBOOK_PATH}")
    print("New cells:")
    print("  Cell 71: [markdown] Section header — noise-augmented prototype conditioning")
    print("  Cell 72: [code] Config + utility functions")
    print("  Cell 73: [code] Train encoder + WGAN-GP with optimal config")
    print("  Cell 74: [code] eta sweep (synthesis-only perturbation)")
    print("  Cell 75: [code] Analysis + figures")
    print("  Cell 76: [code] Cache best-eta results")

if __name__ == "__main__":
    main()
