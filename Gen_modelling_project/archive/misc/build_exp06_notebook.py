"""
Build experiments/06_residual_film/notebook.ipynb from Exp05 notebook.

Changes vs experiments/05_high_capacity_film/notebook.ipynb:
  Cell 41: WGAN_CONFIG — film_hidden=128, gamma_min=0.5, lambda_gamma=0.1, experiment='film_residual'
  Cell 42: FiLMGenerator — residual _film (γ=1+Δγ), zero-bias init, extended get_film_stats
  Cell 44: Training loop — add compute_lgamma, L_G = L_wass + α·L_var + λ_γ·L_gamma
  Cell 50: FiLM param analysis — effective γ = 1+Δγ, fix EXP_FIG_DIR, add γ_min reference lines
  Cell 75: Harvest cell — add delta_gamma1_mean, gamma1_min, gamma2_min

Run from project root:
    python3 experiments/06_residual_film/build_exp06_notebook.py
"""

import json, os

BASE = os.path.join(
    os.path.dirname(__file__), '..', '05_high_capacity_film', 'notebook.ipynb'
)
OUT = os.path.join(os.path.dirname(__file__), 'notebook.ipynb')

with open(BASE) as f:
    nb = json.load(f)

def _src(c):
    s = c['source']
    return ''.join(s) if isinstance(s, list) else s

cells = nb['cells']

# ──────────────────────────────────────────────────────────────────────────────
# CELL 41: WGAN_CONFIG
# ──────────────────────────────────────────────────────────────────────────────
CELL_41 = """\
# =============================================================================
# cWGAN-GP CONFIGURATION — Residual FiLM (Exp 06)
# =============================================================================
# Fixes the γ collapse failure mode identified in Exp05 (γ1_mean → 0.16):
#   1. Residual FiLM: γ = 1 + Δγ  (FiLM MLP learns Δγ, not γ directly)
#      - Identity modulation is now a stable fixed point, not a transient init
#      - Collapsing γ to 0.16 requires Δγ → −0.84, which competes with noise-path gradients
#   2. γ floor regularisation: L_γ = λ_γ · E[ReLU(γ_min − γ)²] summed over both FiLM layers
#      - Active precisely when γ < 0.5 — restoring gradient unlike L_var (zero at variance=0)
#   3. film_hidden: 256 → 128 — limits β capacity; Exp05's β1 encoded all class centroids alone

WGAN_CONFIG = {
    'z_dim':             100,
    'embed_dim':         64,             # Must match ENCODER_CONFIG['embed_dim']
    'hidden_dim':        512,            # Generator main path width (keep from Exp05)
    'film_hidden':       128,            # FiLM MLP intermediate dim (reduced from 256)
    'lr':                1e-4,
    'betas':             (0.0, 0.9),
    'lambda_gp':         10,
    'n_critic':          5,
    'n_steps':           50000,          # Keep 5× training budget
    'batch_size':        256,
    'n_synth_per_class': 20,
    'alpha':             1.0,            # L_var weight (sweep optimum from Exp04c)
    'gamma_min':         0.5,            # γ floor threshold for L_γ regularisation
    'lambda_gamma':      0.1,            # L_γ regularisation weight
    'seed':              42,
    'experiment':        'film_residual',
}

print("=" * 65)
print("cWGAN-GP CONFIGURATION (Residual FiLM + L_var + L_gamma)")
print("=" * 65)
for k, v in WGAN_CONFIG.items():
    print(f"  {k}: {v}")
print("=" * 65)
print("  L_G = L_wasserstein + 1.0·L_var + 0.1·L_gamma_floor")
print("  γ = 1 + Δγ  (residual parameterisation — Perez et al. 2018)")
"""

# ──────────────────────────────────────────────────────────────────────────────
# CELL 42: MODEL DEFINITIONS — Residual FiLM
# ──────────────────────────────────────────────────────────────────────────────
CELL_42 = """\
# =============================================================================
# cWGAN-GP MODEL DEFINITIONS — Residual FiLM Generator
# =============================================================================

class FiLMGenerator(nn.Module):
    \"\"\"
    Residual FiLM-conditioned generator.

    Parameterisation: γ = 1 + Δγ  (Δγ is what the FiLM MLP learns)
      h' = (1 + Δγ(s_c)) ⊙ h + β(s_c)

    Why residual: identity modulation (γ=1, β=0) is a stable fixed point.
    Collapsing γ to 0.16 (Exp05 failure) requires Δγ → −0.84, which competes
    against the noise-path gradient from z throughout training.

    Jacobian: J_G = W3 · D2(z,s_c) · diag(γ2) · W2 · D1(z) · diag(γ1) · W1
    D1 class-agnostic as long as γ1 > 0  (guaranteed by L_gamma floor).

    Init: Δγ = 0, β = 0 → γ = 1 at t=0 (all FiLM MLP last-layer biases zeroed).
    \"\"\"
    def __init__(self, z_dim=100, embed_dim=64, hidden_dim=256, film_hidden=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, embed_dim)
        self.film1_mlp = nn.Sequential(
            nn.Linear(embed_dim, film_hidden), nn.ReLU(),
            nn.Linear(film_hidden, hidden_dim * 2),   # -> [Δγ1 | β1]
        )
        self.film2_mlp = nn.Sequential(
            nn.Linear(embed_dim, film_hidden), nn.ReLU(),
            nn.Linear(film_hidden, hidden_dim * 2),   # -> [Δγ2 | β2]
        )
        self._init_film_identity()

    def _init_film_identity(self):
        \"\"\"Zero-init last layer: Δγ=0 → γ=1, β=0 at t=0.\"\"\"
        for mlp in (self.film1_mlp, self.film2_mlp):
            last = mlp[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            # No bias fill — Δγ=0 means γ=1+0=1, β=0. Identity at init.

    def _film(self, h, mlp, s_c):
        \"\"\"Residual FiLM modulation: h' = (1 + Δγ) * h + β.\"\"\"
        params      = mlp(s_c)
        delta_gamma, beta = params.chunk(2, dim=-1)
        return (1.0 + delta_gamma) * h + beta   # γ = 1 + Δγ

    def forward(self, z, s_c):
        h1 = F.leaky_relu(self.fc1(z), 0.2)
        h1 = self._film(h1, self.film1_mlp, s_c)
        h2 = F.leaky_relu(self.fc2(h1), 0.2)
        h2 = self._film(h2, self.film2_mlp, s_c)
        return F.normalize(self.fc3(h2), p=2, dim=-1)

    def get_film_stats(self, s_c):
        \"\"\"Returns effective γ (=1+Δγ) and Δγ stats for WandB monitoring.\"\"\"
        with torch.no_grad():
            p1 = self.film1_mlp(s_c); dg1, b1 = p1.chunk(2, dim=-1); g1 = 1.0 + dg1
            p2 = self.film2_mlp(s_c); dg2, b2 = p2.chunk(2, dim=-1); g2 = 1.0 + dg2
        return {
            'delta_gamma1_mean': dg1.abs().mean().item(),
            'gamma1_mean':       g1.abs().mean().item(),
            'gamma1_min':        g1.min().item(),
            'gamma1_max':        g1.max().item(),
            'beta1_norm':        b1.norm(dim=-1).mean().item(),
            'delta_gamma2_mean': dg2.abs().mean().item(),
            'gamma2_mean':       g2.abs().mean().item(),
            'gamma2_min':        g2.min().item(),
            'gamma2_max':        g2.max().item(),
            'beta2_norm':        b2.norm(dim=-1).mean().item(),
        }

class Critic(nn.Module):
    \"\"\"Conditional critic: [e, s_c] -> scalar score (unchanged, hidden_dim=256).\"\"\"
    def __init__(self, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * embed_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )
    def forward(self, e, s_c):
        return self.net(torch.cat([e, s_c], dim=-1))


def compute_gradient_penalty(critic, real_samples, fake_samples, s_c, device):
    \"\"\"WGAN-GP gradient penalty: E[(||grad D(e_bar)||_2 - 1)^2].\"\"\"
    batch_size = real_samples.size(0)
    eps   = torch.rand(batch_size, 1, device=device)
    e_bar = (eps * real_samples + (1 - eps) * fake_samples).requires_grad_(True)
    d_int = critic(e_bar, s_c)
    grads = torch.autograd.grad(
        outputs=d_int, inputs=e_bar,
        grad_outputs=torch.ones_like(d_int),
        create_graph=True, retain_graph=True,
    )[0].view(batch_size, -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()

generator = FiLMGenerator(
    z_dim       = WGAN_CONFIG['z_dim'],
    embed_dim   = WGAN_CONFIG['embed_dim'],
    hidden_dim  = WGAN_CONFIG['hidden_dim'],    # 512
    film_hidden = WGAN_CONFIG['film_hidden'],   # 128
).to(device)

critic = Critic(embed_dim=WGAN_CONFIG['embed_dim']).to(device)

n_gen  = sum(p.numel() for p in generator.parameters())
n_crit = sum(p.numel() for p in critic.parameters())
print(f"FiLMGenerator (residual, hidden_dim=512, film_hidden=128): {n_gen:,} params")
print(f"Critic (hidden_dim=256):                                    {n_crit:,} params")
print(f"  Exp05 generator was 906,816 params — this is {n_gen/906816:.2f}x (reduced film_hidden)")
print(f"  Exp04 generator was 256,832 params — this is {n_gen/256832:.2f}x (increased hidden_dim)")
"""

# ──────────────────────────────────────────────────────────────────────────────
# CELL 44: TRAINING LOOP — Residual FiLM + L_var + L_gamma
# ──────────────────────────────────────────────────────────────────────────────
CELL_44 = """\
# =============================================================================
# cWGAN-GP TRAINING — Residual FiLM + L_var + L_gamma (50K steps)
# =============================================================================
# L_G = L_wasserstein + alpha·L_var + lambda_gamma·L_gamma
#
# L_gamma = sum_i E_sc[ReLU(gamma_min - gamma_i(s_c))²]  over both FiLM layers
#   - gamma_i = 1 + delta_gamma_i  (residual parameterisation)
#   - Active when gamma < 0.5; gradient = -2·lambda_gamma·(gamma_min - gamma)
#   - Provides restoring force unlike L_var which has zero gradient at variance=0
#
# Gradient conflict monitoring (grad_cos_sim) computed every 5K steps throughout.
# Checkpoints saved every 10K steps.

import copy as _copy
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os


def compute_lvar(e_fake, y_batch):
    \"\"\"L_var = -mean_c Var_within[G(z,s_c)]. Maximises within-class spread.\"\"\"
    classes = y_batch.unique()
    var_per_class = []
    for c in classes:
        mask = (y_batch == c)
        if mask.sum() > 1:
            var_per_class.append(e_fake[mask].var(dim=0).mean())
    if not var_per_class:
        return torch.tensor(0.0, device=e_fake.device)
    return -torch.stack(var_per_class).mean()


def compute_lgamma(gen, s_c):
    \"\"\"L_gamma = sum_i E_sc[ReLU(gamma_min - (1+Δγ_i))²] over both FiLM layers.

    Gradient: ∂L_γ/∂(Δγ) = -2·λ_γ·(γ_min - γ)  when γ < γ_min, else 0.
    Pushes Δγ upward (toward making γ = 1+Δγ ≥ γ_min).
    This gradient is largest when γ is most collapsed — unlike L_var which
    provides zero gradient at the degenerate zero-variance state.
    \"\"\"
    gamma_min = WGAN_CONFIG['gamma_min']
    total = torch.tensor(0.0, device=s_c.device)
    for mlp in (gen.film1_mlp, gen.film2_mlp):
        params       = mlp(s_c)
        delta_gamma, _ = params.chunk(2, dim=-1)
        gamma        = 1.0 + delta_gamma
        # Penalise gamma below gamma_min (0.5). No penalty when gamma >= gamma_min.
        total        = total + F.relu(gamma_min - gamma).pow(2).mean()
    return total


def grad_cosine_sim(model, loss_w, loss_v):
    \"\"\"Cosine similarity between grad(L_wass) and grad(L_var) w.r.t. generator params.\"\"\"
    g_w = torch.autograd.grad(loss_w, model.parameters(),
                               retain_graph=True, create_graph=False, allow_unused=True)
    g_v = torch.autograd.grad(loss_v, model.parameters(),
                               retain_graph=True, create_graph=False, allow_unused=True)
    g_w = torch.cat([g.flatten() for g in g_w if g is not None])
    g_v = torch.cat([g.flatten() for g in g_v if g is not None])
    return F.cosine_similarity(g_w.unsqueeze(0), g_v.unsqueeze(0)).item()


def compute_var_ratio_quick(gen, s_protos, cls_list, e_real, y_real,
                             z_dim, dev, n_cls=50, n_synth=15):
    gen.eval()
    sample_cls = np.random.choice(cls_list, min(n_cls, len(cls_list)), replace=False)
    s_vars, r_vars = [], []
    with torch.no_grad():
        for c in sample_cls:
            sc = torch.FloatTensor(s_protos[c]).unsqueeze(0).repeat(n_synth, 1).to(dev)
            z  = torch.randn(n_synth, z_dim, device=dev)
            es = gen(z, sc).cpu().numpy()
            er = e_real[y_real == c]
            if len(er) > 1:
                s_vars.append(np.var(es, axis=0).mean())
                r_vars.append(np.var(er, axis=0).mean())
    gen.train()
    rmean = np.mean(r_vars) if r_vars else 0.0
    return float(np.mean(s_vars) / rmean) if rmean > 0 else 0.0


def compute_var_ratio_full(synth_emb, synth_lbl, real_emb, real_lbl):
    classes = np.unique(synth_lbl)
    s_vars, r_vars = [], []
    for c in classes:
        se = synth_emb[synth_lbl == c]; re = real_emb[real_lbl == c]
        if len(se) > 1 and len(re) > 1:
            s_vars.append(np.var(se, axis=0).mean())
            r_vars.append(np.var(re, axis=0).mean())
    rmean = np.mean(r_vars) if r_vars else 0.0
    return float(np.mean(s_vars) / rmean) if rmean > 0 else 0.0


# ── Setup ─────────────────────────────────────────────────────────────────────
alpha        = WGAN_CONFIG['alpha']
lambda_gamma = WGAN_CONFIG['lambda_gamma']

torch.manual_seed(WGAN_CONFIG['seed'])
np.random.seed(WGAN_CONFIG['seed'])

g_opt = torch.optim.Adam(generator.parameters(), lr=WGAN_CONFIG['lr'], betas=WGAN_CONFIG['betas'])
c_opt = torch.optim.Adam(critic.parameters(),    lr=WGAN_CONFIG['lr'], betas=WGAN_CONFIG['betas'])

run = init_run(
    experiment_name="06_residual_film",
    config_dict={**ENCODER_CONFIG, **WGAN_CONFIG},
    tags=["exp06", "film-conditioning", "residual-film", "50k-steps",
          "lvar-alpha-1.0", "lgamma-0.1"],
)

os.makedirs('results', exist_ok=True)

g_losses, c_losses, gp_values = [], [], []
data_iter = iter(wgan_loader)
g_step    = 0

print(f"Training: Residual FiLM | hidden_dim={WGAN_CONFIG['hidden_dim']} | "
      f"film_hidden={WGAN_CONFIG['film_hidden']}")
print(f"  alpha={alpha} (L_var)   lambda_gamma={lambda_gamma} (L_gamma, gamma_min={WGAN_CONFIG['gamma_min']})")
print(f"  n_steps={WGAN_CONFIG['n_steps']:,} | L_G = L_wass + {alpha}·L_var + {lambda_gamma}·L_gamma")
print(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")
print(f"Critic params:    {sum(p.numel() for p in critic.parameters()):,}")
print("=" * 70)

# ── Training loop ─────────────────────────────────────────────────────────────
while g_step < WGAN_CONFIG['n_steps']:

    # Critic update
    for _ in range(WGAN_CONFIG['n_critic']):
        try:   e_real, s_c, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(wgan_loader); e_real, s_c, _ = next(data_iter)
        e_real, s_c = e_real.to(device), s_c.to(device)
        batch_size  = e_real.size(0)
        z           = torch.randn(batch_size, WGAN_CONFIG['z_dim'], device=device)
        e_fake      = generator(z, s_c)
        gp          = compute_gradient_penalty(critic, e_real, e_fake.detach(), s_c, device)
        c_loss      = (-critic(e_real, s_c).mean()
                       + critic(e_fake.detach(), s_c).mean()
                       + WGAN_CONFIG['lambda_gp'] * gp)
        c_opt.zero_grad(); c_loss.backward(); c_opt.step()

    # Generator update
    try:   e_real, s_c, y_b = next(data_iter)
    except StopIteration:
        data_iter = iter(wgan_loader); e_real, s_c, y_b = next(data_iter)
    s_c, y_b   = s_c.to(device), y_b.to(device)
    batch_size = s_c.size(0)
    z          = torch.randn(batch_size, WGAN_CONFIG['z_dim'], device=device)
    e_fake     = generator(z, s_c)

    l_wass  = -critic(e_fake, s_c).mean()
    l_var   = compute_lvar(e_fake, y_b)
    l_gamma = compute_lgamma(generator, s_c)      # γ floor regularisation
    g_loss  = l_wass + alpha * l_var + lambda_gamma * l_gamma

    g_opt.zero_grad(); g_loss.backward(); g_opt.step()

    g_losses.append(g_loss.item())
    c_losses.append(c_loss.item())
    gp_values.append(gp.item())
    g_step += 1

    # Logging
    if g_step % 500 == 0 or g_step == 1:
        film_stats  = generator.get_film_stats(s_c)
        log_payload = {
            "train/step":          g_step,
            "train/L_wasserstein": l_wass.item(),
            "train/L_var":         l_var.item(),
            "train/L_gamma":       l_gamma.item(),
            "train/L_G":           g_loss.item(),
            "train/L_D":           c_loss.item(),
            "train/GP":            gp.item(),
            **{f"train/{k}": v for k, v in film_stats.items()},
        }
        extra_str = ""

        if g_step % 1000 == 0:
            var_r = compute_var_ratio_quick(
                generator, S_seen_prototypes, list(seen_classes),
                E_train_seen, label_train_seen, WGAN_CONFIG['z_dim'], device,
            )
            log_payload["train/VarR_seen"] = var_r
            extra_str += f", VarR_seen={var_r:.3f}"

        if g_step % 5000 == 0:
            # Gradient conflict: L_wass vs L_var (separate forward for clean graph)
            e_gs   = generator(torch.randn(batch_size, WGAN_CONFIG['z_dim'], device=device), s_c)
            lw_gs  = -critic(e_gs, s_c).mean()
            lv_gs  = compute_lvar(e_gs, y_b)
            if lv_gs.abs().item() > 1e-6:
                cos = grad_cosine_sim(generator, lw_gs, lv_gs)
                log_payload["train/grad_cos_sim"] = cos
                extra_str += f", grad_cos={cos:.3f}"
            # γ status
            g1_val = film_stats.get('gamma1_mean', 0.0)
            g1_min = film_stats.get('gamma1_min',  0.0)
            extra_str += f", γ1={g1_val:.3f}(min={g1_min:.3f}), Lγ={l_gamma.item():.4f}"
            # Checkpoint
            if g_step % 10000 == 0:
                ckpt_path = f'results/generator_residual_step{g_step}.pt'
                torch.save(generator.state_dict(), ckpt_path)
                extra_str += "  [ckpt]"
            print(f"  step {g_step:6d}/{WGAN_CONFIG['n_steps']:,}: "
                  f"G={g_loss.item():.4f}, Lvar={l_var.item():.4f}{extra_str}")

        wandb.log(log_payload, step=g_step)

# ── Post-training ─────────────────────────────────────────────────────────────
torch.save(generator.state_dict(), 'results/generator_residual_final.pt')
generator.eval()
print("=" * 70)
print(f"Training complete. Final model: results/generator_residual_final.pt")
print(f"Mean G loss (last 1000 steps): {np.mean(g_losses[-1000:]):.4f}")
print(f"Mean D loss (last 1000 steps): {np.mean(c_losses[-1000:]):.4f}")
final_stats = generator.get_film_stats(s_c)
print(f"Final gamma1_mean={final_stats['gamma1_mean']:.4f} "
      f"(min={final_stats['gamma1_min']:.4f})   "
      f"[Exp05 collapsed to 0.160 — if this is > 0.5, fix succeeded]")
print(f"Final gamma2_mean={final_stats['gamma2_mean']:.4f} "
      f"(min={final_stats['gamma2_min']:.4f})")
"""

# ──────────────────────────────────────────────────────────────────────────────
# CELL 50: FiLM PARAMETER ANALYSIS — Residual parameterisation
# ──────────────────────────────────────────────────────────────────────────────
CELL_50 = """\
# =============================================================================
# FILM PARAMETER ANALYSIS: effective gamma/beta (seen vs unseen prototypes)
# =============================================================================
# Residual FiLM: the MLP outputs Δγ; effective γ = 1 + Δγ.
# Key diagnostics:
#   gamma > 0.5  -> noise path active, γ floor regularisation not binding
#   gamma ~ 0.0  -> γ collapsed (Exp05 failure mode) — should not occur with L_gamma
#   gamma ~ 1.0  -> identity modulation (no learned scaling yet / convergence)
# Seen vs unseen overlap -> FiLM MLPs generalise to unseen prototype space.

EXP_FIG_DIR = 'experiments/06_residual_film/figures'
import os; os.makedirs(EXP_FIG_DIR, exist_ok=True)

generator.eval()

seen_pt   = torch.FloatTensor(np.array([S_seen_prototypes[c]   for c in seen_classes])).to(device)
unseen_pt = torch.FloatTensor(np.array([S_unseen_prototypes[c] for c in unseen_classes])).to(device)

with torch.no_grad():
    p1s = generator.film1_mlp(seen_pt);   dg1s, b1s = p1s.chunk(2, dim=-1); g1s = 1.0 + dg1s
    p1u = generator.film1_mlp(unseen_pt); dg1u, b1u = p1u.chunk(2, dim=-1); g1u = 1.0 + dg1u
    p2s = generator.film2_mlp(seen_pt);   dg2s, b2s = p2s.chunk(2, dim=-1); g2s = 1.0 + dg2s
    p2u = generator.film2_mlp(unseen_pt); dg2u, b2u = p2u.chunk(2, dim=-1); g2u = 1.0 + dg2u

def np_flat(t): return t.cpu().numpy().flatten()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Residual FiLM Parameters: Effective γ = 1 + Δγ  (Seen vs Unseen)", fontsize=13)

for ax, gs, gu, lbl in [
    (axes[0, 0], g1s, g1u, "gamma1 = 1 + Δγ1"),
    (axes[0, 1], b1s, b1u, "beta1"),
    (axes[1, 0], g2s, g2u, "gamma2 = 1 + Δγ2"),
    (axes[1, 1], b2s, b2u, "beta2"),
]:
    ax.hist(np_flat(gs), bins=60, alpha=0.6, label="seen",   color="steelblue", density=True)
    ax.hist(np_flat(gu), bins=60, alpha=0.6, label="unseen", color="coral",     density=True)
    ax.set_title(f"{lbl} distribution"); ax.set_xlabel("value"); ax.legend()
    if "gamma" in lbl:
        ax.axvline(x=0.5, color='red',   linestyle='--', linewidth=1.5, label='γ_min=0.5 (floor)')
        ax.axvline(x=1.0, color='green', linestyle='--', linewidth=1.5, label='γ=1 (identity)')
        ax.axvline(x=0.0, color='black', linestyle=':',  linewidth=1.0, label='γ=0 (collapse)')
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{EXP_FIG_DIR}/film_param_distributions.png", dpi=150, bbox_inches="tight")
plt.show()

print("FiLM parameter summary — residual parameterisation (seen / unseen):")
print(f"  gamma1 (=1+Δγ1): mean {g1s.mean():.4f}/{g1u.mean():.4f}  "
      f"std {g1s.std():.4f}/{g1u.std():.4f}  min {g1s.min():.4f}/{g1u.min():.4f}")
print(f"  beta1:  L2-norm {b1s.norm(dim=-1).mean():.4f}/{b1u.norm(dim=-1).mean():.4f}")
print(f"  gamma2 (=1+Δγ2): mean {g2s.mean():.4f}/{g2u.mean():.4f}  "
      f"std {g2s.std():.4f}/{g2u.std():.4f}  min {g2s.min():.4f}/{g2u.min():.4f}")
print(f"  beta2:  L2-norm {b2s.norm(dim=-1).mean():.4f}/{b2u.norm(dim=-1).mean():.4f}")
print(f"\\n  Exp05 failure: gamma1_mean=0.160. Target here: gamma > 0.5.")
print(f"  gamma=1 is identity. Deviation from 1 = learned prototype-specific scaling.")
print(f"  L_gamma floor (gamma_min={WGAN_CONFIG['gamma_min']}) should prevent collapse.")

generator.train()
"""

# ──────────────────────────────────────────────────────────────────────────────
# CELL 75: HARVEST CELL — Residual FiLM metrics
# ──────────────────────────────────────────────────────────────────────────────
CELL_75 = """\
# =============================================================================
# RESULTS HARVEST — copy-paste this output block to Claude Code
# =============================================================================
import json as _json

def _get(name, default='N/A'):
    try:    return eval(name)
    except: return default

_results = {
    'experiment':            WGAN_CONFIG.get('experiment', 'unknown'),
    'hidden_dim':            WGAN_CONFIG.get('hidden_dim', 512),
    'film_hidden':           WGAN_CONFIG.get('film_hidden', 128),
    'n_steps':               WGAN_CONFIG.get('n_steps', 0),
    'alpha':                 WGAN_CONFIG.get('alpha', 0.0),
    'gamma_min':             WGAN_CONFIG.get('gamma_min', 0.5),
    'lambda_gamma':          WGAN_CONFIG.get('lambda_gamma', 0.1),
    'H_mean_pct':            round(_get('H_gzsl', 0) * 100, 4),
    'AccS_pct':              round(_get('acc_seen_gzsl', 0) * 100, 4),
    'AccU_pct':              round(_get('acc_unseen_gzsl', 0) * 100, 4),
    'routing_pct':           round(_get('routing_rate', 0) * 100, 2),
    'VarR_seen':             round(float(_get('VarR_seen', 0)), 5),
    'VarR_unseen':           round(float(_get('VarR_unseen', 0)), 5),
    'VarR_gap':              round(float(_get('VarR_seen', 0)) - float(_get('VarR_unseen', 0)), 5),
    'rho_sp':                round(float(_get('rho_sp', 0)), 5),
    'kNN10':                 round(float(_get('knn10', 0)), 5),
    # Residual FiLM: g1s/g2s = effective gamma = 1 + delta_gamma
    'gamma1_mean':           round(float(_get('g1s.abs().mean().item()', 0)), 5),
    'gamma1_min':            round(float(_get('g1s.min().item()', 0)), 5),
    'delta_gamma1_mean':     round(float(_get('dg1s.abs().mean().item()', 0)), 5),
    'gamma2_mean':           round(float(_get('g2s.abs().mean().item()', 0)), 5),
    'gamma2_min':            round(float(_get('g2s.min().item()', 0)), 5),
    'beta1_norm':            round(float(_get('b1s.norm(dim=-1).mean().item()', 0)), 5),
    'beta2_norm':            round(float(_get('b2s.norm(dim=-1).mean().item()', 0)), 5),
    'VarR_seen_posttrain':   round(float(_get('VarR_seen_posttrain', 0)), 5),
    'g_loss_final_mean1k':   round(float(np.mean(_get('g_losses', [0])[-1000:])), 5),
    'c_loss_final_mean1k':   round(float(np.mean(_get('c_losses', [0])[-1000:])), 5),
}

print()
print('=' * 60)
print('RESULTS HARVEST — paste this block to Claude Code')
print('=' * 60)
print(_json.dumps(_results, indent=2))
print('=' * 60)
print()
print('KEY DIAGNOSTICS:')
print(f'  gamma1_mean = {_results[\"gamma1_mean\"]:.4f}  '
      f'(Exp05 collapsed to 0.1601; target > 0.5)')
print(f'  gamma1_min  = {_results[\"gamma1_min\"]:.4f}  '
      f'(any element < 0.5 means floor not holding)')
print(f'  routing_pct = {_results[\"routing_pct\"]:.2f}%  '
      f'(Exp05: 54.81%; healthy: ~20%)')
"""

# ── Apply all cell modifications ──────────────────────────────────────────────
cells[41]['source'] = CELL_41
cells[42]['source'] = CELL_42
# cells[43] unchanged (data prep with class labels)
cells[44]['source'] = CELL_44
# cells[45] unchanged (post-training VarR check — uses open WandB run)
cells[50]['source'] = CELL_50
cells[75]['source'] = CELL_75

# Write output
with open(OUT, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Written: {OUT}")
print(f"Total cells: {len(cells)}")
print("Cells modified: 41, 42, 44, 50, 75")
print("Cells unchanged: 43 (data prep), 45 (VarR posttrain), 51-54 (VarR/rho_sp/kNN/tsne)")
