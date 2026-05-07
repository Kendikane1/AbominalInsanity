#!/usr/bin/env python3
"""
Build FiLM conditioning experiment notebooks (Experiment 04).

Creates 3 variants in experiments/04_film_conditioning/:
  notebook.ipynb           — FiLM baseline (pure architecture fix)
  notebook_film_ln.ipynb   — FiLM + LayerNorm after each modulation step
  notebook_film_lvar.ipynb — FiLM + L_var alpha sweep

Archive after running:
  mv experiments/04_film_conditioning/build_film_notebooks.py archive/misc/
"""

import json, copy
from pathlib import Path

BASE = Path("experiments/04_film_conditioning/notebook.ipynb")
OUT  = Path("experiments/04_film_conditioning")

def code_cell(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}

def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

# =============================================================================
# Cell source strings
# =============================================================================

CELL_06_ADDITION = '''
# Import WandB utility helpers (after sys.path is set above)
from shared.wandb_utils import init_run, log_gzsl_results, finish_run
print("wandb_utils imported.")
'''

CELL_41_BASELINE = '''\
# =============================================================================
# cWGAN-GP CONFIGURATION — FiLM Baseline (Exp 04)
# =============================================================================

WGAN_CONFIG = {
    'z_dim': 100,
    'embed_dim': 64,           # Must match ENCODER_CONFIG['embed_dim']
    'lr': 1e-4,
    'betas': (0.0, 0.9),
    'lambda_gp': 10,
    'n_critic': 5,
    'n_steps': 10000,
    'batch_size': 256,
    'n_synth_per_class': 20,
    'film_hidden': 128,        # FiLM MLP hidden dim (gamma/beta networks)
    'seed': 42,
    'experiment': 'film_baseline',
}

print("=" * 60)
print("cWGAN-GP CONFIGURATION (FiLM Baseline)")
print("=" * 60)
for k, v in WGAN_CONFIG.items():
    print(f"  {k}: {v}")
print("=" * 60)
'''

CELL_41_LN = CELL_41_BASELINE.replace("'film_baseline'", "'film_layernorm'").replace(
    "(FiLM Baseline)", "(FiLM + LayerNorm)")

CELL_41_LVAR = CELL_41_BASELINE.replace("'film_baseline'", "'film_lvar'").replace(
    "(FiLM Baseline)", "(FiLM + L_var Sweep)").replace(
    "    'seed': 42,", "    'seed': 42,\n    'alpha_sweep': [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],")

# ── FiLM baseline generator ──────────────────────────────────────────────────

GENERATOR_FILM = '''\
class FiLMGenerator(nn.Module):
    """
    FiLM-conditioned generator: z -> [FiLM1(s_c) -> FiLM2(s_c)] -> e_hat

    Architecture:
      Noise path (class-agnostic): z -> h1 -> h2 -> e_hat
      Prototype conditioning:       s_c -> (gamma1, beta1), (gamma2, beta2)
      Modulation:                   h_i\\'  = gamma_i * h_i + beta_i

    Jacobian: J_G = W3 * D2(z, s_c) * diag(gamma2) * W2 * D1(z) * diag(gamma1) * W1
    D1 depends only on z (not s_c), enabling variance to transfer to unseen prototypes.

    Initialisation: gamma = 1, beta = 0 (identity modulation at t=0).
    """
    def __init__(self, z_dim=100, embed_dim=64, hidden_dim=256, film_hidden=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, embed_dim)
        self.film1_mlp = nn.Sequential(
            nn.Linear(embed_dim, film_hidden), nn.ReLU(),
            nn.Linear(film_hidden, hidden_dim * 2),   # -> [gamma1 | beta1]
        )
        self.film2_mlp = nn.Sequential(
            nn.Linear(embed_dim, film_hidden), nn.ReLU(),
            nn.Linear(film_hidden, hidden_dim * 2),   # -> [gamma2 | beta2]
        )
        self._init_film_identity()

    def _init_film_identity(self):
        for mlp in (self.film1_mlp, self.film2_mlp):
            last = mlp[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            with torch.no_grad():
                last.bias[:self.hidden_dim].fill_(1.0)  # gamma = 1

    def _film(self, h, mlp, s_c):
        params = mlp(s_c)
        gamma, beta = params.chunk(2, dim=-1)
        return gamma * h + beta

    def forward(self, z, s_c):
        h1 = F.leaky_relu(self.fc1(z), 0.2)
        h1 = self._film(h1, self.film1_mlp, s_c)
        h2 = F.leaky_relu(self.fc2(h1), 0.2)
        h2 = self._film(h2, self.film2_mlp, s_c)
        return F.normalize(self.fc3(h2), p=2, dim=-1)

    def get_film_stats(self, s_c):
        with torch.no_grad():
            p1 = self.film1_mlp(s_c);  g1, b1 = p1.chunk(2, dim=-1)
            p2 = self.film2_mlp(s_c);  g2, b2 = p2.chunk(2, dim=-1)
        return {
            'gamma1_mean': g1.abs().mean().item(),
            'beta1_norm':  b1.norm(dim=-1).mean().item(),
            'gamma2_mean': g2.abs().mean().item(),
            'beta2_norm':  b2.norm(dim=-1).mean().item(),
        }
'''

# ── FiLM + LayerNorm generator ───────────────────────────────────────────────

GENERATOR_FILM_LN = '''\
class FiLMGeneratorLN(nn.Module):
    """
    FiLM generator with LayerNorm after each modulation step.

    LayerNorm re-centres h_i\\' = gamma_i * h_i + beta_i, limiting the extent
    to which large beta values create prototype-dependent activation patterns
    in downstream layers. This further decouples D2 from s_c, strengthening
    the variance transfer guarantee relative to the baseline FiLM variant.
    """
    def __init__(self, z_dim=100, embed_dim=64, hidden_dim=256, film_hidden=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, embed_dim)
        self.film1_mlp = nn.Sequential(
            nn.Linear(embed_dim, film_hidden), nn.ReLU(),
            nn.Linear(film_hidden, hidden_dim * 2),
        )
        self.film2_mlp = nn.Sequential(
            nn.Linear(embed_dim, film_hidden), nn.ReLU(),
            nn.Linear(film_hidden, hidden_dim * 2),
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self._init_film_identity()

    def _init_film_identity(self):
        for mlp in (self.film1_mlp, self.film2_mlp):
            last = mlp[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            with torch.no_grad():
                last.bias[:self.hidden_dim].fill_(1.0)

    def _film_ln(self, h, mlp, s_c, ln):
        params = mlp(s_c)
        gamma, beta = params.chunk(2, dim=-1)
        return ln(gamma * h + beta)

    def forward(self, z, s_c):
        h1 = F.leaky_relu(self.fc1(z), 0.2)
        h1 = self._film_ln(h1, self.film1_mlp, s_c, self.ln1)
        h2 = F.leaky_relu(self.fc2(h1), 0.2)
        h2 = self._film_ln(h2, self.film2_mlp, s_c, self.ln2)
        return F.normalize(self.fc3(h2), p=2, dim=-1)

    def get_film_stats(self, s_c):
        with torch.no_grad():
            p1 = self.film1_mlp(s_c);  g1, b1 = p1.chunk(2, dim=-1)
            p2 = self.film2_mlp(s_c);  g2, b2 = p2.chunk(2, dim=-1)
        return {
            'gamma1_mean': g1.abs().mean().item(),
            'beta1_norm':  b1.norm(dim=-1).mean().item(),
            'gamma2_mean': g2.abs().mean().item(),
            'beta2_norm':  b2.norm(dim=-1).mean().item(),
        }
'''

CRITIC_AND_GP = '''\

class Critic(nn.Module):
    """Conditional critic: [e, s_c] -> scalar score (unchanged from concat baseline)."""
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
    """WGAN-GP gradient penalty: E[(||grad D(e_bar)||_2 - 1)^2]."""
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

'''

INSTANTIATION_BASELINE = '''\
generator = FiLMGenerator(
    z_dim=WGAN_CONFIG['z_dim'],
    embed_dim=WGAN_CONFIG['embed_dim'],
    film_hidden=WGAN_CONFIG['film_hidden'],
).to(device)

critic = Critic(embed_dim=WGAN_CONFIG['embed_dim']).to(device)

print(f"FiLMGenerator parameters:  {sum(p.numel() for p in generator.parameters()):,}")
print(f"Critic parameters:          {sum(p.numel() for p in critic.parameters()):,}")
'''

INSTANTIATION_LN = INSTANTIATION_BASELINE.replace(
    "FiLMGenerator(", "FiLMGeneratorLN("
).replace(
    "FiLMGenerator parameters:", "FiLMGeneratorLN parameters:"
)

def make_cell42(variant_label, generator_code, instantiation_code):
    return (
        f"# =============================================================================\n"
        f"# cWGAN-GP MODEL DEFINITIONS — {variant_label}\n"
        f"# =============================================================================\n\n"
        + generator_code
        + CRITIC_AND_GP
        + instantiation_code
    )

# ── Training loop (baseline + LN variants — no auxiliary loss) ───────────────

TRAINING_LOOP_FILM = '''\
# =============================================================================
# cWGAN-GP TRAINING — FiLM Generator
# =============================================================================

run = init_run(
    experiment_name=f"04_{WGAN_CONFIG['experiment']}",
    config_dict={**ENCODER_CONFIG, **WGAN_CONFIG},
    tags=["exp04", "film-conditioning", "wgan-gp"],
)

g_optimizer = torch.optim.Adam(generator.parameters(), lr=WGAN_CONFIG['lr'], betas=WGAN_CONFIG['betas'])
c_optimizer = torch.optim.Adam(critic.parameters(),    lr=WGAN_CONFIG['lr'], betas=WGAN_CONFIG['betas'])

g_losses, c_losses, gp_values = [], [], []


def compute_var_ratio_quick(gen, s_protos, cls_list, e_real, y_real,
                             z_dim, dev, n_cls=50, n_synth=15):
    """Estimate training VarR on a random subset of seen classes."""
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


print(f"Training FiLM cWGAN-GP for {WGAN_CONFIG['n_steps']} generator steps...")
print(f"Critic updates per G step: {WGAN_CONFIG['n_critic']}")

data_iter = iter(wgan_loader)
g_step    = 0

while g_step < WGAN_CONFIG['n_steps']:
    # ── Critic ────────────────────────────────────────────────────────────────
    for _ in range(WGAN_CONFIG['n_critic']):
        try:   e_real, s_c = next(data_iter)
        except StopIteration:
            data_iter = iter(wgan_loader); e_real, s_c = next(data_iter)
        e_real, s_c = e_real.to(device), s_c.to(device)
        batch_size  = e_real.size(0)
        z           = torch.randn(batch_size, WGAN_CONFIG['z_dim'], device=device)
        e_fake      = generator(z, s_c)
        d_real      = critic(e_real, s_c)
        d_fake      = critic(e_fake.detach(), s_c)
        gp          = compute_gradient_penalty(critic, e_real, e_fake.detach(), s_c, device)
        c_loss      = -d_real.mean() + d_fake.mean() + WGAN_CONFIG['lambda_gp'] * gp
        c_optimizer.zero_grad(); c_loss.backward(); c_optimizer.step()

    # ── Generator ─────────────────────────────────────────────────────────────
    try:   e_real, s_c = next(data_iter)
    except StopIteration:
        data_iter = iter(wgan_loader); e_real, s_c = next(data_iter)
    s_c        = s_c.to(device)
    batch_size = s_c.size(0)
    z          = torch.randn(batch_size, WGAN_CONFIG['z_dim'], device=device)
    e_fake     = generator(z, s_c)
    g_loss     = -critic(e_fake, s_c).mean()
    g_optimizer.zero_grad(); g_loss.backward(); g_optimizer.step()

    g_losses.append(g_loss.item())
    c_losses.append(c_loss.item())
    gp_values.append(gp.item())
    g_step += 1

    if g_step % 500 == 0 or g_step == 1:
        film_stats  = generator.get_film_stats(s_c)
        log_payload = {
            "train/step":          g_step,
            "train/L_wasserstein": g_loss.item(),
            "train/L_D":           c_loss.item(),
            "train/GP":            gp.item(),
            **{f"train/{k}": v for k, v in film_stats.items()},
        }
        var_r_str = ""
        if g_step % 1000 == 0:
            var_r = compute_var_ratio_quick(
                generator, S_seen_prototypes, list(seen_classes),
                E_train_seen, label_train_seen, WGAN_CONFIG['z_dim'], device,
            )
            log_payload["train/VarR_seen"] = var_r
            var_r_str = f", VarR_seen={var_r:.3f}"
        wandb.log(log_payload, step=g_step)
        print(f"  Step {g_step:5d}/{WGAN_CONFIG['n_steps']}: "
              f"G={g_loss.item():.4f}, C={c_loss.item():.4f}, "
              f"GP={gp.item():.4f}{var_r_str}")

print("\\nFiLM cWGAN-GP training complete!")
'''

# ── Training loop (L_var sweep variant) ─────────────────────────────────────

TRAINING_LOOP_LVAR = '''\
# =============================================================================
# cWGAN-GP TRAINING — FiLM + L_var Alpha Sweep
# =============================================================================
# L_var = -mean_c Var_within[G(z, s_c)] — maximises within-class variance.
# With FiLM, D1 is class-agnostic so L_var should generalise to unseen classes.
# Sweep alpha values; each alpha = one WandB run.

import copy as _copy
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def compute_lvar(e_fake, y_batch):
    """L_var = negative mean within-class variance over classes in this batch."""
    classes = y_batch.unique()
    var_per_class = []
    for c in classes:
        mask = (y_batch == c)
        if mask.sum() > 1:
            var_per_class.append(e_fake[mask].var(dim=0).mean())
    if not var_per_class:
        return torch.tensor(0.0, device=e_fake.device)
    return -torch.stack(var_per_class).mean()


def grad_cosine_sim(model, loss_w, loss_v):
    """Cosine similarity between grad(L_wass) and grad(L_var) w.r.t. generator params."""
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


ALPHA_SWEEP = WGAN_CONFIG['alpha_sweep']
sweep_results  = []
best_H         = -1.0
best_gen_state = None
best_g_losses  = []
best_c_losses  = []
best_gp_values = []

for alpha in ALPHA_SWEEP:
    print(f"\\n{'='*60}")
    print(f"Alpha = {alpha:.2f}  (L_G = L_wass + {alpha:.2f} * L_var)")
    print('='*60)

    torch.manual_seed(WGAN_CONFIG['seed'])
    np.random.seed(WGAN_CONFIG['seed'])

    gen_a  = FiLMGenerator(
        z_dim=WGAN_CONFIG['z_dim'],
        embed_dim=WGAN_CONFIG['embed_dim'],
        film_hidden=WGAN_CONFIG['film_hidden'],
    ).to(device)
    crit_a = Critic(embed_dim=WGAN_CONFIG['embed_dim']).to(device)

    g_opt = torch.optim.Adam(gen_a.parameters(),  lr=WGAN_CONFIG['lr'], betas=WGAN_CONFIG['betas'])
    c_opt = torch.optim.Adam(crit_a.parameters(), lr=WGAN_CONFIG['lr'], betas=WGAN_CONFIG['betas'])

    run = init_run(
        experiment_name=f"04_film_lvar_alpha{alpha:.2f}",
        config_dict={**ENCODER_CONFIG, **WGAN_CONFIG, 'alpha': alpha},
        tags=["exp04", "film-conditioning", "lvar", f"alpha-{alpha:.2f}"],
    )

    g_losses_a, c_losses_a, gp_values_a = [], [], []
    data_iter = iter(wgan_loader)
    g_step    = 0

    while g_step < WGAN_CONFIG['n_steps']:
        # ── Critic ────────────────────────────────────────────────────────────
        for _ in range(WGAN_CONFIG['n_critic']):
            try:   e_real, s_c, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(wgan_loader); e_real, s_c, _ = next(data_iter)
            e_real, s_c = e_real.to(device), s_c.to(device)
            batch_size  = e_real.size(0)
            z           = torch.randn(batch_size, WGAN_CONFIG['z_dim'], device=device)
            e_fake      = gen_a(z, s_c)
            gp          = compute_gradient_penalty(crit_a, e_real, e_fake.detach(), s_c, device)
            c_loss      = (-crit_a(e_real, s_c).mean()
                           + crit_a(e_fake.detach(), s_c).mean()
                           + WGAN_CONFIG['lambda_gp'] * gp)
            c_opt.zero_grad(); c_loss.backward(); c_opt.step()

        # ── Generator ─────────────────────────────────────────────────────────
        try:   e_real, s_c, y_b = next(data_iter)
        except StopIteration:
            data_iter = iter(wgan_loader); e_real, s_c, y_b = next(data_iter)
        s_c, y_b   = s_c.to(device), y_b.to(device)
        batch_size = s_c.size(0)
        z          = torch.randn(batch_size, WGAN_CONFIG['z_dim'], device=device)
        e_fake     = gen_a(z, s_c)
        l_wass     = -crit_a(e_fake, s_c).mean()

        if alpha > 0:
            l_var  = compute_lvar(e_fake, y_b)
            g_loss = l_wass + alpha * l_var
        else:
            l_var  = torch.tensor(0.0, device=device)
            g_loss = l_wass

        g_opt.zero_grad(); g_loss.backward(); g_opt.step()

        g_losses_a.append(g_loss.item())
        c_losses_a.append(c_loss.item())
        gp_values_a.append(gp.item())
        g_step += 1

        if g_step % 500 == 0 or g_step == 1:
            film_stats  = gen_a.get_film_stats(s_c)
            log_payload = {
                "train/step":          g_step,
                "train/L_wasserstein": l_wass.item(),
                "train/L_var":         l_var.item(),
                "train/L_G":           g_loss.item(),
                "train/L_D":           c_loss.item(),
                "train/GP":            gp.item(),
                **{f"train/{k}": v for k, v in film_stats.items()},
            }
            var_r_str = ""
            if g_step % 1000 == 0:
                var_r = compute_var_ratio_quick(
                    gen_a, S_seen_prototypes, list(seen_classes),
                    E_train_seen, label_train_seen, WGAN_CONFIG['z_dim'], device,
                )
                log_payload["train/VarR_seen"] = var_r
                var_r_str = f", VarR_seen={var_r:.3f}"

                if alpha > 0 and g_step <= 5000:
                    e_gs   = gen_a(torch.randn(batch_size, WGAN_CONFIG['z_dim'], device=device), s_c)
                    lw_gs  = -crit_a(e_gs, s_c).mean()
                    lv_gs  = compute_lvar(e_gs, y_b)
                    if lv_gs.abs().item() > 1e-6:
                        cos    = grad_cosine_sim(gen_a, lw_gs, lv_gs)
                        log_payload["train/grad_cos_sim"] = cos
                        var_r_str += f", cos={cos:.3f}"

            wandb.log(log_payload, step=g_step)
            if g_step % 1000 == 0 or g_step == 1:
                print(f"  a={alpha:.2f} step {g_step:5d}: "
                      f"G={g_loss.item():.4f}, Lvar={l_var.item():.4f}{var_r_str}")

    # ── Per-alpha evaluation ───────────────────────────────────────────────────
    gen_a.eval()
    synth_emb_a, synth_lbl_a = [], []
    with torch.no_grad():
        for c in unseen_classes:
            sc = torch.FloatTensor(S_unseen_prototypes[c]).unsqueeze(0).repeat(
                WGAN_CONFIG['n_synth_per_class'], 1).to(device)
            z  = torch.randn(WGAN_CONFIG['n_synth_per_class'], WGAN_CONFIG['z_dim'], device=device)
            synth_emb_a.append(gen_a(z, sc).cpu().numpy())
            synth_lbl_a.extend([c] * WGAN_CONFIG['n_synth_per_class'])
    E_synth_a = np.vstack(synth_emb_a)
    y_synth_a = np.array(synth_lbl_a)

    var_r_unseen = compute_var_ratio_full(E_synth_a, y_synth_a, E_unseen, y_unseen)

    synth_cents = np.array([E_synth_a[y_synth_a == c].mean(axis=0) for c in sorted(unseen_classes)])
    real_pts    = np.array([S_unseen_prototypes[c]                  for c in sorted(unseen_classes)])
    n_u = len(unseen_classes)
    idx = np.triu_indices(n_u, k=1)
    rho_sp, _ = spearmanr(cdist(synth_cents, synth_cents)[idx], cdist(real_pts, real_pts)[idx])

    # GZSL eval (same balancing as main pipeline)
    seen_cls_arr, seen_cts = np.unique(label_train_seen, return_counts=True)
    med_pc = int(np.median(seen_cts))
    rng = np.random.RandomState(WGAN_CONFIG['seed'])
    ds_idx = np.concatenate([
        rng.choice(np.where(y_synth_a == c)[0],
                   size=min(med_pc, int((y_synth_a == c).sum())), replace=False)
        for c in np.unique(y_synth_a)
    ])
    X_tr = np.vstack([E_train_seen, E_synth_a[ds_idx]])
    y_tr = np.concatenate([label_train_seen, y_synth_a[ds_idx]])

    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000,
                              class_weight='balanced', random_state=WGAN_CONFIG['seed'], n_jobs=-1)
    clf.fit(X_tr, y_tr)

    acc_s = accuracy_score(label_test_seen, clf.predict(E_test_seen))
    acc_u = accuracy_score(y_unseen,        clf.predict(E_unseen))
    H     = 2 * acc_s * acc_u / (acc_s + acc_u) if (acc_s + acc_u) > 0 else 0.0
    unseen_lbl_set = set(int(c) for c in unseen_classes)
    routing = sum(1 for p in clf.predict(E_unseen) if int(p) in unseen_lbl_set) / len(y_unseen)

    log_gzsl_results(H, acc_s, acc_u, routing, VarR=var_r_unseen, rho_sp=rho_sp)
    finish_run()

    sweep_results.append({
        'alpha': alpha, 'H': H, 'AccS': acc_s, 'AccU': acc_u,
        'VarR_unseen': var_r_unseen, 'rho_sp': rho_sp,
    })
    print(f"  alpha={alpha:.2f}: H={H:.4f}, AccS={acc_s:.4f}, AccU={acc_u:.4f}, "
          f"VarR_unseen={var_r_unseen:.4f}, rho_sp={rho_sp:.4f}")

    if H > best_H:
        best_H         = H
        best_gen_state = _copy.deepcopy(gen_a.state_dict())
        best_g_losses  = g_losses_a
        best_c_losses  = c_losses_a
        best_gp_values = gp_values_a
    gen_a.train()

print("\\n" + "="*60)
print("ALPHA SWEEP COMPLETE")
print("="*60)
for r in sweep_results:
    mark = " <-- best" if r['H'] == best_H else ""
    print(f"  alpha={r['alpha']:.2f}: H={r['H']:.4f}, "
          f"VarR_u={r['VarR_unseen']:.4f}, rho_sp={r['rho_sp']:.4f}{mark}")

# Load best generator for downstream pipeline cells
best_alpha = sweep_results[[r['H'] for r in sweep_results].index(best_H)]['alpha']
print(f"\\nBest alpha = {best_alpha:.2f} (H={best_H:.4f})")

generator = FiLMGenerator(
    z_dim=WGAN_CONFIG['z_dim'],
    embed_dim=WGAN_CONFIG['embed_dim'],
    film_hidden=WGAN_CONFIG['film_hidden'],
).to(device)
generator.load_state_dict(best_gen_state)
generator.eval()

g_losses  = best_g_losses
c_losses  = best_c_losses
gp_values = best_gp_values
print("Best generator loaded into `generator` — downstream cells run on best-alpha model.")
'''

# ── Modified cell 43 for lvar (add labels to DataLoader) ────────────────────

CELL_43_LVAR = '''\
# =============================================================================
# cWGAN-GP DATA PREPARATION (with class labels for L_var grouping)
# =============================================================================

def get_prototype_for_labels(labels, prototypes_dict):
    return np.array([prototypes_dict[int(l)] for l in labels])

E_train_tensor      = torch.FloatTensor(E_train_seen)
S_train_conditions  = torch.FloatTensor(get_prototype_for_labels(label_train_seen, S_seen_prototypes))
y_train_tensor      = torch.LongTensor(label_train_seen.astype(int))  # for L_var class grouping

wgan_dataset = TensorDataset(E_train_tensor, S_train_conditions, y_train_tensor)
wgan_loader  = DataLoader(wgan_dataset, batch_size=WGAN_CONFIG['batch_size'], shuffle=True, drop_last=True)

# Load unseen embeddings for per-alpha evaluation
y_unseen = np.load('cached_arrays/y_unseen.npy') if 'y_unseen' not in dir() else y_unseen

print(f"WGAN training data (with labels): {len(wgan_dataset)} samples")
print(f"Batch size: {WGAN_CONFIG['batch_size']}")
print(f"Batches per epoch: {len(wgan_loader)}")
'''

# ── Diagnostic cells ─────────────────────────────────────────────────────────

DIAG_MD_SRC = '''\
---
## FiLM Conditioning Diagnostics

Post-training analysis: (1) gamma/beta parameter distributions, (2) VarR transfer gap,
(3) structural overcoupling rho_sp. These are the metrics that failed in Exp 03 (L_var).
'''

DIAG_FILM_PARAMS = '''\
# =============================================================================
# FILM PARAMETER ANALYSIS: gamma/beta distributions (seen vs unseen prototypes)
# =============================================================================
# Key check: do unseen prototypes produce gamma/beta in the same range as seen?
# If yes  -> smooth interpolation across prototype space -> variance transfer works.
# If no   -> unseen prototypes are OOD for FiLM MLPs -> conditioning collapses.

EXP_FIG_DIR = 'experiments/04_film_conditioning/figures'
import os; os.makedirs(EXP_FIG_DIR, exist_ok=True)

generator.eval()

seen_pt   = torch.FloatTensor(np.array([S_seen_prototypes[c]   for c in seen_classes])).to(device)
unseen_pt = torch.FloatTensor(np.array([S_unseen_prototypes[c] for c in unseen_classes])).to(device)

with torch.no_grad():
    p1s = generator.film1_mlp(seen_pt);   g1s, b1s = p1s.chunk(2, dim=-1)
    p1u = generator.film1_mlp(unseen_pt); g1u, b1u = p1u.chunk(2, dim=-1)
    p2s = generator.film2_mlp(seen_pt);   g2s, b2s = p2s.chunk(2, dim=-1)
    p2u = generator.film2_mlp(unseen_pt); g2u, b2u = p2u.chunk(2, dim=-1)

def np_flat(t): return t.cpu().numpy().flatten()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("FiLM Parameters: Seen vs Unseen Prototypes", fontsize=14)

for ax, gs, gu, lbl in [
    (axes[0, 0], g1s, g1u, "gamma1"),
    (axes[0, 1], b1s, b1u, "beta1"),
    (axes[1, 0], g2s, g2u, "gamma2"),
    (axes[1, 1], b2s, b2u, "beta2"),
]:
    ax.hist(np_flat(gs), bins=60, alpha=0.6, label="seen",   color="steelblue", density=True)
    ax.hist(np_flat(gu), bins=60, alpha=0.6, label="unseen", color="coral",     density=True)
    ax.set_title(f"{lbl} distribution"); ax.set_xlabel("value"); ax.legend()

plt.tight_layout()
plt.savefig(f"{EXP_FIG_DIR}/film_param_distributions.png", dpi=150, bbox_inches="tight")
plt.show()

print("FiLM parameter summary (seen / unseen):")
print(f"  gamma1: mean {g1s.mean():.3f}/{g1u.mean():.3f}, std {g1s.std():.3f}/{g1u.std():.3f}")
print(f"  beta1:  L2-norm {b1s.norm(dim=-1).mean():.3f}/{b1u.norm(dim=-1).mean():.3f}")
print(f"  gamma2: mean {g2s.mean():.3f}/{g2u.mean():.3f}, std {g2s.std():.3f}/{g2u.std():.3f}")
print(f"  beta2:  L2-norm {b2s.norm(dim=-1).mean():.3f}/{b2u.norm(dim=-1).mean():.3f}")
print("\\n  gamma=1.0 at init; deviation from 1 shows learned prototype-specific scaling.")

generator.train()
'''

DIAG_VARR = '''\
# =============================================================================
# VARIANCE RATIO ANALYSIS: Seen (training) vs Unseen (evaluation)
# =============================================================================
# THE primary diagnostic: VarR_seen >> VarR_unseen means variance does not
# transfer from seen to unseen prototypes — the failure mode of Exp 03.
# FiLM goal: VarR_unseen > 0.95, transfer gap < 0.03.

generator.eval()


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


# Training VarR (seen classes)
n_diag = 20
synth_seen_emb, synth_seen_lbl = [], []
with torch.no_grad():
    for c in seen_classes:
        sc = torch.FloatTensor(S_seen_prototypes[c]).unsqueeze(0).repeat(n_diag, 1).to(device)
        z  = torch.randn(n_diag, WGAN_CONFIG["z_dim"], device=device)
        synth_seen_emb.append(generator(z, sc).cpu().numpy())
        synth_seen_lbl.extend([c] * n_diag)
synth_seen_emb = np.vstack(synth_seen_emb)
synth_seen_lbl = np.array(synth_seen_lbl)

VarR_seen   = compute_var_ratio_full(synth_seen_emb, synth_seen_lbl, E_train_seen, label_train_seen)
VarR_unseen = compute_var_ratio_full(E_synth_unseen,  y_synth_unseen, E_unseen,     y_unseen)

print("=" * 60)
print("VARIANCE RATIO ANALYSIS")
print("=" * 60)
print(f"  VarR (seen  classes, training): {VarR_seen:.4f}")
print(f"  VarR (unseen classes, eval):    {VarR_unseen:.4f}")
print(f"  Transfer gap (seen - unseen):   {VarR_seen - VarR_unseen:.4f}")
print()
print("  Reference baselines:")
print("    Exp 01 (concat WGAN-GP):  VarR_unseen = 0.872")
print("    Exp 03 (L_var training):  VarR_seen=0.973, VarR_unseen=0.875, gap=0.098")
print("    FiLM target:              VarR_unseen > 0.95, gap < 0.03")

wandb.log({
    "eval/VarR_seen":   VarR_seen,
    "eval/VarR_unseen": VarR_unseen,
    "eval/VarR_gap":    VarR_seen - VarR_unseen,
})

generator.train()
'''

DIAG_RHO_SP = '''\
# =============================================================================
# STRUCTURAL OVERCOUPLING: rho_sp and kNN@10
# =============================================================================
# rho_sp = Spearman corr(d_synthetic_centroid, d_real_prototype) over unseen pairs.
# Overcoupled (pathological): synthetic centroids mirror prototype geometry too closely.
# Baseline (Exp 01): rho_sp=0.857. Exp 03 (L_var): 0.880 (worse). Real data: ~0.668.
# FiLM target: rho_sp < 0.800.

from scipy.stats import spearmanr
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

synth_cents = np.array([E_synth_unseen[y_synth_unseen == c].mean(axis=0)
                         for c in sorted(unseen_classes)])
real_pts    = np.array([S_unseen_prototypes[c] for c in sorted(unseen_classes)])

n_u = len(unseen_classes)
d_s = cdist(synth_cents, synth_cents)
d_r = cdist(real_pts,    real_pts)
idx_u = np.triu_indices(n_u, k=1)
rho_sp, p_sp = spearmanr(d_s[idx_u], d_r[idx_u])

knn_p = NearestNeighbors(n_neighbors=11).fit(real_pts)
knn_s = NearestNeighbors(n_neighbors=11).fit(synth_cents)
_, ip  = knn_p.kneighbors(real_pts)
_, is_ = knn_s.kneighbors(synth_cents)
knn10 = np.mean([len(set(ip[i, 1:]) & set(is_[i, 1:])) / 10 for i in range(n_u)])

print("=" * 60)
print("STRUCTURAL OVERCOUPLING DIAGNOSTICS")
print("=" * 60)
print(f"  rho_sp (centroid vs prototype):   {rho_sp:.4f}  (p={p_sp:.2e})")
print(f"  kNN@10 neighbourhood preservation: {knn10:.4f}")
print()
print("  Reference: Exp 01 baseline: rho_sp=0.857, kNN@10=0.611")
print("             Exp 03 L_var:    rho_sp=0.880 (worse, overcoupling increased)")
print("             Real seen data:  rho_sp~0.668")
print("  FiLM target: rho_sp < 0.800")

wandb.log({"eval/rho_sp": rho_sp, "eval/kNN10": knn10})
'''

WANDB_LOG_GZSL = '''\
# =============================================================================
# LOG GZSL RESULTS TO WANDB
# =============================================================================

unseen_lbl_set  = set(int(c) for c in np.unique(y_unseen))
routing_rate    = sum(1 for p in y_pred_unseen if int(p) in unseen_lbl_set) / len(y_pred_unseen)

log_gzsl_results(
    H=H_gzsl,
    AccS=acc_seen_gzsl,
    AccU=acc_unseen_gzsl,
    routing=routing_rate,
)
print(f"Logged to WandB: H={H_gzsl:.4f}, AccS={acc_seen_gzsl:.4f}, "
      f"AccU={acc_unseen_gzsl:.4f}, routing={routing_rate:.4f}")
'''

WANDB_FINISH = '''\
# =============================================================================
# FINALISE WANDB RUN
# =============================================================================
finish_run()
print("WandB run closed.")
'''


# =============================================================================
# Build functions
# =============================================================================

def _src(cell):
    """Return cell source as a plain string regardless of list-or-str format."""
    s = cell['source']
    return ''.join(s) if isinstance(s, list) else s


def build_variant(nb_base, cell41_src, cell42_src, cell44_src,
                  cell43_src=None, variant_label=""):
    """Build a FiLM variant notebook from the base notebook."""
    nb = copy.deepcopy(nb_base)
    c  = nb['cells']

    # ── Cell 6: add wandb_utils import ───────────────────────────────────────
    c[6]['source'] = _src(c[6]) + CELL_06_ADDITION

    # ── Cell 41: WGAN config ─────────────────────────────────────────────────
    c[41]['source'] = cell41_src

    # ── Cell 42: model definitions ───────────────────────────────────────────
    c[42]['source'] = cell42_src

    # ── Cell 43: data prep (modified for lvar, unchanged otherwise) ──────────
    if cell43_src is not None:
        c[43]['source'] = cell43_src

    # ── Cell 44: training loop ────────────────────────────────────────────────
    c[44]['source'] = cell44_src

    # ── Insert diagnostic cells after cell 47 (sanity check) ─────────────────
    new_cells = []
    for i, cell in enumerate(c):
        new_cells.append(cell)
        if i == 47:
            new_cells.append(md_cell(DIAG_MD_SRC))
            new_cells.append(code_cell(DIAG_FILM_PARAMS))
            new_cells.append(code_cell(DIAG_VARR))
            new_cells.append(code_cell(DIAG_RHO_SP))
        # Old cell 57 is now at position 57+4=61 after the 4 insertions above.
        # We track it as: original index i == 57, we insert WandB log after it.
        if i == 57:
            new_cells.append(code_cell(WANDB_LOG_GZSL))

    # ── Append WandB finish cell ──────────────────────────────────────────────
    new_cells.append(code_cell(WANDB_FINISH))

    nb['cells'] = new_cells
    return nb


# =============================================================================
# Main build
# =============================================================================

with open(BASE) as f:
    nb_base = json.load(f)

# ── Notebook 1: FiLM baseline ─────────────────────────────────────────────────
cell42_baseline = make_cell42("FiLM Baseline", GENERATOR_FILM, INSTANTIATION_BASELINE)

nb1 = build_variant(
    nb_base,
    cell41_src=CELL_41_BASELINE,
    cell42_src=cell42_baseline,
    cell44_src=TRAINING_LOOP_FILM,
    variant_label="FiLM Baseline",
)
out1 = OUT / "notebook.ipynb"
with open(out1, "w") as f:
    json.dump(nb1, f, indent=1)
print(f"Written: {out1}  ({len(nb1['cells'])} cells)")

# ── Notebook 2: FiLM + LayerNorm ─────────────────────────────────────────────
cell42_ln = make_cell42("FiLM + LayerNorm", GENERATOR_FILM_LN, INSTANTIATION_LN)

nb2 = build_variant(
    nb_base,
    cell41_src=CELL_41_LN,
    cell42_src=cell42_ln,
    cell44_src=TRAINING_LOOP_FILM,  # same training loop, different generator class
    variant_label="FiLM + LayerNorm",
)
out2 = OUT / "notebook_film_ln.ipynb"
with open(out2, "w") as f:
    json.dump(nb2, f, indent=1)
print(f"Written: {out2}  ({len(nb2['cells'])} cells)")

# ── Notebook 3: FiLM + L_var sweep ───────────────────────────────────────────
cell42_lvar = make_cell42("FiLM + L_var Sweep", GENERATOR_FILM, INSTANTIATION_BASELINE)

nb3 = build_variant(
    nb_base,
    cell41_src=CELL_41_LVAR,
    cell42_src=cell42_lvar,
    cell43_src=CELL_43_LVAR,
    cell44_src=TRAINING_LOOP_LVAR,
    variant_label="FiLM + L_var Sweep",
)
out3 = OUT / "notebook_film_lvar.ipynb"
with open(out3, "w") as f:
    json.dump(nb3, f, indent=1)
print(f"Written: {out3}  ({len(nb3['cells'])} cells)")

print("\nAll 3 FiLM notebooks built successfully.")
print("Archive this script: mv experiments/04_film_conditioning/build_film_notebooks.py archive/misc/")
