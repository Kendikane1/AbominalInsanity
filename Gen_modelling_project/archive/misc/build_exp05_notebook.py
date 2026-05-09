"""
Build experiments/05_high_capacity_film/notebook.ipynb from the Exp04 lvar notebook.

Changes vs notebook_film_lvar.ipynb:
  Cell 41: WGAN_CONFIG — hidden_dim=512, film_hidden=256, n_steps=50000, alpha=1.0 (no sweep)
  Cell 42: FiLMGenerator instantiation with hidden_dim=512, film_hidden=256
  Cell 43: Data prep (unchanged — already has class labels for L_var)
  Cell 44: Single-run 50K training loop (no sweep loop), checkpoints every 10K steps
  Cell 45: Post-training VarR_seen check (replaces lvar's WandB reinit cell)
  Cell 75: Harvest cell — removes sweep table

Run from project root:
    python experiments/05_high_capacity_film/build_exp05_notebook.py
"""

import json, copy, os

BASE = os.path.join(
    os.path.dirname(__file__), '..', '04_film_conditioning', 'notebook_film_lvar.ipynb'
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
# cWGAN-GP CONFIGURATION — High-Capacity FiLM (Exp 05)
# =============================================================================
# Three changes from Exp04:
#   1. hidden_dim: 256 → 512  (3.53× generator parameters: 257K → 907K)
#   2. film_hidden: 128 → 256 (FiLM MLPs proportionally scaled)
#   3. n_steps: 10,000 → 50,000 (5× training budget for convergence)
#   4. alpha: 1.0 fixed (sweep optimum from Exp04c, no sweep needed)

WGAN_CONFIG = {
    'z_dim':             100,
    'embed_dim':         64,             # Must match ENCODER_CONFIG['embed_dim']
    'hidden_dim':        512,            # FiLM generator hidden width (was 256)
    'film_hidden':       256,            # FiLM MLP intermediate dim (was 128)
    'lr':                1e-4,
    'betas':             (0.0, 0.9),
    'lambda_gp':         10,
    'n_critic':          5,
    'n_steps':           50000,          # 5× longer training budget
    'batch_size':        256,
    'n_synth_per_class': 20,
    'alpha':             1.0,            # L_var weight (sweep optimum from Exp04c)
    'seed':              42,
    'experiment':        'film_highcap',
}

print("=" * 60)
print("cWGAN-GP CONFIGURATION (High-Capacity FiLM + L_var)")
print("=" * 60)
for k, v in WGAN_CONFIG.items():
    print(f"  {k}: {v}")
print("=" * 60)
"""

# ──────────────────────────────────────────────────────────────────────────────
# CELL 42: MODEL DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────
CELL_42 = """\
# =============================================================================
# cWGAN-GP MODEL DEFINITIONS — High-Capacity FiLM Generator
# =============================================================================

class FiLMGenerator(nn.Module):
    \"\"\"
    FiLM-conditioned generator: z -> [FiLM1(s_c) -> FiLM2(s_c)] -> e_hat

    Architecture:
      Noise path (class-agnostic): z -> h1 -> h2 -> e_hat
      Prototype conditioning:       s_c -> (gamma1, beta1), (gamma2, beta2)
      Modulation:                   h_i' = gamma_i * h_i + beta_i

    Jacobian: J_G = W3 * D2(z, s_c) * diag(gamma2) * W2 * D1(z) * diag(gamma1) * W1
    D1 depends only on z (not s_c), enabling variance to transfer to unseen prototypes.

    Initialisation: gamma = 1, beta = 0 (identity modulation at t=0).
    \"\"\"
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

class Critic(nn.Module):
    \"\"\"Conditional critic: [e, s_c] -> scalar score (hidden_dim=256, unchanged).\"\"\"
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
    z_dim      = WGAN_CONFIG['z_dim'],
    embed_dim  = WGAN_CONFIG['embed_dim'],
    hidden_dim = WGAN_CONFIG['hidden_dim'],    # 512
    film_hidden= WGAN_CONFIG['film_hidden'],   # 256
).to(device)

critic = Critic(embed_dim=WGAN_CONFIG['embed_dim']).to(device)

print(f"FiLMGenerator parameters:  {sum(p.numel() for p in generator.parameters()):,}")
print(f"Critic parameters:          {sum(p.numel() for p in critic.parameters()):,}")
print(f"  hidden_dim={WGAN_CONFIG['hidden_dim']}, film_hidden={WGAN_CONFIG['film_hidden']}")
print(f"  (Exp04 generator was 256,832 params — this is "
      f"{sum(p.numel() for p in generator.parameters())/256832:.2f}x larger)")
"""

# ──────────────────────────────────────────────────────────────────────────────
# CELL 44: TRAINING LOOP (single run, 50K steps)
# ──────────────────────────────────────────────────────────────────────────────
CELL_44 = """\
# =============================================================================
# cWGAN-GP TRAINING — High-Capacity FiLM + L_var (50K steps, alpha=1.0)
# =============================================================================
# Single training run. No sweep loop.
# Checkpoints saved every 10K steps (resilience against Colab disconnects).
# grad_cos_sim monitored every 5K steps throughout (vs first 5K only in Exp04).

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
alpha = WGAN_CONFIG['alpha']  # 1.0

torch.manual_seed(WGAN_CONFIG['seed'])
np.random.seed(WGAN_CONFIG['seed'])

g_opt = torch.optim.Adam(generator.parameters(), lr=WGAN_CONFIG['lr'], betas=WGAN_CONFIG['betas'])
c_opt = torch.optim.Adam(critic.parameters(),    lr=WGAN_CONFIG['lr'], betas=WGAN_CONFIG['betas'])

run = init_run(
    experiment_name="05_film_highcap",
    config_dict={**ENCODER_CONFIG, **WGAN_CONFIG},
    tags=["exp05", "film-conditioning", "highcap", "50k-steps", "lvar-alpha-1.0"],
)

os.makedirs('results', exist_ok=True)

g_losses, c_losses, gp_values = [], [], []
data_iter = iter(wgan_loader)
g_step    = 0

print(f"Training: FiLM-HighCap | hidden_dim={WGAN_CONFIG['hidden_dim']} | "
      f"film_hidden={WGAN_CONFIG['film_hidden']} | alpha={alpha} | "
      f"n_steps={WGAN_CONFIG['n_steps']:,}")
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
    l_wass     = -critic(e_fake, s_c).mean()
    l_var      = compute_lvar(e_fake, y_b)
    g_loss     = l_wass + alpha * l_var

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
            # Gradient conflict diagnostic — computed throughout training (not just first 5K)
            e_gs   = generator(torch.randn(batch_size, WGAN_CONFIG['z_dim'], device=device), s_c)
            lw_gs  = -critic(e_gs, s_c).mean()
            lv_gs  = compute_lvar(e_gs, y_b)
            if lv_gs.abs().item() > 1e-6:
                cos = grad_cosine_sim(generator, lw_gs, lv_gs)
                log_payload["train/grad_cos_sim"] = cos
                extra_str += f", grad_cos={cos:.3f}"
            # Checkpoint save every 10K (save at 5K, 10K, 15K, ...)
            if g_step % 10000 == 0:
                ckpt_path = f'results/generator_highcap_step{g_step}.pt'
                torch.save(generator.state_dict(), ckpt_path)
                extra_str += f"  [ckpt]"
            print(f"  step {g_step:6d}/{WGAN_CONFIG['n_steps']:,}: "
                  f"G={g_loss.item():.4f}, Lvar={l_var.item():.4f}{extra_str}")

        wandb.log(log_payload, step=g_step)

# ── Post-training ─────────────────────────────────────────────────────────────
torch.save(generator.state_dict(), 'results/generator_highcap_final.pt')
generator.eval()
print("=" * 70)
print(f"Training complete. Final model saved: results/generator_highcap_final.pt")
print(f"Mean G loss (last 1000 steps): {np.mean(g_losses[-1000:]):.4f}")
print(f"Mean D loss (last 1000 steps): {np.mean(c_losses[-1000:]):.4f}")
"""

# ──────────────────────────────────────────────────────────────────────────────
# CELL 45: Post-training VarR check (replaces lvar WandB reinit)
# ──────────────────────────────────────────────────────────────────────────────
CELL_45 = """\
# =============================================================================
# POST-TRAINING VarR CHECK (seen classes)
# =============================================================================
# Final VarR_seen after 50K steps — benchmark before full diagnostic suite.
# WandB run is still open; this logs to the same run as training.

VarR_seen_posttrain = compute_var_ratio_quick(
    generator, S_seen_prototypes, list(seen_classes),
    E_train_seen, label_train_seen, WGAN_CONFIG['z_dim'], device,
    n_cls=200, n_synth=20,
)
wandb.log({"eval/VarR_seen_post_training": VarR_seen_posttrain})

print(f"Post-training VarR_seen (200 classes, 20 synth each): {VarR_seen_posttrain:.5f}")
print("Reference: Exp04 lvar α=1.0 → VarR_seen=0.982 (at 10K steps)")
print("Reference: Exp04 baseline  → VarR_seen=0.985 (at 10K steps)")
"""

# ──────────────────────────────────────────────────────────────────────────────
# CELL 75: Harvest cell (remove sweep table, keep everything else)
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
    'experiment':   WGAN_CONFIG.get('experiment', 'unknown'),
    'hidden_dim':   WGAN_CONFIG.get('hidden_dim', 256),
    'film_hidden':  WGAN_CONFIG.get('film_hidden', 128),
    'n_steps':      WGAN_CONFIG.get('n_steps', 0),
    'alpha':        WGAN_CONFIG.get('alpha', 0.0),
    'H_mean_pct':   round(_get('H_gzsl', 0) * 100, 4),
    'AccS_pct':     round(_get('acc_seen_gzsl', 0) * 100, 4),
    'AccU_pct':     round(_get('acc_unseen_gzsl', 0) * 100, 4),
    'routing_pct':  round(_get('routing_rate', 0) * 100, 2),
    'VarR_seen':    round(float(_get('VarR_seen', 0)), 5),
    'VarR_unseen':  round(float(_get('VarR_unseen', 0)), 5),
    'VarR_gap':     round(float(_get('VarR_seen', 0)) - float(_get('VarR_unseen', 0)), 5),
    'rho_sp':       round(float(_get('rho_sp', 0)), 5),
    'kNN10':        round(float(_get('knn10', 0)), 5),
    'gamma1_mean':  round(float(_get('g1s.abs().mean().item()', 0)), 5),
    'gamma2_mean':  round(float(_get('g2s.abs().mean().item()', 0)), 5),
    'beta1_norm':   round(float(_get('b1s.norm(dim=-1).mean().item()', 0)), 5),
    'beta2_norm':   round(float(_get('b2s.norm(dim=-1).mean().item()', 0)), 5),
    'VarR_seen_posttrain': round(float(_get('VarR_seen_posttrain', 0)), 5),
    'g_loss_final_mean1k': round(float(np.mean(_get('g_losses', [0])[-1000:])), 5),
    'c_loss_final_mean1k': round(float(np.mean(_get('c_losses', [0])[-1000:])), 5),
}

print()
print('=' * 60)
print('RESULTS HARVEST — paste this block to Claude Code')
print('=' * 60)
print(_json.dumps(_results, indent=2))
print('=' * 60)
"""

# ── Apply all cell modifications ──────────────────────────────────────────────
cells[41]['source'] = CELL_41
cells[42]['source'] = CELL_42
# cells[43] unchanged (data prep with labels — correct for lvar/exp05)
cells[44]['source'] = CELL_44
cells[45]['source'] = CELL_45
cells[75]['source'] = CELL_75

# Write output notebook
with open(OUT, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Written: {OUT}")
print(f"Total cells: {len(cells)}")
print("Cells modified: 41, 42, 44, 45, 75")
print("Cell 43 unchanged (data prep with class labels)")
