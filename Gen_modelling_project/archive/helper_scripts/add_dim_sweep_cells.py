#!/usr/bin/env python3
"""
Removes hyperparameter sweep cells (67-80) from GZSL_EEG_Pipeline_v2.ipynb
and adds a focused 3-run full-pipeline embedding dimension sweep.

Resulting notebook: 71 cells (67 original + 4 new).
"""

import json, shutil
from datetime import datetime

NOTEBOOK = "GZSL_EEG_Pipeline_v2.ipynb"


def make_source(lines):
    """Join lines into notebook source format."""
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def make_code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": make_source(source_lines),
    }


def make_markdown_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": make_source(source_lines),
    }


def cell_67():
    """Markdown header for dim sweep."""
    return make_markdown_cell([
        "# Embedding Dimension — Full Pipeline Sweep",
        "",
        "The encoder-only sweep (Phases 1-4) found embed_dim nearly irrelevant for retrieval accuracy",
        "(64 ≈ 512). However, the downstream pipeline (WGAN-GP + classifier) operates differently at each",
        "dimension due to geometric crowding effects on the unit hypersphere.",
        "",
        "This sweep runs the **complete pipeline** (encoder → prototypes → WGAN-GP → synthesis → classifier → GZSL eval)",
        "for embed_dim ∈ {64, 128, 256}, using the optimal training dynamics from the hyperparameter sweep:",
        "- tau=0.05, epochs=75, lr=2e-3, wd=1e-4",
        "- Architecture: BrainEncoder [1024, 512], ImageProjector hidden=512 (both validated optimal)",
        "- WGAN-GP: 10000 steps, z_dim=100, unchanged from baseline",
    ])


def cell_68():
    """Full pipeline dim sweep - all 3 runs."""
    return make_code_cell([
        "# ═══════════════════════════════════════════════════════════════",
        "# EMBEDDING DIMENSION — FULL PIPELINE SWEEP",
        "# ═══════════════════════════════════════════════════════════════",
        "",
        "import time, gc, random",
        "from collections import Counter",
        "",
        "def set_seeds(seed=42):",
        "    np.random.seed(seed)",
        "    torch.manual_seed(seed)",
        "    random.seed(seed)",
        "    if torch.cuda.is_available():",
        "        torch.cuda.manual_seed_all(seed)",
        "",
        "def run_full_pipeline_dim(embed_dim, device,",
        "                          X_train_t, I_train_t, Y_train_t,",
        "                          X_test_t, I_test_t, Y_test_t,",
        "                          X_unseen_t, I_unseen_t, Y_unseen_t,",
        "                          lbl_train_seen, lbl_test_seen):",
        '    """Run complete pipeline for a given embed_dim. Returns results dict."""',
        "",
        "    t0 = time.time()",
        "    set_seeds(42)",
        "",
        "    # --- Encoder training (optimal dynamics) ---",
        "    tau, n_epochs, lr, wd = 0.05, 75, 0.002, 0.0001",
        "    bs, warmup_ratio, drop = 256, 0.1, 0.1",
        "",
        "    brain_enc = BrainEncoder(input_dim=X_train_t.shape[1],",
        "                             embed_dim=embed_dim, dropout=drop).to(device)",
        "    img_proj = ImageProjector(input_dim=I_train_t.shape[1],",
        "                              embed_dim=embed_dim, dropout=drop).to(device)",
        "",
        "    opt = torch.optim.AdamW(",
        "        list(brain_enc.parameters()) + list(img_proj.parameters()),",
        "        lr=lr, weight_decay=wd",
        "    )",
        "    ds = TensorDataset(X_train_t, I_train_t, Y_train_t)",
        "    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)",
        "    total_steps = n_epochs * len(loader)",
        "    warmup_steps = int(warmup_ratio * total_steps)",
        "",
        "    def lr_lambda(step):",
        "        if step < warmup_steps:",
        "            return step / max(warmup_steps, 1)",
        "        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)",
        "        return 0.5 * (1.0 + np.cos(np.pi * progress))",
        "",
        "    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)",
        "",
        "    brain_enc.train()",
        "    img_proj.train()",
        "    final_loss = 0",
        "    for ep in range(n_epochs):",
        "        ep_loss, nb = 0, 0",
        "        for xb, ib, yb in loader:",
        "            xb, ib = xb.to(device), ib.to(device)",
        "            e_brain = brain_enc(xb)",
        "            e_image = img_proj(ib)",
        "            loss = contrastive_loss(e_brain, e_image, tau)",
        "            opt.zero_grad()",
        "            loss.backward()",
        "            opt.step()",
        "            sched.step()",
        "            ep_loss += loss.item()",
        "            nb += 1",
        "        final_loss = ep_loss / nb",
        "",
        "    enc_time = time.time() - t0",
        "",
        "    # --- Compute embeddings ---",
        "    brain_enc.eval()",
        "    img_proj.eval()",
        "    with torch.no_grad():",
        "        E_tr = brain_enc(X_train_t.to(device)).cpu().numpy()",
        "        E_te = brain_enc(X_test_t.to(device)).cpu().numpy()",
        "        E_un = brain_enc(X_unseen_t.to(device)).cpu().numpy()",
        "        V_tr = img_proj(I_train_t.to(device)).cpu().numpy()",
        "        V_un = img_proj(I_unseen_t.to(device)).cpu().numpy()",
        "",
        "    # --- Prototypes ---",
        "    y_unseen_np = Y_unseen_t.numpy().astype(int)",
        "    S_seen_dict = compute_prototypes(V_tr, lbl_train_seen)",
        "    S_unseen_dict = compute_prototypes(V_un, y_unseen_np)",
        "    seen_cls = sorted(S_seen_dict.keys())",
        "    unseen_cls = sorted(S_unseen_dict.keys())",
        "    S_seen_arr = np.array([S_seen_dict[c] for c in seen_cls])",
        "    S_unseen_arr = np.array([S_unseen_dict[c] for c in unseen_cls])",
        "",
        "    # --- Top-1 retrieval ---",
        "    E_te_n = E_te / (np.linalg.norm(E_te, axis=1, keepdims=True) + 1e-8)",
        "    S_s_n = S_seen_arr / (np.linalg.norm(S_seen_arr, axis=1, keepdims=True) + 1e-8)",
        "    sims = E_te_n @ S_s_n.T",
        "    top1_preds = np.array(seen_cls)[np.argmax(sims, axis=1)]",
        "    enc_top1 = np.mean(top1_preds == lbl_test_seen)",
        "",
        "    print(f'  Encoder: loss={final_loss:.4f}, top1={enc_top1:.4f}, time={enc_time:.0f}s')",
        "",
        "    # --- WGAN-GP ---",
        "    t1 = time.time()",
        "    z_dim, wgan_lr, wgan_betas = 100, 1e-4, (0.0, 0.9)",
        "    lambda_gp, n_critic, n_steps, n_synth = 10, 5, 10000, 20",
        "",
        "    E_tr_t = torch.tensor(E_tr, dtype=torch.float32).to(device)",
        "    S_seen_t = torch.tensor(S_seen_arr, dtype=torch.float32).to(device)",
        "    S_unseen_t = torch.tensor(S_unseen_arr, dtype=torch.float32).to(device)",
        "",
        "    gen = Generator(z_dim=z_dim, embed_dim=embed_dim).to(device)",
        "    crt = Critic(embed_dim=embed_dim).to(device)",
        "    opt_G = torch.optim.Adam(gen.parameters(), lr=wgan_lr, betas=wgan_betas)",
        "    opt_C = torch.optim.Adam(crt.parameters(), lr=wgan_lr, betas=wgan_betas)",
        "",
        "    cls_to_idx = {c: np.where(lbl_train_seen == c)[0] for c in seen_cls}",
        "    seen_proto_t = {c: S_seen_t[i] for i, c in enumerate(seen_cls)}",
        "",
        "    gen.train()",
        "    crt.train()",
        "    final_G, final_C = 0, 0",
        "",
        "    for step in range(1, n_steps + 1):",
        "        # Critic updates",
        "        for _ in range(n_critic):",
        "            bc = np.random.choice(seen_cls, size=bs, replace=True)",
        "            ri = [np.random.choice(cls_to_idx[c]) for c in bc]",
        "            real = E_tr_t[ri]",
        "            cond = torch.stack([seen_proto_t[c] for c in bc])",
        "",
        "            z = torch.randn(bs, z_dim, device=device)",
        "            fake = gen(z, cond)",
        "",
        "            c_real = crt(real, cond)",
        "            c_fake = crt(fake.detach(), cond)",
        "            gp = compute_gradient_penalty(crt, real, fake.detach(), cond, device)",
        "            c_loss = c_fake.mean() - c_real.mean() + lambda_gp * gp",
        "",
        "            opt_C.zero_grad()",
        "            c_loss.backward()",
        "            opt_C.step()",
        "",
        "        # Generator update",
        "        bc = np.random.choice(seen_cls, size=bs, replace=True)",
        "        cond = torch.stack([seen_proto_t[c] for c in bc])",
        "        z = torch.randn(bs, z_dim, device=device)",
        "        fake = gen(z, cond)",
        "        g_loss = -crt(fake, cond).mean()",
        "",
        "        opt_G.zero_grad()",
        "        g_loss.backward()",
        "        opt_G.step()",
        "",
        "        final_G, final_C = g_loss.item(), c_loss.item()",
        "        if step % 5000 == 0:",
        "            print(f'    WGAN step {step}/{n_steps}: G={final_G:.4f}, C={final_C:.4f}')",
        "",
        "    wgan_time = time.time() - t1",
        "",
        "    # --- Synthesis ---",
        "    gen.eval()",
        "    unseen_proto_t = {c: S_unseen_t[i] for i, c in enumerate(unseen_cls)}",
        "    synth_list, synth_labels = [], []",
        "",
        "    with torch.no_grad():",
        "        for c in unseen_cls:",
        "            z = torch.randn(n_synth, z_dim, device=device)",
        "            cond = unseen_proto_t[c].unsqueeze(0).expand(n_synth, -1)",
        "            out = F.normalize(gen(z, cond), p=2, dim=-1)",
        "            synth_list.append(out.cpu().numpy())",
        "            synth_labels.extend([c] * n_synth)",
        "",
        "    E_synth = np.vstack(synth_list)",
        "    y_synth = np.array(synth_labels)",
        "",
        "    var_real = np.mean(np.var(E_tr, axis=0))",
        "    var_synth = np.mean(np.var(E_synth, axis=0))",
        "    print(f'  WGAN: G={final_G:.4f}, C={final_C:.4f}, time={wgan_time:.0f}s')",
        "    print(f'  Synth: {E_synth.shape}, var_real={var_real:.4f}, var_synth={var_synth:.4f}')",
        "",
        "    # --- Downsample synthetic to seen per-class median ---",
        "    med = int(np.median(list(Counter(lbl_train_seen).values())))",
        "    ds_idx = []",
        "    for c in unseen_cls:",
        "        c_idx = np.where(y_synth == c)[0]",
        "        if len(c_idx) > med:",
        "            c_idx = np.random.choice(c_idx, med, replace=False)",
        "        ds_idx.extend(c_idx)",
        "    E_synth_ds = E_synth[ds_idx]",
        "    y_synth_ds = y_synth[ds_idx]",
        "    print(f'  Downsample: {len(E_synth)} -> {len(E_synth_ds)} (median={med}/class)')",
        "",
        "    # --- Classifier ---",
        "    X_clf = np.vstack([E_tr, E_synth_ds])",
        "    y_clf = np.concatenate([lbl_train_seen, y_synth_ds])",
        "    clf = LogisticRegression(",
        "        multi_class='multinomial', solver='lbfgs', max_iter=2000,",
        "        class_weight='balanced', n_jobs=-1, verbose=0",
        "    )",
        "    clf.fit(X_clf, y_clf)",
        "",
        "    # --- GZSL evaluation ---",
        "    seen_set = set(seen_cls)",
        "    unseen_set = set(unseen_cls)",
        "    res = evaluate_gzsl(clf, E_te, lbl_test_seen, E_un, y_unseen_np,",
        "                        seen_set, unseen_set, phase_name=f'dim={embed_dim}')",
        "",
        "    # --- Weight norm diagnostics ---",
        "    all_classes = clf.classes_",
        "    s_mask = np.array([c in seen_set for c in all_classes])",
        "    u_mask = np.array([c in unseen_set for c in all_classes])",
        "    w_norms = np.linalg.norm(clf.coef_, axis=1)",
        "    wn_s, wn_u = w_norms[s_mask].mean(), w_norms[u_mask].mean()",
        "    print(f'  Weight norm ratio (U/S): {wn_u/wn_s:.2f}x')",
        "",
        "    total_time = time.time() - t0",
        "",
        "    res.update({",
        "        'embed_dim': embed_dim,",
        "        'encoder_loss': final_loss,",
        "        'encoder_top1': enc_top1,",
        "        'g_loss': final_G,",
        "        'c_loss': final_C,",
        "        'var_real': var_real,",
        "        'var_synth': var_synth,",
        "        'wn_ratio': wn_u / wn_s,",
        "        'total_time': total_time,",
        "    })",
        "",
        "    # Cleanup",
        "    del brain_enc, img_proj, gen, crt, E_tr_t, S_seen_t, S_unseen_t",
        "    if torch.cuda.is_available():",
        "        torch.cuda.empty_cache()",
        "    gc.collect()",
        "",
        "    return res",
        "",
        "# ═══════════════════════════════════════════════════════════════",
        "# RUN DIM SWEEP",
        "# ═══════════════════════════════════════════════════════════════",
        "",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
        "dims_to_test = [64, 128, 256]",
        "dim_results = []",
        "",
        "print('=' * 70)",
        "print('EMBEDDING DIMENSION — FULL PIPELINE SWEEP')",
        "print('Config: tau=0.05, epochs=75, lr=2e-3, wd=1e-4, arch=[512]')",
        "print('=' * 70)",
        "",
        "for dim in dims_to_test:",
        "    print(f'\\n[dim={dim}] Running full pipeline...')",
        "    res = run_full_pipeline_dim(",
        "        embed_dim=dim, device=device,",
        "        X_train_t=X_train_tensor, I_train_t=I_train_tensor,",
        "        Y_train_t=Y_train_tensor,",
        "        X_test_t=X_test_tensor, I_test_t=I_test_tensor,",
        "        Y_test_t=Y_test_tensor,",
        "        X_unseen_t=X_unseen_tensor, I_unseen_t=I_unseen_tensor,",
        "        Y_unseen_t=Y_unseen_tensor,",
        "        lbl_train_seen=label_train_seen, lbl_test_seen=label_test_seen",
        "    )",
        "    dim_results.append(res)",
        "    print(f'  => AccS={res[\"acc_seen\"]:.4f}, AccU={res[\"acc_unseen\"]:.4f}, '",
        "          f'H={res[\"H\"]:.4f}, Routing={res[\"routing_rate\"]:.4f}')",
        "",
        "print('\\n' + '=' * 70)",
        "print('DIM SWEEP COMPLETE')",
        "print('=' * 70)",
    ])


def cell_69():
    """Results comparison table + figures."""
    return make_code_cell([
        "# ═══════════════════════════════════════════════════════════════",
        "# DIM SWEEP — RESULTS COMPARISON",
        "# ═══════════════════════════════════════════════════════════════",
        "",
        "import matplotlib.pyplot as plt",
        "",
        "print('=' * 80)",
        "print('EMBEDDING DIMENSION — FULL PIPELINE COMPARISON')",
        "print('=' * 80)",
        "print()",
        "header = f'{\"Metric\":<20} {\"dim=64\":>10} {\"dim=128\":>10} {\"dim=256\":>10}'",
        "print(header)",
        "print('-' * 55)",
        "",
        "metrics_list = [",
        "    ('AccS', 'acc_seen'),",
        "    ('AccU', 'acc_unseen'),",
        "    ('H-mean', 'H'),",
        "    ('F1 Seen', 'f1_seen'),",
        "    ('F1 Unseen', 'f1_unseen'),",
        "    ('Routing Rate', 'routing_rate'),",
        "    ('Encoder top-1', 'encoder_top1'),",
        "    ('Encoder loss', 'encoder_loss'),",
        "    ('G_loss', 'g_loss'),",
        "    ('WN ratio', 'wn_ratio'),",
        "    ('Var (real)', 'var_real'),",
        "    ('Var (synth)', 'var_synth'),",
        "    ('Time (s)', 'total_time'),",
        "]",
        "",
        "for name, key in metrics_list:",
        "    vals = [r[key] for r in dim_results]",
        "    print(f'{name:<20} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f}')",
        "",
        "best_idx = max(range(len(dim_results)), key=lambda i: dim_results[i]['H'])",
        "best_dim = dim_results[best_idx]['embed_dim']",
        "best_h = dim_results[best_idx]['H']",
        "print()",
        "print(f'Best H-mean: dim={best_dim} -> H={best_h:.4f} ({best_h*100:.2f}%)')",
        "print()",
        "",
        "# Original baseline reference (tau=0.15, epochs=50, lr=1e-3, dim=128)",
        "print('Comparison with original baseline (cells 28-66):')",
        "print(f'  Baseline H-mean:  0.0453 (tau=0.15, ep=50, lr=1e-3, dim=128)')",
        "print(f'  Best dim sweep:   {best_h:.4f} (tau=0.05, ep=75, lr=2e-3, dim={best_dim})')",
        "print(f'  Delta:            {best_h - 0.0453:+.4f} ({(best_h - 0.0453)/0.0453*100:+.1f}%)')",
        "print('=' * 80)",
        "",
        "# --- Figures ---",
        "fig, axes = plt.subplots(1, 3, figsize=(15, 5))",
        "",
        "dims = [r['embed_dim'] for r in dim_results]",
        "x = np.arange(len(dims))",
        "w = 0.25",
        "",
        "# Panel 1: AccS, AccU, H-mean grouped bar",
        "ax = axes[0]",
        "accS = [r['acc_seen'] for r in dim_results]",
        "accU = [r['acc_unseen'] for r in dim_results]",
        "hmean = [r['H'] for r in dim_results]",
        "ax.bar(x - w, accS, w, label='AccS', color='steelblue')",
        "ax.bar(x, accU, w, label='AccU', color='coral')",
        "ax.bar(x + w, hmean, w, label='H-mean', color='seagreen')",
        "ax.set_xticks(x)",
        "ax.set_xticklabels([str(d) for d in dims])",
        "ax.set_xlabel('Embedding Dimension')",
        "ax.set_ylabel('Accuracy')",
        "ax.set_title('GZSL Metrics by Dimension')",
        "ax.legend()",
        "for i, h in enumerate(hmean):",
        "    ax.annotate(f'{h:.4f}', (x[i] + w, h), ha='center', va='bottom', fontsize=9)",
        "",
        "# Panel 2: Routing rate",
        "ax = axes[1]",
        "routing = [r['routing_rate'] for r in dim_results]",
        "ax.bar(x, routing, color='darkorange')",
        "ax.axhline(y=200/1854, color='gray', linestyle='--', label=f'Ideal ({200/1854:.3f})')",
        "ax.set_xticks(x)",
        "ax.set_xticklabels([str(d) for d in dims])",
        "ax.set_xlabel('Embedding Dimension')",
        "ax.set_ylabel('Routing Rate')",
        "ax.set_title('Seen→Unseen Routing Rate')",
        "ax.legend()",
        "for i, r in enumerate(routing):",
        "    ax.annotate(f'{r:.3f}', (x[i], r), ha='center', va='bottom', fontsize=9)",
        "",
        "# Panel 3: Encoder top-1 vs H-mean",
        "ax = axes[2]",
        "enc_top1 = [r['encoder_top1'] for r in dim_results]",
        "ax.scatter(enc_top1, hmean, s=100, c='purple', zorder=5)",
        "for i, d in enumerate(dims):",
        "    ax.annotate(f'd={d}', (enc_top1[i], hmean[i]),",
        "                textcoords='offset points', xytext=(10, 5), fontsize=10)",
        "ax.set_xlabel('Encoder Top-1 Retrieval')",
        "ax.set_ylabel('GZSL H-mean')",
        "ax.set_title('Encoder Quality vs GZSL Performance')",
        "ax.axhline(y=0.0453, color='gray', linestyle='--', alpha=0.5, label='Baseline H-mean')",
        "ax.legend()",
        "",
        "plt.tight_layout()",
        "plt.savefig('figures/dim_sweep_full_pipeline.png', dpi=150, bbox_inches='tight')",
        "plt.show()",
        "print('Saved: figures/dim_sweep_full_pipeline.png')",
    ])


def cell_70():
    """Markdown footer."""
    return make_markdown_cell([
        "---",
        "**Dim sweep complete.** Results above determine the final encoder configuration",
        "before proceeding to structure-preserving WGAN-GP research.",
    ])


def run_checks(cells, n_final):
    """Run sanity checks on the modified notebook."""
    checks = [
        ("Cell count == 71", n_final == 71),
        ("Cell 67 is markdown", cells[67]["cell_type"] == "markdown"),
        ("Cell 68 has run_full_pipeline_dim",
         "run_full_pipeline_dim" in "".join(cells[68]["source"])),
        ("Cell 68 has dims_to_test",
         "dims_to_test" in "".join(cells[68]["source"])),
        ("Cell 68 has evaluate_gzsl",
         "evaluate_gzsl" in "".join(cells[68]["source"])),
        ("Cell 68 has compute_prototypes",
         "compute_prototypes" in "".join(cells[68]["source"])),
        ("Cell 68 has WGAN training loop",
         "n_critic" in "".join(cells[68]["source"])),
        ("Cell 69 has comparison table",
         "COMPARISON" in "".join(cells[69]["source"]).upper()),
        ("Cell 69 has figure save",
         "dim_sweep_full_pipeline.png" in "".join(cells[69]["source"])),
        ("Cell 70 is markdown", cells[70]["cell_type"] == "markdown"),
    ]

    print("\n--- Sanity Checks ---")
    all_pass = True
    for name, result in checks:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")
        if not result:
            all_pass = False
    return all_pass


def main():
    with open(NOTEBOOK, "r") as f:
        nb = json.load(f)

    # Backup
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{NOTEBOOK}.backup.dimsweep.{ts}"
    shutil.copy2(NOTEBOOK, backup)
    print(f"Backup: {backup}")

    cells = nb["cells"]
    n_before = len(cells)

    # Remove cells 67-80 (the previous sweep, 14 cells)
    cells_to_remove = list(range(67, min(81, n_before)))
    for idx in sorted(cells_to_remove, reverse=True):
        if idx < len(cells):
            del cells[idx]

    n_after_remove = len(cells)
    print(f"Removed sweep cells: {n_before} -> {n_after_remove} (-{n_before - n_after_remove})")

    # Add dim sweep cells
    new_cells = [cell_67(), cell_68(), cell_69(), cell_70()]
    cells.extend(new_cells)

    n_final = len(cells)
    print(f"Added dim sweep cells: {n_after_remove} -> {n_final} (+{len(new_cells)})")

    # Sanity checks
    if not run_checks(cells, n_final):
        print("\nSome checks failed! Review before proceeding.")
        return

    print("\nAll sanity checks passed.")

    nb["cells"] = cells
    with open(NOTEBOOK, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\nNotebook written: {NOTEBOOK}")
    print("Done!")


if __name__ == "__main__":
    main()
