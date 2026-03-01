# AGENT CONTEXT: BraVL EEG–Text CLIP + cWGAN-GP (Notebook-Only)
**Purpose.** This document is the authoritative implementation guide for the **CLIP encoder** and **conditional WGAN-GP** components of the project. It must be followed to keep the notebook (`COMP2261_ArizMLCW_with_baseline.ipynb`) aligned with the report’s **Model Development** section and the earlier **Data & Paradigm** section.

**Hard constraints**
- **Notebook-only**: implement everything in **new cells** in the notebook. Do **not** create external `.py` files.
- Do **not** modify/delete existing data-loading cells; build on the loaded arrays.
- **No leakage**: do not use unseen EEG trials for training any model. Unseen EEG is **test-only**.
- Keep outputs reproducible: set seeds; log hyperparameters; save key figures.

---

## 0) What the dataset provides (BraVL as loaded in the notebook)
Expected arrays (from baseline summary):
- `brain_seen`: shape `(16540, 561)` — EEG features for seen classes  
- `brain_unseen`: shape `(16000, 561)` — EEG features for unseen classes (**test-only**)  
- `text_seen`: shape `(16540, 512)` — text embeddings paired with seen trials  
- `text_unseen`: shape `(16000, 512)` — text embeddings paired with unseen trials (semantic side info)  
- `label_seen`: shape `(16540, 1)` — integer labels for seen trials (1-indexed)  
- `label_unseen`: shape `(16000, 1)` — integer labels for unseen trials (1-indexed)

Other known facts:
- Seen classes: `|S|=1654` classes, **10 trials per class**
- Unseen classes: `|U|=200` classes, **80 trials per class**
- Baseline split: 80/20 stratified split **within seen** (`13232` train, `3308` test)

---

## 1) Sources used as inspiration (must be cited in the report; agent should follow the design intent)
1) **EEG–Text CLIP**: Frontiers EEG-CLIP article (architecture + CLIP-style alignment idea)
   - Key idea: dual encoders + projection heads + contrastive loss aligning EEG and text in a shared space.
   - They use Deep4 CNN (for raw EEG time series) and BERT (text), projecting to low-dim space.
   - In BraVL we do **MLPs** because EEG is already a 561-D feature vector and text is already 512-D embeddings.

2) **Deep4** (Schirrmeister et al., 2017/2018 arXiv 1708.08012)
   - Canonical EEG deep representation learning reference.
   - We cite it for EEG representation learning rationale (not necessarily reusing its CNN because BraVL is feature-based).

3) **cWGAN-GP external framework**
   - GitHub: `georgehalal/cWGAN-GP` (the included ZIP mirrors the implementation style)
   - We adapt it to **conditional embedding synthesis** in the CLIP semantic space.

---

## 2) Component A: EEG–Text CLIP (dual encoder, shared semantic space)
### 2.1 Goal and operating space
We learn:
- Brain encoder: `f_b: R^561 -> R^d`
- Text projector: `g: R^512 -> R^d`

Outputs live in a shared embedding space `R^d`, and are **L2-normalised**:
- `e = normalize(f_b(b))`
- `s = normalize(g(t))`

**Default choice:** `d = 64` (keeps model light and aligns with EEG-CLIP practice).  
If compute is abundant, `d = 128` is acceptable; must be logged.

### 2.2 Data used for CLIP training
Train CLIP encoders only on seen train split:
- Use `brain_train_seen`, `text_train_seen`, `label_train_seen` from the notebook’s 80/20 split.
- Do **not** train on `brain_unseen`.

### 2.3 Preprocessing
- Apply `StandardScaler` to EEG features **fit on train_seen only**.
- Apply the same scaler to `test_seen` and `unseen` when producing embeddings for evaluation.

Text embeddings (`text_seen`, `text_unseen`) are already numeric vectors; do **not** standardise them globally unless explicitly justified.  
If you add a text projector `g`, it can learn any needed scaling.

### 2.4 Architecture (MLP encoders; minimal but stable)
**Brain encoder `f_b`**
- Input: 561
- Hidden: 1024 -> 512 (ReLU + Dropout optional 0.1–0.3)
- Output: `d`
- Example: Linear(561,1024) → ReLU → Linear(1024,512) → ReLU → Linear(512,d)

**Text projector `g`**
- Input: 512
- Hidden: 512 (or 256)
- Output: `d`
- Example: Linear(512,512) → ReLU → Linear(512,d)

**Normalisation**
- After projection, do: `x = x / (||x||_2 + eps)`

### 2.5 Loss: CLIP / InfoNCE (symmetric)
For a batch of size `B`, compute similarity matrix:
- `S_ij = (e_i · s_j) / tau` (dot product equals cosine similarity due to L2 norm)
- `tau` = temperature; default `tau=0.07` (can be learnable scalar)

Targets: `y = [0..B-1]`

Loss:
- `L_e2t = CE(softmax over rows of S, targets)`
- `L_t2e = CE(softmax over cols of S, targets)`
- `L_clip = (L_e2t + L_t2e)/2`

### 2.6 Training strategy (defaults)
- Optimiser: AdamW (or Adam)  
  - `lr = 1e-4` (start), `weight_decay = 1e-4`
- Batch size: `B = 256` (use 128 if memory-limited)
- Epochs: `20` (start). If convergence is slow: 30–50.
- Scheduler (optional): cosine or step LR; keep simple if limited time.
- Early stopping (optional): monitor validation contrastive loss (use a small split from train_seen).

Logging required:
- `d`, `tau`, `lr`, batch size, epochs, dropout, weight decay.
- Plot training loss curve and save: `figures/clip_loss_curve.png`

### 2.7 Post-training artifacts to compute and cache
After CLIP training, compute embeddings:
- `E_train_seen = normalize(f_b(X_train_seen_scaled))` shape `(n_train_seen, d)`
- `E_test_seen = normalize(f_b(X_test_seen_scaled))` shape `(n_test_seen, d)`
- `E_unseen = normalize(f_b(X_unseen_scaled))` shape `(n_unseen, d)`  (**inference only; encoder fixed**)

Compute semantic prototypes:
- For each class `c` in seen: `s_c = mean_{i: y_i=c} normalize(g(t_i))` then renormalise.
- For unseen classes: same prototype computation from `text_unseen` (by their labels), no EEG used.

Cache (recommended):
- Save tensors/arrays to disk in notebook (e.g., `np.save`):
  - `E_train_seen.npy`, `E_test_seen.npy`, `E_unseen.npy`
  - `S_seen_prototypes.npy`, `S_unseen_prototypes.npy`
- This avoids retraining CLIP every run.

### 2.8 Visual diagnostics (must generate and save)
- PCA or t-SNE scatter of:
  - `E_test_seen` (subset of points) and prototypes `S_seen`
  - include unseen prototypes `S_unseen` (no unseen EEG points required here, but optional to include `E_unseen` later)
- Save: `figures/clip_embedding_tsne.png`
- Ensure code is deterministic (fixed random_state).

---

## 3) Component B: cWGAN-GP for conditional embedding synthesis
### 3.1 Goal in this project
Train a conditional WGAN-GP in the **CLIP embedding space** `R^d` to model:
- Real distribution: `e ~ p_data(e | c)` where `e = f_b(b)` for seen classes
- Generator distribution: `e_hat = G(z, s_c)` where `s_c` is the semantic prototype for class `c`

Then, for each unseen class `u`:
- Generate synthetic EEG embeddings: `e_hat ~ p_G(e | s_u)`
- Use these as training data for a **standard classifier** in GZSL.

### 3.2 Data used for WGAN training
**Only seen embeddings**:
- Real samples: `E_train_seen` paired with their class ids `y_train_seen`
- Condition vectors: `s_{y_i}` using seen prototypes from CLIP text projector
- Do **not** use `E_unseen` for training.

### 3.3 Model interfaces and dimensional constraints
Let:
- `d` = embedding dimension (default 64)
- `z_dim` = noise dimension (default 100)

**Generator**
- Input: concatenation `[z, s_c]` → dimension `(z_dim + d)`
- Output: `e_hat` dimension `(d)`
- Apply L2 normalisation at output to keep generated embeddings on/near unit sphere:
  - `e_hat = normalize(e_hat)`

**Critic (conditional)**
- Input: concatenation `[e, s_c]` → dimension `(2d)`
- Output: scalar score (real-valued), **no sigmoid**.

### 3.4 Loss: conditional WGAN-GP
Critic objective (to maximise):
- `E[D(e_real, s_c)] - E[D(e_fake, s_c)] - lambda_gp * GP`

Generator objective (to minimise):
- `L_G = - E[D(e_fake, s_c)]`

Gradient penalty:
- Sample interpolation: `e_bar = eps*e_real + (1-eps)*e_fake` with `eps~U(0,1)`
- `GP = E[(||∇_{e_bar} D(e_bar, s_c)||_2 - 1)^2]`

### 3.5 Training strategy (defaults consistent with WGAN-GP practice + external repo style)
- Optimiser: Adam  
  - `lr = 1e-4`  
  - `betas = (0.0, 0.9)` (preferred for WGAN-GP stability; if unstable, try (0.5, 0.9))
- `lambda_gp = 10`
- `n_critic = 5` (5 critic updates per generator update)
- Batch size: 256 (or 128)
- Steps/epochs: because this is embedding synthesis, prefer **steps-based** training:
  - `n_steps = 10_000` generator steps (with 5× critic steps each) as a starting point
  - If using epoch terminology, define epochs over `E_train_seen` and target 50–200 epochs depending on batch size.
- Logging:
  - critic loss components (real, fake, GP)
  - generator loss
- Save curves: `figures/wgan_losses.png`

### 3.6 How many synthetic embeddings to generate for unseen classes
You must pick `n_synth_per_class` for unseen classes. Recommendation:
- Start with `n_synth_per_class = 10` or `20` (small, comparable to seen training density).
- Try ablation later: `{5, 10, 20, 50}`.

Total synthetic unseen samples:
- `n_synth_total = |U| * n_synth_per_class` (e.g., 200*20 = 4000)

### 3.7 Sanity checks (required)
Before training classifier:
- For seen classes, generate a small batch and compare:
  - cosine similarity between real embeddings and generated embeddings
  - optional MMD or simple distribution stats (mean/std per dimension)
- For unseen classes, verify:
  - generated embeddings are not collapsing (variance > 0)
  - critic scores are not exploding

Optional visual:
- t-SNE with:
  - real seen embeddings
  - synthetic unseen embeddings
  - prototypes (stars)
Save: `figures/real_vs_synth_tsne.png`

---

## 4) How CLIP and cWGAN-GP connect in the notebook (sequential cell plan)
Agent should add notebook cells in this order (new headings):

1) **CLIP: Setup**
   - set seeds, choose `d`, define MLPs, optimisers, dataloaders

2) **CLIP: Train**
   - train loop for `L_clip`
   - loss curves

3) **CLIP: Embed + Prototypes**
   - compute and cache `E_train_seen`, `E_test_seen`, `E_unseen`
   - compute and cache prototypes `S_seen`, `S_unseen`

4) **WGAN: Setup**
   - define `G`, `D`, gp function, optimisers

5) **WGAN: Train**
   - train critic/gen with `n_critic=5`, `lambda_gp=10`
   - loss curves

6) **WGAN: Synthesize**
   - generate synthetic unseen embeddings
   - store arrays (for classifier step)

(Do not implement the final classifier here unless asked; this doc is for CLIP + cWGAN.)

---

## 5) Reporting alignment (what this enables in Model Development)
- Clear mathematical definitions for:
  - `f_b`, `g`, `L_clip`
  - `G`, `D`, `GP`, `L_G`
- Theoretical expectations to write (no results here):
  - CLIP alignment produces semantically structured, normalised embeddings.
  - Conditional WGAN-GP provides stable conditional synthesis and mitigates seen-class bias by supplying unseen-class training mass in embedding space.
  - L2-normalisation makes dot product equivalent to cosine similarity, consistent with contrastive learning geometry.

---

## 6) Deliverables the agent must output (files + printed summaries)
**Figures to save**
- `figures/clip_loss_curve.png`
- `figures/clip_embedding_tsne.png`
- `figures/wgan_losses.png`
- `figures/real_vs_synth_tsne.png` (optional but recommended)

**Cached arrays**
- `E_train_seen.npy`, `E_test_seen.npy`, `E_unseen.npy`
- `S_seen_prototypes.npy`, `S_unseen_prototypes.npy`
- `E_synth_unseen.npy` (+ corresponding `y_synth_unseen.npy`)

**Console summary**
- hyperparameters chosen
- final CLIP loss (train/val)
- WGAN final losses (critic, GP, generator)
- generated sample counts per unseen class

---

## 7) Parameters (single place to define)
Agent should define a single config dict at top of CLIP and WGAN sections, e.g.:

- `embed_dim d = 64`
- `clip_tau = 0.07`
- `clip_epochs = 20`
- `clip_batch = 256`
- `clip_lr = 1e-4`
- `wgan_z_dim = 100`
- `wgan_lr = 1e-4`
- `wgan_betas = (0.0, 0.9)`
- `lambda_gp = 10`
- `n_critic = 5`
- `wgan_steps = 10000`
- `n_synth_per_class = 20`

This configuration must be printed verbatim for reproducibility.

---
