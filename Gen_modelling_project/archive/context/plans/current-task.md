# Current Task: EEG-to-Image (CORnet-S) Alignment — Paradigm Shift

**Status**: READY FOR EXECUTION
**Date**: 2026-03-03
**Notebook**: `GZSL_EEG_Pipeline_v2.ipynb` (67 cells — already modified by helper script)
**Priority**: Critical path — this is the highest-impact intervention available

---

## 1. Context: Why We're Doing This

### 1.1 The Problem — Text Alignment Ceiling

After exhaustive optimisation across Phases E and F, the EEG-to-text alignment pipeline hit a hard ceiling at ~3% top-1 accuracy on 1654-way seen-only classification (2.87% best, Phase E). Phase F tested every available lever:

| Intervention | Result | Conclusion |
|-------------|--------|------------|
| Augmentation (channel drop, noise, mixup) | 0.24% top-1 (worse) | EEG augmentation hurts more than helps at this resolution |
| SupCon (supervised contrastive) | 0.06% (random chance) | Embedding collapse — 1654 classes is too many for SupCon |
| EEGNet / ShallowConvNet | Worse than MLP | 17×33 resolution insufficient for convolutional extraction |
| All combined | No improvement | ~3% is a hard ceiling for EEG→text |

**Root cause**: The text (CLIP) alignment target itself is the bottleneck. Text embeddings for 1654 object categories have **mean inter-class cosine similarity of 0.668** — i.e., all class prototypes are clustered in a tight cone. The encoder can't learn to separate classes that the target space doesn't separate.

### 1.2 Visual Investigation Results

We ran a visual feature investigation (now archived in `context/Visual_investigation_results.md`) comparing the available alignment targets:

| Property | Visual (CORnet-S) | Text (CLIP) | Winner |
|----------|-------------------|-------------|--------|
| Inter-class cosine similarity | **-0.001** | 0.668 | **Visual** (near-orthogonal classes) |
| Effective rank at 95% variance | **335-D** | 169-D | **Visual** (2x richer supervision) |
| Feature dimensions | **1000-D** | 512-D | **Visual** (more information) |
| Cross-modal correlation with EEG | r = 0.0006 | r = -0.033 | Comparable (both near-zero pre-projection) |

**Key insight**: Visual features provide dramatically better class separation AND a richer supervision signal. The near-orthogonality means the encoder has room to learn fine-grained distinctions.

### 1.3 Literature Support

NICE-EEG (Song et al., ICLR 2024) achieves **15.6% top-1 on 200-way ZSL** using EEG-to-CLIP-image alignment — a 5x improvement over text-based approaches. While they use CLIP image embeddings (not CORnet-S), the fundamental insight is the same: **image features are a superior alignment target for brain decoding**.

### 1.4 CORnet-S vs CLIP Image Features

CORnet-S features are what's available in our dataset. They are biologically-inspired CNN features (recurrent ventral stream model), not CLIP-trained. Key differences:

- **CORnet-S**: 1000-D PCA features, biologically plausible, NOT trained with text-image contrastive learning
- **CLIP Image**: 512-D or 768-D, trained with text-image contrastive learning, shares structure with CLIP text

We're using CORnet-S first because:
1. It's already in the dataset — no external sourcing needed
2. It has excellent class separation (inter-class cosine ~ -0.001)
3. It establishes the image alignment infrastructure
4. If CORnet-S works, CLIP image features (which have even more structure) should work even better

---

## 2. What Changed (Helper Script Already Ran)

The helper script `helper_files/switch_to_image_alignment.py` has **already been executed** on the notebook. The notebook is ready for Colab execution. Below is exactly what changed, cell by cell, so you can verify correctness.

### 2.1 Summary of Changes

| Cell | Section | Change |
|------|---------|--------|
| 7 | Data loading | Removed `image_seen[:,0:100]` and `image_unseen[:, 0:100]` PCA truncation → full 1000-D |
| 28 | Markdown | "Brain-Text CLIP Encoder" → "Brain-Image Contrastive Encoder (CORnet-S)" with paradigm shift rationale |
| 29 | Config | `CLIP_CONFIG` → `ENCODER_CONFIG` with `image_input_dim: 1000` and `alignment_target: 'image (CORnet-S)'` |
| 30 | Models | `TextProjector` → `ImageProjector` (1000→512→128 with LayerNorm), `clip_loss` → `contrastive_loss` |
| 31 | Data prep | `T_*_tensor` (text) → `I_*_tensor` (image), DataLoader pairs brain+image |
| 32 | Training | Uses `image_projector` + `contrastive_loss`, references `ENCODER_CONFIG` |
| 33 | Loss curve | `clip_losses` → `encoder_losses`, updated titles/filename |
| 34 | Embeddings | Added `V_train_embeds`, `V_unseen_embeds` via `image_projector` |
| 35 | Prototypes | Computes from `V_train_embeds`/`V_unseen_embeds` (image) instead of text |
| 36 | Caching | Saves `V_train_embeds.npy`, `V_unseen_embeds.npy` |
| 37 | t-SNE | "Text Prototypes" → "Image Prototypes (CORnet-S)" |
| 38 | WGAN markdown | "text prototypes" → "image prototypes" |
| 47 | WGAN t-SNE | "CLIP Space" → "Brain-Image Space" |
| 48 | Summary | All CLIP references → Brain-Image/ENCODER_CONFIG references |
| 49 | Classifier markdown | "CLIP encoder" → "Brain-Image encoder" |
| 61 | Final summary | "Brain-Text CLIP" → "Brain-Image (CORnet-S)", "text" → "image" |
| 67-75 | Visual investigation | **DELETED** (9 cells removed, investigation complete) |

### 2.2 What Did NOT Change

These components are **identical** to the text alignment version:

- **BrainEncoder** (cell 30): Same architecture (561→1024→512→128, LayerNorm, L2-norm). Untouched.
- **WGAN-GP code** (cells 39-46): Generator, Critic, training loop, synthesis — all use `S_seen_prototypes` and `S_unseen_prototypes` by name. These variables are now populated from image embeddings upstream, so the WGAN-GP automatically operates in image prototype space without any code changes.
- **GZSL classifier** (cells 50-57): LogReg training with sample balancing. Uses `E_train_seen`, `E_synth_unseen` by name — unchanged.
- **Evaluation harness** (cells 62-66): `evaluate_gzsl`, `diagnose_classifier` — unchanged.
- **Sample balancing** (cells 51-52): Downsampling + `class_weight='balanced'` — preserved exactly as is.

---

## 3. Architecture Details

### 3.1 New Data Flow

```
Raw EEG (561-D)         ──► BrainEncoder f_b    ──► Shared space (R^128)
CORnet-S Image (1000-D) ──► ImageProjector g_v  ──► Shared space (R^128)
                                                          │
                                ┌──────────────────────────┤
                                ▼                          ▼
                       cWGAN-GP trains on          Image prototypes
                       seen brain embeddings       for all classes
                                │                          │
                                ▼                          ▼
                       Generator G(z, s_c) ──► Synthetic unseen embeddings
                                │
                                ▼
                       LogReg on real seen + synthetic unseen
                                │
                                ▼
                       GZSL metrics: AccS, AccU, H-mean
```

### 3.2 ImageProjector Architecture

```
Input (1000-D CORnet-S PCA)
  │
  ├── Linear(1000, 512)     ← 512,000 weights + 512 biases
  ├── LayerNorm(512)         ← 512 + 512 (gain + bias)
  ├── ReLU
  ├── Dropout(0.1)
  │
  ├── Linear(512, 128)       ← 65,536 weights + 128 biases
  │
  └── L2-normalize           ← output lives on unit sphere S^127
```

**Total parameters**: ~579,200

**Design rationale**:
- **2-layer MLP**: Mirrors original TextProjector depth. CORnet-S PCA features are clean and structured — no need for BrainEncoder's 3-layer depth.
- **LayerNorm**: Added because variance is concentrated in early PCA components (first 100 dims = 32.4% variance, 95% at 913 dims). Without LN, early components dominate the gradient signal.
- **Dropout 0.1**: Same regularisation as BrainEncoder. With 1000-D input and only 13,232 training samples, overfitting is a real risk — the first layer alone has 512K parameters.
- **L2-normalised output**: Stays on unit sphere, matches BrainEncoder. Both embeddings live in S^127, enabling cosine similarity as the distance metric.
- **No StandardScaling on input**: Image features are already PCA-whitened (mean ~0, std ~0.06 after x50.0 scaling). Additional scaling would be redundant.

### 3.3 BrainEncoder (Unchanged)

```
Input (561-D EEG)
  │
  ├── Linear(561, 1024) → LayerNorm(1024) → ReLU → Dropout(0.1)
  ├── Linear(1024, 512) → LayerNorm(512)  → ReLU → Dropout(0.1)
  ├── Linear(512, 128)
  │
  └── L2-normalize
```

**Total parameters**: ~1,123,200

### 3.4 Contrastive Loss (Renamed from clip_loss)

Same symmetric InfoNCE loss, just renamed for clarity:

```
L = (L_{b→i} + L_{i→b}) / 2

where:
  L_{b→i} = CrossEntropy(brain_embeds @ image_embeds.T / τ, arange(B))
  L_{i→b} = CrossEntropy(image_embeds @ brain_embeds.T / τ, arange(B))
```

Temperature τ = 0.15 (Phase E optimised). This is a "soft" temperature that prevents hard negatives from dominating — important because EEG is noisy and many categories are visually similar.

### 3.5 Hyperparameters (ENCODER_CONFIG)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `embed_dim` | 128 | Phase E optimised (was 64) |
| `image_input_dim` | 1000 | Full CORnet-S PCA dimensionality |
| `tau` | 0.15 | Phase E optimised (was 0.07) |
| `epochs` | 50 | Phase E optimised (was 20) |
| `batch_size` | 256 | Phase E optimised |
| `lr` | 1e-3 | Phase E optimised (was 1e-4) |
| `weight_decay` | 1e-4 | AdamW regularisation |
| `dropout` | 0.1 | Moderate regularisation |
| `schedule` | cosine_warmup | 10% warmup steps |
| `seed` | 42 | Reproducibility |

These are **identical** to the Phase E text alignment config except for `image_input_dim` and `alignment_target`. We start with the same hyperparameters because they were validated to work well for this problem. If results are unsatisfactory, hyperparameter tuning is a follow-up task.

---

## 4. How Prototypes Flow Through the Pipeline

This is the critical mechanism that makes the paradigm shift work with minimal code changes:

### 4.1 Old Pipeline (Text)

```
Cell 34: T_train_embeds = text_projector(T_train_tensor)    ← text in 128-D
Cell 35: S_seen_prototypes = compute_prototypes(T_train_embeds, labels)  ← dict
         S_unseen_prototypes = compute_prototypes(T_unseen_embeds, labels)
         S_seen_array = [S_seen_prototypes[c] for c in seen_classes]     ← (1654, 128)
         S_unseen_array = [S_unseen_prototypes[c] for c in unseen_classes] ← (200, 128)
Cell 41: WGAN data prep uses S_seen_array (pairs real embeds with prototypes)
Cell 44: Synthesis uses S_unseen_array (generates from unseen prototypes)
Cell 51: Classifier combines real seen + synthetic unseen
```

### 4.2 New Pipeline (Image)

```
Cell 34: V_train_embeds = image_projector(I_train_tensor)    ← image in 128-D
Cell 35: S_seen_prototypes = compute_prototypes(V_train_embeds, labels)  ← SAME VARIABLE NAME
         S_unseen_prototypes = compute_prototypes(V_unseen_embeds, labels)
         S_seen_array = [S_seen_prototypes[c] for c in seen_classes]     ← SAME VARIABLE NAME
         S_unseen_array = [S_unseen_prototypes[c] for c in unseen_classes] ← SAME VARIABLE NAME
Cell 41: WGAN data prep uses S_seen_array ← AUTOMATICALLY USES IMAGE PROTOTYPES
Cell 44: Synthesis uses S_unseen_array    ← AUTOMATICALLY USES IMAGE PROTOTYPES
Cell 51: Classifier combines real seen + synthetic unseen ← UNCHANGED
```

Because the prototype variable names (`S_seen_prototypes`, `S_unseen_prototypes`, `S_seen_array`, `S_unseen_array`) are the same, **the entire downstream pipeline (WGAN-GP, synthesis, classifier) automatically operates in image prototype space without any code changes**. This is the key design decision that keeps the modification scoped and clean.

### 4.3 What Changes in Prototype Geometry

| Property | Text Prototypes | Image Prototypes (Expected) |
|----------|-----------------|----------------------------|
| Mean inter-class cosine | 0.668 | ~-0.001 |
| Prototype cloud shape | Tight cone (all pointing ~same direction) | Near-uniform on sphere |
| WGAN conditioning signal | Similar prototypes → similar synthetic embeds | Diverse prototypes → diverse synthetic embeds |
| Generator difficulty | Low (all prototypes ~same) | Higher (must learn 128-D variation) |

**Implication for WGAN-GP**: The generator now receives prototypes that are spread across the sphere rather than clustered. This means the conditioning signal is more informative (each class gets a unique direction), but the generator must learn a more complex mapping. If the WGAN-GP loss diverges, possible fixes: increase `n_steps` from 10000, or widen generator hidden layers.

---

## 5. Notebook Cell Map (67 cells total)

```
Cells  0- 6: Setup (installs, descriptions)
Cell   7   : Data loading — full 1000-D CORnet-S (no PCA truncation), label collision fix
Cells  8- 9: Data loading (continued)
Cells 10-16: Data exploration (structure summary, class distribution, EEG norms)
Cells 17-22: Baseline [A] — LogReg on raw EEG
Cells 23-27: GZSL baseline evaluation (seen-only on unseen)
Cell  28   : [MODIFIED] Brain-Image Encoder markdown header
Cell  29   : [MODIFIED] ENCODER_CONFIG dictionary
Cell  30   : [MODIFIED] BrainEncoder + ImageProjector + contrastive_loss definitions
Cell  31   : [MODIFIED] Data prep — brain + image tensors, DataLoader
Cell  32   : [MODIFIED] Training loop — brain_encoder + image_projector
Cell  33   : [MODIFIED] Loss curve plot
Cell  34   : [MODIFIED] Embedding extraction (brain + image)
Cell  35   : [MODIFIED] Prototype computation from image embeddings
Cell  36   : [MODIFIED] Caching (brain + image embeddings + prototypes)
Cell  37   : [MODIFIED] t-SNE visualisation
Cell  38   : [MODIFIED] cWGAN-GP markdown header
Cells 39-46: cWGAN-GP (config, models, data prep, training, synthesis) — UNCHANGED
Cell  47   : [MODIFIED] Real vs synthetic t-SNE
Cell  48   : [MODIFIED] Implementation summary
Cell  49   : [MODIFIED] GZSL Classifier markdown header
Cell  50   : GZSL classifier accuracy computation — UNCHANGED
Cell  51   : Sample balancing (downsampling synthetic unseen) — UNCHANGED
Cell  52   : LogReg training with class_weight='balanced' — UNCHANGED
Cells 53-57: GZSL evaluation metrics — UNCHANGED
Cell  58   : Baseline vs GZSL comparison — UNCHANGED
Cells 59-60: Figures — UNCHANGED
Cell  61   : [MODIFIED] Final summary
Cells 62-66: Evaluation harness (evaluate_gzsl, diagnostics, phase comparison) — UNCHANGED
```

---

## 6. Verification Checklist

After running the notebook end-to-end on Colab, verify:

### 6.1 Structural Checks
- [ ] Notebook has exactly **67 cells** (no cells 67+ exist)
- [ ] Cell 7 prints `seen_image_features= 1000` (not 100)
- [ ] No `TextProjector`, `T_train_tensor`, `clip_loss(` in code cells
- [ ] `ImageProjector`, `I_train_tensor`, `V_train_embeds`, `contrastive_loss` all present

### 6.2 Training Checks
- [ ] Encoder training loss is **decreasing** over 50 epochs
- [ ] Final loss is reported (compare to text alignment baseline of ~4.2)
- [ ] Loss curve saved to `figures/encoder_loss_curve.png`

### 6.3 Embedding & Prototype Checks
- [ ] `E_train_seen.shape` = (13232, 128)
- [ ] `V_train_embeds.shape` = (13232, 128)
- [ ] `S_seen_array.shape` = (1654, 128)
- [ ] `S_unseen_array.shape` = (200, 128)

### 6.4 WGAN-GP Checks
- [ ] Generator and Critic losses converge (no divergence)
- [ ] `E_synth_unseen.shape` = (4000, 128) — 200 classes x 20 synth per class

### 6.5 GZSL Classifier Checks
- [ ] Sample balancing prints median count (~8/class)
- [ ] `class_weight='balanced'` in LogReg
- [ ] Routing rate is **below 15%** (not 97.9% like the old bug)
- [ ] AccS, AccU, H-mean all printed

### 6.6 Diagnostics
- [ ] Weight norm ratio (unseen/seen) is **below 1.5** (healthy balance)
- [ ] Bias table shows reasonable seen vs unseen distribution

---

## 7. Expected Results

### 7.1 Performance Expectations

| Metric | Text Alignment (baseline) | Image Alignment (expected) |
|--------|---------------------------|----------------------------|
| Encoder top-1 (seen) | 2.87% | ~5-10% (conservative) |
| AccS | 3.45% | ~5-10% |
| AccU | 0.32% | ~1-3% |
| H-mean | 0.58% | ~1-4% |
| Routing rate | 10.8% | ~8-15% |

**Rationale**: The text alignment ceiling was fundamentally about the target space (0.668 inter-class cosine sim). With near-orthogonal visual prototypes (-0.001 cosine sim), the encoder has much more room to discriminate. However, CORnet-S features are not CLIP-trained, so we may not achieve NICE-EEG's 15.6% — that used CLIP image features which have even better structure.

### 7.2 What Would Indicate Success

- **Strong success**: >5% top-1 (significant improvement over text alignment)
- **Moderate success**: 3-5% (improvement but similar magnitude)
- **Concerning**: <3% (worse than text alignment — investigate)
- **Failure signal**: WGAN-GP loss divergence (image prototypes too spread for generator)

### 7.3 If Results Are Concerning

1. **WGAN-GP divergence**: Increase `n_steps` to 20000, or try `lr_g: 2e-4`, `lr_c: 5e-5`
2. **Worse than text**: Check if `image_seen.shape[1]` = 1000 (not 100). Check if prototypes are L2-normalised.
3. **High routing rate**: Sample balancing should be intact. Verify cells 51-52 are correct.
4. **Overfitting**: Try reducing ImageProjector hidden_dim from 512 to 256, or increase dropout to 0.2.

---

## 8. Risks and Mitigations

### 8.1 CORnet-S Seen/Unseen Norm Asymmetry

The raw CORnet-S features have different norms: seen ~75 vs unseen ~50 (after x50.0 scaling). This could bias prototypes.

**Mitigation**: Prototypes are L2-normalised (`compute_prototypes` does mean then normalise). So all prototypes live on the unit sphere regardless of input norm. The x50.0 scaling in cell 7 ensures features are in a reasonable numerical range for the network.

### 8.2 WGAN-GP Under Different Prototype Geometry

Text prototypes were all clustered (cosine ~0.668), so the generator learned to produce embeddings in a tight region. Image prototypes are near-orthogonal, requiring the generator to produce embeddings spread across the sphere.

**Mitigation**: The WGAN-GP architecture (Generator: [z_dim + embed_dim -> 4096 -> 4096 -> embed_dim]) has enough capacity. If divergence occurs, increase `n_steps` or apply spectral normalisation.

### 8.3 1000-D Input Overfitting

The ImageProjector's first layer has 512K parameters with only 13,232 training samples.

**Mitigation**: LayerNorm + Dropout(0.1) provide regularisation. The contrastive loss with batch_size=256 provides ~32K negative pairs per batch. If overfitting is severe, reduce hidden_dim to 256.

---

## 9. File Reference

| File | Purpose |
|------|---------|
| `GZSL_EEG_Pipeline_v2.ipynb` | Active notebook (67 cells, image alignment) |
| `helper_files/switch_to_image_alignment.py` | Script that applied the modifications |
| `context/Visual_investigation_results.md` | Visual investigation outputs (archived) |
| `context/Claude_phaseF_analysis.md` | Phase F analysis (why text alignment failed) |
| `CLAUDE.md` | Project spec and constraints |
| `MEMORY.md` | Persistent memory |

---

## 10. What Comes Next (After This Run)

1. **Analyse results**: Compare AccS, AccU, H-mean with text alignment baseline
2. **If successful**: Consider sourcing CLIP image embeddings for even better performance
3. **If concerning**: Hyperparameter tuning on the image alignment pipeline
4. **Research thread**: Structure-preserving WGAN-GP loss modifications (pen-and-paper math)
5. **Benchmarking**: Compare with NICE-EEG and ATM on standardised metrics
