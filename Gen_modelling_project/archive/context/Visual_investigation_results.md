Cell outputs, starting from the GZSL Classifier Evaluation with Harness :

Cell 1 :
Seen labels: 1654 classes (range 1-1654)
Unseen labels: 200 classes (range 1655-1854)
Overlap: set()

============================================================
GZSL EVALUATION — Pipeline v2
============================================================
  Acc Seen:      0.0033
  Acc Unseen:    0.0248
  Harmonic Mean: 0.0059
  F1 Seen:       0.0034
  F1 Unseen:     0.0160
  Routing Rate (seen→unseen): 0.9791 (3239/3308)

  Bias Table:
                         Pred Seen    Pred Unseen
    True Seen                  69        3239  (97.9% misrouted)
    True Unseen               135       15865
============================================================

============================================================
CLASSIFIER DIAGNOSTICS — Pipeline v2
============================================================
  Weight norms ||w_c||:
    Seen   — mean: 4.1027, std: 0.5200, min: 1.9792, max: 5.2223
    Unseen — mean: 7.2283, std: 0.3864, min: 5.7493, max: 7.9963
    Ratio (unseen/seen mean): 1.76x

  Intercepts (biases) β_c:
    Seen   — mean: 0.0024, std: 0.1170
    Unseen — mean: -0.0199, std: 0.1765
============================================================

Saved: figures/pipeline_v2_diagnostics.png

Cell 2 :
============================================================
VISUAL FEATURE INSPECTION
============================================================

Seen visual feature keys: ['data']
Unseen visual feature keys: ['data']

Full visual feature shapes:
  vis_seen (train): (16540, 1000)
  vis_unseen (test): (16000, 1000)

Currently using first 100 PCA components out of 1000 total
Information retained: 100/1000 dimensions = 10.0%

Visual Feature Statistics (full dimensions):
  Seen  - mean: 0.0000, std: 0.0596, norm: 1.51
  Unseen - mean: -0.0001, std: 0.0319, norm: 1.00

Cell 3 :
  90% variance explained at: 829 dimensions
  95% variance explained at: 913 dimensions
  99% variance explained at: 983 dimensions

  accompanied by figure visual feature

Cell 4 :
============================================================
ALIGNMENT TARGET COMPARISON: Visual vs Text
============================================================

Inter-class cosine similarity (lower = better class separation):
  Visual (CORnet-S): mean=-0.0014, std=0.1722
  Text (CLIP):       mean=0.6680, std=0.0773

  accompanied by figure visual feature vs text feature

Cell 5 :
============================================================
CROSS-MODAL CORRELATION (per-class prototypes)
============================================================
  EEG-Visual:  mean r = 0.0006, std = 0.0307
  EEG-Text:    mean r = -0.0331, std = 0.0328

Note: These are raw feature-space correlations (before any learned projection).
Higher absolute correlation suggests an easier alignment target.

Cell 6 :
============================================================
CLIP IMAGE FEATURE AVAILABILITY CHECK
============================================================

Visual feature directory structure:
  ThingsTrain/pytorch/cornet_s/ (10 subjects)
  ThingsTest/pytorch/cornet_s/ (10 subjects)

Searching for CLIP image features:
  not found: clip
  not found: CLIP
  not found: clip
  not found: CLIP
  not found: openai_clip

--- ASSESSMENT ---
CLIP image embeddings are NOT pre-computed in this dataset.
CORnet-S features are biologically-inspired CNN features, not CLIP.
Options for image alignment:
  1. Use CORnet-S features directly as visual alignment target
  2. Compute CLIP image embeddings from original THINGS stimuli (requires images)
  3. Download pre-computed CLIP features from NICE-EEG or BraVL repos

Cell 7 :
============================================================
EFFECTIVE DIMENSIONALITY ANALYSIS
============================================================

Raw feature dimensions:
  EEG (brain):            561-D
  Text (CLIP):            512-D
  Visual (CORnet-S full): 1000-D
  Visual (current slice): 100-D

  Visual (CORnet-S):
    95% variance at 335 dimensions
    99% variance at 510 dimensions

  Text (CLIP):
    95% variance at 169 dimensions
    99% variance at 328 dimensions

  EEG (Brain):
    95% variance at 116 dimensions
    99% variance at 300 dimensions

--- IMPLICATIONS ---
Higher effective rank = richer supervision signal for alignment.
If visual features have higher effective rank than text, they provide
more information per sample for the encoder to learn from.

Cell 8 :
============================================================
VISUAL FEATURE INVESTIGATION SUMMARY
============================================================

1. CORnet-S features: 1000-D total, using 100 PCA components
   100 dims explain 32.4% variance

2. Text features: 512-D

3. Inter-class separation (mean cosine sim, lower = better):
   Visual: -0.0014
   Text:   0.6680

4. Cross-modal correlation with EEG (higher = easier alignment):
   EEG-Visual: r = 0.0006
   EEG-Text:   r = -0.0331

5. CLIP image features: NOT available locally

============================================================

Cell 9 :
