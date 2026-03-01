
# Project Context: Brain–Text CLIP + GAN + GZSL on BraVL (COMP2261 Coursework)

This document exists **purely to give agentic coding tools (e.g. Google Antigravity, Cursor) full context** about the project, dataset, goals, and architecture.  
It is **not** meant to contain the implementation itself, only the *design and constraints*.

The user is working on a graded university coursework for:

> **COMP2261 – Artificial Intelligence / Machine Learning Coursework**  
> Dataset: **BraVL-like brain–vision–language dataset** (EEG + text + image features)  
> Environment: **Google Colab**, possibly with helper tools (Cursor, Antigravity, etc.)

The key goal is to design, implement, and analyse a **novel or hybrid paradigm** that improves over a **baseline model** using advanced methods from zero-shot learning, generative modelling, and contrastive representation learning.

---

## 1. High-Level Objective

We want to decode **visual object category** labels from **EEG/brain signals**, under a **Generalised Zero-Shot Learning (GZSL)** setting, using:

1. A **baseline model [A]**:  
   - Multiclass **Logistic Regression** on **raw EEG features**.
   - Purely supervised, no zero-shot or generative component.

2. **Methodological improvements [B]** (the hybrid paradigm):  
   - A **Brain–Text CLIP-style encoder** that aligns EEG and text embeddings in a shared semantic space via contrastive learning.  
   - A **conditional GAN** that synthesises **brain embeddings for unseen classes** in that shared space, conditioned on semantic text prototypes.  
   - A **GZSL evaluation protocol** with metrics that balance seen vs unseen performance (harmonic mean).

3. A **customised model [A+B]**:  
   - Still uses **multinomial Logistic Regression** (same family as the baseline).  
   - But now trained on:
     - Real embeddings for **seen** classes (from brain encoder).  
     - Synthetic embeddings for **unseen** classes (from GAN).  
   - Evaluated on **both seen and unseen EEG samples**.

The **core research question** is:

> Does aligning EEG with text semantics and augmenting unseen classes via generative modelling improve generalisation to unseen categories (GZSL) compared to a simple supervised baseline?

---

## 2. Dataset and Existing Notebook Context

The user already has an existing Colab notebook:

- Likely named something like:  
  `COMP2261_ArizMLCW.ipynb`

This notebook:

- Uses a BraVL-like multimodal dataset with:
  - `brain_seen`, `brain_unseen` – EEG tensors.
  - `text_seen`, `text_unseen` – text embeddings (e.g. CLIP text or sentence-transformer embeddings).
  - `image_seen`, `image_unseen` – image features (not essential to the main pipeline, can be ignored or used as auxiliary analysis).
  - `label_seen`, `label_unseen` – class indices (integers).

- Performs some or all of the following operations:
  - Loads dataset via `mmbra` / `mmbracategories` utilities.
  - Filters to a subset of classes (e.g. first 20 or 50 categories) using a threshold on labels.
  - Splits data into per-class train/test sets for seen vs unseen classes.
  - Runs a simple baseline classifier (e.g. SVM or LR) on brain features.

> **Agent note:**  
> You must respect the existing data structure: EEG, text, and labels already exist and are aligned.  
> Do **not** assume raw images are available or necessary for the project — the main modalities are **EEG and text**.

---

## 3. Key Concepts and Terminology

### 3.1 Classes

- A **class** = a category label, such as `"dog"`, `"cat"`, `"car"`, `"flower"`, etc.
- In code, classes are represented as integers (`0, 1, 2, ...`), via `label_seen`, `label_unseen`.
- Each class has:
  - One or more **EEG samples** (for seen classes).
  - One or more **text embeddings** (for both seen and unseen classes).

We define:

- **Seen classes** 𝓢: have EEG during training.
- **Unseen classes** 𝓤: have **no EEG training samples**, but may appear in the test set.

### 3.2 Text Embeddings and Semantic Prototypes

- `text_seen` / `text_unseen` are text feature arrays, typically derived from a **text encoder** (e.g. CLIP’s text tower or a sentence-transformer).
- Each row corresponds to a **text description of a class**. Usually, these are constructed from the class name, e.g.:

  - `"dog"` → `"a photo of a dog"`  
  - `"cat"` → `"a photo of a cat"`

- A **semantic prototype** for class `c` is a single vector summarising that class, commonly:

  - The **text embedding** `g(t_c)` for `"a photo of a <class_name>"`, or  
  - The **average** of multiple text embeddings for that class.

These prototypes are critical:

- During CLIP-style training, to align EEG and text.  
- As conditioning vectors `s_c` for the GAN: `G(z, s_c)`.

---

## 4. Experimental Structure (Aligned with Coursework Specification)

The coursework requires:

1. A **baseline model [A]** using existing packages.  
2. A **methodological improvement [B]** with theoretical justification.  
3. Implementation of **[A+B]** as a customised, improved system.

### 4.1 Baseline Model [A] – Logistic Regression on Raw EEG

- **Input:** `train_brain` (EEG features, possibly flattened).  
- **Output:** class labels `train_label` (over the selected classes).  
- **Model:** `sklearn.linear_model.LogisticRegression` with:
  - `multi_class='multinomial'`
  - `solver='lbfgs'`
  - `max_iter ~ 1000`
- **Training setting:** standard supervised learning, using all selected classes as “seen”.
- **Evaluation:** plain accuracy, macro-F1, confusion matrix on held-out EEG test set.

This is the simplest supervised EEG→label pipeline and is **important as the anchor** for evaluation.

### 4.2 Methodological Improvements [B] – Hybrid Paradigm

The improvements [B] consist of three major parts:

1. **Brain–Text CLIP-style Contrastive Encoder**  
2. **GAN-based Embedding Synthesis for Unseen Classes**  
3. **GZSL Setting and Metrics**

#### 4.2.1 Brain–Text CLIP-style Encoder

**Goal:**  
Learn a shared semantic space where EEG embeddings and text embeddings are aligned.

**Components:**

- Brain encoder `f_b`:  
  - Input: EEG vector `b ∈ ℝ^{D_b}`  
  - Output: embedded vector `z^b ∈ ℝ^d`, then L2-normalised.

- Text projector `g`:  
  - Input: text vector `t ∈ ℝ^{D_t}` (precomputed, e.g. CLIP text features)  
  - Output: embedded vector `z^t ∈ ℝ^d`, then L2-normalised.

**Loss:**

- CLIP-style InfoNCE / contrastive loss on mini-batches of matched `(EEG, text)` pairs.
- Enforces:

  - `z_i^b` close to `z_i^t` for the same class.  
  - `z_i^b` far from `z_j^t` for different classes.

**Outcome:**

- A shared space where:
  - Brain embeddings cluster around their class text prototypes.
  - Text-only semantics (for unseen classes) have a meaningful position in the same space.

#### 4.2.2 GAN-based Embedding Synthesis

**Goal:**  
Generate plausible **EEG embeddings** for unseen classes using only their **text descriptions**, in the shared space learned above.

**Training data for GAN:**

- Real embeddings: `e = f_b(b)` for EEG from seen classes.  
- Semantic condition: `s_c = z_c^t` for each seen class `c`.

**GAN structure:**

- Generator `G(z, s_c)`:
  - Input: random noise `z` + semantic prototype `s_c`.  
  - Output: synthetic embedding `\tilde{e}` in the same ℝ^d space (optionally L2-normalised).

- Discriminator / critic `D(e, s_c)`:
  - Input: embedding (real or fake) + semantic prototype.  
  - Output: scalar “real/fake” score.

**Loss:**

- Vanilla GAN or Wasserstein GAN with gradient penalty (WGAN-GP).
- Optional regularisers:
  - Feature matching to align distributions of real vs synthetic embeddings.
  - Cosine alignment to ensure `\tilde{e}` stays close to `s_c`.
  - Diversity penalties to avoid mode collapse.

**Synthesis for unseen classes:**

- At inference / data-generation time:
  - For each unseen class `u`:
    - Compute semantic prototype `s_u` (text embedding).
    - Sample multiple `z ~ N(0, I)` and generate `\tilde{e}_u = G(z, s_u)`.

These synthetic embeddings simulate “fake EEG” in embedding space for unseen categories.

#### 4.2.3 GZSL paradigm and metrics

**Setting:**

- Train the encoder and GAN **only on seen classes** (EEG + text).  
- At test time, EEG may come from both **seen** and **unseen** classes.

**Classifier (same family as baseline):**

- Multinomial logistic regression on embeddings, not raw EEG.

**Training set for the classifier:**

- Real embeddings for seen classes:
  - `e = f_b(b)` for `y ∈ 𝓢`.
- Synthetic embeddings for unseen classes:
  - `\tilde{e}_u = G(z, s_u)` for `u ∈ 𝓤`.

**Evaluation:**

- `Acc_seen`: accuracy on seen-class test samples.  
- `Acc_unseen`: accuracy on unseen-class test samples.  
- Harmonic mean:

  \[
  H = \frac{2 \cdot Acc_{\text{seen}} \cdot Acc_{\text{unseen}}}
           {Acc_{\text{seen}} + Acc_{\text{unseen}}}
  \]

This ensures the model is judged on **balance** between seen and unseen classes.

---

## 5. Customised Model [A+B]

The **final model** is:

> A multinomial logistic regression classifier (same model family as the baseline) trained on a **mixture of real and synthetic embeddings** produced by the brain–text encoder and GAN, evaluated under GZSL.

Pipeline:

1. **EEG → embedding** via `f_b`.  
2. **Class names → text embeddings → semantic prototypes** via `g`.  
3. **GAN** generates unseen-class embeddings in the same space.  
4. **Logistic regression** trained on:
   - Real embeddings from seen classes.
   - Synthetic embeddings from unseen classes.
5. **GZSL test:** All EEG test samples are embedded with `f_b`, classified with LR, metrics computed per seen/unseen and harmonic mean.

---

## 6. Implementation Strategy for Agentic Tools

This section is **for agents** like Cursor / Google Antigravity to know how to operate on the repo / notebook.

### 6.1 Repository / File Structure (Recommended)

If possible, aim for a structure like:

```text
project_root/
│
├─ data/
│   └─ ThingsEEG-Text
│   └─ ThingsEEG-Text.zip
├─ COMP2261_ArizMLCW.ipynb   # The original coursework notebook
│
└─ README.md / CONTEXT.md       # This document
```

However, the user might want to do all of this inside a single `.ipynb` in Colab. 

### 6.2 If the Agent Cannot Edit the `.ipynb` Directly

If an agentic environment cannot safely patch `.ipynb` JSON, it can:


1. If absolutely necessary, the agent can:
   - Load the `.ipynb` as JSON,
   - Append new code cells at the end,  
   - And save a modified file (e.g. `_with_gzsl.ipynb`).

**Important constraints:**

- **Do not silently break or delete original coursework cells.**  
- All new code should be **additive** and clearly separated (e.g. by headings in Markdown cells).

---

## 7. Libraries, Dependencies, and Environment

Environment assumptions (Colab-like):

- Python 3.x
- Libraries:
  - `numpy`, `scipy`
  - `pandas` (optional)
  - `torch` and `torchvision`
  - `scikit-learn`
  - `matplotlib` / `seaborn` (for visualisation)
  - `mmbra` / `mmbracategories` (dataset utilities provided by the coursework)
  - A text model (CLIP text or sentence-transformer); might already be embedded as `text_seen`, `text_unseen`.

Agents should:

- Reuse the **existing imports and installed packages** in the notebook where possible.
- Only install extra packages if absolutely necessary (and they must work in Colab).

---

## 8. What the Agent Should **Not** Change

- Do **not** alter the original dataset loading logic. It is part of the coursework.
- Do **not** rename or reorder labels in a way that breaks alignment of:
  - `brain_*`, `text_*`, `image_*`, `label_*`.
- Do **not** delete or overwrite original cells that the student may still need for baselines or for referencing in the report.
- Do **not** change random seeds or splits in a way that makes reproducing results impossible without the student’s consent.

---

## 9. Summary for the Agent

If you (the agent) are dropped into a project directory containing `COMP2261_ArizMLCW.ipynb` and this `CONTEXT.md`, here is what you should understand:

1. The main learning target is **GZSL EEG decoding** with **seen and unseen classes**.
2. The **baseline** is multinomial logistic regression on raw EEG features (no zero-shot).
3. The **improved model**:
   - Uses a **brain–text contrastive encoder** to learn a shared semantic space.
   - Uses a **conditional GAN** to synthesise unseen-class embeddings conditioned on text.
   - Uses the **same classifier family (logistic regression)** as the baseline, but on improved data (real seen embeddings + synthetic unseen embeddings).
4. The **evaluation** uses:
   - Seen accuracy, unseen accuracy, and harmonic mean.
5. The user will build the model in **stages** (baseline → CLIP encoder → GAN → GZSL classifier).  
   Agents should respect this staging and avoid “doing everything at once” in an unstructured way.

This document is the **authoritative context** for what the project is trying to achieve and how to structure the implementation.

