# The Inverse Bias Phenomenon: A Mathematical Autopsy

**Context**: This document is a theoretical deep-dive into why the full GZSL model (Method D) collapses on seen-class accuracy while improving unseen-class accuracy. It routes 99.6% of true-seen test trials to unseen labels — an *inverse* of the classic GZSL failure mode.

**How to use**: Paste or reference this in Claude Desktop App for rendered LaTeX. Continue the theoretical discussion there, then bring conclusions back to Claude Code for implementation planning.

---

## 1. The Stage — Geometry of the Embedding Space

After CLIP training, every embedding lives on the unit hypersphere:

$$\mathbb{S}^{d-1} = \{x \in \mathbb{R}^d : \|x\|_2 = 1\}, \quad d = 64$$

Because both $f_b$ and $g$ L2-normalise their outputs, the natural metric is **angular distance**, and the inner product $e^\top s$ equals $\cos(e, s)$.

With 1854 total classes (1654 seen + 200 unseen) in $\mathbb{R}^{64}$, the prototypes $\{s_c\}$ are approximately mutually orthogonal — a consequence of concentration of measure. Random unit vectors in $\mathbb{R}^d$ have expected pairwise cosine similarity $\approx 0$, with standard deviation $\sim 1/\sqrt{d} \approx 0.125$. Prototypes are roughly 90° apart with small fluctuations.

---

## 2. Two Very Different Distributions on the Same Sphere

The GZSL classifier sees two fundamentally different kinds of training data:

### Real seen embeddings

$$e_i = \text{normalize}(f_b(b_i)), \quad y_i \in \mathcal{S}$$

The brain encoder $f_b$ maps raw EEG through an MLP. EEG carries substantial measurement noise — sensor drift, trial-to-trial variability, neural noise. CLIP training pulls $e_i$ toward $s_{y_i}$, but cannot eliminate this irreducible noise. The class-conditional distribution on the tangent space of $\mathbb{S}^{d-1}$ is:

$$e \mid y = s \;\sim\; p_{\text{real}}(\cdot \mid s), \quad \text{mean} \approx \mu_s, \quad \text{covariance} \;\Sigma_s$$

where $\mu_s$ is offset from the prototype $s_s$ (imperfect alignment) and $\Sigma_s$ has **substantial trace** (EEG noise propagated through $f_b$). The angular spread around each prototype is **large**.

### Synthetic unseen embeddings

$$\tilde{e}_j = \text{normalize}(G(z, s_u)), \quad u \in \mathcal{U}$$

The generator $G$ is a smooth MLP mapping $(z, s_u) \to \tilde{e}$. It was trained to fool the critic $D$, which enforces that generated embeddings resemble real *seen* embeddings. But $G$ operates in a **clean computational graph** — no sensor noise, no biological variability. The WGAN-GP gradient penalty enforces Lipschitz continuity on $D$, which implicitly encourages $G$ to produce smooth, well-behaved output distributions:

$$\tilde{e} \mid y = u \;\sim\; p_G(\cdot \mid u), \quad \text{mean} \approx s_u, \quad \text{covariance} \;\tilde{\Sigma}_u$$

where $\operatorname{tr}(\tilde{\Sigma}_u) \ll \operatorname{tr}(\Sigma_s)$. The angular spread is **tight**. Synthetic embeddings cluster closely around their conditioning prototype.

### Alignment quality

Define the alignment quality for class $c$ as:

$$\alpha_c = \mathbb{E}[e^\top s_c \mid y = c]$$

Empirically (from the report's Figure 3):
- Real seen: $\alpha_s \approx$ moderate (positive/negative histograms overlap)
- Synthetic unseen: $\alpha_u \approx$ high (tight, well-separated from negatives)

---

## 3. What Multinomial Logistic Regression Actually Learns

The classifier predicts:

$$p(y = c \mid e) = \frac{\exp(w_c^\top e + \beta_c)}{Z(e)}, \qquad Z(e) = \sum_{c' \in \mathcal{S} \cup \mathcal{U}} \exp(w_{c'}^\top e + \beta_{c'})$$

Training minimises cross-entropy: $\mathcal{L} = -\sum_i \log p(y_i \mid e_i)$.

For each class $c$, the model learns a weight vector $w_c \in \mathbb{R}^d$ and bias $\beta_c \in \mathbb{R}$. The logit for class $c$ given input $e$:

$$\ell_c(e) = w_c^\top e + \beta_c$$

### Gradient dynamics on the two data types

**On a synthetic unseen example** $(\tilde{e}_j, u)$: The model wants $\ell_u(\tilde{e}_j) \gg \ell_c(\tilde{e}_j)$ for all $c \neq u$. Because $\tilde{e}_j \approx s_u$ and is well-separated from other prototypes, this is **easy**. The gradient pushes $w_u$ toward $s_u$ and inflates $\beta_u$ with little resistance from the loss landscape. The model reaches near-zero loss on these samples quickly.

**On a real seen example** $(e_i, s)$: The model wants $\ell_s(e_i) \gg \ell_c(e_i)$ for all $c \neq s$. But $e_i$ is noisy — it has significant components orthogonal to $s_s$, and non-trivial cosine similarity with other prototypes. There is **irreducible classification error**. Gradients for $w_s$ are noisier, and $\beta_s$ cannot be inflated aggressively without misclassifying other seen examples.

### Result after training

$$\|w_u\| \gg \|w_s\| \quad \text{(on average)}, \qquad \beta_u > \beta_s \quad \text{(on average)}$$

This is the **asymmetric confidence** problem.

---

## 4. The Routing Failure at Test Time

At test time, ALL inputs are real EEG embeddings. For a real seen-test embedding $e_{\text{test}}$ with true label $s \in \mathcal{S}$:

$$\ell_s(e_{\text{test}}) = w_s^\top e_{\text{test}} + \beta_s \quad \leftarrow \text{moderate}$$
$$\ell_u(e_{\text{test}}) = w_u^\top e_{\text{test}} + \beta_u \quad \leftarrow \text{for each } u \in \mathcal{U}$$

### Signal decomposition

Decompose the test embedding in terms of prototype alignment and noise:

$$e_{\text{test}} = (e_{\text{test}}^\top s_s)\, s_s + \varepsilon_\perp$$

where $\varepsilon_\perp \perp s_s$ and $|e_{\text{test}}^\top s_s|^2 + \|\varepsilon_\perp\|^2 = 1$ (unit sphere constraint).

Now expand the unseen logit:

$$w_u^\top e_{\text{test}} = w_u^\top \big[(e_{\text{test}}^\top s_s)\, s_s + \varepsilon_\perp\big] = (e_{\text{test}}^\top s_s)(w_u^\top s_s) + w_u^\top \varepsilon_\perp$$

In 64 dimensions with 200 unseen classes, it is highly likely that $w_u^\top \varepsilon_\perp > 0$ for several $u$ — the noise component of the real embedding projects positively onto some unseen weight vectors **by chance**. Combined with the inflated $\beta_u$, this pushes $\ell_u > \ell_s$.

### The max-over-classes amplifier

The predicted class is $\hat{y} = \arg\max_{c \in \mathcal{S} \cup \mathcal{U}} \ell_c(e_{\text{test}})$. With 200 unseen classes, you need:

$$\ell_s(e_{\text{test}}) > \max_{u \in \mathcal{U}} \ell_u(e_{\text{test}})$$

The probability that at least one of 200 unseen logits exceeds the correct seen logit:

$$P\!\Big(\exists\, u \in \mathcal{U} : \ell_u(e_{\text{test}}) > \ell_s(e_{\text{test}})\Big) = 1 - \prod_{u \in \mathcal{U}} P\!\Big(\ell_u(e_{\text{test}}) \leq \ell_s(e_{\text{test}})\Big)$$

Even if each individual $P(\ell_u > \ell_s)$ is moderate (say 0.02), the product over 200 classes:

$$1 - (1 - 0.02)^{200} = 1 - 0.98^{200} \approx 1 - 0.018 \approx 0.982$$

This is the **max-of-many-random-variables** phenomenon — the maximum of 200 moderate random variables easily exceeds a single moderate value.

---

## 5. The Sample Count Amplifier

A second mechanism compounds the problem:

| | Total samples | Classes | Samples per class |
|---|---|---|---|
| Seen training | 13,232 | 1,654 | $\approx 8$ |
| Unseen training (synthetic) | 4,000 | 200 | $20$ |

Unseen classes have **2.5× more training samples per class**. LogReg's MLE yields more confident (higher-magnitude) parameters for classes with more data and cleaner signal. This further inflates $\|w_u\|$ and $\beta_u$ relative to seen counterparts.

---

## 6. What "Geometrically Easy" Precisely Means

On $\mathbb{S}^{d-1}$, the class-conditional distributions for synthetic unseen embeddings occupy a **small solid angle** around each prototype. The classifier separates them with decision boundaries **far from the cluster centres** — the margin is large.

For real seen embeddings, the clouds are diffuse. Decision boundaries must pass *through* the noisy tails of neighbouring class distributions. The margin is small or negative (classes overlap).

The **Fisher discriminant ratio** for class $c$:

$$J_c = \frac{\|\mu_c - \mu_{\text{overall}}\|^2}{\operatorname{tr}(\Sigma_c)}$$

- Unseen: high $J_c$ (tight clusters, well-separated means) → **easy**
- Seen: low $J_c$ (diffuse clouds, overlapping) → **hard**

The optimiser rationally "spends" its capacity making unseen classes very classifiable (because it can) and assigns leftover capacity to the harder seen classes. This minimises training loss but is **catastrophic** for test-time seen-class generalisation.

---

## 7. Summary: Three Compounding Mechanisms

| Mechanism | Mathematical signature | Effect on classifier |
|---|---|---|
| **Variance mismatch** | $\operatorname{tr}(\tilde{\Sigma}_u) \ll \operatorname{tr}(\Sigma_s)$ | $\|w_u\| \gg \|w_s\|$, $\beta_u > \beta_s$ |
| **Sample count imbalance** | 20 vs 8 per class | More statistical power → higher confidence for unseen |
| **Max-over-classes** | $\max_{u \in \mathcal{U}} \ell_u$ over 200 classes | Even small per-class leakage compounds to route ~100% of seen inputs to unseen |

These three effects compound **multiplicatively**, producing the 99.6% misrouting.

---

## 8. Candidate Fix Directions (for discussion)

The diagnosis suggests several intervention points:

1. **Match the variances**: Inject calibrated noise into synthetic embeddings so $\tilde{\Sigma}_u \approx \Sigma_s$
2. **Match the sample counts**: Balance per-class training counts between seen and unseen
3. **Decouple the logit scales**: Cosine classifier (normalise $w_c$ too) or separate temperature parameters $\tau_S, \tau_U$ for seen vs unseen logits
4. **Post-hoc calibration**: Learn a scalar/bias correction on a held-out set (Chao et al.'s AUSUC — Area Under Seen-Unseen Curve)
5. **Covariance-aware generation**: Train $G$ to match not just the mean but the second-order statistics of real embeddings
6. **Prototype retrieval instead of LogReg**: Your Method B worked better precisely because nearest-prototype retrieval doesn't suffer from this calibration pathology — it has no learned biases $\beta_c$

---

*Continue this discussion in Claude Desktop App for better rendering. Bring conclusions back to Claude Code for implementation planning.*
