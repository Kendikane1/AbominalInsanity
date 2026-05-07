# Mathematics Curriculum — GZSL EEG Project

*Your personal learning roadmap. Each lesson maps to a concept you will need to understand our project deeply. Work through them in order within each level. For each lesson, ask the math teacher: "Teach me L[X.Y] — [topic name]."*

*You don't need to master Level 0 before touching Level 1 — but do the Level 0 check-in first so you know where your gaps are.*

---

## How to Use This Curriculum

Ask the teacher to start a lesson with:
> "Teach me L1.1 — Jacobian matrices"

Or to check your current understanding:
> "Give me a quick Level 0 diagnostic — ask me one question per topic to find my gaps"

Or to connect a lesson to a specific experiment:
> "Why did the perturbation sweep (L3.1) fail? Explain using the Jacobian."

---

## Level 0 — Prerequisites

*These are assumed. Ask for a diagnostic, not a full lesson, unless you hit a gap.*

**L0.1 — Vectors and vector spaces**
What a vector is geometrically (a direction and magnitude in space). The concept of a *basis* (a set of vectors that can describe all others). What *dimension* means (how many basis vectors you need). The *inner product* a·b and what it measures (alignment). The *L2 norm* ||a||_2 and what it measures (length).

*Why it matters for our project*: Every embedding in our project is a vector in R^{64}. The inner product of two unit-norm embeddings is exactly the cosine similarity — the core similarity measure throughout the entire pipeline.

---

**L0.2 — Matrices as linear transformations**
A matrix is not just a grid of numbers — it is a *function* that takes a vector in one space and produces a vector in another. W ∈ R^{m×n} maps R^n → R^m. Matrix multiplication is function composition. The *column space* of W (what outputs are reachable). The *null space* (what inputs get sent to zero).

*Why it matters*: Every layer of our encoder and generator is a matrix multiplication followed by a nonlinearity. Understanding weight matrices as geometric transformations is essential for understanding what a neural network actually does.

---

**L0.3 — Eigenvalues and eigenvectors**
An eigenvector v of a matrix A satisfies Av = λv — it points in a direction that A only *scales* (doesn't rotate). λ is the eigenvalue (the scaling factor). The eigendecomposition A = QΛQ^T (for symmetric A) expresses A as a sum of rank-1 outer products. PCA uses eigendecomposition of the data covariance matrix.

*Why it matters*: The within-class covariance matrix Σ_c for our embeddings has eigenvectors that point in the directions of maximum variance. These directions are what we want our generator to explore (and currently doesn't).

---

**L0.4 — Gaussian distributions and variance**
The univariate Gaussian N(μ, σ²): what μ and σ² mean geometrically (centre and spread). The multivariate Gaussian N(μ, Σ): what the covariance matrix Σ means (it encodes the shape and orientation of the distribution ellipsoid). Marginal distributions. Expectation E[X]. Variance Var[X] = E[(X - μ)²].

*Why it matters*: The generator's output distribution (for a fixed conditioning prototype s_c) is what we want to shape. VarR = 0.872 is a statement about the variance of this distribution compared to real brain embeddings.

---

**L0.5 — Gradients and the chain rule**
The gradient ∇_θ L is the vector of partial derivatives of a loss L with respect to all parameters θ. It points in the direction of steepest ascent. Gradient descent moves opposite to the gradient. The *chain rule* for composed functions: d/dx [f(g(x))] = f'(g(x)) · g'(x). In multiple dimensions, this becomes the product of Jacobians.

*Why it matters*: Every training step computes ∇_{θ_G} L_G and ∇_{θ_D} L_D and takes a step. Understanding gradients is essential for understanding why any loss modification changes the generator's behaviour.

---

## Level 1 — Core ML Mathematics

*These are directly needed to understand our pipeline components.*

**L1.1 — Jacobian matrices** ← START HERE
The Jacobian J_f of a function f : R^n → R^m is the matrix of all partial derivatives:
J_f[i,j] = ∂f_i / ∂x_j

It is the *best linear approximation* of f near a point x. Geometrically: if you draw a tiny ball of radius ε around x in the input space, J_f maps it to an ellipsoid in the output space. The *singular values* of J_f are the lengths of the ellipsoid's axes. Large singular values = directions of high sensitivity. Small singular values = directions the function barely responds to.

*Lesson topics*: Definition → geometric meaning as input-ball deformation → singular value decomposition of J → product structure for composed functions → why the generator Jacobian J_G determines everything about output diversity

*Project hook*: Our core finding is that isotropic prototype perturbation fails because it propagates through J_G into output perturbations along the generator's learned sensitivity directions (columns of J_G), not along real brain variability directions. This is why synthesis-only perturbation cannot help.

---

**L1.2 — Backpropagation as Jacobian chain multiplication**
When we compute ∂L/∂θ for a neural network L = l(f_n(f_{n-1}(...f_1(x, θ)...))), the chain rule gives:
∂L/∂x = J_{f_1}^T · J_{f_2}^T · ... · J_{f_n}^T · ∂l/∂y

Each J_{f_k}^T is the transpose of a layer's Jacobian. Backpropagation IS this matrix product, computed efficiently using the computational graph.

*Project hook*: When we add a variance regularisation term L_var to L_G, the gradient ∂L_var/∂θ_G flows backward through this chain and updates W_1, W_2, W_3 — which directly reshapes the columns of J_G. This is how training-time loss modifications work at a fundamental level.

---

**L1.3 — Unit sphere and high-dimensional geometry**
S^{d-1} is the set of all unit vectors in R^d: {x ∈ R^d : ||x||_2 = 1}. It is a (d-1)-dimensional manifold. The geodesic (shortest path) between two points on the sphere is an arc, not a line.

*Concentration of measure*: In high dimensions, almost all volume of the sphere concentrates near the equator (for any chosen "north pole"). A pair of random unit vectors in R^d has cosine similarity ≈ 0 with high probability — they are nearly orthogonal. For d=64, the expected cosine similarity between two random unit vectors is 0 with standard deviation 1/√64 = 0.125.

*Why our prototypes are nearly orthogonal*: We compute 1654 class prototypes in R^{64} by averaging image embeddings and L2-normalising. Even if the raw prototypes had structure, the concentration of measure on S^{63} means that with 1654 points in 64 dimensions, most pairs will be nearly orthogonal.

*Project hook*: The mean prototype cosine similarity ≈ 0.014 is NOT a coincidence. It is essentially what you would expect from 1654 random unit vectors in R^{64}. This confirms the embedding space is being used well.

---

**L1.4 — Covariance matrices and variance decomposition**
The *covariance matrix* Σ of a set of vectors {e_1,...,e_n} is:
Σ = (1/n) Σ_i (e_i - μ)(e_i - μ)^T

It is symmetric positive semi-definite. Its eigenvectors are the principal axes of the distribution. Its eigenvalues are the variances along those axes.

*Within-class covariance* Σ_c: covariance of all embeddings belonging to class c.
*Between-class covariance* Σ_B: covariance of the class means (prototypes) around the global mean.

VarR = 0.872 means: averaged across dimensions, the diagonal entries of Σ_c for synthetic embeddings are 87.2% of those for real unseen embeddings. The ellipsoid is slightly too tight.

*Project hook*: When we add a variance regularisation loss, we are trying to inflate the ellipsoid represented by Σ_c to better match the target. But we need to inflate it in the RIGHT DIRECTIONS (aligned with brain variability), not isotropically.

---

## Level 2 — Project-Specific Mathematics

*These are the exact mathematical components of our pipeline.*

**L2.1 — InfoNCE contrastive loss**
InfoNCE (Noise-Contrastive Estimation) is the loss we use to train the encoder. It maximises the similarity between matching (brain, image) pairs while minimising similarity to all non-matching pairs in the batch.

*Lesson topics*: Derivation from noise-contrastive estimation → temperature τ as a sharpness control → what "hard negatives" are and why small τ forces the model to care about them → gradient analysis: which pairs get pushed apart vs pulled together → the alignment-uniformity decomposition of InfoNCE on the sphere

*Project hook*: τ=0.05 works for image prototypes (nearly orthogonal, so small angles between correct pairs are meaningful) but τ=0.15 worked for text prototypes (highly clustered, so larger τ is needed to avoid over-sharpening). Temperature IS a function of the geometry of the prototype space.

---

**L2.2 — GAN theory from scratch**
The GAN minimax game: min_G max_D E[log D(x)] + E[log(1-D(G(z)))].

*Lesson topics*: What the optimal discriminator D* looks like for fixed G → what the generator loss becomes at optimality (Jensen-Shannon divergence between P_real and P_G) → why JS divergence causes mode collapse (saturating gradients when distributions are disjoint) → Nash equilibrium and why it's hard to reach with simultaneous gradient descent

*Project hook*: This is why we switched to WGAN — JS divergence has vanishing gradients when P_real and P_G have non-overlapping support, which happens early in training when the generator produces garbage.

---

**L2.3 — WGAN and the Wasserstein distance** ← HIGH PRIORITY
The Wasserstein-1 distance (earth mover's distance) W(P,Q) measures the minimum amount of "work" needed to transform distribution P into distribution Q, where work = mass × distance moved.

*Lesson topics*: Geometric definition (imagine moving piles of sand) → the Kantorovich-Rubinstein dual: W(P,Q) = sup_{||f||_L ≤ 1} E_P[f] - E_Q[f] → this turns into a critic (discriminator) objective → 1-Lipschitz constraint: why it's needed, what it means geometrically → weight clipping (original WGAN) → gradient penalty (WGAN-GP): why interpolated points ê = α·x_real + (1-α)·G(z), and why penalising ||∇D(ê)||_2 = 1 enforces 1-Lipschitz → λ=10 and why it matters

*Project hook*: Our critic L_D = E[D(G(z,s_c))] - E[D(e_real)] + λ·GP is exactly the Kantorovich-Rubinstein dual plus the gradient penalty. The critic is trying to be the 1-Lipschitz function that maximises the Wasserstein distance; the generator tries to minimise it.

---

**L2.4 — Zero-shot learning mathematics**
The ZSL transfer assumption: visual features and semantic features (text, images) share a common relational structure. If class A and class B are semantically similar, their visual representations should also be geometrically close.

*Lesson topics*: Semantic embedding space and prototype construction → why ZSL works when this assumption holds → the GZSL problem: classifying into seen AND unseen classes simultaneously → the seen-bias problem: classifiers trained on real seen + synthetic unseen prefer seen classes → the harmonic mean H = 2·AccS·AccU/(AccS+AccU) as a penalisation of routing bias → why H goes to zero if either AccS or AccU is zero

*Project hook*: Our routing rate is 20.4% — meaning 20.4% of all test samples are predicted to belong to an unseen class. With 200 unseen out of 1854 total classes, "fair" routing would be ~10.8%. We slightly over-route to unseen because our sample balancing (matching ~8/class for both seen and unseen) compensates for the fact that there are many more seen classes.

---

**L2.5 — Spearman rank correlation**
Spearman ρ measures the degree to which two variables are monotonically related (not necessarily linearly). It is computed by ranking both variables and then applying Pearson correlation to the ranks.

*Lesson topics*: Why rank correlation? (robust to outliers, captures any monotonic relationship) → How to compute it → Geometric interpretation: ρ=1 means "larger in one ↔ larger in the other", always → What ρ=0.857 vs ρ=0.668 means in our context (synthetic centroids are much more correlated with prototypes in rank-order distance than real brain centroids are)

*Project hook*: We use Spearman ρ (not Pearson) because we're comparing cosine similarity *rankings* between two sets of points, not their absolute values. Two points might both have cosine-sim=0.3 to their prototype in different spaces, but whether that's "close" or "far" depends on the distribution of all similarities in that space.

---

## Level 3 — Current Research Mathematics

*These are the mathematical tools needed for the next phase: training-time loss modifications.*

**L3.1 — Generator Jacobian: full derivation**
For our specific generator architecture (164→256→256→64 with LeakyReLU, then L2 normalisation):

```
G(z, s_c) = L2_norm(W_3 · σ(W_2 · σ(W_1 · [z ; s_c])))
```

The Jacobian with respect to s_c (the conditioning input):
```
J_G = J_{L2} · W_3 · D_2 · W_2 · D_1 · W_1^{(s)}
```
where:
- W_1^{(s)} ∈ R^{256×64} is the sub-block of W_1 that multiplies s_c (the bottom 64 rows of the input)
- D_1 ∈ R^{256×256} is the diagonal Jacobian of the first LeakyReLU (d_i = 1 if pre-activation > 0 else 0.01)
- W_2 ∈ R^{256×256} is the second weight matrix
- D_2 ∈ R^{256×256} is the diagonal Jacobian of the second LeakyReLU
- W_3 ∈ R^{64×256} is the output weight matrix
- J_{L2} ∈ R^{64×64} is the Jacobian of L2-normalisation: J_{L2}(u) = (I - ê_0 ê_0^T) / ||u||_2

*Lesson topics*: Full derivation of each factor → singular value decomposition of J_G → what the singular vectors mean (directions of high vs low generator sensitivity to conditioning) → why the perturbation experiment must fail (output perturbation ∝ columns of J_G, not aligned with brain variability)

---

**L3.2 — Variance regularisation loss**
How to write a loss that forces the generator to produce samples with appropriate within-class variance.

*Lesson topics*: What the loss term L_var looks like mathematically → derivation of ∂L_var/∂ê (gradient with respect to a single generated sample) → how this flows back through G to give ∂L_var/∂θ_G → how adding this term to L_G changes which directions the generator explores (it effectively adds a "repulsion from mean" term to the generator gradient) → why this is fundamentally different from post-hoc perturbation

---

**L3.3 — Diversity loss and embedding-space repulsion**
Adding L_div = -(2/(K(K-1))) Σ_{k<k'} ||ê_{c,k} - ê_{c,k'}||² to L_G forces the generator to spread its K outputs for class c.

*Lesson topics*: Gradient of L_div with respect to ê_{c,k} (it is a repulsion force pointing away from all other generated samples of the same class) → how this reshapes J_G over training → connection to maximum mean discrepancy (MMD) as a distributional divergence → potential conflict with Wasserstein objective: the critic wants to maximise distributional distance, the diversity loss wants to spread samples — are these aligned?

---

**L3.4 — Graph Laplacian and inter-class structure preservation**
The Laplacian L = D - A (where A is an adjacency matrix and D is the degree matrix) measures the "smoothness" of a signal over a graph. The quadratic form f^T L f = Σ_{(i,j)∈E} (f_i - f_j)² penalises large differences between adjacent nodes.

*Lesson topics*: Definition and construction of A, D, L → the quadratic form as a smoothness penalty → how to construct A from prototype cosine similarities → writing L_struct as a Laplacian-style penalty on synthetic centroids → gradient analysis

---

**L3.5 — Training dynamics: balancing new loss terms**
Adding multiple terms to L_G creates gradient conflicts: different terms push weights in different directions.

*Lesson topics*: Gradient magnitude scaling (why you need loss weights α · L_var + β · L_div + L_G) → how to estimate α, β to balance gradient magnitudes → the Nash equilibrium of the modified game: does it still exist? Is it still the Wasserstein minimum? → early stopping heuristics: watch for collapse (VarR suddenly drops back to near-zero) or divergence (G_loss suddenly spikes)

---

## Suggested Starting Sequence (Given Current Project State)

The project is at the threshold of training-time loss modifications. The most important gaps to close right now:

1. **L1.1 Jacobian matrices** — This is the core mathematical object explaining why the perturbation failed and why training-time modifications can succeed. Start here.
2. **L2.3 WGAN-GP** — To modify the generator loss, you need to understand what the current loss is doing. This is the second priority.
3. **L3.1 Generator Jacobian derivation** — Once you have L1.1, this gives you the full picture of why the columns of J_G don't align with brain variability.
4. **L3.2 Variance regularisation** — The first candidate loss modification to implement.
5. **L1.3 Unit sphere geometry** — To understand why VarR=0.872 matters and what "appropriate variance on S^{63}" means.

The remaining lessons can be done in any order as questions arise during implementation.
