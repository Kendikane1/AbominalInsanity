# Phase 1: Sample Count Balancing Analysis

**Task**: Assess mechanism M2 (sample imbalance: 20 unseen vs 8 seen) by balancing counts per class.
**Methods**:
- **P1A**: Downsample unseen to 8 per class (target median seen class length).
- **P1B**: Upsample seen (using WGAN generator + text prototypes) to 20 per class.

---

### Key Findings

#### 1. The Routing Rate Collapsed
- **Baseline (Method D)**: 99.70% of seen test inputs routed to unseen.
- **P1A (Downsample)**: 9.16% of seen test inputs routed to unseen.
- **P1B (Upsample)**: 8.80% of seen test inputs routed to unseen.

This is a **massive and immediate shift**. Merely balancing the sample counts largely eliminates the catastrophic inverse bias (misrouting to unseen) that dominated the base model.
> **Conclusion:** M2 (sample imbalance) was the primary driver artificially pushing predictions towards the unseen categories.

#### 2. Accuracy Trade-off (AccS vs AccU)
| Phase | AccS | AccU | H | Routing Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 0.0006 | 0.0213 | 0.0012 | 0.9970 |
| **P1A** | 0.0290 | 0.0022 | 0.0042 | 0.0916 |
| **P1B** | 0.0239 | 0.0026 | 0.0046 | 0.0880 |

- The collapse of inverse routing rate allowed **AccS to drastically improve** (from practically 0% back up to ~2-3%).
- However, **AccU plummeted** (from ~2% down to 0.2%).
- Overall Harmonic Mean (H) improved from ~0.1% to ~0.4%, but performance is still quite bad.
> **Conclusion:** While balancing fixes the routing bias issue, the model is still inherently poor at classification overall. The original 2% unseen accuracy was an illusion caused by predicting unseen for *almost everything* (if you guess unseen 99.7% of the time, you'll naturally catch some true unseen correctly, inflating AccU artificially).

#### 3. P1A vs P1B equivalence
P1A and P1B achieve near-identical routing rates (9.16% vs 8.80%), accuracy profiles, and H scores.
> **Conclusion:** From the instructions: "If P1B matches P1A → variance mismatch is symmetric and M2 is the main driver". Because the generator produces tightly clustered low-variance samples for both seen (in P1B) and unseen (in P1A and P1B), adding low-variance synthetic seen data doesn't meaningfully change the classifier's boundary compared to just removing low-variance synthetic unseen data. M2 (the raw counts of samples) dominated the decision boundaries.

#### 4. Classifier Diagnostics: Weight Norms & Biases
**P1A (Downsample):**
- Seen ||w|| mean: 3.76
- Unseen ||w|| mean: 3.96 (Ratio: 1.05x)
- Seen bias mean: -0.0005
- Unseen bias: 0.0043

**P1B (Upsample):**
- Seen ||w|| mean: 6.98
- Unseen ||w|| mean: 7.24 (Ratio: 1.04x)
- Seen bias mean: -0.0031
- Unseen bias: 0.0256

- The weight norm ratios between unseen and seen are *very* close to 1.0 in both balanced scenarios (1.05x and 1.04x).
- The intercepts/biases are also near zero.
> **Conclusion:** The classifier is no longer artificially amplifying weight magnitudes or biases specifically for the unseen classes. This confirms that log-regression inherently scaled up weights/biases for the class sets that simply had more training samples in the original unbalanced dataset.

### Summary
The inverse bias (seen->unseen) was almost entirely an artefact of sample count imbalance (M2), heavily amplified by Logistic Regression's tendency to favor majority classes (or sets of classes with larger priors in the training empirical distribution).

Balancing the dataset corrects the model calibration, returning it to a state where it no longer guesses "unseen" for everything. However, the true discriminatory power of the features (or the capacity of the model to generalize zero-shot) remains very weak, exposing the very low harmonic mean accuracy. 

The next phases will need to address the fundamental discriminative capability of the features / generator, now that the calibration issue is isolated.
