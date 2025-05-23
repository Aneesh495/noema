# Mathematical Derivations and Algorithmic Foundations

This document provides rigorous mathematical derivations for the core algorithms, geometric embeddings, heuristic search formulas, and statistical calibration guarantees implemented in Noema.

---

## 1. Rotary Position Embeddings (RoPE) and YaRN Interpolation

### 1.1 RoPE Geometric Formulation
In self-attention, query $\mathbf{q}$ and key $\mathbf{k}$ vectors at sequence positions $m$ and $n$ must encode relative spatial offset $m - n$ while preserving inner product invariance under orthogonal transformations.

Let $\mathbf{x} \in \mathbb{R}^d$ be a token embedding vector with even dimension $d$. We partition $\mathbf{x}$ into two-dimensional subvectors:
$$\mathbf{x} = [\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \dots, \mathbf{x}^{(d/2)}]^T, \quad \mathbf{x}^{(i)} = [x_{2i-1}, x_{2i}]^T \in \mathbb{R}^2$$

For each subvector coordinate $i \in \{1, \dots, d/2\}$, define frequency:
$$\theta_i = \theta_{\text{base}}^{-2(i-1)/d}, \quad \text{where } \theta_{\text{base}} = 10000.0$$

The position-dependent rotation operator $\mathbf{R}_{\Theta, m} \in \mathbb{R}^{d \times d}$ acts as a block-diagonal orthogonal matrix:
$$\mathbf{R}_{\Theta, m}^{(i)} = \begin{pmatrix} \cos(m \theta_i) & -\sin(m \theta_i) \\ \sin(m \theta_i) & \cos(m \theta_i) \end{pmatrix}$$

Applying $\mathbf{R}_{\Theta, m}$ to query $\mathbf{q}$ at position $m$ and $\mathbf{R}_{\Theta, n}$ to key $\mathbf{k}$ at position $n$:
$$\langle \mathbf{R}_{\Theta, m} \mathbf{q}, \, \mathbf{R}_{\Theta, n} \mathbf{k} \rangle = \sum_{i=1}^{d/2} (\mathbf{q}^{(i)})^T (\mathbf{R}_{\Theta, m}^{(i)})^T \mathbf{R}_{\Theta, n}^{(i)} \mathbf{k}^{(i)}$$

Since rotation matrices are orthogonal and commute in two dimensions:
$$(\mathbf{R}_{\Theta, m}^{(i)})^T \mathbf{R}_{\Theta, n}^{(i)} = \mathbf{R}_{\Theta, -m}^{(i)} \mathbf{R}_{\Theta, n}^{(i)} = \mathbf{R}_{\Theta, n - m}^{(i)}$$

Thus:
$$\langle \mathbf{R}_{\Theta, m} \mathbf{q}, \, \mathbf{R}_{\Theta, n} \mathbf{k} \rangle = g(\mathbf{q}, \mathbf{k}, m - n)$$
The attention dot product depends strictly on relative offset $m - n$.

### 1.2 YaRN (Yet another RoPE extensioN) Frequency Scaling
When extending context length by factor $s > 1$, naive positional extrapolation causes high-frequency attention spikes. YaRN interpolates frequencies smoothly across three wavelength regimes:
- High-frequency components ($\lambda_i < \alpha$): No interpolation is applied to preserve local token precision.
- Low-frequency components ($\lambda_i > \beta$): Complete linear interpolation $\theta_i' = \theta_i / s$ is applied.
- Intermediate components ($\alpha \le \lambda_i \le \beta$): Smooth ramp function $\gamma(r) = \frac{r - \alpha}{\beta - \alpha}$ blends the regimes:
$$\theta_i^{\text{YaRN}} = (1 - \gamma_i) \theta_i + \gamma_i \frac{\theta_i}{s}$$

In addition, attention logits are scaled by temperature factor $\tau = 0.1 \ln(s) + 1.0$ to counteract entropy reduction across extended context horizons.

---

## 2. PUCT on AND/OR Proof Hypergraphs

### 2.1 Hyperedge Selection Formula
In a proof hypergraph, an action hyperedge $e = (u, \mathcal{V})$ transitions from source goal $u$ to a set of conjunct subgoals $\mathcal{V} = \{v_1, \dots, v_k\}$.

To balance exploration and exploitation, Noema extends the Polynomial Upper Confidence Tree (PUCT) formula to hyperedges:
$$\text{PUCT}(e) = Q(e) + c_{\text{puct}} \cdot P(e) \cdot \frac{\sqrt{\sum_{e' \in \text{Edges}(u)} N(e')}}{1 + N(e)}$$
where:
- $P(e)$: Prior policy probability predicted by the Transformer policy head.
- $N(e)$: Visit count of hyperedge $e$.
- $Q(e)$: Estimated probability that traversing hyperedge $e$ leads to a complete proof.

### 2.2 Conjunctive Value Propagation
For an OR-node with multiple outgoing hyperedges:
$$V(u) = \max_{e \in \text{Edges}(u)} Q(e)$$

For an AND-hyperedge $e$ targeting subgoals $\{v_1, \dots, v_k\}$:
Because all subgoals must be proved simultaneously for $e$ to succeed, assuming conditional independence of sibling subgoals yields:
$$Q(e) = \prod_{j=1}^k V(v_j)$$
For base-case terminal tactics ($k = 0$), $Q(e) = 1.0$.

When a child goal is proved ($V(v) = 1.0$), solvability propagates bottom-up. When a child goal is recognized as unsatisfiable ($V(v) = 0.0$), $Q(e)$ drops to zero and triggers dependency-directed backtracking.

---

## 3. Conformal Prediction Calibration

### 3.1 Non-Conformity Scoring
Given goal embedding $\mathbf{x}$ and candidate tactic action $y \in \mathcal{A}$, the model outputs softmax probability $\hat{\pi}(y \mid \mathbf{x})$.

The non-conformity score measures how unusual action $y$ is under the model policy:
$$s(\mathbf{x}, y) = 1 - \hat{\pi}(y \mid \mathbf{x})$$

### 3.2 Finite-Sample Coverage Guarantee
Given a calibration set of $n$ verified proof steps $(\mathbf{x}_i, y_i^*)_{i=1}^n$, we compute calibration non-conformity scores:
$$S = \{s(\mathbf{x}_1, y_1^*), \dots, s(\mathbf{x}_n, y_n^*)\}$$

For target error rate $\alpha \in (0, 1)$, we compute the empirical quantile:
$$\hat{q} = \text{Quantile}_{\lceil (n + 1)(1 - \alpha) \rceil / n}(S)$$

The conformal prediction set for an unseen goal $\mathbf{x}_{\text{new}}$ is:
$$C(\mathbf{x}_{\text{new}}) = \{y \in \mathcal{A} \mid s(\mathbf{x}_{\text{new}}, y) \le \hat{q}\} = \{y \in \mathcal{A} \mid \hat{\pi}(y \mid \mathbf{x}_{\text{new}}) \ge 1 - \hat{q}\}$$

Under exchangeability of calibration and test examples, the coverage guarantee holds non-parametrically:
$$\mathbb{P}(y_{\text{new}}^* \in C(\mathbf{x}_{\text{new}})) \ge 1 - \alpha$$

---

## 4. Generalized Advantage Estimation (GAE) for Theorem Proving

### 4.1 Temporal Difference Residuals
Let $r_t$ denote the intermediate step reward (rewarding reduction in open subgoals), $V(s_t)$ the estimated state value, and $d_t \in \{0, 1\}$ the termination indicator.

The one-step temporal difference residual is:
$$\delta_t = r_t + \gamma V(s_{t+1}) (1 - d_t) - V(s_t)$$

### 4.2 Multi-Step Exponentially Discounted Advantage
Generalized Advantage Estimation aggregates multi-step returns using discount factor $\gamma \in (0, 1]$ and exponential weighting parameter $\lambda \in [0, 1]$:
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}$$

Recursively:
$$\hat{A}_t = \delta_t + \gamma \lambda (1 - d_t) \hat{A}_{t+1}$$

Target value returns are computed as:
$$\hat{R}_t = \hat{A}_t + V(s_t)$$
Normalizing advantages across the rollout batch stabilizes the policy gradient update:
$$\tilde{A}_t = \frac{\hat{A}_t - \mu_A}{\sigma_A + \epsilon}$$

---

## 5. Unbiased pass@k Estimator

When sampling $n$ independent candidate proofs per theorem with $c$ correct proofs, the probability that at least one of $k \le n$ randomly chosen samples is correct is estimated without bias:
$$\text{pass}@k = \mathbb{E}\left[ 1 - \frac{\binom{n - c}{k}}{\binom{n}{k}} \right]$$

To prevent numerical overflow when computing large factorials, Noema evaluates this quantity using log-gamma functions:
$$\ln \binom{m}{k} = \ln \Gamma(m + 1) - \ln \Gamma(k + 1) - \ln \Gamma(m - k + 1)$$
$$\text{ratio} = \exp\left( \ln \binom{n - c}{k} - \ln \binom{n}{k} \right)$$
$$\text{pass}@k = 1 - \text{ratio}$$
This formulation avoids high-variance empirical sample-and-average Monte Carlo simulations.
