# Dynamic Sparse Hypergraph Routing for Self-Assembling Neural Fields

## 1. Motivation and Design Thesis

The progression from Models 1804, 1806, 1401, and 1805 suggests a clear architectural direction.

Model **1804**, as a dense model, demonstrates that exhaustive information exchange is computationally and representationally feasible: module-to-module, module-to-environment, and environment-to-module interactions can all be queried densely, and a query point \(q\) can in principle interact directly with every relevant low-level entity.

Model **1806** further suggests that much of this dense information exchange is redundant. In other words, the accuracy achieved by dense global querying does not necessarily require every possible pairwise interaction to be instantiated.

Taken together, these observations motivate the following question:

> Can the interaction structure be reorganized into a more physically interpretable sparse topology, such that each query point \(q\) accesses only the information that is physically relevant to it, while preserving the fine-scale geometric interactions required for field accuracy?

We propose to treat a **hypergraph primarily as a routing mechanism rather than a compression mechanism**.

The intended role of the hypergraph is not to replace the underlying module-level information with pooled hyperedge embeddings. Instead, it should answer a structural question:

\[
\text{Which low-level entities should interact with which queries?}
\]

The hypergraph therefore defines a sparse interaction mask, while the final high-fidelity interaction is still computed directly between the query \(q\) and the selected underlying modules or environment entities.

This leads to the central design principle:

\[
\boxed{
\text{Sparse routing for efficiency}
\quad+\quad
\text{fine-grained pairwise interaction for physical fidelity}
}
\]

---

## 2. Lessons from Previous Architectures

### 2.1 Model 1401: Globally Fixed Hyperedge Capacity

Model 1401 explored hypergraph-style organization, but imposed a globally fixed number of hyperedges.

This introduces an architectural mismatch for generative or self-assembling design problems. If the number of active modules is variable, and if the number of distinct physical interaction mechanisms also varies from one configuration to another, then a fixed global hyperedge count \(K\) becomes an artificial bottleneck.

If \(K\) is too small, distinct physical mechanisms are forced into shared latent groups. If \(K\) is unnecessarily large, the model reserves capacity that may remain unused and may still incur memory, routing, and optimization overhead.

The deeper limitation is that the topology is not allowed to grow, shrink, split, or merge according to the actual physical system.

### 2.2 Model 1805: Geometry-Driven Partitioning with Premature Pooling

Model 1805 explored structured grouping based primarily on geometric partitioning, including B-spline-like spatial grouping.

Its main weakness is that geometric proximity is not equivalent to physical connectivity.

For example, two fluid regions may be spatially adjacent yet separated by an impermeable or adiabatic wall. A purely Euclidean partition can incorrectly group them despite the absence of a physically meaningful interaction path.

A second and more fundamental limitation is **premature pooling**.

Both 1401 and 1805 implicitly treat a hyperedge or grid group as a **compressor**. Module and environment features are aggregated into a single representation,

\[
h_k
=
\operatorname{Pool}
\left(
\{m_i\}_{i\in\mathcal{E}_k},
\{e_j\}_{j\in\mathcal{B}_k}
\right),
\]

after which a query point \(q\) interacts with \(h_k\) rather than with the original low-level entities.

This creates an information bottleneck. Fine-scale geometric and physical interaction features may be lost before the query-specific decoding stage.

The problem can be summarized as:

\[
\boxed{
\text{routing} \neq \text{compression}
}
\]

A hyperedge should determine **which interactions are instantiated**, not replace those interactions with a pooled surrogate.

---

## 3. Desired Computation Graph

The target architecture is:

1. A lightweight topology-construction mechanism groups modules and environment entities into one or more latent physical mechanisms.
2. A query point \(q\) determines which mechanisms are relevant.
3. The selected mechanisms define a sparse support set of low-level entities.
4. The decoder computes high-fidelity nonlinear interactions directly between \(q\) and those selected entities.

Let the active modules be

\[
\mathcal{M}
=
\{(x_i,m_i)\}_{i=1}^{M},
\]

where \(x_i\in\mathbb{R}^{d_x}\) denotes geometry or location and \(m_i\in\mathbb{R}^{d_m}\) denotes module features.

Let the environment entities be

\[
\mathcal{E}
=
\{(y_\ell,e_\ell)\}_{\ell=1}^{L}.
\]

Let \(q\in\mathbb{R}^{d_x}\) be a field query location, with optional query embedding \(z_q\).

The topology layer should produce a sparse routing weight

\[
\Pi_{qi}\ge 0,
\]

where

\[
\Pi_{qi}=0
\]

means that the expensive fine-scale interaction between \(q\) and module \(i\) need not be instantiated.

The decoded context is then

\[
c(q)
=
\sum_{i:\Pi_{qi}>0}
\Pi_{qi}
\,
\Phi_{\mathrm{fine}}
\left(
q-x_i,\,
m_i,\,
e_{\mathrm{local}}(q,x_i)
\right),
\]

followed by

\[
\widehat{U}(q)
=
\operatorname{FieldHead}
\left(
\operatorname{LayerNorm}(c(q))
\right).
\]

The essential point is that \(\Phi_{\mathrm{fine}}\) is evaluated **after sparse routing**, not before it.

---

## 4. Core Architectural Requirements

A successful topology mechanism should satisfy the following requirements.

### 4.1 Exact Computational Sparsity

The routing mechanism should produce exact zeros so that irrelevant interactions can be omitted from the forward graph.

A standard Softmax attention vector,

\[
p_i
=
\frac{\exp z_i}{\sum_j \exp z_j},
\]

satisfies

\[
p_i>0
\]

for every finite logit \(z_i\). Therefore, Softmax alone cannot provide an exact sparse support set.

Sparse alternatives such as Sparsemax or \(\alpha\)-Entmax are more appropriate because they can produce exact zeros.

### 4.2 Fine-Scale Interaction Preservation

No hyperedge-level pooling operation should replace the final \(q\)-to-module interaction.

Hyperedges may provide routing summaries or latent routing descriptors, but the high-fidelity decoder should preserve explicit pairwise or local-set interactions of the form

\[
\Phi_{\mathrm{fine}}(q,x_i,m_i,\cdots).
\]

### 4.3 Dynamic Topology

The effective number of active hyperedges should be determined by the current physical configuration rather than globally fixed.

We denote the active hyperedge set by

\[
\mathcal{K}(X,M,E),
\]

with cardinality

\[
K_{\mathrm{eff}}
=
|\mathcal{K}(X,M,E)|.
\]

The architecture should permit \(K_{\mathrm{eff}}\) to vary across samples and, ideally, across optimization steps.

### 4.4 Physics-Aware Connectivity

Topology formation should depend on both geometry and physical connectivity.

A valid affinity should therefore depend on terms such as

\[
\|x_i-x_j\|,
\]

but also on learned or explicit indicators of barriers, interfaces, transport pathways, material discontinuities, characteristic directions, shear structures, thermal connectivity, or other domain-specific physical constraints.

### 4.5 Differentiability for Inverse or Generative Design

Because module geometry may itself be optimized through gradient-based inverse design, routing should allow useful gradients with respect to module features and positions.

However, this requirement must be stated carefully.

Sparsemax, Entmax, ReLU, and support-selection mechanisms are generally **piecewise differentiable** or **differentiable almost everywhere**, not globally smooth. When the active support changes, the mapping can be non-smooth.

Therefore, the realistic target is:

\[
\boxed{
\text{exact sparsity}
+
\text{useful almost-everywhere gradients}
}
\]

rather than an unrealistically strong claim of globally smooth differentiation through every topology birth, death, split, or merge event.

---

## 5. Sparsemax as a Common Routing Primitive

For \(z\in\mathbb{R}^{K}\), Sparsemax is the Euclidean projection onto the probability simplex:

\[
\operatorname{sparsemax}(z)
=
\arg\min_{p\in\Delta^{K-1}}
\frac{1}{2}\|p-z\|_2^2,
\]

where

\[
\Delta^{K-1}
=
\left\{
p\in\mathbb{R}^{K}
\;:\;
p_k\ge 0,\;
\sum_{k=1}^{K}p_k=1
\right\}.
\]

Equivalently,

\[
p_k
=
\max(z_k-\tau(z),0),
\]

where \(\tau(z)\) is chosen such that

\[
\sum_k p_k = 1.
\]

Hence Sparsemax creates a support set

\[
S(z)
=
\{k:p_k>0\}.
\]

Within a region of logit space where \(S(z)\) is unchanged, its Jacobian is

\[
\frac{\partial p_i}{\partial z_j}
=
\begin{cases}
1-\frac{1}{|S|}, & i=j,\ i\in S,\\[4pt]
-\frac{1}{|S|}, & i\neq j,\ i,j\in S,\\[4pt]
0, & \text{otherwise}.
\end{cases}
\]

Thus coordinates outside the active support receive exactly zero local gradient.

This property is useful for sparse routing, but it also means that inactive routes do not automatically receive a revival gradient through Sparsemax alone. Any self-assembling mechanism must therefore be designed so that support changes can still occur through changes in the logits, hub scores, geometry, or auxiliary continuous variables.

---

# 6. Strategy I: Module-Elected Dynamic Hubs

## 6.1 Concept

In this strategy, hyperedges are not represented by a globally fixed set of latent nodes.

Instead, modules themselves are potential hyperedge hubs.

A module becomes a hub when the model determines that it occupies a physically important region, such as a burner center, strong shear layer, localized heat-release region, recirculation core, or another dynamically important structure.

The number of active mechanisms is therefore induced from the current module set.

---

## 6.2 Hub Propensity

Each module \(i\) computes a hub propensity score

\[
r_i
=
f_{\mathrm{hub}}
\left(
m_i,
x_i,
e_{\mathrm{local}}(x_i)
\right).
\]

A nonnegative activation can be defined as

\[
\gamma_i
=
\operatorname{ReLU}(r_i).
\]

The active hub set is

\[
\mathcal{H}
=
\{i:\gamma_i>0\}.
\]

The effective number of hubs is

\[
K_{\mathrm{eff}}
=
|\mathcal{H}|.
\]

If the selected hub indices are denoted \(i_k\), then

\[
s_k=x_{i_k},
\qquad
h_k=
g_{\mathrm{hub}}
\left(
m_{i_k},
e_{\mathrm{local}}(x_{i_k})
\right).
\]

### Important mathematical qualification

The mapping

\[
\gamma_i \mapsto \mathbb{I}(\gamma_i>0)
\]

is discrete. Therefore, the **cardinality**

\[
K_{\mathrm{eff}}
=
\sum_i \mathbb{I}(\gamma_i>0)
\]

is not itself differentiable.

The continuous scores \(\gamma_i\) are differentiable almost everywhere, but converting them into an explicitly variable-size tensor requires either:

- dynamic indexing,
- a straight-through or relaxed estimator,
- thresholded execution with continuous surrogate training,
- or a dense candidate representation with sparse support.

This distinction should be preserved in implementation and evaluation.

---

## 6.3 Physics-Aware Barrier or Resistance Function

To suppress physically invalid cross-boundary routing, define a continuous resistance between two spatial locations \(a\) and \(b\).

Let \(\overline{ab}\) denote the line segment connecting \(a\) and \(b\). One possible environment-conditioned resistance is

\[
R(a,b)
=
\sum_{\ell=1}^{L}
\rho_\ell
\exp
\left(
-\frac{
d(y_\ell,\overline{ab})^2
}{
2\sigma_R^2
}
\right),
\]

where

\[
\rho_\ell
=
\operatorname{softplus}
\left(
R_{\mathrm{env}}(e_\ell)
\right)
\ge 0.
\]

The distance from environment point \(y_\ell\) to the segment \(\overline{ab}\) is denoted

\[
d(y_\ell,\overline{ab}).
\]

This is a soft differentiable approximation of barrier accumulation.

If exact topological blocking is required, a sharper gate may be used, but sharper gates introduce stronger non-smoothness.

---

## 6.4 Module-to-Hub Routing

For a module \(j\) and active hub \(k\), define

\[
\ell^{M}_{jk}
=
\frac{
\left(W_M m_j\right)^\top
\left(W_H h_k\right)
}{
\sqrt{d_h}
}
+
b_{\mathrm{geo}}(x_j,s_k)
-
\eta R(x_j,s_k).
\]

Then

\[
A_{j:}
=
\operatorname{sparsemax}
\left(
\ell^{M}_{j:}
\right).
\]

Hence

\[
A_{jk}\ge 0,
\qquad
\sum_k A_{jk}=1,
\]

with many \(A_{jk}=0\).

---

## 6.5 Query-to-Hub Routing

Similarly,

\[
\ell^{Q}_{qk}
=
\frac{
\left(W_Q z_q\right)^\top
\left(W_H h_k\right)
}{
\sqrt{d_h}
}
+
b_{\mathrm{geo}}(q,s_k)
-
\eta R(q,s_k),
\]

and

\[
\alpha_{q:}
=
\operatorname{sparsemax}
\left(
\ell^{Q}_{q:}
\right).
\]

---

## 6.6 Hypergraph-Induced Query-to-Module Mask

The query-to-module routing matrix is obtained through the two-hop factorization

\[
\Pi
=
\alpha A^\top,
\]

or elementwise,

\[
\Pi_{qj}
=
\sum_{k\in\mathcal{H}}
\alpha_{qk}A_{jk}.
\]

This quantity should be interpreted as a **routing compatibility** rather than a compressed physical representation.

The active module neighborhood of \(q\) is

\[
\mathcal{N}(q)
=
\{j:\Pi_{qj}>0\}.
\]

The fine decoder then evaluates

\[
c(q)
=
\sum_{j\in\mathcal{N}(q)}
\widetilde{\Pi}_{qj}
\,
\Phi_{\mathrm{fine}}
\left(
q-x_j,\,
m_j,\,
e_{\mathrm{local}}(q,x_j)
\right).
\]

Here \(\widetilde{\Pi}_{qj}\) may equal \(\Pi_{qj}\), or may be renormalized over \(\mathcal{N}(q)\).

Finally,

\[
\widehat{U}(q)
=
\operatorname{FieldHead}
\left(
\operatorname{LayerNorm}(c(q))
\right).
\]

---

## 6.7 Interpretation

This construction preserves the desirable behavior of the dense 1804-style model:

\[
q
\longrightarrow
i
\]

remains an explicit high-fidelity interaction.

The hypergraph only changes the set of pairs for which that interaction is instantiated.

The architecture therefore implements:

\[
\boxed{
\text{hyperedges as routing pointers, not feature compressors}
}
\]

---

# 7. Strategy II: Differentiable Mean-Shift Attractors

## 7.1 Concept

The second strategy removes the requirement that a physical module itself serve as a hyperedge center.

Instead, modules generate continuous latent attractor coordinates, and those coordinates evolve through a differentiable clustering process.

The effective hyperedges are then associated with emergent attractors.

---

## 7.2 Joint Physical-Latent Coordinates

A stronger formulation than purely spatial mean shift is to construct a joint embedding

\[
u_i^{(0)}
=
\begin{bmatrix}
\lambda_x x_i\\
\lambda_m \psi(m_i)
\end{bmatrix}
\in\mathbb{R}^{d_u},
\]

where \(\psi\) is a learned feature embedding.

Alternatively, the model may predict a drifted initial attractor coordinate,

\[
u_i^{(0)}
=
\begin{bmatrix}
\lambda_x(x_i+\Delta x_i)\\
\lambda_m \psi(m_i)
\end{bmatrix},
\]

with

\[
\Delta x_i
=
f_{\Delta}(m_i,e_{\mathrm{local}}(x_i)).
\]

---

## 7.3 Physics-Aware Kernel

Define the pairwise kernel

\[
K_{ij}^{(t)}
=
K_{\mathrm{geo}}
\left(
u_i^{(t)},u_j^{(t)}
\right)
\,
K_{\mathrm{phys}}
(i,j).
\]

For example,

\[
K_{\mathrm{geo}}
=
\exp
\left(
-\frac{
\|u_i^{(t)}-u_j^{(t)}\|_2^2
}{
2\sigma_u^2
}
\right),
\]

while the physical connectivity term may be

\[
K_{\mathrm{phys}}(i,j)
=
\exp
\left(
-\eta R(x_i,x_j)
\right).
\]

This yields

\[
K_{ij}^{(t)}
=
\exp
\left(
-\frac{
\|u_i^{(t)}-u_j^{(t)}\|_2^2
}{
2\sigma_u^2
}
-\eta R(x_i,x_j)
\right).
\]

---

## 7.4 Mean-Shift Update

For \(t=0,\dots,T-1\),

\[
u_i^{(t+1)}
=
\frac{
\sum_j
K_{ij}^{(t)}
u_j^{(t)}
}{
\sum_j
K_{ij}^{(t)}
+\varepsilon
}.
\]

After several iterations, the points

\[
\{u_i^{(T)}\}
\]

may concentrate near a small number of modes.

The original proposal interprets these modes as dynamically generated hyperedge centers.

### Important mathematical qualification

Continuous mean-shift iterations are differentiable with respect to their inputs when the kernel is smooth and denominators remain nonzero.

However, converting the converged points into an **exact set of unique cluster centers**

\[
\{s_k\}_{k=1}^{K_{\mathrm{eff}}}
\]

requires a mode-merging or equivalence operation such as

\[
\|u_i^{(T)}-u_j^{(T)}\|<\epsilon,
\]

which is discrete or non-smooth.

Therefore, a practical differentiable implementation should avoid relying on an exact “number of unique coordinates” during training.

More robust alternatives include:

1. retaining all \(M\) attractor candidates but using sparse routing so only a few receive mass;
2. using differentiable density weighting and pruning only at inference;
3. using soft mode-merging during training and exact merging only for execution;
4. treating the attractors as a sparse continuous basis rather than requiring a literal discrete \(K\) during optimization.

---

## 7.5 Routing Through Attractors

Let \(s_k\) denote retained attractor centers.

Module-to-attractor routing can be defined as

\[
\ell^M_{jk}
=
-\frac{
\|\xi_j-s_k\|_2^2
}{
\tau_M
}
-\eta R(x_j,\operatorname{pos}(s_k)),
\]

followed by

\[
A_{j:}
=
\operatorname{sparsemax}
\left(
\ell^M_{j:}
\right).
\]

Similarly,

\[
\ell^Q_{qk}
=
-\frac{
\|\xi_q-s_k\|_2^2
}{
\tau_Q
}
-\eta R(q,\operatorname{pos}(s_k)),
\]

and

\[
\alpha_{q:}
=
\operatorname{sparsemax}
\left(
\ell^Q_{q:}
\right).
\]

Then

\[
\Pi_{qj}
=
\sum_k
\alpha_{qk}A_{jk}.
\]

Because the attractors arise from local physics-aware drift, \(\Pi\) is expected to exhibit approximately block-structured sparsity when the underlying physical system decomposes into local mechanisms.

---

## 7.6 Fine-Scale Decode

As before,

\[
\mathcal{N}(q)
=
\{j:\Pi_{qj}>0\},
\]

and

\[
c(q)
=
\sum_{j\in\mathcal{N}(q)}
\Pi_{qj}
\,
\Phi_{\mathrm{fine}}
\left(
q-x_j,\,
m_j,\,
e_{\mathrm{local}}(q,x_j)
\right).
\]

The hypergraph or attractor layer does not pool away module-level detail.

---

# 8. Strategy III: Large Sparse Dictionary / Mixture-of-Experts Routing

## 8.1 Concept

The third strategy is an engineering compromise.

Instead of explicitly creating a variable-size set of hyperedges, allocate a large candidate dictionary

\[
K_{\max}\gg K_{\mathrm{eff}}.
\]

Only a small number of candidates are allowed to become active for each module or query.

This approximates an effectively unbounded mechanism vocabulary while preserving static tensor shapes that are convenient for GPU execution.

---

## 8.2 Candidate Hyperedges

For

\[
k=1,\dots,K_{\max},
\]

let each candidate hyperedge contain a learned latent vector

\[
V_k\in\mathbb{R}^{d_h}
\]

and optionally a spatial anchor

\[
S_k\in\mathbb{R}^{d_x}.
\]

A module-routing logit may be

\[
\ell^{M}_{jk}
=
\frac{
(W_Mm_j)^\top V_k
}{
\sqrt{d_h}
}
-
\lambda_x
\|x_j-S_k\|_2^2
-
\eta R(x_j,S_k).
\]

Similarly,

\[
\ell^{Q}_{qk}
=
\frac{
(W_Qz_q)^\top V_k
}{
\sqrt{d_h}
}
-
\lambda_q
\|q-S_k\|_2^2
-
\eta R(q,S_k).
\]

---

## 8.3 Top-\(C\) Sparse Routing

Let

\[
\operatorname{TopC}(z)
\]

retain the \(C\) largest coordinates and assign

\[
-\infty
\]

to the others.

Then

\[
A_{j:}
=
\operatorname{sparsemax}
\left(
\operatorname{TopC}(\ell^M_{j:})
\right),
\]

and

\[
\alpha_{q:}
=
\operatorname{sparsemax}
\left(
\operatorname{TopC}(\ell^Q_{q:})
\right).
\]

Thus

\[
\|A_{j:}\|_0\le C,
\qquad
\|\alpha_{q:}\|_0\le C.
\]

### Differentiability qualification

The explicit Top-\(C\) operator is discrete at ranking boundaries.

Therefore this mechanism is not globally differentiable with respect to route membership.

In practice, one may use:

- straight-through top-\(C\) estimators,
- noisy top-\(k\) routing,
- continuous sparse routing during training followed by top-\(C\) execution,
- or differentiable sorting / selection relaxations.

Sparsemax alone is differentiable almost everywhere, but Top-\(C\) selection adds a separate discrete operation.

---

## 8.4 Environment-Based Candidate Suppression

For candidate \(k\), define an environment validity score

\[
g_k
=
G_{\mathrm{env}}(S_k,E).
\]

A hard invalidation rule may be written as

\[
\widetilde{\ell}_{jk}^{M}
=
\begin{cases}
\ell_{jk}^{M}, & g_k>0,\\
-\infty, & g_k\le 0.
\end{cases}
\]

For training, a continuous penalty is often preferable:

\[
\widetilde{\ell}_{jk}^{M}
=
\ell_{jk}^{M}
-
\lambda_g
\operatorname{softplus}(-g_k).
\]

This permits the candidate to become progressively suppressed while retaining useful gradients.

---

## 8.5 Sparse Query-to-Module Routing

The combined mask is

\[
\Pi
=
\alpha A^\top,
\]

with

\[
\Pi_{qj}
=
\sum_{k=1}^{K_{\max}}
\alpha_{qk}A_{jk}.
\]

A critical correction is required here:

> The fact that both \(\alpha_{q:}\) and \(A_{j:}\) are \(C\)-sparse does **not** imply that each query \(q\) connects to only \(C^2\) modules.

Instead,

\[
\Pi_{qj}>0
\]

whenever query \(q\) and module \(j\) share at least one active hyperedge.

If a selected hyperedge contains many modules, the number of active \(q\)-to-\(j\) pairs can still be large.

Therefore the true pairwise cost depends on **hyperedge occupancy**.

For a query \(q\), if its active hyperedges are

\[
\mathcal{K}_q
=
\{k:\alpha_{qk}>0\},
\]

then a useful upper-bound-like quantity is

\[
|\mathcal{N}(q)|
\le
\sum_{k\in\mathcal{K}_q}
|\mathcal{M}_k|,
\]

where

\[
\mathcal{M}_k
=
\{j:A_{jk}>0\}.
\]

Thus practical efficiency requires controlling both:

\[
\boxed{
\text{number of selected hyperedges per query}
}
\]

and

\[
\boxed{
\text{number of modules assigned to each active hyperedge}.
}
\]

This may require capacity constraints, load balancing, geometric locality, or occupancy regularization.

---

## 8.6 Delayed Pairwise Instantiation

After routing, construct

\[
\mathcal{P}
=
\{(q,j):\Pi_{qj}>0\}.
\]

The expensive fine interaction is evaluated only for pairs in \(\mathcal{P}\):

\[
p_{qj}
=
\Phi_{\mathrm{fine}}
\left(
q-x_j,m_j,e_{\mathrm{local}}(q,x_j)
\right),
\qquad
(q,j)\in\mathcal{P}.
\]

Then

\[
c(q)
=
\sum_{j:(q,j)\in\mathcal{P}}
\Pi_{qj}p_{qj},
\]

and

\[
\widehat{U}(q)
=
\operatorname{FieldHead}
\left(
\operatorname{LayerNorm}(c(q))
\right).
\]

In an efficient implementation, \(\mathcal{P}\) should be represented by packed index arrays or a sparse adjacency structure.

A generic sparse COO tensor is conceptually convenient, but a custom packed representation or fused segmented kernel may be substantially more efficient on GPUs.

---

# 9. Physics-Aware Manifold Connectivity

The central problem with purely Euclidean grouping is that

\[
\|x_i-x_j\|_2
\]

does not determine whether information should propagate between \(i\) and \(j\).

We therefore seek an affinity of the general form

\[
\mathcal{A}_{ij}
=
f
\left(
x_i,x_j,
m_i,m_j,
E
\right),
\]

with the intended property

\[
\mathcal{A}_{ij}\approx 0
\]

across physical barriers or disconnected regions.

A generic log-affinity is

\[
\log \mathcal{A}_{ij}
=
-\lambda_x
\|x_i-x_j\|_2^2
+
\lambda_m
\operatorname{sim}
\left(
\psi(m_i),\psi(m_j)
\right)
-
\eta R(x_i,x_j).
\]

The resistance term \(R\) may encode:

- solid walls,
- material interfaces,
- adiabatic boundaries,
- impermeable regions,
- discontinuities,
- anisotropic transport resistance,
- learned local connectivity,
- characteristic directions,
- streamline or path-based proximity,
- recirculation structure,
- combustion-zone connectivity,
- or other domain-specific physical constraints.

The desired topology should follow physical structures rather than axis-aligned geometric partitions.

---

# 10. Avoiding Early Pooling

The following architecture should be considered undesirable:

\[
h_k
=
\sum_i A_{ik}m_i,
\]

followed by

\[
c(q)
=
\sum_k \alpha_{qk}\Psi(q,h_k).
\]

This architecture compresses all modules belonging to hyperedge \(k\) into a single vector before the query interacts with them.

The preferred architecture is instead

\[
\Pi_{qi}
=
\sum_k \alpha_{qk}A_{ik},
\]

followed by

\[
c(q)
=
\sum_{i:\Pi_{qi}>0}
\Pi_{qi}
\Phi_{\mathrm{fine}}
(q-x_i,m_i,\cdots).
\]

The first formulation uses the hyperedge as a latent content carrier.

The second uses the hyperedge as an **indexing and routing structure**.

This distinction is fundamental.

---

# 11. Dynamic \(K\): What “Self-Assembling” Should Mean

A model should not be called fully self-assembling merely because many entries of a fixed \(K_{\max}\) dictionary remain unused.

There are at least three levels of adaptivity:

### Level 1: Fixed capacity, dense use

\[
K=K_{\max},
\]

with most candidates active.

This is not self-assembling.

### Level 2: Fixed capacity, sparse effective use

\[
K_{\mathrm{eff}}
\ll K_{\max}.
\]

This is adaptive sparse routing, but the representational capacity is still globally preallocated.

### Level 3: Dynamically instantiated topology

The number and identity of topological units arise from the current physical state:

\[
K_{\mathrm{eff}}
=
K_{\mathrm{eff}}(X,M,E).
\]

This most closely matches the intended notion of self-assembly.

However, Level 3 creates significant implementation difficulty because variable-cardinality topology operations are typically discrete.

Therefore, the central research question is not simply whether \(K\) can vary, but:

> Can the model achieve dynamically varying effective topology while retaining stable optimization, exact execution sparsity, scalable GPU implementation, and useful gradients for inverse design?

---

# 12. Computational Complexity

Assume:

- \(Q\): number of query points,
- \(M\): number of modules,
- \(K\): number of active hyperedges or routing centers,
- \(C_Q\): average number of hyperedges selected per query,
- \(C_M\): average number of hyperedges selected per module,
- \(D_k\): number of modules assigned to hyperedge \(k\).

A dense query-to-module decoder requires approximately

\[
\mathcal{O}(QM)
\]

fine interactions.

The sparse routed decoder cost is approximately

\[
\mathcal{O}
\left(
\sum_{q=1}^{Q}
|\mathcal{N}(q)|
\right).
\]

If each query activates at most \(C_Q\) hyperedges,

\[
|\mathcal{N}(q)|
\lesssim
\sum_{k\in\mathcal{K}_q}D_k.
\]

If occupancy is bounded by

\[
D_k\le D_{\max},
\]

then

\[
|\mathcal{N}(q)|
\le
C_QD_{\max},
\]

and the expensive decoding stage becomes approximately

\[
\mathcal{O}
\left(
Q C_Q D_{\max}
\right).
\]

This is the computational regime the architecture should target.

Therefore, **sparse hyperedge selection alone is insufficient**. Hyperedge occupancy must also be controlled.

---

# 13. Suggested Regularization Terms

Several regularizers may be required to prevent degenerate routing.

## 13.1 Hyperedge Occupancy Penalty

Let

\[
n_k
=
\sum_i \mathbb{I}(A_{ik}>0).
\]

A soft surrogate may penalize oversized hyperedges:

\[
\mathcal{L}_{\mathrm{cap}}
=
\sum_k
\operatorname{ReLU}
\left(
\widetilde{n}_k-D_{\mathrm{target}}
\right)^2,
\]

where

\[
\widetilde{n}_k
=
\sum_i A_{ik}.
\]

## 13.2 Routing Entropy or Concentration Control

Depending on the desired behavior,

\[
\mathcal{L}_{\mathrm{route}}
=
\sum_i H(A_{i:})
+
\sum_q H(\alpha_{q:})
\]

may be minimized or targeted toward a desired entropy range.

For Sparsemax, ordinary Shannon entropy is not always the most natural choice; Tsallis entropy is especially relevant for Entmax-family mappings.

## 13.3 Spatial Compactness

\[
\mathcal{L}_{\mathrm{compact}}
=
\sum_{i,k}
A_{ik}
\|x_i-s_k\|_2^2.
\]

This should not be used alone, otherwise the model may regress toward the geometric failure mode of 1805.

## 13.4 Physics-Barrier Penalty

\[
\mathcal{L}_{\mathrm{barrier}}
=
\sum_{i,k}
A_{ik}
R(x_i,s_k)
+
\sum_{q,k}
\alpha_{qk}
R(q,s_k).
\]

## 13.5 Load Balancing for MoE-Style Routing

For a large static dictionary, define mean usage

\[
u_k
=
\frac{1}{M}
\sum_i A_{ik}.
\]

A balancing objective may prevent collapse onto only one or two candidates:

\[
\mathcal{L}_{\mathrm{balance}}
=
K_{\max}
\sum_k
\left(
u_k-\frac{1}{K_{\max}}
\right)^2.
\]

However, this objective should be used carefully because strict uniformity may contradict the desired physical adaptivity.

---

# 14. Candidate Architecture for First Implementation

A pragmatic first implementation should preserve the scientific objective while minimizing engineering risk.

A recommended initial form is:

1. Use a moderately large candidate hub set rather than fully dynamic tensor cardinality.
2. Derive hub candidates from actual modules rather than free latent tokens.
3. Use physics-aware continuous logits.
4. Apply Sparsemax or \(\alpha\)-Entmax for exact support sparsity.
5. Control both per-query fan-out and hyperedge occupancy.
6. Construct explicit sparse \(q\)-to-module pair indices.
7. Evaluate \(\Phi_{\mathrm{fine}}\) only on those indices.
8. Avoid all hyperedge-level pooling in the final field-decoding path.

One possible formulation is:

\[
r_i
=
f_{\mathrm{hub}}
(m_i,x_i,e_{\mathrm{local}}),
\]

\[
w_i
=
\operatorname{ReLU}(r_i),
\]

\[
\ell^M_{ji}
=
\frac{
(W_Mm_j)^\top(W_Hm_i)
}{
\sqrt{d_h}
}
-\lambda_x\|x_j-x_i\|^2
-\eta R(x_j,x_i)
+
\lambda_h\log(w_i+\varepsilon),
\]

\[
A_{j:}
=
\operatorname{entmax}_{\alpha}
(\ell^M_{j:}),
\]

\[
\ell^Q_{qi}
=
\frac{
(W_Qz_q)^\top(W_Hm_i)
}{
\sqrt{d_h}
}
-\lambda_q\|q-x_i\|^2
-\eta R(q,x_i)
+
\lambda_h\log(w_i+\varepsilon),
\]

\[
\alpha_{q:}
=
\operatorname{entmax}_{\alpha}
(\ell^Q_{q:}),
\]

and

\[
\Pi
=
\alpha A^\top.
\]

This formulation keeps all modules as *candidate* hubs during training, but allows exact sparse support without introducing an explicit non-differentiable variable-length hub extraction step.

At inference or in an optimized sparse kernel, modules with zero hub usage may be removed from the active hub set.

This is not yet a mathematically literal dynamic-\(K\) object, but it provides a useful intermediate architecture that is much closer to the intended self-assembling behavior than a fixed set of arbitrary global hyperedges.

---

# 15. Open Research Questions for Model Development

The AI agent reviewing this proposal should explicitly investigate the following questions.

## 15.1 Does hypergraph routing preserve 1804-level accuracy?

The key empirical test is whether sparse routing can retain the accuracy of direct dense \(q\)-to-module interaction.

An ablation should compare:

\[
\text{Dense direct interaction}
\]

against

\[
\text{Sparse routing}
+
\text{direct interaction on the selected support}.
\]

The comparison must isolate routing effects from decoder-capacity effects.

## 15.2 How sparse can routing become before accuracy degrades?

Measure performance as a function of

\[
\mathbb{E}[|\mathcal{N}(q)|].
\]

The most important efficiency metric is not merely the number of hyperedges but the actual number of instantiated fine-scale \(q\)-module pairs.

## 15.3 Does physics-aware routing outperform geometric routing?

Compare:

\[
\ell_{ij}^{\mathrm{geo}}
=
-\lambda\|x_i-x_j\|^2
\]

against

\[
\ell_{ij}^{\mathrm{phys}}
=
-\lambda\|x_i-x_j\|^2
-\eta R(x_i,x_j)
+
\lambda_m\operatorname{sim}(m_i,m_j).
\]

Evaluation should specifically include cases with thin walls, nearby disconnected regions, material interfaces, recirculation zones, and other geometries where Euclidean proximity is misleading.

## 15.4 Is explicit dynamic \(K\) necessary?

A central empirical question is whether true variable-cardinality topology provides measurable benefit over a sufficiently large sparse candidate basis.

The comparison should distinguish:

- accuracy,
- memory,
- wall-clock runtime,
- GPU utilization,
- inverse-design stability,
- topology interpretability,
- and extrapolation to larger numbers of modules.

## 15.5 Which sparse activation is most suitable?

Compare at least:

\[
\text{Sparsemax},
\]

\[
\alpha\text{-Entmax},
\]

and, if appropriate,

\[
\text{Top-}C
\text{ gating with a surrogate gradient}.
\]

Relevant metrics include:

- support size,
- gradient stability,
- route turnover during training,
- convergence,
- inference sparsity,
- and inverse-design behavior.

## 15.6 How should topology changes occur during inverse design?

The agent should investigate whether support transitions create optimization pathologies.

In particular:

- Can a previously inactive hub become active?
- Can two mechanisms merge?
- Can one mechanism split?
- Do gradients vanish before a necessary topological transition occurs?
- Is stochastic exploration or temperature scheduling needed?
- Would Entmax with \(\alpha<2\) provide smoother support changes than Sparsemax?

## 15.7 What is the correct physical connectivity operator?

The proposed line-segment resistance

\[
R(a,b)
\]

is only one candidate.

More physically meaningful alternatives may include:

- learned graph geodesics,
- differentiable shortest-path surrogates,
- local anisotropic metrics,
- signed-distance fields,
- mesh connectivity,
- visibility-like operators,
- PDE-informed transport metrics,
- streamline distance,
- characteristic-direction kernels,
- or local message passing over an environment graph.

This component should not be accepted without dedicated ablation.

---

# 16. Recommended Ablation Matrix

A minimal serious experimental program should include:

| Variant | Routing | Dynamic topology | Physics-aware connectivity | Early pooling | Fine \(q\)-module interaction |
|---|---|---:|---:|---:|---:|
| Dense baseline | none | no | no | no | yes |
| 1401-like | fixed hyperedges | no | limited | yes | no / reduced |
| 1805-like | geometric groups | limited | no | yes | reduced |
| Sparse fixed dictionary | Sparsemax / Entmax | effective only | optional | no | yes |
| Module-elected hubs | sparse hub routing | yes / approximate | yes | no | yes |
| Mean-shift attractors | attractor routing | yes / approximate | yes | no | yes |

The principal evaluation axes should be:

\[
\text{field error},
\]

\[
\text{memory},
\]

\[
\text{latency},
\]

\[
\text{number of active }(q,i)\text{ pairs},
\]

\[
\text{topology stability},
\]

and

\[
\text{inverse-design optimization quality}.
\]

---

# 17. Key Failure Modes to Watch

The proposed architecture may fail in several ways.

### 17.1 Hyperedge Collapse

Many modules may route to one dominant hyperedge.

### 17.2 Hyperedge Fragmentation

The model may create too many tiny groups, eliminating computational benefit.

### 17.3 Physically Incorrect Shortcuts

Learned routing may bypass walls or interfaces unless physical barriers are represented explicitly and effectively.

### 17.4 Dead Hubs

Exact sparse activations may create permanently inactive candidates with no useful gradient path for reactivation.

### 17.5 Routing Churn

Small geometry changes may cause abrupt support changes, destabilizing inverse design.

### 17.6 Sparse Routing Without Sparse Execution

A mathematically sparse \(\Pi\) does not automatically imply runtime savings if the implementation still computes all logits, all pair interactions, or dense intermediate tensors.

The implementation must therefore distinguish:

\[
\text{representational sparsity}
\]

from

\[
\text{computational sparsity}.
\]

### 17.7 Large Hyperedge Occupancy

Even if each query selects only a few hyperedges, the method may remain expensive if those hyperedges contain many modules.

---

# 18. Final Design Principle

The desired architecture can be summarized by the following computation:

\[
\boxed{
\begin{aligned}
&\text{physical state}
\\
&\quad\downarrow
\\
&\text{dynamic / sparse topology construction}
\\
&\quad\downarrow
\\
&\text{query-to-mechanism routing}
\\
&\quad\downarrow
\\
&\text{sparse support }
\mathcal{N}(q)
\\
&\quad\downarrow
\\
&\text{direct fine-scale }
(q,i)
\text{ interactions}
\\
&\quad\downarrow
\\
&\widehat{U}(q)
\end{aligned}
}
\]

The hypergraph should answer:

\[
\boxed{
\text{Who should interact with whom?}
}
\]

It should **not** answer that question by first destroying the low-level information required by the interaction.

The main architectural hypothesis is therefore:

> Dense pairwise physics need not imply dense global computation. A dynamically generated, physics-aware sparse routing topology may preserve the high-fidelity query-to-module interaction of the dense model while avoiding the redundant global computation identified by later sparse experiments.

The most important implementation criterion is that the sparse topology must be converted into **actual sparse execution**. The expensive fine-scale interaction operator should be instantiated only on the routing support, while the topology layer itself remains lightweight enough that the cost of discovering sparsity does not negate the benefit of exploiting it.
