# Next-generation HONF: hypergraph-conditioned sparse quadrature

**Status:** research design, not an implemented model or a demonstrated speedup.  
**Suggested experiment label:** Run 1408, subject to the user's authorization and the existing run allocator.  
**Repository reference:** `cosmos2w/ModularDT`, `agent/honf-core-next`, commit `07a255ea0acfd7e61d46aa58324baf6535242b2b`.  
**Primary evidence:** `HONF_Run1406_Run1407_BestBefore5000_Comparative_Evaluation.md`.

## 1. Decision in one paragraph

Keep the successful phase-shared, low-dimensional many-body controller and the fine, phase-dependent source states. Replace only the environmental query reader's **enumerate-all-sources-then-weight** formulation with a **predict-a-small-integration-stencil-then-read** formulation. Each hyperedge controls a small set of query-dependent sampling locations and weights. Fine keys and values are obtained from the existing environmental response grid using local interpolation, not by reading a pooled group value. The environmental micro-kernel is evaluated only at these locations. A fixed stencil budget makes its cost bounded before training; entmax is no longer expected to discover a hardware-efficient number of environment interactions accidentally.

The decisive change is mathematical, not a new sorting method or custom sparse kernel:

\[
\text{learn weights on an }E\text{-source candidate set}
\quad\longrightarrow\quad
\text{learn a small integration rule for that response}.
\]

This is a learned cubature approximation of the environmental read. It is not an exact rewrite of Run 1407, not an unbiased Monte Carlo estimator, and not a claim that arbitrary field detail can be recovered from a fixed number of samples.

## 2. What the mature evidence actually says

Use the report's **best-by-validation-field at or before epoch 5000** policy as the primary result. Retain exact epoch 5000 as a separate sensitivity analysis.

| Primary checkpoint result | 1404 | 1406 | 1407 | Dense 1804 |
|---|---:|---:|---:|---:|
| Pooled fluid relative L2 | 0.03460 | 0.03250 | 0.03065 | 0.02896 |
| Case p95 | 0.06028 | 0.05439 | 0.05731 | 0.05281 |
| Near-interface relative L2 | 0.03337 | 0.04122 | 0.03749 | 0.03504 |
| Field temperature relative L2 | 0.04873 | 0.04419 | 0.04198 | 0.03867 |
| Heat-flux relative L2 | 0.13784 | 0.11712 | 0.11259 | 0.10615 |
| Full forward, milliseconds | 25.38 | 38.90 | 38.21 | 33.98 |
| Prepared P2, milliseconds | 6.39 | 12.03 | 14.08 | 13.02 |

Run 1407 improves selected-checkpoint accuracy over 1406 and reuses its controller as intended. It does not dominate Dense, and exact-endpoint ordering reverses between 1406 and 1407. Do not turn a 50- or 500-epoch result into a final architectural verdict.

The central cost finding is not the number of visible hyperedges. Run 1407 has 73.22% environmental logical support but executes the full QE rectangle. Valid module support remains complete. Only about 2.2% of the combined fine rows are removed, predominantly padding rather than physical source selection.

Both group-control models use only about two module-bearing groups. This is functional organization with partial collapse, not evidence for six identifiable physical mechanisms. Frozen interventions establish dependence, not superiority or physical causality. The report's secondary all-domain aggregate has less secure attribution and must not replace the fluid, internal-temperature, or interface metrics.

### Current-code observations

- `phase_shared_group_control.py` constructs the controller at P0 and reuses it at later phases while refreshing fine states/K/V.
- Its hybrid support policy mainly changes QM execution. QE remains rectangular.
- `group_control_pairwise.py` computes environmental content and geometry for all sources in the normal rectangular reader, then applies overlap masks/weights.
- `environment.py` in ThermalChannel supplies a cell-centred regular `24 x 8` grid, with x varying fastest. A reusable core must obtain this information explicitly from the adapter, not assume an arbitrary token sequence can be reshaped into an image.
- The accepted full-QE activation checkpoint must be retained conceptually for the new reader. Reopening rejected diagnostic-elision and Fourier micro-optimizations is not the main task.

## 3. Scope: one new operator, not another collection of branches

Preserve:

1. `InterfaceFieldCore` and the physical P0/P1/P2 coupling;
2. Dense fine MM/ME/EM preparation and phase-dependent module/environment response states;
3. the phase-shared P0 controller, fixed `K=6`, and control width `D=16`;
4. the current module reader `C_M(q)` and the simple background `C_g(q)`;
5. the final three-term assembly and field head;
6. the local surrogate, physical losses, data, normalization, optimizer policy, and historical models.

Replace only the environmental receiver computation:

\[
C(q)=C_g(q)+C_M(q)+C_E^{\mathrm{SQ}}(q).
\]

There is no new coarse bank, local correction, group-value bypass, count estimator, routing penalty, temperature schedule, top-k truncation, or tree. No sparsity claim is made for MM/ME/EM preparation or for the small-module QM branch.

## 4. Core interpretation: a hyperedge specifies an integration rule

A hyperedge should no longer be an unbounded source list that must be searched before computation. In this proposal it contains:

- a shared many-body control state, built from modules, environment, and operating conditions;
- a small, learned rule mapping a receiver to environmental sampling sites;
- positive integration weights;
- the same rule reused at module ports, refinement points, and field queries.

The rule is stable within one physical forward, but its locations depend on the receiver. Fine sampled values change with each physical preparation. This is **a phase-shared program with query-dependent incidence**, not one immutable spatial subset for an entire design.

The sparse environmental read set is explicit:

\[
\mathcal N_E(q)
=
\bigcup_{k,r}\mathcal C(\xi_{qkr}),
\]

where `C(xi)` is the small interpolation-cell vertex set. It is known before environmental content dots and geometry MLP calls.

## 5. Notation

- `B`: number of cases.
- `M`: number of active physical modules; storage padding is separate.
- `E`: fine environmental tokens, currently 192.
- `Q`: requested receivers.
- `K=6`: group count; not attention-head count.
- `D=16`: low-dimensional controller width.
- `H=256`: fine hidden width.
- `n_h=4`, `d_h=H/n_h`: attention heads and head width.
- `p in {0,1,2}`: physical preparation phase.
- `R=4`: samples per group in the initial candidate.
- `J=K R=24`: maximum sample slots per receiver, shared across attention heads.
- `y_j`, `omega_j`: environment coordinates and normalized quadrature masses, with `sum_j omega_j=1`.
- `A^E_jk`: P0 environment/group assignment; rows sum to one.
- `h_k`: phase-shared group control.
- `alpha_qk`: phase-shared-controller query routing; rows sum to one.
- `K_j^(p), V_j^(p)`: current-phase projected environmental keys and values, including inherited source-side value modulation.
- `xi_qkr`: learned sample coordinate.
- `beta_qkr`: within-group positive sample weight, normalized over r.

Use different symbols in code/docs for group keys, attention keys, group count, and sample count.

## 6. Start from the exact environmental operator

For one attention head, suppressing the head index, Run 1407 calculates

\[
\rho_{qj}=\sum_k\alpha_{qk}A^E_{jk},
\qquad
\mu_k=\sum_j\omega_j A^E_{jk},
\qquad
G_q=\sum_k\alpha_{qk}\mu_k.
\]

Let `s_qj^(p)` be the inherited content/geometry score **before** adding log source measure and log overlap. Then its environmental response before output projection is

\[
 o_q^{(p)}=
 \frac{\sum_j\omega_j\rho_{qj}e^{s_{qj}^{(p)}}V_j^{(p)}}
 {\sum_j\omega_j\rho_{qj}e^{s_{qj}^{(p)}}}.
\tag{1}
\]

Its context includes the inherited amplitude and output-projection bias:

\[
 C_E^{(p)}(q)=K G_q\,\operatorname{Out}(o_q^{(p)}).
\tag{2}
\]

This amplitude convention should be preserved in the new candidate so the experiment does not also alter branch scaling.

Because all assignments are nonnegative, Equation (1) can be regrouped **exactly**:

\[
\begin{aligned}
N_q &=\sum_k\alpha_{qk}\sum_j\omega_jA^E_{jk}
       e^{s_{qj}}V_j,\\
D_q &=\sum_k\alpha_{qk}\sum_j\omega_jA^E_{jk}
       e^{s_{qj}}.
\end{aligned}
\tag{3}
\]

For nonempty groups define the probability measure

\[
\nu_k=\sum_j\frac{\omega_j A^E_{jk}}{\mu_k}\delta_{y_j}.
\tag{4}
\]

The current network therefore evaluates a group mixture of integrals, but computes them by exhaustive enumeration. This is the mathematical location where sparse quadrature enters. No division by `mu_k` is needed in the deployed new operator.

## 7. Replace exhaustive group integration by a small learned rule

For each receiver and group, learn

\[
\int F_q(y)\,d\nu_k(y)
\;\approx\;
\sum_{r=1}^{R}\beta_{qkr}F_q(\xi_{qkr}),
\qquad
\beta_{qkr}>0,\quad\sum_r\beta_{qkr}=1.
\tag{5}
\]

Here `F_q` is either the attention numerator integrand or denominator integrand. The same sites and weights are used for both. This avoids inconsistent numerator/denominator approximation.

Define unnormalized sample masses

\[
\lambda_{qkr}=\alpha_{qk}\mu_k\beta_{qkr}.
\tag{6}
\]

Then

\[
\sum_{k,r}\lambda_{qkr}=G_q.
\tag{7}
\]

The new response is

\[
\boxed{
\widehat o_q^{(p)}=
\frac{\sum_{k,r}\lambda_{qkr}e^{s^{(p)}(q,\xi_{qkr})}
                   \mathcal I[V^{(p)}](\xi_{qkr})}
     {\sum_{k,r}\lambda_{qkr}e^{s^{(p)}(q,\xi_{qkr})}}.
}
\tag{8}
\]

Finally,

\[
\boxed{C_E^{\mathrm{SQ},(p)}(q)
 =K G_q\,\operatorname{Out}(\widehat o_q^{(p)}).}
\tag{9}
\]

When `G_q=0`, return exactly zero including the output bias. Do not introduce a learned null gate or a tiny pseudo-source. Implement positive sample masses using stable masked log normalization; never take an unguarded `log(0)` or all-negative-infinity softmax.

### What is retained and what is approximated

The regrouping in Equation (3) is exact. Replacing each group measure by R query-conditioned points is the **new modeling approximation**. It does not preserve Run-1407 predictions without retraining.

Sampling sites need not coincide with original tokens or exact support members of `A^E`. `A^E` describes the measure the learned rule aims to represent; the executed interpolation incidence is defined by `xi`. Do not render the old `A^E` matrix as the new executed support. There is no theorem that the learned quadrature is accurate for arbitrary feature fields.

## 8. A small, bounded, differentiable site generator

Keep the existing query-control vector `v_q` and group control `h_k`. Add one shared small network, not a network per group:

\[
(t^a_{qkr},t^g_{qkr},t^w_{qkr})_{r=1}^{R}
=\operatorname{MLP}_{\mathrm{sample}}([v_q,h_k]).
\tag{10}
\]

Use input width `2D`, hidden width `2D`, and output width `R(d+2)` in dimension d. The output consists of d anchor-coordinate logits, one mixing logit, and one weight logit per site.

Let `l_c,u_c` be the min/max environment-grid centre coordinates and `l_O,u_O` the physical rectangular-domain bounds supplied by the adapter. Define

\[
\bar q=l_c+(u_c-l_c)\odot\frac{q-l_O}{u_O-l_O},
\tag{11}
\]

\[
a_{qkr}=l_c+(u_c-l_c)\odot\sigma(b^{\rm ref}_{kr}+t^a_{qkr}),
\qquad
g_{qkr}=\sigma(t^g_{qkr}),
\tag{12}
\]

\[
\boxed{\xi_{qkr}=(1-g_{qkr})\bar q+g_{qkr}a_{qkr},}
\qquad
\beta_{qk:}=\operatorname{softmax}_r(t^w_{qk:}).
\tag{13}
\]

For receivers inside the physical rectangle, these sites stay inside the environmental-centre hull. There is no hard distance cutoff and no assertion that long-range physics is negligible. Learned `g` allows sites to remain close to the receiver or move toward remote anchors.

`b_ref[K,R,d]` is a small learnable anchor-logit table initialized to a deterministic stratified set of J interior locations. Use an approximately uniform 6-by-4 normalized layout for K=6/R=4 in 2D. Initialize site-weight and mixing biases at zero and the final sampler weights at small nonzero scale. Do not run an initialization sweep. Group permutations must permute the corresponding reference-anchor table as well as the group/controller entries.

Equation (11) is an explicit inward coordinate map because the supplied environment is cell-centred and contains no boundary tokens. It is not a physical boundary condition. Inspect near-interface/boundary errors for bias from this choice. The first implementation supports regular rectangular environmental grids only; do not claim arbitrary-domain sampling is already solved.

## 9. Fine information is sampled, not replaced by h

`I` denotes bilinear interpolation on the supplied 2D grid:

\[
\mathcal I[F](\xi)=\sum_{j\in\mathcal C(\xi)}b_j(\xi)F_j,
\qquad b_j\ge0,\quad\sum_jb_j=1,
\quad |\mathcal C(\xi)|\le4.
\tag{14}
\]

Interpolate **prepared projected keys and modulated values**, not pooled group values. Interpolation after projection/normalization is a specified part of this model and must not be silently replaced with interpolation before nonlinear normalization.

The score uses the same inherited ingredients at the sampled point:

\[
 s_a^{(p)}(q,\xi)
 =\frac{Q_a(q)^\top\mathcal I[K_a^{(p)}](\xi)}{\sqrt{d_h}}
   [1+\tanh\zeta_a(q,\xi)]
 +b_a(\Phi((q-\xi)/s)).
\tag{15}
\]

The inherited low-dimensional head control has the continuous extension

\[
\zeta_a(q,\xi)
=\sum_{k'}\alpha_{qk'}\,
 \mathcal I[A^E_{:k'}\,b_a^{\mathrm{ctrl}}(h_{k'})](\xi).
\tag{16}
\]

Prepare the scalar/head-control maps once in the shared controller. Do not form a query-by-E-by-D tensor. Query keys, sampled fine keys/values, and scalar geometry remain differentiable.

There is no direct addition of `h_k` to `C_E` or the final field. The controller changes **where and how the fine field is read**.

### Local interpolation is an approximation too

Bilinear interpolation is continuous and differentiable almost everywhere, not globally C1. Gradients can change across cell boundaries. It preserves constant/affine sample fields, but does not preserve arbitrary high-frequency content. A fixed 24-site budget is an expressivity tradeoff, not free information. Do not dismiss this as harmless or claim mesh convergence without experiments.

## 10. Exact reference identities and useful error interpretation

### Dense-node reference

For a toy check only, give group k every original node and set

\[
\xi_{qkr}=y_j,
\qquad
\beta_{qkr}=\omega_j A^E_{jk}/\mu_k.
\]

Since the fine score/value at a node is independent of the label of the group proposing it, Equation (8) reduces to Equation (1). This is an arithmetic reference, not a deployed K-times-dense executor.

### Constants

If all fine values are the same vector, the normalized environmental response returns that vector for any valid sites/weights. The original `K G_q` amplitude and output projection still apply.

### Error is about the integrand, not a pretty sampling plot

With exact numerator/denominator `(N,D)` and their quadrature versions `(Nhat,Dhat)`, for positive denominators:

\[
\left\|\frac{\widehat N}{\widehat D}-\frac ND\right\|
\le
\frac{\|\widehat N-N\|}{\widehat D}
+
\frac{\|N\|\,|\widehat D-D|}{\widehat D D}.
\tag{17}
\]

Consequently, preserving sample mass alone does not establish response accuracy. Measure numerator/denominator or context discrepancies in bounded diagnostics and, ultimately, actual physical errors after the complete feedback loop.

## 11. Where the many-body dependence lives

The P0 controller observes multiple module and environmental states jointly. The learned site generator receives this shared controller:

\[
\{z_i, e_j, g\}\rightarrow h_k
\rightarrow (\xi_{qkr},\beta_{qkr})
\rightarrow \widehat U(q).
\]

A second module can change a group's read sites and hence the fine information acquired for a first module's surrounding field. This supplies nonlinear many-body dependence without executing a fine `(q,j,k)` function for every environmental j.

It does not prove many-body **physical correctness**. Attention models can implement equivalent conditioning. The hypergraph-specific claim requires showing that case-dependent group conditioning improves the accuracy/cost/compositional tradeoff relative to an equal-budget query-only sampler. An attractive group diagram or a large frozen ablation effect is not sufficient.

## 12. The budget becomes structural

For K=6, R=4, J=24 in 2D:

- environmental content/geometry kernels: at most `Q * 24` sample rows;
- bilinear source-bank corner loads: at most `Q * 24 * 4 = Q * 96` incidences per bank;
- unique accessed environment cells per query: at most `min(E,96)`;
- the old environmental reader: `Q * 192` content/geometry rows on the present grid.

Therefore the fine sample-score/geometry row budget is 12.5% of the old QE candidate count; the upper bound on interpolation-corner incidences is 50%. Neither number is an overall FLOP, memory, or latency ratio.

Use the same sample locations across all attention heads. Head-specific sites would multiply the memory-access budget and change this calculation.

Evaluate all J slots initially, including those with zero group mass. That preserves regular tensor shapes. Their zero-weight contribution does not mean their sample execution was free. Do not count them as pruned unless the code truly avoids their work.

Repeated or nearly identical sites are not sorted/deduplicated in the first implementation. Their total count remains J. Report wasted/repeated sampling; do not reintroduce the historical raw-path compiler to fix a tiny fixed-budget set.

### Costs not reduced

The new read does not eliminate full fine environmental preparation, fine K/V refresh, MM/ME/EM, the complete valid-module read, or local-surrogate calls. It does not imply that a globally omitted physical module has zero influence. With h and other prepared quantities live, unsampled source inputs may influence sampled values through earlier preparation.

## 13. Hardware implementation: fixed tensor shape before custom kernels

Use one normal framework sampler call on a packed source bank rather than E-sized support sorting or millions of edge-list operations.

Suggested packed map per phase:

`[B, 2*H + K*n_h, Ny, Nx]`

containing keys, values, and shared source/head-control maps. For a receiver chunk Qc, the sampling grid is:

`[B, Qc, J, 2]`.

Sampling returns roughly:

`[B, 2*H + K*n_h, Qc, J]`.

Reshape once into attention-head dimensions, compute the J scores and normalize over the flattened `(k,r)` axis per head, then accumulate values.

No `[B,Q,E]` candidate-score or overlap matrix is needed in the new QE read. `G_q = alpha @ mu_E` is only `[B,Q]`. No `[B,Q,E,H]` and no `[B,Q,E,K]` tensor should appear.

### Do not ignore sampled-bank materialization

For B=1, Qc=2048, J=24, H=256, a sampled key bank or value bank occupies about 48 MiB in FP32. Keys and values together are about 96 MiB before other tensors. That can exceed the attention-score storage of an efficient dense GEMM reader even though J is much smaller than E.

Use the existing outer query chunk; measure sampled forward/backward memory. Keep non-reentrant checkpointing around the whole sampled QE block to avoid retaining every sampled key/value/score across all physical reads. Do not checkpoint dense preparation again by default.

Start with `torch.nn.functional.grid_sample` and a four-corner reference implementation for tests. Record the installed PyTorch version and sampling conventions. CUDA sampling backward can be nondeterministic; do not demand bitwise matching or disable an existing reproducibility policy silently. The initial task needs first derivatives; second-order support is a separate capability to test before any Hessian/PDE-derivative use.

Use `align_corners=True` only with explicit centre-hull coordinates: map xi to `2*(xi-l_c)/(u_c-l_c)-1`. Singleton grid axes need a defined constant-axis implementation. Because Equation (13) keeps sites in the hull, padding does not supply fabricated physical values. Test exact node positions, cell centres, and affine interpolation. Do not rely on the default `align_corners` value.

Custom fused sampling/attention is a later optional executor improvement if a measured hotspot justifies it. Do not make custom CUDA/Triton or a new dependency part of the first scientific design.

## 14. Generic versus case-specific code

### Generic architecture

Suggested opt-in name:

`forward_architecture = "hypergraph_quadrature_honf"`.

Add narrowly scoped files:

- `interface_fields/hypergraph_quadrature.py`: phase-shared backend with the replacement QE read;
- `interface_fields/environment_sampling.py`: typed regular-grid layout and interpolation helpers;
- sampler network can live in `hypergraph_quadrature.py`; do not create a second routing framework.

Reuse `PhaseSharedGroupControlPairwiseField` where inheritance is clean. Override the environmental read and phase preparation that packs the sampled bank. Preserve the parent QM, controller construction, module overlap/control equations, and output context. Existing phase state may be wrapped in a small runtime object for sampling metadata; never persist autograd caches in checkpoint state.

If parent class preparation creates unused environment support tables, leave their removal as a localized exact optimization, not an excuse to refactor every family. The sampled QE path must not call them.

### Core plumbing

Add the architecture to the existing factory/config allowlists and phase-shared-controller recognition. Reuse `ThreeTermInterfaceContext`. Extend `BatchData`/`EncodedInterfaceCase` with an optional sampler-layout field appended with a `None` default, or use an already compatible metadata seam. Historical constructors, checkpoint keys, and behavior must remain unchanged.

The field and physical-port interfaces must call the same sampled environmental reader at P0, P1, P2, and consistency reads. Diagnostic flags must not switch to a different mathematical predictor.

### ThermalChannel ownership

The environment builder owns:

- physical rectangle bounds;
- x/y centre coordinates and lattice sizes;
- a validated token-to-lattice map;
- input sample weights;
- interpretation of environment/boundary features.

The generic reader must not hard-code 24, 8, channel length, module radius, thermal walls, or token order. The first supported implementation is the regular rectangular grid already present. Unsupported irregular/obstructed layouts should fail with an informative adapter-support message, not silently execute a guessed grid reshape or a dense fallback.

For token-permutation tests, permute the mapping consistently. For duplicate-weight tests, exact duplicate physical samples should be coalesced measure-consistently in the adapter's reference mapping or tested through the quadrature identity; do not force an arbitrary duplicated sequence into a grid.

### Configuration

Inherit the scientific profile of 1407. The only new scientific setting needed is:

```text
samples_per_group = 4
```

Keep K=6, D=16, H=256, four attention heads, P0/P1/P2, the parent losses, optimizer, seed, and data. The bounded site generator and interpolation convention are part of the documented model. Do not introduce a broad sampler configuration tree, multiple backends, or tuning schedules.

## 15. Compact implementation sequence

### Task 1 — establish one actual sampled reader

Read the parent report and relevant source. Implement the new QE equation and adapter layout. Execute:

- dense-node reference identity and constant/affine interpolation;
- first-gradient checks away from interpolation knots;
- an actual predicted-port physical backward/update;
- one parent/new untrained-architecture execution comparison on the same data/GPU.

The latter measures cost only; it does not predict trained accuracy. A smaller sample-row count cannot substitute for actual timing and memory measurement.

### Task 2 — one short scientific trial

After user authorization, create one fresh candidate (proposed Run 1408) from scratch on a free GPU. Preserve other running jobs. Use the same sample budget throughout.

Train to epoch 50, inspect learning and actual execution, and normally continue the same optimizer/run to 500 if numerically healthy and reasonably competitive. The report shows why early slow convergence alone is not a reliable rejection criterion. Do not require strict dominance of every parent metric by epoch 50. Do not use this observation to excuse NaNs, a frozen sampler, or large persistent physical-error deterioration.

No automatic second candidate, seed, schedule, controller freeze, or sample-count sweep. No changes to the scientific model inside one run. Tensor-layout improvements that preserve the same sampled operator may be documented separately.

### Task 3 — answer the scientific question

At 500 evaluate exact endpoint and saved-best-through-500 separately against matched 1407, 1406, and Dense 1804 evidence. Include historical 1404 near/interface behavior. Do not mix mature parent checkpoints into the matched-budget table.

The full 5000-epoch continuation requires a later user decision. Keep milestone checkpoints for 50, 100, 250, 500, 1000, 2500, and 5000 as supported by the existing workflow.

## 16. Minimal but decisive evaluation

### Fidelity

Retain pooled/equal-case fluid errors, p95/worst, near/far, all physical channels, internal/surface temperature, heat flux, final ports, and existing composition strata. Focus especially on temperature, heat flux, wakes/vorticity, and interface gradients. Do not let pressure improvement conceal thermal degradation.

Use 0273/0653 and unfavorable cases 0295/0297/0298 as bounded anchors. The full 90-case development population remains the main existing reconstruction comparison; it is not new CFD evidence.

### Cost

Measure actual full forward, P0/P1/P2 preparation/read, prepared P2, B48/Q1024 M1/M12 optimizer steps, allocated/reserved memory, and sampled-bank peaks. Use the maintained warmups and synchronized repetitions, both model orders when the difference is small. Do not add nested event times into a new synthetic total.

Record independently:

- sample slots executed;
- nonzero sample masses;
- fine content-dot and geometry-MLP rows;
- interpolation-corner incidences and unique grid cells touched;
- old-style source memberships (descriptive only);
- common preparation work and QM work;
- backward checkpoint recomputation.

Evaluate actual 192/768/3072-source shapes only as execution scaling, not larger-domain physical accuracy. A fixed sample budget gives an environmental-read complexity advantage, not demonstrated domain-extrapolation quality.

### Group usefulness

Use a few frozen counterfactuals, without training more models:

1. Freeze sampling coordinates to a reference case or fixed initial geometry while keeping current fine values and other branches.
2. Keep sites fixed but neutralize learned within-group quadrature weights.
3. Perturb two modules, recomputing the full physical loop, and inspect changes in group programs, sampled cells, interfaces, and field.
4. Test phase-specific removal of the sampled environmental term, keeping the existing P0/P1/P2 definitions explicit.

Always report ground-truth error change alongside prediction discrepancy. These establish trained-model reliance only. If the candidate is promising, one later same-budget query-only deformable sampler control is necessary for a strong claim that hypergraph conditioning is uniquely useful; it is not automatically authorized here.

### Gradients and physical interpretation

Check first derivatives to sampling coordinates, module coordinates, fine K/V, controller parameters, and sampler parameters. Test crossing interpolation knots with one-sided and two-sided finite differences; do not call a piecewise-differentiable field globally smooth.

The module-number/type inverse problem remains discrete. Continuous sampling does not solve discrete self-assembly. The existing physical perturbation requests remain the path to independent multi-body/sensitivity validation. Never use surrogate outputs as CFD truth.

## 17. Visualization that reflects the executed model

For the same anchor cases and selected queries, show:

- fine response grid and module geometry;
- group-colored learned sampling sites;
- the four actual interpolation neighbours of each site;
- site weights and the union of cells actually read;
- site displacement when one or two modules are perturbed;
- raw fine-row and interpolation counts, with measured latency.

Keep the familiar controller incidence plot as secondary context. Label it as controller construction, not executed QE support. Group labels remain permutation ambiguous. Prefer a few readable selected-query diagrams to dense fans of thousands of lines.

## 18. Failure modes that must not trigger automatic architectural accumulation

**Sample collapse:** several sites coincide. Measure unique cells/site separation and physical errors. Do not immediately add repulsion or coverage losses.

**Missing boundary detail:** interpolation and finite sampling may damage gradients or narrow wakes. Assess errors and sample geometry before adding a near branch or second latent bank.

**No hardware gain:** grid sampling can create expensive H-wide gathered tensors. Profile the sampled bank and checkpoints; do not claim an 8x speedup from 8x fewer score rows.

**Sampler saturation:** sigmoid anchors/mixing can saturate; entmax can leave empty groups. Inspect site-coordinate gradients and actual updates, not just total network gradients.

**False interpretation:** learned sites may be effective mathematical quadrature but not identifiable physical regions. Report them accordingly.

**Accuracy-cost tradeoff remains ordinary:** possible and scientifically informative. Do not relabel it a hypergraph breakthrough.

## 19. Amdahl limit and interpretation of success

The mature report attributes about 8.30 ms to P2 QE out of a roughly 38.21 ms full forward. These are instrumented/nested and unprofiled quantities respectively, so they cannot be combined as an exact performance prediction. As a rough ceiling illustration, even removing 26.78% of all QE work perfectly would save only about 2.22 ms at that phase, before indexing overhead. The current small mask change is therefore an inadequate basis for a dramatic speed claim.

The new proposal seeks a different scale of reduction at the environmental micro-kernel. Its other costs remain. Report preparation and one-shot versus repeated-query results separately. Dominance over Dense is not guaranteed.

Success requires actual environmental micro-work reduction, competitive multi-channel/interface fidelity, and useful case-conditioned sampling. A speedup obtained only by using fewer informative measurements with unacceptable thermal error is not success. A sampler with good fidelity and no measured speedup is an accurate model but not the intended computational advance.

## 20. Compatibility and research workflow

Do not change old architectures or accepted checkpoint loaders. Do not disable safety checks or reproducibility settings silently. Preserve historical profiles and output definitions. Use ordinary Git, existing configs/run IDs, and standard targeted tests. Add no cryptographic hashes, snapshots, contract freezes, approval services, or monitoring daemons. Numerical checks are ordinary research verification, not a new production-gating framework.

Reuse existing report/evaluation paths and avoid duplicating source runs or the entire 90-case evidence bundle. Save only new diagnostic arrays needed to establish sample locations, measures, gradients, and actual work. No long training is launched by creating this design document.

## 21. Methodological position and references

This is not a claim to have invented deformable attention or adaptive quadrature. The intended novelty, if demonstrated, is a shared many-body-conditioned integration program that couples reusable local physical models and a continuous global field under an explicit fine-read budget.

Primary sources consulted:

1. Zhu et al., **Deformable DETR: Deformable Transformers for End-to-End Object Detection**, ICLR 2021. Sparse query-conditioned sampling with bilinear interpolation is an important precedent. Detection results do not transfer as accuracy claims for HONF. https://arxiv.org/abs/2010.04159
2. Xia et al., **Vision Transformer with Deformable Attention**, CVPR 2022. Data-dependent positions of keys and values provide another relevant comparison. https://openaccess.thecvf.com/content/CVPR2022/html/Xia_Vision_Transformer_With_Deformable_Attention_CVPR_2022_paper.html
3. Chien et al., **You are AllSet: A Multiset Function Framework for Hypergraph Neural Networks**, ICLR 2022. Supports the group-as-shared-set-function interpretation, not uniqueness relative to attention. https://arxiv.org/abs/2106.13264
4. Li et al., **Geometry-Informed Neural Operator for Large-Scale 3D PDEs**, NeurIPS 2023. Relevant separation of geometry handling, fine/latent representations, and continuous-domain queries; its guarantees are not inherited by this proposal. https://arxiv.org/abs/2309.00583
5. PyTorch, **torch.nn.functional.grid_sample** documentation. Check the installed-version semantics, align-corners convention, gradient order, and CUDA nondeterminism. https://docs.pytorch.org/docs/main/generated/torch.nn.functional.grid_sample.html

Repository files inspected at the pinned commit:

- `src/honf_forward_core/interface_fields/phase_shared_group_control.py`
- `src/honf_forward_core/interface_fields/group_control_pairwise.py`
- `src/honf_forward_core/interface_fields/types.py`
- `Case_ThermalChannel/src/channelthermal/environment.py`

## 22. Independent algebra checks performed for this proposal

A small standalone CPU float64 prototype, not the HONF repository or dataset, checked:

- all-node quadrature recovery of the parent mixture: maximum absolute discrepancy `1.11e-15`;
- group-sample prior mass identity: error `2.78e-17`;
- affine bilinear interpolation: maximum discrepancy `2.22e-16`;
- interpolation-coordinate gradient on an affine field: exact within that calculation;
- interpolation first-gradient check: passed;
- numerator/denominator error inequality on a toy instance: passed.

No model was trained; no checkpoint was loaded into this runtime; no GPU or CFD measurement was performed. The accompanying algebra JSON is optional evidence, not an approval prerequisite.
