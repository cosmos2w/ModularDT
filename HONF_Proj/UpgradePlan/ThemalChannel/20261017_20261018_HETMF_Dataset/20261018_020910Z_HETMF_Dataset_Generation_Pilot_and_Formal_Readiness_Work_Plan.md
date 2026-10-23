# HET-MF: heterogeneous electro-thermal contact arrays
## Physical specification, GPU2 pilot, dataset design, storage, and formal-generation readiness

**Status:** proposed implementation and bounded dataset pilot. No HET-MF simulation, throughput measurement, or ML training has been performed by the author of this plan. Numerical bands, parameter ranges, sample counts, and resource limits below are prospective choices to test, not measured facts.

**Requested outcome:** implement an independent case-specific generator; generate and visualize a small, diverse pilot; verify numerical quality; measure generation and I/O costs; propose compact/standard/expanded formal datasets; provide tested scripts for formal generation, but **stop for the user's approval before generating a formal release**.

**Repository context reviewed:** `cosmos2w/ModularDT`, `agent/honf-core-next`, remote tip `53cba42bc73661306e469c7905be9422b235d672`. This tip documents an amended Round9 horizon and device arrangement. Recheck the local and remote states before coding. This dataset task does not change, resume, stop, or reinterpret Round9.

**Case/package names:** `HONF_Proj/Case_ElectroThermal/`, Python package `hetmf`; long name **Heterogeneous Electro-Thermal Modular Field (HET-MF)**. Use a new dataset family `hetmf_contact_sheet_v1` and an independent HET-MF solver ledger. Do not reuse ThermalChannel's physical capability, target units, normalization, or historical solver accounting.

---

## 0. Executive decisions

1. **Use a mixture of controlled morphology and bounded Gaussian-random-field variation.** Neither a single GRF family nor a few hand-designed motifs are sufficient. The proposed core environment mixture is 40% correlated random mosaics, 20% layered/barrier media, 20% inclusion media, and 20% conducting-corridor media. Sample module layouts independently of those environments. Hand-designed alignment between a corridor and contacts is allowed only in clearly identified diagnostic examples, not throughout the training distribution.
2. **Make the environmental input complete and numerically resolved.** A 12 × 12 material-control catalogue defines electrical sheet conductance, thermal sheet conductance, and cooling. The primary physical field is a fixed log-bilinear reconstruction of those 144 records, representing a graded composite. There is no unobserved fine random field. Sharp piecewise-constant material jumps are a separately labelled numerical diagnostic, not an unannounced substitute for the core field.
3. **Keep the physics simple but genuinely coupled.** Solve steady electrical potential, compute conservative sheet/contact Joule losses, and solve steady temperature rise. Material coefficients are temperature-independent. Potential/current are linear in terminal voltages at fixed geometry, whereas heat generation and temperature are quadratic. Do not import the old heat-affine temperature head into this dataset.
4. **Do not prescribe point sources or almost-perfect Dirichlet pads.** Use smooth finite-area contact footprints and finite total contact conductances. This makes layout variation inexpensive to rasterize and easier to verify on modest grids.
5. **Select resolution from evidence.** Start at 96 × 96, compare the same physical cases at 48² and 192², and use a small 384² reference subset if necessary. Adopt 48² only if it passes the declared field, current, power, and hotspot checks. Do not lower resolution merely to improve throughput.
6. **Count independent environments and layouts, not just rows.** Proposed formal profiles have 2,304 / 4,608 / 9,216 physical states, including explicit generalization panels and counterfactual quartets. The standard profile has 2,880 ordinary TRAIN states built from 240 independent environments and 480 independent contact layouts; it is not 2,880 independent environments.
7. **GPU2 only, with no disruption to other work.** Verify its UUID and processes before allocation. If another task owns GPU2, produce CPU-side implementation/configuration work and report the conflict; do not kill, move, or share with that job without explicit approval.
8. **Everything generated lives on `/data2`.** Keep source/configuration/schemas in Git, and expose raw/processed/pilot/figure directories through ignored browsing symlinks below `Case_ElectroThermal/Dataset/`.
9. **A complete pilot, not a nominal smoke, is the deliverable.** Produce twelve representative scenes, six controlled counterfactual quartets, numerical/refinement evidence, and a 256-state stratified throughput/coverage panel, with overlapping cases deduplicated. This is normally at most 292 distinct physical states. No neural-model fitting is required in this round.

---

## 1. Scientific purpose and boundaries

### 1.1 Position in the planned paper

ThermalChannel provides relatively simple prescribed environmental geometry with fluid, interface, and intra-module thermal outputs. HET-MF instead provides an independently varying, fully given material environment, configurable electrical contacts, and coupled fields on the surrounding sheet. WindFarm and the separately developed deep-geothermal dataset provide realistic 3D evidence. HET-MF should be presented as a controlled engineering-inspired benchmark, not a validated battery or current-collector design simulator.

A representative question is:

> For this set of electrical contacts, material map, and receiver, which contacts and environmental regions must be considered together to explain current pathways and temperature?

Useful behaviors include current rerouting around resistive barriers, remote coupling through conductive corridors, contact-current redistribution after moving another contact, and different electrical versus thermal pathways when electrical and thermal conductivities are not perfectly correlated.

### 1.2 Explicit simplifications

The reference is a square, thermally thin sheet with scalar, positive, spatially varying coefficients. Contacts inject/extract current through the sheet thickness into external ideal-voltage terminals via finite interface conductances. Their internal electrical and thermal fields are not resolved. Lateral boundaries are electrically and thermally insulating; distributed cooling removes heat out of plane.

No electrochemistry, state of charge, phase change, thermoelectric effects, radiation, mechanics, fluid flow, contact deformation, or temperature feedback into conductivity is included. A temperature below a pad is a **sheet temperature**, not an intra-module temperature prediction. Finite contact loss is deposited entirely into the sheet in version 1; this is a declared heat-partition assumption, not a universal interface law.

The varied coefficient fields describe a generic conductive composite/effective sheet. They are not claimed to be a calibrated set of metallic materials satisfying a specific electrical–thermal constitutive relation.

### 1.3 Interpretation limits

Elliptic conduction generally produces distributed influences. A true physical interaction matrix may be dense. The generator must not invent sparse causal edges or use HONF's desired graph to manufacture targets. Save physical interventions, terminal response matrices, and influence rankings that a future model can be evaluated against; do not label a thresholded streamline picture as ground-truth hypergraph membership.

Nonzero voltage–voltage interactions in temperature arise from quadratic Joule heating. At fixed geometry/environment, third voltage derivatives vanish in this one-way model. Do not claim arbitrary irreducible higher-order voltage nonlinearity. Layout and environmental changes alter the operators and can produce more general non-additive responses.

---

## 2. Physical and mathematical formulation

### 2.1 Dimensional interpretation

Let the physical sheet occupy `[0,L]²`, with fixed thickness `t`. Define sheet electrical conductance `Sigma = t sigma_bulk` [S], sheet thermal conductance `K = t k_bulk` [W/K], and distributed cooling `H` [W/(m² K)]. Contact `i` has total conductance `G_i` [S], terminal voltage `V_i` [V], and footprint density `w_i` [m^-2] satisfying `integral w_i dx = 1`.

The dimensional electrical equation is

\[
-\nabla\cdot(\Sigma\nabla\phi)
+\sum_i G_i w_i(\phi-V_i)=0,
\qquad \mathbf n\cdot\Sigma\nabla\phi=0\ \text{on }\partial\Omega.
\]

The in-plane sheet current density is `j = -Sigma grad(phi)` [A/m], not the 3D bulk current density [A/m²]. Total terminal current supplied to the sheet is

\[
I_i=\int_\Omega G_iw_i(V_i-\phi)\,dx.
\]

Positive `I_i` means delivery from terminal to sheet. An intermediate-voltage contact's current sign is solved, not assigned from an arbitrary source/sink label.

The areal heat generation [W/m²] is

\[
q_{\rm sheet}=\Sigma|\nabla\phi|^2,
\qquad
q_{\rm contact}=\sum_i G_iw_i(V_i-\phi)^2,
\qquad q=q_{\rm sheet}+q_{\rm contact}.
\]

Temperature rise `Theta = T - T_ambient` obeys

\[
-\nabla\cdot(K\nabla\Theta)+H\Theta=q,
\qquad \mathbf n\cdot K\nabla\Theta=0\ \text{on }\partial\Omega.
\]

All contact Joule loss is included once. There is no separate ideal-voltage boundary on the pad in addition to the finite-contact term.

### 2.2 Nondimensional dataset law

Use fixed reference values `L`, `Sigma_0`, `K_0`, and `V_0` only to define units. Set

\[
x'=x/L,\quad v=\phi/V_0,\quad u_i=V_i/V_0,\quad
s=\Sigma/\Sigma_0,\quad \kappa=K/K_0,\quad
\gamma_i=G_i/\Sigma_0,
\]
\[
\theta=\Theta/(\Sigma_0V_0^2/K_0),\quad
\beta=HL^2/K_0,\quad \chi_i=L^2w_i.
\]

Drop the prime on coordinates. The solver and canonical targets use dimensionless units on `[0,1]²`:

\[
 a(x)=\sum_i\gamma_i\chi_i(x),\qquad
 b(x)=\sum_i\gamma_i\chi_i(x)u_i,
\]
\[
\boxed{-\nabla\cdot(s\nabla v)+a v=b,}\tag{1}
\]
\[
\boxed{\mathbf j=-s\nabla v,\quad
q=s|\nabla v|^2+\sum_i\gamma_i\chi_i(v-u_i)^2,}\tag{2}
\]
\[
\boxed{-\nabla\cdot(\kappa\nabla\theta)+\beta\theta=q.}\tag{3}
\]

Both lateral normal fluxes are zero. Main cases require `s > 0`, `kappa > 0`, `beta > 0`, at least two finite-conductance contacts, and a nonzero terminal-voltage range. These conditions make both discrete systems positive definite. Do not ground a grid cell as an extra boundary condition: the external terminals already anchor the potential through `a`.

Store the nondimensionalization explicitly. Default numerical reference units can all equal one, but outputs must then be labelled **dimensionless**, not kelvin, volts, watts, or A/m. An optional user-selected SI interpretation is a separate postprocessing transform, not per-case normalization.

### 2.3 Smooth, finite-area contact footprint

For a pad of radius `r_i` centered at `x_i`, use

\[
\chi_i(x)=\frac{3}{\pi r_i^2}
\left[1-\frac{\|x-x_i\|^2}{r_i^2}\right]_+^2.
\]

Its continuous integral is one when the disk lies inside the sheet. Compute cell-averaged weights using 4 × 4 Gauss–Legendre quadrature per affected cell, with a bounding-box stencil around each pad. Compare 2 × 2, 4 × 4, and 8 × 8 quadrature in the pilot. Normalize the discrete weights to exactly `sum_p chi_i,p A_p = 1`, and record the pre-normalization integral error. Refinement must keep the same physical center/radius; only quadrature and sampling resolution change.

`gamma_i` is **total** contact conductance, independent of radius. Changing radius at fixed `gamma_i` changes spreading geometry rather than silently changing total conductance. Fixed areal contact admittance is a possible future variant, not the version-1 convention.

### 2.4 Exact reference identities

At fixed material fields and pad geometry, potential and current depend linearly on terminal voltage. Joule power and temperature depend quadratically. Test:

- Common shift: `u -> u+c`, `v -> v+c`, while `j`, `q`, `theta`, and terminal currents do not change.
- Scaling: `u -> alpha*u` gives `v,j,I -> alpha*(v,j,I)` and `q,theta -> alpha²*(q,theta)`.
- Reversal: all signs reversed leaves heat/temperature unchanged.
- Equal voltages: `v` is that constant, `I=j=q=theta=0`.
- Changing only `kappa` or `beta` cannot change the electrical solution.
- Permuting complete contact records cannot change the mathematical solution.

For an input-only unit-voltage basis on four representative scenes, compute a terminal conductance matrix `C` with `I=C u`. Check symmetry, positive semidefiniteness, and `C 1=0` within numerical tolerance. Its entries are responses of the physical reference, not learned attention. Save this small matrix and the basis-solve accounting; do not store large per-contact field bases for every formal case.

---

## 3. Conservative finite-volume reference

### 3.1 Grid and coefficient interfaces

Use a cell-centered Cartesian grid, native array order `[y,x]`, and cell ID `iy * nx + ix`. Cell centers are `((ix+0.5)/nx,(iy+0.5)/ny)`. Evaluate the same 12 × 12 log-bilinear material-control field (§4.1) on candidate grids 48², 96², and 192². The physical environment is not regenerated or blurred differently at each grid. Sample its positive coefficients at cell centers for the declared finite-volume discretization; refinement quantifies that approximation. For separately labelled piecewise-constant diagnostics, these grids also align with material-cell boundaries.

Use a two-point flux approximation for **scalar** coefficients on this orthogonal grid. Rotated patterns mean spatial morphology, not tensor conductivity. Do not later introduce rotated anisotropic conductivity tensors while retaining a scalar harmonic-face rule.

For an internal face between cells `p` and `r`, face length `l_f`, and center-to-face distances `d_p,d_r`, define

\[
R_{p,f}=d_p/(s_p l_f),\quad R_{r,f}=d_r/(s_r l_f),\quad
T^s_f=(R_{p,f}+R_{r,f})^{-1}.
\]

Thus a uniform mesh uses harmonic face conductance, including at material jumps. Thermal transmissibility `T^k_f` uses `kappa` in the same formula. Exterior normal flux is exactly zero; never allow tensor-roll periodic wraparound.

The integrated electrical operator is

\[
(A_v v)_p=\sum_{r\sim p}T^s_{pr}(v_p-v_r)+A_p a_pv_p,
\qquad (b_v)_p=A_pb_p.
\]

The integrated thermal operator is

\[
(A_T\theta)_p=\sum_{r\sim p}T^k_{pr}(\theta_p-\theta_r)
+A_p\beta_p\theta_p,
\qquad (b_T)_p=A_pq_p.
\]

Assembly, matrix-free application, and all residual calculations must use this same area convention. Document that the thermal equation is solved after, not simultaneously with, the electrical equation.

### 3.2 Joule heat must come from the discrete electrical energy

Do **not** define reference heating by squaring an averaged cell-centered current or a generic image gradient. That can violate energy conservation, especially at coefficient jumps.

Each unique internal electrical face dissipates

\[
P_f=T^s_f(v_p-v_r)^2.
\]

Partition that loss between its two half-cells in proportion to their resistances:

\[
P_{f\to p}=\frac{R_{p,f}}{R_{p,f}+R_{r,f}}P_f,
\qquad
P_{f\to r}=\frac{R_{r,f}}{R_{p,f}+R_{r,f}}P_f.
\]

Equal splitting is correct for equal conductivities, **not** generally at a material jump. Set `q_sheet,p = sum_f P_(f->p)/A_p` and

\[
q_{{\rm contact},p}=\sum_i\gamma_i\chi_{i,p}(v_p-u_i)^2.
\]

This yields nonnegative cell heat, honors local resistive heating, and conserves total network energy. Keep `q_sheet` and `q_contact` separately so the pilot can detect an uninteresting contact-loss-dominated regime.

### 3.3 Global and local balances

Compute terminal currents with the same quadrature:

\[
I_i=\gamma_i\sum_p A_p\chi_{i,p}(u_i-v_p).
\]

The expected discrete equalities are

\[
\sum_iI_i=0,\qquad
\sum_i u_iI_i=\sum_f P_f+\sum_p A_pq_{{\rm contact},p},
\]
\[
\sum_p A_p\beta_p\theta_p=\sum_p A_pq_p.
\]

Report normalized residuals, absolute residuals, and denominator definitions. In a near-zero-power test, use an explicit absolute tolerance and label relative errors unresolved; do not divide by a small arbitrary number and report a dramatic relative failure.

Also retain cellwise electrical and thermal residual RMS/max. Global cancellation alone cannot establish accurate local fields. Recheck balances after the chosen raw-data dtype is written and reloaded.

### 3.4 Native face and cell outputs

Store oriented electrical face current **densities** along +x and +y with shapes `[ny,nx+1]` and `[ny+1,nx]`; these equal integrated face current divided by face length. Exterior normal components are zero. Store thermal face fluxes on the same staggered geometry.

For convenience, define cell-centered `j_x,j_y` as arithmetic averages of the two parallel face densities surrounding each cell. These are declared reconstruction targets. Their use for plots or ML does not replace native face fluxes in conservation tests or the energy-based Joule source.

---

## 4. Heterogeneous environment: a hybrid distribution, not a full Cartesian sweep

### 4.1 What is physically given

The canonical environment is an array of 144 records on centers `((ix+0.5)/12,(iy+0.5)/12)`:

`(record_id, center_x, center_y, spacing_x, spacing_y, sampling_area, s, kappa, beta)`.

The record spacing is 1/12 and sampling area is 1/144. This area is an input quadrature convention, not the assertion that the coefficient is constant over that entire square.

The three property arrays are actual inputs. Family labels, random seeds, latent GRF coefficients, and pattern-construction parameters are generation provenance, **not model features**. Module locations, terminal voltages, and solved fields do not enter the core environmental generator.

**Design refinement from the exploratory sketch:** use a graded field as the primary release instead of sharp discontinuous tiles. Abrupt multi-material corners can make local current/Joule peaks especially resolution-sensitive; the primary problem should not spend its numerical budget on unresolved corners. Sharp-interface tests remain valuable, but they are a distinct renderer/schema rather than a post-hoc fix to a failing sample.

Let `B_a(x)` be ordinary tensor-product piecewise-linear interpolation weights on the 12 × 12 center lattice, with coordinates clamped to the outermost centers in the half-cell boundary strips. Then

\[
\log s(x)=\sum_a B_a(x)\log s_a,\quad
\log\kappa(x)=\sum_a B_a(x)\log\kappa_a,\quad
\log\beta(x)=\sum_a B_a(x)\log\beta_a.
\]

The weights are nonnegative and sum to one, so the renderer is positive and does not overshoot the supplied coefficient bounds. Interpolate **logs**, then exponentiate; do not silently switch to interpolation of linear coefficients. Freeze this renderer as `log_bilinear_centers_border_v1` and test it at arbitrary points and across all grid sizes.

A fine-grid baseline receives a raster of this exact reconstruction, while HONF receives the original 144 records and the same known rendering convention. Neither model gets hidden finer environmental information. Input pattern descriptions use the word "tile" below for the control-lattice scale, not a claim of piecewise-constant physical coefficients.

A `piecewise_constant_tiles_v1` renderer may be used in data-free coefficient-jump tests and a bounded diagnostic copy of an existing representative scene. Record it as a different physical variant, exclude it from the primary formal core unless the user explicitly selects that variant, and never compare grids with different renderers as a refinement study.

### 4.2 Primary morphology mixture

| Family | Core share | Generator | Parameters / constraints |
|---|---:|---|---|
| `correlated_mosaic` | 40% | Gaussian field sampled at 12 × 12 tile centers, transformed to positive bounded coefficients | Correlation lengths 0.12–0.35; correlation-length ratio 1–3; orientation uniform in [0,pi); realized lengths and contrasts reported |
| `layered_barrier` | 20% | 2–4 oblique bands or a low-conductance barrier with a resolved opening, plus weak texture | Effective band/opening width at least 2 tiles; angles not restricted to horizontal/vertical; opening position independent of modules |
| `inclusions` | 20% | 2–5 higher/lower-conductance elliptical inclusions rasterized to the canonical tiles | Semi-axes 0.10–0.24; random signs/orientations/centers; no sub-tile island silently added |
| `corridors` | 20% | 1–2 bent, connected high-conductance paths in a lower-conductance matrix, plus weak texture | 2–4 control segments; width 2–3 tiles; path endpoints independent of contact endpoints; record connectivity after rasterization |

These shares are initial design choices. The pilot may recommend an explicit revision before formal generation, with a reason and a new plan hash. Do not silently rebalance after inspecting ML results.

Use modest textural variation on the structured families, for example 10–20% of the main bounded latent amplitude. The texture must not erase the intended barriers/corridors. A uniform medium is a diagnostic/null family, not 20% of the ordinary training set.

### 4.3 GRF construction without unresolved randomness

At the 144 tile centers, an inexpensive and transparent implementation is direct sampling from a Gaussian covariance matrix:

\[
K_{ab}=\exp[-\tfrac12(x_a-x_b)^TR(\varphi)
\operatorname{diag}(\ell_1^{-2},\ell_2^{-2})R(\varphi)^T(x_a-x_b)].
\]

Use FP64 Cholesky with a documented diagonal jitter initially `1e-10`, then a bounded increase to `1e-8` only on factorization failure. Sample independent normal vectors with stable per-environment RNG keys. This is a 144-dimensional input-generation operation; CPU generation is acceptable and need not occupy GPU2. Reuse a covariance factor for the three latent property draws within an environment.

Do not standardize every realization to its own min/max or force its own sample variance to one. That changes the stochastic law. Record empirical mean, variance, correlations, and realized contrasts. If FFT generation is substituted, specify padding/cropping and covariance normalization; do not accidentally impose a periodic environment or call arbitrary spectral filtering an exact Matérn field.

### 4.4 Positive bounded property transforms

For a latent field `g_s`, set

\[
\log s=\tfrac12\log C_s\,\tanh(g_s),\qquad s_0=1.
\]

Thus the possible ratio is bounded by `C_s`, but an individual realization may have a smaller actual max/min ratio. Structured families construct a bounded latent morphology with weak GRF texture before applying the declared transform. Save both the requested bound and realized contrast.

Use initial parameter bounds:

| Parameter | Core proposal | Sampling |
|---|---|---|
| Electrical contrast bound `C_s` | 3–30 | log-uniform / space-filling within family |
| Thermal contrast bound `C_k` | 1.5–6 | log-uniform |
| Electrical–thermal latent sharing `rho` | 0 or 0.7, half of core environments each | balanced discrete factor |
| Cooling level `beta_0` | 2–12 | log-uniform |
| Cooling contrast bound `C_beta` | 1–3 | log-uniform; a resolved constant-cooling subset is allowed |
| Texture amplitude, structured families | 0.10–0.20 of morphology scale | bounded uniform |

For thermal properties use a fresh random field with the same covariance family and

\[
g_k=\rho g_s+\sqrt{1-\rho^2}\,\widetilde g_k,
\qquad
\log\kappa=\tfrac12\log C_k\,\tanh(g_k).
\]

`rho` specifies latent mixing, not a guaranteed empirical correlation after morphology and nonlinear transformation. Cooling has its own independent latent field:

\[
\beta=\beta_0\exp[\tfrac12\log C_\beta\tanh(g_\beta)].
\]

For structured `g_s`, the mixture is a designed correlated property family, not a claim that all channels are joint Gaussian. Check and report actual spatial correlations.

### 4.5 Reserved challenges and scope of generalization

Keep the main training distribution bounded. Reserved physical challenges can include electrical contrast 30–100, negative electrical–thermal latent sharing, combined barrier-plus-corridor patterns, and larger contact counts. Do not simultaneously vary every difficulty parameter in the sole OOD panel: include labelled single-axis subsets.

Generalization means performance on declared withheld environments/layouts and combinations. It cannot mean arbitrary spatial material fields from a few hundred environments. A future sample-efficiency study should enlarge independent environment families before multiplying voltage-scaled copies of the same solution.

---

## 5. Contact layouts and operating conditions

### 5.1 Main parameter space

| Quantity | Core proposal | Reason / safeguard |
|---|---|---|
| Contact count `M` | {2,3,4,6,8,12}, balanced | Distinguishes several interaction densities; no global fixed twelve-contact requirement |
| Radius `r_i` | 0.06–0.09 | Finite, resolved contacts; audit the lower end at every candidate grid |
| Total contact conductance `gamma_i` | 1–10 | Finite contact resistance, not a huge penalty approximation to Dirichlet pads |
| Edge clearance | `r_i + 0.025` from each side | Footprints remain inside the sheet |
| Pairwise clearance | distance >= `r_i+r_j+0.015` | No overlapping contacts at different terminal voltages |
| Terminal voltage range | exactly 1 in ordinary core cases | Avoids spending cases on pure scaling symmetries |
| Voltage common mode | center `(min(u)+max(u))/2` to zero | Removes a redundant gauge-like input offset; raw diagnostic shifts are still tested |
| Electrical/thermal coefficient temperature dependence | none | Two steady linear solves remain sufficient |

Use three geometry families, approximately equally represented **within every M stratum**: dispersed random layouts, perturbed rows/arrays with continuously varying orientation, and legal two-cluster layouts. Do not tie one layout family or polarity assignment to one environmental morphology.

Generate with deterministic rejection or a constructive sampler. Log failed geometry proposals and retries. If dense clusters at M12 make the requested radius distribution infeasible, recommend a pre-formal change explicitly; never shrink pads silently as M increases. Main layouts must remain independent of environment.

### 5.2 Voltage patterns

Use balanced two-level patterns and multilevel patterns for M>=3. For two-level cases, require at least one terminal at each extreme; randomly assign the polarity to physical contacts. For multilevel cases, choose two geometry-independent contacts at -0.5/+0.5 and sample others in [-0.4,0.4]. Contact ordering is correspondence only.

For M2, changing the amplitude or reversing both voltages yields only a scaling/sign transformation. Do not count those as new independent ordinary cases. For all M, detect duplicate voltage vectors up to common shift, global scaling, and sign when they would be stored as nominally independent excitations of the exact same layout/environment.

The canonical layout catalogue includes geometry, radii, conductances, and the assigned voltage pattern. Later counterfactual voltages retain the original scale rather than renormalizing each endpoint; otherwise the intended perturbation would also alter every other voltage.

### 5.3 Parameter sampling

Use stratification for discrete factors (environment family, M, layout family, voltage family, correlation mode). Within each stratum, use scrambled Latin-hypercube sampling for continuous hyperparameters plus independent keyed random draws for high-dimensional field and layout realizations. LHS is a space-filling design aid, not a guarantee of neural-network generalization [R4].

Prepare the largest *input-only* candidate index deterministically, then choose balanced nested profile subsets without changing completed case identities. Avoid a full Cartesian product of all scalar values. If Sobol is used instead, respect power-of-two sampling and disclose loss of balance after rejection/subsampling [R5].

## 6. Proposed dataset sizes and leakage-aware organization

### 6.1 Three prospective profiles

These are **input-manifest proposals**, not authorization to solve every row.

| Profile | Ordinary TRAIN | DEV | Joint-unseen TEST | Axis-specific/generalization panel | Counterfactual states | Total physical states |
|---|---:|---:|---:|---:|---:|---:|
| Compact | 1,440 | 240 | 240 | 256 | 128 = 32 quartets | **2,304** |
| Standard | 2,880 | 480 | 480 | 512 | 256 = 64 quartets | **4,608** |
| Expanded | 5,760 | 960 | 960 | 1,024 | 512 = 128 quartets | **9,216** |

For ordinary core data, use **twelve states per environment**: two independently generated layouts from each M stratum `{2,3,4,6,8,12}`, with voltage patterns already included in each layout's input record. Do not obtain the twelve states by rescaling one voltage vector.

| Profile | TRAIN environment roots | DEV environment roots | TEST environment roots | TRAIN layout roots | DEV/TEST layout roots, each |
|---|---:|---:|---:|---:|---:|
| Compact | 120 | 20 | 20 | 240 = 40 per M | 48 = 8 per M |
| Standard | 240 | 40 | 40 | 480 = 80 per M | 96 = 16 per M |
| Expanded | 480 | 80 | 80 | 960 = 160 per M | 192 = 32 per M |

In the standard TRAIN design, each layout appears with six independently sampled environments: `480 * 6 = 2,880`. Each environment appears with twelve layouts: `240 * 12 = 2,880`. Use balanced randomized bipartite pairing with no duplicate environment/layout pair. ENV and layout family assignments must remain independent apart from declared stratum balance.

This reuse serves an experimental purpose: the model sees multiple environments for the same layout and multiple layouts for the same environment. It does not make every row statistically independent. Report independent root counts in every summary.

### 6.2 Core split rules

Split environment roots and layout roots before pairing. Core TRAIN, DEV, and joint-unseen TEST have disjoint roots on **both** axes. Every perturbation, rotated copy, resolution repeat, and alternate voltage state of a root keeps its root lineage. Never split near-duplicate rows randomly.

All pilot inputs live in a separate `pilot` namespace and remain development-exposed. Do not silently promote the twelve curated examples or throughput scenes into a supposedly untouched formal TEST.

Freeze the final manifest before formal solves. Persist `split`, `environment_root_id`, `layout_root_id`, `voltage_pattern_id`, `interaction_family_id`, `generation_seed`, exact physical input hash, and solver configuration hash. IDs are provenance only and never learned features.

### 6.3 Axis-specific panels

Divide the standard 512-state supplementary panel into four 128-state panels (halve/double for the other profiles):

1. **New environment / known layout:** new environment roots paired with some TRAIN layout roots. Tests environmental transfer while intentionally retaining known modules.
2. **Known environment / new layout:** TRAIN environment roots paired with new layout roots. Tests modular placement/count variation within the declared core support.
3. **New pair / known components:** withheld environment–layout pairs, both components present elsewhere in TRAIN. This is compositional recombination, not unseen-environment generalization.
4. **Extrapolation:** four explicitly labelled 32-state subsets: M10 count interpolation, M16 count extrapolation, electrical contrast 30–100 at familiar M, and withheld composite morphology at familiar contrast/M. The numeric quality of each subset must be qualified separately.

These panels intentionally differ in component exposure. They are not interchangeable with the joint-unseen TEST. Record `environment_seen_in_train` and `layout_seen_in_train`; those flags must match the actual catalogue joins. Reusing a TRAIN component here is deliberate, but using these panel targets for fitting or normalization is prohibited.

No unavailable larger-count or high-contrast case may be replaced by an easier case without updating the manifest/status. If a challenge needs a finer grid, retain a separate declared resolution group and report its cost; do not mix resolutions invisibly into a fixed-grid target array.

### 6.4 Counterfactual families

The standard proposal contains 64 independently owned quartets: 32 module–environment (ME), 16 module–module (MM), and 16 environment–environment (EE). Compact/expanded scale proportionally. Keep all four states in the same audit split and retain shared root IDs. These roots are separate from the ordinary core TEST.

For independent interventions `A` and `B`, solve

\[
F_{00}, F_{10}, F_{01}, F_{11},\qquad
\mathcal I_{AB}=F_{11}-F_{10}-F_{01}+F_{00}.
\]

- **ME:** move one legal pad by a fixed small displacement, or change its finite conductance; independently multiply the electrical conductance values of a fixed 2 × 2 or 3 × 3 material-control patch by a bounded factor. Save its actual interpolation-support footprint as well as its record mask.
- **MM:** change two independently selected terminal voltages by small increments, or make two legal geometry changes. Voltage cases should include M>=3 and retain all unaffected values exactly. Do not renormalize voltage range separately at each endpoint.
- **EE:** modify two coefficient-control patches with disjoint interpolation supports, leaving at least the required control-lattice gap. Include same-physics electrical changes and a cross-physics example (one electrical, one thermal/cooling patch). A thermal-only perturbation's electrical null is a useful control, not a reason to discard the example.

Freeze contact/patch selection from input geometry and RNG before solving. Do not search physical targets for the most attractive quartet. If an interaction is below the measured numerical floor, report it as unresolved/consistent with zero. Estimate that floor by tighter solves and refinement of the **quartet**, not just one endpoint.

Do not recenter or renormalize an entire coefficient map after a local patch intervention: that would change the supposedly untouched environment. Check final property bounds and save the exact intervention masks and transforms.

### 6.5 How to recommend a final size

The pilot report should recommend the smallest profile justified by measured physical diversity, implementation cost, and future model-input size. Dataset sufficiency cannot be certified without ML learning curves. Propose later curves at nested fractions of independent TRAIN environment/layout roots, not random rows that preserve nearly every environment at each size.

If coverage is inadequate, first enlarge independent environments or the relevant layout stratum. Do not create thousands of global-voltage-scaled copies to claim increased generalization. Conversely, do not jump from 4,608 to 50,000 cases because the input representation has hundreds of coordinates.

---

## 7. The pilot that Codex must actually run

### 7.1 Stage P0: preflight and physical contract

Before allocating GPU2 or writing data:

- Inspect current source/worktree status, active device UUIDs/processes, `/data2` mount identity/free space, and the Python environment. Record versions, not assumptions inherited from old reports.
- Establish an isolated worktree/branch or a demonstrably disjoint development arrangement; see §12. Do not change the live Round9 worktree's branch, common files, environment packages, or GPU jobs.
- Validate a complete pilot config; missing solver tolerances, output paths, and resource limits are errors.
- Create an input-only `pilot_plan.json`, code/config fingerprints, and an owned resource reservation below `/data2`.
- Run data-free stencil/energy/shape tests on CPU before a GPU solver trial.

### 7.2 Stage P1: twelve representative scenes

Create the following first. Their exact inputs must be saved before solving. A paired scene may intentionally reuse another scene's geometry, but the report must identify that fact.

| ID | Environment and modules | Main diagnostic |
|---|---|---|
| R00 | Uniform coefficients, symmetric M2 contacts | Potential/current symmetry, terminal balance, zero transverse symmetry response |
| R01 | Uniform coefficients, irregular M6 multilevel contacts | Multiple-terminal response without heterogeneous E |
| R02 | Isotropic correlated mosaic, M4 | Independent spatial material conditioning |
| R03 | Oriented/anisotropic-correlation mosaic, M8 | Pathways not aligned with geometric nearest neighbors |
| R04 | Oblique layered medium, M6 | Along-layer versus cross-layer conduction |
| R05 | Low-conductance barrier with a resolved gap, M4 | Bottleneck and rerouting, not zero-conductance disconnection |
| R06 | Bent conductive corridor, M6 | A physically connected remote pathway |
| R07 | Multiple inclusions, M8 | Interacting material features, detours, distributed current |
| R08 | Same electrical setup as one earlier case, changed thermal map | Electrical null with changed thermal spreading |
| R09 | Same electrical setup and kappa as that case, patchy cooling | Electrical null with changed hotspot pattern |
| R10 | Legal clustered M12, upper-core contrast | Dense-layout and conditioning stress within the core proposal |
| R11 | M12 with contrast bound 100 / composite pattern | Out-of-core numerical stress; not automatic formal-core admission |

Solve at 96² first and immediately render the representative gallery. Then solve the same physical inputs at 48² and 192². Use 384² on at most four declared difficult/representative scenes when needed to qualify the 192² reference. Do not change pad radius or the 12 × 12 material map between grids.

The report must show R08/R09 side by side with their exact electrical parent, not merely assert that only one field changed.

### 7.3 Stage P2: six counterfactual quartets

Produce six fixed quartets, two each of ME/MM/EE, normally 24 physical states before deduplication. Include both a structured environment and a random mosaic. At least one quartet changes only thermal/cooling properties on one axis, demonstrating the appropriate electrical null. All four states use identical grid/precision/solver settings and fixed visualization scales.

Use these six intervention categories: one ME local-coefficient patch plus pad displacement; one ME electrical-map rotation plus contact-conductance change; one MM two-voltage perturbation; one MM two-pad displacement; one EE two-electrical-patch change; and one EE electrical-patch plus cooling-patch change. Freeze legal examples from inputs, without searching targets.

For the map-rotation ME quartet, rotate the canonical electrical-control array by 90 degrees at fixed contacts and unchanged thermal/cooling arrays. It preserves electrical coefficient histograms while changing spatial organization; all descendants share an environment root. This prevents the entire environmental-interest audit from being explained only by changes in mean conductivity. The local-patch rules in §6 remain the default formal ME family; this whole-map intervention is an explicitly identified pilot diagnostic.

Measure field differences, interaction residuals, terminal-current changes, total-power changes, and hotspot changes. Repeat two quartets at finer resolution/tighter tolerance to quantify numerical floors. Save all endpoints and intervention definitions, including small or null interactions.

For four representative base scenes, additionally solve the full terminal unit-voltage basis to form `C`. These electrical-only solves count in the ledger but do not inflate the number of independent physical dataset scenes.

### 7.4 Stage P3: 256-state stratified coverage and throughput panel

After the mathematical reference passes, generate 256 distinct, input-frozen pilot states. Use 64 environments and four independently paired layouts per environment, with M counts balanced as closely as divisibility permits across all four environment families. Use a rotating balanced assignment so that all six M strata, three layout families, voltage modes, and correlation modes occur in every sizeable morphology group.

Approximate environment-family counts may be 26/13/13/12 for 40/20/20/20; record the integer rounding. This panel is not the final dataset split design. Its purpose is to measure:

- solver success/failure and true residuals;
- iteration and complete-coupled-solve time distributions;
- actual property contrasts and spatial correlations;
- contact packing/rejection rates, clearances, current/power ranges;
- contact-versus-sheet Joule-power fractions;
- field diversity at fixed modules and changed environments;
- actual file sizes, compression ratios, and loader throughput.

Test scalar CPU FP64 and batched GPU execution on a fixed timing subset first. Qualify the fastest correct precision/backend, then time the full panel at batch sizes 1, 4, 16, and 32 where feasible. Use at least three timing repeats, disable visualization during timed generation, and do not persist duplicate outputs from timing-only repetitions. Record all repeated solver attempts.

An optional B64 test is allowed only if the measured B32 throughput is still improving and the reservation has ample headroom. Do not spend the pilot implementing a sophisticated solver solely to maximize GPU utilization.

### 7.5 Bounds for this pilot

Proposed ceiling: **320 distinct physical pilot states, 4,000 complete coupled evaluations including timing/refinement repeats, and 10,000 individual linear-system attempts**. Input proposals rejected before solving are counted separately. Basis solves, failed solves, CPU references, refinement runs, and timing repeats all debit the appropriate system-attempt counter.

Proposed resource envelope: **6 elapsed hours for implementation/QA/pilot, at most 2 GPU-associated hours, 15 GiB retained generated files, 5 GiB temporary files, and 20 GiB peak new generated data**. Preserve at least 50 GiB disk headroom. These are conservative task bounds, not predicted usage. Reserve time for reporting rather than exhaust the budget on compiler work.

If a correctness issue blocks the pilot, stop at the smallest reproducible failure and provide the implementation/diagnostic report. If GPU2 is busy, finish CPU-side preparation and clearly state that no GPU throughput is established. Do not substitute a three-case smoke and call the full pilot complete.

---

## 8. Solver implementation and GPU acceleration

### 8.1 Required baseline

Implement matrix-free **batched preconditioned conjugate gradient** for both SPD systems. Use a Jacobi diagonal preconditioner initially. The matrix-vector product should be a small number of tensor slice/neighbor operations with explicit exterior-face handling. No dense `[N²,N²]` GPU matrix is allowed.

Coefficient/face arrays are prepared once per environment/layout state. The thermal matrix depends on the environment, not voltages or pad geometry, in this version; its prepared coefficients may be reused across matching environment states. Electrical matrix coefficients depend on environment and pad geometry/conductance, not the voltage pattern. Reuse those only under exact input hashes. Charge preparation separately in warm/cold benchmarks.

Input generation can remain on CPU using a few threads. Batch transfer of complete coefficients and compact contact records is preferable to tiny repeated transfers. All solving and postprocessing should run with gradients disabled; no full autograd trajectory is needed for a reference dataset.

### 8.2 Batched stopping and numerical safety

Each case has its own active mask, residual, denominator, and iteration count. Converged or zero-RHS cases must not produce `0/0` in an inactive CG update. Apply safe denominator substitutions before arithmetic, not only after NaNs have been generated. Detect nonpositive/nonfinite `p^T A p` and report the individual failing state.

Use residual checks every 10–20 iterations, not a Python `.item()` every iteration. Recompute the true residual before acceptance. A recurrence residual is not sufficient. Group only by shape and, optionally, input-known contrast/contact bounds; case ordering or batch composition must not change the mathematical problem.

Start with a maximum of 4,000 iterations per linear solve, serialized in the config. Record nonconvergence rather than return the last iterate as a successful label. A retry can increase precision or use the independent reference backend with exactly the same physical inputs; it must not soften the environment.

### 8.3 Precision policy

Use CPU FP64 sparse direct solves as an independently assembled small-panel reference. Qualify GPU FP64 and GPU FP32 with FP64 residual/energy evaluation; select the fastest policy that passes the prospective numerical bands.

Initial targets are `rtol=1e-10` for FP64 and `rtol=1e-6` for FP32/mixed solves, using the integrated operator and a separately declared absolute tolerance for zero/near-zero RHS. FP32 may hit a true-residual floor. In that event use bounded mixed-precision iterative refinement (FP64 residual, FP32 correction solve, at most three corrections) or a FP64 retry. Do not silently loosen tolerances after seeing target discrepancies.

Material inputs and geometry should be stored canonically in FP64 because they are small. If coefficients are narrowed for a fast solve, the acceptance residual and reference comparison must be evaluated against the **canonical** operator; otherwise one merely verifies a different rounded problem. Store precision, retry count, pre-storage and post-storage audits per case.

No autocast, TF32 change, PyTorch upgrade, or new compiler path is introduced merely to claim speed. Equivalent batched/sliced floating-point calculations need not be bitwise identical [R6]; compare numerical and conservation errors under declared tolerances.

### 8.4 Optional optimization, only after a correct pilot

An optional compiled matrix-vector/CG block or symmetric geometric multigrid preconditioner may be tested if Jacobi-PCG materially threatens the measured formal forecast. Limit this to one bounded engineering extension. A multigrid preconditioner used by CG must be symmetric positive definite (compatible restriction/prolongation, symmetric smoothing, and a correctly anchored coarse solve).

Do not add custom CUDA kernels, distributed solvers, or a large dependency stack unless the simple implementation demonstrably cannot complete the bounded pilot. Compile/setup time belongs in the cold forecast. Cache paths must be below `/data2`.

### 8.5 GPU2 ownership and controls

The intended physical device is **GPU2**, historically UUID `GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39`. Verify the actual index/UUID mapping using `nvidia-smi` before setting `CUDA_VISIBLE_DEVICES`; never assume device index persistence. Expose only that UUID and use logical `cuda:0` inside the process.

Check active compute processes, GPU memory, and current user jobs before allocation. Round9's separate authorization for GPU1/GPU2 does not authorize this generator to displace its jobs or use GPU1. Do not send signals to unrelated processes. If an unrelated GPU2 job appears during the pilot, checkpoint at a safe batch boundary and report the conflict instead of claiming uncontended timing.

Keep at least `max(8 GiB, 20% of physical capacity)` free. Start with a 4 GiB process-allocation budget; the small-grid generator should not require a mostly full 48 GiB device. Increasing that cap requires a measured need and a new reservation, not an automatic attempt to fill memory.

Record UUID, software versions, CPU thread counts, batch size, precision, allocated/reserved/device memory, and all timing conditions. Pin CPU thread counts to a modest value such as two for initial comparisons; report actual choices. The CUDA environment-variable behavior and synchronization rules should follow the installed PyTorch version [R7].

---

## 9. Numerical qualification and resolution decision

### 9.1 Independent implementations

Required checks include:

- An independent small CPU sparse assembler using explicit face contributions and a sparse direct solution, not a wrapper around the same matrix-free function.
- Literal small-grid energy identities, zero-flux boundary tests, positivity, and operator symmetry.
- A smooth manufactured-solution test for each elliptic operator, with analytically consistent sources/boundaries. This is a numerical unit test, not an extra physical dataset family.
- Homogeneous contact examples, coefficient-jump examples, and nonoverlapping-pad quadrature checks.

Scalar coefficients, an orthogonal grid, and positive reaction/contact terms are the assumptions behind the selected finite-volume/CG approach [R3,R8].

### 9.2 Proposed solver-error acceptance bands

For nondegenerate pilot cases, apply these before considering grid error:

| Check | Prospective acceptance criterion |
|---|---|
| True canonical electrical and thermal relative residual | <= 1e-6 fast path; FP64 reference target <= 1e-10 |
| Net current imbalance | `abs(sum I) / sum abs(I) <= 1e-5` |
| Electrical power closure | <= 1e-5 of positive supplied power |
| Thermal power closure | <= 1e-5 of total Joule heat |
| GPU vs independent CPU FP64, v/theta relative L2 | <= 5e-5 |
| GPU vs CPU, current / total power / peak temperature | <= 1e-4 relative where resolved |
| Potential maximum principle | within [min u,max u] plus a 1e-5 voltage-span tolerance |
| Nonnegative q and theta | no material negative values; tolerance tied to fixed reference scale, not hidden clipping |
| Repeated/batched/permuted state agreement | comparable to the same solver-reference envelope; exact input hashes/IDs |

Near-zero fields use a sealed absolute tolerance derived from the dimensionless benchmark scale and FP64 fixtures. Record maximum errors and per-field residuals, not only a scalar mean. A missed fast-path band triggers a precision/backend repair, not deletion of the case. Report the fallback rate and include it in runtime estimates.

### 9.3 Spatial and contact-quadrature error

Compare identical physical cases at 48², 96², and 192². Restrict fine scalar/areal fields by conservative cell averaging and native face fluxes by face integration/averaging. The fixed log-bilinear environmental renderer is part of the physical input definition; do not introduce extra resolution-dependent smoothing of coefficients or target currents to improve the comparison. In sharp-interface diagnostic variants, do not smooth away material jumps.

Proposed main-resolution acceptance versus a qualified finer reference:

| Quantity | Suggested grid-discrepancy band |
|---|---:|
| Potential and temperature relative L2 | <= 1% |
| Terminal-current vector and total electrical power | <= 2% |
| Current/thermal-flux field relative L2 on declared reconstructed/native support | <= 5% |
| Cell-averaged Joule source relative L2 | <= 8%, with contact/interface strata visible |
| Maximum temperature and top-5%-temperature regional mean | <= 3% / 5% |
| Contact pre-normalization quadrature-mass error | <= 0.2%; tighten quadrature if larger |

These are prototype dataset-quality targets, not estimates of continuum error from a theorem. Include hotspot regions, contact neighborhoods, strong material-transition regions (or exact interfaces in the sharp diagnostic variant), and bulk separately. Do not demand a stable single argmax location for a flat plateau; show top-1% hot regions and their overlap/centroid in that case.

Check 192² against 384² on up to four input-selected difficult/representative cases when necessary to ensure the supposed reference is not also underresolved. Do not presume second-order convergence across discontinuous coefficients.

Choose one primary native resolution for the core release. If 48² fails and 96² passes, recommend 96². If neither passes, report which physical features/fields fail and recommend finer resolution or a prospectively revised minimum feature/contact scale. **Do not silently blur, shrink contrasts, or drop difficult cases after generation.** Keep high-contrast OOD qualification separate.

### 9.4 Scientific-interest audit

Report, without target-based filtering of ordinary cases:

- conditional field variation across environments at the same layout;
- conditional variation across layouts at the same environment;
- sheet/contact Joule-power share distributions;
- terminal-current and hotspot distributions by morphology and M;
- spatial temperature variation after subtracting its mean as an additional diagnostic, while retaining unmodified absolute dimensionless fields and losses; this reveals a nearly uniform-temperature regime that a low global relative error could hide;
- measured ME/MM/EE interaction magnitude relative to numerical uncertainty;
- examples where conductivity paths and Euclidean proximity disagree;
- covariance/effective spatial complexity of the input maps and actual property contrast coverage.

If almost all heat is in contacts, or thermal smoothing makes every temperature nearly constant, explain the regime and propose a limited pre-formal adjustment to `gamma`, `beta`, or contrast. Save original pilot results. Never discard individual correct cases because HONF would find them inconvenient.

## 10. Raw data and processed-data contract

### 10.1 Input manifests are immutable scientific records

Before solving a state, save its exact canonical input. Use deterministic, independent RNG namespaces for environment roots, layouts, voltage patterns, pair assignment, and intervention families. Suggested key material is `(dataset_version, root_seed, namespace, family, root_index)`, hashed with SHA256 to seed a local NumPy generator. Never use Python's process-randomized `hash()`.

Persist actual realized coefficient tiles and contact inputs, not seeds alone. Seeds, library versions, and generation recipes enable reproduction; the saved numerical input defines the actual reference problem. Changes in precision, grid, discretization, contact quadrature, or physics must change the solver/configuration hash.

Every state record should include:

```text
state_id, physical_input_sha256, physics_schema, solver_schema
split, environment_root_id, layout_root_id, voltage_pattern_id
environment_seen_in_train, layout_seen_in_train
morphology_family, layout_family, M, parameter_stratum
contact_ids, contact_xy, contact_radius, contact_gamma, terminal_voltage
environment_catalogue_key, coefficient_tile_hash, grid_nx_ny
interaction_family_id, endpoint_bits, intervention_definition
status, accepted_attempt_id, retry_history, raw_shard_key, processed_shard_key
```

Rejected geometry proposals and numerical attempts have status records. A failed numerical attempt is not a completed case, and a successful retry does not erase the failure. Do not use file existence alone as a resume criterion; require matching hashes, complete datasets, and the accepted-state marker.

### 10.2 Raw payload

Prefer HDF5 for compact, portable array storage, consistent with the existing project's use of external data and manifests. A formal raw shard holds up to 64 cases of the same native grid, with datasets chunked primarily by case. Store:

| Dataset | Shape per state | Recommended persisted dtype / meaning |
|---|---|---|
| `potential` | `[ny,nx]` | FP32 after storage-quality qualification |
| `temperature_rise` | `[ny,nx]` | FP32, dimensionless theta |
| `q_sheet` | `[ny,nx]` | FP32, resistance-partitioned face energy per cell area |
| `q_contact` | `[ny,nx]` | FP32, contact loss per cell area |
| `jx_face` | `[ny,nx+1]` | FP32, oriented sheet-current density |
| `jy_face` | `[ny+1,nx]` | FP32, oriented sheet-current density |
| `heat_flux_x_face` | `[ny,nx+1]` | FP32, oriented in-plane thermal flux |
| `heat_flux_y_face` | `[ny+1,nx]` | FP32, oriented in-plane thermal flux |
| `terminal_current` | `[M]`, ragged or padded with explicit mask | FP64 diagnostics |
| `diagnostics` | small named arrays / tables | FP64 errors, currents, powers, extrema, solver costs |

Canonical tile coefficients, contacts, and shared coordinate catalogues are stored once in input catalogues, with exact references. Do not repeat three dense environment maps or coordinate meshes per formal case when they can be reconstructed exactly. For small FP64 qualification examples, keep the higher-precision raw arrays in a separately labelled QA group.

The raw fields represent two coupled PDE solutions and their derived quantities, not five independent physical systems. Retain native face quantities for future conservation evaluation even when ordinary ML predicts only cell fields.

### 10.3 Processed payload for future HONF training

A minimal processed shard contains `fields[case,ny,nx,5]` ordered

`[potential, jx_cell, jy_cell, temperature_rise, q_total]`.

Here `q_total=q_sheet+q_contact` and cell currents follow §3.4. Keep fields in physical dimensionless units. Provide a loader-side optional TRAIN-fitted normalizer, stored separately, rather than duplicate raw and normalized copies of every field.

For a later inverse study, distinguish temperature minimization at fixed nonzero electrical service from the trivial solution obtained by turning the device off. A design request should constrain terminal-voltage range, supplied current/power, or required receiver response. This pilot does not run inverse optimization.

The model-facing input contains:

- module records `(x,y,r,gamma,u)` and source IDs/masks;
- original environmental records `(x,y,side_x,side_y,s,kappa,beta)` and tile IDs/areas;
- arbitrary query coordinates or native grid IDs;
- explicit coordinate/feature/units schema.

Family names, seeds, test labels, conductivity-path diagnostics, solver errors, terminal currents, and target-derived hotspots are excluded from predictor inputs. Pad counts are variable; padding is a batching convention, not a physical extra module. Keep modules and E records separate instead of merging contacts into the environmental coefficient map and losing their identities.

For a grid baseline, create **input-only** raster channels from the same E records and contact geometry/controls. Include `a=sum gamma*chi`, `b=sum gamma*chi*u`, and `d=sum gamma*chi*u²`, or preserve the full per-contact footprint/voltage channels. The third moment `d` is needed to determine contact Joule heat as `a*v²-2*b*v+d` in a general rasterized contact mixture; `a,b` alone are not always an information-complete electro-thermal input. For numerical reference heating, retain the nonnegative per-contact squared-drop formula rather than that cancellation-prone expanded expression. `d` is a known input statistic, **not** solved Joule heat. Do not allow a baseline to read q, current, terminal-current solutions, or finer hidden coefficients as conditioning.

### 10.4 Sampling and normalization

The pilot should implement and test a simple case/query loader but perform no HONF fitting. Use exact native cells for primary quantitative targets. For a future training configuration, start with 1,024–2,048 queries per case, with an input-only mixture such as 70% uniform cells, 20% contact neighborhoods, and 10% strong input-material-transition neighborhoods. Preserve sampling probabilities and distinguish this training draw from uniform full-field evaluation. Do not let solved hotspot targets determine inference queries.

Future normalization statistics must use ordinary TRAIN targets only, accumulated in FP64 with stable moments. Do not refit on DEV/TEST, divide by each sample's own peak, or fit formal transforms from the exposed pilot. Pilot pack/loader testing can use a clearly marked pilot-only transform. Evaluate full native grids and all physical diagnostics without normalization ambiguities.

### 10.5 Compression and writing

Benchmark uncompressed, LZF, and gzip level 1 on a small representative subset; select based on measured write/read time and size. HDF5 chunks of one case/field or one case of the five-channel processed field are a reasonable starting point at 96² [R9]. Report the actual chunk and compression settings.

Use a **single writer per shard**. Workers may compute in parallel but cannot independently write the same HDF5 file. Write to a round-owned temporary file, flush/close, validate, then atomically rename on the same `/data2` filesystem. Mark completed case/shard hashes in the append-only journal. Interrupted partial shards are resumed/rebuilt under declared hashes; they are never treated as ready data.

For the pilot, per-case files are acceptable for inspection. Test formal sharding on a small pilot subset. Avoid thousands of uncompressed `.npz`/JSON arrays or image files in the final dataset.

---

## 11. Storage estimates and measured runtime forecasts

### 11.1 Analytic storage baseline

With the raw schema above, a square N × N state stores approximately `8*N²+4*N` float32 values: four cell fields plus two electrical and two thermal staggered-face arrays. The minimal processed state adds `5*N²` float32 values.

Therefore:

\[
S_{\rm raw}(N)=4(8N^2+4N),\quad
S_{\rm processed}(N)=20N^2,\quad
S_{\rm combined}(N)=52N^2+16N\ \text{bytes/state}.
\]

These counts exclude small input/diagnostic tables, FP64 qualification arrays, figures, manifests, and compression. They do not count timing-only repeat outputs because those should not be persisted.

| Grid | Raw bytes/state | Processed bytes/state | Compact combined GiB | Standard combined GiB | Expanded combined GiB |
|---|---:|---:|---:|---:|---:|
| 48² | 74,496 | 46,080 | 0.259 | 0.517 | 1.035 |
| 96² | 296,448 | 184,320 | 1.032 | 2.063 | 4.126 |
| 192² | 1,182,720 | 737,280 | 4.120 | 8.240 | 16.479 |

For provisional budgeting, add 25% overhead before separately adding retained figures and high-resolution QA. At 96² this gives roughly **1.29 / 2.58 / 5.16 GiB** for compact/standard/expanded raw-plus-processed payload. These are schema-based estimates, not measured filesystem sizes or guaranteed compression ratios. The pilot report must replace/supplement them with actual logical bytes, allocated filesystem bytes, and per-family compression statistics.

If raw arrays must remain FP64 for accuracy, the raw part doubles, not necessarily the entire dataset. Show that variant explicitly. If a stress panel needs a finer native grid, estimate it separately rather than multiply every row by the core-grid rate.

### 11.2 Measure complete case cost, not only the CG kernel

Record disjoint scopes:

`input generation -> material/contact preparation -> H2D -> electrical solve -> face/current/Joule postprocess -> thermal solve -> diagnostics -> D2H -> raw write -> pack/write`.

Measure GPU compute intervals with CUDA events/synchronization as appropriate and complete host elapsed time separately. Do not sum overlapping asynchronous intervals as though they were disjoint. Include setup/compilation, retries, warm-cache differences, and CPU input generation. Plots are timed separately and generated only for declared examples.

Report batch-level throughput and per-case amortized time, p50/p90 iteration counts, tail/fallback cases, CPU-versus-GPU performance, and warm versus cold preparation. An apparent speedup on a reused environment is not the cost of a completely fresh case.

### 11.3 Forecast script

Implement `forecast` to join an exact prospective input manifest with measured strata and produce compact/standard/expanded tables for:

- distinct environment/layout/state counts;
- electrical and thermal system counts, including planned physical counterfactuals;
- best-supported mean total wall time and a conservative tail forecast;
- setup, I/O, packing, selected visualization, QA, and retry allowances;
- raw/processed/QA/figure storage and peak temporary disk requirement;
- measured GPU and host-memory limits.

One useful formula is

\[
\widehat T=T_{\rm setup}+
\sum_g n_g\widehat t_g+
T_{\rm QA}+T_{\rm pack}+T_{\rm selected\ plots}+T_{\rm retry\ allowance},
\]

where `g` identifies measured grid/morphology/contrast/M/backend strata. Use batch measurements for the intended batching policy. A simple `case_count * fastest_single_kernel_time` is unacceptable. For unmeasured strata, label the extrapolation and give a range; do not invent a precise p90.

For scale only, 4,608 cases at an **assumed** complete rate of 0.2 / 1 / 5 seconds per case would take 15.36 minutes / 1.28 hours / 6.4 hours before QA and setup. These are sensitivity examples, **not throughput predictions**. The deliverable must quote the actual pilot-supported rate and replace such placeholders.

The forecast should also show incremental cost per additional 60 core TRAIN environments (720 ordinary states), making a size decision about new physical diversity rather than arbitrary file counts easy.

---

## 12. Repository and `/data2` directory management

### 12.1 Source code stays separate from payload

Proposed tracked layout:

```text
HONF_Proj/
  Case_ElectroThermal/
    README.md
    pyproject.toml
    .gitignore
    configs/
      pilot_v1.yaml
      formal_compact_v1.yaml
      formal_standard_v1.yaml
      formal_expanded_v1.yaml
    Dataset/
      README.md
      PHYSICS_AND_DATA.md
      dataset_manifest.json                # registry/status; no fabricated ready hashes
      dataset_locations.example.json
      schemas/
        physical_case_v1.schema.json
        environment_catalogue_v1.schema.json
        raw_shard_v1.schema.json
        processed_shard_v1.schema.json
      dataset_locations.local.json          # ignored, machine-local
      pilot -> /data2/.../pilots/<run_id>/    # ignored browsing links
      raw -> /data2/.../pilots/<run_id>/raw/
      processed -> /data2/.../pilots/<run_id>/processed/
      figures -> /data2/.../pilots/<run_id>/figures/
      catalogues -> /data2/.../pilots/<run_id>/inputs/
      links/
        active_manifest.json -> /data2/.../pilots/<run_id>/manifest.json
    src/hetmf/
      __init__.py
      __main__.py
      cli.py
      config.py
      environment.py
      layouts.py
      contacts.py
      finite_volume.py
      solvers.py
      physics.py
      validation.py
      study_design.py
      interventions.py
      pipeline.py
      io.py
      resources.py
      processing.py
      dataset.py
      visualization.py
      forecasting.py
    scripts/
      run_pilot.py                         # thin wrappers, not copied solver loops
      generate_dataset.py
      postprocess_dataset.py
      visualize_dataset.py
    tests/
      test_physics.py
      test_numerics.py
      test_design_and_splits.py
      test_io_and_resume.py
      test_resources_and_links.py
    docs/
      HETMF_PILOT_REPORT.md
      HETMF_GENERATION_GUIDE.md
```

Generated code caches, data, model artifacts, figures, logs, and per-case metadata do not belong in Git. A short report, schema, and real summary measurements can be committed. A registry entry for an ungenerated formal release must say `planned/unavailable`, with null/absent actual size/hash fields; never manufacture a ready resource.

### 12.2 Physical disk layout

Use a verified root such as:

```text
/data2/wanglz/ModularDT/
  datasets/hetmf/
    pilots/<run_id>/
      inputs/
      raw/
      processed/
      diagnostics/
      figures/
      logs/
      temporary/
      receipts/
      manifest.json
      solver_ledger.jsonl
    releases/<approved_release_id>/
      inputs/
      raw/shards/
      processed/shards/
      diagnostics/
      figures/selected/
      manifests/
      receipts/
  caches/hetmf/<run_id>/
  worktrees/hetmf-dataset-pilot/             # only if using an isolated worktree
```

The physical raw/processed files are always under `/data2`, irrespective of the terminal's current directory. Do not create output first below the repository and then migrate it. Do not duplicate Thermal/Wind datasets, move their historical roots, or reuse the active Round9 diagnostics directory.

### 12.3 Symlink safety

Implement a case-local link helper that verifies `/data2` is genuinely mounted, the target exists below the allowed owned root, and the destination lies under this case's `Dataset/`. Distinguish absent, regular file/directory, live symlink, and dangling symlink. Refuse to overwrite a regular file/directory or an unrelated link. Updating an existing HET-MF-owned symlink requires an explicit `--replace-owned-links` flag, an ownership record, and atomic replacement.

Use `findmnt`/mount metadata plus resolved paths; an ordinary `/data2` directory on the root disk is not an acceptable substitute when the drive is unmounted. Check ownership, writability, free capacity, and symlink traversal before payload creation.

**Repository finding:** the reviewed `HONF_Proj/tools/link_datasets.py` still hard-codes ThermalChannel's link destination even when another locations file is supplied. Do not invoke it with HET-MF locations or modify it underneath another running task. Implement the narrow case-local helper first; later genericization can be a separately reviewed change.

Browsing symlinks are conveniences. The loader uses the machine-local location map and logical dataset ID, not the current working directory or mandatory symlink traversal. Generated manifests live on `/data2`; the tracked registry points to them without embedding machine-specific ready data paths in portable source defaults.

### 12.4 Protect ongoing Round9 development

Prefer a separate branch `codex/hetmf-dataset-pilot` and an isolated worktree based on the inspected current repository state. Never switch branches or reset the working tree in which Round9 runs. An isolated worktree has its own `HONF_Proj/Case_ElectroThermal/Dataset/` browsing view with exactly the structure above; provide its absolute path in the report.

After review/integration, the same case directory and links can be exposed in the user's main `HONF_Proj` tree. If immediate browsing there is needed, add only a clearly named, ignored, nonconflicting preview link after checking it does not shadow an existing directory. Do not insert an untracked copy of source code that will collide with a later merge. Do not cherry-pick or merge into the active Round9 worktree automatically.

If an isolated worktree cannot preserve needed uncommitted context, record the limitation and use only new case-local files with an explicitly disjoint change inventory. No global registry refactor is required for this pilot.

### 12.5 Caches, cleanup, and failure handling

Set `TMPDIR`, `MPLCONFIGDIR`, CUDA/compilation caches, and Python bytecode cache policy to owned `/data2` locations before importing heavy libraries. Avoid editable installs that modify the shared Round9 environment; use a local `PYTHONPATH` or a separate verified environment. Do not upgrade PyTorch/CUDA or install unrelated packages.

Cleanup is limited to pinned, round-owned disposable files. Never recursively clean a shared `diagnostics`, dataset, worktree, or cache root. Keep successful raw states, all representative/interaction evidence, input manifests, error receipts, and numerical summaries. Report actual logical and allocated bytes, not just apparent path size. No automatic deletion of raw data after packing.

---

## 13. Case-specific script guidance and CLI

### 13.1 Architectural boundaries

`environment.py` owns coefficient generation; `layouts.py` owns independent contact geometry/voltage generation; `contacts.py` owns compact footprint quadrature; `finite_volume.py` owns transmissibilities and operator application; `solvers.py` owns CPU reference and batched GPU solves; `physics.py` owns current/Joule/thermal coupling and conservation.

`study_design.py` freezes root pools, splits, and pairings. `interventions.py` applies exact local changes without global renormalization. `pipeline.py` orchestrates generation, journaling, resumption, and resource stops. `processing.py` derives canonical ML fields but never changes the physical reference. `dataset.py` is a minimal read-only loader, not integration into HONF's training engine.

Keep configuration declarative and validated. Typed dataclasses or equivalent are appropriate. Reject unknown fields and negative/nonfinite coefficients. Avoid separate subtly different solver implementations in each script.

### 13.2 Proposed commands

The exact parser implementation may differ, but provide these operations and test their installed/module entry points. The commands below are **interfaces to implement**, not claims that the scripts already exist.

```bash
# Run with the verified interpreter and the case source on PYTHONPATH.
python -m hetmf preflight --config configs/pilot_v1.yaml --device-uuid <verified-GPU2-UUID>
python -m hetmf plan --profile pilot --output <data2-pilot-root>/inputs/pilot_plan.json
python -m hetmf pilot --stage representatives --plan <pilot-plan> --device cuda:0
python -m hetmf pilot --stage qualify --plan <pilot-plan> --device cuda:0
python -m hetmf pilot --stage interactions --plan <pilot-plan> --device cuda:0
python -m hetmf pilot --stage throughput --plan <pilot-plan> --device cuda:0
python -m hetmf validate --root <data2-pilot-root> --full
python -m hetmf pack --root <data2-pilot-root> --scope pilot
python -m hetmf visualize --root <data2-pilot-root> --gallery representatives
python -m hetmf forecast --pilot <data2-pilot-root> --profiles compact standard expanded
python -m hetmf link --root <data2-pilot-root> --case-dir <workspace>/HONF_Proj/Case_ElectroThermal
python -m hetmf status --root <data2-pilot-root>
```

Provide one convenience `pilot --all` orchestration that respects stage gates and resume hashes rather than bypassing validation. It must save visible progress after representative cases and resumable state after each completed batch.

Implement and CPU-dry-run formal commands without launching them:

```bash
python -m hetmf plan --profile standard --output <data2-plan-root>/formal_standard_plan.json
python -m hetmf generate --plan <formal-plan> --dry-run
# Later, only after the user approves exact size/grid/resources:
python -m hetmf generate --plan <formal-plan> --approval <approved-json> --yes --device cuda:0
python -m hetmf generate --plan <formal-plan> --approval <approved-json> --yes --resume
python -m hetmf pack --release <release-root> --verify-source-hashes
```

Formal generation requires an approval record bound to the manifest hash, maximum states, grid/resolution groups, coefficient bounds, physical GPU UUID, disk root, and time/storage limits. A pilot approval is not formal approval. A formal `--dry-run` can validate plans and forecast costs but cannot begin PDE solves or create populated formal shards.

Resumption is input/hash-based and independent of batch size. A clean stop preserves completed states and journal entries. SIGINT or a task-owned stop file requests a safe batch boundary; it must not signal another task.

### 13.3 Public reusable functions

Suggested narrow API:

```text
generate_environment(spec, seed) -> EnvironmentRecords
sample_layout(spec, seed) -> ContactRecords
rasterize_environment(records, grid) -> CoefficientFields
rasterize_contacts(records, grid, quadrature) -> ContactWeights
build_electrical_operator(coefficients, weights) -> OperatorAndRHS
solve_electrical(operator, rhs, solver_config) -> SolutionWithResiduals
compute_currents_and_joule(solution, operator, weights) -> ElectricalOutputs
solve_thermal(coefficients, joule, solver_config) -> ThermalOutputs
validate_coupled_case(inputs, outputs, config) -> ValidationReceipt
write_case_atomic(inputs, outputs, receipt, root) -> CommittedCaseRecord
```

Operator definitions should support multiple right-hand sides for diagnostic terminal bases. Do not expose the learned model's router to the generator or use neural predictions as reference initial data unless it becomes an explicitly separate future study.

### 13.4 Minimal configuration inventory

The pilot config must serialize at least:

```yaml
schema: hetmf_pilot_config_v1
rng:
  root_seed: 20261017
  key_hash: sha256_canonical_namespace_v1
  pilot_namespace_disjoint_from_formal: true
physics:
  domain: [1.0, 1.0]
  sheet_model: scalar_thin_sheet_one_way_electrothermal
  lateral_electrical_bc: insulating
  lateral_thermal_bc: insulating
  contact_heat_fraction_to_sheet: 1.0
  temperature_dependent_coefficients: false
  units: dimensionless_fixed_reference_scales
environment:
  tile_shape: [12, 12]
  reconstruction: log_bilinear_centers_border_v1
  family_weights: {correlated_mosaic: 0.40, layered_barrier: 0.20, inclusions: 0.20, corridors: 0.20}
  electrical_contrast_bound: [3.0, 30.0]
  thermal_contrast_bound: [1.5, 6.0]
  cooling_level: [2.0, 12.0]
  cooling_contrast_bound: [1.0, 3.0]
  latent_correlation_modes: [0.0, 0.7]
contacts:
  counts: [2, 3, 4, 6, 8, 12]
  radius: [0.06, 0.09]
  total_conductance: [1.0, 10.0]
  footprint: normalized_compact_quartic_v1
  quadrature_order_per_axis: 4
  edge_extra_clearance: 0.025
  interpad_extra_clearance: 0.015
  core_voltage_range: 1.0
grid:
  first_representative: [96, 96]
  refinement_levels: [48, 96, 192]
  optional_reference: 384
solver:
  method: batched_matrix_free_pcg
  preconditioner: jacobi
  precision_policy: qualify_fp64_and_fp32_refined
  fast_true_rtol: 1.0e-6
  fast_absolute_residual_tolerance: 1.0e-10
  reference_true_rtol: 1.0e-10
  reference_absolute_residual_tolerance: 1.0e-13
  max_iterations: 4000
  max_refinement_corrections: 3
  stopping_check_interval: 10
pilot:
  representative_count: 12
  counterfactual_quartets: 6
  throughput_states: 256
  batch_sizes: [1, 4, 16, 32]
  timing_repeats: 3
  max_distinct_states: 320
  max_coupled_evaluations: 4000
  max_linear_system_attempts: 10000
resources:
  physical_gpu: 2
  required_gpu_uuid: GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39
  min_free_gpu_gib: 8
  min_free_gpu_fraction: 0.20
  initial_process_allocation_cap_gib: 4
  elapsed_budget_hours: 6
  gpu_associated_budget_hours: 2
  retained_budget_gib: 15
  temporary_budget_gib: 5
  minimum_free_disk_gib: 50
  data2_base: /data2/wanglz/ModularDT/datasets/hetmf
formal_generation_authorized: false
ml_training_authorized: false
```

Complete the output schema, near-zero field/energy absolute tolerances, diagnostics, and plotting settings before running. The explicit residual tolerances above apply to the integrated operator; validate them on FP64/zero-response fixtures. Do not leave undefined numerical conventions to implicit defaults. Templates are not receipts of completed experiments.

---

## 14. Postprocessing and visualization

### 14.1 Representative case pages

Create one consistent, readable page per representative state showing:

1. Electrical conductance map with labelled contact disks and prescribed voltages.
2. Thermal conductance and cooling maps, with the same geometry.
3. Potential and current magnitude/direction.
4. Sheet and contact Joule heating separately, plus total heating if useful.
5. Temperature rise and thermal-flux direction.
6. Terminal-current bars and a compact power/residual/iteration table.

Use correct `[y,x]` orientation, physical domain extent, equal aspect ratio, and visible material-control lattice markers where useful. Do not draw the primary graded medium as piecewise-constant physical tiles without identifying that view as the input-control catalogue. Electrical and thermal material maps can use logarithmic color scales with truthful labels. Use a diverging scale for signed potential/current components and sequential scales for nonnegative heat/temperature. Plot smoothing is a display operation only and may not replace the target arrays.

Current streamlines, when included, are qualitative displays of the reconstructed vector field. Use native face data for quantitative balances. Contact outlines should remain visible rather than conceal the sheet temperature under them. Label sheet-current units as dimensionless here, never bulk A/m².

### 14.2 Paired and interaction pages

For each quartet, use the same color scale across its four endpoints, and a separate symmetric signed scale for differences/interaction residuals. Show the exact changed pad/patch, its magnitude, all unchanged inputs, physical endpoint hashes, and the measured numerical floor. Do not normalize every panel independently.

For a current/temperature perturbation too small to resolve, say so in the caption rather than autoscale it to look dramatic. Include an electrical-null/thermal-change example alongside nonzero ME examples.

### 14.3 Numerical and cost boards

Produce compact plots/tables for refinement errors, conservation errors, PCG iterations, throughput versus batch size, per-family time tails, contact-power fraction, and measured storage. Show failed/retried cases and the selected precision/resolution policy. Do not substitute a visually smooth contour for numerical evidence.

Use lossless PNGs and vector PDF masters for a small selected figure set, plus a local HTML index linking saved arrays and figures. PDF creation is postprocessing only; it is not a required simulator dependency. Captions state dimensionless units, exact native grid, case identity, and whether a figure is a physical solve, derived field, or numerical comparison.

The formal script should default to plotting only a stratified gallery (for example 24–48 states), all counterfactual summary boards, and dataset-wide histograms—not every ordinary formal case. The report must include the plotting/storage forecast separately from PDE generation.

### 14.4 Export ready for paper development

Retain plots' source arrays, case/manifest hashes, color-range settings, and generation commands. Save full numerical field arrays, not only screenshots. The user should be able to reopen `Dataset/figures/` and locate a case's raw input/output from the figure caption or HTML link.

Do not overlay imagined HONF groups: no HONF model has been trained on this new dataset in the pilot. Physical current paths and terminal conductance matrices are diagnostic reference evidence, not learned hypergraph visualizations.

---

## 15. Focused tests and acceptance checklist

### W0 — scope and storage

- GPU UUID/index checks, busy-device refusal, no GPU1 fallback, and no signals to other jobs.
- `/data2` unmounted/wrong-root/full-disk rejection before writes.
- Owned-link creation/update, dangling links, destination regular directories, path traversal, and repeated invocation.
- No source reset, shared training-file edits, model training, or formal generation from a pilot command.

### W1 — generator and physical model

- Positive bounded fields, canonical tile reconstruction, independent environment/layout RNG namespaces.
- Footprint integral, quadrature convergence, nonoverlap, geometry rejection reporting.
- Common-voltage shift, reversal, scaling, equal-voltage zero, terminal permutation, thermal-only electrical null.
- Electrical source/conductance sign convention and correct dimensionless scaling.

### W2 — numerical solver

- Operator symmetry, positive quadratic form, explicit boundary-flux behavior, independent sparse-matrix agreement.
- Cell and global balances, face-energy heating, resistance-weighted heat splitting at coefficient jumps.
- Zero RHS, near-zero RHS, inactive batched cases, heterogeneous batch convergence, maximum-iteration failure.
- Mixed-precision true-residual verification against canonical coefficients, numerical retry and storage reload.

### W3 — design, data, and resume

- Exact planned profile counts, environment/layout degrees, no duplicate ordinary pairs.
- Core root-disjoint splits and correctly labelled deliberate component reuse in supplemental panels.
- Quartets preserve untouched values, do not globally renormalize, and retain common root/split metadata.
- Input/schema hashes, interrupted shard detection, atomic writer, batch-size-independent resumption, and single writer locks.
- Loader input whitelist, correct field order, native query IDs, and no target-derived conditioning.

### W4 — pilot and forecasts

- Twelve representative pages and six physical quartets are present, with numerical quality status.
- All declared throughput strata and unsuccessful attempts are represented.
- Independent CPU reference/refinement comparisons and maximum-principle checks are recorded.
- Forecast arithmetic agrees with the planned counts and measured byte/time units.
- Formal generation dry-run succeeds without doing physical solves.

A small number of meaningful tests plus real numerical/pilot evidence is preferred to an enormous infrastructure test suite. Report actual tests passed/failed; do not claim global repository qualification from a case-local collection.

---

## 16. Required final report and stopping point

Write `Case_ElectroThermal/docs/HETMF_PILOT_REPORT.md` and an evidence index. The first page should answer:

- Is the physical solver numerically qualified, and on which cases/grids/precisions?
- Which environmental families produce useful independent conditioning and resolved interactions?
- What native grid is recommended, and what failed at the lower grid?
- How many independent environments, layouts, and physical states are proposed?
- What measured complete generation time and raw/processed/temporary size are predicted for each profile?
- Which raw/processed/figure directories are visible through the requested case-specific browsing links?
- What remains blocked or uncertain, and what exact user choice is needed before full generation?

Include a recommendation table:

| Profile | States / env roots / layout roots | Grid / precision | Measured forecast: mean / conservative time | Raw / processed / QA / temporary GiB | Status |
|---|---|---|---|---|---|
| Compact | actual manifest counts | qualified proposal | pilot-based | measured extrapolation | awaiting approval |
| Standard | actual manifest counts | qualified proposal | pilot-based | measured extrapolation | recommended or not |
| Expanded | actual manifest counts | qualified proposal | pilot-based | measured extrapolation | awaiting approval |

Cite exact source/config/manifest hashes, sampled GPU UUID, timing policy, real accepted/failed/repeated solve counts, and storage accounting. Field plots and simple statistics cannot establish future ML sample sufficiency; state this explicitly.

Commit/push only durable case-local source, tests, configs, schemas, guides, and concise reports on the separate branch. Audit the outgoing files to prevent raw data, caches, local path maps, and symlinks from entering Git. Report the commit SHA and remote readback. Do not merge into the active Round9 branch automatically.

**Stop here:** formal scripts and manifests ready, small pilot complete, runtime/storage forecast delivered. Ask the user to approve or revise profile size, primary resolution, any out-of-core challenge scope, and the measured resource forecast. Do not start the 2,304/4,608/9,216-case generation because a dry-run passed.

---

## 17. Source basis and design provenance

The user selected the electro-thermal contact-array scenario and requested GPU2 pilot generation, representative visual checks, a measured time/size recommendation, and `/data2` payloads exposed beneath a case-specific `Dataset/` directory. The following sources support the physical/numerical background or implementation conventions. The proposed mixture weights, contact model, parameter intervals, counts, and acceptance bands are **new benchmark-design choices in this plan**, not results claimed by those sources.

**Repository conventions checked at `53cba42bc73661306e469c7905be9422b235d672`:**

- `HONF_Proj/Case_ThermalChannel/Dataset/README.md`: external raw datasets, a logical registry, and machine-local locations; browsing links are optional.
- `HONF_Proj/Case_WindFarm/Dataset/dataset_manifest.json`: explicit schemas, units, native shapes, and layout-level grouping.
- `HONF_Proj/tools/link_datasets.py`: ThermalChannel-specific destination; not a generic HET-MF link helper.
- `HONF_Proj/docs/guides/Joint_Native_Hypergraph.md`: active Round9 is a separate workflow and the current device/horizon boundary is versioned. No GPU availability was inferred from that document.

Pinned repository base: https://github.com/cosmos2w/ModularDT/tree/53cba42bc73661306e469c7905be9422b235d672/HONF_Proj

**External primary / official sources:**

[R1] COMSOL, *Which Study Type Should I Use for My Electrothermal Analysis?* Electric-current/heat-transfer coupling and Joule heat `J dot E`. This supports the generic one-way coupling background, not the particular pad-source law or material calibration proposed here.
https://www.comsol.com/blogs/which-study-type-should-i-use-for-my-electrothermal-analysis

[R2] COMSOL, *Modeling Heat Transfer in Thin Layers* and *Designing Heating Circuits with Multiphysics Simulation*. Thin-layer simplification, conductive heating, and electrical/thermal power accounting as engineering precedents.
https://www.comsol.com/blogs/modeling-heat-transfer-in-thin-layers-via-layered-material-technology
https://www.comsol.com/blogs/designing-heating-circuits-with-multiphysics-simulation

[R3] Droniou, J. (2014), *Finite volume schemes for diffusion equations: introduction to and review of modern methods*, Mathematical Models and Methods in Applied Sciences, DOI `10.1142/S0218202514400041`; author preprint.
https://arxiv.org/abs/1407.1567

[R4] SciPy official `scipy.stats.qmc.LatinHypercube` documentation. Stratified marginal sampling and explicit RNG semantics. Use syntax appropriate to the actually installed SciPy version.
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html

[R5] SciPy official `scipy.stats.qmc.Sobol` documentation. Power-of-two balance and the limitations of skipping/thinning sequences.
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html

[R6] PyTorch 2.6, *Numerical accuracy*. Equivalent execution arrangements need not be bitwise equal; numerical agreement is distinct from mathematical equivalence.
https://docs.pytorch.org/docs/2.6/notes/numerical_accuracy.html

[R7] PyTorch 2.6, *CUDA semantics*. Device management, asynchronous execution, and correct timing/synchronization. Verify the installed version before selecting APIs; no environment upgrade is authorized.
https://docs.pytorch.org/docs/2.6/notes/cuda.html

[R8] SciPy official `scipy.sparse.linalg.cg` documentation. CG requires a Hermitian positive-definite operator and a compatible preconditioner; an independent sparse direct solver remains the pilot reference.
https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html

[R9] h5py official dataset documentation. Chunking, lossless compression, and dataset I/O design; actual performance must be measured for these arrays and storage hardware.
https://docs.h5py.org/en/stable/high/dataset.html

[R10] Müller, S., Schüler, L., Zech, A., and Heße, F. (2022), *GSTools v1.3: a toolbox for geostatistical modelling in Python*, Geoscientific Model Development 15, 3161–3182. DOI `10.5194/gmd-15-3161-2022`. Reference for covariance-controlled spatial random fields; GSTools is not a required dependency for the small direct-covariance generator here.
https://gmd.copernicus.org/articles/15/3161/2022/

[R11] Takamoto et al. (2022), *PDEBench: An Extensive Benchmark for Scientific Machine Learning*, NeurIPS Datasets and Benchmarks. Reproducible generation and multi-metric evaluation precedent; it does not establish an appropriate sample count for HET-MF.
https://arxiv.org/abs/2210.07182

The previous exploratory conversation included tentative battery-reference bibliographic details. Do not copy unverified author/title/DOI pairings from that conversation into this dataset's paper. A future formal bibliography should verify each engineering reference against its publisher record; the physical model here is fully specified without depending on such a pairing.

---

## 18. Final development principle

**Generate a small, physically complete and numerically defensible benchmark before generating a large one.** Preserve enough controlled randomness to test unseen environments, enough designed morphology to expose pathway changes, and enough paired interventions to evaluate joint interaction organization. Keep every future model's given environmental information equivalent. Report measured speed and storage, retain failures, and leave the final dataset-size decision with the user.
