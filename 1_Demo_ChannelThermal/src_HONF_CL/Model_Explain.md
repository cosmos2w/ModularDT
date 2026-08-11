# Mathematical and Data Definition of HONF-CL

This document defines the current Hypergraph Operator Neural Field (HONF) as a
neural surrogate for modular physical designs, specifies the ChannelThermal
prediction problem and its packed datasets, and explains how the reusable HONF
is adapted to that problem.

The presentation follows the implementation in:

- `honf_core/model.py`, `honf_core/organizer.py`, and
  `honf_core/decoder.py` for the reusable HONF;
- `data/datasets.py` for the training data contract;
- `channelthermal/input_adapter.py`, `environment.py`,
  `local_coupling.py`, and `model.py` for the thermal-channel adaptation;
- `local_surrogate/model.py` for the Stage-A single-module surrogate; and
- `train.py` and `training_tools/losses.py` for the coupled objective.

Only two global decoder modes are discussed:

1. `hyper_plus_global_near`, used here as the base HONF formulation; and
2. `enhanced_honf_pairwise`, the real default in
   `configs/train_global_honf_template.json`.

## 1. General mathematical definition of HONF

### 1.1 Modular operator-learning problem

Let a design contain a variable-size set of modules

$$
\mathcal{M}
=
\left\{
  \left(\mathbf{x}_m,\mathbf{s}_m\right)
\right\}_{m=1}^{N_M},
$$

where $\mathbf{x}_m\in\Omega\subset\mathbb{R}^{d_x}$ is a module position and
$\mathbf{s}_m\in\mathbb{R}^{F_M}$ contains module properties. Let
$\mathbf{c}\in\mathbb{R}^{F_G}$ contain case-level/global properties, and let

$$
\mathcal{E}
=
\left\{
  \left(\mathbf{y}_e,\mathbf{r}_e\right)
\right\}_{e=1}^{N_E}
$$

be environment samples that describe the domain and boundary context.

The desired physical solution is a field-valued operator

$$
\mathcal{G}^{\star}:
(\mathcal{M},\mathcal{E},\mathbf{c})
\longmapsto
\mathbf{u}(\cdot),
\qquad
\mathbf{u}:\Omega\rightarrow\mathbb{R}^{F}.
$$

HONF approximates this operator by

$$
\widehat{\mathbf{u}}_{\Theta}(\mathbf{q})
=
\mathcal{G}_{\Theta}
\left(\mathcal{M},\mathcal{E},\mathbf{c};\mathbf{q}\right),
\qquad \mathbf{q}\in\Omega.
$$

The design is encoded once, while any number of continuous query coordinates
$\mathbf{q}$ can subsequently be decoded. Module slots are padded to a common
maximum $M$ and accompanied by a mask $a_m\in\{0,1\}$. Consequently, inactive
slots contribute neither to hyperedge aggregation nor to local context.

Because module encoders are shared and module contributions are reduced by
weighted sums, the field prediction is invariant to a consistent permutation
of module slots. Module-indexed intermediate tensors are correspondingly
equivariant.

### 1.2 Tensor notation and primary data contract

| Symbol | Code tensor | Shape | Meaning |
|---|---|---:|---|
| $B$ | — | scalar | batch size |
| $M$ | — | scalar | padded module-slot count |
| $E$ | — | scalar | environment-token count |
| $K$ | — | scalar | latent hyperedge count |
| $Q$ | — | scalar | global query count |
| $H$ | — | scalar | hidden/token dimension |
| $F$ | — | scalar | output-field dimension |
| $\mathbf{X}$ | `module_centers` | $[B,M,2]$ | module centers |
| $\mathbf{S}$ | `module_features` | $[B,M,F_M]$ | module properties |
| $\mathbf{a}$ | `module_present` | $[B,M]$ | active-slot mask |
| $\mathbf{c}$ | `global_context` | $[B,F_G]$ | case-level properties |
| $\mathbf{Y}$ | `env_coords` | $[B,E,2]$ or $[E,2]$ | environment coordinates |
| $\mathbf{R}$ | `env_features` | $[B,E,F_E]$ | environment properties |
| $\mathbf{Q}$ | `query_xy` | $[B,Q,2]$ | coordinates to predict |
| $\widehat{\mathbf{U}}$ | `pred_field` | $[B,Q,F]$ | predicted field values |

These tensors are held by `honf_core.config.BatchData`. The organizer state is
case-dependent but query-independent; only the decoder scales with $Q$.

### 1.3 Fourier coordinate representation and input encoders

For a coordinate vector $\mathbf{x}\in\mathbb{R}^{d}$ and $J$ frequency
bands, the implemented Fourier map appends powers-of-two sinusoidal features:

$$
\gamma_J(\mathbf{x})
=
\left[
  \mathbf{x},
  \left\{
    \sin(2^j\pi\mathbf{x}),
    \cos(2^j\pi\mathbf{x})
  \right\}_{j=0}^{J-1}
\right].
$$

Coordinates are normalized by the domain lengths before encoding. With learned
MLPs $f_M$, $f_X$, $f_E$, and $f_G$, the initial states are

$$
\begin{aligned}
\mathbf{g}_b
&=f_G(\mathbf{c}_b)
&&\in\mathbb{R}^{H},\\
\mathbf{z}^{(0)}_{bm}
&=a_{bm}\left[
f_M(\mathbf{S}_{bm})
+f_X\!\left(\gamma_J(\bar{\mathbf{X}}_{bm})\right)
\right]
&&\in\mathbb{R}^{H},\\
\mathbf{e}_{be}
&=f_E\!\left(
  [\gamma_J(\bar{\mathbf{Y}}_{be}),\mathbf{R}_{be}]
\right)+\mathbf{g}_b
&&\in\mathbb{R}^{H}.
\end{aligned}
$$

The addition of $\mathbf{g}_b$ to environment tokens occurs in both requested
modes because both enable global context. The global token is also added later
as a distinct decoder context; these are separate learned pathways.

### 1.4 Learned hypergraph organizer

HONF introduces $K$ latent hyperedges. It does not require a predefined graph
or module adjacency matrix.

#### Module-to-environment context

When `use_A_me_auxiliary=true`, module/environment attention is

$$
\ell^{ME}_{bme}
=
\frac{
  (W_Q^M\mathbf{z}^{(0)}_{bm})^{\mathsf T}
  (W_K^E\mathbf{e}_{be})
}{\sqrt{H}},
$$

$$
A^{ME}_{bme}
=a_{bm}\operatorname{softmax}_{e}(\ell^{ME}_{bme}),
\qquad
\mathbf{c}^{ME}_{bm}
=\sum_e A^{ME}_{bme}\mathbf{e}_{be}.
$$

The state assigned to hyperedges is

$$
\widetilde{\mathbf{z}}_{bm}
=a_{bm}\left(
\mathbf{z}^{(0)}_{bm}
+0.25\,W_{ME}\mathbf{c}^{ME}_{bm}
\right).
$$

Thus $A^{ME}\in\mathbb{R}^{B\times M\times E}$ describes which environment
tokens provide context to each module. It is an auxiliary organizer relation,
not the final query-routing map.

#### Module-to-hyperedge incidence

For learned assignment,

$$
A^{MH}_{bmk}
=
a_{bm}\operatorname{softmax}_{k}
\left([W_{MH}\widetilde{\mathbf{z}}_{bm}]_k\right),
$$

so every active module distributes unit mass over $K$ hyperedges and inactive
rows are zero. Define the raw module mass and normalized source weights as

$$
s^M_{bk}=\sum_m A^{MH}_{bmk},
\qquad
\bar A^{MH}_{bmk}
=\frac{A^{MH}_{bmk}}{s^M_{bk}+\varepsilon}.
$$

The physical source centroid of hyperedge $k$ is

$$
\boldsymbol{\mu}^{S}_{bk}
=\sum_m \bar A^{MH}_{bmk}\mathbf{X}_{bm}.
$$

#### Environment-to-hyperedge incidence

Each environment token receives learned logits plus a distance bias to each
source centroid:

$$
\ell^{EH}_{bek}
=
[W_{EH}\mathbf{e}_{be}]_k
-\frac{\|\mathbf{Y}_{be}-\boldsymbol{\mu}^{S}_{bk}\|_2}{
0.25\sqrt{L_x^2+L_y^2}},
$$

$$
A^{EH}_{bek}=\operatorname{softmax}_{k}(\ell^{EH}_{bek}).
$$

The environment mass, normalized weights, and region centroid are

$$
s^E_{bk}=\sum_e A^{EH}_{bek},
\qquad
\bar A^{EH}_{bek}=\frac{A^{EH}_{bek}}{s^E_{bk}+\varepsilon},
\qquad
\boldsymbol{\mu}^{R}_{bk}
=\sum_e \bar A^{EH}_{bek}\mathbf{Y}_{be}.
$$

The organizer also reports normalized module/environment masses and

$$
\operatorname{strength}_{bk}
=
\sqrt{
  \frac{s^M_{bk}}{\sum_j s^M_{bj}}
  \frac{s^E_{bk}}{\sum_j s^E_{bj}}
  +\varepsilon
}.
$$

#### Hyperedge state

Module and environment messages are aggregated separately:

$$
\mathbf{h}^{M}_{bk}
=\sum_m\bar A^{MH}_{bmk}W_M\widetilde{\mathbf{z}}_{bm},
\qquad
\mathbf{h}^{E}_{bk}
=\sum_e\bar A^{EH}_{bek}W_E\mathbf{e}_{be}.
$$

The latent hyperedge state is

$$
\mathbf{h}_{bk}
=f_H\left(\mathbf{h}^{M}_{bk}+\mathbf{h}^{E}_{bk}\right)
\in\mathbb{R}^{H}.
$$

The code additionally builds generic source-region geometry and mass
descriptors. The maintained default has
`use_hyper_mechanism_encoder=false`, so these descriptors remain available for
diagnostics but do not alter $\mathbf{h}_{bk}$.

### 1.5 Query state and hyperedge routing

Let $\bar{\mathbf{q}}=(q_x/L_x,q_y/L_y)$. The query feature vector includes
normalized coordinates, a constant no-time encoding, Fourier features, and—in
this problem—normalized distances to the rectangular boundaries. A query MLP
produces

$$
\mathbf{d}_{bq}=f_Q(\phi_Q(\mathbf{q}_{bq}))\in\mathbb{R}^{H}.
$$

For each query/hyperedge pair, the decoder constructs ten relative-geometry
features $\boldsymbol{\psi}_{bqk}$ from offsets and distances to
$\boldsymbol{\mu}^{S}_{bk}$ and $\boldsymbol{\mu}^{R}_{bk}$. The routing logits
are

$$
\ell^{QH}_{bqk}
=
\frac{
  (W_Q\mathbf{d}_{bq})^{\mathsf T}(W_K\mathbf{h}_{bk})
}{\sqrt H}
+\beta\,w_G^{\mathsf T}\boldsymbol{\psi}_{bqk},
$$

and the default dense learned routing is

$$
\alpha_{bqk}
=
\operatorname{softmax}_{k}
\left(\ell^{QH}_{bqk}/\tau\right).
$$

The implementation also supports top-$k$ sparsification, but the maintained
default uses `hyper_attention_topk=0`, hence all $K$ hyperedges participate.
The hyperedge value context is

$$
\mathbf{c}^{H}_{bq}
=\sum_k\alpha_{bqk}W_V\mathbf{h}_{bk}.
$$

### 1.6 Base mode: `hyper_plus_global_near`

The global context is

$$
\mathbf{c}^{G}_{bq}=W_G\mathbf{g}_b.
$$

For local geometric context, the module weights are

$$
w^{N}_{bqm}
=
\frac{
  a_{bm}\exp\left(-\|\mathbf{q}_{bq}-\mathbf{X}_{bm}\|_2^2/(2r^2)\right)
}{
  \sum_j a_{bj}\exp\left(-\|\mathbf{q}_{bq}-\mathbf{X}_{bj}\|_2^2/(2r^2)\right)
  +\varepsilon
},
$$

where $r$ is `module_radius`. The near-module context is

$$
\mathbf{c}^{N}_{bq}
=W_N\sum_m w^N_{bqm}\mathbf{z}_{bm}.
$$

Therefore the base-mode decoder is

$$
\mathbf{c}^{\text{base}}_{bq}
=
\operatorname{LayerNorm}
\left(
\mathbf{c}^{H}_{bq}
+\mathbf{c}^{G}_{bq}
+\mathbf{c}^{N}_{bq}
\right),
$$

$$
\widehat{\mathbf{u}}_{bq}
=f_{\mathrm{out}}(\mathbf{c}^{\text{base}}_{bq})
\in\mathbb{R}^{F}.
$$

This mode combines a nonlocal latent mechanism ($\mathbf{c}^{H}$), a single
case-level state ($\mathbf{c}^{G}$), and explicit local geometry
($\mathbf{c}^{N}$).

### 1.7 Default mode: `enhanced_honf_pairwise`

The real default adds an explicit query-module interaction before reducing
through the same hypergraph.

For query $q$ and module $m$, define the six relative features

$$
\boldsymbol{\rho}_{bqm}
=
\left[
\frac{\Delta x}{L_x},
\frac{\Delta y}{L_y},
\frac{\sqrt{\Delta x^2+\Delta y^2}}{\sqrt{L_x^2+L_y^2}},
\frac{\max(\Delta x,0)}{L_x},
\frac{\max(-\Delta x,0)}{L_x},
\frac{|\Delta y|}{L_y}
\right],
$$

where $\Delta\mathbf{x}=\mathbf{q}_{bq}-\mathbf{X}_{bm}$. The pair embedding is

$$
\mathbf{p}_{bqm}
=
a_{bm}f_P\left(
  [\gamma(\boldsymbol{\rho}_{bqm}),a_{bm},
   \mathbf{z}_{bm},\mathbf{S}_{bm}]
\right)
\in\mathbb{R}^{H}.
$$

The module-to-hyperedge incidence first pools pair embeddings:

$$
\mathbf{r}_{bqk}
=\sum_m\bar A^{MH}_{bmk}\mathbf{p}_{bqm}.
$$

The query-to-hyperedge routing then selects among those edge-specific pair
responses:

$$
\mathbf{c}^{P}_{bq}
=
\sigma(\eta)\sum_k\alpha_{bqk}\mathbf{r}_{bqk},
$$

where $\eta$ is a learned scalar logit and the maintained configuration
initializes $\sigma(\eta)=0.1$. The default context and output are therefore

$$
\boxed{
\mathbf{c}^{\text{enh}}_{bq}
=
\operatorname{LayerNorm}
\left(
\mathbf{c}^{H}_{bq}
+\mathbf{c}^{P}_{bq}
+\mathbf{c}^{G}_{bq}
+\mathbf{c}^{N}_{bq}
\right)
},
$$

$$
\boxed{
\widehat{\mathbf{u}}_{bq}
=f_{\mathrm{out}}(\mathbf{c}^{\text{enh}}_{bq})
}.
$$

Compared with `hyper_plus_global_near`, the pairwise term preserves detailed
query-to-module geometry until after hyperedge aggregation. This is valuable
when the field at a point depends not only on a hyperedge-level summary, but on
the point's distinct relative position to each contributing module.

### 1.8 Intermediate and output structure

| Output key | Shape | Mathematical role |
|---|---:|---|
| `A_me` | $[B,M,E]$ | $A^{ME}$, module/environment attention |
| `module_env_context` | $[B,M,H]$ | $\mathbf{c}^{ME}$ |
| `A_mh` | $[B,M,K]$ | $A^{MH}$, module/hyperedge incidence |
| `A_eh` | $[B,E,K]$ | $A^{EH}$, environment/hyperedge incidence |
| `hyper_state` | $[B,K,H]$ | latent hyperedge state $\mathbf{h}$ |
| `hyper_source_coords` | $[B,K,2]$ | source centroids $\boldsymbol{\mu}^{S}$ |
| `hyper_region_coords` | $[B,K,2]$ | region centroids $\boldsymbol{\mu}^{R}$ |
| `hyper_module_mass`, `hyper_env_mass` | $[B,K]$ | normalized incidence masses |
| `hyper_strength` | $[B,K]$ | joint module/environment participation |
| `module_tokens` | $[B,M,H]$ | encoded or domain-refined module state |
| `env_tokens` | $[B,E,H]$ | encoded environment state |
| `global_token` | $[B,H]$ | encoded case-level state |
| `query_hyper_attention` | $[B,Q,K]$ | $\alpha$, returned only on request |
| `pairwise_edge_contribution` | $[B,Q,K]$ | edge-wise pair-context magnitude, optional |
| `pred_field` | $[B,Q,F]$ | continuous field prediction |

Static organizer tensors do not depend on the query set. Evaluation can retain
them in `PreparedChannelThermalCase` and decode a large grid in chunks without
re-encoding the design.

For the maintained training template and the current packed dataset, the
resolved dimensions are

| Quantity | Current value |
|---|---:|
| maximum module slots $M$ | 12 |
| environment tokens $E$ | $24\times 8=192$ |
| hyperedges $K$ | 6 |
| hidden dimension $H$ | 256 |
| global output channels $F$ | 5 |
| interface points $P$ | 64 |
| sampled global queries per case $Q$ | 1024 |
| local disk queries $Q_l$ | 3096 |

The domain and radius fields are resolved from the dataset rather than hard
coded by the template; the inspected cases use $L_x=12$, $L_y=6$, and
$r=0.45$.

## 2. ChannelThermal problem and datasets

### 2.1 Design domain and prediction targets

For a rectangular channel

$$
\Omega=[0,L_x]\times[0,L_y],
$$

let the $m$-th circular module occupy

$$
\Omega_m^s
=
\left\{
\mathbf{x}:\|\mathbf{x}-\mathbf{X}_m\|_2\le r_m
\right\},
$$

and let the fluid region be

$$
\Omega^f=\Omega\setminus\bigcup_{m=1}^{N_M}\Omega_m^s.
$$

A design/case is represented by

$$
\mathcal{D}
=
\left(
Re,U_{in},L_x,L_y,
\nu,\alpha_s,\alpha_f,k_s,k_f,r,
\{\mathbf{X}_m,Q_m\}_{m=1}^{N_M}
\right).
$$

The global prediction target is the five-channel steady field

$$
\mathbf{u}(\mathbf{x};\mathcal{D})
=
\left[
u(\mathbf{x}),v(\mathbf{x}),p(\mathbf{x}),
\omega(\mathbf{x}),T(\mathbf{x})
\right],
$$

with two-dimensional vorticity

$$
\omega=\frac{\partial v}{\partial x}-\frac{\partial u}{\partial y}.
$$

For each solid module, the model also predicts

$$
T_m^s(\boldsymbol{\xi}),
\qquad \boldsymbol{\xi}\in[-1,1]^2,
\quad \|\boldsymbol{\xi}\|_2\le 1,
$$

and the angular interface response

$$
\mathbf{b}_m(\theta)
=
\left[T_{m,\Gamma}(\theta),q_{m,n}(\theta)\right].
$$

### 2.2 Physical interpretation and fidelity of the supplied labels

A high-fidelity continuum reference for the five predicted global channels
would start from the steady incompressible flow equations. With
$\mathbf{v}_f=(u,v)$,

$$
\nabla\cdot\mathbf{v}_f=0,
\qquad
(\mathbf{v}_f\cdot\nabla)\mathbf{v}_f
=-\frac{1}{\rho}\nabla p+\nu\nabla^2\mathbf{v}_f
\qquad \text{in }\Omega^f,
$$

$$
\omega=\partial_xv-\partial_yu,
\qquad
Re=\frac{U_{in}L_{ref}}{\nu}.
$$

The thermal part represents conjugate advection/diffusion in the fluid and
diffusion with internal heat generation in each solid:

$$
\mathbf{v}_f\cdot\nabla T_f
=\nabla\cdot(\alpha_f\nabla T_f)
\qquad \text{in }\Omega^f,
$$

$$
0=\nabla\cdot(k_s\nabla T_m^s)+Q_m
\qquad \text{in }\Omega_m^s,
$$

with the ideal conjugate interface conditions

$$
T_f=T_m^s,
\qquad
q_{m,n}
=-k_s\nabla T_m^s\cdot\mathbf{n}_m
=-k_f\nabla T_f\cdot\mathbf{n}_m.
$$

The current local/global neural coupling represents that interface through the
reduced Robin relation

$$
q_{m,n}\approx h_m(\theta)
\left(T_{m,\Gamma}(\theta)-T_{m,env}(\theta)\right).
$$

The last expression is also the exact Robin anchor used by the current neural
coupling. The sign convention in code is positive outward flux when
$T_{surface}>T_{env}$.

It is important not to overstate the fidelity of the current dataset. The
packed case metadata explicitly records that:

- the velocity and pressure labels use a deterministic `analytic_wake`
  nonperiodic channel approximation, with no-slip module masks and wake
  deficits;
- the default generator does not perform a full Navier--Stokes solve, and its
  optional projection is a diagnostic divergence cleanup;
- fluid cells advect and diffuse temperature;
- solid cells diffuse temperature and receive internal heat generation;
- one shared temperature grid covers fluid and solid cells; and
- interface transfer is approximated by diffusion between neighboring grid
  cells.

Thus HONF-CL is trained to emulate this data-generating operator. The variables
$Re$, $\nu$, $U_{in}$, $p$, and $\omega$ have their usual fluid-mechanics
interpretation, but the present labels should be treated as a reduced-order
channel-flow benchmark rather than high-fidelity CFD ground truth.

The observed generator metadata uses cold inlet/walls ($T_{in}=T_{wall}=0$),
nonperiodic geometry, circular modules, and steady targets taken after thermal
convergence. The HDF5 root identifies the target as `converged_final` /
`steady_final_window`.

### 2.3 Packed global dataset

The current file is

```text
Data_Saved/Processed_ChannelThermal_Dataset/packed_dataset.h5
```

It contains 690 cases: 600 training and 90 test cases. The packed constants are
$M=12$ maximum module slots, $P=64$ interface points, a $64\times128$ global
grid, and a $64\times64$ local module grid.

At the root:

| HDF5 item | Shape | Meaning |
|---|---:|---|
| `case_ids`, `splits` | $[690]$ | case identity and train/test split |
| `channel_order` | $[5]$ | `u, v, p, omega, temperature` |
| `sampled_point_feature_names` | $[7]$ | `x, y` plus five field channels |
| `interface_condition_feature_names` | $[8]$ | interface input/diagnostic schema |
| `interface_target_names` | $[2]$ | `T_surface, q_normal` |
| `normalization/*` | per-channel vectors | global input/target mean and standard deviation |
| `cases/<case_id>` | group | arrays for one modular design |

Each case group contains:

| HDF5 item | Shape | Role |
|---|---:|---|
| `module_centers` | $[12,2]$ | padded module centers |
| `heat_powers` | $[12]$ | padded module heat inputs |
| `module_present` | $[12]$ | active-module mask |
| `material_parameters` | attributes | $Re,U_{in},\nu,\alpha_s,\alpha_f,k_s,k_f,r$ |
| `sampled_points` | $[4096,7]$ | `[x,y,u,v,p,omega,T]` training pool |
| `sampled_point_weights` | $[4096]$ | loss weights, stored separately for compatibility |
| `sampled_point_group` | $[4096]$ | uniform/near-module/boundary/gradient source label |
| `steady_field` | $[64,128,5]$ | full steady evaluation grid |
| `rms_field` | $[64,128,5]$ | RMS field retained by preprocessing |
| `x_grid`, `y_grid` | $[64,128]$ | global grid coordinates |
| `module_mask` | $[64,128]$ | solid occupancy on the global grid |
| `interface_condition` | $[12,64,8]$ | angular outside-flow/thermal conditions |
| `interface_condition_valid_mask` | $[12,64]$ | valid $h_{effective}$ supervision |
| `interface_response` | $[12,64,9]$ | full sampled interface diagnostic record |
| `interface_target` | $[12,64,2]$ | $T_{surface},q_n$ targets |
| `module_internal_temperature` | $[12,64,64]$ | solid temperature grids |
| `module_internal_mask` | $[64,64]$ | normalized disk mask |
| `structure_env_token_coords` | $[288,2]$ | solved-structure supervision grid |
| `env_module_influence_target` | $[288,12]$ | optional organizer target |
| `env_region_label` | $[288]$ | optional discrete environment-region label |
| `module_affinity_target` | $[12,12]$ | optional module-affinity target |
| `active_edge_count_target` | $[1]$ | optional active-hyperedge target |
| `selected_frame_ids`, `selected_times` | $[1]$, $[1]$ | source snapshot provenance |
| `steady_time` | $[1]$ | time associated with the steady target |
| `case_config_json` | scalar string | generator configuration and fidelity metadata |

The eight interface-condition channels are

$$
[\theta,n_x,n_y,T_{outside},u_n,u_t,h_{proxy},h_{effective}],
$$

while the teacher port tensor used by Stage A selects

$$
[\theta,n_x,n_y,T_{outside},h_{effective}].
$$

The packed file excludes module interiors from global sampled points. The
default training reader selects $Q=1024$ of the 4096 points without replacement
for each case and changes the deterministic subset with the epoch. Full grids
and organizer targets are loaded only when explicitly requested.

`GlobalChannelThermalDataset.__getitem__` exposes the following model-facing
sample before `DataLoader` adds the batch dimension:

| Dictionary entry | Shape |
|---|---:|
| `structure.module_centers` | $[M,2]$ |
| `structure.heat_powers` | $[M]$ |
| `structure.module_present` | $[M]$ |
| `structure.material_params` | $[6]$ |
| `structure.re`, `structure.u_in` | $[1]$, $[1]$ |
| `structure.domain_length_x/y` | $[1]$, $[1]$ |
| `query_xy` | $[Q,2]$ |
| `field_targets` | $[Q,5]$ |
| `point_weights`, `point_group` | $[Q]$, $[Q]$ |
| `interface_condition` | $[M,P,8]$ |
| `interface_target` | $[M,P,2]$ |
| `teacher_port_tokens` | $[M,P,5]$ |
| `local_module_params` | $[M,7]$ |
| `module_internal_query_points` | $[Q_l,2]$ |
| `module_internal_temperature_points` | $[M,Q_l]$ |

Here $Q_l=3096$ for the current $64\times64$ disk mask.

### 2.4 Packed Stage-A local dataset

The standalone local dataset is

```text
Data_Saved/Processed_LocalModule_Dataset/packed_dataset.h5
```

It contains 1034 single-module cases: 919 training and 115 test cases. Its
target kind is `steady_robin_conduction`.

| HDF5 item | Shape | Meaning |
|---|---:|---|
| `module_params` | $[1034,7]$ | module scalar descriptors |
| `port_tokens` | $[1034,64,5]$ | angular Robin conditions |
| `internal_query_points` | $[1034,3096,2]$ | points inside normalized disk |
| `internal_temperature_targets` | $[1034,3096]$ | solid temperatures |
| `interface_targets` | $[1034,64,2]$ | surface temperature and normal flux |
| `local_grid` | $[1034,64,64,2]$ | local coordinate grid |
| `local_mask` | $[1034,64,64]$ | disk mask |
| `local_target_stats` | $[1034,4]$ | $T$ mean/max/min/standard deviation diagnostics |
| `local_target_roughness` | $[1034,4]$ | interface roughness/high-frequency diagnostics |
| `normalization/*` | vectors | Stage-A input/target statistics |

The seven module parameters are

$$
[Q_m,k_s,\alpha_s,\mu_h,\sigma_h,\mu_{T_{env}},\sigma_{T_{env}}],
$$

and every port is

$$
[\theta,\cos\theta,\sin\theta,T_{env}(\theta),h(\theta)].
$$

`GlobalModuleAlignmentDataset` supplies a second Stage-A view by extracting
every active module from the global HDF5 and converting it to exactly this
schema. In mixed Stage-A training, one normalizer is fitted over all raw
training samples from both sources, then reused by validation and stored in the
checkpoint.

### 2.5 Normalization

For every supported tensor group, the reader applies component-wise z-score
normalization

$$
\widetilde{x}_j=\frac{x_j-\mu_j}{\max(\sigma_j,\varepsilon)}.
$$

The global HDF5 stores statistics for field channels, heat power, interface
conditions, interface targets, and internal temperature. Stage-A checkpoints
store statistics for module parameters, port tokens, interface targets, and
internal temperature. At coupled inference, Stage-A inputs are mapped into its
own normalized space; its outputs are returned to physical units and then, if
needed, mapped into the global target-normalization space. This prevents the
same physical temperature or flux from acquiring incompatible meanings across
the two stages.

## 3. Adaptation of HONF to ChannelThermal

### 3.1 Physical design to generic HONF features

`ChannelThermalInputAdapter` transforms the problem-specific design into the
generic $(\mathbf{X},\mathbf{S},\mathbf{a},\mathbf{c})$ contract.

Let $Q_m$ denote physical heat power. With the maintained
`normalize_inputs=true` setting, the value received by the global adapter is

$$
\widetilde Q_m=\frac{Q_m-\mu_Q}{\max(\sigma_Q,\varepsilon)}.
$$

If input normalization is disabled, simply take $\widetilde Q_m=Q_m$. For
each module, the ten global-HONF features are

$$
\mathbf{S}_m=
[\widetilde Q_m,|\widetilde Q_m|,
\widetilde Q_m/\widetilde Q_{max},
|\widetilde Q_m/\widetilde Q_{max}|,a_m,
\alpha_s,\alpha_f,k_s,k_f,r],
$$

where

$$
\widetilde Q_{max}=\max_{m:a_m=1}|\widetilde Q_m|.
$$

The 14-dimensional global vector is

$$
\mathbf{c}=
[Re,U_{in},N_M/M,\sum_m a_m\widetilde Q_m,
\operatorname{mean}_{a_m=1}\widetilde Q_m,
\widetilde Q_{max},L_x,L_y,
\nu,\alpha_s,\alpha_f,k_s,k_f,r].
$$

This scaling distinction is intentional in the existing data path:
`local_module_params` is assembled before global input normalization, so the
Stage-A descriptor below retains physical $Q_m$, while the generic global
HONF and port head use $\widetilde Q_m$.

The environment builder places a cell-centered $n_x\times n_y$ grid in the
channel. The maintained default uses $24\times8$, hence $E=192$. Each token has
seven features:

$$
[x/L_x,y/L_y,y/L_y,(L_y-y)/L_y,x/L_x,(L_x-x)/L_x,
1-|y-L_y/2|/(L_y/2)].
$$

These explicitly expose bottom/top wall, inlet/outlet, and centerline context
without embedding ChannelThermal semantics in the reusable core.

### 3.2 Stage-A single-module operator

For module $m$, let

$$
\mathbf{d}_m
=[Q_m,k_s,\alpha_s,\mu_h,\sigma_h,
\mu_{T_{env}},\sigma_{T_{env}}]
\in\mathbb{R}^{7},
$$

and let $\mathbf{p}_{mj}\in\mathbb{R}^{5}$ be its $j$-th port token. Stage A
encodes

$$
\mathbf{a}_m=f_d(\mathbf{d}_m),
\qquad
\mathbf{t}_{mj}=f_p(\mathbf{p}_{mj}).
$$

Learned latent queries $\boldsymbol{\lambda}_{ml}$ are conditioned on
$\mathbf{a}_m$ and repeatedly cross-attend to all port states:

$$
\boldsymbol{\lambda}^{(0)}_{ml}
=\boldsymbol{\lambda}^{learned}_{l}+W_a\mathbf{a}_m,
$$

$$
\boldsymbol{\lambda}^{(r+1)}_m
=
\operatorname{CrossAttentionBlock}
(\boldsymbol{\lambda}^{(r)}_m,\mathbf{t}_m).
$$

The response latent is

$$
\mathbf{z}^{loc}_m
=f_z\left(
[\operatorname{mean}_l\boldsymbol{\lambda}_{ml},\mathbf{a}_m]
\right).
$$

The continuous solid field and port response are decoded as

$$
\widehat T_m^s(\boldsymbol{\xi})
=f_T([\gamma(\boldsymbol{\xi}),\mathbf{z}^{loc}_m]),
$$

$$
[\widehat T_{m,\Gamma,j},\widehat q^{sur}_{m,n,j}]
=f_{\Gamma}([\mathbf{t}_{mj},\mathbf{z}^{loc}_m]).
$$

Only active padded modules are gathered and evaluated; results are scattered
back to $[B,M,\ldots]$ tensors with zero-filled inactive slots.

### 3.3 Predicted port conditions

The global model must operate autonomously without solved interface conditions.
For fixed angles $\theta_j=2\pi j/P$, `PortConditionHead` predicts

$$
[\widehat T^{env}_{bmj},\widehat h_{bmj}^{raw}]
=f_{port}
\left([
\mathbf{z}^{(0)}_{bm},\mathbf{c}^{ME}_{bm},\mathbf{g}_b,
\gamma(\theta_j,\cos\theta_j,\sin\theta_j),
\widetilde Q_{bm},a_{bm}
]\right),
$$

$$
\widehat h_{bmj}
=\operatorname{softplus}(\widehat h_{bmj}^{raw})+10^{-4}.
$$

Hence the autonomous port tensor is

$$
\widehat{\mathbf{p}}_{bmj}
=
[\theta_j,\cos\theta_j,\sin\theta_j,
\widehat T^{env}_{bmj},\widehat h_{bmj}].
$$

Training can use teacher, predicted, or convexly mixed ports. The maintained
configuration uses predicted ports directly, so the deployed forward path does
not require solved interface inputs.

When `local_module_params_from_used_ports=true`, the four statistical entries
of $\mathbf{d}_m$ are recomputed from the selected $T_{env}$ and $h$ ports
before Stage A is called.

### 3.4 Physics-anchored interface response

The default `corrected_physics` mode first forms the Robin flux

$$
q^{phys}_{bmj}
=
\widehat h_{bmj}
\left(
\widehat T_{bmj}^{surface}-\widehat T_{bmj}^{env}
\right).
$$

A residual head receives the global module state, local response latent,
$T_{surface}$, $T_{env}$, $\log(1+h)$, and Stage-A raw flux:

$$
\Delta q_{bmj}
=f_{\Delta q}
\left([
\mathbf{z}^{(0)}_{bm},\mathbf{z}^{loc}_{bm},
\widehat T^{surface}_{bmj},\widehat T^{env}_{bmj},
\log(1+\widehat h_{bmj}),\widehat q^{sur}_{bmj}
]\right).
$$

The final interface prediction is

$$
\widehat{\mathbf{b}}_{bmj}
=
\left[
\widehat T^{surface}_{bmj},
q^{phys}_{bmj}+\Delta q_{bmj}
\right].
$$

The last residual layer is initialized at zero, so the initial coupled model is
exactly anchored at the Robin expression and learns only a correction.

### 3.5 Local response fused into the global modular state

The local fields are reduced to six physical statistics:

$$
\mathbf{r}_m=
[\operatorname{mean}T_{surface},\max T_{surface},
\operatorname{mean}q_n,\max q_n,
\operatorname{mean}T_s,\max T_s].
$$

The global module token is updated by

$$
\mathbf{z}^{(1)}_{bm}
=a_{bm}\left[
\mathbf{z}^{(0)}_{bm}
+f_L([\mathbf{z}^{(0)}_{bm},\mathbf{z}^{loc}_{bm}])
+f_R(\mathbf{r}_{bm})
\right].
$$

The organizer is then recomputed with $\mathbf{z}^{(1)}$, which means the final
hyperedges depend on predicted solid/interface behavior rather than only on the
original geometric and heat descriptors.

### 3.6 One local/global interaction-refinement pass

The maintained configuration uses `interaction_refinement_steps=1`. After the
first Stage-A response is fused, the model builds a provisional organizer and
queries the global temperature just outside every module port:

$$
\mathbf{x}^{out}_{bmj}
=
\mathbf{X}_{bm}
+(r+\delta_r)
[\cos\theta_j,\sin\theta_j],
$$

where $\delta_r$ is `port_global_consistency_radius_offset` (default $0.05$).
The provisional HONF yields $\widehat T^G(\mathbf{x}^{out}_{bmj})$.

A zero-initialized residual head updates $T_{env}$ and $\log(1+h)$ using the
current port, provisional outside temperature, module state, angular features,
and local-response summary. Stage A is then evaluated once more, its response
is fused with the original base module token, and the final organizer and
global decoder are evaluated. This is a fixed one-pass learned coupling, not an
iterative convergence loop.

### 3.7 Complete default forward map

The real default computation can be summarized as

$$
\mathcal{D}
\xrightarrow{\text{physical adapter}}
(\mathbf{S},\mathbf{X},\mathbf{a},\mathbf{R},\mathbf{Y},\mathbf{c})
\xrightarrow{\text{base HONF organizer}}
(\mathbf{z}^{(0)},\mathbf{h}^{(0)},A^{(0)})
$$

$$
\xrightarrow{\text{port head}}
(\widehat T_{env},\widehat h)
\xrightarrow{\text{Stage A + Robin correction}}
(\widehat T_s,\widehat T_{surface},\widehat q_n,\mathbf{z}^{loc})
$$

$$
\xrightarrow{\text{response fusion + one refinement}}
\mathbf{z}^{(1)}
\xrightarrow{\text{final organizer}}
(\mathbf{h},A^{MH},A^{EH})
\xrightarrow{\text{enhanced pairwise decoder at }\mathbf{q}}
\widehat{[u,v,p,\omega,T]}(\mathbf{q}).
$$

The primary output dictionary is:

| Key | Shape | Meaning |
|---|---:|---|
| `pred_field` | $[B,Q,5]$ | global `[u,v,p,omega,T]` |
| `pred_internal_temperature` | $[B,M,Q_l,1]$ | solid temperature at local disk points |
| `pred_interface` | $[B,M,P,2]$ | corrected `[T_surface,q_normal]` |
| `pred_port_condition` | $[B,M,P,5]$ | final autonomous port conditions |
| `module_response_latent` | $[B,M,D_{loc}]$ | Stage-A response encoding |
| `organizer_aux` | dictionary | final incidences, centroids, masses, and tokens |
| `base_organizer_aux` | dictionary | pre-local-response organizer state |
| `routing_aux` | dictionary | query routing and pairwise diagnostics |

### 3.8 Coupled training objective

Let $w_q$ be the packed point weight and $\lambda_f$ the optional field-channel
weight. The global field term is

$$
\mathcal{L}_{field}
=
\frac{
\sum_{b,q,f}w_{bq}\lambda_f\,
(\widehat U_{bqf}-U_{bqf})^2
}{F\sum_{b,q}w_{bq}}.
$$

The coupled objective has the form

$$
\mathcal{L}
=
\lambda_{field}\mathcal{L}_{field}
+\lambda_{int}\mathcal{L}_{int}
+\lambda_{\Gamma}\mathcal{L}_{\Gamma}
+\lambda_{port}\mathcal{L}_{port}
+\lambda_{smooth}\mathcal{L}_{smooth}
+\lambda_{G\Gamma}\mathcal{L}_{G\Gamma}
+\lambda_{pred}\mathcal{L}_{pred}
+\mathcal{L}_{org}.
$$

For the maintained template:

$$
(\lambda_{field},\lambda_{int},\lambda_{\Gamma},
\lambda_{port},\lambda_{smooth},\lambda_{G\Gamma})
=(1.0,1.0,0.2,0.3,0.01,0.2).
$$

$\mathcal{L}_{port}$ supervises $T_{env}$ after division by a temperature
scale of 10 and supervises $h$ in $\log(1+h)$ space. The interface target
weights are $[1.0,0.25]$ for $[T_{surface},q_n]$. The predicted-port consistency
weight increases linearly to $0.05$ over 100 epochs. Generic organizer
anti-collapse regularization is present in code but disabled in the maintained
template, and solved-field structure targets are not required as inference
inputs.

## 4. Code-to-mathematics index

| Mathematical component | Main implementation |
|---|---|
| generic batch and mode definitions | `honf_core/config.py` |
| $\gamma$, input encoders, static encode/decode separation | `honf_core/model.py` |
| $A^{ME}$, $A^{MH}$, $A^{EH}$, centroids, masses, $\mathbf{h}$ | `honf_core/organizer.py` |
| $\alpha$, $\mathbf{c}^H$, $\mathbf{c}^G$, $\mathbf{c}^N$, pairwise kernel, output field | `honf_core/decoder.py` |
| physical module/global feature definitions | `channelthermal/input_adapter.py` |
| channel environment coordinates and seven boundary features | `channelthermal/environment.py` |
| Stage-A cross-attention and local neural fields | `local_surrogate/model.py` |
| port head, normalization bridge, Robin correction, refinement, response fusion | `channelthermal/local_coupling.py` |
| full two-stage execution and final output dictionary | `channelthermal/model.py` |
| HDF5 schemas, splits, point sampling, local/global alignment, normalization | `data/datasets.py` |
| weighted global-field MSE | `training_tools/losses.py::weighted_field_mse` |
| coupled local/interface losses and curriculum application | `train.py` |

The reusable HONF therefore defines a continuous set-to-field operator for
arbitrary modular designs, while the ChannelThermal layer supplies the exact
physical features, local solid surrogate, interface variables, and coupling
logic needed for this particular multi-field prediction problem.
