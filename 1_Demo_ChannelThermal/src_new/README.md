# ChannelThermal NewHONF

`src_new` is a standalone, supervised neural surrogate for the ChannelThermal
demo. It predicts a five-channel global field around heated circular modules,
predicts each module's internal temperature and fluid/solid interface response,
and exposes the learned hypergraph organization used to couple modules to
thermal regions. It does not import the old `src/` model or
`Unified_Forward_Model_0` at runtime.

The implementation has two training stages:

1. **Stage A — local module surrogate:** learn steady conduction inside one
   circular solid module from module properties and angular Robin boundary
   conditions.
2. **Stage B — global NewHONF model:** learn the channel field and the local
   boundary conditions, optionally using a frozen Stage-A model to couple the
   global flow/temperature prediction to module-scale conduction.

The reusable HONF code is kept in `_models_core`; ChannelThermal geometry,
thermal features, local coupling, and plotting remain outside that core.

## 1. Model goal and physics problem

### 1.1 What is predicted

For each channel case, the global model consumes Reynolds number, inlet
velocity, module locations, heat powers, active-module masks, material
properties, and arbitrary query coordinates. It predicts

```text
[u, v, p, omega, temperature]
```

at every query coordinate. With the Stage-A local surrogate attached, it also
predicts, for every active module:

- internal solid temperature at local disk points;
- interface surface temperature `T_surface(theta)`;
- outward normal heat flux `q_normal(theta)`;
- the local port condition `[theta, cos(theta), sin(theta), T_env, h_effective]`;
- a compact module-response latent used by the global hypergraph organizer.

### 1.2 Physical interpretation

The packed global data represents two-dimensional, nonperiodic channel flow
with heated circular solid modules:

- the fluid field contains velocity, pressure, vorticity, and temperature;
- fluid temperature is governed by advection and diffusion;
- solid temperature is governed by diffusion and internal heat generation;
- the fluid/solid boundary exchanges heat through surface temperature and
  normal flux.

The Stage-A local problem is steady conduction in a normalized disk:

```text
-(alpha_s / R^2) Laplacian_xi(T) = q
-(alpha_s / R^2) dT/dn_xi = h(theta) [T_surface - T_env(theta)]
```

In the default `corrected_physics` coupling mode, the interface flux is anchored
to the Robin relation and receives a learned, zero-initialized correction:

```text
q_physics = h_effective * (T_surface - T_env)
q_normal  = q_physics + delta_q
```

This directory does **not** solve these PDEs during training or inference and
does not add PDE-residual losses. It learns a surrogate from packed simulator
outputs and uses physical feature construction, the Robin flux relation, and
an optional global/local consistency probe as inductive biases. The upstream
global data generator is a lightweight channel approximation, not a
high-fidelity Navier–Stokes CFD solver.

### 1.3 Why the hypergraph is used

Modules can interact through shared downstream thermal regions, near-wall
regions, and common inlet/outlet context. HONF learns `H` latent hyperedges
that organize active modules and environment tokens before decoding a field.
For the enhanced decoder, the principal field path is

```text
U(q) = f_out[c_H(q) + g_pair c_pair(q) + c_global(q) + c_near(q)]
```

where `c_H` is hyperedge value context, `c_pair` is query/module detail routed
through the same hyperedges, `c_global` is case context, and `c_near` is a
distance-weighted local-module context. The unrestricted direct module/env
decoder is disabled in the supplied enhanced-HONF profiles.

## 2. Directory and code structure

Paths below are relative to `1_Demo_ChannelThermal/src_new`.

```text
src_new/
├── train.py                         Stage-B/global trainer
├── train_local.py                   Stage-A/local trainer
├── evaluate.py                      one-case global evaluation and exports
├── evaluate_local.py                one-case local evaluation
├── compare_models.py                multi-checkpoint quantitative comparison
├── _bootstrap_imports.py            direct-script import-path bootstrap
├── _data/
│   ├── channelthermal_datasets.py   global and local packed-HDF5 readers
│   └── local_module_datasets.py     same readers; local-focused module docstring
├── _models_core/
│   ├── honf_types.py                generic configs and BatchData
│   ├── honf_core.py                 input encoders and core composition
│   ├── honf_organizer.py            A_me/A_mh/A_eh and hyperedge state
│   └── honf_decoder.py              query routing and field decoder
├── _models_channelthermal/
│   ├── channelthermal_config.py     combined core/domain configuration
│   ├── channelthermal_input_adapter.py
│   ├── channelthermal_environment.py
│   ├── channelthermal_full_model.py global/local orchestration wrapper
│   ├── local_coupling.py            port, surrogate, flux, and refinement logic
│   └── internal_fallback_heads.py   global-only internal/interface heads
├── _models_local/
│   └── model_local.py               checkpoint-compatible Stage-A model
├── _helpers/
│   ├── model_utils.py               paths, JSON, normalization, AMP, MLPs
│   ├── training_losses.py           reusable weighted field MSE
│   ├── checkpointing.py             compact checkpoint wrappers
│   ├── honf_diagnostics.py          scalar health metrics and regularization
│   ├── hypergraph_plan.py           canonical inverse-ready plan I/O
│   ├── evaluation_plots.py          global metrics and quicklooks
│   ├── local_module_plots.py        local disk/interface plots
│   ├── organizer_viz_channelthermal.py
│   └── routing_viz_channelthermal.py
├── _tests/                          smoke and regression scripts
├── MIGRATION_MANIFEST.md            implementation provenance
└── README.md                         this guide
```

Adjacent directories used by these scripts are:

```text
1_Demo_ChannelThermal/
├── Configs_new/                     training configurations
├── Data_Saved/                      packed HDF5 inputs
├── Saved_Model_LocalModule*/        Stage-A runs
└── Saved_Model_NewHONF/             Stage-B runs and evaluations
```

`_bootstrap_imports.py` adds the demo directory and `src_new` to `sys.path`, so
the entrypoints are intended to be launched as direct scripts. Package
`__init__.py` files only expose the main types and mark package boundaries.

## 3. Setup and required data

Run commands from the demo directory unless stated otherwise:

```bash
cd 1_Demo_ChannelThermal
```

The existing project examples use the `ModularDT` Conda environment:

```bash
conda run -n ModularDT python -m compileall -q src_new
```

The runtime imports PyTorch, NumPy, h5py, Matplotlib, and tqdm. CUDA is
optional; pass `--device cpu`, `--device cuda`, or `--device cuda:N`.

The trainers do not generate or preprocess data. By default they expect:

```text
Data_Saved/Processed_LocalModule_Dataset/packed_dataset.h5
Data_Saved/Processed_ChannelThermal_Dataset/packed_dataset.h5
```

Relative data, checkpoint, and output paths in JSON files are resolved against
`1_Demo_ChannelThermal`, not against `src_new`.

Important configuration-path note: `train.py` and two smoke scripts currently
default to `Configs_new/train_global_honf_template.json`, but that file is
stored at `Configs_new/_tests/train_global_honf_template.json`. Always pass an
explicit existing `--config`, as the examples below do.

## 4. Configure the models

### 4.1 Stage-A local configuration

Start from `Configs_new/train_local_module_template.json`. The active sections
are:

| Section | Important settings |
| --- | --- |
| `Run_ID` | Numeric run serial; normalized to four digits. |
| `dataset` | `source`, HDF5 paths, train/validation splits, batch sizes, workers, and input/target normalization. |
| `model` | Dimensions, hidden/latent size, learned latent count, attention heads/layers, Fourier frequencies, and dropout. Dimension fields may be `"auto"`. |
| `loss` | Internal/interface MSE weights, optional per-interface-channel weights, and cyclic interface smoothness. |
| `training` | Epochs, AdamW learning rate/weight decay, gradient clipping, AMP, seed, device, and optional initialization checkpoint. |
| `paths.saved_model_dir` | Stage-A output root. |

`dataset.source` controls the sample source:

- `local`: synthetic/standalone local packed cases;
- `global_alignment`: each active module in the packed global dataset becomes a
  local sample;
- `mixed`: concatenates the two datasets.

For training from scratch, set `training.init_checkpoint_path` to `null`.
For alignment/fine-tuning, point it to a compatible Stage-A checkpoint or pass
`--init-checkpoint`.

The template contains `local_synthetic_weight` and
`global_alignment_weight`, but the current `MixedLocalDataset` performs plain
concatenation and does not read those weights.

### 4.2 Stage-B global configuration

Use one of the existing root profiles:

- `Configs_new/train_global_honf_validated_core.json`: enhanced HONF with the
  Stage-A coupling and corrected physics;
- `Configs_new/train_global_honf_old_parity.json`: settings intended for an
  old/new comparison;
- `Configs_new/train_global_honf_validated_core_global_only_200ep.json`:
  global-only fallback heads, no attached local surrogate;
- `Configs_new/_tests/train_global_honf_template.json`: smaller reference and
  smoke-test profile.

The global JSON schema is:

| Section | What it controls |
| --- | --- |
| `model.core_honf` | Field/domain sizes, token grid, hidden size, number of hyperedges, organizer assignment modes, query routing, Fourier features, mechanism encoder, pairwise kernel, and decoder contexts. Dataset-dependent sizes can be `"auto"`. |
| `model.channelthermal` | Material dimension, internal prediction selection, and global fallback-head sizes. |
| `model.local_coupling` | Whether to attach Stage A, its checkpoint, freeze mode, local latent dimension, port-derived parameter refresh, and default angular resolution. |
| `model.physical_correction` | Flux mode, blend factor, zero/one interaction-refinement pass, and global-temperature probe radius/count. |
| `dataset` | Packed HDF5 path, splits, sampled field points, batches, workers, normalization, and random training-point sampling. |
| `loss` | Global field/channel weights; internal, interface, port, smoothness, global-port, predicted-autonomous, and optional generic organizer losses. |
| `training` | Run ID/name, seed, device, epochs, AdamW settings, AMP, clipping, optional batch caps, and port curriculum. |
| `paths.saved_model_dir` | Stage-B output root. |

Authoritative setting precedence is:

```text
model.local_coupling      overrides duplicate local keys in model.channelthermal
model.physical_correction overrides duplicate physics keys in model.channelthermal
CLI --device/--epochs/--Run_ID and batch-limit flags override training JSON
```

Before Stage B with local coupling, update
`model.local_coupling.local_surrogate_checkpoint_path` to a real Stage-A
checkpoint. Also make sure `local_surrogate_latent_dim` matches that Stage-A
model's `latent_dim`. Saved Stage-B checkpoints embed the local architecture,
normalization statistics, and weights, so later evaluation is self-contained.

Useful behavior switches:

- `internal_prediction_mode="auto"`: use the attached local surrogate, else the
  global fallback heads;
- `"local_surrogate"`: require an attached Stage-A model;
- `"global_head"`: bypass local outputs and use learned fallback heads;
- `local_surrogate_flux_mode`: `surrogate`, `physics_from_port`,
  `corrected_physics`, or `blend`;
- `interaction_refinement_steps`: only `0` or `1`;
- `hyper_module_assignment_mode` and `hyper_query_attention_mode`: `learned`
  or `uniform` for ablation studies;
- `hyper_attention_topk=0`: dense query-to-H attention; a positive value keeps
  only the query-local top-k hyperedges;
- `organizer_regularization.enabled=false`: the supplied default; enable only
  for explicit generic anti-collapse experiments.

The active port curriculum is `training.port_curriculum`:

- `schedule="none"` uses its fixed `mode` (`teacher`, `mixed`, or `predicted`);
- `schedule="teacher_to_predicted"` uses `teacher_epochs`,
  `predicted_after_epoch`, and the mixed-ratio endpoints.

In teacher mode, internal/interface loss weights are zero; in mixed mode they
are scaled by `1 - mixed_teacher_ratio`; in predicted mode they use their full
configured values. Validation always includes a separate predicted-port pass.

Current implementation details worth knowing:

- `checkpointing` flags in JSON are informational; `train.py` currently writes
  all best/latest checkpoint variants unconditionally.
- `dataset.require_converged` is not passed into the dataset reader and does not
  filter cases in this stack.
- `model.channelthermal.heat_scale` and `field_names` are not used in the
  forward computation; field order comes from the HDF5/checkpoint.
- `random_point_sampling=true` uses a fresh NumPy generator for training-point
  selection, so `training.seed` alone does not make sampled points fully
  deterministic. Validation sampling is deterministic because it is disabled.

Every Stage-B run writes a concrete `config_resolved.json`; inspect it rather
than the source template when reproducing a run.

## 5. Train the models

### 5.1 Stage A: train or align the local surrogate

```bash
conda run -n ModularDT python src_new/train_local.py \
  --config Configs_new/train_local_module_template.json \
  --Run_ID 0007 \
  --run-name local_aligned \
  --device cuda:0
```

For a quick CPU smoke train:

```bash
conda run -n ModularDT python src_new/train_local.py \
  --config Configs_new/train_local_module_template.json \
  --Run_ID 9001 \
  --device cpu \
  --epochs 1 \
  --max-train-batches 1 \
  --max-val-batches 1
```

Stage-A outputs are placed under
`paths.saved_model_dir/Run_<ID>[_suffix]_<timestamp>/`:

```text
best_model.pt
latest_model.pt
resolved_train_config.json
loss_history.csv
loss_curve.png
```

The checkpoint contains the model config, state dict, feature names, and local
input/target normalization metadata.

### 5.2 Stage B: train the global NewHONF model

After setting the local checkpoint path in the selected config:

```bash
conda run -n ModularDT python src_new/train.py \
  --config Configs_new/train_global_honf_validated_core.json \
  --Run_ID 0008 \
  --run-name predicted_ports \
  --device cuda:0
```

Global-only example:

```bash
conda run -n ModularDT python src_new/train.py \
  --config Configs_new/train_global_honf_validated_core_global_only_200ep.json \
  --Run_ID 0009 \
  --device cuda:0
```

CPU smoke train with the smaller test profile:

```bash
conda run -n ModularDT python src_new/train.py \
  --config Configs_new/_tests/train_global_honf_template.json \
  --Run_ID 9002 \
  --device cpu \
  --epochs 1 \
  --max-train-batches 1 \
  --max-val-batches 1
```

Resume in the same run directory by supplying its checkpoint. `--epochs` is
the final epoch number, not a number of extra epochs:

```bash
conda run -n ModularDT python src_new/train.py \
  --config Saved_Model_NewHONF/Run_0008_YYYYMMDD_HHMMSS_predicted_ports/config_resolved.json \
  --resume-checkpoint Saved_Model_NewHONF/Run_0008_YYYYMMDD_HHMMSS_predicted_ports/latest_model.pt \
  --epochs 1000 \
  --device cuda:0
```

Stage-B outputs include:

```text
best_model.pt                     lowest validation total loss
best_by_field_mse_model.pt        lowest validation field MSE
best_by_temperature_mse_model.pt  lowest validation temperature MSE
best_predicted_model.pt           lowest autonomous predicted-port validation loss
latest_model.pt
config_resolved.json
metrics.csv
loss_curve.png
diagnostic_plots/*.png
summary.json
```

The total training loss can include weighted global field MSE, local internal
temperature loss, interface loss, supervised port loss, cyclic port
smoothness, global/port temperature consistency, predicted-port consistency,
and optional organizer regularization. `metrics.csv` also records assignment
entropy, active-edge counts, mass concentration, query-routing entropy,
pairwise gate/context norms, and decoder feature flags.

## 6. Post-processing and evaluation

### 6.1 Evaluate one Stage-A case

Use a direct checkpoint path, or use `--Run_ID` plus the correct saved root.
The supplied local template saves to `Saved_Model_LocalModule_NewHONF`, while
the evaluator's default root is `Saved_Model_LocalModule`, so pass the root
explicitly when necessary.

```bash
conda run -n ModularDT python src_new/evaluate_local.py \
  --Run_ID 0007 \
  --checkpoint best \
  --saved-root ./Saved_Model_LocalModule_NewHONF \
  --split test \
  --case-index 0 \
  --device cpu
```

It writes `internal_temperature_comparison.png`,
`interface_curve_comparison.png`, and `evaluation_summary.json` below an
`eval_local/<case>_<timestamp>/` directory next to the checkpoint, unless
`--output-dir` is supplied.

### 6.2 Evaluate one Stage-B case

Named checkpoint selectors require `--Run_ID`:

```bash
conda run -n ModularDT python src_new/evaluate.py \
  --Run_ID 0008 \
  --checkpoint best_predicted \
  --saved-root ./Saved_Model_NewHONF \
  --split test \
  --case-index 0 \
  --device cpu \
  --local-port-condition-mode predicted \
  --temperature-display-mode composite_internal \
  --organization-view all
```

Alternatively, pass a direct `.pt` path without `--Run_ID`. Available named
selectors are `best`, `best_by_field_mse`, `best_by_temperature_mse`,
`best_predicted`, and `latest`. A case can be selected with `--case-id` or
`--case-index`.

`--local-port-condition-mode` controls the local boundary input during
evaluation:

- `predicted`: autonomous inference;
- `teacher`: dataset boundary conditions;
- `mixed`: linear teacher/predicted blend;
- `both`: produce predicted and teacher outputs; predicted mode remains the
  primary source for organization, routing, and plan exports.

Standard outputs are written to
`<run>/eval_global/<case>_<timestamp>/` unless `--output-dir` is provided:

- `global_field_quicklook_<mode>.png` with GT, prediction, and absolute error;
- `module_internal_temperature_<mode>.png` and
  `interface_curves_<mode>.png` when those tensors are nonempty;
- `evaluation_outputs_<mode>.npz` and `metrics_<mode>.csv`;
- `organization_overview.png`, `organization_summary_matrices.png`, and
  `organization_schematic.png`, depending on `--organization-view`;
- `hypergraph_diagnostics.json`, `summary.json`, and `summary_compact.json`.

`--temperature-display-mode fluid_only` masks module interiors in the global
temperature plot. `composite_internal` inserts the local-surrogate internal
temperature into each module disk for visualization; field metrics remain
fluid-mask aware.

### 6.3 Export query routing and a static hypergraph plan

```bash
conda run -n ModularDT python src_new/evaluate.py \
  --Run_ID 0008 \
  --checkpoint best_predicted \
  --saved-root ./Saved_Model_NewHONF \
  --case-index 0 \
  --device cpu \
  --query-batch-size 4096 \
  --local-port-condition-mode predicted \
  --return-routing-maps \
  --routing-view all \
  --export-hypergraph-plan
```

Routing export adds:

```text
routing_maps.npz
routing_summary.json
routing_dominant_edge.png
routing_context_norms.png
routing_attention_maps.png                 with --routing-view all
routing_pairwise_contribution_maps.png     with --routing-view all
```

Dense routing arrays scale as `[Q,H]` per case and are intentionally disabled
during normal training/evaluation. Reduce `--query-batch-size` if field
decoding is memory-bound; note that the pairwise kernel internally also forms
query/module features.

Plan export adds `hypergraph_plan.npz` and
`hypergraph_plan_summary.json`. The plan stores canonicalized static arrays
(`A_mh`, `A_eh`, source/region coordinates, masses, strengths, active mask,
environment coordinates, module mask, and edge provenance). It intentionally
excludes learned `hyper_state`, module tokens, and query-dependent `alpha_qk`.
Load and validate it with
`_helpers.hypergraph_plan.load_hypergraph_plan()` and
`validate_hypergraph_plan()`.

### 6.4 Compare multiple Stage-B checkpoints

`compare_models.py` evaluates the same sampled fraction of a split for every
model and produces reconstruction, local-response, and hypergraph tables and
figures:

```bash
conda run -n ModularDT python src_new/compare_models.py \
  --Run_ID 0004 \
  --Run_ID 0005 \
  --Run_ID 0008 \
  --label NoPretrained \
  --label UniformH \
  --label ValidatedCore \
  --checkpoint-selector best_predicted \
  --case-ratio 1.0 \
  --local-port-condition-mode predicted \
  --device cuda:0
```

`--checkpoint-path` may be repeated instead of `--Run_ID`. Optional
`--return-routing-maps` adds routing summaries, and `--save-debug-npz` stores
case/model arrays. The default output is
`Saved_Model_NewHONF/CompareModels/Run_<timestamp>/` with `logs/`, `tables/`,
`figures/`, and optional `debug_npz/` subdirectories.

## 7. Detailed tensor flow

### 7.1 Symbols and dataset tensors

| Symbol | Meaning |
| --- | --- |
| `B` | batch size |
| `M` | padded module slots |
| `E` | environment tokens, `num_env_tokens_x * num_env_tokens_y` |
| `H` | hyperedges |
| `Q` | global field query points |
| `Qi` | local disk query points |
| `Ntheta` | angular interface points |
| `D` | HONF hidden dimension |
| `L` | Stage-A response latent dimension |

`GlobalChannelThermalDataset` returns the following primary tensors after
DataLoader batching:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `structure.re`, `structure.u_in` | `[B,1]` | Reynolds number and inlet speed |
| `module_centers` | `[B,M,2]` | global `(x,y)` centers |
| `heat_powers` | `[B,M]` | dataset-scaled module heating |
| `module_present` | `[B,M]` | active-slot mask |
| `material_params` | `[B,6]` | `[nu, alpha_s, alpha_f, k_s, k_f, R]` |
| `query_xy` | `[B,Q,2]` | global query coordinates |
| `field_targets` | `[B,Q,5]` | `[u,v,p,omega,T]` |
| `interface_condition` | `[B,M,Ntheta,8]` normally | `[theta,nx,ny,T_out,u_n,u_t,h_proxy,h_effective]` |
| `teacher_port_tokens` | `[B,M,Ntheta,5]` | `[theta,nx,ny,T_out,h_effective]` |
| `interface_target` | `[B,M,Ntheta,2]` | `[T_surface,q_normal]` |
| `local_module_params` | `[B,M,7]` | heat, solid properties, and port summary statistics |
| `module_internal_query_points` | `[B,Qi,2]` | shared normalized disk queries |
| `module_internal_temperature_points` | `[B,M,Qi]` | local temperature targets |

Older packed files without `h_effective` copy `h_proxy` into that column and
warn. Missing interface-validity and learned-structure targets also receive
documented fallbacks in the dataset reader.

### 7.2 ChannelThermal input and environment adapters

`ChannelThermalInputAdapter` maps physical inputs to generic HONF inputs:

```text
heat/material/mask -> module_features [B,M,10]
case scalars        -> global_context [B,14]
```

The ten module feature columns are dataset-scaled heat, absolute heat,
case-relative heat, absolute case-relative heat, active flag, solid/fluid
diffusivity, solid/fluid conductivity, and radius. Coordinate features are not
duplicated here; the core encodes module positions separately.

`ChannelThermalEnvironmentBuilder` creates a cell-centered token grid:

```text
env_coords   [B,E,2]
env_features [B,E,7]
```

The seven environment features are normalized x/y, bottom/top wall distance,
inlet/outlet distance, and centerline proximity.

### 7.3 Core encoding and static organization

`HONFNeuralField.encode_and_organize()` performs:

```text
global_context [B,14] -> global_encoder -> global_token [B,D]
module_features + Fourier(position)     -> module_tokens [B,M,D]
env_features + Fourier(position) + global_token -> env_tokens [B,E,D]
```

`HypergraphOrganizerCore` then computes:

1. optional module-to-environment attention `A_me [B,M,E]` and an environment
   context added weakly to each module token;
2. module-to-hyperedge assignment `A_mh [B,M,H]`, normalized across `H` for
   every active module;
3. each hyperedge source coordinate as the `A_mh`-weighted module-center mean;
4. environment-to-hyperedge assignment `A_eh [B,E,H]`, with a geometric bias
   toward each source coordinate;
5. each hyperedge region coordinate as the `A_eh`-weighted environment mean;
6. normalized module/environment masses and
   `hyper_strength = sqrt(module_mass * env_mass)`;
7. source/region geometry and mass descriptors;
8. assigned module and environment summaries mixed into
   `hyper_state [B,H,D]`.

Static organization is case-dependent but query-independent. It is therefore
computed once per model pass, except that the ChannelThermal wrapper recomputes
it after local-response fusion.

### 7.4 Port prediction and Stage-A local flow

The port head combines base module state, `A_me` environment context, heat,
global state, and fixed angular features. It predicts two values per angle:

```text
T_env, softplus(h_effective) -> port_tokens [B,M,Ntheta,5]
```

Teacher, predicted, or blended tokens are selected. When configured,
`local_module_params_for_ports()` refreshes the four `h`/`T_env` mean and
standard-deviation columns from the selected ports.

`call_local_surrogate()` flattens active slots into the Stage-A batch:

```text
module_params [B,M,7]       -> [B*M,7]
port_tokens [B,M,Ntheta,5]  -> [B*M,Ntheta,5]
local queries               -> [B*M,Qi,2]
```

Inside `LocalModuleSurrogate`:

```text
module parameters -> param_state [B*M,D]
port tokens        -> port_state  [B*M,Ntheta,D]
learned latent queries + param_state
                  -> repeated cross-attention over port_state
                  -> pooled module response z_module [B*M,L]

[Fourier(local_xy), z_module] -> internal T [B*M,Qi,1]
[port_state, z_module]         -> interface [B*M,Ntheta,2]
```

The coupling reshapes these back to `[B,M,...]`, handles local/global
normalization spaces, selects/corrects `q_normal`, summarizes mean/max internal
and interface values into six features, and fuses both the response latent and
summary into the base module token.

With one interaction-refinement step, the wrapper provisionally reorganizes
the fused module state, decodes global temperature just outside every module
angle, predicts residual updates to `T_env` and log-`h`, reruns Stage A, and
fuses the final response. A frozen Stage-A model remains in `eval()` even while
the coupling heads train.

### 7.5 Query-dependent decoder flow

After final module-state fusion, the wrapper recomputes the organizer and calls
`HypergraphFieldDecoder` once for the requested global field:

```text
query_xy [B,Q,2]
  -> normalized coordinates + Fourier/boundary features
  -> query_state [B,Q,D]

dot(query_state, hyper_state) + source/region geometry bias
  -> logits [B,Q,H]
  -> dense or top-k softmax
  -> alpha_qk [B,Q,H]

alpha_qk * hyperedge values
  -> c_H [B,Q,D]
```

The pairwise path forms normalized relative query/module geometry and optional
module token/raw features:

```text
pair features       [B,Q,M,*]
  -> pair MLP       [B,Q,M,D]
  -> A_mh aggregate [B,Q,H,D]
  -> alpha_qk sum   [B,Q,D]
  -> sigmoid gate   c_pair [B,Q,D]
```

The decoder adds the enabled global, near-module, and optional direct contexts,
normalizes the combined context, and maps it to `pred_field [B,Q,5]`. Normal
training retains scalar routing diagnostics only. `return_routing_maps=True`
additionally returns `alpha_qk`, dominant edge, entropy, per-edge pairwise
contribution magnitude, and `c_H`/`c_pair` norm maps.

### 7.6 Final wrapper outputs

`ChannelThermalHONFModel.forward()` returns:

| Key | Shape/type | Meaning |
| --- | --- | --- |
| `pred_field` | `[B,Q,5]` | global field |
| `pred_internal_temperature` | `[B,M,Qi,1]` | Stage-A or fallback internal T |
| `pred_interface` | `[B,M,Ntheta,2]` | `T_surface`, selected `q_normal` |
| `pred_port_condition` | `[B,M,Ntheta,5]` | final predicted ports |
| `local_port_condition_used` | same | teacher/mixed/predicted ports actually sent to Stage A |
| `module_response_latent` | `[B,M,L]` or `[B,M,D]` | local response or fallback state |
| `organizer_aux` | dictionary | final static organization |
| `base_organizer_aux` | dictionary | organization before local fusion |
| `routing_aux` | dictionary | scalar diagnostics and optional dense maps |
| `pred_port_global_*` | `[B,M,Nprobe]` | optional global/port consistency probe |

Fallback mode uses Fourier-encoded local disk coordinates and angular tokens
with per-module MLP heads. It preserves the output contract but does not use
the pretrained local conduction surrogate.

## 8. Helper-function reference

### 8.1 `_helpers/model_utils.py`

- `ensure_dir`, `resolve_demo_path`: create directories and resolve paths
  relative to the demo root.
- `read_json`, `write_json`: JSON configuration/summary I/O.
- `load_trusted_checkpoint`: loads local PyTorch checkpoints with compatibility
  for the PyTorch `weights_only` default change.
- `current_timestamp`, `set_seed`, `select_device`: run naming, RNG seeding, and
  CPU/CUDA choice.
- `dataclass_from_dict`, `dataclass_to_dict`, `deep_update`: tolerant config
  conversion and recursive dictionary override.
- `decode_string_array`: decodes byte/string arrays from HDF5.
- `recursive_to_device`: moves arbitrarily nested tensors to a device.
- `count_parameters`: counts trainable initialized parameters and tolerates
  lazy layers.
- `safe_std_np`, `safe_std_torch`: replace near-zero standard deviations with
  one before normalization.
- `masked_mean`, `masked_mse`, `masked_softmax`: mask-aware tensor reductions.
- `make_grad_scaler`, `autocast_context`: version-compatible CUDA AMP helpers.
- `strip_module_prefix`: removes a DataParallel `module.` checkpoint prefix.
- `FourierEncoder`: appends power-of-two sine/cosine coordinate features.
- `MLP`: shared configurable feed-forward network.
- `save_loss_curve`: renders all numeric CSV loss columns on a log scale.

### 8.2 Loss, checkpoint, and diagnostic helpers

- `training_losses.weighted_field_mse`: channel-weighted and optionally
  point-weighted field MSE. `train.py` currently contains an equivalent local
  `field_loss()` implementation.
- `checkpointing.save_newhonf_checkpoint`, `load_newhonf_checkpoint`, and
  `write_checkpoint_summary`: small wrappers around checkpoint and JSON I/O;
  the current trainer uses its richer local `save_checkpoint()` function.
- `honf_diagnostics.compute_honf_diagnostics`: converts static assignments and
  scalar decoder summaries into CSV-ready entropy, mass, activity, and context
  metrics.
- `honf_diagnostics.organizer_regularization_loss`: optional generic penalties
  for active-edge count, low mass entropy, excessive mass concentration, and
  duplicate source/region vectors. It is a no-op unless explicitly enabled.
- Private `_scalar`, `_entropy`, and `_entropy_norm` functions standardize
  scalar extraction and entropy calculations.

### 8.3 Hypergraph plan helpers

- `extract_hypergraph_plan`: removes batch dimensions, zeros inactive module
  rows, adds schema/domain metadata, and canonicalizes hyperedge order.
- `save_hypergraph_plan`, `load_hypergraph_plan`: compressed NPZ round-trip.
- `validate_hypergraph_plan`: checks required keys, finite values, shapes,
  row/mass normalization, binary active masks, and canonical ordering.
- `summarize_hypergraph_plan`: JSON-friendly key/shape/count summary.
- Private conversion/permutation helpers convert tensors to NumPy and sort
  active edges first by source, region, and strength while retaining original
  edge-index provenance.

### 8.4 Evaluation and visualization helpers

- `evaluation_plots.error_metrics` and `masked_error_metrics`: L2, MSE, RMSE,
  normalized RMSE, MAE, and relative-L2 metrics.
- `module_radius_from_sample`, `module_and_fluid_masks`,
  `draw_module_outlines`: recover geometry and construct evaluation masks.
- `composite_temperature_grid`: inserts predicted local disk temperatures into
  the global temperature image.
- `plot_field_quicklook`, `plot_internal`, `plot_interface`: global field,
  module-internal, and angular-interface comparisons.
- `local_module_plots.raster_from_points`, `plot_local_internal`, and
  `plot_local_interface`: reusable Stage-A versions of the local plots.
  `evaluate_local.py` currently carries equivalent plotting functions inline.
- `organizer_viz_channelthermal` public renderers produce the physical
  overview, tripartite module/hyperedge/region schematic, and assignment/mass
  matrix summary. Its private helpers compute hulls, dominant assignments,
  colors, summaries, and draw module, environment, source, region, and link
  overlays.
- `routing_viz_channelthermal.save_routing_diagnostics`: saves routing NPZ/JSON
  plus requested PNGs. Private helpers reshape query arrays to the physical
  grid, choose active edges/colors/panel layouts, overlay geometry, and render
  per-edge attention/contribution, dominant-edge, and context-norm maps.

### 8.5 Dataset helper behavior

`channelthermal_datasets.py` and `local_module_datasets.py` contain the same
implementation; only their top-level documentation differs. Their private
helpers decode metadata, select split indices, build normalized disk query
coordinates, read case JSON, resolve feature indices, and provide legacy
`h_effective` fallbacks.

- `H5Normalizer` centralizes every supported HDF5 mean/std transformation.
- `LocalModuleDataset` reads native Stage-A cases.
- `GlobalModuleAlignmentDataset` converts each active global module into a
  Stage-A-compatible sample and computes alignment statistics.
- `GlobalChannelThermalDataset` samples global field points, loads local and
  interface targets, constructs teacher ports/local parameters, and supplies
  stored or geometry-only structure targets.

## 9. Validation utilities

The plan test is self-contained:

```bash
conda run -n ModularDT python src_new/_tests/test_hypergraph_plan_stability.py
```

The remaining checks require the packed global dataset and/or the configured
Stage-A checkpoint. Pass the existing test template explicitly:

```bash
conda run -n ModularDT python src_new/_tests/test_local_checkpoint_compat.py \
  --checkpoint ./Saved_Model_LocalModule/Run_0003_20260507_224352/latest_model.pt \
  --device cpu

conda run -n ModularDT python src_new/_tests/smoke_global_modes.py \
  --config Configs_new/_tests/train_global_honf_template.json \
  --device cpu \
  --points 32

conda run -n ModularDT python src_new/_tests/test_newhonf_hardening.py \
  --config Configs_new/_tests/train_global_honf_template.json \
  --device cpu \
  --points 32
```

These verify strict old Stage-A checkpoint compatibility; finite global-only,
teacher, mixed, and predicted forward modes; one-decoder-pass behavior;
deterministic frozen Stage A; mechanism features on auxiliary queries;
self-contained Stage-B checkpoint loading; and stable canonical plan export.
