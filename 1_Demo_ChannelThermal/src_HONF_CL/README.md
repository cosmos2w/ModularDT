# HONF-CL: Hypergraph Operator Neural Field for Channel Thermal Flow

`src_HONF_CL` is a self-contained code package for training and evaluating a
two-scale neural surrogate of channel flow with heated circular modules. It
keeps reusable hypergraph neural-field code separate from ChannelThermal input,
local-solid, and interface-physics code. The packed HDF5 datasets and optional
pretrained checkpoints remain external data dependencies under
`1_Demo_ChannelThermal`; all code and configuration templates needed to launch
the workflows are contained here.

## Model goal and physics problem

The model predicts the steady global field
`[u, v, p, omega, temperature]` at arbitrary points in a rectangular channel,
and predicts the solid temperature and boundary response of every active heated
module. A case is defined by Reynolds number/inlet speed, fluid and solid
material properties, module locations, module heat powers, and a padded module
mask.

The reference data represent coupled fluid/solid heat transfer. In physical
terms, the fluid satisfies steady incompressible momentum and energy transport,
while each solid module satisfies heat conduction with an internal heat source:

```text
div(u) = 0
(u . grad)u = -grad(p) + viscous diffusion
u . grad(T_f) = alpha_f laplacian(T_f)
alpha_s laplacian(T_s) + heat source = 0
```

At a module boundary, temperature and normal heat flux couple the two scales.
The learned port contract uses angle, outward normal, outside temperature, and
effective heat-transfer coefficient:

```text
port = [theta, cos(theta), sin(theta), T_env, h_effective]
q_normal,Robin = h_effective * (T_surface - T_env)
```

HONF-CL is a data-driven surrogate for these solved fields
The Stage-A local model learns one module's internal temperature and interface response. 
The Stage-B global HONF organizes modules and environment samples into latent
hyperedges, incorporates Stage-A responses into module state, and decodes the
global field continuously at requested coordinates.

## Repository layout

```text
src_HONF_CL/
├── train_local.py                 Stage-A local training CLI
├── train.py                       Stage-B coupled global training CLI
├── evaluate_local.py              local evaluation and plots
├── evaluate.py                    global evaluation, plots, routing, plan export
├── compare_models.py              aligned multi-checkpoint comparison
├── requirements.txt               direct Python dependencies
├── configs/                       production and experiment JSON templates
├── data/datasets.py               HDF5 readers, sampling, normalization
├── local_surrogate/model.py       single-module Stage-A neural field
├── honf_core/                     domain-reusable HONF
│   ├── config.py                  strict modes and generic batch contract
│   ├── model.py                   input encoders and encode/decode API
│   ├── organizer.py               A_me, A_mh, A_eh and hyperedge state
│   └── decoder.py                 query routing and field decoder
├── channelthermal/                domain-specific adapters and physics
│   ├── config.py                  combined strict configuration
│   ├── input_adapter.py           physical case -> generic HONF features
│   ├── environment.py             channel wall/inlet/outlet tokens
│   ├── local_coupling.py          port prediction and Stage-A coupling
│   ├── fallback_heads.py          global-only comparison heads
│   └── model.py                   complete coupled model
├── training_tools/                losses and scalar HONF diagnostics
├── evaluation_tools/              plots, routing, organization, plan I/O
├── common/runtime.py              paths, JSON, devices, AMP, small networks
└── tests/                          functional and checkpoint regressions
```

The package uses ordinary top-level imports. Run entry points from
`1_Demo_ChannelThermal`, as shown below, so `src_HONF_CL` and the data paths are
resolved consistently.

## Installation and external inputs

From the repository root:

```bash
cd 1_Demo_ChannelThermal
python -m venv .venv-honf-cl
source .venv-honf-cl/bin/activate
python -m pip install -r src_HONF_CL/requirements.txt
```

The default templates expect:

```text
Data_Saved/Processed_LocalModule_Dataset/packed_dataset.h5
Data_Saved/Processed_ChannelThermal_Dataset/packed_dataset.h5
```

The coupled template also points to an existing Stage-A checkpoint. Either
train Stage A first and set
`model.local_coupling.local_surrogate_checkpoint_path` to its checkpoint, or
use a global-only config with `use_local_surrogate: false`.

All relative paths in JSON are resolved from `1_Demo_ChannelThermal`, not from
the shell's current directory. CUDA is selected automatically when
`training.device` is `null`; use `--device cpu` or `--device cuda:0` to force a
device.

## Configuration

Start from one of the two primary templates:

- `configs/train_local_module_template.json`: Stage-A training. Its `mixed`
  dataset source combines standalone local samples and module samples extracted
  from global cases. One normalizer is fitted on all physical training samples,
  reused by validation, and embedded in checkpoints. The configured source
  weights control `WeightedRandomSampler` probability mass.
- `configs/train_global_honf_template.json`: coupled Stage-B training. Values
  marked `"auto"` are resolved from the packed dataset and written to the run's
  resolved config.

Use a new numeric `Run_ID` for each experiment and change `training.run_name`
to a short descriptive suffix. Set data paths, splits, normalization, batch
sizes, point budgets, worker count, optimizer settings, loss weights, and output
root in their corresponding JSON sections. CLI flags override device, epochs,
batch limits, run name, and Run_ID without editing the template.

Important global settings are:

| Section | Setting | Meaning |
|---|---|---|
| `model.core_honf` | `num_hyperedges`, `hidden_dim` | hypergraph capacity and token width |
| `model.core_honf` | `decoder_mode` | exact combination of hyper, pairwise, global, direct, and near contexts |
| `model.core_honf` | `hyper_module_assignment_mode` | learned or uniform module-to-hyperedge `A_mh` |
| `model.core_honf` | `hyper_query_attention_mode`, `hyper_attention_topk` | learned/uniform query routing and optional sparse top-k |
| `model.local_coupling` | `use_local_surrogate`, checkpoint, freeze | enable and attach Stage A |
| `model.channelthermal` | `internal_prediction_mode` | `auto`, forced `local_surrogate`, or `global_head` |
| `model.physical_correction` | `local_surrogate_flux_mode` | select interface normal-flux construction |
| `model.physical_correction` | `interaction_refinement_steps` | zero or one local/global port refinement |
| `training.port_curriculum` | mode/schedule/ratios | teacher, predicted, or mixed ports during training |
| `checkpointing` | five save switches | total, field, temperature, autonomous-predicted, and latest checkpoints |

Unknown configuration keys raise an error. Historical redundant core flags are
accepted only when old checkpoints/configs are loaded; `decoder_mode` is the
single authority in cleaned configs.

### Supported modes and their code locations

`honf_core/config.py` declares every decoder mode and its enabled components:

| Decoder mode | Context components |
|---|---|
| `hyper_only` | hyperedge value context |
| `hyper_plus_global` | hyper + global |
| `hyper_plus_direct_residual` | hyper + direct module/environment memory |
| `hyper_plus_near_module` | hyper + Gaussian near-module context |
| `hyper_plus_global_near` | hyper + global + near |
| `hyper_plus_global_direct` | hyper + global + direct |
| `hyper_plus_near_direct` | hyper + near + direct |
| `no_hyper_global_near` | global + near, no query/hyperedge value path |
| `no_hyper_current_like_direct`, `current_like` | global + near + direct |
| `enhanced_honf_pairwise` | hyper value + routed pairwise + global + near |
| `enhanced_honf_pairwise_only` | routed pairwise + global + near; hyper value disabled |

The component branches are implemented in `honf_core/decoder.py`.
`geometry_mode` is `nonperiodic` or `periodic`; `query_time_mode` is `none`,
`phase`, or `physical_time`; `boundary_feature_mode` is `none` or `channel`.
Module assignment and query attention each support `learned` or `uniform`.

`channelthermal/local_coupling.py` implements port modes `teacher`, `predicted`,
and `mixed`. Evaluation additionally accepts `both` and runs teacher and
predicted views. Its interface flux modes are:

- `surrogate`: use Stage-A predicted normal flux.
- `physics_from_port`: use the Robin expression exactly.
- `corrected_physics`: Robin flux plus a learned residual initialized at zero.
- `blend`: weighted surrogate/Robin combination using
  `local_surrogate_flux_blend_alpha`.

`interaction_refinement_steps` supports `0` or `1`. With one predicted-port
step, the provisional global field is sampled just outside every module, ports
are refined, Stage A is rerun, and final module tokens are rebuilt.

The copied experiment templates under `configs/` and `configs/experiments/`
preserve old-parity, global-only, uniform-assignment, and no-pretrained-module
variants. Treat `train_global_honf_template.json` and
`train_local_module_template.json` as the maintained starting points; the
others are controlled ablations or historical reproduction settings.

## Training

### 1. Train or fine-tune Stage A

Review the optional `training.init_checkpoint_path`. Set it to `null` for a
fresh run or to a compatible local checkpoint for fine-tuning, then launch:

```bash
python src_HONF_CL/train_local.py \
  --config src_HONF_CL/configs/train_local_module_template.json \
  --Run_ID 0001 \
  --run-name local_mixed
```

The run is written under `Saved_Model_HONF_CL/Local/Run_<id>_<name>_<time>/`
with `resolved_train_config.json`, `loss_history.csv`, `loss_curve.png`,
`best_model.pt`, and `latest_model.pt`. Checkpoints contain architecture,
training config, and the shared training-fitted normalization statistics.

### 2. Train the coupled global model

Put the selected Stage-A checkpoint in
`model.local_coupling.local_surrogate_checkpoint_path`, then launch:

```bash
python src_HONF_CL/train.py \
  --config src_HONF_CL/configs/train_global_honf_template.json \
  --Run_ID 0002 \
  --run-name coupled_predicted
```

For a global-only baseline use an appropriate bundled global-only config or set
`use_local_surrogate: false`, clear the checkpoint path, and select
`internal_prediction_mode: global_head`.

Global runs are written under
`Saved_Model_HONF_CL/Global/Run_<id>_<time>_<name>/`. The directory contains
`config_resolved.json`, `metrics.csv`, training plots, `summary.json`, and the
enabled checkpoints:

- `best_model.pt`: lowest configured total validation objective.
- `best_by_field_mse_model.pt`: lowest global field MSE.
- `best_by_temperature_mse_model.pt`: lowest temperature MSE.
- `best_predicted_model.pt`: best autonomous predicted-port validation metric.
- `latest_model.pt`: most recent epoch.

Checkpoint writes are atomic. Global checkpoints embed the attached Stage-A
model and its normalization, so evaluation does not depend on the original
local checkpoint path. Resume in place with:

```bash
python src_HONF_CL/train.py \
  --config src_HONF_CL/configs/train_global_honf_template.json \
  --resume-checkpoint Saved_Model_HONF_CL/Global/Run_.../latest_model.pt
```

## Evaluation and post-processing

Evaluate one local case by Run_ID:

```bash
python src_HONF_CL/evaluate_local.py \
  --Run_ID 0001 --checkpoint best --split test --case-index 0
```

This produces internal-temperature maps, angular interface curves, error
metrics, and `evaluation_summary.json` under the checkpoint run directory (or
`--output-dir`). Evaluation uses checkpoint normalization statistics; it does
not refit statistics on the evaluation split.

Evaluate a global checkpoint and request the complete formal post-processing
set:

```bash
python src_HONF_CL/evaluate.py \
  --Run_ID 0002 \
  --checkpoint best_predicted \
  --split test \
  --case-index 0 \
  --local-port-condition-mode predicted \
  --temperature-display-mode composite_internal \
  --organization-view all \
  --organization-style presentation \
  --return-routing-maps \
  --routing-view all \
  --export-hypergraph-plan
```

The evaluator decodes a full grid in `--query-batch-size` chunks while encoding
and organizing the case only once. Outputs include field and error quicklooks,
solid internal-temperature maps, interface curves, NPZ arrays, CSV/JSON
metrics, organization overview/schematic/matrices, optional query-routing maps,
hypergraph diagnostics, and a canonical `hypergraph_plan.npz` plus summary.
Use `--case-id` for stable case selection, `--organization-view none` to omit
organizer figures, or `--routing-view summary` for a smaller routing report.

Compare multiple checkpoints on exactly the same sampled case set:

```bash
python src_HONF_CL/compare_models.py \
  --Run_ID 0002 \
  --Run_ID 0003 \
  --label coupled \
  --label global_only \
  --checkpoint-selector best_predicted \
  --split test \
  --case-ratio 1.0
```

The comparison writes a manifest, selected-case log, per-case/per-module and
summary CSV tables, reconstruction figures, hypergraph figures, and aggregate
violin/bar plots under `Saved_Model_HONF_CL/Global/CompareModels/`. It preserves
declared model order and verifies aligned case IDs and field-channel order.
Requested checkpoint substitution is disabled by default; use
`--allow-checkpoint-fallback` only when fallback behavior is intentional.

## Detailed tensor and code flow

Notation: `B` batch cases, `M` padded module slots, `E` environment nodes, `K`
hyperedges, `P` angular ports, `Q` global query points, `Ql` local disk query
points, `L` local latent queries, `H` hidden width, and `F` global field width.

### Data layer

`GlobalChannelThermalDataset` returns physical structure tensors
`module_centers [M,2]`, `heat_powers [M]`, `module_present [M]`, material/case
features, sampled `query_xy [Q,2]`, and `target_field [Q,F]`. When Stage A is
needed it also returns local parameters `[M,7]`, teacher ports `[M,P,5]`, local
query points `[Ql,2]`, internal targets `[M,Ql,1]`, and interface targets
`[M,P,2]`. Random point selection is reproducible for `(seed, case, epoch)`.
Training calls `set_epoch` so the subset changes between epochs without losing
reproducibility. Expensive full grids and structure targets are loaded only when
evaluation/comparison explicitly requests them.

`LocalModuleDataset` returns one module per item. `GlobalModuleAlignmentDataset`
extracts active modules from global cases using the same Stage-A schema.
`fit_local_normalizer` streams all selected training sources to calculate one
mean/std transform; validation and checkpoint evaluation receive that same
immutable transform.

### Stage A: local module surrogate

In `local_surrogate/model.py`:

1. `module_params [B,7] -> param_state [B,H]` through an MLP.
2. `port_tokens [B,P,5] -> port_state [B,P,H]` through a second MLP.
3. Learned latent queries `[L,H]` are conditioned on `param_state`, broadcast to
   `[B,L,H]`, and repeatedly cross-attend to the `P` port tokens.
4. Mean-pooled latents and the parameter state produce
   `module_response_latent [B,Dlocal]`.
5. Fourier-encoded local points `[B,Ql,2]` plus the response latent decode
   `internal_temperature [B,Ql,1]`.
6. Port states plus the response latent decode interface
   `[T_surface,q_normal] [B,P,2]`.

The architecture and parameter names remain strictly compatible with existing
Stage-A checkpoints. During global coupling only active modules are gathered,
evaluated, and scattered back into padded `[B,M,...]` tensors.

### Physical inputs and HONF encoding

`channelthermal/input_adapter.py` converts physical inputs to
`module_features [B,M,10]` and `global_context [B,14]`; exact column names are
declared beside the code. `environment.py` creates `E=nx*ny` cell-centered
coordinates `[B,E,2]` and seven normalized wall/inlet/outlet/centerline
features `[B,E,7]`.

`honf_core/model.py` encodes those arrays as:

```text
module features + Fourier(module centers) -> module_tokens [B,M,H]
environment coordinates/features          -> env_tokens    [B,E,H]
case context                               -> global_token   [B,H]
```

Per-case environment coordinates remain per-case; they are not incorrectly
replaced by the first case's grid.

### Hypergraph organizer

`honf_core/organizer.py` builds three incidence tensors:

```text
A_me [B,M,E]  optional module-to-environment attention
A_mh [B,M,K]  module-to-hyperedge assignment
A_eh [B,E,K]  environment-to-hyperedge assignment with geometry bias
```

`A_mh` respects the padded module mask. Weighted aggregation produces module
and environment summaries `[B,K,H]`, source and region centroids `[B,K,2]`,
normalized masses/strengths, and generic source-region mechanism descriptors.
Their sum passes through `hyper_mix` to form `hyper_state [B,K,H]`.

### Query decoder

`honf_core/decoder.py` maps normalized/Fourier query features
`[B,Q,Dq] -> query_state [B,Q,H]`. In hyper modes, dot-product routing produces
`alpha_qk [B,Q,K]`, optionally with source/region geometry bias, uniform
routing, temperature scaling, or sparse top-k selection. Hyperedge values
reduce to context `[B,Q,H]`.

The enhanced pairwise kernel separately forms relative query-module embeddings
`[B,Q,M,H]`, pools them through `A_mh` into `[B,Q,K,H]`, and reduces them with
the same `alpha_qk` to `[B,Q,H]`. Enabled global, direct-memory, and near-module
contexts are added, normalized, and decoded to `pred_field [B,Q,F]`. Training
returns scalar routing diagnostics; dense `[B,Q,K]` maps are materialized only
when explicitly requested during evaluation.

### Coupled ChannelThermal wrapper

`channelthermal/model.py` first creates base module/environment/hyperedge state.
`PortConditionHead` predicts `[theta,cos,sin,T_env,h] [B,M,P,5]`. The chosen
teacher/predicted/mixed ports and `[B,M,7]` local parameters enter Stage A.
`local_response_summary` reduces local fields to six physical statistics, and
`fuse_module_state` combines those statistics and the local response latent
with the base `[B,M,H]` token. The final organizer and decoder then produce the
global field. A global-only run uses `fallback_heads.py` instead.

The evaluator requests `PreparedChannelThermalCase`, which retains only final
case-static organizer state and `global_token`; subsequent query chunks call
`decode_prepared` and never rerun Stage A or the organizer.

## Helper reference

| File | Helpers and responsibility |
|---|---|
| `common/runtime.py` | resolve demo-relative paths; read/write JSON; create directories; load trusted metadata-bearing checkpoints; seed RNGs; select devices; recursively move batches; count parameters; guard near-zero standard deviations; create AMP scaler/autocast contexts; remove DataParallel prefixes; provide shared Fourier and MLP blocks |
| `data/datasets.py` | decode HDF5 strings/configs, choose split indices and feature columns, construct disk query points, provide `H5Normalizer`, stream Stage-A moments, lazily open worker-local HDF5 handles, sample global points, extract material/domain/teacher-port data, and derive optional fallback organizer targets |
| `training_tools/losses.py` | compute field MSE with configured channel and point weights |
| `training_tools/honf_diagnostics.py` | compute assignment/routing entropy, edge-use and context-norm scalars; optionally add generic anti-collapse regularization |
| `evaluation_tools/plots.py` | compute full/masked error metrics, build fluid/module masks and composite temperature grids, and render global/local field, internal, and interface plots |
| `evaluation_tools/organizer_visualization.py` | render physical hyperedge regions, source/region links, module memberships, and incidence/mass summary matrices |
| `evaluation_tools/routing_visualization.py` | reshape query maps, select active edges, overlay geometry, and export dominant-edge, per-edge, entropy, and context-norm figures/data |
| `evaluation_tools/hypergraph_plan.py` | extract a query-independent organizer plan, canonicalize hyperedge order, validate/save/load NPZ plans, and produce a JSON-safe summary |
| `channelthermal/local_coupling.py` | translate interface conditions to teacher ports, assemble/update local parameters, predict ports, attach/freeze Stage A, apply stored normalization, select ports, gather active modules, construct physical/corrected flux, summarize local response, and fuse module state |
| `evaluate.py` | resolve checkpoints/cases, load embedded Stage A, tensorize samples, run prepared-state chunked inference, denormalize predictions, compute diagnostics, and dispatch all global exports |
| `evaluate_local.py` | resolve a local checkpoint, inject checkpoint statistics, run one local case, and dispatch local plots/metrics |
| `compare_models.py` | resolve ordered model specifications, retry checkpoint reads without silent substitution, align cases, compute reconstruction/module/hypergraph metrics, aggregate tables, and render comparison figures |

Every module, class, and function contains an in-code docstring; major model
forwards additionally document their tensor contracts next to the
implementation.
