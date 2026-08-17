# HONF Project

`HONF_Proj` is the installable home of the Hypergraph organized Neural Field
(HONF): a neural operator for predicting continuous physical fields around a
variable-size set of interacting modules. The repository separates reusable
model/runtime code from physical-case code and includes one complete example,
`ThermalChannel`.

The project currently supports two connected directions:

| Direction | Question | Status |
|---|---|---|
| Forward HONF | Given a modular design and operating context, what fields and module responses occur? | Established, checkpoint-compatible workflow with migration parity evidence |
| Hierarchical inverse | Given desired behavior and context, which mechanisms and modular layouts should be tried? | Initial research implementation for dataset, staged training, and verified candidate studies |

The forward model is the foundation. The inverse model never replaces it: it
generates candidate designs and uses a frozen autonomous forward checkpoint to
verify every reported candidate. Neither direction replaces a high-fidelity
solver or engineering validation.

## 1. The project in one picture

For a physical design `D`, operating context `c`, and query coordinates `q`,
the forward model predicts the channel field and local module response:

```text
physical design D + context c + query coordinates q
                         │
                         ▼
           case adapter and environment tokens
                         │
                         ▼
                 reusable HONF core
                         │
              predicted module ports
                         │
                         ▼
           frozen local-module surrogate
                         │
          local response fused into HONF state
                         │
                         ▼
   global field + internal temperature + interface response
```

The inverse hierarchy works in the opposite design direction but closes its
loop through the same forward model:

```text
structured request R + context c
              │
              ▼
    compact mechanism plan G          sample several plausible plans
              │
              ▼
      physical modular design D       sample several layouts per plan
              │
              ▼
      frozen autonomous HONF
              │
              ├── exact request functionals and physical outputs
              └── realized mechanism G_hat
                            │
                            └── optional one-pass bounded correction
```

The main symbols used throughout the inverse code are:

| Symbol | Meaning |
|---|---|
| `D` | physical design: module presence, centers, and heat powers |
| `c` | operating/material/domain context |
| `R` | unordered functional request tokens plus separate geometry constraints |
| `G` | generated compact plan over the forward model's fixed hyperedges |
| `G_hat` | compact plan realized by the final organizer when HONF evaluates `D` |

`Model_Explain.md` contains the full forward and inverse equations. This README
focuses on how the pieces fit together and how to run them safely.

## 2. Code organization and ownership

```text
HONF_Proj/
├── train.py                         generic forward/local training dispatcher
├── evaluate.py                      generic forward/local/compare dispatcher
├── src/
│   ├── honf_forward_core/           reusable encoder, organizer, decoder, losses
│   ├── honf_inverse_core/           request encoder, flows, corrector, sampling
│   ├── honf_runtime/                config composition, plugins, paths, runs
│   └── config_core/
│       ├── forward/                 maintained forward launch profiles
│       ├── inverse/                 inverse build/train/evaluation templates
│       └── schemas/                 strict configuration schemas
├── Case_ThermalChannel/
│   ├── src/channelthermal/
│   │   ├── data/                    packed-HDF5 readers and batch construction
│   │   ├── local_surrogate/         Stage-A single-disk model and contract
│   │   ├── workflows/               case-owned train/evaluate implementations
│   │   ├── evaluation_tools/        forward plots and organization views
│   │   ├── inverse/                 ThermalChannel inverse physics and artifacts
│   │   ├── input_adapter.py         physical tensors -> reusable HONF contract
│   │   ├── local_coupling.py        predicted ports and Stage-A coupling
│   │   ├── model.py                 complete coupled ThermalChannel wrapper
│   │   └── plugin.py                runtime integration and strict case config
│   ├── scripts/inverse/             thin inverse command-line launchers
│   ├── inverse_requests/            request/context JSON examples
│   ├── configs/                     case-owned physics/data/loss settings
│   ├── Dataset/                     manifest, schemas, and local path template
│   └── artifacts/                   optional local checkpoint location
├── Trained_Results/                 generated runs and evaluations
├── docs/                            extension, config, checkpoint, result contracts
├── tests/                           reusable runtime/model/inverse tests
└── Case_ThermalChannel/tests/       case-specific contract and integration tests
```

The dependency direction is strict:

```text
generic entry points -> runtime/plugin protocol -> installed case package
                                             └──> reusable forward/inverse core
```

- `honf_forward_core` and `honf_inverse_core` do not import ThermalChannel,
  HDF5, or plotting code.
- The case package owns physical feature meaning, datasets, losses,
  local-module coupling, exact inverse functionals, and plots.
- `train.py` and `evaluate.py` discover a case through its configured dotted
  plugin factory; they do not branch on a case name.
- The inverse launchers are currently ThermalChannel-owned because inverse
  dataset construction and verification require case physics.

This boundary is what allows the same core to support another physical case.

## 3. Installation, data, and required artifacts

Run all commands below from the `HONF_Proj` directory. Python 3.10 or newer is
required. Install PyTorch for the local CUDA/CPU platform first if necessary.

The maintained development environment is `ModularDT`:

```bash
cd HONF_Proj
conda activate ModularDT
python -m pip install -e . -e ./Case_ThermalChannel
```

For test and lint dependencies:

```bash
python -m pip install -e '.[dev]'
```

### 3.1 Configure the ThermalChannel datasets

Datasets are external and are not synchronized by Git. Copy the location map
and replace both example paths:

```bash
cp Case_ThermalChannel/Dataset/dataset_locations.example.json \
   Case_ThermalChannel/Dataset/dataset_locations.local.json
```

| Logical dataset ID | Purpose | Cases and splits |
|---|---|---|
| `thermal_disk_local_v1` | Stage-A isolated/local disk responses | 1,034: 919 train, 115 test |
| `thermal_channel_global_v1` | Stage-B coupled channel fields | 690: 600 train, 90 test |

The local map is ignored by Git. As an alternative, `HONF_DATA_ROOT` may point
to a directory containing `Processed_LocalModule_Dataset/` and
`Processed_ChannelThermal_Dataset/`.

No trained checkpoint is bundled with the source tree; use a trusted existing
artifact or train the required stage locally.

Validate required HDF5 keys, size, and full SHA-256 after copying data:

```bash
python tools/inspect_dataset.py --dataset-id thermal_disk_local_v1 --sha256
python tools/inspect_dataset.py --dataset-id thermal_channel_global_v1 --sha256
```

The manifest, field order, module tensors, interface features, padding, and
normalization rules are documented in
`Case_ThermalChannel/Dataset/PHYSICS_AND_DATA.md` and `Dataset/schemas/`.

### 3.2 Understand checkpoint dependencies

There are three checkpoint levels:

1. Stage A writes `best_model.pt`, a local heated-disk surrogate.
2. Stage B consumes that Stage-A checkpoint and writes a self-contained global
   checkpoint. The local model and normalizers are embedded, so later forward
   evaluation needs no separate Stage-A file.
3. The inverse dataset/trainer consumes `best_predicted_model.pt`, and inverse
   evaluation checks that the configured forward checkpoint matches the SHA
   recorded by the inverse checkpoint.

For a new Stage-B run, pass the local checkpoint explicitly with
`--local-checkpoint`, or update the case configuration's
`model.local_coupling.local_surrogate_checkpoint.path`. Generated checkpoints
are trusted PyTorch pickle artifacts; never load an untrusted `.pt` file.

### 3.3 Paths and configuration composition

A forward launch combines:

- one core profile under `src/config_core/forward/`, which owns the workflow,
  architecture, optimizer, checkpoint policy, dataset ID, and run identity;
- the referenced case profile
  `Case_ThermalChannel/configs/case_default.json`, which owns physical data,
  local coupling, losses, evaluation defaults, and local-module settings;
- an optional strict experiment overlay; and
- allow-listed command-line overrides.

`project://...` paths are anchored at `HONF_Proj`; `config://...` paths are
anchored at their configuration file. Unknown settings and ownership mistakes
fail before a run directory is created. Every managed run stores both source
profiles, overrides, the resolved configuration, hashes, software information,
and Git state.

## 4. A beginner's ThermalChannel walkthrough

The demo is a steady incompressible channel containing a variable number of
circular heated solid modules. The coupled reference solution contains fluid
momentum/energy, solid conduction, and temperature/heat-flux interaction at
every module boundary.

The global output field order is:

```text
[u, v, p, omega, temperature]
```

The easiest way to understand the system is to follow its actual dependency
order: inspect data, obtain Stage A, run Stage B, evaluate the autonomous
forward checkpoint, and only then try inverse design.

### 4.1 Always validate a launch first

Forward and local training are confirmation-gated. `--dry-run` resolves and
validates configuration, dataset, checkpoint dependencies, run ID, device, and
destination without writing anything:

```bash
python train.py \
  --config src/config_core/forward/local_module_thermal_disk.json \
  --device cuda:0 --dry-run
```

Use `--yes` only after reviewing that output. A numeric run ID is unique within
its result family and is never silently overwritten.

### 4.2 Quick wiring smoke versus meaningful training

A bounded smoke confirms data/model/checkpoint wiring; it does not produce a
useful scientific model. Choose unused numeric IDs on your machine:

```bash
# Stage-A one-batch smoke
python train.py \
  --config src/config_core/forward/local_module_thermal_disk.json \
  --run-id 9001 --epochs 1 --max-train-batches 1 --max-val-batches 1 \
  --device cuda:0 --yes

# Stage-B one-batch smoke using the resulting Stage-A checkpoint
python train.py \
  --config src/config_core/forward/enhanced_honf_pairwise.json \
  --local-checkpoint Trained_Results/ThermalChannel/Local_Module_Runs/thermal_disk/Run_9001_.../best_model.pt \
  --run-id 9002 --epochs 1 --max-train-batches 1 --max-val-batches 1 \
  --device cuda:0 --yes
```

Replace the ellipsis with the actual timestamped directory. Formal experiments
use the full profile budgets and should be launched deliberately; do not infer
model quality from smoke outputs.

### 4.3 Evaluate an existing autonomous forward checkpoint

If you already have the packed global dataset and a trusted self-contained
`best_predicted_model.pt`, this is the shortest meaningful demo:

```bash
python evaluate.py \
  --config src/config_core/forward/enhanced_honf_pairwise.json \
  --workflow forward \
  --checkpoint /absolute/path/to/best_predicted_model.pt \
  --dataset /absolute/path/to/Processed_ChannelThermal_Dataset/packed_dataset.h5 \
  --device cuda:0 --case-index 0 \
  --organization-view all --routing-view summary \
  --export-hypergraph-plan
```

This reconstructs the checkpoint-owned architecture and normalization,
evaluates a complete test case, plots the predicted field and local responses,
shows the final hypergraph organization, and exports the canonical static plan
used by inverse tooling.

## 5. Forward model: from one module to the coupled channel

### 5.1 Stage A: local thermal-disk surrogate

Stage A learns one reusable module operator. Its physical input consists of:

- seven module/material/port-summary scalars;
- a sequence of angular Robin-condition tokens
  `[theta, cos(theta), sin(theta), T_env, h]`; and
- normalized query coordinates inside the circular module.

The local model uses shared token/coordinate encoders and cross-attention to
predict:

- internal solid temperature at arbitrary disk coordinates; and
- interface response `[T_surface, q_normal]` at the angular ports.

The mixed local workflow combines standalone local training samples with active
modules extracted from global channel cases, then fits one training-only
normalizer shared by validation and checkpoint evaluation.

Train Stage A:

```bash
python train.py \
  --config src/config_core/forward/local_module_thermal_disk.json \
  --device cuda:0 --dry-run

python train.py \
  --config src/config_core/forward/local_module_thermal_disk.json \
  --device cuda:0 --yes
```

Evaluate a managed local run:

```bash
python evaluate.py \
  --config src/config_core/forward/local_module_thermal_disk.json \
  --workflow local_module --run-id 0001 --checkpoint best \
  --device cuda:0 --case-index 0
```

The run is saved under
`Trained_Results/ThermalChannel/Local_Module_Runs/thermal_disk/` and contains
`best_model.pt`, `latest_model.pt`, `loss_history.csv`, summary/config files,
and local internal/interface plots.

### 5.2 Stage B: coupled global HONF

For each channel case, the ThermalChannel adapter converts physical tensors
into the case-neutral HONF contract:

- padded module centers/features and `module_present` mask;
- case-level flow, material, heat, and geometry context;
- a fixed `24 x 8` environment-token grid in the maintained profiles; and
- arbitrary global query coordinates.

The complete Stage-B path is:

1. Shared encoders create module, environment, and global-context tokens.
2. The HONF organizer assigns modules and environment tokens to six latent
   hyperedges and constructs source/region centroids, masses, and edge states.
3. A learned port head predicts autonomous outside temperature and transfer
   coefficient for each module boundary point.
4. Frozen Stage A predicts internal temperature and interface response. A
   physically anchored Robin flux is blended with a learned correction.
5. Six response statistics and a local latent are fused back into each global
   module token.
6. One configured local/global refinement pass samples provisional outside
   temperature, updates the ports, and reevaluates the local response.
7. The final organizer is recomputed after local-response fusion.
8. The continuous decoder predicts `[u,v,p,omega,T]` at every requested point.

The main model outputs are:

| Output | Meaning |
|---|---|
| `pred_field [B,Q,5]` | global velocity, pressure, vorticity, temperature |
| `pred_internal_temperature [B,M,Ql,1]` | per-module solid temperature |
| `pred_interface [B,M,P,2]` | corrected surface temperature and normal heat flux |
| `pred_port_condition [B,M,P,5]` | final autonomous local boundary condition |
| `organizer_aux` | final post-fusion incidences, centroids, masses, and states |
| `routing_aux` | query routing and pairwise decoder diagnostics |

Inactive padded slots are masked from organization, local inference, losses,
and metrics.

### 5.3 Maintained Stage-B profiles

The two main profiles share width 256, six hyperedges, the same environment
tokens, frozen Stage A, coupled losses, and predicted-port training. They differ
in the global decoding mechanism being studied:

| Profile | Decoder idea | Default run ID |
|---|---|---:|
| `hyper_plus_global_near.json` | hyperedge context plus global and local near-module context | `0001` |
| `enhanced_honf_pairwise.json` | hyperedge organization plus a learned module-query pairwise kernel | `0002` |

Train either profile with an explicit Stage-A dependency:

```bash
python train.py \
  --config src/config_core/forward/hyper_plus_global_near.json \
  --local-checkpoint /path/to/stage_a/best_model.pt \
  --device cuda:0 --dry-run

python train.py \
  --config src/config_core/forward/hyper_plus_global_near.json \
  --local-checkpoint /path/to/stage_a/best_model.pt \
  --device cuda:0 --yes

python train.py \
  --config src/config_core/forward/enhanced_honf_pairwise.json \
  --local-checkpoint /path/to/stage_a/best_model.pt \
  --device cuda:0 --yes
```

The coupled objective includes weighted global-field MSE, internal-temperature
loss, interface loss, supervised port loss, angular port smoothness,
global/interface consistency, and a warm predicted-port consistency term.
Organizer anti-collapse regularization exists for experiments but is disabled
in the maintained templates.

Strict ablation overlays are under `src/config_core/forward/experiments/`. For
example, the global-only fallback needs no Stage-A checkpoint:

```bash
python train.py \
  --config src/config_core/forward/enhanced_honf_pairwise.json \
  --experiment-overlay src/config_core/forward/experiments/global_only.json \
  --run-id 0003 --device cuda:0 --dry-run
```

### 5.4 Forward checkpoint selection and evaluation

Stage-B training can write:

| Selector | Root filename | Selection criterion |
|---|---|---|
| `best` | `best_model.pt` | total validation objective |
| `best_by_field_mse` | `best_by_field_mse_model.pt` | global field MSE |
| `best_by_temperature_mse` | `best_by_temperature_mse_model.pt` | temperature MSE |
| `best_predicted` | `best_predicted_model.pt` | autonomous predicted-port validation |
| `latest` | `latest_model.pt` | latest resumable state |

`best_predicted` is the normal checkpoint for autonomous deployment and the
required foundation for inverse data/verification.

Evaluate a managed forward run by its unique numeric ID:

```bash
python evaluate.py \
  --config src/config_core/forward/enhanced_honf_pairwise.json \
  --workflow forward --run-id 0002 --checkpoint best_predicted \
  --device cuda:0 --case-index 0 \
  --organization-view all --routing-view summary \
  --export-hypergraph-plan
```

Forward evaluation supports `teacher`, `predicted`, `mixed`, or `both` local
port conditions; full-grid decoding in query chunks; physical, matrix, and
schematic organizer views; optional dense routing maps; and compact NPZ plan
export. Run `python evaluate.py --help`, then append ThermalChannel-specific
options after the generic arguments.

Compare any number of compatible runs or explicit checkpoints on one dataset:

```bash
python evaluate.py \
  --config src/config_core/forward/enhanced_honf_pairwise.json \
  --workflow compare --Run_ID 0001 --Run_ID 0002 \
  --checkpoint-selector best_predicted \
  --label hyper_plus_global_near --label enhanced_honf_pairwise \
  --device cuda:0
```

Missing checkpoint selectors and ambiguous run IDs fail by default. Use an
exact checkpoint path to disambiguate. Checkpoint fallback is available only
when evaluation or comparison explicitly requests it.

### 5.5 Resume a managed forward/local run

Interrupted managed runs resume in place:

```bash
python train.py \
  --config src/config_core/forward/enhanced_honf_pairwise.json \
  --resume-checkpoint Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_0002_.../latest_model.pt \
  --device cuda:0 --yes
```

Resume validates case, workflow, model family, immutable model/data/loss
sections, feature schemas, dataset identity, and normalization. Current
checkpoints restore model, optimizer, AMP scaler, epoch, best metrics, and
Python/NumPy/Torch/CUDA random states. `--local-checkpoint` is not resume; it
selects Stage A for a new Stage-B run or initializes a new local run.

## 6. Hierarchical inverse design

### 6.1 Purpose and current scope

The inverse model is an initial, bounded research implementation around the
current ThermalChannel forward model. It is usable for contract tests, dataset
construction, staged experiments, and verified candidate studies, but it is
not yet as mature as the forward workflow.

It factorizes the one-to-many problem as:

```text
p(D,G | R,c) = p(D | G,R,c) p(G | R,c)
```

Independent Gaussian noise in both conditional rectified flows allows several
mechanisms for one request and several layouts for one mechanism. There is no
iterative design optimization.

The reusable inverse core contains:

```text
src/honf_inverse_core/
├── request_schema.py, contracts.py, normalization.py
├── models/
│   ├── request_encoder.py            unordered request/context encoder
│   ├── plan_flow.py                  (R,c) -> fixed-edge compact G
│   ├── layout_flow.py                (G,R,c) -> padded layout D
│   ├── joint_corrector.py            optional one bounded residual proposal
│   └── hierarchical_inverse.py       public model/checkpoint API
├── training/                         losses, stages, trainer, checkpoints
└── sampling/                         result contracts, ranking, serialization
```

ThermalChannel-specific vocabulary, exact functionals, compact-plan extraction,
geometry, frozen HONF adapter, HDF5 builder, and plots live under
`Case_ThermalChannel/src/channelthermal/inverse/`.

### 6.2 Structured requests

Request schema v1 accepts one to four active tokens and intentionally supports
only these functionals:

| Functional | Meaning |
|---|---|
| `environment_temperature_max` | maximum predicted fluid temperature |
| `pressure_drop` | mean inlet pressure minus mean outlet pressure |
| `outlet_temperature_nonuniformity` | outlet-temperature standard deviation |
| `internal_temperature_max` | largest active-module internal temperature |
| `internal_temperature_spread` | spread of active-module peak temperatures |
| `regional_temperature_mean` | mean fluid temperature in a normalized rectangle |
| `regional_temperature_max` | maximum fluid temperature in that rectangle |

Relations are `upper_bound`, `lower_bound`, `target_range`, and `minimize`.
Geometry remains separate: module-count bounds, minimum center distance,
wall/inlet/outlet clearances, and optional total heat. Schema v1 permits at
most one regional token.

Start from the strict examples in `Case_ThermalChannel/inverse_requests/`:

```text
balanced_cooling.json
low_internal_temperature.json
downstream_region_avoidance.json
low_pressure_drop_and_uniform_outlet.json
mixed_global_local_request.json
contexts/reference_operating_context.json
```

Targets are expressed in physical units and normalized from training
statistics embedded in the inverse artifact. Unknown fields, duplicate active
functionals, invalid regions/ranges, and unsupported versions fail early.

### 6.3 Compact mechanism plan

For each fixed forward hyperedge, `G` stores activity, module-side source
location, environment-region location and scale, module/environment mass,
mass-derived strength, heat fraction, and hard module/source fraction. It does
not generate dense organizer states, query routing, raw module tokens, or full
incidence matrices.

The generated `G` and verified `G_hat` use the same versioned 12-feature schema
and canonical active-first edge order. This gives the hierarchy an
interpretable intermediate target and exposes whether a generated layout
realizes the mechanism it was conditioned on.

### 6.4 Build the inverse dataset

Edit the checkpoint, source-dataset, and output paths in
`src/config_core/inverse/thermalchannel_inverse_data_v1.json`, including every
placeholder. The forward checkpoint should be the trusted self-contained
`best_predicted_model.pt` you intend to keep frozen:

```bash
python Case_ThermalChannel/scripts/inverse/build_inverse_dataset.py \
  --config src/config_core/inverse/thermalchannel_inverse_data_v1.json \
  --dry-run

python Case_ThermalChannel/scripts/inverse/build_inverse_dataset.py \
  --config src/config_core/inverse/thermalchannel_inverse_data_v1.json \
  --device cuda:0 --yes
```

For each source case, the builder loads `D,c`, runs frozen HONF once in
predicted-port mode, exports the final canonical plan, derives `G`, evaluates
the supported physical functionals, and saves geometry/provenance. It then
creates 16 request variants by default without another forward call.

Splits are assigned before augmentation, so variants of one design cannot leak
between train, validation, and test. The case-major HDF5 stores each `D,c,G`
once and adds a request-variant axis; the training reader flattens
`(case, variant)` only when batching.

Outputs include `inverse_dataset_v1.h5`, `dataset_summary.json`, split-ID
hashes, and functional/request/plan histograms. A build limited by
`--max-cases-per-split` is marked partial diagnostic data and normal training
rejects it unless `--allow-partial-debug` is explicit.

### 6.5 Train the four-stage hierarchy

Edit dataset/checkpoint paths in
`src/config_core/inverse/train_inverse_hierarchical_template.json`, then run:

```bash
python Case_ThermalChannel/scripts/inverse/train_inverse_hierarchical.py \
  --config src/config_core/inverse/train_inverse_hierarchical_template.json \
  --device cuda:0 --yes
```

Use `--smoke` only for a small one-epoch-per-stage wiring diagnostic.

| Stage | Learned behavior |
|---|---|
| `stage_plan` | request/context to true compact plan |
| `stage_layout_teacher_plan` | true plan to physical layout |
| `stage_layout_mixed_plan` | layout generation under gradually mixed true/generated plans |
| `stage_joint_consistency` | sparse frozen-HONF request/plan/geometry consistency and optional corrector |

Plan and layout flows use default hidden width 256, four residual blocks, and
24 Heun sampling steps. The joint-stage consistency contribution is capped so
it cannot dominate flow matching. Current inverse training supports selected
stages and warm initialization but does not promise exact interrupted-run
resume.

Inverse runs contain `best_plan_model.pt`, `best_layout_model.pt`,
`best_unguided_model.pt`, `best_corrected_model.pt`, `latest_model.pt`,
`metrics.csv`, a live `loss_curve.png`, atomic `training_status.json`,
`config_resolved.json`, and `summary.json`. Training displays nested stage,
epoch, and batch progress with running total/flow losses; every completed epoch
prints its summary and the latest/best checkpoint decisions. A failed epoch
records its exception and traceback in `training_status.json`. Every checkpoint
records the forward checkpoint identity, inverse dataset hash, schema versions,
normalization, and model configuration.

### 6.6 Sample, verify, correct once, and rank

Edit `src/config_core/inverse/evaluate_inverse_hierarchical_template.json` and
keep its forward checkpoint consistent with the inverse checkpoint provenance:

```bash
python Case_ThermalChannel/scripts/inverse/evaluate_inverse_hierarchical.py \
  --config src/config_core/inverse/evaluate_inverse_hierarchical_template.json
```

Defaults sample 8 plans and 4 layouts per plan: 32 raw candidates and 32 exact
HONF calls. With correction enabled, every lineage receives at most one
bounded proposal and one additional HONF call. A proposal is accepted only if
exact geometry remains valid and exact request violation improves.

The result preserves four meanings:

- `raw_unguided`: generated and verified candidates before correction;
- `corrected`: every one-pass proposal, including worse proposals;
- `accepted_one_pass`: one raw-or-corrected representative per lineage;
- `final_ranked`: up to `top_k` representatives selected by request violation,
  geometry, plan consistency, and diversity-aware tie breaking.

Raw generator success is reported before reranking. Evaluation writes summary
JSON, all/top CSVs, compressed candidate arrays, a SHA-inventoried manifest,
population comparison plots, and detailed field/layout/mechanism plots for top
candidates.

The equivalent public API attaches a case runtime because exact verification
and correction require ThermalChannel physics:

```python
from honf_inverse_core.models.hierarchical_inverse import HierarchicalInverseDesigner
from honf_inverse_core.training.checkpointing import load_inverse_checkpoint
from channelthermal.inverse.context import load_context
from channelthermal.inverse.evaluation.candidate_evaluator import ThermalChannelCandidateEvaluator
from channelthermal.inverse.request import make_request_codec
from channelthermal.inverse.verifier import FrozenThermalChannelVerifier
from channelthermal.workflows.evaluate_inverse_hierarchical import normalizers_from_checkpoint

inverse_path = ".../best_corrected_model.pt"
checkpoint = load_inverse_checkpoint(inverse_path)
normalizers = normalizers_from_checkpoint(checkpoint)
request = make_request_codec(normalizers.functional).load(".../balanced_cooling.json")
context = load_context(".../reference_operating_context.json")

designer = HierarchicalInverseDesigner.load(inverse_path, device="cuda:0")
frozen = FrozenThermalChannelVerifier(
    ".../best_predicted_model.pt",
    device="cuda:0",
    dataset_path=".../packed_dataset.h5",
)
runtime = ThermalChannelCandidateEvaluator(designer, frozen, normalizers)
designer.attach_verifier(runtime)
result = designer.sample_candidates(
    request=request,
    context=context,
    num_plans=8,
    layouts_per_plan=4,
    correct_once=True,
    top_k=8,
    seed=0,
)
serializable = result.to_dict()
```

### 6.7 Initial inverse limitations

Schema/model v1 fixes the forward hyperedge count, supports at most 12 modules,
generates only centers and heat powers, and assumes one module family. The
geometry decoder has one analytic fallback rather than a general constraint
solver. Stage-four differentiable probes are coarser training surrogates; final
evaluation uses the exact frozen verifier. The evaluator currently aborts a
request if a candidate's forward call fails instead of retaining a failed
candidate record. There is no iterative correction, feasibility guarantee, or
claim beyond the frozen forward surrogate's accuracy.

The bounded audit entry point can study request sensitivity, plan realization,
layout diversity, and correction acceptance without launching a formal run:

```bash
python Case_ThermalChannel/scripts/inverse/audit_inverse_hierarchy.py --help
```

## 7. Runs, checkpoints, and saved results

Managed output is grouped by case and workflow:

```text
Trained_Results/ThermalChannel/
├── Local_Module_Runs/thermal_disk/Run_<id>_<timestamp>_<name>/
├── HONF_Forward_Runs/Run_<id>_<timestamp>_<name>/
│   ├── configs/                       source/resolved/provenance JSON
│   ├── environment/                   software and source-state snapshots
│   ├── checkpoints/                   canonical checkpoint aliases
│   ├── metrics/, plots/, logs/
│   ├── run_manifest.json
│   ├── best_model.pt, best_predicted_model.pt, latest_model.pt, ...
│   └── eval_global/<timestamp>/...
├── Inverse_Dataset_Builds/            generated inverse HDF5/diagnostics
├── HONF_Inverse_Runs/                 staged inverse runs and evaluations
└── Baselines/
```

Run directories use `Run_<four-digit-id>_<local-timestamp>_<safe-name>` and a
UUID in `run_manifest.json`. Status records created/running/completed/failed
transitions. Failures include the exception and last recoverable epoch.
Completion inventories checkpoint selectors and mirrors workflow-compatible
root files into canonical `checkpoints/`, `metrics/`, and `plots/` subtrees.

Generated results, datasets, checkpoints, local resource maps, diagnostic
configs, and inverse evaluation artifacts are ignored by Git. Source code,
schemas, maintained templates, request examples, and documentation remain
trackable.

## 8. Testing and validation

Run the full root and ThermalChannel suites in the maintained environment:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n ModularDT pytest -q tests Case_ThermalChannel/tests
```

Focused forward diagnostics:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n ModularDT \
  python Case_ThermalChannel/tests/check_global_modes.py --device cuda:0
CUDA_VISIBLE_DEVICES=0 conda run -n ModularDT \
  python Case_ThermalChannel/tests/check_honf_hardening.py --device cuda:0
conda run -n ModularDT \
  python Case_ThermalChannel/tests/check_hypergraph_plan_stability.py
```

Focused inverse regressions:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n ModularDT pytest -q \
  tests/test_inverse_*.py tests/test_joint_corrector.py \
  tests/test_layout_flow.py tests/test_plan_flow.py tests/test_request_set_encoder.py \
  Case_ThermalChannel/tests/test_inverse_*.py
```

`VALIDATION.md` records forward migration parity and bounded smoke evidence.
The inverse evidence in `CHANGELOG.md` is explicitly diagnostic, not a claim of
production convergence.

## 9. Extending HONF to another physical case

A new case is a physical-data/adapter/workflow plugin, not a fork of the HONF
core and not a new `if case == ...` branch in the entry points.

### 9.1 Minimum forward-case implementation

1. Create an installable sibling package such as `Case_MyPhysics/`.
2. Define a factory like `myphysics.plugin:create_plugin` implementing
   `honf_runtime.case_protocol.CasePlugin`.
3. Add a versioned dataset manifest, exact schema documentation, an ignored
   machine-local location map, and reproducible train/validation readers.
4. Adapt case tensors to `honf_forward_core.config.BatchData`: module centers,
   features, presence mask, context, environment tokens/features, queries, and
   targets.
5. Wrap the reusable HONF core only where case-specific preprocessing,
   auxiliary outputs, or coupling are required.
6. Keep physical losses, metrics, visualization, and post-processing in the
   case package.
7. Add a strict case profile and a core launch profile selecting the plugin,
   model family, logical dataset ID, and architecture.
8. Prove installation, dry-run, batch adaptation, inference, loss, checkpoint
   reconstruction, and evaluation with synthetic contract tests.

If the case has a reusable local physics component, expose it as a
`LocalModuleSpec` with stable input/port/query/target schemas, model/checkpoint
factories, coupling adapter, freeze policy, and embedding policy.

### 9.2 Adding inverse support for the new case

Reuse `honf_inverse_core` only after the forward contract is stable. The case
must then supply:

1. a small versioned functional vocabulary and strict request codec;
2. a versioned operating-context contract;
3. physical design canonicalization, generated-design decoding, and exact
   geometry checks;
4. a compact plan derived from that forward model's canonical organizer plan;
5. exact functional evaluation in physical units;
6. a frozen autonomous forward verifier that returns `G_hat` and requested
   physical outputs without borrowing hidden source-case inputs;
7. split-before-augmentation dataset construction and diagnostics;
8. case-owned staged training/evaluation adapters, plots, and tests; and
9. provenance checks tying inverse data/checkpoints to the exact forward
   checkpoint and schema versions.

Do not runtime-import an older demo tree or place case physics in the reusable
inverse package. Start with the smallest request vocabulary and one module
family, then expand only after end-to-end verification is reliable.

The detailed plugin checklist is in `docs/case_plugin.md`. Configuration,
checkpoint, result, and model-family rules are in `docs/configuration.md`,
`docs/checkpoints.md`, `docs/results.md`, and `docs/model_family.md`.

## 10. Further reading and project status

- `Model_Explain.md`: detailed forward/inverse mathematics and tensor meaning.
- `Case_ThermalChannel/Dataset/PHYSICS_AND_DATA.md`: physical and HDF5 contract.
- `VALIDATION.md`: numerical migration and workflow validation evidence.
- `CHANGELOG.md`: current forward and inverse implementation history.
- `INVERSE_CODING.md`: development roadmap and design decisions for the first
  inverse hierarchy.
- `SUPPORT.md`: compatibility and support expectations.

The package version is `0.1.0`. The current license is deliberately
all-rights-reserved because the parent repository did not provide an
open-source license to carry forward. Replace it with an owner-approved license
and update the placeholder repository URL in `CITATION.cff` before public
distribution.
