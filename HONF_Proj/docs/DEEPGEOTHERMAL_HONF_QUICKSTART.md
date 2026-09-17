# DeepGeoThermal: quick start for a new HONF case

This is a coworker-and-AI handoff for adding a new `DeepGeoThermal` dataset and case package while reusing the case-neutral HONF implementation. The intended first comparison is:

- a **1401-like legacy HONF**: `forward_architecture="legacy_honf"`, fixed `K=6`, context-fusion decoder;
- an **1804-like dense baseline**: `forward_architecture="dense_pairwise_field"` with dense module/environment preparation and receiver reads.

These names describe architecture families, not checkpoint transfer or an exact reproduction of the ThermalChannel experiments. Start both models from scratch unless a separately validated, shape-compatible initialization is deliberately requested.

## 1. Recommended ownership boundary

Keep DeepGeoThermal physics in a new installable case package. Reuse the HONF core without adding `if case_id == "DeepGeoThermal"` branches to the core or runtime.

```text
raw DeepGeoThermal data
    │
    ├─ case-owned validation, split, normalization and query sampling
    ▼
DeepGeoThermal adapter → generic BatchData
    │
    ├─ legacy_honf              # 1401-like
    └─ dense_pairwise_field     # 1804-like
    ▼
case-owned loss, physical inverse transform, metrics and plots
```

Use these current implementations as templates:

- [`Case_WindFarm`](../Case_WindFarm/src/windfarm/) for a 3-D dataset, geometry-only environment tokens, dynamic module padding, `BatchData`, a two-backend wrapper, and native-field training/evaluation;
- [`Case_ThermalChannel`](../Case_ThermalChannel/src/channelthermal/) only for genuine port/local-surrogate coupling;
- [`docs/case_plugin.md`](case_plugin.md) for the plugin boundary;
- [`honf_runtime.case_protocol`](../src/honf_runtime/case_protocol.py) for `CasePlugin` and `LocalModuleSpec`;
- [`honf_forward_core.config.BatchData`](../src/honf_forward_core/config.py) for the reusable input contract.

The smallest good first milestone is a field-only DeepGeoThermal package that can dry-run, execute one real backward/update batch with each architecture, and evaluate one held-out case. Add a local surrogate and ports only when they correspond to a real physical interface.

## 2. Pull the maintained branch and create a work branch

The maintained integration branch is `agent/honf-core-next`. It was verified against `origin` at commit `53e2e54` on 2026-09-15; always fetch again rather than assuming that snapshot is still current.

For a new checkout:

```bash
git clone git@github.com:cosmos2w/ModularDT.git
cd ModularDT
git fetch origin
git switch --create agent/honf-core-next --track origin/agent/honf-core-next
git pull --ff-only
git switch -c agent/deepgeothermal
cd HONF_Proj
```

For an existing checkout:

```bash
git status
git fetch origin
git switch agent/honf-core-next
git pull --ff-only
git switch -c agent/deepgeothermal
cd HONF_Proj
```

Do not discard a dirty worktree to make these commands succeed. Commit, stash, or move the work intentionally first.

## 3. Create and verify the environment

Python 3.10 or newer is required. Use the existing `ModularDT` environment when available:

```bash
conda activate ModularDT
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pip install -e '.[dev]'
```

After creating the new case package, install it too:

```bash
python -m pip install -e ./Case_DeepGeoThermal
python -c "from deepgeothermal.plugin import create_plugin; print(create_plugin().case_id)"
```

If a fresh environment is needed, create a Python 3.10+ Conda environment, install the correct PyTorch build for that machine, and then install the two editable packages above. Do not copy another machine's CUDA-specific wheel blindly.

## 4. Scaffold the case package

Create this initial shape:

```text
Case_DeepGeoThermal/
├── pyproject.toml
├── README.md
├── configs/
│   ├── case_config.schema.json
│   └── forward_field.json
├── Dataset/
│   ├── PHYSICS_AND_DATA.md
│   ├── dataset_manifest.json
│   ├── dataset_locations.example.json
│   └── dataset_locations.local.json       # ignored, never commit
├── scripts/
│   └── inspect_deepgeothermal.py
├── src/deepgeothermal/
│   ├── __init__.py
│   ├── plugin.py
│   ├── io.py
│   ├── splits.py
│   ├── normalization.py
│   ├── adapter.py
│   ├── data.py
│   ├── environment.py
│   ├── model.py
│   ├── loss.py
│   ├── ports.py                          # only if physically justified
│   ├── local_surrogate/                  # optional second milestone
│   └── workflows/
│       ├── train_forward.py
│       └── evaluate_forward.py
└── tests/
```

Also add two complete core profiles:

```text
src/config_core/forward/deepgeothermal_legacy_k6.json
src/config_core/forward/deepgeothermal_dense_pairwise.json
```

Let the coworker's AI copy structure and signatures from WindFarm, then rename domain concepts and delete assumptions that do not apply. Do not copy turbine constants, ThermalChannel feature columns, 2-D circular ports, or their loss terms.

## 5. Lock the data contract before implementing the model

Write `Dataset/PHYSICS_AND_DATA.md` first. It should answer:

1. What is one statistically independent case: reservoir realization, geology, operating schedule, time window, or something else?
2. Which quantities are known at inference time?
3. Which entities are HONF modules: wells, fractures, completion segments, heat exchangers, or another repeated object?
4. Which continuous fields are targets, with units and channel order?
5. Are grids shared, native/ragged, structured, or unstructured?
6. What physical measure should the loss approximate: volume, area, interface area, or an explicitly chosen sampling distribution?
7. Which identifier prevents leakage between train, validation, and test?

### Suggested first-pass mapping

| Physical concept | Generic HONF field | Expected shape |
|---|---|---|
| Module/well centres in fixed physical units | `module_centers` | `[B,M,3]` |
| Active-module mask | `module_present` | `[B,M]` |
| Geometry and operating controls known at inference | `module_features` | `[B,M,Fm]` |
| Reservoir/global controls known at inference | `global_context` | `[B,Fg]` |
| Geometry/boundary support tokens | `env_coords`, `env_features` | `[B,E,3]`, `[B,E,Fe]` |
| Environmental quadrature/measure | `env_weights` | `[B,E]` |
| Sampled receiver coordinates | `query_xy` | `[B,Q,3]` |
| Known receiver geometry/boundary descriptors | `query_features` | `[B,Q,Fq]` |
| Standardized target fields | `target_field` | `[B,Q,C]` |

`field_dim=C` must equal the declared target-channel count. Keep a single canonical channel order, for example `[temperature, pressure]`; do not silently change it between preprocessing, loss, checkpoint metadata, and evaluation.

### Preprocessing rules worth enforcing in code

- Split by the independent physical group, not by rows or sampled points. All times/controls derived from one reservoir realization should stay together unless the scientific question explicitly defines a temporal extrapolation split.
- Fit input and target normalization on the training partition only. Save the statistics and their schema/hash; use the same frozen transform for both architectures.
- Never place target fields, solved-field summaries, future controls, or validation/test statistics in module, environment, global, or query features.
- Preserve native coordinates and masks. For large or ragged volumes, use memory maps/offsets and sample target points without loading or resampling the entire dataset.
- If sampling combines bulk-volume, near-well, boundary, or high-gradient strata, return explicit loss weights. The weighted loss should match the stated physical or evaluation measure.
- Generate deterministic split indices and fixed validation queries. Training queries may change by epoch from a recorded seed.
- Dynamically pad modules only to the maximum `M` in the current batch; zero padded features and test padding invariance.

The preprocessing command can follow the WindFarm pattern:

```bash
python Case_DeepGeoThermal/scripts/inspect_deepgeothermal.py \
  --dataset-root /path/to/deepgeothermal \
  --output-dir Case_DeepGeoThermal/Dataset/derived/forward_v1
```

It should emit, at minimum, split indices, normalization statistics, sampling metadata, a compact audit JSON/CSV, and finite/shape/unit checks. Generated arrays and local paths stay ignored by Git; schemas, manifests, and example maps are committed.

## 6. Design environment tokens and coordinate scaling

The case adapter owns `env_coords`, `env_features`, and optional `env_weights`. They should describe geometry, boundaries, geology descriptors, and operating conditions available at inference—not a downsampled target field.

For the first 3-D baseline:

- use `spatial_dim: 3`, nonperiodic geometry, and no periodic axes;
- choose a fixed per-axis `coordinate_scale=[Lx,Ly,Lz]` from documented physical reference lengths, not test-set extrema;
- start with roughly 128–512 environment tokens and increase only after a measured accuracy/cost study;
- attach quadrature mass when token cells represent unequal physical volume;
- encode variable domain bounds/distances in environment/query features instead of forcing every case into a common cube.

The existing 3-D reusable core currently supports `legacy_honf` and `dense_pairwise_field`, which is exactly the requested pair.

## 7. Decide whether DeepGeoThermal really has ports

HONF does not require ports. A well location and its controls can be a module without being a port-coupled local model.

Use ports only when there is a genuine interface across which the global field supplies conditions to a reusable local solver/surrogate and receives a local response. Possible geothermal examples include a borehole-wall interface, completion segments, or a heat-exchanger boundary. The domain expert must choose the interface; the AI should not invent it.

### If the first study is field-only

Leave out `ports.py`, `local_surrogate/`, and P0/P1 refinement. Build the wrapper like [`WindFarmForwardModel`](../Case_WindFarm/src/windfarm/model.py):

- convert case batches to `BatchData`;
- select `HONFNeuralField` for `legacy_honf`;
- select `InterfaceFieldCore` for `dense_pairwise_field`;
- expose `prepare_case`, chunked `decode`, `forward`, and physical inverse transformation;
- materialize lazy layers on a real geometry batch before creating the optimizer.

This is the recommended first comparison unless a validated local-module dataset and physical coupling contract already exist.

### If a real local module is required now

Define a case-owned `LocalModuleSpec` and keep the port schema explicit. For a 3-D interface, a useful abstract contract is:

```text
port coordinates       [B,M,P,3]
port local coordinates [B,M,P,Dlocal]
port normals/tangents   [B,M,P,3] or a documented local frame
port present mask       [B,M,P]
predicted conditions    [B,M,P,Cport]
local response latent   [B,M,Hlocal]
local interface output  [B,M,P,Cinterface]
```

Define and test:

- fixed port ordering and orientation;
- units and sign convention for pressure, temperature, mass/heat flux, and normal direction;
- how variable port counts are masked;
- which condition channels the global model predicts;
- which local outputs are fused back into module state;
- whether one refinement pass is physically meaningful;
- conservation/continuity losses and their quadrature weights;
- autonomous inference with predicted ports, with teacher ports restricted to training/diagnostics.

The typical coupled flow is:

```text
P0: prepare global state → read at physical ports → predict port conditions
    → frozen local surrogate → local fields/interface response → fuse response
P1: optional re-prepare → probe outside/global field → refine port conditions
    → rerun local surrogate → fuse refreshed response
P2: final prepare → decode requested global field coordinates
```

Use [`interface_field_coupling.py`](../Case_ThermalChannel/src/channelthermal/interface_field_coupling.py) as a control-flow reference, not as reusable geothermal physics. In particular, its `[theta, cos(theta), sin(theta), T_env, h]` disk-port contract is ThermalChannel-specific and must not be copied into a 3-D geothermal interface.

Train and validate the local surrogate separately, freeze it in the parent runs initially, and embed its state/normalization in the parent checkpoint. If no trustworthy local checkpoint exists, do not hide a randomly initialized local surrogate inside the two baseline runs.

## 8. Implement the plugin and wrapper

`deepgeothermal.plugin:create_plugin` must return an object with:

```python
case_id = "DeepGeoThermal"
display_name = "Deep geothermal field"
version = "0.1.0"

def validate_config(bundle): ...
def inspect_launch(bundle, request): ...
def train(bundle, request, *, run_dir): ...
def evaluate(bundle, request): ...
```

`validate_config` should reject unknown keys and assert the actual domain contract: field count, spatial dimension, target order, split/group key, positive batch/query sizes, coordinate scale, allowed architectures, and port/local-checkpoint requirements.

`inspect_launch` should open enough of the real resource to prove that paths, schemas, split artifacts, and local checkpoints are usable, without scanning a multi-terabyte dataset. Return human-readable facts for `--dry-run`.

The model wrapper should keep only the two core call sequences. All target transformation, port logic, loss weighting, physical metrics, and plotting remain in the case package. The generic core should receive tensors and return learned field context; it should not know what a well, reservoir, or geothermal port means.

## 9. Create matched core profiles

Start by adapting the two proven 3-D WindFarm profiles:

- [`windfarm_classic_k6_b16_q8192.json`](../src/config_core/forward/windfarm_classic_k6_b16_q8192.json)
- [`windfarm_dense_pairwise_b16_q8192.json`](../src/config_core/forward/windfarm_dense_pairwise_b16_q8192.json)

Replace their case ID/config, field count, coordinate scale, module radius/reference scale, environment sizes, batch/query budget, run IDs, and names. Keep the comparison controlled: dataset, split, normalization, sampled queries, target/loss weights, seed, training budget, and checkpoint rule must match.

Recommended starting values are baselines, not truths:

| Setting | Legacy K6 | Dense pairwise | Note |
|---|---:|---:|---|
| `hidden_dim` | 256 | 256 | Keep matched initially |
| `num_hyperedges` | 6 | n/a | 1401-like fixed bottleneck |
| `message_hidden_dim` | n/a | 128 | Dense typed messages |
| `attention_heads` | n/a | 4 | Must divide hidden dimensions where required |
| `coarse_latent_count` | n/a | 8 | Shared coarse route in dense backend |
| Fourier frequencies | 4 | 4 | Position/query/relative geometry |
| dropout | 0 | 0 | Match historical baselines |
| learning rate | 3e-4 | 3e-4 | AdamW |
| weight decay | 1e-5 | 1e-5 | Keep matched |
| gradient clip | 1.0 | 1.0 | Record pre/post-clip diagnostics |
| AMP | off initially | off initially | Enable only after numerical comparison |
| receiver chunk | case-owned 128 initially | 128 initially | Increase after memory probe |
| seed | 0 | 0 | Add more seeds after pipeline validity |

Important configuration detail: **do not add `interface_model` to the legacy profile**. Put a legacy receiver-chunk execution setting in the case dataset/runtime configuration, as the WindFarm implementation does. The dense profile owns its `interface_model.receiver_chunk_size`.

For initial engineering, use a small real workload such as `B=2`, `Q=256`, and a modest `E`. Before full launch, scale toward the intended workload only after one real forward/backward/update measurement for both models. Dense and legacy can have very different memory bottlenecks.

## 10. Tests and definition of “ready to launch”

At minimum, add tests for:

- manifest/location resolution and missing-resource failure;
- group-safe split determinism and no overlap;
- training-only normalization and inverse transformation;
- dataset shapes, finite values, masks, units, and query/target alignment;
- module permutation and padding invariance;
- environment-weight mass accounting;
- both wrapper backends on the same synthetic and real mini-batch;
- query chunk consistency;
- weighted loss against a direct reference calculation;
- checkpoint round-trip and evaluation artifact creation;
- port ordering, normals, masking, sign convention, and predicted-only inference, if ports exist.

Run:

```bash
pytest -q Case_DeepGeoThermal/tests
ruff check Case_DeepGeoThermal/src Case_DeepGeoThermal/tests
python -m compileall -q Case_DeepGeoThermal/src
git diff --check
```

Then run a disposable real batch for each model and record peak memory, loss, gradient finiteness, pre-clip norm, and a nonzero sampled parameter update. A configuration parse or forward-only smoke test is not enough evidence to start a long run.

## 11. Dry-run, bounded smoke runs, and full launches

All commands run from `HONF_Proj`. Choose unused run IDs; the examples below are placeholders.

First validate without creating run directories:

```bash
python train.py \
  --config project://src/config_core/forward/deepgeothermal_legacy_k6.json \
  --workflow forward --run-id 2200 --run-name deepgeothermal_legacy_k6 \
  --device cuda:0 --dry-run

python train.py \
  --config project://src/config_core/forward/deepgeothermal_dense_pairwise.json \
  --workflow forward --run-id 2201 --run-name deepgeothermal_dense_pairwise \
  --device cuda:0 --dry-run
```

Check that both summaries show the same dataset, split artifacts, channel order, normalization, query sampling, optimizer budget, and case wrapper. Only the architecture-specific blocks, run identity, and unavoidable parameter/resource counts should differ.

Optional bounded lifecycle runs consume their IDs, so use throwaway IDs such as 2290/2291:

```bash
python train.py \
  --config project://src/config_core/forward/deepgeothermal_legacy_k6.json \
  --workflow forward --run-id 2290 --run-name deepgeothermal_legacy_smoke \
  --epochs 1 --max-train-batches 1 --max-val-batches 1 \
  --device cuda:0 --yes

python train.py \
  --config project://src/config_core/forward/deepgeothermal_dense_pairwise.json \
  --workflow forward --run-id 2291 --run-name deepgeothermal_dense_smoke \
  --epochs 1 --max-train-batches 1 --max-val-batches 1 \
  --device cuda:0 --yes
```

After the tests and real memory/update probe pass, launch the full matched runs. On separate physical GPUs, each process should still use logical `cuda:0`:

```bash
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --config project://src/config_core/forward/deepgeothermal_legacy_k6.json \
  --workflow forward --run-id 2200 --run-name deepgeothermal_legacy_k6 \
  --device cuda:0 --yes

CUDA_VISIBLE_DEVICES=1 python -u train.py \
  --config project://src/config_core/forward/deepgeothermal_dense_pairwise.json \
  --workflow forward --run-id 2201 --run-name deepgeothermal_dense_pairwise \
  --device cuda:0 --yes
```

Use the coworker's normal supervised process manager or terminal multiplexer rather than relying on a fragile shell session. Preserve the literal commands, source commit, GPU mapping, and logs in an ignored launch record.

## 12. Monitoring and the first comparison

A started process is not a completed run. For each managed directory under `Trained_Results/DeepGeoThermal/HONF_Forward_Runs/`, inspect:

- `run_manifest.json` for source/config hashes and lifecycle status;
- `metrics.csv` for real epoch progress and finite train/validation metrics;
- retained checkpoints and `latest_model.pt`/named best checkpoints;
- GPU memory and update time under the intended workload.

Select checkpoints using the declared validation metric only. Evaluate both architectures under both policies when useful:

1. the same exact terminal epoch;
2. each run's validation-selected checkpoint.

Keep a reserved test partition untouched until the design and checkpoint rule are frozen. Compare whole-field and per-channel physical errors, near-module/interface errors if relevant, matched inference time/memory, parameter count, and failure cases. Do not call model predictions simulator/CFD truth.

The case plugin may expose commands such as:

```bash
python evaluate.py \
  --config project://src/config_core/forward/deepgeothermal_legacy_k6.json \
  --workflow forward --run-id 2200 --checkpoint best_field --device cuda:0

python evaluate.py \
  --config project://src/config_core/forward/deepgeothermal_dense_pairwise.json \
  --workflow forward --run-id 2201 --checkpoint best_field --device cuda:0
```

The exact checkpoint selector is case-owned; implement and test it before relying on these example commands.

## 13. Compact coworker/AI execution checklist

- [ ] Pull `origin/agent/honf-core-next` with `--ff-only`; create a feature branch.
- [ ] Document the physical sample, inference-known inputs, targets, units, and leakage group.
- [ ] Commit manifest/schema/example map; keep raw data and local paths ignored.
- [ ] Build deterministic preprocessing, train-only normalization, and weighted query sampling.
- [ ] Adapt every batch to canonical `BatchData` and test padding/permutation invariance.
- [ ] Decide explicitly: field-only modules, or genuine port-coupled local physics.
- [ ] Implement one thin wrapper over `HONFNeuralField` and `InterfaceFieldCore`.
- [ ] Add matched legacy-K6 and dense core profiles; no `interface_model` in legacy.
- [ ] Pass focused tests and one real backward/update resource probe for each architecture.
- [ ] Review both `--dry-run` summaries, then launch fresh matched runs with unused IDs.
- [ ] Confirm progress from `run_manifest.json` and `metrics.csv`, not from process existence.
- [ ] Freeze the checkpoint rule before consuming the reserved test set.

If the coworker and AI preserve these boundaries, most work stays in `Case_DeepGeoThermal`; the general HONF code should need little or no modification.
