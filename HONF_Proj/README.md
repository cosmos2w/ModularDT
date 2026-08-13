# HONF Project

`HONF_Proj` is the installable, extensible home of the Hypergraph Operator
Neural Field (HONF). The reusable forward model and runtime are independent of
the physical application. Each application supplies an installed case plugin;
the included `Case_ThermalChannel` plugin preserves the complete coupled
thermal-channel workflow, including its Stage-A disk surrogate, Stage-B global
training, evaluation, plots, routing diagnostics, hypergraph-plan export, and
multi-model comparison.

The inverse namespace is reserved but intentionally has no implementation in
this release.

## Repository map

```text
HONF_Proj/
├── train.py, evaluate.py          generic entry points
├── src/
│   ├── honf_forward_core/         reusable HONF model/organizer/decoder
│   ├── honf_inverse_core/         documented placeholder
│   ├── honf_runtime/              config, plugins, paths, runs, checkpoints
│   └── config_core/forward/       maintained launch profiles
├── Case_ThermalChannel/
│   ├── src/channelthermal/        case model, data, Stage A, workflows, plots
│   ├── configs/                   case-owned settings
│   ├── Dataset/                   manifest, schemas, machine-local map
│   └── artifacts/                 untracked external checkpoints
├── Trained_Results/               generated runs and post-processing
├── docs/                          extension and artifact contracts
└── tests/                         generic and ThermalChannel regression tests
```

Core code has no dependency on `channelthermal`, HDF5, or Matplotlib. The case
package depends on the core/runtime, never the reverse. Top-level dispatchers
discover the case through the configured dotted plugin factory.

## Install

Python 3.10 or newer is required. PyTorch may be installed separately to match
the local CUDA or CPU platform.

```bash
cd HONF_Proj
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . -e ./Case_ThermalChannel
```

For tests and lint dependencies, install `-e '.[dev]'` for the root package.
Run commands from `HONF_Proj`; `project://` paths are anchored here and do not
depend on the shell's current directory.

## Configure the datasets

The HDF5 data stay outside Git. Copy the example location map and edit both
values:

```bash
cp Case_ThermalChannel/Dataset/dataset_locations.example.json \
   Case_ThermalChannel/Dataset/dataset_locations.local.json
```

The two logical resources are:

| ID | Purpose | Expected cases |
|---|---|---:|
| `thermal_disk_local_v1` | Stage-A local disk surrogate | 1,034 |
| `thermal_channel_global_v1` | Stage-B coupled channel field | 690 |

Paths in the local map may be absolute or `project://` paths. The map is
ignored by Git. If it is absent, `HONF_DATA_ROOT` may point to the common data
root containing `Processed_LocalModule_Dataset/` and
`Processed_ChannelThermal_Dataset/`. Validate file size, HDF5 keys, and the full
SHA-256 fingerprint with:

```bash
python tools/inspect_dataset.py --dataset-id thermal_disk_local_v1 --sha256
python tools/inspect_dataset.py --dataset-id thermal_channel_global_v1 --sha256
```

Optional human-readable symlinks can be generated under `Dataset/links/`:

```bash
python tools/link_datasets.py
```

See `Case_ThermalChannel/Dataset/PHYSICS_AND_DATA.md` and its `schemas/`
directory for physical definitions, feature order, array shapes, splits, and
normalization.

## Always dry-run first

A dry-run validates the core/case profiles, dataset contract, dependent local
checkpoint, run ID, and output destination. It writes nothing.

```bash
python train.py \
  --config src/config_core/forward/enhanced_honf_pairwise.json \
  --local-checkpoint /path/to/thermal_disk.pt \
  --device cuda:0 --dry-run
```

Without `--yes`, an interactive launch prints the same summary and waits for
`yes` (or `y`). Non-interactive jobs must use `--yes`. CLI overrides
are saved as provenance; a duplicate numeric run ID is rejected instead of
overwriting an earlier result.

## ThermalChannel training

### 1. Train the Stage-A local module

```bash
python train.py \
  --config src/config_core/forward/local_module_thermal_disk.json \
  --device cuda:0 --dry-run

python train.py \
  --config src/config_core/forward/local_module_thermal_disk.json \
  --device cuda:0 --yes
```

The run appears under
`Trained_Results/ThermalChannel/Local_Module_Runs/thermal_disk/`. Use its
`best_model.pt` or `latest_model.pt` for Stage B. Either pass that path with
`--local-checkpoint`, or symlink/copy the selected file to
`Case_ThermalChannel/artifacts/thermal_disk.pt` so the default case config can
find it.

### 2. Train the two Stage-B global modes

The maintained profiles preserve the same core capacity: 24 by 8 environment
tokens, 6 hyperedges, hidden width 256, and all local coupling/physical loss
terms. They differ only in the decoder mechanism being studied.

```bash
# Global plus local near-field decoder
python train.py \
  --config src/config_core/forward/hyper_plus_global_near.json \
  --local-checkpoint /path/to/stage_a/best_model.pt \
  --device cuda:0 --yes

# Enhanced HONF pairwise decoder
python train.py \
  --config src/config_core/forward/enhanced_honf_pairwise.json \
  --local-checkpoint /path/to/stage_a/best_model.pt \
  --device cuda:0 --yes
```

Use `--epochs`, `--run-id`, or `--run-name` only as explicit experiment
overrides. For a wiring smoke test, add `--epochs 1 --max-train-batches 1
--max-val-batches 1` and choose an unused run ID.

Established ablations are strict overlays under
`src/config_core/forward/experiments/`. For example, the no-Stage-A fallback
run needs no local checkpoint:

```bash
python train.py \
  --config src/config_core/forward/enhanced_honf_pairwise.json \
  --experiment-overlay src/config_core/forward/experiments/global_only.json \
  --run-id 0003 --device cuda:0 --dry-run
```

Interrupted managed runs resume in place from `latest_model.pt`:

```bash
python train.py \
  --config src/config_core/forward/local_module_thermal_disk.json \
  --resume-checkpoint Trained_Results/ThermalChannel/Local_Module_Runs/thermal_disk/Run_0001_.../latest_model.pt \
  --device cuda:0 --yes
```

Resume validates the case/model/workflow, immutable model/data/loss sections,
feature schemas, and normalization before loading. Current checkpoints restore
model, optimizer, AMP scaler, epoch, best metrics, and Python/NumPy/Torch/CUDA
random streams. Historical checkpoints lacking optimizer/RNG state remain
loadable but print an explicit fresh-state warning.

### 3. Evaluate and post-process

```bash
# Forward run selected by numeric ID
python evaluate.py \
  --config src/config_core/forward/enhanced_honf_pairwise.json \
  --workflow forward --run-id 0002 --checkpoint best_predicted \
  --device cuda:0 --case-index 0 --organization-view all \
  --routing-view summary --export-hypergraph-plan

# Stage-A local run
python evaluate.py \
  --config src/config_core/forward/local_module_thermal_disk.json \
  --workflow local_module --run-id 0001 --checkpoint best \
  --device cuda:0 --case-index 0

# Compare any number of run IDs or explicit checkpoint paths
python evaluate.py \
  --config src/config_core/forward/enhanced_honf_pairwise.json \
  --workflow compare --Run_ID 0001 --Run_ID 0002 \
  --checkpoint-selector best_predicted \
  --label hyper_plus_global_near --label enhanced_honf_pairwise \
  --device cuda:0
```

Forward evaluation supports `teacher`, `predicted`, `mixed`, or `both` port
conditions, chunked full-grid decoding, physical/matrix/schematic organizer
views, optional dense routing maps, and compact NPZ hypergraph-plan export.
Run `python evaluate.py --help` and append case options after the generic ones.
Missing named checkpoints and ambiguous Run IDs fail by default. Pass an exact
checkpoint path to disambiguate; use `--allow-checkpoint-fallback` only when a
documented `best_predicted` to `best` substitution is intentional.

## Results and checkpoints

Generated runs follow this structure:

```text
Trained_Results/<CaseID>/
├── HONF_Forward_Runs/Run_<id>_<timestamp>_<name>/
│   ├── configs/{core_source,case_source,cli_overrides,resolved_config}.json
│   ├── configs/config_provenance.json and environment/{software,source_state}.json
│   ├── run_manifest.json
│   ├── checkpoints/, metrics/, plots/, logs/
│   ├── best_model.pt, best_predicted_model.pt, latest_model.pt, ... (compatibility aliases)
│   └── eval_global/<timestamp>/...
├── Local_Module_Runs/<module_id>/Run_<id>_.../
│   └── ... plus eval_local/<timestamp>/...
├── HONF_Inverse_Runs/
└── Baselines/
```

Every run gets a UUID, configuration hash, source and resolved profiles,
status, checkpoint inventory, and evaluation children. Checkpoints embed the
effective model configuration, training normalization, and frozen Stage-A
state needed for standalone global evaluation. `.pt` files are trusted local
artifacts and are loaded with pickle semantics; do not load untrusted files.

## Add a new physical case

No edit to `train.py`, `evaluate.py`, `honf_forward_core`, or `honf_runtime` is
required.

1. Create an installable case package beside `Case_ThermalChannel`.
2. Implement a factory such as `mycase.plugin:create_plugin` satisfying
   `honf_runtime.case_protocol.CasePlugin`.
3. Keep the physical dataset reader, batch adapter, field names, losses,
   local-module definitions, and visualizations inside that package.
4. Add a committed dataset manifest and schema, plus an ignored machine-local
   location map.
5. Add a case JSON profile naming the factory and case-owned settings.
6. Copy a core launch profile, change only `case.id`, `case.config`, the logical
   dataset ID, and any permitted core architecture/training settings.
7. Install the package, run a dry-run, then add synthetic contract and
   end-to-end smoke tests.

The core `BatchData` contract and detailed plugin checklist are documented in
`docs/case_plugin.md`. New model families/baselines are a separate axis from a
case and must not be implemented as case IDs.

## Configuration and testing

Configuration precedence is: committed core profile + referenced case profile,
then an optional strict experiment overlay, then allow-listed CLI overrides.
The exact effective configuration and its hash are stored before training.
Unknown sections fail early. JSON schemas are
provided under `src/config_core/schemas/` and `Case_ThermalChannel/configs/`.

```bash
pytest -q tests Case_ThermalChannel/tests
python Case_ThermalChannel/tests/check_global_modes.py --device cuda:0
python Case_ThermalChannel/tests/check_honf_hardening.py --device cuda:0
python Case_ThermalChannel/tests/check_hypergraph_plan_stability.py
```

`Model_Explain.md` gives the model equations and execution path. Additional
contracts are in `docs/`, and `Creat_Proj.md` records the design and validation
plan used for this migration.

## Release and support

The package version is `0.1.0`; changes are summarized in `CHANGELOG.md`,
compatibility/support expectations are in `SUPPORT.md`, and citation metadata
is in `CITATION.cff`. The current `LICENSE` is deliberately all-rights-reserved
because the parent repository did not provide an open-source license to carry
forward. Replace it with an owner-approved license and update the placeholder
repository URL in `CITATION.cff` before public distribution.
