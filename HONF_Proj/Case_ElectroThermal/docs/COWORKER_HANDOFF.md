# HET-MF coworker handoff

**Status checked 26 October 2026.** Standard and Expanded HDF5 archives are complete and reviewed, and the case adapter has a CPU forward/backward interface qualification. The qualification made zero optimizer updates; no trained field accuracy, organizer result, inverse result, or training-runtime saving has been measured.

## Shared paths

The shared read-only case repository is `/data/public/ModularDT-HETMF`; the dataset and evidence bundle is `/data/public/ModularDT-HETMF-Dataset/v1`, with `standard.h5` and `expanded.h5`. The H5 files and generated evidence are outside Git. Browse the shared repository read-only; for code changes, clone the GitHub repository into your own private worktree and use a working branch.

After the bundle has been published, run this from `HONF_Proj/Case_ElectroThermal` to create the case-local links used by report figures:

```bash
python -m hetmf link-shared --bundle /data/public/ModularDT-HETMF-Dataset/v1 --case-dir .
```

Keep checkpoints, logs, and generated evaluations under your own `/data2/<account>/ModularDT/` area by default. Use `/data/public/` for outputs only when deliberately publishing shared material. Keep all generated training outputs and figures out of Git.

## Problem and H5 contract

Each input describes steady current flow and Joule heating in a heterogeneous sheet connected to external voltage contacts. The electrical solve supplies heat to a thermal conduction and cooling solve. All inputs, targets, and coordinates are dimensionless; the archive is not calibrated to a particular device or physical unit system.

The prescribed inputs are variable-count contacts with columns `(x, y, radius, total_gamma, terminal_voltage)` and 144 material-control records with columns `(x, y, dx, dy, sampling_area, s, kappa, beta)`. Material controls are input features, not hidden fields. The target order is `potential`, `jx_cell`, `jy_cell`, `temperature_rise`, `q_total`, stored on the native 96 × 96 grid.

| Profile | TRAIN | DEV | TEST | Four challenge panels (combined) | Counterfactual audit | Total |
|---|---:|---:|---:|---:|---:|---:|
| Standard | 2,880 | 480 | 480 | 512 | 256 | 4,608 |
| Expanded | 5,760 | 960 | 960 | 1,024 | 512 | 9,216 |

The four panels cover environmental transfer, layout transfer, new pairings of known components, and extrapolation challenges including M10, M16, high contrast, and composite morphology. The 256-state Standard counterfactual audit contains controlled four-state comparisons. These populations are separate from ordinary TRAIN/DEV/TEST use. Expanded doubles Standard while retaining every Standard input and its split, so the profiles are nested rather than independent.

`HETMFDataset(..., normalize=True)` applies the archive's saved TRAIN-only statistics to target fields only. Inputs remain in canonical physical form; the case adapter maps contacts to `(radius, log(total_gamma), terminal_voltage)` and all 144 material records to `(log(s), log(kappa), log(beta))` before model interactions. Coordinates, measures, masks, and contact identity remain separate. Preserve the selected profile's normalizer with its checkpoint and invert it before reporting physical-space field errors or balances.

```python
from hetmf.training import HETMFTask, read_config

config = read_config("configs/train_six_hyperedge_standard.yaml")
task = HETMFTask("/path/to/standard.h5", config)
dataset = task.dataset("train", epoch=0)
batch = task.collate([dataset[0], dataset[1]])
dataset.close()
```

Sampled native queries are deterministic from the configured seed, epoch, stable case identity, and physical-input hash. Keep the same seed, epoch, case order, and query count across model arms; call `dataset.set_epoch(epoch)` when the training query draw changes. Full-grid field reporting uses all 9,216 native cell centers.

## Code map and model comparison

| Component | Responsibility |
|---|---|
| `HONF_Proj/src/honf_forward_core/` | Case-neutral HONF contracts, general interaction models, organization, and field decoding |
| `HONF_Proj/src/honf_runtime/field_training.py` | Shared field-training engine |
| `src/hetmf/training_data.py` | H5 reader, saved target normalization, and deterministic native-query sampling |
| `src/hetmf/honf_adapter.py` | HET-MF contact/material features and case-to-HONF batch mapping |
| `src/hetmf/training.py` | Case task binding for the shared training engine |
| `configs/train_*_standard.yaml` | Matched Standard-profile model and training settings |

| Model | What the comparison measures |
|---|---|
| **Six-hyperedge organization controlling contextualized member reads** | Six learned hyperedge slots feed the query/read path with contextualized contact and material members while preserving individual record identity. Forward computation is currently dense; six is a model setting, not a discovered physical group count. |
| **Dense MM/ME/EM interaction and direct field reads** | Dense contact–contact, contact–environment, and environment–contact interactions provide the comparator, with direct reads of individual source records for field queries. |
| **A2-D** | General full-access HONF baseline with no explicit six-hyperedge organizer. |

All three models use the same five-field output contract and prescribed physical inputs. Applied voltage and positive material coefficients enter their owning objects before interactions. Do not add Thermal StageA/port-specific branches, affine thermal heads, or pretrained Thermal checkpoints; HET-MF heat and temperature respond quadratically to terminal voltage.

The loader metadata includes provenance, split labels, family/root identities, solver data, and target-derived diagnostics. Keep those values out of model features. Literal contact IDs preserve identity and correspondence; they are not learned numeric features.

## Install and dry-run commands

Until this PR is merged into `main`, clone the preparation branch into your own account:

```bash
git clone --branch codex/hetmf-main-handoff https://github.com/cosmos2w/ModularDT.git ModularDT
cd ModularDT
```

After merge, a normal `main` checkout contains the same case. The host's shared checkout already uses the preparation branch; it is for read-only browsing. From the ModularDT repository root, install the general core and case packages, then work from the case directory:

```bash
python -m pip install -e HONF_Proj -e HONF_Proj/Case_ElectroThermal
cd HONF_Proj/Case_ElectroThermal
```

Set the input path and a per-user output root. The commands below are CPU dry runs and create no training trajectory:

```bash
DATASET=/data/public/ModularDT-HETMF-Dataset/v1/standard.h5
RUN_ROOT="/data2/$USER/ModularDT/hetmf/runs"

python -m hetmf train --config configs/train_six_hyperedge_standard.yaml --dataset "$DATASET" --output "$RUN_ROOT/six-hyperedge-standard-seed0" --device cpu --dry-run
python -m hetmf train --config configs/train_dense_pairwise_standard.yaml --dataset "$DATASET" --output "$RUN_ROOT/dense-pairwise-standard-seed0" --device cpu --dry-run
python -m hetmf train --config configs/train_a2d_standard.yaml --dataset "$DATASET" --output "$RUN_ROOT/a2d-standard-seed0" --device cpu --dry-run
```

No model is trained by this handoff. A real fit requires the trainer's explicit `--yes` flag and a separately selected experiment. The first full-data segment is capped at 100 epochs for each explicitly selected model; there is no automatic extension. Resume with the same config, dataset, and output directory using `--resume <run>/latest.pt`; use `--epochs` only for an explicitly chosen epoch horizon.

After an explicitly approved fit, use DEV for model selection and hold TEST for the selected checkpoint. The following example evaluates a sampled DEV panel, then all 9,216 TEST cell centers; the evaluator uses the run's `best_field.pt` by default.

```bash
DATASET=/data/public/ModularDT-HETMF-Dataset/v1/standard.h5
RUN_DIR="/data2/$USER/ModularDT/hetmf/runs/six-hyperedge-standard-seed0"
python -m hetmf evaluate --run "$RUN_DIR" --dataset "$DATASET" --split dev --device cpu --queries 1024
python -m hetmf evaluate --run "$RUN_DIR" --dataset "$DATASET" --split test --device cpu --queries 9216
```

Report channel-wise physical-unit errors with the normalizer inverted. Retain full loss history, update loss curves every 10 epochs, and keep the declared latest, best-field, and 100-epoch milestone checkpoints.

## Interpretation and comparison limits

Use the same Standard archive, initialization seed, case order, sampled queries, physical objective, schedule, and epoch budget for the three model arms. The shipped configs pin seed 0; preserve their shared settings when creating a comparison. Expanded is a separate data-size study. Its shared states also appear in Standard, so a Standard/Expanded comparison is not an independent-population test.

TRAIN has a known layout–morphology association: each individual TRAIN layout appears with environments from one morphology family. Report this limitation when interpreting generalization. The challenge panels deliberately reuse some TRAIN components and must not be added to the core training set or normalization fit.

`jx_cell` and `jy_cell` are cell-centered reconstructed targets. The reference's exact discrete conservation checks use native face currents and heat fluxes, which the five model target channels do not contain. Full-grid evaluation means querying every cell; it does not establish conservative face fluxes or exact balance for a learned field. Report per-field errors and residual/balance diagnostics separately, using the reference's native-face quantities for reference-solver conservation.

Learned hyperedges describe model organization, not physical causality. Claim sparse work or executor savings only from measured selected records, executed pair calculations, preparation/index cost, and complete runtime. Sampler diversity alone does not validate inverse designs; any inverse candidate needs fresh physical feasibility checks.

## Existing scientific report

The repository includes [HETMF_DATASET_AND_TRAINING_READINESS_REPORT.md](../HETMF_DATASET_AND_TRAINING_READINESS_REPORT.md), which records the physical equations, verified archive memberships, representative saved fields, numerical limits, and tested CPU adapter connection. The report's generated figures and HDF5 payloads remain outside Git; run `python -m hetmf link-shared --bundle /data/public/ModularDT-HETMF-Dataset/v1 --case-dir .` from the case directory to connect the report to the shared evidence.
