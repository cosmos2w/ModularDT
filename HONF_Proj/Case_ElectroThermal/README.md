# HET-MF: heterogeneous electro-thermal contact sheets

HET-MF is a dimensionless steady electro-thermal field-prediction case for the general HONF model. Each example supplies variable-count electrical contacts and 144 environmental material records; the model predicts potential, two cell-centered current components, temperature rise, and total Joule heat.

Shared read-only locations are `/data/public/ModularDT-HETMF` for the case repository and `/data/public/ModularDT-HETMF-Dataset/v1` for the dataset and evidence bundle. Browse the shared repository read-only. For code changes before this PR merges, clone `codex/hetmf-main-handoff` from GitHub into your own account and create a working branch; a plain `main` checkout does not contain this case until merge.

## Start here

| Need | Resource |
|---|---|
| Coworker setup, data contract, model comparison, and command examples | [Coworker handoff](docs/COWORKER_HANDOFF.md) |
| Dataset design, equations, split audit, saved physical fields, and measured limits | [HETMF dataset and training-readiness report](HETMF_DATASET_AND_TRAINING_READINESS_REPORT.md) |
| Archive schemas and input/target meaning | [Dataset contract](Dataset/PHYSICS_AND_DATA.md) and [dataset index](Dataset/README.md) |
| Physical reference generation and archive commands | [Generation guide](docs/HETMF_GENERATION_GUIDE.md) |
| Pilot and Standard release history | [Pilot report](docs/HETMF_PILOT_REPORT.md) and [Standard release report](docs/HETMF_STANDARD_RELEASE_REPORT.md) |

## Install the Python packages

From the ModularDT repository root, install the case-neutral HONF package and the HET-MF package into the active Python environment:

```bash
python -m pip install -e HONF_Proj -e HONF_Proj/Case_ElectroThermal
cd HONF_Proj/Case_ElectroThermal
python -m hetmf --help
python -m hetmf link-shared --bundle /data/public/ModularDT-HETMF-Dataset/v1 --case-dir .
```

The H5 archives and model outputs stay outside Git. Use the shared `/data/public/ModularDT-HETMF-Dataset/v1/` files as read-only inputs. Place training runs below your own `/data2/<account>/ModularDT/` area by default; use `/data/public/` for outputs only when deliberately publishing shared material.

## Current evidence

The Standard archive has 4,608 states and Expanded has 9,216. The case loader and adapter have passed CPU forward/backward interface checks through the general HONF core, with finite gradients and unchanged parameters; those checks made zero optimizer updates. No trained field accuracy, learned-organizer result, sparse-work or executor saving, or inverse-design validity has been measured. The readiness report records these limits and the reviewed archive evidence.

The two training archives contain physical dimensionless fields and TRAIN-only target statistics. `HETMFDataset(normalize=True)` applies the saved target transform on read; the case adapter applies fixed logarithms to positive conductance inputs. Preserve the archive profile and its stored TRAIN normalizer when comparing or resuming models.
