# HET-MF dataset storage and browsing

The shared training bundle lives at `/data/public/ModularDT-HETMF-Dataset/v1`. Run `python -m hetmf link-shared --bundle /data/public/ModularDT-HETMF-Dataset/v1 --case-dir .` from the case directory to create six ignored browsing symlinks. This real `Dataset` directory keeps tracked contracts and schemas; preserve it as a directory. The original simulation archive remains protected beneath `/data2/wanglz/ModularDT/datasets/hetmf/` and is not required to read or train on the standalone H5 files.

## Current data and evidence

| Browsing entry | Path in the public bundle | Contents |
|---|---|---|
| `standard.h5` | `standard.h5` | Standalone Standard archive: 4,608 states |
| `expanded.h5` | `expanded.h5` | Standalone Expanded archive: 9,216 states |
| `formal_standard` | `evidence/formal_standard/` | Selected Standard release receipts and saved-field figures |
| `formal_expanded` | `evidence/formal_expanded/` | Source-lineage information; full provenance is embedded in the archive |
| `readiness` | `evidence/readiness/` | Archive/physics reviews, selected figures, and compact receipts |
| `pilot` | `evidence/pilot/` | Historical pilot report figures and inventory |


Open the current H5 files with `hetmf.training_data.HETMFDataset`, choosing an explicit split. The original FP64 references retain native face currents, heat fluxes, separate heat contributions, and terminal currents; the training targets use the five-channel contract in [PHYSICS_AND_DATA.md](PHYSICS_AND_DATA.md). The two archives have separate frozen memberships and TRAIN normalizers. Expanded sharing does not authorize deleting Standard or editing shared files in place.

## Owner archive and processing history

The following owner-only commands refer to the protected original archive, not the public training bundle. Run `python -m hetmf link-training --standard-h5 <data2-standard.h5> --expanded-h5 <data2-expanded.h5> --standard-release <release-root> --expanded-release <release-root> --readiness <audit-root> --case-dir <absolute-case-directory>` to create the five current data/evidence links. Run `python -m hetmf link --root <pilot-root> --case-dir <absolute-case-directory>` to create only `Dataset/pilot`. Both helpers refuse regular destinations and foreign symlinks. Pilot material is browsed through `pilot/raw`, `pilot/processed`, `pilot/figures`, `pilot/inputs`, and `pilot/manifest.json`.

Simulation, processing, verification, and visualization are maintained in `src/hetmf/`, exposed through `python -m hetmf`, and documented in the [generation guide](../docs/HETMF_GENERATION_GUIDE.md). Exact past launch wrappers and their logs remain in the `/data2` `approvals/` and readiness `inputs/` directories as provenance. Source worktrees remain retained; their retirement is deferred.

## Local retention

Keep original simulations, packed training archives, declared memberships, saved TRAIN normalizers, numerical reference quantities and provenance outside Git. Never overwrite a shared archive in place or infer that an old packaged shard is disposable from its name. Historical cleanup ledgers, plots and readiness reports remain local; this source release provides readers and schemas rather than dataset payloads.
