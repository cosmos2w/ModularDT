# HET-MF dataset storage and browsing

All numerical payloads live beneath `/data2/wanglz/ModularDT/datasets/hetmf/`. This real `Dataset` directory contains tracked contracts, schemas, and six ignored browsing symlinks. Do not replace it with a symlink or store generated arrays on `/home`.

## Current data and evidence

| Browsing entry | Canonical path relative to the `/data2` HET-MF root | Contents |
|---|---|---|
| [standard.h5](standard.h5) | `processed/hetmf_standard_v1.h5` | Self-contained processed Standard archive: 4,608 states |
| [expanded.h5](expanded.h5) | `processed/hetmf_expanded_v1.h5` | Self-contained processed Expanded archive: 9,216 states |
| [formal_standard](formal_standard) | `releases/standard_20261017_v1/` | Original FP64 cases, frozen inputs, approvals, solver ledgers, and release evidence |
| [formal_expanded](formal_expanded) | `releases/expanded_20261018_v1/` | Expanded originals and lineage, including 4,608 Standard sources shared through hardlinks |
| [readiness](readiness) | `readiness/readiness_20261018_v1/` | Physical and archive audits, fourteen native-192 references, interface checks, report figures, and receipts |
| [pilot](pilot) | `pilots/pilot_20261017_v1/` | Historical pilot inputs, 320 raw files across four grids, 286 processed cases, diagnostics, and figures |

Open the current H5 files with `hetmf.training_data.HETMFDataset`, choosing an explicit split. The original FP64 references retain native face currents, heat fluxes, separate heat contributions, and terminal currents; the training targets use the five-channel contract in [PHYSICS_AND_DATA.md](PHYSICS_AND_DATA.md). The two archives have separate frozen memberships and TRAIN normalizers. Expanded sharing does not authorize deleting Standard or editing shared files in place.

## Linking and processing

Run `python -m hetmf link-training --standard-h5 <data2-standard.h5> --expanded-h5 <data2-expanded.h5> --standard-release <release-root> --expanded-release <release-root> --readiness <audit-root> --case-dir <absolute-case-directory>` to create the five current data/evidence links. Run `python -m hetmf link --root <pilot-root> --case-dir <absolute-case-directory>` to create only `Dataset/pilot`. Both helpers refuse regular destinations and foreign symlinks. Pilot material is browsed through `pilot/raw`, `pilot/processed`, `pilot/figures`, `pilot/inputs`, and `pilot/manifest.json`.

Simulation, processing, verification, and visualization are maintained in `src/hetmf/`, exposed through `python -m hetmf`, and documented in the [generation guide](../docs/HETMF_GENERATION_GUIDE.md). Exact past launch wrappers and their logs remain in the `/data2` `approvals/` and readiness `inputs/` directories as provenance. Source worktrees remain retained; their retirement is deferred.

## Retention after the approved 2026-10-24 cleanup

Keep both standalone training archives, every original simulation and refinement reference, frozen plans and approvals, source identities and inheritance, normalizers and integrity seals, solver/attempt journals, audit receipts, and selected PDF figures with the PNG companions embedded in reports. Preserve the historical pilot processed files because the supported `PilotDataset` reader consumes them. Historical lock files and completion receipts are retained.

The 73 old Standard raw shards and 73 old processed shards were redundant packaged exports. Their retirement requires exact checks of every numerical array and membership against the original FP64 files and current training archive. The [cleanup receipt](readiness/receipts/tidy_20261024_v1/cleanup_receipt.json) and [shard verification inventory](readiness/receipts/tidy_20261024_v1/shard_redundancy_verification.json) record the approved removal, original checksums, H5 metadata, preserved sources, and validation. Original packing receipts remain unchanged as historical records; their shard counts describe the earlier packed release. The current [dataset registry](dataset_manifest.json) records retained counts.

Identical frozen plan/approval copies share inodes while retaining their existing paths. Do not edit these records in place. Unused empty directories and the incomplete forecast staging file are removed only through the exact cleanup allowlist. Temporary directories are recreated when writes need them. Future processing should reuse the verified training archives; the explicit legacy `pack --release` command remains available for a separately budgeted need and can recreate the retired shards.

The dated maintenance removed 146 shard files totaling 3,293,749,386 bytes and consolidated 46,846,304 bytes of identical plans/approvals. It also removed the incomplete forecast staging file, five legacy main-checkout aliases, unused empty directories, and round-owned test fixtures. Both retained archives passed full semantic/payload/TRAIN-normalizer verification, all eight split loaders were exercised, and 18 focused tests passed. The [post-cleanup preservation check](readiness/receipts/tidy_20261024_v1/postflight_verification.json) records retained-file hashes, unchanged source worktrees, remaining capacity, and the explicit process-inspection limits.
