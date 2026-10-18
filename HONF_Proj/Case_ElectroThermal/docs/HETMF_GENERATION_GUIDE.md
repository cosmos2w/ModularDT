# HET-MF generation guide

The case is independent of ThermalChannel and Round9. Use `/home/wanglz/miniconda3/envs/ModularDT/bin/python` with `PYTHONPATH=src` from `HONF_Proj/Case_ElectroThermal`; the system Python's torch installation was not used for the pilot. Set `PYTHONDONTWRITEBYTECODE=1` so Python bytecode is not written into the source tree. Generated input plans, arrays, diagnostics, figures, caches, and logs belong below the verified `/data2/wanglz/ModularDT/datasets/hetmf/` root.

## Pilot

The pilot root is `/data2/wanglz/ModularDT/datasets/hetmf/pilots/pilot_20261017_v1`. Inspect `python -m hetmf preflight --config configs/pilot_v1.yaml` before generating; the command reports mount identity, free bytes, and physical GPU2 status. `python -m hetmf plan --profile pilot --output <pilot-root>/inputs/pilot_plan.json` freezes the 286 distinct physical inputs. `python -m hetmf pilot --stage representatives --plan <pilot-plan> --root <pilot-root> --backend cpu` runs only the twelve 96² representative scenes. The `qualify`, `qualify384`, `interactions`, `interaction_refine`, and `throughput` stages are separate bounded commands. `pilot --stage all` covers the normal representatives, 48²/192² refinement, interactions, and 256-scene throughput stages, with each stage checked before proceeding; the optional 384² and interaction refinement stages remain explicit.

Use `python -m hetmf validate --root <pilot-root> --full` for hash, field-shape, finite-value, and post-storage conservation checks. `python -m hetmf pack --root <pilot-root> --scope pilot` writes processed native-96 targets while retaining raw data. `python -m hetmf visualize --root <pilot-root> --gallery representatives` and `--gallery quartets` render saved arrays. `python -m hetmf forecast --pilot <pilot-root>` computes CPU-supported profile projections from measured complete host timings and bytes. `python -m hetmf link --root <pilot-root> --case-dir <absolute-case-directory>` creates safe ignored browsing links; it refuses regular destinations or unrelated symlinks. `python -m hetmf status --root <pilot-root>` reports the local payload inventory.

The optional `python -m hetmf gpu-benchmark --pilot <pilot-root> --allow-concurrent-gpu2` replays only frozen pilot inputs, with three repeats of the 256-state B32 panel and 32-state B1/B4/B16 panels. It binds the verified physical GPU2 UUID before importing torch, requires at least 16 GiB initially free, caps this process at 25% of device memory, and rechecks free memory at every batch boundary. The concurrency switch must reflect explicit user authorization for this pilot; its timings are labelled shared-device measurements. It writes timing/quality journals below the pilot root and no duplicate field files.

The pilot plan is input-only, bound to a SHA-256 hash and separate from all formal roots. A successful raw case requires matching scene and solver hashes, complete HDF5 fields, an accepted marker, and a post-storage conservation audit. A `receipts/STOP` file requests a clean batch-boundary stop. The single-writer lock prevents concurrent HDF5 mutation. The ledger retains failed attempts; resumption never treats mere file existence as success.

## Formal proposal and approval gate

`python -m hetmf plan --profile standard --output <data2-plan-directory>/formal_standard_plan.json` creates a proposed input manifest and performs no physical solves. Compact and expanded use the corresponding profile names. `python -m hetmf generate --plan <formal-plan> --dry-run` validates the manifest without creating a release or solving any PDE. This dry run is safe before approval.

A later formal run requires a separately supplied JSON approval record with `schema=hetmf_formal_approval_v1`, exact `plan_hash`, `max_states`, `grid`, every `resolution_groups` entry (`default`, `m10`, `m16`, `high_contrast`, `composite`), positive `coefficient_bounds` min/max pairs for `s`, `kappa`, and `beta`, `precision=fp64`, physical `gpu_uuid`, exact `disk_root` below the HET-MF `releases/` directory, `max_elapsed_seconds`, `retained_budget_bytes`, `temporary_budget_bytes`, `minimum_free_bytes`, and `batch_size`. The runner requires `--approval <record> --yes --root <approved-release-root> --device cuda:0` together. Exclusive GPU2 is the default. A separately authorized shared run must bind `gpu_sharing={"mode":"concurrent","minimum_free_mib":16384,"process_memory_fraction":0.25}` in its approval; the runner caps its own CUDA allocation and checks the physical GPU2 free-memory floor at every batch. Sharing slows or perturbs timing, so label its measurements accordingly. `--resume` only accepts matching saved input and solver hashes.

A complete approved release may be packed into at most 64 same-grid cases per raw and processed shard. The packer verifies each source hash and accepted marker, writes temporary HDF5 files, verifies them, and atomically renames the pair. It retains the source cases for audit; their disk cost must be included in the approved storage budget. The model-facing five-channel field order is fixed by `Dataset/PHYSICS_AND_DATA.md`; normalization and training are outside this generator.

After an approved release completes, `python -m hetmf pack --release <approved-release-root> --plan <formal-plan> --approval <record> --verify-source-hashes` checks every accepted source and its independent solver ledger before writing shards. This command is unavailable for an unapproved or incomplete release. The ignored `Dataset/formal_standard` link in the home checkout points to the Standard release root on `/data2`; the real `Dataset` directory retains tracked schemas and documentation.

## Reviewed CPU generation and nested profiles

The formal command also accepts an approval with `backend=cpu_sparse_direct_fp64`, `gpu_uuid=null`, and `--device cpu`. This route uses the independently qualified CPU sparse-direct reference and never binds or initializes CUDA. Do not include `gpu_sharing` in a CPU approval. The numerical recipe records exact physics-source hashes and NumPy/SciPy versions; resuming after a recipe change is rejected. Set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` for this background preparation while other work is active.

An explicitly bound `inheritance={"root":"<standard-release>","plan_hash":"<standard-plan-hash>","approval_hash":"<standard-approval-hash>"}` permits unchanged states from an exact nested source release. Each source is checked against its original input, solver, approval, file checksum, and post-storage physics before linking. The destination retains the original H5 attributes and records the original approval and source path; a new profile never relabels those labels as newly solved. Source and destination must have identical physical inputs, split membership, and grid. New states use the CPU recipe. Both single-writer ownership and resource limits remain enforced.

## Single-file training archives

Use `PYTHONPATH=src:../src` from the case directory for commands that connect to the general HONF core. A complete release can be exported without creating another set of per-case arrays or raw shards:

```bash
python -m hetmf export-training --release <release-root> --output <data2-training.h5> \
  --retained-budget-gib 3 --temporary-budget-gib 3 --minimum-free-gib 100
python -m hetmf verify-training --dataset <data2-training.h5>
python -m hetmf honf-smoke --dataset <data2-training.h5> --queries 17 \
  --output <data2-readiness-root>/diagnostics/honf_smoke.json
```

The exporter writes one self-contained `hetmf_training_h5_v1` archive and refuses an existing output path. It stores unnormalized physical targets, separate prescribed inputs, native queries and measures, explicit split indices, source lineage, and exact TRAIN-only population statistics. It verifies every source and the complete staged archive before atomic publication. Reuse an existing verified archive instead of exporting comparison-arm copies. The full verifier checks the entire numerical payload and metadata seal and recomputes the TRAIN normalizer. Loader construction performs cheaper structural checks; run full verification at dataset admission and after any suspected storage change.

The read-only `HETMFDataset(path, split="train", query_count=1024, seed=0, normalize=True)` exposes prescribed inputs, queries, targets, and separate metadata. It binds sampled queries to stable physical case identity and supports pickle/fork/spawn workers. `ElectroThermalHONFAdapter.collate` makes CPU `SceneBatch`/`QueryBatch` tensors, and `HONFReadyBatch.to(device)` moves a complete batch in the main process. Worker collation should remain on CPU; use spawn after Torch threads are initialized. The general A2-D smoke uses five nonlinear outputs and zero optimizer steps; it establishes an interface connection, not predictor accuracy or a trained organizer. A later campaign still needs a case TaskProvider and explicit training recipe.

```bash
python -m hetmf link-training --case-dir <home-case-directory> \
  --standard-h5 <data2-standard.h5> --expanded-h5 <data2-expanded.h5> \
  --standard-release <standard-root> --expanded-release <expanded-root> \
  --readiness <data2-readiness-root>
python -m hetmf diagnose-releases --standard-release <standard-root> \
  --expanded-release <expanded-root> --output-root <data2-readiness-root> \
  --retained-budget-gib 24 --temporary-budget-gib 4 --minimum-free-gib 100
```

The saved-field diagnostic freezes six representative inputs and two complete quartets, reads the native-96 release arrays, and creates fourteen same-input CPU native-192 references. It retains the selected numerical references and PDF/PNG figures and performs no model fitting. Its fixed selection is reused on rerun. See the [detailed readiness report](../HETMF_DATASET_AND_TRAINING_READINESS_REPORT.md) for the measured limitation of the frozen TRAIN layout–morphology pairing and the distinction between solver residual, mesh discrepancy, and learned-model accuracy.
