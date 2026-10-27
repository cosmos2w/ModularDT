# Thermal segmented qualification and fair model comparison

**Policy revision, 27 October 2026:** the [standing project instructions](../../../AGENTS.md) govern new rounds. Training duration is expressed only in epochs. The fixed quarter membership below is a reusable segmented execution/debugging set, with at most 500 segmented epochs across a round and only as many as necessary to establish real execution. Scientific model assessment requires full-training-population evidence and adequate training age. The first full-data segment is at most 100 epochs per selected model. The user now authorizes Codex to decide, after the completed 100-epoch review, whether the round's proposed final formal candidate should resume in the same lineage to 500 total epochs, within the round's declared budget. Other extensions still require explicit instruction. A short segmented fit or full-data 100-epoch endpoint is preliminary evidence, not a mature architectural verdict. The old development100→500→1000 promotion workflow below remains historical.

An epoch visits its declared membership completely. Retain the100-epoch checkpoint cadence,10-epoch curve refresh, initial/final curves and resumable states, including the requested final stop for short qualification. Existing runs, frozen Stage-A normalization, mature references and historical launch recipes retain their identities; this policy revision launches or alters none of them. All new HONF architectures use prescribed controls in their owning input objects and a generic joint physical-field predictor. Thermal dataset simplifications do not justify separate flow/temperature architectures or universal affine/invariance constraints; case adapters own targets, units and native observations. The same principle applies to future Wind and other cases.

## Fixed membership

The read-only generated manifest is [fixed25_v1/manifest.json](../../../../../../../../data/wanglz/ModularDT/thermal_development/fixed25_v1/manifest.json). It references the existing [packed dataset](../../../../../../../../data/wanglz/ModularDT/1_ChannelThermal/Processed_ChannelThermal_Dataset/packed_dataset.h5); it does not copy cases or field arrays.

- Selection seed: `20261004`; model/query seed: `0`.
- Semantic manifest SHA256: `933b0138ba2f8447a1ecadfe31fd0bb2cb4a05607d3ac3d9f0dc79419f196044`.
- Physical-input metadata SHA256: `1bba5ab5c0535fabab2f2434de33eccb67184ef3fd7d7881eb01f5088e15e211`.
- Native runtime dataset binding: `thermal_channel_global_v1`, schema `thermal_channel_global_hdf5_v1`, catalog fingerprint `4224093c22a67af4adfecc8b21d53548e4263ec2254c230dc83c89526b36da05`.
- Training: **150/600 cases, exactly 25%**. Validation: **22/89 canonical cases, 24.72%**; this is 22/90, or 24.44%, of the original test partition. Known duplicate `0273` is excluded before selection.

The catalog record supplies the native dataset ID/fingerprint; the packed HDF5 does not store those attributes. The manifest therefore binds its source using actual physical-input metadata and file size/mtime, and does not invent HDF5 attributes. Its semantic hash is independent of location. Runtime validation also checks recorded source path/stat; relocating or replacing the source requires deliberate regeneration and rebinding.

| Module count M | Original train | Selected train | Canonical validation | Selected validation |
|---:|---:|---:|---:|---:|
| 1 | 90 | 23 | 0 | 0 |
| 2 | 90 | 23 | 0 | 0 |
| 3 | 17 | 4 | 24 | 6 |
| 4 | 90 | 22 | 0 | 0 |
| 5 | 15 | 4 | 25 | 6 |
| 6 | 90 | 22 | 0 | 0 |
| 7 | 15 | 4 | 25 | 6 |
| 9 | 90 | 22 | 0 | 0 |
| 10 | 15 | 4 | 15 | 4 |
| 12 | 88 | 22 | 0 | 0 |
| **Total** | **600** | **150** | **89** | **22** |

Selection allocates counts proportionally by M using largest remainders. Within each M it preserves available Reynolds bands `Re≤70`, `70<Re≤130`, and `Re>130`, then samples without replacement across within-band total-heat rank terciles where the quota permits. All ten training M categories and all four validation M categories remain represented. All available M/Re bands are represented in this manifest. A one- or two-case stratum cannot represent all three heat bins; full exact-Re/layout coverage is not claimed. Selection reads physical inputs and original partitions only, never fields, losses or model errors. The complete strata table is in the local [generation receipt](../../../../../../../../data/wanglz/ModularDT/thermal_development/fixed25_v1/generation_receipt.md).

The fixed validation set is repeatedly exposed development evidence. Report per-field and per-M metrics with sample counts; four or six cases per validation M give limited precision. Do not repeatedly redraw a better-looking validation cohort. Keep full canonical validation for a final formal check, with original 90-case compatibility results separately labelled if needed.

## Fair qualification and full-data budgets

Compared segmented models use this exact manifest and matching train-only normalization, seeds, case ordering/query policy, native losses, effective batch48, FP32, Q1024, schedule and epoch budget. Full-data comparisons instead use the entire source training membership and one common training-only normalizer, with separately named lineages. Initialization must match the new architecture and the declared comparison; historical B-fine or affine-model initialization is not mandatory for a general joint-field model. Frozen historical normalization remains unchanged. No silent segmented/full-data resume or transfer is allowed.

Use segmented data only for actual prepare/decode/backward, learning-path, checkpoint and resume qualification. Stop when those are demonstrated, usually well before the whole-round 500-epoch ceiling; retries do not renew that allowance. When execution of an agreed comparison is requested, fit selected models on full TRAIN for at most 100 initial epochs and retain saved evidence/resume commands. After that review, Codex may use the standing authorization for the round's proposed final formal candidate to continue to 500 total epochs; explicitly record the decision, evidence and unchanged lineage. Controls and historical A/B runs are not implicitly extended. Do not repeat short fits until a metric passes or add a portfolio. New models declare one coherent objective; they do not inherit a legacy epoch 101 objective change by accident.

One segmented epoch presents150 selected cases and153,600 primary fluid queries atQ1024. Segmented validation has22 cases and22,528 primary queries. One full-source epoch presents600 training cases. Native surface/material/stencil and response work remains additional and is reported separately. Keep complete declared case passes and distinguish measured elapsed time from a projection; do not equate100 segmented epochs with100 full-data epochs or describe either using another training-duration unit.

Historical profiles used no response fitting in their first100 epochs, then admitted selected TRAIN reference anchors for continuation. The fixed manifest intersects the eight stored anchors at0348 only (M10, Re70, total heat11.9141576 in benchmark scales). Those historical profiles and their loader behavior remain intact. New rounds explicitly declare whether native-only or TRAIN-only response fitting is shared across comparison arms; an absent optional response diagnostic is not a new research admission gate. Never insert validation anchors into fitting or claim one-anchor coverage generalizes across all M.

Every 100 epochs, retain an epoch milestone, update latest state and consider the best-field alias. Best-field selection compares only saved review checkpoints; unsaved epochs do not silently replace the selector. Disable total-loss, temperature and predicted-mode checkpoint aliases while retaining those scalar metrics. Keep the declared 100,200,…,1000 milestones, one latest alias and one best-field alias; an explicitly requested final stop is the only non-multiple exception. Current loss curves refresh every 10 epochs by default, while validation and checkpoint selection keep their declared cadence. Separate validation files are written only at checkpoint milestones and final or clean-stop reviews; every review remains in full history. Tqdm progress is enabled by default, including non-TTY output. Do not create 25-epoch snapshots or per-evaluation checkpoint copies. Historical checkpoint cleanup needs a separate, explicit retention decision.

Per-case field resolution is unchanged. Batch composition can affect peak memory despite the same microbatch size. Record actual elapsed train/validation time and contention; do not infer fourfold speedup from subset size. Initial formal 100-epoch and segmented launches receive a read-only Luna startup/epoch-time/ETA check and a near-completion check, with earlier inspections only for a concrete failure or required milestone. Luna stays quiet while nothing actionable changes; the parent does necessary independent work or rests and does not duplicate polling. For the final formal 100→500 continuation only, confirm resume/startup once and finish the task without a Luna monitor, observer, polling, completion wait or follow-up automation. Ordinary in-process outputs remain enabled. No new monitoring daemon is needed.

Every new Thermal forward formal/development TRAIN600 run must have its workspace alias at `HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/<run_name>`, pointing to the verified DATA2 payload. Do not create aliases directly under `Trained_Results`; segmented qualification can remain under the round's diagnostics root. Preserve historical payloads and identities when repairing aliases. Historical examples below do not override this new-run placement rule.

## Historical preparation and manual-launch examples

The commands in this section document the original profile helper and are not an instruction to launch Tree or extend new segmented work. Use the maintained runner and the current round's declared model/epoch stop for new tasks. Do not alter active historical configurations to impose new policy.

Use the `ModularDT` environment from the project directory. The tracked [selection specification](../../src/config_core/data/thermal_development25_v1.json), [manifest builder](../../tools/thermal_development_split.py), [profile helper](../../tools/thermal_development.py) and [development templates](../../src/config_core/forward/thermal_development) are durable. Generated memberships, bound configurations and receipts stay outside Git. Templates are deliberately unbound; materialize them before launch.

The current manifest already exists. On a fresh machine, create it once with the same source/spec; the tool refuses to overwrite an existing manifest:

```bash
conda activate ModularDT
cd /home/wanglz/Desktop/src/ModularDT/HONF_Proj
python tools/thermal_development_split.py \
  --dataset ../../../../../../../../data/wanglz/ModularDT/1_ChannelThermal/Processed_ChannelThermal_Dataset/packed_dataset.h5 \
  --output ../../../../../../../../data/wanglz/ModularDT/thermal_development/fixed25_v1/manifest.json
```

For example, prepare only Tree, then inspect its launch. `--first-run-id` is the first ID of the maintained five-arm mapping; 3101 maps Tree to **3103**. Choose unused IDs when starting actual work. Preparation writes configurations and prints dry-run commands; it launches no models:

```bash
python tools/thermal_development.py --arm H-tree \
  --manifest ../../../../../../../../data/wanglz/ModularDT/thermal_development/fixed25_v1/manifest.json \
  --dataset ../../../../../../../../data/wanglz/ModularDT/1_ChannelThermal/Processed_ChannelThermal_Dataset/packed_dataset.h5 \
  --first-run-id 3101 --stage 100 \
  --output-dir /data/wanglz/ModularDT/thermal_development/tree_preparation_e100
CUDA_VISIBLE_DEVICES=1 python train.py \
  --config /data/wanglz/ModularDT/thermal_development/tree_preparation_e100/h-tree_e100.json \
  --device cuda:0 --dry-run
```

Only when that candidate's training is requested, replace `--dry-run` with `--yes`. Physical GPU 2 uses `CUDA_VISIBLE_DEVICES=2` with the same logical `cuda:0`. Development run output defaults to `/data/wanglz/ModularDT/thermal_development/HONF_Forward_Runs`, keeping new runs separate from historical full-data campaigns. Override `--run-output-root` deliberately if needed.

The original helper also supports historical `--stage 500` and1000 recipes. Their availability is not authorization to extend a new run. On an explicitly requested continuation, preserve the exact parent's state, membership, normalization and schedule. Never turn a historical full-data checkpoint into an undeclared quarter-data run or reuse debugging weights as a full-data initialization without a declared transfer protocol.

## Evaluation and artifact budget

For a bound development checkpoint, forward evaluation defaults to its selected cohort, including at stage 100. Normalized and physical loaders use identical indices. A routine statistical comparison evaluates **all 22 selected validation cases once** at a retained checkpoint, with the normal intervention only. Detailed field arrays and graph/phase archives default to a fixed representative panel of four cases spanning available M. Additional diagnostic interventions must use an explicitly budgeted small panel. Reuse completed evidence when checkpoint, membership and evaluation protocol have not changed.

```bash
python tools/thermal_campaign_evaluate.py \
  --checkpoint PATH_TO_SAVED_DEVELOPMENT_CHECKPOINT \
  --stage 100 --evaluation-scope development --device cuda:0 \
  --interventions normal --capture-phase-graphs \
  --output-dir /data/wanglz/ModularDT/thermal_development/evaluation/EXACT_RUN_AND_CHECKPOINT
```

`--quick-diagnostic` explicitly requests a smaller metric panel; label those results accordingly. `--save-field-arrays all` and `--phase-graph-scope all` explicitly expand output budgets. The default calibration control, when required for an intervention, uses a small selected-training panel. It must not read excluded training cases.

Frozen heat inference and shared benchmark/topology loaders inherit the checkpoint's subset. Response evaluation accepts only the selected-training response anchors by default for development checkpoints. Inverse-head training also binds selected train/validation cohorts and fits its own public-input normalization on selected training only. Fixed inverse trails, twelve-task paired-head panels and separately labelled archived observation controls remain additional explicit diagnostic budgets; they are not the 22-case forward statistical validation result. Do not launch an inverse head or a battery of ablations merely because a helper exists.

`--evaluation-scope formal-full` deliberately expands forward/heat/response evaluation where supported. It does not change the checkpoint's training history or normalization; report this as full-population evaluation of a development-trained checkpoint. Formal full-data training of the chosen model is a new run using the original training population and its own train-only normalization. Existing full-data profiles and 5,000-epoch recipes keep their historical schedules and must not be repurposed into development runs by silently editing dataset fields.

## Verification record

The local [CPU verification receipt](../../../../../../../../data/wanglz/ModularDT/thermal_development/fixed25_v1/cpu_loader_verification.json) records an actual selected-train normalization fit and native loader traversal: 150 unique training cases, 22 unique validation cases, 19 microbatches, four optimizer boundaries, 153,600/22,528 primary queries and one shared train normalizer. With one CPU thread, fitting took 0.457 s, training-data traversal 0.729 s and validation-data traversal 0.102 s. These are data-path measurements; no native model fitting or checkpoint execution occurred.

The combined CPU suite passed **199 tests**; two CUDA fixture tests were skipped and six native-forward/resource tests were deliberately excluded. A synthetic missing-mask fixture emitted its expected compatibility warning. Tests cover deterministic M/Re/heat stratification, source/membership tampering, excluded-case isolation, normalization leakage and Stage-A invariance, full selected-case accounting, checkpoint cadence/best selection, resume identity, actual native profile composition, profile fairness and bounded evaluation scope. All five actual bound development profiles passed metadata-only native dry-runs and reported 150/22 selected cases, four optimizer batches and horizon 1,000. Their proposed run output root remained absent. These checks validate preparation separately from scientific training completion. Generated manifests, profiles, receipts and future scientific outputs remain local under the repository upload rule.
