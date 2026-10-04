# Faster, fair Thermal model development

Future Thermal development uses one fixed input-stratified quarter of the existing training and validation partitions. A development epoch visits that selected training set completely. Statistical validation uses the same selected validation cases for every model. Checkpoints and plots are retained every 100 epochs. Full-data training is a separately labelled formal run after architecture selection.

This changes the development budget, not the native Thermal physics. Frozen Stage-A weights and their normalization remain checkpoint-owned. Existing runs, mature Run1804, historical full-data results and manual formal launch recipes retain their original identities. Wind training remains paused. This implementation prepared data views and configurations; it launched no training and created no scientific checkpoints.

## Fixed membership

The read-only generated manifest is [fixed25_v1/manifest.json](/data/wanglz/ModularDT/thermal_development/fixed25_v1/manifest.json). It references the existing [packed dataset](/data/wanglz/ModularDT/1_ChannelThermal/Processed_ChannelThermal_Dataset/packed_dataset.h5); it does not copy cases or field arrays.

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

Selection allocates counts proportionally by M using largest remainders. Within each M it preserves available Reynolds bands `Re≤70`, `70<Re≤130`, and `Re>130`, then samples without replacement across within-band total-heat rank terciles where the quota permits. All ten training M categories and all four validation M categories remain represented. All available M/Re bands are represented in this manifest. A one- or two-case stratum cannot represent all three heat bins; full exact-Re/layout coverage is not claimed. Selection reads physical inputs and original partitions only, never fields, losses or model errors. The complete strata table is in the local [generation receipt](/data/wanglz/ModularDT/thermal_development/fixed25_v1/generation_receipt.md).

The fixed validation set is repeatedly exposed development evidence. Report per-field and per-M metrics with sample counts; four or six cases per validation M give limited precision. Do not repeatedly redraw a better-looking validation cohort. Keep full canonical validation for a final formal check, with original 90-case compatibility results separately labelled if needed.

## Fair development budget

All compared models use this exact manifest, train-only normalization, seeds, case ordering/query policy, native losses, effective batch 48, FP32, Q1024, learning-rate schedule and stopping budget. Common physical tensors in HONF candidates use the maintained matched fresh B-fine initializer; Dense uses its own seeded native architecture. Normalization streams all stored sampled points and active supervised values from the selected 150 training cases only. Validation receives that same transform. Frozen Stage-A normalization remains unchanged.

Start one explicitly selected candidate or agreed comparison at **100 development epochs**. Review physical fidelity and measured work before extending selected runs to **500**. A **1,000** stop requires an explicit budget; it is an available horizon, not an automatic obligation. All these stops share one absolute 1,000-epoch development schedule. Do not shorten the schedule between comparison arms or reset it during continuation. The maintained common physical-objective amendment at epoch 101 applies consistently to reviewed continuations.

At the default batch sizes, one development epoch has **150 unique training visits, four native optimizer updates, 19 microbatches and 153,600 primary fluid queries**. Validation has **22 unique visits and 22,528 primary queries**, in one batch of at most 48 cases. Native training enforces complete selected-case passes and rejects train/validation batch caps. A 100-epoch development screen therefore presents 15,000 training cases, versus 60,000 for 100 full-source epochs. Auxiliary physical queries, calibration and responses are additional work and must be reported separately.

The initial 100 screen does not run response-loss callbacks. For continuations, the response pool is the first at most four established atlas anchors that belong to selected training IDs, shared across arms. This manifest intersects the eight available anchors at **0348 only** (M10, Re70, total heat 11.9141576 in benchmark scales). That one-anchor response coverage must be labelled; it does not demonstrate response fidelity across all M. Do not force excluded anchors into this split. If a future fixed manifest has no available response anchor, preparation refuses continuation beyond 100 until a reviewed training-only response pool exists.

Every 100 epochs, retain an epoch milestone, update latest state and consider the best-field alias. Best-field selection compares only saved review checkpoints; unsaved epochs do not silently replace the selector. Disable total-loss, temperature and predicted-mode checkpoint aliases while retaining those scalar metrics. Keep the declared 100,200,…,1000 milestones, one latest alias and one best-field alias; an explicitly requested final stop is the only non-multiple exception. Plots use the same cadence. Do not create 25-epoch snapshots or per-evaluation checkpoint copies. Historical checkpoint cleanup needs a separate, explicit retention decision.

Case/primary-query work falls to approximately one quarter; actual training speedup and peak GPU memory have **not** been measured here. Per-case field resolution is unchanged. Batch composition can affect peak memory despite the same microbatch size. Record real elapsed train/validation time and GPU contention for each run; do not infer fourfold wall-time improvement from row counts.

## Preparation and manual launch

Use the `ModularDT` environment from the project directory. The tracked [selection specification](../../src/config_core/data/thermal_development25_v1.json), [manifest builder](../../tools/thermal_development_split.py), [profile helper](../../tools/thermal_development.py) and [development templates](../../src/config_core/forward/thermal_development/) are durable. Generated memberships, bound configurations and receipts stay outside Git. Templates are deliberately unbound; materialize them before launch.

The current manifest already exists. On a fresh machine, create it once with the same source/spec; the tool refuses to overwrite an existing manifest:

```bash
conda activate ModularDT
cd /home/wanglz/Desktop/src/ModularDT/HONF_Proj
python tools/thermal_development_split.py \
  --dataset /data/wanglz/ModularDT/1_ChannelThermal/Processed_ChannelThermal_Dataset/packed_dataset.h5 \
  --output /data/wanglz/ModularDT/thermal_development/fixed25_v1/manifest.json
```

For example, prepare only Tree, then inspect its launch. `--first-run-id` is the first ID of the maintained five-arm mapping; 3101 maps Tree to **3103**. Choose unused IDs when starting actual work. Preparation writes configurations and prints dry-run commands; it launches no models:

```bash
python tools/thermal_development.py --arm H-tree \
  --manifest /data/wanglz/ModularDT/thermal_development/fixed25_v1/manifest.json \
  --dataset /data/wanglz/ModularDT/1_ChannelThermal/Processed_ChannelThermal_Dataset/packed_dataset.h5 \
  --first-run-id 3101 --stage 100 \
  --output-dir /data/wanglz/ModularDT/thermal_development/tree_preparation_e100
CUDA_VISIBLE_DEVICES=1 python train.py \
  --config /data/wanglz/ModularDT/thermal_development/tree_preparation_e100/h-tree_e100.json \
  --device cuda:0 --dry-run
```

Only when that candidate's training is requested, replace `--dry-run` with `--yes`. Physical GPU 2 uses `CUDA_VISIBLE_DEVICES=2` with the same logical `cuda:0`. Development run output defaults to `/data/wanglz/ModularDT/thermal_development/HONF_Forward_Runs`, keeping new runs separate from historical full-data campaigns. Override `--run-output-root` deliberately if needed.

After reviewing 100, generate a separate `--stage 500` preparation for the same arm and mapping. Continue from the exact development parent's saved checkpoint with `--resume-checkpoint`; preserve its run lineage. Apply the same process for a budgeted 1,000 stop. Native resume rejects changed membership, source binding, schedule or normalization, and rejects full-data/development transitions. Never initialize or resume a quarter-data run from a historical full-data checkpoint without a separately reviewed transfer-learning protocol.

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

The local [CPU verification receipt](/data/wanglz/ModularDT/thermal_development/fixed25_v1/cpu_loader_verification.json) records an actual selected-train normalization fit and native loader traversal: 150 unique training cases, 22 unique validation cases, 19 microbatches, four optimizer boundaries, 153,600/22,528 primary queries and one shared train normalizer. With one CPU thread, fitting took 0.457 s, training-data traversal 0.729 s and validation-data traversal 0.102 s. These are data-path measurements; no native model, optimizer update or checkpoint was executed.

The combined CPU suite passed **199 tests**; two CUDA fixture tests were skipped and six native-forward/resource tests were deliberately excluded. A synthetic missing-mask fixture emitted its expected compatibility warning. Tests cover deterministic M/Re/heat stratification, source/membership tampering, excluded-case isolation, normalization leakage and Stage-A invariance, full selected-case accounting, checkpoint cadence/best selection, resume identity, actual native profile composition, profile fairness and bounded evaluation scope. All five actual bound development profiles passed metadata-only native dry-runs and reported 150/22 selected cases, four optimizer batches and horizon 1,000. Their proposed run output root remained absent. These checks validate preparation separately from scientific training completion. Generated manifests, profiles, receipts and future scientific outputs remain local under the repository upload rule.
