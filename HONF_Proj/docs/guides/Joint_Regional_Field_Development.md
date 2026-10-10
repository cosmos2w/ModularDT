# Joint fields and registered regional hypergraphs

This opt-in family learns one trainable source/environment representation and its physical heads from fresh initialization. Thermal has four configuration-field outputs and one source-resolved affine temperature field; Wind uses the same executed core with a nonlinear velocity head and its TRAIN-fitted empirical height-profile transformation. No old learned field checkpoint is an initialization or inference dependency.

The two shallow collective blocks encode typed physical nodes, gather source and environment content into spatially registered source/regional anchors, update collective states, and return content to the nodes. A receiver reads collective regional context before the shared source-preserving feature evaluation. A pooled global bypass remains explicit. All active heads and shared parameters train together; learned incidence is a computational organization, not an identified physical causal graph.

| Mode | Executed organization | Receiver context |
| --- | --- | --- |
| J-direct | Trainable source/environment scene context | Source/global path |
| J-geometry | Typed node→edge→node updates with fixed smooth geometry incidence | Fixed smooth geometry access |
| J-H | Same registered edge budget and typed collective updates with learned incidence | Learned receiver-specific access |

Thermal heat is absent from context preparation, support, and flow prediction. At fixed weights, heating changes leave flow unchanged and temperature increments equal the physical-source coefficient contraction. Configuration and weight changes invalidate prepared state. Temperature extraction roles share one field and native stencils. The native sample helper uses 64 angular interface points by default; a smaller catalogue requires an explicit `ntheta` argument. Sampled maxima and effective ratios retain their nonlinear semantics. The complete joint model is not declared affine; only its qualified temperature/control block is affine.

Wind keeps physical rotor-diameter coordinates, native direction/vector transformations, and three velocity components in m/s. Environment records are geometry quadrature, not observed flow sensors. Wind endpoint and local derivative calls follow a nonlinear output law; an exact affine increment request is rejected.

## Fixed development recipe

Thermal uses the existing fixed25_v1 manifest with 150 selected TRAIN cases and 22 exposed DEV cases; Wind uses wind_shared_fixed24_v1 with 72 TRAIN direction rows from 24 layouts and 24 DEV rows from eight layouts. The latter is 17.1% of source TRAIN and omits M18. WindTEST targets remain locked. Normalization, background profiles, component scales, response calibration, and any fixed objective adjustment use TRAIN only. The selected IDs and case_epoch_v1 query streams remain identical across the three modes.

Initial matched settings are H128/message 128, depth 2, receiver tile 512, Thermal E192/Q1024 plus native roles and operator128, and Wind E64/Q4096 with the maintained five-role component-balanced objective. Effective batches are 48 Thermal and 24 Wind. Packing is selected from measured TRAIN resource benchmarks with a memory reserve; changing packing does not change the effective batch or loss denominators. Common compatible tensors initialize identically across modes.

The ordinary joint schedule uses AdamW, gradient clipping 1, a 20-epoch learning-rate warmup to 3e-4, a hold through 1000, and cosine decay to 3e-6 at 2500. Every selected case is visited once per development epoch. All physical heads train from epoch 1. Monitoring, latest, best-field, and declared milestone artifacts use a 100-epoch cadence. Each extension follows a recorded 100/500/1000 review; healthy selected paired fits may mature to 2500 within the authorized campaign budget.

The Thermal objective gives equal weight to TRAIN-scaled flow and temperature families, with explicit u/v/p/omega and fluid/surface/material roles. The native q-proxy coefficient 0.05 is outside that balanced temperature family; qualified TRAIN response and stored-velocity operator terms are separate. Auxiliary coefficients are calibrated once on a fixed TRAIN panel from the new joint initialization and reused across compared modes. The selector remains the declared equal-family field score; fluid-T and each flow/temperature role remain visible separately. Wind displays all 15 role/component means and tails rather than hiding transverse errors in a scalar.

Use the ModularDT environment from the HONF project directory. Preparation binds inputs and transforms without taking an optimizer step; the development dry-run takes one disposable CPU update and writes no scientific checkpoint:

```bash
python tools/joint_regional_train.py prepare --task thermal --mode J-H
python tools/joint_regional_train.py dry-run --task thermal --mode J-H
python tools/joint_regional_train.py dry-run --task wind --mode J-H
```

Use a sealed JSON recipe for scientific starts and exact resumes, including measured packing and the common TRAIN auxiliary calibration. Generated recipes, run directories, arrays, figures, profiles, and receipts stay in ignored local paths. A stop is an absolute epoch, and exact resume keeps the same dataset, normalization, model, optimizer, sampling, and schedule identity:

```bash
CUDA_VISIBLE_DEVICES=1 python tools/joint_regional_train.py start \
  --recipe-json PATH_TO_SEALED_DEVELOPMENT_RECIPE --device cuda:0 \
  --output-dir PATH_TO_NEW_RUN --stop-after 100
CUDA_VISIBLE_DEVICES=1 python tools/joint_regional_train.py resume \
  --recipe-json PATH_TO_SAME_RECIPE --device cuda:0 \
  --output-dir PATH_TO_SAME_RUN --stop-after 500
```

The optional J-H locality revision adds a fixed Gaussian log prior to the learned typed incidence and receiver access scores; its strength is omitted at zero to preserve v1 identities. Positive strength retains learned residual scores, every physical source column, the same trainable tensors, and the same registered anchors. Changing it invalidates prepared contexts and heat operators. It is a model revision rather than an exact resume option.

A shape-compatible locality child requires an immutable pre-fit diagnosis, its SHA256-bound declaration, a new run directory and the exact parent monitoring checkpoint. The `branch` command preserves parent weights, named optimizer moments, RNG, case/query streams, absolute schedule and cumulative case/update age. It permits only the declared locality scalar change and rejects changes in datasets, normalization, objectives, packing, width or selection policy. Selection is reset because parent scores belong to the old function; inherited history and new-function age remain separately labelled.

```bash
CUDA_VISIBLE_DEVICES=1 python tools/joint_regional_train.py branch \
  --recipe-json PATH_TO_REVISED_J_H_RECIPE --device cuda:0 \
  --branch-from-checkpoint PATH_TO_COMMON_PARENT_CHECKPOINT \
  --revision-declaration PATH_TO_HASH_BOUND_DECLARATION \
  --output-dir PATH_TO_NEW_CHILD_RUN --stop-after 1000
```

## Manual fullTRAIN5000 preparation

The [Thermal recipe](../../src/config_core/forward/joint_regional/thermal_full5000_v1.json) and [Wind recipe](../../src/config_core/forward/joint_regional/wind_full5000_v1.json) are experimental manual-only recipes. Thermal selects the learned family with locality strength 1; Wind selects the geometric regional family after the mature learned locality revision still worsened near-wake Ux and Uy. Both execute the same source-preserving collective core. Effective and micro batches are 48/48 Thermal and 24/24 Wind, following measured development packing. They fit transforms on the original 600 Thermal TRAIN cases or 420 Wind TRAIN rows, initialize all trainable weights freshly, and preserve the test lock. They do not resume segmented checkpoints or import an external learned flow model. FullTRAIN membership, transforms and the 5000-epoch schedule have a distinct identity from development. The inert validation checks actual fullTRAIN preparation and two TRAIN native forwards with an optimizer-constructor guard; it does not measure a fitted formal model or formal GPU throughput. Both recipes remain experimental, requiring a separately requested manual launch.

The formal dry-run is deliberately inert: it constructs the fullTRAIN provider, validates its identity, optimizer coverage, output heads, transforms, and 5000 schedule, and performs no fullTRAIN optimization. Preparation does not authorize launch:

```bash
python tools/joint_regional_train.py dry-run \
  --recipe-json src/config_core/forward/joint_regional/thermal_full5000_v1.json
python tools/joint_regional_train.py dry-run \
  --recipe-json src/config_core/forward/joint_regional/wind_full5000_v1.json
```

A separately requested manual launch additionally requires the explicit --manual-formal-launch option, a fresh output directory, a reviewed GPU, and a declared absolute stop. No formal run is launched by the development workflow.

## Matched J-geometry formal comparison

The separate [Thermal J-geometry recipe](../../src/config_core/forward/joint_regional/thermal_geometry_full5000_v1.json) and [Wind J-geometry recipe](../../src/config_core/forward/joint_regional/wind_geometry_full5000_v1.json) support explicitly requested fresh 5000-epoch comparisons to Thermal3904 and Wind2202. They retain original TRAIN600/420, initialization seeds 0/42, effective and micro batches 48/24, fluid Q1024 or five-role Q4096, and TRAIN-only transforms. Thermal monitors canonical89, excluding the known TRAIN duplicate0273, while preserving original90 indices for the baseline's fixed validation query seeds; Wind monitors original VALID90/30 layouts with WindTEST targets locked. The formal native sampling protocol derives the dataset key from the baseline's full-TRAIN fingerprint rather than the development manifest and retains the baseline native query streams. Preparation receipts must verify the actual dataset keys, memberships, transforms, budgets and optimizer groups before launch.

Both comparison recipes use AdamW with learning rate 3e-4 from initialization through epoch2000, then cosine decay to 3e-6 at epoch5000, weight decay 1e-5, betas (0.9,0.999), epsilon 1e-8 and clipping 1. Every physical head and shared block trains from epoch1. This optimizer clock is separate from the older models' routing curriculum; the joint models have no detail-gate phase curriculum. Immutable checkpoints are limited to 100/500/1000/2000/2500/5000, with latest and best-field aliases overwritten. Epoch1 writes initial latest state and loss curves without adding a milestone or validation review; curves, latest and exposed validation subsequently update every100 epochs. The old experimental manual recipes and their completed development evidence retain their original bindings.

These are matched data and training-budget comparisons of different model recipes. Thermal3904 optimizes temperature refinement with an external frozen flow component; the new Thermal model learns fresh flow and temperature heads together, with the joint equal-family objective and TRAIN-calibrated response/operator terms. Widths, parameter groups, collective organization and objectives differ, so scalar total losses and old selector scores are not common accuracy measures. Compare common native physical fields on the same primary panels and report measured training/inference cost separately. Saved Thermal3904 validation files do not contain native point digests; reconstructing its unchanged original90 sampler demonstrates point agreement but does not add missing historical query receipts.

```bash
CUDA_VISIBLE_DEVICES=1 python tools/joint_regional_train.py start \
  --recipe-json src/config_core/forward/joint_regional/thermal_geometry_full5000_v1.json \
  --manual-formal-launch --device cuda:0 --stop-after 5000 --output-dir PATH_TO_NEW_THERMAL_RUN
CUDA_VISIBLE_DEVICES=2 python tools/joint_regional_train.py start \
  --recipe-json src/config_core/forward/joint_regional/wind_geometry_full5000_v1.json \
  --manual-formal-launch --device cuda:0 --stop-after 5000 --output-dir PATH_TO_NEW_WIND_RUN
```

## Evidence and retention

The [Thermal J-H formal recipe](../../src/config_core/forward/joint_regional/thermal_jh_full5000_v1.json) and [Wind J-H formal recipe](../../src/config_core/forward/joint_regional/wind_jh_full5000_v1.json) change only the model mode from the matched J-geometry formal recipes. They retain the full TRAIN populations, transforms, seeds, query streams, losses, batch sizes, 5000-epoch optimizer clock, primary validation panels and checkpoint/curve cadence. They use the standard learned J-H scores with locality prior strength zero. Wind automatically uses the [geometry-verified persistent catalogue store](Wind_Native_Catalogue_Cache.md), reusing verified entries and publishing missing geometry once; catalogue reuse never includes targets, query draws or learned state. Launch requires the same explicit manual-formal flag and a fresh run directory.

The [completed 2026-10-09 development report](../reports/HONF_Joint_Field_Regional_Development_20261009.md) binds the mature selected comparisons, both locality revisions, native fields, actual graph interventions, numerical limits and complete cost receipts. Thermal prefers learned locality organization for fluid-temperature/vorticity gains with retained flow and response misses; Wind prefers geometric regional organization after the mature learned revision still loses near-wake Ux/Uy. These are fixed exposed development results, and both manual fullTRAIN recipes remain experimental.

Report literal endpoints and selected checkpoints separately, using all 22/all 24 DEV statistics and a small fixed input-selected native field/graph panel. Record actual case visits, updates, queries, complete epoch/inference time, prepared reads, input VJPs, allocated/reserved peaks, and process memory. Receiver-access mass truncation is a model approximation; dense logical masks do not establish executor savings. Any subset implementation requires measured frontier evidence, dense/subset parity for the truncated function, and separate physical-reference errors.

The inference APIs default to the full dense function. An explicit `retained_access_mass` of 0.99, 0.95 or 0.90 selects access by cumulative mass; `receiver_edge_executor="subset"` performs a packed gather and weighted accumulation of retained receiver-edge values. Scoring, typed collective context formation and all original physical-source reads remain dense. Work receipts count each primitive separately, including padded source slots; a full-access subset request records a dense fallback. This executor has no implied wall-time advantage: measure complete native inference and memory before choosing it. The core also accepts an explicit boolean `fixed_receiver_edge_mask` for conditional derivatives with fixed edge identities; do not combine it with mass/count selection or interpret its derivative as differentiating support switches. Mutating the prepared support invalidates its affine operator.

The 2026-10-09 measured packed reader reduces receiver-edge value pairs but is slower than the full dense reader on the retained native panels. Thermal high-M extraction also misses the recorded strict same-truncation tolerance at two interface values because native q differencing amplifies FP32 reduction roundoff; this failure remains recorded. The packed variant remains an opt-in approximation experiment. The preferred and manual recipes use the full smooth dense function; no sparse speedup or physical omission guarantee is claimed.

Keep one compact report with the six figure families: joint learning, native fields/residuals, actual donor→edge→receiver chains, operational interventions, saved physical responses, and accuracy versus complete cost. PDF masters and necessary small Markdown companions remain ignored. Preserve existing scientific artifacts and the exact protected inventory, use no new solves or inverse campaign, audit all outgoing commits, and push only durable code/tests/configuration/documentation.
