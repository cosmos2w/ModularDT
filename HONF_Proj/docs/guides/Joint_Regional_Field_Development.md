# Joint fields and registered regional hypergraphs

This opt-in family learns one trainable source/environment representation and its physical heads from fresh initialization. Thermal has four configuration-field outputs and one source-resolved affine temperature field; Wind uses the same executed core with a nonlinear velocity head and its TRAIN-fitted empirical height-profile transformation. No old learned field checkpoint is an initialization or inference dependency.

The two shallow collective blocks encode typed physical nodes, gather source and environment content into spatially registered source/regional anchors, update collective states, and return content to the nodes. A receiver reads collective regional context before the shared source-preserving feature evaluation. A pooled global bypass remains explicit. All active heads and shared parameters train together; learned incidence is a computational organization, not an identified physical causal graph.

| Mode | Executed organization | Receiver context |
| --- | --- | --- |
| J-direct | Trainable source/environment scene context | Source/global path |
| J-geometry | Typed node→edge→node updates with fixed smooth geometry incidence | Fixed smooth geometry access |
| J-H | Same registered edge budget and typed collective updates with learned incidence | Learned receiver-specific access |

Thermal heat is absent from context preparation, support, and flow prediction. At fixed weights, heating changes leave flow unchanged and temperature increments equal the physical-source coefficient contraction. Configuration and weight changes invalidate prepared state. Temperature extraction roles share one field and native stencils. Sampled maxima and effective ratios retain their nonlinear semantics. The complete joint model is not declared affine; only its qualified temperature/control block is affine.

Wind keeps physical rotor-diameter coordinates, native direction/vector transformations, and three velocity components in m/s. Environment records are geometry quadrature, not observed flow sensors. Wind endpoint and local derivative calls follow a nonlinear output law; an exact affine increment request is rejected.

## Fixed development recipe

Thermal uses the existing fixed25_v1 manifest with 150 selected TRAIN cases and 22 exposed DEV cases; Wind uses wind_shared_fixed24_v1 with 72 TRAIN direction rows from 24 layouts and 24 DEV rows from eight layouts. The latter is 17.1% of source TRAIN and omits M18. WindTEST targets remain locked. Normalization, background profiles, component scales, response calibration, and any fixed objective adjustment use TRAIN only. The selected IDs and case_epoch_v1 query streams remain identical across the three modes.

Initial matched settings are H128/message128, depth2, receiver tile512, Thermal E192/Q1024 plus native roles and operator128, and Wind E64/Q4096 with the maintained five-role component-balanced objective. Effective batches are 48 Thermal and 24 Wind. Packing is selected from measured TRAIN resource benchmarks with a memory reserve; changing packing does not change the effective batch or loss denominators. Common compatible tensors initialize identically across modes.

The ordinary joint schedule uses AdamW, gradient clipping1, a 20-epoch learning-rate warmup to 3e-4, a hold through1000, and cosine decay to3e-6 at2500. Every selected case is visited once per development epoch. All physical heads train from epoch1. Monitoring, latest, best-field, and declared milestone artifacts use a100-epoch cadence. Each extension follows a recorded100/500/1000 review; healthy selected paired fits may mature to2500 within the authorized campaign budget.

The Thermal objective gives equal weight to TRAIN-scaled flow and temperature families, with explicit u/v/p/omega and fluid/surface/material roles. The native q-proxy coefficient0.05 is outside that balanced temperature family; qualified TRAIN response and stored-velocity operator terms are separate. Auxiliary coefficients are calibrated once on a fixed TRAIN panel from the new joint initialization and reused across compared modes. The selector remains the declared equal-family field score; fluid-T and each flow/temperature role remain visible separately. Wind displays all15 role/component means and tails rather than hiding transverse errors in a scalar.

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

The [Thermal recipe](../../src/config_core/forward/joint_regional/thermal_full5000_v1.json) and [Wind recipe](../../src/config_core/forward/joint_regional/wind_full5000_v1.json) are experimental manual-only recipes. They fit transforms on original600 Thermal TRAIN cases or original420 Wind TRAIN rows, initialize all trainable weights freshly, and preserve the test lock. They do not resume segmented checkpoints or import an external learned flow model. The development evidence and final report determine whether a formal recommendation is justified.

The formal dry-run is deliberately inert: it constructs the fullTRAIN provider, validates its identity, optimizer coverage, output heads, transforms, and5000 schedule, and performs no fullTRAIN optimization. Preparation does not authorize launch:

```bash
python tools/joint_regional_train.py dry-run \
  --recipe-json src/config_core/forward/joint_regional/thermal_full5000_v1.json
python tools/joint_regional_train.py dry-run \
  --recipe-json src/config_core/forward/joint_regional/wind_full5000_v1.json
```

A separately requested manual launch additionally requires the explicit --manual-formal-launch option, a fresh output directory, a reviewed GPU, and a declared absolute stop. No formal run is launched by the development workflow.

## Evidence and retention

Report literal endpoints and selected checkpoints separately, using all22/all24 DEV statistics and a small fixed input-selected native field/graph panel. Record actual case visits, updates, queries, complete epoch/inference time, prepared reads, input VJPs, allocated/reserved peaks, and process memory. Receiver-access mass truncation is a model approximation; dense logical masks do not establish executor savings. Any subset implementation requires measured frontier evidence, dense/subset parity for the truncated function, and separate physical-reference errors.

Keep one compact report with the six figure families: joint learning, native fields/residuals, actual donor→edge→receiver chains, operational interventions, saved physical responses, and accuracy versus complete cost. PDF masters and necessary small Markdown companions remain ignored. Preserve existing scientific artifacts and the exact protected inventory, use no new solves or inverse campaign, audit all outgoing commits, and push only durable code/tests/configuration/documentation.
