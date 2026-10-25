# Thermal interaction-group reuse development recipe

This recipe implements the bounded [Round12-3 plan](../../UpgradePlan/HONF_Round12-3_Interaction_Group_Reuse_Work_Plan.md). The new `interaction_groups_v1` core prepares information-bearing E aggregates inside two interaction layers and exposes the same final group key/value tensors to a bounded query reader. It preserves original M identities and heat columns. It has one joint u/v/p/T head, case-owned native extraction, no direct M-value read, no independently learned membership C, no ordinary original-E query expansion and no optional refinement. This exclusive connected partition is a restricted organizational hypothesis, not unrestricted overlapping hypergraph learning.

The [completed first100 report](../reports/HONF_Round12-3_Interaction_Group_Reuse_Development_Report.md) records the surface-preservation failure and the decision not to start G-H100. The adaptive recipe remains provisional; its outstanding calibration is not implied by native forward/backward qualification.

## Matched arms and decisions

The versioned configurations are [G-EE](../../configs/research/thermal_interaction_groups_gee_dev_v1.json), [G-F](../../configs/research/thermal_interaction_groups_gf_dev_v1.json) and [G-H](../../configs/research/thermal_interaction_groups_gh_dev_v1.json). G-EE alone adds E→E after M→E. G-F and G-EE use known singleton partitions and execute neither candidate construction nor the stop policy. G-H nominates a spatial merge tree from the applied M→E relation and contextual E states and samples a reversible frontier from epoch 1. The common named model tensors have identical seeded initialization; extra E→E tensors are initialized afterward.

Use the existing fixed25 TRAIN150/exposed DEV22 binding and selected-TRAIN normalization. Every arm uses seed 0, FP32, width 128, four heads, two layers, Q1024, effective batch 48, nominal microbatch 24 and AdamW 3e-4 with weight decay 1e-5. An epoch visits all 150 selected TRAIN cases. The initial endpoint is 100, with 15,000 primary case visits, 400 optimizer boundaries and 15.36 million primary fluid queries. Native surface/material queries and finite-response endpoints are additional work. The absolute schedule is 1,000 epochs; it does not authorize continuation. Native loss weights remain the declared fixed weights at and after epoch 101, rather than silently inheriting another runner's amendment.

Existing physical-reference family 0348 is the only response fitting anchor. A no-step response gradient calibration at the first macro fixes the coefficient targeting 10% of the shared encoder/head native gradient, with the existing downward cap applied at every response event. Every fourth primary update adds one alternating heat/single-geometry response pair before the same Adam step. There are 100 response microbatches and 200 endpoint-scene visits in a completed 100-epoch arm, plus the separately counted calibration pair. Other atlas families remain reference-only or response-validation-only. No solver calls, M1/M2 response references, mixed-geometry training or flow-invariance constraints are introduced.

Train G-EE and G-F first. Compare their explicit epoch-100 checkpoints on all nine native DEV22 boards, the fixed six-M1/six-M2 TRAIN panel and existing physical responses. Priority temperature/peak preservation bands are mean ratio ≤1.10 and p90 ratio ≤1.15; response bands use the ratio of pooled-record medians and p90 absolute RMSE for each priority component. Always retain absolute errors, strata, tails and near-null exclusions. Compare with the fixed selected-TRAIN mean native predictor and zero-response predictor. A preservation ratio alone cannot qualify an inaccurate surrogate. A material field/response failure stops the adaptive training stage and is reported as a negative communication qualification.

## Adaptive frontier and counted work

G-H has no forced-leaf warmup, organizer freeze, target K, risk head, stopped tie loss or expected-K reward. Temperature decreases continuously from 2 to 1 over absolute epochs 1–500; the work multiplier follows the same cosine from 0 to 1. Its value near epoch 100 is only 0.094. The connected 24×8 grid has 352 candidate edges per layer. Tree storage consists of compact children and additive statistics, followed by one ancestry-label traversal. Expensive group transformations and key/value projections apply only to active frontier nodes.

Before any adaptive training, fit positive relation/state/coordinate distance scales on at most 12 input-selected TRAIN scenes, measure stage prices, and qualify the task/work policy VJPs on the same complete encoder/organizer parameter inventory. Record the calibration receipt and fixed prices in an ordinary executable recipe. A null or unstable work gradient leaves its coefficient zero and work learning unqualified; do not launch an apparently work-trained arm from the provisional zero-price configuration. The work policy contribution targets 2% of the nonzero native-policy norm and is capped downward at 10% of the full native gradient norm. Conditional native gradients and score-function terms are separate; previous-update TRAIN baselines are action-independent and checkpointed. Epoch 100 reviews do not silently retune prices.

Query work scores at most all M lightweight proposal keys, selects two M addresses, visits at most eight postings and nine spatial cells, retains at most four geometric group entrances and at most eight total group candidates, then scores and reads only these retained group values. A deterministic cyclic alternative occupies one proposal slot on one in sixteen TRAIN query IDs when possible. Evaluation uses deterministic learned proposals. The ordinary reader has zero original-E and zero M-value reads. Preparation still embeds and contextualizes original E and charges both layers' group formation, relation reduction and index construction. A small final K alone is neither a complete work metric nor evidence of physical causality.

## Maintained execution

Use the `ModularDT` environment. The maintained `train-contextual` command dispatches the new recipe schema to `ThermalInteractionGroupTask`, `ResearchTrainingEngine` and its named Adam/RNG/provider restoration and OOM rollback. Existing contextual checkpoints retain their own adapter/workflow schemas. Fresh runs use a single visible GPU UUID and logical `cuda:0`; preserve that visibility on resume. GPU0 and an idle GPU1 have equal priority; shared GPU0 is permitted, while other users' jobs remain untouched. Device assignment is recorded by each recipe, and timing comparisons require an isolated compatible GPU.

```bash
CUDA_VISIBLE_DEVICES=GPU_UUID python tools/honf_research.py train-contextual \
  --config configs/research/thermal_interaction_groups_gf_dev_v1.json \
  --device cuda:0 --devices cuda:0 --epochs 100 \
  --output /data2/wanglz/ModularDT/relocated_workspace/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/thermal_interaction_groups_gf_dev_v1
```

The endpoint retains epoch100, latest and best-field, with selection only over declared monitoring points. Latest/checkpoint cadence is 100; curves refresh every 10 epochs, full history remains and tqdm stays enabled. Do not add selector copies, intermediate scientific snapshots or a separate validation file per ordinary review.

```bash
python tools/honf_research.py evaluate-contextual --checkpoint CHECKPOINT \
  --device cuda:0 --kind native --output NEW_NATIVE_OUTPUT
python tools/honf_research.py evaluate-contextual --checkpoint CHECKPOINT \
  --device cuda:0 --kind responses --output NEW_RESPONSE_OUTPUT
python tools/honf_research.py evaluate-contextual --checkpoint CHECKPOINT \
  --device cuda:0 --kind group-execution --output NEW_EXECUTION_OUTPUT
```

The native evaluator retains nine equal-case metric boards, module-count strata, worst IDs and four input-selected detailed cases; it computes the selected-TRAIN mean baseline on the same masks without another model run. The existing response evaluator preserves physical atlas identities and endpoint maxima. Group execution evaluation measures fixed M1/M2 regions, early-layer group→M clamping, final address clamping with unchanged candidates, zero-value and exhaustive-group-search interventions, cached/complete Q32/Q1024/native calls, persistent tensor bytes and disposable full native/response/Adam timing macros. Its temperature-only zero-heat and original-column JVP/VJP probes do not evaluate heat-independent flow or certify physical gradients. Engineering timing optimizer steps are restored afterward and never counted as scientific training updates.

## Allocation and closeout

The initial GPU ceiling is 24 hours: 2 for qualification, at most 6 per justified 100-epoch arm and 4 for scientific evaluation. The 12-hour recovery reserve requires a measured reason and explicit allocation; it is not automatically spent. The overall ceiling is 36 GPU-hours. No full-data training, extension, extra seed, optional refinement, inverse campaign or new physical data is included. Monitor each launch once at startup and again near its estimated completion; use the delegated read-only Luna monitor and avoid duplicate parent polling.

All generated payloads stay under verified `/data2/wanglz/ModularDT` bindings, with at most 4 GiB retained, 512 MiB peak temporary storage and at least 200 GiB filesystem headroom. Reuse the existing packed dataset and atlas stores without copying them. At round closeout retain scientific checkpoints, full histories, selected numerical/reference evidence and PDF figure masters with necessary Markdown raster companions. Remove only exact round-owned disposable outputs after dependency/open-handle checks, retain a local inventory/receipt, report actual removed bytes and remaining capacity, and preserve historical runs and protected migration backups. Upload durable code, tests, configuration and documentation only after auditing the entire outgoing history and running the existing pre-push artifact gate.
