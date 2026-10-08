# Unified interaction training and source-preserving refinement

The maintained refinement family keeps an inexpensive read from every active physical source and uses receiver-specific gates to add selected fine corrections. Thermal and Wind execute the same context, base, organizer and training engine with separate weights. Thermal retains its qualified affine heating law; Wind retains its nonlinear physical velocity readout. An execution contract or a successful software test does not establish field fidelity or useful organization; the development report supplies those measurements.

## Shared computation and dataset responsibilities

The opt-in implementation is [interaction_refinement.py](../../src/honf_forward_core/interface_fields/interaction_refinement.py). Existing fine classes and their checkpoint keys remain available unchanged. The shared context contains every physical source and the full environmental ancestry. The cheap path projects source and query features to sixteen dimensions before combining them; it does not first allocate the full fine hidden-width receiver/source stack. The organizer has hidden width32 and reads cheap contextual features, receiver features and relative geometry. Targets are confined to the task providers' supervision objects.

For each receiver/source pair, the executed read is `B + g*(F-B)`. A hard route evaluates the expensive fine head only on selected pairs during inference. Every original positive-support near pair remains selected. The diagnostic `all_base` setting uses the coarse read outside this protected near support; it keeps those near reads fine. Unselected sources retain their cheap read and their effects on the contextual states. Environmental contextualization remains full; the source gate does not prune environment records. Physical source IDs remain attached to individual reads rather than an averaged latent bank.

| Responsibility | Thermal | Wind |
| --- | --- | --- |
| Inference inputs | Geometry, materials and prescribed boundary/flow conditions | Native turbine geometry, direction, prescribed conditions, support features and measures |
| Separate control | Heating, applied after preparing coefficients | None; configuration changes require new preparation |
| Output law | Zero-offset affine temperature coefficients for the qualified benchmark | Refined vector messages, physical-measure reduction and nonlinear velocity head |
| Fine initialization | Preserved development R-direct2500 | Fresh H64/message64 |
| Environmental representation | Existing geometry-only environment; optional predicted-flow injection disabled | Explicit eight records, `token_shape=(2,2,2)` |
| Output transform | Existing fixed25 TRAIN scales and native shared-grid extraction | Selected-TRAIN velocity normalization plus a common TRAIN-fitted empirical height profile |
| Physical reference | Analytic-flow/shared-grid Thermal benchmark | Stored OpenFOAM fields |
| Legal derivatives | Fixed-route heat increment/VJP and live local geometry/query derivatives | Local physical JVP/VJP; no exact affine finite increment |

`RefinedSourceResponseOperator.from_fine(parent.core)` preserves the original fine parameters and adds only `refinement.*` state. All-fine inference dispatches to the original fine reader; it does not subtract and re-add the cheap path in FP32. Mixed precise Thermal reads retain raw factors and widen them before kernel assembly and forcing contraction. Heating, action radii and observed fields do not enter the Thermal base or gate.

`RefinedNonlinearFieldReadout.read_refinement` returns the deployed field, an already-paid full field when training requests replay, and execution diagnostics. Sparse inference does not silently compute a full teacher field. Exported fine tensors at omitted pairs are placeholders; `complete_fine_values` identifies whether a complete fine tensor was actually computed. Wind messages are latent computational values, not velocity contributions or physical control derivatives.

## One optimizer loop

[unified_training.py](../../src/honf_runtime/unified_training.py) owns epoch visitation, deterministic sampling keys, microbatch accumulation, exact final-boundary denominators, AdamW, absolute schedules, stage changes, train/eval mode, monitoring, selection and checkpoint/resume. [Thermal's provider](../../Case_ThermalChannel/src/channelthermal/training/unified_task.py) owns native scalar extraction, inherited reconstruction/response/operator objectives and response-development guards. [Wind's provider](../../Case_WindFarm/src/windfarm/training/unified_task.py) owns native role sampling, physical transforms and velocity objectives. Neither provider has an independent optimizer loop.

| Sealed development contract | Thermal | Wind |
| --- | --- | --- |
| Membership | `fixed25_v1`, 150 TRAIN /22 DEV | `wind_shared_fixed24_v1`, 24 TRAIN layouts /72 direction rows; eight DEV layouts /24 rows |
| Micro/effective cases | 8/48 | 4/24 |
| Complete-epoch updates | Four; final boundary has six cases | Three |
| Shared all-fine warmup | New epochs1–500 | New epochs1–500 |
| Matched branches | Full-detail and Adaptive-detail from the identical epoch500 state | Full-detail and Adaptive-detail from the identical epoch500 state |
| Requested ages | Total-new1000; healthy maturation to2500, including warmup | Total-new1000; healthy maturation to2500, including warmup |
| Monitoring | Every100 epochs and an explicit final/clean stop | Every100 epochs and an explicit final/clean stop |

The sampler excludes the arm from its seed. Both branches reuse the same membership, normalizer, queries, native objectives and absolute physical-model schedule. Checkpoints bind the provider identity, engine configuration, parameter-group membership, schedule, RNG, epoch cursor and provider calibration state. A branch is permitted only from the shared warmup endpoint; an exact resume requires the same arm and identity. Development and formal memberships remain distinct identities.

The native task objective combines the deployed main prediction with already-paid full-detail replay using weights0.75/0.25 in adaptive training. Full-detail uses the same fine prediction with total unit weight. The cheap approximation target is detached fine output/message state with a fixed TRAIN scale. Training computes full fine reads for this target and replay, so a hard forward mask is not a sparse-training-speed claim.

The adaptive router opens forward gates during501–600 while learning a detached residual-importance target. During601–800 it uses a soft route with temperature annealed from1 to0.1. From801 onward, inference uses a hard gate and training supplies a zero-valued restoring surrogate to the organizer only. Input features of that surrogate are detached; physical derivative consumers use the actual fixed hard route or an explicitly labelled soft model. One TRAIN-only gradient calibration after600 targets a5% router-native gradient share with coefficient cap0.1. Thermal achieves5%; Wind hits the cap and achieves0.37912%, recorded explicitly in its calibration receipt. There is no forced K or quota on partial receivers.

Thermal fine parameters retain parent AdamW moments, warm from3e-6 to5e-5 over new epochs1–20, hold through1000 and decay to3e-6 by2500. New cheap/router parameters use fresh moments and3e-4 peak. Fresh Wind parameters use3e-4 through1000 and decay to3e-6 by2500. The router's optimizer age begins when it first receives a gradient, separately from the physical predictor's epoch clock.

Thermal field selection and response-guarded selection are separate. The guard checks mean fluid/surface/material response-development RMSE against the starting development parent with limit1.10 and fixed TRAIN near-zero scales. Its four response-development families differ from the four fitted TRAIN addenda and from counted replay outcomes. If no child qualifies, response use retains the parent. Wind has no independent finite-response label guard. Literal1000/2500 checkpoints remain separately reportable from selectors.

## Development runner lifecycle

`tools/unified_interaction_train.py` is the maintained entry point for these development profiles. The default profile seed is0 for Thermal and42 for Wind; the seed, initial model-state hash, provider identity and engine schedule are sealed into every run. Run IDs must stay the same for warmup, branches and exact resumes.

`prepare` writes a provider, parent and membership receipt without starting an optimizer. `dry-run` exercises one disposable CPU engine update and writes no checkpoint. `start` begins the shared warmup or branches a matched arm from its epoch500 warmup checkpoint in a new output without existing training history. `resume` requires that output's current `latest_model.pt`, the same task, run ID, profile seed, arm, provider identity and schedule; it rejects historical checkpoints that would rewind history. A process-owned run lock prevents concurrent writers. `status` reads the process, checkpoint and history receipts and checks PID/start identity; an unverified historical running receipt is labelled accordingly. Runtime failures record `failed`, and `clean-stop` preserves any pending request identity and requests a stop after the current complete epoch.

For example, from the ModularDT checkout root, the Thermal development sequence is:

```bash
CUDA_VISIBLE_DEVICES="" python HONF_Proj/tools/unified_interaction_train.py prepare --task thermal --run-id thermal_refinement
CUDA_VISIBLE_DEVICES="" python HONF_Proj/tools/unified_interaction_train.py dry-run --task thermal --run-id thermal_refinement
CUDA_VISIBLE_DEVICES=0 python HONF_Proj/tools/unified_interaction_train.py start --task thermal --run-id thermal_refinement --arm warmup --stop-after 100 --device cuda:0
CUDA_VISIBLE_DEVICES=0 python HONF_Proj/tools/unified_interaction_train.py resume --task thermal --run-id thermal_refinement --arm warmup --stop-after 500 --device cuda:0
CUDA_VISIBLE_DEVICES=0 python HONF_Proj/tools/unified_interaction_train.py start --task thermal --run-id thermal_refinement --arm full_detail --stop-after 1000 --branch-from HONF_Proj/diagnostics/generated/unified_refinement_20261007/runs/thermal/thermal_refinement/warmup/latest_model.pt --device cuda:0
CUDA_VISIBLE_DEVICES=0 python HONF_Proj/tools/unified_interaction_train.py start --task thermal --run-id thermal_refinement --arm adaptive_detail --stop-after 1000 --branch-from HONF_Proj/diagnostics/generated/unified_refinement_20261007/runs/thermal/thermal_refinement/warmup/latest_model.pt --device cuda:0
```

Continue either arm from its own `latest_model.pt` with `resume --arm full_detail` or `resume --arm adaptive_detail`; set `--stop-after 2500` only after the saved review gate and budget approval. Wind uses the same commands with `--task wind`, its fixed profile seed42, and the assigned Wind GPU. The [manual5000 guide](Unified_Interaction_Manual_5000.md) provides the distinct full-data factory and lifecycle commands; the development factory and this development CLI do not create or launch that run.

## Compiled packet and derivative consumers

[receiver_packets.py](../../src/honf_forward_core/interface_fields/receiver_packets.py) retains the historical value-only packet API. `compile_receiver_request` copies immutable external IDs, coordinates and masks at a validated trust boundary, builds tensor source/receiver maps and partitions row-local fallback. Unsupported rows use full access while qualified rows retain subset execution, joined in the original order. Repeated calls validate bounded parameter/input/version/configuration ownership without reconstructing every host coordinate tuple or issuing one CUDA scalar write per pair.

`apply_differentiable_fixed_route` is the separate autograd-capable affine packet consumer. Its exact baseline, scene, source order, receiver identity, units, action box and prepared context must agree. Physical tensors remain live; a geometry move requires rebuilding context, route and baseline. The public refinement readers accept an explicit `fixed_route` only for adaptive hard inference without a training surrogate. Such derivatives hold membership fixed and do not differentiate through a discrete gate transition.

The Wind adapter binds velocity-transform identity, scalar reference velocity and normalization-array identities/contents to prepared ownership. It also binds the empirical background profile. Mutation or replacement invalidates endpoint and local-linearization use; callers must prepare again explicitly. Normalization is part of the physical model contract, not incidental postprocessing.

## Evaluation and interpretation

Separate native physical error, same-weight full-model distortion, active fine rows, padded rows, cheap/gate rows, protected near rows and full context pairs. Rows from layers of different width are not interchangeable FLOPs. Full training pays auxiliary, replay, native extraction and optimizer work. Cold catalogue initialization, new-scene calls, prepared reads and physical derivatives have different scopes; report complete outputs and mandatory guards in each measured scope.

Same-weight all-fine/all-base/learned routes, equal-count nearest/upstream controls and one degree-matched shifted source-order reversal test the deployed correction. The reversal is a permutation for every source count. Label permutation with physical objects is instead an invariance check. Neither control proves physical causality. A field-preserving fine removal is relevant only when the native predictor itself is adequate.

Static Wind geometry/role catalogues cache native indices, measures and coordinates rather than target volumes. Fresh epoch queries use the fixed source rows and all three directions. Wind development role-loss scales are inherited Run2111 Stage-A objective constants from `configs/active_packet_organizer_reuse.json`, separately sealed by source SHA; they were not newly fitted on fixed24 TRAIN. The fresh physical normalizer and height profile use selected TRAIN rows only. The subset omits TRAIN M18 and must retain that disclosure. The older H32 pilot actually used512 environment atoms despite its eight-atom receipt label; the new eight-atom/H64/background recipe is a different input/capacity/history combination, so gains relative to that pilot cannot be assigned solely to routing.

Generated configurations, checkpoints, numerical tables, one-time campaign drivers and inspected figures remain in ignored local paths. Durable implementations, tests, profiles, guides and the report may be uploaded after the repository's complete outgoing-history audit. No software contract authorizes new solves, formal training or an inverse-generation campaign.
