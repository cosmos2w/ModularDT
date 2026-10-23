# Multiview native routing evidence index — 2026-10-12

This index accompanies the [main report](20261013_035603Z_HONF_Multiview_Native_Routing_20261012.md). It records a bounded common2000 comparison, not six completed2500 endpoints. All artifact links resolve through the ignored campaign symlink `HONF_Proj/diagnostics/generated/multiview_20261012` to `/data/wanglz/ModularDT/diagnostics/multiview_20261012`; numerical artifacts, figures and one-time helpers remain local and are not uploaded. Prose/captions use continuous source lines.

## Run and data identity

| Domain/arm | Fresh run identity | Actual saved age | Common native/selected age |
|---|---|---:|---:|
| Thermal P | TMV1-8009e2e99fab4429966ef47d5d89319a | 2500 | 2000 |
| Thermal R1 | TMV1-b1787b1230814d23b8820363b0dfc872 | 2500 | 2000 |
| Thermal R2 | TMV1-72419885a38d4886886047dbd1dc8204 | 2100 | 2000 |
| Wind P | WMV1-963fb63fbd7a4685a645d0cf5dd9bc40 | 2000 | 2000 |
| Wind R1 | WMV1-621070cb7f8e479684adbcd2be68fd06 | 2000 | 2000 |
| Wind R2 | WMV1-6b347fc96e6f401ebb7079bf42ce438a | 2000 | 2000 |

Thermal comparison manifest raw SHA256 is `553d9dba0d7d13a6b897b7ed82db4ea111662c1c62512e80da52b73533064719`; Wind is `b41fd202686afefc52cc7cbe2640145732aa910aaad9f9c884b46e0490e78658`. Full TRAIN source-set digests are Thermal `02e821e9f10028e10dc9680f25edffc3f698840fb277b42b70ac1d1ea6b44d5c` (32 files) and Wind `27fcb208340de65534743e059f031d9f2ca3d1a559d5fb0b8e229ada3cd8490e` (31 files), sealed at8c13d09. Dataset bindings are `ThermalChannel:fee942000f78d6ac6a24e9865c2d7e77853b904ca11bf65ca8f9cbf1c87d1144` and `WindFarm:1322db58917ec1d4aea5b64c6d07481541051e879fcc087f0f533b1d545171f0`.

The [primary fit accounting](../../../../diagnostics/generated/multiview_20261012/receipts/primary_fit_lifetime_accounting_bounded_v1.json) (`f3bac7c5…45b211`) verifies complete actual histories, saved 100 cadence, exact checkpoint identities, all owned primary lifetimes and observed work. The [bounded resource stop](../../../../diagnostics/generated/multiview_20261012/receipts/root_primary_resource_bound_stop_v3.json) (`ea19b9fc…db2da1`) retains the requested2500 horizon and explicitly records the missing endpoints. [Thermal shared-prefix audit](../../../../diagnostics/generated/multiview_20261012/receipts/thermal_poststop_observed_shared_prefix_e2000_stream_audit_v2.json) (`cd5428d3…d25e8`) checks the common1–2000 observed stream despite unequal completed histories. [Wind stream audit](../../../../diagnostics/generated/multiview_20261012/wind_fresh_primary_01/matched_stream_audit_e2000_2b79a8ab52ece0ed.json) checks all2000 epochs.

The original [1900 stop acknowledgment](../../../../diagnostics/generated/multiview_20261012/receipts/root_original_e1900_ack_snapshot_v1.json) remains byte-preserved; the [2100 watcher result](../../../../diagnostics/generated/multiview_20261012/receipts/root_clean_stop_e2100_watch_result_v1.json) records the exact new request/acknowledgment. Neither stop is described as a completed2500 fit. [Original-cutoff user reply](../../../../diagnostics/generated/multiview_20261012/receipts/keep_original_cutoff_user_reply_v1.json), [shared-GPU amendment](../../../../diagnostics/generated/multiview_20261012/receipts/shared_gpu2_user_authorization_v1.json) and [original authorization](../../../../diagnostics/generated/multiview_20261012/authorization.json) preserve the unchanged caps and computational-latency waiver.

## Native fields, tails and selection

The completed [common native manifest](../../../../diagnostics/generated/multiview_20261012/evaluation/common_e2000_fallback/common_intervention_payload_v5/native_execution_manifest_common_intervention_payload_v5.json) is raw SHA256 `6fd09e3fc4c19ecc5e0dd35b5ef4b56d0ffc5d940e78634ac87af11d944090c7`; its [native metrics](../../../../diagnostics/generated/multiview_20261012/evaluation/common_e2000_fallback/common_intervention_payload_v5/native_results_v5_01/native_metrics.json) are `5d82062f6ddb6e4b4d2b02fca13ee81ba468b43c4fc3b5ffa59e5389fb4be593`. Twelve logical rows comprise six common terminal states and six saved 20 selected rows; all select 2000, so their ordinary predictions reuse identical physical arrays while four selected R1/R2 trace jobs execute separately.

The [complete native analysis](../../../../diagnostics/generated/multiview_20261012/evaluation/root_final_common_e2000_native_analysis_v6.json) (`7ed4c85e…a8fd4b`) contains all 22 Thermal cases, nine Thermal primary/auxiliary fields, p90/worst/M strata, and all 24 Wind rows,20 component/vector role metrics, reference amplitudes, row-p95 and equal-layout paired differences. The [selected20 analysis](../../../../diagnostics/generated/multiview_20261012/evaluation/root_final_selected20_native_analysis_v6.json) (`53e37920…1dc3`) applies the same bounded20-age selection opportunities. Prospective2500 physical gates are explicitly null/unassessed; descriptive common2000 diagnostics are not converted into endpoint passes.

| Wind layout | R1/P four-priority geomean | R2/P four-priority geomean |
|---|---:|---:|
| 23 | 0.958593 | 0.959447 |
| 42 | 0.859873 | 0.862067 |
| 43 | 0.991585 | 0.988600 |
| 57 | 0.982480 | 0.984733 |
| 142 | 0.978944 | 0.983090 |
| 154 | 0.993066 | 0.986701 |
| 160 | 0.990627 | 0.990732 |
| 182 | 0.890000 | 0.888709 |

These per-layout diagnostic geomeans use exactly near/downstream Uy/Uz; the overall headline geomean uses the four equal-layout mean errors. Individual worsening priority cells remain: R1 layout 43 downstream Uy ratio 1.004137 and154 near Uy1.011054; R2 layout 57 near Uy1.008301,142 downstream Uz1.003100,154 near Uy1.001158,160 near Uz1.000045 and downstream Uz1.002911. Every layout's overall four-field score improves, but not every field within a layout does.

The [fixed1000 probe](../../../../diagnostics/generated/multiview_20261012/evaluation/native_e1000_fixed_gpu2_v19_01/native_metrics.json) (`d1634928…11817d`) and common2000 representative arrays use Thermal TRAIN0002/0327/0345/0358 and DEV0277/0291/0294/0687; Wind TRAIN456/273/534/531 and DEV69/126/426/462. The [available Thermal2500 supplement](../../../../diagnostics/generated/multiview_20261012/evaluation/thermal_available_e2500_fixed_probe_v1/native_results_v1/native_metrics.json) (`66b29de9…ba9e3f`) contains only literal P/R1 on that same four-case panel. Its40 recorded input/query/support hash crosswalks pass; it is never merged into full22-case common metrics.

## Mature references

The [mature native evaluation](../../../../diagnostics/generated/multiview_20261012/evaluation/native_references_gpu2_12/native_metrics.json) (`76c439136426da0f9325526abf2a6eed221f457755c1dd839df2063ffa8d2f15`) contains the same authorized DEV22/DEV24 fields and representative supports, with own saved normalizers and native context. Thermal1404 and1804 are literal5000. Wind2102 and2103 are selected2440/2475 from2500 histories, not5000. Migrated Wind2202 is literal5000, E8, fullTRAIN420. Its state digest is `5df438fcbe812e36d457a8f76b0c4e808b30837f8ab44cea91f45395368e8b31`. The canonical saved result receipts bind raw checkpoint/state hashes and historical local-model pretraining; the main report never treats this as matched fresh training.

## Actual gathering and restoration

The selected common native result contains 32 baseline R1/R2 trace scenes,80 R2 interventions and16 exact restorations. Native destination/source/environment IDs, presence, coordinates, measures, bounded scores, densities, actual weights, 64-coordinate weighted content and final direct source reads are saved. The exporter labels feature views only by index. All executed-value and source-coordinate crosswalks are checked; no graph support is interpreted as physical causality, independent K response truth or sparse execution.

The [display-ID adapter qualification](../../../../diagnostics/generated/multiview_20261012/receipts/root_common_intervention_payload_qualification_v5.json) (`7a816ad4…bc138c`) accepts only the fixed intervention dictionary keys, removes the display-only `id` before the core intervention API and preserves caller dictionaries. Five positive/four negative CPU checks pass. The core's own safety and full-rebuild semantics remain unchanged.

## Complete inference/update work and memory

| Actual Thermal training work | P2500 | R12500 | R22100 |
|---|---:|---:|---:|
| Primary label receivers | 384,000,000 | 384,000,000 | 322,560,000 |
| Native surface queries | 66,659,232 | 66,659,232 | 56,000,768 |
| Native material queries | 133,318,464 | 133,318,464 | 112,001,536 |
| Supervised native operator rows | 48,000,000 | 48,000,000 | 40,320,000 |
| Response temperature union receivers | 17,356,247 | 17,356,247 | 14,579,604 |

These are observed engine work counters at each actual achieved horizon, not independent query samples or additive information units. Native curl expands/deduplicates primary receivers and the response/operator paths have their own counts; full padded/active reader and typed-pair work is retained in fit accounting. Each Wind arm executes589,824,000 primary queries at2000 with the complete five-role objective. The primary total2,860,032,000 does not absorb these distinct Thermal auxiliary counters.

The successful [memory/work result](../../../../diagnostics/generated/multiview_20261012/evaluation/common_e2000_fallback/memory_work_common_e2000_v3/result_common_e2000_v6_context/memory_work_result_v6_context.json) is raw SHA256 `d1f15d426a314411bef55c1bed618a49e9f71cfbbd1e5414195eb2295828da33` (2,932,560bytes). It strict-loads five mature and six common2000 models, completes 44 native forwards at TRAIN low/high M and Q1024/Q8192, six full-objective cloned macro0 AdamW updates and12 prepared/rebuild comparison forwards. Its144 observed value hooks and 72 prepared-reuse hooks count executed128-wide values, not one value network per view. All six prepared states reject stale parameter changes and restore model bytes exactly.

The [final dispatch](../../../../diagnostics/generated/multiview_20261012/receipts/root_dispatched_common_e2000_memory_v6_context.json), [full source-identity qualification](../../../../diagnostics/generated/multiview_20261012/receipts/root_memory_source_identity_adapter_qualification_v5.json) and [native E8 context qualification](../../../../diagnostics/generated/multiview_20261012/receipts/root_memory_native_context_metadata_qualification_v6.json) retain the exact physical cost-v3, native-v12/fresh-v19 loader and source identities. Fresh Wind context is explicitly8; mature context is its saved512/8. The context validator still rejects an incorrect count. The [completed six-update journal](../../../../diagnostics/generated/multiview_20261012/evaluation/common_e2000_fallback/memory_work_common_e2000_v3/macro0_attempt_journal_common_e2000_v6_context.json) records six entries/completions and zero checkpoint/solver/TEST writes or reads.

Macro updates use actual saved 2000 schedule/case draws, Thermal48/Wind24 effective batch with micro8, the complete Thermal primary/operator/response objective and all five Wind roles. Thermal has seven context preparations per typed route (six primary microbatches plus auxiliary context); Wind has three. Every preparation evaluates each of the six round/type values once at width128. Sampling and denominator digests match within each dataset. Peak allocation is measured before microbatch preparation through the optimizer boundary, before receipt observers; it is distinct from whole-fit peaks and resident-parameter storage. Full process RSS/HWM, shared loaded-model CUDA allocation/reservation, public input/context tensor inventory, incremental call workspace and logical/padded route/final-read work are retained separately.

The result's embedded original delivery budget still records prospective disposable85→91 for one profile. It is an immutable planning snapshot, not the campaign total: four six-slot reservations now give109 conservatively charged disposable updates; the first two attempts entered zero, the incomplete third completed six and the final fourth completed six, so97 were actually executed. These totals remain below192. Comparative latency is omitted by user authorization, not inferred from memory or parameter counts; the engineering price gate remains unassessed.

The completed [Wind native plane result](../../../../diagnostics/generated/multiview_20261012/evaluation/wind_native_plane_detail_common-e2000-actual-completion-v3_v10/result.json) (`f63c8ad1e815ed55a8d30dd1d214de58fe54c628134fdd1ad21bd26dc312cfad`) contains 24 native NPZ cuts for six systems on rows 69/M8 and 426/M30, direction270°, with 1,090,044 total valid points and2,136 forward chunks of512. Exact frozen coordinate/hash checks pass for all 24 arrays; no double rotation or interpolated cut is introduced. The native hub plane is z70.869949m; transverse native x is89.380272m for row69 and−19.546692m for row426. The selected indices may differ from requested physical coordinates and are labelled as the actual native planes.

## Saved responses

The [response manifest](../../../../diagnostics/generated/multiview_20261012/evaluation/saved_thermal_responses_common-e2000-actual-completion-v3_v12_manifest_retry2.json) (`96882513…c8ed99`) binds the selected common2000 checkpoints and all six stored signed pairs. The [response metrics](../../../../diagnostics/generated/multiview_20261012/evaluation/saved_thermal_responses_common-e2000-actual-completion-v3_v12/saved_signed_response_metrics.json) (`a1429306…f8af39`) and [response arrays](../../../../diagnostics/generated/multiview_20261012/evaluation/saved_thermal_responses_common-e2000-actual-completion-v3_v12/saved_signed_response_arrays.npz) (`3434e160…88008d`) retain666 numerical members, per-support weighted errors, signed mean/peak responses, exact heat-null flow, FP64 increments and legacy FP32 endpoint differences.

The saved source is development/audit exposure, not independent test evidence or a new physical solve. Provenance preserves the original0291 H5 `split=test` metadata and its adapter classification, current versus missing historical solver-source identity, no asserted SI conversion, q_normal proxy scope, one independent signed transfer direction per layout and0277 missing-baseline status. No nominal H5 array substitutes for that baseline.

## Failures and adapter qualifications

Failed execution is preserved and charged. Common native-v3 completed one ordinary model receipt before a trace-dispatch mismatch; native-v4 completed five model receipts and nine baseline trace captures before a display `id` was correctly rejected by the core API. These are incomplete executions, not fabricated predictions or model failures. Their partial scientific arrays remain retained. Native-v5 adds only qualified evaluator serialization/dispatch fixes; the sealed training/model definitions are unchanged.

Memory-v3 failed before model/inference/update work because a helper referenced an absent cost-module member; memory-v4 strict checkpoint loading rejected an outer source-provenance wrapper rather than the required inner identity, entering zero updates. The corrected-v5 strict source loader completed all six full-objective cloned updates, then failed an inference context receipt that used `prod([])=1` for fresh E8. All six successful updates from that incomplete attempt remain charged and its journal retained. The corrected source passes the complete provenance wrapper to metadata validation and only the exact inner identity to physical loaders; no source guard is relaxed.

Wind-plane-v9 retained20 partial native NPZs for fresh P/R1/R2 and classic2102/2103, then rejected an unsupported `receiver_features` keyword at migrated2202's public API. The v10 adapter removes that exact unsupported keyword; raw coordinates, E8 preparation, saved normalizer, chunk512, queries and model values are unchanged. A strict CPU-loaded migrated checkpoint gave byte-identical adapter/direct-public predictions on16 frozen native queries. Old source/geometry receipts are retained separately from the explicit repaired API identity.

The same-function audit preserves failed/complete rows, the old23% Q8192 overhead, the old20% target miss and legacy derivative/curl/proxy failures. Smaller validation overhead is not a physical improvement or a requalified historical gate. Fixed flags reject invalid inputs/intermediates before output or update commitment.

## Storage, protection and upload boundary

Before materialization the declared budgets were18GB on/data and2GB on/home,20GB total, with6GB estimated peak plus1GB temporary and20GB minimum headroom on each filesystem. Existing verified shared caches are reused; there is no per-arm geometry catalogue, full-dataset copy or persistent new dataset cache. The common endpoint delivery sub-budget is3GB. One metadata preparation exceeded its1MB estimate by 205,688bytes; the explicit4MB/8MB metadata addenda preserve that planning miss without erasing old receipts, and the overall delivery/campaign limits remain intact.

Cleanup is constrained to exact round-owned disposable allowlists with inode/link/ownership/size/hash checks, active process/open-handle rechecks and preserved receipts. Scientific checkpoints, source/scripts, histories, reference arrays, failed numerical evidence, reusable shared caches and provenance are protected. The historical inventory includes67 already-absent bindings. The protected shared-cache manifests are content-verified; one older baseline inode/mtime discrepancy with identical content is disclosed, rather than claiming every cache byte or all metadata were newly hashed.

The [GPU lifetime ledger](../../../../diagnostics/generated/multiview_20261012/gpu_process_ledger.json) charges every owned wrapper lifetime once, including failures; nested engine/forward/update timings are not added again. All work uses physical GPU2, UUID `GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39`, and the ModularDT environment. Foreign jobs and unrelated PhyCoFlow GPU0/1 jobs were observed without signalling. Actual primary/auxiliary work and disposal reservations remain distinct.

## Figure index

Six PDF masters contain60 pages of saved numerical evidence; the main report embeds19 essential PNG companions. All other PNG exports and superseded presentation versions are disposable after final visual/HTML QA; their immutable render receipts and source/failure evidence remain retained.

| Family | Retained PDF | Scope |
|---|---|---|
| 1 | [family1_learning_and_endpoint_boards.pdf](../../../../diagnostics/generated/multiview_20261012/evaluation/multiview_common_e2000_actual_completion_figures_v25/family1_learning_and_endpoint_boards.pdf) | Actual fresh learning/identity and contextual mature systems |
| 2 | [family2_thermal_native_fields_and_auxiliary_misses.pdf](../../../../diagnostics/generated/multiview_20261012/evaluation/multiview_common_e2000_actual_completion_figures_v25/family2_thermal_native_fields_and_auxiliary_misses.pdf) | Native T/v fields, signed residuals and material/interface misses |
| 3 | [family3_wind_native_hub_and_transverse_cuts.pdf](../../../../diagnostics/generated/multiview_20261012/evaluation/multiview_common_e2000_actual_completion_figures_v25/family3_wind_native_hub_and_transverse_cuts.pdf) | Low/high-M native hub/transverse Ux/Uy/Uz and residuals |
| 4 | [family4_actual_native_routing_reads_interventions_restoration.pdf](../../../../diagnostics/generated/multiview_20261012/evaluation/multiview_common_e2000_actual_completion_figures_v25/family4_actual_native_routing_reads_interventions_restoration.pdf) | Actual typed gathers/content/direct reads, five interventions and16 exact restores |
| 5 | [family5_full_native_accuracy_work_memory_and_fit_scopes.pdf](../../../../diagnostics/generated/multiview_20261012/evaluation/multiview_common_e2000_actual_completion_figures_v25/family5_full_native_accuracy_work_memory_and_fit_scopes.pdf) | Physical error, dense executed work, memory, full updates and actual fit scope; latency unassessed |
| 6 | [family6_saved_signed_responses_and_null_status.pdf](../../../../diagnostics/generated/multiview_20261012/evaluation/multiview_common_e2000_actual_completion_figures_v25/family6_saved_signed_responses_and_null_status.pdf) | Saved signed response maps, means, nonlinear peaks and interface limits |

The final [figure-data manifest](../../../../diagnostics/generated/multiview_20261012/evaluation/multiview_figure_data_manifest_common_e2000_v25_final.json) and [render receipt](../../../../diagnostics/generated/multiview_20261012/evaluation/multiview_common_e2000_actual_completion_figures_v25/figure_render_receipt.json) bind all physical numerical sources and exact producer/consumer versions. Presentation fixes select ordinary numeric NPZ exports instead of trace JSON, mask absent auxiliary rows, distinguish metadata scalars from source/environment catalogues, join work only within each dataset, bind the already-parsed response through its manifest, shorten labels and wrap plot footers. No pickle is enabled and no training/model/source physics, support, normalization, color limits or error denominator is changed. Superseded partial/complete render attempts remain documented even after their redundant visual payloads expire.

## Resource and storage closeout

All67 owned GPU ledger entries are closed:20,337.598621 associated seconds (5.649333hours) against the10-hour cap, counted once per owned lifetime. Scientific updates are46,400; actual disposable updates97 and conservative reserved/charged disposable109 against192. There were no pre-seal scientific pilots, second seed, formal fit, inverse search or new solves. The final protection check has zero new mismatches, preserves all67 historical absences and solver326/326, and verifies the original plan plus full32/31 TRAIN source sets.

Final cleanup removed742 regular files,7 disposable-fixture symlinks and76 directories:379,956,404 logical bytes (382,267,392 allocated inventory bytes). The earlier large qualification cleanup separately removed77 files/419,499,162 logical bytes, with smaller cleanup receipts retained individually. Selected19 PNG companions and six PDF masters, all scientific checkpoints/numerical arrays, source/scripts, historical failures and provenance remain. Actual remaining capacity is110,763,442,176bytes on/data (about103.1GiB) and42,771,361,792bytes on/home and/ (about39.8GiB). The final removal receipt identifies every expired payload; no source data, code or scientific checkpoint was deleted.

The [final cleanup receipt](../../../../diagnostics/generated/multiview_20261012/cleanup/root_final_multiview_cleanup_receipt_v1.json), [final preservation receipt](../../../../diagnostics/generated/multiview_20261012/receipts/protection_final_multiview_round_v4.json), [source/plan seal check](../../../../diagnostics/generated/multiview_20261012/receipts/root_final_training_source_and_plan_seal_check_v1.json) and [browser QA](../../../../diagnostics/generated/multiview_20261012/receipts/root_browser_final_report_qa_v1.json) retain the closeout evidence. Browser QA loaded19/19 figures at1280/390px with no overflow, broken images or console errors/warnings; final cleanup prose was added afterward and static link/continuous-paragraph checks were repeated. Figures and one-time renderers remain ignored locally.
