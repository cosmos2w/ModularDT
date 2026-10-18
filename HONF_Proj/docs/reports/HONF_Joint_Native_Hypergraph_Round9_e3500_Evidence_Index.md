# Round9 e3500 evidence index

This index accompanies the [decision-stop report](HONF_Joint_Native_Hypergraph_Round9_e3500_Report.md). Generated checkpoints, arrays, figures, one-time audit helpers and receipts remain local under the verified `/data2` root, reached through the ignored workspace symlink. Their absence from Git is deliberate; the report and durable implementation are versioned. All eight scientific lineages stop at literal epoch 3500. The original e5000 horizons remain unfinished; no extension or formal full-data run is authorized.

## Exact endpoint checkpoints

| Literal arm | Saved checkpoint | SHA256 | Bytes |
|---|---|---|---:|
| thermal_D9_M | [e3500](../../diagnostics/generated/joint_native_round9_20261017T210035Z/runs/thermal_D9_M/checkpoints/epoch_3500_model.pt) | `9fde741d332b6fbba7081023d5f75330e3a0e552462a4ea06751a4f4d288e208` | 25,583,522 |
| thermal_D9_ME | [e3500](../../diagnostics/generated/joint_native_round9_20261017T210035Z/runs/thermal_D9_ME/checkpoints/epoch_3500_model.pt) | `3ef9f8f8cdda780fa56c83ccbc2000de929986b53616e597093926a12f359d46` | 26,832,286 |
| thermal_H9_ME | [e3500](../../diagnostics/generated/joint_native_round9_20261017T210035Z/runs/thermal_H9_ME/checkpoints/epoch_3500_model.pt) | `843e17df16d56ed52335122b39dac7f8d8652f770d2b3b2e71b25cc11db8afab` | 27,961,082 |
| thermal_S9_ME | [e3500](../../diagnostics/generated/joint_native_round9_20261017T210035Z/runs/thermal_S9_ME/checkpoints/epoch_3500_model.pt) | `b5a266412d9df7dd9f08dc047f3cc705423d40d6119cfa9d8e712eedf2ec886d` | 27,852,706 |
| wind_D9_M | [e3500](../../diagnostics/generated/joint_native_round9_20261017T210035Z/runs/wind_D9_M/checkpoints/epoch_3500_model.pt) | `4abd8977bdc97085c49be4119b19189238d1e1627bb066f55af3ee2653d1255c` | 21,719,250 |
| wind_D9_ME | [e3500](../../diagnostics/generated/joint_native_round9_20261017T210035Z/runs/wind_D9_ME/checkpoints/epoch_3500_model.pt) | `f499b8df220c1a773548556be3ab8fb6d62824ec6eb7838ba937aaab94c064c9` | 22,900,238 |
| wind_H9_ME | [e3500](../../diagnostics/generated/joint_native_round9_20261017T210035Z/runs/wind_H9_ME/checkpoints/epoch_3500_model.pt) | `8bda53d5a5f7f6ab20db395b7cff4b6b9509640280c82250ac08d4d9feac0f96` | 24,026,346 |
| wind_S9_ME | [e3500](../../diagnostics/generated/joint_native_round9_20261017T210035Z/runs/wind_S9_ME/checkpoints/epoch_3500_model.pt) | `d0381d8c36b67133b40dc5cda3772e93491ea27e17d360b99e001e2c7bbd7520` | 23,975,186 |

The [scientific closure](../../diagnostics/generated/joint_native_round9_20261017T210035Z/receipts/scientific3500_complete_runtime_monitor_v2.json) has SHA256 `023f4bbba1bc8e245f3cb5e1c5748dce26f018e32a219a98520541dfcfce1f22`. It checks finite model/named Adam state, unchanged histories and sources, exact final-stage case/query streams, actual successful updates and continuous resource observations. Latest/best aliases, declared milestones and original source checkpoints remain retained. All eight [standalone inference exports](../../diagnostics/generated/joint_native_round9_20261017T210035Z/inference_exports_e3500/) passed the [guarded packaging check](../../diagnostics/generated/joint_native_round9_20261017T210035Z/receipts/terminal_inference_export_e3500_validation_v1.json); these are inference packages, not replacement scientific optimizer checkpoints.

## Parent and scientific bindings

| Binding | SHA256 / identity |
|---|---|
| Verified implementation ancestor | `6ecc1086454705dc791d9ea782dba0f68b459e55` |
| Thermal D8 e2500 | `d9e47e6f70d44c5d06b357ef64b11fe5113a5382e6c2d95a9148bba146805b4e` |
| Wind D8 e2500 | `cad56856c30b89177a3cf72cbd04e0c6baeccd2c351366f421d305b06b3fa763` |
| Shared Thermal M+E e3000 | `4afbefa80d8a9aaa1ed3deba5cc7032b914e85f9fc20484e81b39a622e2d904c` |
| Shared Wind M+E e3000 | `f367756a2292f04342991d2b9db788ae3bbbb048a9bacf67840f21910e914fbf` |
| D9-M scientific source | `a13329d85b0895a58f87e4883f65b8f741f71a35` |
| Warm/fork scientific source | `96ca8bef53a44c267799e9a97e145e7a2d6732a6` |

Source inventories, native contracts and exact migration are retained in [parents](../../diagnostics/generated/joint_native_round9_20261017T210035Z/receipts/parents.json), [source contracts](../../diagnostics/generated/joint_native_round9_20261017T210035Z/receipts/source_native_contracts.json) and [migration](../../diagnostics/generated/joint_native_round9_20261017T210035Z/receipts/exact_source_migration.json). Thermal/Wind train memberships, original normalization/query policies and all eight recipes remain pinned. The original constant-LR recipe horizon 5000 is preserved as identity metadata, not a command to continue.

## Architecture sizes

| Dataset | D9-M | D9-ME | H9-ME | S9-ME |
|---|---:|---:|---:|---:|
| Thermal | 1,520,493 | 1,621,553 | 1,621,588 | 1,621,565 |
| Wind | 1,525,675 | 1,627,695 | 1,627,730 | 1,627,707 |

These are actual loaded architecture counts from the [parameter inventory](../../diagnostics/generated/joint_native_round9_20261017T210035Z/receipts/actual_parameter_inventory_v1.json). H adds 35 parameters and S adds 12 over the common dense M+E value function, while M+E adds roughly 101,000 over dense M. Small router parameter count does not imply small runtime or memory overhead. D9-ME versus D9-M changes architecture/capacity as well as direct E access; the fixed-state E ablations answer a separate dependence question.

## Endpoint manifests and reconciled attempts

| Artifact | SHA256 |
|---|---|
| [terminal_physical_manifest_v9.json](../../diagnostics/generated/joint_native_round9_20261017T210035Z/receipts/terminal_physical_manifest_v9.json) | `17c509ca3635254e3b08ee0d828e88fad9aed619b45d50637c692daa1758ba16` |
| [e3500_board_reconciliation_registry_v2.json](../../diagnostics/generated/joint_native_round9_20261017T210035Z/receipts/e3500_board_reconciliation_registry_v2.json) | `18aaa5764b301ec964558907604b670aa67a4131a1e8d4ec05079e13228e9acf` |
| [terminal_execution_manifest_v5.json](../../diagnostics/generated/joint_native_round9_20261017T210035Z/receipts/terminal_execution_manifest_v5.json) | `a94c8964f3c11f36ae93af07ca61d59078ec8b8f28ee03784e9ebce05f911d8e` |
| [execution_e3500_attempt_registry_v1.json](../../diagnostics/generated/joint_native_round9_20261017T210035Z/receipts/execution_e3500_attempt_registry_v1.json) | `ce4c866b8b66ed03e95df893ac08196b70aac98e9ab8dcca0fe295a1597fac22` |
| [resource_reservation.json](../../diagnostics/generated/joint_native_round9_20261017T210035Z/receipts/resource_reservation.json) | `3f6ffad68c1cbbc1905992207c38bdc17e538b69e7597719d9b150ff2b143be7` |
| [resource_reservation_gpu1_v1.json](../../diagnostics/generated/joint_native_round9_20261017T210035Z/receipts/resource_reservation_gpu1_v1.json) | `6b9dd0d6e34675abd6cafe14308a246eb0cf6eb1e0671b0de6447df55e451242` |
| [user_GPU1_GPU2_authorization_v1.json](../../diagnostics/generated/joint_native_round9_20261017T210035Z/receipts/user_GPU1_GPU2_authorization_v1.json) | `fcf47b880f4dc45cb4c8739500439fa482c42b5322af2e1e6dc414ebd9e6a395` |
| [user_e3500_decision_stop_amendment_v1.json](../../diagnostics/generated/joint_native_round9_20261017T210035Z/receipts/user_e3500_decision_stop_amendment_v1.json) | `21e1446f0d6656228f63ecaa6ab0e8cf1932f6f4419d0f69a827a10e29464ad2` |

The board registry reconciles three completed v5 boards, twelve completed v6 boards and the final Thermal D9-ME TRAIN v7 attempt. Its first timeout and subsequent stale output-token rejection remain historical failed attempts. Six mandatory physical audits completed under v7 before an old transitive GPU2 admission check rejected the first typed-control job; v8 changes that admission only and retains the completed owners. Fourteen more physical jobs then completed before Wind seam discovery found an evaluator access to a nonexistent scene `.support` attribute. Version 9 rebuilds the same native query features from the real target-free scene support bounds; it retains the twenty completed physical owners and reissues only the eighteen remaining scopes, with a fresh namespace for the failed seam discovery. The original evaluator, failed receipt and completed measurements remain preserved; no physical target, query draw, model or training state changes. Execution v4 owns the fourteen completed memory audits. The first latency attempt failed before timed repetitions because host tensors were not transferred; v5 repairs transfer accounting and reissues only the 84 uncompleted inference/reuse scopes under new attempt2 IDs. No completed measurement is repeated for these repairs, and all failed time is charged.

The complete endpoint comprises **156 applicable logical scopes**: 16 full native boards, four CPU field comparisons, 38 physical/read/control audits, fourteen memory audits and 84 complete inference/reuse measurements. The [final numerical summary](../../diagnostics/generated/joint_native_round9_20261017T210035Z/arrays/final_e3500_summary_v1/summary.json), SHA256 `9c3eed676224140346d319bcc254e6feb9a2113e9b18331344fd2916c1bf355d`, authenticates complete mandatory physical and execution scope. Historical e3000 boards and failed/reissued attempts are retained separately. The [artifact index](../../diagnostics/generated/joint_native_round9_20261017T210035Z/receipts/final_e3500_artifact_index_v2.json) has SHA256 `9b1f5cb0abcd9ec499ee0dbce52e9e899237d8b7604842ca62a68d7a4f0bd522` and records checkpoint, output, native-array, source and owner hashes with zero errors. It curates exactly eight retained PDF masters (22 pages) and nine PNG companions; the original v1 index and superseded figure metadata remain historical evidence. Successful execution means a measurement completed; it does not mean every scientific gate passed.

After terminal authentication, the Wind bounds-only seam fix was applied to the public evaluator and verified by 18 CPU tests, including real `WindContentSceneInputs`, the native adapter and moved 3D query features. The [regression receipt](../../diagnostics/generated/joint_native_round9_20261017T210035Z/receipts/public_wind_seam_regression_v1.json), SHA256 `374d6bea0e6e7fce35f5d1f81ae4d1063fda996216c44ebddb2992d1c58ba174`, records the new code/test hashes and zero GPU/optimizer work. The original evaluator hash `0905a776b85d2911237a83f1f880117686d0ca7c04f168d68207c1805075dc0e` remains in an immutable local archive through the [explicit historical source-resolution receipt](../../diagnostics/generated/joint_native_round9_20261017T210035Z/receipts/public_wind_seam_original_source_archive_v1.json). Old manifests remain unchanged; their historical path/hash resolves to those archived bytes, rather than claiming the current repaired file is byte-identical.

## Numerical evidence and figure masters

| Evidence | Retained source |
|---|---|
| Full endpoint mean/p90/worst/strata comparisons, 26,352 rows | [Comparison CSV](../../diagnostics/generated/joint_native_round9_20261017T210035Z/arrays/final_e3500_summary_v1/all_roles_mean_p90_worst_strata_comparisons.csv) |
| Per-field E output changes, error changes and controls, 696 rows | [E utility CSV](../../diagnostics/generated/joint_native_round9_20261017T210035Z/arrays/final_e3500_summary_v1/per_field_E_effects_and_error_change.csv) |
| Native endpoint cohorts, physical receivers and representative arrays | [Full-cohort table](../../diagnostics/generated/joint_native_round9_20261017T210035Z/arrays/report_tables_e3500_v1/per_field_full_cohort.csv), [native boards](../../diagnostics/generated/joint_native_round9_20261017T210035Z/arrays/physical_boards/) |
| All typed controls, E interventions, native invariance and shared seams | [Physical audit arrays and receipts](../../diagnostics/generated/joint_native_round9_20261017T210035Z/arrays/physical_interactions/), [physical reconciliation](../../diagnostics/generated/joint_native_round9_20261017T210035Z/receipts/e3500_physical_reconciliation_registry_v2.json) |
| Saved-reference response channels | [Response CSV](../../diagnostics/generated/joint_native_round9_20261017T210035Z/arrays/response_report_tables_e3500_v1/physical_response_channels.csv) |
| Four-epoch training memory and low/high-M parity probes | [Selected execution owners](../../diagnostics/generated/joint_native_round9_20261017T210035Z/receipts/execution_e3500_attempt_registry_v1.json), [phase-memory table](../../diagnostics/generated/joint_native_round9_20261017T210035Z/figures/graph_work_e3500_attempt2_v1/work_selected_v2/phase_memory.csv) |
| 48 descriptive calls, twelve paired calls and 24 reuse scopes | [Actual inference summary](../../diagnostics/generated/joint_native_round9_20261017T210035Z/receipts/actual_inference_attempt2_summary_v1.json) |
| S and dense scientific-cycle observations | [e3497–3500 observational receipt](../../diagnostics/generated/joint_native_round9_20261017T210035Z/receipts/scientific_S_observational_cycle_e3497_3500_GPU1_v1.json) |
| Conditional ranking disposition | [Unqualified status](../../diagnostics/generated/joint_native_round9_20261017T210035Z/receipts/endpoint_conditional_saved_candidate_status_e3500_v2.json); no candidate evaluation/solver launched |
| Selected visual evidence | [Report figure index](HONF_Joint_Native_Hypergraph_Round9_e3500_Report.md#figure-index), [graph/work visual review](../../diagnostics/generated/joint_native_round9_20261017T210035Z/receipts/graph_work_final_visual_review_v1.json) |

The final summary and its two CSVs occupy 14,934,069 bytes within the unchanged 16 MiB consumer-output cap. Raw native arrays remain separate and source-referenced; they are not duplicated into the report summary.

## Claim-by-claim disposition

| Claim | e3500 disposition |
|---|---|
| Eight literal matched scientific endpoints | Completed amended horizon only; original e5000 horizons unfinished |
| Actual typed M+E nonlinear value reads | Measured; original IDs, eligible rows, protection and executed counters retained |
| E-conditioned native output dependence | Measured with fixed-support E interventions; restoration checked |
| E-conditioning predictive utility | Resolved on parts of the exposed TRAIN intervention panel; field-dependent |
| Joint graph organization/adaptation | Actual saved typed incidence and legal-coordinate controls; model organization, not physical causality |
| Field preservation versus both dense controls | Mixed; Thermal and Wind TRAIN/tail/stratum misses remain visible |
| H superiority over trained S | Not established; H/S work is not uniformly matched |
| Actual versus logical sparsity | Both measured separately; padding and dense estimator work are not learned omission |
| Selected-step memory saving | Declared 20% goal missed at this age |
| Complete four-epoch capacity saving | Not observed; dense-router steps and selected peaks retained |
| Complete native-call speed | Measured; paired H calls slower, with no required speedup substituted |
| Physical geometry/heat response fidelity | Mixed, including a substantial mixed fluid-temperature response miss and visible hard-route spatial seams |
| Independent environmental interaction | Unavailable: E is geometry-derived and independent forcing references are absent |
| Conditional saved-design ranking | Not qualified by measured-memory prerequisite; not executed |
| New inverse/design success | Untested; no new inverse campaign or solver calls |
| Full-data/locked TEST generalization | Untested; development cohorts remain exposed and TEST remains locked |

## Preserved failures, exclusions and conditional scope

The [Thermal native boundary audit](../../diagnostics/generated/joint_native_round9_20261017T210035Z/receipts/thermal_native_boundary_audit_v3.json) fails strict one-step parameter equality (maximum 2.9802322387695312e−8), while its separately scoped forward recovery passes. The corresponding [Wind audit](../../diagnostics/generated/joint_native_round9_20261017T210035Z/receipts/wind_native_boundary_audit_v3.json) passes. Neither changes the literal saved-state inheritance or justifies relabelling failed strict parity.

The [historical e3300 fixed-panel diagnostic](../../diagnostics/generated/joint_native_round9_20261017T210035Z/receipts/thermal_H9_ME_historical_e3300_fixed_panel_replay_v1.json) fails exact native-loss equality; later packed-support/denominator comparisons were not reached. Its original monitor did not trigger the same test under its own equal-case statistic. The failure and restored state are retained, without a loosened tolerance or repeat. Original e3000 boards remain historical all-open boundary evidence and are not called sparse learning. Superseded unexecuted e4000/e5000 plans remain unexecuted.

Independent E-forcing physics, a full geometry-plus-heat factorial, full-population generalization, locked TEST performance, a new solver validation and a new inverse/design campaign are unavailable. The saved-design ranking remains conditional and is not executed without its measured-memory prerequisite. The inherited closed 326/326 solver receipt is preserved, with the bounded-search limitation that no independent global ledger was located. New Round9 solver attempts are zero.

## Cost, cleanup and final preservation

| Receipt | SHA256 |
|---|---|
| [Complete cost v4](../../diagnostics/generated/joint_native_round9_20261017T210035Z/receipts/complete_cost_exposure_e3500_v4.json) | `e6b3f3cd25c15bf1b3a21121d4e01b6bcf7fcc54d10c95e5c5a20f61ee004f30` |
| [Cumulative cleanup counts v2](../../diagnostics/generated/joint_native_round9_20261017T210035Z/receipts/closeout_cleanup_cumulative_counts_v2.json) | `fdcb2e28bc7afb45caf3bc87e81b05e8f5b65228759aea0cc974ce524b1766ae` |
| [Protected closeout](../../diagnostics/generated/joint_native_round9_20261017T210035Z/receipts/final_cost_cleanup_protected_closure_v1.json) | `f3575beebdf9a70443ffa341665e8ee7c01e25750fed8ca71c17c9389211d9e9` |
| [Final retained temporary evidence and capacity](../../diagnostics/generated/joint_native_round9_20261017T210035Z/receipts/final_tmp_retention_and_capacity_v1.json) | `240bedcde438c76a8c0e6be1e20c20071560cdc0e5efc831c972d87d5d2feacd` |

The complete cost is 24,357.599 seconds of deduplicated GPU-associated process time, with 5.802 seconds of terminal CPU comparisons and 47.815 seconds of recorded CPU rendering/export/regression intervals kept separate. Uninstrumented CPU summary/index and visual-review time remain unavailable. The report gives all scientific exposure, failed attempts, nested scopes and contention limitations. Cleanup removes only verified disposable payloads; the retained 2,406,008 bytes under `tmp` are source and scientific binding provenance. All endpoint checkpoints, protected history, TEST/solver bindings and unrelated user changes remain preserved. GPU1 is idle and further training remains unauthorized pending the user's decision.
