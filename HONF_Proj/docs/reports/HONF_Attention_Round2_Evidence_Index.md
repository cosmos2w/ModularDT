# HONF attention Round2: evidence index

This index belongs to the geometry/content study specified in [the Round2 execution plan](../../UpgradePlan/HONF_Attention_Round2_Geometry_and_Content_Development_Plan.md). The local evidence root is `HONF_Proj/diagnostics/generated/attention_round2_20261014T135200Z`, a symlink to `/data/wanglz/ModularDT/diagnostics/attention_round2_20261014T135200Z`. Numerical evidence, figures, receipts and one-time helpers remain ignored local artifacts. The reviewed starting lineage is `852f995403a91d8feb0e5df94f3022c7b1f3467b`; run identities retain their actual recipe, imported-source inventory, data and normalization hashes.

## Scientific contracts and source

| Contract | Maintained source or retained evidence |
|---|---|
| Mathematical amendment and bounded protocol | [Round2 plan](../../UpgradePlan/HONF_Attention_Round2_Geometry_and_Content_Development_Plan.md), [Oct13 mathematical specification](../../UpgradePlan/HONF_Redesign_Oct13.md), [Oct13 execution record](../../UpgradePlan/HONF_Redesign_Implementation_and_Evaluation_Oct13.md) |
| Shared v2 preparation and decoding | `src/honf_forward_core/content_hypergraph/model_v2.py`, `staged_attention_v2.py`, `spatial_features_v2.py`, `content_bank_v2.py`, `readers_v2.py`, and versioned `contracts.py` |
| Native Thermal | `Case_ThermalChannel/src/channelthermal/content_hypergraph.py`, `training/content_hypergraph_task_v2.py`; maintained native interpolation, q and curl modules |
| Native Wind | `Case_WindFarm/src/windfarm/content_hypergraph.py`, `training/content_hypergraph_task.py`; corrected physical residual law and native role catalogue |
| Training and strict loading | `src/honf_runtime/unified_training.py`, `src/honf_runtime/content_hypergraph.py`, `tools/content_hypergraph_train.py` |
| Full-run qualification | `src/honf_runtime/content_hypergraph_qualification.py`; measured native fields, content use, numerical checks, complete costs, B8 feasibility, partition/source binding and resource reserve |
| Recipes and usage | `src/config_core/forward/content_hypergraph/*a2-*.json`, [Round2 guide](../guides/Content_Hypergraph_Round2.md) |
| Input/resource seals | Local `authorization.json`, `receipts/input_seal.json`, `receipts/pre_outcome_criteria_seal.json`, `receipts/storage_peak_forecast_v2.json` |

Thermal uses fixed25 TRAIN150/exposed DEV22, manifest `933b0138ba2f8447a1ecadfe31fd0bb2cb4a05607d3ac3d9f0dc79419f196044`. Wind uses fixed24 TRAIN72/exposed DEV24, manifest `b223e174d7262682a7e1ac1977cb38f721993e14c0340b2dc0f5fb5f95bfa715`. Wind's exact DEV Q8192 panel SHA256 is `594681b0f270eb82ed271adf5d962c00be0b1885f9e99b157662c5b2c3f028ed`, with four roles of1640 rows and background1632. This corrects the obsolete820-row metadata from the first-round copied summary; the actual arrays and metrics are not rewritten.

## Qualification and segmented fitting

The executable environment is `/home/wanglz/miniconda3/envs/ModularDT/bin/python`, Python3.12.13, Torch2.6.0+cu124, FP32, TF32 off and four CPU threads. Physical GPU0 UUID is `GPU-233fcd85-5c6a-6212-3f44-655252afec70`; GPU1 UUID is `GPU-3ceda40c-fd5c-4b88-6c47-b3301711571e`. Each GPU command exposes one UUID as logical `cuda:0`. GPU2 is excluded from round computation.

| Evidence | Local path / interpretation |
|---|---|
| Combined native/core/runtime tests | `logs/modulardt_centroid_final_combined.log` and matching receipt:321 passed in21.12s; `HONF_Proj/tests/test_content_hypergraph_wind_r1.py`:8 nonoverlapping tests passed in1.62s;329 unique tests across these commands, with earlier overlapping suites excluded |
| Actual TRAIN gradient/native checks | `source/thermal_v2_gradient_coverage_modulardt.py`, `source/thermal_v2_affine_adjoint_modulardt.py` and their retained receipts/logs; both datasets' GPU qualification records in `gpu_process_ledger.json` |
| Smooth geometry fixture | `source/geometry_derivative_v2_cpu.py`; synthetic smooth-core derivatives with stable local membership, not native catalogue/stencil physical geometry labels |
| All disposable optimizer work | `disposable_optimizer_updates.json`;73 actual updates, including the interrupted Thermal H6 attempt, none used to initialize scientific fitting |
| Segment accounting | `gpu_process_ledger.json`; all owned GPU-associated attempts, failures, completed stages, wall time and child resource use |
| Paired1000 review | `receipts/thermal_a2d_h6_e1000_pair_review_v1.json`, `receipts/wind_a2d_a2h6_e1000_pair_review.json`; exact epochwise case/query hashes, finite named moments and expected visits/updates |
| Fixed-weight TRAIN replay | `evaluation/{thermal,wind}_a2{d,h6}_e{100,500,1000}_fixedtrain/result.json`; fixed saved weights, complete selected TRAIN membership and declared monitor queries |
| Full stop guards | `receipts/full_stop_cli_guard_checks.json`; actual stop5000 rejection before allocation and stop1000 rejection without measured qualification for both full recipes |
| Single typed-centroid revision | `receipts/typed_centroid_r1_allocation_and_budget.json`, `r1_prelaunch_correctness_review.json`, `r1_e500_main_review.json`, `r1_e1000_main_review.json`, `r1_e1000_to2500_forecast.json`; fresh organized runs, unchanged direct controls after default-math verification |
| Default preservation | `receipts/default_backward_isolated_v1.json`: all four native outputs and813 parameter gradients match original26d6363 code bitwise in controlled single-threaded autograd; earlier four-thread hash mismatches remain in raw captures and are not called a bitwise pass |
| Native R1 macro-objective reachability | `evaluation/thermal_r1_centroid_native_gradient/result.json` and `wind_r1_centroid_native_gradient/result.json`: all237/220 intended parameters receive finite nonzero gradients, native Thermal first48/final6 and Wind24 packed12+12, zero optimizer updates |
| R1 actual full-stop negative tests | `receipts/r1_full_stop_cli_guard_checks.json`; both revised full recipes reject5000 before allocation and reject an unqualified1000 start |

The four primary runs are `runs/thermal_a2d_dev`, `thermal_a2h6_dev`, `wind_a2d_dev` and `wind_a2h6_dev`. Each keeps `configs/experiment_identity.json`, named Adam state, `metrics/history.json`, exact case/query hashes, monitor milestones every100 epochs, latest and best-field aliases. Final literal-age and selected-age results are recorded separately; the selected alias comes only from the saved field monitors.

The fresh revised organized runs are `runs/thermal_a2h6_r1_centroids_dev` and `runs/wind_a2h6_r1_centroids_dev`. They explicitly add `model.query_edge_geometry_mode="typed_w_centroids_v1"`, compute live module/environment coordinate first moments from the final executed W without another measure multiplier, and retain one128-wide message with four heads. They preserve the original anchor, typed validity masks, six contents, Thermal rank4 and affine law, local support, losses, schedules and fixed streams. No parent epoch or optimizer state is added to a revised age. Primary checkpoints/histories are frozen by `receipts/protected_primary_before_r1.json`.

The input-only TRAIN geometry/content diagnostics are retained at initialization,100,1000 and2500 for each organized fit. Correct-runtime initial content files are `evaluation/{thermal,wind}_initial_content_geometry_details_modulardt.json`; Thermal e100 is `thermal_h6_e100_geometry_content_v1.json` with its absolute-ModularDT command in `receipts/thermal_h6_e100_geometry_command_v1.json`, while Wind e100 and both mature primary ages use `{thermal,wind}_h6_e{age}_geometry_content_modulardt.json`. Revised files use `{thermal,wind}_r1_{init,e100,e1000,e2500}_geometry_content_modulardt.json`. Their smooth-coordinate gradients and far-K spectra are model diagnostics over128 receiver rows; they are not independent physical geometry labels or physical excitation-rank measurements.

The bounded four-case learnability check occurred after the first100 review, a documented ordering deviation. Wind's actual fixed objective decreased in both arms. Thermal H6's fresh ten-update fixed objective decreased from4.82306 to1.26235; its interrupted predecessor contributed one counted update. Thermal D completed ten updates, but its final fixed-checkpoint four-case replay was not saved, so its rotating pre-step losses are not presented as a fixed before/after measurement. Scientific1000 checkpoints and fixed-TRAIN replay provide separate mature learning evidence.

## Frozen references and native common-panel boards

The first-round root is `/data/wanglz/ModularDT/diagnostics/content_hypergraph_oct13_20261014T015930Z`. `receipts/protected_prior_round.json` protects its993 files; the source/cache/checkpoint inventory also preserves earlier expected absences. Native reference loading retains each checkpoint's own input factory, transforms, environment catalogue and physical output law. Invalid older Wind mean-law runs and wrong-input Thermal classic arrays remain quarantined.

| Reference | Literal2500 checkpoint SHA256 / native evidence |
|---|---|
| Thermal A-D | `30da0a857a1b23f649250f02aaa9d65eab1d314ba958fa058b584ff3b36a5bae`; old `evaluation/thermal_ad_e2500_native_dev22/result.json` |
| Thermal A-H6 | `0d4eb5eeb120c8be630dc439e7fc39e2c625e8fad3a28b53a644836ae032e117`; old `evaluation/thermal_ah6_e2500_native_dev22/result.json` |
| Wind A-D | `79fea32bcf618928b1a54ae519f6f9919fb0aa0a200a93051d06b447886e53cd`; old `evaluation/wind_e2500_native_board_v1/result.json`, label `wind_ad_e2500` |
| Wind A-H6 | `6c4ff6ecff90150c04f6382cbf1c23609fc10559e071dbd1ebd874492b3984cc`; same old board, label `wind_ah6_e2500` |
| Thermal classic native references | Old `evaluation/thermal_reference_tool_public_qualification_v1/result.json`: public CPU qualification on0291/0687 for actual1404/1804 e1000/e5000; `thermal_classic_cpu_board_v2/result.json`: complete DEV22 mature5000 board |
| Wind classics on exact Round2 DEV24 | New `results/wind_classic_dev24_endpoint_board_v2/result.json`, `receipts/wind_classic_dev24_execution_receipt.json`; actual2102 e1000/e2440 and2103 e1000/e2475, endpoint metrics and prediction hashes |

The previous Wind full VALID90 receiver draws do not match the DEV24 draws, even for common row IDs. `receipts/wind_classic_dev24_reuse_audit.json` documents the mismatch. The bounded CPU classic replay therefore uses the exact sealed DEV24 panel, each classic's E512 native inputs and normalizer, and its public physical predictor. All196608 receiver support features were checked against native `support_features`; a public prediction parity replay agreed bitwise on8192 points. Raw classic endpoint vectors were not retained; per-row physical metrics, source/prediction hashes and provenance were retained. Classic full-TRAIN420 exposure, batch16/Q8192, historical losses and schedules differ from the new TRAIN72 batch24/Q4096 study.

## Completed literal2500 and selected-age evidence

All six fresh development fits completed their 100/500/1000/2500 segmented stops. Primary H6 and the single fresh centroid revision each match all 2500 D case/query hashes. Each Thermal fit has375000 case visits,10000 updates,17500 microbatches and15000 tail-case visits; each Wind fit has180000 visits,7500 updates,15000 microbatches and no tail. `receipts/r1_e2500_main_review.json` and dataset checkpoint-delta receipts verify finite model tensors/named Adam moments, exact steps, source/locks and fixed-TRAIN improvement. All 237/220 R1 trainable tensors changed since 1000; no parameter is passed merely because it has an optimizer entry.

| Literal2500 model | Literal checkpoint SHA256 | Saved field-selected age |
| --- | --- | --- |
| Thermal A2-D | `2c407b9fcffa31a9802f23586116e7271910ddda384cf062d240d88caa09b338` | 1700 |
| Thermal primary H6 | `40c561365eb7204c8aa926a7a3726ca5d2290eb5a1a38935387a8ac5145b9867` | 1500 |
| Thermal H6 R1 | `e2dc40c05cebce03ed7f2d5437f8514dbf016dd788592575f4e47ce6dad42f31` | 1200 |
| Wind A2-D | `bd275b634a1d4794cca19650401d9c98f7e7a606d71ac4fc948d1e56e5becc2b` | 1600 |
| Wind primary H6 | `3a22149dbd482fc2fdf3456e4ee8974249d665228c8a790e47f481e4206e6fa8` | 700 |
| Wind H6 R1 | `48bd9a0ab2e3d135d8943d7136adcee31b0b2f628444402d606bdbee8a379a02` | 800 |

The primary native results are `evaluation/thermal_a2d_e2500_native_dev22/result.json`, `thermal_a2h6_e2500_native_dev22/result.json`, and `wind_e2500_native_board_v2/result.json`. R1 results are `evaluation/thermal_a2h6_r1_centroids_e2500_native_dev22/result.json` and `wind_r1_e2500_native_board/result.json`. The unchanged D arrays are reused; R1 Wind is a new-only board, not an invented new D replay. `results/complete_native_tables/{metrics.json,tables.md}` retains all 8 Thermal roles, all 5 Wind vector roles,6 priority components, native broad scores, priority p90, module-count/layout groups and all exact IDs. `results/thermal_primary_native_distributions.json`, `wind_primary_native_distributions.json`, and complete native tables retain descriptive medians/worst errors. Statistical units are22 Thermal cases or24 Wind rows, with8 Wind layouts; p90 is across case/row RMSEs.

Native field-selected replays are `evaluation/thermal_a2d_selected_e1700_native_dev22/result.json`, `thermal_a2h6_selected_e1500_native_dev22/result.json`, `thermal_r1_selected_e1200_native_dev22/result.json`, `wind_a2d_selected_e1600_native_dev24_retry1/result.json`, `wind_h6_selected_e0700_native_dev24/result.json`, and `wind_r1_selected_e0800_native_dev24/result.json`. Their native broad scores are0.01189792/0.03912227/0.04387094 and0.04287159/0.16052818/0.16704920 respectively. Literal qualification never uses those selected-age substitutions. The selected R1 alias hashes are Thermal `2bd4726de8e8bb06a31a4b50516cc55d864195ed3a551a84e09465ab6f5b1b08` and Wind `7bbfc8abb5f1c2c390240d2e068f83536cf1bd97605a37a2de1f6c695d9b13b1`.

## Final improvement decision, content and costs

`receipts/final_section10_decisions.json` binds the literal R1 checkpoints, architecture/native-law hashes, native field decision sources, inspection effects, current-v1 costs and fixed-B8 points. Both datasets are **not qualified** for full-TRAIN 1000. Thermal R1 fluid/broad ratios to matched D are1.95092/3.64317, above 1.15; Wind R1 GM6 ratio to old A-D is0.99510 rather than at most 0.90, broad1.13340, matched near/downstream2.50675/2.84394, and downstream row-RMSE p90 old-baseline ratio1.26477. Cost/correctness/content passes do not override field failure. No full run directory, full-data normalization fit, optimizer allocation or 5000 job is authorized by this decision. Primary candidates also failed; no near-miss approval is issued.

Actual same-weight interventions/traces are `evaluation/{thermal,wind}_r1_inspection_e2500/result.json` and their retained NPZ files. Full, uniform receiver access, input-declared edge0 removal without renormalization, and local-zero controls all preserve full-six/local executed rows. Read-only syntheses are `receipts/thermal_r1_e2500_saved_dev22_org_intervention_summary.json` and `wind_mature_organizer_cost_summary_20261014.json`. All 22/24 native organization summaries use actual W/content/head traces; the 128-query receiver summary is input-selected. W is a donor-normalized gather, not old source-across-edge B or causal influence. Signed Thermal C and explicit K have distinct units/factorization interpretation.

R1 complete costs plus target-free native B8 forward/backward points are `evaluation/{thermal,wind}_r1_e2500_cost_and_density/result.json`; current same-device/same-output retained-v1 comparisons are `{thermal,wind}_r1_e2500_current_baseline_cost/result.json`. Physical GPU UUIDs and exact checkpoint/input/output hashes are recorded. High-M R1/v1 median ratios are1.52952 Thermal and1.78032 Wind, below2. Captured profiling has9 efficient-attention operators per candidate high-M call; the actual CUDA dispatch is CUTLASS memory-efficient attention. No Flash claim, sparse savings or timing-scope summation is made. Setup/fixture, fresh GPU, prepared decode, host-to-native-host, repeated controls, training loop and whole process costs are separate.

For primary Wind, keep native timings from `wind_e2500_cost_and_density_v1/result.json`, but use only `wind_e2500_density_workspaces_v2/result.json` for density. The initial density resident baseline grew on first backward and is explicitly superseded for that scope. The final v2 density warmup/allocator policy produces a common resident baseline for both arms at all requested Q. R1 density uses the same corrected policy. Absolute repeated invocation timings vary; compare the paired current-v1 ratio in its own invocation and retain all ranges. Thermal auxiliary stencils/deduplicated union and Wind role-prefix packing limit pure-Q scaling interpretation.

## Saved response, figures and validation

`evaluation/thermal_a2d_selected_e1700_saved_response/{result.json,result.npz}` and `thermal_r1_selected_e1200_saved_response/{result.json,result.npz}` use only the saved0291/0294/0687 families:3 independent layout/direction families,6 signed changes, no invented0277 baseline. The request SHA is `4128b0a56efb7fb065fd123097ad752edab811908232ef8685e66f2b7cc78f08`;22 saved-record hashes are retained. Both compare387616 elements across18 role/direction groups: FP64 has0 failures, maximum discrepancy2.089e-13 D/3.269e-13 R1; legacy FP32 has8954/10904 failures. Figure6's arithmetic panel is the 245760 fluid-channel subset, with642/1128 FP32 failures. These arithmetic counts are not physical-response accuracy. The reference is explicitly analytic-wake/shared-grid, not CFD; no inverse optimizer or new solver is used.

The [main report](HONF_Attention_Round2_Report.md) embeds15 selected PNGs and provides a short six-PDF index. Selected local folders are `figures/learning_final_v2`, `thermal_native_final`, `wind_native_final`, `content_final`, `cost_final`, and `responses_final_v3` (the last has a nested `figures/`). PDF masters have2/3/3/3/3/1 pages respectively. All figures are rendered from retained arrays/metrics, inspected visually and source/hash bound; superseded visual payloads are removed only after replacement/report dependency validation. Main-thread review corrects the response metadata lookup from adapter-level to nested core geometry, shortens overlapping arithmetic annotations and gives learning footers sufficient page space; earlier failure logs/metrics remain retained.

The main measurement synthesis is `results/final_measurement_summary.json`. CPU native/inspection execution receipts are `receipts/r1_{thermal,wind}_{native,inspection}_execution.json`; selected Thermal execution receipts identify age1200 frozen by field monitors before the response audit. Resource/cleanup/protected-closeout receipts are kept locally, with actual capacities in the final report. Source paragraphs/captions remain one continuous line. The unrelated pre-existing report deletion is preserved and is never staged.

## Tested command forms

These are the command forms actually exercised in this round. Evidence outputs are immutable; a later replay requires a fresh output path. Historical resume commands also require the original imported-source binding. The round-owned `session_run.py` wraps every GPU-associated process with the declared time/storage/device limits; it is a local supervisor, not a second training engine.

```bash
HONF_MODULARDT_PY=/home/wanglz/miniconda3/envs/ModularDT/bin/python
HONF_ROUND2_ROOT=/data/wanglz/ModularDT/diagnostics/attention_round2_20261014T135200Z
"$HONF_MODULARDT_PY" -B "$HONF_ROUND2_ROOT/session_run.py" --label wind_a2d_dev_e2500 --gpu 1 --stage training --maximum-seconds 4200 --run-dir "$HONF_ROUND2_ROOT/runs/wind_a2d_dev" -- "$HONF_MODULARDT_PY" -B HONF_Proj/tools/content_hypergraph_train.py --recipe HONF_Proj/src/config_core/forward/content_hypergraph/wind_a2-d_development.json --device cuda:0 --output "$HONF_ROUND2_ROOT/runs/wind_a2d_dev" --stop-after 2500 --resume "$HONF_ROUND2_ROOT/runs/wind_a2d_dev/checkpoints/latest_model.pt" --yes
```

The100/500/1000 segments use the same maintained trainer and corresponding stop, fresh versus resume flags, dataset recipe and run identity. Exact executed argv is retained in the process ledger; strict case/query/source binding is also stored in each run. CPU evidence commands set `CUDA_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4` and `OPENBLAS_NUM_THREADS=4`.

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$HONF_MODULARDT_PY" -B HONF_Proj/tools/content_hypergraph_train.py --recipe HONF_Proj/src/config_core/forward/content_hypergraph/thermal_a2-h6_full1000.json --device cpu --stop-after 5000 --dry-run
```

The last command is an intentionally rejected negative test, not a launch instruction: `validate_stop` rejects5000 before construction/allocation. A1000 full start additionally requires the independently measured v2 qualification. The retained full recipes are inert prospective5000 schedules with a hard1000 execution boundary. Future continuation needs a later explicit user decision, source-bound continuation support and tests; no runnable5000 command or automatic job is supplied.


The revised endpoint commands below were executed with the same source-bound checkpoint and supervisor. CPU outputs and logs are retained; choose a new output path for a future replay.

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$HONF_MODULARDT_PY" -B HONF_Proj/tools/content_hypergraph_evaluate.py --checkpoint "$HONF_ROUND2_ROOT/runs/thermal_a2h6_r1_centroids_dev/checkpoints/epoch_2500_model.pt" --output "$HONF_ROUND2_ROOT/evaluation/thermal_a2h6_r1_centroids_e2500_native_dev22" --device cpu --split validation --panel native --export-case 0291 --export-case 0687
"$HONF_MODULARDT_PY" -B "$HONF_ROUND2_ROOT/session_run.py" --label thermal_r1_e2500_cost_density --gpu 0 --stage cost --maximum-seconds 600 -- "$HONF_MODULARDT_PY" -B HONF_Proj/tools/content_hypergraph_benchmark.py --dataset thermal --checkpoint-ad "$HONF_ROUND2_ROOT/runs/thermal_a2d_dev/checkpoints/epoch_2500_model.pt" --checkpoint-ah6 "$HONF_ROUND2_ROOT/runs/thermal_a2h6_r1_centroids_dev/checkpoints/epoch_2500_model.pt" --typed-centroid-revision --device cuda:0 --output "$HONF_ROUND2_ROOT/evaluation/thermal_r1_e2500_cost_and_density"
```

Wind uses its retained R1 checkpoint, `--dataset wind`, and physical GPU1 through the supervisor. Exact actual argv for all24 scientific fit segments and all evaluation/cost attempts is in `gpu_process_ledger.json`; no GPU2 job is included. The per-dataset fresh R1 start used `*_a2-h6_r1_centroids_development.json` with stop100, then strict resumes to500/1000/2500. All declared milestones, latest and best aliases remain retained; there are no extra25-epoch or duplicate selector checkpoints.


## Verified storage and protection closeout

`receipts/final_storage_closeout.json` records 538 quantified removed files / 264308485 logical bytes, the retained approximately 4.57 GiB local round, initial/final filesystem capacities, budget use and the one earlier unquantified failed-macro unlink. `owned_closeout_cleanup_execution.json` contains the exact 343-entry final allowlist, inode/link/hash/ownership checks, readable process FD/mapping and lsof review, report-link exclusion, and all 396 scientific run files plus 27 selected figure files unchanged. The final selected figure freeze is `final_selected_figure_acceptance.json`; all 21 PDF/PNG exports and their 6 metric/identity JSONs remain protected. Superseded exports are absent; compact failure evidence and code/scripts remain local.

`receipts/final_protection_audit_v2.json` verifies 993 prior-round files, 264 primary files, four sealed inputs and 1919 inherited observations (1827 files, 18 resolved symlinks, 74 preserved absences). It preserves the failed scanner/schema guard attempts and records their corrected target-stat/cache/solver interpretations. Native source bindings and solver 326/326 metadata match; Wind cache 6337-file metadata fingerprint matches with zero writes. Its limitation is no historical regular-file content SHA baseline; the Thermal H5 original record also supplies stat binding rather than a baseline SHA. PID 248340 remains alive with its original command and was only inspected. No protected source data/checkpoints, solver arrays or unrelated jobs are modified.

Final Markdown/source-link/browser QA and push/tip receipts remain local under `receipts/`. The existing user deletion stays unstaged. Generated evidence is intentionally not uploaded; the durable branch contains code, tests, recipes, guide and the two reports only.
