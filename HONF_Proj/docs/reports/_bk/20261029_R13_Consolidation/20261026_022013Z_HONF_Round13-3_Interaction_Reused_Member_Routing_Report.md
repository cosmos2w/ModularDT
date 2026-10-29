# HONF Round13-3: applied interaction reuse and original-member routing

Historical endpoint report. The implemented correction, native diagnosis, new TRAIN600 epoch-100 lineage, measured approximation costs and tested continuation are recorded in the [corrective relative-member report](20261026_044023Z_HONF_Round13-3_Corrective_Relative_Member_Routing_Report.md). The measurements below remain the original failed Route and Full evidence.

Completed first segments, 26 October 2026 UTC. ThermalChannel only. Both scratch TRAIN600 lineages stopped at epoch 100; no later training is authorized or running for this round.

The new versioned model applies MM, EM, EE and ME contextual exchanges to original M/E records, reuses the final applied M←M and M←E gates as an M-address directory, and performs query-conditioned original-member selection with independent M/E null routes before one joint physical head. Full reads all eligible original members; Route executes values only for exact-positive selected weights. The implementation and recovery paths work, but Route loses all final member reads by epoch 3. Its lower training runtime at epoch 100 accompanies a serious loss of input sensitivity, rather than useful learned sparse retrieval.

## Predictor, organizer and inverse outcomes

**Predictor gains and evidence.** The maintained native objective, all nine observables, complete TRAIN600 trajectories, exposed VAL89 evaluation, fixed field/detail panels, checkpoint aliases and executable recovery are available for both arms. The two arms share the complete training membership, TRAIN-only normalizer, case order, query draws, common initialization and 100-epoch horizon. Full has lower equal-case mean RMSE on all nine VAL89 observables; material temperature is 2.354 versus Route 6.204, q_proxy 6.024 versus 14.434. The tables below report actual fieldwise performance rather than treating one aggregate selector as fidelity.

**Predictor misses and next step.** Route outputs a member-free query-dependent field at this endpoint. Its saved heat responses are effectively zero and its homogeneous-zero-heat temperature residual is large. Full is a predictive-capacity reference and also misses physical diagnostics: its M10 zero-heat material-temperature RMSE is 7.096 and mixed-geometry q-normal response RMSE is 14.28. Neither first-100 endpoint establishes convergence, mature architectural success or definitive failure. Any further fitting or changed null-support prescription needs the user’s explicit choice; no follow-on arm or automatic continuation was launched.

**Organizer gains and evidence.** The decoder uses live applied B coefficients, original typed IDs, deduplicated source reads and independent null normalizations; native initialization gradients and focused path/permutation/tiling tests pass. All four contextual exchanges really run. The final directory reuses only last-stage MM/ME; it does not represent every upstream dependency.

**Organizer miss and next step.** No final M/E values execute on VAL89 or the four-case low-M TRAIN panel. Positive candidates and broad B rows remain, but their learned member weights are zero. B-only and comparable-count geometry-B interventions therefore leave fields unchanged; this endpoint supplies no evidence that learned B improves physical prediction beyond geometry. Geometry member scores can reopen VAL reads at fixed B/proposals, exposing a dormant path. A later null-collapse investigation should preserve this negative lineage and change one measured cause at a time, without forcing a winner, K, support fraction or physical interpretation.

**Inverse result.** No inverse optimizer, design fitting or design-validity experiment was executed. There are no inverse gains to claim. The failed routed response diagnostic does not support promoting this endpoint to inverse design; any inverse work is a separate user decision.

## Identity, architecture and bounded execution

Source capability: `interaction_reused_member_routing_v1`, input capability `joint_prescribed_records_v1`; adapter: ThermalChannel. Both arms use width 128, four heads, two ordered MM→EM→EE→ME contextual cycles, E192, FP32, AdamW 3e-4 with constant schedule, weight decay 1e-5, epsilon 1e-8, gradient clipping 1, effective batch 48, microbatch 24 and 1,024 primary training queries. The native loss retains the existing flow, fluid/surface/material temperature and q-proxy weights; it adds no response-fitting, organizer or analytical correction objective. The field head jointly predicts u/v/p/T; native vorticity and interface observables use the maintained extraction.

At each present M address, B_E is the actual applied final `me` gate and B_M is the actual final nonself `mm` gate plus identity self-access. Present M has final-read count measure one even when its contextual donor measure is zero. Positive E physical measures are divided by their scene positive mean and applied once in the final typed read. Original heat controls enter their owning module before interaction. Cached original-record projections and contextual residual reinjection are preparation work, not independent query-value branches. There are no pre-query pooled group values, group-function bottleneck, nearest-32 cutoff, learned optimal K, independent raw-value bypass or new training engine.

Proposal entmax1.5 has a fixed zero-logit null; M/E member entmax distributions have separate nulls. The union of positive proposal/B paths defines candidate IDs, and only beta>0 causes deduplicated query-member value execution. Address coefficients use deterministic batched matrix multiplication; this corrected a real CUDA checkpoint-replay support mismatch caused by sparse atomic accumulation. Its dense scalar coefficient products are charged explicitly. Empty/all-null M addresses give zero typed contexts, with no unorganized E fallback. Query coordinates still enter the permitted joint trunk, so zero member context does not force zero physical output.

The scratch comparison verified 186 exactly equal common state tensors, identical TRAIN600 normalization, case order and sampled native query/control inputs. Heat RMS is 1.319898732. Both objects contain 1,402,049 parameters; Full has 1,368,384 trainable parameters and Route 1,402,049. Full does not exercise/follow gradients through the proposal path. This is not an equal-trainable-parameter or equal-work causal comparison. [Initialization evidence](../../../../diagnostics/round13_3_20261025/initialization_comparison.json).

Four segmented fitting epochs were used in total: Full epoch 1 then actual resume to epoch 2, and Route epoch 1 then actual resume to epoch 2, on the existing fixed input-stratified TRAIN150/DEV22 manifest. These qualified execution, learning paths, save and recovery only; their weights were not promoted into TRAIN600. Native TRAIN M1 case 0006 and M12 case 0231 checks measured nonzero applicable proposal, member-score/value, joint-head and applied-gate gradients. The finite gate bias is 1.5; proposal output bias −1.5 was selected from TRAIN-only gradient health, not validation errors. No forced exposure or dormant-support fitting was used. [Gradient evidence](../../../../diagnostics/round13_3_20261025/native_gradient_probe.json).

| Arm | Recorded epochs / endpoint | Case visits | TRAIN seconds | Recorded review seconds | Engine active interval s | CLI logged span s | Review epochs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full | 100 / 100 | 60,000 | 2432.1 | 1.3203 | 2454.4 | 2462.9 | [100] |
| Route | 100 / 100 | 60,000 | 1328.8 | 1.214 | 1350.5 | 1359.1 | [100] |

The engine active interval includes maintained execution/save/plot overhead but excludes earlier imports/provider preparation. CLI logged spans (Full 2462.9 s, Route 1359.1 s) use unchanged log-creation-to-final-write timestamps and approximate launch-to-output elapsed time; the two main fits total 1.062 logged process-associated GPU hours. These clocks differ from sums of TRAIN and recorded review times and are not GPU-utilization measurements. Initial and routine review artifacts remain in each run. Main fitting used physical GPU 0 for Full and GPU 1 for Route; both saw CUDA_VISIBLE_DEVICES=0,1. Luna performed startup ETA and a near-completion check without duplicate parent polling. No third fit, TEST evaluation, solver call, dataset copy, persistent comparison cache or inverse fit was added. Both histories contain exactly epochs 1–100 and 60,000 case visits. The 12 process-associated GPU-hour envelope was not exhausted; phase costs and approximate evaluation/qualification log windows are retained in `cost_accounting.json`.

## Native field evidence

Every number below is equal-case native RMSE in that observable’s packed ThermalChannel units; no unverified SI conversion is implied. q_proxy is the maintained interface target and is not a new independently solved heat-flux reference. Worst is the maximum case RMSE, not a maximum point error. Canonical exposed VAL89 contains M3/M5/M7/M10 counts 24/25/25/15. DEV22 is an exposed subset; neither is independent TEST or a guarantee of physical-family independence. Low-M TRAIN is shown separately: cases 0003, 0004 (M1), 0048, 0049 (M2); the retained result JSON is authoritative for literal IDs. Detail cases are selected only by input module count and stable source ID, never field error.

### Exposed VAL89

| Observable | N | Full mean | Full p90 | Full worst (ID) | Route mean | Route p90 | Route worst (ID) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| u | 89 | 0.15959 | 0.18205 | 0.22572 (0297) | 0.26199 | 0.32023 | 0.34954 (0281) |
| v | 89 | 0.026332 | 0.029776 | 0.033698 (0285) | 0.027758 | 0.032989 | 0.034399 (0678) |
| p | 89 | 0.09702 | 0.11177 | 0.12209 (0302) | 0.10863 | 0.12199 | 0.13441 (0292) |
| omega | 89 | 0.33173 | 0.38343 | 0.44523 (0285) | 0.42359 | 0.48329 | 0.51143 (0663) |
| temperature_fluid | 89 | 2.8635 | 3.5015 | 5.5122 (0298) | 5.3734 | 6.6504 | 7.9381 (0295) |
| temperature_surface | 89 | 2.6929 | 3.7046 | 6.6016 (0298) | 5.6665 | 7.1203 | 9.7756 (0679) |
| temperature_material | 89 | 2.3538 | 3.3657 | 6.1028 (0298) | 6.2036 | 8.9563 | 11.91 (0679) |
| q_proxy | 89 | 6.024 | 8.4637 | 12.482 (0298) | 14.434 | 17.794 | 19.591 (0672) |
| module_peak_endpoint_rmse | 89 | 2.1942 | 3.2823 | 6.7365 (0298) | 7.7916 | 11.523 | 14.882 (0679) |

Populated input-count strata; each cell is mean / p90 / worst RMSE. Empty categories are not measured.

| Arm / observable | M3 (N=24) | M5 (N=25) | M7 (N=25) | M10 (N=15) |
| --- | --- | --- | --- | --- |
| Full / u | 0.15468 / 0.17793 / 0.18256 | 0.15685 / 0.18177 / 0.20877 | 0.16989 / 0.19445 / 0.22572 | 0.15485 / 0.17008 / 0.17775 |
| Full / v | 0.024005 / 0.027664 / 0.029005 | 0.02602 / 0.029251 / 0.033698 | 0.027779 / 0.031112 / 0.032657 | 0.028164 / 0.030305 / 0.030422 |
| Full / p | 0.088048 / 0.11373 / 0.11918 | 0.095017 / 0.10299 / 0.11175 | 0.10241 / 0.11033 / 0.12209 | 0.10573 / 0.11408 / 0.11554 |
| Full / omega | 0.30002 / 0.33941 / 0.38111 | 0.3184 / 0.35682 / 0.44523 | 0.35689 / 0.39031 / 0.42725 | 0.36273 / 0.38068 / 0.40578 |
| Full / temperature_fluid | 2.5154 / 3.1312 / 3.5306 | 2.8147 / 3.3665 / 3.7819 | 3.1558 / 3.8837 / 5.5122 | 3.0144 / 3.4793 / 3.5641 |
| Full / temperature_surface | 1.8678 / 2.4196 / 2.8671 | 2.6679 / 3.5148 / 5.0555 | 3.3554 / 4.6686 / 6.6016 | 2.9504 / 3.2371 / 3.4988 |
| Full / temperature_material | 1.6104 / 2.2941 / 2.4122 | 2.2846 / 3.2608 / 4.4627 | 3.0027 / 4.5887 / 6.1028 | 2.5769 / 2.8776 / 3.316 |
| Full / q_proxy | 4.2276 / 5.2063 / 5.5247 | 5.7509 / 8.0569 / 9.1329 | 7.3376 / 9.9826 / 12.482 | 7.1638 / 8.4208 / 9.8789 |
| Full / module_peak_endpoint_rmse | 1.4826 / 2.5467 / 2.8597 | 2.0398 / 3.0234 / 3.4249 | 2.9261 / 4.162 / 6.7365 | 2.3701 / 3.1535 / 3.5573 |
| Route / u | 0.31547 / 0.34181 / 0.34954 | 0.27021 / 0.2902 / 0.31593 | 0.24435 / 0.27291 / 0.28665 | 0.1921 / 0.20841 / 0.23353 |
| Route / v | 0.02302 / 0.024454 / 0.026733 | 0.027154 / 0.029831 / 0.030116 | 0.029646 / 0.031972 / 0.034289 | 0.033196 / 0.03411 / 0.034399 |
| Route / p | 0.10364 / 0.11159 / 0.11601 | 0.10627 / 0.11955 / 0.13441 | 0.11076 / 0.12228 / 0.12689 | 0.11699 / 0.12575 / 0.12775 |
| Route / omega | 0.36722 / 0.40308 / 0.43273 | 0.42039 / 0.46448 / 0.47727 | 0.44687 / 0.49758 / 0.51143 | 0.48034 / 0.49989 / 0.50851 |
| Route / temperature_fluid | 6.1847 / 6.8353 / 7.2802 | 5.1326 / 6.1883 / 6.6686 | 4.747 / 5.7279 / 7.9381 | 5.5204 / 6.7921 / 7.0322 |
| Route / temperature_surface | 5.1792 / 6.3436 / 7.2086 | 5.1315 / 6.2977 / 6.7884 | 5.7836 / 7.061 / 9.6743 | 7.143 / 9.3858 / 9.7756 |
| Route / temperature_material | 4.7818 / 5.6351 / 6.8517 | 5.381 / 7.6419 / 8.1481 | 6.8319 / 8.8957 / 11.656 | 8.8023 / 11.384 / 11.91 |
| Route / q_proxy | 13.105 / 17.454 / 19.064 | 14.786 / 18.134 / 18.206 | 15.214 / 18.101 / 19.591 | 14.673 / 15.919 / 17.111 |
| Route / module_peak_endpoint_rmse | 5.5668 / 7.8533 / 9.2523 | 6.9029 / 10.491 / 11.246 | 8.8472 / 11.648 / 14.725 | 11.073 / 13.821 / 14.882 |

### Four low-M TRAIN cases

| Observable | N | Full mean | Full p90 | Full worst (ID) | Route mean | Route p90 | Route worst (ID) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| u | 4 | 0.1461 | 0.159 | 0.16049 (0049) | 0.34704 | 0.37193 | 0.38251 (0004) |
| v | 4 | 0.022374 | 0.023488 | 0.024082 (0049) | 0.019539 | 0.021418 | 0.021538 (0048) |
| p | 4 | 0.080991 | 0.087775 | 0.088448 (0049) | 0.065913 | 0.070389 | 0.071666 (0048) |
| omega | 4 | 0.27696 | 0.30886 | 0.31496 (0049) | 0.31876 | 0.34819 | 0.35233 (0049) |
| temperature_fluid | 4 | 2.1962 | 2.9232 | 3.3193 (0049) | 7.4838 | 8.2319 | 8.3086 (0004) |
| temperature_surface | 4 | 1.4137 | 1.6232 | 1.684 (0004) | 6.9477 | 8.4603 | 9.0753 (0048) |
| temperature_material | 4 | 1.1257 | 1.4826 | 1.5946 (0049) | 5.164 | 6.0471 | 6.4314 (0048) |
| q_proxy | 4 | 3.1294 | 4.0413 | 4.256 (0049) | 12.523 | 16.439 | 17.664 (0049) |
| module_peak_endpoint_rmse | 4 | 0.44469 | 0.76385 | 0.88654 (0049) | 3.4246 | 5.053 | 6.0048 (0049) |

Populated input-count strata; each cell is mean / p90 / worst RMSE. Empty categories are not measured.

| Arm / observable | M1 (N=2) | M2 (N=2) |
| --- | --- | --- |
| Full / u | 0.13418 / 0.14165 / 0.14352 | 0.15801 / 0.15999 / 0.16049 |
| Full / v | 0.021656 / 0.02183 / 0.021874 | 0.023091 / 0.023884 / 0.024082 |
| Full / p | 0.074655 / 0.080962 / 0.082539 | 0.087327 / 0.088224 / 0.088448 |
| Full / omega | 0.26735 / 0.28916 / 0.29462 | 0.28657 / 0.30928 / 0.31496 |
| Full / temperature_fluid | 1.7334 / 1.8085 / 1.8272 | 2.6591 / 3.1872 / 3.3193 |
| Full / temperature_surface | 1.3649 / 1.6202 / 1.684 | 1.4624 / 1.4776 / 1.4813 |
| Full / temperature_material | 0.99977 / 1.1768 / 1.2211 | 1.2517 / 1.526 / 1.5946 |
| Full / q_proxy | 2.3607 / 2.4992 / 2.5339 | 3.8982 / 4.1845 / 4.256 |
| Full / module_peak_endpoint_rmse | 0.20733 / 0.24546 / 0.255 | 0.68206 / 0.84564 / 0.88654 |
| Route / u | 0.36488 / 0.37899 / 0.38251 | 0.32921 / 0.33328 / 0.3343 |
| Route / v | 0.01774 / 0.018551 / 0.018753 | 0.021338 / 0.021498 / 0.021538 |
| Route / p | 0.062287 / 0.062386 / 0.06241 | 0.069538 / 0.07124 / 0.071666 |
| Route / omega | 0.29208 / 0.30997 / 0.31445 | 0.34543 / 0.35095 / 0.35233 |
| Route / temperature_fluid | 8.1808 / 8.283 / 8.3086 | 6.7867 / 7.4289 / 7.5894 |
| Route / temperature_surface | 6.5379 / 6.9277 / 7.0252 | 7.3575 / 8.7317 / 9.0753 |
| Route / temperature_material | 4.7838 / 5.0772 / 5.1505 | 5.5442 / 6.254 / 6.4314 |
| Route / q_proxy | 9.4246 / 10.25 / 10.457 | 15.622 / 17.255 / 17.664 |
| Route / module_peak_endpoint_rmse | 2.7909 / 2.8239 / 2.8322 | 4.0583 / 5.6155 / 6.0048 |

Numerical fields, residuals, physical heat and original module/environment coordinates/IDs are retained in four `case_*.npz` files per arm; only those fixed representatives were exported. The reference fluid-valid/physical-weight mask, interface validity and present material masks determine plotted/error supports. [Full metrics, source IDs and traces](../../../../diagnostics/round13_3_20261025/evaluation_full/result.json); [Route metrics, source IDs and traces](../../../../diagnostics/round13_3_20261025/evaluation_route/result.json).

## Routing and fixed-checkpoint interventions

Route TRAIN exact-positive M/E primary value rows are 11,909,412 / 5,556,762 at epoch 1, 3,241,394 / 0 at epoch 2, then 0 / 0 at every epoch 3–100. This is a measured early collapse; initial gradient health did not prevent it. At epoch 100, VAL89 scores 4,276,224 M and 139,984,896 E candidate pairs and executes zero M/E value rows. Low-M TRAIN also executes zero. These are primary native-query counters; scene projections, contextual work and native stencil extraction remain separately charged.

The 12 fixed traced locations comprise three physical positions in each TRAIN M1 / VAL M3 / M7 / M10 case. M1 proposes null at all three; the VAL cases retain positive M-address proposals and broad actual B support, but both member distributions are null at all nine positions. Traces retain candidate and selected original typed IDs/weights, omitted IDs, original coordinates, proposal/member nulls and actual stage provenance. A positive address/B path is an access permission, not an executed value read, and learned support is not physical causality.

| Source ID | No-fit intervention | Native fluid ΔRMSE | Max abs Δ | Baseline M/E at 3 q | Changed M/E at 3 q | Counts matched |
| --- | --- | --- | --- | --- | --- | --- |
| 0003 | b_fixed_candidates_scores | 0 | 0 | 0/0 | 0/0 | True |
| 0003 | geometry_b_comparable_counts | 0 | 0 | 0/0 | 0/0 | True |
| 0003 | geometry_member_scores_fixed_b_pi | 0 | 0 | 0/0 | 0/0 | True |
| 0003 | all_candidates_open | 0 | 0 | 0/0 | 0/0 | True |
| 0003 | remove_contextual_message_recompute | 0 | 0 | 0/0 | 0/0 | True |
| 0274 | b_fixed_candidates_scores | 0 | 0 | 0/0 | 0/0 | True |
| 0274 | geometry_b_comparable_counts | 0 | 0 | 0/0 | 0/0 | True |
| 0274 | geometry_member_scores_fixed_b_pi | 3.0606 | 15.101 | 0/0 | 9/272 | False |
| 0274 | all_candidates_open | 0 | 0 | 0/0 | 0/0 | True |
| 0274 | remove_contextual_message_recompute | 0 | 0 | 0/0 | 0/0 | True |
| 0293 | b_fixed_candidates_scores | 0 | 0 | 0/0 | 0/0 | True |
| 0293 | geometry_b_comparable_counts | 0 | 0 | 0/0 | 0/0 | True |
| 0293 | geometry_member_scores_fixed_b_pi | 3.1 | 15.158 | 0/0 | 21/274 | False |
| 0293 | all_candidates_open | 0 | 0 | 0/0 | 0/0 | True |
| 0293 | remove_contextual_message_recompute | 0 | 0 | 0/0 | 0/0 | True |
| 0678 | b_fixed_candidates_scores | 0 | 0 | 0/0 | 0/0 | True |
| 0678 | geometry_b_comparable_counts | 0 | 0 | 0/0 | 0/0 | True |
| 0678 | geometry_member_scores_fixed_b_pi | 3.1864 | 15.477 | 0/0 | 30/275 | False |
| 0678 | all_candidates_open | 0 | 0 | 0/0 | 0/0 | True |
| 0678 | remove_contextual_message_recompute | 0 | 0 | 0/0 | 0/0 | True |

B-only scales eligible nonself M and E coefficients by 0.7 while preserving M identity; fixed-query comparisons hold proposal, candidate IDs and cheap scores fixed. Geometry-B changes the function on the same actual support and chooses a bounded geometry floor solely by three-query selected counts, retaining its trial list; no field/reference error is used. With zero baseline reads, a count match is degenerate and supplies no attribution. Geometry-member scoring keeps B/proposals fixed and reopens VAL reads, but changes the model function. The dense candidate-union executor retains the same beta weights, including zeros: fields agree exactly while extra zero-weight value rows execute. Removing one first-cycle applied ME message recomputes all dependent stages; at this endpoint the final all-null route masks its output effect. Full-native selected counts and the exact frozen-query trace scope are retained separately; a three-query count match is not a full-field equal-work match.

## Complete costs and selected versus executed work

Both final cost probes used the same physical GPU 1 (RTX 6000 Ada), one warmup and three measured repetitions for each phase. M1 source 0003 and M10 source 0348 are TRAIN inputs. Prepare includes actual contextual formation/indexing and original key/value projections; cached decode excludes prepare; complete native includes preparation, decode and native extraction. The representative 48-case training macro uses the maintained engine, two 24-case microbatches, real native losses, recovery snapshots, backward, clipping and AdamW, including fresh disposable optimizer construction and sampling. Its fixed distinct source IDs and per-repeat times are saved. Disposable changes to model/provider/RNG are restored and add zero scientific fitting epochs; there is no new training engine.

| Arm | Anchor | ID | Prepare mean / p90 s | Cached decode mean / p90 s | Complete native mean / p90 s | 48-case macro mean / p90 s | Peak allocated bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full | M1 | 0003 | 0.0338 / 0.03401 | 0.049338 / 0.049595 | 0.086154 / 0.086319 | 1.347 / 1.3481 | 1.2535e+10 |
| Full | M10 | 0348 | 0.036535 / 0.036704 | 0.050012 / 0.050158 | 0.090073 / 0.090293 | 1.381 / 1.3827 | 1.2524e+10 |
| Route | M1 | 0003 | 0.034338 / 0.034642 | 0.060008 / 0.060054 | 0.096665 / 0.096955 | 0.47945 / 0.48044 | 3.957e+09 |
| Route | M10 | 0348 | 0.036401 / 0.036442 | 0.06945 / 0.069548 | 0.10927 / 0.10938 | 0.50258 / 0.50285 | 3.973e+09 |

| Complete native work counter | Full M1 | Full M10 | Route M1 | Route M10 |
| --- | --- | --- | --- | --- |
| query_M_proposal_score_rows | 0 | 0 | 8192 | 81920 |
| query_M_cheap_score_rows | 8192 | 81920 | 0 | 81920 |
| query_E_cheap_score_rows | 1572864 | 1572864 | 0 | 1572864 |
| query_M_candidate_rows | 8192 | 81920 | 0 | 81920 |
| query_E_candidate_rows | 1572864 | 1572864 | 0 | 1572864 |
| query_M_selected_value_rows | 8192 | 81920 | 0 | 0 |
| query_E_selected_value_rows | 1572864 | 1572864 | 0 | 0 |
| query_M_value_execution_rows | 8192 | 81920 | 0 | 0 |
| query_E_value_execution_rows | 1572864 | 1572864 | 0 | 0 |
| query_address_coefficient_pair_rows | 1581056 | 16547840 | 1581056 | 16547840 |
| cached_scene_projection_rows_M | 1 | 10 | 1 | 10 |
| cached_scene_projection_rows_E | 192 | 192 | 192 | 192 |
| context_mm_score_rows | 2 | 200 | 2 | 200 |
| context_em_score_rows | 384 | 3840 | 384 | 3840 |
| context_ee_score_rows | 73728 | 73728 | 73728 | 73728 |
| context_me_score_rows | 384 | 3840 | 384 | 3840 |
| context_mm_positive_value_edges | 0 | 116 | 0 | 171 |
| context_em_positive_value_edges | 384 | 2262 | 384 | 3840 |
| context_ee_positive_value_edges | 55860 | 73646 | 73728 | 73728 |
| context_me_positive_value_edges | 384 | 3836 | 384 | 3840 |

Context EE scores 73,728 pairs per scene across two cycles, even when the decoder reads none. Its segment backend executes positive-gate value edges only; these measured counts differ from all scored pairs (Full M1: 55,860 values; Route M1: 73,728). Cached E projections cover all 192 original E records. Scalar address-coefficient products are charged despite zero selected values. Entmax support/sorting, cheap geometric scores, query features and preparation remain real runtime costs. Route is slower even with zero final reads: complete native M1 is 0.09667 s versus Full 0.08615 s, and M10 is 0.10927 s versus 0.09007 s. Route’s training macro is cheaper (M10 0.5026 s versus 1.3810 s) and its combined benchmark peak allocation is lower (3.973 versus 12.524 billion bytes), but these savings accompany null collapse and poorer fidelity. Complete native and training-macro timings measure executor behavior; this failed all-null endpoint does not establish fidelity-preserving acceleration. Full-access value execution is the separately labelled Full arm; the fixed-weight dense-union probe is not another trained model.

## Physical-response and zero-heat limits

Only four existing source-TRAIN atlas families were read: 0001, 0318, 0333, 0348. They are labelled `train_reference_only`, with zero response-fitting epochs, zero new solver calls and no response-validation atlas or TEST access. They provide finite heat/geometry and mixed-geometry controls on their saved receiver supports. They do not establish generalization, infinitesimal derivatives, physical causality or valid inverse designs. Relative errors exclude near-zero reference components as recorded; zero heat-only flow response is consistent with these saved case-specific references, not a generic architectural heat-independence law.

| Arm | Saved control | Observable | Mean RMSE of predicted vs saved Δresponse | Mean reference-normalized response error | Valid relative families |
| --- | --- | --- | --- | --- | --- |
| Full | heat_transfer_plus | fluid_fields/u | 0.0086631 | — | 0 |
| Full | heat_transfer_plus | fluid_fields/v | 0.0013325 | — | 0 |
| Full | heat_transfer_plus | fluid_fields/p | 0.004683 | — | 0 |
| Full | heat_transfer_plus | fluid_fields/omega | 0.020587 | — | 0 |
| Full | heat_transfer_plus | fluid_fields/temperature | 0.28795 | 1.1531 | 4 |
| Full | heat_transfer_plus | interface/T_surface | 0.3728 | 0.91893 | 4 |
| Full | heat_transfer_plus | interface/q_normal | 0.78351 | 0.85283 | 4 |
| Full | heat_transfer_plus | solid_temperature/temperature | 0.44086 | 0.84238 | 4 |
| Full | heat_transfer_plus | module_peak_endpoint/temperature | 0.5357 | 0.75348 | 4 |
| Full | i_plus | fluid_fields/temperature | 0.31986 | 4.2211 | 4 |
| Full | i_plus | interface/T_surface | 0.42008 | 3.5564 | 4 |
| Full | i_plus | interface/q_normal | 1.0222 | 1.4653 | 4 |
| Full | i_plus | solid_temperature/temperature | 0.39294 | 3.3635 | 4 |
| Full | i_plus | module_peak_endpoint/temperature | 0.37685 | 3.1552 | 4 |
| Full | mixed_central_derivative | fluid_fields/temperature | 6.1939 | 18.588 | 4 |
| Full | mixed_central_derivative | interface/T_surface | 7.1126 | 10.457 | 4 |
| Full | mixed_central_derivative | interface/q_normal | 14.28 | 11.568 | 4 |
| Full | mixed_central_derivative | solid_temperature/temperature | 6.8792 | 12.348 | 4 |
| Full | mixed_central_derivative | module_peak_endpoint/temperature | 6.7864 | 16.148 | 4 |
| Route | heat_transfer_plus | fluid_fields/u | 0 | — | 0 |
| Route | heat_transfer_plus | fluid_fields/v | 0 | — | 0 |
| Route | heat_transfer_plus | fluid_fields/p | 0 | — | 0 |
| Route | heat_transfer_plus | fluid_fields/omega | 0 | — | 0 |
| Route | heat_transfer_plus | fluid_fields/temperature | 0.28406 | 1 | 4 |
| Route | heat_transfer_plus | interface/T_surface | 0.42831 | 1 | 4 |
| Route | heat_transfer_plus | interface/q_normal | 0.91009 | 1 | 4 |
| Route | heat_transfer_plus | solid_temperature/temperature | 0.54018 | 1 | 4 |
| Route | heat_transfer_plus | module_peak_endpoint/temperature | 0.72265 | 1 | 4 |
| Route | i_plus | fluid_fields/temperature | 0.095487 | 1 | 4 |
| Route | i_plus | interface/T_surface | 0.29341 | 2.463 | 4 |
| Route | i_plus | interface/q_normal | 0.82327 | 1.1041 | 4 |
| Route | i_plus | solid_temperature/temperature | 0.26526 | 2.2039 | 4 |
| Route | i_plus | module_peak_endpoint/temperature | 0.26566 | 1.5458 | 4 |
| Route | mixed_central_derivative | fluid_fields/temperature | 0.60072 | 1 | 4 |
| Route | mixed_central_derivative | interface/T_surface | 1.6016 | 1 | 4 |
| Route | mixed_central_derivative | interface/q_normal | 3.5538 | 1 | 4 |
| Route | mixed_central_derivative | solid_temperature/temperature | 1.3001 | 1 | 4 |
| Route | mixed_central_derivative | module_peak_endpoint/temperature | 0.78966 | 1 | 4 |

The zero-heat probe re-prepares TRAIN M1 case 0003 and M10 case 0348 with original module heat set to zero, after verifying prescribed t_in=t_wall=0. The known homogeneous-zero-temperature solution is the temperature limit; no new independently solved zero-heat arrays were created. Fluid RMSE uses positive native physical-weight rows, surface/material use valid original record supports. Predicted heat-only flow change is recorded separately and cannot substitute for temperature fidelity.

| Arm | ID / M | Fluid T RMSE from zero | Surface T RMSE from zero | Material T RMSE from zero | q_proxy RMS from zero |
| --- | --- | --- | --- | --- | --- |
| Full | 0003 / 1 | 1.8295 | 3.1574 | 3.7159 | 7.855 |
| Full | 0348 / 10 | 5.12 | 6.871 | 7.096 | 6.2172 |
| Route | 0003 / 1 | 8.687 | 9.601 | 9.4768 | 2.5474 |
| Route | 0348 / 10 | 8.5104 | 10.311 | 10.376 | 3.5549 |

## Selected figures

The following figures are rendered from retained numerical arrays and histories, with PDF masters and one PNG companion per displayed page. Source IDs, partitions, literal epoch-100 checkpoint, native units/masks, controls and interpretation limits are on the pages/captions. All pages are embedded below; PDF links do not require viewer-side conversion. The short index is: learning/read collapse; M1 fields/detail; M10 fields/detail; three-query routing and applied B; nine-field fidelity; timing; executed work/context/interventions; physical responses/zero heat. The final figure manifest provides the literal inventory.

![Actual TRAIN600 weighted native losses and primary selected-value work through epoch 100](../../../../diagnostics/round13_3_20261025/figures/training_curves_work.png)

Figure 1. Actual TRAIN600 weighted native losses and primary selected-value work through epoch 100. Route M/E rows become zero after epochs 2/1 and remain zero; Full keeps original-member reads and continues reducing native losses. These are training histories, not a reconstructed validation trajectory or a convergence claim.

[PDF master, page 1](../../../../diagnostics/round13_3_20261025/figures/training_curves_work.pdf#page=1)

![TRAIN600 source 0003, M1, literal epoch-100 checkpoints: u/p/T references, predictions and signed residuals, surface temperature/q_proxy by original module/sample ID, and material maxima](../../../../diagnostics/round13_3_20261025/figures/native_fields_M1.png)

Figure 2. TRAIN600 source 0003, M1, literal epoch-100 checkpoints: u/p/T references, predictions and signed residuals, surface temperature/q_proxy by original module/sample ID, and material maxima. Fluid-temperature RMSE is Full 1.640 versus Route 8.053; material RMSE is 0.778 versus 5.151. The failed routed field is visible despite an allowed all-null computation; saved native units and physical masks apply.

[PDF master, page 1](../../../../diagnostics/round13_3_20261025/figures/native_fields.pdf#page=1)

![Exposed VAL89 source 0678, M10, epoch 100, with the same field/reference/residual conventions and original module IDs](../../../../diagnostics/round13_3_20261025/figures/native_fields_M10.png)

Figure 3. Exposed VAL89 source 0678, M10, epoch 100, with the same field/reference/residual conventions and original module IDs. Fluid-temperature RMSE is Full 2.654 versus Route 6.472; material RMSE is 2.669 versus 10.913. Module-peak endpoint RMSE is 2.329 versus 12.912. This is a fixed validation representative, distinct from TRAIN cost anchor 0348.

[PDF master, page 2](../../../../diagnostics/round13_3_20261025/figures/native_fields.pdf#page=2)

![Three fixed physical query locations each for TRAIN M1 0003 and VAL M10 0678](../../../../diagnostics/round13_3_20261025/figures/routing_support.png)

Figure 4. Three fixed physical query locations each for TRAIN M1 0003 and VAL M10 0678. Every final typed read is null, so no positive-beta value edge is drawn; all omitted M/E IDs are retained. Actual B_E remains broad (192/192 M1 and 1,920/1,920 M10 positive entries). Proposal pi, member nulls and final applied MM/ME gates are separate quantities. These are computational access relations, not physical causality or the complete upstream influence graph.

[PDF master, page 1](../../../../diagnostics/round13_3_20261025/figures/routing_support.pdf#page=1)

![All nine equal-case native RMSE observables, overall and over populated VAL89 strata M3/M5/M7/M10 (24/25/25/15 cases)](../../../../diagnostics/round13_3_20261025/figures/native_fidelity.png)

Figure 5. All nine equal-case native RMSE observables, overall and over populated VAL89 strata M3/M5/M7/M10 (24/25/25/15 cases). Solid/dashed/dotted lines show mean/p90/worst. Full has lower overall mean for all nine; material-temperature mean is 2.354 versus Route 6.204. Exposed validation and native unit keys are labelled; categories with no cases are omitted.

[PDF master, page 1](../../../../diagnostics/round13_3_20261025/figures/native_fidelity.pdf#page=1)

![Same physical GPU 1, TRAIN anchors 0003 (M1) and 0348 (M10), one warmup and three repetitions](../../../../diagnostics/round13_3_20261025/figures/costs_selection_work.png)

Figure 6. Same physical GPU 1, TRAIN anchors 0003 (M1) and 0348 (M10), one warmup and three repetitions. Bars show medians and observed min–max, while the report table gives mean/p90. Route M10 complete native is about 109 ms versus Full 90 ms, despite zero selected M/E values; its 48-case macro is about 503 ms versus 1,380 ms. Native query work and the mixed-case macro are distinct. Lower allocation/training cost accompanies failed reads and does not establish useful acceleration.

[PDF master, page 1](../../../../diagnostics/round13_3_20261025/figures/cost_work_interventions.pdf#page=1)

![Actually scored, eligible and positive-gate executed contextual pairs for the complete-native anchors and maintained mixed-case 48-case macro](../../../../diagnostics/round13_3_20261025/figures/context_pair_work.png)

Figure 7. Actually scored, eligible and positive-gate executed contextual pairs for the complete-native anchors and maintained mixed-case 48-case macro. Both arms score 73,728 EE pairs per scene; Full M1 executes 55,860 EE value edges and Route 73,728 through the segment backend. Decoder null collapse does not remove contextual exchange, cached projections or preparation cost. Route names are destination first; these tables are execution evidence, not causal graphs.

[PDF master, page 2](../../../../diagnostics/round13_3_20261025/figures/cost_work_interventions.pdf#page=2)

![Five no-fit Route interventions on fixed sources 0003/0274/0293/0678 (TRAIN M1 and exposed VAL M3/M7/M10)](../../../../diagnostics/round13_3_20261025/figures/route_interventions.png)

Figure 8. Five no-fit Route interventions on fixed sources 0003/0274/0293/0678 (TRAIN M1 and exposed VAL M3/M7/M10). B-only, count-matched geometry-B, dense zero-weight candidate execution and dependent-stage recomputation have zero field effect at this endpoint. Geometry member scores reopen 9/272, 21/274 and 30/275 M/E rows over the three VAL queries, with native fluid-field delta RMSE 3.061/3.100/3.186. This changes the function and is not a recovered trained model; the baseline zero-count match is degenerate.

[PDF master, page 3](../../../../diagnostics/round13_3_20261025/figures/cost_work_interventions.pdf#page=3)

![Existing saved TRAIN reference families 0001/0318/0333/0348 and known homogeneous-zero-heat diagnostics on TRAIN 0003/0348, epoch 100, without response fitting or new solves](../../../../diagnostics/round13_3_20261025/figures/saved_responses_limits.png)

Figure 9. Existing saved TRAIN reference families 0001/0318/0333/0348 and known homogeneous-zero-heat diagnostics on TRAIN 0003/0348, epoch 100, without response fitting or new solves. Full mixed-geometry q-normal response error is 14.28; Route is 3.554 with a failed member path. Zero-heat material-temperature RMSE for M10 is Full 7.096 and Route 10.376. Predicted positive-heat flow changes are separate from saved-reference fidelity. These bounded controls do not establish general response or inverse readiness.

[PDF master, page 1](../../../../diagnostics/round13_3_20261025/figures/saved_responses_limits.pdf#page=1)

## Verification and recovery handoff

Final focused verification passed 113 CPU tests, 10 plotting/progress/output tests, a final five-test plot rerun, and the actual M12 CUDA checkpoint-replay regression (one test on physical GPU 1). Coverage includes safe loading/recovery, output-only policies, applied-stage provenance, original-ID path witnesses and coalescing, present zero-donor-measure M, independent typed nulls, all-null/absent-M behavior, E eligibility, live gradients, permutation and query ordering/tiling. The native M1/M12 backward and actual four segmented fitting epochs are additional execution evidence, not mature quality qualification. A plotting-classification defect for the new native names/counters was corrected without changing fitting math; final curves are regenerated from unchanged complete histories. [Verification receipt](../../../../diagnostics/round13_3_20261025/test_results.json).

Both retained epoch-100 checkpoints were loaded and restored in memory using exact named optimizer moments and Python/NumPy/Torch/CUDA RNG state, preserving TRAIN600/VAL89 membership. This checks endpoint recoverability without executing a post-100 epoch. Actual epoch-advancing resume was tested separately on each qualification lineage from epoch 1 to epoch 2. Latest, best-field and declared epoch-100 checkpoints remain (both best-field aliases select epoch 100); use latest for interruption recovery and the explicit epoch-100 file for this report’s measurements.

Executed qualification recovery commands, from `HONF_Proj` in the ModularDT environment:

```bash
export CUDA_VISIBLE_DEVICES=0,1
python tools/honf_research.py resume-member-routing --run /data2/wanglz/ModularDT/round13_3_20261025/qualification_full --device cuda:0 --devices cuda:0 --stop-after-epoch 2
python tools/honf_research.py resume-member-routing --run /data2/wanglz/ModularDT/round13_3_20261025/qualification_route --device cuda:1 --devices cuda:1 --stop-after-epoch 2
```

Those completed qualification commands are records of tests and now correctly reject a nonadvancing stop. To recover an interrupted scientific lineage before its saved horizon, use `resume-member-routing --run RUN --device DEVICE --devices DEVICE --stop-after-epoch 100`, with the original visible-device inventory. Both scientific lineages have already reached 100, so running that command now also correctly rejects a nonadvancing stop. An explicit future user-authorized endpoint is required to use the tested extension interface below; the placeholder is not an executed continuation or a recommendation for a particular horizon.

```bash
conda activate ModularDT
cd /home/wanglz/Desktop/src/ModularDT/HONF_Proj
export CUDA_VISIBLE_DEVICES=0,1
# Only after explicit user authorization; substitute the chosen absolute epoch.
python tools/honf_research.py resume-member-routing --run /data2/wanglz/ModularDT/round13_3_20261025/full --device cuda:0 --devices cuda:0 --stop-after-epoch AUTHORIZED_EPOCH --extend-horizon
python tools/honf_research.py resume-member-routing --run /data2/wanglz/ModularDT/round13_3_20261025/route --device cuda:1 --devices cuda:1 --stop-after-epoch AUTHORIZED_EPOCH --extend-horizon
```

The narrow extension changes only the maintained engine endpoint/milestone inventory; it preserves provider membership, normalization, constant optimizer schedule and named optimizer/RNG state. Ordinary identity/recovery tests reject unrelated provider or schedule changes. No new hash/freeze/admission service was added. Launch and evaluator interfaces are documented in [the maintained guide](../../../guides/_bk/20261029_Guide_Consolidation/03_R12_R13_Predecessor_Recipes/Interaction_Reused_Member_Routing.md).

## Storage, preservation and closeout

Payload root: `/data2/wanglz/ModularDT/round13_3_20261025`, exposed through the ignored workspace diagnostics symlink. DATA2 was verified mounted before materialization; the declared new payload limit was 4 GiB, peak temporary limit 512 MiB, minimum free headroom 200 GiB on DATA2 and 20 GiB on home. No datasets or historical geometry catalogues were copied and no new persistent cache was created. Protected historical results/checkpoints/shared data, migration backups and unrelated jobs were excluded from cleanup. The original unrelated user deletion/report edit/untracked Round12 plan remain unstaged.

Closeout removed 893 exact-allowlisted disposable files (61,088,598 logical bytes) and 588 empty directories, including successful test fixtures, temporary browser/render caches, superseded figure sets and redundant original response PNG exports. The response summaries retain their original export names; [the cleanup manifest](../../../../diagnostics/round13_3_20261025/cleanup_manifest.json) and [receipt](../../../../diagnostics/round13_3_20261025/cleanup_receipt.json) identify every removed visual and its validated replacement. No historical evidence was rewritten. All numerical arrays, metrics/provenance/manifests, logs/failure receipts, complete histories, required checkpoints and the nine selected PDF/PNG pages remain. Protected checkpoint metadata was identical before/after cleanup; every allowlisted payload is absent. Before/after free space was DATA2 3,214,405,898,240 / 3,214,471,663,616 bytes and home 83,003,346,944 / 83,003,346,944 bytes. The retained payload is approximately 303.3 MiB, below the 4 GiB limit; actual DATA2 capacity remains 3.214 TB free and home 77.30 GiB free. The unrelated existing GPU-2 U-S process (PID 2032806) continues unchanged. Required DATA2 bindings remain in place.

The Markdown was rendered to standalone HTML and visually checked in the existing headless Firefox; all nine selected figure pages were inspected, all relative links/PNG decodes/PDF page counts passed, and prose/captions remain on one continuous source line. [Render/link validation](../../../../diagnostics/round13_3_20261025/report_validation.json) and [browser inspection](../../../../diagnostics/round13_3_20261025/report_browser_validation.json) are retained locally. Generated figures, data and one-time helpers remain ignored on DATA2; only durable implementation, tests, configurations, guide/specification and this report are uploaded on the current working branch.
