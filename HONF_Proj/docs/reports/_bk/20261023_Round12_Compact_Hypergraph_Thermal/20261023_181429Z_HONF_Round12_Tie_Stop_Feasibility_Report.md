# Round12 tie-loss amendment and bounded feasibility check

**Subsequent completed work:** The [compact recovery and validation continuation](20261023_195951Z_HONF_Round12_Compact_Recovery_and_Validation_Report.md) continues the exact amended local41 state through local50 and matched local100, with separate selected-best90 evaluation. Its terminal100 compact bank recovers global priority field quality and preserves primary response-error bands, while mixed geometry responses, candidate ranking and speed remain limited. This report preserves the historical local41 feasibility scope; its one-epoch recovery and then-missing evaluations are not the final scientific status.

The sibling tie loss now stops the gradient through its outer probability weight. The probability inside the live descriptor/geometry blend and the capacity term remains differentiable. This removes the demonstrated incentive to avoid tying by rejecting a mismatched pair; it changes neither the scalar loss nor predictions at fixed weights. The amendment follows the [saved-run diagnosis](20261023_171813Z_HONF_Round12_Sibling_Policy_Diagnosis.md). A bounded 273-update qualification now halves the E bank, while the M policy still rejects all candidates and field recovery remains incomplete. Existing common20, R0, R1 and the literal U-H e1000 source are preserved.

## Change and tests

For a pair with probability p and live blended membership discrepancy D(p), the tie contribution is now stopgrad(p)D(p). Its policy derivative retains p times the derivative of D, and removes the explicit D times the derivative of p. Capacity retains its negative logit derivative, so minimizing capacity still encourages acceptance. This is an intentional change to the training gradient convention; it does not guarantee that native task gradients will permit useful merges.

The only model-code change is the outer detach in `CompactGroupRetrieval.structural_losses`. Original record values, source columns, encoder, direct K-by-N construction, query-to-K scoring, Thermal source law, initialization, strict b>0 cut, optimizer, ramp and training samples are unchanged. The new named recipes record `round12.tie_loss_gradient=stop_outer_probability_v1` in the existing full-config run identity. That field documents the amendment; it introduces no alternate engine or runtime admission mechanism. Training uses source commit `2de91a9`.

Focused compact tests passed 18 checks. They verify exact old/new scalar tie-value equality, absence of a direct policy gradient when discrepancy is fixed, closed-form live membership gradients, live gradients through actual soft descriptors to the policy, a negative capacity gradient, and masking of padded pairs. Another 59 integration checks passed across the adapter, inherited normalization, additive objective, research runner and evaluator. The mutation check temporarily substitutes the original formula in an isolated process: the focused regression fails at the intended fixed-discrepancy assertion, with a nonzero logit derivative of 0.1321. The committed formula passes. Test logs and the [independent review receipt](../../../../diagnostics/thermal_compact_round12_20261023/tie_fix_20261023_v1/evidence/independent_validation_receipt.json) are retained with the local evidence.

## Saved-weight predictions

The actual CPU native comparison uses literal common20/e20 and original R1/e100 weights on fixed TRAIN0006/M1 and 0254/M12. All fluid u/v/p/omega/T, interface T/q-proxy and material-T prediction arrays are bitwise identical between the old-formula and amended-formula copies. Compact memberships, controls, geometry, pair logits/probabilities, bounded query-to-group/access weights and affine response coefficients also match exactly. State tensors, normalizer tensors and checkpoint digests remain unchanged. This verifies fixed-weight behavior on the declared panel; it does not imply identical future training updates. The [parity receipt](../../../../diagnostics/thermal_compact_round12_20261023/tie_fix_20261023_v1/evidence/saved_weight_prediction_parity.json) retains shapes, hashes and zero-difference statistics. An in-memory common20 alpha=0.05 loss-gradient check also gives exactly equal scalar tie/capacity values and capacity derivatives, while changing the tie derivative as intended; it performs no optimizer step.

## Training boundary

This is a short feasibility qualification from the unchanged full-TRAIN common20 state, not a new matched local 100 comparison. Only the amended R1 is trained; the existing fork command also creates an untrained R0 bootstrap. The pilot preserves the original full-TRAIN600 membership, inherited e1000 normalizer, seed 0, query keys, effective batch 48, microbatch 24, Q=1024, constant source-terminal learning rate, native objectives and work controllers. It recalibrates both weak structural coefficients using the original next-TRAIN-macro procedure and the changed gradient. The recalibrated tie coefficient is 0.193474576 and capacity coefficient 9.98895787e-5; the old coefficients were 0.178774665 and 9.98902780e-5. Calibration visits the same 48 TRAIN cases without updating parameters. Its weak M coverage remains a limitation: 45 M1, one M2 and two M3 cases provide only 3 M pairs against 4,608 E pairs. No validation-dependent coefficient, threshold, initialization or candidate tuning is performed.

The hard new-training limit is local 21–41: 20 soft epochs, the existing single strict cut after 40, and one recovery epoch. That is 273 new optimizer updates, 12,600 case visits and 12,902,400 primary-query presentations, excluding auxiliaries. The 20-epoch ramp and cut 40 are retained, making this the shortest check of the complete existing transition followed by an actual recovery epoch. The sealed recipe horizon remains 100 for ordinary user-controlled continuation; this request stops at 41.

The run completed exactly through local 41: 273 new updates, 12,600 new TRAIN visits and 12,902,400 primary-query presentations. R0 bootstrap remains at 20 with zero new updates. The sum of measured epoch training times is 924.27 s (15.40 min), excluding startup and monitoring. Final clocks are sampler 1041, controller 4041 and joint macro 13533; cumulative child updates are 533, including 260 inherited updates. All 21 new case-order/query-sampling signatures match the original R1 schedule. The hard cut changes only `compact_cut_active` and `compact_policy_frozen`; the eight policy tensors and their Adam moments stay exact through recovery, with Adam step 260, while the other 283 stepped parameters reach 533. Normalization and original source bindings remain unchanged. No new solver, TEST, Wind or U-S experiment was launched.

The exposed validation monitoring field score is 0.00228624 at 30, 0.00672148 before the cut at 40, and 0.00332416 after the one recovery epoch 41. Thus the gradient amendment enables a compact bank, but late tying worsens the predictor and one epoch does not establish recovered field quality. The pre/post-cut states, final41, latest and existing best aliases are retained. The best alias remains the inherited common-prefix state, so all current-candidate results here use explicit 41 rather than relabeling that inherited predictor as a trained compact success.

## Predictor, organizer and inverse interpretation

**Predictor.** The complete native VAL89 board at 41 has finite predictions and the following equal-case errors in dataset-native units. These are exposed validation results after a short recovery, with no new matched R0 training or independent TEST. They do not establish a predictor improvement over the preserved references. Keep the existing validated predictors as references; further recovery is a user-controlled next experiment, not work performed here.

| Native quantity | VAL89 mean RMSE | VAL89 p90 RMSE |
| --- | --- | --- |
| u | 0.01218948 | 0.01643939 |
| v | 0.001739947 | 0.002205539 |
| p | 0.0066506 | 0.009331562 |
| omega | 0.03076274 | 0.04236111 |
| temperature_fluid | 0.4814722 | 0.6882571 |
| temperature_surface | 0.4528929 | 0.7490425 |
| temperature_material | 0.4875595 | 0.782557 |
| q_proxy | 1.442387 | 2.298044 |
| module_peak_endpoint | 0.4574988 | 0.8527962 |

The combined cut is a finite operator change, not an exact merge. On TRAIN0006/M1 and 0254/M12, its fluid-temperature prediction-difference RMSE is 0.0299633 and 0.255963, respectively. On0254, fluid-T error against the stored native target changes 1.43897→1.29229 across the cut, then reaches 0.557453 at 41. Detailed component deltas use saved physical masks, including separate surface-T and outside q-proxy validity; mixed-unit aggregates are not interpreted as a scalar physical error. Six scenes have before/after native RMSE boards; detailed physical-output delta arrays are retained for the fixed two TRAIN scenes.

**Organizer.** The learned E rule accepts every eligible E sibling on TRAIN600; it still rejects every M sibling. The geometry-only survey reconstructs the maintained candidate/logit APIs and agrees with the measured native TRAIN2 cuts. Its counts are explicitly geometry-derived, while the fixed six before/after boards and complete VAL89 board measure actual live K. The native M1 scene changes 193→97 groups, M12 changes 204→108, and all 89 validation scenes retain96 E groups with total K=99–106.

| TRAIN600 bank | Original records N | Eligible pairs P | Accepted pairs after cut | Groups K after cut |
| --- | --- | --- | --- | --- |
| M | 3,417 | 1,595 | 0 | 3,417 |
| E | 115,200 | 57,600 | 57,600 | 57,600 |
| Total | 118,617 | 59,195 | 57,600 | 61,017 |

The total geometry-derived K/N is 0.51440. Mean E/M logits are +2.47617/−3.36168, with mean probabilities 0.91966/0.03353. All candidate logits are bitwise equal at 40 and 41; 81 other compact-retrieval tensors change during recovery, confirming the descriptor/scorer remains trainable. This shows that removing the direct rejection shortcut permits actual E contraction. It does not establish selective functional redundancy, M compression, layout-adaptive E cuts or physical causality: this input-only E topology remains fixed across layouts at fixed operating inputs. Original donor records and source columns remain on the N axis. No inference-cost probe was run; smaller K is not an end-to-end speed claim, and the dense final executor continues to evaluate original M/E value rows.

**Inverse and responses.** No inverse designs, new response/ranking campaign or current-candidate inverse-reliability claim is made. Saved-weight affine response factors match exactly in the parity check; those unchanged-weight checks do not validate responses after retraining. The original response/ranking evidence remains intact and belongs to its original checkpoints. Physical-only pilot comparison receipts explicitly mark response-K unevaluated. Qualify the new weights before using them for inverse decisions.

## Key figures

Figure index: [organization PDF](../../../../diagnostics/thermal_compact_round12_20261023/tie_fix_20261023_v1/evidence/tie_stop_organization.pdf) and [native-field PDF](../../../../diagnostics/thermal_compact_round12_20261023/tie_fix_20261023_v1/evidence/tie_stop_native_fields.pdf). The PNG companions below support direct Markdown embedding; these two PDF masters and their companions are the selected exports.

![Expected capacity during tying, actual cut bank counts and direct K-by-N memberships](../../../../diagnostics/thermal_compact_round12_20261023/tie_fix_20261023_v1/evidence/tie_stop_organization.png)

Figure 1. TRAIN epoch means of the capacity surrogate diverge after the amendment: expected reduction rises to 43.69% of N rather than collapsing toward zero under the old gradient. This is a surrogate, shown separately from measured K in two TRAIN and four exposed VAL scenes. TRAIN0254 direct memberships change204×204→108×204 at the cut with the same trained weights; original donor columns remain 204. M/E separators are cyan. The membership heatmaps show executed learned support, not physical-causal edges or a timing measurement.

![Native temperature target, before-cut and recovery predictions with physical residuals and cut-induced change](../../../../diagnostics/thermal_compact_round12_20261023/tie_fix_20261023_v1/evidence/tie_stop_native_fields.png)

Figure 2. TRAIN0254/M12 uses the same stored native target, physical coordinates and 7,318 valid exterior-fluid receivers throughout. Top panels share a dataset-temperature color scale; the lower residual panels share a signed scale, while the cut-change panel has its own labeled scale. The cut changes fluid-T predictions by 0.255963 RMSE (maximum 1.12518); error against the target falls from 1.43897 before 40 to 1.29229 after 40 and 0.557453 after 41. This demonstrates finite cut effects and partial recovery on one TRAIN geometry, not validation-population accuracy or recovered surface/material/peak fidelity. No new physical solver reference was generated.

## Commands and retained evidence

The executed sequence was ordinary `fork-common` from protected common20, `resume --stop-after-epoch 40`, `activate-cut`, and `resume --stop-after-epoch 41`. A first fork attempt with a one-GPU visibility mask failed before child creation or optimizer updates because the checkpoint stores three CUDA RNG streams; the successful sequence preserves the original visible-device list and explicitly computes on GPU2. Native evaluation used GPU1. Compact failed-attempt logs are retained.

The following command advances only the new candidate to 50 if the user chooses further recovery. It is not launched by this round. The existing resume/restore path and stop 50 request were [qualified without updates](../../../../diagnostics/thermal_compact_round12_20261023/tie_fix_20261023_v1/evidence/recovery41_stop50_readonly_qualification.json); the actual40→41 continuation also exercised post-cut training. A stop at 50 adds 117 updates and 5,400 TRAIN visits. Preserve the three-device visibility when restoring this checkpoint.

```bash
cd /home/wanglz/Desktop/src/ModularDT
python_bin=/home/wanglz/miniconda3/envs/ModularDT/bin/python
pilot_root=/data2/wanglz/ModularDT/diagnostics/thermal_compact_round12_20261023/tie_fix_20261023_v1
env -u CUDA_VISIBLE_DEVICES PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR="$pilot_root/temporary/mpl" \
  "$python_bin" HONF_Proj/tools/honf_research.py resume \
  --run "$pilot_root/runs/R1_tie_stop" --device cuda:2 --devices cuda:2 --stop-after-epoch 50
```

Inspect storage/headroom and review native fields at the chosen next stop before considering a longer extension. The recipe horizon 100 is a retained protocol boundary, not an automatic launch authorization. Historical soft-phase replay with the original gradient uses the pre-amendment source `59514b5`; saved inference and the preserved completed post-cut runs remain unchanged.

Selected local evidence: [gradient tests](../../../../diagnostics/thermal_compact_round12_20261023/tie_fix_20261023_v1/evidence/focused_18tests.stdout.log), [integration tests](../../../../diagnostics/thermal_compact_round12_20261023/tie_fix_20261023_v1/evidence/integration_tests.log), [fork equality](../../../../diagnostics/thermal_compact_round12_20261023/tie_fix_20261023_v1/evidence/fork_state_equality.json), [TRAIN600 policy survey](../../../../diagnostics/thermal_compact_round12_20261023/tie_fix_20261023_v1/evidence/policy_recovery_geometry_e40_e41_v2.json), [TRAIN2 cut deltas](../../../../diagnostics/thermal_compact_round12_20261023/tie_fix_20261023_v1/evidence/compare40_TRAIN2.json), [VAL4 cut metrics](../../../../diagnostics/thermal_compact_round12_20261023/tie_fix_20261023_v1/evidence/compare40_VAL4.json), [current native VAL89](../../../../diagnostics/thermal_compact_round12_20261023/tie_fix_20261023_v1/evidence/final41_VAL89/result.json), and [figure metrics](../../../../diagnostics/thermal_compact_round12_20261023/tie_fix_20261023_v1/evidence/figure_metrics.json).

All generated evidence, one-time helpers and checkpoints remain under the ignored [pilot directory](../../../../diagnostics/thermal_compact_round12_20261023/tie_fix_20261023_v1). The shared dataset and existing caches are reused. The new-output budget is 2 GiB, peak temporary estimate 1 GiB, minimum /data2 free headroom 200 GiB, and new persistent dataset-cache budget zero. The preflight records exact workspace bindings, free capacity and digests for 63 protected original-run/source files.

Cleanup removed 111 disposable files/links and 60 empty directories (9,033,920 logical file bytes; 9,584,640 allocated bytes). This includes pytest fixtures, font caches, redundant diagnostic stdout, a superseded survey export and two accidental bytecode caches from an interrupted diagnostic. Exact recursive inode/link/owner/checksum allowlists, accessible open-handle/mapping checks and dependency review preceded removal. All 63 protected original-run/source files and every new scientific run file retain their checksums. The [cleanup receipt](../../../../diagnostics/thermal_compact_round12_20261023/tie_fix_20261023_v1/cleanup_receipt.json) retains 214,680,147 bytes at its snapshot, including all scientific checkpoints/aliases, histories, provenance, selected numerical evidence, PDF/PNG pairs, helpers and compact failure logs. Remaining capacity at cleanup is 3,221,759,393,792 bytes on /data2 and 97,315,852,288 bytes on the home filesystem. Workspace symlink bindings remain exact. No owned training/evaluation worker or automatic launcher remains; completed task-owned tmux shells were closed. User changes remain unstaged and unrelated runs are untouched.

Durable code, tests, recipes and this report are committed and uploaded through the existing pre-push artifact gate; generated artifacts remain local. The [delivery verification](../../../../diagnostics/thermal_compact_round12_20261023/tie_fix_20261023_v1/delivery_verification.json) records the audited outgoing history and remote/local tip equality.
