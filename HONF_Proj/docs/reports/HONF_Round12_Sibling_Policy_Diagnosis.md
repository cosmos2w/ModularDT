# Round12 follow-up: why R1 rejected merges and regressed at the terminal checkpoint

**Subsequent completed work:** The [compact recovery and validation continuation](HONF_Round12_Compact_Recovery_and_Validation_Report.md) measures the amended genuinely compact terminal100 and selected-best90. The original diagnosis below concerns the preserved zero-merge run; original best80 is a singleton predictor reference. The follow-up establishes recovered compact fields and heat-response utility, with mixed geometry/decision and efficiency limitations reported separately.

The saved runs reveal two distinct failures. The absence of merges is an objective-design failure: the implemented tie loss can be reduced by rejecting a pair, and the measured structural gradient strongly favors that escape. The large local100 predictor regression is a later, nonmonotone optimization outcome; it is not evidence of destructive coarsening, because no groups were coarsened. The saved R1 best-by-field checkpoint at local80 has comparable native field accuracy to R0’s best checkpoint at100. No training, optimizer update, solver call or checkpoint modification was performed for this diagnosis.

I found no evidence of a candidate-index, strict-cut, optimizer-membership, fork/clock, freeze/recovery or additive-loss denominator bug. The relevant implementation matches the supplied work plan, including its problematic tie-loss definition. This distinguishes a flaw in the specified learning incentive from an implementation failing to execute that specification. It does not prove every implementation path is bug-free or identify the unique cause of the late optimization spike.

## 1. The policy learned rejection; candidate generation and cut execution worked

A CPU geometry-only survey reconstructs the actual input transform, native records, same-type adjacent sibling metadata and compatibility MLPs on TRAIN600 plus canonical exposed VAL89. It covers 67,963 eligible candidates: TRAIN has 1,595 M and 57,600 E candidates, and VAL89 has 224 M and 8,544 E candidates. At common20 all logits are exactly −2, corresponding to sigmoid probability 0.1192. At R1 before40 every logit is negative: range −6.3832 to −4.9157, median −5.4528, with mean probability about 0.00423 for M and 0.00437 for E. The strict b>0 rule therefore correctly accepts zero pairs. This is not a lack of eligible pairs or a silent threshold reset.

The geometry-only survey was checked against actual CPU core.prepare on TRAIN0006/M1 and0254/M12 at R1 before40: candidate topology and float32 logits match exactly, with maximum difference zero. Saved policy tensors are equal at40 and100. The eight compatibility parameters are in the AdamW group; they acquire 260 optimizer steps during local21–40 and remain at260 after freezing, while other trainable parameters reach1,300 at100. Thus the policy was optimized and subsequently frozen as intended. Common20 fork, case/query sampling and all100 clocks match between arms; the cut changes only the two cut/freeze buffers.

The topology and policy use input-only same-type features as specified. E topology is intentionally insensitive to the module layout at fixed domain/material/operating conditions, while live C and query routing retain contextual-state dependence. That limits what this policy can distinguish, but it does not explain away the measured reject-all incentive. The duplicated normalized separation feature is redundant; it is not a demonstrated cause of failure.

## 2. The tie objective has a cheaper path than learning compatible groups

Let p=sigmoid(b), let D(p) be the equal-donor-type mean absolute difference between the two blended membership rows, let P be the scene’s eligible sibling-pair count and N its original-record count. The implemented contribution for one pair, omitting the outer equal-case average, is:

\[
L_{struct}=\alpha\left[\frac{\lambda_{tie}}{P}pD(p)-\frac{\lambda_{cap}}{N}p\right]+\text{constant}.
\]

Its logit derivative is:

\[
\frac{\partial L_{struct}}{\partial b}=\alpha p(1-p)\left[\frac{\lambda_{tie}}{P}\left(D+p\frac{\partial D}{\partial p}\right)-\frac{\lambda_{cap}}{N}\right].
\]

The positive D term penalizes probability itself whenever sibling rows differ. Gradient descent can reduce the loss by making b more negative, avoiding both tying and merging instead of making the rows compatible. The capacity term has the correct opposite sign, but the actual calibration gives lambda_tie=0.178774665 and lambda_cap=9.9890278e-5. Ignoring the row-change derivative, the explicit gate rejects a pair whenever D exceeds roughly 2.8e-4 for these scenes. Actual initial mismatch medians range from 0.0042 to0.0728 in the two fixed representatives, well above that threshold. As p falls, both its tying effect and sigmoid gradient shrink; the fixed b>0 threshold then becomes increasingly remote.

The existing two-term 2%-of-native calibration does not balance their opposition on b: the tie norm includes many compact membership/control parameters, while capacity acts on the compatibility policy. Its reported scaled gradient norms are 6.0129e-4 for tie and9.7825e-6 for capacity on their respective participating sets. These are different parameter sets, so neither the shared “2%” label nor scalar loss size establishes a weak tie gradient on the policy. In addition, the first48 TRAIN calibration cases contain 45 M1, one M2 and two M3 cases: only3 M candidate pairs against4,608 E pairs. This is a coverage weakness in the chosen calibration sample, not an invalid data split or proof of an optimizer bug.

I differentiated the actual saved common20 organization on the fixed TRAIN representatives0006/M1 and0254/M12 at alpha=0.05 (the first soft epoch) andalpha=1 (the calibration/ramp endpoint). No optimizer was stepped. The raw tie value matches the maintained structural-loss function, and its derivative equals the sum of the explicit probability-weight derivative and the blend/membership derivative. For all198 eligible pairs at each alpha, the original weighted tie-plus-capacity derivative with respect to b is positive: the structural objective locally encourages rejection. Native/work VJPs are excluded from this diagnostic, so it does not claim the exact total optimizer direction for every case.

| Alpha | TRAIN case/type | Pairs | Explicit rejection / capacity gradient L2 | Current dL/db >0 | Proposed dL/db <0 |
| --- | --- | --- | --- | --- | --- |
| 0.05 | 0006/E | 96 | 19.6× | 96 | 96 |
| 0.05 | 0254/M | 6 | 279.2× | 6 | 6 |
| 0.05 | 0254/E | 96 | 326.5× | 96 | 96 |
| 1.0 | 0006/E | 96 | 17.2× | 96 | 96 |
| 1.0 | 0254/M | 6 | 258.2× | 6 | 6 |
| 1.0 | 0254/E | 96 | 296.6× | 96 | 96 |

Positive dL/db means descent reduces b. The direct rejection gradient is17–327 times the capacity gradient on these pair-logit axes. This explains the observed direction of learning more specifically than saying capacity was merely too weak. Saved training history independently agrees: the unweighted capacity surrogate rises from approximately0.9412 at21 to0.9976 at40, while the expected reduction fraction falls from5.88% to0.239% of N. Hard accepted pairs remain zero at the cut.

## 3. Terminal predictor regression is not a uniform R1 failure

R1 first differs modestly at local30 and40, can recover to better field scores at70–80, then deteriorates toward100. The pre/post40 probe retains K=N in all six scenes and changes fluid-T predictions only by at most0.0026446 RMSE, so it does not account for the large terminal gap. With zero accepted pairs, the post-cut model has the singleton forward form; R1 differs from R0 in learned tensors, Adam moments and controller trajectory established during soft training, not in a smaller bank.

| Local epoch | R0 exposed VAL field score | R1 exposed VAL field score | R1/R0 |
| --- | --- | --- | --- |
| 30 | 0.00218874 | 0.00224755 | 1.027 |
| 40 | 0.00214337 | 0.00231978 | 1.082 |
| 60 | 0.00229347 | 0.00471532 | 2.056 |
| 70 | 0.00250762 | 0.00216758 | 0.864 |
| 80 | 0.00217806 | 0.00206739 | 0.949 |
| 90 | 0.00223762 | 0.00243188 | 1.087 |
| 100 | 0.00214090 | 0.00506705 | 2.367 |

The late instability is visible in TRAIN as well as exposed VAL: R1’s recorded epoch sum of 13 macro-update pre-clip global gradient norms reaches18.57 at98 versus3.53 for R0, and remains12.56 at100 versus2.06. These are epoch sums, not the largest individual update norm. R1 already had a separate spike at60 and recovered afterward. Both runs use constant LR3e-4 and the same data/query schedule. These facts support an unstable optimization trajectory and sensitivity to the final iterate. They do not identify whether soft perturbations, structural gradients, adaptive moments, sampled queries or numerical reduction differences triggered that trajectory; no training replay/ablation was run.

A same-weight diagnostic compares saved R1-final100 with a disposable singleton-mode copy under identical execution options, plus an ordinary repeated R1 preparation. C differs by only about1.2–1.3e-6 in either comparison; fluid-T output-difference RMSE is approximately1.1e-5 on TRAIN0254 in both the mode comparison and the ordinary repeat. This is consistent with the existing FP32 segment-reduction variability, rather than a hidden field-changing cut switch. The comparison is numerical, not bitwise; early exact-equality assertions were rejected and their compact failure logs are preserved. It does not attribute the long optimization trajectory to rounding.

The existing best_by_field_mse aliases, selected by the same monitoring rule, point to R0-local100 and R1-local80. I ran a fresh full-native VAL89 field board on saved R1-local80 without training and compared it with the unchanged R0-local100 board. The terminal100 negative comparison remains the authorized matched-horizon result; selected-best comparisons are a separately labeled, already-exposed validation analysis, not a replacement endpoint or independent TEST.

| Native field | R0 best100 mean | R1 best80 mean | R1 final100 mean | Best80 vs R0 mean | Best80 vs R0 p90 |
| --- | --- | --- | --- | --- | --- |
| u | 0.0105283 | 0.00994911 | 0.0137592 | -5.50% | -6.05% |
| v | 0.00153183 | 0.00153675 | 0.0018379 | +0.32% | +4.98% |
| p | 0.00575354 | 0.00587325 | 0.00700722 | +2.08% | -1.24% |
| omega | 0.0275909 | 0.0275151 | 0.0310708 | -0.27% | -2.15% |
| temperature_fluid | 0.365555 | 0.369392 | 0.564668 | +1.05% | -4.60% |
| temperature_surface | 0.316832 | 0.287659 | 0.626164 | -9.21% | +2.90% |
| temperature_material | 0.343554 | 0.306267 | 0.704775 | -10.85% | -0.48% |
| q_proxy | 1.21355 | 1.2042 | 1.47613 | -0.77% | -0.24% |
| module_peak_endpoint | 0.31949 | 0.314344 | 0.680675 | -1.61% | -3.64% |

All four priority temperature/peak quantities meet the plan’s descriptive +10% mean and+15% p90 bands under the existing selected-best rule. Surface/material means improve by9.2%/10.9%, fluid mean is1.05% higher, and peak mean is1.61% lower than R0-best100. This corrects an overbroad interpretation of “R1 regressed”: its terminal checkpoint regressed strongly, but its best saved predictor is comparable. R1-best80 still has zero merges. Its saved-response/ranking and inverse reliability have not been reevaluated at80 and must not inherit claims from either final100 or the field board.

## 4. Smallest reasonable fix to propose

For a future, separately authorized candidate, change only the explicit probability weight in the tie loss to a stopped-gradient weight. Keep the current probability differentiable inside the soft descriptor/geometry blend and inside capacity; keep the native field/work objectives, original record values, encoder, Thermal law, b initialization and strict cut rule unchanged. The proposed line is:

```python
tie = (probability.detach() * pair_discrepancy).sum(-1) / compact.pair_mask.sum(-1).clamp_min(1)
```

This removes the direct positive D term in the policy derivative while preserving gradients that improve the blended memberships. At identical saved weights it changes no forward predictions and no scalar tie-loss value. With the original coefficients, the no-step probe reverses the structural derivative toward merging for every tested pair at both alpha values. That is evidence that the proposal removes the demonstrated escape, not evidence that a retrained model would produce useful K<N groups or retain fields.

The focused regression check for an eventual implementation should assert that the explicit probability-weight escape is absent, gradients through D and the blend remain live, capacity still pushes b upward, and saved-weight forward/source-law behavior remains unchanged. Apply the existing TRAIN-only calibration procedure to the changed gradient before any future experiment; record per-bank policy-head coverage and signed competition. Its present first48 sample barely covers M, so a representative input-only calibration selection within the same48-case limit is a sensible preparation improvement. Do not jointly tune capacity strength, ramp, threshold and initialization against VAL89 or force any merge.

For the already trained weights, the smallest accuracy action requires no model change: use the existing selected-best checkpoints for predictive reference and keep final100 explicitly labeled as terminal-horizon evidence. Keep R0 as the matched compression control and R1-best80 as a comparable singleton predictor reference. I do not recommend a long R1 continuation to obtain merges: its input-only compatibility policy is frozen and rejects all current TRAIN/VAL candidates. No policy fix has been applied or trained in this follow-up.

## Figure and retained diagnostic evidence

![Saved field trajectory, selected checkpoints, capacity collapse and structural gradient direction](../../diagnostics/thermal_compact_round12_20261023/diagnosis_20261023_v1/evidence/round12_diagnosis_v2.png)

Figure1. A: all ten saved exposed VAL monitoring scores, with stars at the existing selected-best states and the single cut at40. B: fresh full-native priority-field mean ratios distinguish comparable R1-best80 from degraded R1-final100. C: the TRAIN expected-capacity reduction surrogate collapses toward zero before the strict cut; it is not measured allocated K. D: the saved-common20 structural-only gradient on the two fixed TRAIN representatives changes sign after detaching only the outer probability weight, with original coefficients and alpha1. These are saved-run arithmetic and no-step diagnostics, not a new trained comparison or a physical-causality result.

[PDF master](../../diagnostics/thermal_compact_round12_20261023/diagnosis_20261023_v1/evidence/round12_diagnosis_v2.pdf); [policy survey](../../diagnostics/thermal_compact_round12_20261023/diagnosis_20261023_v1/evidence/agent_policy_logits.json); [actual prepare parity](../../diagnostics/thermal_compact_round12_20261023/diagnosis_20261023_v1/evidence/agent_prepare_parity.json); [structural gradients](../../diagnostics/thermal_compact_round12_20261023/diagnosis_20261023_v1/evidence/structural_gradient_probe.json); [trajectory](../../diagnostics/thermal_compact_round12_20261023/diagnosis_20261023_v1/evidence/trajectory.json); [R1-best80 native board](../../diagnostics/thermal_compact_round12_20261023/diagnosis_20261023_v1/evidence/R1_best80_VAL89/result.json); [selected checkpoint comparison](../../diagnostics/thermal_compact_round12_20261023/diagnosis_20261023_v1/evidence/selected_checkpoint_comparison.json); [same-weight singleton check](../../diagnostics/thermal_compact_round12_20261023/diagnosis_20261023_v1/evidence/same_weight_singleton_equivalence.json). The original terminal experiment report remains the source for full before/after-cut arrays, responses and rankings.

## Boundaries and closeout

Only existing TRAIN and exposed VAL data/checkpoints were read. This follow-up performed zero optimizer updates and created no model or optimizer checkpoint; existing runs, hashes, normalizers, source inputs, user changes and unrelated jobs were preserved. New generated evidence uses the verified /data2 mount with a256 MiB hard new-output budget,128 MiB temporary estimate,200 GiB free-headroom requirement and zero persistent new dataset-cache budget. Existing data recipes and caches were reused. One selected PDF/PNG diagnostic pair and compact numeric evidence are retained locally; one-time helpers and generated outputs are excluded from Git.

Cleanup removed 3 diagnosis-owned cache/superseded-export files (476,155 logical bytes) after exact allowlist, active-handle/mapping and dependency checks. The diagnosis retains 16,754,941 bytes at the receipt snapshot, within its 256 MiB budget. Selected PDF/PNG, numerical diagnostics and compact failure logs remain; scientific checkpoint digests match the original Round12 closeout and source U-H e1000/workspace bindings are unchanged. Actual remaining free space is 3,221,974,482,944 bytes on /data2 and 97,338,175,488 bytes on the home filesystem. No training or evaluation worker remains owned by this follow-up.

Implementation remains unchanged at source revision1608b74; this round proposes the objective amendment only. Durable diagnostic documentation is uploaded through the existing artifact gate; [delivery verification](../../diagnostics/thermal_compact_round12_20261023/diagnosis_20261023_v1/delivery_verification.json) records the normal push and remote/local tip equality.

Subsequent implementation: the proposed outer-probability stop-gradient amendment was implemented and qualified in a separate bounded run through local41; see the [tie-loss feasibility report](HONF_Round12_Tie_Stop_Feasibility_Report.md). This does not alter the saved runs or the zero-update scope of the diagnosis above.
