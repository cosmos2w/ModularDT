# ThermalChannel physical interaction and numerical reliability evidence

**Status:** train-exposed reference-solver evidence for the M0 baseline decision and bounded M3 interaction panels. This report supplements [the M0 baseline decision](20260926_195912Z_baseline_decision.md); it does not establish held-out performance, engineering feasibility, CFD causality, or learned graph edges.

## Decision summary

Run 1804 selected e4738 remains the provisional full-access incumbent `B_inc` for an explicit response refit. None of the replayed checkpoints qualifies as `B_response`; see the linked baseline report for exact checkpoint identities and absolute, finite-response, pressure, and peak metrics.

The frozen M3/0001 `i_plus`/`j_plus` pair produces numerically resolved material-temperature mixed responses on the two moved receivers at the tested step. The hot spectator module controls the observed maximum temperature but has a mixed response near zero, so the measured pair is response-relevant and is not shown to improve the current peak-temperature objective. No graph-guided design utility follows.

The M3/0304 h=0.15 cross-grid pilot failed its predeclared mixed-signal and peak-stability gates. Its 34-state coarse expansion was not run. The separate M7/0340 sensitivity check also shows that peak finite-move signs and ranking can change with mesh resolution even when an anchored mixed peak scalar appears stable. These results prevent transferring a single family’s numerical floor or local peak sensitivity to another family.

## Evidence boundary and frozen inputs

All M3 interaction results below are from existing train-split atlas families. The signed input proposal is `diagnostics/generated/baseline_decision_m0_20260926/reference_coverage_m3_m7_20260926/thermal_train_coverage_inputs.json`, SHA-256 `6625db040308ac36272df4eed55211d9d2400eb71b72a9f08e067981536f05d0`. It was not changed. The 0304 coarse atlas was reused exactly at 128×64; its new pilot inputs differed only by the frozen grid change to 256×128. No h=0.075 cases were added.

The four-state anchored response is

`R_pp - R_i+ - R_j+ + R_baseline`.

Material comparisons join by physical receiver-module ID and local coordinate after FP32 coordinate canonicalization, with duplicate checks, one-to-one coverage, and common validity masks. Eulerian fluid grids have no exactly coincident cell centers between 128×64 and 256×128, so the report gives per-grid fluid responses and does not interpolate to invent a cross-grid floor. `q_normal` is a flux proxy, not a conservation certificate.

## M3/0001: a supported moved-receiver response with a hot null spectator

The panel is `duplicate_family:0001+0273`, train split, h=0.15, with the 128×64 states `baseline`, `i_plus`, `j_plus`, and `pp`. The donors are `0001:module:0` moved +0.150000095 in x and `0001:module:1` moved +0.149999976 in y. Its exact saved-output metrics are in `diagnostics/generated/baseline_decision_m0_20260926/reference_coverage_m3_m7_20260926/m3_0001_donor_receiver_metrics.json`.

The table reports signed means in physical dataset units. RMS values are over the stated per-receiver common support; mixed RMS is the anchored mixed response.

| Receiver role | Physical receiver | Samples | `i_plus` mean | `j_plus` mean | Mixed mean | Mixed RMS |
|---|---|---:|---:|---:|---:|---:|
| Interface `T_surface` | module 0 | 64 | −0.231086 | +0.0009781 | +0.014759 | 0.021922 |
| Interface `T_surface` | module 1 | 64 | +0.011379 | +0.216281 | +0.015793 | 0.023525 |
| Interface `T_surface` | hot module 2 | 64 | +0.0000904 | +4.45e−7 | +7.64e−8 | 5.74e−7 |
| Solid temperature | module 0 | 3,096 | −0.271677 | +0.0008380 | +0.014873 | 0.018769 |
| Solid temperature | module 1 | 3,096 | +0.011265 | +0.293378 | +0.015931 | 0.020093 |
| Solid temperature | hot module 2 | 3,096 | +0.0000875 | +3.95e−7 | +2.70e−8 | 7.41e−7 |

The family-level exact-coordinate cross-grid comparison supports the moved-receiver temperature responses: `T_surface` has 192 matched samples, coarse/fine mixed RMS 0.018565/0.016300, and discrepancy RMS 0.003220, giving signal-to-discrepancy ratios 5.77/5.06. Solid temperature has 9,288 matches, RMS 0.015874/0.013578, discrepancy 0.002692, and ratios 5.90/5.04. Both exceed the predeclared 3× numerical-resolution criterion at both grids. These are family-level floors; the per-module rows do not have independent replicate-derived floors.

Because each pooled comparison has three equally sized receiver partitions (64 interface points or 3,096 solid points per module), any one module's RMS grid discrepancy is bounded above by `sqrt(3)` times the pooled RMS discrepancy. This gives conservative upper bounds of 0.005577 for interface `T_surface` and 0.004662 for solid temperature. Against those bounds, module 0/1 mixed RMS ratios are at least 3.93/4.22 for interface temperature and 4.03/4.31 for solid temperature. These are mathematical bounds from the pooled comparison, not separately measured per-module floors. Module 2 remains below the bounds and unresolved.

The hot spectator module 2 has baseline sampled peak 18.866768, above module 0 at 15.492541 and module 1 at 13.155111. Module 2 remains the maximum in baseline, `i_plus`, `j_plus`, and `pp`. The maximum-temperature finite changes are +0.0002079 (`i_plus`), +0.00000191 (`j_plus`), and +0.0002079 (`pp`); the anchored mixed change is −0.00000191. The module-2 interface and solid mixed RMS values, 5.74e−7 and 7.41e−7, are well below their family-level cross-grid discrepancies. Thus the paired signal is present on the moved modules, while the current maximum-temperature objective is effectively insensitive to the pair at this tested neighborhood.

The pressure-drop baseline is 0.074143511. The `pp` value is 0.074422975, with finite changes +0.000279436 (`i_plus`) and +0.0000000277 (`j_plus`); the anchored mixed pressure change is −7.65e−10. The 1.05×baseline value 0.077850687 is only a temporary study screen. No engineering allowable is recorded, so this panel cannot certify pressure feasibility. The `q_normal` proxy fails the same 3× cross-grid criterion: coarse/fine RMS 0.028662/0.022895 versus discrepancy 0.021912 (ratios 1.31/1.04).

This record measures a physical response of the local reference solver. It is not evidence of a learned relation, causal hyperedge, or design benefit.

## M3/0304: family-specific cross-grid gate failed

The M3/0304 pilot is `stored_family:0304`, train split, Re=70 and runtime `nu=0.012857142857142857`, with h=0.15. Nine exact coarse states were reused from the 128×64 atlas and nine frozen 256×128 inputs were attempted. The atlas NPZ SHA-256 is `f75311979d1c44b0b99fdaa2db686142fa62ffa1d5dad69bd0cccc2fb48b16f9`. The no-solve audit found zero exact and zero near matches between these fine inputs and the 52 newly created 0001 coarse raw configurations.

All nine fine solves converged with no retry. The predeclared gate failed for interface temperature, solid temperature, and maximum-peak sign/ranking stability:

| Role/channel | Matched samples | Coarse mixed RMS | Fine mixed RMS | Cross-grid discrepancy RMS | Coarse/fine signal-to-discrepancy | Gate |
|---|---:|---:|---:|---:|---:|---|
| Interface `T_surface` | 192 | 1.2030e−6 | 2.1900e−7 | 1.2221e−6 | 0.984 / 0.179 | fail |
| Solid temperature | 9,288 | 1.3270e−6 | 3.0957e−7 | 1.3605e−6 | 0.975 / 0.228 | fail |
| Interface `q_normal` proxy | 192 | 1.6912e−6 | 2.8585e−7 | 1.7187e−6 | 0.984 / 0.166 | unresolved proxy |

No maximum-peak finite move reaches the required 3× grid-resolution rule, so there are no resolved moves for a sign/ranking statistic. For example, `j_minus` changes from −0.490272 coarse to −0.277960 fine (grid discrepancy 0.212312), while `j_plus` changes from +0.238274 to +0.080597 (discrepancy 0.157677). Their signs agree, but the effects are smaller than three times the grid discrepancy. The measured Spearman statistic is therefore unavailable; the gate fails from insufficient resolved moves, not from a misleading rank computed on unresolved values.

Pressure-drop baselines are 0.054767540 coarse and 0.054777323 fine. The anchored mixed changes are 1.13e−10 and 1.18e−11. No fixed allowable pressure limit is present for this family; pressure values cannot establish engineering feasibility. Fluid mixed responses are reported separately on 7,920 coarse and 31,659 fine valid cells. Their cross-grid discrepancy was not computed because the Eulerian cell centers do not coincide.

The complete attempt ledger, configs, raw outputs, common-support masks, metrics, and no-solve reuse audit are retained under `diagnostics/generated/baseline_decision_m0_20260926/reference_coverage_m3_m7_20260926/pilot_m3_0304_fine/`. The failed gate is a stop: the frozen 34-state 0304 coarse expansion was not run, and no later reference solves were started.

## M7/0340: peak sensitivity is not portable from field mixed signals

The separate M7/0340 calibration-development sensitivity check used a layout that is train-exposed; this is not a claim that the calibration panel belongs to the train split. It used seven states at h=0.1, reused original-tolerance 128×64 atlas states, and made 14 new NumPy CPU attempts: seven 128×64 tightened-tolerance calls and seven 256×128 default-tolerance calls. All converged. The recovered first coarse/tight baseline call has unknown per-call CPU and adapter-wall timing; the other 13 calls used 136.853 process-CPU seconds and 136.877 runner-wall seconds. A distinct 600-second CPU and wall reserve is retained for that unknown call.

At the fixed coarse grid, tightening tolerances barely changed the main material mixed signals. Mesh doubling changed them more: interface `T_surface` anchored RMS was 0.029862 / 0.029857 / 0.029163 for default coarse / tight coarse / default fine, with coarse-to-fine discrepancy 0.003983; solid temperature was 0.026836 / 0.026829 / 0.026247, discrepancy 0.002808. The `q_normal` proxy was 0.040620 / 0.040620 / 0.043145, discrepancy 0.026017. Those values are a single-family sensitivity check, not a portable uncertainty floor. Exact local material coordinates were paired; Eulerian fluid fields have no exact shared cells across meshes.

Peak finite changes expose a stronger decision risk. For `j_plus`, the sampled maximum changes −0.01152 at default coarse, −0.01155 at tightened coarse, and +0.12605 at default fine. For `pp`, the changes are −0.02850 / −0.02853 / +0.10040. Both reverse sign after mesh doubling and the finite-change order changes, while the anchored maximum-peak mixed scalar remains about +0.04705 / +0.04701 / +0.04717. The latter does not resolve the finite-move instability. Pressure baselines are about 0.08245, but this family has no allowable pressure limit; do not import the M3/0001 temporary screen.

Full per-role counts and metrics, attempt provenance, and raw results are in [the M0 baseline report](20260926_195912Z_baseline_decision.md) and ignored `diagnostics/generated/baseline_decision_m0_20260926/mixed_floor_m7_0340/`.

## Attempt and resource ledger

All attempt counts below include recovered physical solves exactly once. No solve was retried.

| Phase | Attempts charged | Converged | Measured process CPU | Measured runner wall |
|---|---:|---:|---:|---:|
| M7/0340 floor pilot | 14 | 14 | 136.853 s across 13 calls | 136.877 s across resumed/recorded work |
| M3/0001 fine pilot | 9 | 9 | 74.552510 s | 74.571935 s |
| M3/0001 coarse panel | 52 | 52 | 142.753216 s for phase | 148.809972 s for phase, including the recovered-smoke outer command |
| M3/0304 fine pilot | 9 | 9 | 96.532221 s | 96.548110 s |
| **Total** | **84** | **84** | **450.690947 s measured** | **456.807017 s measured** |

Two completed calls have missing per-call timing and remain separate reservations: the initial M7/0340 coarse/tight baseline and the M3/0001 coarse smoke whose solver finished before a post-solve serializer error. The latter was recovered from its preserved raw directory without another solver call. Reserve 600 CPU seconds and 600 adapter-wall seconds for each unknown call; do not add either reserve to the measured totals. The smoke command’s 5.945-second outer runner wall is already included in its phase’s measured wall total.

The cap for this coverage sequence is 5,400 process-CPU seconds and 5,400 runner-wall seconds. After the 0304 pilot, measured totals plus the two separate reserves leave 3,749.309 CPU seconds and 3,743.193 wall seconds. These are remaining accounting capacity, not authorization to launch more solves. The 0304 gate failed, and all further reference solving is held.

The frozen proposal contains 129 new-solve rows total, including the already completed nine-state 0001 fine pilot. Completed proposal phases are 0001 coarse 52, 0001 fine 9, and 0304 fine 9 (70 proposal rows); the separate M7/0340 floor pilot adds 14 attempts, yielding 84 charged attempts. The explicitly unspent frozen rows are 0304 coarse 34, M7/0333 nominal-grid 16, and M7/0333 spectator-shift nominal-grid 9. The nine reused 0304 coarse states are reuse records, not new attempts. No further family was sampled.

Counting all 129 manifest rows plus the separate 14-call M7 pilot gives 143 total attempts, leaving 17 under the 160-attempt coverage ceiling. A prior 152-attempt working estimate double-counted the nine 0001 fine-pilot rows, which are already included in the manifest’s 129. The 17-attempt slack remains unspent.

All interaction/floor pilots used the local NumPy analytic-wake/shared-grid reference solver. `CUDA_VISIBLE_DEVICES=2` was set for scoped visibility, but these solves used CPU and recorded `gpu_used=false`; no GPU training or compute is attributed to these panels. Raw outputs and one-time harnesses remain under ignored `diagnostics/generated/` paths. The 0304 run manifest and full metrics are `diagnostics/generated/baseline_decision_m0_20260926/reference_coverage_m3_m7_20260926/pilot_m3_0304_fine/run_manifest.json` and `pilot_metrics.json`.

## Readiness conclusion

M3/0001 shows a local paired thermal response on receivers 0 and 1 but a null interaction at the hot receiver that controls the current sampled-peak objective. M3/0304 does not resolve its mixed signal above its family-specific mesh discrepancy. M7/0340 shows that an apparently stable anchored peak interaction can coexist with sign reversals in peak finite moves. Pressure feasibility remains uncalibrated because the train-split M3 families and the calibration-development M7 family supply no engineering allowable. Keep Run 1804 e4738 as a provisional refit incumbent, keep `B_response` unassigned, and do not make graph-guided or inverse-design claims from these records.
