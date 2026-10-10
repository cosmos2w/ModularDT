# Completed joint formal models: fidelity, cost, organization, and response diagnosis

The new models replace the earlier refinement organization with a spatially registered source/environment → hyperedge → node core and receiver access to those edges. Thermal 3905/3906 also replace the frozen flow partner with fresh flow and temperature heads sharing that core, preserving heat-independent flow and affine heating; Wind uses the same core family with a nonlinear velocity readout. The completed 5000-epoch results show a substantial regression in Thermal flow and Wind wake fidelity. Thermal 3906 has a useful improvement in average fluid temperature, but worse temperature tails and peak errors; it is not an overall replacement for the mature references.

This is a separate comparison of Thermal 3905 J-geometry / 3906 J-H and Wind 2203 J-geometry / 2204 J-H against the classic models and previous Thermal 3903 / Wind 2202. The [joint development goal report](HONF_Joint_Field_Regional_Development_20261009.md) is preserved unchanged. Results below distinguish physical predictions, measured execution cost, organization, and the limited inverse-use evidence. No model training, physical solves, or inverse search was performed for this report; WindTEST remained locked.

## 1. What can be compared fairly

**Checkpoint identity.** The primary comparison uses the literal epoch-5000 checkpoints of all four new runs and 3903/2202. Thermal classic references are 1404, 1501, and Dense 1804 at epoch 5000. Wind classic references are the retained E512 K6 2102 checkpoint at epoch 2440 and Dense 2103 checkpoint at epoch 2475, selected during their mature 2500-epoch runs. Their physical predictions can be compared on the same panel; they are not a controlled equal-training-age architecture ablation. New best-field aliases are secondary checks, not replacements for the requested endpoints.

| Formal family | Training / exposed validation | Shared new-run settings | Completed budget per new model |
|---|---|---|---|
| Thermal 3905 / 3906 | Original TRAIN600 / canonical89 | Seed 0; width/message 128; two core blocks; 16 regional anchors; 192 environment records; Q1024 fluid queries; effective batch 48 | 3,000,000 case visits; 65,000 optimizer updates; 5000 full-TRAIN epochs |
| Wind 2203 / 2204 | Original TRAIN420 / fullVALID90 | Seed 42; width/message 128; two core blocks; 32 regional anchors; E64 environment records; Q4096 total over five roles; effective batch 24 | 2,100,000 row visits; 90,000 optimizer updates; 5000 full-TRAIN epochs |

Within each new J-geometry/J-H pair, dataset membership, TRAIN-fitted normalization, losses, sampling budgets, and schedule match: AdamW, initial LR 3×10⁻⁴ held through epoch 2000, cosine decay to 3×10⁻⁶ at epoch 5000, weight decay 10⁻⁵, and gradient clipping 1. J-H adds parameters and changes the fresh initialization; this is one seed per architecture, not repeated-run statistical evidence. Both formal J-H models use locality-prior strength **0**. Thermal uses response/operator coefficients **0.1/0.1**, with no saved auxiliary-calibration receipt. The preferred Thermal development child in the goal report had locality strength 1 and calibrated auxiliaries on the segmented development protocol. Its results cannot be transferred to formal 3906; the formal run did not reproduce that preferred recipe.

**Physical evaluation.** Thermal uses the same canonical89 cases and native protocol as the saved classic/3903 evaluation: 8192 fluid-grid coordinates with solid interiors excluded, 64 interface angles per active module, and all 3096 material points per module. This covers 522 modules, 33,408 interface points, and 1,616,112 material points. References are the stored analytic-wake/shared-grid packed-H5 fields, not independent CFD. Canonical89 excludes the original TRAIN duplicate 0273 and was exposed during model selection. Thermal errors retain each channel's stored native units; temperature, pressure, vorticity, and flux-proxy errors must not be added as one physical quantity.

Wind uses the exact frozen Q8192 role-balanced panel across fullVALID90, with the native cell/coordinate/target binding rechecked. Roles are near turbine, downstream envelope, hub slab, background, and volume. They overlap, and the balanced mixture is not an unbiased whole-volume estimate. Wind components are physical m/s. The 90 rows include correlated directions from the same layouts; row quantiles describe this exposed panel, not independent-test confidence intervals. Tables average per-case or per-row RMSE after computing each RMSE; pooled-query RMSE is kept separately in the evidence files.

**A loss-display mismatch exaggerates the apparent Wind gap.** Wind 2202 trains with component-specific role scales but its saved validation display/selector uses one scalar per role. The new models use component-specific scales in both phases. Reconstructing all three validation curves from saved per-row component errors and the identical TRAIN component scales gives endpoint objectives 0.008242 / 0.024830 / 0.020680 for 2202 / 2203 / 2204: approximately 3.01× and 2.51× the prior model, not the 8–10× ratio implied by unlike displayed scores. Thermal's old selector measures temperature, whereas the new selector also includes fresh flow, so its total scores are likewise unsuitable for a cross-family headline. Physical errors below avoid both problems.

![Completed training histories with matched physical monitors and corrected Wind validation scaling](../../diagnostics/generated/joint_formal_comparison_20261010/figures/learning_and_generalization.png)

**Figure 1.** All histories reach epoch 5000. The upper panels show physical monitoring errors on the fixed Q1024 Thermal and Q4096 Wind review panels; the full native evaluation below uses Q8192 and must not be numerically mixed with those monitors. Lower panels exclude auxiliary losses; TRAIN is a 25-epoch mean over changing model weights and equally weighted optimizer updates, while VALID uses fixed review weights and equal cases/rows. The gap is descriptive rather than an exactly matched population estimator. Wind TRAIN objectives keep falling while physical wake validation plateaus and deteriorates; correcting the score convention does not remove that regression.

## 2. Predictor: physical field fidelity

### Thermal: weak new flow, mixed temperature gains

| Model, endpoint e5000 | u RMSE | v RMSE | p RMSE | ω RMSE |
|---|---:|---:|---:|---:|
| Classic 1404 | 0.01137 | 0.000509 | 0.003426 | 0.02578 |
| Classic 1501 | 0.00655 | 0.000445 | 0.002570 | 0.02660 |
| Dense 1804 | 0.00650 | 0.000332 | 0.002249 | 0.02443 |
| Prior 3903, frozen flow partner | 0.00433 | 0.000459 | 0.001975 | 0.03756 |
| 3905 J-geometry, fresh joint flow | 0.01300 | 0.001553 | 0.005909 | 0.08703 |
| 3906 J-H, fresh joint flow | 0.01444 | 0.001883 | 0.006436 | 0.08154 |

These are equal-case native fluid errors, with the same valid masks for all models. Relative to 3903, 3905 has roughly 3.00× u, 3.38× v, 2.99× p, and 2.32× ω error; 3906 has 3.34×, 4.10×, 3.26×, and 2.17×. Both new models lose on u and v in **all 89 cases**. Their v, pressure, and vorticity are also substantially worse than the classic references. Removing the frozen dependency achieved the intended model identity, but did not retain its physical fidelity.

A bounded CPU consistency check on saved fields for 0291/0653/0687 finds RMS of ω−(∂xv−∂yu) of 0.0247–0.0322 for 3903, 0.0549–0.1002 for 3905, and 0.0680–0.0940 for 3906. Centered native stencils require a valid fluid center and four valid neighbors; the stored reference's mismatch is only about 1.4–1.6×10⁻⁷. This supports an output-consistency revision, without proving that it caused the validation regression. The analytic reference itself has nonzero divergence RMS of 0.266–0.315 on these stencils, so a zero-divergence penalty would change this benchmark's target assumptions.

| Model, endpoint e5000 | Fluid T mean / p90 | Surface T mean | Material T mean | Module peak mean | Normal-flux proxy mean |
|---|---:|---:|---:|---:|---:|
| Classic 1404 | 0.2933 / 0.4519 | 0.5111 | 0.4094 | 0.4187 | 2.1203 |
| Classic 1501 | 0.2549 / 0.4261 | 0.4695 | 0.3604 | 0.3645 | 1.4482 |
| Dense 1804 | 0.2374 / 0.3891 | 0.4295 | 0.3293 | 0.3176 | 1.5338 |
| Prior 3903 | 0.3211 / 0.4339 | 0.3067 | 0.3017 | 0.3181 | 1.4557 |
| 3905 J-geometry | 0.3405 / 0.6557 | 0.3788 | 0.3649 | 0.4084 | 1.2633 |
| 3906 J-H | 0.2862 / 0.5308 | 0.3082 | 0.3121 | 0.3564 | 1.2932 |

The table reports equal-case RMSE; “module peak” first takes the maximum predicted and reference temperature on each module's full material grid, then measures their errors within each case. Flux is a native dataset proxy, not a measured CFD wall flux. 3905 loses to 3903 on all mean temperature roles while improving mean flux-proxy error. 3906 improves mean fluid T by **10.9%**, wins on that role in **74/89 cases**, and improves mean flux-proxy error by 11.2%. It nevertheless loses to Dense 1804 on mean fluid T by **20.6%**; 3905 loses by 43.4%.

Robustness is the main temperature miss: 3906 fluid-T p90 rises by 22.3% and its worst-case RMSE rises from 0.6122 to 0.9245. Module-peak p90 rises from 0.7497 to 1.1739 and worst-case peak RMSE from 1.2808 to 2.3508; 0297/M7 is the new worst peak case. Its mean material error rises 3.5%, despite improving 63/89 cases. Thus a majority of improved cases can coexist with worse average material error and severe tails. The best-field aliases of both new Thermal runs are at epoch 4300; fluid-T means 0.3413 / 0.2875 remain close to their endpoints and do not repair flow or tail fidelity.

![Thermal native fluid-temperature predictions and signed errors with per-panel relative L2 error](../../diagnostics/generated/joint_formal_comparison_20261010/thermal/figures/thermal_native_temperature_0291_0687.png)

**Figure 2.** Fixed input-selected cases 0291/M5 and 0687/M10, native 64×128 grid; reference, Dense 1804, 3903, and the new e5000 models. Fields and signed residuals use common scales across models/cases, with solid interiors masked. Every residual panel marks **relative L2 = 100‖prediction−reference‖₂/‖reference‖₂** on all valid fluid cells; display clipping does not affect the metric. Similar-looking global plumes conceal spatially coherent wake errors. These two examples illustrate structure rather than estimate the 89-case mean or its worst tail.

![Thermal native vorticity predictions and signed errors with per-panel relative L2 error](../../diagnostics/generated/joint_formal_comparison_20261010/thermal/figures/thermal_native_omega_0291_0687.png)

**Figure 3.** The same geometries, checkpoint ages, masks, and L2 definition as Figure 2, now for ω in stored native units. The new models show broader near-body and downstream residual structure on a common symmetric error scale. The canonical89 mean ω errors are 0.02443 for Dense, 0.03756 for 3903, and 0.08703 / 0.08154 for 3905 / 3906. Accurate plume temperature therefore does not establish accurate joint flow.

### Wind: regression across wake roles and transverse components

| Model / retained checkpoint | Near vector mean / p95 | Downstream vector mean | Hub vector mean | Near Uy RMSE | Near Uz RMSE |
|---|---:|---:|---:|---:|---:|
| Classic K6 2102, e2440 | 0.09550 / 0.13617 | 0.06378 | 0.05209 | 0.01669 | 0.01118 |
| Dense 2103, e2475 | 0.07445 / 0.09075 | 0.03910 | 0.03297 | 0.01909 | 0.01557 |
| Prior 2202, e5000 | 0.09295 / 0.11145 | 0.05414 | 0.04399 | 0.02397 | 0.01833 |
| 2203 J-geometry, e5000 | 0.17097 / 0.26992 | 0.10991 | 0.09232 | 0.03951 | 0.02864 |
| 2204 J-H, e5000 | 0.15916 / 0.25088 | 0.09968 | 0.08256 | 0.03519 | 0.02453 |

All errors are equal-row physical RMSE in m/s; vector RMSE is √mean(ΔUx²+ΔUy²+ΔUz²), with no division by three. Relative to 2202, near-turbine errors increase **83.9% / 71.2%** and downstream errors **103.0% / 84.1%** for J-geometry / J-H. The regression is broad: 2203 loses near-turbine fidelity in 89/90 rows and 2204 in 90/90; both lose downstream fidelity in every row. Near Ux also worsens from 0.08782 to 0.16366 / 0.15318. Transverse errors worsen independently, so a large background Ux cannot stand in for wake fidelity.

Both new endpoints lose to 2202 and Dense in all five vector roles. Their volume errors are 0.04482 / 0.04015 versus 2202's 0.02310; background errors are 0.01707 / 0.01647 versus 0.01114. They improve background relative to classic K6, while losing its other four vector roles. Dense is the strongest listed vector predictor, but K6 has lower near Uy/Uz error than Dense; that component tradeoff should be preserved.

J-H partially recovers J-geometry fidelity, reducing endpoint near error by 6.9% and downstream error by 9.3%, but remains far behind 2202. Selected e1400/e2000 checkpoints have near errors 0.16314 / 0.14483 and downstream errors 0.10483 / 0.09081. These aliases were selected on the same exposed VALID using the new component-scaled objective; 2202 used its old scalar-scaled selector. The primary comparison remains literal e5000. Earlier stopping helps, particularly J-H tails, without making either model competitive. The corrected common monitoring objective deteriorates by 26.4% / 19.6% from each new model's best review to epoch 5000.

![Wind native Ux and Uy fields and signed residuals with per-panel relative L2](../../diagnostics/generated/joint_formal_comparison_20261010/wind/wind_native_hub_field_residuals_row426_UxUy.png)

**Figure 4.** Native hub-height Ux and Uy fields and signed residuals on the fixed row426/layout142/M30/WD270 geometry, z/D≈0.886. All methods use the exact native plane cells and common field/error scales per component. Each error panel reports the component's relative L2 on the full plane using the unclipped reference and prediction; this whole-plane metric is distinct from the near-turbine role RMSE in the table. Uy exposes structure that a Ux-only image hides: whole-plane relative L2 is 8.39% for Dense, 15.74% for 2202, 18.92% for 2203, and 21.51% for 2204. J-H therefore remains worse on this example's Uy despite its better 90-row role means. Native references are the stored Wind validation simulation fields; no new CFD was run.

![Wind native Ux Uy Uz centerline comparisons for fixed row426](../../diagnostics/generated/joint_formal_comparison_20261010/wind/wind_native_wake_centerline_row426_UxUyUz.png)

**Figure 5.** Ux/Uy/Uz cuts through the nearest native y row to the input-selected upstream-most turbine, using exact native x cells without interpolation. The large-scale Ux recovery can look plausible while local wake and transverse structure differ. A single fixed cut is illustrative; the fullVALID90 role errors establish the population of observed failures.

## 3. Computation: savings must be measured alongside lost fidelity

| Run | Recorded TRAIN time, h | Complete engine time, h | Median TRAIN epoch, last 1000, s |
|---|---:|---:|---:|
| Thermal 3903 | 6.779 | Not recoverable cumulatively | 4.001 |
| Thermal 3905 | 3.975 | 4.021 | 2.862 |
| Thermal 3906 | 5.081 | 5.128 | 3.658 |
| Wind 2202 | 7.396 | 7.553 | 5.017 |
| Wind 2203 | 5.835 | 5.994 | 3.795 |
| Wind 2204 | 7.345 | 7.518 | 4.887 |

Times are recorded execution, not FLOP estimates. Engine time includes its reviews/checkpoint work but excludes preparation before the engine. Thermal 3903's engine summary covers only its last resumed invocation; its TRAIN time above is reconstructed over the full 5000-row history and excludes the separate frozen-flow training. Although Thermal case/update budgets match, 3903 accumulated 211,016 microbatches versus 65,000 for each new run. These training-clock differences are therefore deployment-history observations, not a controlled architectural speedup. Wind first epochs included large catalogue preparation costs; the late-epoch statistic avoids conflating that startup with steady training.

The new J-H engine costs **27.5% more than J-geometry in Thermal and 25.4% more in Wind**. This buys partial temperature/wake recovery while leaving severe physical misses. Wind's persistent native-catalogue implementation verifies geometry, role recipe, dtype/shape, and array checksums and supports read-only reuse of coordinates, role indices, quadrature CDFs, and support volumes. It stores no targets, predictions, learned model weights, or sampled query draws. The preparation snapshots show zero entries/hits at capture and the fit metadata records capacity rather than final hit statistics; reuse is an implemented capability, not a measured cache-hit rate or cold-to-warm speedup in these formal runs. It concerns setup/I/O rather than physical accuracy.

| Native deployed model | Parameters, million | Median ms (observed min–max) | Peak CUDA allocated, total / additional MiB |
|---|---:|---:|---:|
| Thermal classic 1404 | 3.509 | 54.58 (54.43–56.12) | 77.32 / 55.79 |
| Thermal classic 1501 | 3.987 | 228.61 (227.97–232.18) | 81.54 / 58.18 |
| Thermal Dense 1804 | 5.431 | 168.39 (168.29–168.70) | 85.68 / 56.83 |
| Thermal 3903 + frozen 3901 | 0.441 | 42.07 (41.53–42.24) | 29.57 / 19.75 |
| Thermal 3905 J-geometry | 0.971 | 42.75 (42.43–96.38) | 46.40 / 34.57 |
| Thermal 3906 J-H | 1.065 | 55.61 (55.49–55.90) | 47.49 / 35.30 |
| Wind K6 2102 | 1.696 | 71.09 (70.44–71.95) | 73.12 / 58.18 |
| Wind Dense 2103 | 3.629 | 44.52 (43.72–44.63) | 319.08 / 296.77 |
| Wind 2202 | 0.675 | 14.13 (14.01–20.71) | 55.10 / 44.40 |
| Wind 2203 J-geometry | 0.962 | 22.38 (22.36–22.48) | 68.96 / 57.06 |
| Wind 2204 J-H | 1.061 | 29.39 (29.29–29.50) | 70.33 / 58.05 |

Fresh GPU1 RTX 6000 Ada measurements use two warmups and five synchronized wall-time calls per model, with context rebuilt, full physical output conversion, and CPU result copy on every call. Thermal uses 0687/M10 with Q8192 plus interface/material/port extraction; Wind uses row426/M30 and the frozen Q8192 role panel. Checkpoint/data loading and target metrics are excluded. Classic Thermal follows its retained predicted-port/local-surrogate path; 3903 includes its exact frozen 3901 flow weights. This measures native deployment paths, not equal-function executor backends. CUDA peaks are PyTorch allocated memory, including resident tensors; “additional” subtracts the pre-call baseline. They are not process-wide GPU memory, cache usage, or CPU RAM. All five samples, including outliers, are retained.

The new Wind calls are **58.4% / 108.1% slower than 2202** while less accurate. Thermal 3905 has a similar median to 3903; 3906 is 32.2% slower. J-H adds 30.1% / 31.3% over J-geometry in Thermal / Wind. The new models are smaller and often faster than the classic deployments, but those savings coexist with the fidelity losses above; Wind Dense trades greater memory/time for substantially stronger vector accuracy. This one-input benchmark does not establish scaling across module counts or a whole-dataset throughput ratio.

![Matched fixed-input fidelity versus complete native inference time](../../diagnostics/generated/joint_formal_comparison_20261010/figures/native_accuracy_cost.png)

**Figure 6.** Physical error and cost on the identical input in each domain: Thermal0687 fluid T/u and Wind426 near-turbine vector velocity. Points use the measured median; horizontal spans preserve the observed five-call min–max, not confidence intervals. Thermal time axes are logarithmic. Thermal3906 improves this scene’s fluid T at the cost of u fidelity and latency; Wind2204 is slower and has worse near error than 2203 on this scene despite improving the 90-row mean. These fixed scenes illustrate accuracy–cost choices; the full validation tables establish broader fidelity.

The new graphs execute dense donor/edge and receiver/edge work and retain individual source reads. Small displayed access weights are not skipped execution. At Thermal0687, 26 active edges sit in a 28-edge allocation with 10 active sources in 12 slots; Wind426 has 62 edges and 30 sources. Across two blocks, source/environment aggregation has 520/9984 active Thermal pairs and 3720/7936 Wind pairs; Thermal allocated capacities are 672/10752. Those counts describe work shapes, not physical support quality, FLOPs, or measured sparse savings. No new sparse executor benefit was demonstrated.

## 4. Organizer: real organization and use, limited physical meaning

The checkpoints contain two actual typed donor→edge→node blocks and spatially registered source/regional anchors. J-geometry uses fixed Gaussian memberships/access; J-H learns them. Both preserve physical source identities, read every source at receivers, and retain a pooled scene-context bypass. Heat does not enter the Thermal organization or flow representation. This is a stronger organization contract than detail-only refinement gating, but a recognizable graph does not by itself preserve physics or explain which source caused a wake.

![Actual saved typed memberships and receiver access for new formal endpoints](../../diagnostics/generated/joint_formal_comparison_20261010/organization_cost/actual_organization_graph_chain.png)

**Figure 7.** Exact e5000 source/environment memberships and three receiver-access rows, on input-sealed Thermal0687/M10 and Wind426/M30. Source IDs and source/regional edge identities are retained; masks remove only absent slots in the display. Weights share a 0–1 scale and are not thresholded causal supports. J-H has positive access at all 78/186 active probe–edge pairs, so visible concentration does not demonstrate sparse execution. These formal strength-0 weights have no inherited locality result from the preferred development child.

![Spatially registered J-H edges and saved receiver access in native coordinates](../../diagnostics/generated/joint_formal_comparison_20261010/organization_cost/registered_spatial_access.png)

**Figure 8.** The same J-H inputs plotted in native Thermal coordinates and Wind x–y/D. Line width follows saved receiver-access weight; only the five largest links per probe are drawn for readability, with all edges still executed. The Wind graph uses 3D access although this map projects x–y. Near-probe entropy-effective edge counts are 4.77/26 for Thermal and 2.17/62 for Wind, versus J-geometry's 12.70/26 and 22.57/62. This establishes checkpoint-specific selectivity at those probes, not a global small-K policy. Wind background access spans a weighted registered-anchor distance of 21.86D, illustrating why registration alone does not guarantee local physical interpretation.

![Compact same-checkpoint organization intervention effects on valid physical field RMSE](../../diagnostics/generated/joint_formal_comparison_20261010/organization_cost/intervention_compact.png)

**Figure 9.** Counterfactual/ordinary RMSE ratios at the same e5000 weights: removal of collective edge content/update, or replacement of learned J-H membership and receiver access with fixed geometry. Thermal uses only the sealed 7463 valid fluid cells of 0687; Wind uses the frozen row426 Q8192 role panel, with vector errors over all three components. This is one geometry per dataset. Both interventions retain pooled context and individual source reads. J-geometry→geometry is an exact identity; restoring ordinary context gives bit-exact original outputs in every model. The detailed component changes are retained in the numerical receipts.

Every listed channel/role worsens under these interventions on the sealed inputs. For example, Thermal3906 fluid-T RMSE rises from 0.2604 to 2.2082 after geometry replacement and to 2.4240 after collective removal. The collective path is therefore used even with the source/global bypass intact. These severe same-weight changes can also reflect taking a trained representation outside its training distribution: they do not prove that learned organization is better than a separately trained geometric model, that each learned edge has a physical cause, or that the bad formal generalization is caused by graph collapse. Physical organizer quality remains limited by the predictor errors and untested geometry responses.

## 5. Inverse use and physical response: bounded saved-pool evidence

The response comparison reuses only three previously counted Thermal geometries, 0291/0294/0687, each with baseline and two signed constant-total heating transfers. Each new endpoint evaluates nine saved states and six prepared signed increments, with no optimizer update, inverse generator, placement search, or solve. References are the existing analytic-wake/shared-grid records. Flow remains exactly unchanged under these heat changes; that confirms the heat-independent contract while saying nothing about its absolute flow accuracy. Affine temperature increments are evaluated at fixed geometry, and maximum-temperature changes are computed from the complete endpoint material fields, because taking a maximum is not an affine operation.

| e5000 model | Fluid ΔT field RMSE | Material ΔT field RMSE | Maximum-T change RMSE | Correct saved-pool choice |
|---|---:|---:|---:|---:|
| Classic 1404 | 0.02051 | 0.03574 | 0.09310 | 3/3 |
| Classic 1501 | 0.01093 | 0.01659 | 0.04209 | 3/3 |
| Dense 1804 | 0.01056 | 0.01879 | 0.02742 | 3/3 |
| Prior 3903 | 0.00764 | 0.01402 | 0.18932 | 2/3 |
| 3905 J-geometry | 0.01076 | 0.01611 | 0.28898 | 2/3 |
| 3906 J-H | 0.00970 | 0.01466 | 0.14261 | 3/3 |

Field-response RMSE is the arithmetic mean over six signed-state RMSEs on the same native queries; maximum-T change RMSE is over six endpoint functional differences. All temperature quantities use stored native units. New fluid ΔT errors rise 40.7% / 26.9% over 3903, while material ΔT errors rise 14.9% / 4.6%. J-H improves the maximum-T change over 3903 and chooses the reference-best state in all three pools, but classic Dense remains much stronger on the maximum-T increment. 3903 and 3905 choose baseline instead of transfer-plus for 0291, incurring reference regret 0.39035; J-H has zero regret within these three-state pools. Every model remains materially better than a zero-temperature-response control on fluid/material increments.

![Saved signed Thermal responses and finite candidate-pool peak decisions](../../diagnostics/generated/joint_formal_comparison_20261010/thermal/figures/thermal_saved_heating_response_and_peak_pool.png)

**Figure 10.** Left: signed global-peak changes in 0291, where 3903 and 3905 miss the sign of transfer-plus. Center: transfer-plus responses of three modules whose own heating is unchanged, exposing collective thermal response. Right: minimum-peak choices in the three existing baseline/minus/plus pools; B denotes baseline and + denotes transfer-plus, with reference regret 0.39035 in the two red cells. All plotted predictions use the retained e5000 endpoints. This is finite-pool rescoring against stored reference maximum material temperatures, not an inverse search success rate or independent physical certification. No geometry perturbation or Wind inverse/placement response was measured, and organizer interventions do not substitute for such physical response evidence.

## 6. Diagnosis and next-round revision priorities

| Measured finding | Supported diagnosis | Next controlled check |
|---|---|---|
| New Wind TRAIN error improves while wake validation is much worse; earlier checkpoints help only partly | A generalization/representation problem remains after fixing the validation display convention; numerical divergence is not supported by the finite histories | Freeze a common component-scaled monitor and physical role metrics; compare source/local decoder capacity and context resolution on identical fixed development IDs, queries, and update budgets |
| Fresh Thermal flow loses u/v in every case; saved ω/curl inconsistency is larger | Shared latent features do not enforce consistency among physical output channels | Match the reference's native derivative convention and ω/curl relation, then check temperature advection using predicted flow; preserve one shared core and heat-independent flow while improving its output constraint/local decoder |
| Formal J-H uses strength 0 and Thermal auxiliaries have no calibration receipt, unlike the preferred development child | The formal candidate changed several consequential conditions; development preference was not directly tested | Reproduce the intended prior/calibration recipe as a named development control before attributing the miss to learned organization; do not infer it is the sole cause because J-geometry also regresses |
| Collective removal changes predictions; learned access is concentrated at some probes but broad elsewhere | The collective path is used, with usefulness depending on checkpoint and physical role | Test the same core with a calibrated locality prior and a matched-capacity collective control; judge wake/temperature tails and same-weight role errors, not heatmap appearance or forced small K |
| J-H improves mean Thermal T and a tiny saved-pool ranking, while peak tails and field increments still miss mature references | Average fidelity, tail robustness, and inverse objectives can disagree | Retain fluid-T, per-module peak tails, and saved signed responses as separate acceptance checks; use existing references within the solver boundary |

These are prioritized hypotheses for controlled revision, not causal conclusions from a single training pair. The Thermal operator auxiliary uses stored TRAIN velocity coefficients; sharing the representation does not enforce predicted-flow advection, continuity, or ω/curl consistency. Wind also changes context granularity: classic E512, prior E8, new E64. The present comparison cannot isolate context resolution, normalization/readout choice, capacity, locality, and objective balance from one another.

For predictor use, retain mature references: Dense 1804 for the strongest listed average fluid T, 3903 for its frozen-partner u/pressure and temperature/response strengths, and Dense 2103 for Wind vector fidelity, with K6's transverse tradeoff explicit. For continued joint development, 3906/2204 provide partial recovery over their geometric siblings and useful structure to diagnose; their current accuracy–cost results do not justify replacing the references. For organizer use, keep the actual registered graph/export/intervention machinery but require physical role and tail improvements before calling it better. For inverse use, preserve the saved-pool evidence and exact Thermal increment machinery, while leaving new geometry/design validity unclaimed.

The next round should start on the fixed stratified quarter-data development protocol, with 100-epoch reviews and a common saved selector, and extend only on physical evidence within an explicit budget. No formal retraining or inverse campaign is launched by this report.

## 7. Evidence and compact figure index

| Figure | Main evidence | Retained PDF master |
|---|---|---|
| 1 | Completed histories and common Wind monitor | [Histories](../../diagnostics/generated/joint_formal_comparison_20261010/figures/learning_and_generalization.pdf) |
| 2–3 | Native Thermal fields/residuals, annotated L2 | [Fluid T](../../diagnostics/generated/joint_formal_comparison_20261010/thermal/figures/thermal_native_temperature_0291_0687.pdf), [ω](../../diagnostics/generated/joint_formal_comparison_20261010/thermal/figures/thermal_native_omega_0291_0687.pdf) |
| 4–5 | Native Wind fields/residuals and wake cuts | [Ux/Uy; row69 on second PDF page](../../diagnostics/generated/joint_formal_comparison_20261010/wind/wind_native_hub_field_residuals.pdf), [Ux/Uy/Uz cuts](../../diagnostics/generated/joint_formal_comparison_20261010/wind/wind_native_wake_centerline_row426_UxUyUz.pdf) |
| 6 | Same-input measured native accuracy–cost | [Cost](../../diagnostics/generated/joint_formal_comparison_20261010/figures/native_accuracy_cost.pdf) |
| 7–8 | Actual typed graphs and registered receiver access | [Graph matrices](../../diagnostics/generated/joint_formal_comparison_20261010/organization_cost/actual_organization_graph_chain.pdf), [Spatial map](../../diagnostics/generated/joint_formal_comparison_20261010/organization_cost/registered_spatial_access.pdf) |
| 9 | Same-weight collective/geometry interventions | [Interventions](../../diagnostics/generated/joint_formal_comparison_20261010/organization_cost/intervention_compact.pdf) |
| 10 | Signed thermal responses and finite-pool choices | [Responses](../../diagnostics/generated/joint_formal_comparison_20261010/thermal/figures/thermal_saved_heating_response_and_peak_pool.pdf) |

The [local evidence directory](../../diagnostics/generated/joint_formal_comparison_20261010/) retains native arrays, checkpoints' hashes, full per-case/per-row metrics, corrected monitoring scores, sealed organizer/intervention inputs, and numerical cost receipts. Generated figures and one-time evaluation/rendering scripts stay ignored and local; only this durable report is versioned. Core physical evidence is in the [Thermal native comparison](../../diagnostics/generated/joint_formal_comparison_20261010/thermal/native/results/native_comparison_metrics.json), [Wind physical comparison](../../diagnostics/generated/joint_formal_comparison_20261010/wind/wind_formal_physical_comparison.json), [common Wind monitor reconstruction](../../diagnostics/generated/joint_formal_comparison_20261010/wind_common_monitor_scores.json), [training accounting](../../diagnostics/generated/joint_formal_comparison_20261010/training_accounting_compact.json), [organizer receipts](../../diagnostics/generated/joint_formal_comparison_20261010/organization_cost/findings.json), [intervention receipts](../../diagnostics/generated/joint_formal_comparison_20261010/organization_cost/intervention_findings.json), and [saved response comparison](../../diagnostics/generated/joint_formal_comparison_20261010/thermal/figures/thermal_response_comparison_findings.json).

Bindings: Thermal canonical89 SHA256 `51f0bea2278c26b24a994a29a52ad6b8af1e3e85941dc75d9b20bbebce5d8728`; Wind frozen panel `188e1d2cf0eace3cacb6c6dfd721829bbe8dea0478a96d041b2ee29291f3ae74`, validation-row fingerprint `ed7295dd2650687c599c13e29a6bae3492508ec8346517e39f4cacf60d0eff72`. Full run identities and checkpoint SHA256 hashes are in the [native cost receipt](../../diagnostics/generated/joint_formal_comparison_20261010/organization_cost/native_cost_gpu1.json); the [saved-field stencil receipt](../../diagnostics/generated/joint_formal_comparison_20261010/thermal/fixed_panel_finite_difference_diagnostics.json) binds the bounded ω/curl check. The [post-analysis protection receipt](../../diagnostics/generated/joint_formal_comparison_20261010/protected_after.json) verifies all 647 protected bindings and the original goal-report hash unchanged. Solver usage remains 326/326; this report used zero new solves and no WindTEST targets.
