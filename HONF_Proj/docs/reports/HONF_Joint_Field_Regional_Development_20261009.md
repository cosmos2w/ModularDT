# Joint field and regional hypergraph development — 9 October 2026

The new family trains its own Thermal flow and temperature heads on one shared source/environment/collective representation. It removes the frozen learned-flow dependency, replaces detail-only organization with two executed node→edge→node blocks and spatial receiver access, and runs the same core implementation in Wind with a nonlinear physical velocity output. Thermal flow remains exactly independent of actual heating inputs, while one source-resolved affine temperature field supplies the native fluid, surface, material and q-proxy roles. Thermal and Wind train separate weights. The selected recipe is learned locality-prior organization for Thermal and geometric regional organization for Wind. Both have completed 2500-epoch fits; the final learned Wind child also reaches 2500 and remains a measured losing hypothesis.

## Gains, misses, measured evidence and next steps

**Predictor.** The preferred Thermal development recipe is J-H with locality strength 1 and full smooth access. At its selected e2500 checkpoint, mean fluid-T RMSE improves 5.52%, vorticity 15.33%, surface T 7.94% and material T 9.94% over the strongest matched geometry control selected at e2400. Mean u and p worsen 14.43% and 18.96%; the M10 fluid-T worst case worsens 13.38% and worst monitored q proxy worsens 14.41%. These are useful difficult-field gains with important local misses, not broad physical qualification. Wind prefers J-geometry selected e2500: the final learned child selected e2400 has near Ux/Uy .359528/.071139 m/s versus .311602/.060174 m/s (+15.38%/+18.22%), and downstream Ux/Uy .230627/.037735 versus .212591/.032557 (+8.48%/+15.90%). Its better Uz in those roles does not remove the wake and transverse misses. The next predictor investment is the remaining velocity/pressure and native heat-flux weakness, using these fixed development identities before any separately authorized formal fit.

**Organizer.** Saved trained states show actual typed M/E memberships, registered anchors, two rounds of collective messages and node updates, and different access for three receivers chosen from inputs. Same-weight membership/access replacements and collective-content removal change native predictions, and exact restoration returns the baseline. Learned routing is consequential but is not physical causality; the source-preserving and pooled-global bypasses remain explicit. Wind learned content is used, but the independent geometric regional model remains more accurate in the important wake roles; learned-incidence benefit over that control is unproved. The next organization decision should use the field and cost evidence rather than a prescribed small edge count.

**Inverse interface.** Heat-null, affine increments, source columns, native extraction, JVP/VJP and stale-state rejection are measured on the selected models. Prepared FP64 heat contraction retains precision, and same-weight FP64 derivative replays qualify arithmetic separately from the retained FP32 finite-difference and permutation failures. Thermal 0291 still selects the wrong hottest physical module; its reference is an already exposed analytic-wake record. Existing geometry-response examples are TRAIN-fitted auxiliary evidence, and Wind has no new finite turbine-displacement/yaw/thrust reference. No inverse campaign or valid-design claim follows. The next inverse step remains physical forward-response qualification.

| Decision | Result |
| --- | --- |
| Joint-model contract | Met: fresh trainable heads and shared blocks, self-contained loading, actual-input dependency laws, source/native identities and stale-state checks. Numerical precision qualifications below remain explicit. |
| Predictor direction | Thermal promising for fluid-T/vorticity with u/p, M10 and q misses; Wind learned-incidence superiority is not supported; retain its stronger geometric regional predictor. |
| Learned collective organization | Consequential in same-weight interventions; Wind learned-versus-geometric advantage remains unproved, despite clear same-weight content dependence. |
| Sparsifiability | Measured at 100/99/95/90% mass with identical-count geometry controls; actual packed value reads are measured, complete inference is slower, and Thermal native q has a strict parity miss. |
| Inverse interface numerics | Usable within the declared smooth/precision or fixed-support scope; no support-switch derivative or blanket FP32 finite-difference pass. |
| Physical inverse qualification | Limited: wrong0291 hot-module ranking, exposed/fitted response records, no new Wind turbine-control reference and no inverse designs. |

## Protocol and actual learning age

Thermal uses the sealed fixed25_v1 manifest: 150 TRAIN and 22 exposed DEV cases, TRAIN-fitted normalization, seed 0, effective/micro batch 48/48, Q1024 fluid queries, E192, H128/message 128, depth 2, tile 512, 16 regional anchors plus every source anchor. Surface monitoring uses 16 angles per module, material monitoring 32 fixed samples, and the stored-velocity operator uses 128 queries. TRAIN response calibration fits only four existing TRAIN families (0001/0318/0333/0348). The objective balances the flow and temperature families equally, with q-proxy coefficient 0.05 and separate TRAIN-calibrated response/operator terms. Selected TRAIN covers M1/2/3/4/5/6/7/9/10/12; DEV has M3/5/7/10 with 6/6/6/4 cases, so no DEV M12 generalization is claimed.

Wind uses wind_shared_fixed24_v1: 72 TRAIN direction rows from 24 layouts and 24 exposed DEV rows from eight layouts, seed 42, TRAIN-fitted transforms/background/component scales, effective/micro batch 24/24, E64, H128/message 128, depth 2, tile 512 and 32 regional anchors plus every physical turbine. Q4096 is the total query count per row: 820 each for four physical roles and 816 background queries. It is not five times 4096. DEV has M8/11/13/18/21/24/27/30, three stored directions 270°/285°/300° per layout. The selected TRAIN omits M18 and is 17.1% of original TRAIN. Correlated directions and overlapping role strata are not independent samples; WindTEST stays locked.

Common compatible linear tensors initialize from the same qualified parameter names and dataset seed. Compared arms reuse case IDs, transforms, query streams, losses, effective batches and schedule: AdamW, clip 1, 20-epoch warmup to 3e-4, hold through 1000, cosine decay to 3e-6 at 2500. Every development epoch visits every selected case. Saved checkpoints and reviews occur every 100 epochs. Direct arms stop at 500 after screening; the relevant geometry and learned candidates mature to 2500. No second seed or expanded portfolio was started.

Revision 1 adds one fixed Gaussian log prior to learned Thermal membership/access scores, branching from the common H e500 state. The child preserves weights, named Adam moments, RNG, streams, objective and absolute schedule; it has 500 inherited plus 2000 revised-function epochs. Revision 2 applies that same single core-law change to Wind after diffuse mature source/environment gathering and poor wakes were observed; it branches from H e1000 and has 1000 inherited plus 1500 revised-function epochs. Its first 100 new epochs miss the control, and that miss is retained. Healthy falling training loss justifies the authorized maturation; neither acceptance bands nor validation membership move. Both revision allowances are used.

This compact JSON is the durable experiment index; common profiles apply to every corresponding row. Actual direct regional edge count is zero; organized models execute every physical source anchor plus the declared regional capacity. Selected native metrics are in the full physical-role tables below, and the [complete eight-run machine-readable receipt](../../diagnostics/generated/joint_regional_20261009/audit/final_experiment_table_v2.json) retains checkpoint/source hashes and every counter.

```json
{
  "schema":"joint_regional_experiment_summary_v1",
  "scope":"single-seed fixed exposed DEV; direct500 is an earlier-age screen",
  "profiles":{
    "Thermal":{"protocol":"fixed25_v1","seed":0,"train_dev_cases":[150,22],"manifest_sha256":"933b0138ba2f8447a1ecadfe31fd0bb2cb4a05607d3ac3d9f0dc79419f196044","effective_micro_cases":[48,48],"Q":1024,"E":192,"hidden_message":[128,128],"depth":2,"regional_anchor_capacity":16},
    "Wind":{"protocol":"wind_shared_fixed24_v1","seed":42,"train_dev_rows":[72,24],"manifest_sha256":"b223e174d7262682a7e1ac1977cb38f721993e14c0340b2dc0f5fb5f95bfa715","effective_micro_cases":[24,24],"Q_total_five_roles":4096,"E":64,"hidden_message":[128,128],"depth":2,"regional_anchor_capacity":32}
  },
  "runs":[
    {"id":"T4001","dataset":"Thermal","model":"J-direct","initialization":"fresh","completed_selected_epoch":[500,500],"completed_selected_updates":[2000,2000],"completed_case_visits":75000,"completed_primary_queries":76800000,"completed_selected_new_function_epoch":[500,500],"selected_active_Adam_age":2000,"charged_owned_training_seconds":521.302,"training_allocated_reserved_GiB":[8.737,10.084],"decision":"Direct e500 lower-age screen; not a mature fair-comparison claim.","selected_checkpoint_sha256":"334452c3ddf0210bcbe8d3f87d093b057f51afc7007d040004f6f53c1b5e5f31","selected_DEV_native_metrics":{"field_score":0.1505131497979164,"fluid_T_mean_p90_native":[0.9880188798362558,1.5257962465286259],"omega_mean_p90_native":[0.6895716298710216,0.7528498053550721]}},
    {"id":"T4002","dataset":"Thermal","model":"J-geometry","initialization":"fresh","completed_selected_epoch":[2500,2400],"completed_selected_updates":[10000,9600],"completed_case_visits":375000,"completed_primary_queries":384000000,"completed_selected_new_function_epoch":[2500,2400],"selected_active_Adam_age":9600,"charged_owned_training_seconds":2290.523,"training_allocated_reserved_GiB":[9.667,10.73],"decision":"Selected geometry comparator; root prefers T4004.","selected_checkpoint_sha256":"92dfb8131dc02406587b4a719387e3cb464bb7b9244a3ada712b79d1cd077992","selected_DEV_native_metrics":{"field_score":0.0318199716,"fluid_T_mean_p90_native":[0.7258719111,1.1587930202],"omega_mean_p90_native":[0.3836056177,0.4368281931]}},
    {"id":"T4003","dataset":"Thermal","model":"J-H-v1 (H0)","initialization":"fresh","completed_selected_epoch":[1000,1000],"completed_selected_updates":[4000,4000],"completed_case_visits":150000,"completed_primary_queries":153600000,"completed_selected_new_function_epoch":[1000,1000],"selected_active_Adam_age":4000,"charged_owned_training_seconds":1378.585,"training_allocated_reserved_GiB":[16.869,18.355],"decision":"Unregularized H0 negative branch baseline, stopped at e1000.","selected_DEV_native_metrics":{"field_score":0.12219138054007833,"fluid_T_mean_p90_native":[0.9064007862047716,1.2951177358627322],"omega_mean_p90_native":[0.6427536755800247,0.7043873965740204]}},
    {"id":"T4004","dataset":"Thermal","model":"J-H locality prior λ=1","initialization":"branch T4003 e500","completed_selected_epoch":[2500,2500],"completed_selected_updates":[10000,10000],"completed_case_visits":375000,"completed_primary_queries":384000000,"completed_selected_new_function_epoch":[2000,2000],"selected_active_Adam_age":10000,"charged_owned_training_seconds":2035.18,"training_allocated_reserved_GiB":[16.869,18.373],"decision":"Preferred Thermal selection; full-native profile is slower and strict interface parity still fails.","selected_checkpoint_sha256":"52ce5831c30b3ad61525509bc56ba667b66b47dcba8f2e548751e271a80c6846","selected_DEV_native_metrics":{"field_score":0.0271367399,"fluid_T_mean_p90_native":[0.6857825585,1.0790271223],"omega_mean_p90_native":[0.3247914368,0.3555148691]}},
    {"id":"W2301","dataset":"Wind","model":"J-direct","initialization":"fresh","completed_selected_epoch":[500,500],"completed_selected_updates":[1500,1500],"completed_case_visits":36000,"completed_primary_queries":147456000,"completed_selected_new_function_epoch":[500,500],"selected_active_Adam_age":1500,"charged_owned_training_seconds":1889.119,"training_allocated_reserved_GiB":[8.876,9.793],"decision":"Direct e500 lower-age screen; not a mature fair-comparison claim.","selected_checkpoint_sha256":"d0af2df919d89fc00be2f6380cb7a2d01af80b227d5c924afbc33a20aa1f3cfd","selected_DEV_native_metrics":{"field_score":0.34655136316265206,"near_turbine_rmse_mps":{"Ux":{"mean":0.6378285462469057,"p95":0.7481607831487878,"worst":0.7796578739278777},"Uy":{"mean":0.16946691771183128,"p95":0.1903597959959755,"worst":0.19295836940713212},"vector":{"mean":0.6721616207843869,"p95":0.7795292542759319,"worst":0.8121110994541263}}}},
    {"id":"W2302","dataset":"Wind","model":"J-geometry","initialization":"fresh","completed_selected_epoch":[2500,2500],"completed_selected_updates":[7500,7500],"completed_case_visits":180000,"completed_primary_queries":737280000,"completed_selected_new_function_epoch":[2500,2500],"selected_active_Adam_age":7500,"charged_owned_training_seconds":3990.614,"training_allocated_reserved_GiB":[9.356,10.303],"decision":"Preferred Wind selection; better aggregate score and near-wake Ux/Uy than W2304.","selected_checkpoint_sha256":"5330f614fec1b93edb54f67fefc41dd384752a58617e574c6d3897333ee3db1b","selected_DEV_native_metrics":{"field_score":0.06563511279718807,"near_turbine_rmse_mps":{"Ux":{"mean":0.3116023096732628,"p95":0.3879147434463033,"worst":0.4172690950060509},"Uy":{"mean":0.060173913728029625,"p95":0.08581577248469384,"worst":0.1122402962661968},"vector":{"mean":0.320861,"p95":0.40943,"worst":0.425554}}}},
    {"id":"W2303","dataset":"Wind","model":"J-H-v1 (H0)","initialization":"fresh","completed_selected_epoch":[2500,2200],"completed_selected_updates":[7500,6600],"completed_case_visits":180000,"completed_primary_queries":737280000,"completed_selected_new_function_epoch":[2500,2200],"selected_active_Adam_age":6600,"charged_owned_training_seconds":4141.037,"training_allocated_reserved_GiB":[16.693,17.637],"decision":"Completed e2500, selected e2200; negative wake-role reference.","selected_checkpoint_sha256":"42747b60e99ad28985efa05ccc6a8928ad9d7d0464430414e8ef8ef2aeb0f7c0","selected_DEV_native_metrics":{"field_score":0.08171054611662038,"near_turbine_rmse_mps":{"Ux":{"mean":0.37196694941436,"p95":0.44358511787512855,"worst":0.44690844896125753},"Uy":{"mean":0.08558626203096635,"p95":0.09993785582862573,"worst":0.10365740262673528},"vector":{"mean":0.384156,"p95":0.456667,"worst":0.459574}}}},
    {"id":"W2304","dataset":"Wind","model":"J-H locality prior λ=1 revision 2","initialization":"branch W2303 e1000","completed_selected_epoch":[2500,2400],"completed_selected_updates":[7500,7200],"completed_case_visits":180000,"completed_primary_queries":737280000,"completed_selected_new_function_epoch":[1500,1400],"selected_active_Adam_age":7200,"charged_owned_training_seconds":2343.12,"training_allocated_reserved_GiB":[16.693,17.637],"decision":"Completed e2500 but loses near-wake Ux/Uy/vector to G; not root-preferred.","selected_checkpoint_sha256":"55800ab1ad2512c88351047e78b7b0753c0cac56476699ef5d5f56e54b00e040","selected_DEV_native_metrics":{"field_score":0.06750357268826225,"near_turbine_rmse_mps":{"Ux":{"mean":0.35952837392480946,"p95":0.45564885284888085,"worst":0.5081093256332668},"Uy":{"mean":0.07113858460594918,"p95":0.08712499289451306,"worst":0.08920758188855205},"vector":{"mean":0.3685639698170909,"p95":0.4637972224562855,"worst":0.5181143179250072}}}}
  ],
  "metric_units":{"Thermal":"stored native units; monitoring Q1024/native16/32, not full Q8192/64/3096","Wind":"m/s, Q4096 total, fixed24 exposed DEV"}
}
```

Each Thermal 2500 lineage has 375,000 cumulative case visits and 10,000 optimizer updates; the Thermal child adds 300,000 visits and 8,000 updates. Each Wind 2500 lineage has 180,000 visits and 7,500 updates; the Wind child adds 108,000 visits and 4,500 updates. Every active optimizer tensor in each completed selected lineage has the same applicable moment age. A selected monitoring checkpoint can have an earlier age than the completed run; Thermal G2400 has 9,600 updates, while H2500 has 10,000. Wind G2500 has 7500 updates and final learned H2400 has 7200; the original H-v1 selected e2200 has 6600. The completed Wind child has 1000 inherited and 1500 revised-function epochs, while its selected checkpoint has 1000 inherited plus 1400 revised-function epochs. Checkpoint aliases and monitoring files can have different container hashes; the selected-binding receipts verify bitwise-equal model-state tensors, not byte equality of different serializations.

## Six figure families

| Family | Retained PDF master and evidence |
| --- | --- |
| 1. Joint model and learning | [Architecture and physical learning, 2 pages](../../diagnostics/generated/joint_regional_20261009/figures/final_learning/joint_architecture_and_learning.pdf) |
| 2. Native fields and residuals | [Thermal native fields, pages 1–5](../../diagnostics/generated/joint_regional_20261009/figures/thermal_selected_fields_and_responses.pdf); [Wind matched native fields, pages 1–2](../../diagnostics/generated/joint_regional_20261009/audit/wind_revision2_compact_best2400_pages_v1/figures/compact_paired_native_pages.pdf); [paired native wake cuts, 2 pages](../../diagnostics/generated/joint_regional_20261009/audit/wind_revision2_selected2400_paired_wake_cuts_final_v1/figures/native_wake_cuts_paired.pdf) |
| 3. Actual donor→edge→node→receiver chains | [Thermal full low/high graph and intervention audit, 6 pages](../../diagnostics/generated/joint_regional_20261009/audit/core_thermal_org_selected_Hlambda1_e2500_v1/figures/thermal_selected_Hlambda1_e2500_final_v4_actual_graph_and_intervention_audit.pdf); [Wind final learned child graph and intervention audit, 6 pages](../../diagnostics/generated/joint_regional_20261009/audit/core_wind_org_selected_child_v2_mature_v1/figures/wind_CHILD_v2_inherited1000_new1400_of1500_Hmonitor_e2400_overviewfix_v5_actual_graph_intervention_probe_audit_v1.pdf) |
| 4. Executed organizational interventions | Same graph-audit masters, native error changes and exact restorations. |
| 5. Stored physical response examples | [Thermal existing response records, pages 6–8](../../diagnostics/generated/joint_regional_20261009/figures/thermal_selected_fields_and_responses.pdf); [Stored Wind directions, page 3](../../diagnostics/generated/joint_regional_20261009/audit/wind_revision2_compact_best2400_pages_v1/figures/compact_paired_native_pages.pdf) |
| 6. Accuracy against measured work/cost | [Error–cost and retention frontiers, 2 pages](../../diagnostics/generated/joint_regional_20261009/figure6_saved_cost_closeout_v1/figure6_accuracy_cost_work.pdf); [exact scope/cost companion](../../diagnostics/generated/joint_regional_20261009/figure6_saved_cost_closeout_v1/figure6_accuracy_cost_tables.md) |

Figures and numerical evidence remain in ignored local paths under the upload rule; this checkout's Markdown renders them directly. Only the durable report, guide, implementation, tests and recipe configuration are uploaded. Every embedded selected panel was rendered and independently visually inspected; PDF masters retain the complete pages, including low-M counterparts and failures.

## 1. One joint model actually learns

![Executed shared joint core, fresh Thermal heads and nonlinear Wind head](../../diagnostics/generated/joint_regional_20261009/figures/final_learning/joint_architecture.png)

Figure 1a. The executed representation supplies both fresh Thermal heads; heating enters only the final source-column contraction. Two nonlinear collective blocks return edge content to both typed populations, with explicit source/global bypasses. Wind uses the same class and nonlinear physical output with its TRAIN-fitted profile, not a second full predictor or an external learned field. At fixed weights and geometry, the shared context c supplies F(q)=flow_head(q,c) and K_j(q)=temperature_head(q,c,source_j), with T(q,h)=sum_j K_j(q)h_j over original physical source IDs. Heating is excluded from c, F and K; the same T field and its native linear stencils supply surface/material/q roles. This is an exact control-linear temperature block, while the complete configuration-to-field model and Wind head remain nonlinear.

![Saved fixed development learning curves for selectors and difficult physical fields](../../diagnostics/generated/joint_regional_20261009/figures/final_learning/joint_learning.png)

Figure 1b. Saved 100-epoch monitors expose physical fluid-T/vorticity and near-wake Ux/Uy beside each selector. The locality curves start after their branch boundaries rather than reusing old-function scores. All shown statistics are fixed exposed DEV, one seed. Parent learning and new-function age are distinguished; late misses remain visible.

The selected checkpoint's named Adam states confirm that both Thermal heads and the shared representation learned together. Every tensor in each row below has a finite model/optimizer state and a nonzero first moment, at the recorded update age. These are read-only checkpoint measurements; the child ages include inherited updates under the prior function.

| Selected checkpoint | Parameter group | Active tensors | Scalar parameters | Updates per tensor |
| --- | --- | ---: | ---: | ---: |
| Thermal H2500 | Shared representation | 82 | 996,544 | 10,000 |
| Thermal H2500 | Fresh flow head | 4 | 52,100 | 10,000 |
| Thermal H2500 | Fresh affine temperature head | 4 | 16,641 | 10,000 |
| Wind G2500 | Shared representation | 68 | 908,416 | 7,500 |
| Wind G2500 | Nonlinear physical velocity head | 4 | 54,019 | 7,500 |
| Wind child H2400 | Shared representation | 82 | 1,007,424 | 7,200 |
| Wind child H2400 | Nonlinear physical velocity head | 4 | 54,019 | 7,200 |

## 2. Native field quality

### Thermal

| Physical role | G mean / p90 / worst | H mean / p90 / worst | H mean error change vs G (%) |
|---|---:|---:|---:|
| u | 0.046894 / 0.054565 / 0.059023 | 0.05366 / 0.058855 / 0.062008 | +14.43% |
| v | 0.0077793 / 0.0094223 / 0.01034 | 0.0073119 / 0.0089082 / 0.0093798 | -6.008% |
| p | 0.024324 / 0.028925 / 0.031976 | 0.028935 / 0.034874 / 0.035493 | +18.96% |
| omega | 0.38361 / 0.43683 / 0.45128 | 0.32479 / 0.35551 / 0.39172 | -15.33% |
| fluid temperature | 0.72587 / 1.1588 / 1.5513 | 0.68578 / 1.079 / 1.3896 | -5.523% |
| surface temperature | 0.76715 / 1.2643 / 1.7794 | 0.70623 / 1.0354 / 1.4512 | -7.941% |
| material temperature | 0.71568 / 1.1236 / 1.7406 | 0.64457 / 0.92343 / 1.405 | -9.935% |
| q proxy | 2.3662 / 3.3768 / 3.9343 | 2.2851 / 3.0946 / 4.5013 | -3.427% |
| sampled material peak | 0.78711 / 1.1948 / 2.1405 | 0.687 / 1.1954 / 2.1352 | -12.72% |

Cells are equal-case mean/p90/worst RMSE in dataset-native physical units after the maintained TRAIN transform. No unsupported Kelvin or SI label is assigned. The primary fluid panel uses exactly the saved weighted Q1024 support; q is the native proxy, not an independently measured heat-flux field. Literal e2500 versus e2500 preserves the result: fluid-T .725420→.685783 and vorticity .384119→.324791; u/p still worsen. Selecting each arm's best saved field checkpoint therefore does not create the reported gain.

| Module count M | Cases | G fluid-T mean / p90 / worst | H fluid-T mean / p90 / worst | Near / far valid Q records |
|---:|---:|---:|---:|---:|
| 3 | 6 | 0.50705 / 0.66656 / 0.67318 | 0.48215 / 0.59791 / 0.65688 | 2,519 / 3,625 |
| 5 | 6 | 0.76293 / 1.2789 / 1.5513 | 0.70704 / 1.1089 / 1.3896 | 2,787 / 3,357 |
| 7 | 6 | 0.80376 / 1.1567 / 1.3069 | 0.72565 / 0.9579 / 1.1007 | 3,016 / 3,128 |
| 10 | 4 | 0.88168 / 1.0861 / 1.1757 | 0.89955 / 1.1983 / 1.3331 | 2,097 / 1,999 |
| All DEV22 | 22 | — | — | 10,419 / 12,109 of 22,528 valid primary temperature queries |

The temperature query panel has 10,419 near and 12,109 far valid records across 22 cases; these counts describe sampling support rather than separate independent cases. On the input-defined near-body/overlap/far regions, selected H mean/p95 vorticity RMSE is .397995/.461642, .140428/.186472 and .093301/.137799; fluid-T is .700716/1.385572, .790310/1.468724 and .574651/1.014126. These weighted per-case regional summaries use the same Q1024 arrays and retain the tails. M10 fluid-T mean .881677→.899548 and worst 1.175716→1.333090 miss the caution bands. The full native scene 0684 also worsens fluid-T RMSE .91020→1.17575; averaging over DEV does not remove that failure.

![Low-M Thermal native fluid temperature and shared-scale residuals](../../diagnostics/generated/joint_regional_20261009/audit/thermal_selected_final_response_bundle_v2/figures/native_selected/low_m_0277_fluid_temperature_fluidmasked.png)

Figure 2T-a. Fixed DEV0277, M3: geometry/H fluid-T RMSE is 0.41632/0.43098 in dataset-native units. Reference, G2400 and H2500 retain identical Q8192 coordinates and body masks. This low-M scene slightly worsens despite the all-DEV mean improvement.

![High-M Thermal native temperature with visible local error](../../diagnostics/generated/joint_regional_20261009/audit/thermal_selected_final_response_bundle_v2/figures/native_selected/high_m_0684_fluid_temperature_fluidmasked.png)

Figure 2T-b. Fixed DEV0684, M10: geometry/H fluid-T RMSE is 0.91020/1.17575, a 29.2% deterioration. The common residual scale exposes the retained local miss; bodies are masked in gray and excluded from fluid errors.

![High-M Thermal native vorticity and residuals](../../diagnostics/generated/joint_regional_20261009/audit/thermal_selected_final_response_bundle_v2/figures/native_selected/high_m_0684_omega_fluidmasked.png)

Figure 2T-c. On the same M10 native scene, vorticity RMSE improves 0.24949→0.23913. The wake-shaped structure is evaluated against the maintained physical reference, separately from the temperature miss.

![High-M Thermal material temperature on unmasked native material support](../../diagnostics/generated/joint_regional_20261009/audit/thermal_selected_final_response_bundle_v2/figures/native_selected/high_m_0684_material_temperature_unmasked.png)

Figure 2T-d. Native material temperature uses 3,096 points per physical module without the fluid-body mask, 30,960 material records for M10. It is a separate extraction of the same affine field; its units remain the stored Thermal units. This panel does not relabel the 32-point monitoring role as full material evaluation.

![High-M Thermal u, v and p prediction errors retained beside difficult-field gains](../../diagnostics/generated/joint_regional_20261009/audit/thermal_selected_final_response_bundle_v2/figures/native_selected/high_m_0684_flow_fields_uvp_weakness.png)

Figure 2T-e. On native DEV0684, u RMSE worsens 0.04139→0.05524 and p worsens 0.01894→0.02713. The shared physical/residual scales and native pressure gauge make these weaker fields visible beside the vorticity gain.

The fixed input-selected native scenes are 0277(M3) and 0684(M10), Q8192 at the maintained 128×64 coordinates, with native 64-angle interface and 3096 material points per module. The original 4096 raw labels cross-check exactly against the 8192 saved coordinate/value subset. Body masks match exactly and gray body cells are excluded from fluid errors; material T is shown on its own unmasked physical support. Every comparison preserves coordinates, pressure gauge, output ordering, extraction and scales. The high-M u/v/p panel makes the weaker configuration fields visible alongside the improved aggregate vorticity and temperature measures.

### Wind

| Role / quantity | G2500 mean / p95 / worst (m/s) | H-locality2400 mean / p95 / worst (m/s) | H mean change (%) |
| --- | ---: | ---: | ---: |
| near turbine / Ux | 0.311602 / 0.387914 / 0.417269 | 0.359528 / 0.455649 / 0.508109 | +15.38 |
| near turbine / Uy | 0.0601739 / 0.0858158 / 0.11224 | 0.0711386 / 0.087125 / 0.0892076 | +18.22 |
| near turbine / Uz | 0.0454238 / 0.0687003 / 0.0799623 | 0.03736 / 0.0496505 / 0.0511495 | -17.75 |
| near turbine / vector | 0.320861 / 0.40943 / 0.425554 | 0.368564 / 0.463797 / 0.518114 | +14.87 |
| downstream envelope / Ux | 0.212591 / 0.26877 / 0.278015 | 0.230627 / 0.303842 / 0.310607 | +8.48 |
| downstream envelope / Uy | 0.0325572 / 0.0522635 / 0.0700012 | 0.0377348 / 0.0481951 / 0.0519584 | +15.90 |
| downstream envelope / Uz | 0.0241215 / 0.039813 / 0.0429006 | 0.0220907 / 0.0296106 / 0.0311531 | -8.42 |
| downstream envelope / vector | 0.216625 / 0.272294 / 0.289885 | 0.234841 / 0.308272 / 0.314866 | +8.41 |
| hub slab / Ux | 0.178228 / 0.254486 / 0.263733 | 0.196306 / 0.273083 / 0.294008 | +10.14 |
| hub slab / Uy | 0.0270375 / 0.0510995 / 0.0573551 | 0.0310425 / 0.0425896 / 0.0437098 | +14.81 |
| hub slab / Uz | 0.0178624 / 0.0321462 / 0.0371568 | 0.0153001 / 0.0221655 / 0.0269633 | -14.34 |
| hub slab / vector | 0.181381 / 0.257213 / 0.266335 | 0.19945 / 0.276337 / 0.298355 | +9.96 |
| volume / Ux | 0.0922949 / 0.122904 / 0.134745 | 0.095088 / 0.138379 / 0.14674 | +3.03 |
| volume / Uy | 0.0150804 / 0.0247032 / 0.0277553 | 0.0164217 / 0.0200925 / 0.0276655 | +8.89 |
| volume / Uz | 0.0117517 / 0.0189063 / 0.0317789 | 0.0106187 / 0.0151073 / 0.0162147 | -9.64 |
| volume / vector | 0.0944534 / 0.125261 / 0.136425 | 0.0971886 / 0.139837 / 0.148747 | +2.90 |
| background / Ux | 0.0408025 / 0.0563779 / 0.0658317 | 0.0390486 / 0.0528287 / 0.0613209 | -4.30 |
| background / Uy | 0.00829597 / 0.0113666 / 0.0129172 | 0.00894953 / 0.0104424 / 0.0112091 | +7.88 |
| background / Uz | 0.00721475 / 0.0100313 / 0.0147419 | 0.00749404 / 0.0100745 / 0.0104632 | +3.87 |
| background / vector | 0.0422862 / 0.0577748 / 0.0686876 | 0.0408148 / 0.0544449 / 0.0623633 | -3.48 |

Cells are equal-row mean/p95/worst RMSE in m/s on the sealed all 24 Q4096 panel. The table retains all 15 component-role metrics and five vector-role summaries.

| DEV stratum | Rows | G near Ux / Uy / vector mean (m/s) | Child near Ux / Uy / vector mean (m/s) |
| --- | ---: | ---: | ---: |
| M 8 | 3 | 0.255335 / 0.0444275 / 0.261729 | 0.355512 / 0.0655937 / 0.363532 |
| M 11 | 3 | 0.261493 / 0.0507299 / 0.268661 | 0.321233 / 0.0719509 / 0.330979 |
| M 13 | 3 | 0.265098 / 0.0473136 / 0.271894 | 0.31097 / 0.0636624 / 0.319198 |
| M 18 | 3 | 0.290642 / 0.0509531 / 0.297733 | 0.353038 / 0.0618976 / 0.359826 |
| M 21 | 3 | 0.332605 / 0.0637487 / 0.343548 | 0.334371 / 0.0704956 / 0.344195 |
| M 24 | 3 | 0.350954 / 0.0931531 / 0.368932 | 0.334567 / 0.0792753 / 0.346171 |
| M 27 | 3 | 0.344699 / 0.0625599 / 0.353256 | 0.398839 / 0.0730081 / 0.407323 |
| M 30 | 3 | 0.391992 / 0.0685055 / 0.401136 | 0.467697 / 0.0832252 / 0.477288 |
| direction 270° | 8 | 0.321054 / 0.0585692 / 0.329426 | 0.37062 / 0.0715261 / 0.379621 |
| direction 285° | 8 | 0.298253 / 0.0592884 / 0.307382 | 0.353134 / 0.0697836 / 0.362226 |
| direction 300° | 8 | 0.3155 / 0.0626642 / 0.325776 | 0.354831 / 0.0721061 / 0.363845 |

The linked component companion retains all 15 components and five vectors, with per-M and direction tails. M24 is the only stratum where every component mean improves; the other wake/transverse strata retain losses.  The child’s selected field score is 0.06750357 versus G’s 0.06563511 (+2.85%). Literal child/G e2500 scores are 0.06758964/0.06563511; literal child improves about 17.8% over old H-v1 e2500 (0.0822387), but does not beat geometry in the prioritized wake roles. Selection cannot turn this losing learned comparison into a positive. The original H-v1 selected e2200 remains a negative mature result: near Ux .37197 versus geometry .31160, near Uy .085586 versus .060174, and downstream Uy .043285 versus .032557. It is not hidden by any later revision.

![Low-M Wind native Ux Uy Uz reference geometry control learned child and residuals](../../diagnostics/generated/joint_regional_20261009/audit/wind_revision2_compact_best2400_pages_v1/figures/low_M_native_comparison.png)

Figure 2W-a. Stored DEV row 69, M8, 270°, Q52,548 at z=70.869949 m: G2500/H2400 RMSE is Ux 0.14910/0.17520, Uy 0.019620/0.026759 and Uz 0.005844/0.006742 m/s. The final learned child is worse in all three components on this native scene.

![High-M Wind native Ux Uy Uz reference and matched selected models](../../diagnostics/generated/joint_regional_20261009/audit/wind_revision2_compact_best2400_pages_v1/figures/high_M_native_comparison.png)

Figure 2W-b. Stored DEV row 426, M30, 270°, Q99,750: G2500/H2400 RMSE is Ux 0.28795/0.33384, Uy 0.036307/0.045783 and Uz 0.007317/0.007844 m/s. The child retains wakes but loses native component accuracy; each component shares its physical and signed residual limits across low/high scenes and both models.

![Paired high-M native source0 wake cuts at existing plus5D and plus10D columns](../../diagnostics/generated/joint_regional_20261009/audit/wind_revision2_selected2400_paired_wake_cuts_final_v1/figures/high_M_native_wake_cuts_paired.png)

Figure 2W-c. At row 426 source 0+5D/+10D, actual absolute native x/D is −2.306/2.642. Geometry→child Ux cut RMSE improves 0.53586→0.38497 and 0.35336→0.28251 m/s, while Uy worsens 0.05384→0.06635 and 0.01795→0.04641. The reference/control/child traces and cells are unchanged, and shared scales preserve these local gains and transverse misses. They do not overturn the all 24 and full-plane preference for geometry; the paired PDF also retains M8 cuts.

Fixed low/high scenes are row 69(gen_0023_wd270,M8) and row 426(gen_0142_wd270,M30); the existing nearest hub-height plane is z=70.869949 m with 52,548 and 99,750 native cells. Coordinates are in rotor-diameter units and vectors in m/s, using each stored row's native direction transformation. Source 0 wake cuts use the nearest existing columns to source-relative +5D/+10D and cross-wake offsets±3D; any printed actual x is the absolute native x/D coordinate. No field interpolation, extra rotation or new solve supplies these maps or cuts. The selected geometry model wins all three components on both fixed native planes. On high-M, its Ux/Uy/Uz relative L2 is 0.03332/0.38402/0.45627 versus the child’s 0.03863/0.48425/0.48916; small Ux aggregate error does not imply accurate transverse velocities.

## 3–4. Actual collective organization and useful interventions

![Actual registered Thermal high-M hyperedge, physical donors and native receiver access](../../diagnostics/generated/joint_regional_20261009/audit/core_thermal_org_selected_Hlambda1_e2500_v1/figures/page_review_selected_Hlambda1_e2500_final_v4/page-3.png)

Figure 3T-a. H2500 on DEV0684 retains M10 active physical sources, all E192 geometry nodes and 16 regional anchors plus source anchors. The complete typed membership and regional access arrays show different donor mixtures in the high-M scene. Source IDs remain joined to the original native coordinates; the six-page PDF also retains the low-M scene.

![Actual Thermal three-receiver access rows, full M E incidence and two-block update norms](../../diagnostics/generated/joint_regional_20261009/audit/core_thermal_org_selected_Hlambda1_e2500_v1/figures/page_review_selected_Hlambda1_e2500_final_v4/page-6-probe-chain-0684.png)

Figure 3T-b. Three input-sealed native probes show every receiver-to-edge weight, every M/E incoming row and actual messages/node updates in both trained blocks. This is the recorded H2500 computation, with both typed populations normalized by their own measure; the figure does not omit low-mass donors to imply physical locality.

Figures 3T. Selected Thermal H2500 uses all physical source identities and the full E192 catalogue; three receiver probes are sealed from input-only near/overlap/background regions before errors are inspected. The maps show registered edges, separate typed donor masses, full receiver access and both blocks' actual messages/node updates. The low-M regional memberships concentrate heavily on one source, whereas the high-M pattern varies; concentration is descriptive routing and does not remove the other source columns. Fixed-geometry heat transfer leaves B/R/flow/K unchanged and changes only affine T contributions, which is the required dependency.

![Actual final Wind child high-M registered hypergraph with physical turbine identities](../../diagnostics/generated/joint_regional_20261009/audit/core_wind_org_selected_child_v2_mature_v1/figures/page_review_CHILD_v2_inherited1000_new1400_of1500_Hmonitor_e2400_overviewfix_v5_v1/page-04-graph-426.png)

Figure 3W-a. The final learned child selected e2400, inherited 1000 plus 1400 new-function epochs, has M30, E64 and G62 active edges on stored row 426. Regional edge 54 gathers distributed source mass (largest donor about 0.075); source anchors preserve individual turbine identity. This mechanism belongs to the losing learned comparison, while the independently trained geometric model is preferred.

![Full final Wind child M E incidence, three actual native receiver reads and block updates](../../diagnostics/generated/joint_regional_20261009/audit/core_wind_org_selected_child_v2_mature_v1/figures/page_review_CHILD_v2_inherited1000_new1400_of1500_Hmonitor_e2400_overviewfix_v5_v1/page-06-probe-chain-426.png)

Figure 3W-b. All 30 turbine rows, 64 environment rows and 62 active edge columns are displayed, along with three pre-sealed native probes at flat IDs 1,637,477/1,645,885/1,596,349. Both collective blocks carry nonzero messages and node updates. Receiver probabilities describe the trained model’s access and do not identify physical causes.

Figures 3W. The final child uses row 69 (M8/G40/Q52,548) and row 426 (M30/G62/Q99,750), with the selected checkpoint, input seal and complete typed arrays bound by SHA256. Every physical turbine and all 64 environment nodes remain represented. The three actual receiver rows show access to registered source and regional edges, and the complete incoming membership plus outgoing updates is visible. Geometric region names identify location, not causal upstream/downstream exclusivity.

![Thermal selected high-M native errors under executed organization interventions](../../diagnostics/generated/joint_regional_20261009/audit/core_thermal_org_selected_Hlambda1_e2500_v1/figures/page_review_selected_Hlambda1_e2500_final_v4/page-4.png)

Figure 4T. Replacing membership or receiver access, averaging receiver access, removing collective content or node updates, and frozen matched-mass shuffling execute through the affected two-block model. Native errors and model distortions are recorded, and all eight exact restoration checks recover the baseline. Source reads and pooled context remain available; the intervention is consequential despite those bypasses. On high-M0684, baseline native relative L2(omega,T) is(.323009,.091353); geometric membership changes it to(.532811,.258390), geometric access to(.845282,.111154), all-block content removal to(2.396644,.253999), and matched-mass shuffling to(.954428,.242623). Low-M content removal likewise changes(.233125,.118318) to(2.501021,.331221).

![Wind selected child native errors under eight executed and restored organization interventions](../../diagnostics/generated/joint_regional_20261009/audit/core_wind_org_selected_child_v2_mature_v1/figures/page_review_CHILD_v2_inherited1000_new1400_of1500_Hmonitor_e2400_overviewfix_v5_v1/page-05-intervention-426.png)

Figure 4W. On row 426, native near/downstream/far vector relative L2 is 0.07197/0.04376/0.02495 at baseline. Geometric membership raises it to 0.48541/0.18442/0.11172, geometric access to 0.12007/0.05743/0.02285, scene-average access to 0.17303/0.10785/0.10686, all-block content removal to 0.76004/0.69982/0.80868 and frozen matched-mass shuffle to 3.35393/1.33969/0.81959. All eight interventions restore exactly. The broader report separately compares independently trained controls; this page shows same-weight organization interventions. Changes in learned probabilities alone would not establish utility. Small source-center changes preserve IDs and rebuild graph states, but without new physical references they establish model sensitivity rather than correct turbine-control physics.

FP32 consistent edge-label permutation checks retain their strict failures: Thermal max differences 5.72e-6/7.63e-6 and original Wind H-v1 max 4.77e-5/1.43e-4 on low/high native panels. Same-weight FP64 reprepare passes the unchanged strict 1e-10 relative/1e-11 absolute criterion, with Thermal max 1.27e-14 and original Wind max 2.18e-13. The final Wind child likewise retains FP32 failures (max 3.8624e-5/1.1444e-4, 117/157,644 and 1,964/299,250 values outside the unchanged 2e-6 relative/absolute criterion). Its same-weight FP64 reprepare passes 1e-10/1e-11 with max 6.57e-14/1.90e-13. A target-free +0.01D x shift of source 0 on row 426 preserves IDs/capacity and changes receiver access by relative L2 0.001523 and velocity by 0.0001963, max 0.03892 m/s. The algebraic symmetry has supporting precision evidence; a blanket native FP32 strict-pass claim would be false.

## 5. Responses to existing physical changes

![Physical module peaks and hot-module ranking under existing 0291 heat transfers](../../diagnostics/generated/joint_regional_20261009/audit/thermal_selected_final_response_bundle_v2/figures/counted_0291/0291_module_peaks.png)

Figure 5a. Existing 0291 baseline/minus/plus heat transfers preserve total 4.634584 and use the same exposed analytic-wake reference geometry. The plus control changes physical source 1 by−.0575153 and source 4 by+.0575153. All module peaks below retain physical IDs; the record is neither a new CFD solve nor independent qualification.

| Physical module | Reference baseline→plus peak | Selected H baseline→plus peak |
| ---: | ---: | ---: |
| 0 | 13.7298→13.8884 | 15.9805→16.2033 |
| 1 | 18.5202→18.1299 | 20.1607→19.8776 |
| 2 | 11.0069→11.2701 | 12.6573→13.0085 |
| 3 | 17.7801→17.9829 | 21.2257→21.5069 |
| 4 | 10.6464→11.1684 | 11.0344→11.5794 |

Reference hot-module identity stays 1; selected H incorrectly stays 3 for baseline/minus/plus. The reference global maximum decreases 18.5202→18.1299 while H increases 21.2257→21.5069; reference and predicted hot locations are(9.6165,4.1647) and(6.0538,3.6118). The transfer therefore still misses the known ranking weakness despite the primary field gains. Peaks/global maxima are nonlinear endpoint quantities, not the linear heat increment.

![Existing 0291 normal heat-flux proxy and temperature transfer increments](../../diagnostics/generated/joint_regional_20261009/audit/thermal_selected_final_response_bundle_v2/figures/counted_0291/0291_q_normal_transfer_increments.png)

Figure 5b. Plus-transfer increment RMSE is .02612 fluid/.04246 surface/.04641 material/.10488 q, with max absolute discrepancies .11316/.09761/.09763/.28415 in native units. Heat-induced flow increment is exactly zero. Endpoint q errors 4.44/4.52 and wrong hotspot ranking still block physical inverse qualification; a small increment error alone cannot validate the endpoint or design.

![Sixteen saved fitted TRAIN geometry-response increments and errors](../../diagnostics/generated/joint_regional_20261009/audit/thermal_selected_final_response_bundle_v2/figures/saved_geometry16/saved_atlas_geometry_increment_fit.png)

Figure 5c. Sixteen pre-existing geometry increments from 0001/0318/0333/0348 compare 20 endpoints across the four fitted TRAIN families, with no solver or optimizer call. Paired increment RMSE min/median/max is fluid .0473/.1080/.3717, surface .0442/.1766/.5759, material .0521/.1793/.6003 and q .3072/.9047/1.5085. All 20 sampled hot-module IDs agree in this small fitted panel; that agreement is exposed auxiliary fit evidence, not an independent response gate.

![Stored 270 and 285 degree native operating-context Ux fields for reference and selected models](../../diagnostics/generated/joint_regional_20261009/audit/wind_revision2_compact_best2400_pages_v1/figures/stored_270_285_operating_context.png)

Figure 5d. Stored 270° and 285° operating rows of the same low-M layout use their own native cells and transforms. Rows 69/70 retain 52,548/53,940 native cells, the same eight physical turbine identities and the nearest stored hub-height plane. The figure compares their actual Ux wakes in m/s against the corresponding stored references with a common physical scale. These are existing operating-context field comparisons; center JVP/VJP below remain model derivatives, with no finite turbine-displacement/yaw/thrust physical reference.

## 6. Accuracy, complete costs and sparsification

![Exposed development physical errors and independently measured complete inference times](../../diagnostics/generated/joint_regional_20261009/figure6_saved_cost_closeout_v1/figure6_accuracy_cost.png)

Figure 6a. Fixed DEV mean error and p90/p95-to-worst tails are shown against complete synchronized inference medians for the separate low/high native cost scenes. Thermal H2500 buys 5.52% fluid-T and 15.33% vorticity mean improvements at roughly 59–60 ms versus geometry’s 44 ms; Wind geometry retains lower near-Ux/Uy errors at about 15 ms versus the child’s 19.4–19.5 ms on Q4096. Direct e500 is an earlier-age screen, not a mature fair comparison.

![Four access-mass retention levels with physical-reference errors and identical-count geometric controls](../../diagnostics/generated/joint_regional_20261009/figure6_saved_cost_closeout_v1/figure6_retention_work.png)

Figure 6b. Targets 90/95/99/100% plot positive receiver value pairs against saved dense score slots; typed context, scoring and all physical-source paths still execute densely. Thermal uses weighted DEV22 Q1024, and Wind uses selected child e2400 DEV24 Q4096. At 99%, learned access preserves important fields far better than the same-count geometry intervention; 90% substantially damages vorticity and transverse wakes. These x values are logical work, not complete executor savings.

Figure 6. The matched smooth models and actual packed reader are timed separately, and field evidence remains tied to each exact selected checkpoint. Matched smooth measurements use RTX 6000 Ada on authorized GPU1; the actual subset experiment and declared repeat explicitly use GPU2 with synchronized complete calls; the 2-warm/7-call median and p90 timings are not compared against differently scoped historical fits. The old 82 ms Thermal/25ms Wind values remain unmatched engineering context only.

| Model / scene | Complete median / p90 ms | Prepared read median ms | Useful native VJP median ms | CUDA allocated / reserved GiB |
| --- | ---: | ---: | ---: | ---: |
| Thermal H locality e2500, high_m, Q8192 + native64/3096 | 59.8717 / 59.9897 | 25.3821 | 119.94 | 1.308 / 1.414 |
| Thermal H locality e2500, low_m, Q8192 + native64/3096 | 59.2151 / 59.63 | 25.6157 | 121.116 | 1.308 / 1.414 |
| Thermal G e2400, high_m, Q8192 + native64/3096 | 44.1582 / 44.5764 | 18.3394 | 89.0206 | 0.786 / 0.877 |
| Thermal G e2400, low_m, Q8192 + native64/3096 | 44.1322 / 44.1781 | 18.4433 | 90.9239 | 0.787 / 0.877 |
| Wind G e2500, high_M, five roles Q4,096 | 14.5906 / 14.6779 | 9.34081 | 30.7182 | 8.935 / 9.982 |
| Wind G e2500, high_M, native plane Q99,750 | 209.706 / 211.226 | 203.365 | 482.561 | 8.935 / 9.982 |
| Wind G e2500, low_M, five roles Q4,096 | 14.9608 / 15.7211 | 9.49627 | 34.3616 | 1.419 / 1.498 |
| Wind G e2500, low_M, native plane Q52,548 | 112.367 / 112.686 | 105.38 | 255.229 | 1.419 / 1.498 |
| Wind H locality e2400, high_M, five roles Q4,096 | 19.5334 / 19.5475 | 12.4177 | 38.5967 | 16.066 / 17.043 |
| Wind H locality e2400, high_M, native plane Q99,750 | 287.347 / 288.638 | 287.054 | 647.874 | 16.066 / 17.043 |
| Wind H locality e2400, low_M, five roles Q4,096 | 19.3889 / 19.4307 | 12.0826 | 43.9796 | 3.860 / 3.932 |
| Wind H locality e2400, low_M, native plane Q52,548 | 157.257 / 185.305 | 149.519 | 341.822 | 3.860 / 3.932 |

Warm epochs include all train objectives, backward and AdamW updates and exclude each owned process-segment first epoch; child percentiles exclude inherited old-function epochs. Charges belong to the actual owning process, with failed attempts counted and parent work counted once. Training peaks below are each run’s own invocation maxima; training host RSS was not recorded.

| Run | Warm epoch median / p90 (s) | Cold process-start epoch:seconds | Charged owned process seconds | Training CUDA allocated / reserved GiB |
| --- | ---: | --- | ---: | ---: |
| T4001 J-direct | 1.027 / 1.103 | e1:1.0, e101:1.7 | 521.30 | 8.74 / 10.08 |
| T4002 J-geometry | 0.769 / 1.160 | e1:1.1, e101:1.8, e501:1.5, e1001:1.3 | 2290.52 | 9.67 / 10.73 |
| T4003 J-H-v1 (H0) | 1.368 / 1.535 | e1:1.3, e101:1.9, e501:2.4 | 1378.59 | 16.87 / 18.36 |
| T4004 J-H locality prior λ=1 | 0.976 / 1.019 | e501:1.5, e1001:1.4 | 2035.18 | 16.87 / 18.37 |
| W2301 J-direct | 0.877 / 1.020 | e1:316.9, e101:442.0 | 1889.12 | 8.88 / 9.79 |
| W2302 J-geometry | 0.666 / 0.832 | e1:465.6, e101:485.6, e501:318.5, e1001:316.0 | 3990.61 | 9.36 / 10.30 |
| W2303 J-H-v1 (H0) | 0.854 / 0.866 | e1:411.0, e101:332.8, e501:316.4, e1001:315.5 | 4141.04 | 16.69 / 17.64 |
| W2304 J-H locality prior λ=1 revision 2 | 0.851 / 0.858 | e1001:314.5, e1101:311.1 | 2343.12 | 16.69 / 17.64 |

| Selected model | Standalone DEV2 setup (s) | Resident weights MiB | Profile process peak RSS MiB |
| --- | ---: | ---: | ---: |
| T4002 e2400 | 0.1468 | 3.706 | 1381.7 |
| T4004 e2500 | 0.0362 | 4.064 | 1381.7 |
| W2302 e2500 | 0.2255 | 3.671 | 1120.7 |
| W2304 e2400 | 0.2130 | 4.049 | 1121.7 |

Standalone setup is checkpoint/native-catalogue construction, not full training-provider setup. These setup values depend on measured process/cache order; repeated RSS values are the shared profiling process peak, not independent per-model host allocations. Thermal precise prepared heat increments take about 1.77–1.86 ms at the same native scopes. Wind useful VJPs are measured separately on Q4096 and Q52,548/99,750 full planes.  CUDA peaks above cover each scene's complete native forward plus useful VJP audit, not Q4096 inference alone. Resident weights, standalone DEV2 catalogue setup, full training/provider cold starts and process RSS are distinct scopes. Cold startup and failed attempts are charged to the campaign; the Wind first-epoch/validation data-loading stalls are not presented as warm GPU throughput.

Receiver retention at 100/99/95/90% cumulative access mass is a same-weight approximation, with identical retained-count geometry controls and no prescribed K. Both deviation from full predictions and stored-reference error are saved for every role and case. Thermal 99% fluid-T mean .687023 versus .685783 and vorticity .327261 versus .324791 are near the full function;90% vorticity .489070 versus .324791 is a substantial miss. Same-count geometry access is much worse for the learned checkpoint. For final Wind H2400, near Ux mean/p95/worst changes 0.359528/0.455649/0.508109→0.360695/0.453083/0.504080 at 99%, Uy 0.071139/0.087125/0.089208→0.073062/0.088156/0.090795, and vector 0.368564/0.463797/0.518114→0.370360/0.461701/0.515420. The same-count geometry vector mean is 0.775368. Learned 99% uses 2,099,595 positive value pairs against 5,013,504 dense score slots on the all 24 Q4096 panel; original full positive mass has 4,138,506 pairs because FP32 softmax underflows some values to zero. Full dense execution still scores/reads its declared capacities. Positive receiver-edge pairs are counted separately from dense QG scores, all-source QM reads and typed collective node-edge work. On the full native Wind low/high planes, two-block typed M→G membership capacities are 640/3,720 and E→G capacities 5,120/7,936; dense receiver score capacities are 2,101,920/6,184,500 and physical-source read capacities 420,384/2,992,500. These recorded primitives do not count only the retained edges as the complete computation.

The actual Thermal 99% packed reader cuts value pairs about 62% on the Q8192 low/high native flow panel, while complete inference increases from roughly 57–60ms to 80–81ms. Scores, context and all physical-source reads remain dense. All-source native extraction is preserved, but high-M q/interface parity fails at 2 of 1536 values under the unchanged rtol 3e-5/atol 3e-6 criterion (max 4.245e-5); that miss remains in the receipt. On GPU2, final Wind H2400 high-M Q4096 full/99%-packed medians are 19.384/24.618 ms with 253,952 score pairs and 86,358 retained value pairs; the full native plane increases 284.038→422.452 ms with 6,184,500 scores and 2,046,546 packed values. Low-M native plane increases 156.851→225.129 ms. Its first low-Q full median was noisy at 28.440 ms; a sealed repeat on the same GPU gives full/packed/fallback 19.514/24.650/19.499 ms. Both measurements remain saved. All final Wind same-truncation output checks pass, and every physical source still executes. The final geometry preference is based on full smooth timing on GPU1, not this repeat. Full-access subset requests use the dense fallback and recover bitwise equality. Smooth dense access remains the preferred/manual default; no sparse wall-clock speedup, total-ancestry reduction or physical omission guarantee is claimed.

## Numerical interface and tests

On selected Thermal H e2500, cold heat gives zero T/q/material values, all 12 native source columns remain available, heat-flow increments are exactly zero, and the prepared FP64 contraction reproduces the actual native field/stencil increments. Heat finite differences at .01/.001 have about 1e-13–1e-12 discrepancy and 1e-8 retains about 1.7e-8–7.5e-8 relative error. Legacy FP32 .001 failures(.001116/.007321) and 1e-8 failures(relative error 1) remain. Geometry/query dots and the declared finer finite-difference checks pass; the coarse radius.001 misses(.105817/.097402), with fixed-stencil high-M.001 also missing(.082168). Eulerian receiver and module-attached derivatives remain different and support changes are explicit.

Selected Wind geometry and child pass native center/query JVP–VJP dot checks and reject stale geometry, receiver, profile, source, weight and locality states. Their native FP32 finite differences retain scale-dependent failures: geometry high-M query/center at 1e-4 have relative errors 0.05090/0.12658, and child high-M at 1e-4 have 0.07262/0.10257; child high-M query at 5e-5 reaches 0.46235. Child full smooth same-weight FP64 replay passes both 1e-4/5e-5 scales (largest relative error 2.76e-6). Native FP32 finite-difference tolerance remains 5%; no old failed proxy tolerance is relaxed. The same trained full smooth Wind geometry model cast to FP64 passes both 1e-4 and 5e-5 center/query finite-difference scales with relative errors 3.2e-8–8.8e-7, and all dot tests pass. This is precise arithmetic replay, not a different fitted physical model or an erased FP32 failure.

Fixed-support packed/dense trained-model derivatives are independently checked in FP64 on nine input-selected native receivers per scene. Thermal fields plus every source kernel and Wind physical velocities retain strict endpoint/JVP/VJP parity, finite-difference checks and dot checks. The Thermal conditional check promotes native FP32 typed features into the FP64 core; it does not establish a fully FP64 feature pipeline. These derivatives condition on frozen edge identities and do not differentiate mass-selection support switches. All declared geometry/query/normalization/source/weight/locality invalidations, including Wind empirical-profile state, reject stale preparation. Loading/target-poison/actual-gradient/resume/batch-denominator/atom-split/padding/native tests substantiate the joint contract; the self-contained checkpoint predicts with access to old learned model files denied.

The final affected CPU suite has 133 passes; the final focused joint training/manual runtime suite additionally has 10 passes in 2.38 s. Twenty-four originally CUDA-hidden inherited failures pass on authorized GPUs; fourteen persistent unrelated inherited failures remain classified in saved receipts(10 core,4 Thermal). Neither broad legacy suites nor numerical proxy panels are falsely called fully green. The final Wind geometric fullTRAIN 5000 recipe passes real CUDA-hidden inert validation in 4.60 s with 962,435 active parameters, zero frozen parameters, zero optimizer construction/updates and two native TRAIN forwards; its recipe SHA256 is dd4c72e05c316ccd809a815fdffe44638d6bca7c5b1a54b29a9e3435d236221d. The unchanged Thermal locality recipe retains its final inert receipt and SHA256 25632a499b6752d3f2ebf576fc0f1fb4040b7f46e4633a385a717aeb29fe26fc.

## Preferred recipes and delivery boundary

Use Thermal J-H locality strength 1, full smooth dense access, with the measured child lineage and the geometry control retained. Use Wind J-geometry, full smooth dense access, retaining the final H-locality child as a tested comparison rather than a second preferred recipe. Fresh starts of a locality recipe do not reproduce the measured child transfer history; the recorded parent boundary, moments and schedule are part of that development lineage. The tested [Thermal manual 5000 configuration](../../src/config_core/forward/joint_regional/thermal_full5000_v1.json) and [Wind manual 5000 configuration](../../src/config_core/forward/joint_regional/wind_full5000_v1.json) are experimental preparation artifacts. They initialize all trainable weights freshly and fit transforms on original 600 Thermal TRAIN cases or 420 Wind TRAIN rows; fullTRAIN and development identities/schedules are distinct and cannot be silently resumed across. Inert checks construct actual providers and native TRAIN forwards with an optimizer-constructor guard, zero optimizer steps and zero formal checkpoints. Formal GPU memory/throughput and mature formal quality are unmeasured. No formal run is launched; a future manual launch requires its separately requested scope and explicit launch flag.

At computational closeout (2026-10-09 19:58:46 UTC), total charged GPU-associated wall is 5.399331 hours and elapsed wall is 5.524022 hours, within the 24 GPU-hour/16 elapsed-hour ceiling. All candidate GPU work has ceased on the authorized physical GPUs 1/2, preserving the final two-hour reserve. This conservative accounting includes data stalls, preparation, failed attempts and the Wind child wrapper exit 143: the engine independently saved all 2500 history rows, checkpoints and finite optimizer states, while its wrapper completion was interrupted. Its original running receipt is preserved, raw exit 143 is retained, and 1,821.582 seconds are charged through verified process absence; an unknown engine exit code is not rewritten as zero. Review/source-format/retention/Git closeout uses no further candidate GPU work.

The exact 647 goal-start logical bindings, including archived/symlinked 3903, are compared with identical membership at closeout. All 647 exact logical paths have identical SHA256, sizes, nanosecond timestamps and resolved paths; the separate end snapshot SHA256 is 663523174259083ec589156aa20abf2bad53c8bfcb634c832b073f6d86d873cc, with zero changes. User-owned plan hashes and the pre-existing report deletion are unchanged. The older inventory-evidence limitation is retained rather than retroactively repaired. WindTEST targets remain locked; solver usage stays 326/326, with zero new solves, inverse campaign or automatic formal run. Unrelated jobs and the user's deleted report/untracked plans remain untouched. The [compact implementation/training guide](../guides/Joint_Regional_Field_Development.md) records the reusable APIs and manual boundary; local audit receipts bind checkpoints, protocols, measurements and selected figures.

## Bound evidence index

Generated numerical evidence, figures and one-time scripts are local ignored artifacts. The seven retained PDF masters in the six-family index and required inline raster companions are the presentation set; earlier receipts and scientific arrays/checkpoints remain available independently.

| Evidence | Bound local receipt |
| --- | --- |
| Thermal G2400/H2500 selected-state equivalence | [receipt](../../diagnostics/generated/joint_regional_20261009/audit/thermal_mature_selected_binding_v1.json) — SHA256 `33dd10c02716d95fc0d1fe00cb6824710083496affdc57003930a98e5d0db3cd` |
| Wind G2500/original H2200 selection | [receipt](../../diagnostics/generated/joint_regional_20261009/audit/wind_mature_v1_selected_binding_v1.json) — SHA256 `dc4e6d10adb34f625b1e654b8c1935dcb7f144b1c048c6d79bc7add0952cb2b3` |
| Wind child H2400/completed2500 lineage and optimizer integrity | [receipt](../../diagnostics/generated/joint_regional_20261009/audit/wind_mature_revision2_selected_binding_v1.json) — SHA256 `9250ebabef2fb8f13f241ee7446debc8b23fb37df9ca0e5d2681b89faf73a5f8` |
| Thermal primary all22 and literal2500 physical role metrics | [receipt](../../diagnostics/generated/joint_regional_20261009/audit/thermal_joint_e2500_final_dev22_summary_v1.json) — SHA256 `1c1aa309cacf678cc655d935d568e616b5e4f2889fda6c8cf96c9a7a0b14cd5f` |
| Wind all24 selected/literal matched child versus geometry | [receipt](../../diagnostics/generated/joint_regional_20261009/audit/wind_revision2_matched_child_geometry_comparison_v1.json) — SHA256 `8e5fdcdd3f2408ecc6bf1f198a8f4383f945cbc92ab38dba861c243bcb8a712e` |
| Wind full component/M/direction history tables | [receipt](../../diagnostics/generated/joint_regional_20261009/figure6_saved_cost_closeout_v1/wind_revision2_selected_e2400_validation_tables.md) — SHA256 `3297d540ebae76c01d0dfc1b3f899ad0012257922c4a5631c66cebf2079a5d69` |
| Thermal native fields and exposed/fitted response scope | [receipt](../../diagnostics/generated/joint_regional_20261009/audit/thermal_selected_final_response_bundle_v2/selected_response_evidence.json) — SHA256 `9c76f31302986b0de64c5ab5a30d70a615a4d89f7afca3b70424895b35241584` |
| Exact cost/retention/parity companion | [receipt](../../diagnostics/generated/joint_regional_20261009/figure6_saved_cost_closeout_v1/figure6_accuracy_cost_tables.md) — SHA256 `0d9cede4110b8082cd1a6ae383d9ffdcf6f5d4a8958f1af495ec94c9aa8ed9c4` |
| Eight-run lineage/work/training and profile resources | [receipt](../../diagnostics/generated/joint_regional_20261009/audit/final_experiment_table_v2.json) — SHA256 `e847578b57c54b1428730a6b3b4e36054bcb3dfe48d4d46eb88626c3e553e216` |
| Selected shared/head update ages and finite nonzero Adam moments | [receipt](../../diagnostics/generated/joint_regional_20261009/audit/selected_head_optimizer_evidence_v1.json) — SHA256 `175b2f58c3f5d315b33d116d3c63579fafe9db5c731f921e993825e328837648` |
| 133 affected CPU passes and bitwise legacy continuation replay | [test/commit receipt](../../diagnostics/generated/joint_regional_20261009/audit/push_round6.json); [native parity receipt](../../diagnostics/generated/joint_regional_20261009/audit/round6_default_parity/comparison.json) |
| Final focused runtime recheck, 10 passes in 2.38 s | [observed terminal-result receipt](../../diagnostics/generated/joint_regional_20261009/audit/final_focused_runtime_test_observation_v1.json) |
| Broad inherited failure classifications | [receipt](../../diagnostics/generated/joint_regional_20261009/audit/broad_tests/classification.json) |
| Numerical scales and conditional precision scope | [declaration](../../diagnostics/generated/joint_regional_20261009/audit/final_numeric_scale_declaration_v1.json); [Thermal trained conditional derivatives](../../diagnostics/generated/joint_regional_20261009/audit/thermal_selected_h2500_conditional_subset_v1.json) |
| Root final selected Wind and cost visual review | [receipt](../../diagnostics/generated/joint_regional_20261009/audit/root_wind_child_and_cost_final_visual_review_v1.json) — SHA256 `9b3e3002ec2bfaa35d5985526c0ca7c77f31e18085dd6ace86831aa95e985599` |
| Root selected Thermal graph visual review | [receipt](../../diagnostics/generated/joint_regional_20261009/audit/root_thermal_final_visual_review_v1.json) — SHA256 `7c8804a8bdc45d9e2ef1dc435f792edbefbd9e16a41f802fef3f75578e12919b` |
| Protected exact647 and user-state/solver closeout | [receipt](../../diagnostics/generated/joint_regional_20261009/audit/protected_scope_closeout_v1.json) — SHA256 `234c491b0d9e1c289d17ba9b0608685ccca503e8a15cdc79c48ab3caa30f2c09` |
| Conservative aggregate resource closeout | [receipt](../../diagnostics/generated/joint_regional_20261009/audit/resource_closeout_v1.json) — SHA256 `3cc2c146c7a1cd703fbff6a4e05fa657cac2e8592e8206c744700b751ef47e6b` |
| Thermal fullTRAIN5000 inert native validation | [receipt](../../diagnostics/generated/joint_regional_20261009/audit/manual_thermal_5000_final_v2.json) — SHA256 `17882b34f729813c477f11227b0546160ad4ba504b847efcf181e1aab3cdb2b4` |
| Preferred Wind geometry fullTRAIN5000 inert native validation | [receipt](../../diagnostics/generated/joint_regional_20261009/audit/manual_wind_5000_geometry_final_v3.json) — SHA256 `a89502ca70815e739d7a05ae185192fc18ff0c124422534d36947f5e25e4a15c` |

The durable full-source range is recorded by the final push audit; generated evidence is intentionally not contained in the remote repository. Locally selected weights are Thermal H `52ce5831c30b3ad61525509bc56ba667b66b47dcba8f2e548751e271a80c6846` / control G `92dfb8131dc02406587b4a719387e3cb464bb7b9244a3ada712b79d1cd077992`, and Wind preferred G `5330f614fec1b93edb54f67fefc41dd384752a58617e574c6d3897333ee3db1b` / learned child `55800ab1ad2512c88351047e78b7b0753c0cac56476699ef5d5f56e54b00e040`. Full lineage, partition, transform and native-input hashes remain in the linked receipts.
