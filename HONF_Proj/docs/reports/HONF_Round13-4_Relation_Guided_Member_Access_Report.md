# HONF R13-4: relation-guided original-member access

**Status: development review complete; independent scratch candidate in progress.** Both transferred arms completed 100 additional TRAIN600 epochs and their native evaluations. The selected Guide is now being trained independently from scratch for its separately reserved 100 epochs. Scratch results, final cleanup and closeout remain pending.

R13-4 moves the E candidate decision ahead of the expensive query–member scorer. The decision uses a prior composed from the model's actual applied final-stage relations, query proposals and retained record information measures. Selected original M/E records retain their identities, geometry, controls and contextual values, and feed one general joint physical head. The new weighted member-score calibration also removes the exact-positive-count boundary found in the previous numerical reference. Historical readers and checkpoints retain their loading behavior.

For the predictor, the restricted Guide at transferred epoch 600 improves eight of nine exposed-VAL89 mean errors against the new Full, including 3.24% lower material-temperature error, while surface-temperature error is 0.65% higher and the worst material/peak errors are about 6.6% higher. For the organizer, the relation prior restricts original-member scores before the MLP and preserves scene dependence, but remains broad: 174.75 E scores per query versus 190.37 in the same Guide checkpoint's full reference. Profiled native inference is slower after restriction, so no end-to-end saving is established. The recommended next candidate is the existing simple Guide, with omission tolerance 0.01 and zero auxiliary loss. No new inverse experiment or solver campaign has been performed.

## Figure index

The selected evidence is grouped below as qualification health, matched learning/VAL89 work, three-way organization, literal M7/M0 evolution, relation-to-access, four fixed native fields, original-module detail and complete native costs. Each figure is embedded at its relevant measurement with a linked PDF master; numerical sources remain on DATA2.

## Model and comparison

The [applied reader amendment](../../UpgradePlan/HONF_Round13-4_Applied_Reader_Amendment.md) gives the actual equations, expected derivatives and their limits. Both new arms retain two ordered MM→EM→EE→ME contextual cycles, the same original M/E object ownership, width 128, four heads, FP32, native physical objectives and one joint head. M and E remain input roles. No separate temperature/flow responsibility, heat-affine correction, pooled replacement or unrestricted record-read branch is introduced.

Full evaluates all relation-eligible original members with the new calibration. Guide retains all eligible M and chooses the shortest E prior-mass prefix with conditional prior omission at most 0.01, before its member MLP. Its deployed prediction reads only that independently calculated candidate set. Guide training averages native Full-reference and restricted physical losses on the same queries and parameters, with tiled checkpointed decoder intermediates; this additional reference work is charged. The auxiliary KL coefficient is zero. The old absolute rho-scaled value truncation is absent from both new executions.

Weighted score calibration uses prior-weighted mean and variance, the inherited additive unit scale floor and bounded tanh. Conditional preference and optional typed read strength remain separate. A common score shift cancels in the mean. Exact-zero applied relations and the hard prefix decision have no invented recovery derivative; the training Full reference supplies task feedback through positive eligible members omitted by the restricted path. A missing type or a singleton M scene can legitimately lack a particular gradient.

The actual Thermal input inventory is unchanged from the retained joint adapter. Physical coordinates, original IDs and measures are separate from normalized encoder features. E cell IDs follow the existing 24-by-8 catalogue. The encoder's 128-dimensional contextual records retain these owners through the applied exchanges; the decoder accesses the resulting original-member values through B.

The decoder's actual information measure is one unit per present M record and E physical cell area divided by the scene's positive mean E area. This is distinct from the preserved physical module-area metadata. In the uniform E192 catalogue the normalized E measures are one, so variation in its prior comes from the live relation mixture and proposals, rather than hidden geometric pruning.

Read necessity is represented continuously by rho. This execution has no learned hard typed-skip shortcut: a small contribution still incurs the reported candidate/value work. Exact empty contexts occur when the relevant path is absent or ineligible. The implementation does not claim that a low rho automatically saves an entire typed read.

| Inventory | Actual fields and ownership |
| --- | --- |
| M features, width 9 | Center x/Lx and y/Ly; radius/Lx and radius/Ly; solid/fluid diffusivities scaled by 100; solid/fluid conductivities; prescribed module heat divided by the TRAIN heat scale |
| E features, width 6 | Normalized cell coordinates followed by the existing four normalized boundary-distance components; no nearest-E cutoff or independent geometric member branch |
| Operating features, width 12 | Re/100, inlet velocity, viscosity and solid/fluid diffusivities scaled by 100, solid/fluid conductivities, radius, domain lengths scaled by 12 and 6, inlet and wall temperatures; supplied before contextual formation |
| Physical record metadata | Original M/E IDs and presence masks, native positions and frame, module area pi*r², environment cell area Lx*Ly/192, and characteristic lengths |
| Joint head / native observations | One u/v/p/T head; native extraction provides u, v, p, vorticity, fluid/surface/material temperature, interface q_proxy and original-module material-peak endpoint error |

## Identities, state and allocation

The fixed source is `round13_3_20261025/corrective_20261026/member_routing_relative_train600_100_v2/checkpoints/epoch_0500_model.pt` on DATA2, read in place. It is the retained R13-3-Relative epoch-500 state, not the changing latest alias. Both development arms strict-load the same 194 model-state tensors and 190 named AdamW states. The source has an inherited AdamW clock of 6500 per parameter and 300000 TRAIN sample visits and its own TRAIN600 normalization. The constant learning-rate schedule retains its original schedule horizon of 100 and rate 0.0003; the new execution horizon is a separate budget. Python, NumPy, CPU and two logical CUDA RNG streams and provider numerical counters are inherited. Only the changed reader's health baseline is reset at the parent epoch.

The new lineages are `full_e500_development` and `guide_e500_development` under the [round root](../../diagnostics/round13_4_20261026). Parent history remains explicitly labelled as Relative epochs 1–500; new reader/objective rows begin at absolute epoch 501. The first paired review ends at absolute epoch 600, representing 100 additional epochs per arm. The older `thermal_r13_3_relative_train600_1000_v2` lineage remains independently clean-paused at epoch 756 and is not resumed by this work.

The user's explicit allocation permits at most 1000 additional full-data epochs in aggregate, reserving 100 for one separately initialized final TRAIN600 candidate with its own TRAIN-only normalization. Development reviews occur every 100 additional epochs, with autonomous evidence-based continuation within the aggregate ceiling. The segmented ceiling is 500 epochs; actual qualification has consumed 12. Neither transferred nor segmented weights will be presented as the final scratch fit. The final scratch run stops at 100 and requires user direction for any later extension.

## Completed native diagnosis and selection

The initial replay used four fixed input-selected TRAIN cases: 0006/M1, 0051/M2, 0322/M5 and 0231/M12. It evaluated complete native objectives and backward paths, with bounded eight-query route traces at full access and omission tolerances 0.01 and 0.05. At 0.01, mean E candidates were 165, 153.875, 186.125 and 184.375; minimum captured all-eligible conditional preference mass was 0.987689, 0.989357, 0.992804 and 0.990528. Native E-relation gradients remained positive. The larger tolerance discarded more useful preference and caused larger physical-field changes. The conservative 0.01 decision and zero auxiliary coefficient were saved before fixed VAL detail evaluation.

The saved three-member arithmetic example reproduced the old exact-positive-count calibration jump; prior-weighted moments faded the negligible member continuously. A TRAIN-only four-scene, eight-query simplex oracle found almost no proposal-ranking room in M1 and substantially more in M12. These measurements support the initial simple candidate without a new address bank or selector network. They do not establish that prior mass is a physical error bound or that a different learned proposal is guaranteed to recover every omitted contribution.

The fixed TRAIN0003 and VAL0274/0293/0678 complete native detail replay saved fields, residuals, surface/material arrays, original IDs and native task gradients for the changed Full and restricted equations at the same parent weights. Weighted calibration changes the function immediately; those arrays are explicitly a new numerical reference, not a claim of exact equivalence to v2. No VAL target selected the tolerance. Native units follow the packed ThermalChannel fields and maintained interface target; no unsupported SI conversion or independent optimized physical maximum is implied.

## Qualification, failures and recovery

Guide qualification used the existing input-only TRAIN150/DEV22 membership and TRAIN150 normalization. A six-epoch segment covered twice the earlier three-epoch absorbing-null onset. A same-lineage checkpoint recovery then reached epoch 12, retaining contiguous history and exercising actual resume. The initial twelve-epoch health window covers four fixed low/high-M TRAIN cases at every epoch; subsequent development observations use the maintained cadence. This window establishes persistence over the observed onset and recovery, not a guarantee of future stability or architectural maturity.

Two zero-epoch failures were preserved and corrected. The first initial-health call requested a trace over the entire native union, exceeding the explicit decoder trace tile. The repair keeps complete native prediction, loss and control measurements while capping route traces at eight union queries. The second failure occurred before resumed optimization: sorted JSON manifest keys reordered a list of loss-role names in identity metadata. The repair uses the native canonical role order in v3 metadata; physical weights, parameter state and ordinary resume checks remain unchanged. Focused regression tests cover both failures. The failed roots, logs and time receipts remain in DATA2.

Across epochs 7–12, E native gradients through B, keys, scores and values were positive in all 24 panel observations. M value gradients were positive in all 24; M ranking gradients were positive in 18, with zeros confined to singleton M scenes. Native heat-control sensitivity remained measurable: mean Full-path fieldwise RMS change normalized by TRAIN scales was 0.003969 in epochs 1–6 and 0.003436 in epochs 7–12, while repeatability was approximately 0.0000022. Full and restricted sensitivities remained close. These measurements establish active information paths on this panel; they are sensitivity diagnostics rather than saved-reference response accuracy.

The scratch qualification remained broad: approximately 190 of 192 E candidates, mean relation-prior capture 0.990145 then 0.990237, and conditional-preference capture 0.989227 then 0.987784. The actual epoch-12 checkpoint, optimizer/RNG/sampler state, full history and visually inspected loss curves are retained. No end-to-end sparse savings have been established by this qualification.

![Native TRAIN150 information-path observations across twelve epochs, showing candidates, captured preference, typed strengths, task gradients and material-temperature sensitivity](../../diagnostics/round13_4_20261026/qualification_health_figure/native_health.png)

All 48 case observations retain native E-relation and original-value gradients. Candidate counts remain 190/192, while captured conditional preference and learned strengths vary during fitting. Material-temperature heat sensitivity remains above the corresponding repeated-input variation on this fixed panel. No native physical response accuracy or speedup follows from these curves. [PDF master](../../diagnostics/round13_4_20261026/qualification_health_figure/native_health.pdf); [saved plotted values](../../diagnostics/round13_4_20261026/qualification_health_figure/source.json).

![Guide TRAIN150 qualification through twelve epochs: native field losses, exposed DEV scores and the sum of Full and restricted reader work](../../diagnostics/round13_4_20261026/guide_train150_qual_retry1/plots/training/loss_curves.png)

The qualification curve retains epochs 1–12 across the actual six-to-twelve recovery. Exposed DEV field score decreases from 0.85614 at epoch 6 to 0.78934 at epoch 12; the summed paired-reader work remains broad. This is execution and learning-path qualification, not a mature predictor comparison. [PDF master](../../diagnostics/round13_4_20261026/guide_train150_qual_retry1/plots/training/loss_curves.pdf).

## Paired TRAIN600 development and the recommendation

Full and Guide both completed absolute epochs 501–600 from the same retained Relative epoch-500 state. Each retains 600 contiguous history rows, the epoch-600 numerical state and latest/best aliases. Source normalization, native losses, sampling, constant AdamW and all named optimizer moments were preserved. Full executed its first additional epoch on physical GPU 1 and recovered the same state for the remaining 99 on physical GPU 2, sharing with U-S. Guide executed its 100 additional epochs on physical GPU 1. No collapse pause, nonfinite computation, OOM or numerical recovery occurred in either arm. The older Relative lineage remains clean-paused at epoch 756.

The original low/high-M panel was observed at every one of the first twelve additional epochs and every ten thereafter, giving 21 saved observations per case over the full development segment. E native task gradients through applied B, contextual keys, member scores and original values remain active; M value gradients remain active, while singleton M proposal/ranking zeros are expected. Guide's eight logged native role losses all decreased from the first new epoch to epoch 600, by approximately 24–61%. The repeated native-control measurements retain fieldwise sensitivity above repeated-input variation. B/key/value and parameter gradients use the complete sampled Q1024 native objective: the deployed restricted reader for Guide and the all-eligible reader for Full. Proposal/member score gradients use the actual first decoder tile under a separate eight-fluid-query plus native surface/material objective; stored route-support summaries are capped at eight query rows. These observations establish sustained task connectivity on the declared panel, not a guarantee for every query or future epoch.

![Guide information-path health throughout the actual 100 additional TRAIN600 epochs, stratified by fixed module counts](../../diagnostics/round13_4_20261026/guide600_health_figure/native_health.png)

All four input-fixed cases retain native task gradients and nontrivial heat sensitivity over the observed trajectory. Candidate scope remains broad and varies by scene; the plotted conditional mass is an eight-query diagnostic, whereas actual complete-native work is measured separately. Native field changes are RMS divided by TRAIN scales; latent typed-vector changes elsewhere in the histories are L2 norms. This diagnostic is not physical response accuracy. [PDF master](../../diagnostics/round13_4_20261026/guide600_health_figure/native_health.pdf); [Guide summary](../../diagnostics/round13_4_20261026/guide600_health_review.json); [Full summary](../../diagnostics/round13_4_20261026/full600_health_review.json).

VAL89 is the existing exposed validation membership: the packed dataset originally calls this source partition `test`, and the protocol excludes duplicate 0273. Its module-count strata are M3/n24, M5/n25, M7/n25 and M10/n15. It is not a newly untouched test set. The separately fixed TRAIN health panel includes M1, M2, M5 and M12. No additional locked TEST membership is opened by these measurements.

| Native observable | Full e600 | Guide full reference e600 | Guide restricted e600 | Restricted / Full |
| --- | ---: | ---: | ---: | ---: |
| u | 0.04576779 | 0.04569694 | 0.04572214 | 0.99900 |
| v | 0.006949129 | 0.006929996 | 0.006935338 | 0.99802 |
| p | 0.02251879 | 0.02245543 | 0.0224513 | 0.99700 |
| Vorticity | 0.1213538 | 0.1211794 | 0.1212591 | 0.99922 |
| Fluid temperature | 1.115755 | 1.10498 | 1.105524 | 0.99083 |
| Surface temperature | 0.9576652 | 0.9570705 | 0.9639086 | 1.00652 |
| Material temperature | 0.9316697 | 0.8965248 | 0.9014434 | 0.96756 |
| Interface q_proxy | 2.489325 | 2.464278 | 2.470098 | 0.99228 |
| Module peak endpoint RMSE | 0.9601825 | 0.9500684 | 0.9580429 | 0.99777 |

These are equal-case mean native RMSEs from the maintained complete extraction, not a scalar ranking of the architecture. Temperature and q_proxy use their packed native targets; no unsupported SI conversion is applied. The [full mean/tail table](../../diagnostics/round13_4_20261026/review600/native_VAL89.csv) and [stratified review](../../diagnostics/round13_4_20261026/review600/review.json) retain all nine means, p90, worst values, case IDs, physical units and module-count strata. All new means improve over the historical v2 continuous epoch-500 parent by approximately 8–24%, but that comparison combines changed calibration and 100 further epochs. Full versus Guide shares the retained start; Guide's averaged two-reader objective differs from Full's objective, so this is a method comparison rather than an isolated causal estimate of one equation.

The same-checkpoint restriction changes Guide's mean errors by −0.018% to +0.839% relative to its full reference. That is much smaller than the prior v2 absolute-mass truncation penalty, but it is a measured approximation effect, not a mass-derived error guarantee. Tails remain mixed: worst material error is 4.8173 versus Full's 4.5189, and worst module-peak endpoint error is 5.1801 versus 4.8580, both at exposed VAL0283. This case was not added to TRAIN fitting or selected for a new diagnostic panel.

![Complete native VAL89 mean errors, tails, material-temperature strata and executed E member scoring work for the new Full and Guide](../../diagnostics/round13_4_20261026/review600/native_accuracy_work.png)

Guide restricted improves eight of nine mean errors against Full, with material temperature lower by 3.24% and surface temperature higher by 0.65%. E scoring averages 174.75 rows per native query versus 190.37 for Guide full; all eligible M remain. Only the connected Guide points share exactly the same endpoint parameters. The work axis displays forward member scoring, while the complete cost measurement below includes preparation and backward separately. [PDF master](../../diagnostics/round13_4_20261026/review600/native_accuracy_work.pdf).

![Full history of fieldwise native TRAIN contributions, inherited Relative epochs 1–500 and actual Full and Guide epochs 501–600](../../diagnostics/round13_4_20261026/review600/matched_learning.png)

The grey history belongs to the retained Relative parent; the two new colors begin at epoch 501. Both branches continue learning all eight native physical roles. Guide's curve averages full and restricted physical contributions, and should not be interpreted as full-VAL89 fidelity. The checkpoint histories also retain the separate paired-reader normalized native losses. [PDF master](../../diagnostics/round13_4_20261026/review600/matched_learning.pdf).

Development stops after these 200 aggregate additional epochs. The saved [review decision](../../diagnostics/round13_4_20261026/review600/decision.json) selects the existing Guide with epsilon 0.01, lambda zero, no refiner and no new address capacity. The evidence is sufficient to select the coherent next scratch candidate: sustained information paths, small same-checkpoint approximation, modest mean-fidelity gains and real early member restriction. It does not establish convergence or rule out later gains. The remaining 700 development epochs are deliberately unused. Larger omission, a KL campaign or more capacity is not justified by a diagnosed selection failure here; the measured limitations are broad prior coverage, mixed physical tails and indexing/calibration overhead. Independent scratch learning is the required next measurement.

## Original identities, relation evolution and useful access

The figures use saved original IDs and the historical fixed physical queries, matched into the actual native union. In VAL0678, present M IDs are 0–9 and E IDs 0–191; historical padded absent M records are excluded by identity. Contextual final ME is E donors sending to M receivers. Its B_E rows become M-address-to-E access views; a reverse index is not evidence of another contextual exchange.

![Run1404 latent membership, U-H original-record gates and Guide applied E-to-M contextual relation rows on the same VAL0678 scene](../../diagnostics/round13_4_20261026/figures/organization600_v2/three_way_organizer_Run1404_UH_Guide.png)

At shared native query ID 4289, Guide chooses 186/192 E before member scoring and executes those 186 original values, with rho_E 0.061. Its displayed M6/M7/M0 rows have 117/124/68 positive E relations. Run1404's latent membership and U-H's original-record gates are reused historical epoch-500 evidence, with different address and readout meanings; they are context references, not matched controls or physical-causal maps. [PDF master](../../diagnostics/round13_4_20261026/figures/organization600_v2/three_way_organizer_Run1404_UH_Guide.pdf).

![Literal original M7 and M0 relation rows from retained Relative epoch 500 to the new Guide epoch 600, with support overlap and normalized mass changes](../../diagnostics/round13_4_20261026/figures/organization600_v2/relation_evolution_Relative_e500_to_Guide_e600.png)

M7 changes from 126 to 124 positive E members, sharing 122 and adding two while dropping four; its support Jaccard is 0.9531 and raw row mass changes from 29.6259 to 26.5367. M0 changes from 76 to 68, sharing all 68 retained members, with Jaccard 0.8947 and mass 20.8052 to 16.8087. These distinct, overlapping rows evolve through applied contextual relations without a forced count or locality objective. [PDF master](../../diagnostics/round13_4_20261026/figures/organization600_v2/relation_evolution_Relative_e500_to_Guide_e600.pdf).

![Relation-derived full E prior, early permitted original IDs and final conditional preference at two fixed queries in TRAIN0003 and VAL0678](../../diagnostics/round13_4_20261026/figures/access600/relation_to_access_Guide_e600_TRAIN0003_VAL0678.png)

TRAIN0003/M1 retains 189/192 E at both fixed queries, capturing 98.12% and 97.84% of full-reference conditional preference; rho_E changes from 0.0151 to 0.0783. VAL0678/M10 retains 185/192, capturing 99.54% and 99.57%, with rho_E 0.0129 and 0.0771. Prior mass is plotted before candidate renormalization; the final p is conditional on actual candidates, and rho is reported separately. M1 cannot change its sole address ranking, while its member preference and read strength still respond to q. Broad access remains an explicit outcome. [PDF master](../../diagnostics/round13_4_20261026/figures/access600/relation_to_access_Guide_e600_TRAIN0003_VAL0678.pdf).

Small no-fit TRAIN0006/M1 and TRAIN0231/M12 interventions hold checkpoints and query draws fixed. Reassigning whole B_E rows among present M, while preserving B_M, context and original identities, is an identity control in M1; in M12 it changes normalized material-temperature prediction RMS by 0.2131 and increases native material RMSE from 0.8200 to 2.1668. Removing row-mass differences changes M12 material prediction RMS by 0.06520 and RMSE to 0.8988. Thus row assignment and mass are numerically used by this reader. These out-of-distribution interventions demonstrate dependence, not unique physical necessity or causal discovery.

Keeping only the strong 90% measure-weighted prefix within each applied E row changes M12 material prediction RMS by 0.03049; restoring the omitted tail returns it within 2.1e-7 normalized RMS of the baseline. A graph-selected positive long-range final-ME message was then removed before contextual aggregation and all dependent context was re-prepared. The actual selected edges are M0<-E20 in M1 and M4<-E25 in M12, with original applied coefficients 0.7051 and 0.7217; the recomputed coefficient is zero. Material prediction changes by 0.001076 and 0.002053 normalized RMS. Complete field metrics, changed coefficients, stage provenance, original IDs, repetition variation and executed work are retained in the [intervention evidence](../../diagnostics/round13_4_20261026/counterfactuals_guide600.json). No physical response reference or solver validates these synthetic message changes.

## Native physical reconstruction and original-module detail

The following input-fixed panel uses complete native fields and validity masks. Columns are target, Full all-eligible, Guide restricted and their residuals; u and fluid/valid-material T share target-derived scales, with residual scales shared across cases. Source paths, checkpoint ages, original IDs and prescribed heat controls appear in the saved figures and manifests. Native module detail uses all 3096 material receivers at original module ID 0, aligned by original module_slot_ids; geometry comes from the unchanged historical extraction, never from borrowed historical predictions.

![Native u and temperature fields and residuals for fixed TRAIN0003 with one original module](../../diagnostics/round13_4_20261026/figures/physical600/native_fields_case_0003.png)

TRAIN0003 has native material-temperature RMSE 0.4399 for Full and 0.4059 for Guide restricted, while fluid-temperature error is 0.8433 and 0.8581. The wake and heat plume are reconstructed with remaining spatial residuals; a mean improvement on one role does not imply every field improves. [PDF master](../../diagnostics/round13_4_20261026/figures/physical600/native_fields_case_0003.pdf).

![Native fields and residuals for fixed exposed VAL0274 with three original modules](../../diagnostics/round13_4_20261026/figures/physical600/native_fields_case_0274.png)

VAL0274 has u RMSE 0.03477 versus 0.03455 and material-temperature RMSE 0.3814 versus 0.4194, Full versus Guide. Guide improves the fluid-temperature field here but worsens original-module thermal detail. [PDF master](../../diagnostics/round13_4_20261026/figures/physical600/native_fields_case_0274.pdf).

![Native fields and residuals for fixed exposed VAL0293 with seven original modules and interacting wakes and heat plumes](../../diagnostics/round13_4_20261026/figures/physical600/native_fields_case_0293.png)

VAL0293 remains a difficult fixed-panel field: u RMSE is 0.07056 versus 0.07141, and material-temperature RMSE is 2.1782 versus 2.2208. Both readers miss parts of the downstream wake/plume structure; these residuals are not resolved by healthy routing alone. [PDF master](../../diagnostics/round13_4_20261026/figures/physical600/native_fields_case_0293.pdf).

![Native fields and residuals for fixed exposed VAL0678 with original modules 0–9](../../diagnostics/round13_4_20261026/figures/physical600/native_fields_case_0678.png)

VAL0678 has u RMSE 0.04429 versus 0.04385 and material-temperature RMSE 0.6792 versus 0.7113. The organized original-member reader retains the multi-module plume, while local thermal residuals remain. [PDF master](../../diagnostics/round13_4_20261026/figures/physical600/native_fields_case_0678.pdf).

![Aligned original-module material residual maps, per-original-ID material peaks and native surface-temperature residuals across all four fixed cases](../../diagnostics/round13_4_20261026/figures/physical600/native_module_surface_detail.png)

The top row shows actual material detail at original ID 0, the middle row compares receiver-defined peaks across every original module, and the bottom row shows all 64 native surface receivers. These are saved native receiver maxima, not independently optimized physical maxima. VAL0293 retains peak underprediction and a pronounced angular surface residual; the two new methods remain close in that case. [PDF master](../../diagnostics/round13_4_20261026/figures/physical600/native_module_surface_detail.pdf).

## Work, runtime and memory

On complete VAL89 native extraction, Guide's full reference scores 190.37 E per union query and restricted execution scores 174.75, a reduction of 8.20%; all 5.865 mean M records remain. The scalar B composition still charges 1166.18 address/member pairs per query on average, and the restricted executor sorts the E catalogue before scoring. Context exchange, original key/value construction, index preparation, all candidate scores, calibration, gather/reduction and native extraction are retained; a small rho does not cancel these costs. Paired Guide fitting explicitly pays for both physical readers and checkpoint recomputation.

![Measured complete native inference and backward time with executed original E scoring on fixed low- and high-module TRAIN cases](../../diagnostics/round13_4_20261026/figures/cost600_v2/native_costs.png)

On isolated physical GPU 1, FP32 native union Q8192, median complete loaded-batch inference is 129.9 versus 176.0 ms in M1 and 174.1 versus 181.2 ms in M12, full versus restricted. E scores decrease 190 to 167 and 192 to 183.33 per query, respectively. One native forward/backward without an optimizer update costs 858 versus 769 ms in M1 and 583 versus 588 ms in M12; this single backward timing is not a statistical speed estimate. Three measured inference repetitions follow one warmup. The full-process receipts additionally charge HDF5 loading, setup and instrumented profiling; inclusive profiler ranges overlap and must not be added. [PDF master](../../diagnostics/round13_4_20261026/figures/cost600_v2/native_costs.pdf); [timings, memory, executed hooks and profiler ranges](../../diagnostics/round13_4_20261026/profile_guide600.json).

Peak allocated GPU memory in the native profile is 0.213/0.197 GiB for M1 full/restricted inference and 0.225/0.218 GiB for M12. Native forward/backward peaks are 0.528/0.480 GiB and 0.567/0.553 GiB. Reserved memory ranges from 0.266 to 0.586 GiB and carries the allocator history of earlier calls; it is not independent per-mode fresh-process memory. Actual member-E MLP hooks count 3,112,960/2,736,128 rows in the M1 full/restricted backward calls and 3,145,728/3,003,706 in M12, including checkpoint recomputation; forward semantic counters alone would charge half those rows. Training uses much larger effective batches, and its per-epoch memory is reported separately from this single-scene profile.

These measurements identify a broad-candidate and execution-overhead limitation, not a sparse speedup. At this scale, prefix sorting, indexing, reductions, native preparation and host/device bookkeeping offset the small member-MLP saving. The nonlinear member MLP remains a substantial cost, but replacing it or adding a different selector is not justified as a rescue of a healthy physical reader. Exact affine factoring, caching and reduced scalar synchronization remain focused future optimization options, to be verified separately; none is silently applied to this comparison.

Complete training processes took 50.11 seconds for Full's initial epoch-501 attachment, 5423.33 seconds for its subsequent 99 epochs on shared GPU 2, and 5035.66 seconds for Guide's 100 epochs on GPU 1. These device-contention differences prevent interpreting training wall time as an isolated Full/Guide speed ratio. Across the new 100-epoch history rows, Full reaches 12.532 GiB allocated and 31.986 GiB reserved, including its initial epoch on GPU 1; Guide reaches 12.518 GiB allocated and 45.564 GiB reserved. These are complete training-epoch peaks excluding the separately recorded monitoring callbacks. The native profile above uses one loaded scene, not the training effective batch. The histories retain per-epoch device memory and native work. No GPU-utilization integration is claimed by process-associated accounting.

## Independent final candidate: in progress

The recommended Guide is being trained as `guide_scratch_train600` with the maintained [scratch configuration](../../configs/research/thermal_member_access_r13_4_guide_scratch_train600_v3.json): independent seed-0 initialization, no source checkpoint, its own freshly built TRAIN600 normalization, identical native targets and constant AdamW, paired physical losses and epsilon 0.01. The first twelve full-TRAIN600 epochs completed in 651.90 seconds, preserving twelve contiguous history rows and 190 named AdamW states, each at clock 156, plus model, RNG, sampler and provider state. Original M/E paths remained active across all four fixed low/high-M cases, with conditional preference capture 0.993514–0.995867 at epoch 12. The same lineage has now actually resumed from epoch 12 to its fixed epoch-100 endpoint; this run inherits neither transferred nor segmented weights. It will stop at 100 and remain resumable; any later extension requires user direction. Its final endpoint fields and closeout remain pending.

## Focused software validation

The initial integration command passed 66 checks covering the new core/task/transfer capability, Relative behavior, health, maintained recovery and training output. After the real resume metadata failure, its focused task module passed nine tests including the new sorted-manifest regression. The legacy interaction/Relative reader command passed 18 tests with one CUDA-only replay fixture skipped in the CPU command; real CUDA native checkpointed training and the new capability's actual six-to-twelve recovery completed separately. These are overlapping command scopes, not a sum of unique tests. Their logs remain in the round root. The completed transferred trajectories and native evaluation provide empirical evidence beyond these software checks; independent scratch completion remains pending.

A focused final review found that the v2-to-v3 attachment omitted comparison of the provider's already saved dataset/manifest source binding while comparing other identity fields. The repair compares that existing metadata without new hashes, snapshots or services; it affects only future attachments and changes no reader equation or active scratch execution. Both actual parent/Full/Guide saved bindings match, and CPU replay of the repaired attachment on their literal identities passes. Nine transfer checks, including changed/missing-binding regressions and the horizon-only extension check, passed in 1.13 seconds. [Saved binding comparison](../../diagnostics/round13_4_20261026/existing_binding_comparison.json).

## Evidence, costs and continuation

Verified superseded-export cleanup has removed 24 additional PDF/PNG files totaling 12,503,517 bytes after their selected replacements were decoded and inspected. The exact allowlist and receipt remain on DATA2; numerical evidence, source helpers, all checkpoints and active readers were preserved. Final temporary-cache cleanup waits for the scratch run to finish.

The [attempt ledger](../../diagnostics/round13_4_20261026/attempt_ledger.json), [initial TRAIN decision](../../diagnostics/round13_4_20261026/initial_decision.json), [qualification decision](../../diagnostics/round13_4_20261026/qualification_decision.json), [resume diagnosis](../../diagnostics/round13_4_20261026/resume_identity_diagnosis.json) and [process-cost receipts](../../diagnostics/round13_4_20261026/process_costs.json) preserve actual attempts and their scope. Complete GPU-associated process time includes CPU setup, extraction, diagnosis, validation and saving; inherited parent compute is not double-charged as new fitting. CPU tests, rendering and quiet read-only Luna monitoring are separate. The initial retained-payload budget is 4 GiB, disposable peak 512 MiB, with DATA2/home headroom of 100/20 GiB. No dataset, geometry catalogue or parent checkpoint copy was created.

The maintained entrypoint is `HONF_Proj/tools/honf_member_access.py`. The actual qualification commands below completed from the project directory using the ModularDT environment. Device ordinals are resolved through the declared visible-device mapping; `cuda:1` here is physical GPU 1.

```bash
cd /home/wanglz/Desktop/src/ModularDT/HONF_Proj
CUDA_VISIBLE_DEVICES=0,1 /home/wanglz/miniconda3/envs/ModularDT/bin/python tools/honf_member_access.py train --config configs/research/thermal_member_access_r13_4_guide_qual_train150_v3.json --output /data2/wanglz/ModularDT/round13_4_20261026/guide_train150_qual_retry1 --device cuda:1 --stop-after-epoch 6
CUDA_VISIBLE_DEVICES=0,1 /home/wanglz/miniconda3/envs/ModularDT/bin/python tools/honf_member_access.py resume --run /data2/wanglz/ModularDT/round13_4_20261026/guide_train150_qual_retry1 --device cuda:1 --stop-after-epoch 12
```

The independent final candidate's real train/resume sequence is:

```bash
cd /home/wanglz/Desktop/src/ModularDT/HONF_Proj
CUDA_VISIBLE_DEVICES=0,1 /home/wanglz/miniconda3/envs/ModularDT/bin/python tools/honf_member_access.py train --config configs/research/thermal_member_access_r13_4_guide_scratch_train600_v3.json --output /data2/wanglz/ModularDT/round13_4_20261026/guide_scratch_train600 --device cuda:1 --stop-after-epoch 12
CUDA_VISIBLE_DEVICES=0,1 /home/wanglz/miniconda3/envs/ModularDT/bin/python tools/honf_member_access.py resume --run /data2/wanglz/ModularDT/round13_4_20261026/guide_scratch_train600 --device cuda:1 --stop-after-epoch 100
```

The first command completed; the second restored epoch 12 and is running to 100. No extension beyond 100 is launched. A later user-directed horizon-extension command will be recorded with the completed final checkpoint; its execution-horizon-only amendment is covered by the focused transfer test, while native recovery is being exercised by this actual same-lineage run.

The qualification commands completed; existing run directories must not be reused for a fresh `train`. Both development lineages ended at epoch 600, including Full's actual state-preserving resume from 501 to 600. Scratch recovery has restored epoch 12 and remains bounded by epoch 100. Final scratch evidence, closeout costs and temporary-cache cleanup remain pending.
