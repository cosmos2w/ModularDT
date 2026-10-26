# HONF R13-4: relation-guided original-member access

**Status: experiment in progress.** The maintained implementation and native qualification are complete; the first paired TRAIN600 development segment is running. This document distinguishes those completed measurements from the pending field comparison, recommendation and independent scratch candidate. It will be updated from saved evidence at the development review.

R13-4 moves the E candidate decision ahead of the expensive query–member scorer. The decision uses a prior composed from the model's actual applied final-stage relations, query proposals and retained record information measures. Selected original M/E records retain their identities, geometry, controls and contextual values, and feed one general joint physical head. The new weighted member-score calibration also removes the exact-positive-count boundary found in the previous numerical reference. Historical readers and checkpoints retain their loading behavior.

For the predictor, completed evidence establishes finite native objectives, task gradients and input dependence through real training and recovery; it does not yet establish improved full-population field accuracy. For the organizer, the early selector executes and retains live task feedback, but the qualified scratch organizer is broad. Broad healthy access is a measured limitation to investigate, rather than an automatic failure. No new inverse experiment or solver campaign has been performed.

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

Across epochs 7–12, E native gradients through B, keys, scores and values were positive in all 24 panel observations. M value gradients were positive in all 24; M ranking gradients were positive in 18, with zeros confined to singleton M scenes. Native heat-control sensitivity remained measurable: mean full-path fieldwise L2 change was 0.003969 in epochs 1–6 and 0.003436 in epochs 7–12, while repeatability was approximately 0.0000022. Full and restricted sensitivities remained close. These measurements establish active information paths on this panel; they are sensitivity diagnostics rather than saved-reference response accuracy.

The scratch qualification remained broad: approximately 190 of 192 E candidates, mean relation-prior capture 0.990145 then 0.990237, and conditional-preference capture 0.989227 then 0.987784. The actual epoch-12 checkpoint, optimizer/RNG/sampler state, full history and visually inspected loss curves are retained. No end-to-end sparse savings have been established by this qualification.

![Native TRAIN150 information-path observations across twelve epochs, showing candidates, captured preference, typed strengths, task gradients and material-temperature sensitivity](../../diagnostics/round13_4_20261026/qualification_health_figure/native_health.png)

All 48 case observations retain native E-relation and original-value gradients. Candidate counts remain 190/192, while captured conditional preference and learned strengths vary during fitting. Material-temperature heat sensitivity remains above the corresponding repeated-input variation on this fixed panel. No native physical response accuracy or speedup follows from these curves. [PDF master](../../diagnostics/round13_4_20261026/qualification_health_figure/native_health.pdf); [saved plotted values](../../diagnostics/round13_4_20261026/qualification_health_figure/source.json).

![Guide TRAIN150 qualification through twelve epochs: native field losses, exposed DEV scores and the sum of Full and restricted reader work](../../diagnostics/round13_4_20261026/guide_train150_qual_retry1/plots/training/loss_curves.png)

The qualification curve retains epochs 1–12 across the actual six-to-twelve recovery. Exposed DEV field score decreases from 0.85614 at epoch 6 to 0.78934 at epoch 12; the summed paired-reader work remains broad. This is execution and learning-path qualification, not a mature predictor comparison. [PDF master](../../diagnostics/round13_4_20261026/guide_train150_qual_retry1/plots/training/loss_curves.pdf).

## Transferred development: measured start and pending review

Full completed one real TRAIN600 epoch after attachment, ending at absolute epoch 501 with 600 case visits and 13 new optimizer updates. Its complete history retained the 500 parent rows. Native training took 26.97 seconds, and the complete process took 50.11 seconds including setup, health and saving. Peak allocated/reserved memory was 13.32/34.35 GB in that process on GPU 1. All four panel cases retained active M/E paths and measured heat-control sensitivity; singleton ranking zeros remained legitimate.

Full is now recovering that same lineage through absolute epoch 600 on physical GPU 2, sharing with the preserved U-S job. Guide starts independently from the identical retained parent and runs through 600 on physical GPU 1. Two new epochs measured roughly 54–56 seconds of Full training per epoch under contention, versus 49–50 seconds for paired Guide on its primary device. These training times are not a fair isolated inference benchmark; complete process costs and later fixed-workload measurements are reported separately. Neither launch changes the protected GPU 0 job or the U-S writer.

The pending 100-additional-epoch review will measure all nine canonical VAL89 observables, tails and module-count strata, the fixed native detail panels, applied-relation evolution, candidate utility and actual work. Three-way Run1404/U-H/new-Guide organizer maps will reuse the historical numerical evidence and align actual original IDs, including absent padded records. Those historical models have different readout priors and are context references rather than matched causal controls. A recommendation and final scratch training remain pending.

VAL89 is the existing exposed validation membership: the packed dataset originally calls this source partition `test`, and the protocol excludes the known duplicate 0273. It is not a newly untouched test set. No additional locked TEST membership is opened by these measurements.

## Focused software validation

The initial integration command passed 66 checks covering the new core/task/transfer capability, Relative behavior, health, maintained recovery and training output. After the real resume metadata failure, its focused task module passed nine tests including the new sorted-manifest regression. The legacy interaction/Relative reader command passed 18 tests with one CUDA-only replay fixture skipped in the CPU command; real CUDA native checkpointed training and the new capability's actual six-to-twelve recovery completed separately. These are overlapping command scopes, not a sum of unique tests. Their logs remain in the round root. The paired development and independent final scratch outcomes are still empirical requirements beyond these software checks.

## Evidence, costs and continuation

The [attempt ledger](../../diagnostics/round13_4_20261026/attempt_ledger.json), [initial TRAIN decision](../../diagnostics/round13_4_20261026/initial_decision.json), [qualification decision](../../diagnostics/round13_4_20261026/qualification_decision.json), [resume diagnosis](../../diagnostics/round13_4_20261026/resume_identity_diagnosis.json) and [process-cost receipts](../../diagnostics/round13_4_20261026/process_costs.json) preserve actual attempts and their scope. Complete GPU-associated process time includes CPU setup, extraction, diagnosis, validation and saving; inherited parent compute is not double-charged as new fitting. CPU tests, rendering and quiet read-only Luna monitoring are separate. The initial retained-payload budget is 4 GiB, disposable peak 512 MiB, with DATA2/home headroom of 100/20 GiB. No dataset, geometry catalogue or parent checkpoint copy was created.

The maintained entrypoint is `HONF_Proj/tools/honf_member_access.py`. The actual qualification commands below completed from the project directory using the ModularDT environment. Device ordinals are resolved through the declared visible-device mapping; `cuda:1` here is physical GPU 1.

```bash
cd /home/wanglz/Desktop/src/ModularDT/HONF_Proj
CUDA_VISIBLE_DEVICES=0,1 /home/wanglz/miniconda3/envs/ModularDT/bin/python tools/honf_member_access.py train --config configs/research/thermal_member_access_r13_4_guide_qual_train150_v3.json --output /data2/wanglz/ModularDT/round13_4_20261026/guide_train150_qual_retry1 --device cuda:1 --stop-after-epoch 6
CUDA_VISIBLE_DEVICES=0,1 /home/wanglz/miniconda3/envs/ModularDT/bin/python tools/honf_member_access.py resume --run /data2/wanglz/ModularDT/round13_4_20261026/guide_train150_qual_retry1 --device cuda:1 --stop-after-epoch 12
```

These commands document completed execution and recovery; the existing run directory must not be reused for a fresh `train`. Development and final scratch extension commands will be recorded with their actual checkpoints after their respective reviews. Cleanup, selected final figures, complete endpoint costs and the final recommendation are not yet complete.
