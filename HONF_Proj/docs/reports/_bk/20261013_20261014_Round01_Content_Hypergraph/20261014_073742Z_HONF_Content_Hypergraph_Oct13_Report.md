# Oct13 shared attention and content-hypergraph research report

The new opt-in family uses jointly trained typed attention and a late, content-bearing edge bank. A-D reads contextualized physical sources directly; A-H6 reads six learned edge contents and a compact local branch. Thermal has one affine heat-response model for all native temperature observations; Wind predicts three-dimensional native velocities through its own physical transform. TrainingEngine, deterministic sampling, TRAIN normalization and native observation code are reused. The new path contains no frozen D-sep, Stage-A fit or pair-value contextualizer.

The first board uses the literal e2500 matched development arms. Accuracy is measured on native exposed DEV22/DEV24; latency is the complete high-M B1 application; training memory is measured at effective batch 48/24 and Q1024/4096, respectively. Decimal GB are used, and reserved memory includes allocator residency.

| Measured primary result, A-D / A-H6 | Thermal | Wind |
|---|---|---|
| Priority field RMSE | Fluid T: 2.4744 / 3.3028 native T units | Near-turbine vector: 0.6431 / 0.6428 m/s |
| Flow detail | u: 0.06172 / 0.08847; v: 0.01037 / 0.01061 native units | Downstream vector: 0.4555 / 0.4771 m/s |
| Wind secondary components | Not applicable | Near Uy: 0.10993 / 0.10735; Uz: 0.06436 / 0.06153 m/s |
| Complete high-M native prediction | 17.20 / 19.59 ms | 7.72 / 8.70 ms |
| Actual training peak allocated | 2.420 / 1.619 GB | 1.886 / 0.648 GB |
| Actual training peak reserved | 11.148 / 4.452 GB | 3.345 / 0.847 GB |
| Edge content affects physical output | H6: yes, native temperature interventions | H6: yes, up to 1.404 m/s probe change |

**Predictor — gains, misses, evidence and next step.** All four primary fits completed the same segmented 2,500-epoch schedule. Both models improve substantially over inexpensive input-only baselines. On native exposed DEV, Wind A-H6 is close to A-D (field score 0.20558 versus 0.19483; near-turbine vector RMSE 0.64282 versus 0.64306 m/s), while Thermal A-H6 falls behind (0.12543 versus 0.06240; fluid-temperature RMSE 3.3028 versus 2.4744 native units). The original Thermal pair therefore fails the automatic full-TRAIN guard. The sole Thermal revision widens the shared global/local reader message width from 64 to 128 while preserving the context, six edges, rank 4, losses, queries and schedule. It improves final native field score to 0.11080 but remains 1.775 times A-D and also fails the follow-up bound; no full-TRAIN Thermal run launches. The qualified Wind architecture completes one fresh full-TRAIN run at the enforced 1,000 stop, with one documented numerical pooling repair and saved-state continuation; its final native/reference board is reported separately below. Mature classics remain markedly more accurate on the development board.

**Organizer — gains, misses, evidence and next step.** Saved incidence, measure-weighted pooling, edge contents, receiver access and actual messages are connected to the physical predictions by unchanged-weight interventions. Edge removal changes Wind probe velocities by up to 1.404 m/s and strongly changes Thermal temperature. The bank is used, even though module incidence is concentrated. Thermal's original temperature contribution is almost entirely carried by edge 4 in the inspected scenes; Wind reads several contents despite module-incidence concentration on edge 0. This is descriptive computational organization, without a physical-causality or uniform-occupancy claim. The next useful question is whether content diversity and broad-field fidelity improve together, rather than whether an incidence histogram looks balanced.

**Inverse interface — gains, misses, evidence and next step.** The implementation exposes prepared affine heat application, heat JVP/VJP, native transpose extraction and geometry derivatives. Saved heat-transfer replays distinguish exact software algebra from physical response accuracy: the precise factorized FP64 path passes the predeclared numerical law, while old FP32 endpoint-subtraction failures remain visible. Physical response errors and missed endpoint peaks remain; three saved independent transfer families and smooth software VJP fixtures do not establish valid inverse designs. No inverse generator, new physical solve or new solver attempt ran; solver use remains 326/326.

The mathematical specification is [HONF_Redesign_Oct13.md](../../../../UpgradePlan/ThemalChannel/20261013_20261014_Round01_Content_Hypergraph/20261013_HONF_Redesign_Oct13.md), SHA256 `6887c5436f6006fa8b6451cca5abe753e08a2d27ca3fed08cbbd336caeb63a44`; the operating contract is [the execution plan](../../../../UpgradePlan/ThemalChannel/20261013_20261014_Round01_Content_Hypergraph/20261014_025503Z_HONF_Redesign_Implementation_and_Evaluation_Oct13.md). The [evidence index](20261014_073742Z_HONF_Content_Hypergraph_Oct13_Evidence_Index.md) maps exact paths, checkpoint identities, commands and retained failures. All measurements in this report come from this round's saved artifacts. Figures and numerical arrays remain local and ignored, under `$ROUND=/data/wanglz/ModularDT/diagnostics/content_hypergraph_oct13_20261014T015930Z`; the standard Markdown embeds resolve through the repository's ignored round symlink.

## Implementation and scientific identity

The shared core has width 128, four attention heads and two simultaneous typed MM/ME/EM updates, with no E→E update. Donor K/V projections are reused; physical measures enter attention/pooling, source masks and self-exclusion remain explicit, and empty types are handled. Both decoders use query-consistent pointwise reads and radius-limited local rows. A-D omits only the final environment destination update that has no A-D consumer; its simultaneous module update still reads the same old environment state. Common named tensors have bitwise identical initialization under a common seed. A-H6 retains the final environment states because its organizer consumes them.

The default bank has six content edges with full softmax access, width 64 global/local messages and signed Thermal rank 4 modes per edge. Pooling incidence B, normalized donor weights W, edge masses, anchors/scales and C[B,H,r,M] source coefficients are prepared once per scene. Source heat enters only the late affine contraction; heat-null flow and T(h=0)=0 hold. There is no mandatory dense K[Q,M] materialization. Prepared ownership, weight/buffer/scene versions, legal physical-control axes and query versions reject stale reuse. Cached affine factors support repeated controls and transpose products without rerunning the neural readers.

Thermal uses a 24×8 environment and the native cell-centered grid. u/v/p are denormalized before the maintained native curl; one predicted temperature field feeds fluid, 64-angle surface, all 3,096 material points per module and the conductivity/spacing-dependent q proxy. The compact radius is 3r. Native endpoint metrics exclude solids using the actual shared mask and use endpoint maxima over common native material receivers. Monitoring uses sampled auxiliary receivers, so its module maxima are not relabelled as full-native maxima. Temperature/flow units below are packed-dataset native units, and q is a proxy rather than a new calibrated flux claim.

Wind uses E64=4×4×4 geometry-only tokens and prescribed operating inputs, seven query features, eleven operating features and its native wind-aligned 3D frame. It does not rotate the already aligned native frame again. The physical transform is `U=height_profile(z)+normalized_residual×safe_std×u_ref`; adding the absolute-velocity normalizer mean would introduce a false offset and is rejected by adapter-state version 2. An independent oracle invokes the maintained native physical transform, including the zero-residual→height-profile limit. The local radius is 1.5D. Historical Wind references keep their own E512 catalogues and transforms.

Only physical GPU0 (`GPU-233fcd85-5c6a-6212-3f44-655252afec70`) and GPU1 (`GPU-3ceda40c-fd5c-4b88-6c47-b3301711571e`) were authorized for computation. Scientific commands use `/home/wanglz/miniconda3/envs/ModularDT/bin/python`, PyTorch 2.6.0+cu124, FP32, TF32 off and no autocast. Training and main CPU checks use four threads; the historical Wind CPU latency matrix uses two, with comparisons confined to that matrix. Thermal fits and native cost matrices use GPU0; Wind fits and native cost matrices use GPU1. The data-free Wind optimizer-equivalence diagnostic uses idle GPU0. Costs are compared within a dataset on the same device. GPT-6 Luna max subagents implemented and monitored the assigned components; the main thread reviewed interfaces, numerical contracts, native boards and figures.

## Matched development protocol and measured work

Thermal uses the frozen stratified fixed25 manifest: TRAIN150 and exposed DEV22, preserving all module-count categories and original split separation. Wind uses fixed24: TRAIN72 rows from 24 layouts and DEV24 rows from 8 layouts, all three native wind directions per selected layout. TRAIN-only normalization is shared within each pair, with Thermal seed 0 and Wind seed 42. These are exposed development populations; the Thermal subset comes from the source H5 TEST catalogue by the declared development protocol, while unexposed Thermal targets and Wind TEST targets remain locked. Development evidence is neither full-population nor independent-test evidence.

Both arms use identical case/epoch/query keys, losses, schedules and budgets. Thermal uses primary Q1024, effective batch 48 and packing 24; Wind uses Q4096 total across five roles, effective batch 24 and packing 12. AdamW uses 20 warmup epochs, peak 3e-4, hold through 1000 and cosine decay to 3e-6 at 2500; decay is 1e-4 Thermal and 1e-5 Wind, with gradient clipping 1. Thermal explicitly excludes biases/norms; the initial Wind implementation incorrectly put them in the decay group. Both Wind primary arms and its fresh full-data run share that recorded declaration error. Thermal flow terms each have weight 1/8; fluid/surface/material temperature each 1/6, q proxy 0.05, and the retained original-TRAIN response addendum 0.1 is attached once at the final macro-update per epoch. No DEV response record is used for training.

The corrected maintained Wind recipes explicitly exclude biases/norms. Original recipes, identities and named optimizer groups remain unchanged for replay. For the recorded FP32 AdamW implementation, the maximum decay product is 3e-9, below half an FP32 ULP beneath one; every declared learning-rate epoch therefore gives a decay multiplier exactly equal to one. Actual CPU AdamW steps from both trained e2500 parameter/moment inventories, at four declared learning-rate values, give bitwise identical parameters and moments under the original and corrected groups (16 disposable steps). The matching CUDA check also passes bitwise for all parameter/moment tensors (16 further disposable steps). This arithmetic exception does not make the original metadata compliant and does not establish equivalence for FP64, other learning rates or another optimizer backend.

| Completed primary arm | Epochs | Case visits | Optimizer updates | Microbatches | Primary receiver queries | Measured train-loop time |
|---|---:|---:|---:|---:|---:|---:|
| Thermal A-D | 2500 | 375,000 | 10,000 | 17,500 | 384,000,000 | 1,807.51 s |
| Thermal A-H6 | 2500 | 375,000 | 10,000 | 17,500 | 384,000,000 | 1,826.33 s |
| Wind A-D | 2500 | 180,000 | 7,500 | 15,000 | 737,280,000 | 1,630.02 s |
| Wind A-H6 | 2500 | 180,000 | 7,500 | 15,000 | 737,280,000 | 1,622.43 s |

The paired case-order and query-sampling hashes match at every completed primary epoch. Each Thermal arm actually processed 1,213,375,693 native union rows, 1,002,693,616 compact local pairs and 11,201,859 separate response-addendum union rows; the union count includes native interpolation/stencil requirements and exceeds the primary-query count. Each Wind arm processed 737,280,000 native receivers; H6 executed 4,423,680,000 receiver-edge slots. Wind's 13,271,040,000 active source-receiver capacity is a logical support count, not a claim about padded executor rows; benchmark traces separately record actual padded reads. Train-loop times include sampling/preparation, forward/backward, optimizer and the Thermal TRAIN response addendum, and exclude monitoring, process startup and separate evaluation.

Reviews at 100/500/1000 found finite gradients, actual shared-parameter updates, nontrivial loss reduction and matched streams, allowing both original pairs to mature to 2500. Fixed-weight TRAIN evaluation is separate from the changing-weight training average. At 500, Thermal TRAIN/DEV monitoring scores were 0.2810/0.3042 A-D and 0.2246/0.2606 H6; at 1000, 0.1125/0.1464 and 0.1110/0.1527. At 2500, 0.03576/0.08089 and 0.05322/0.15651. Wind's corresponding 2500 TRAIN/DEV monitor scores were 0.17667/0.19331 A-D and 0.15818/0.20466 H6. These comparisons use the same checkpoint weights and declared epoch sampling, and expose a Thermal fit/generalization limitation.

Literal common-age comparisons remain separate from selected aliases. The selected Thermal A-D/H6 aliases are epochs 2100/1300, with native DEV22 scores 0.06235/0.10281; Wind aliases are 2300/2400, with native DEV24 scores 0.19471/0.20500. Every declared 100-epoch milestone, latest state, selected state and history is retained; no 25-epoch or duplicate selector campaign was introduced.

The [experiment registry](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/evaluation/final_tables/experiment_registry.json) retains every run's recorded/final identities, completed and selected ages, milestone hashes, exact memberships/transforms, query/loss/schedule/precision contracts, measured work and source amendments. Failed and repeated work remains separate from the retained checkpoint trajectory.

## Native field accuracy and mature-reference distance

Thermal field selection is one-half the mean TRAIN-standardized squared error of u/v/p/omega and one-half the mean of fluid/surface/material temperature errors. q and endpoint maxima are reported separately. The native board uses all 8192 cell-grid receivers before applying the shared fluid mask, all 64 surface angles and all 3096 stored material receivers per active module. The table reports means of case RMSEs on the same 22 cases; mature references use separately qualified original public CPU inference and their own TRAIN transforms.

| Thermal native DEV22 metric | A-D e2500 | A-H6 e2500 | Run 1404 e5000 | Run 1804 e5000 |
|---|---:|---:|---:|---:|
| u RMSE | 0.06172 | 0.08847 | 0.01122 | 0.006451 |
| v RMSE | 0.01037 | 0.01061 | 0.0005143 | 0.0003281 |
| p RMSE | 0.02965 | 0.03646 | 0.003232 | 0.002184 |
| Native curl RMSE | 0.16717 | 0.20748 | 0.02502 | 0.02332 |
| Fluid T RMSE | 2.47444 | 3.30279 | 0.30004 | 0.22349 |
| Surface T RMSE | 2.03836 | 3.30899 | 0.47095 | 0.39093 |
| Material T RMSE | 1.92250 | 3.09481 | 0.37256 | 0.29915 |
| q-proxy RMSE | 4.29254 | 6.28358 | 2.11580 | 1.50705 |
| Shared new-family field score | 0.06240 | 0.12543 | Different historical training | Different historical training |

The TRAIN-fitted input-only Thermal field score is 0.73929 on this panel. Original H6 learns substantial structure but its final field-score ratio to A-D is 2.010 and fluid-T RMSE ratio 1.335, failing both 1.25 follow-up bounds. Its material endpoint maximum absolute-error mean/p90/p95/worst is 2.424/5.611/7.307/14.888, versus 1.634/3.538/4.839/11.143 for A-D, over 130 active modules. Errors are retained for M3/M5/M7/M10 separately. Worst field-score cases are 0686/0288/0677 for A-D and 0686/0648/0663 for H6; no case was added or removed using these errors.

The new Thermal reference board first reproduces normalized global heat/interface inputs and raw-derived local inputs from the actual H5 reader. Public/helper and repeated public CPU predictions agree byte-for-byte on 0291/0687 for both 1404/1804 at literal 1000 and 5000 (eight bounded qualifications, 20.25 CPU seconds). The completed mature DEV22 board took 38.8 CPU seconds. The older incomplete wrong-input board and failed GPU interface comparison remain quarantined; this new CPU evidence does not retrospectively establish GPU parity. Full-data e5000 classic results and quarter-data e2500 results have different training, so their distance is descriptive rather than an architecture ablation.

Wind reports absolute Ux/Uy/Uz and vector errors for every role, with layout means, p95/worst tails and native wake cuts. A small whole-velocity relative error is insufficient evidence of accurate transverse or vertical components. The endpoint panel has 8192 total queries per row: 1640 each for volume/hub/downstream/near-turbine and 1632 background. The actual saved role IDs and all metric rows agree; an early result metadata field mistakenly copied Q4096 monitoring counts. Raw JSONs are preserved and a correction receipt supplies the proper counts; predictions and metrics were unchanged.

| Wind native DEV24 vector RMSE, m/s | A-D e2500 | A-H6 e2500 | Run 2102 e2440 | Run 2103 e2475 |
|---|---:|---:|---:|---:|
| Volume | 0.18769 | 0.20095 | 0.03624 | 0.01767 |
| Hub slab | 0.39786 | 0.41878 | 0.05463 | 0.03402 |
| Downstream envelope | 0.45548 | 0.47714 | 0.06702 | 0.03963 |
| Near turbine | 0.64306 | 0.64282 | 0.09828 | 0.07478 |
| Background | 0.06728 | 0.08137 | 0.02682 | 0.01126 |
| Field-selection score | 0.19483 | 0.20558 | 0.006531 | 0.004639 |

Mean absolute component RMSEs keep the small secondary velocities visible. Each cell lists Ux / Uy / Uz in m/s on the same native DEV24 receivers. Complete relative-error, p95/worst-component and layout-tail tables are retained in the [machine-readable Wind development table](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/evaluation/final_tables/wind_development_native.json).

| Wind role | A-D e2500 component RMSE | A-H6 e2500 component RMSE |
|---|---:|---:|
| Volume | 0.18466 / 0.02748 / 0.01860 | 0.19792 / 0.02866 / 0.01901 |
| Hub slab | 0.39374 / 0.04988 / 0.02664 | 0.41478 / 0.05103 / 0.02626 |
| Downstream | 0.44961 / 0.06146 / 0.03805 | 0.47131 / 0.06320 / 0.03841 |
| Near turbine | 0.63027 / 0.10993 / 0.06436 | 0.63075 / 0.10735 / 0.06153 |
| Background | 0.06431 / 0.01480 / 0.01271 | 0.07848 / 0.01650 / 0.01323 |

Near-turbine vector RMSE p95/worst is 0.6822/0.6978 m/s A-D and 0.7263/0.7475 H6. The worst layout-mean near-turbine error is 0.6647 at layout 160 for A-D and 0.7255 at layout 182 for H6. Thus H6’s almost equal mean retains a worse exposed tail; the modest follow-up qualification does not imply uniform case quality.

The input-only height-profile score is 0.64827. Original H6 is within the broad follow-up guards: field ratio 1.0551, near-turbine vector ratio 0.99963 and maximum resolved role/component mean-RMSE ratio 1.2203. All 15 component comparisons retain physical absolute errors and the predeclared 3e-6×TRAIN-scale numerical floor; none relies on a near-zero denominator. The representative planes show both new models learning body-local perturbations and broad background structure while missing much of the downstream deficit depth and secondary velocity detail. The near-turbine mean errors remain far above mature references. Mature Wind ages are the actual saved 2440/2475 checkpoints from 2500 histories, never relabelled 5000. Their metrics are reused only after exact native-panel-content SHA and current checkpoint-file SHA agreement.

## The sole Thermal capacity revision

The one substantive revision changes only `model.local_hidden_dim=64→128`, widening edge/global and compact-local messages plus the physical field-head hidden space. It uses fresh seed-0 weights and moments, identical fixed membership, normalization, loss weights, query/case streams and the 2500 schedule; context128/depth2, six edges, rank4, radius3r and native laws remain fixed. The unchanged same-shape named initial tensors match bitwise. Parameter count increases from 1,295,349 to 1,386,421. This capacity comparison is separate from the unchanged matched A-D/A-H6 pair.

| Thermal exposed DEV22 native / fixed TRAIN monitor | A-D e2500 | Original H6 e2500 | Reader128 H6 e2500 |
|---|---:|---:|---:|
| Native field score | 0.062400 | 0.125428 | 0.110795 |
| Native flow / thermal group | 0.03538 / 0.08942 | 0.05112 / 0.19974 | 0.04331 / 0.17828 |
| Native fluid / surface / material T RMSE | 2.474 / 2.038 / 1.922 | 3.303 / 3.309 / 3.095 | 3.097 / 3.326 / 3.193 |
| Fixed TRAIN monitor field score | 0.03576 | 0.05322 | 0.03521 |
| Fixed TRAIN flow / thermal group | 0.05383 / 0.01769 | 0.07542 / 0.03102 | 0.04571 / 0.02470 |

The revision completes 2500 epochs, 375,000 case visits, 10,000 updates and 17,500 microbatches, with all paired stream hashes matching the original arms and finite losses/gradient aggregates. Its measured train loop takes 1880.49 seconds; actual training peak allocated/reserved is 2.565/11.545 GB. Native endpoint peak absolute-error mean/p90/p95/worst is 2.479/5.666/6.824/9.706 over 130 modules. It improves aggregate native flow/field fit over original H6 and comes close to A-D’s fixed TRAIN aggregate, but its native thermal group is nearly twice A-D and surface/material errors remain high. This supports a persistent thermal/generalization limitation; it does not establish a modal-rank diagnosis.

The selected reader128 alias is e1500, with separately measured native field score 0.08924 and fluid/surface/material RMSE 2.844/2.647/2.504. It is reported without replacing the literal e2500 comparison. The final e2500 field ratio to A-D is about 1.775, exceeding 1.25; the Thermal automatic full-TRAIN follow-up is blocked and no second revision is tried. A first cost attempt correctly rejected unequal reader widths; the explicit revision-only retry preserves every other recipe/data/transform guard and labels unequal capacity. Complete low/high-M medians are 17.16/19.60 ms for reader128 H6 versus 14.75/17.38 ms for the retained A-D, with zero optimizer updates. The unchanged width64 fixed-B density results below do not claim to measure reader128 scaling.

## Six figure groups

| Group | Contents | Retained masters |
|---|---|---|
| 1 | Matched learning, fixed-weight TRAIN/DEV and fidelity versus measured work | [Learning PDF](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/figures/learning_cost/learning.pdf) |
| 2 | Thermal native T/v residuals and interface/material detail | [Native fields](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/evaluation/figures/thermal_content_hypergraph_e2500_v3/thermal_content_hypergraph_e2500_fluid.pdf), [surface/material](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/evaluation/figures/thermal_content_hypergraph_e2500_v3/thermal_content_hypergraph_e2500_receiver_detail.pdf), [width revision](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/figures/thermal_reader128_e2500/thermal_reader128_e2500_T_v_native.pdf) |
| 3 | Wind native hub planes, component residuals and 2D/4D/8D cuts | [Four-page Wind PDF](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/figures/wind_e2500/Wind_e2500_and_full1000_native_planes_and_cuts_v3.pdf) |
| 4 | Actual pooling/content/read graph, B/access/messages and four interventions | [Two-page organization PDF](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/figures/organization_e2500/content_and_interventions.pdf) |
| 5 | Complete native costs, real training memory and fixed-B density | [Cost/scaling PDF](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/figures/learning_cost/cost_and_scaling.pdf) |
| 6 | Saved physical responses, endpoint peak changes, arithmetic precision and decision | [Response/decision PDF](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/figures/responses/saved_responses.pdf) |

### 1. Learning and fidelity versus work

![Matched development learning curves and fidelity versus measured train-loop work](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/figures/learning_cost/learning.png)

All four primary arms and the sole fresh Thermal reader128 revision complete literal e2500. Curves use the frozen exposed DEV monitor and circles use separately evaluated fixed-weight TRAIN checkpoints at 500/1000/2500. Primary Thermal train-loop time is 30.13/30.44 minutes A-D/H6 and Wind 27.17/27.04; the revised Thermal model takes 31.34 minutes. Monitoring and separate endpoint evaluation are excluded from those horizontal axes. The final native boards retain the Thermal organized-model gap and the much smaller Wind aggregate gap.

![Qualified full Wind follow-up learning, actual cold costs and native fullVALID90 reference board](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/figures/learning_cost/wind_full1000_learning.png)

The fresh full-TRAIN420/VALID90 trajectory is separate from fixed24 development and stops at literal e1000. The page shows measured train-loop/monitor seconds, native Q8192 vector errors for each role and actual saved historical ages. A vertical marker identifies the audited pooling repair from saved epoch 800; the retained trajectory has 420,000 visits and 18,000 updates, while failed and repeated work is separately charged. The prospective 5000-horizon schedule does not make this a converged fit.

The fixed populations and identical sampling streams isolate the primary decoder comparison within this new contextualizer/recipe. The curves use monitoring receivers, while the tables above use complete native endpoint receivers. Thermal's original organized-model gap persists on TRAIN and DEV; Wind H6 is slightly worse on DEV but better on fixed-weight TRAIN at 2500. Lower monitoring error at 1000 was sufficient to justify maturation, but was not substituted for the final 2500 Thermal decision.

### 2. Thermal fields and native receivers

![Thermal native temperature fields and absolute residuals on exposed DEV cases 0291 and 0687](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/evaluation/figures/thermal_content_hypergraph_e2500_v3/thermal_content_hypergraph_e2500_T_fluid.png)

Cases 0291(M5) and 0687(M10), source H5 exposed DEV22, literal A-D/A-H6 e2500 and public-CPU Run 1804 e5000; physical and residual scales are shared in native dataset units, with exact common fluid masks. On 0291, fluid T RMSE/rL2 is 2.487/0.465 A-D, 2.349/0.439 H6 and 0.2945/0.0551 classic; on 0687, 1.516/0.129,1.898/0.161 and 0.2799/0.0238. Both new models recover coarse thermal structure but miss plume shape and amplitude; the two examples do not replace the full 22-case statistics.

![Thermal transverse-velocity native fields and absolute residuals on the same receivers](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/evaluation/figures/thermal_content_hypergraph_e2500_v3/thermal_content_hypergraph_e2500_v_fluid.png)

The same native cases/checkpoints and common masks show transverse flow v without rescaling away its smaller amplitude. v rL2 on 0291 is 0.419/0.435 for A-D/H6 versus 0.0125 for classic; on 0687, 0.373/0.396 versus 0.0150. Compact body structure is learned, but its residual is substantial even where the larger flow component looks plausible.

![Thermal source 0 native 64-angle surface curve and all 3096 material receivers with residuals](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/evaluation/figures/thermal_content_hypergraph_e2500_v3/thermal_content_hypergraph_e2500_receiver_detail.png)

Case 0291 source 0 is joined by typed physical source ID and exact center. All 64 native angles and 3096 stored material points are included. Surface RMSE is 3.33 A-D, 5.41 H6 and 0.284 classic; material RMSE is 3.63, 5.03 and 0.256 in native T units. This local source is underpredicted by A-D and overpredicted by H6; the material endpoint functional is evaluated from maxima over these aligned receivers, rather than from a maximum of increments.

![Thermal reader128 capacity comparison with native temperature and transverse-velocity residuals](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/figures/thermal_reader128_e2500/thermal_reader128_e2500_case0291_preview.png)

The sole Thermal revision is compared with both untouched primary arms at literal e2500. The embedded page is 0291; the [two-page PDF also includes 0687](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/figures/thermal_reader128_e2500/thermal_reader128_e2500_T_v_native.pdf). On 0291, reader128 T/v RMSE improves to 1.956/0.00808 versus original H6’s 2.349/0.01185. On 0687, T worsens to 2.810 versus 1.898 while v improves to 0.01050 versus 0.01298. Shared physical/error scales, masks and source controls are fixed; negative T predictions reaching −8.845 on the selected receivers remain visible without clipping. The revised fit is heterogeneous rather than a uniform thermal improvement.

### 3. Wind native fields and cuts

![Wind low-module-count native hub plane and Ux Uy Uz signed residuals](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/figures/wind_e2500/wind_low_M_native_plane.png)

Exposed DEV row 69, layout 23, 8 turbines, wind 270°, literal e2500 states; the actual stored z plane is 70.869949 m for requested 70 m. Shared component scales and signed-error scales retain all 52,548 common plane receivers. Ux RMSE is 0.4195/0.4359 m/s A-D/H6; Uy rL2 is 0.614/0.623 and Uz 0.808/1.016. Both new models produce body-local responses but miss long downstream deficits, so a modest whole-vector relative error cannot be read as component fidelity.

![Wind high-module-count native hub plane and all three component residuals](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/figures/wind_e2500/wind_high_M_native_plane.png)

Exposed DEV row 426, layout 142, 30 turbines, the same actual native z plane and 99,750 common receivers. Ux RMSE is 0.4515/0.5154 m/s A-D/H6; Uy rL2 is 0.601/0.635 and Uz 0.676/0.725. The systematic Ux wake deficit is shallow in both models, with a larger high-M H6 residual. These are full native planes, without thick-slab averaging or a second wind rotation.

![Wind native crosswind cuts at requested2D4D8D downstream distances](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/figures/wind_e2500/wind_native_wake_cuts.png)

The same two layouts use an input-selected upstream source and nearest stored x slices; actual distances and native indices are printed on each column. Ux/Uy/Uz curves expose missed deficit depth, shape and crosswind/vertical detail at 2D/4D/8D. No cut is an interpolated XY projection or a model-error-selected receiver.

![Fresh full-TRAIN Wind H6 e1000 high-module native hub plane and component residuals](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/figures/wind_e2500/Wind_fullTRAIN1000_highM_native_plane_v3.png)

The separately bound full model uses exposed VALID90 row 426, layout 142, 30 turbines and wind 270°, on the same actual native z=70.869949 m plane with 99,750 common cells. Ux RMSE/rL2 is 0.41915 m/s / 0.04850, Uy 0.04662 / 0.49312 and Uz 0.01074 / 0.67000. Shared GT/H6 scales preserve every component’s amplitude; signed residuals expose missed elongated deficits and vertical structure despite the small Ux relative error. The source amendment begins at epoch 801. This page uses saved native arrays only; historical references are compared statistically on the common Q8192 board, without inventing unavailable historical plane predictions.

### 4. Actual organization and same-weight interventions

![Thermal actual donor-to-content-to-receiver paths and same-weight interventions](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/figures/organization_e2500/thermal_content_and_interventions.png)

Thermal H6 literal 2500, cases 0291/0687; shown graph is 0687(M10). Lines are actual top-three pooling links per type/edge and actual receiver access; dashed paths are executed local rows. The shown full native fluid read executes 49,152 edge rows and 6,482 local pairs. Module incidence concentrates on edge 4, and its signed temperature term dominates despite varied receiver access. With unchanged weights, full/uniform/edge 4-zero/local-zero native fluid-T RMSE on 0291 is 2.349/5.922/5.257/2.316; on 0687, 1.898/22.200/11.547/2.312 (exact values in the linked arrays/metrics). All six edges execute, edge removal keeps original access without renormalization, and local zeroing preserves pair rows. These are model interventions and do not identify physical source-response causality.

![Wind actual content use and same-weight native velocity interventions](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/figures/organization_e2500/wind_content_and_interventions.png)

Wind H6 literal 2500, rows 69/426; the graph is high-M 426. Module incidence concentrates on edge 0, while near/overlap probes read edge 3 and the background reads edge 4; actual weighted latent message norms are shown before the nonlinear head. The shown read executes 49,152 edge rows and 2,214 compact local rows. Removing edge 3 changes probe fields by up to 1.404 m/s, raises near-turbine vector RMSE and leaves the original access probabilities unnormalized. Local zeroing also worsens near-body accuracy while preserving executed pairs. A latent message norm is not an additive physical velocity contribution.

The reader128 revision is inspected separately at literal e2500 on the same two native cases. Full/uniform/largest-edge-zero/local-zero fluid-T RMSE is 1.956/6.926/4.272/2.026 for 0291 and 2.810/12.024/7.695/3.161 for 0687. The strongest probe contribution is edge 4 on 0291 and edge 1 on 0687; the revision uses its content bank and local path without uniformly improving physical fields. Thermal intervention arithmetic accumulates unchanged FP32 neural factors and native extraction in FP64; it is not an FP64 neural forward. The exact arrays and identities are in the [revision inspection](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/evaluation/thermal_reader128_inspection_e2500_v1/result.json).

The completed full-TRAIN Wind H6 e1000 is inspected independently on input-selected VALID90 rows 69/426. Full/uniform-access/largest-edge-zero/local-zero near-turbine vector RMSE is 0.5404/0.7172/0.5649/0.9838 m/s on 69 and 0.5049/0.6421/0.5619/0.9211 on 426. Edge zeroing changes a selected probe by up to 0.8940 m/s; it preserves the original access probabilities without renormalizing. Each condition still executes all 49,152 edge rows and 1,984/2,190 local pairs. The numerical source repair therefore did not leave a content-free model, while these conditional effects remain distinct from physical causality. Exact rows and arrays are in the [full e1000 content-use evidence](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/evaluation/wind_full1000_inspection_v1/result.json).

### 5. Complete costs and fixed-batch query-density memory

![Complete native costs, complete optimizer updates and fixed-batch query-density CUDA memory](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/figures/learning_cost/cost_and_scaling.png)

Measured full six-edge execution uses physical GPU0 for Thermal and GPU1 for Wind. High-M complete native medians are 17.20/19.59 ms Thermal A-D/H6 and 7.72/8.70 ms Wind; post-repair complete warm full-update medians are 209.24/215.71 and 210.21/217.58 ms. The other panels separately show actual campaign allocated/reserved memory, fixed B8 native forward/backward at Q1024/4096/8192 and a target-free Q32768 core stress. At fixed B8/Q8192, peak allocated memory is 1.506/1.004 GB Thermal and 2.474/0.809 GB Wind. Sources, checkpoint ages, repeated/native query distinctions and timing exclusions are explicit.

All native latency entries use literal 2500 checkpoints, B1, two warmups and five alternating timed repeats, at Q8192 total Wind receivers or 8192 Thermal grid nodes plus native auxiliary extraction. Input-only TRAIN fixtures are Thermal 0006(M1)/0631(M12) and Wind rows 456(M6)/533(M30). The scopes separately measure prepared decode, fresh learned scene/native prediction and complete in-memory application including native extraction and CPU return; their medians must not be added. HDF5/static-case input packing and checkpoint loading are excluded from this latency matrix, while actual train-loop throughput includes its provider work. Complete high-M CPU-return latency is 17.201/19.590 ms A-D/H6 for Thermal and 7.723/8.701 ms for Wind, ratios 1.139/1.127. H6 has cheaper prepared reads but a modestly slower complete application at these small-M fixtures.

The separate complete fixed-update measurement starts from the literal e2500 parameters, AdamW moments and scheduled learning rate and uses the final stable pooling implementation. Each timed regular full macro-update freshly samples the same deterministic cases/queries, gathers native targets, packs/transfers inputs, runs native prediction and loss, backpropagates, clips, steps AdamW and returns metrics. Neither timing includes checkpoint/provider construction or cold geometry priming. The Thermal response addendum runs on the last partial six-case macro-update, so it is absent from this regular full-update fixture and included in the measured complete campaign epochs. Scratch parameters/moments are discarded, saved checkpoints remain unchanged, and allocator baselines identify shared-process residency. Earlier pre-repair Thermal complete and prepacked medians, 0.21304/0.23437 s and 0.18940/0.20103 s respectively, remain separate measurements.

| Complete warm full-macro update after pooling repair | A-D | A-H6 | Effective / micro batch; primary Q |
|---|---:|---:|---|
| Thermal median, seconds | 0.20924 | 0.21571 | 48 / 24; 1024 |
| Wind median, seconds | 0.21021 | 0.21758 | 24 / 12; 4096 |

Each entry uses two warmups and five alternating timed repeats with identical case/query hashes. These small fixtures measure a full regular update, while the actual campaign epoch includes every selected case and its partial final update. The pre-repair native inference, training-memory and density traces retain their original source bindings. A separate post-repair Wind native cost check still meets the continuation bound, with a complete high-M H6/A-D ratio of 1.2314; it is not substituted into the earlier primary trace.

The primary parameter counts are 856,132/1,295,349 Thermal A-D/H6 and 855,895/1,270,163 Wind. H6 is not parameter-sparse. Saved AdamW state tensor bytes (including step scalars) are 6,849,480/10,363,364 for Thermal A-D/H6 and 6,847,568/10,161,836 for Wind; corresponding weight bytes are 3,424,528/5,181,396 and 3,423,580/5,080,652. These are measured from literal trained checkpoints. Live gradient/activation memory is included in the CUDA backward/optimizer peaks, not stored as checkpoint gradients. It replaces Q×M direct reads with Q×6 edge messages and uses a factorized response; at high-M native Wind Q8192 the actual padded read rows are 245,760 A-D versus 49,152 H6. Source support, edge rows, local rows, requested/unique queries and allocated/reserved memory are reported separately; no sparse executor savings are inferred from these logical reductions.

| Fixed B8 native forward/backward peak allocated bytes | Q1024 | Q4096 | Q8192 |
|---|---:|---:|---:|
| Thermal A-D | 570,462,208 | 958,960,640 | 1,506,147,840 |
| Thermal A-H6 | 417,461,760 | 655,561,728 | 1,003,815,424 |
| Wind A-D | 364,916,736 | 1,264,596,480 | 2,474,300,928 |
| Wind A-H6 | 139,902,976 | 430,319,104 | 808,506,880 |

Density measurements use the unchanged architecture at e1000, fixed B8 and increasing requested Q, without optimizer updates. Thermal's actual per-case union counts are 2810/5020/8192; Wind's unique native IDs are 1024/4092/8171 although all requested Wind receiver rows still execute. Reserved bytes and the premeasurement allocator baseline are retained because an H6 measurement can inherit A-D's allocator cache in the same process. Standalone optimizer-boundary campaign memory is separate from that shared-process reservation.

The declared larger Q32768 probe is a target-free core stress test, including actual backward and finite parameter gradients but no optimizer/native-accuracy claim. Thermal B8/M12 A-D/H6 peak allocated memory is 5.76/3.71 GB with 262,144 generated in-frame core query rows. Wind B8/M30 is 9.72/3.01 GB with 262,144 executed receiver rows but only 8171 unique native IDs per repeated case; its actual direct/edge row counts are 7,864,320/1,572,864 and local rows 76,480. Passing this probe establishes memory/work scaling for the tested functions, not new physical fidelity.

Complete CPU runtime context uses each historical model's native transforms and prepared geometry. On the matched high-M panels, new Thermal A-D/H6 e1000 medians are 53.46/41.98 ms versus 820.93 ms for Run 1804 e5000. Wind new e500 medians are 55.76/40.48 ms versus approximately 1.018 s classic2102 and 2.547 s dense2103 at their actual mature states. Run 2202 e5000 costs 183.24 ms under `full_detail, hard`, explicitly a full-detail control/fallback measurement. These mixed-age CPU timings do not establish an accuracy-matched speedup or a replacement result. No BF16, compiler, FlashAttention-kernel-specific or sparse-executor speedup was measured.

### 6. Saved response, arithmetic precision and research decision

![Saved native heat-transfer responses, endpoint peak changes and separate FP32/FP64 arithmetic checks](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/figures/responses/saved_responses.png)

Selected A-D e2100, original H6 e1300 and reader128 H6 e1500 on saved cases 0291/0294/0687; the top row shows the same 0291 plus transfer on 7837 common valid fluid receivers with a shared unclipped native-T scale. The bottom row separates physical response error, endpoint maximum changes and numerical affinity. Checkpoint ages and all six signed replays per model remain explicit; neither geometry diversity nor exact software algebra establishes inverse-design validity.

Saved cases 0291/0294/0687 contain three independent geometry/transfer families and six signed endpoint replays per model. The 0277 baseline is absent and is never fabricated. Geometry/context is prepared once per case, and physical FP64 heat controls are preserved before contraction; no dense source kernel or repeated neural read is required per transfer. Fluid/surface/material differences use the original valid masks and physical quadrature. A module-peak change is `max T(h+δh)−max T(h)` on common support, not `max δT`.

For selected A-D e2100, all 18 precise native role comparisons pass FP64 atol 1e-9/rtol 1e-7; the largest discrepancy is 2.70e-13. The old FP32 subtraction/direct-increment comparison is retained at its own atol 2e-6/rtol 2e-5 and fails on several surface/material and small fluid responses (worst interface discrepancy 1.42e-4). No tolerance was tuned to those failures. This arithmetic improvement does not repair physical response error: 0291 fluid-response RMSE is about 0.1209 native T units and its predicted signed mean has the wrong sign. Endpoint peak-change absolute-error mean is 0.1638/0.0171/0.00567 for M5/M7/M10, respectively. The response source is the saved analytic-wake/shared-grid reference, not new CFD or a new counted solve.

Both organized models also pass all 18 precise native role comparisons, while the original FP32 comparison retains 9109 failed elements for original H6 and 12029 for reader128 across fluid/surface/material roles. Fluid response RMSE on 0291/0294/0687 is 0.07592/0.04687/0.005979 for original H6 and 0.07603/0.03678/0.006441 for reader128. Their M5/M7/M10 mean absolute endpoint-peak change errors are 0.09978/0.03281/0.004602 and 0.09605/0.02382/0.004947, respectively. These selected-checkpoint responses improve some cases and miss others; the width revision does not uniformly repair physical responses.

One smooth geometry-VJP fixture per dataset retains physical extraction and native receiver semantics. Thermal VJP −0.311632830945 agrees with finite differences at steps 2e-5/4e-5 to below 1e-9 absolute; Wind VJP −0.6922090 agrees with steps 0.02/0.01D to 2.03e-5/1.43e-5 absolute. Both pass the prespecified scaled gradient atol 1e-5/rtol 1e-4. These are software interfaces with fixed receiver/support rules, not physical displacement truth, global kernel coverage or inverse-design validation.

## Conditional fresh full-TRAIN 1,000 follow-ups

The full recipe binds transforms fitted only on original TRAIN, initializes fresh parameters and moments, and preserves the selected development architecture, physical law, losses, effective batch, query counts and backend. Its prospective LR schedule has horizon 5000, warmup 20 and hold through 2000, but the approved execution stop is 1000. Actual CLI calls with stop 5000 fail before model/data construction or run allocation. There is no queued continuation and no automatic 5000 permission.

Wind's measured Sections 9/13 receipt qualifies it independently at common 2500: field 1.05513, priority 0.99963, maximum resolved component 1.22032 and high-M complete-latency 1.12662 relative to A-D; correctness, actual edge use and nontrivial fixed-weight TRAIN/DEV fields also pass. Before launch, a CUDA-hidden current-code dry-run allocates no run and performs zero updates, verifies fresh full 420 TRAIN/90 VALID membership and TRAIN transforms, and prints the resolved development-versus-full configuration diff. The new full cohort has its own manifest/normalizer identity, rather than silently resuming the 72-row model. Wind TEST90 remains locked.

Full Wind uses the verified full-TRAIN420 normalization/profile cache with exact membership and transform hashes, rather than the quarter-data state. Its e1000 budget is 420,000 case visits, 18,000 updates and 1,720,320,000 primary queries. Retained Runs 2102/2103 e1000 each record 27,000 updates, batch 16, Q8192 (6144 volume + 2048 near-band), seed 0 with sampler seed 42, a volume/band loss weighted 0.75/0.25, E512 and declared constant LR 3e-4 over a 2500 horizon. Their saved configs/CLI declare no pretraining or frozen partner. The new model uses seed 42, effective batch 24, equal five-role/component-balanced loss, Q4096, E64 and its explicit warm/hold prospective schedule. Literal age, common source membership and identical endpoint receivers therefore do not establish an equal-budget architecture ablation; exact saved update/recipe metadata are in the [historical protocol table](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/evaluation/final_tables/wind_historical_training_protocols.json).

The Wind wrapper exits zero at exactly e1000, with ten literal 100-epoch milestones and latest/best aliases at e1000; no later checkpoint or continuation job exists. Literal e1000 SHA256 is `1e2f8974b83c9e606c668410950f849c151379a151781b69fcdcd67f25a1886a`. The retained trajectory has 420,000 visits, 18,000 updates, 35,000 microbatches and 1,720,320,000 primary queries. The original failure adds 1,230 unsaved successful updates and 28,704 successful-update visits, plus one 24-case failed backward with no step. Full-Wind scientific training across the original attempt and continuation totals 19,230 successful updates and 448,728 case forwards / 1,837,989,888 primary queries; these excess costs are charged rather than hidden in the retained history.

The retained full-data train loop takes 6,311.25 s and monitoring 860.36 s. Cold initial/resumed epochs take 1,352.04/1,367.13 s; their first monitors take 428.13/430.90 s, while the final warm epoch/monitor take 3.613/0.173 s. All full-run process attempts, including the failed original run and pre-update routing failure, total 7453.95 associated seconds. Peak full training CUDA allocated/reserved is 0.651/0.847 GB and maximum recorded child RSS is 90.56 GiB; the 128 GiB RAM catalogue cap is distinct from the zero-new-persistent-cache policy. All epochs in the retained history have finite losses; the final exposed monitor score is 0.158863 versus profile 0.729228. This is a completed bounded research checkpoint under a prospective 5000 schedule, with no convergence or independent-test claim.

Across the entire round, all 63 GPU-associated process attempts total 17477.29 s (4.855 hours), within 16 hours. The first GPU experiment to final GPU exit spans 16859.02 s (4.683 hours), within the 14-hour envelope. The preceding CPU-only preparation window is 1487.58 s (24 min 47.58 s); later coding, reconciliation and evaluation overlap training and are reported through their measured per-command elapsed/CPU records. These wall/process durations are not GPU or CPU utilization, and no aggregate CPU total is fabricated.

The completed [fullVALID90 native board](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/evaluation/wind_full1000_native_board_v1/result.json) evaluates the literal new e1000 and four retained references on identical 8192-query native rows, with role quotas 1640/1640/1640/1640/1632. It includes all 90 exposed VALID rows / 30 layouts and uses panel SHA256 `ea04882030aec3993f302553f054ec7e8dc092300a3ed752051d4bb4c29253c0`; target support, units and output receivers match. Each reference retains its own qualified loader, E512 catalogue and TRAIN transform before physical m/s metrics. The four-thread CUDA-hidden command completes in 465.96 s with zero optimizer/solver calls and TEST90 locked.

| Full VALID90 mean native vector RMSE, m/s | H6 full e1000 | Classic 2102 e1000 | Classic 2102 e2440 | Dense 2103 e1000 | Dense 2103 e2475 |
|---|---:|---:|---:|---:|---:|
| Volume | 0.17528 | 0.04028 | 0.03327 | 0.02601 | 0.01869 |
| Hub slab | 0.35709 | 0.06317 | 0.05159 | 0.04106 | 0.03261 |
| Downstream | 0.41534 | 0.07226 | 0.06356 | 0.05197 | 0.03899 |
| Near turbine | 0.51648 | 0.11663 | 0.09542 | 0.08843 | 0.07393 |
| Background | 0.06413 | 0.02941 | 0.02379 | 0.01701 | 0.01119 |
| Shared full-TRAIN-scaled field score | 0.158517 | 0.008319 | 0.006251 | 0.006202 | 0.004405 |

All table values are means of case/row errors on common native receivers. H6 improves on the input-only full-VALID height-profile field score 0.729797 but remains materially worse than both actual e1000 references and their mature states. This is the full-data early-age/practical-distance board required by Section 13, with differing objectives, initialization, updates and schedules retained above. Its full-TRAIN scales and 90-row population differ from the quarter-data board, so the lower score is not a matched estimate of a data or architecture gain.

| Full H6 e1000 native role | Ux / Uy / Uz mean RMSE, m/s | Vector p95 / worst RMSE, m/s |
|---|---|---|
| Volume | 0.17286 / 0.02411 / 0.01556 | 0.21101 / 0.21980 |
| Hub slab | 0.35382 / 0.04283 / 0.02085 | 0.42241 / 0.44252 |
| Downstream | 0.41045 / 0.05418 / 0.03188 | 0.48479 / 0.50888 |
| Near turbine | 0.50779 / 0.08154 / 0.04693 | 0.60537 / 0.67941 |
| Background | 0.06161 / 0.01338 / 0.01122 | 0.08329 / 0.09595 |

The near-turbine worst row is 548 at 0.67941 m/s; the worst layout mean is layout 182 at 0.63316 m/s, with layout-mean p95 0.59915. All component/tail/layout records and saved row predictions remain in the native board. The inspected high-M plane retains shallow deficits and substantial transverse/vertical error. Full six-edge content use and finite training do not resolve this field-fidelity gap.

## Correctness, failed work and reproducibility

The final combined family/adapters/loading/launch/evaluation/benchmark/inspection/response/native-reference/legacy-curl and TrainingEngine suite passes 239 tests in 16.18 s in the ModularDT environment. Earlier snapshots and focused suites remain individually logged rather than added into an inflated total. Ruff and diff whitespace checks pass. Tests cover typed attention/reference arithmetic, masks/measure/symmetries, common initialization, gradients, native input/transform oracles, query subset/chunk consistency, capability laws, prepared reuse/staleness, source-order/units, physical extraction, transpose products, strict old loading, numerical-source amendment routing and hard full1000 stops. CPU toy fixtures test Engine behavior and are labelled as such; real checkpoints, native evaluations and measured CUDA traces supply the experiment evidence.

The initial Wind implementation incorrectly added an absolute normalizer mean in residual output conversion. Its 900 scientific optimizer updates and old checkpoints/logs are preserved and explicitly invalidated; adapter-state version 2 rejects that old physical law. Corrected `wind_*_native_v1` roots are fresh scientific lineages. The late metadata-count correction changed no numerical Wind predictions. A superseded Thermal revision rendering omitted the display mask in some panels; metrics already used the correct mask. The retained renderer now masks every field/error grid and checks all excluded-cell display centers, with the failed version recorded separately. The first real saved-response replay exposed an attempt to mutate a read-only saved validity mask; the metric now copies that local mask, a regression asserts record immutability, the failure log remains, and the subsequent actual replays complete.

The full Wind run first stopped at the strict gradient-norm guard after 868 completed epochs and six successful updates in epoch 869; its last saved checkpoint was 800. The lost failing parameter state was not reconstructed, so the guard's original nonfinite-versus-zero ambiguity and unique cause remain unresolved. An independent CUDA stress reproduces finite forward pooling with NaN score gradients at subnormal edge mass. The stable normalized-sum repair preserves normal-mass FP32 forward rounding and uses the analytical log-domain derivative; genuinely zero/subnormal masses also use the stable forward value. Learned layers remain FP32; the continuation from saved epoch 800 uses FP64 arithmetic in the small pooling reductions for epochs 801–1000. The original development and Thermal-revision fits precede this repair. Empty types remain exactly zero and no physical mass or occupancy penalty is added.

Unchanged trained checkpoints on two frozen native cases per dataset pass the original output envelope after the repair, including full Wind e800; the same e869 query key has finite backward at the retained e800 weights. FP64 reference values/gradients, JVPs and higher derivatives pass. Earlier pure log-domain variants failed a few Wind output elements and remain recorded; their tolerances were never relaxed. This is Wind's one numerical revision within the same architecture, with no new matched campaign. An explicit source-hash amendment permits the same full run's latest-state continuation while sealing every scientific binding and named moment; it forbids a second source amendment and any stop beyond 1000. The first continuation preparation failed before updates because its audit lacked an artifact category. The corrected Engine path is tested through an actual CPU toy resume, and the failed preparation remains in the ledger.

Other failed preparation/cost attempts are retained: an unintended base-Python environment, initial data-lock/overly strict arithmetic preflight, a named-milestone resume rejected in favor of the current latest-state contract, a physical1/logical1 device mismatch rejected in favor of a single-UUID logical0 child, and an initial Thermal stress adapter mismatch. Successful repairs retain the declared equations/precision budgets; historical failure artifacts were not overwritten. The discarded learnability/packing checks used 640 optimizer updates, CPU/CUDA optimizer-equivalence checks add 32, and four fixed-update cost diagnostics add 56, for 728 of the 800 permitted disposable updates. These scratch updates never modify scientific checkpoints. Density, native endpoint, pooling-equivalence and response evaluations add zero optimizer updates. All associated process lifetimes, failed work and startup costs are charged to the campaign ledger, which measures process association rather than GPU utilization.

The maintained [family guide](../../../guides/Content_Hypergraph_Family.md) contains executable command forms for dry preparation, scientific start/latest-state resume, native endpoint/fixed TRAIN evaluation, cost/density, organization, saved responses and native classic qualification. The [tested-command index](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/evaluation/final_tables/tested_commands.json) links serialized GPU/CPU argv, exits and logs and labels reconstructed reproduction forms explicitly. Run from the repository root with the absolute ModularDT interpreter and explicit import roots when running the combined test collection; GUI/base Python is not the scientific environment.

```bash
export PYTHONPATH="$PWD/HONF_Proj/src:$PWD/HONF_Proj/Case_ThermalChannel/src:$PWD/HONF_Proj/Case_WindFarm/src:$PWD/HONF_Proj/tests"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
  /home/wanglz/miniconda3/envs/ModularDT/bin/python -B -m pytest -q \
  HONF_Proj/tests/test_content_hypergraph_*.py \
  HONF_Proj/tests/test_thermal_content_hypergraph.py \
  HONF_Proj/tests/test_thermal_content_response.py \
  HONF_Proj/tests/test_thermal_native_reference_evaluate.py \
  HONF_Proj/tests/test_wind_native_reference_evaluate.py \
  HONF_Proj/tests/test_thermal_native_curl_readout.py \
  HONF_Proj/Case_WindFarm/tests/test_content_hypergraph_wind.py \
  HONF_Proj/tests/test_unified_training.py \
  HONF_Proj/tests/test_unified_training_progress.py \
  --basetemp=/absolute/owned/disposable/pytest_family
```

## Storage and preservation closeout

The round began with 110,356,488,192 free bytes on `/data` and 39,385,210,880 on `/home`. The declared hard new-disk ceilings were 12 GiB / 2 GiB, estimated peaks 8 GiB / 512 MiB and temporary peaks 768 MiB / 256 MiB, with 20 GiB minimum headroom on each filesystem. No dataset copy or separate comparison-arm geometry store was created; exact verified shared role catalogues were read-only, with bounded RAM fallbacks and no persistent geometry-cache writes. Browser and plotting previews use explicit round-owned temporary budgets and expiry.

The [exact science-cleanup allowlist](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/receipts/cleanup_science_exact_allowlist_v1.json) and [removal receipt](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/receipts/cleanup_science_receipt_v1.json) record ownership, sizes, devices/inodes/links, hashes, dependencies and accessible process/open-mapping checks before removal. Closeout removes 2,738 regular disposable files and 251 symlinks: 62,940,872 logical bytes / 71,888,896 allocated bytes, plus empty owned directories. These are expired pytest fixtures/caches, superseded field/PDF previews, compiled caches and exact owned review/tool-log files. The original Wind three-page content is preserved identically inside the selected four-page master; old masters and failed layout payloads are removed with their compact failure/metric/hash evidence retained.

Earlier receipt-backed task cleanups separately removed 364 temporary regular files (7,564,758 bytes) and 16 symlinks, and replaced four superseded Thermal visual payloads (1,287,837 bytes). Their exact receipts remain under `$ROUND/receipts/thermal_reader128_*cleanup*`, the corresponding layout-replacement receipts and the Wind pytest cleanup receipts. Those counts are separate from the science-closeout allowlist, so the same deletion is not counted twice.

At 2026-10-14T07:29:07.529560+00:00, actual remaining free capacity after science cleanup is 107,747,074,048 bytes on `/data` (100.347 GiB), 39,057,666,048 on `/home` (36.375 GiB) and 270,325,571,584 on `/dev/shm`. Elapsed time from the first GPU experiment to this closeout checkpoint is 5.060 hours. The final browser preview is removed after validation; its exact removal/latest capacity record is `$ROUND/receipts/cleanup_browser_receipt_v1.json`, and the final end-to-end timestamp is retained in the repository-closeout receipt.

The [preservation verification](../../../../diagnostics/generated/content_hypergraph_oct13_20261014T015930Z/receipts/preservation_science_v1.json) rechecks all 1,919 initial observations: 1,845 existing-file observations retain metadata and SHA256, all 74 initial absences remain absent, source H5/manifest bindings are unchanged, and the unrelated user deletion is preserved. The 976 frozen retained round files pass pre/post cleanup identity checks. All declared scientific milestones, latest/best aliases, original failure histories, physical-reference arrays, native metric/prediction archives, source helpers and manifests remain; retained scientific/visual payload is 2,467,728,423 bytes. The selected report contains six figure groups, eight PDF masters and fourteen necessary small PNG companions. All allowlisted obsolete payloads are absent, all associated GPU jobs have ended, WindTEST remains locked and solver usage remains 326/326 with zero new attempts. Final preview and repository receipts retain source/link/render and upload verification.

**Research decision: attention foundation works but edge decoding needs revision.** A further authorized study should reduce broad-field and near-body errors while retaining the lawful shared computation. The current evidence supports a tested family and measured content use; classic replacement, physical causal groups, sparse execution savings, independent-test generalization and inverse readiness remain unestablished.
