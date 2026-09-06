# HONF interface-operator study: Stage-3 evaluation and continuation decision

## Decision

At the matched 500-epoch budget, the **dense pairwise field adaptation (Run 1804)** is the strongest candidate and the only run recommended for a user-launched continuation. It has the lowest pooled fluid-field normalized MSE and relative L2, the best equal-case field-error distribution, and the best field result in every predefined physical stratum. The geometry-latent adaptation is useful on several local/module and thermal KPIs, but its far-field error is materially worse. Sparse-interface HONF constructs and executes the intended bounded-support topology, yet its learned main-group read has collapsed to essentially zero magnitude; it is the slowest model on the present 3–10-module cases and has the worst field accuracy at epoch 500.

This does **not** establish a publication-level superiority claim. There is one training seed, the 90 cases are the established development holdout, the physical collective-response reference is pending because no maintained project-integrated CFD case is available, and a long Run-1804 continuation would eventually need a comparable-budget new-family baseline or an explicitly cost-budgeted comparison.

The dense and geometry-latent models are adaptations of contextual pairwise and Set-Transformer/Perceiver-style ideas, respectively; neither is an exact paper reproduction.

## What was implemented

Stages 1 and 2 added three candidates beside the unchanged `legacy_honf` family:

- `dense_pairwise_field`: simultaneous contextual module–module, module–environment, and environment–module updates, followed by nonlinear receiver–source messages. It is not an additive straw man.
- `geometry_latent_field`: geometry-aware latent cross-attention and processing, with the latent state read continuously at physical ports and field receivers.
- `sparse_interface_honf`: layout-generated overlapping cubic supports, retained typed module/environment incidences, learned membership factors within geometric support, one nonlinear shared state per group, and sparse receiver lookup bounded by 16 groups in 2-D.

All three new families share the same encode/prepare/read facade, eight-token coarse path, compact local correction, case wrapper, autonomous port prediction, frozen Stage-A local operator, one physical refinement, features, losses, optimizer policy, and prepared decoding path. Sparse HONF reuses one geometry layout across P0/P1/P2, caches environment work, refreshes the shared group states after local-operator responses, and uses those same states for port and field reads. Legacy checkpoints and execution remain on their historical path. The recommended profile is still `stage7_structured_context`.

Stage-3 reporting adds raw pooled error components, deterministic physical KPIs, physical strata, and arbitrary matched visual anchors. These are diagnostics, not gates or new approval machinery.

Implementation handoffs: [Stage 1](stage1_interface_fields/README.md) and [Stage 2](stage2_sparse_interface_honf/README.md).

## What ran

### Training record inherited from Stages 1–2

No managed run was trained or resumed in Stage 3. The only optimizer updates were the explicitly requested nonpersistent training-step profiles described below; no checkpoint or run state was saved. The primary checkpoints are the exact epoch-500 endpoints from the earlier managed runs.

| Model | Managed run | Epochs / updates | Trainable scalars | Train + validation work | Manifest wall | Peak CUDA allocation |
|---|---|---:|---:|---:|---:|---:|
| Dense pairwise adaptation | `Run_1804_20260905_081349_dense_pairwise_field_adaptation` | 500 / 6,500 | 4,395,409 | 9,888.42 s | 9,967.15 s | 27,208.66 MiB |
| Geometry latent adaptation | `Run_1801_20260905_110056_geometry_latent_field_adaptation` | 500 / 6,500 | 4,542,489 | 8,949.67 s | 9,039.02 s | 23,039.53 MiB |
| Sparse interface HONF | `Run_1802_20260905_170523_sparse_interface_honf` | 500 / 6,500 | 3,778,064 | 9,332.69 s | 9,413.42 s | 29,582.44 MiB |

Each epoch used 13 real optimizer batches. The observed mean training-batch wall time, including the full physical loop, backward, clipping, optimizer update, and scheduled diagnostics, was 1.2923 s for sparse HONF. Dense and latent total training work above is the corresponding end-to-end evidence; no synthetic kernel time is substituted for training cost.

Selected pre-clip and actual-update observations:

| Model / epoch | Pre-clip total | Encoder | backend | head | local coupling | clip scale | update norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense / 1 | 547.461 | 246.558 | 94.809 | 477.653 | 42.218 | 0.001827 | 0.5646 |
| Dense / 20 | 22.313 | 8.770 | 4.067 | 19.731 | 3.888 | 0.04482 | 0.09184 |
| Dense / 500 | 0.569 | 0.343 | 0.213 | 0.397 | 0.058 | 1.000 | 0.1148 |
| Latent / 1 | 449.255 | 204.533 | 65.332 | 392.228 | 43.424 | 0.002226 | 0.5853 |
| Latent / 20 | 26.633 | 7.312 | 4.434 | 25.074 | 2.731 | 0.03755 | 0.1104 |
| Latent / 500 | 1.449 | 0.906 | 0.410 | 1.045 | 0.136 | 0.6903 | 0.09279 |
| Sparse / 1 | 369.056 | 153.887 | 55.152 | 329.582 | 29.240 | 0.002710 | 0.52825 |
| Sparse / 20 | 28.058 | 8.252 | 3.338 | 26.366 | 3.579 | 0.035641 | 0.08368 |
| Sparse / 500 | 2.890 | 1.551 | 0.794 | 1.815 | 1.421 | 0.346004 | 0.06484 |

Run 1800 was the proposed dense ID but failed before epoch 1 with a real later-bucket CUDA OOM. Run 1803 also exposed a peak-memory site and remains preserved as failed. The completed matched dense run is therefore Run 1804; no occupied run was overwritten.

### Stage-3 evaluation scope

The four primary endpoint checkpoints were each evaluated once on all 90 cases in the fixed `test` development holdout, with the checkpoint-owned normalization, all 8,192 grid queries per case, predicted ports, one physical refinement, and routing diagnostics. This produced 360 aligned per-case rows. A data-quality audit confirmed identical 90-case ID sets, finite required metrics, and exact `pooled MSE = sum(SSE) / sum(value count)` arithmetic.

Run 1401 best-field at epoch 4585 was evaluated separately on the same 90 cases. It is a mature reference, not a fifth matched-budget competitor.

Primary output:

`Trained_Results/ThermalChannel/HONF_Forward_Runs/CompareModels/Stage3_Run1401_1804_1801_1802_Epoch500_90Case`

Mature output:

`Trained_Results/ThermalChannel/HONF_Forward_Runs/CompareModels/Stage3_Run1401_MatureBestField_Epoch4585_90Case`

## Matched accuracy at epoch 500

Normalized MSE and relative L2 are pooled from raw fluid-domain SSE, target SSE, and value counts. Mean/median/p95/worst are equal-case summaries of each case's normalized relative L2. “Worst” means the maximum observed among these 90 development cases.

| Model | Pooled norm. MSE | Pooled rel. L2 | Case mean | median | p95 | worst | Near surface distance 0–0.25 | Far distance >=1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Run 1401 legacy HONF @500 | 0.012705 | 0.11715 | 0.11345 | 0.11086 | 0.14088 | 0.15024 | 0.11731 | 0.12306 |
| Dense pairwise adaptation @500 | **0.009026** | **0.09874** | **0.09611** | **0.09674** | **0.11253** | **0.12476** | **0.09759** | **0.09940** |
| Geometry latent adaptation @500 | 0.019554 | 0.14533 | 0.14025 | 0.13546 | 0.20345 | 0.23552 | 0.10287 | 0.15681 |
| Sparse interface HONF @500 | 0.028678 | 0.17600 | 0.17049 | 0.16495 | 0.23887 | 0.26319 | 0.12066 | 0.19114 |

Dense improves pooled relative L2 by 15.7% versus Run 1401, 32.1% versus latent, and 43.9% versus sparse. Sparse is 78.2% above dense. Latent is comparatively good near interfaces but loses accuracy in the far field.

### Physical-space field channels

The values below are pooled physical-space relative L2 per channel. Every channel is scaled back with the checkpoint/dataset normalizer before error calculation; unlike physical units are never combined into one physical error.

| Model | u | v | p | omega | temperature |
|---|---:|---:|---:|---:|---:|
| Run 1401 @500 | 0.04599 | 0.07742 | 0.09677 | 0.15560 | 0.11575 |
| Dense @500 | **0.02936** | **0.10376** | **0.08898** | **0.11367** | **0.09023** |
| Latent @500 | 0.04748 | 0.13930 | 0.17167 | 0.15948 | 0.12030 |
| Sparse @500 | 0.05786 | 0.13623 | 0.20823 | 0.20856 | 0.15627 |

Dense wins four of five channels; Run 1401 is best on transverse velocity `v`.

### Ports, interfaces, internal state, and fixed physical KPIs

| Model | Port T_env rel. L2 | Port h rel. L2 | Internal T rel. L2 | Surface T rel. L2 | Flux rel. L2 | Delta-p MAE | Outlet T MAE | Mean module T MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Run 1401 @500 | 0.08804 | 0.04225 | 0.07172 | 0.09151 | 0.21691 | 0.00913 | 0.91553 | 0.45940 |
| Dense @500 | **0.06892** | **0.04163** | 0.06949 | 0.08724 | 0.22007 | **0.00719** | 1.01958 | 0.60572 |
| Latent @500 | 0.10701 | 0.04904 | **0.05814** | **0.07766** | **0.20395** | 0.01320 | **0.52436** | **0.19514** |
| Sparse @500 | 0.13227 | 0.05004 | 0.07113 | 0.09346 | 0.21810 | 0.01112 | 0.54376 | 0.31993 |

Pressure drop is the mean pressure on the fixed 64-point inlet grid column minus the fixed 64-point outlet column; the difference is gauge-invariant. Mean outlet temperature uses that fixed outlet column. Mean module temperature averages all sampled internal cells of active modules; it is stable within a fixed case but is not an identity-matched KPI across changing module counts.

Latent's local/module and outlet-temperature results are genuine strengths, but do not offset its 47.2% higher pooled field MSE than dense for the forward-field objective. Dense's worse outlet and module-temperature KPIs are the main reason not to collapse the decision into one aggregate score.

### Physical strata

Strata are fixed reporting bins, not model gates: exact active counts `{3,5,7,10}`; minimum surface gap `<1r`, `1–2.5r`, `>=2.5r`; wall clearance `<1.5r`, `1.5–2.5r`, `>=2.5r`; heat-power coefficient of variation `<0.25`, `0.25–0.35`, `>=0.35`.

| Stratum | Cases | Best mean field rel. L2 | Sparse mean field rel. L2 |
|---|---:|---:|---:|
| M=3 / 5 / 7 / 10 | 25 / 25 / 25 / 15 | Dense: 0.0821 / 0.0951 / 0.1042 / 0.1076 | 0.1418 / 0.1715 / 0.1951 / 0.1754 |
| Crowded / intermediate / separated | 45 / 30 / 15 | Dense: 0.1037 / 0.0919 / 0.0819 | 0.1802 / 0.1674 / 0.1476 |
| Near wall / middle / interior | 28 / 50 / 12 | Dense: 0.0978 / 0.0981 / 0.0838 | 0.1774 / 0.1730 / 0.1440 |
| High / medium / low heat CV | 20 / 42 / 28 | Dense: 0.1006 / 0.0973 / 0.0911 | 0.1935 / 0.1684 / 0.1572 |

Dense is best in every listed stratum. All are in-distribution development cases; difficult crowded/high-heating cases are not called OOD.

## Mature Run-1401 reference

Run 1401's trusted best-field checkpoint records epoch 4585. On the same 90 cases it obtains pooled normalized MSE 0.000954 and pooled relative L2 0.03210; equal-case mean/median/p95/worst are 0.02982/0.02610/0.05056/0.07415. Near/far relative L2 are 0.03254/0.03476. This large maturity advantage is reported separately and is not used to demote a 500-epoch candidate by unequal-budget comparison.

Its physical relative L2 values are `u=0.01067`, `v=0.01559`, `p=0.03134`, `omega=0.03880`, and `temperature=0.03723`; fixed-KPI MAEs are `delta-p=0.00187`, outlet temperature `0.19251`, and mean module temperature `0.13063`.

## Does the sparse group path participate?

Sparse HONF's topology is genuinely constructed and executed:

| Retained quantity, 90 cases | Min | Mean | median | p95 | max |
|---|---:|---:|---:|---:|---:|
| Distinct occupied group states K | 28 | 44.97 | 44.5 | 56.0 | 58 |
| Module/group incidence rows | 56 | 117.57 | 105.5 | 203.2 | 209 |
| Cached environment/group rows | 1,870 | 2,658.03 | 2,696.5 | 2,931.7 | 2,960 |
| Initial-port/group read rows | 3,765 | 9,232.72 | 9,618 | 11,794 | 11,794 |
| Final-query/group read rows | 80,762 | 113,753.67 | 115,383 | 125,496 | 127,259 |
| Mean unique modules/group | 1.22 | 2.55 | 2.50 | 3.92 | 4.17 |
| Mean field-read degree | 9.86 | 13.89 | 14.08 | 15.32 | 15.53 |
| Mean port-read degree | 4.90 | 12.02 | 12.52 | 15.36 | 15.36 |

The observed receiver maximum is exactly 16, the `4^2` cubic-support bound. These are actual retained rows processed by the sparse membership/message/read networks, not dense dimensions multiplied by a mask.

However, the learned group branch is functionally underused. Its mean context-norm fraction is `4.37e-26` at final field queries and `1.47e-24` at physical ports. Coarse/local fractions are 0.8851/0.1149 at queries and 0.7268/0.2732 at ports. Context norms are magnitude-based reliance proxies, not causal effects; the intervention results below test the same trained network more directly.

Three physical preparations P0/P1/P2 refresh K group states on one layout; cached environment transforms/pools are reused. Receiver incidence lookup is recomputed for each coordinate set. The common local correction first forms a dense receiver-by-module distance tensor before gathering neighbours, so only the main group path—not the entire model—currently has support-count sparse execution.

## Interventions, conditional influence, and learned-model derivatives

Four sparse cases were used: 0273, 0653, crowded case 0298, and worst larger case 0302. Removing main-group context from port prediction and then recomputing the entire local/refinement sequence changes the final field by only `7.73e-12` mean absolute value across cases (maximum element `4.77e-7`); port, interface, and internal tensors are unchanged at recorded precision. Removing the main-group final field read produces `4.24e-8` mean absolute field change (maximum element `7.63e-6`). This agrees with, and gives functional support to, the near-zero context-norm diagnosis.

For the third intervention, each case receives a valid `0.1r` module displacement; the unperturbed coarse contexts are replayed at the matching P0/P1/P2 reads while sparse layout/group states, local responses, and local reads rebuild normally. Relative to the normally perturbed prediction, coarse clamping changes field/port/interface/internal tensors by mean absolute values `0.00264/0.00340/0.00192/0.00157`; maxima are `0.0649/0.2883/0.0601/0.0321`. The coarse route therefore carries material geometry response even though the fine group route does not.

The full sparse model was differentiated on four layouts at fixed physical probes, with every displaced point rebuilding encoding, topology, P0/P1/P2, local coupling, and decoding. Directional autograd versus central finite differences at `h=0.01r` and `0.005r` has relative discrepancies:

| Case | 0.01r | 0.005r |
|---|---:|---:|
| 0273 | 1.07e-4 | 6.73e-4 |
| 0653 | 1.98e-2 | 5.89e-3 |
| 0298 | 3.28e-4 | 2.04e-3 |
| 0302 support transition | 1.83e-4 | 7.07e-5 |

Case 0302 was adjusted by a valid `0.04503` displacement in x so one physical port lay on a lattice boundary. The selected module's active incidence set changes from 16 to 20 group keys across the central difference, with four keys in the symmetric difference. Agreement remains good because the cubic basis enters/leaves continuously.

For a single prepared group update, the tool perturbs one encoded module state while reading only the main fine path; global/coarse state is held fixed and local correction is excluded. Module-specific disconnected receivers exist in cases 0298 and 0302. Their autograd derivative is exactly zero and central finite differences are zero or at most `9.63e-32`; connected receivers also have near-zero response because the trained receiver weights collapsed. This validates conditional sparsity, while simultaneously showing that topology is not being used productively at this checkpoint.

The interventions are trained-model reliance diagnostics, not unbiased retraining ablations. Autograd agreement with the same model's finite differences checks implementation consistency, not physical derivative accuracy.

## Quadrature consistency and execution scaling

Exact 2x environment duplication on case 0273 preserves total quadrature volume and repeats identical features. All four complete physical models agree to float32 arithmetic: maximum absolute field changes are `3.58e-6` (Run 1401), `5.25e-6` (dense), `6.20e-6` (latent), and `8.58e-6` (sparse); mean absolute field changes range from `9.33e-8` to `1.48e-7`. Ports, interfaces, and internal outputs are similarly stable. This is exact-duplicate consistency, not a finer-feature resolution study.

The required five synthetic layouts use actual nonoverlapping module footprints, environment quadrature, physical support construction, neural preparation, and every requested query. Domain area scales with M at constant nominal module/environment density. With one warmup and five synchronized measurements, median seconds are:

| (M,E,Q) | Dense | Latent | Sparse | Sparse K | Sparse module/env incidences | Sparse query/group rows |
|---|---:|---:|---:|---:|---:|---:|
| (8,192,8,192) | 0.178 | **0.150** | 0.246 | 60 | 171 / 3,024 | 131,072 |
| (32,768,8,192) | 0.227 | **0.191** | 0.323 | 160 | 703 / 12,212 | 131,072 |
| (32,768,65,536) | 1.171 | **0.967** | 1.769 | 160 | 703 / 12,212 | 1,048,576 |
| (32,768,262,144) | 4.553 | **3.657** | 6.748 | 160 | 703 / 12,212 | 4,194,304 |
| (128,3,072,262,144) | 7.149 | **3.896** | 7.099 | 480 | 2,775 / 48,872 | 4,194,304 |

The maximum sparse read degree is 16 at every shape, and the final two rows have the same Q and therefore the same `Q*16` main-read row count despite M increasing from 32 to 128. Peak allocated memory at the largest shape is 5,116 MiB dense, 270 MiB latent, and 275 MiB sparse. Sparse is roughly tied with dense there but not faster; latent remains 1.82x faster. Every family completes all shapes. All-at-once versus 32,768-chunk decoding on the smallest shape is bit-exact for all three. These are execution-only synthetic results and carry no physical-accuracy conclusion.

The present 90-case synchronized, no-warmup end-to-end comparison measured one complete prepared full-grid prediction with routing diagnostics per model/case:

| Model | Mean wall s | median | p95 | Mean queries/s | Mean incremental allocated MiB |
|---|---:|---:|---:|---:|---:|
| Run 1401 @500 | 0.0542 | 0.0517 | 0.0541 | 163,332 | 362.81 |
| Dense @500 | 0.2037 | 0.2035 | 0.2250 | 40,342 | 81.44 |
| Latent @500 | 0.1716 | 0.1717 | 0.1889 | 47,884 | 54.94 |
| Sparse @500 | 0.2869 | 0.2868 | 0.3040 | 28,581 | 55.50 |

The first-model CUDA allocation makes these incremental memory numbers observations, not a controlled memory ranking. Repeated warm measurements provide the stronger phase-level timing evidence.

Five warmups and 20 synchronized measurements on the two real anchors give these median milliseconds (`0273 / 0653`):

| Model | Encoding + layout | Physical preparation + one query | Prepared decode 8,192 | Full forward 8,192 |
|---|---:|---:|---:|---:|
| Run 1401 @500 | not separable on legacy path | 21.89 / 25.53 | 6.55 / 6.80 | 25.46 / 26.64 |
| Dense @500 | 1.17 / 1.21 | 39.26 / 40.10 | 115.91 / 118.58 | 152.96 / 153.68 |
| Latent @500 | 1.20 / 1.19 | 36.40 / 37.47 | 93.84 / 92.84 | 128.18 / 128.97 |
| Sparse @500 | 3.82 / 3.64 | 58.33 / 54.15 | 186.33 / 186.37 | 234.47 / 232.82 |

`Physical preparation + one query` includes case encoding/layout, all P0/P1/P2 preparations, frozen local-operator calls, one refinement, and one decode. The encoding/layout row is measured separately and is not subtracted from it. The sparse environment pool is cached across its three preparations; receiver lookups are recomputed. Dense activation checkpointing is configured for training, not these inference-mode measurements.

A separate nonpersistent profile used one warmup and three measured repetitions of the canonical real 48-case x 1,024-query training batch, loading endpoint AdamW moments and discarding the model/optimizer without saving. Median full-step times are `88.5 ms` Run 1401, `801.6 ms` dense, `755.1 ms` latent, and `800.8 ms` sparse; p05–p95 intervals are `87.8–99.2`, `796.5–841.2`, `747.1–809.7`, and `784.5–808.4 ms`. AMP is off in all four checkpoint policies. These 16 disposable profiling updates are not managed training, a resume, or a checkpoint continuation.

## Collective response and physical-reference status

No maintained, project-integrated CFD case or conversion path is present in `HONF_Proj`. The maintained verifier is a frozen neural model and is not a physical solver. Host OpenFOAM 10 includes `chtMultiRegionFoam`, but the repository lacks the required mesh, region dictionaries, boundary conditions, material maps, and output conversion. Constructing an unvalidated case would not be use of an existing trusted solver. The ignored historical demo explicitly uses analytic wake fields and neighbouring-cell diffusion rather than full coupled CFD, so it was not relabelled as reference truth.

Accordingly, no solver was run. Case 0298 supplies exactly 16 valid `0.1r` positional-perturbation requests: a separated pair on modules 0/6, a crowded pair on modules 4/5, and a compact triple on modules 1/2/6. Operating/material conditions and fixed probe coordinates remain unchanged. The export contains corresponding predictions from all four primary checkpoints and confirms that their state-dict structures remain unchanged. Their physical reference fields and interaction errors remain **pending**. Model-predicted collective interactions are small (roughly `1e-6` to `3e-4` in normalized fixed-probe KPIs), but are hypotheses for later solver comparison, not validated higher-order physics or evidence of accuracy.

Physical derivative quality is also pending. Learned-model autograd versus learned-model finite differences does not fill that gap.

## Matched cases and figures

Cases 0273 and 0653 remain the established anchors. Case 0302 was selected after the 90-case summary because it is the sparse model's worst field-error case (`M=7`, normalized fluid relative L2 0.26319); it was not selected to improve a preferred model's narrative.

The coherent comparison tree contains matched physical panels, correctly labelled dense and latent interaction maps, sparse geometric/learned memberships, common group IDs for port/query routes, bounded read-degree maps, and summary plots. Case 0302 is under `selected_large_case_0302/` inside the same tree.

## What remains unproven

- Collective pair/triple interactions and physical coordinate sensitivities await trustworthy coupled-solver outputs.
- One seed cannot establish generalization or optimization stability.
- No support-matched sparse pairwise control has been trained, so results cannot isolate reusable nonlinear group state from locality alone. If attribution later matters, specify the plan's deduplicated `sparse_pairwise_control` using identical supports/coarse/local/physical heads, but do not launch Run 1803 without explicit user authorization.
- The coarse branch dominates sparse predictions; a longer sparse run is not justified by current matched-budget evidence alone.
- Synthetic 32/128-module execution is not accuracy evidence, and a fixed eight-token coarse bank may become the fidelity bottleneck as the physical domain grows.
- The shared local correction still materializes receiver-by-module distances; it is not yet fully sparse by retained-neighbour count.

## Recommendation and Stage-3 handoff

Continue **Run 1804 dense pairwise adaptation** first if the user chooses to spend a long-run budget. It is the clear matched-budget field winner and has the best port errors and pressure-drop error, while latent remains an informative local-physics comparator. Do not extend sparse HONF automatically: its topology is valid, but its learned group route contributes essentially nothing at epoch 500 and its accuracy/cost do not justify priority.

The following command is a handoff only and was **not executed**:

```bash
rtk conda run --no-capture-output -n ModularDT python train.py \
  --config project://src/config_core/forward/dense_pairwise_interface_context.json \
  --workflow forward --device cuda:0 --epochs 2500 \
  --resume-checkpoint /home/wanglz/Desktop/src/ModularDT/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1804_20260905_081349_dense_pairwise_field_adaptation/latest_model.pt --yes
```

A long Run 1804 compared only with 500-epoch latent/sparse checkpoints would be an unequal-budget development result. A strong final claim would require comparable-budget training of the strongest relevant baseline or a predeclared compute-budget comparison; mature Run 1401 @4585 is already labelled separately for the same reason.

## Reproduction commands

Primary 90-case evaluation (line breaks shortened only for readability):

```bash
rtk env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:Case_ThermalChannel/src \
/home/wanglz/miniconda3/envs/ModularDT/bin/python evaluate.py --config src/config_core/forward/sparse_interface_honf_context.json \
  --workflow compare \
  --checkpoint-path Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1401_20260823_151126_stage7_modern_structured_context/epoch_0500_model.pt \
  --checkpoint-path Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1804_20260905_081349_dense_pairwise_field_adaptation/epoch_0500_model.pt \
  --checkpoint-path Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1801_20260905_110056_geometry_latent_field_adaptation/epoch_0500_model.pt \
  --checkpoint-path Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1802_20260905_170523_sparse_interface_honf/epoch_0500_model.pt \
  --label 'Run 1401 legacy HONF @500' --label 'Dense pairwise adaptation @500' \
  --label 'Geometry latent adaptation @500' --label 'Sparse interface HONF @500' \
  --split test --case-ratio 1.0 --device cuda:0 --query-batch-size 32768 \
  --local-port-condition-mode predicted --return-routing-maps \
  --anchor-case-id 0273 --anchor-case-id 0653 \
  --output-dir Trained_Results/ThermalChannel/HONF_Forward_Runs/CompareModels/Stage3_Run1401_1804_1801_1802_Epoch500_90Case
```

Mature Run-1401 evaluation used the same split/query/normalization settings and the trusted checkpoint that records epoch 4585:

```bash
rtk env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:Case_ThermalChannel/src \
/home/wanglz/miniconda3/envs/ModularDT/bin/python evaluate.py --config src/config_core/forward/sparse_interface_honf_context.json \
  --workflow compare \
  --checkpoint-path Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1401_20260823_151126_stage7_modern_structured_context/best_by_field_mse_model.pt \
  --label 'Run 1401 mature best-field @4585' --split test --case-ratio 1.0 \
  --device cuda:0 --query-batch-size 32768 --local-port-condition-mode predicted \
  --return-routing-maps --anchor-case-id 0273 --anchor-case-id 0653 \
  --output-dir Trained_Results/ThermalChannel/HONF_Forward_Runs/CompareModels/Stage3_Run1401_MatureBestField_Epoch4585_90Case
```

Focused ordinary verification:

```bash
PYTHONPATH=src:Case_ThermalChannel/src python -m pytest -q \
  tests/test_sparse_supports.py \
  Case_ThermalChannel/tests/test_interface_field_models.py \
  Case_ThermalChannel/tests/test_sparse_evaluation_diagnostics.py \
  Case_ThermalChannel/tests/test_stage3_interface_study.py
```

Observed result after adding the Stage-3 study tests: `31 passed`.

The complete ordinary repository suite also passed: `360 passed, 1 skipped`; the skip is the pre-existing inverse integration test whose local inverse artifacts are unavailable.

The single Stage-3 study orchestrator and the extended phase profiler were run with these checkpoint bindings and task arguments (all paths are relative to `HONF_Proj` except the Python executable):

```bash
P1401=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1401_20260823_151126_stage7_modern_structured_context/epoch_0500_model.pt
DENSE=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1804_20260905_081349_dense_pairwise_field_adaptation/epoch_0500_model.pt
LATENT=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1801_20260905_110056_geometry_latent_field_adaptation/epoch_0500_model.pt
SPARSE=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1802_20260905_170523_sparse_interface_honf/epoch_0500_model.pt
PY=/home/wanglz/miniconda3/envs/ModularDT/bin/python

rtk env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:Case_ThermalChannel/src $PY tools/diagnostics/run_stage3_interface_study.py interventions \
  --sparse-checkpoint "Sparse interface HONF @500=$SPARSE" \
  --case-id 0273 --case-id 0653 --case-id 0298 --case-id 0302 \
  --query-count 8192 --device cuda:0 \
  --output diagnostics/generated/interface_operator_study/stage3/interventions/interventions.json

rtk env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:Case_ThermalChannel/src $PY tools/diagnostics/run_stage3_interface_study.py gradients \
  --sparse-checkpoint "Sparse interface HONF @500=$SPARSE" \
  --case-id 0273 --case-id 0653 --case-id 0298 --case-id 0302 \
  --query-count 256 --device cuda:0 \
  --output diagnostics/generated/interface_operator_study/stage3/gradients/gradients_four_layouts.json

rtk env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:Case_ThermalChannel/src $PY tools/diagnostics/run_stage3_interface_study.py quadrature \
  --checkpoint "Run 1401 legacy HONF @500=$P1401" \
  --checkpoint "Dense pairwise adaptation @500=$DENSE" \
  --checkpoint "Geometry latent adaptation @500=$LATENT" \
  --checkpoint "Sparse interface HONF @500=$SPARSE" \
  --case-id 0273 --query-count 8192 --device cuda:0 \
  --output diagnostics/generated/interface_operator_study/stage3/quadrature/quadrature_case0273.json

rtk env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:Case_ThermalChannel/src $PY tools/diagnostics/run_stage3_interface_study.py scaling \
  --checkpoint "Dense pairwise adaptation @500=$DENSE" \
  --checkpoint "Geometry latent adaptation @500=$LATENT" \
  --checkpoint "Sparse interface HONF @500=$SPARSE" \
  --device cuda:0 --query-batch-size 32768 --warmup 1 --repetitions 5 \
  --chunk-equality-limit 8192 \
  --output diagnostics/generated/interface_operator_study/stage3/scaling/five_real_support_shapes.json

rtk env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:Case_ThermalChannel/src $PY tools/diagnostics/run_stage3_interface_study.py requests \
  --checkpoint "Run 1401 legacy HONF @500=$P1401" \
  --checkpoint "Dense pairwise adaptation @500=$DENSE" \
  --checkpoint "Geometry latent adaptation @500=$LATENT" \
  --checkpoint "Sparse interface HONF @500=$SPARSE" \
  --case-id 0298 --query-count 256 --perturbation-scale 0.1 \
  --max-requests 16 --device cuda:0 \
  --output diagnostics/generated/interface_operator_study/stage3/reference_requests/pending_reference_requests.json
```

Inference phases used the same four checkpoint bindings with `tools/profile_stage2_sparse_inference.py`, `--case-id 0273 --case-id 0653 --warmups 5 --repetitions 20 --prepared-query-count 8192`. The disposable training-step profile used `--training-step --warmups 1 --repetitions 3`; its separate output directory prevents mixing repetition protocols.

## Artifact index

- Primary aggregate, per-case, per-module, physical-stratum, interaction, topology, and cost tables: `.../Stage3_Run1401_1804_1801_1802_Epoch500_90Case/tables/`.
- Probe/stratum definitions, checkpoint manifest, and exact selected case IDs: `.../logs/`.
- Matched figures: `.../figures/`; larger anchor: `.../selected_large_case_0302/figures/`.
- Mature tables: `.../Stage3_Run1401_MatureBestField_Epoch4585_90Case/tables/`.
- Stage-3 diagnostic tables, figures, and pending solver requests: `diagnostics/generated/interface_operator_study/stage3/`.
