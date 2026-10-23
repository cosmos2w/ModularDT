# ThermalChannel M0 baseline decision

**Decision:** use Run 1804 selected e4738 as the provisional full-access incumbent (`B_inc`) for the response-aware refit. Do not promote any replayed checkpoint as `B_response`: the existing models miss the default finite-response target on the material receiver roles and on pressure/peak changes. This is a compact train-exposed diagnostic, not a held-out performance claim or a feasibility guarantee.

## Exact models and scope

Each checkpoint was loaded by the maintained ThermalChannel loader with its saved input/target normalization, predicted local-port mode, and embedded local surrogate. The absolute panel and response atlas both come from the train split. The response families are calibration neighborhoods on teacher-seen layouts, so they test local response replay rather than unseen-layout generalization. All four calibration contexts are Re=90 with nu=0.01; the sampled train layouts include Re=50 (0001, nu=0.018) and Re=70 (0318/0333/0348, nu=0.012857). The local response panel therefore uses train-exposed geometries at an operating-context shift, not a same-context training replay.

| Replay label | Exact checkpoint | SHA-256 | Epoch / architecture |
|---|---|---|---|
| Run1804 selected e4738 | `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1804_20260905_081349_dense_pairwise_field_adaptation/best_by_field_mse_model.pt` | `71ed480ff0396813491c650dd11d887195174019b373fbd9a1fb25505142c066` | 4738 / `dense_pairwise_field` |
| Run1502 selected e4794 | `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1502_20260923_004751_sparse_incidence_environment_sparsemax/best_by_field_mse_model.pt` | `08d86a573c7f7d86463bde231eb9f84a745602fca2b8f97142f39540d33a85bb` | 4794 / `sparse_incidence_group_control_honf` |
| Run1508 selected e496 | `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1508_20260925_114332_source_conditioned_pairwise_core/best_by_field_mse_model.pt` | `57f6762091c861e9eea2d703c9ebc70470603c3c0988a7be406cc769b8af74c0` | 496 / `source_conditioned_pairwise_honf` |
| Run1508 exact e500 | `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1508_20260925_114332_source_conditioned_pairwise_core/epoch_0500_model.pt` | `f40a16aaba1918b3befc2a4fc0d94deb0b7055bd7f103fd113482d12224549ab` | 500 / `source_conditioned_pairwise_honf` |

Run1508 e496 is the selected best-by-field-MSE state; `epoch_0500_model.pt` is a distinct exact e500 state. The e500 replay is a convenience comparison, not a substitute for the selected checkpoint.

The absolute development panel contains existing train cases `0001` (M3), `0318` (M5), `0333` (M7 hot-receiver layout), and `0348` (M10). The saved response panel contains four existing calibration families `0310` (M3), `0325` (M5), `0340` (M7), and `0355` (M10), each with 11 recorded states. It evaluates absolute values, four finite transitions (`i_plus`, `j_plus`, `heat_transfer_plus`, `pp`), anchored mixed change `pp - i_plus - j_plus + baseline`, and the centered four-corner mixed derivative from `pp`, `pm`, `mp`, and `mm`.

| Family | Active modules | Fluid role | Interface role | Material-solid role |
|---|---:|---:|---:|---:|
| 0310 | 3 | 8,192 × 5 | 192 × 2 | 9,288 × 1 |
| 0325 | 5 | 8,192 × 5 | 320 × 2 | 15,480 × 1 |
| 0340 | 7 | 8,192 × 5 | 448 × 2 | 21,672 × 1 |
| 0355 | 10 | 8,192 × 5 | 640 × 2 | 30,960 × 1 |

The solid role has 3,096 material-coordinate samples per active module. Every finite comparison uses the stencil's aligned common mask and channel schema. The response-relative error below is `||prediction_delta - reference_delta||2 / ||reference_delta||2` on that mask. Its denominator is the actual paired response, not the baseline field norm. Exact zero reference responses have an undefined response-relative error and are excluded from medians; their absolute prediction errors remain in the CSV.

## Absolute development results

The following are four-case means from the existing evaluator using each checkpoint's native normalizers. They are separate from the physical-unit response metrics below.

For `Fluid-field`, `Near-interface`, and `Internal-cell normalized L2`, each case first normalizes predictions and targets with that checkpoint's saved global statistics. It then flattens the selected values and computes `||prediction - target||2 / max(||target||2, 1e-12)`. Fluid uses the module-excluded fluid mask and all five normalized field channels together; near-interface adds the geometry-only mask `0 <= min_module(distance_to_center - radius) <= 0.25`; internal-cell uses every sampled cell in active modules. Thus the fluid-role aggregate is a pooled normalized norm, not an equal-weight average of five channel scores. `Internal-temperature physical relative L2` instead uses denormalized physical temperatures at every sampled cell in active modules and divides the error norm by `max(||physical target||2, 1e-12)`. The reported entries are arithmetic means of the four per-case scores, not a single norm pooled across cases.

| Checkpoint | Fluid-field normalized L2 | Near-interface normalized L2 | Internal-cell normalized L2 | Internal-temperature physical relative L2 | Fixed-grid pressure-probe relative error |
|---|---:|---:|---:|---:|---:|
| Run1804 e4738 | 0.019023 | 0.016684 | 0.041731 | 0.019469 | 0.574% |
| Run1502 e4794 | 0.018999 | 0.015762 | 0.042211 | 0.019735 | 2.384% |
| Run1508 e496 | 0.099307 | 0.110355 | 0.097673 | 0.046085 | 10.376% |
| Run1508 e500 | 0.115282 | 0.132113 | 0.094091 | 0.043645 | 9.229% |

The last column is the existing comparison probe: mean pressure at the 64 fixed fluid-grid coordinates on each outermost x edge, with pressure drop defined as inlet mean minus outlet mean. Its relative error is `abs(predicted_drop - target_drop) / max(abs(target_drop), 1e-12)`, averaged over the four cases. This is not the maintained geometry-derived pressure functional used for the response replay. Run1502 is close to Run1804 on absolute fields, with a slightly better interface mean and slightly worse internal-cell mean; the old fixed-grid pressure probe is worse. Run1508 e496/e500 are materially behind both mature references on fluid and near-interface fields.

## Finite-response panel

There are 4 families × 4 finite transitions = 16 paired transitions per checkpoint. The role medians pool channel/family/transition L2 ratios: 80 fluid-channel rows, 32 interface-channel rows, and 16 solid-temperature rows. Of the 80 fluid rows, 16 are exact-zero target changes, leaving 64 nonzero response denominators; all 32 interface and 16 solid rows are nonzero. The plan's default target is median normalized finite-response error below 0.10.

| Checkpoint | Fluid role median | Interface role median | Solid role median |
|---|---:|---:|---:|
| Run1804 e4738 | 0.112 | 0.724 | 0.519 |
| Run1502 e4794 | 0.093 | 0.699 | 0.553 |
| Run1508 e496 | 0.297 | 0.670 | 0.575 |
| Run1508 e500 | 0.286 | 0.696 | 0.570 |

Only the Run1502 fluid-role median falls below 0.10. Channel separation matters; the fluid-role median does not pass Run1804's thermal response:

| Channel / role | Run1804 e4738 | Run1502 e4794 | Run1508 e496 | Run1508 e500 |
|---|---:|---:|---:|---:|
| Fluid `u` | 0.1097 | 0.0905 | 0.2993 | 0.2874 |
| Fluid `v` | 0.0388 | 0.0382 | 0.1369 | 0.1463 |
| Fluid `p` | 0.0808 | 0.0675 | 0.2005 | 0.1821 |
| Fluid `omega` | 0.1798 | 0.2034 | 0.3701 | 0.3968 |
| Fluid temperature | 0.4407 | 0.4042 | 0.4914 | 0.5362 |
| Interface `T_surface` | 0.4481 | 0.5045 | 0.5483 | 0.5541 |
| Interface `q_normal` proxy | 0.9316 | 0.8743 | 0.9595 | 0.9494 |
| Material solid temperature | 0.5192 | 0.5527 | 0.5746 | 0.5699 |

Each channel has 16 family-transition rows. For fluid `u`, `v`, `p`, and `omega`, the `heat_transfer_plus` reference change is exactly zero in all four families, leaving 12 nonzero denominators per channel; 16 exact-zero rows are omitted from the 80-row fluid role median. These nulls require an error-vs-floor assessment. The material thermal errors are not hidden by fluid channels: for M7 calibration family `0340`, the `i_plus` fluid-temperature response-relative error is 0.975 for Run1804, 1.051 for Run1502, 1.534 for Run1508 e496, and 1.567 for e500. Run1508 e496's worst interface-surface and solid-temperature finite ratios occur in M10 family `0355` (`i_plus`): 1.359 and 1.328. Values at or above 1 mean the prediction error norm equals or exceeds the measured response norm.

## Mixed-response panel and numerical limits

There are four family examples per mixed role. The anchored mixed pooled row counts are 20 fluid (18 nonzero target denominators), 8 interface, and 4 solid channel-family rows; the centered derivative uses the same counts, with 18 nonzero fluid denominators. Centered response units are divided by coordinate squared. The mixed-response denominator is the actual mixed-signal L2. Median target mixed-signal L2 relative to the baseline field L2 is near zero: anchored fluid `1.73e-7`, interface `6.87e-5`, solid `3.87e-5`; centered fluid `1.19e-5`, interface `7.99e-4`, and solid `4.96e-4`.

Consequently, the median anchored-mixed response-relative error is numerically unstable: fluid/interface/solid are 3364.6/40.8/22.3 for Run1804, 2878.8/20.6/13.6 for Run1502, 3374.9/13.2/10.5 for e496, and 3480.6/13.7/10.9 for e500. Centered-mixed derivative ratios are also large (Run1804 2384.5/66.4/40.9; Run1502 1499.2/32.5/24.5; e496 1366.0/19.6/17.1; e500 1303.3/20.8/18.4). These ratios alone do not measure useful fidelity for near-null signals.

The typed atlas records no per-family/per-role noise floors or replicate-derived uncertainty for these mixed labels. The default `<0.25` observed-mixed gate therefore cannot be declared passed or failed from the response-relative ratio alone. Earlier numerical checks on other explicitly repeated ThermalChannel probes are not portable floors for these four families; see [the earlier interaction-response study](20260926_064922Z_HONF_Interaction_Response_and_Inverse_Design_Study.md).

## Pressure, peaks, and decision relevance

Pressure is computed from the predicted fluid field on geometry-derived inlet/outlet sections, each occupying the maintained 8% of domain length. For baseline pressure and peak, relative error is `abs(prediction - reference) / max(abs(reference), 1e-12)`; each checkpoint's mean and worst entry are over the four families. Finite pressure and peak errors are scalar `abs(predicted_delta - reference_delta) / max(abs(reference_delta), 1e-12)`, summarized by the median over 16 family-transition pairs. The true peak is the maximum sampled material-coordinate solid temperature per module, then the maximum over active modules. `q_normal` is a recorded flux proxy, not a conservation certificate.

| Checkpoint | Pressure baseline mean / worst relative error | Max-active true-peak baseline mean / worst relative error | Median pressure finite-delta relative error | Median max-peak finite-delta relative error |
|---|---:|---:|---:|---:|
| Run1804 e4738 | 1.38% / 2.72% | 1.70% / 2.37% | 73.3% | 69.4% |
| Run1502 e4794 | 1.01% / 2.60% | 1.68% / 2.59% | 45.7% | 83.8% |
| Run1508 e496 | 4.22% / 8.47% | 3.85% / 5.91% | 71.4% | 67.0% |
| Run1508 e500 | 3.82% / 6.53% | 6.77% / 10.97% | 57.6% | 75.8% |

Pressure baseline denominators are the four family drops (mean reference about `0.0793` in dataset pressure units). Across the 16 finite transitions, mean absolute pressure-drop change is `3.95e-4`; the median relative errors above divide by each transition's measured pressure change. The mean absolute error on maximum-module peak changes is `0.134`, `0.182`, `0.138`, and `0.156` temperature units in checkpoint order, against a mean measured absolute peak change of `0.194`. These finite functional errors do not meet a decision-readiness bar even when baseline absolute pressure and peak values are close.

The calibration atlas carries no fixed allowable-pressure margin, so this replay cannot assess margin-scaled pressure error or false-feasible frequency. The prior independent M3 common-pool report remains separate evidence: selected candidate `c04` was predicted feasible but had actual pressure `0.14047` above the `0.13649` limit; the present four-family replay did not rerun that pool. It also does not estimate decision ranking or uncertainty on an independent candidate set.

## M7/0340 numerical-reliability pilot

A focused numerical-sensitivity pilot used the existing M7 calibration family `0340` at Re=90, `nu=0.01`, with the exact seven states `baseline`, `i_plus`, `j_plus`, `pp`, `pm`, `mp`, and `mm`. The two position perturbations use `h=0.1`. The original 128×64 default-tolerance states were reused. Fourteen new reference-solver attempts covered the same seven states at (1) 128×64 with tightened tolerances (`convergence_tol=1e-5`, `convergence_rel_tol=1e-6`) and (2) 256×128 with the original default tolerances (`1e-4`, `1e-5`). All 14 attempts converged. This is one train-exposed calibration family, not an independent validation set.

The solver is the local NumPy analytic-wake/shared-grid thermal reference, not CFD. `CUDA_VISIBLE_DEVICES=2` was set for process hygiene; no GPU computation was used. The first coarse/tight baseline solve completed and its raw case was recovered after the attempt-result logger failed on a `mappingproxy`; that physical input was not rerun. Its adapter wall and process-CPU times were not preserved. For the other 13 calls, the summed adapter wall was `136.556 s`, resumed-process CPU time was `136.853 s`, and resumed-runner wall was `136.877 s`. The resumed process used 3,000-second CPU and wall caps, reserving 600 seconds against the first call's unknown time; timing is consequently incomplete for the full 14-call pilot.

The following are physical-unit mixed-signal RMS values. `D` is reused 128×64 default, `T` is new 128×64 tightened, and `F` is new 256×128 default. The anchored change is `pp - i_plus - j_plus + baseline`; the centered quantity is `(pp - pm - mp + mm)/(4h²)`. Counts are common valid samples per channel and condition.

| Role / channel | Anchored RMS D / T / F | Centered derivative RMS D / T / F | Valid count D / T / F |
|---|---:|---:|---:|
| Fluid temperature | 0.019906 / 0.019901 / 0.020898 | 2.84986 / 2.84987 / 3.07463 | 7,645 / 7,645 / 30,579 |
| Fluid `omega` | 0.006605 / 0.006605 / 0.010795 | 0.59729 / 0.59729 / 0.92401 | 7,645 / 7,645 / 30,579 |
| Fluid `u` | 0.000898 / 0.000898 / 0.000949 | 0.069849 / 0.069849 / 0.074033 | 7,645 / 7,645 / 30,579 |
| Fluid `p` | 5.81e-9 / 5.81e-9 / 5.30e-9 | 1.46e-7 / 1.46e-7 / 1.60e-7 | 7,645 / 7,645 / 30,579 |
| Fluid `v` | 9.14e-10 / 9.14e-10 / 8.96e-10 | 2.28e-8 / 2.28e-8 / 2.30e-8 | 7,645 / 7,645 / 30,579 |
| Interface `T_surface` | 0.029862 / 0.029857 / 0.029163 | 4.22541 / 4.22543 / 4.33953 | 448 / 448 / 448 |
| Interface `q_normal` proxy | 0.040620 / 0.040620 / 0.043145 | 6.05210 / 6.05210 / 6.65526 | 448 / 448 / 448 |
| Material solid temperature | 0.026836 / 0.026829 / 0.026247 | 3.73461 / 3.73463 / 3.82297 | 21,672 / 21,672 / 21,672 |

The RMS ranking across all three settings is stable: fluid temperature > `omega` > `u` > `p` > `v`, and interface `q_normal` proxy > `T_surface`. Fluid `p` and `v` are effectively null channels here, so their tiny RMS and perfect-looking rankings are not evidence of useful mixed-response signal. The aggregate signed means keep the same direction for fluid temperature, `omega`, `u`, pressure, interface temperature, the `q_normal` proxy, and solid temperature. The fluid-`v` mean changes sign at magnitudes around `1e-12`, which is also a null response.

At fixed 128×64 resolution, tightening tolerance changes anchored RMS by only `8.25e-6` for fluid temperature (0.041% of its D signal RMS), `9.30e-6` for interface temperature (0.031%), `1.18e-5` for the `q_normal` proxy (0.029%), and `1.07e-5` for solid temperature (0.040%). The corresponding pressure state outputs are unchanged at the recorded precision; max-peak state shifts are at most 0.00121 temperature units. This measures tolerance sensitivity for this family and solver setup, not a portable noise floor.

Exact paired material coordinates show larger mesh sensitivity. Coarse-to-fine anchored mixed RMS shifts are 0.003983 for interface temperature (13.3% and 13.7% of the D and F signal RMS), 0.026017 for the `q_normal` proxy (64.0% and 60.3%), and 0.002808 for solid temperature (10.5% and 10.7%). For centered derivatives, those shift-to-signal ratios are 9.9%/9.6%, 39.4%/35.8%, and 8.8%/8.6%, respectively. Interface and solid comparisons pair exact local coordinates and physical receiver-module IDs; all 448 interface and 21,672 solid samples pair. The Eulerian 128×64 and 256×128 cell centers have no exact common coordinates, so no cross-mesh fluid-vector shift is reported and no interpolation is used. Per-grid fluid summaries remain available.

Pointwise sign and magnitude-rank stability should be read separately. On the coarse grid, default-to-tight sign agreement ranges 0.861–1.000 for fluid channels, 0.904–0.935 for interface channels, and 0.908–0.929 for solid temperature; Spearman correlation of absolute pointwise mixed magnitudes is 0.934–1.000, 0.989–0.993, and 0.990–0.992, respectively. Across coarse-to-fine exact material coordinates, interface sign agreement is 0.750–0.855 and solid-temperature agreement is 0.755–0.865, while absolute-magnitude rank correlations remain 0.959–0.982 and 0.982–0.984. Near-zero pointwise values make sign agreement fragile; the rank results describe spatial ordering of magnitudes, not predictive fidelity. Cross-grid fluid sign and rank comparisons are unavailable because there are no exact shared cell centers.

The sampled max-module peak is more concerning for decisions. Baseline pressure drops are 0.08244767 at D/T and 0.08245022 at F; no allowable pressure threshold is recorded for family `0340`, so `pressure_limit - pressure_drop` and feasibility margin are unavailable at every grid. Do not borrow the distinct M3 limit. Pressure finite-change ordering and signs are stable across the two resolutions, but the anchored pressure mixed values are near zero (`-7.34e-10` at D/T and `-3.34e-10` at F), so they do not support a pressure-interaction claim.

| Move | Pressure change D/T | Pressure change F | Max-peak change D | Max-peak change T | Max-peak change F |
|---|---:|---:|---:|---:|---:|
| `i_plus` | +2.72787e-4 | +2.72806e-4 | -0.06403 | -0.06399 | -0.07282 |
| `j_plus` | -1.861e-8 | -1.924e-8 | -0.01152 | -0.01155 | +0.12605 |
| `pp` | +2.72768e-4 | +2.72786e-4 | -0.02850 | -0.02853 | +0.10040 |
| `pm` | +2.72795e-4 | +2.72812e-4 | -0.18230 | -0.18233 | -0.25983 |
| `mp` | -2.59660e-4 | -2.59680e-4 | +0.02500 | +0.02498 | +0.17328 |
| `mm` | -2.59637e-4 | -2.59655e-4 | +0.10818 | +0.10815 | +0.06804 |

The peak-change order is `pm < i_plus < pp < j_plus < mp < mm` at D/T, but `pm < i_plus < mm < pp < j_plus < mp` at F; `j_plus` and `pp` both reverse sign after mesh doubling. The max remains `0340:module:3` in every state, so this is mesh sensitivity of that module's sampled peak, not a switch in which module owns the maximum. Although the anchored max-peak mixed change stays positive and close (`+0.04705`, `+0.04701`, `+0.04717` at D/T/F), it does not resolve the finite-move sign/order instability. This pilot therefore supplies a family-specific numerical sensitivity check, not a portable floor or a decision-readiness pass.

The ignored harness, exact per-attempt configs, all raw cases, recovered first-call lineage, metrics, and common-support masks are retained under `diagnostics/generated/baseline_decision_m0_20260926/mixed_floor_m7_0340/`. The initial serializer failure and postprocessing recovery are recorded in `attempts.jsonl`, `postprocess_events.jsonl`, and the run-manifest snapshots. No further families or reference solves were run.

## M0 gate and next use

- **Provisional `B_inc` reference for refit:** Run1804 selected e4738, because it is the mature dense baseline with strong absolute fluid, receiver, internal-temperature, pressure, and sampled-peak values on this compact panel. This label selects the reference for Stage A/B; it does not mean the checkpoint passed a response or decision-readiness gate. Its fluid-temperature, interface, solid-temperature, pressure-change, and peak-change errors remain material. Run1502 remains a close mature comparator, not a reason to replace the provisional reference. The refit must pass the response and constraint gates before any model is promoted as decision-ready.
- **`B_response`:** none of these checkpoints qualifies. Keep the exact e496 Run1508 checkpoint as a possible explicit-refit initialization; do not treat e500 as the selected state or claim prediction identity after conversion to the new core.
- **Warm-start inventory:** an initial CPU-only core check strict-loaded all 86 target core tensors from e496, with 0 initialized target keys, 26 source-only `backend.*` keys discarded, and 0 unmaterialized tensors. A later one-update preflight on physical GPU 2 repeated the 86/0/26 transfer through the native ThermalChannel wrapper, attached and froze e496's embedded local surrogate, and verified target-free inputs, optimizer coverage, module permutation, and whole-wrapper heating/position derivative checks. Its one optimizer update and checkpoint round trip establish execution correctness, not response fit or prediction identity. Both checks are explicit refits from e496; neither claims an exact output conversion.
- **Decision limit:** the response panel is train-exposed local calibration, not held testing. Do not claim reliable feasibility, mixed-response fidelity, generalization, learned causal interactions, adaptive-K benefit, or runtime improvement from M0.

## Reproduction and retained evidence

No training was run. The four-checkpoint replay used existing labels and ran on physical GPU2, UUID `GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39`; synchronized prediction wall was `35.919 s` and end-to-end wall was `51.844 s`. It emitted 4,404 metric rows across four exact checkpoints. Separately, the M7/0340 sensitivity pilot made 14 new NumPy CPU reference-solver attempts; 13 calls have recorded solver and process timing, while the recovered first call's timing is unknown. A first checkpoint-adapter preflight had failed before prediction after `0.849 s`; its failure manifest is preserved separately.

The replay script and exact checkpoint manifest are retained locally under ignored `diagnostics/generated/baseline_decision_m0_20260926/replay_harness/`. Raw response metrics, predictions, manifests, absolute panel tables, and the failed-preflight manifest are under ignored `diagnostics/generated/baseline_decision_m0_20260926/`. Key tables are `absolute/tables/model_summary_metrics.csv`, `response/response_metrics.csv`, and `response/response_macro_summary.csv`.

Replay command (from `HONF_Proj`):

```bash
rtk proxy conda run -n ModularDT env CUDA_VISIBLE_DEVICES=2 python diagnostics/generated/baseline_decision_m0_20260926/replay_harness/replay_baseline_response.py \
  --device cuda:0 --physical-gpu-index 2 --physical-gpu-uuid GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39 \
  --manifest diagnostics/generated/baseline_decision_m0_20260926/replay_harness/baseline_m0_manifest.json \
  --atlas-root diagnostics/generated/interactions/physical_response_atlas_20260926/families \
  --output-dir diagnostics/generated/baseline_decision_m0_20260926/response --query-batch-size 32768
```
