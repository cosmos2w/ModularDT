# HONF decision-aware dynamic-K study

## M0 — ThermalChannel local truth and baseline

**Gate result:** Run1804 selected e4738 is the provisional dense full-access incumbent (`B_inc`) for the next response-aware refit. No replayed checkpoint is ready to serve as `B_response`; the current mature references miss the default material-response target, and their finite pressure-drop and true-peak changes are not decision-ready. This result comes from train-exposed development and calibration data. It does not establish held-out generalization, feasibility reliability, or the value of adaptive K.

### Exact checkpoints and absolute panel

| Model | Exact state | SHA-256 | Epoch / architecture |
|---|---|---|---|
| Run1804 selected | `Run_1804_20260905_081349_dense_pairwise_field_adaptation/best_by_field_mse_model.pt` | `71ed480ff0396813491c650dd11d887195174019b373fbd9a1fb25505142c066` | e4738 / dense pairwise |
| Run1502 selected | `Run_1502_20260923_004751_sparse_incidence_environment_sparsemax/best_by_field_mse_model.pt` | `08d86a573c7f7d86463bde231eb9f84a745602fca2b8f97142f39540d33a85bb` | e4794 / sparse incidence |
| Run1508 selected | `Run_1508_20260925_114332_source_conditioned_pairwise_core/best_by_field_mse_model.pt` | `57f6762091c861e9eea2d703c9ebc70470603c3c0988a7be406cc769b8af74c0` | e496 / source-conditioned pairwise |
| Run1508 exact e500 | same Run1508 directory, `epoch_0500_model.pt` | `f40a16aaba1918b3befc2a4fc0d94deb0b7055bd7f103fd113482d12224549ab` | e500 / source-conditioned pairwise |

The Run1508 selected e496 state and exact e500 state are distinct; e500 is a convenience comparison, not the selected checkpoint. The absolute development sample is 4 train cases: `0001` M3, `0318` M5, `0333` M7 hot receiver, and `0348` M10. The existing evaluator used checkpoint-native normalizers and predicted local ports.

| Model | Fluid normalized L2 | Near-interface normalized L2 | Internal-cell normalized L2 | Internal temperature physical relative L2 | Fixed 64-point pressure-probe relative error |
|---|---:|---:|---:|---:|---:|
| Run1804 e4738 | 0.019023 | 0.016684 | 0.041731 | 0.019469 | 0.574% |
| Run1502 e4794 | 0.018999 | 0.015762 | 0.042211 | 0.019735 | 2.384% |
| Run1508 e496 | 0.099307 | 0.110355 | 0.097673 | 0.046085 | 10.376% |
| Run1508 e500 | 0.115282 | 0.132113 | 0.094091 | 0.043645 | 9.229% |

Each value above averages 4 cases. The old pressure column uses fixed outermost 64 fluid-grid coordinates and is not the maintained geometry-derived section functional below.

The fluid, near-interface, and internal-cell scores are computed after normalizing targets and predictions with each checkpoint's saved global statistics. For each case, the evaluator flattens selected values and computes `||prediction - target||2 / max(||target||2, 1e-12)`. Fluid uses the module-excluded fluid mask over all five normalized channels together; near-interface adds `0 <= min_module(distance_to_center - radius) <= 0.25`; internal-cell uses all sampled cells in active modules. Internal-temperature physical relative L2 instead uses denormalized temperatures over active-module cells and denominator `max(||physical target||2, 1e-12)`. The table reports the arithmetic mean of four per-case scores, not one pooled norm. The fixed-probe pressure relative error is `abs(predicted_drop - target_drop) / max(abs(target_drop), 1e-12)`, with each drop formed from the mean of 64 fixed fluid-grid points at the inlet edge minus the mean at the outlet edge.

### Response panel and finite gate

The replay used only the existing `calibration_0310/0325/0340/0355` response NPZ/JSON records. These are four local-response neighborhoods on teacher-seen train layouts (M3/5/7/10), not unseen-layout tests. All four calibration contexts are Re=90 with nu=0.01; sampled train layouts include Re=50 (0001, nu=0.018) and Re=70 (0318/0333/0348, nu=0.012857), so this panel is train-exposed by geometry but shifted in operating context. Each family has 11 recorded states, so each checkpoint has 44 absolute design-state predictions and 16 paired finite transitions (4 families × `i_plus`, `j_plus`, `heat_transfer_plus`, `pp`). Per state, fluid output has 8,192 points × 5 channels; interface output has 64 angles per module × 2 channels; solid output has 3,096 material-coordinate samples per module × 1 channel.

Finite error is `||prediction_delta-reference_delta||2 / ||reference_delta||2` on the stencil common mask. The default target is median `<0.10`. The medians pool 80 fluid channel/family/transition entries, 32 interface entries, and 16 solid entries; they are not independent test-case counts. Sixteen of the 80 fluid entries are exact-zero target changes, leaving 64 nonzero response denominators; all interface and solid entries are nonzero.

| Model | Fluid role | Interface role | Solid-temperature role |
|---|---:|---:|---:|
| Run1804 e4738 | 0.112 | 0.724 | 0.519 |
| Run1502 e4794 | 0.093 | 0.699 | 0.553 |
| Run1508 e496 | 0.297 | 0.670 | 0.575 |
| Run1508 e500 | 0.286 | 0.696 | 0.570 |

Channel medians expose the thermal miss and distinguish nonzero changes from exact null controls:

| Channel / role | Run1804 | Run1502 | Run1508 e496 | Run1508 e500 |
|---|---:|---:|---:|---:|
| Fluid `u` | 0.1097 | 0.0905 | 0.2993 | 0.2874 |
| Fluid `v` | 0.0388 | 0.0382 | 0.1369 | 0.1463 |
| Fluid `p` | 0.0808 | 0.0675 | 0.2005 | 0.1821 |
| Fluid `omega` | 0.1798 | 0.2034 | 0.3701 | 0.3968 |
| Fluid temperature | 0.4407 | 0.4042 | 0.4914 | 0.5362 |
| Interface `T_surface` | 0.4481 | 0.5045 | 0.5483 | 0.5541 |
| Interface `q_normal` proxy | 0.9316 | 0.8743 | 0.9595 | 0.9494 |
| Material solid temperature | 0.5192 | 0.5527 | 0.5746 | 0.5699 |

Each channel has 16 transitions. For fluid `u`, `v`, `p`, and `omega`, the four `heat_transfer_plus` reference changes are exactly zero; those transitions have no response-relative denominator, leaving 12 nonzero reference changes per channel. The worst high-peak M7 family `0340` fluid-temperature transition is `i_plus`: relative errors are 0.975 (Run1804), 1.051 (Run1502), 1.534 (e496), and 1.567 (e500). Run1508 e496's worst material temperature and interface surface-temperature cases occur in M10 family `0355`, `i_plus`: 1.328 and 1.359. The `q_normal` outputs remain a proxy.

### Mixed responses, pressure, and sampled peaks

There are 4 family examples per mixed role. Pooled row counts are 20 fluid (18 nonzero target denominators), 8 interface, and 4 solid channel/family rows for both anchored mixed and centered derivative metrics; centered fluid also has 18 nonzero denominators. Their denominator is the measured mixed-signal L2 (centered derivative units are divided by coordinate squared). Median target-signal L2 relative to baseline L2 is near zero: anchored fluid `1.73e-7`, interface `6.87e-5`, solid `3.87e-5`; centered fluid `1.19e-5`, interface `7.99e-4`, solid `4.96e-4`. The atlas includes no noise-floor or replicate-uncertainty fields, so the default `<0.25` observed-mixed criterion cannot be decided from these unstable ratios. Earlier numerical checks on other explicit probes are not portable floors for these four families; see [baseline details](20260926_195912Z_baseline_decision.md) and [the earlier interaction-response study](20260926_064922Z_HONF_Interaction_Response_and_Inverse_Design_Study.md).

Pressure uses fluid-only inlet/outlet sections that each occupy the maintained 8% of domain length. For pressure and peak baselines, relative error is `abs(prediction - reference) / max(abs(reference), 1e-12)`; each mean and worst value is over four calibration families. Finite functional errors are `abs(predicted_delta - reference_delta) / max(abs(reference_delta), 1e-12)`, summarized by the median over 16 family-transition pairs. True peak means the sampled material-coordinate solid-temperature maximum per module, followed by the maximum over active modules. There are 4 baseline functional values and 16 finite changes per checkpoint. Mean baseline drop is about `0.0793`; mean absolute finite pressure change is `3.95e-4`. Mean absolute finite max-peak change is `0.194` temperature units.

| Model | Pressure baseline mean / worst error | Max true-peak baseline mean / worst error | Median pressure finite-delta error | Median max-peak finite-delta error |
|---|---:|---:|---:|---:|
| Run1804 e4738 | 1.38% / 2.72% | 1.70% / 2.37% | 73.3% | 69.4% |
| Run1502 e4794 | 1.01% / 2.60% | 1.68% / 2.59% | 45.7% | 83.8% |
| Run1508 e496 | 4.22% / 8.47% | 3.85% / 5.91% | 71.4% | 67.0% |
| Run1508 e500 | 3.82% / 6.53% | 6.77% / 10.97% | 57.6% | 75.8% |

The pressure relative-error denominator is each measured pressure change; its mean absolute magnitude is 0.5% of the mean baseline drop. The peak finite-change mean absolute prediction errors, in model order, are 0.134, 0.182, 0.138, and 0.156 temperature units against the 0.194-unit mean target. The calibration data contains no allowable-pressure margin, so margin-scaled error and false-feasible rate are unavailable. The earlier M3 common-pool evidence is separate: selected `c04` was predicted feasible but actual pressure was `0.14047` above the `0.13649` limit ([prior study](20260926_064922Z_HONF_Interaction_Response_and_Inverse_Design_Study.md)). This replay did not rerun that pool.

### Warm-start and M0 disposition

Run1508 e496's checkpoint loader attaches its embedded local model (`internal_prediction_mode=local_surrogate`); its local-model provenance is `Trained_Results/ThermalChannel/Local_Module_Runs/thermal_disk/Run_0000_base/best_model.pt`. An initial CPU-only materialized-core check strict-loaded all 86 target `three_term_full_access_honf` tensors (0 initialized, 26 source-only `backend.*` tensors discarded, 0 unmaterialized) but did not attach the local surrogate. A subsequent physical-GPU-2 one-update preflight repeated that transfer through the native wrapper with the embedded local surrogate attached and frozen. It passed target-free input, optimizer coverage, module permutation, whole-wrapper heating/position AD/FD, and optimizer/checkpoint round-trip checks. This establishes execution correctness after an explicit refit transition, not prediction identity or response fidelity.

**Disposition:** Run1804 is the provisional `B_inc` reference for Stage A/B refit, not a response- or decision-ready checkpoint; its thermal-role and finite pressure/peak errors remain material. None of the four exact snapshots is `B_response`-ready, and the refit must pass response and constraint gates before promotion. Do not claim mixed-response fidelity, constraint reliability, held-out accuracy, causal interactions, adaptive-K benefit, or speedup from this M0 evidence. The four-checkpoint replay used existing reference labels. A separate M7/0340 numerical-sensitivity pilot made 14 new NumPy CPU solver attempts; its family-specific tolerance, mesh, sign/order, timing, and pressure-margin limits are documented in [20260926_195912Z_baseline_decision.md](20260926_195912Z_baseline_decision.md). The local raw replay artifacts are ignored under `diagnostics/generated/baseline_decision_m0_20260926/`.

## Stage A paired-response fit: frozen e100 review protocol

Before launching the paired fit, the training panel was fixed to three existing physical-response stencils: train `0001` (M3, Re50), train `0304` (a second M3, Re70), and train `0333` (M7, Re70), with 11 states per stencil. The separate development panel is calibration `0310/0325/0340/0355` (M3/M5/M7/M10, Re90); it is train-exposed by layout and provides no held claim. The source is exact Run1508 selected e496 with its frozen embedded local surrogate. A strict GPU2 native one-update preflight passed before this paired fit. Two one-update preflight attempts consumed optimizer work across the failed and successful harness runs; neither is part of the matched arms.

`B_value` and `B_response` start from the same materialized three-term full-access state, use the same deterministic family/value-sample order, and stop after 100 actual updates for review. The first 20 `B_response` updates are value-only; updates 21–100 add observed finite responses. Decision and pressure-constraint losses begin only after u100 if a continuation is justified. Mixed labels lack a measured train-family noise floor and remain unknown, so no mixed loss is enabled. The source-to-target tensor transfer is an explicit refit, not an identity conversion. Training uses 2,048 fluid rows per family while retaining all 1,280 inlet/outlet-band rows, all interface ports, and sampled material rows that include every observed per-state/module peak. The sample seed is 2317; development evaluation uses full stencil grids.

The e100 review compares per-family and per-role finite physical-unit errors against `B_value` and the zero-change predictor, while protecting absolute near-interface/receiver quality. The plan's defaults are median resolved finite normalized error below 0.10, no more than 5% relative degradation of the protected near-interface/receiver metric, and lower resolved-response error than zero change; near-null entries require a measured floor. Pressure absolute and incremental errors are compared with each family's declared 5%-of-original-baseline benchmark allowance; false-feasible selections are counted rather than hidden by a mean. The four development families cannot establish a reliable false-feasible rate. Mixed-response metrics are exploratory until their numerical floor is established. No checkpoint selection, dynamic cover training, or inverse trial follows automatically from a finite gradient or an e100 training-loss decrease.

The checked-in Stage A recipe allows at most 500 epochs, 2,000 recipe updates, 20,000 formal-arm updates, or the time allocation, whichever binds first. With only three training stencils, the effective epoch cap is 1,500 updates per arm. This e100 launch has a 900-second fit-wall cap per arm and a 2,100-second whole-process timeout; it makes zero new physical solves. The ignored run directory is `diagnostics/generated/response_control_runs/Run1508_e496_stageA_train0001_0304_0333_paired100_20260926/`, including source SHA-256 inventory, numbered model/optimizer/RNG checkpoints, and the eventual paired curves and development metrics.

### Stage A u100 review: recipe 1 did not pass promotion

Both matched arms completed exactly 100 optimizer updates from the same explicit Run1508 e496-to-three-term refit initialization, on physical GPU 2 (UUID `f6a4ddbb-ad44-5ef5-0421-eecf7120df39`). The paired fit took 722.16 s end to end (684.34 s fit; B_value 337.71 s, B_response 340.41 s), with 34.28 GB peak allocated and 36.59 GB reserved. The full-grid read-only replay of both arms on three train and four calibration-development stencils took 58.52 s and peaked at 3.62 GB allocated. It made no new physical solves. Exact source, u100 checkpoint, and code hashes, outputs, and metrics are retained in the ignored `diagnostics/generated/response_control_runs/Run1508_e496_stageA_train0001_0304_0333_paired100_20260926/` directory; B_value and B_response u100 checkpoint hashes are respectively `237eb0280515a784dfeeefedb83c345251854bd40cdb2b0ae9e6a6de2c2d3878` and `10a99aca507a10cb0c2b55831a9f86c1b19e906434f9b7f1c0e40d470f8300dc`.

The following entries are means of the saved full-grid `signal_relative_rmse` channel metrics over each split's families and stored absolute or finite labels. They are descriptive summaries, not independent point-level samples. Calibration remains local-response development, not a held set.

| Metric | Train B_value → B_response | Calibration-development B_value → B_response | Review |
|---|---:|---:|---|
| Finite interface `T_surface` | 0.7643 → 0.5808 | 0.6328 → 0.6552 | Train gain did not transfer consistently. |
| Finite interface `q_normal` proxy | 0.8551 → 0.8160 | 0.8591 → 0.8172 | Small relative gain; remains far above the 0.10 target and is not a conservation result. |
| Finite solid temperature | 0.7720 → 0.5563 | 0.6399 → 0.6677 | Clear train gain, but worse on the calibration panel. |
| Absolute near-interface fluid `u` | 0.0661 → 0.0899 | 0.1026 → 0.1204 | Protected receiver guard worsened in both splits. |
| Absolute full-grid fluid `p` | 0.0651 → 0.0977 | 0.1576 → 0.1826 | Absolute pressure-field accuracy worsened. |

Train finite thermal loss is still learning: B_response improves `T_surface` and solid-temperature finite metrics, and also reduces fluid-temperature finite error (0.4994 → 0.4517). However, `q_normal` remains weak, several absolute roles regress, and the 5% near-interface guard fails. Development finite results improve only for `q_normal`; `T_surface`, solid temperature, and most fluid channels are level or worse. At u100 this is a fit-capability signal, not a promotion result. Exact-null finite `v`/`p` entries have no measured physical noise floor and are not counted as resolved evidence; mixed labels remain unknown.

#### Corrected near-interface sampling audit

The original recipe sampled 2,048 of 8,192 fluid rows and protected all 1,280 pressure-band rows, but it did not explicitly protect the full `0 <= min(distance to active module center) - radius <= 0.25` band. A recomputation on the typed full-grid coordinates and each record's own geometry found the following per-state full versus sampled near-interface rows:

| Train family | Full-grid near rows across its 11 states | Recipe-1 sampled near rows | Union of near rows over states | Pressure + near protected union |
|---|---:|---:|---:|---:|
| 0001 (M3, Re50; duplicate family `0001+0273`) | 301–314 | 36–46 | 462 | 1,742 |
| 0304 (M3, Re70) | 305–317 | 26–28 | 468 | 1,748 |
| 0333 (M7, Re70) | 720–733 | 87–92 | 877 | 2,146 |

The earlier `pre_fit_train_scale_and_sampling_audit.json` incorrectly labeled sampled-coordinate near counts as full-grid counts and reported 100% coverage. That file is preserved; the corrected row-by-row calculation is in `pre_fit_train_scale_and_sampling_audit_corrected.json`. Recipe 1 retained only about 8–15% of full near-interface rows, so its near-interface regression may be sampler-limited. This correction is not evidence that a better sampler will pass the guard.

The frozen recipe-2 sampler now protects the union of all valid near-interface rows across all stencil states and all pressure bands, with a fixed 3,072 fluid rows per family for both arms. It leaves 1,330, 1,324, and 926 inverse-probability-weighted nonprotected rows for the three families, respectively. To isolate the sampling amendment, recipe 2 reuses the exact recipe-1 train-derived channel/functional scales and calibrated response multipliers rather than deriving new values from the changed row set; the run manifest records both source-file hashes. The original audit's `frozen_scales` object remains the numeric scale source, while its near-interface counts are superseded by the separate corrected audit. This preserves the full-grid evaluation, initialization, response schedule, held calibration panel, and matched B_value/B_response comparison. It is a pre-launch evidence-driven amendment; recipe-1 checkpoints and metrics remain the result for that recipe.

#### Pressure-limit and constraint review

The per-family pressure threshold is 1.05 times the original physical baseline and is a study benchmark, not an engineering limit. Every train reference state was below it (33/33), as were all 44 calibration-development reference states. The zero false-feasible count on development is therefore vacuous; there are no independent near-boundary infeasible labels. At the absolute baseline, B_response has 22/44 false-infeasible decisions and B_value has 33/44; neither result establishes calibrated feasibility. B_response median absolute pressure error is 0.54–2.61 times the 5% study-margin allowance across the four development families.

At u100 the frozen B_response constraint multiplier is `0.0025616145519238133`. On the train-only sampled panel, omitting a single-class feasibility BCE changes the unweighted constraint gradient RMS from 67.2757 to 97.5577, versus 0.2060 for value; the saved multiplier gives an amended weighted RMS about 1.21 times the value gradient. The multiplier is retained for recipe 2 rather than recalibrated at u100. Its first decision review must continue to omit the BCE for any all-one-class stencil while preserving continuous absolute and finite pressure losses.

**Decision:** do not promote recipe 1 and do not continue its u100 checkpoints to u300. The finite thermal gains justify one bounded matched fit-capability rerun from the exact e496 refit initialization with the corrected, fixed-3,072-row sampler. Review both arms at u100 before any continuation. No inverse study, held-out claim, constraint-reliability claim, or mixed-response claim follows from this u100 evidence.

### Stage A u100 review: recipe 2 also fails promotion

Recipe 2 used the corrected, deterministic 3,072-fluid-row panel: all pressure-band rows and the full near-interface-row union across each family's 11 states were protected, with inverse-probability weighting on the remaining fluid rows. Interface ports and protected solid-peak locations were retained. It reused recipe 1's frozen train-only loss scales and response multipliers, so this run tests the corrected sampling panel at the same numeric objective calibration. The training families were `0001` (M3, Re50; physical family `0001+0273`), `0304` (M3, Re70), and `0333` (M7, Re70); the four calibration families remained development-only and are not a held-out split.

Both paired arms started from the explicit Run1508 e496 refit and stopped at exactly 100 successful optimizer updates. The source checkpoint SHA-256 is `57f6762091c861e9eea2d703c9ebc70470603c3c0988a7be406cc769b8af74c0`. The u100 `B_value` and `B_response` checkpoint hashes are `66db6c43b07ae2fd1af3e3f1c22eff179a7614e278bb2a67134ffc3a88e21651` and `9e2b6a431a1338d673ca3a4c01fa0fc5d1f4fe1cfadf9d117ff98074b409e82d`, respectively. Exact numbered checkpoints, optimizer/RNG/sampler state, curves, read-only full-grid train replay, and manifests are under the ignored `diagnostics/generated/response_control_runs/Run1508_e496_stageA_recipe2_near_guard_3072_20260926/paired_u100_attempt01/` directory.

The paired fit used physical GPU 2 (UUID `f6a4ddbb-ad44-5ef5-0421-eecf7120df39`): 761.16 s fit wall, 799.66 s whole-process wall including development replay, 4.41 GB peak allocated during fit, and 34.27 GB peak allocated during development replay. A separate read-only full-grid replay of both u100 checkpoints on the three train stencils took 16.32 s; it made zero optimizer updates and zero physical solves. Its peak allocation was 86.2 MB per arm. No reference solve was run for recipe 2.

On train finite responses, the median paired change in signal-relative error (negative is an improvement) was:

| Train family | Interface `T_surface` | Interface `q_normal` proxy | Solid temperature |
|---|---:|---:|---:|
| 0001 / M3 | −18.1% | −5.2% | −21.5% |
| 0304 / M3 | −26.4% | −4.9% | −23.0% |
| 0333 / M7 | −48.3% | −5.3% | −53.0% |

The response arm's finite objective was active at updates 21–100 and declined from 0.415 at update 21 to 0.231 at update 100. The `T_surface` and solid-temperature finite improvements are a fit-capability signal, but remain well above the 0.10 target; the `q_normal` proxy remains weak. Calibration-development changes were heterogeneous by family. Near-null finite channels and mixed responses have no measured per-role train noise floor, so they remain unresolved rather than being counted as successful fidelity.

The protected near-interface fluid guard uses the full-grid band `0 <= min(surface distance) <= 0.25`; cells are compared by channel and family. Entries below are median paired changes in signal-relative error for `B_response` versus `B_value` (positive means worse). Several channels exceed the 5% degradation allowance, including large train regressions in velocity and vorticity:

| Split / family | `u` | `v` | `p` | `omega` | Temperature |
|---|---:|---:|---:|---:|---:|
| Train 0001 / M3 | −3.4% | +52.0% | +7.8% | +24.1% | −4.8% |
| Train 0304 / M3 | +26.2% | +42.9% | −3.6% | +38.6% | +8.0% |
| Train 0333 / M7 | +10.1% | +15.7% | +21.8% | +14.0% | +18.7% |
| Calibration 0310 / M3 | +0.6% | +10.7% | +14.6% | +17.0% | +10.9% |
| Calibration 0325 / M5 | −18.1% | +10.8% | −3.9% | +6.0% | −10.4% |
| Calibration 0340 / M7 | +5.4% | −1.0% | −12.6% | −3.3% | −5.2% |
| Calibration 0355 / M10 | +3.0% | +8.3% | +13.8% | +4.2% | −12.1% |

The worst paired per-state degradation was +71.8% (train 0001, `v`) and +49.5% (calibration 0310, `omega`). The u100 near-interface guard therefore fails; an aggregate channel mean would hide important tails.

The pressure threshold remains each family's `1.05 × original baseline` study benchmark, not an engineering limit. All 33 train and 44 development reference states are feasible, so false-feasible rates of 0/33 and 0/44 are vacuous. On train, B_response median absolute pressure error is 1.84, 1.27, and 0.69 times the 5%-of-baseline study margin for 0001, 0304, and 0333; B_value is 0.45, 0.65, and 0.32 times those margins. On development B_response has 22/44 false-infeasible decisions versus 33/44 for B_value, with no false-feasible decisions in either arm. B_response median absolute pressure error spans 0.64–2.81 times the study margin by family. These labels do not establish calibrated feasibility because the reference panel contains no infeasible boundary contrast.

**Decision:** recipe 2 also fails the promotion gates: near-interface errors breach the 5% guard, absolute and finite pressure errors are large relative to the study margin, and the reference panel cannot calibrate feasibility. The declining train finite loss supports only a bounded fit-capability diagnosis. One matched u100-to-u300 continuation is authorized from these exact paired checkpoints, carrying optimizer, RNG, and sampler state, frozen numeric scales, and saved response weights. At updates 101–300, the predeclared decision and continuous-constraint terms activate; single-class feasibility BCE remains omitted and mixed-response loss remains excluded. The continuation stops at u300 for a new full-grid review. It does not promote either arm and does not authorize u1000 or u2000 training automatically.

### Stage A u300 review: response fit improved on train; promotion guards still fail

The bounded continuation resumed the exact recipe-2 u100 optimizer/RNG/sampler checkpoints and stopped both arms at absolute update 300: 200 successful updates per arm, no update 301, and no timeout. The source remained Run1508 selected e496, SHA-256 `57f6762091c861e9eea2d703c9ebc70470603c3c0988a7be406cc769b8af74c0`. The selected `B_value` u300 checkpoint is `response_control_B_value_training_checkpoint_u00300.pt` (SHA-256 `ce470895b0e8039a03254797eb9d054e5242abc3feed09af5512d2c5fe6695d1`); `B_response` is `response_control_B_response_training_checkpoint_u00300.pt` (SHA-256 `da3f850f524c1f9b5e446548e7c960b7e5866e8e98a64215d3661af2c171c41`). Both contain 300 total optimizer updates, optimizer state at step 300, and preserved arm-specific RNG/sampler state. The ignored run directory is `diagnostics/generated/response_control_runs/Run1508_e496_stageA_recipe2_near_guard_3072_20260926/paired_u300_resume_attempt01/`.

The paired process ran on physical GPU 2 (UUID `f6a4ddbb-ad44-5ef5-0421-eecf7120df39`) and passed its manifest: 1,583.69 s total (1,546.29 s fit plus 34.69 s four-family development replay), with arm walls of 755.34 s (`B_value`) and 790.25 s (`B_response`). The paired-fit peak was 4.41 GB allocated / 4.54 GB reserved; development replay peaked at 34.27 GB / 36.78 GB. A separate read-only full-grid replay of both u300 checkpoints on the three train families passed in 16.09 s, used zero optimizer updates and zero new physical solves, and peaked at 86.2 MB allocated / 113.2 MB reserved per arm. The measured response-control GPU 2 ledger through this review is 3,262.30 s (0.906 h), including 1,662.51 s recorded before this continuation. The replay and fit manifests, per-step curve, and full-grid metric summaries remain beside the checkpoints.

The curve confirms the paired schedule was executed. Updates 101–300 in `B_value` have only `value` active. `B_response` updates 101–300 have `value`, `finite`, `decision`, and continuous `constraint` terms active. All 25 rows for updates 101–125 contain finite, decision, and constraint losses in the response arm, while the corresponding value-arm rows contain only the value loss. The constraint multiplier stays at `0.0025616145519238133`; feasibility BCE is omitted on the all-feasible training labels. Mixed-response loss remains absent because its train-family noise floor is unknown. B_response's mean sampled finite loss decreases from 0.218 over updates 101–125 to 0.154 over updates 276–300, while the guard-level full-grid results below remain inadequate.

The table gives the median finite-response signal-relative RMSE per family and arm; lower is better. These are measured response errors, and `q_normal` remains a flux proxy.

| Split / family | `T_surface` B_value → B_response | `q_normal` proxy B_value → B_response | Solid temperature B_value → B_response |
|---|---:|---:|---:|
| Train 0001 / M3 | 0.991 → 0.607 | 0.944 → 0.697 | 1.000 → 0.522 |
| Train 0304 / M3 | 0.806 → 0.422 | 0.969 → 0.704 | 0.838 → 0.425 |
| Train 0333 / M7 | 0.649 → 0.228 | 0.889 → 0.699 | 0.674 → 0.224 |
| Calibration 0310 / M3 | 0.635 → 0.808 | 0.943 → 0.989 | 0.713 → 0.791 |
| Calibration 0325 / M5 | 0.682 → 0.684 | 1.001 → 1.017 | 0.701 → 0.684 |
| Calibration 0340 / M7 | 0.439 → 0.454 | 0.672 → 0.700 | 0.447 → 0.466 |
| Calibration 0355 / M10 | 0.593 → 0.641 | 1.023 → 1.026 | 0.599 → 0.645 |

Training finite errors improve substantially, but `T_surface`, `q_normal`, and solid-temperature errors still exceed the 0.10 response target. Development has no comparable `T_surface` or flux-proxy improvement, and solid-temperature response improves only slightly on 0325. Exact-null pressure/velocity responses still have no measured resolution floor; mixed responses remain unknown. Absolute full-fluid and receiver errors are mixed: train fluid-u error increases by median +57%, +47%, and +35% over `B_value` for 0001/0304/0333; train absolute `T_surface` increases by +11%, +31%, and +17%. On development, absolute `T_surface` worsens +38.6% on 0310 but improves on the other three families, while absolute `q_normal` worsens on 0310 (+26.6%) and 0355 (+15.5%).

The protected near-interface metric uses the full-grid band `0 <= min(surface distance) <= 0.25` and all 11 absolute stencil states per family. Table cells are median paired changes in signal-relative RMSE for `B_response` versus `B_value` (positive is worse). The tail columns count per-state channel entries over the 5% allowance among 55 entries, then give the worst response/control ratio and its label/channel.

| Split / family | `u` | `v` | `p` | `omega` | Temperature | Entries >5% | Worst ratio / label / channel |
|---|---:|---:|---:|---:|---:|---:|---|
| Train 0001 / M3 | +30.7% | +3.7% | +9.5% | +10.8% | +23.1% | 43/55 | 1.405 / `mm` / `u` |
| Train 0304 / M3 | +35.5% | −9.4% | −7.7% | −0.7% | +18.6% | 31/55 | 1.660 / `mm` / `u` |
| Train 0333 / M7 | +20.8% | +9.6% | +10.8% | +5.0% | +11.8% | 45/55 | 1.230 / `pp` / `u` |
| Calibration 0310 / M3 | +12.4% | +15.3% | +12.7% | +27.9% | +46.2% | 55/55 | 1.705 / `pp` / `omega` |
| Calibration 0325 / M5 | +20.6% | +5.4% | −5.3% | +16.4% | −12.5% | 27/55 | 1.303 / `mp` / `omega` |
| Calibration 0340 / M7 | +37.4% | −2.5% | +11.2% | −5.2% | −3.7% | 22/55 | 1.414 / `heat_transfer_plus` / `u` |
| Calibration 0355 / M10 | +30.3% | +5.1% | +22.3% | −5.1% | −8.0% | 29/55 | 1.313 / `mm` / `u` |

This guard fails on every family, including 0310 where all 55 channel/state entries degrade by more than 5%. The sampler already included the complete train near-interface union, so this failure is not explained by omitted near-interface rows.

Pressure uses each family's `1.05 × original-start pressure drop` as a study benchmark; it is not an engineering limit. The table reports predicted-infeasible counts and median absolute/finite pressure error relative to 5% of the baseline reference pressure. False-feasible counts are zero throughout, but all 33 train references and all 44 development references are feasible, so that result is vacuous.

| Split / family | Reference infeasible | Predicted infeasible B_value / B_response | Median absolute error / margin B_value / B_response | Median finite error / margin B_value / B_response |
|---|---:|---:|---:|---:|
| Train 0001 / M3 | 0/11 | 0 / 0 | 0.674 / 1.316 | 0.086 / 0.005 |
| Train 0304 / M3 | 0/11 | 0 / 0 | 0.075 / 0.558 | 0.124 / 0.019 |
| Train 0333 / M7 | 0/11 | 0 / 0 | 0.043 / 0.324 | 0.053 / 0.008 |
| Calibration 0310 / M3 | 0/11 | 11 / 11 | 3.317 / 2.177 | 0.115 / 0.102 |
| Calibration 0325 / M5 | 0/11 | 11 / 11 | 1.817 / 1.394 | 0.036 / 0.023 |
| Calibration 0340 / M7 | 0/11 | 0 / 0 | 0.035 / 0.202 | 0.021 / 0.015 |
| Calibration 0355 / M10 | 0/11 | 0 / 0 | 2.446 / 1.475 | 0.104 / 0.121 |

Thus each arm predicts 22/44 development states infeasible and 22 feasible, with 22 false-infeasible decisions and no false-feasible decisions. The training panel has no pressure-feasibility class contrast at all; development also has no infeasible reference states. Continuous finite pressure supervision improves in three training families and three development families, but this cannot validate a feasibility classifier.

**Decision:** u300 does not pass the forward promotion gates, and Stage B promotion, Stage C cover training, and the inverse comparison remain no-go. Train finite response errors improve while development finite errors, absolute roles, near-interface errors, and pressure behavior remain inconsistent; near-interface tails violate the 5% guard by up to 70.5%, and the pressure panel cannot establish feasibility calibration. Do not continue these checkpoints to u1000 or u2000. If a third Stage A recipe is selected after review, the next bounded test should start from fresh e496 and add an explicitly scored near-interface absolute-value term to both matched arms while keeping the response contrast, sample panel, and numeric scales fixed. This tests the remaining measured near-interface regression with one objective amendment; pressure BCE stays off until train evidence includes both feasibility classes. Stage B stays disabled unless every protected family/channel guard passes and pressure feasibility is evaluated against a boundary-contrast panel.

## WindFarm native transfer and decision evidence

The native 3-D implementation uses the existing velocity wrapper, categorical wind directions, row-specific support, and 512 geometry-only environmental tokens. The dense `B_inc` is the manifest-selected Run 2103 e2475, after a matched 90-row validation replay showed lower physical volume, hub-height-slab, and downstream RMSE than the documented Run 2101 e495 on every row. Run 2101's top-level e580 alias is a distinct historical continuation whose selection metadata conflict with its e495 manifest; Run 2104's interrupted state also remains unpromoted. The previously opened reserved-test outputs cannot serve as fresh held evidence. See the [WindFarm baseline decision](../../../Case_WindFarm/docs/baseline_decision.md) for identities, native-data lineage, and per-row ledgers.

`W1` explicitly refit a clean three-term full-access model from e2475: 74 compatible encoder/fine-backend tensors transferred, 73 dense common-head tensors discarded, and 12 new common-head tensors initialized. The managed Run 2105 completed e300/1,200 cumulative optimizer updates on physical GPU 2, with optimizer and RNG continuity checked at e100; validation selected e295/update 1,180. On the matched Q=40,960/E512/90-row native validation panel, mean physical RMSE in m/s was:

| Native region | Dense e2475 | Three-term e295 | e295 rows better |
|---|---:|---:|---:|
| Volume | 0.010773 | 0.043472 | 0/90 |
| Hub-height slab | 0.019497 | 0.073290 | 0/90 |
| Downstream envelope | 0.022652 | 0.088763 | 0/90 |

Run 2105's within-run validation MSE improved from 0.0398403 at e100 to selected 0.01530734 at e295, but the physical-unit paired gate rejects this refit as a replacement for dense `B_inc`. Its e295 evaluation took 20.92 s external wall, including 12.301 s synchronized prediction and 16.762 s case-loop wall, with 348/468 MiB peak allocated/reserved CUDA memory. These are measured W1 costs, without a matched dense timing claim. No further WindFarm refit follows from this no-go.

`W2` used the frozen dense e2475 model for a finite stored-library choice, not a continuous inverse or adaptive-graph test. A train-only geometry ridge had validation `wake_loss_pct` RMSE 1.9200 percentage points; adding predicted rotor-neighborhood velocity features reduced it to 1.3053 points. Across 24 frozen validation cohorts, the feature control selected mean stored wake loss 15.8445% versus 15.8777% for geometry alone, changed only 2 of 24 selections, and had a paired mean difference of −0.033254 percentage points. The metric is an empirical stored scalar with undocumented formula; the 17 receiver query points per turbine do not have exact native reference samples. No power, AEP, or physical design-improvement conclusion follows. `W3` is held because no independent new-layout CFD source, mesh/solver log, or wake-loss formula is available. Existing native velocity checks and W2 stored labels cannot substitute for that reference.

## Local reference interactions and numerical resolution

The bounded local NumPy reference work is documented in [ThermalChannel physical interaction evidence](20260926_195912Z_HONF_Thermal_Physical_Interaction_Evidence.md), with the signed input manifest and raw outputs retained in ignored `diagnostics/generated/` paths. These are local analytic-wake/shared-grid reference solves, not external CFD or surrogate-teacher responses. The 84 new attempts all converged: 14 M7/0340 mesh/tolerance checks, 9 M3/0001 fine checks, 52 M3/0001 coarse coverage states, and 9 M3/0304 fine checks. Measured solver-process CPU is 450.691 s; two completed calls with missing per-call times each retain a separate conservative 600 s reserve. The frozen 0304 coarse and M7/0333 expansions remain unspent because prerequisite numerical-resolution and model-fit gates have not passed.

At the tested h=0.15 M3/0001 pair, material-coordinate mixed `T_surface` and solid-temperature responses on moved modules 0 and 1 exceed the family-specific coarse-to-fine discrepancy criterion. The hot spectator module 2 controls the sampled true peak and has a mixed response below the family-wide discrepancy, while the pressure mixed change is near zero. This is evidence of a local effective thermal interaction but not of a useful peak-temperature design factor, a learned edge, or microscopic causality. M3/0304's temperature mixed signals are of the same order as its mesh discrepancy and fail the predeclared 3× rule; no coarse expansion was run. In M7/0340, mesh doubling reverses signs of two finite peak changes despite a stable anchored mixed peak scalar. Those distinct outcomes prevent exporting the 0001 support or numerical floor to another family.

No engineering pressure allowable is recorded for these panels. The per-family `1.05 × original baseline` threshold used by the response-control diagnosis is a fixed study benchmark. Its existing train/development stencil labels are all feasible, so they cannot calibrate false-feasible behavior near a decision boundary. Physical reference outcomes, stored WindFarm scalar labels, frozen-surrogate teachers, and synthetic cover tests are recorded as separate evidence sources throughout this study.

| Evidence source | Used here for | Does not establish |
|---|---|---|
| New local ThermalChannel reference solves | Mesh/tolerance sensitivity and measured material/pressure changes | External CFD validity or a learned hyperedge |
| Existing typed ThermalChannel atlas | Matched response training and replay | New held-out response generalization |
| Stored WindFarm CFD exports and `wake_loss_pct` | Native velocity validation and finite-library choice | Continuous new-layout design or turbine power/AEP |
| Frozen forward surrogates and response-factor teachers | Baseline/fit controls and model-side factors | Independent physical response truth |
| Synthetic cover fixtures | Algebra, provenance, row-accounting, and protocol tests | Native adaptive-K behavior or speedup |

## Adaptive cover and inverse protocol status

The registered `three_term_full_access_honf` mode exposes the clean common MM/ME/EM preparation and fine QM/QE reads for an explicit response refit. The separate opt-in `adaptive_interaction_cover_honf` prototype builds a case-local receiver tree from physical input anchors, uses typed module/environment membership and continuous split access, and compiles a deduplicated union of actual fine query-source rows. Its ledger distinguishes active groups, raw paths, unique versus rectangular rows, fallback environmental reads, and unchanged dense preparation work. Full-access mode calls the dense fine reader directly; external plans remain offline diagnostics. Synthetic endpoint, overlap, permutation, empty-environment, provenance, and physical-wrapper tests establish software behavior only. There is no fitted input-only organizer, demonstrated native case-dependent K, observed native support switch, or measured end-to-end cover speedup.

The ThermalChannel inverse protocol accepts a baseline observation and only the sum of physically indexed anchored response factors for each candidate. Trial geometry is used to check clearance and fixed pressure-section occupancy; the protocol has no trial-design forward-surrogate method. Tests poison a dense trial predictor and verify that it is never called. Matched graph-guided, size-matched random, and ungrouped policies share the unary/pair order schedule, candidate dimension, trust radius, reference-call limits, and selected-policy acceptance rules; best evaluated feasible outcomes remain separate from selected-policy outcomes. This is a tested experiment harness, not a physical inverse result. The Stage B near-interface and pressure gates have not passed, so Stage C oracle training, Stage D amortization, and the matched new-reference inverse study are held. WindFarm W2 is a separate finite stored-library association, not a graph-guided continuous trial.

## Final promotion decisions

| Component | Decision | Measured reason or missing gate |
|---|---|---|
| Thermal full-access forward model | Hold `B_inc` as the provisional comparison; do not promote the response refit | Train finite responses improve at u300, but development response targets, protected near-interface errors, and pressure feasibility fail. The e496-to-three-term transfer is an explicit refit, not an equivalent checkpoint conversion. |
| Adaptive organizer | Hold | The input-anchored cover and bounded oracle search are implemented, but no physically adequate full-access response checkpoint exists to supply train-only cover targets. No organizer has been fitted or shown to vary exact active count natively. |
| Sparse executor | Hold; retain dense full-access fallback | Synthetic row accounting and deduplicated packed execution are checked, but no native target-shape cover has passed the accuracy gate or shown a complete preparation-plus-read wall-time benefit. |
| Thermal inverse policy | Hold | The matched factor-only protocol passes software checks, but no new matched reference design trial is justified with the current pressure and receiver errors. Selected-policy outcomes and best evaluated opportunities therefore remain unmeasured for this round. |
| WindFarm transfer | Reject the W1 refit as a dense replacement; retain W2 as a finite-library diagnostic | W1 loses to dense on all 90 matched native rows. W2 changes 2 of 24 stored-cohort selections with a small paired wake-loss difference; new-layout CFD and the stored scalar's exact physical definition are unavailable. |

The identified limit is inadequate full-access response and constraint prediction, compounded by family-specific numerical resolution and a feasibility panel containing no infeasible reference states. This round cannot determine whether an adequate adaptive cover, its amortization, its native support-switch behavior, or its design utility exists. The next bounded experiment, if undertaken, is the fresh-e496 paired Stage A near-interface absolute-value remedy specified in the u300 review. A boundary-contrast pressure panel would still be required before any feasibility claim. This experiment was not launched in this round.

## Resource accounting and retained artifacts

The measured Thermal response-control GPU 2 ledger, including its M0 replay and bounded preflights, is 3,262.30 s (0.906 h). WindFarm has about 499.9 s (0.139 h) of recorded or explicitly estimated GPU 2 job wall: Run 2105 training contributes about 387.8 s, its e100/e295 evaluation jobs 104.09 s, and W2 feature extraction 8.00 s. The e100–e300 training segment is estimated at about 247.9 s from a 3:58 process snapshot near e294 plus the last six epochs' recorded 9.86 s; it was not independently time-wrapped. These disjoint accounted segments sum to about 3,762.2 s (1.045 h). The two WindFarm dense baseline replay job walls, its strict warm-start preflight wall, and brief focused tests have no retained job-wall measurements, so this is a partial total, not a claim of exact aggregate GPU use. Synchronized model times nested within a job wall are not added again. No new WindFarm CFD solve was run.

ThermalChannel made 84 new local-generator reference attempts, all converged, with 450.691 s measured solver-process CPU. Two completed calls have missing individual timing and retain separate 600 s reserves rather than fabricated point estimates. The signed manifest froze 143 possible rows including the 14-call M7 pilot, leaving 59 unspent; the 84 attempted calls are below the 512-attempt and six-measured-CPU-hour ceilings. Existing atlas, surrogate, and stored CFD replays are not charged as new reference solves. Checkpoints, per-step curves, physical raw outputs, and generated metrics remain in ignored run and `diagnostics/generated/` locations; only reusable code, configurations, tests, and evidence reports are uploaded.
