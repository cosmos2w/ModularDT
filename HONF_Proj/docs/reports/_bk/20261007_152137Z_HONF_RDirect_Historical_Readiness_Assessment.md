# Historical R-Direct readiness assessment — superseded on 2026-10-07

This is the preserved pre-launch assessment. Its DEV22/e2500 evidence and no-launch status describe preparation before the subsequently authorized Run3901/Run3902 full-TRAIN fits, which are complete. Use the [current four-model comparison](20261007_031743Z_HONF_RDirect_Formal_Readiness_and_Receiver_Local_Organization_Report.md) for completed e5000 results. Historical scientific evidence remains valid within its original protocol.

The following assessment retains the earlier bounded development evidence and figures. Its checkpoint ages, DEV22 metrics and no-launch decision belong to that earlier stage; they are not the completed formal comparison above.


**Model decision: use R-direct as the preferred response-family reference, with ordinary D-sep as its flow recipe. Launch decision: the fresh full-TRAIN flow5000 → fixed-flow thermal5000 research recipe is implemented and available for a manual launch; no 5,000-epoch fit was launched.** This round adds precise prepared increments, replays the historical Dense1804/HONF1502 models in their native paths, completes only the matched +500 D-sep refinement pair, and separates full-TRAIN identities from the unchanged fixed25_v1 development workflow. R-group, Run3801 and all historical checkpoints remain preserved.

**Predictor.** Precise fixed-kernel arithmetic passes all 288 endpoint/increment checks while retaining 250 failing legacy FP32 checks. R-direct lowers primary fluid heat-response RMSE from H-add’s 0.037067 to 0.011587 and surface-temperature RMSE from 0.7883 to 0.7123. The historical classics are substantially better absolute predictors on this DEV22 replay; R-direct’s module-peak mean RMSE is 0.7826 versus H-add’s 0.6477 (+20.8%). Its material-peak, near-vorticity and geometry-response misses remain visible. The near-boundary flow refinement was not selected; the ordinary continuation was better.

**Organizer.** All 64 receiver-patch matrices were measured with TRAIN-box omission bounds and equal-K nearest/upstream controls. Mean retained donor count is 5.75/5.58 at the 1%/2% budgets, with many patches retaining every donor. This is a bounded explanatory view. Adaptive global grouping, causal donor truth and sparse executor savings remain unproved.

**Inverse reuse.** Fixed-layout heat changes and native maxima can be replayed with stable arithmetic. Geometry remains inaccurate, and the existing solved-pool choices were already available to older controls. No inverse generator, continuous search or new design validation was performed. These missing capabilities do not block the research recipe.

**Price and execution evidence.** Three real full-TRAIN600 epochs ran for each fresh component on GPU 2, with 600 visits, 13 updates and 614,400 primary fluid queries per epoch. Mean training times were 0.715 s for flow and 3.653 s for thermal. The conditional training-only forecast is 0.99 + 5.07 = **6.07 GPU-associated hours** on one GPU, before future canonical89/original90 monitoring and larger checkpoint saves. Thermal stopped at e2 and continued from saved optimizer state through e3. Flow ran e1–e3 uninterrupted and has a zero-update endpoint restore check; continuing flow resume was not measured because its three-epoch allowance was already consumed. The guide records that limitation explicitly rather than adding unauthorized epochs. See the [tested manual guide](../../guides/Thermal_RDirect_Formal5000.md), [flow profile](../../../src/config_core/forward/thermal_source_response/d-sep_full5000.json) and [thermal profile](../../../src/config_core/forward/thermal_source_response/r-direct_full5000.json).

### Five main result figures

The short figure index is: **1**, absolute temperature and residuals on fixed representative 0291 and high-M 0687; **2**, receiver-local source strength and signed heat response; **3**, grouping/work comparison and bounded local covers; **4**, saved heat/design changes and nonlinear maxima; **5**, measured work and manual launch price. PDF is the retained master; the small PNG companions enable direct Markdown display. All retained pages were visually inspected, use at most six panels, and label native dataset units rather than inventing SI units. The references are the existing analytic-wake/shared-grid generator and its saved alternatives, not independent CFD.

#### 1. What field does each model predict?

![Stored temperature, direct response, Classic Dense and signed residuals on DEV0291](../../../diagnostics/generated/rdirect_readiness_20261007/figures/01_fields_0291.png)

**Figure 1a.** The same native fluid coordinates/masks and shared temperature/residual scales show the fixed representative 0291/M5; numbered disks identify modules and the star marks the measured material-temperature maximum. Across all 22 cases, fluid-temperature RMSE is 0.7746 for R-direct, 0.1979 for Classic Dense and 0.2467 for Classic HONF1502. The classic advantage is measured, but training population, objective and component ages differ, so this is not an isolated architecture effect. [PDF](../../../diagnostics/generated/rdirect_readiness_20261007/figures/01_fields_0291.pdf).

![Stored temperature, direct response, Classic Dense and signed residuals on high-M DEV0687](../../../diagnostics/generated/rdirect_readiness_20261007/figures/01_fields_0687.png)

**Figure 1b.** The predeclared high-M representative 0687/M10 retains its own shared physical and residual scales. The direct response reproduces the broad thermal wake but leaves structured errors; the classic Dense residual is smaller. The marked material maximum does not turn a fluid map into a peak-temperature certification. [PDF](../../../diagnostics/generated/rdirect_readiness_20261007/figures/01_fields_0687.pdf).

| Physical row | H-add e500 | Direct e2500 | Grouped e2400 | Dense e4738 | HONF1502 e4794 |
|---|---|---|---|---|---|
| Fluid u | 0.022037 / 0.024914 | 0.017068 / 0.020804 | 0.017068 / 0.020804 | 0.0057149 / 0.0070266 | 0.0057348 / 0.0080422 |
| Fluid v | 0.0030797 / 0.0039929 | 0.0017125 / 0.0022202 | 0.0017125 / 0.0022202 | 0.00031904 / 0.00041079 | 0.0004142 / 0.00054682 |
| Fluid p | 0.012222 / 0.015606 | 0.0090926 / 0.011129 | 0.0090926 / 0.011129 | 0.0021 / 0.0027101 | 0.0024947 / 0.0030259 |
| Fluid omega | 0.10457 / 0.13261 | 0.13146 / 0.16482 | 0.13146 / 0.16482 | 0.023461 / 0.045707 | 0.022921 / 0.048003 |
| Fluid temperature | 0.77031 / 1.1015 | 0.77463 / 1.0177 | 0.84435 / 1.1176 | 0.19794 / 0.24768 | 0.24672 / 0.35577 |
| Surface temperature | 0.78833 / 1.1782 | 0.71231 / 1.1134 | 0.71674 / 1.1906 | 0.40316 / 0.49392 | 0.43369 / 0.5295 |
| Material temperature | 0.66308 / 1.1367 | 0.67773 / 1.0164 | 0.65899 / 0.97331 | 0.30043 / 0.34331 | 0.33453 / 0.46917 |
| Module material peak | 0.64773 / 1.213 | 0.78262 / 1.2034 | 0.74378 / 1.1139 | 0.28529 / 0.40907 | 0.34595 / 0.54235 |

Each cell in the table is equal-case mean / case p90 physical RMSE on the exact fixed25_v1 DEV22 IDs. Fluid rows use the identical fluid masks, surface/material rows use valid native receivers, and the module-peak row first computes each module's maximum over saved material queries, then per-case RMSE across modules. New-family initial-port history is NA; it is not replaced with final-port values. The classic paths use their own checkpoint-native TRAIN input/target transforms and their full predicted-port/local-surrogate evaluation, then denormalize before comparison. These metrics do not mix checkpoint-specific normalized errors.

#### 2. Which sources matter at this receiver?

![Input-chosen receiver patches, ranked learned donor strengths, signed contributions and saved heat-response fields](../../../diagnostics/generated/rdirect_readiness_20261007/figures/02_receiver_sources.png)

**Figure 2.** Three patches were selected from input geometry and channel bounds: upstream, near module 0 and downstream overlap. Learned source strength is ranked within each patch, with physical source IDs at the points; signed current contributions retain those same IDs. The lower row compares the stored 0291 plus transfer, direct kernel response and signed residual. Stored mean fluid changes are −0.0289189/+0.0289138 for minus/plus, versus direct −0.0369188/+0.0369188: the direction is right but amplitude is about 27.7% too large. Environment dots are context only. One excitation direction per layout does not independently validate each donor column or establish causality. [PDF](../../../diagnostics/generated/rdirect_readiness_20261007/figures/02_receiver_sources.pdf).

#### 3. Is grouping buying anything?

![Direct and grouped source counts and work, bounded receiver-local donor counts and equal-K controls](../../../diagnostics/generated/rdirect_readiness_20261007/figures/03_grouping.png)

**Figure 3.** At M10, R-group has 11 valid modes and all 110 memberships are positive; M+1 is not learned adaptive K. Complete cold M12/Q8192 prediction is 25.450 ms for direct and 31.604 ms for grouped. The primary six signed heat responses give fluid-temperature RMSE 0.011587/0.013830. The new local view covers 16 fluid patches per case and 1%/2% of sealed TRAIN response RMS: mean retained K is 5.75/5.58, range 3–10. At matched K, mean covered physical response RMSE is 0.011678/0.011653 for learned strength, 0.012768/0.012768 for nearest geometry and 0.011757/0.011861 for upstream geometry. These descriptive four-state patch averages do not establish general superiority. The final panel separates actual model omission from its triangle bound. No sparse executor was run. [PDF](../../../diagnostics/generated/rdirect_readiness_20261007/figures/03_grouping.pdf).

#### 4. What happens when the design changes?

![Saved signed heat transfers, actual material maxima and a difficult saved geometry response](../../../diagnostics/generated/rdirect_readiness_20261007/figures/04_design_changes.png)

**Figure 4.** The fixed 0291 geometry uses the explicitly displayed minus/base/plus heating values; native material maxima are evaluated from endpoints rather than applying a linear increment to a maximum. The lower row uses the saved original-TRAIN 0348/M10 `i_plus` obstacle move on the common fluid mask. Its geometry-response RMSE is 0.19245 for direct and 0.18775 for grouped; the reference mean change is +0.087545 while direct predicts −0.003275. Geometry is therefore unqualified. Case 0277 has no fresh counted baseline and remains a secondary minus-to-plus span; its packed nominal case is never substituted into that span. Replaying stored alternatives is not new inverse-design evidence. [PDF](../../../diagnostics/generated/rdirect_readiness_20261007/figures/04_design_changes.pdf).

#### 5. How much work, and what can be launched?

![Complete cold and prepared reuse timings, precise arithmetic price, real full-TRAIN epoch timings and manual forecast](../../../diagnostics/generated/rdirect_readiness_20261007/figures/05_work_and_launch.png)

**Figure 5.** Existing complete cold M12/Q8192 timings are 120.973/25.450/31.604 ms for H-add/direct/grouped. Three-heat direct reuse costs 38.188 ms including preparation, versus 77.155 ms for three cold calls; grouped costs 43.865 versus 97.068 ms. In the separate new typed-record M10 precision audit, ordinary prepared/precise endpoint/precise increment calls cost 6.525/6.874/1.958 ms, with 43.783 ms preparation and 3,285,760 transient FP64 kernel bytes. The startup plot measures all scheduled objectives in genuine full-TRAIN epochs. Training-only flow5000/thermal5000 forecasts are 0.99/5.07 h; full-panel monitoring and later save overhead were not measured. These scopes do not justify adding overlapping inference timers or promising a six-hour end-to-end fit. [PDF](../../../diagnostics/generated/rdirect_readiness_20261007/figures/05_work_and_launch.pdf).

### What the measured increments changed

The neural weights and ordinary FP32 path are unchanged. The opt-in `accumulation_dtype=torch.float64` path converts raw far/near factors before kernel construction, contracts physical heating in FP64, interpolates native temperatures in FP64 and recomputes the native harmonic-conductivity/delta proxy in FP64. It contracts Δh directly, before adding offsets or subtracting large endpoints. It does not merely cast a rounded final FP32 kernel, retrain in FP64 or change near/far semantics. Geometry remains bound to the prepared context and is rejected if stale.

The actual GPU2 audit covers four existing TRAIN response families (0001, 0318, 0333, 0348) and all four fixed representatives (0277, 0291, 0294, 0687), balanced and nonbalanced signed changes at 0.25, 0.001 and 1e-6. At the retained rtol 2e-5/atol 2e-6, all 288 precise endpoint/increment role checks pass; 250/288 legacy FP32 endpoint checks still fail and remain recorded separately. Maximum precise endpoint discrepancy is 1.51e-14 and native physical-kernel VJP discrepancy is 8.88e-16. Cold/prepared precise outputs are equal. The receipt also records input-heat quantization separately, so coefficient preservation is not confused with accuracy against physical truth. The CPU replay independently passed, with CUDA hidden.

Linear increments provide fluid/interface/material temperatures, q proxy and outside temperature. Effective h ratios and module maxima remain nonlinear endpoint functionals: evaluate both endpoints with the precise path when those quantities are needed. Increment results deliberately omit `pred_port_condition` and mark those functionals NA. Flow has zero heat increment under the audited prescribed-flow capability. Kernel factors remain learned estimates: passing arithmetic checks does not validate geometry, donor causality or continuum heat flux. See [precision summary](../../../diagnostics/generated/rdirect_readiness_20261007/precision_gpu2/summary.json) and the [maintained inference guide](../../guides/Thermal_Source_Response.md).

### The only refinement pair: ordinary versus near consistency

The pair starts from the same D-sep2500 flow tensors and AdamW state. The retained epoch_2500 and latest aliases have different ZIP filenames but identical loaded state and all 156 uncompressed members; that equivalence is recorded before selection. Both children visit the same 150 TRAIN cases for 500 additional epochs, with 2,000 updates and 75,000 case visits per arm. Their schedule is fixed at warmup 1e-6→3e-5 over 20 epochs, hold through 200, then cosine decay to 3e-6 at +500. The refinement adds 128 near-fluid centers per case and their generator-matched stencil; it does not enable heat access or change the four flow channels.

The native generator definition is omega = dv/dx − du/dy with dx=Lx/nx, dy=Ly/ny, first-order boundary differences and solid masking. It reproduced saved TRAIN omega with RMSE 1.74e-7/1.36e-7 on low/high-M qualification cases. Near receivers mean fluid centers within two radii of a module center, or surface distance at most one radius; far is the remaining fluid mask. The additional stencil visits 96,000 query rows per epoch; the ordinary primary sampler visits 153,600. These qualification and work measurements are not new physical solves.

The ordinary e3000 control improves the parent all-fluid mean u/v/p/omega RMSE by 2.46%/4.86%/4.32%/1.68%; near omega improves 1.48%. The near-consistency child is 12.68%/13.34% worse than control in mean/p90 fluid u, 13.05%/13.25% worse in near u, 0.17%/0.27% worse in near omega and 6.89%/5.98% worse in far omega. The final flow recipe therefore keeps the ordinary objective. No additional remedy, D-open run or thermal fit was started. Both histories and the failed CUDA lazy-initialization attempt are retained.

![Native u and omega reference, ordinary flow e3000 and signed residuals on DEV0291](../../../diagnostics/generated/rdirect_readiness_20261007/refinement_pair/flow_control3000_u_omega_0291.png)

**Flow companion.** The ordinary control retains the large-scale flow but leaves near-wall omega errors on exposed 0291/M5. The same reference/prediction scale and symmetric residual scale are used per channel. This fresh continuation is not silently inserted into the selected R-direct2500 composition used in Figures 1–4, whose flow remains e2500. [PDF](../../../diagnostics/generated/rdirect_readiness_20261007/refinement_pair/flow_control3000_u_omega_0291.pdf). The separately inspected [high-M companion](../../../diagnostics/generated/rdirect_readiness_20261007/refinement_pair/flow_control3000_u_omega_0687.pdf) and [closeout/alias receipt](../../../diagnostics/generated/rdirect_readiness_20261007/refinement_pair/workstream_c_closeout_cpu.json) preserve detailed evidence.

| DEV22 physical flow row | Parent e2500 | Ordinary e3000 | Near consistency e3000 |
|---|---|---|---|
| fluid u | 0.017068 / 0.020804 | 0.016648 / 0.02003 | 0.018759 / 0.022702 |
| fluid v | 0.0017125 / 0.0022202 | 0.0016293 / 0.0020779 | 0.0016924 / 0.0021622 |
| fluid p | 0.0090926 / 0.011129 | 0.0086996 / 0.010446 | 0.0088306 / 0.010644 |
| fluid omega | 0.13146 / 0.16482 | 0.12926 / 0.16177 | 0.1304 / 0.16218 |
| near u | 0.034137 / 0.036876 | 0.033693 / 0.036274 | 0.03809 / 0.041081 |
| near v | 0.0029301 / 0.0035319 | 0.0027584 / 0.0032862 | 0.0029117 / 0.0034778 |
| near p | 0.013814 / 0.014708 | 0.013578 / 0.014576 | 0.013726 / 0.0142 |
| near omega | 0.32075 / 0.34269 | 0.31599 / 0.33643 | 0.31654 / 0.33733 |
| far u | 0.01173 / 0.01486 | 0.011227 / 0.013929 | 0.01257 / 0.015542 |
| far v | 0.0013823 / 0.0018378 | 0.0013294 / 0.001729 | 0.0013619 / 0.0017624 |
| far p | 0.0078157 / 0.010539 | 0.0073872 / 0.0097244 | 0.0075161 / 0.009895 |
| far omega | 0.047188 / 0.05551 | 0.045657 / 0.053453 | 0.048804 / 0.056651 |

### Receiver-local bounds, controls and physical limits

For each native 4×4 equal-area fluid patch P, normalized fluid-only quadrature W_P defines q_Pi = r_i ||W_P^(1/2) K_i||_2. Physical source IDs/centers are retained. The input-only balanced radius is r_i = max(0, min(h_i−h_min, h_max−h_i, 0.20 mean active h)); h_min=0.5036706 and h_max=1.9991537 are rederived from the exact 150 selected TRAIN inputs, with four TRAIN addendum source hashes verified. Budget is 0.01/0.02 × the retained TRAIN-only temperature-response RMS 0.4408216, or 0.0044082/0.0088164 native temperature units. Counted/DEV response errors do not set radii or budgets.

The learned cover sorts q and retains the smallest prefix whose omitted-score sum is within budget. Nearest and upstream geometry controls retain exactly the same K. All 112 saved response/patch rows per budget lie inside their per-source radii and satisfy the triangle inequality for model omission. This bound limits distortion of the learned model, not error against the physical reference. Covered/full physical errors are reported separately. Case 0277/0294 retains every source, 0291 averages about 4.2 of 5, and 0687 about 8.1–8.8 of 10. FP64 kernel re-export differs from the retained FP32 operator by at most 4.95e-6; its stored cold-response replay differs by at most 5.59e-6. Those ordinary-rounding differences remain explicit.

This tool is post-fit and explanatory: the authoritative forward path remains source-resolved and full-access. It measures cover size, omitted bound, measured omission and reference error, but not a sparse executor or saved forward work. No patch-SVD/training search was added. Full matrices, source contributions and every equal-K control are in the [local-view summary](../../../diagnostics/generated/rdirect_readiness_20261007/local_view/summary.json) and its four checkpoint-bound kernel arrays.

### Formal software, data boundaries and launch status

The two maintained formal profiles resolve all 600 original TRAIN IDs from packed-H5 metadata and fit one global normalizer on those cases alone. Flow and thermal carry identical memberships, transforms and canonical89/original90 bindings. The historical training duplicate 0273 is excluded from canonical89 and remains in the original90 compatibility panel. Startup is explicitly disposable, validates only on DEV22, cannot exceed e3 and cannot be resumed or promoted into a formal identity. The legacy fixed25_v1 150/22 loaders and same-arm resume checks remain strict. The new formal loader accepts valid monitored thermal ages with an exact e5000 formal flow partner; final endpoint reporting requires both e5000 ages explicitly.

Both stages start from fresh weights, seed 0, FP32 training, effective batch 48/microbatch 8 and Q1024 fluid sampling. Formal LR holds 3e-4 for 2,000 epochs, then cosine-decays to 3e-6 by e5000. Thermal retains the three standardized temperature roles, 0.05 q proxy, the same four saved TRAIN response anchors and qualified TRAIN-only discrete-balance objective; calibration is recomputed at fresh initialization using the full-TRAIN normalizer. Every 100 epochs, each stage saves monitoring/latest/best-field state and its declared milestone. Nonempty preparation destinations are rejected before writes; resumes require matching profile, schedule, data, normalizer, partner and source identities. Trusted checkpoint loading and atomic saves remain in use.

The actual thermal startup recorded 109,344 material-query and 54,672 surface-query rows per epoch, 76,800 qualified operator rows and 437,376 operator source-column rows, plus 17,152–17,392 response-target rows. These counters derive from actual sampled cases and native extraction; neural and stencil work is recorded separately rather than replaced with logical sparsity. Thermal's 52 Adam states progress from step 26 at the clean stop to step 39 after resume; the frozen e3 flow remains unchanged. DEV22 thermal selector improves 0.5071→0.4307 and common five-field standardized MSE 1.0009→0.9965. Physical u/v/p/omega/T RMSE at e3 is 0.3183/0.05373/0.1700/1.1327/5.2885, demonstrating startup composition rather than mature accuracy.

Measured initial flow training epochs are 0.843/0.652/0.651 s; thermal epochs are 3.799/3.402/3.758 s. Flow DEV22 validation/save/loading are 0.0107/0.4848/0.2236 s; thermal's two validation/save/loading intervals total 0.1256/0.9980/1.0030 s. Peak allocated CUDA memory is 247,451,648 bytes flow and 896,102,400 bytes thermal. Data loading and other preparation overhead are separately recorded in process receipts and are not added twice. The original three training process intervals total 41.411 s; actual training is 13.104 s. The later zero-update flow verification is separate. Full90 truth was not evaluated during startup. The 6.07 h forecast is conditional on the same GPU/schedule/contention and includes training objectives only; it does not guarantee accuracy or end-to-end runtime. Exact manual prepare/start/status/stop/resume commands and the save/resume limitation are in the [guide](../../guides/Thermal_RDirect_Formal5000.md).

### Historical identities and detailed numerical evidence

The common exposed DEV22 manifest is `933b0138ba2f8447a1ecadfe31fd0bb2cb4a05607d3ac3d9f0dc79419f196044`. Selected checkpoints are Dense1804 e4738 (`71ed480ff0396813491c650dd11d887195174019b373fbd9a1fb25505142c066`), HONF1502 e4794 (`08d86a573c7f7d86463bde231eb9f84a745602fca2b8f97142f39540d33a85bb`), R-direct thermal e2500 (`05974d2fbc367bad8f2092063818ce9073c21783131753b3648fb3ea11a870aa`) with D-sep e2500, R-group selected thermal e2400 with the same flow, and retained H-add Run3801 e500. Classical/global and development-quarter histories differ; normalization and training age are preserved rather than relabelled as matched training.

#### Same saved heat responses: historical controls

Mean state RMSE below uses the same six primary baseline-relative changes on 0291/0294/0687. Direct is better than both classics on surface/material response error, while Dense is slightly better on fluid-temperature response error. This distinction supports a reusable response reference without claiming all-role superiority. All models retain their native training history and extraction; the secondary 0277 span is separate.

| Primary physical response | Direct e2500 | Grouped e2400 | Dense e4738 | HONF1502 e4794 |
|---|---|---|---|---|
| Fluid temperature | 0.011587 | 0.013830 | 0.011020 | 0.012818 |
| Surface temperature | 0.014868 | 0.015816 | 0.017235 | 0.025743 |
| Material temperature | 0.016767 | 0.017369 | 0.017839 | 0.026818 |

| Saved layout / change | Direct e2500 | Grouped e2400 | Dense e4738 | HONF1502 e4794 |
|---|---|---|---|---|
| 0277 / transfer_plus (secondary span) | 0.022235 | 0.029147 | 0.065340 | 0.030881 |
| 0291 / transfer_minus | 0.025253 | 0.029382 | 0.018120 | 0.019319 |
| 0291 / transfer_plus | 0.025260 | 0.029391 | 0.018712 | 0.021256 |
| 0294 / transfer_minus | 0.008177 | 0.010514 | 0.013547 | 0.015087 |
| 0294 / transfer_plus | 0.008177 | 0.010514 | 0.011307 | 0.015420 |
| 0687 / transfer_minus | 0.001329 | 0.001590 | 0.002225 | 0.003093 |
| 0687 / transfer_plus | 0.001329 | 0.001590 | 0.002211 | 0.002733 |

The heat-response error of H-add is 0.037067/0.050119/0.049496 for fluid/surface/material temperature on the same primary states. Its wrong-sign mean changes on 0291 remain retained (+0.010420/−0.015508 for minus/plus). The precise increment repair changes arithmetic, not those historical physical predictions.

Literal adjacent e5000 `latest.pt` checkpoints exist and were additionally replayed read-only on DEV22. Dense e5000 SHA is `9d0b83c562cecc2ffc52c3a08c993dfa8f47ae0e769f54ed6ffdd966047d63d9`; HONF1502 e5000 SHA is `20af85200796f639039d4853425fe91ee76e19e7caf5b5b68863badfe7690895`. They are endpoint comparisons and do not replace the selected states. Their fluid T mean/p90 RMSE is 0.22349/0.29301 and 0.26577/0.38331; module-peak RMSE is 0.26767/0.41649 and 0.43650/0.58220. Native models, weights and source files remain unchanged.

| Physical row | H-add e500 | Direct e2500 | Grouped e2400 | Dense e4738 | HONF1502 e4794 |
|---|---|---|---|---|---|
| Near u | 0.030499 / 0.035402 | 0.034137 / 0.036876 | 0.034137 / 0.036876 | 0.0070298 / 0.011735 | 0.007657 / 0.013494 |
| Near v | 0.0042195 / 0.0053616 | 0.0029301 / 0.0035319 | 0.0029301 / 0.0035319 | 0.00053293 / 0.00064699 | 0.00073675 / 0.0010112 |
| Near p | 0.015464 / 0.018712 | 0.013814 / 0.014708 | 0.013814 / 0.014708 | 0.002796 / 0.0043792 | 0.0032513 / 0.004893 |
| Near omega | 0.23768 / 0.28737 | 0.32075 / 0.34269 | 0.32075 / 0.34269 | 0.052648 / 0.10597 | 0.047963 / 0.10457 |
| Near temperature | 0.77834 / 1.1666 | 0.7695 / 1.0917 | 0.80461 / 1.2566 | 0.22798 / 0.35022 | 0.26507 / 0.48669 |
| Far u | 0.020042 / 0.022867 | 0.01173 / 0.01486 | 0.01173 / 0.01486 | 0.0053391 / 0.0063835 | 0.0052205 / 0.006428 |
| Far v | 0.0027983 / 0.003557 | 0.0013823 / 0.0018378 | 0.0013823 / 0.0018378 | 0.00025768 / 0.00032377 | 0.00031425 / 0.00040713 |
| Far p | 0.011501 / 0.015156 | 0.0078157 / 0.010539 | 0.0078157 / 0.010539 | 0.0019279 / 0.0023644 | 0.0022952 / 0.0028058 |
| Far omega | 0.053536 / 0.06514 | 0.047188 / 0.05551 | 0.047188 / 0.05551 | 0.011211 / 0.014088 | 0.0118 / 0.015393 |
| Far temperature | 0.76757 / 1.1307 | 0.76836 / 0.99889 | 0.84797 / 1.1612 | 0.19208 / 0.24296 | 0.24127 / 0.32262 |
| Native q proxy | 3.352 / 4.1176 | 2.559 / 3.4789 | 2.6864 / 3.624 | 1.5561 / 1.9344 | 1.6105 / 2.0886 |
| Final outside temperature | 1.1274 / 1.5114 | 0.72857 / 1.1378 | 0.75146 / 1.2569 | 0.8159 / 0.94122 | 0.83456 / 0.94682 |
| Final effective h | 0.82872 / 1.0807 | 4.1115e-05 / 7.6852e-05 | 0.036806 / 7.6874e-05 | 0.62013 / 0.87519 | 0.59819 / 0.84696 |
| Initial outside temperature | 1.413 / 2.2311 | NA | NA | 0.77516 / 0.92808 | 0.85839 / 1.035 |
| Initial effective h | 11.871 / 11.902 | NA | NA | 2.7701 / 2.988 | 0.56746 / 0.7316 |
| Inlet–outlet pressure difference | 0.0070759 / 0.012437 | 0.0069276 / 0.01176 | 0.0069276 / 0.01176 | 0.00095241 / 0.0019682 | 0.0018654 / 0.0030195 |

#### Module-count appendix

These are the same exposed DEV22 reductions, split by actual active module count. Cells are mean / p90 physical RMSE; small stratum sizes are shown explicitly and do not support population-wide tail estimates. Other fields and H-add strata remain in the linked saved summaries.

| Active modules / cases / physical row | Direct e2500 | Grouped e2400 | Dense e4738 | HONF1502 e4794 |
|---|---|---|---|---|
| M3 / 6 / u | 0.014531 / 0.015756 | 0.014531 / 0.015756 | 0.0051136 / 0.0056195 | 0.0045919 / 0.0049996 |
| M3 / 6 / Near omega | 0.32832 / 0.36859 | 0.32832 / 0.36859 | 0.043927 / 0.078918 | 0.021586 / 0.037905 |
| M3 / 6 / Fluid T | 0.57057 / 0.79406 | 0.63639 / 0.7882 | 0.14506 / 0.15383 | 0.18918 / 0.2233 |
| M3 / 6 / Module peak | 0.54545 / 0.76102 | 0.4851 / 0.61712 | 0.20025 / 0.25225 | 0.22274 / 0.26516 |
| M5 / 6 / u | 0.015954 / 0.016332 | 0.015954 / 0.016332 | 0.0060189 / 0.0071249 | 0.0061768 / 0.0083826 |
| M5 / 6 / Near omega | 0.32669 / 0.33691 | 0.32669 / 0.33691 | 0.063764 / 0.11985 | 0.063588 / 0.11565 |
| M5 / 6 / Fluid T | 0.83382 / 1.0168 | 0.79538 / 0.96317 | 0.18209 / 0.23368 | 0.28124 / 0.38244 |
| M5 / 6 / Module peak | 0.93638 / 1.3883 | 0.86917 / 1.5013 | 0.23847 / 0.29096 | 0.35436 / 0.55996 |
| M7 / 6 / u | 0.017868 / 0.018718 | 0.017868 / 0.018718 | 0.0058217 / 0.0069568 | 0.0062221 / 0.0081936 |
| M7 / 6 / Near omega | 0.31565 / 0.33898 | 0.31565 / 0.33898 | 0.067117 / 0.12138 | 0.069183 / 0.12465 |
| M7 / 6 / Fluid T | 0.7788 / 0.91716 | 0.91846 / 1.0505 | 0.25565 / 0.36275 | 0.29168 / 0.45488 |
| M7 / 6 / Module peak | 0.74722 / 1.0451 | 0.7637 / 0.90061 | 0.32167 / 0.38529 | 0.38754 / 0.467 |
| M10 / 4 / u | 0.021346 / 0.022324 | 0.021346 / 0.022324 | 0.0060005 / 0.0067948 | 0.006055 / 0.0063178 |
| M10 / 4 / Near omega | 0.30815 / 0.31688 | 0.30815 / 0.31688 | 0.02735 / 0.02974 | 0.032263 / 0.040567 |
| M10 / 4 / Fluid T | 0.98571 / 1.127 | 1.1186 / 1.286 | 0.21449 / 0.23151 | 0.2138 / 0.23 |
| M10 / 4 / Module peak | 0.96085 / 1.1683 | 0.91383 / 1.0894 | 0.4285 / 0.58887 | 0.45579 / 0.58342 |

![Native angular surface temperature, original normal-q proxy and outside-port temperature](../../../diagnostics/generated/rdirect_readiness_20261007/figures/appendix_native_roles.png)

**Native-role appendix.** Source 0 on exposed 0687 retains the original 64-angle receiver ordering. The plotted normal q is the saved native −harmonic(k)/delta × (outside−surface) proxy, not continuum flux. New-family initial-port history is NA. Surface, q and outside-temperature errors remain separate rather than being collapsed into fluid-temperature success. [PDF](../../../diagnostics/generated/rdirect_readiness_20261007/figures/appendix_native_roles.pdf).

Detailed arrays and receipts are local ignored artifacts: [R-direct physical fields and strata](../../../diagnostics/generated/response_operator_20261006/evaluation/R-direct_selected_fields/summary.json), [R-group fields and strata](../../../diagnostics/generated/response_operator_20261006/evaluation/R-group_selected_fields/summary.json), [selected Dense DEV22](../../../diagnostics/generated/response_operator_20261006/evaluation/classic_Run1804_DEV22_fields/summary.json), [selected HONF1502 DEV22](../../../diagnostics/generated/response_operator_20261006/evaluation/classic_Run1502_DEV22_fields/summary.json), [Dense endpoint](../../../diagnostics/generated/response_operator_20261006/evaluation/classic_Run1804_e5000_DEV22_fields/summary.json), [HONF1502 endpoint](../../../diagnostics/generated/response_operator_20261006/evaluation/classic_Run1502_e5000_DEV22_fields/summary.json), [direct counted responses](../../../diagnostics/generated/response_operator_20261006/evaluation/R-direct_selected_counted/summary.json), [Dense counted responses](../../../diagnostics/generated/response_operator_20261006/evaluation/classic_Run1804_counted_responses/summary.json), [HONF1502 counted responses](../../../diagnostics/generated/response_operator_20261006/evaluation/classic_Run1502_counted_responses/summary.json), [flow startup history](../../../diagnostics/generated/rdirect_readiness_20261007/startup/D-sep-startup/history.json), and [thermal startup history](../../../diagnostics/generated/rdirect_readiness_20261007/startup/R-direct-startup/history.json). These are locally usable links; generated assets are deliberately absent from Git under the repository artifact rule.

### Delivery and next steps

No new reference solves were attempted: the ledger remains 326/326. GPU-associated time includes both successful flow children, the failed initialization, read-only classic/precision replays and startup processes; the local closeout receipt accounts for the bounded round below 8 GPU-associated hours and 6 elapsed hours. Only GPUs 0 and 2 were used, in the ModularDT environment. No R-group retraining, forced K, inverse/Wind campaign, formal3501/3502 restart or automatic5000 launch occurred. Scientific raw arrays, checkpoints, historical failures and security controls remain preserved; only superseded newly generated 12-panel visual copies were replaced with readable six-panel pages.

Next predictor step is the manual fresh full-TRAIN comparison with exact component ages, canonical89 and original90 evidence, and explicit peak/vorticity/geometry limitations. Next organizer step, if separately authorized, would measure actual execution work at the same fidelity rather than infer it from local K. Next inverse step, if separately authorized, would require independent saved-response or physical design validation. None of those future studies is claimed as completed here.

Validation completed with **109 focused CPU tests passed**, Ruff and Python compilation on all 18 changed Python files, unchanged hashes for six retained selected/endpoint checkpoints, and `git diff --check`. The Markdown browser preview loaded all eight embeds with no horizontal overflow; figures and table screenshots were inspected, all local targets were checked, and every report prose paragraph/caption stays on one source line. The closeout at 0.648 elapsed hours records 0.0787 hours from measured inner GPU process receipts. To cover imports and uninstrumented overhead, charging both authorized GPUs for the entire round gives a conservative 1.296-hour ceiling, still below eight. See the [resource closeout](../../../diagnostics/generated/rdirect_readiness_20261007/round_closeout.json) and [retained-model hashes](../../../diagnostics/generated/rdirect_readiness_20261007/retained_model_hashes.json). Durable source/tests/profiles/docs are committed and pushed after a complete outgoing-history artifact audit; checkpoints, numerical arrays, figures and one-time renderers remain local and ignored.
