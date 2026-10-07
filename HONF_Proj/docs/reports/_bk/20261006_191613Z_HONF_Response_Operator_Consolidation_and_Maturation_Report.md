# Learned source-response interface: consolidation and maturation

The forward interface now prepares a learned temperature-response operator from the complete layout and prescribed environment, then applies physical heating separately. Two independently trained readouts use the same encoder and near-source architectures and common initialization, then update separate weights: R-direct evaluates individual far-source coefficients; R-group represents them through source memberships and signed receiver functions. Both compose with one frozen, heat-independent D-sep flow reader. The protected Run3801 thermal predictor remains unchanged.

**Predictor:** the trained source-response formulation is useful. R-direct2500 has fluid/surface/material temperature RMSE **0.774634 / 0.712311 / 0.677731** on all 22 exposed DEV cases, meeting the declared mean and p90 tolerances. Its saved fixed-audit response RMSEs are **0.0115873 / 0.0148677 / 0.0167672**, better than both H-add and the stronger Tensor-H control. Both 0291 signs are corrected in the new pair, with remaining mean-amplitude excess of 27.7% direct / 36.4% group. Protected Run3801 keeps its original wrong signs. Complete high-M Q8192 cold inference is **25.450 ms direct /31.604 ms group** versus 120.973 ms optimized H-add. The shared D-sep reader still fails the vorticity replacement target, so this is not an unqualified five-field replacement.

**Organizer:** the factors implement actual, measurable source-response routes, but grouping adds no demonstrated value over the competent direct arm. Group temperature-response errors are generally worse; its selected cold call is 24.2% slower. Neither nonzero gradients nor active memberships establish a unique grouping advantage or physical causality.

**Inverse:** prepared full outputs agree with cold outputs on tested inputs, and heat VJPs agree with exported learned kernels in physical units. Strict FP32 finite-increment checks retain failures under the original tolerances. Saved heat allocations can be replayed within measured limits; obstacle-move thermal responses still miss important amplitudes and signs. Rank-one excitation, exposed panels and no new physical validation prohibit a broad inverse claim.

| Question | Measured verdict |
|---|---|
| A: grouping adds value beyond a strong predictor | Not established: selected fixed-audit fluid response 0.0138301 group vs 0.0115873 direct, and group cold call 24.2% slower. The response formulation itself improves over archival controls. |
| B: interpretation matches actual information flow | Supported as software: input-only coefficient preparation, measured cold flow null, physical-unit K–VJP agreement and actual factor/geometry interventions. Memberships are not causal physics. |
| C: response knowledge transfers beyond fitted heat families | Positive on existing exposed DEV/fixed-audit heat directions; incomplete for geometry moves, unexcited directions and independent layouts. No new physical attempts. |


## Experiment identity and completed work

This is fixed25_v1 development, not a formal or independent-test result. The semantic fingerprint is `933b0138ba2f8447a1ecadfe31fd0bb2cb4a05607d3ac3d9f0dc79419f196044`: 150 primary TRAIN cases and 22 repeatedly exposed validation cases. Original-TRAIN response families 0001/0318/0333/0348 form the declared addendum; only 0348 is already primary TRAIN, so there are three auxiliary families. Response-development0304/0320/0335/0350 and the fixed audit 0277/0291/0294/0687 supplied no response-fitting gradients, coefficient calibration or operator-residual construction. They have historical exposure. Input-only fixed sampling, normalization fitted on selected TRAIN, seed 0, losses, case order, query streams and ordinary update budgets are matched between the new thermal arms.

R-direct has 202,026 active thermal parameters and R-group has 202,114 (0.044% difference). Common initial encoder/near tensors match exactly. Their far-head hidden widths are 227/64, chosen before results to match real parameter counts. The thermal arms start from new random thermal weights; Run3801 supplies the unchanged TRAIN normalization and a protected reference, not initialization or teacher-generated heat-response labels. Comparisons with mature H-add and Tensor-H have different historical training and input representations. The isolated grouping comparison is R-group versus R-direct.

The sealed recipe uses AdamW, weight decay 1e-5 and gradient clipping 1.0. Learning rate is 3e-4 through epoch 1000, then cosine-decays to 3e-6 at 2500. Every development epoch visits all 150 cases: microbatch 8, effective batch 48, four ordinary optimizer updates. The equal-case/equal-module objective averages separately standardized fluid/surface/material temperature MSE, with 0.05 standardized q-proxy MSE. Each update also visits one rotating positive TRAIN response family. TRAIN response RMS scales are 0.287101/0.440027/0.554004 for fluid/surface/material temperature. Response and operator coefficients are both 1.0, calibrated once using mean per-case/per-family TRAIN gradient norms across both initialized arms, under the declared cap 1 and half-temperature-gradient rule. The coefficient is the minimum of the cap and each arm's half-mean-temperature-norm divided by its mean auxiliary norm; cap 1 is active for both terms. This initial norm statistic differs from the pooled mean-gradient-vector cosines reported by later checkpoint probes and does not guarantee a later or combined auxiliary half-gradient bound. Neither statistic tunes coefficients on DEV.

The qualified training-only discrete balance applies 128 stratified stencil rows per case/epoch across boundary, solid, interface and fluid, with all active source columns. Stored TRAIN velocities are supervision coefficients; they never enter the inference encoder. Initial qualification used 23 operator applications, one adjoint check and zero solves/integrations/inversions. Stored-field residual RMS was 2.46e-5–3.16e-5, or 6.59e-5–1.14e-4 relative to forcing. Stopping and FP32 indicators qualify this result; they are not a physical discretization-error certificate. A synthetic fractional-forcing q comparison exceeded the retained strict arithmetic tolerance in one element; its failure was recorded before sealing rather than hidden.

Both arms completed 2500 genuinely new thermal epochs: **375,000 primary case visits and 10,000 optimizer updates per arm**. Recorded target work per arm is 384M fluid +68.08M material +34.04M surface reconstruction queries, 13.24M response rows,48M operator rows and 272.32M operator source-column rows. Actual native neural reconstruction work is 810,720,302 rows over 2500 epochs. The logging-only extension after review100 records auxiliary neural work for epochs 101–2500:16,664,097 response rows,204,087,586 padded operator receiver rows (183,062,624 unique), and 999,057,261 total neural rows over those 2400 covered epochs. The initial 100 auxiliary neural counters are unavailable and are not extrapolated. The two arms' recorded streams match exactly. There was no mathematical recipe change after sealing.

Direct selects literal 2500 (score 0.00962795548); group selects literal 2400 (0.00984464539), with matched group2500 endpoint0.00985780176. Native endpoint/selected evaluations are reused when weights coincide. The final CPU audit verifies actual moments,10,000 endpoint AdamW steps, 9,600 group-selected steps, aliases, all 25 milestones, source digests and normalization. Final pooled checkpoint gradient probes include conflicts; they did not trigger coefficient searches or extra training.


All declared 100-epoch checkpoints, latest and one best-field alias are retained. Best selection uses only the common all 22 Q1024 three-temperature reconstruction score at saved monitoring epochs. Native all 22 measurements were completed at 100/500/1000/final. Reviews at 100/500/1000 authorized continued healthy fitting under the unchanged recipe. At 100, direct/group native fluid/surface/material RMSE was 3.03178/3.31361/3.73911 and 3.29780/3.49108/3.02540; at 500 it was 1.36281/1.21953/1.40834 and 1.57231/1.45711/1.74412; at 1000 it was 0.953094/0.832877/0.793579 and 1.12010/0.928811/0.907165. These were valid underfit states, not final qualification passes.

![Matched measured learning curves, actual updates, schedule and training time](../../../diagnostics/generated/response_operator_20261006/figures/predictor/01_learning.png)

Figure 1. Actual 2500-epoch histories, fixed 150 TRAIN/22 exposed DEV and four TRAIN response families; the full objective learns throughout maturation. Monitoring selects direct2500/group2400, separately from terminal 2500. Complete training-epoch times include all scheduled objectives and updates, excluding separate loading/monitoring/saves; outer process times are priced below. [PDF](../../../diagnostics/generated/response_operator_20261006/figures/predictor/01_learning.pdf).

## Predictor: fields, physical responses and dependency

|State|Fluid T mean / p90|Surface T mean / p90|Material T mean / p90|
|---|---|---|---|
|Run3801 Add500|0.770312 / 1.10148|0.788326 / 1.17822|0.66308 / 1.13675|
|R-direct selected/end2500|0.774634 / 1.01772|0.712311 / 1.1134|0.677731 / 1.01644|
|R-group selected 2400|0.844351 / 1.1176|0.716744 / 1.19063|0.658986 / 0.973309|
|R-group endpoint2500|0.842532 / 1.10339|0.717484 / 1.19259|0.657391 / 0.974099|

These native-unit equal-case RMSEs use all 22 exposed DEV cases. Both selected arms pass the prospective mean+10% and p90+15% temperature tolerances. Direct is 0.56% worse in fluid mean,9.64% better in surface and 2.21% worse in material than Run3801; group fluid mean is 9.61% worse, close to the limit. Direct/group q-proxy means are 2.55899/2.68638 versus 3.35197 incumbent. Outside-temperature means are 0.728568/0.751461. Derived h-effective means are 4.11153e-5/0.0368062; ratio cancellation and clipping prevent using these as evidence of independent interface physics. Two historical initial-port roles are NA, rather than an invented all 24 pass. Full role tails and M strata appear in the appendix.


Temperature, surface, material and outside roles are extracted from the same learned shared native grid using the original cell-center bilinear stencils and module-coordinate transforms. The flux proxy is `-k_interface*(T_outside-T_surface)/delta`, with harmonic conductivity and `delta=min(dx,dy,0.15*radius)`. h-proxy and h-effective retain their own epsilon/clipping/validity rules and need not be heat-affine. Material maxima are computed after reconstructing the material field. Historical initial ports are explicitly **not applicable**; this family has no P0/P1/P2 trajectory. Final native port quantities remain measured under their original definitions.

![Saved native reference and learned temperatures with signed residuals](../../../diagnostics/generated/response_operator_20261006/figures/predictor/02_native_fields.png)

Figure 2. Original 128×64 physical grid on exposed DEV 0277/0291/0294/0687, direct2500/group2400; common temperature scales per layout and unclipped residuals in dataset units. Large plumes are reconstructed but downstream bias remains. All 22 fluid means 0.774634/0.844351; group is weaker at M10 (1.11858 vs direct0.985713). The packed 0277 nominal field does not replace its missing fresh-response baseline. [PDF](../../../diagnostics/generated/response_operator_20261006/figures/predictor/02_native_fields.pdf).

![All actual material receivers and original angular surface and flux residuals](../../../diagnostics/generated/response_operator_20261006/figures/predictor/03_native_roles.png)

Figure 3. Same selected states/DEV cases, all stored material points and input-selected module 0 at original 64 surface angles. Material mean RMSE 0.677731/0.658986; surface0.712311/0.716744; q-proxy2.55899/2.68638 in its native proxy units. The 0291 angular residual remains substantial despite aggregate gains. No continuum AD-normal flux substitutes for the stored proxy. [PDF](../../../diagnostics/generated/response_operator_20261006/figures/predictor/03_native_roles.pdf).

The active source is analytic-wake flow plus a shared thermal grid, not Navier–Stokes CFD. Source inspection of all 690 embedded configurations found projection disabled, prescribed inlet 1, 17 Reynolds numbers, zero thermal inlet/wall/initial state, fixed alpha 0.01/0.02 and conductivity 1/1. Fixed-coefficient diffusion and prescribed-velocity upwind advection make point-temperature updates affine in heating at a common integration step. Convergence stopping and saved FP32 endpoints qualify exact endpoint affinity. Ports are derived quantities, not a separately solved Robin interface. See the maintained [source audit](20261006_154229Z_HONF_Dependency_Correct_Source_Audit.md).

The generic core and Thermal input whitelist contain geometry, physical source identities/measures, material and prescribed operating context, environment coordinates and geometry-derived features. Current heating, observed temperatures/velocities, effective ports and target arrays are absent from coefficient preparation. Heating enters only forcing application. Zero heating preserves the obstacle in the layout; removing it changes context. Geometry changes rebuild the operator and retain live input derivatives. There is no physical generator, analytic inference substitution, truth cache, detached-input trick or second old thermal wrapper.

The fixed audit contains three baseline-relative layouts and six signed responses, with original masks;0277 is reported separately. Temperature columns are fluid/surface/material; flux uses dataset proxy units. The archival controls' historical training differs from the new matched pair.

|Fixed-audit state|Fluid ΔT RMSE|Surface ΔT RMSE|Material ΔT RMSE|q Δ RMSE|
|---|---|---|---|---|
|Zero response|0.051185|0.097755|0.117894|—|
|Archival G-fast1000|0.043103|0.054331|0.053634|—|
|Archival Tensor-H1000|0.027871|0.033235|0.034309|—|
|Retained H-add500|0.037067|0.050119|0.049496|—|
|R-direct2500|0.0115873|0.0148677|0.0167672|0.043885|
|R-group2400 selected|0.0138301|0.0158156|0.0173693|0.0389086|
|R-group2500 matched endpoint|0.0138215|0.0157713|0.0173|0.0388083|

Direct improves these temperature errors by 68.7%/70.3%/66.1% versus H-add, and 58.4%/55.3%/51.1% versus stronger Tensor-H. Group also beats both archival temperature controls but does not beat direct. Both signs/roles enter these macros; detailed sign arrays remain available.

|State/cohort|Fluid ΔT|Surface ΔT|Material ΔT|q Δ|
|---|---|---|---|---|
|R-direct selected fit|0.01489|0.0102708|0.0142343|0.136655|
|R-direct selected development|0.0515914|0.0352876|0.0351455|0.156499|
|R-group selected fit|0.0158129|0.0122941|0.0160224|0.146965|
|R-group selected development|0.0541741|0.0448767|0.0412751|0.153293|
|R-group2500 DEV|0.0540315|0.0446872|0.0410185|0.15318|

Each fit/DEV aggregate has four layouts/eight correlated signed states. DEV was withheld from these response gradients, but remains historically exposed. Direct DEV temperature responses improve46.4%/66.1%/66.3% versus retained H-add's0.096167/0.103982/0.104342. Heat-only cold flow changes and 8%-inlet/outlet pressure-drop changes are exactly zero for the new pair; actual inputs and AD verify this imposed dependency independently.

![Actual saved0291 minus and plus heat responses and signed residuals](../../../diagnostics/generated/response_operator_20261006/figures/predictor/04_0291_both_signs.png)

Figure 4. Exposed sentinel 0291, original physical coordinates/common fluid mask; direct2500/group2400. Reference means−0.02891885/+0.02891378, direct−0.03691884/+0.03691884, group−0.03944306/+0.03944283. Both signs improve over protected Add500, but absolute mean-amplitude errors are 0.00800/0.01053 (about 27.7%/36.4%). These are measured directions, not a certified sign floor. [PDF](../../../diagnostics/generated/response_operator_20261006/figures/predictor/04_0291_both_signs.pdf).

Central closure below uses `T_plus+T_minus−2*T_baseline` after FP64 widening. The evaluator stores the half-mean convention, so both prediction and reference RMS are multiplied by two before comparison. On 0291/0294/0687, reference fluid closure is 1.26082e-5/4.42661e-7/7.39003e-7; direct 5.24849e-7/6.42185e-7/1.12344e-6; group7.66365e-7/1.00679e-6/1.30975e-6. This removes the old artificial curvature (Add5000291:0.0178700), but does not replace the response-amplitude tests.


Run3801's unresolved 0291 signs remain intact: reference minus/plus mean temperature changes are -0.02891885/+0.02891378, while protected Add500 predicts +0.01042027/-0.01550809. Its prior wrong signs are not a new flow-fit failure. Near-zero closure in the new family is a model-class identity; correct signed amplitudes require the separate saved-response measurements.

### Supporting flow continuation

D-sep is a literal schedule child of the saved epoch 1000 flow reader. Its weights and AdamW moments were preserved; only the declared learning-rate schedule changed. New ages 1–20 warm from 1e-6 to 1e-4, hold through new 500, then cosine-decay to 1e-6 at new 1500/absolute 2500. The continuation adds 1500 epochs, 6000 updates,225,000 case visits and 230.4M fluid queries. Its measured GPU-associated process time is 225.702 s. D-open1000 remains the frozen, shorter-age dependency control; no new age-matched heat-policy accuracy claim is made.

| Fluid role | Run3801 mean / p90 | D-sep2500 mean / p90 | Prospective mean+5% / p90+10% |
|---|---:|---:|---|
| u |0.0220367 /0.0249139|0.0170684 /0.0208044|Pass|
| v |0.00307966 /0.00399294|0.00171246 /0.00222025|Pass|
| p |0.0122219 /0.0156063|0.00909258 /0.0111285|Pass|
| omega |0.104567 /0.132608|0.131462 /0.164818|Fail|

These are equal-case native-grid RMSE in dataset units over all 22 DEV cases. D-sep's near u/omega means are 0.0341367/0.320751; far means are 0.0117296/0.0471879. Vorticity mean is 25.7% worse than the incumbent. There is no parent-flow fallback. The independent candidate remains available with this disclosed fidelity limitation; more near-source resolution is a hypothesis, not a verified remedy or an authorized new architecture.

![D-sep native velocity and vorticity reference maps and signed residuals](../../../diagnostics/generated/response_operator_20261006/figures/flow/flow_fields.png)

Figure 5. Saved D-sep2500 native fields on fixed exposed DEV 0277/0291/0294/0687, with solid cells masked and dataset velocity/vorticity units. u and omega residuals concentrate near module boundaries; all 22 near-omega RMSE 0.320751 versus far 0.0471879 supports that localization. The analytic-wake references do not certify CFD accuracy. [PDF](../../../diagnostics/generated/response_operator_20261006/figures/flow/flow_fields.pdf).

## Organizer: actual response routes and the matched direct control


The grouped readout is an input-conditioned factorization of the response operator. Current heating enters only when the prepared coefficients are applied. Whole-layout source and environment context, the geometry near weights, and learned receiver functions remain separate paths; source memberships do not describe physical causality.

On the input-declared **0687** layout, the complete scalar grid kernel has shape **[1,8192,10,1]**. Group factors have shapes **b_M=[1,11,10]**, **b_E=[1,11,192]** and **a=[1,8192,11,1]**: ten physical sources, eleven valid response modes and 192 environment context donors. All **110** source membership edges are positive. The full forward model allocates **M+1** modes; this is not learned sparse executor selection. At TRAIN M12/Q8192, direct/group far coefficient rows are **98,304/106,496**, with the same **13,522** exact near coefficient rows.

![Same physical donors and near/far contributions for selected direct and grouped response models](../../../diagnostics/generated/response_operator_20261006/figures/operator_cost/direct_group_operator_0687.png)

Figure 6. Matched donor operators. Direct epoch 2500 and grouped epoch 2400 use identical stored0687 geometry, heat, source IDs and receiver coordinates. Donors 0687:module:0 and 0687:module:9 are selected by input order. Both donor columns share one temperature/heating scale across arms, and both near/far columns share one temperature scale. The plotted terms are actual saved K_near h and (K-K_near)h, excluding the affine offset. Independently solved excitation remains rank one per layout: an individual donor column is a model estimate, not separately validated physical attribution. [PDF](../../../diagnostics/generated/response_operator_20261006/figures/operator_cost/direct_group_operator_0687.pdf).

![Actual grouped physical source support, environment context weights and corrected mode contributions on0687](../../../diagnostics/generated/response_operator_20261006/figures/operator_cost/group_information_flow_0687.png)

Figure 7. Actual information flow. Eleven modes read all ten physical sources through normalized b_M and 192 context donors through b_E. Every positive source edge is displayed without an outcome-selected cutoff; line width/opacity represent dimensionless membership. The environment map displays the input-first valid mode. Corrected contributions are a_e sum_i (1-w_ri)b_ei h_i, and their sum agrees with the saved executed far response at the retained numerical check. Membership is neither energy nor causal flow. [PDF](../../../diagnostics/generated/response_operator_20261006/figures/operator_cost/group_information_flow_0687.pdf).

Two fixed-weight controls change the prepared grouped readout while preserving learned context, geometry and receiver functions. Geometry reassignment preserves original source column marginals and valid-mode row normalization. Removing the first valid source-anchored mode preserves all other terms. Both recompute the exact near subtraction. On the exposed0291 baseline and saved plus state, temperature changes are quadrature-weighted RMS on valid native receivers:

| Fixed-weight control | Max abs change in K | Baseline temperature RMS change: fluid / surface / material | Plus-response temperature RMS change: fluid / surface / material |
|---|---:|---:|---:|
| Geometry marginal matched membership | 4.24375 | 0.70335 / 1.22441 / 1.25857 | 0.06510 / 0.15211 / 0.16171 |
| Remove input-first valid mode | 5.02623 | 0.81502 / 0.22782 / 0.17028 | 7.74e-6 / 5.70e-6 / 5.57e-6 |

The second control substantially changes the absolute field but barely changes this saved response, whose first source has unchanged own heating. That distinction prevents absolute-field sensitivity from being mistaken for response utility. Joint permutation of both factors is a regression rather than a physical intervention: the kernel check passes, with maximum K difference1.43e-6 across fixed 4, while retained native-output differences reach5.72e-5. These controls establish readout sensitivity; they do not establish an accuracy or work advantage over the independently trained direct model.

Complete control arrays, all-role errors and source joins are retained in [factor controls](../../../diagnostics/generated/response_operator_20261006/evaluation/selected_counted_compression/factor_interventions.json); temperature-only summaries are in [final metrics](../../../diagnostics/generated/response_operator_20261006/report_preparation/final_cost_compression_metrics.json). Units are dataset temperature/heating, and the physical reference is existing analytic-wake/shared-grid evidence rather than CFD.


![Learned donor kernels before and after moving the same physical module](../../../diagnostics/generated/response_operator_20261006/figures/geometry_native_ad/R-group_geometry_donor_kernels.png)

Figure 8. Selected group2400, saved fit-anchor 0001 donor module 0 and 0348 donor module 4, same receiver grid and physical donor identity before/after the declared move. Temperature/heating units and common before/after scales expose changed coefficients; separate difference scales show the actual change. These are learned estimates, with no separately solved physical donor-kernel reference. Their movement establishes live geometry conditioning; the physical finite-response misses are reported below. [PDF](../../../diagnostics/generated/response_operator_20261006/figures/geometry_native_ad/R-group_geometry_donor_kernels.pdf).


There is one valid response proposal per active module plus one background proposal. Allocated capacity is not a learned target K. Environmental tokens (192 in this adapter) are prescribed/geometric **context donors**, not 192 independently actuated heating sources. Module source IDs remain identifiable in physical K and its actual contributions. Signed receiver functions carry response units; normalized source memberships are neither conservation certificates nor uniquely identifiable physical causes.

The declared near weight is 1 within two radii and smoothly reaches0 at four radii. Its local MLP evaluates actual active near pairs, while feature construction and far preparation remain priced. Group application includes the exact subtraction of its far response within this near support. Membership controls hold learned receiver functions and context fixed; degree/measure-matched geometry assignment and one removed input-selected mode recompute that subtraction. Joint factor permutation is an invariance test, not a physical intervention. Nonzero group gradients prove training activity, not grouping-added value.

## Inverse: prepared response use and its measured boundary

![Measured unchanged-own-heat module peak responses and finite saved-pool absolute bias](../../../diagnostics/generated/response_operator_20261006/figures/predictor/05_nonlocal_and_pool.png)

Figure 9. Selected direct2500/group2400 on fixed 4 existing pools, temperature units. All four selections have zero realized saved-benchmark regret, including secondary0277; mature archival controls already achieved zero regret, so this is no new policy advantage. Unchanged-own-heat modules have measured nonlocal responses, while absolute peaks—especially0291—remain biased. No new design trajectory or physical solve is depicted. [PDF](../../../diagnostics/generated/response_operator_20261006/figures/predictor/05_nonlocal_and_pool.pdf).

Prepared reuse stores model coefficients for the exact layout/catalogue, not truth or temperatures from a different state. Full cold/prepared native outputs are bitwise equal in the M1/M12 cost checks. Exported heat VJPs pass all tested fluid/surface/material/q roles with maximum discrepancy1.19209e-6 direct /4.76837e-7 group. Actual cold flow changes and composed heat gradients are zero; standalone flow heat inputs are unused. Center/query gradients remain finite and nonzero. These are model-gradient checks, not proof of physical geometry derivative accuracy.

Strict finite increments fail `rtol=2e-5, atol=2e-6`: on 0348, direct fails 1/128 fluid,3/160 surface,82/160 q and 0/80 material elements; group fails 2/128,6/160,90/160 and 3/80. On 0001, q fails8/48 direct and 11/48 group. Largest native q discrepancies are 8.19266e-5/7.58171e-5. Full-grid high-M cost checks also fail strict comparisons for all four tested linear role categories, with interface maximum1.28869e-4/1.36346e-4. Algebraic affinity and exact full-output reuse do not turn those FP32 subtraction failures into passes or physical noise floors. Nonlinear h ratios are excluded from the affine-increment assertion.

![Actual fixed-heat geometry response reference, direct, group and signed residual](../../../diagnostics/generated/response_operator_20261006/figures/geometry_native_ad/geometry_response_comparison.png)

Figure 10. Saved input-declared i_plus obstacle moves on exposed fit-anchor0001/M3 and 0348/M10, common 11-state support, direct2500/group2400; T and pressure have separate native units/scales. ΔT RMSE direct/group is 0.113934/0.141787 and 0.192446/0.187753.0348 reference mean+0.0875451 contrasts with direct−0.00327478/group+0.0107849. Shared D-sep Δp RMSE 0.00330682/0.00157017 is much closer. Thermal geometry transfer remains unqualified. [PDF](../../../diagnostics/generated/response_operator_20261006/figures/geometry_native_ad/geometry_response_comparison.pdf).

Across all four i±/j± saved moves, direct/group temperature RMSE means are 0001:0.141591/0.151717;0348:0.116158/0.120860; DEV 0304:0.162745/0.146559;DEV 0350:0.0993807/0.104477. These geometry response endpoints were not thermal-fit targets. Pressure means are 0.00239709/0.00140514/0.00237896/0.00233061 for both arms. The live coefficient derivatives and actual kernel changes do not repair these observed physical geometry-response misses in the selected models.


Each saved complete family excites one balanced heating direction and its opposite: rank 1 out of M−1=2/4/6/9 for M3/5/7/10. More epochs, receivers and known-operator residual columns do not create independently solved response directions. 0277 has no successful fresh baseline and remains a secondary minus-to-plus endpoint-span comparison; the packed nominal field is not a replacement baseline. Individual source kernels and inferred memberships are estimates outside the measured excitation span.

Compression is restricted to balanced increments within a TRAIN-input-only heat box (0.503671–1.999154). Its radii also respect the declared 20%-of-case-mean bound. Distortion budgets are 0/0.5/1/2% of TRAIN temperature-response RMS 0.440822. Group omission bounds use actual effective `(1-w)*b` weights and source-only centering. They bound approximation to the learned model, not error against physical truth. Absolute baseline predictions remain unchanged; out-of-domain increments fall back explicitly to the full model. Direct SVD is an algebraic approximation of its learned K, not a physical operator inversion. Physical prediction error and compression distortion are reported separately through the triangle decomposition.

No new inverse model, continuous design optimization, denoiser, sampler or physical validation was run. Existing finite pools are replay evidence; correct rank ordering or sampler diversity would not establish valid new designs. The solver allowance remains326/326 with **zero new attempts**. Wind, formal campaigns, restarts, Tree and alternative training portfolios were not launched. Existing mature scientific checkpoints/history remain protected.

## Complete-call cost, validation and artifact receipt


The native benchmark executes optimized H-add Run3801 epoch 500, selected direct epoch 2500 and selected group epoch 2400 on physical GPU0 in FP32. Each cold call rebuilds context, coefficients, native stencils and flow preparation; model/device kernels are resident. It includes native fluid, every stored material point, surface channels and final ports. The incumbent actually executes P0/P1/P2; candidates have zero old thermal-wrapper calls, and initial thermal-refinement ports are not applicable.

The fixed TRAIN conditions have actual M1 or M12 cases in every batch. Five alternating warmed cold-call samples and two warmed full heat/query input-VJP samples per arm/condition give the following medians, in milliseconds. VJPs price differentiation with one common physical-role adjoint; they are not physical-gradient validation.

| Condition | M | H-add cold | Direct cold | Group cold | H-add forward+VJP | Direct forward+VJP | Group forward+VJP |
|---|---:|---:|---:|---:|---:|---:|---:|
| B8Q1024_low | 1 | 89.953 | 17.586 | 19.356 | 168.504 | 22.614 | 24.707 |
| B8Q1024_high | 12 | 130.335 | 21.492 | 23.487 | 234.472 | 28.962 | 31.808 |
| B1Q8192_low | 1 | 109.707 | 24.884 | 31.078 | 236.463 | 34.546 | 39.838 |
| B1Q8192_high | 12 | 120.973 | 25.450 | 31.604 | 267.687 | 34.511 | 40.885 |

High-M Q8192 cold inference is **25.450 ms direct /31.604 ms grouped**, versus **120.973 ms H-add**: both meet the prospective cost ceiling of 1.20 times the optimized incumbent. Group cold inference is **24.18% slower than direct**. Geometry forward+VJP, measured only on this high-M condition, costs **55.903/71.030 ms direct/group versus 267.455 ms H-add**. Peak extra heat/query VJP allocations are **101.6/101.2 MiB versus 2004.8 MiB**; each subtracts invocation baseline with all models resident, and reserved memory includes allocator history. These are measured tensor allocations and physical-row work, not isolated process memory or hardware kernel counts.

![Measured complete native cold calls and prepared three-forcing reuse with formation included](../../../diagnostics/generated/response_operator_20261006/figures/operator_cost/complete_native_cost.png)

Figure 11. Complete cost and reuse. Median bars and observed ranges cover all four conditions. Three distinct high-M heat allocations cost **77.155/97.068 ms cold** for direct/group. Prepared applications cost **17.660/18.183 ms**, plus measured full preparation **20.528/25.681 ms**: **38.188/43.865 ms total**, or **2.02/2.21 times faster** than their respective three cold calls after formation is priced. Formation is one measured preparation; these total bars sum separately measured scopes. Native thermal application, flow read and role extraction are nested scopes and must not be added again. [PDF](../../../diagnostics/generated/response_operator_20261006/figures/operator_cost/complete_native_cost.pdf).

**Retained numerical misses.** Full cold/prepared outputs agree bitwise for both candidates. At M12, however, all four strict increment-versus-subtraction checks fail for both arms at the original rtol2e-5/atol2e-6: fluid temperature, interface including q-proxy, material temperature and final-port outside temperature. Maximum error is **1.28869e-4 direct /1.36346e-4 grouped**. These flags were not relaxed or described as a physical noise floor. M1 balanced changes are zero, so its three applications use identical heat and its passing increment check is trivial.

Compression is an optional view of **increments**, with the full absolute baseline preserved. Its source-only box uses selected-TRAIN heat limits **0.503670633–1.999153733**, and r_i=max(0,min(h_i-h_min,h_max-h_i,**0.20 mean active heat**)), with sum_i delta_h_i=0. The sealed TRAIN-addendum temperature-response RMS is **0.440821614**; distortion budgets are **0,0.5%,1%,2%** of that RMS. No development/audit outcome selects radii, ranks or mode masks. The omission bound uses the actual corrected weights (1-w_ri)b_ei; native temperature interpolation retains the bound. q-proxy, effective h, ports and flow are excluded from the scalar temperature compression budget.

All **eleven** original fixed 4 states are reused: six primary baseline-relative increments and one **0277 minus-to-plus secondary span**, whose failed primary baseline remains absent. All seven saved changes lie inside the sealed box, with **zero actual full-response fallbacks**. Out-of-box requests retain the full response. All **56** existing model-bound and physical-error-triangle flags pass with their original FP32 allowance. Zero-budget direct application differs from the native exact route by at most1.19e-7; zero-budget group application is exact. Physical error is reported separately from approximation error, since compression cannot repair model error.

![Adaptive response-use distortion, storage and measured costs with all-four full formation table](../../../diagnostics/generated/response_operator_20261006/figures/operator_cost/adaptive_compression_fixed4.png)

Figure 12. Qualified compression. Curves use the input-declared 0687 layout, and the table retains all four cases at the 2% budget. Grouped retained receiver-mode slots are **85.74/80.76/89.05/86.74%** for0277/0291/0294/0687; maximum approximation error is respectively **0.002889/0.002245/0.000546/0.000163** dataset temperature. Group coefficients retain dense shapes and exact near subtraction: omission adds a dense mask, and **no sparse executor savings are established**. Positive-budget direct SVD ranks are **2/4/6/9 ofM3/5/7/10**, the balanced common-mode reduction; further rank reduction is not forced. Grouped representation storage can be smaller on 0687, but original coefficients remain resident; this is not measured memory savings. Application is substantially slower. [PDF](../../../diagnostics/generated/response_operator_20261006/figures/operator_cost/adaptive_compression_fixed4.pdf).

A bounded preparation supplement closes the full-formation pricing scope on these exact saved receiver catalogues: one warm-up and **three alternating genuine cold prepare_record timings per arm/case**, totaling32 preparations, with zero output applications or physical evaluations. Role-value access is blocked by a forbidden-access sentinel; input state/catalogue hashes, work counts and unchanged frozen weights are saved. Each preparation rebuilds complete native thermal context/coefficients/stencils and flow state.

| Saved case / baseline | Full native preparation direct / group (ms) | Native + matrix/SVD or mask formation direct / group (ms) | Saved increment application direct / group (ms) | Formation + application direct / group (ms) |
|---|---:|---:|---:|---:|
| 0277 / transfer_minus, secondary | 21.720 / 27.554 | 22.937 / 28.036 | 0.091 / 0.844 | 23.028 / 28.880 |
| 0291 / baseline | 25.346 / 31.505 | 26.524 / 31.946 | 0.131 / 1.575 | 26.655 / 33.520 |
| 0294 / baseline | 32.224 / 38.513 | 33.503 / 38.989 | 0.143 / 1.654 | 33.646 / 40.643 |
| 0687 / baseline | 43.953 / 49.193 | 45.381 / 49.635 | 0.140 / 1.584 | 45.522 / 51.219 |

These totals sum separate measured cold-preparation medians, existing matrix-export/SVD or mask-formation timings, and existing application medians; they are **not a new end-to-end timing**. Direct SVD factors only the learned K on the identical receiver/forcing domain: no physical operator is factorized or inverted. On every fixed 4 case, complete formation plus saved-increment application is more expensive for group than direct. Thus these measurements support response-core reuse while supplying no unique economic benefit for learned grouping.

Sources: [native cost and unchanged-state receipt](../../../diagnostics/generated/response_operator_20261006/evaluation/selected_complete_native_cost/cost.json), [sealed compression/domain and triangles](../../../diagnostics/generated/response_operator_20261006/evaluation/selected_counted_compression/compression.json), [same-catalogue full preparation measurements](../../../diagnostics/generated/response_operator_20261006/evaluation/selected_counted_preparation_cost/preparation_cost.json), and [current-byte visual inspection](../../../diagnostics/generated/response_operator_20261006/figures/operator_cost/visual_inspection.json). Three central GPU0 processes add **0.00698504 GPU-associated hours**, including imports; internal receipts are retained without double-counting. Solver attempts, optimizer updates and new inverse designs are all zero.


All GPU jobs completed successfully. At 2026-10-06 19:04 UTC, the round ledger records **1.427512 aggregate GPU-associated process-hours** (5139.045 s) and **2.057 elapsed hours** since the 17:01:11 UTC goal start, within the 12/8-hour ceilings. Final CPU report/upload closeout follows this timestamp and does not add GPU work. Outer process receipts include import, loading, computation, monitoring, writing and exit; actual PIDs are counted once. Initial inner-only fit receipts receive a conservative 2-second import/exit charge, and the early D-sep1500 native check receives a conservative 60-second charge. The GPU-associated ledger includes training, every CUDA evaluation, the three cost jobs, and the extra direct2200 evaluation dispatched before terminal selection was known. That extra result is retained under its literal age; final direct2500 was measured separately. CPU response evaluations and checkpoint-gradient probes are reported separately. Two CPU preparation integration failures were fixed before sealing; there were zero failed GPU jobs.

Physical GPUs **0 and 2**, mapped by UUID, follow the user's explicit override of the plan's 1/2 assignment. The unrelated GPU1 job was preserved. No new reference attempt, physical solve, inverse search, formal run or restart occurred. [Final process ledger](../../../diagnostics/generated/response_operator_20261006/accounting/final_gpu_ledger.json) and [dispatch correction](../../../diagnostics/generated/response_operator_20261006/accounting/direct_terminal_dispatch_correction.json) retain the individual measured scopes and conservative charges.


The focused CPU regression union passed 132 tests; Ruff and whitespace checks passed. Checks cover input/target poisoning, affine forcing and physical kernel derivatives, actual cold heat-independent flow, native role units, padding/permutation/measure splitting, query/chunk consistency, grouped near subtraction, live geometry/query gradients, stale preparation,2-D/3-D and explicit nonlinear-forcing API contracts, strict schedule-child and data/ normalization identity guards. Existing dependency/executor and read-only Wind APIs passed. Numerical failures retained under old tolerances are reported separately from scientific prediction errors.

Durable implementation and regression tests live in the shared response core, Thermal adapter and maintained fit/evaluation tools. The [interface guide](../../guides/Thermal_Source_Response.md) describes loading, capabilities and replay. Checkpoints, numerical evidence, PDFs, small PNG companions required for direct Markdown embedding and round-local renderers remain under ignored local storage. Figure links work in this workspace; Git contains no generated scientific artifact. Selected PDFs are retained as masters; superseded round-only visual exports are removed after final inspection. Security/trusted-loading and artifact-history gates remain enabled; the entire outgoing commit range is audited before branch push.


## Appendix: native role tails and strata

All values are equal-case native RMSE mean / p90 / maximum on the same22 exposed DEV cases; units follow the named field or proxy. Initial-port NA is a missing historical trajectory, not zero error.

|Role|Run3801|R-direct2500|R-group2400|
|---|---|---|---|
|far/omega|0.0535358 / 0.0651397 / 0.0720683|0.0471879 / 0.0555103 / 0.0607514|0.0471879 / 0.0555103 / 0.0607514|
|far/p|0.0115006 / 0.0151563 / 0.015746|0.00781572 / 0.0105389 / 0.013553|0.00781572 / 0.0105389 / 0.013553|
|far/temperature|0.767569 / 1.13071 / 1.50403|0.768357 / 0.998893 / 1.08233|0.847972 / 1.1612 / 1.30793|
|far/u|0.0200417 / 0.0228665 / 0.0260496|0.0117296 / 0.0148604 / 0.0160129|0.0117296 / 0.0148604 / 0.0160129|
|far/v|0.00279828 / 0.00355701 / 0.00363614|0.00138234 / 0.00183775 / 0.00193388|0.00138234 / 0.00183775 / 0.00193388|
|final_port/h_effective|0.828716 / 1.0807 / 1.08589|4.11153e-05 / 7.68518e-05 / 0.000404154|0.0368062 / 7.68738e-05 / 0.518588|
|final_port/outside_temperature|1.12741 / 1.51141 / 1.88325|0.728568 / 1.13779 / 1.36638|0.751461 / 1.25692 / 1.37092|
|fluid/omega|0.104567 / 0.132608 / 0.145743|0.131462 / 0.164818 / 0.170153|0.131462 / 0.164818 / 0.170153|
|fluid/p|0.0122219 / 0.0156063 / 0.0166512|0.00909258 / 0.0111285 / 0.0142451|0.00909258 / 0.0111285 / 0.0142451|
|fluid/temperature|0.770312 / 1.10148 / 1.52178|0.774634 / 1.01772 / 1.1725|0.844351 / 1.1176 / 1.31186|
|fluid/u|0.0220367 / 0.0249139 / 0.0267397|0.0170684 / 0.0208044 / 0.0229582|0.0170684 / 0.0208044 / 0.0229582|
|fluid/v|0.00307966 / 0.00399294 / 0.00435644|0.00171246 / 0.00222025 / 0.00254663|0.00171246 / 0.00222025 / 0.00254663|
|inlet_outlet_pressure_difference|0.00707586 / 0.0124374 / 0.019937|0.00692762 / 0.01176 / 0.0165952|0.00692762 / 0.01176 / 0.0165952|
|material_temperature|0.66308 / 1.13675 / 1.22715|0.677731 / 1.01644 / 1.22428|0.658986 / 0.973309 / 1.27349|
|module_material_peak|0.647734 / 1.21301 / 1.44839|0.782622 / 1.20335 / 1.47058|0.74378 / 1.11393 / 1.77773|
|near/omega|0.237676 / 0.287368 / 0.347686|0.320751 / 0.342685 / 0.393869|0.320751 / 0.342685 / 0.393869|
|near/p|0.0154642 / 0.0187119 / 0.020403|0.0138137 / 0.0147079 / 0.0172851|0.0138137 / 0.0147079 / 0.0172851|
|near/temperature|0.778342 / 1.16663 / 1.56744|0.7695 / 1.09168 / 1.38157|0.804615 / 1.2566 / 1.32213|
|near/u|0.0304989 / 0.0354021 / 0.0365452|0.0341367 / 0.0368755 / 0.0402244|0.0341367 / 0.0368755 / 0.0402244|
|near/v|0.00421953 / 0.00536157 / 0.00618517|0.00293008 / 0.0035319 / 0.00399479|0.00293008 / 0.0035319 / 0.00399479|
|q_normal_proxy|3.35197 / 4.11759 / 4.56089|2.55899 / 3.47887 / 3.8751|2.68638 / 3.624 / 4.01289|
|surface_temperature|0.788326 / 1.17822 / 1.53758|0.712311 / 1.1134 / 1.35098|0.716744 / 1.19063 / 1.34747|
|initial_port/outside_temperature|1.41298 / 2.23114 / 2.35028|NA|NA|
|initial_port/h_effective|11.8709 / 11.9017 / 11.9177|NA|NA|

|State|M|Cases|Fluid T|Surface T|Material T|Near T|Far T|
|---|---|---|---|---|---|---|---|
|Run3801|3|6|0.602765|0.572658|0.495641|0.570269|0.60368|
|Run3801|5|6|0.706282|0.815118|0.690007|0.783946|0.691801|
|Run3801|7|6|0.846868|0.845531|0.717572|0.853757|0.842591|
|Run3801|10|4|1.00285|0.98583|0.792111|0.968924|1.01452|
|R-direct2500|M3|6|0.570566|0.439268|0.43675|0.49095|0.575156|
|R-direct2500|M5|6|0.833818|0.818709|0.789405|0.875957|0.823333|
|R-direct2500|M7|6|0.778798|0.69013|0.665626|0.742075|0.786256|
|R-direct2500|M10|4|0.985713|0.995547|0.889847|1.06878|0.948848|
|R-group2400|M3|6|0.636389|0.491513|0.434315|0.524599|0.644952|
|R-group2400|M5|6|0.795384|0.721919|0.725185|0.844604|0.785335|
|R-group2400|M7|6|0.918458|0.714323|0.659583|0.84791|0.930048|
|R-group2400|M10|4|1.11858|1.05046|0.895799|1.09971|1.12334|

Detailed per-case/tail/mask arrays remain in the linked numerical summaries; none is relabelled independent test evidence.

## Selected figures and replay evidence

Selected PDF masters and small PNG companions required for inline Markdown are indexed here; numerical arrays, checkpoints and local renderers are retained.

| Evidence | Retained PDF |
|---|---|
|Learning, schedule and measured work|[PDF](../../../diagnostics/generated/response_operator_20261006/figures/predictor/01_learning.pdf)|
|Native fields and residuals|[PDF](../../../diagnostics/generated/response_operator_20261006/figures/predictor/02_native_fields.pdf)|
|Material/surface/flux roles|[PDF](../../../diagnostics/generated/response_operator_20261006/figures/predictor/03_native_roles.pdf)|
|0291 signed heat responses|[PDF](../../../diagnostics/generated/response_operator_20261006/figures/predictor/04_0291_both_signs.pdf)|
|Flow fidelity limitation|[PDF](../../../diagnostics/generated/response_operator_20261006/figures/flow/flow_fields.pdf)|
|Actual donor kernels and near/far contributions|[PDF](../../../diagnostics/generated/response_operator_20261006/figures/operator_cost/direct_group_operator_0687.pdf)|
|Actual memberships and environment context|[PDF](../../../diagnostics/generated/response_operator_20261006/figures/operator_cost/group_information_flow_0687.pdf)|
|Moved-geometry learned donor kernels in physical units|[PDF](../../../diagnostics/generated/response_operator_20261006/figures/geometry_native_ad/R-group_geometry_donor_kernels.pdf)|
|Saved-pool/nonlocal response limits|[PDF](../../../diagnostics/generated/response_operator_20261006/figures/predictor/05_nonlocal_and_pool.pdf)|
|Physical geometry response misses|[PDF](../../../diagnostics/generated/response_operator_20261006/figures/geometry_native_ad/geometry_response_comparison.pdf)|
|Complete native cold/prepared costs|[PDF](../../../diagnostics/generated/response_operator_20261006/figures/operator_cost/complete_native_cost.pdf)|
|Bounded approximation and full formation costs|[PDF](../../../diagnostics/generated/response_operator_20261006/figures/operator_cost/adaptive_compression_fixed4.pdf)|

The local artifact root is `/data/wanglz/ModularDT/thermal_development/response_operator_20261006`. The declaration follows the source-response consolidation plan, SHA256 `28c7e848d48e26b33a061347bb14e9c1e31334b907f416368d0563694f246adf`. The original dependency round is retained as provenance; this report covers the subsequent explicitly authorized maturation scope.

| Literal model | Checkpoint relative to artifact root | SHA256 |
|---|---|---|
| R-direct selected/endpoint2500 | `thermal_pair/R-direct/epoch_2500_model.pt` | `05974d2fbc367bad8f2092063818ce9073c21783131753b3648fb3ea11a870aa` |
| R-group selected 2400 | `thermal_pair/R-group/epoch_2400_model.pt` | `6175bc9c2959066ee80c4dd6d4dec0f7f91194174f4ec7a3369891898164bced` |
| R-group endpoint2500 | `thermal_pair/R-group/epoch_2500_model.pt` | `34f9b2ce73820604dd05c3b94fbe8c528fcdf3fdba1dbd18e86e190361a758d9` |
| Shared D-sep child2500 | `D-sep2500/epoch_2500_model.pt` | `914fe0b4e07c2b805a4f1df1e0d1e54a53180165d61349278acff05c335abdda` |

The literal D-sep parent is `/data/wanglz/ModularDT/thermal_development/dependency_correct_20261006/D-sep/epoch_1000_model.pt`, SHA256 `c9b03e0cffbfdbe388c482c89e79cbdfb28ca53d4b149dbb734e78838adfa6b0`. Protected Run3801 is `/data/wanglz/ModularDT/thermal_development/response_refinement_20261005/ThermalChannel/HONF_Forward_Runs/Run_3801_20261005_214458_thermal_response_refinement25_h-add_v1/epoch_0500_model.pt`, SHA256 `13e2c8bd7c32a8a939b4d18b065300c04c73854412c203efabd728b5cd93006f`; its unchanged digest is verified again during closeout.

Replay summaries: [Round identity](../../../diagnostics/generated/response_operator_20261006/accounting/round_contract.json), [Sealed paired recipe](../../../diagnostics/generated/response_operator_20261006/thermal_pair/paired_recipe.json), [TRAIN-only operator qualification](../../../diagnostics/generated/response_operator_20261006/qualification/operator_decision.json), [Endpoint CPU state audit](../../../diagnostics/generated/response_operator_20261006/thermal_pair/checkpoint2500_CPU_audit.json), [Direct native role summaries](../../../diagnostics/generated/response_operator_20261006/evaluation/R-direct_selected_fields/summary.json), [Group native role summaries](../../../diagnostics/generated/response_operator_20261006/evaluation/R-group_selected_fields/summary.json), [Direct saved fixed-audit responses](../../../diagnostics/generated/response_operator_20261006/evaluation/R-direct_selected_counted/summary.json), [Group saved fixed-audit responses](../../../diagnostics/generated/response_operator_20261006/evaluation/R-group_selected_counted/summary.json), [Direct response-development](../../../diagnostics/generated/response_operator_20261006/evaluation/R-direct_selected_development/summary.json), [Group response-development](../../../diagnostics/generated/response_operator_20261006/evaluation/R-group_selected_development/summary.json), [Group matched2500 response audit](../../../diagnostics/generated/response_operator_20261006/evaluation/R-group2500_counted/summary.json), [Geometry/native derivative summaries](../../../diagnostics/generated/response_operator_20261006/report_preparation/selected_geometry_native_ad_metrics.json), [Complete native costs](../../../diagnostics/generated/response_operator_20261006/evaluation/selected_complete_native_cost/cost.json), [Counted compression and controls](../../../diagnostics/generated/response_operator_20261006/evaluation/selected_counted_compression/compression.json), [Fresh full native formation supplement](../../../diagnostics/generated/response_operator_20261006/evaluation/selected_counted_preparation_cost/preparation_cost.json), [GPU process receipt](../../../diagnostics/generated/response_operator_20261006/accounting/final_gpu_ledger.json).


## Retained models and the next decision

| Model | Status | Exact remaining qualification |
|---|---|---|
| Protected Run3801 Add500 | Retained incumbent | Original wrong 0291 slopes and heat-sensitive flow remain disclosed; weights unchanged. |
| R-direct2500 + D-sep2500 | Candidate | Thermal/heat-response development gains; omega target, strict FP32 increment arithmetic, geometry transfer and independent response directions remain unresolved. |
| R-group2400, endpoint2500 + same flow | Candidate | Meets temperature tolerances, but no grouping advantage over direct; same numerical/geometry/flow evidence limits. |
| D-open1000, Tensor-H1000, G-fast1000, earlier joint/readout controls | Archival control | Frozen scientific comparisons with different historical exposure; no new fits. |

**Next decision: retain the useful source-response formulation and R-direct as the competent candidate; grouping is not better in this mature bounded comparison.** Keep the fixed response representation and exact flow partner for any later confirmation. Preserve Run3801. Do not switch to an unqualified full-field default, repeat an identical portfolio or launch a formal/inverse campaign. The prior17-attempt rank-completion request remains unexecuted; additional fit excitation and independent new-layout confirmation need separate authorization and exposure definitions. No further run is launched by this report.
