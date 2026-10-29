# Round4: attention-inherited gathering, native fidelity and executed work

This round replaces A2-D's repeated nonlocal query–source value MLP with jointly trained associations inherited from the last EM/EE context stages. Thermal uses cached anchor aggregates (IAG-C); Wind uses the one diagnosed query-conditioned alternative (IAG-Q). All four soft/sparse trajectories finish at literal e2500. They reduce dense-query memory and demonstrably use the inherited associations, but lose required field accuracy and increase complete latency. **No candidate qualifies for the conditional full-TRAIN e1000 probe; none was launched.** The [Round2 formal results](../20261014_20261015_Round02_Attention_Geometry/20261014_193935Z_HONF_Attention_Round2_Report.md) and [Round3 report](../20261015_20261016_Round03_A2D_Sparse_Readout/20261016_004810Z_HONF_A2D_Sparse_Readout_Round3_Report.md) remain unchanged. This is a segmented development study against unchanged Round3 D0, with mature full-TRAIN classics as practical context.

| Question at the measured e2500 endpoint | Thermal C | Wind Q |
|---|---|---|
| Are native fields preserved against matched D0? | No: soft/sparse v and p regress; sparse also misses own-soft preservation | No: soft/sparse role-vector means are 2.64–3.67× D0 |
| Does the inherited atlas affect actual outputs? | Yes: association interventions change fields; exact restoration is bitwise | Yes: controls change native velocities, with mixed benefits from individual learned assignments |
| Was work removed? | Nonlocal pair values are removed; canonical scene contractions remain dense; optional selected sparse preparation skips 32.0% of eligible associations | Nonlocal pair values are removed; positive associations fall, but cell/head unions retain full candidate widths and padded dot products |
| Is a complete call faster? | No: soft +16.8%, sparse +18.3% at the high-M fixture | No: soft +36.3%, sparse +38.1% |

**Predictor:** measured gains are lower Thermal q error and 58–61% lower B8/Q8192 allocated memory; misses are the native flow and Wind wake errors. The next model question is how to retain query/source discrimination within the reusable representation; no further training or retuning is included here. **Organizer:** inherited reads are active and sparse normalization creates exact zeros, but the selected backends do not accelerate these complete calls. A useful next executor must reduce head/cell unions or grouping overhead while preserving the same function. **Inverse:** saved Thermal material-response increments improve, but no inverse search or new physical perturbation labels were produced. Native baseline accuracy and endpoint validation remain prerequisites for that separate work.

## What changed in the model

The shared width-128, four-head, two-cycle contextualizer still executes MM → EM → EE → ME. Here EM means an environment destination receives module donors. Only the last EM/EE raw content-plus-geometry scores are exported for reuse; donor log-measure is excluded from that export and applied once by the new normalization. The contextualizer retains its original SDPA computation. Exporting the scores requires a charged score recomputation; fused attention does not expose a free hidden weight buffer.

The exported scores come from their actual last EM/EE stages, while their payloads are projected from the final contextualized module/environment states after subsequent updates. This distinction matters: the reused atlas is a trainable association law, rather than an exact accounting of every token's residual ancestry. MM/ME and the remaining contextualizer routes retain full access. No utility scorer, rank-16 completion, teacher refresh, H6 bank or nonlocal pair-value bypass is instantiated in the new family. The compact geometry-local reader remains.

For each typed donor set, let \(m_i=\mu_i/\sum_j\mu_j\) over valid positive measures. The soft association is \(P_{ahi}=m_i\exp(s_{ahi})/\sum_jm_j\exp(s_{ahj})\). Separate module/environment tensors have shapes \([B,4,E,M]\) and \([B,4,E,E]\). Invalid donors/destinations have zero mass; an empty donor type produces zero aggregate. Final typed values have head width 32. Module and environment measures are normalized separately, preserving their relative measure convention within each type.

The adapter supplies a continuous input-only cover \(\Lambda_{qa}\). Tensor-product smoothstep weights \(S(t)=3t^2-2t^3\) read at most four receiving anchors in Thermal and eight in Wind. Axes and native IDs determine the corners, independently of storage order. The cover is C1 on the regular catalogue, with constant extension from the anchor-center box to the declared legal native bounds; live queries outside those bounds are rejected. Smoothstep has flat slopes at anchors and does not exactly reproduce arbitrary linear fields. It does not make discrete native masks, identities or stencils differentiable.

IAG-C prepares \(G^t_a=\sum_iP^t_{ai}V^t_i\) for each donor type and reads \(c^t_q=\sum_a\Lambda_{qa}G^t_a\). Thus \((\Lambda P)V=\Lambda(PV)\) replaces a query-by-source-by-128 nonlinear value tensor with scene aggregation and indexed cover reads. The query encoder, compact local reader and nonlinear field head still operate at each receiver. Both trained Thermal recipes use dense scene contractions and indexed query gathering. Selected positive-edge scene execution is a separately measured identical-probability diagnostic, including for the sparse child.

Q queries are unit-normalized projections multiplied by a learned bounded per-head temperature; keys are unit-normalized final-token projections. The dot product below includes that temperature. Wind's IAG-Q forms \(\beta_{qhi}=\sum_a\Lambda_{qa}P_{ahi}\), then \(w_{qhi}\propto\beta_{qhi}\exp(q_{qh}\cdot k_{hi})\) over positive inherited support and reads cached final values. This restores query-specific discrimination without constructing nonlinear pair values. Geometry-only cover signatures group queries; each structural cell gathers its inherited union of cached keys/values and executes batched attention. The implementation records logical candidates, padded dot products, cell widths, key/value rows and atlas reads separately. This function does not admit IAG-C's scene/query reassociation. Wind uses cached-token preparation and the cell-selected query executor; its same-weight dense reference is an execution control. Thermal IAG-Q was not activated in this bounded round.

### Native output laws

Thermal retains heat-independent nonlinear u/v/p, native derived curl and a source-indexed, zero-intercept heating law. With eight signed carrier channels per head, \(R_{ahr}(h)=\sum_i(P^M_{ahi}/m_i)t_{ihr}h_i\) and the far temperature is \(\sum_{a,h,r}\Lambda_{qa}\phi_{qhr}R_{ahr}(h)\), plus the maintained local source term. Carriers, \(\phi\), atlas and context are independent of heating at fixed structure and weights. Zero heat therefore gives exactly zero temperature response and no flow change. The signed coefficients and latent carrier norms have different meanings from nonnegative association probabilities.

Native extraction multiplies temperature by the stored TRAIN scale only, without adding a temperature mean. Fluid, surface, outside and material reads share that field; conductivity and spacing define the existing q proxy, and native curl follows the maintained flow transform. Temperature, pressure and q remain **packed-native units**; Kelvin or SI heat-flux labels are not established. Source-indexed affine transport is a software parameterization, rather than a conservation or causality result.

Wind retains \(U(q)=U_{\mathrm{profile}}(z)+U_{\mathrm{ref}}s_{\mathrm{safe}}\odot r_\theta(q)\) in its already aligned native frame. It adds neither an absolute-velocity normalizer mean nor another rotation. Vector RMSE is \(\sqrt{\operatorname{mean}\|U-\widehat U\|_2^2}\), without dividing the squared vector norm by three. The classic checkpoints keep their own native input factories, catalogue sizes, transforms and objectives.

## Protocol and comparison identity

| Setting | Thermal | Wind |
|---|---|---|
| Fixed selected TRAIN / exposed DEV | 150 / 22 cases, from full 600 / 89 | 72 / 24 direction rows, grouped into 24 / 8 layouts, from full 420 / 90 |
| Initial representation | IAG-C | IAG-C, stopped after the documented e500 capacity review |
| Mature representation | IAG-C | One fresh IAG-Q replacement with its own e500 parent |
| Initialization / deterministic sampling | Named initialization seed 0; case/query stream seed 0 | Named initialization seed 0; case/query stream seed 42 |
| TRAIN query budget | 1024 fluid plus unchanged native auxiliaries and scheduled response | 8192: 6144 volume + 2048 hub-band |
| Effective / microbatch cases | 48 / 24; 150 visits and 4 updates per epoch | 16 / 8; 72 visits and 5 updates per epoch |
| Optimizer | All-parameter AdamW, constant 3e-4, weight decay 1e-5, betas .9/.999, eps 1e-8, clip 1 | Same |
| Native objective | Maintained flow/curl, three-temperature, q and original-TRAIN response terms | Maintained .75 volume / .25 hub, equal standardized channel loss |
| Normalization | Fitted only on the selected TRAIN membership | Selected TRAIN normalizer and profile only |
| Native endpoint board | Full DEV22 native fluid/surface/material support | Fixed DEV24 Q8192 five-role board: 1640 × 4 + 1632 |
| Fixed-weight TRAIN check | Cases 0006/0097/0342/0631 | Rows 456/273/536/533 |
| Monitoring / outputs | Validation and declared checkpoints every 100, full history, latest/best aliases, curves every 10, tqdm | Same |

The original TRAIN/TEST separation, exact shared caches and native reference loading remain intact. Wind's selected TRAIN omits M18; direction rows sharing a layout are correlated and are grouped in summaries. The five-role endpoint board differs from the two-role TRAIN loss and the 10240-query volume/hub monitor. All are labelled separately. Rotating pre-update TRAIN loss and fixed-weight native TRAIN error are different measurements.

The final macro-batch of an epoch contains six Thermal or eight Wind rows, exactly as in D0. No case is dropped or repeated to fill an artificial batch. Histories retain actual visits, updates, native receiver counters and deterministic case/query fingerprints, rather than inferring exposure from the nominal effective batch alone.

Wind's single representation revision was justified at e500 using the four fixed TRAIN rows. C's near/downstream vector errors were 0.641/0.485 m/s against same-age D0's 0.333/0.268. After subtracting the same TRAIN profile and centering within cover cells, wake-residual error/target SSE ratios were .526/.716 for C versus .141/.220 for D0, with correlations .688/.533 versus .928/.885. The input-fixed diagnosis supports testing more query discrimination; it does not prove that C has no asymptotic capacity. The C run stops at literal 500 with all evidence retained. Q starts fresh at seed 0 with empty Adam state and the same data/sampling/objective, rather than inheriting C weights. No Wind-C sparse campaign is added.

Each sparse child inherits its own soft e500 parameters, named Adam moments, RNG and 500-row history. Epochs 501–800 use \(P_e=(1-\xi_e)P_{\mathrm{soft}}+\xi_eP_{1.5}\), \(\xi_e=(e-500)/300\); every positive blend entry counts as executed support. Exact measure-relative 1.5 normalization applies from e801 through literal e2500: \(P_i=m_i[(s_i-\tau)/2]_+^2\), with \(\tau\) chosen so the row sums to one. This differs from ordinary entmax applied to score plus log-measure. No additional sparsity penalty or degree target is fitted.

Board A compares independently trained soft/sparse trajectories at literal e2500 and their common e500 state. Board B compares each with unchanged Round3 D0 on the same selected membership, objective, deterministic streams and native endpoints. Board C retains full-TRAIN Thermal1804/1404 at their actual selected ages 4738/4890 and Wind2103/2102 at 2475/2440 on the same DEV22/24 native panels. Those classic columns have unequal training exposure, transforms, objectives and ages; they are practical reference distance. They are not the Round2 full-population formal summaries.

The primary decision is literal e2500. The prospectively declared secondary selector uses the lowest monitoring field_score for Thermal or val_volume_mse for Wind, earliest tie, over common saved ages 600\:100\:2500. It does not search native endpoint errors. Existing engine aliases retain their original domains. The matched fixed TRAIN4 D0 board is newly evaluated from unchanged weights; older R3 two-case TRAIN checks are excluded from the four-case learning curves.

### Qualification and correctness

The plan's modest research criteria require priority means within 10% of D0, priority p90 within 20%, other field/component means within 15%, and either at least 10% lower complete high-M latency or at least 20% lower absolute B8/Q8192 allocated memory with latency no more than 15% slower. An alternative requires at least 5% improvement in the named fluid-T or near-vector priority, all other mean guards, and cost/memory no more than 10% higher. Sparse-versus-own-soft priority means within 5% and p90 within 10% are additional diagnostic targets. Classic replacement and inverse readiness are not qualification gates.

Selective atlas organization requires at least 20% absent eligible high-M associations, with head/query unions disclosed. Sparse-executor acceleration additionally requires the selected backend to beat dense execution of exactly the same probabilities while removing actual work. A faster cached model with a dense atlas can be useful, but zeros alone establish neither skipped computation nor speed.

Strict software checks cover native transformations, source identities, masks, affine zero/increment/transpose laws, live score/value/measure gradients, prepared-state ownership, deterministic checkpoint/resume and immutable source/data bindings. The normalization root was corrected to use FP64 prefix moments and a safe inactive discriminant after a real e501 failure; the failed partial trajectory is retained. Thermal's corrected child restarts from its untouched own-soft e500, rather than reusing those failed updates. This is a numerical correctness repair, not a new representation or sparsity retuning.

Later integration failures were caught at the runtime boundary: the corrected-child CLI compared an unsealed partial resume identity with a fully sealed identity, the actual TrainingEngine attachment guard initially admitted C but rejected Q, and Q's continuation verifier did not yet admit the declared four-file numerical/dispatch correction. The durable fixes preserve strict full-identity validation and allow only matching C/Q architecture identities. Focused actual TrainingEngine attachment/resume tests exercise Adam/RNG/history restoration; seven additional continuation-guard tests pass without optimizer updates. Thermal's frozen-v5 child uses an explicitly hashed driver that independently reconstructs its full sealed identity before invoking unchanged engine code. Wind's e1000→2500 segment uses frozen v6 plus an explicitly declared admission-validator overlay from pushed commit dc442f1. That overlay changes one verifier binding, pins the old/new function hashes and all frozen files, then runs the original validators and unchanged engine; it changes no model, objective, numerical kernel, data, optimizer or sampling state. This segment is not described as byte-for-byte v6 execution. Frozen source trees are never edited in place. Failure logs and narrowly pinned compatibility receipts remain retained.

## Predictor: Thermal native fields

Literal e2500 completes both C trajectories at 375,000 case visits and 10,000 optimizer updates each, counting their common 500-epoch exposure once within each trajectory. Qualification uses the full exposed DEV22 native board, equal case weighting, native masks and stored surface/material receivers. The primary means and p90 values below use packed-native units.

| Native role | D0 mean / p90 | C soft mean / p90 | C sparse mean / p90 | Sparse / soft mean, p90 |
|---|---:|---:|---:|---:|
| Fluid T | 0.87718 / 1.3362 | 0.86976 / 1.3028 | 0.94782 / 1.4858 | 1.090, 1.141 |
| Transverse v | 0.0030744 / 0.0038693 | 0.0043575 / 0.0050909 | 0.0044803 / 0.0051772 | 1.028, 1.017 |
| Pressure p | 0.011296 / 0.014276 | 0.01662 / 0.018715 | 0.017439 / 0.021338 | 1.049, 1.140 |
| Streamwise u | 0.020205 / 0.025148 | 0.026998 / 0.033424 | 0.03225 / 0.040481 | 1.195, 1.211 |
| Native curl | 0.051771 / 0.06195 | 0.068925 / 0.07975 | 0.078636 / 0.092719 | 1.141, 1.163 |
| Surface T | 0.90268 / 1.3866 | 1.0097 / 1.6763 | 1.0434 / 1.8097 | 1.033, 1.080 |
| Material T | 0.91965 / 1.4358 | 0.98982 / 1.6776 | 1.0829 / 1.6816 | 1.094, 1.002 |
| Normal q proxy | 3.297 / 5.391 | 2.5978 / 3.5025 | 2.704 / 3.569 | 1.041, 1.019 |
| Module endpoint-peak RMSE | 0.95248 / 1.9978 | 1.0619 / 1.8513 | 1.0884 / 1.7096 | 1.025, 0.923 |

Soft C preserves fluid-T mean within 1% of D0 and reduces q-proxy mean by 21.2%, but v/p means rise by 41.7/47.1%; u and curl also regress by about one third. Sparse C raises fluid-T mean/p90 by 9.0/14.1% relative to its own soft parent trajectory and raises pressure p90 by 14.0%. It therefore misses the own-soft preservation target. Against D0, its fluid-T mean/p90 remain within the stated priority bands, while v/p and other mean guards fail. Lower q and endpoint-peak tails remain useful measured gains; they do not satisfy the named fluid-T alternative qualification.

The miss is not confined to one example: sparse/soft fluid-T means rise in every available DEV module-count group (M3/5/7/10, respectively 6/6/6/4 cases). The worst fluid-T case for both is 0663: soft 1.582 and sparse 1.642. All role medians, worst cases, per-case metrics and module-count summaries remain in the native JSON boards. The two input-fixed maps below are locally favorable temperature examples and show the remaining flow/pressure error explicitly.

![Thermal0291 native temperature transverse velocity and pressure with aligned residual panels](../../../../diagnostics/generated/attention_round4_20261016T035217Z/evaluation/family1_thermal_native_e2500_v2/family1_thermal_0291_native_fields.png)

Figure 1a. DEV0291, M5, literal e2500, native 128×64 grid with identical fluid mask and unsmoothed coordinates. D0/soft/sparse fluid-T RMSE is 0.827/0.709/0.675, while v is 0.00326/0.00458/0.00468. Prediction and signed-error scales are shared within each physical field, and the residual columns align with their predictors. Local T improvement coexists with worse transverse flow.

![Thermal0687 native temperature transverse velocity and pressure with aligned residual panels](../../../../diagnostics/generated/attention_round4_20261016T035217Z/evaluation/family1_thermal_native_e2500_v2/family1_thermal_0687_native_fields.png)

Figure 1b. DEV0687, M10, same native support and literal ages. D0/soft/sparse T RMSE is 0.861/0.647/0.622; v RMSE is 0.00245/0.00482/0.00497. The shared pressure residual scale also shows that the visual T gain does not establish uniformly better fields. The population table remains the decision board.

![Thermal native surface temperature q proxy and material endpoint maxima](../../../../diagnostics/generated/attention_round4_20261016T035217Z/evaluation/family1_thermal_native_e2500_v2/family1_thermal_native_surfaces_material.png)

Figure 1c. Both fixed scenes retain all 64 stored surface angles and original material receivers. Surface T and q curves average valid modules; shading is 10–90% module variation, not a confidence interval. Endpoint maxima use the actual material field. Good local curves and lower population q error do not remove the fluid/flow misses. Q is the native conductivity/spacing proxy, without an SI flux calibration.

The matching TRAIN4 fluid-T means are D0 .6178, soft .5686 and sparse .5860, while TRAIN v/p remain worse for both C models. Thus the DEV misses cannot be inferred from a single training-loss curve. The declared common-age secondary selector chooses soft e2500 and sparse e2400. At e2400, sparse fluid-T mean/p90 is .9195/1.448, v .004655/.005365 and p .01794/.02209; the selector does not recover the D0 flow capability. These are secondary selected-age results; the e2500 table determines qualification.

## Predictor: Wind native fields

Both Q trajectories finish at 180,000 case visits and 12,500 updates. Their full 2500-row histories have the same case order, query fingerprints, visits and updates as D0; sparse's first 500 rows exactly equal its own soft parent. The exposed DEV24 board has identical input-fixed native queries and targets across all three models, independently bound by the target-parity receipt. The following are equal-direction-row summaries, with grouped-layout checks retained separately; they are not independent TEST evidence.

| Vector RMSE role, m/s | D0 mean / p90 | Q soft mean / p90 | Q sparse mean / p90 | Sparse / soft mean, p90 |
|---|---:|---:|---:|---:|
| Volume | 0.073227 / 0.086414 | 0.23619 / 0.2642 | 0.22875 / 0.26859 | 0.968, 1.017 |
| Hub band | 0.13535 / 0.1656 | 0.49609 / 0.58215 | 0.47588 / 0.53996 | 0.959, 0.928 |
| Downstream | 0.17175 / 0.21666 | 0.55882 / 0.61238 | 0.5357 / 0.58843 | 0.959, 0.961 |
| Near turbine | 0.23481 / 0.28773 | 0.74351 / 0.83491 | 0.72494 / 0.785 | 0.975, 0.940 |
| Background | 0.038907 / 0.046817 | 0.10592 / 0.14133 | 0.10265 / 0.12102 | 0.969, 0.856 |

Q sparse improves the five vector means by about 2.5–4.1% relative to its own soft trajectory. It passes the declared own-soft priority preservation bands, including the six near/downstream components. That result preserves a weak predictor: its five vector means remain 2.64–3.52 times D0, and every priority component mean/p90 fails the D0 bands. Sparse near Ux remains 3.15× D0, Uy 1.90× and Uz 1.62×; downstream component ratios are 3.14×, 2.52× and 2.15×. A lower vector error does not imply that every component improves over soft.

| Priority component RMSE, m/s | D0 mean / p90 | Q soft mean / p90 | Q sparse mean / p90 |
|---|---:|---:|---:|
| Near Ux | 0.22782 / 0.27971 | 0.7364 / 0.82873 | 0.71748 / 0.77833 |
| Near Uy | 0.047723 / 0.05727 | 0.089407 / 0.096568 | 0.090774 / 0.099312 |
| Near Uz | 0.030223 / 0.03646 | 0.048917 / 0.054731 | 0.048992 / 0.055246 |
| Downstream Ux | 0.1689 / 0.21303 | 0.55399 / 0.608 | 0.53047 / 0.58304 |
| Downstream Uy | 0.025576 / 0.032875 | 0.062912 / 0.069349 | 0.064576 / 0.071614 |
| Downstream Uz | 0.016978 / 0.023129 | 0.036972 / 0.04038 | 0.036515 / 0.041272 |

The fixed TRAIN4 checks also remain above D0: volume/hub/downstream/near/background vector means are .038/.060/.081/.118/.029 for D0, .083/.146/.202/.268/.055 for soft and .080/.139/.189/.251/.049 m/s for sparse. These four training rows do not establish a population fit score, but the DEV gap is not explained by holdout generalization alone. Between e1000 and e2500, Qsoft's TRAIN4 near/downstream means improve from .4814/.3859 to .2684/.2021 m/s, while DEV24 worsens from .6572/.5199 to .7435/.5588. Sparse shows the same fit-versus-DEV direction. This supports both a fit limitation relative to D0 and an exposed-DEV generalization problem; it does not establish a unique capacity or optimization cause.

Sparse's worst DEV24 vector cases differ by role: volume row172/M27/layout57 (.280 m/s), hub and downstream row426/M30/layout142 (.565/.677), near row547/M24/layout182 (.868), and background row69/M8/layout23 (.155). Highest layout means are volume layout142 (.264), hub57 (.540), downstream142 (.581), near182 (.822), and background23 (.132). Error does not rise uniformly with module count. The saved boards retain all row/component metrics; grouping shared-layout direction rows avoids treating 24 directions as 24 independent geometries. This is one initialization and one fixed selection, without seed uncertainty or an unseen-test claim.

The prospectively fixed monitoring selector chooses Qsoft e800 and Qsparse e600. The latter is the transition mixture with alpha=1/3, before exact sparse normalization; it is not a mature sparse checkpoint. Their secondary DEV volume means are .2322 and .2387 m/s. Soft e800 near/downstream means are .6293/.4942, still far above D0 e2500. Monitoring uses the maintained 8192-volume/2048-hub panel, an .8/.2 weighted monitor and a volume-only selector, whereas TRAIN uses .75/.25 and this endpoint board has five roles. These distinctions explain why no native-board-driven reselection is made.

![Wind069 native three-component fields aligned residuals and stored crosswind cuts](../../../../diagnostics/generated/attention_round4_20261016T035217Z/visualizations/wind_iag_q_e2500_native_planes_v2/W069/wind_W069_hub_native_endpoints.png)

Figure 1d. Exposed DEV row69, literal e2500, 52,548 common stored plane cells at z=.885874D (about 70.87 m), rather than an interpolated 80 m slice. Ux plane RMSE is .157/.540/.496 m/s for D0/soft/sparse; relative L2 is .0176/.0608/.0558. All three velocity components, prediction-aligned signed errors and native cuts at the actual stored offsets are shown. Shared scales expose diffuse or displaced IAG wakes; all coordinates and targets are original tuples without resampling.

![Wind426 native three-component fields aligned residuals and stored crosswind cuts](../../../../diagnostics/generated/attention_round4_20261016T035217Z/visualizations/wind_iag_q_e2500_native_planes_v2/W426/wind_W426_hub_native_endpoints.png)

Figure 1e. Exposed DEV row426/M30, 99,750 common stored plane cells and original cuts, same checkpoint ages and native frame. Ux plane RMSE is .145/.660/.650 m/s, with relative L2 .0168/.0763/.0752. The weaker IAG wake reconstruction remains visible in both maps and crosswind profiles, including small transverse/vertical components. Plane metrics are a different support from the Q8192 five-role population board.

## Learning: rotating loss and fixed-weight native checks

![Thermal complete histories and fixed-weight TRAIN4 DEV22 field learning](../../../../diagnostics/generated/attention_round4_20261016T035217Z/visualizations/family2_learning/family2_thermal_learning_v1.png)

Figure 2a. Complete saved Thermal histories and native checkpoint boards. Top panels isolate each architecture's rotating pre-update objective; they are not overlaid as a common native accuracy score. Bottom panels show fixed-weight physical fields with mean/p90, not confidence bands. Sparse execution begins at e501 and reaches exact1.5 at e801. D0 TRAIN4 is measured only at e500/e2500; unmatched historical TRAIN2 points are omitted. Lines connect measured ages and do not create intermediate measurements.

![Wind complete histories and fixed-weight TRAIN4 DEV24 wake learning](../../../../diagnostics/generated/attention_round4_20261016T035217Z/visualizations/family2_learning/family2_wind_learning_v1.png)

Figure 2b. Wind C's diagnosed e500 stop, fresh Qsoft and its own e500 sparse child, together with unchanged D0. Native TRAIN4 errors fall while exposed-DEV near/downstream errors worsen after the early Q checkpoints. Rotating TRAIN loss therefore does not establish wake generalization or qualify a formal run. Direction rows share layouts, no TEST rows are used, and no additional architecture revision follows this result.

## Practical distance to the mature classics

The retained classic checkpoints are evaluated through their original native loaders on the same exposed DEV22/24 panels. Their full-TRAIN data, selected ages, normalization and objectives differ from this segmented study. The following means establish practical distance; they do not isolate an architectural effect. Thermal uses packed-native RMSE, and Wind uses m/s vector RMSE.

| Thermal native role | Dense1804 e4738 | Organized1404 e4890 | Segmented D0 e2500 | C soft e2500 | C sparse e2500 |
|---|---:|---:|---:|---:|---:|
| Fluid T | .19794 | .29587 | .87718 | .86976 | .94782 |
| v | .0003190 | .0004868 | .003074 | .004358 | .004480 |
| p | .002100 | .003131 | .01130 | .01662 | .01744 |
| u | .005715 | .01067 | .02020 | .02700 | .03225 |
| Curl | .02346 | .02430 | .05177 | .06892 | .07864 |
| Surface T | .40316 | .45028 | .90268 | 1.00972 | 1.04344 |
| Material T | .30043 | .35989 | .91965 | .98982 | 1.08293 |
| q proxy | 1.55607 | 2.08649 | 3.29703 | 2.59782 | 2.70401 |
| Module endpoint peak | .28529 | .36583 | .95248 | 1.06188 | 1.08838 |

The classics remain stronger across these Thermal mean fields. C's lower q error than segmented D0 therefore does not imply recovery of the classic q capability. The [Round2 report](../20261014_20261015_Round02_Attention_Geometry/20261014_193935Z_HONF_Attention_Round2_Report.md) contains the separately completed full-TRAIN A2-D Thermal e5000 and Wind e2500 comparisons, with their original populations, native laws and costs. Those formal results remain intact and are not substituted by the segmented D0 columns here.

| Wind vector role | Dense2103 e2475 | Organized2102 e2440 | Segmented D0 e2500 | Q soft e2500 | Q sparse e2500 |
|---|---:|---:|---:|---:|---:|
| Volume | .017674 | .036241 | .073227 | .236186 | .22875 |
| Hub band | .034015 | .054632 | .135345 | .496087 | .47588 |
| Downstream | .039630 | .067020 | .171749 | .558821 | .53570 |
| Near turbine | .074779 | .098276 | .234814 | .743510 | .72494 |
| Background | .011262 | .026824 | .038907 | .105916 | .10265 |

Classic organization also has a different meaning: 1404/2102 retain six-slot route mechanisms, while 1804/2103 retain their native dense environment attention. Neither is the new typed EM/EE atlas. Comparing their edge counts, latent norms or old timing fixtures directly with IAG would mix graph definitions and execution scopes. Their already measured formal field, organizer, tiling and memory results remain in the protected Round2 report. Current same-device, same-fixture efficiency decisions use unchanged segmented D0 and same-function IAG backends.

## Organizer: actual inherited reads and intervention effects

Each receiver reads its geometry-owned four/eight cover slots, the last EM/EE association probabilities and the final typed values. The saved organizer arrays preserve all heads, valid donor/anchor IDs, raw scores, measures, cover weights, actual Q query probabilities and values. The graph view uses all positive head-0 EM edges touched by the three input-fixed probes, without top-k drawing. Companion panels retain the other heads and the complete EM/EE atlases. These are learned information routes through a globally contextualized payload, not a physical causal graph or a claim of independent modules.

Atlas replacement, permutation and degree-matched geometry controls keep model weights, contextualizer outputs and final typed values fixed, then rerun the resulting reads and field heads. Exact restoration recovers every baseline output bitwise. This establishes active use of the inherited associations. For a dense soft atlas, degree preservation includes every donor; the nearest-geometry and measure-only controls have the same reported effects here, although their probability byte hashes need not coincide. They are not independent restrictions of soft support. These controls do not establish that every learned edge is beneficial, that a probability is a physical response coefficient, or that sparse training improves field fidelity.

| Thermal two-scene native field score | C soft | C sparse |
|---|---:|---:|
| Baseline | .007803 | .009557 |
| Typed measure only | .145868 | .164312 |
| Permute anchor rows | .026775 | .029671 |
| Remove one input-selected positive EE edge | .007984 | .009734 |
| Degree/measure-matched nearest geometry | .145868 | .126115 |
| Exact association restoration | .007803 | .009557 |

The Thermal score is the maintained native standardized reduction across the two fixed scenes, not a physical-unit RMSE or the full DEV22 board. Large measure/permutation effects show dependence on the atlas, and the matched geometry control is worse on this panel. Sparse's own baseline remains worse than soft, so the controls do not establish an accuracy gain from zeroing edges. The C edge-removal control does not renormalize the row; it also reruns nonlinear flow/φ paths, so its total output change is not a fixed-φ single-edge contribution.

For Wind-Q soft, the same two-scene baseline near/downstream/volume vector errors are .7919/.6164/.2668 m/s. Measure-only and degree-matched geometry worsen near error by 28.2%, while anchor-row permutation improves it by 0.6%; the input-selected single-edge removal slightly improves all five role means. Q renormalizes the surviving inherited distribution after edge removal. These mixed effects demonstrate use, while limiting a claim that the learned anchor assignment is uniformly better than alternatives. The full DEV24 trained-model comparison remains the primary added-value test.

For Wind-Q sparse, the two-scene near baseline is .7445 m/s; measure-only, row permutation, one-EE-edge removal and matched geometry change it by +13.72%, +2.77%, +.013% and +5.54%, respectively. The complete output RMS changes on W69/W426 are .259/.184 m/s for measure-only, .091/.138 for permutation and .247/.236 for matched geometry. Both arms' exact restorations recover every native output bitwise. The sparse model therefore depends on its inherited associations and is better than those geometry controls on these two scenes, while its primary DEV24 accuracy remains worse than D0.

The Thermal signed response matrix K is exported separately from P and latent V norms at the same three probes. On DEV0687 its soft/sparse ranges are −.541…5.697 and −.297…5.742 packed-native temperature per raw heat_power unit; 7/6 of the 30 coefficients are negative. Sparse has one exact-zero coefficient here. Negative learned coefficients do not establish physical cooling. All-head/query unions and the local path can keep a source influential even when an individual association is zero.

The input-only geometry perturbations rebuild organization without receiving physical labels. Moving source0 by +.03 in y on the single-module Thermal TRAIN0006 leaves its module distribution unchanged, as expected with one donor, while EE-P RMS changes by .001279/.000527 for soft/sparse and sparse changes 450 EE support cells. Moving source0 by +.14019245 rotor diameters in x on Wind TRAIN456 changes soft module/EE P RMS by .01000/.0002978. This is a measured graph response to input geometry; the correctness of the corresponding physical perturbation remains untested.

![Thermal actual three-query inherited reads typed values and signed response kernel](../../../../diagnostics/generated/attention_round4_20261016T035217Z/visualizations/family3_organizer/thermal_three_query_gather_e2500.png)

Figure 3a. DEV0687, both literal-e2500 checkpoints, three input-fixed native probes. The graph shows all positive head0 EM paths through the actual cover; the heatmaps retain all four heads' query mixtures and final typed value norms. Graph-node colorbars are per arm; the companion norm heatmaps use a shared scale. The bottom signed K matrix is separate from probabilities and latent norms, uses a common signed native scale, and includes local plus inherited response. Its negative entries are learned coefficients, without a physical-cooling interpretation.

![Thermal complete four-head typed inherited atlases](../../../../diagnostics/generated/attention_round4_20261016T035217Z/visualizations/family3_organizer/thermal_all_head_typed_atlas_e2500.png)

Figure 3b. All eligible EM/EE associations on DEV0687, with original donor/anchor ordering and no top-k filter. Soft is fully positive; sparse removes 31.34% of module and 33.35% of environment head/anchor/donor cells. Blue-gray denotes eligible exact zero; invalid padding is excluded. Uniting heads leaves only 7.7% of module and .55% of environment anchor/donor pairs absent, so an individual sparse head does not imply a comparably sparse union or executor.

![Wind actual three-query inherited paths final query probabilities and values](../../../../diagnostics/generated/attention_round4_20261016T035217Z/visualizations/family3_organizer/wind_three_query_gather_e2500.png)

Figure 3c. DEV W426/M30, actual soft/sparse Q paths. The graph's widths show inherited head0 P; the middle heatmap instead shows the actual query-conditioned weights after beta and QK normalization, scattered to original source IDs. These are distinct tensors. Three-probe module beta support ranges from 17–30 of 30 per head/query, yet its union over those probes contains every module in every head. Latent V norms have no physical units.

![Wind complete four-head module and environment inherited atlases](../../../../diagnostics/generated/attention_round4_20261016T035217Z/visualizations/family3_organizer/wind_all_head_typed_atlas_e2500.png)

Figure 3d. Complete W426 typed atlases and all four heads, literal e2500. Sparse retains 5,310/7,680 EM and 13,515/16,384 EE cells; soft retains all. Across the three fixed probes, environment donor unions by head contain 64/64/64/57 of 64 donors. This is a three-probe organizer observation; full-query execution counts are measured separately and retain full structural-cell widths on the high-M TRAIN fixture.

![Native association intervention effects with fixed weights and typed values](../../../../diagnostics/generated/attention_round4_20261016T035217Z/visualizations/family3_organizer/native_organizer_interventions_e2500.png)

Figure 3e. Complete reruns on two fixed exposed-DEV scenes per dataset, normalized by each arm's own baseline. Thermal uses the maintained standardized native field score; Wind uses near-turbine vector RMSE. Measure/permutation/geometry effects establish computational reliance, with mixed benefits from particular learned assignments; restoration is exactly one and the native outputs restore bitwise. These intervention targets contain no new physical perturbation labels.

The next organizer question is whether a specific inherited read restriction can preserve the native fields and reduce complete work. This round adds no utility, teacher or third trained geometry arm. Atlas zeros, query/head unions and executed counts are reported independently below.

## Execution: complete costs and query-density memory

All new latency/backend/memory comparisons use physical GPU1 (NVIDIA RTX 6000 Ada Generation, 50,875,793,408 reported device bytes), identified by its full UUID in each receipt, and ModularDT Python3.12.13/Torch2.6.0+cu124 with four host threads. The paired low/high-M inputs are selected from TRAIN independently of prediction errors. A complete call starts from a host native request, transfers it, constructs scene/context/atlas and local organization, decodes the full native receiver union, applies physical extraction and returns native arrays to the CPU. Wind local-pair packing is rebuilt inside each call and checked against the original b/q/module arrays. The shared catalogue is already warm; checkpoint loading and disk catalogue reads are excluded. There are two warmups and seven alternating measured repetitions per arm. P90 describes these repeated samples, not uncertainty across machines or datasets.

| Completed candidate | Low-M D0 → IAG median [p90], ms | High-M D0 → IAG median [p90], ms | High-M median ratio |
|---|---:|---:|---:|
| Thermal C soft | 62.17 [64.16] → 71.45 [74.01] | 68.20 [69.21] → 79.66 [83.02] | 1.168 |
| Thermal C sparse | 61.21 [62.46] → 74.79 [76.00] | 68.62 [70.15] → 81.19 [83.90] | 1.183 |
| Wind Q soft | 46.51 [60.56] → 66.77 [89.39] | 54.59 [57.00] → 74.40 [75.71] | 1.363 |
| Wind Q sparse | 46.90 [48.47] → 69.02 [71.10] | 54.78 [56.25] → 75.64 [77.30] | 1.381 |

The displayed D0 values are paired separately with each candidate. Qualification uses those paired ratios, rather than pooling different invocations. Thermal fixtures are TRAIN0006/M1 and 0631/M12; Wind fixtures are TRAIN456/M6 and 533/M30. All four comparisons are slower despite removing repeated nonlocal pair-value construction.

Prepared-scene/new-query and prepared-heating application timings are independent scopes. They do not include all scene work and must not be added to the complete-call time. Atlas score export, normalization, projected tokens, local pairs, cover construction, query grouping, native auxiliary receivers and padding remain charged. A zero nonlocal pair-value counter means that particular MLP work was removed; it does not mean contextualization, token attention, local work or query-specific computation disappeared.

| Paired prepared-scope median, D0 → IAG ms | Low-M new query | High-M new query | Low-M heating application | High-M heating application |
|---|---:|---:|---:|---:|
| Thermal C soft | 23.94 → 29.58 | 30.15 → 35.98 | 5.14 → 7.58 | 5.61 → 7.59 |
| Thermal C sparse | 24.09 → 29.37 | 29.63 → 34.36 | 5.38 → 7.33 | 5.25 → 7.69 |
| Wind Q soft | 10.82 → 26.85 | 18.61 → 31.83 | — | — |
| Wind Q sparse | 11.49 → 26.36 | 18.20 → 33.09 | — | — |

The new-query scope caches scene/context/organization only; each timed call packs and transfers a fresh identical receiver/control request, rebuilds the query/support plan, decodes/extracts native fields and returns them to CPU. Wind's local pairs are rebuilt and checked inside that scope too. Heating application additionally caches native query unions and affine factors, then transfers the unchanged TRAIN heat controls and returns native outputs. Its speed therefore does not price preparation or a new physical solve. These scopes are also slower than paired D0 here; their raw samples, p90 values, parity checks and exact boundaries remain retained.

On Thermal TRAIN0631/M12/Q8192, D0 constructs 98,304 direct source rows and 12,582,912 width-128 message entries. C replaces those repeated values with 204 typed token projections, 156,672 scene associations and indexed cover reads. Sparse has 106,504 positive associations: 50,168/156,672, or 32.0%, are exact zeros. Its canonical dense scene contraction still executes all 156,672 entries and 9,216 source-response associations. Cover counts are 131,072 head/slot positions, 120,000 positive geometric weights and 393,216 actual head/type anchor reads; the geometry limit of four anchors is not a claim of four scalar memory reads. Both retain 7,812 local pairs. The optional selected clone executes 106,504 scene associations, but its complete median is 80.78 ms versus 78.76 ms for the identical sparse probabilities executed densely. This is measured work removal without measured acceleration.

The indexed-query same-function controls are modestly faster than literal dense query execution: soft 77.37 versus 79.07 ms, sparse 80.23 versus 82.07 ms on that native fixture. Soft selected-scene preparation gives 78.83 versus dense 79.95 ms, but has no zero associations to omit. Neither observation establishes sparse acceleration. These are separate paired backend invocations; they are not substituted for the D0-versus-model cost gate, and the generic Thermal interface-screen limitation remains explicit below.

On Wind TRAIN533/M30/Q8192, D0 constructs 245,760 source rows and 31,457,280 width-128 message entries. Q soft instead projects 94 final tokens plus their keys and computes 24,064 inherited associations. Its 27 geometry cells still execute 983,040 module and 2,097,152 environment query/head dot products; padding raises those counts to 2,060,640 and 4,396,032. Maximum widths are the full 30/64 donors. Both arms rebuild the same 608 local pairs inside the timer. The same-P dense Q reference is faster than cell-selected soft Q, 67.90 versus 73.70 ms. Removing the pair-value MLP therefore buys memory, while the remaining attention, grouping and complete native path still cost time. These row, dot-product and latent-entry counts have different units and are not presented as interchangeable FLOPs.

Wind sparse retains 18,901/24,064 atlas associations on this TRAIN fixture, omitting 21.5%. Its positive query/head associations fall to 914,093 module and 1,941,795 environment, but all 27 structural cells still have full 30/64 candidate widths. Executed logical dot products and padded counts therefore remain the same exhaustive counts as soft. The same-P dense Q reference takes 71.24 ms versus 76.40 ms for cell-selected sparse Q. Sparse probabilities and fewer positive associations do not produce dot-product savings or complete-call acceleration here.

The fixed-B8 stress matrix uses one model/provider per process and the identical input hash at Q1024/4096/8192 across arms. Thermal retains fixed native surface/material auxiliaries, so the native query union is larger than the requested fluid budget. Wind uses nested 3:1 volume/hub prefixes from the TRAIN recipe, rather than substituting the five-role DEV board. Each point runs model.eval() with autograd through native outputs and a target-free stress objective, checks finite parameter gradients, and takes absolute allocated/reserved peaks without subtracting model residency. It is a forward/backward memory measurement with zero optimizer updates, not complete Adam-step memory or pure-Q asymptotics.

| Completed B8 arm | Parameters | Q1024 allocated GiB | Q4096 | Q8192 | Q8192 reserved GiB |
|---|---:|---:|---:|---:|---:|
| Thermal D0 | 1,520,493 | 1.252 | 1.988 | 3.011 | 3.438 |
| Thermal C sparse | 1,559,912 | 0.738 | 0.959 | 1.260 | 1.486 |
| Thermal C soft | 1,559,912 | 0.640 | 0.861 | 1.161 | 1.301 |
| Wind D0 | 1,525,675 | 0.865 | 3.067 | 5.994 | 6.332 |
| Wind Q soft | 1,506,471 | 0.462 | 1.262 | 2.315 | 2.631 |
| Wind Q sparse | 1,506,471 | 0.477 | 1.277 | 2.330 | 2.646 |

Thermal soft/sparse reduce Q8192 peak allocated memory by 61.5/58.1% relative to D0; Wind soft/sparse reduce it by 61.4/61.1%. Parameter counts change by +2.6% for C and −1.3% for Q, so this is primarily a measured execution/activation tradeoff. The complete high-M latency ratios exceed the ≤1.15 memory-led qualification band in all four candidates. Their field guards also fail independently. The normalizations have separate trained weights; the soft-versus-sparse columns are model comparisons, not same-function backend tests.

Real TrainingEngine pilots separately exercised the native optimizer boundary: Thermal effective48/micro24 with scheduled responses reached 2,392,907,776 allocated / 3,483,369,472 reserved bytes; Wind-Q effective16/micro8 reached 2,644,155,392 / 6,769,606,656. Those pilot weights were discarded and their updates charged to engineering. Sparse normalization and support unions can change memory during later training; these pilot maxima are measured configurations, not a bound on every future epoch.

| Full saved e1–2500 trajectory | Sum TRAIN seconds | Sum monitor seconds | Maximum training allocated / reserved GiB |
|---|---:|---:|---:|
| Thermal historical D0 | 3616.40 | 4.24 | 5.158 / 30.088 |
| Thermal C soft | 4931.78 | 5.51 | 2.260 / 4.641 |
| Thermal C sparse | 5118.37 | 5.51 | 2.576 / 5.029 |
| Wind historical D0 | 3675.01 | 11.63 | 5.971 / 15.088 |
| Wind Q soft | 6835.43 | 13.93 | 2.575 / 8.154 |
| Wind Q sparse | 6956.77 | 14.03 | 2.582 / 8.154 |

These are actual complete TrainingEngine epoch telemetry, including optimizer-boundary memory, rather than the target-free B8 stress. Reserved peaks are allocator reservations, not live tensor requirements. D0's historical hardware/process state is not a newly matched full-training timing benchmark; the same-GPU 14-call endpoint matrix is the efficiency comparison. A child's full history includes its inherited prefix for trajectory interpretation; resource accounting below counts that prefix once. TRAIN/monitor clocks omit other process work. For the final e1001–2500 segments, total process times are 3637.24/3785.45/4590.70/4670.34 seconds for Thermal soft/sparse and Wind soft/sparse, versus TRAIN+monitor sums 2944.68/3084.62/4108.94/4197.83. The gap includes output/setup and other engine work; no unmeasured I/O-only time is inferred.

Same-function backend tests keep one checkpoint and the complete association array fixed. Thermal compares canonical dense scene contraction with a selected-edge clone, and indexed query reads with a literal dense reference. Wind compares cell-selected Q with a dense query reference. Full native B1/Q8192 forward checks pass in all completed controls under prospectively sealed TRAIN-scale and propagated-native-operator bounds. For Thermal, the separate generic-Q8 screen fails on `pred_interface` and stops before gradient comparison; its gradient equivalence is untested, not failed. Both Wind-Q normalizations pass their generic normalized-output and gradient checks. No universal Thermal backend-gradient qualification is inferred from the full-native forward result.

Three diagnostic harness failures remain preserved: a CPU scene passed to the GPU D0 prepared scope, a requirement for a response tensor absent from the maintained native fixture, and hashing a scalar int64 through a zero-dimensional byte view. The corrections fix device setup, optional-fixture admission and scalar byte serialization. They do not change trained weights, native kernels, measured tolerances, query budgets or saved failed outcomes. Successful Thermal-soft cost-v2 and backend-v3 records are used explicitly.

![Thermal native accuracy complete call cost same-function backends and B8 query-density memory](../../../../diagnostics/generated/attention_round4_20261016T035217Z/visualizations/family4_cost_memory_v2/thermal_native_accuracy_complete_cost_memory_e2500.png)

Figure 4a. Literal e2500 on physical GPU1: field mean/p90 ratios, complete native latency, absolute B8 forward/backward allocated/reserved memory and same-function backend ratios. D0 plotting bars pool the 14 samples from the two paired candidate invocations; qualification uses each separate paired D0 ratio in the table. Whiskers are repeated-call p90. Memory inputs match exactly across all three arms at every Q. Selected sparse scene execution omits work but is slower than its identical-P dense contraction; the separate generic Thermal interface/gradient limitation is retained in the text.

![Wind native accuracy complete call cost same-function backends and B8 query-density memory](../../../../diagnostics/generated/attention_round4_20261016T035217Z/visualizations/family4_cost_memory_v2/wind_native_accuracy_complete_cost_memory_e2500.png)

Figure 4b. Same physical GPU, measured native scopes and fixed B8 inputs, including every required near/downstream component. Q removes pair-value activation growth, reducing Q8192 allocated memory by 61.4/61.1% for soft/sparse. Cell-selected execution is slower than the same-P dense Q reference for both models. The sample-p90 variability at low M remains visible; neither the pooled plotting summary nor a logical support count replaces the paired qualification decision.

## Inverse and response: bounded saved evidence

There is no inverse optimization, new design search, new PDE solve or new Wind perturbation target in this round. The response check reuses saved Thermal0291/0294/0687 heat-transfer pairs with both signed directions: six replays of the analytic-wake/shared-grid reference. The missing 0277 baseline remains explicitly unavailable. Each model prepares its context and response factors once per scene and reuses them for all heat vectors. Flow is unchanged by heat, zero heat gives zero temperature, and material peak changes are computed as max(T endpoint) minus max(T baseline), never max(ΔT).

| Mean of six saved weighted response RMSEs | D0 | C soft | C sparse |
|---|---:|---:|---:|
| Fluid-role T increment | .013254 | .013123 | .014289 |
| Interface-role T increment | .017368 | .015645 | .017238 |
| Material-role T increment | .022121 | .014105 | .015323 |

These are arithmetic means of six saved role-specific weighted RMSEs, not a pooled receiver statistic. Soft/sparse material response errors improve by 36.2/30.7% relative to D0, while sparse fluid response worsens by 7.8%. A good increment prediction does not imply an accurate baseline temperature or a valid inverse design. Individual cases and endpoint-peak changes remain visible below and in the saved numerical record.

![Saved Thermal temperature-response maps and aligned increment residuals](../../../../diagnostics/generated/attention_round4_20261016T035217Z/evaluation/thermal_family5_saved_responses_v2/thermal_family5_saved_responses_page1.png)

Figure 5a. Two input-fixed scenes and signed transfer directions, literal e2500, original native fluid masks and coordinates. Each scene has its own reference/increment and signed-error limits, shared across D0/soft/sparse within that scene. The stored native arrays are shown without spatial interpolation. The reference is analytic-wake/shared-grid, not independent CFD.

![All saved heat-transfer response errors precision checks and material endpoint peak changes](../../../../diagnostics/generated/attention_round4_20261016T035217Z/evaluation/thermal_family5_saved_responses_v2/thermal_family5_saved_responses_page2.png)

Figure 5b. All six replay errors and material peak changes, with FP32 and FP64 arithmetic results kept separate. Under the unchanged saved strict thresholds, FP32 subtraction of endpoints fails all 18 role checks for each arm: 9,181/12,069/10,658 failed elements out of 387,616 for D0/soft/sparse. FP64 contraction and endpoint subtraction pass all 18 checks in each arm. Learned factors remain FP32; this is precise accumulation of prepared factors, not FP64 training or restored PDE accuracy.

The legacy response harness labels all three methods `other_factorization` with dense-K bytes zero. That label does not make their internal work identical. IAG additionally constructs the inherited atlas, signed carriers and receiver φ factors; D0 has its maintained factorization. Their actual preparation and application costs must be read from their own receipts. No response speedup is inferred from the zero dense-K counter. The next useful inverse evidence would require accurate native fields and validated endpoint responses first; no inverse readiness is claimed here.

## Reproducibility and bounded execution

All scientific runs use the ModularDT interpreter and physical GPUs1/2 only, each exposed by its authorized UUID as logical cuda:0. The latest explicit device instruction supersedes the plan's inherited 0/2 note. GPU0's unrelated PhyCoFlow process and all unrelated tmux sessions remain protected. Source freezes are immutable copies; contextualizer, inherited associations, typed values and field heads remain jointly trainable.

| Trajectory | Source used for newly executed epochs | Exact retained continuation command record |
|---|---|---|
| Thermal C soft | Frozen v3, fb73d18f | [Soft e1000→2500](../../../../diagnostics/generated/attention_round4_20261016T035217Z/source/thermal_soft_to2500_launch_v1.json) |
| Thermal C sparse, corrected | Frozen v5, 573bf45d; sealed-identity driver | [Corrected child e1000→2500](../../../../diagnostics/generated/attention_round4_20261016T035217Z/source/thermal_sparse_corrected_to2500_launch_v2.json) |
| Wind Q soft | Frozen v4, 5c81333a | [Q soft e1000→2500](../../../../diagnostics/generated/attention_round4_20261016T035217Z/source/wind_q_soft_to2500_launch_v1.json) |
| Wind Q sparse | Frozen v6, 9f2c00f1; declared dc442f1 validator overlay for e1000→2500 | [Q sparse continuation](../../../../diagnostics/generated/attention_round4_20261016T035217Z/source/wind_q_sparse_to2500_guard_overlay_launch_v1.json) |

The records contain executed argv, environment, storage forecasts, physical device, source/checkpoint hashes and literal final stops. The sparse children reuse their own e1–500 parent histories; those epochs are not newly executed child work. The [cost command matrix](../../../../diagnostics/generated/attention_round4_20261016T035217Z/source/round4_final_cost_commands_v5.json) binds the 14 endpoint cost/backend/memory calls and their fixed sources. Evaluation writes fresh output destinations and refuses to overwrite retained evidence.

Focused maintained-environment commands pass 26 primitive/evidence tests, the 168-test combined gate, the 181-test Q/model/native/lifecycle group, the expanded 196-test normalization group and seven continuation-guard tests. These groups overlap and are not added into a fabricated unique total. Test logs and exact argv remain in the [validation index](../../../../diagnostics/generated/attention_round4_20261016T035217Z/evaluation/round4_cost_work_training_ledger_summary_v2.json); the [completed cost/resource summary](../../../../diagnostics/generated/attention_round4_20261016T035217Z/evaluation/round4_final_cost_resource_summary_v1.json) binds all 14 final measurement outputs and ledgers. Earlier resource-summary v1 has a retained CPU/GPU classification error corrected in v2; final totals use the completed summary. A failed early system-Python NCCL import is retained and excluded from passing evidence. The final aggregator additionally rejects wrong Wind representation, normalization, runtime source and native panel, and binds saved checkpoint metadata to the exact recipe, TRAIN transform, source map and own-soft e500 lineage.

Disposable engineering uses 243 of the 400 allowed optimizer updates: 112 CPU and 131 GPU. The GPU count is 43 Thermal, 44 original Wind-C and 44 Wind-Q pilot/resume updates, with all pilot weights discarded or restored before fresh scientific initialization. The ledger's `stage_a_*` names refer to this plan's software qualification/pilot phase; the [wording clarification](../../../../diagnostics/generated/attention_round4_20261016T035217Z/receipts/round4_cost_summary_stage_a_wording_clarification_v1.json) preserves their actual updates and distinguishes them from the prohibited old frozen D-sep/Stage-A model. The failed original Thermal sparse partial adds two scientific updates and 96 case visits before the normalization repair; those are separately retained and are not hidden in engineering or inherited by the corrected child. Native evaluation, plotting and cost/memory replays perform zero optimizer updates.

Before materialization, the round declares a 12 GiB new-persistent ceiling on /data, 1 GiB on /home, 1 GiB /data temporary and 256 MiB /tmp temporary ceilings, 20 GiB minimum available space on each affected filesystem, and a separate 64 GiB RAM cache limit. Existing exact-recipe/geometry shared catalogues are reused. No dataset copy or new persistent dataset cache is created. Selected figures share a 256 MiB total budget, with 64 MiB per family and an expiring 16 MiB shared Matplotlib font-cache allowance.

## Decision, retained evidence and limits

The [sealed qualification sheet](../../../../diagnostics/generated/attention_round4_20261016T035217Z/evaluation/round4_final_research_qualification_v1.json) combines the literal-e2500 native board with paired complete costs and exact-input B8 memory. Thermal soft/sparse miss v/p means and tails; Wind soft/sparse miss all required near/downstream component means and tails. All four also exceed the 1.15 complete-latency ceiling for the memory-led route, and none obtains the named accuracy-led gain. Wind sparse passes own-soft preservation; Thermal sparse does not. Neither selected executor is faster than its identical-probability dense control on the declared high-M fixture. These are the plan's first-version research decisions, without imposing a mature-classic replacement or inverse-readiness gate. **The conditional full-TRAIN allowance is unused for both datasets.** No later run, evaluation campaign or monitoring automation is scheduled.

Unique newly executed scientific work is 9,500 completed epochs, 1,035,000 case visits and 43,000 optimizer updates, plus the retained failed Thermal partial's 96 visits/two updates. This counts the stopped Wind-C500 separately, because Q starts fresh, and counts each child's inherited 500 prefix only once. The GPU-associated process ledger totals 26,422.70 seconds, or 7.33964 process-hours (3.16795 on physical GPU1 and 4.17169 on GPU2), including setup, failures, pilots and benchmarks. Its 48 attempts comprise 34 exit-zero and 14 failed attempts; a failed admission or harness attempt is not silently relabelled scientific learning. These are sums of device-associated process lifetimes, not measured GPU busy-time or energy. The 243 engineering updates remain separately charged within 400. The first scientific GPU command is 2026-10-16 04:48:09 UTC; completion/closeout fits comfortably inside the 20 process-hour/14 elapsed-hour ceiling and its final two-hour reporting reserve.

| Retained primary endpoint | Literal checkpoint | SHA256 prefix |
|---|---|---|
| Thermal C soft | [thermal_iag_c_soft_dev e2500](../../../../diagnostics/generated/attention_round4_20261016T035217Z/runs/thermal_iag_c_soft_dev/checkpoints/epoch_2500_model.pt) | 5345ddb5a437 |
| Thermal C sparse, corrected | [thermal_iag_c_entmax15_corrected_dev e2500](../../../../diagnostics/generated/attention_round4_20261016T035217Z/runs/thermal_iag_c_entmax15_corrected_dev/checkpoints/epoch_2500_model.pt) | 4ee7ff59fac4 |
| Wind Q soft | [wind_iag_q_soft_dev e2500](../../../../diagnostics/generated/attention_round4_20261016T035217Z/runs/wind_iag_q_soft_dev/checkpoints/epoch_2500_model.pt) | c1e88c2adf87 |
| Wind Q sparse | [wind_iag_q_entmax15_dev e2500](../../../../diagnostics/generated/attention_round4_20261016T035217Z/runs/wind_iag_q_entmax15_dev/checkpoints/epoch_2500_model.pt) | 49d6a6b2b91d |

The [final evidence index](../../../../diagnostics/generated/attention_round4_20261016T035217Z/evaluation/round4_final_evidence_index_v1.json) binds all four rehashed final checkpoints and contiguous histories, all 14 completed endpoint cost/backend/memory records, selected numerical summaries, all 16 PNG companions and 15 PDF masters. It contains full hashes; the table shows prefixes only. The [final resource summary](../../../../diagnostics/generated/attention_round4_20261016T035217Z/evaluation/round4_final_cost_resource_summary_v1.json) additionally binds the full timing/work/memory tables and both ledgers; the 14 final GPU measurements consume 195.35 associated process-seconds with zero optimizer updates. Raw histories, latest/best aliases, declared 100-epoch milestones, input/query/target provenance, native arrays, controls, failed attempts and immutable source snapshots remain locally retained. The software runtime ends at pushed code revision dc442f1; the later report commit does not change training source. Exact tested continuation/evaluation/cost argv and environments remain in the command records and receipts linked above.

The scientific limits are material: one seed and one exposed segmented selection; only four fixed TRAIN cases/rows; correlated Wind directions; no unseen TEST evaluation; unequal classic exposure/objectives/normalization/ages; a prescribed regular cover with finite receiving resolution; globally contextualized payloads that prevent physical-causal attribution; incomplete generic Thermal alternate-backend gradient equivalence; FP32 saved-response cancellation failures; and no new CFD/PDE or inverse-design evidence. The failed Wind-C capacity test motivates Q but does not prove an asymptotic impossibility theorem. No revision beyond the single diagnosed Wind replacement is used to erase these misses.

## Selected figure index

Only five main figure families are retained for this report, totaling 12,231,580 payload bytes. PDF is the master format; small PNG companions provide direct Markdown embedding. Numerical arrays and original reference artifacts remain separate from presentation exports. Every displayed page was visually inspected, all 16 embedded PNGs decode, and all local report links and MathML/HTML conversion are checked. The generated evidence remains local under the ignored diagnostic root; Git contains durable source/tests/policy/report text, so a fresh clone needs the retained local artifact store to resolve these figure links.

| Family | PDF masters |
|---|---|
| 1. Native fields, aligned residuals and surfaces/cuts | [Thermal0291](../../../../diagnostics/generated/attention_round4_20261016T035217Z/evaluation/family1_thermal_native_e2500_v2/family1_thermal_0291_native_fields.pdf), [Thermal0687](../../../../diagnostics/generated/attention_round4_20261016T035217Z/evaluation/family1_thermal_native_e2500_v2/family1_thermal_0687_native_fields.pdf), [surfaces/material](../../../../diagnostics/generated/attention_round4_20261016T035217Z/evaluation/family1_thermal_native_e2500_v2/family1_thermal_native_surfaces_material.pdf), [Wind69](../../../../diagnostics/generated/attention_round4_20261016T035217Z/visualizations/wind_iag_q_e2500_native_planes_v2/W069/wind_W069_hub_native_endpoints.pdf), [Wind426](../../../../diagnostics/generated/attention_round4_20261016T035217Z/visualizations/wind_iag_q_e2500_native_planes_v2/W426/wind_W426_hub_native_endpoints.pdf) |
| 2. Learning and fixed native TRAIN/DEV checks | [Thermal](../../../../diagnostics/generated/attention_round4_20261016T035217Z/visualizations/family2_learning/family2_thermal_learning_v1.pdf), [Wind](../../../../diagnostics/generated/attention_round4_20261016T035217Z/visualizations/family2_learning/family2_wind_learning_v1.pdf) |
| 3. Actual read paths, all-head atlases and controls | [Thermal routes/K](../../../../diagnostics/generated/attention_round4_20261016T035217Z/visualizations/family3_organizer/thermal_three_query_gather_e2500.pdf), [Thermal atlas](../../../../diagnostics/generated/attention_round4_20261016T035217Z/visualizations/family3_organizer/thermal_all_head_typed_atlas_e2500.pdf), [Wind routes](../../../../diagnostics/generated/attention_round4_20261016T035217Z/visualizations/family3_organizer/wind_three_query_gather_e2500.pdf), [Wind atlas](../../../../diagnostics/generated/attention_round4_20261016T035217Z/visualizations/family3_organizer/wind_all_head_typed_atlas_e2500.pdf), [interventions](../../../../diagnostics/generated/attention_round4_20261016T035217Z/visualizations/family3_organizer/native_organizer_interventions_e2500.pdf) |
| 4. Accuracy, complete costs, backend controls and memory | [Thermal](../../../../diagnostics/generated/attention_round4_20261016T035217Z/visualizations/family4_cost_memory_v2/thermal_native_accuracy_complete_cost_memory_e2500.pdf), [Wind](../../../../diagnostics/generated/attention_round4_20261016T035217Z/visualizations/family4_cost_memory_v2/wind_native_accuracy_complete_cost_memory_e2500.pdf) |
| 5. Saved heat responses and precision limits | [Thermal, two pages](../../../../diagnostics/generated/attention_round4_20261016T035217Z/evaluation/thermal_family5_saved_responses_v2/thermal_family5_saved_responses.pdf) |

## Storage cleanup and protection

The contract records initial available capacity 96,473,116,672 bytes on /data and 44,416,684,032 bytes on /home and /tmp, with an 8 GiB forecast persistent peak below the 12 GiB hard /data limit. Existing exact-key shared caches were reused; no data copy or new persistent dataset cache was made. Final measured retention, exact removed-payload totals and remaining available capacity are recorded in the closeout receipt below.

The [protected-input verification](../../../../diagnostics/generated/attention_round4_20261016T035217Z/receipts/protected_inputs_final_continuity_verification_v2.json) matches 6,318 sealed path metadata records, rehashes 17 critical inputs/reports/anchors, verifies six links and 529 shared-cache manifests, and preserves the source/cache metadata digest. Round2/Round3 reports, historical/formal scientific checkpoints, native reference arrays and solver usage 326/326 remain unchanged; Wind TEST stays locked and new solver attempts are zero. The 72.6 GB shared cache payload and source H5 were not fully rehashed in the final continuity pass; that coverage limit is explicit. GPU0's unrelated PID 472981 retains the same executable/command/device identity, and no R4 worker remains on the GPUs.

Cleanup uses frozen exact allowlists, hashes/link/inode ownership, dependency and reader checks, then pre/post receipts. Retained old manifests continue to describe removed superseded presentation payloads; historical evidence is not rewritten. Two early cleanup records have incomplete recursive metadata: the small owned pytest109/110 trees and two py_compile files. They are disclosed in local receipts and excluded from the verified exact removed-payload subtotal. Later post-unlink receipt bookkeeping failures were recovered against their frozen inventories and independently rechecked; they did not broaden deletion scope. Same-UID process checks found no candidate readers, with cross-UID and unrelated FUSE visibility limits retained in the receipts.

The [storage/cleanup closeout receipt](../../../../diagnostics/generated/attention_round4_20261016T035217Z/receipts/round4_storage_cleanup_closeout_v1.json) records an exact verified subtotal of 461 removed regular files / 13,687,514 logical bytes across its eight listed receipt groups: owned test/cache payloads, superseded field/response/cost exports, expired font caches and one precisely inventoried compile file. It excludes the two early audit-gap groups described above. The large temporary-fixture cleanup additionally removes 71 internal fixture symlinks; the protected shared-data links are untouched. All allowlisted payloads are absent. Retained round payload is 2,774,883,137 logical bytes / 2,795,151,360 allocated bytes before the compact closeout receipt, within 12 GiB; it includes scientific checkpoints, histories, numerical evidence, selected figures, failure evidence and immutable sources. No disposable cache remains under temporary/: its sole 2,378-byte file is the explicitly protected failed-freeze pre-push source copy.

At that closeout sample, actual available capacity is 93,674,524,672 bytes on /data and 43,951,214,592 bytes on /home and /tmp (the latter share a filesystem), measured with f_bavail rather than reserved-root free blocks. Each remains above the 20 GiB floor. The final cache/export pass has a persisted PRE before deletion, verifies 26 protected hashes afterward and preserves every selected v2 output; its [receipt](../../../../diagnostics/generated/attention_round4_20261016T035217Z/receipts/round4_visualization_cleanup_v1.json) retains exact removed identities and pre/post capacity. Concurrent writers and receipt writes prevent attributing the whole filesystem delta solely to deleted payloads. Final Git closeout audits the entire outgoing range, runs the repository artifact gate, pushes only durable changes to the current non-default branch and verifies matching local/remote tips; generated data, figures and one-time helpers remain local and ignored.
