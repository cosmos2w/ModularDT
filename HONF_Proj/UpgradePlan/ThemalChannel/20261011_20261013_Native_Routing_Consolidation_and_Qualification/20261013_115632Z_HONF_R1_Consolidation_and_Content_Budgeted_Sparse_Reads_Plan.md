# HONF next phase: consolidate R1 and test useful sparse reads

**Work mode:** Codex Goal mode; bounded implementation, controlled continuation, and learned execution-policy development. **Reviewed branch:** `agent/honf-core-next`. **Reviewed commit:** `a6e898d20cf5a1ac2f60fc0d20eba41b97302ffd`. **Document status:** proposed work, not completed results. A later explicit user instruction overrides the proposed resource envelope and optional stages below.

## 0. The decision in one page

Keep the self-contained source-preserving field model. Keep one-view native routing, R1, as a conditional Wind learning direction. Keep P as the current Thermal development preference. Stop active development of R2: the second view did not improve the matched Wind priority score, worsened several Thermal quantities, and average-density collapse had little effect. Do not reinterpret this negative result as a reason to add more heads, diversity penalties, regional slots, or another field network.

The next organizing question is not whether another set of weights can be drawn as a hypergraph. It is whether the existing source-resolved computation contains a receiver-specific subset of reads that can be selected before expensive value evaluation, executed at lower real work, and verified against the complete physical output. Preserve the strong learned model while testing that question.

There are four mandatory deliverables:

1. A bounded reconciliation of the suspicious Thermal classic-reference replay, with an explicit disposition of the affected comparison board.
2. Completion of the existing Wind P/R1 pair to its original e2500 horizon, if its exact resumable states and remaining resources permit; read-only full-DEV evaluation of the already available Thermal P/R1 e2500 states. R2 is not extended.
3. A reusable active-only source-read executor and a measured, TRAIN-only sparse-read feasibility frontier that separates padding removal, learned content selection, physical error, and execution price.
4. At most one small receiver–source importance predictor per dataset, only where that frontier demonstrates useful removable work. This predictor controls execution of existing values; it does not replace or reweight the dense field model.

The primary sparse-execution hosts are **mature migrated Wind P/Run2202 e5000** and **fresh Thermal P e2500**. This deliberately tests execution on a competent retained Wind function rather than only on an undertrained development predictor. The Wind P/R1 continuation answers a separate representation-learning question. Never combine their benefits into an imaginary mature R1 checkpoint.

A shared computational core does not require both datasets to select the same preferred routing policy or the same sparse budget. One implementation with adapter-owned input/output laws and separately trained weights is the goal. Retaining dense execution in a difficult dataset or output capability is an admissible result.

## 1. What the completed round establishes

### 1.1 Predictor

The matched main board is e2000, not e2500. Thermal P/R1/R2 actually completed 2500/2500/2100; all Wind arms completed 2000. The schedule still points to the original e2500 horizon. The missing results are resource-censored, not evidence of convergence or algorithmic failure. [S1: “Comparison identities and bounded horizon.”]

| Common-e2000 result | P | R1 | R2 | Interpretation |
|---|---:|---:|---:|---|
| Thermal fluid-T mean RMSE, native units | 0.613898 | 0.623151 | 0.650975 | P is preferred on this priority field. |
| Thermal v mean RMSE, native units | 0.003482 | 0.003684 | 0.003610 | Neither routing version beats P. |
| Thermal M5 peak error, native units | 0.674542 | 0.768820 | 0.818754 | Local failures cannot be hidden by material-average gains. |
| Wind four-priority error ratio to P | 1.000000 | 0.953847 | 0.953971 | R1 gives a modest useful signal; R2 adds no priority benefit. |
| Wind near vector RMSE, m/s | 0.295884 | 0.288580 | 0.287420 | Improvement is smaller than the transverse-only headline. |

Wind's eight layout-level priority geometric means improve under both routed models, but some individual components worsen. These are eight exposed layout clusters with correlated directions and one fresh seed, not 24 independent experiments or an independent-test result. The earlier warm-start R/P/C result is supporting history, not a second fresh seed. [S1, S2.]

The available Thermal e2500 TRAIN4/DEV4 probes still show a fitting/generalization tradeoff: R1 improves TRAIN temperature while worsening the corresponding DEV temperature. This is evidence on four fixed examples, not proof of a specific overfitting mechanism across the whole training population. It does not justify another blind loss-weight or width search. [S1: supplementary e2500 probe.]

### 1.2 Organizer

R2 is implemented as intended: one 128-wide pair value, two 64-coordinate blocks, and two donor distributions. It adds 96 parameters to R1 rather than duplicating the value network. Both datasets use the actual shared implementation and TrainingEngine. The negative R2 result is therefore informative; do not “repair” it by forcing the two views to differ. [S1, S3, S4.]

The 80 complete-rebuild interventions and 16 exact restorations show model reliance and correct restoration. They do not show that the extra view improves training, that an edge is a physical causal link, or that concentrated weights eliminate any executed value rows. [S1, S2.]

Native context gathers and final source–query reads are different operations. Context donors can influence a final source state through both rounds and the global state. Omitting one final read is not deleting that source from the physical scene, the context, or all causal ancestry.

### 1.3 Physical responses

The 18 saved Thermal evaluations preserve heat-null flow and measure nonzero changes at modules whose own heat is unchanged. They recover the displayed 0291 response signs, but overpredict amplitudes. For example, the saved fluid mean changes are approximately −0.028919/+0.028914, while R1 gives −0.039107/+0.039107. These are useful but limited benchmark-response observations. The opposite transfers supply one independent direction per layout; neither K-column truth nor inverse-design validity follows. [S1: section C.]

### 1.4 Cost

Fresh comparative latency was waived because GPU2 was shared. Fitting totals reflect different horizons and changing contention. They do not establish that R2 has a generally slower kernel, or that any new arm has a qualified latency advantage. The synchronization audit already found only a modest 1–4% same-function improvement on its completed rows. Do not spend another campaign assuming scalar validation checks explain the remaining cost. [S1, S2.]

## 2. A reference inconsistency must be contained before interpretation

The latest board reports Thermal1404/1804 fluid-T mean errors of **2.68863/3.22798** on DEV22. Earlier formal reporting gives approximately **0.2933/0.2374** on canonical89 for the stated mature families. The large classic auxiliary errors in the new 0291 figure also deserve verification. These observations are not automatically reconciled by saying the current subset is harder. [S1: mature-system board; S7: formal physical tables.]

For exactly the same nonnegative per-case error and a 22-case subset of an 89-case panel,

\[
\overline E_{22}\leq\frac{89}{22}\overline E_{89}.
\]

For the earlier Dense1804 mean 0.237375, that ceiling is about 0.960290, not 3.22798. This is a **conditional consistency check**, not proof that either historical table is correct: checkpoint, target, mask, normalization, output definition, or native adapter semantics may differ. The current stencil-valid mask also has to be compared explicitly before applying the bound literally. A changed mask requires per-case SSE/count reconciliation, not a direct reuse of the mean inequality.

### Task R0 — bounded native-path reconciliation

Budget at most **60 elapsed minutes** for this task. Do not create a new global audit framework or rerun every historical model.

Use the exact stated Thermal1404 and 1804 checkpoint files and their original checkpoint-native public predictor on DEV0291 and DEV0687. Compare with the evaluator path that produced `native_references_gpu2_12/native_metrics.json`. Work from existing source data and saved arrays; no solver calls and no optimizer construction.

For each case bind and compare:

- checkpoint file/state identity, model family and local-surrogate checkpoint;
- raw physical geometry, heating, prescribed context, source IDs and presence;
- checkpoint-native global and local normalization, including mean subtraction/addition and output channel order;
- exact receiver coordinates and ordering, native fluid/stencil masks, and target frame;
- predicted ports and the local-surrogate evaluation path, without replacing either by truth;
- standardized outputs, denormalized outputs, and per-channel error SSE/count before final reduction;
- interface/material support indices, active-slot mapping, and target normalization separately from fluid T.

Check the same exact input through the native public path and the new adapter. A matching file hash or matching metadata field does not prove that the adapter performs the same physical transformation. The maintained metric helper in `tools/diagnostics/multiview_native_evidence.py` reduces supplied arrays; it does not load the historical model or prove native-predictor parity.

**Disposition:** if a narrowly identified evaluation bug is fixed, add a small native-parity regression fixture and publish a versioned correction/addendum; retain old arrays and clearly invalidate affected claims. Recompute only affected common-panel references. If the discrepancy cannot be resolved inside the bound, label the affected Thermal mature-reference board **unreconciled** and exclude it from model-selection and classic-replacement claims. Continue the internally matched P/R1 work once its own input/output bindings are sound. Do not let an unresolved historical comparison consume the sparse-development budget, and do not declare the new models dramatically superior to the Thermal classics from the suspect board.

If this check instead exposes an error in the new P/R1 prediction or target path, freeze the affected measurements and repair/requalify that path before using it as an execution host.

## 3. Preserve a stable research line and finish only useful unfinished work

### Task M1 — controlled maturation without a new portfolio

Exact-resume the existing Wind P and R1 from their own e2000 states to the original **e2500** endpoint. Preserve the original seed42, TRAIN72/DEV24 membership, normalization/calibration, observed query namespace, AdamW moments, effective/micro batch24/8, Q4096 total five-role sampling, clipping, decay and original LR schedule. There are 500 additional epochs and 1,500 updates per arm: **3,000 additional scientific updates total**.

Use the original sealed training source/configuration when strict resume requires it; do not hot-edit the code underneath these runs to insert the new sparse reader. Keep a read-only worktree or equivalent existing source-bound mechanism rather than weakening source identity. Write continuation receipts and snapshots as additions; preserve the original resource-stop record and e2000 board.

Review the first new saved milestone at e2100, then continue healthy runs at the established 100-epoch cadence. Do not require an undertrained model to beat mature full-data references at each review. Do not restart the LR schedule, insert a warm restart, add sampling categories, or extend beyond e2500.

Evaluate the already available Thermal P/R1 e2500 pair on the full existing DEV22 native board and the same M strata, not just TRAIN4/DEV4. This is read-only evaluation; no new Thermal field fit is requested. R2 remains an archived e2000 comparison and e2100 Thermal endpoint, with no further training.

If exact Wind resume is unavailable or contention makes both tails infeasible within the new envelope, retain the shared e2000 endpoint and explicitly report the blocked continuation. Do not start a replacement fresh pair merely to finish a table. This blocked tail need not prevent frozen-host executor development.

### Comparison policy

Keep three separate questions:

- **Matched mechanism:** new Wind P/R1 at the same literal endpoint; Thermal P/R1 at the same literal endpoint. Report selected states separately and use equal selection opportunities.
- **Practical system distance:** native mature references on the exact same physical panel, only after relevant reconciliation. Keep their own normalizers, native E512/E192 contexts, ages and pretraining.
- **Execution policy:** sparse versus active-dense versus legacy-dense at the **same host weights** and requested output scope.

The old requirement that all six P/R1/R2 arms reach2500 before any subsequent research is superseded for this new phase. The old unassessed gates remain unassessed historically. This is not a retroactive “pass.” Likewise, a 4.6% effect does not become a completed5% threshold, but it is legitimate evidence when deciding a new scoped experiment.

No full-data5k fit, second fresh architecture portfolio, or new physical solver run is part of this phase.

## 4. The important code-level distinction: gathering versus expensive reading

The reviewed code locations are:

| Source at a6e898d | Reviewed behavior | Use in this phase |
|---|---|---|
| `src/honf_forward_core/interface_fields/native_interaction_routing.py` | Width32 scalar/two-view scores; bound `2*tanh(raw/2)`; measure-preserving normalization; reset/reassignment semantics | Retain R1 and old R2 loading. Do not enlarge heads or change the bound. |
| `src/honf_forward_core/interface_fields/interaction_core.py::prepare_context` | Two dense MM/ME/EM rounds; scores precede values; values still run on padded/unsupported slots; explicit measures and self exclusion | Count context work and preserve it in the primary sparse-read experiment. |
| `interaction_core.py::_native_gather` | Same full128 message or two64 blocks | R2 implementation already answers the previous hypothesis. No additional view work. |
| `interaction_core.py::NonlinearFieldReadout.predict` | Every source is read nonlinearly for each receiver, then source-measure reduction and nonlinear field head | Primary candidate for meaningful sparse work. |
| `interaction_core.py::_read_selected_features` | Existing primitive gathers selected physical pairs before constructing wide fine inputs | Reuse rather than create a tree-based query directory. |
| `interaction_preserving_joint.py::predict_fields` | P/R modes dispatch to the maintained nonlinear reader; collective paths remain optional | Keep one shared physical reader and historical modes intact. |
| `interaction_preserving_joint.py::prepare_receivers` | Separate affine-source features and K preparation | Leave Thermal affine reads dense in the primary experiment. |
| `tools/diagnostics/multiview_native_evidence.py` | Physical reducers and declared cache-binding checks, not historical checkpoint loading | Reuse reducers; add native-path parity checks at the actual evaluator boundary. |
| Dataset `training/joint_task.py`, `tools/joint_regional_train.py`, `src/honf_runtime/unified_training.py` | Shared engine with dataset-owned laws, objectives, sampler and strict fresh/resume identities | Reuse for continuation and a small auxiliary read-policy task. No separate ad-hoc field trainer. |

For Wind M30, the report records **2,760 context value rows** across the two rounds. At Q8192 the final nonlinear reader has **245,760 source–query rows**. This row comparison is not a FLOP or latency estimate, but it shows why pruning only context weights cannot be advertised as eliminating most of the computation. Instrument layer-level work before making a total-cost claim. Thermal additionally has distinct affine and stencil work. [S1, S4, S5.]

The current score law also limits a naive high-mass cut. For N equal-base-weight donors, scores in [−2,2] give a largest/smallest exponential-weight ratio of at most exp(4). Even in the most favorable extreme, k retained donors carry at most

\[
\frac{k e^4}{k e^4+N-k}
\]

of normalized mass. A99% mass cover thus requires at least

\[
\left\lceil \frac{0.99N}{e^4(1-0.99)+0.99}\right\rceil
\]

donors:19 of29 or124 of192. This is an analytical bound under equal base measures, not a measurement of trained support; unequal measures require the actual weighted calculation. It explains why99% mass coverage should not be treated as a promise of a tiny graph. Nor does1% omitted attention mass imply1% physical-output error.

Do not increase score range, force top-K, penalize entropy, or retrain the predictor merely to evade that observation.

## 5. The organizing object to build

Use a **native receiver read plan**, not a new latent group bank. Its meaning is: “for this physical receiver, these source-conditioned value evaluations are retained by this execution policy.” A receiver and its selected donors form a directed receiver-anchored hyperedge representation, but the implemented operator is still a selective reduction of source-resolved messages. Do not claim an independently identified higher-order physical interaction.

A plan should carry the minimum needed information through the existing prepared-state machinery:

- host/model and policy identity and generation;
- physical source IDs, receiver IDs/coordinates, native order and measures;
- admitted pair indices/mask and any mandatory pairs;
- original reduction weights and selected-read scores, with distinct names;
- retained count, active capacity, padding count, chosen request/threshold and dense-dispatch status;
- input/configuration/parameter versions and output capability scope.

Do not create a second provenance service, global registry, permission hierarchy or persistent error database. Extend the existing prepared-state conventions. A diagnostic trace can be detailed, but the public hot path should not materialize it when unused.

The policy sees only legal native inputs, current encoded source/global states, receiver geometry and the explicit compute request. No target, error map, case-ID embedding, physical solution, or held response enters inference. Physical IDs label and align records; they are not numerical learned features.

## 6. Stage E0 — active-only execution before learned omission

Implement an opt-in execution path for the existing nonlinear final source reader that gathers only active physical source–query pairs, evaluates the unchanged MLP on those rows, and reduces with the original source weights in the original physical identity space. Use `_read_selected_features` or a minimal extension. Keep the default legacy path unchanged.

The first control removes only absent/padded contributions. It does not remove an active source, change a measure denominator, normalize over the retained subset, change the source/global context, or introduce new trainable field parameters. Preserve bias handling and nonlinear order.

Compare:

1. **Legacy dense:** original padded computation.
2. **Active dense:** every legal active source read, with avoidable padded reads removed.
3. **Selected active:** a declared subset of those same active pairs.

Only3 versus2 measures active-read omission. A Thermal low-M reduction from padded12 to active3 is not a learned75% sparsity gain. Wind's existing active source dimension already avoids much of that padding; do not invent an equivalent Wind gain.

For all-active coverage, a direct legacy dispatch is allowed and should be explicitly counted. For partial coverage, gather before expensive value evaluation. “Compute every value and mask afterward” is a dense-masked reference, not a sparse executor.

Use vectorized selection and segment/scatter reduction; no Python loop per receiver or per donor, no recursive receiver tree, and no CPU round trip for each decision. Fixed source-ID order and tie handling must be reproducible. Batch size, query tiling and diagnostic flags must not change which physical pairs are selected for the same query.

### Numerical and interface checks

Use the existing same-function numerical tolerances, with CPU FP64 and native GPU FP32 low/high-M coverage, all required outputs, query/source VJPs, and at least one disposable optimizer-boundary check where the executor is proposed for training. This phase primarily qualifies inference; do not claim sparse-training acceleration from inference tests.

Keep same-function numerical qualification separate from deliberate approximation budgets. Do not widen an inherited parity tolerance to admit a different execution path. If a reordered active-dense path misses a strict derived-omega/proxy criterion, retain the failure and legacy default, investigate one bounded implementation remedy, and keep any unqualified executor experimental.

Thermal affine K, surface/material temperature and q paths remain dense and unchanged in this primary experiment. Check exact K/temperature preservation at fixed weights and the original heat-null flow law. This prevents the sparse investigation from purchasing a lower cost by sacrificing the known thermal response contract.

## 7. Stage E1 — measure sparsifiability before fitting a selector

### Hosts and frozen populations

Use the self-contained migrated Wind2202/P e5000 for the primary Wind execution study. Strict-load the exact preserved artifact and respect its E8 input contract and original normalization. Its archival moments are not a resumable P optimizer: do not use them for field training. Use Thermal P's completed fresh e2500 state if its own prediction path is qualified. Keep the P/R1 maturation results on a separate board.

Select eight TRAIN scenes per dataset from input metadata before reading new sparse outcomes. Cover low/high M and available intermediate categories, plus prescribed context/direction diversity. Use the already authorized primary TRAIN inventory; do not expand to all600/all420. Use Q1024 TOTAL requested receivers initially, with the full native stencil/auxiliary expansion recorded. No outcome-selected scene removal is allowed.

### A content diagnostic rather than an attention picture

For the nonlinear final reader,

\[
c(q)=\sum_i\mu_i m(q,i),\qquad
\widehat y(q)=\Psi(c(q),z_g,\phi(q),f_q).
\]

Compute and retain the actual measure-weighted message `mu_i*m(q,i)`. Then measure the **single-read deletion effect** through the actual nonlinear head at fixed context:

\[
d_i(q)=\Psi(c(q),\ldots)-\Psi(c(q)-\mu_i m(q,i),\ldots).
\]

This deletes one read, not one physical module. It is cheap to calculate in a vectorized diagnostic once the dense messages are available, but it is not free and must be counted as oracle/calibration work. Apply the dataset's physical output conversion. For a dimensionless ranking, use fixed TRAIN-owned component scales and a declared norm; retain the individual physical components in the evidence. Do not silently use per-case reference amplitudes or DEV-derived scales.

Compare two diagnostic rankings: measure-weighted message norm and this decoder-aware single-read deletion magnitude. If deleting several reads causes cancellation or nonlinear amplification, the combined endpoint decides; the sum of single-read effects is **not** an error bound.

For each ranking, construct declared active retention fractions1.00/0.90/0.75/0.60/0.50, with mandatory pairs added afterward and actual counts recorded. Retention fractions are diagnostic workload probes, not a learned-K claim. Recompute complete native outputs for each proposed support, including Thermal curl from its required neighbor values. The dense model and physical targets remain separate references.

At the same actual per-receiver active count and same mandatory pairs, compare an input-only nearest-source control and a deterministic source-ID permutation control. Add upstream geometry only where the dataset adapter explicitly defines the frame; do not import Wind's directional convention into a universal core.

No new selector is trained unless a useful frontier exists. The proposed feasibility requirement is: at least **30% fewer active final value rows** across the input-fixed oracle panel, while preserving each priority physical-error mean within2% and relevant case/row tail within5% of the host, with no>10% individual-scene regression beyond a declared numerical floor. Show all output channels, not only the selector's norm. These are prospective experiment criteria, not universal physical tolerances or inverse-readiness bounds.

A strong host is essential to interpreting this result: retaining a weak teacher does not establish physical adequacy. Report the host's existing absolute reference errors beside every distortion/work result.

### Bounded context probe

On two of the same TRAIN scenes per dataset, optionally show R1 context mass covers at95%/99% and their actual retained weighted content. This is a small explanatory comparison, not another learned context-executor project. Keep all context routes dense in the primary final-read study. If they represent little total work, say so and move on.

### Branch decision

If neither diagnostic ranking yields useful heavy-read reduction under the physical guards, do not fit a selector to impossible labels and do not force lower K. Deliver the active-only executor and the measured frontier. The resulting next hypothesis would concern how to learn a different compressible representation; it is not licensed as an additional architecture in this round.

If a useful frontier exists for only Wind, continue only Wind policy fitting. The common implementation still supports Thermal and its measured full-access outcome. Do not force symmetry between dataset outcomes.

## 8. Stage E2 — one small learned read-importance policy

### Architecture and training scope

For each dataset passing E1, fit one width32 pair scorer with a single nonnegative importance output. Reuse the existing native pair features or a shared feature-construction helper: encoded physical source state, global context, signed receiver–source geometry, source length/domain normalization, and existing receiver features. Spatial dimension and widths are configuration-owned. Do not branch the common scorer on dataset name or on output channel labels.

The field model, encoders, message values, context router and physical heads are frozen only for this **execution-policy calibration**. This is not a return to frozen D-sep plus a separately trained thermal field. The underlying predictor remains one self-contained jointly trained field model, and its dense predictions must stay byte-identical throughout scorer fitting.

Train the small scorer against the existing model's decoder-aware single-read deletion magnitudes on TRAIN. These are model-derived importance labels, not independent physical causal labels. Use a declared stable log-scale regression or equivalent positive-score loss; choose the formula once from TRAIN statistics, not a portfolio of loss functions. A numerical floor for near-zero targets comes from TRAIN and is recorded.

Use an input-only internal TRAIN partition for fitting and calibration: normally120/30 Thermal cases and18/6 Wind layouts, preserving all directions together and covering available M strata. These partitions are inside the already authorized source TRAIN inventory and do not change the field model's original training membership. The existing field host may have seen calibration layouts during field learning; disclose this when interpreting the new policy. No DEV target selects training examples or margins.

Budget at most **600 scorer optimizer updates per qualified dataset**, review at100,300,600, and stop earlier for a clear inability to learn the importance ordering. Use the existing TrainingEngine with an auxiliary read-policy task/provider, fresh scorer-only optimizer state, deterministic native query streams and explicit host identity. Do not implement a second field trainer. Stream labels or use bounded reusable memory, rather than saving Q×M×H tensors for every case/epoch. Record teacher forward/head-deletion work as well as scorer updates.

### Input-only selection

A complete request obtains all cheap importance scores before expensive selected value evaluations. Rank scores within a receiver, apply the chosen cumulative predicted-importance coverage from the fixed grid **0.90/0.95/0.99/0.995/1.00**, and add mandatory near reads supplied by the adapter's existing geometry rule. Use the same rule in oracle and learned comparisons. If the existing near definition is unavailable, document a single input-defined rule before E1 rather than inventing a physics prior after errors are seen.

The final source aggregation retains the original source measures. **Do not renormalize the source measure over selected donors.** R1's context-weight normalization and final-read source omission are different operations. Scalar importance is only a selection signal; it is not multiplied into the retained messages and does not change the dense field function.

Choose one operating threshold per dataset on the internal TRAIN calibration panel using the few fixed coverage options, full-output distortion and physical guards. Deploy only that frozen threshold to DEV. If scores are invalid, the sum is uninformative, a request is unsupported, or the proposed retained fraction exceeds a predeclared break-even dispatch level, use the active-dense path and count that as a dense dispatch. Suggested starting break-even level is85% active retention; it may be fixed from TRAIN timing or a layer-work model before DEV evaluation, not retuned to hide DEV failures.

Do not use physical reference error at inference to decide whether an individual DEV case receives sparse or full evaluation. Do not pay for a hidden dense forward to certify each sparse forward. Offline calibration and final validation are permitted; an online teacher-dependent safeguard is a different, fully charged algorithm and is out of scope.

### Scope of prediction and derivatives

The first implementation sparsifies only the existing **nonlinear final source read**. Thermal affine K construction/application stays dense. Wind retains its nonlinear velocity head and physical normalizer/profile. Context/global paths remain full and visible.

The same physical query must get the same selected support irrespective of evaluator role labels, chunk boundaries, duplicated query occurrence or whether it is needed by a Thermal curl stencil. Wind role IDs organize the loss/evaluation; do not create five inconsistent fields at the same coordinate by feeding evaluation-role names into the policy.

A hard selection is piecewise smooth. Fixed-support derivatives can be checked, but they do not establish derivatives through support switches or geometry-response fidelity. Dense smooth execution remains the default for inverse/geometry derivative consumers. A changed host, normalization, policy, receiver set or scene must reject the stale plan. No force-map trick, current-heat input or geometry-detached fake derivative is allowed.

### Optional implementation simplification, not a new experiment

If the decoder-aware targets cannot be predicted better than the declared message-norm or geometry control on TRAIN calibration, preserve that result and stop the learned policy. Do not add more heads or train the physical model to make the selector look successful. A simpler control that is equally accurate at equal work is the preferred execution policy, with no unique learned-organizer claim.

## 9. Stage E3 — complete physical, work and execution assessment

Freeze host, scorer, threshold, mandatory-pair rule and dispatch rule before new DEV sparse outcomes. Evaluate all existing DEV22 or DEV24 queries with the exact established supports. Keep the same native field, masks, TRAIN-owned scales and physical reducers for dense, selected and matched-count controls.

Report four quantities independently:

1. **Dense-host physical error** against the stored reference.
2. **Sparse physical error** against that same reference.
3. **Sparse-minus-dense model distortion**, including field maps and tails.
4. **Executed work and complete price**, including all selector and fallback costs.

The same-weight executive test is not an accuracy-learning win. If omission improves a particular reference metric, report it without selecting that policy retrospectively from DEV or claiming causal discovery.

### Required numerical work counters

Count source/context encoders, all six context value networks, score projections, selected index formation, final source MLP rows, field-head rows, affine K rows, auxiliary/stencil rows, padded/self-excluded rows, CPU transfers and dense dispatch. Forward hooks must show that omitted active final values were not evaluated. A rectangular union that recomputes many excluded rows must report its executed rows rather than the ideal mask count.

The useful work target is at least30% fewer **active final value rows**, with selector FLOPs included in the total layer-work account. This does not by itself imply30% whole-model savings. Report realized counts separately at low/high M and Q1024/Q8192, and keep dense dispatch in aggregate totals.

### Latency and memory

The previous user-authorized shared-GPU latency waiver remains applicable when the device is still contended. Do not stop unrelated jobs, seize a new GPU, or reinterpret variable shared-GPU observations as controlled prices. If a quiet authorized window is available, run a small interleaved matrix: two TRAIN scenes per dataset, Q1024/Q8192, legacy/active-dense/selected, two warmups and five measured repeats. Reduce scope rather than spend the campaign waiting for a timing window.

Measure complete CPU-input→context→scores/selection→value reads→physical outputs→CPU-return time. Prepared repeated-query time is a second, explicitly named workload and is not substituted for cold complete calls. Report median and full trial range, private/incremental allocated and reserved CUDA memory with their baselines, and the scorer's extra parameter bytes. Never compare sparse prepared decode against dense cold construction.

A prospective **hardware-acceleration** result requires approximately10% lower median complete high-M/Q8192 time than active dense under the same valid physical guards, with no>10% low-Q regression unless the predeclared dense dispatch avoids it. A smaller measured change is reported numerically, not inflated into a strong acceleration claim. If latency is waived, the result can be **physical/work-qualified; latency unassessed**. It is not an execution failure merely because no uncontended timing was authorized.

### Limited response/interface replay

For Thermal, verify the unchanged dense affine K and temperature/interface/material outputs on the three previously bound heat-transfer layouts. Reuse the six saved signed pairs; do not solve again. Sparse nonlinear flow must remain exactly heat-independent. Check source IDs, precise increments, endpoint-derived peak changes and stale rejection at the declared precisions. The sparse flow policy must not silently alter the temperature path through shared prepared-state mutation.

No new inverse generator, optimization trajectory, geometry-reference campaign, Wind turbine-displacement claim or increase in solver326/326 is authorized. If current source records cannot bind a physical comparison, report unavailable rather than substituting nominal arrays or unrelated references.

## 10. How to interpret the result as HONF development

There are three distinct possible organizing benefits:

- **Learning benefit:** R1 changes how full context is gathered and improves the matched field fit. The Wind continuation addresses this; Thermal may remain neutral/P-preferred.
- **Information-use meaning:** an actual read or gather has a measured downstream output effect, with IDs, units and correct restoration. The established interface and new deletion-effect labels address this.
- **Execution benefit:** a cheap input-only policy avoids expensive active reads while preserving physical output and reducing work, potentially latency. E0–E3 address this.

A method need not win all three in the same round. A source-preserving model with useful learned execution but no field-accuracy gain is a legitimate result. Conversely, R1's learning gain does not confer sparse capability automatically.

Use “directed receiver-anchored computational interaction” for the new read plan. Do not advertise latent physical groups that are not present, independently validated Green functions, a discovered physical causal hypergraph, or a new higher-order operator solely because a drawing has several arrows.

Do not use entropy, view separation, visually sparse arrows, smaller logicalK, or a tally of passed tests as the scientific objective. The decisive quantities are physical fields/responses, actual avoided expensive work, and complete measured price.

## 11. Decision rules and stop conditions

| Decision | Required evidence | What happens if not met |
|---|---|---|
| R2 remains active? | Not proposed; existing evidence does not justify it. | Preserve and stop. |
| R1 is a useful Wind learning direction? | Matched endpoint mean/tail/layout results and unchanged objective/streams; report effect size without rewriting the earlier gate. | Retain P and the complete negative/neutral R1 result. |
| Thermal classic board is interpretable? | Native predictor/adapter/target/normalizer reconciliation. | Quarantine affected board; do not block sound within-family research. |
| Useful reads can be omitted? | TRAIN oracle meets explicit physical guards at≥30% active final-row reduction. | Do not fit a selector to this host; retain exact execution improvements. |
| A learned selector adds value? | Frozen TRAIN-selected policy passes DEV physical/work checks and is preferable to same-count geometry/simple controls. | Prefer the simpler policy or dense execution. |
| Sparse executor is real? | Gather occurs before fineMLP, hooks confirm row reduction, no hidden full pass, correct fallback accounting. | Label masks diagnostic only. |
| Hardware acceleration is qualified? | Same-host complete-call benchmark in an authorized quiet window. | Report work/memory only; latency unassessed or measured slower. |
| Research5k launch is justified? | Stable field recipe and a useful scientific question with honest cost/risk context. It does not require R2 maturation, sparse success or classic replacement. | Deliver the exact next experiment recommendation; do not launch automatically. |
| Inverse ready? | Not targeted by this phase; independent physical and design validation is missing. | Dense derivative default; no readiness claim. |

Do not co-adapt the field model to sparse reads in this round. That would be a different learning experiment needing its own matched dense continuation, restoration of both optimizer states, and exposure accounting. This phase first determines whether that investment has a credible execution opportunity.

## 12. Resource, ownership and scheduling

Proposed ceiling: **8 aggregate GPU-associated hours and10 elapsed hours**, final2 elapsed hours reserved for report, validation, preservation and cleanup. Time starts at the first substantive local inspection. Honor any tighter user instruction. These are ceilings, not instructions to fill the allocation. Use the currently authorized device; do not assume that previously named GPUs are idle or that free VRAM implies exclusive compute.

Scientific optimizer caps:3,000 Wind continuation updates plus at most600 scorer updates per dataset, for **4,200 maximum scientific updates**. No new Thermal field updates, R2 extension, new seeds or full-data field fits. Allow up to64 separately counted disposable numerical/optimizer checks; most execution tests require no optimizer.

Allocate work approximately as follows: R0≤1hour; maturation and E0 may proceed independently on source-separated paths after sound host binding; E1≤1hour of active investigation per dataset; E2/E3 consume the remainder only when their upstream feasibility criteria pass. Forecast from actual complete epoch/fit times and changing contention before starting each bounded block. Never start a large tail assuming the best observed epoch time will persist.

Keep field continuation ahead of optional visual or timing portfolios. If resources tighten, drop optional context probes and unpromising scorer fitting first. Preserve a complete readable report of fewer decisive experiments rather than another60-page package accompanying censored primary work.

Before allocating artifacts, inspect current/data and/home capacity. Do not assume the previous103.1/39.8GiB figures remain current. Set a proposed additional disk cap of **8GiB under/data and1GiB under/home**, with at least20GiB free headroom on each affected filesystem where available. Reduce materialization if those limits cannot be met. Reuse verified shared caches and source arrays; no per-arm dataset or geometry copies, and no permanent dense Q×M×H cache. Retain required100-epoch milestones, actual scientific states, failure evidence and metric/provenance arrays.

Apply `AGENTS.md` cleanup and entire-outgoing-range audit requirements. Do not delete scripts/source, historical checkpoints, the only copy of a reusable cache, raw physical references or declared scientific failures. The protected inventory includes67 already-absent observations; preservation means no new mismatch, not953 present files. Solver326/326 and all data locks remain unchanged.

## 13. Required report and intuitive evidence

Produce a new main report and compact evidence index; preserve the previous multiview report and add only an explicit correction link if R0 requires one. The report begins with what changed and the measured Predictor/Organizer/Inverse gains, misses and next step. Keep A/B/C meanings distinct from the comparison boards.

Use at most **five main figure families and roughly8–10 embedded panels/pages**, with additional necessary arrays in the evidence index rather than a large visual portfolio:

1. **Stable learning and current accuracy:** Wind P/R1 actual continuation; Thermal P/R1 actual e2500 fields/tails. Show native units and the literal comparison age. Mark any unreconciled classic row rather than plotting it as established truth.
2. **Which read matters here?** One low/high-M Wind scene and the Thermal challenge: physical receiver/source positions, current context weights, exact single-read deletion effect in native outputs, and retained/omitted final reads. These are different quantities, each labelled.
3. **Can the model be approximated cheaply?** Physical error and model distortion against actual active value rows for dense, oracle, learned and same-count simple controls, including dense-dispatch points. Include one sparse-minus-dense map so smooth global plots cannot hide a local wake/curl artifact.
4. **What actually executes?** Legacy padding, active dense, selected active, scorer/index work, total layer-work proxy, allocated memory, and complete timing only when valid. No ideal-mask bar standing in for executed rows.
5. **Preserved responses and remaining limits:** concise heat-null/unchanged-K evidence, the saved signed mean and nonlinear peak response, and the explicit absence of new geometry/design validation.

Keep physical field/error scales common within a meaningful channel, retain all unclipped metric values, and state mask/support definitions. Show spatial native fields rather than only index-space heatmaps when the coordinate binding is known. Where interface row-to-module correspondence is genuinely unavailable, label the index-space display explicitly and do not fabricate coordinates.

A single physical temperature/error label should not contain long machine-precision decimals. Keep essential numerical labels readable. Top-k display thinning, if needed for readability, must be distinguished from the complete executed selection, with full data retained. Write prose/captions on continuous source lines per the repository rule. Keep generated PDFs/PNGs and one-time renderers ignored.

## 14. Durable deliverables and closeout

Deliver the maintained selected-read primitive, optional small scorer/policy task, strict policy/host bindings, focused tests and a short usage guide using real supported commands. Extend existing interfaces rather than creating parallel trainer or model registries. Keep historical scalar/two-view loading, migration, fresh and architecture-child guards strict.

Provide an executable exact-resume handoff for the existing Wind pair and a stable dense-default inference entrypoint. For any qualified policy, document the host checkpoint, native scope, sparse request, dense-dispatch behavior and derivative limitations. An unsuccessful dataset gets an explicit full-access default, not a silently different fallback model.

Report attained updates/epochs, actual GPU-associated time and elapsed time, work and memory, and all skipped stages with reasons. Count failed starts and disposable updates; do not add nested timers twice. Report files/bytes removed and retained, final free capacity and unchanged protected observations.

Commit and push durable code/tests/guides/reports to the non-default branch after auditing the entire outgoing range and running the repository pre-push checks. Do not push generated evidence or one-time campaign helpers. Verify the remote and local tips match. Completion means the bounded experiment has been executed and honestly interpreted, not that every research hypothesis succeeded.

## 15. Reviewed sources and evidence boundaries

The planning review read the attached main report, fetched the companion evidence index and relevant current code through GitHub, and verified the remote branch tip at a6e898d. Ignored local checkpoints, arrays and one-time evaluator implementations were not independently rerun in that review; the available Desktop Commander device was offline. Codex must verify local bindings at execution time. No new GPU training, physical solver call or numerical reproduction is claimed by this plan.

- **S1:** `docs/reports/_bk/20261013_035603Z_HONF_Multiview_Native_Routing_20261012.md`, especially common-e2000 tables, same-panel mature-system distance, sections A/B/C, execution audit, decisions and closeout.
- **S2:** `docs/reports/_bk/20261013_035603Z_HONF_Multiview_Native_Evidence_Index_20261012.md`, run identities, per-layout Wind results, native-reference and memory/work receipts, saved-response manifests, adapter failures and repairs.
- **S3:** `src/honf_forward_core/interface_fields/native_interaction_routing.py` at a6e898d: scalar/two-view initialization, score law, density normalization and interventions.
- **S4:** `src/honf_forward_core/interface_fields/interaction_core.py` at a6e898d: native gather/prepare, selected-pair features, nonlinear final reader, finite checks and prepared ownership.
- **S5:** `src/honf_forward_core/interface_fields/interaction_preserving_joint.py` at a6e898d: direct dispatch, separate affine-source head, receiver preparation, forcing/increment application.
- **S6:** `tools/diagnostics/multiview_native_evidence.py`, `docs/guides/HONF_Fresh_Matched_Multiview_Comparison.md`, and the relevant Thermal/Wind joint training providers at a6e898d.
- **S7:** `docs/reports/_bk/20261010_154719Z_HONF_Joint_Formal_Comparison_20261010.md` and `docs/reports/_bk/20261008_004514Z_HONF_Unified_Training_and_Adaptive_Interaction_Refinement_Report.md`: historical mature checkpoint identities, native output definitions and canonical89/fullVALID comparison tables. Their scope is historical context until the current Thermal reference discrepancy is reconciled.
- **S8:** `AGENTS.md` at a6e898d: development membership, reporting, artifact storage, cleanup and upload rules.

All thresholds and stage choices in this document are proposed prospective decisions for the next phase. All mathematical observations are deductions under their stated assumptions. Neither is an already measured result.
