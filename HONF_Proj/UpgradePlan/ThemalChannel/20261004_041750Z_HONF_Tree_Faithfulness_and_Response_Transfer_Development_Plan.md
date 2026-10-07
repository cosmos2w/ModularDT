# HONF Tree development: measure-consistent organization and response-faithful reuse

## 0. Goal, scope, and the decision this round must support

**Central project goal:** learn a hypergraph-inspired organization of module–environment interactions through multi-field reconstruction, then reuse that representation in flexible inverse design. Better field accuracy is supporting evidence; it is not, by itself, evidence of a useful organizer.

**This round's question:** can we make the promising Tree computation insensitive to equivalent source discretizations, make its exported dependencies match the computation that produces its controls, and demonstrate value against a separately trained non-hypergraph control?

This is an implementation-and-learning round. It must deliver new, actually trained development models, not only another audit of Run2203. It is not a formal-training campaign, a second five-architecture sweep, or a diffusion-model project.

Reviewed repository: `cosmos2w/ModularDT`, branch `agent/honf-core-next`, HEAD `378de02b65e8c0200627102ae1aef1ba20089eb4`. The scientific 500-epoch closeout is `fb6f598`; the later commit installs the subset development procedure. Re-read current `AGENTS.md` before executing. If HEAD has moved, inspect the relevant changes and record the actual base rather than enforcing an obsolete commit equality.

### Concrete deliverables

1. An opt-in **Tree-F** (faithful Tree) that repairs representation dependence and separates global planning information from membership-local control content.
2. A competent, independently trained **Pair-F** control with the same fine physical backbone, local feature semantics, and training data, but no collective hyperedge state.
3. A fresh **Tree-L** legacy-Tree development control, not a resumed full-data model.
4. Matched 100-epoch results on the established 150/22 subset, with a documented review. An optional matched extension of Tree-F and Pair-F to **at most 500 total development epochs per candidate** is permitted only under Section 10. Tree-L stops at 100.
5. Direct evidence for A/B/C below, including a small completed multi-step inverse comparison, and an explicit statement of missing physical response labels.
6. Reusable source/tests/configurations, a readable report, and preparation instructions for the *next development review*. Do not launch or recommend a formal 5,000-epoch job merely because a test passes.
7. A separately authorized fresh **Dense-D25** baseline, using the Run1804 dense architecture and maintained native configuration on the identical fixed25_v1 subset, through **1,000 development epochs**. Save milestones/latest/best-field and plots every 100. It is a fourth, explicitly requested comparator, not permission for another portfolio. Do not initialize it from Run1804 or change that historical run.

**Execution amendment, 2026-10-04:** the user raised the original 200-epoch candidate cap to 500 because development now uses the fixed quarter subset, and explicitly requested the fresh 1,000-epoch dense baseline. All other solver, data, security and artifact boundaries remain in force. Implementation and monitoring use GPT-6.1 Sol high subagents; the main agent reviews math, integration and scientific decisions. Actual repository base: `378de02b65e8c0200627102ae1aef1ba20089eb4`.

**Completed execution, 2026-10-04:** Tree-L3103 completed100; the reviewed matched Tree-F3204/Pair-F3202 extension completed200 and stopped there; fresh Dense3101 completed1,000. The final200 review retains thermal predictor gains and faithful rebuilt/conditional dependency evidence, but stops further unchanged training because known-null response transfer misses and positive-response labels remain absent. The bounded32-trail inverse comparison completed at those same200 weights. The optional500 ceiling was not automatically spent. See the [measured closeout report](../../docs/reports/_bk/20261004_091519Z_HONF_Tree_Faithfulness_Fixed25_Development_Report.md) for A/B/C, Predictor/Organizer/Inverse, contention, measured work and the six selected local figures. No new solves, Wind training, full-data expansion or formal run occurred.

## 1. What the completed campaign does and does not establish

The source report is `docs/reports/HONF_Shared_Core_Thermal_500_Epoch_Conclusions.md`.

- All five matched arms completed 500 genuine full-data epochs. Tree was the strongest new H arm on the main thermal quantities, but Native and Fine controls retained stronger principal physical errors. Mature Run1804 is an unequal-age reference, not a same-age control.
- On four inspected cases, normal Tree outperformed fixed-access zero-control and full-access retained-control interventions on all 24 equal-case role means. This establishes **dependence on the trained organization at those weights**. It does not establish that a separately trained simpler model could not do equally well.
- Tree's rectangular subset reader executed substantially fewer fine-MLP rows, but all measured complete-wrapper subset contrasts were slower. Preserve the row reduction; do not call it a latency win.
- Fixed-prepared environmental atom splitting passed; rebuilding the organizer after equivalent splitting failed all four reported case/parent checks. A representation-invariant fine integral does not imply a representation-invariant learned operator.
- All arms introduced nonzero flow/pressure changes on stored fixed-geometry heat-transfer controls whose reference changes were zero.
- The matched 500-stage inverse panel used one start and one update. Tree improved observed residuals but worsened held-sensor residuals. Those measurements establish a working interface, not inverse quality.

These findings motivate a **faithfulness and transfer experiment**, not another search for a slightly lower sampled field MSE.

### A/B/C evidence must remain separate

| Requirement | Evidence sought in this round | Insufficient substitute |
|---|---|---|
| A. Added value | Trained Tree-F versus trained Pair-F; same-weight, work-aware interventions; held-observation behavior | Large degradation when a specialized network is ablated |
| B. Faithful interpretation | Whole-rebuild atom-refinement invariance; disclosed value/control/planning dependencies; phase-correct path tests | A binary membership map, fixed-prepared parity, or a single K count |
| C. Response transfer | Heat-null behavior on unseen-to-current-fit layouts; any available subset-aligned nonzero reference responses; held-sensor inverse checks | Fitting the sole available training response anchor, or surrogate self-consistency alone |

## 2. Standing development rules — carry these into future plans

Use the existing procedure in `docs/guides/Thermal_Model_Development_Protocol.md`, not the old full-data campaign defaults.

- Manifest: `/data/wanglz/ModularDT/thermal_development/fixed25_v1/manifest.json`.
- Exactly **150 selected training cases and 22 selected validation cases**. Do not rebuild, resample, or enlarge these memberships.
- Selection seed is `20261004`; initialization/query seed is `0` unless a separately identified replication is authorized.
- Global normalization is fitted on the selected 150 training cases only. The frozen Stage-A surrogate and its own normalization remain unchanged.
- Matched case order, query stream, effective batch, precision, losses applicable to the comparison, and absolute development schedule. Record intentional architecture-specific terms rather than implying that their entire scalar objectives are identical.
- Default native settings remain Q1024, FP32, effective batch 48, and the existing absolute **1,000-epoch development schedule**. Stopping at 100, 200 or 500 must not rescale the schedule.
- A normal development epoch has 150 unique visits, 19 microbatches, four optimizer boundaries, and 153,600 primary fluid queries. Count auxiliary forwards separately. Do not relabel optimizer updates as epochs.
- Monitoring checkpoints, latest/best-field updates and plots occur at **100-epoch boundaries**. At 100 there is only one eligible monitoring checkpoint. Candidate selection at 500 considers e100/e200/e300/e400/e500; dense selection at 1,000 considers its ten declared milestones. Unsaved favorable intermediate epochs are ineligible.
- Scalar training telemetry may remain per epoch. Do not generate checkpoint copies or plots every 10/25 epochs. An emergency interruption save is a recovery artifact, not a new model-selection opportunity.
- Statistical forward evaluation uses **all 22 cases**. Detailed field and graph arrays use four fixed input-selected representatives spanning the available validation M values. Extra interventions use only an explicit bounded panel.
- Full-data training and full-data statistical evaluation remain separate, explicit formal actions. Historical references keep their own original normalization and training lineage.
- Wind scientific training stays paused. A small shared-core 3-D compatibility test is allowed, not a Wind experiment.

**Important coverage fact:** selected training has only four cases each at M=3,5,7,10, whereas the 22 validation cases all belong to those four categories. Report these counts. Do not claim that validation measures every training M category.

**Important response fact:** the fixed subset intersects the existing eight training-atlas anchors at **0348 only**. Do not add excluded atlas families to training. Reusing the old four Re90 neighborhoods is not a new held-layout test, and default development readers must not be bypassed to obtain them.

This plan uses the first 100 epochs with the existing reconstruction objective and no atlas response-loss callback. Section 9 adds a clearly identified benchmark-null auxiliary objective only in the optional **101–500** paired extension. The existing common objective amendment and selected-anchor response callback remain shared between those two arms; no silent schedule replacement is authorized. Dense-D25 uses the maintained ordinary dense continuation objective and 0348 callback after 100, without the new null auxiliary term; report that difference in comparisons after e100.

## 3. Source-level diagnoses to verify once, then act on

### 3.1 The reader is measure-aware; the receiver index is not fully so

`adaptive_receiver_hypergraph.py` uses weighted source pooling and measure-aware membership densities. Nevertheless, `receiver_tree_access.py::build_receiver_tree_geometry` explicitly uses **unweighted child-coordinate means** for split boundaries. `CaseLocalReceiverTree.build` chooses cuts in a sequence of individual anchor rows and can separate duplicate/coincident atoms.

For example, coordinates 0 and 2 with masses 1 and 1 have centroid 1. Replacing the atom at 0 by two identical atoms of mass 0.5 leaves the physical measure unchanged, but the unweighted mean becomes 2/3. A boundary computed from that mean moves. This is a concrete representation dependence. It need not be the sole cause of the report's observed end-to-end discrepancy.

QM/QE fall back to environmental atoms as receiver anchors when no independent query-anchor universe is supplied. Consequently a change in source representation can also change the receiver index. EM has the same issue on its environmental receivers.

Also, `typed_hypergraph_state.py::structural_cost` averages smooth occupancy over **counts of tensor entries**, not physical source/receiver measures. Even a repaired prediction could receive a different regularization signal after equivalent atom splitting.

### 3.2 A displayed typed membership is not the complete control dependency

The current organizer's node descriptors contain all-module summary, all-environment summary, a global token, and a receiver-state summary. The resulting `packed_embeddings` are inputs to every collective control head.

Each route also computes BOTH module and environment memberships for its control, but normal `state.memberships[tau]` retains only the route's direct source type. The other control membership is available only in special topology captures. Thus a plotted QM module membership omits some environmental control content, and both can receive all-source information through the packed node embedding.

This is not necessarily an invalid predictive model. It means that the current picture cannot be interpreted as the complete list of content donors for the control.

### 3.3 The Thermal adapter introduces further cross-source paths

`ChannelThermalInputAdapter` places `q_i / max_j |q_j|` in each module's features and includes total/mean/maximum heat in the shared global context. Changing one source can therefore alter other source tokens before any hyperedge is applied.

A support-local interpretation requires either disclosing these paths or making them explicit, optional design choices. Tree-F does the latter; historical adapters remain intact.

### 3.4 Do not equate phase-local support with global physical causality

P1 module states may already contain P0 transport and local thermal responses. An environmental token after EM may contain several modules' information. Even correctly exported sparse reads have **multi-hop ancestry**. The new interface must describe this ancestry and distinguish planning from content transmission.

## 4. Main model: Tree-F

### 4.1 Keep the valuable parts

Retain the shared `InterfaceFieldCore`, existing three-term fine backbone, all five typed mechanisms, native source-resolved nonlinear messages, one environmental source-union normalization, predicted ports, frozen Stage-A, and P0/P1/P2 assembly.

Retain the learned bounded receiver hierarchy and its source-density/control construction. Do not replace fine physical sources with averaged group field values. Do not reintroduce a separate conservative action-risk selector as the main learning problem. Do not introduce a new convex merger solver, fixed target-K histogram, or permanently killed group slots.

### 4.2 Treat an environmental source population as a measure

Write the environmental catalogue as

\[
\mu_E=\sum_j w_j\,\delta_{(x_j,z_j)}.
\]

An exact refinement replaces one atom by identical-location, identical-feature children with weights summing to the parent weight. Geometry, boundary features, physical characteristic length and external conditions remain unchanged. This is **not** a new physical source or a different grid solution.

Desired behavior:

\[
F_\theta(d,c,\mu_E;Q)
\approx F_\theta(d,c,\mathcal R\mu_E;Q),
\]

including rebuilding all P0/P1/P2 states, and with the same effective receiver/source density and control moment after child contributions are mapped back to their parent.

Implement an opt-in measure-consistent receiver index:

1. Build a canonical **index view** from exact coordinate/role equivalence classes. Sum their anchor mass. Preserve a mapping back to every original atom. This does not delete fine sources or establish a computation saving.
2. Do not split one coincident geometric/role block between children. Determine cuts between distinguishable blocks. If an axis has no separating boundary, try another geometrically valid axis or keep a leaf.
3. Use weighted child centroids, or boundaries between adjacent distinct coordinate blocks, consistently in construction and live access. Choose one convention and test it. The default is weighted child centroids.
4. Use mass-weighted node state/role summaries. Do not let input row count, tie ordering or a padded row affect node content.
5. Min-leaf and depth rules count distinct support blocks, not arbitrary copies of the same atom.
6. Keep query-anchor identity independent of requested query batches. For Thermal, prefer a case-owned, fixed physical receiver-anchor catalogue. Where EM/query anchors derive from environmental support, use its canonical measure view. Neither target queries nor solved fields define the tree.
7. Preserve physical source characteristic lengths under atom refinement. Do not replace a duplicated parent's length by the square root of each child's reduced quadrature weight.
8. Explicitly pass environmental masses in the refinement test. Letting the historical `area/E` fallback assign new uniform weights would change the physical measure and would not be an equivalence test.

The legacy Tree path remains replayable. Do not silently change its boundary convention when loading historical checkpoints.

### 4.3 Make the structural objective measure-consistent too

For route tau, let receiver masses be nu_r, source masses be mu_s, validity V_rs and smooth occupancy function psi. Use a case-local normalized integral:

\[
C_\tau=
\frac{\sum_{r,s}\nu_r\mu_s V_{rs}\,\psi(\rho_{rs})}
     {\sum_{r,s}\nu_r\mu_s V_{rs}},
\qquad \psi(t)=1-e^{-3t}.
\]

Average cases and then eligible mechanisms. Count physical modules with their declared counting measure; environmental receivers/sources use their physical or explicitly declared index measure. Query-role measures are declared separately from source quadrature.

Use analogous source-measure weighting for membership occupancy, including BOTH control donor types where they are used. Report control-pooling work separately from native value-pair work; a cheap global control summary must not be mistaken for the absence of global information. Keep active-frontier complexity separate. A root with all sources still pays for full physical access. Do not equate this representation cost with GPU latency. Retain actual integer row/call telemetry independently.

No new cost coefficient sweep. Start with the existing coefficients after a train-only magnitude/gradient sanity check; document the changed definition. If a scalar adjustment is necessary to avoid a gross scale change, use one fixed train-only calibration, not validation-driven tuning.

### 4.4 Separate the planner from the content it organizes

The planner may inspect the whole current design. Efficient information retrieval generally requires knowing which sources exist. That does not permit unrestricted all-source content to be silently passed through every group value.

Maintain separate objects:

- **Planning inputs:** current input-derived summaries, geometry, candidate receiver information and phase; used to form topology and memberships.
- **Value membership:** sources whose native messages are exposed by each typed route.
- **Control membership:** module and environmental sources used to build that route's collective control.
- **External conditioning:** named prescribed context, not solved targets or an undeclared summary of all heat sources.
- **Upstream ancestry:** module/environment states after earlier permitted interactions or physical refinement.

For route tau and group e, define declared control summaries

\[
\bar z^M_{e,\tau}=
\frac{\sum_i \mu_i^M b^M_{e i,\tau}\phi_M(z_i)}
     {\sum_i\mu_i^M b^M_{e i,\tau}},
\quad
\bar z^E_{e,\tau}=
\frac{\sum_j \mu_j^E b^E_{e j,\tau}\phi_E(z_j)}
     {\sum_j\mu_j^E b^E_{e j,\tau}}.
\]

Empty summaries return zero plus an explicit absence flag. Do not use an epsilon-created artificial source.

Construct

\[
h_{e,\tau}=H_\tau(
\bar z^M_{e,\tau},\bar z^E_{e,\tau},
\gamma_e, c_{\rm ext},\text{phase}),
\]

where gamma contains receiver geometry and role, not the all-source-content node embedding. **Remove `packed_embeddings` as an unrestricted content input to the new control heads.** A geometry-only encoding may replace that input at the same width. Keep the richer node embeddings in the planner.

Export both b^M and b^E for every route whenever both enter its control. The existing direct-source membership alone is insufficient. Separate plotting styles must show native value donors and collective-control donors.

### 4.5 Opt-in source-local Thermal features

Add a separately named adapter mode, preserving old modes and checkpoint behavior.

- Each initial module token uses its own heating, position and material descriptors, with fixed training-only scales. Replace per-case maximum-heat normalization by a fixed selected-training scale.
- Prediction-background context uses an explicit whitelist of prescribed operating/domain/material quantities and declared non-heat geometric context, such as active module count and occupied area. These geometry summaries are fixed during the heat-transfer controls; they are not asserted independent of a geometry-changing intervention. Remove total/mean/max heat from the **content/background** path of Tree-F and Pair-F; source heat reaches predictions through the source pathways.
- Aggregate heat and source summaries may still be constructed as **planner-only** inputs. They must not be fed back into the collective value heads or background field head.
- Do not change the physical heat supplied to Stage-A, the local boundary conditions, or the local normalizer. Feature isolation is not a physical-data rewrite.
- Use names from the adapter schema rather than hard-coded column guesses. Check for other source-relative features in wrapper/query/port construction and disclose them.

This is an intended architectural intervention, not an exact conversion of old checkpoints. Start Tree-F and Pair-F freshly under the new feature semantics. If this source-local representation loses accuracy, report the tradeoff; do not quietly restore the heat-summary bypass.

### 4.6 Preserve source-resolved access algebra

For receiver r, source s and route tau:

\[
\rho^\tau_{rs}=\sum_e a^\tau_{re}b^{\tau}_{es},
\qquad
m^\tau_{rs}=\sum_e a^\tau_{re}b^{\tau}_{es}h^\tau_e.
\]

The effective control is m/rho on positive support, with a safe zero otherwise. Continue using the tested physical-measure normalization in `source_moments`. Preserve the unique environmental-source attention normalization and its source quadrature. Repeated group paths are not repeated physical donors.

Interpretation is conditional computational organization, not microscopic causality. In particular, an omitted value source can still be a declared control donor, and an admitted source state can contain earlier-phase information.

### 4.7 A useful locality test

At P0, hold planner outputs/membership weights fixed for an attribution test. Change only a source's content. If that source is absent from BOTH control donor sets for an edge, the edge control must not change through an undisclosed initial normalization or background path.

This is a conditional content-path property, not a claim that the source is physically irrelevant. Do not apply the same zero test blindly at P1/P2: upstream transport/refinement may legitimately carry that source through an admitted state. Show the relevant ancestry instead.

For actual inverse optimization freeze only the declared discrete topology during a local proposal; recompute continuous features, positive memberships and controls as specified by the native fixed-topology interface. The stronger frozen-weight attribution test is diagnostic only.

## 5. Three fresh development arms, not a new large portfolio

| Arm | Purpose | Initialization / core |
|---|---|---|
| Tree-L | New-subset legacy Tree control | Current repaired historical Tree implementation, fresh seed0, existing feature semantics |
| Tree-F | Main faithful-organizer candidate | New measure-consistent index/cost, source-local adapter and membership-local controls; same fine backbone |
| Pair-F | Strong non-hypergraph comparator | Same source-local adapter and fine backbone as Tree-F; independent pairwise/sourcewise modulation, no collective hyperedge control |

All attach the same frozen Stage-A and use identical selected cases, primary queries, physical objective, effective batch and horizon. Common same-shaped physical tensors use the maintained fresh B-fine initializer. New organizer/direct-control parameters have separately seeded initialization so their creation does not disturb common physical initialization.

### Pair-F must be competent

Use the native full-access fine computation, with a trainable small modulation MLP per typed mechanism depending on receiver state/geometry, individual source state/geometry and the same prescribed context. Give it the same identity-start gain convention as Tree-F. It is not Tree-F with controls zeroed after training.

Use a width chosen from parameter counts before viewing outcomes to keep total active trainable capacity reasonably close, aiming within 10% where practical. Preserve and report exact counts. Do not add unused parameters to manufacture a match.

This control may use dense access; it is a **trained representation-capacity control**, not a pair-count-matched timing control. It retains ordinary nonlinear many-source message aggregation: “no hyperedges” does not mean “no many-body dependence.” Report its additional source work. Use the separate same-weight geometry controls to test source assignment at matched degrees/weights.

Do not implement another expensive learned Q-by-S pruning network merely to manufacture a perfectly matched sparse control in this round. If Pair-F achieves comparable quality and inverse behavior, Tree-F has not shown an added representational benefit; if Tree-F preserves quality with less source work, report representational economy separately from elapsed time.

No new Overlap/Local training. Dense-D25 is the explicitly requested Native exception. Read-only mature Run1804 and existing 500-stage results remain differently trained context. Compare Dense-D25 at exact e100 and, if the pair extends, exact e500; its e1000 and selected-best results are longer-budget references, not age-matched candidate results.

## 6. Coding route

Prefer small opt-in components over edits that change old checkpoint semantics.

| Existing location | Planned work |
|---|---|
| `interface_fields/adaptive_receiver_hypergraph.py` | Reuse planning/sparse memberships; add a named faithful mode or sibling class for local content heads and richer provenance |
| `interface_fields/receiver_tree_access.py` and the tree builder | Add explicitly selected measure-consistent construction/access; leave legacy convention accessible |
| `interface_fields/typed_hypergraph_state.py` | Export direct/control memberships and declared dependencies; add measure-consistent structural reduction |
| `interface_fields/typed_hypergraph_field.py` | Retain fine/source-union algebra; support new state export, direct modulation control and lean diagnostics |
| `interface_fields/core.py`, `capabilities.py`, config validation | Register new opt-in identifiers for 2-D/3-D without reviving dimensional workarounds |
| `Case_ThermalChannel/src/channelthermal/input_adapter.py` | Named source-local feature mode and separate prescribed versus planning context |
| `Case_ThermalChannel/src/channelthermal/interface_field_coupling.py` | Carry the new named inputs and phase provenance without changing Stage-A physics |
| Native train workflow and `tools/thermal_development.py` | Bound three explicit profiles to fixed25_v1; maintain cadence and subset checks |
| Existing response/inverse/evaluation tools | Add scoped diagnostics and the Section9 auxiliary term; no new parallel experiment framework |

Possible new filenames such as `measure_consistent_receiver_tree.py` or `source_local_control.py` are suggestions, not claims that those files already exist. Reuse existing helpers where their semantics fit.

The two Thermal feature paths must not create a second learned core inside the case package. Domain semantics belong in the adapter; measure/integration/tree/control algorithms belong in `honf_forward_core`.

Cache only geometry/index data that remain unchanged in the actual prepared input. Do not cache learned summaries, controls or trial-dependent tokens across inverse candidates. Existing object/version conventions suffice; do not introduce a new fingerprint or contract system.

## 7. Execution checks with a small, real native footprint

Execute tests and a real optimizer step; do not replace these with assertions about architecture.

### Representation tests

- Environmental source permutation, query permutation and query chunking.
- One atom split 1/2+1/2, one unequal split 0.3+0.7, and several atom splits with fixed parent features/lengths and explicit masses.
- First compare fixed prepared state; then rebuild encoder/organizer/preparation and every Thermal phase.
- Compare input measures, canonical receiver partitions, effective density, control moments, field/port/material outputs and the structural objective.
- Pull back source gradients correctly: duplicated feature/coordinate gradients sum to their parent; gradients for child weights use the declared refinement coefficients. Independent child gradients need not individually equal the parent's gradient.
- Use the existing declared floating-point tolerances and record margins. Exact equivalence in algebra does not require identical summation order in FP32.
- Treat a hard split-logit sign change near zero as measured topology fragility, not a reason to secretly loosen field tolerances. Repair the representation path or report the unresolved limit.

### Content-path tests

- New initial module token does not change when a different module's heat changes, holding the selected-training scale fixed.
- Prescribed/background token does not change under fixed-geometry heat reallocation.
- P0 fixed-plan excluded-source control test from Section4.7.
- Both control donor types are present in exported state if used by a control head.
- P2-only intervention does not change already produced P0/P1 ports.
- Visible module/source IDs, masks, native measures and physical units remain aligned.

### Runtime tests

- Real low-M and high-M train examples, one forward and one actual optimizer update per new path.
- Nonzero finite task gradients for the relevant organizer/direct modules; preserve the existing hard-value/soft-organizer gradient boundary.
- Regression of the current FP64 soft-permission quotient repair; do not revert it merely for speed. Distinguish the optimizer's surrogate topology gradient, the derivative of the evaluated fixed-topology predictor, and a physical-reference response: these are not interchangeable.
- Save/reload and a resumed update at an ordinary checkpoint boundary.
- One small generic 3-D shared-core test or retained Wind Q17 compatibility replay. No Wind training or native-grid evaluation matrix.

Limit diagnosis to actual causes. Two bounded implementation remedies per issue are reasonable. If an optional optimization is difficult, use the tested dense-mask path and continue the learning experiment.

## 8. Stage A: 100-epoch reconstruction screen

Run the three fresh profiles through **100 development epochs**. This is 15,000 primary case visits and 400 primary optimizer updates per arm, plus clearly counted setup/auxiliary work. Do not initialize them from any historical 600-case physical checkpoint.

Use the current physical loss and first-100 curriculum. No atlas response callback or new heat-null loss is active here. Train Tree-L on one GPU and Tree-F on the other; queue Pair-F on the first available authorized device. Use only physical GPUs1/2.

At e100:

1. Save milestone/latest/best and the ordinary plots.
2. Compute the same physical statistics over all22, in native units and with common role definitions. Keep field temperature, velocity/pressure, surface temperature, proxy flux and material peaks separate.
3. Run a light whole-rebuild refinement diagnostic on all22 at small Q; full detailed paths only for the four representatives.
4. Compare Tree-F with Tree-L as a compound architectural intervention, and Tree-F with Pair-F as the cleaner test of collective organization. Do not attribute a Tree-F/Tree-L difference to just one change.
5. Measure the interventions in Section11 on the four representatives.
6. Evaluate label-free fixed-geometry heat-null response on all22 and thermal sensitivity on the available declared panels. No physical-temperature labels are invented for perturbed designs.
7. Write a short e100 decision note before any extension.

### Do not reject a healthy model only because it has not beaten mature Run1804

At100 the goal is to establish learning, faithful semantics and an emerging comparative signal. Conversely, passing invariance tests does not establish physical usefulness. Preserve both findings.

If Tree-F is functionally disconnected, has consistently nonfinite steps, or fails its central source-equivalence property after bounded repairs, fix/revise it before authorizing more epochs. An ordinary unfavorable metric is research evidence, not a software gate.

## 9. Stage B: bounded response-consistency learning, only after e100 review

This section is an explicit opt-in extension, **not** a replacement of the repository development protocol.

If Stage A supports continued study, continue **Tree-F and Pair-F together to at most total e500**, with unchanged memberships, normalization, physical tensors, optimizer/RNG and absolute1,000 schedule. Tree-L remains an e100 control. Review the matched pair at e200, e300, e400 and e500; halt both at a reviewed boundary if further learning no longer tests a useful question. Every cross-arm claim uses equal exposure and a disclosed objective.

Keep the existing common physical-objective amendment and the allowed0348 response callback matched. Its one-family scope must remain prominent. Add the following common auxiliary objective to both models.

### 9.1 Benchmark-supported null response across many selected training cases

The stored benchmark has heat-transfer controls at fixed geometry with unchanged flow and pressure. Confirm this assumption against the available generator implementation and stored controls before enabling the loss. If that property cannot be verified over the intended operating regime, leave the new null term disabled in BOTH arms and state the limitation; generic prose describing coupled momentum/energy is not sufficient evidence for it. It is a property of this benchmark, not a general assertion about thermally coupled or buoyant flows.

For each eligible selected training case with at least two active positive-heat modules, generate a bounded, fixed-total heat transfer

\[
q'=q+\delta(e_i-e_j),\qquad \mathbf1^Tq'=\mathbf1^Tq,
\]

preserving geometry, module identity, material properties, Reynolds number, inlet conditions and nonnegativity. Select donors/pairs from inputs and a shared training RNG, not outcome errors. Use small bounded transfers inside declared training-range support; log realized amplitudes. M1 or unavailable donors are ineligible, not successful null examples.

Supervise only the known-null channels:

\[
L_{\rm null}=\frac14\sum_{k\in\{u,v,p,\omega\}}
\frac{\|F_k(d,q')-F_k(d,q)\|^2_{W_Q}}
     {\sigma^2_{k,\rm train}}.
\]

Use fixed training channel scales. Do not divide by the zero reference increment. Do not apply this zero target to temperature, surface temperature, material temperatures or the flux proxy.

Use two selected training cases per epoch in a shuffled cycle, common across arms, with Q256 geometry-fixed fluid queries plus the maintained pressure sections where required. Integrate their mean auxiliary loss into the final ordinary optimizer boundary of the epoch; do not quietly add a new sequence of optimizer steps. Report all extra wrapper/shadow calls and queries. Exercise the established hard/soft gradient route where needed; do not detach the organizer and then claim it learned the new objective.

Use a single fixed train-calibrated coefficient, with a default starting scale0.1 against normalized field losses. One small training-only gradient check may rescale it to keep the initial auxiliary gradient around at most10% of the reconstruction gradient. Do not run a lambda sweep or tune the coefficient on the22 validation outcomes.

Do not reduce null error by masking the model's outputs at evaluation. Every reported null response comes from its ordinary learned forward computation.

### 9.2 What this can establish

At every continued 100-epoch review boundary through e500, evaluate the same null transfers plus alternate input-selected donor pairs/amplitudes on all22 without fitting to them. Compare each model to its own e100 checkpoint and to its same-age paired control. This tests transfer of a known invariance to layouts outside this round's global training set.

Track nonzero thermal sensitivities and ordinary thermal value errors simultaneously. A model that becomes insensitive to all heat is not a successful null-response model.

### 9.3 The missing nonzero-response evidence must remain explicit

Read selected-case metadata to determine whether the150 or22 already contain physically matched fixed-geometry/context, different-heat pairs. Use only genuinely matched pairs and existing stored labels. Do not search the excluded600/90 populations to fill the gap. Pair families, not individual rows, are the units of response evidence.

Use any existing subset-aligned reference responses, with training labels confined to selected-training IDs. By default the legacy atlas supplies only0348 for training and cannot establish independent response transfer.

If no nonzero held response labels exist, report **C-positive-response transfer unavailable**, even if heat-null transfer improves. A reduced surrogate sensor residual is not an independent physical response label.

Prepare, but do not automatically execute, a small optional follow-on reference request: four fixed validation representatives, each baseline plus two opposite fixed-total heat transfers, at most12 local benchmark solves. Record exact input-only pair selection and receiver alignment. This needs separate explicit authorization and reconciliation with the existing solver allowance. No new solver attempt, CFD job or full-data expansion is included in this round's default execution.

Do not infer continuum flux conservation or mesh-independent response from the proxy q-normal channel.

## 10. Extension and model-selection decisions

The candidate ceiling is **3×100 + 2×400 = 1,100 combined development candidate-epochs**: Tree-L100, Tree-F≤500 and Pair-F≤500. Dense-D25 adds its separately requested 1,000, for **2,100 combined epochs maximum**. This corresponds to 315,000 primary training case visits, 8,400 optimizer boundaries and 322,560,000 primary fluid queries; setup, validation, physical auxiliary and response work are additional. Candidates never exceed 500. Only Dense-D25 may reach 1,000. No extra seed, Overlap/Local revival or formal launch.

Before the optional extension beyond100, record:

- actual case/query/update exposure and resource forecast;
- which Tree-F semantic properties are working and which remain unresolved;
- physical learning trends versus Pair-F and the shape of difficult-case errors;
- whether controls/gates receive task gradients and affect actual predictions;
- whether continued learning tests an unresolved question rather than a known broken implementation; forecast the next100 boundary and the maximum500 budget from measured elapsed times and current contention.

A numerically healthy but still immature Tree-F may continue even without a field-error win. A few percent lower field RMSE is not sufficient evidence on its own. No K diversity quota, zero-false-reject selector rule, or every-role-must-beat-Dense requirement controls research continuation.

If e100 already shows an unhelpful direction, spend the remaining scope on one clearly identified repair version rather than an unrelated new architecture. Major changes get a new candidate version and honest initialization/exposure accounting. They do not erase the old100 result or inherit its100 trained epochs without qualification. Do not automatically use the whole allowance.

At each e200/e300/e400/e500 review, distinguish exact endpoint from best among saved monitoring checkpoints. Primary comparisons use exact matched epochs; secondary selected-model comparisons use the same saved-checkpoint selection rule. Evaluate all claimed properties at the same selected weights.

### Concrete continuation and evaluation budget

| Boundary | Training action | Required evaluation and decision |
|---|---|---|
| e100 | Complete all three fresh candidate screens; Dense-D25 retains e100 | All22 ordinary physical statistics, light rebuilt refinement and two heat-null directions; four fixed detailed representatives, utility controls and low/high-M execution measurements. Write the review before extending the pair. |
| e200 | First matched response-consistency continuation | All22 physical/null statistics; reuse the same representatives and saved arrays. Check invariance, finite task/control gradients, thermal sensitivity and auxiliary work. Record continue/stop with measured forecast. |
| e300, e400 | Continue only the reviewed healthy matched pair | All22 physical/null statistics at each milestone, light semantic checks, ordinary cadence plots. Re-render detailed physical/graph panels only when a changed checkpoint supports a new claim. |
| e500 or earlier reviewed final stop | Stop both candidates at the declared boundary | Full bounded A/B/C closeout at matched endpoint and consistently selected monitoring weights; detailed four-case field/graph/refinement panels, measured execution, and completed inverse comparison at one declared matched age. |
| Dense e100…e1000 | Train the single authorized fresh baseline; retain every100 | Record selected-validation statistics/ordinary plots each100; perform the common detailed forward comparison at e100, matched e500 when applicable, and final e1000. No dense graph portfolio or repeated full evaluation for figure styling. |

The two-direction null check uses input-selected eligible donors, one fixed declared amplitude and one alternate pair/amplitude; ineligible cases are counted. Native-unit all22 role/per-M summaries and tails remain separate from these label-free response checks. Fix the four detailed representative IDs before reading model errors, reuse them across every arm, and record actual visits, optimizer updates, primary/auxiliary queries, wrapper/VJP calls, elapsed time and contention. The bounded inverse panel runs once at the chosen matched candidate age; it does not repeat at every checkpoint.

## 11. Evaluation that answers A and B

### 11.1 Forward statistics: all22, one ordinary pass per retained review

Report equal-case native-unit error by role, per-M sample counts, p90/max and the same pressure functional. Retain pooled metrics separately. Compare to mature Run1804 only as a differently trained reference, using its checkpoint-owned transform; do not refit its normalizer.

Detailed Q8192 field/port/material maps are for four fixed representatives, not every case. Select using geometry/M metadata before viewing errors. Use the same points, masks, color scales and reference arrays across models.

### 11.2 Same-weight controls: four representatives

At the selected Tree-F checkpoint evaluate:

1. Ordinary learned computation.
2. Zero collective control at fixed learned access.
3. Full source access with the supported learned controls retained under the existing intervention semantics.
4. Geometry-based membership changes preserving per-receiver positive-weight multisets and measures as closely as the existing exact control permits.
5. One source-assignment rewire that actually changes effective rho/control, with work differences explicitly recorded.
6. Root-union/collapse at its actual additional source work.

Reuse existing helpers and their exact validity/near conventions. If a control cannot be exactly matched, report the mismatch instead of calling it matched. Never deduplicate groups merely because binary supports agree; membership densities and controls may differ.

Separate membership-only interventions, full control-content interventions and planner rebuilds. Large fixed-weight ablation degradation establishes reliance, not uniqueness of the representation. The trained Pair-F comparison is indispensable.

### 11.3 Report K as an outcome

Report allocated capacity, actual frontier size, source-bearing groups, nonredundant actions, receiver participation and native unique source pairs separately by phase and mechanism. Do not merge these into one effectiveK.

Compare effective density/control fields, not raw local group IDs, across case or source permutations. Case-dependent membership with constantK is different from adaptiveK. A known invariant heat perturbation need not force graph topology to change; continuous controls can change while a useful topology remains stable.

### 11.4 Representation and derivative statistics

Light tests on all22: permutation and one unequal-mass duplicate refinement, with full organizer rebuild. Four representatives: multiple atoms, phase-local outputs, and fixed-topology first gradients.

Show the before/after physical output discrepancy and graph-density discrepancy. Include latent numbers only as secondary diagnostic quantities; they are not temperature errors. Report any finite prediction discontinuity when rebuilding a genuinely changed geometric tree separately from exact atom equivalence.

### 11.5 Efficiency without a kernel project

Measure one lower-M and one higher-M representative at small-Q inverse scope and Q8192. Use existing dense-mask and subset paths with several interleaved repeats. Report encoder/organizer/preparation and decode scopes, complete wrapper time, fine rows, fine calls and memory separately.

A single bounded profiler pass may remove an obvious unnecessary repeated host operation or diagnostic export. Do not rewrite the executor, relax numerical precision globally, or start custom CUDA/Triton work. Speed is secondary to A/B/C in this round.

## 12. A small, completed frozen inverse comparison

No new denoising head. Test reusable forward organization through a bounded heat-allocation inverse task at known geometry and supplied total heat. This is a preparatory inverse test, not the final generative claim.

### Normal inverse path

Use the selected Tree-F and matched Pair-F at the same review age. Observations are existing stored values at input-defined sensor positions. Sensor and held-sensor coordinates are fixed before optimization, and never selected using hidden heat or reference-error maps.

For four representatives, use two common starts: uniform allocation and one input-seeded interior simplex allocation. Compare Tree-F full-joint, Tree-F graph-block and size-matched randomized-block modes. Pair-F needs the matched full-joint control. Maximum: **4 cases × 2 starts × (3+1) modes × 10 updates = 320 attempted updates**. This is explicitly a four-case diagnostic, not a22-case inverse statistic.

Graph blocks come from the current P0/P1-relevant organization and exposed control donor sets, not just a final P2 picture. Select them by a deterministic schedule or observed-residual/model derivative information available to the inverse method; never by hidden target heat. Random blocks preserve block size. Full-joint fallback is recorded and is not a graph action.

Use the current fixed-total feasible parameterization or simplex-tangent steps. At each local step, use a bounded trust step with at most two trial evaluations for step acceptance, against the **observed** objective only. Do not use held sensors for line search. If neither trial helps, retain the current design and count a rejected step. Do not enlarge a step solely to make the trail visible.

Freeze the declared discrete topology for an inner proposal while recomputing continuous physical states, then rebuild after an accepted proposal and record any change. Preserve the actual native fixed-topology validity policy; do not freeze all tokens or the whole forward model state across candidates.

### Required outputs

Persist each completed trail immediately. Plot observed and disjoint held residuals versus **charged forward/VJP calls**, not just iteration. Report actual graph/full/random update differences, accepted steps, constraints and native heat allocations. Record observation-Jacobian rank without claiming uniqueness. In M10, six observations cannot identify nine unconstrained heat directions; a different recovered heat vector can still fit observations.

Candidate predictions and stored reference at the original design are not ground truth for the newly generated allocation. Report surrogate fit, held stored-observation fit, and hidden-heat discrepancy with that limitation. No physical design or generative-diversity claim follows without independently evaluated candidates.

If the per-trajectory cost is unexpectedly high, reduce the number of modes before launches while retaining a matched completed comparison and explicitly report the reduced scope. Do not start hundreds of trajectories and return no complete condition.

## 13. Resources, scheduling and bounded autonomy

This is a finite subset campaign, not a requirement to consume a fixed duration.

- Primary science: three100-epoch fits, optional paired continuation to at most500 after actual review, and the explicitly requested dense1000 baseline.
- Forecast elapsed and aggregate GPU-associated time from actual first-five-epoch measurements for each architecture, including contention. The original6–8-hour/12-GPU-hour estimate covered the smaller200-epoch plan and is superseded by the larger explicit budget. Do not extend epoch/data scope to consume time. Record forecast revisions and stop a broken candidate for repair rather than spending its allowance blindly.
- Forecast actual per-epoch runtime after the first five epochs. Report expected stage completion; do not infer exactly4× speedup from quarter case count.
- Use only the two authorized physical GPUs1/2. Log device mapping once at launch; do not use GPU0 or interrupt external jobs.
- Under external contention, progress on the other GPU, use safe co-residency if memory permits, or reduce microbatch size with matched gradient accumulation. Preserve effective batch and selected-case coverage. Record both contention-affected and uncontended timing where available; do not wait indefinitely for exclusive access.
- Checkpoint and halt only our own jobs if an actual external/time boundary requires it. Label incomplete epochs/recovery state honestly; do not call a censored screen complete.
- Reuse saved prediction arrays for figures and tables. No repeated full evaluations for alternative captions.
- Reserve the final45–60 minutes for the report, tests of changed paths, and repository closeout. Do not start an optional extension or inverse panel that cannot leave a completed useful unit.
- No extra solver calls, new data population, fifth trained architecture, additional seed or formal-run launch. Dense-D25 is the only fourth-architecture exception requested in this amendment.

Codex may revise implementation details and make at most two bounded, evidence-driven remedies per central failure. Document reason, actual change, source/optimizer lineage, attempted work and resulting measurement. Routine bugs should be repaired locally; only genuine unresolved limits need escalation. A smoke test, a test count or preparation command does not substitute for the required training and evaluation.

## 14. Required final report and figures

Begin with a plain-language table:

| Goal | What changed | What improved | What did not improve / remains untested | Evidence scope |
|---|---|---|---|---|
| Predictor | Actual trained architecture and objective | Physical values/response findings | Tails, approximation and data limits |150/22, saved age, native units |
| Organizer | Measure and content-path changes | Added utility / refinement consistency | Trained-control deficits, unused groups or hidden paths |All22 light checks, four detailed cases |
| Inverse | Frozen interface and actual step method | Held-observation/call-efficiency result | Underdetermination and missing candidate reference |Four cases, two starts, completed updates |

Answer A, B and C explicitly. Do not combine them into a single pass flag. “Null response transferred; nonzero response generalization unavailable” is a valid and useful conclusion when that is what the evidence supports.

Retain at most six main figures, visually inspected:

1. Learning and budget: physical role learning curves against true development epochs, visits and measured GPU-hours; e100/e200/e300/e400/e500 review decisions marked, with the separately budgeted dense e1000 reference labelled.
2. Representative physical fields: reference/prediction/residual at the four fixed cases; all22 numerical summary alongside.
3. Equivalent-representation test: original versus refined atoms, receiver boundaries, effective access, physical output differences before/after repair.
4. Value/control/planning paths and utility: both control donor types, actual receiver support, intervention effects and trained Pair-F comparison. Distinguish schematics from measured graphs.
5. Response transfer: fixed-total heat perturbation direction, per-channel null changes and nonzero thermal changes; clearly separate reference-labelled and model-only quantities.
6. Inverse: actual heat/design trajectories, observed and held residuals versus charged calls, accepted/rejected steps and failure cases.

Immediately below each figure include a quantitative description and a plain-language interpretation. Do not plot packet tensor indices as spatial coordinates. Do not make all good examples detailed and leave failures only in an appendix. Statistical development summaries always use the22, not only favorable representatives.

Preserve existing artifact/security rules. Generated arrays/figures/checkpoints and one-time renderers stay in ignored local paths; reusable code/tests/configuration/report are committed and pushed on the non-default branch. Use existing source-control and checkpoint primitives. Add no cryptographic fingerprint system, contract freeze, baseline snapshot suite or blocking research gate. Preserve every existing security boundary and historical run.

## 15. Completion and next decision

Completion means new trained development evidence and explicit A/B/C findings, not successful formal promotion.

Prefer Tree-F as the next research base only if it demonstrates its intended representation/content semantics and retains a credible physical/response tradeoff against its controls. A useful graph-reuse signal on the small inverse panel strengthens the case but is not mandatory for reporting the trained result. If Pair-F matches it, acknowledge that collective grouping has not yet shown extra value.

Do not recommend5k simply because Tree-F is the best of these three. Prepare a short proposed next100-epoch continuation or the separate12-solve response-evidence request if warranted. **Formal training remains separately authorized and normally manually launched by the user.** Do not convert a subset checkpoint into a full-data resume, overwrite its normalizer or silently reuse its schedule under a formal label.

If Tree-F succeeds on B but not A/C, the correct conclusion is that the interface is now more faithful but not yet shown more useful. That is more substantive than an unexplained few-percent reconstruction change and tells us what to test next.

## Appendix A. Reviewed code and report basis

All paths are relative to `HONF_Proj/` unless stated otherwise; inspected at378de02.

- Root `AGENTS.md`: fixed-development and reporting obligations.
- `docs/guides/Thermal_Model_Development_Protocol.md`:150/22 membership, cadence,1000 horizon,0348-only response overlap.
- `docs/reports/HONF_Shared_Core_Thermal_500_Epoch_Conclusions.md`: matched500 results, utility, atom-splitting failure, inverse readiness.
- `docs/guides/Shared_Core_Thermal_Wind_Architecture_Map.md`: shared core versus domain assembly and information-path differences.
- `src/honf_forward_core/interface_fields/adaptive_receiver_hypergraph.py`: source/receiver pooling, node features, dual-type control summaries and typed export.
- `src/honf_forward_core/interface_fields/adaptive_interaction_cover.py`: receiver tree construction and legacy split convention.
- `src/honf_forward_core/interface_fields/receiver_tree_access.py`: unweighted child boundary calculation and live batched access.
- `src/honf_forward_core/interface_fields/typed_hypergraph_state.py`: density/moment normalization, state export and count-based structural objective.
- `src/honf_forward_core/interface_fields/typed_hypergraph_field.py`: actual fine reader, control modulation, same-weight interventions, dense/subset execution.
- `src/honf_forward_core/interface_fields/core.py`: source encoders, environmental weight fallback and phase preparation.
- `Case_ThermalChannel/src/channelthermal/input_adapter.py`: case-maximum heat normalization and global source summaries.
- `Case_ThermalChannel/src/channelthermal/environment.py`: physical environment features, source measures and characteristic lengths.

The diagnoses above are source-derived; the proposed Tree-F/Pair-F models and future benefits are hypotheses. No new native checkpoint replay or GPU training was conducted in preparing this document.
