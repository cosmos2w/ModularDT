# HONF Goal Mode Plan: Soft-Organized, Sparse-Executed HONF vs. Fixed-Softmax Fallback

## 0. Mission

Work only in:

- repository: `cosmos2w/ModularDT`
- branch: `agent/honf-adaptive-correctness`
- current branch head at plan preparation: `7a493b96649bbf78192d88c737a48239f28f14c5`

At the start of the task, verify the actual checked-out branch and current `HEAD`. If the branch has advanced, treat the newer branch head as authoritative and record the difference. Do **not** use `main` as the implementation reference.

This round has one scientific goal:

> Compare two modern HONF forward-model strategies under matched data, decoder, loss, optimizer, and evaluation conditions:
>
> **A. Soft-Organized, Sparse-Executed HONF** — learn meaningful roles with soft assignments during training; derive sparse execution only after roles exist.
>
> **B. Fixed-Softmax Modern Fallback** — keep the proven fixed softmax organizer, combine it with the current modern decoder/training stack, and apply the same post-training retained-mass pruning.

The round must preserve the positive results established by Runs 1102/1103/1201–1203 while directly correcting the topology failure diagnosed in Run 1202.

Do not launch a long formal run. Codex should use existing run artifacts first, perform evaluation-only work, make only gated code changes, run ordinary tests and very short smoke/diagnostic screens, and prepare the two final comparison profiles/commands for the user.

---

# 1. Evidence already established — do not re-prove it by retraining

Treat the following as established unless new evidence contradicts it.

## 1.1 Run 1202 is a predictive baseline, not a topology-qualified HONF

Run 1202 demonstrated:

- numerically stable training;
- valid probability/support conservation;
- exact additive closure;
- strong convergence with split learning rates;
- competitive matched-budget field accuracy;
- useful modern diagnostics and training infrastructure.

However, conditional topology analysis of case `0653`, with case `0273` as a robustness check, showed that the selected Run-1202 edges are nearly interchangeable.

For case `0653`:

- module assignment edge-profile cosine: about `0.998`;
- module effective rank: about `1.01`;
- environment edge-profile cosine: about `0.739`;
- environment effective rank: about `1.29`;
- normalized region-center separation: about `0.0246`;
- query normalized entropy: about `0.9985`;
- query edge-profile cosine: about `0.9995`;
- query spatial standard deviation: about `0.00382`;
- one edge wins 100% of queries by a small near-uniform margin;
- pairwise contribution-map cosine: about `0.998`;
- additive per-edge physical fields are close to scaled copies.

Therefore selected-edge count, total mass entropy, maximum total edge mass, and balanced edge-energy fractions are **not sufficient evidence of mechanism specialization**.

## 1.2 Run 1000 remains the organizer/routing structural reference

Run 1000 has a fixed organizer and context-fusion decoder, so it is not an architecture-controlled baseline. Nevertheless, its exported organization is materially more structured:

- environment profile cosine about `0.093`;
- environment effective rank about `5.36`;
- normalized region-center separation about `0.306`;
- query edge-profile cosine about `0.424`;
- query spatial standard deviation about `0.158`;
- multiple coherent dominant-query regions.

Its module organization is still soft and broad; do not overstate module specialization. Its strongest advantage is in environment and query structure.

## 1.3 Split optimizer is retained

Use as the modern optimizer reference:

```text
prediction/default LR = 3e-4
organizer LR          = 1e-4
```

The current branch already has optimizer-group inventory and resume-safety support.

## 1.4 Dense residual background is retained

Use:

```json
"additive_background_mode": "dense_query_attention"
```

`global_pooled_attention` failed the Stage-4 accuracy gate and produced no useful measured 2-D speed gain. Keep it as an ablation only.

## 1.5 Entmax and transition schedules are no longer assumed necessary

The current diagnosis shows that candidate organization is already close to symmetric before hard selection can create meaningful roles. Entmax cannot create specialization from nearly equal logits.

Therefore the default scientific direction for this round is:

> softmax role learning during training; no forced exact zeros; no arbitrary sparsity transitions.

## 1.6 Current host-side efficiency hardening is retained

The current branch includes prediction-preserving training improvements such as packed scalar CPU transfers, validation reuse, reduced plot cadence, and reduced latest-checkpoint write cadence. Preserve these.

---

# 2. Existing artifacts to reuse before any new training

Codex must discover and evaluate existing local runs before deciding that a new diagnostic training screen is needed.

Known references include:

```text
Run 1000:
HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/
Run_1000_20260817_214356_enhanced_honf_pairwise

Run 1201:
Run_1201_*stage4_uniform_lr2e4_dense_background

Run 1202:
Run_1202_20260820_185410_stage4_split_lr_dense_background

Run 1203:
Run_1203_*stage4_split_lr_pooled_background
```

Also search the same run root for existing staged bridge runs corresponding to:

- `Run_1005_*` or another run using `stage1_fixed_additive_soft.json`;
- `Run_1007_*` or another run using `stage2_exchangeable_soft.json`.

Do not rely only on run number. Read each run's `config_resolved.json` / saved provenance and classify it by actual architecture.

The highest-value existing ladder is:

| Ladder position | Organizer | Assignment | Decoder/assembly | Purpose |
|---|---|---|---|---|
| Run 1000 | fixed projection | softmax | classic context fusion | structural reference |
| Stage-1 / likely Run 1005 | fixed projection | softmax | descriptor-first + exact additive | test whether modern decoder destroys organization |
| Stage-2 / likely Run 1007 | exchangeable slots | all-soft softmax | descriptor-first + exact additive | locate exchangeable-slot failure before schedules |
| Run 1201 | exchangeable | scheduled | additive | optimizer-rate structural check |
| Run 1202 | exchangeable | scheduled | additive | predictive baseline |
| Run 1203 | exchangeable | scheduled | pooled additive | rejected background branch |

If a staged run is missing, say so explicitly. Do not silently replace it with a new formal run.

---

# 3. Two target model families

## 3.1 Model A — Soft-Organized, Sparse-Executed HONF

Scientific principle:

\[
oxed{	ext{learn roles softly; prune routes only after roles exist}}
\]

Training target:

- exchangeable organizer;
- softmax module assignment throughout training;
- softmax environment assignment throughout training;
- softmax query routing throughout training;
- all candidate edges available during training;
- no entmax schedule;
- no hard-selection schedule;
- dense exact execution as the training/reference path;
- descriptor-first mechanism state;
- exact additive background + edge field assembly;
- dense residual background;
- current pairwise kernel;
- split optimizer `3e-4 / 1e-4`;
- no organizer entropy/count/diversity/load-balancing regularization.

Primary matched comparison should use **six runtime slots** initially so Model A and Model B have the same nominal edge count and decoder route dimension. Do not use eight slots in the primary final comparison unless the staged evaluation provides a strong reason.

A six-slot research profile makes the comparison interpretable:

```text
exchangeable vs fixed organizer
```

rather than:

```text
exchangeable + larger K vs fixed + smaller K
```

Capacity-adaptive `K=8` behavior can be evaluated later as a secondary exchangeable-only test.

### Model A baseline configuration

The intended starting point is approximately:

```json
{
  "organizer_mode": "exchangeable_slots",
  "edge_capacity": 6,
  "initial_active_edges": 6,
  "minimum_active_edges": 1,
  "edge_selection_mode": "all",

  "module_assignment_normalizer": "softmax",
  "environment_assignment_normalizer": "softmax",
  "query_assignment_normalizer": "softmax",

  "environment_locality_mode": "none",
  "query_locality_mode": "none",

  "mechanism_state_mode": "descriptor_first",
  "field_assembly_mode": "edge_additive",
  "additive_background_mode": "dense_query_attention",

  "routing_execution": "dense"
}
```

Training:

```json
{
  "learning_rate": 0.0003,
  "organizer_learning_rate": 0.0001
}
```

Do not assume this baseline is sufficient. First determine whether the existing Stage-2 all-soft run already learns meaningful organization.

---

## 3.2 Model B — Fixed-Softmax Modern Fallback

Purpose:

> Preserve the proven structured fixed softmax organizer while retaining the modern data path, descriptor diagnostics, additive decoder, dense residual background, split optimizer support, and later retained-mass sparse execution.

Target configuration:

```json
{
  "organizer_mode": "fixed_projection",
  "num_hyperedges": 6,
  "edge_selection_mode": "all",

  "module_assignment_normalizer": "softmax",
  "environment_assignment_normalizer": "softmax",
  "query_assignment_normalizer": "softmax",

  "mechanism_state_mode": "descriptor_first",
  "field_assembly_mode": "edge_additive",
  "additive_background_mode": "dense_query_attention",

  "routing_execution": "dense"
}
```

Training:

```json
{
  "learning_rate": 0.0003,
  "organizer_learning_rate": 0.0001
}
```

The current fixed organizer contains its own persistent learned edge scoring channels and source-centered geometry bias. Preserve that behavior in the fallback; do not rewrite it to imitate the exchangeable organizer.

The preferred fallback is **fixed organizer + modern additive decoder**. If existing Stage-1 evaluation proves that fixed + additive already loses the useful Run-1000 organization, stop and report before launching a fallback formal run. In that case, context fusion remains the structural fallback and the additive decoder itself requires separate diagnosis.

---

# 4. Core evaluation infrastructure — implement first

Before model revisions, create or generalize an evaluation tool that can compare organizer/routing quality over a full split and over selected cases.

Prefer one maintained tool rather than several one-off notebooks/scripts.

Suggested location:

```text
HONF_Proj/diagnostics/evaluate_topology_quality.py
```

or an equivalent maintained diagnostics location.

It must support both fixed and exchangeable organizers and both context-fusion and additive models.

## 4.1 Inputs

Support:

- one or multiple checkpoint paths;
- named labels;
- split selection;
- case subset or complete split;
- fixed query grid;
- query chunking;
- optional base/provisional/final organizer capture when available;
- optional exact additive edge fields when available.

## 4.2 Required organizer metrics

For `A_mh` and `A_eh`, after restricting to active/present rows and active edges as appropriate:

1. row-normalized entropy;
2. mean row maximum probability;
3. effective edges per row:
   \[
   \exp(H)
   \]
4. dominant occupancy fraction per edge;
5. largest dominant occupancy;
6. row-to-row cosine similarity;
7. edge-column/profile cosine similarity;
8. uncentered effective rank;
9. centered effective rank where meaningful;
10. candidate and selected versions when exchangeable candidates exist;
11. source-center pairwise separation;
12. region-center pairwise separation;
13. separation normalized by domain diagonal;
14. region/source scale statistics.

For environment organization also compute:

- local neighbor agreement of dominant owner;
- neighbor L1 assignment variation;
- spatial smoothness of assignment probabilities.

## 4.3 Required query-routing metrics

For `query_hyper_attention`:

- normalized row entropy;
- maximum probability;
- effective edges per query;
- largest dominant-query occupancy;
- edge-map/profile cosine;
- per-edge spatial standard deviation;
- mean L1 deviation from the case-global mean routing vector;
- neighbor L1 routing variation;
- effective rank of the query × edge map.

Do not use the dominant-edge plot without probability-margin statistics.

## 4.4 Pairwise and additive mechanism metrics

When available:

- pairwise contribution-map edge-profile cosine;
- coefficient of variation of mean pairwise contribution;
- additive edge-field cosine by physical channel;
- edge-field effective rank by physical channel;
- near-interface vs far-field contribution energy;
- edge contribution energy by physical channel;
- whether the same edge dominates every variable and region.

For context-fusion models where exact `pred_field_by_edge` is unavailable, clearly mark those metrics unavailable rather than inventing proxies. Pairwise routing maps may be used as the nearest internal comparison.

## 4.5 Visualization fixes

Produce both:

```text
organization_environment_unsorted.png
organization_environment_sorted_by_dominant_edge.png
```

The sorted plot must be clearly labeled as sorted.

Pass the actual active-edge mask to routing visualizers. Do not render inactive zero candidates as active routing panels.

For each case, save:

- module assignment matrix;
- unsorted environment assignment matrix;
- sorted environment assignment matrix;
- physical environment-token map;
- query attention maps;
- dominant-edge map;
- query probability margin map;
- region/source centers;
- pairwise contribution maps if available;
- additive per-edge physical fields if available.

## 4.6 Machine-readable outputs

Write:

```text
topology_quality_summary.json
topology_quality_per_case.csv
topology_quality_per_case_edge.csv
```

Include checkpoint provenance, epoch, source SHA when available, field assembly, organizer mode, edge capacity/count, normalizers, background mode, and optimizer settings.

---

# 5. First decision stage: use existing runs only

Run the new evaluator over the existing ladder before changing training/model code.

At minimum evaluate:

- Run 1000 best-field;
- Stage-1 fixed-additive best/latest if available;
- Stage-2 exchangeable-all-soft best/latest if available;
- Run 1201 best/latest;
- Run 1202 best/latest;
- Run 1203 best/latest.

Use the complete 90-case test split if the evaluator is efficient enough. If that is expensive, first evaluate a deterministic stratified subset of at least 20 cases including `0653` and `0273`, then run the complete split only for the finalists.

No training is allowed in this stage.

## 5.1 Causal decision table

### Outcome A

**Stage-1 fixed-additive preserves Run-1000 organization, but Stage-2 exchangeable-all-soft is rank-one.**

Conclusion:

> Exchangeable slot formation/refinement is the primary failure.

Proceed to the minimal exchangeable symmetry/locality repair stages below.

### Outcome B

**Stage-2 exchangeable-all-soft is well organized, but Run 1202 is rank-one.**

Conclusion:

> Scheduled selection/entmax/locality or later ThermalChannel reorganization destroys a valid soft topology.

Do **not** redesign exchangeable slot identity. Model A becomes the all-soft Stage-2 concept with modern optimizer settings. Proceed directly to base/provisional/final diagnosis and then final comparison preparation.

### Outcome C

**Stage-1 fixed-additive already loses Run-1000 organization.**

Conclusion:

> Fixed organizer alone is insufficient to preserve the old structure after the modern decoder/mechanism changes.

Do not call fixed-additive a safe fallback yet. Compare fixed context-fusion and fixed additive carefully. Stop before formal comparison profiles and report which modern component first destroys structure.

### Outcome D

**Base organizer is meaningful but provisional/final organizer becomes rank-one.**

Conclusion:

> ThermalChannel repeated reorganization after local-surrogate fusion is the primary failure.

Proceed to topology-content decoupling instead of modifying slot formation.

---

# 6. Base/provisional/final organizer diagnosis

The current ThermalChannel path can organize more than once:

1. base organization;
2. provisional organization after local-response fusion;
3. final organization.

For exchangeable models, capture candidate-level organization before selection/masking for each available pass.

Because exchangeable codes are unordered, align candidates across passes by permutation before comparing. Use a deterministic Hungarian/min-cost alignment based on a composite distance between:

- candidate module profile;
- candidate environment profile;
- source coordinate;
- region coordinate;
- slot/code state where appropriate.

Export:

```text
A_mh_base
A_eh_base
A_mh_provisional
A_eh_provisional
A_mh_final
A_eh_final
```

and aligned change metrics.

### Gate

If base organization is structurally good and final organization collapses, do not introduce persistent-code or anchor changes yet.

Instead implement an optional **topology-content decoupled mode**:

- discover topology once from base geometry/module state;
- retain the soft incidences and source/region geometry;
- after local-surrogate fusion, recompute hyperedge content/state using the retained incidences;
- do not rediscover edge ownership from scratch.

This mode must be optional and must preserve the current behavior as a compatibility choice.

Only implement this if the evaluation demonstrates that repeated reorganization is the failure location.

---

# 7. Minimal exchangeable repair path — only if Stage-2 all-soft is already rank-one

Do not implement all repairs at once.

Each repair is conditional on the previous screen failing.

## 7.1 Repair A — softmax + stronger physical locality using existing code if possible

First avoid code changes if a config-only screen can answer the question.

Use:

- all-soft exchangeable training;
- six slots;
- no selection;
- no entmax;
- split LR;
- dense background;
- dense execution.

Test one stronger locality candidate, not a broad sweep.

Suggested first screen:

```json
"environment_locality_mode": "gaussian_bounded",
"environment_locality_strength": 1.0
```

If the current self-broadening region scale makes this ineffective, do not keep increasing strength blindly.

### Short topology screen only

Maximum:

- 300 epochs;
- same seed/data;
- save snapshots around epochs 25, 50, 100, 200, 300;
- no need for complete-split field evaluation.

At each snapshot evaluate a deterministic topology subset.

Stop early if the organizer remains clearly rank-one by epoch 100–200.

### Pass signal

At least by epoch 300, environment/query metrics should move strongly away from the Run-1202 regime:

- environment profile cosine < `0.6`;
- environment effective rank > `2.0`;
- normalized region-center separation > `0.08`;
- query profile cosine < `0.9`;
- query global-deviation L1 > `0.05`.

These are screening gates, not final scientific gates.

---

## 7.2 Repair B — persistent candidate-code residual

Implement only if Repair A is insufficient.

Add a legacy-preserving config field, for example:

```json
"slot_code_residual_scale": 0.0
```

Current behavior must remain exactly `0.0`.

For a nonzero value, after each shared slot-refinement update, reintroduce the corresponding deterministic candidate code through a permutation-equivariant residual:

\[
h_k^{t+1}
=
\operatorname{Norm}
\left(
\operatorname{GRU}(u_k^t,h_k^t)
+
\lambda_c c_k

ight).
\]

Prefer no new edge-specific learned parameters. The code should permute with the slot and preserve code-permutation equivariance.

Suggested first research value:

```text
slot_code_residual_scale = 0.15
```

Do not sweep many values in this round.

### Required tests

- `0.0` reproduces current exchangeable output exactly;
- state-dict/checkpoint compatibility is unchanged if no new trainable tensor is introduced;
- code permutation equivariance remains within the existing tolerance;
- forward/backward finite;
- six-slot and eight-slot runtime capacities remain valid;
- all-soft assignment rows remain normalized.

Then run at most the same 300-epoch topology screen.

---

## 7.3 Repair C — fixed physical locality scale

Implement only if candidate codes remain distinct but environment regions still self-broaden/collapse.

Add a legacy-preserving locality scale choice, e.g.:

```json
"environment_locality_scale_mode": "adaptive_region"
```

with a new option such as:

```json
"environment_locality_scale_mode": "fixed_domain"
```

and a fixed physical scale parameter such as:

```json
"environment_locality_scale_fraction": 0.25
```

The fixed-domain mode must not normalize distances by assignment-derived region scale.

A simple physically interpretable formulation is preferred:

\[
b_{ek}
=
-\lambda
rac{\|x_e-r_k\|}
{s \,\mathrm{diag}(\Omega)}
\]

or an equivalently controlled Gaussian using a fixed domain-derived length.

Do not add learned edge-specific anchors in this stage.

### Required tests

- legacy `adaptive_region` path exact parity;
- fixed scale behaves consistently under coordinate scaling;
- finite gradients;
- permutation equivariance;
- spatial locality bias differs among already-separated candidates;
- no checkpoint-key drift unless strictly necessary.

Run at most one 300-epoch topology screen.

---

## 7.4 Repair D — physical candidate anchors

This is **deferred** unless A–C all fail.

If needed, implement deterministic physical anchors generated from candidate codes and domain geometry, with no edge-index-specific learned parameters.

Do not implement this in the same commit as Repairs B/C unless the prior gates explicitly fail and the Goal Mode report justifies continuing.

---

# 8. Safe fallback profile preparation

If existing Stage-1 fixed-additive organization is acceptable, create a modern fallback experiment profile.

Suggested name:

```text
stage5_fixed_softmax_modern.json
```

Use:

- fixed projection;
- six hyperedges;
- all softmax;
- descriptor-first mechanism;
- edge-additive field assembly;
- dense query-attention background;
- current pairwise kernel;
- dense execution;
- prediction LR `3e-4`;
- organizer LR `1e-4`;
- current dataset/loss/Stage-A settings;
- no sparsity/selection schedule;
- no organizer regularization.

Do not modify the canonical fixed-organizer implementation.

If the split optimizer has not been tested with fixed projection, add focused optimizer-group tests showing that `core.organizer.*` contains the fixed organizer tensors and both parameter groups update.

---

# 9. Soft-organized research profile preparation

Once Model A passes the topology screen, create a single comparison profile representing the **smallest successful exchangeable change**.

Suggested name:

```text
stage5_exchangeable_soft_organized.json
```

It must use:

- six exchangeable slots;
- all-soft softmax training;
- no hard selection;
- no entmax;
- dense training/reference execution;
- dense background;
- split LR;
- only the minimal topology repair(s) proven necessary by prior gates.

Do not carry failed experimental knobs into the final research profile.

Document exactly which changes separate it from the fixed fallback.

---

# 10. Sparse execution is evaluation/deployment behavior, not training behavior

Once a soft model is topology-qualified, compare dense exact execution with retained-mass sparse execution.

Use the current gathered/pruning infrastructure where possible rather than adding a new sparse normalizer.

For query-to-edge routing:

\[
S_q = \min\{S:\sum_{k \in S}lpha_{qk}\ge 
ho_q\}
\]

with initial retained-mass target:

```text
rho_q = 0.98
```

For routed module incidence:

```text
rho_m = 0.95
```

Use an explicit minimum route limit only as a safety floor, not as the main sparsity mechanism.

Do not prune softmax probabilities during training in this round.

## 10.1 Evaluation-only runtime override

If checkpoint-owned config currently prevents switching a saved dense checkpoint into gathered/pruned evaluation cleanly, add an evaluation-only tool that:

1. loads checkpoint-owned model architecture;
2. constructs an identical parameter-shape model with routing-execution overrides only;
3. strictly loads the same state dict;
4. evaluates dense and pruned outputs on identical prepared states;
5. never writes back to the source checkpoint.

## 10.2 Required sparse-execution metrics

For both Model A and Model B:

- dense field error;
- pruned field error;
- per-channel error delta;
- near/far error delta;
- retained query-edge mass min/p05/mean;
- retained module mass min/p05/mean;
- mean evaluated query-edge routes;
- mean evaluated query-module routes;
- route-count reduction;
- decoder median time;
- peak allocated memory;
- exact output difference when no pruning is active.

### Sparse execution promotion gates

Initial gates:

- aggregate fluid MSE degradation <= `2%`;
- no physical channel MSE degradation > `5%`;
- retained query-edge p05 >= `0.98` unless an explicit fallback expands routes;
- retained routed-module p05 >= `0.95`;
- route-count reduction >= `20%`.

Timing on the small 2-D case is secondary. A measured speedup is desirable, but a topology-qualified route-count reduction is the more important scaling proxy for future 3-D cases.

If route-count reduction is <20%, report that the learned soft representation is not sparse-execution-ready even if its topology looks meaningful.

---

# 11. Final scientific comparison — user-controlled, not Codex-launched

Codex must prepare the two final profiles and commands but **must not launch the long comparison runs**.

The final pair should be:

### Model A

```text
Soft-Organized, Sparse-Executed HONF
```

using the smallest topology-qualified exchangeable configuration.

### Model B

```text
Fixed-Softmax Modern Fallback
```

using six fixed softmax edges and the same modern decoder/training stack.

## 11.1 Matched conditions

Both must share:

- dataset and split;
- Stage-A checkpoint;
- seed;
- batch/query sampling;
- hidden width;
- pairwise kernel;
- descriptor-first mechanism state;
- additive decoder;
- dense residual background;
- prediction LR `3e-4`;
- organizer LR `1e-4`;
- training loss;
- training precision;
- dense reference execution;
- six nominal hyperedges/slots;
- checkpoint cadence;
- evaluation protocol.

The primary difference should be the organizer:

```text
exchangeable soft-organized vs fixed softmax
```

plus only the minimal repair needed for exchangeable identifiability.

## 11.2 Minimize formal training

Use **two runs total**, one per model.

First stage:

```text
1500 epochs
```

Evaluate both at the same budget.

Only if both remain scientifically viable and still improving, **resume the same runs** to:

```text
2500 epochs
```

Do not launch new run IDs for the extension.

No automatic 5000-epoch continuation is required.

### Save diagnostic snapshots

For the two final comparison runs, retain checkpoints at approximately:

```text
250
500
1000
1500
2500 (if resumed)
```

This round has no arbitrary sparsity transitions, so snapshots are for role-formation analysis rather than schedule-boundary analysis.

---

# 12. Final acceptance matrix

## 12.1 Predictive quality

At 1500 epochs:

- centered/trailing field MSE should be within about `15%` of the Run-1202 matched-budget reference;
- no sustained numerical instability;
- per-channel full-split error should remain physically reasonable.

At 2500 epochs if resumed:

- field MSE should be within about `10%` of Run 1202 at the same budget, unless a documented topology improvement justifies a small tradeoff.

## 12.2 Topology quality

Use complete-split medians/distributions, not only case `0653`.

Provisional gates:

### Environment

- normalized row entropy < `0.65`;
- mean row maximum > `0.50`;
- edge-profile cosine < `0.55`;
- effective rank > `2.5`;
- normalized region-center separation > `0.10`;
- largest dominant-token occupancy < `0.70`.

### Query

- normalized query entropy < `0.90`;
- query edge-profile cosine < `0.80`;
- largest dominant-query occupancy < `0.80`;
- global-route L1 deviation > `0.10`;
- clear nonzero spatial variation.

### Pairwise/edge fields

- pairwise map cosine < `0.85`;
- additive edge-field cosine < `0.90` where available;
- no single edge should be the same dominant mechanism for nearly every variable, case, and near/far region.

These gates are intentionally much stricter than Stage-4 aggregate mass/count gates.

## 12.3 Generalization / adaptivity

For Model A only, after the main six-slot comparison:

- evaluate runtime capacity changes only if the code supports them without parameter-shape changes;
- test at least one alternative capacity such as `K=8` on a subset;
- treat this as a secondary generalization study, not part of the primary model-vs-fallback fairness comparison.

Do not claim capacity generalization merely because the model loads.

## 12.4 Sparse execution

Apply the same retained-mass pruning evaluator to both finalists.

Compare:

- accuracy loss;
- routes retained;
- runtime;
- memory.

The scientific question is:

> Does the exchangeable soft organizer yield a more prune-able representation than the fixed softmax organizer at comparable field accuracy and topology quality?

This is the ultimate comparison, not raw training-time selected-edge count.

---

# 13. Tests and run-budget rules for Codex

## 13.1 Ordinary code tests

Run relevant maintained unit/integration tests after each code change.

At minimum cover:

- config parsing and strict rejection;
- legacy config parity;
- fixed organizer behavior;
- exchangeable code-permutation equivariance;
- additive closure;
- prepared/chunked decoding;
- optimizer grouping;
- checkpoint loading/resume where touched;
- routing retained-mass logic;
- diagnostics outputs.

## 13.2 Smoke tests

Smoke tests exist only to confirm correctness.

Maximum per configuration:

- 2 epochs;
- <=4 train batches per epoch;
- <=2 validation batches.

Do not infer topology quality from these smoke runs.

## 13.3 Diagnostic training screens

Only allowed when existing runs cannot answer the topology question.

For exchangeable repair screens:

- <=300 epochs;
- one seed;
- no more than one active candidate change at a time;
- use a deterministic topology subset;
- stop early when a screen is clearly rank-one.

Do not run a 650-, 1000-, 1500-, 2500-, 5000-, or 10000-epoch experiment from Goal Mode.

## 13.4 Formal runs

Codex does not launch them.

Prepare commands only.

---

# 14. Explicit non-goals

Do not add in this round:

- edge-count loss;
- entropy regularization;
- diversity loss;
- orthogonality loss;
- load-balancing loss;
- variable-specific edge labels;
- separate learned decoder heads per edge;
- new background architectures;
- pooled background promotion;
- entmax as the main research path;
- scheduled softmax-to-entmax transitions;
- scheduled hard edge selection;
- query-locality experiments before environment roles exist;
- broader structural optimizer groups unless a later diagnosis explicitly requires them;
- inverse-model modifications.

Do not merge or rewrite the fixed organizer to look exchangeable.

---

# 15. Required deliverables

Codex should produce:

## 15.1 Evaluation report

Suggested:

```text
HONF_Proj/UpgradePlan/Stage5_SoftOrganization_Evaluation.md
```

Include:

- branch/SHA;
- discovered run inventory;
- topology metrics for existing ladder;
- causal decision;
- base/provisional/final diagnosis;
- which changes were or were not required;
- figures/examples;
- complete-split or subset scope;
- limitations.

## 15.2 Machine-readable topology outputs

Keep under a diagnostics/results directory, not inside historical run directories unless the evaluator's normal convention writes a new evaluation subdirectory.

Do not modify historical checkpoints.

## 15.3 Final model profiles

Prepare:

```text
stage5_exchangeable_soft_organized.json
stage5_fixed_softmax_modern.json
```

Only include topology repair fields in the exchangeable profile if the staged gates proved them necessary.

## 15.4 Sparse-execution evaluation utility

Either extend an existing evaluator or add a focused tool that compares dense vs retained-mass-pruned execution from the same checkpoint.

## 15.5 Closeout report

State clearly:

- whether Stage-1 fixed-additive preserves Run-1000 organization;
- whether Stage-2 exchangeable-all-soft is already rank-one;
- whether repeated ThermalChannel reorganization is responsible;
- which minimal exchangeable repair, if any, passed;
- whether Model A and Model B profiles are ready for matched formal comparison;
- exact commands for the two user-controlled 1500-epoch runs;
- exact commands to resume the same runs to 2500 epochs if needed;
- explicit statement that no long formal run was launched by Codex.

---

# 16. Goal Mode stopping rules

Stop rather than expanding scope when:

1. existing run evaluation already answers the causal question;
2. Stage-1 fixed-additive fails to preserve structured organization — report before calling it a safe fallback;
3. a repair screen remains rank-one after its 300-epoch budget;
4. a proposed fix would require entropy/diversity/count regularization;
5. a proposed fix would introduce edge-specific learned identities and violate exchangeability;
6. fixed and exchangeable models cannot be compared under matched decoder/optimizer conditions without a larger redesign;
7. external local artifacts required for evaluation are missing.

If a stage fails, write the evidence and stop at that stage unless this document explicitly provides the next gated repair.

---

# 17. Completion definition

The Goal Mode task is complete when:

- existing runs have been mined before new training;
- the first stage of organizer collapse is identified;
- topology diagnostics are maintained and machine-readable;
- visualization artifacts no longer obscure sorted/active-edge semantics;
- the smallest necessary exchangeable repair, if any, has passed only short topology screens;
- one soft-organized exchangeable profile is prepared;
- one fixed-softmax modern fallback profile is prepared;
- dense-vs-pruned evaluation is ready for both;
- matched 1500-epoch user-controlled run commands are prepared;
- no long formal run has been launched.

The final research question for the user-controlled comparison is:

Can soft exchangeable role learning produce topology at least as meaningful as the fixed organizer?

while

retained-mass pruning yields equal or better sparse execution at comparable field accuracy?

If yes, Soft-Organized, Sparse-Executed HONF becomes the preferred forward architecture.

If no, the fixed-softmax modern model becomes the reliable base, while exchangeable organization remains a separate research track rather than blocking the broader ModularDT program.
