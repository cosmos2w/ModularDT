# Stage 7 Goal Mode Plan — Modern Structured Context HONF

## 0. Mission

Stage 7 is a **forward-model consolidation stage**, not another architecture search.

The formal objective is to build a clean modern reproduction of the successful Run-1000 architectural principle while retaining only later revisions that improve engineering correctness, generality, observability, and run management.

The target model is:

**Fixed Softmax Structured Context HONF**

with:

`fixed soft hypergraph organizer -> context-fusion latent bottleneck -> field decoder`

The scientific goals are, in order:

1. physically meaningful and distinguishable hyperedge mechanisms;
2. predictive accuracy competitive with Run 1000;
3. dense computational efficiency close to Run 1000;
4. preserve the infrastructure needed for later sparse execution and inverse design.

Do **not** implement sparse execution changes in the main Stage-7 task. First prove that the modern context-fusion model recovers Run-1000-like accuracy and structure. Sparse execution is the next step only after that gate passes.

Codex must not launch the long formal training run. Prepare the code/profile, validate it, and provide the exact user-controlled Run-1401 command.

---

# 1. Scientific references

Use three frozen references for interpretation.

## 1.1 Run 1000 — primary benchmark

Run 1000 remains the main target because it already demonstrates the desired combination:

- fixed six-edge softmax organization;
- strong environment-region differentiation;
- strong spatially varying query routing;
- high predictive accuracy;
- relatively low parameter count and full-forward cost.

Important Run-1000 complete-split topology reference values:

```text
environment edge-profile cosine  ~0.094
environment effective rank       ~5.24
largest environment occupancy    ~0.292
normalized region separation     ~0.304
query edge-profile cosine        ~0.436
query effective rank             ~3.70
query spatial std                ~0.155
pairwise-map effective rank      ~2.80
```

Run-1000 best complete-split pooled normalized fluid MSE:

```text
~0.000731
```

Historical Run-1000 architecture:

```text
organizer              fixed_projection
K                      6
module assignment      softmax
environment assignment softmax + intrinsic source-centered geometry bias
query routing          softmax
mechanism state        residual/raw organizer state
hyper mechanism encoder OFF
field assembly         context_fusion
learning rate          3e-4 shared/single optimizer
training routing       dense
```

## 1.2 Run 1304 — modern predictive-capacity reference

Run 1304 proves that the modern residual/additive stack has adequate predictive capacity but poor mechanism identifiability.

Its best pooled MSE is about:

```text
0.000952
```

but its mature query/pairwise/additive mechanisms approach rank one:

```text
query effective rank          ~1.16 at epoch 5000
pairwise effective rank       ~1.24
temperature-edge rank         ~1.22
query edge-profile cosine     ~0.956
```

Thus Stage 7 must not trade mechanism quality for lower field loss.

## 1.3 Run 1301 — mechanism-diversity reference only

Run 1301 demonstrates that the current codebase and topology diagnostics can represent high-rank query and edge roles, but exchangeable organization is not part of Stage 7.

Do not develop Run 1301 further in this round.

---

# 2. Stage-7 architectural decision

## 2.1 Keep the Run-1000 structural bottleneck

The primary field path must use context fusion.

Conceptually:

`c_H(q) = sum_k alpha_k(q) V(h_k)`

`c_pair(q) = sum_k alpha_k(q) c_pair,k(q)`

followed by:

`c(q) = c_H(q) + c_pair(q) + c_global(q) + c_direct(q) + c_near(q)`

and:

`F(q) = f_pred(c(q))`

The hypergraph therefore remains a latent information bottleneck. Distinct edge roles increase useful representational capacity; identical edges reduce it.

This is the key inductive pressure Stage 7 is trying to recover.

## 2.2 Fixed softmax organizer

Use:

```text
organizer_mode = fixed_projection
num_hyperedges = 6
edge_selection_mode = all

module_assignment_normalizer = softmax
environment_assignment_normalizer = softmax
query_assignment_normalizer = softmax
```

Keep the fixed organizer's existing source-centered environment geometry bias.

Do not add:

- exchangeable slots;
- adaptive edge selection;
- entmax;
- hard sparsity;
- assignment schedules;
- role regularization;
- edge-count regularization.

## 2.3 Mechanism state

Use:

```text
mechanism_state_mode = residual_concat
use_hyper_mechanism_encoder = false
```

The primary query router should consume the organizer's learned hyperedge state directly, as in Run 1000.

Continue computing/exporting mechanism geometry, source/region, mass, scale, and purity descriptors for diagnostics and future inverse use, but do not force those descriptors through the primary mechanism state.

## 2.4 Primary field assembly

Use:

```text
field_assembly_mode = context_fusion
```

Do not instantiate/use the additive primary prediction path in Stage 7.

Therefore the Stage-7 formal model does not use:

- exact additive edge-field assembly;
- additive edge gate;
- separate additive dense background branch;
- pooled additive background.

Do not delete these implementations. They remain archived/ablation capabilities.

## 2.5 Query routing

Use ordinary learned softmax query routing:

```text
hyper_query_attention_mode = learned
hyper_attention_topk = 0
query_assignment_normalizer = softmax
query_locality_mode = none
```

Keep the existing learned hyper-geometry bias used by the context-fusion decoder.

No new query-locality prior is introduced.

## 2.6 Optimizer

Use one shared optimizer relationship, matching Run 1000:

```text
learning_rate = 3e-4
organizer_learning_rate = null
weight_decay = 1e-5
```

This must use the literal one-group AdamW path.

No LR scheduler.

No staged optimizer changes.

No role freezing.

No curriculum.

---

# 3. Later improvements that Stage 7 explicitly retains

Stage 7 must preserve the following modern revisions where they are applicable to the context-fusion model.

## 3.1 Data/model generality

Retain:

- variable-module-count support;
- module-present masking;
- dynamic module padding/bucketing;
- per-case environment coordinates;
- current generic environment feature handling;
- modern case/domain handling.

## 3.2 ThermalChannel physical coupling correctness

Retain:

- current frozen Stage-A local surrogate integration;
- current predicted-port behavior;
- current local-response fusion;
- current one-step interaction refinement;
- current final-organizer ownership/correctness where applicable to the fixed organizer;
- current interface/port consistency logic.

Do not revert to an older ChannelThermal wrapper merely to reproduce Run 1000.

## 3.3 Training/run engineering

Retain:

- current run manifests and provenance;
- dataset fingerprint recording;
- best-total/best-field/best-temperature/best-predicted checkpoint selection;
- milestone checkpoints;
- prepared/chunked decoding;
- packed scalar CPU transfers;
- current reduced plotting cadence;
- current latest-checkpoint cadence;
- current validation reuse/host-side efficiency fixes;
- resume safety.

## 3.4 Diagnostics

Retain:

- topology evaluator;
- complete-split organizer metrics;
- physical-order and sorted environment plots;
- active-edge-aware routing visualizations;
- source/region coordinates and scales;
- module/environment masses and purities;
- query-routing diagnostics;
- pairwise contribution diagnostics;
- inverse-facing mechanism descriptors.

## 3.5 Sparse-execution infrastructure

Do not remove:

- retained-mass routing utilities;
- gathered-routing code;
- route-count diagnostics.

However, do not change or promote sparse context-fusion execution in Stage 7.

Training and primary Stage-7 evaluation use dense execution.

---

# 4. Later directions explicitly excluded from the Stage-7 formal profile

The following capabilities remain in the codebase but are **not active** in the Stage-7 model:

```text
exchangeable slots
adaptive edge selection
quality/coverage selection
entmax
softmax-to-entmax schedules
sparsity curricula
descriptor-first primary state
hyper-mechanism refinement encoder
exact additive primary prediction
additive background branch
additive edge gate
pooled background
gathered training
query/module top-k training
```

Do not delete these implementations.

The objective is a clean scientific profile, not repository cleanup.

---

# 5. First task: compatibility audit, not model invention

Before editing model code, compare the current fixed/context-fusion path against the historical Run-1000 path.

Use Run-1000 source commit/provenance and the current branch.

Produce a concise table with these categories:

```text
A. mathematically identical / intentionally preserved
B. modern engineering change expected to be prediction-neutral
C. modern scientific/physical change intentionally retained
D. unresolved difference that could contaminate the reproduction
```

At minimum compare:

- fixed organizer equations;
- intrinsic environment geometry bias;
- module-environment auxiliary attention A_me;
- hyper-state construction;
- query-to-hyper attention;
- hyper-value context;
- pairwise context;
- global/direct/near context;
- final context normalization/prediction head;
- current Stage-A coupling;
- current input encoding;
- optimizer construction;
- dataset/normalization path.

### Gate

If the audit discovers a mathematical change in the fixed/context-fusion core that is not required by the retained modern improvements, restore compatibility behind an explicit config choice before formal launch.

If no such change exists, **do not modify organizer.py or decoder.py** merely to create Stage-7 code churn.

---

# 6. Create one explicit full Stage-7 profile

Prefer a dedicated **full forward profile**, not an overlay on `adaptive_sparse_additive.json`.

Suggested path:

```text
HONF_Proj/src/config_core/forward/stage7_structured_context.json
```

The purpose is to avoid inheriting irrelevant adaptive/additive settings.

Base it on the current `enhanced_honf_pairwise.json`, but make the Stage-7 scientific choices and modern run-management settings explicit.

Expected core:

```json
{
  "organizer_mode": "fixed_projection",
  "num_hyperedges": 6,
  "edge_capacity": 0,
  "initial_active_edges": 6,
  "minimum_active_edges": 1,
  "edge_selection_mode": "all",

  "module_assignment_normalizer": "softmax",
  "environment_assignment_normalizer": "softmax",
  "query_assignment_normalizer": "softmax",

  "environment_locality_mode": "none",
  "query_locality_mode": "none",

  "mechanism_state_mode": "residual_concat",
  "use_hyper_mechanism_encoder": false,

  "field_assembly_mode": "context_fusion",

  "routing_execution": "dense",
  "query_edge_limit": 0,
  "query_module_limit": 0,

  "use_hyper_value_context": true,
  "hyper_query_attention_mode": "learned",
  "hyper_attention_topk": 0,
  "hyper_attention_temperature": 1.0,

  "use_hyper_geometry_bias": true,
  "hyper_geometry_bias_scale": 1.0,
  "use_A_me_auxiliary": true,

  "direct_residual_gate_init": 0.0,
  "output_mean_residual_split": false
}
```

Keep the same hidden width, Fourier settings, pairwise kernel configuration, and physical model settings as the Run-1000/current enhanced-HONF path unless the compatibility audit identifies a justified modern difference.

Training:

```json
{
  "learning_rate": 0.0003,
  "organizer_learning_rate": null,
  "weight_decay": 0.00001,
  "amp": false,
  "gradient_clip_norm": 1.0,
  "plot_every_epochs": 50
}
```

Checkpointing:

```json
{
  "save_best": true,
  "save_best_field_mse": true,
  "save_best_temperature_mse": true,
  "save_best_predicted": true,
  "save_latest": true,
  "save_latest_every_epochs": 10,
  "save_epoch_milestones": [500, 1000, 2500, 5000, 7500, 10000]
}
```

Use the current frozen Stage-A checkpoint:

```text
Trained_Results/ThermalChannel/Local_Module_Runs/thermal_disk/Run_0000_base/best_model.pt
```

This makes Run 1401 a **modern Run-1000 counterpart**, not a byte-identical historical reproduction. Document the Stage-A provenance difference explicitly.

---

# 7. Minimal implementation/test policy

The target is to reach a formal launch quickly.

## 7.1 Prefer zero model-code changes

If the current fixed/context-fusion model already matches the intended mathematics, Stage-7 implementation should primarily be:

- one explicit full profile;
- compatibility/provenance report;
- any minimal config/schema additions needed for modern checkpoint cadence;
- launch/dry-run validation.

Do not refactor working core code.

## 7.2 Required tests

Run only focused tests needed to establish launch safety.

At minimum:

1. Stage-7 profile parses/resolves strictly.
2. Fixed organizer forward is finite.
3. Context-fusion prepared/chunked decoding matches one-shot decoding within existing tolerance.
4. A current evaluator can load the historical Run-1000 checkpoint and reproduce its known topology/accuracy outputs on a small deterministic case subset.
5. One-group optimizer inventory is confirmed at `3e-4`.
6. Current Stage-A checkpoint resolves.
7. Dataset fingerprint and 600/90 split resolve.
8. `train.py --dry-run` passes.

## 7.3 Smoke training

If no model prediction code changes are required:

```text
DO NOT launch a smoke training run.
```

A dry-run plus focused maintained tests is sufficient.

If a compatibility bug requires model-code changes, run at most:

```text
1 epoch
<=2 train batches
<=1 validation batch
```

only to confirm correctness.

Do not infer scientific performance from the smoke run.

---

# 8. Formal experiment: Run 1401

Codex prepares this command but does not launch it.

Run from `HONF_Proj`.

Suggested command:

```bash
python train.py   --config src/config_core/forward/stage7_structured_context.json   --local-checkpoint Trained_Results/ThermalChannel/Local_Module_Runs/thermal_disk/Run_0000_base/best_model.pt   --run-id 1401   --run-name stage7_modern_structured_context   --epochs 5000   --device cuda:0   --yes
```

Use the user's chosen GPU if not `cuda:0`.

No `--initialize-checkpoint`.

This must be a fresh forward-model run with the current frozen Stage-A dependency.

No staged learning.

No optimizer changes during the run.

No routing-mode changes during the run.

No topology loss.

---

# 9. Why 5000 epochs first

Run 1304 shows that 5000 epochs are enough to establish modern predictive capacity.

Run 1000 also has a meaningful matched-budget validation history through epoch 5000.

Therefore the first decision point is:

```text
epoch 5000
```

If Run 1401 is structurally strong and follows the Run-1000 accuracy trajectory, resume the **same run** to 10000 epochs.

Do not start a second formal run for the extension.

Suggested resume:

```bash
RUN_DIR=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1401_<timestamp>_stage7_modern_structured_context

python train.py   --config src/config_core/forward/stage7_structured_context.json   --resume-checkpoint "$RUN_DIR/latest_model.pt"   --local-checkpoint Trained_Results/ThermalChannel/Local_Module_Runs/thermal_disk/Run_0000_base/best_model.pt   --epochs 10000   --device cuda:0   --yes
```

---

# 10. Milestone evaluation

Evaluate the saved checkpoints at:

```text
500
1000
2500
5000
```

Use the existing complete-split evaluators.

At 5000, compare:

- Run 1401;
- Run 1000;
- Run 1304 best/latest where useful.

Do not use selected-edge count as a mechanism-quality metric.

Primary structure metrics:

```text
environment edge-profile cosine
environment effective rank
largest environment occupancy
region separation
query edge-profile cosine
query effective rank
query spatial std
pairwise-map cosine/rank
```

Because Stage 7 uses context fusion, exact additive `pred_field_by_edge` is not a primary metric.

Do not invent additive proxies.

---

# 11. Stage-7 acceptance gates

These are intended as practical decision gates, not optimization targets.

## 11.1 Environment organization

Preferred by epoch 5000:

```text
environment profile cosine      < 0.20
environment effective rank      > 3.5
largest environment occupancy   < 0.50
normalized region separation    > 0.20
```

Run 1000 remains the stronger reference:

```text
cosine ~0.094
rank ~5.24
occupancy ~0.292
region separation ~0.304
```

## 11.2 Query organization

Preferred:

```text
query profile cosine   < 0.55
query effective rank   > 3.0
clear spatial variation
```

Run 1000 reference:

```text
query cosine ~0.436
query rank ~3.70
```

## 11.3 Accuracy

At matched training epochs, compare the validation curve directly against Run 1000.

Preferred at epoch 5000:

```text
Run-1401 trailing field-MSE median <= ~1.10 x Run-1000 matched-epoch median
```

Run 1401 should also be competitive with or better than Run 1304 at comparable budget.

For complete-split frozen checkpoints, report the values rather than forcing an artificial threshold where no historical Run-1000 epoch-5000 frozen checkpoint exists.

If Run 1401 remains structurally strong and its accuracy trajectory is close to Run 1000 at 5000, continue the same run to 10000 for the definitive comparison.

## 11.4 Efficiency

Dense full-forward target:

```text
<= 1.10 x Run-1000 full-forward latency
```

Parameter count/checkpoint size should remain close to the Run-1000/context-fusion class and should be materially below the exchangeable model.

No new dense `Q x E` additive-background path should execute.

---

# 12. Sparse execution is explicitly deferred

Do not spend Stage-7 implementation time modifying context-fusion sparse execution before Run 1401 proves the base model.

After Run 1401 passes accuracy/topology gates, open a separate sparse-execution step using the existing retained-mass infrastructure.

The intended future inference rule is retained-mass soft pruning, for example:

```text
query retained mass target rho_q = 0.98
```

with renormalization over retained routes.

The future sparse implementation should prune context-fusion hyper-value and pairwise edge reductions without changing training.

Do not add entmax or hard sparse training.

---

# 13. Mechanism interpretation and inverse readiness

Exact additive physical fields are no longer required for the primary forward model.

For Stage 7, preserve:

```text
hyper_state
A_mh
A_eh
hyper_source_coords
hyper_region_coords
source/region scales
module/environment masses
module/environment purities
query routing
pairwise contribution diagnostics
mechanism descriptors
prepared organizer state
```

These provide a stable fixed-K mechanism coordinate for the next inverse model.

Do not implement a counterfactual edge-attribution tool in the main Stage-7 task unless it is trivial and does not delay Run 1401.

Counterfactual mechanism effects can be added after the formal context-fusion model is validated.

---

# 14. Stop conditions

Stop and report rather than broadening Stage 7 if:

1. the current fixed/context-fusion path cannot load/evaluate Run 1000 correctly;
2. the compatibility audit finds an unexplained scientific difference between historical and current context-fusion mathematics;
3. implementing Stage 7 would require major organizer/decoder rewrites;
4. the dedicated profile cannot reproduce the intended single-group `3e-4` optimizer;
5. current Stage-A or dataset artifacts are unavailable.

Do not respond to these failures by adding new losses, locality terms, or another organizer design.

---

# 15. Required deliverables

Codex should produce:

1. `HONF_Proj/UpgradePlan/Stage7_Modern_Structured_Context_GoalMode.md`
2. `HONF_Proj/diagnostics/Stage7_Run1000_Compatibility_Audit.md`
3. `HONF_Proj/src/config_core/forward/stage7_structured_context.json`
4. Stage-7 dry-run provenance JSON
5. focused validation/test report
6. exact Run-1401 launch command
7. exact same-run resume command to 10000 epochs
8. concise closeout stating that no long formal run was launched

If no model-code change is required, say so explicitly. A profile-only Stage-7 implementation is a successful outcome.

---

# 16. Completion definition

Stage 7 is complete when the repository is ready for one clean formal experiment:

```text
Run 1401 — modern structured context HONF
```

with:

- fixed six-edge softmax organizer;
- Run-1000-style context-fusion bottleneck;
- residual/raw hyperedge state;
- no extra mechanism encoder;
- uniform `3e-4` optimization;
- current modern ThermalChannel/data/checkpoint infrastructure;
- dense training/reference execution;
- complete topology evaluation available;
- sparse execution deferred until the base model is proven.

The formal scientific question is:

> Can the modernized Run-1000 architecture naturally recover meaningful hyperedge roles while matching Run-1000 predictive accuracy and retaining modern engineering improvements?

If yes, stop forward-architecture redesign and move next to sparse inference and inverse-model development.
