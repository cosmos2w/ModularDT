# Stage 6 Closeout

## Outcome

Stage 6 closes as a **scientifically informative no-go for the proposed frozen role interventions**.

- Selected profile: **S0 control** (`scale=0.35`, `query_locality_mode=none`).
- Rejected: S1 (`scale=0.70`) because pooled MSE increased by about 98–104% in the 20-case screen.
- Rejected: S2 (`gaussian_bounded`, strength `0.25`) because full-split best-checkpoint MSE increased 2.59%, `v` MSE increased 8.57%, and routing changes were not material.
- Not run: S3, because neither constituent intervention was individually promising.

The final profile is [stage6_fixed_role_consistent_additive.json](../src/config_core/forward/experiments/stage6_fixed_role_consistent_additive.json). It is intentionally an explicit, checkpoint-compatible S0 control. It does not claim that Stage 6 achieved Run-1000-like role separation.

## Compatibility and dry-run gates

The Run-1302 best checkpoint (`epoch 2497`) was dry-run on case `0273` with 256 queries and GPU 2.

| Gate | Result |
|---|---|
| Prediction after explicit S0 override | bitwise equal; maximum difference `0.0` |
| State-dict structure | unchanged, 251 keys |
| Exact additive closure | maximum absolute residual `5.96e-8` |
| Prediction finite | yes |
| Fixed mechanism order | `K=6` |
| Prepared state | available |
| Per-edge field output | `[1,256,6,5]` |

Inverse-facing prepared-state shapes:

| Tensor | Shape |
|---|---|
| `global_token` | `[1,256]` |
| `hyper_state` | `[1,6,256]` |
| `mechanism_descriptor_features` | `[1,6,16]` |
| `A_mh` | `[1,12,6]` |
| `A_eh` | `[1,192,6]` |
| source/region coordinates | `[1,6,2]` each |
| source/region scales | `[1,6,2]` each |
| module/environment masses and purities | `[1,6]` each |

The prepared organizer also retains query-routing inputs and diagnostics. Fixed-projection channels give the future inverse model a stable ordered mechanism coordinate. No inverse code was changed.

## Code/configuration result

The only public configuration addition is:

```text
query_locality_strength: float | null = null
```

- missing or `null`: inherit `environment_locality_strength`, exactly preserving historical arithmetic;
- nonnegative float: independently scale the existing query-locality bias;
- no parameter tensors or state-dict keys are added.

Evaluation tools now support label-scoped frozen overrides for content scale, query-locality mode, and query-locality strength. They record override and checkpoint provenance. No training loss, optimizer, organizer, exchangeable path, additive assembly, or persisted checkpoint behavior changed.

Cheap topology scalars were not added to every training batch. The existing full-split topology evaluator already computes the decisive environment, region, query, pairwise, and additive metrics without adding training synchronization or memory cost.

## Validation

Focused configuration/override tests:

```bash
conda run -n ModularDT pytest -q \
  tests/test_config_and_registry.py \
  tests/test_forward_upgrade_config.py \
  Case_ThermalChannel/tests/test_forward_variable_modules.py \
  Case_ThermalChannel/tests/test_topology_quality_evaluator.py
```

Result: `47 passed in 2.67s`.

Maintained additive, organizer, chunked/prepared decoding, topology, checkpoint, and retained-mass regression set:

```bash
conda run -n ModularDT pytest -q \
  tests/test_forward_additive.py \
  tests/test_core_contract.py \
  tests/test_gathered_routing.py \
  tests/test_sparse_routing.py \
  tests/test_topology_signature.py \
  tests/test_config_and_registry.py \
  tests/test_forward_upgrade_config.py \
  Case_ThermalChannel/tests/test_forward_variable_modules.py \
  Case_ThermalChannel/tests/test_resources_and_checkpoint.py \
  Case_ThermalChannel/tests/test_topology_quality_evaluator.py
```

Result: `121 passed in 4.75s`.

Profile dry-run:

```bash
conda run -n ModularDT python diagnostics/verify_stage6_profile.py \
  --checkpoint Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1302_20260821_142505_stage5_fixed_softmax_modern/best_by_field_mse_model.pt \
  --device cuda:2 \
  --case-id 0273 \
  --queries 256 \
  --output diagnostics/stage6_profile_dry_run.json
```

Result: passed all compatibility, closure, finiteness, shape, and inverse-readiness checks.

## User-controlled launch command

Codex did **not** launch a long training run. If a clean S0 replication/control is still desired, use:

```bash
cd HONF_Proj

python train.py \
  --config src/config_core/forward/adaptive_sparse_additive.json \
  --experiment-overlay src/config_core/forward/experiments/stage6_fixed_role_consistent_additive.json \
  --local-checkpoint Trained_Results/ThermalChannel/Local_Module_Runs/thermal_disk/Run_0000_base/best_model.pt \
  --run-id 1401 \
  --run-name stage6_fixed_role_consistent_additive_s0_control \
  --epochs 1500 \
  --device cuda:2 \
  --yes
```

Do not use `--initialize-checkpoint` for a clean formal comparison. This command is explicitly a Run-1302-class replication, not an evidence-backed S1/S2 role intervention.

## Milestone evaluation plan

Evaluate epochs `250`, `500`, `1000`, and `1500` together with the complete 90-case split. Replace `<RUN_DIR>` with the created Run-1401 directory.

```bash
python diagnostics/evaluate_stage5_accuracy.py \
  --checkpoint e250=<RUN_DIR>/epoch_250_model.pt \
  --checkpoint e500=<RUN_DIR>/epoch_500_model.pt \
  --checkpoint e1000=<RUN_DIR>/epoch_1000_model.pt \
  --checkpoint e1500=<RUN_DIR>/epoch_1500_model.pt \
  --device cuda:2 --split test --max-cases 90 \
  --output-dir diagnostics/stage6_run1401_accuracy

python diagnostics/evaluate_topology_quality.py \
  --checkpoint e250=<RUN_DIR>/epoch_250_model.pt \
  --checkpoint e500=<RUN_DIR>/epoch_500_model.pt \
  --checkpoint e1000=<RUN_DIR>/epoch_1000_model.pt \
  --checkpoint e1500=<RUN_DIR>/epoch_1500_model.pt \
  --device cuda:2 --split test --max-cases 90 \
  --output-dir diagnostics/stage6_run1401_topology

python diagnostics/evaluate_retained_mass_pruning.py \
  --checkpoint e1500=<RUN_DIR>/epoch_1500_model.pt \
  --device cuda:2 --split test --max-cases 90 \
  --query-mass-floor 0.98 --module-mass-floor 0.95 \
  --minimum-query-routes 1 --minimum-module-routes 1 \
  --output-dir diagnostics/stage6_run1401_pruning
```

Only continue the same run to 2500 epochs if field accuracy remains healthy **and** environment/query/additive topology does not repeat Run 1302's late deterioration. The scientific next-step focus should be a separately specified, controlled late-role-preservation experiment; the frozen screen does not support spending a formal run on S1, S2, or S3.
