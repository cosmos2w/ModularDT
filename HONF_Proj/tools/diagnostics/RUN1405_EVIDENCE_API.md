# Run-1405 evidence API

The Run-1405 comparison entry point requests one untimed debug forward after
the timed phases. The model must expose the following tensors in its ordinary
`interaction_aux` output (the prepared state's `interaction_aux` is accepted
as a fallback):

| Key | Shape | Meaning |
|---|---|---|
| `fixed_group_module_incidence` | `[B,M,6]` | non-negative `A_m`; inactive module rows must be zero |
| `fixed_group_environment_incidence` | `[B,E,6]` | non-negative `A_e` |
| `fixed_group_module_centres` | `[B,6,2]` | module-weighted group centres `r_m` |
| `fixed_group_environment_centres` | `[B,6,2]` | quadrature-weighted environment centres `r_e` |
| `fixed_group_query_routing` | `[B,Q,6]` | non-negative query routing `alpha_qk` |

The prepared state should retain the encoded geometry at
`prepared_state.prepared.encoded`:

```text
module_centers [B,M,2]
module_present [B,M]
env_coords     [B,E,2]
env_weights    [B,E]
```

The backend may additionally return these explicit triple tensors, although
the evidence tool derives and validates the same relations from the positive
supports:

```text
fixed_group_module_triple_{batch,query,group,source}
fixed_group_environment_triple_{batch,query,group,source}
```

`P_M` and `P_E` are computed exactly as

```text
P_M = sum(q,k: alpha[q,k] > 0) count(i: A_m[i,k] > 0 and module_present[i])
P_E = sum(q,k: alpha[q,k] > 0) count(j: A_e[j,k] > 0)
R_M = P_M / (Q * M_active)
R_E = P_E / (Q * E)
```

The support summaries are:

```text
sQ = mean_q count(k: alpha[q,k] > 0)
sM = mean_i count(k: A_m[i,k] > 0), over active modules
sE = mean_j count(k: A_e[j,k] > 0), over environment sources
```

The saved NPZ also contains exact selected-query triples and source fan-out
arrays for the board. These are learned interaction routes; they must not be
described as physical causality or field-value influence.

## Epoch-50 training-health gate

The comparison entry point requires an explicit Run-1405 `metrics.csv` via
`--metrics-1405`. The CPU parser requires unique rows for epochs 1 through 50,
finite train/validation losses and field MSE, and finite aggregate plus
`encoder`, `backend`, `head`, and `local_coupling` gradient/update diagnostics
at epoch 50. Each major group must have positive finite gradient and update
evidence. It also requires the strict last-10 versus first-10
`val_field_mse` median improvement and uses the predeclared instability rule:
at most one last-10 value may exceed twice the last-10 median.

The health result is included in `training_health` and in the continuation
criteria. Missing or malformed health evidence cannot pass continuation; the
Run-1405-versus-Run-1804 fidelity gate remains a separate explicit metric gate.

## CPU-only map/board workflow

`fixed_group_evidence.py` validates the contract and computes the metrics
without importing a model or touching CUDA. `render_fixed_group_interaction_board.py`
renders incidence, centres, query routing, and selected actual triples. The
comparison manifest can be supplied directly:

```text
python tools/diagnostics/render_fixed_group_interaction_board.py \
  --comparison diagnostics/generated/run1405_epoch50/comparison.json \
  --output-dir diagnostics/generated/run1405_epoch50/board
```

Plan-only comparison validation also requires the explicit metrics path:

```text
python tools/diagnostics/run_run1405_epoch50_comparison.py \
  --checkpoint-1405 /path/run1405/epoch_0050_model.pt \
  --checkpoint-1804 /path/run1804/epoch_0050_model.pt \
  --checkpoint-1401 /path/run1401/context.pt \
  --metrics-1405 /path/run1405/metrics.csv \
  --output diagnostics/generated/run1405_epoch50/plan.json \
  --plan-only
```

The timing entry point has an explicit `--plan-only` mode. It requires paths
for all three labels (`--checkpoint-1405`, `--checkpoint-1804`, and
`--checkpoint-1401`) and records the 1401 path/epoch as context without
silently substituting a checkpoint or using it for the Run-1405-vs-1804
epoch-50 fidelity gate.
