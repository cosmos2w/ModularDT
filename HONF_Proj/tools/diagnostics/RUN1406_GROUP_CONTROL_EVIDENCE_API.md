# Run-1406 group-control evidence API

This contract belongs to the CPU-side evidence tools. It does not require a
model or device and does not alter the Run-1406 backend. The comparison tool
requests the debug payload in one untimed full-Q forward per checkpoint/case;
all timed full-forward, preparation, prepared P2 decode, and training-step
calls disable map materialization and profiling.

## Opt-in output

The preferred forward keyword is:

```text
return_group_control_maps=True
```

The tooling falls back to `return_routing_maps=True` only when that keyword is
explicitly exposed by the loaded model. A backend may return the payload in
`group_control_debug` or `interaction_aux`; the same mappings are accepted at
`prepared_state.prepared.interaction_aux` and equivalent prepared containers.
Missing maps are reported as unavailable; they are not reconstructed from
dense counts.

Required Run-1406 dimensions are `K=6` and control width `D=16`. The preferred
array names are:

| Key | Shape | Meaning |
|---|---|---|
| `group_control_module_incidence` | `[B,M,K]` | non-negative module memberships `A_m` |
| `group_control_environment_incidence` | `[B,E,K]` | non-negative environment memberships `A_e` |
| `group_control_query_routing` | `[B,Q,K]` | non-negative query routing `alpha_qk` |
| `group_control_module_overlap` | `[B,Q,M]` | `rho_M`; may be derived as `alpha @ A_m.T` |
| `group_control_environment_overlap` | `[B,Q,E]` | `rho_E`; may be derived as `alpha @ A_e.T` |
| `group_control_module_control_moment` | `[B,Q,M,16]` | mass-weighted module control moment `n_M` |
| `group_control_environment_control_moment` | `[B,Q,E,16]` | mass-weighted environment control moment `n_E` |
| `group_control_h` | `[B,K,16]` | group control state; exact `n` maps are derived as `alpha * A * h` when explicit moment maps are absent |
| `group_control_module_centres` | `[B,K,2]` | detached positive-occupancy group centres |
| `group_control_environment_centres` | `[B,K,2]` | detached positive-occupancy group centres |

The evidence helper also accepts documented spelling variants and derives
positive-support overlaps when only memberships and query routing are present.
Empty-group centres should be NaN/unavailable, never silently assigned to an
origin.

## Phase ledger

`group_control_phase_ledger` is an optional mapping with `P0`, `P1`, `P2`, and
optional `P2_consistency` entries. Each phase should contain `module` and
`environment` records with these fields:

```text
logical_path_count
unique_pair_count
actual_fine_call_count
module_mlp_rows
environment_geometry_network_rows
environment_content_rows
scalar_control_rows
source_projection_rows
forward_call_count
checkpoint_recompute_count
valid_pair_denominator
padded_pair_denominator
```

The environment row names are still used in both source records so the report
can present one stable schema. Backends may emit receiver-chunk records under
`chunks`, `receiver_chunks`, `chunk_records`, or `records`. The helper sums
each numerator/denominator over those records before reporting multiplicity or
ratios. It never averages per-chunk ratios. A missing P0/P1 actual record
remains `status=unavailable`; the tool does not infer calls from the P2 map.
For the current Run-1406 reader, an explicit module `fine_rows` field is also
reported as module-MLP rows because the module fine function has one MLP row
per retained module pair; no environment rows are relabelled as module rows.
`fine_rows` is not relabelled as a valid-pair denominator: that denominator
must be emitted explicitly (`valid_pair_denominator`, `valid_pairs`, or
`dense_valid_pair_count`) so executed work cannot be mistaken for a dense
reference population.

For the current Thermal wrapper, the same fields may be emitted as explicit
raw keys in an auxiliary mapping. The adapter maps only these prefixes:

```text
initial_port_group_control_*  -> P0
provisional_group_control_*   -> P1
group_control_*               -> P2
port_global_group_control_*   -> P2_consistency
```

Thus raw unprefixed P2 keys never synthesize P0/P1, and absent provisional or
consistency keys remain unavailable.

The ledger distinguishes:

```text
logical q -> group -> source paths       cheap control incidence
unique q -> source pairs                 one fine interaction candidate
actual_fine_call_count                  executed backend fine rows
forward_call_count / checkpoint_recompute_count
```

The exact support summaries are:

```text
P_logical = sum(q,s,k) 1[alpha_qk > 0] 1[A_sk > 0]
P_unique  = sum(q,s)   1[rho_qs > 0]
D         = P_logical / max(P_unique, 1)
R         = P_unique / (Q * N_valid)
```

`D` is cheap-control multiplicity, not fine-call count. `R<1` is not a
Run-1406 continuation requirement: the model may read all sources once and
still win by avoiding duplicated H-wide group evaluations or by reducing
other measured work.

## Board semantics

`render_group_control_interaction_board.py` shows module/environment incidence,
centres, query routing, selected logical triples, one solid line per unique
query/source candidate with `rho`, control moments when available, phase rows,
and measured prepared-decode time copied from the map metadata. Dashed paths
are logical learned interaction routes; solid paths are unique fine candidates.
Neither routing magnitude, a group centre, nor a line on the board establishes
physical causality.
