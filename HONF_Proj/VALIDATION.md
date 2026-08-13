# Migration validation record

This file records reproducible evidence used to accept the move from
`1_Demo_ChannelThermal/src_HONF_CL`. Generated runs contain the detailed
configuration and metrics; this document keeps the compact release-level
result.

## Reference resources

| Resource | Size | SHA-256 | Contract |
|---|---:|---|---|
| Stage-A packed HDF5 | 60,424,808 bytes | `51a875857e3b163b9a2fcb1e4a04ea3f343f6ff4c71b3709232e395e36bac5cf` | 1,034 cases; 919/115 train/test |
| Global packed HDF5 | 265,224,376 bytes | `4224093c22a67af4adfecc8b21d53548e4263ec2254c230dc83c89526b36da05` | 690 cases; 600/90 train/test |

Both files passed required-root-key, exact-size, and full-file SHA-256 checks.

## Exact checkpoint parity

On 2026-08-12, the historical global `best_predicted_model.pt` was evaluated
on test case `0273` in `predicted` mode by the old and modular evaluators on the
same CUDA device. All five exported arrays were bit-for-bit equal:

- `pred_field_grid [64,128,5]`;
- `gt_field_grid [64,128,5]`;
- `pred_internal_temperature [12,3096,1]`;
- `pred_interface [12,64,2]`;
- `pred_port_condition [12,64,5]`.

The metrics CSV, hypergraph diagnostics JSON, and three PNG figures also had
identical SHA-256 hashes. The only summary differences were timestamped output
paths. The fixed-case fluid field RMSE was `0.06501651920455319`, temperature
RMSE was `0.1450419914324`, and fluid relative L2 was
`0.026078676735220532` in both implementations.

The historical Stage-A `latest_model.pt` was likewise evaluated on local test
case `local_0920`. Every reported scalar was equal, including internal RMSE
`0.12067130128755493`, interface RMSE `0.05104559924212458`, surface-temperature
RMSE `0.062422842751670954`, and normal-flux RMSE `0.03625872456079145`. Both
generated PNG files were byte-identical.

## Automated and smoke checks

- Editable root and ThermalChannel installations import successfully from
  outside the repository.
- The final combined root/case suite passes (`49 passed` on 2026-08-12),
  including a synthetic second case loaded through a dotted plugin factory
  without core or dispatcher changes. Both source trees also pass
  `compileall`, and every JSON document included in the static pass parses.
- Consolidated neural utilities have explicit equivalence tests for both
  historical Fourier conventions and both checkpoint-visible MLP layouts.
- Forward and Stage-A one-epoch/one-batch smoke runs completed on both CUDA and
  CPU and wrote configs, manifests, histories, plots, and enabled checkpoints.
- Both maintained decoder profiles have direct CUDA coverage: the
  `enhanced_honf_pairwise` run/resume smoke completed two epochs, and the
  `hyper_plus_global_near` smoke completed one bounded train batch plus both
  validation passes with the new Stage-A checkpoint. Its autonomous checkpoint
  was then reconstructed for test case `0273`; the evaluator wrote ten
  inventoried artifacts without checkpoint fallback.
- Forward and local evaluation smoke runs completed under their source runs;
  manifest evaluation-child recording was verified.
- The generic evaluation dispatcher preserved repeated `--Run_ID` and
  `--label` arguments in an actual two-run comparison. The resulting immutable
  manifest inventories 18 tables, figures, and log artifacts.
- Global fallback/teacher/predicted/mixed modes, frozen Stage A, corrected
  flux/refinement, decoder call efficiency, self-contained checkpoint loading,
  plan canonicalization, inactive slots, permutation behavior, and NPZ plan
  round-trip checks pass.
- Source boundary scan finds no `channelthermal`, HDF5, or Matplotlib import in
  `honf_forward_core`, `honf_runtime`, or the generic entry points.

## User-run performance gate

Long validation runs are intentionally not part of this repository-build
check. Per the final validation instruction, the user will execute the
2,000-epoch Stage-A run and the subsequent `hyper_plus_global_near` and
`enhanced_honf_pairwise` runs. The maintained profiles under
`src/config_core/forward/` carry those full budgets; the README gives the exact
launch, resume, evaluation, and comparison commands.

Before that instruction changed, Stage A reached epoch 343 and wrote a valid
resumable checkpoint. It was stopped deliberately and its manifest records the
reason; it is a diagnostic artifact, not a completed performance result. A
bounded CUDA evaluation loaded that checkpoint and produced both local plots
and the complete metric document. No full global performance run was started.

### Frozen equal-budget references

The matching Stage-A reference is the historical mixed-source
`Saved_Model_LocalModule/Run_0003_20260507_224352` run (same 128-wide,
16-port-latent, four-layer model and weighted losses). Through epoch 2,000 its
best validation values are:

| Metric | Best value | Epoch |
|---|---:|---:|
| `val_loss_total` | 0.030583525069479672 | 1772 |
| `val_loss_internal` | 0.004394844487168879 | 1894 |
| `val_loss_interface` | 0.025256912649240133 | 1360 |

The matching pairwise Stage-B reference is
`Saved_Model_HONF_CL/Global/Run_0000_20260811_104337_global_honf_validated_core`
(hidden width 256, six hyperedges, frozen Stage A, predicted-port mode).
Through epoch 2,000 its best values are:

| Metric | Best value | Epoch |
|---|---:|---:|
| `val_loss_total` / `val_predicted_loss_total` | 0.01402169931679964 | 1990 |
| `val_field_mse` / `val_predicted_field_mse` | 0.0021629363182000816 | 1987 |
| `val_temperature_mse` / `val_predicted_temperature_mse` | 0.0028102484066039324 | 1990 |

These archived CSVs are immutable inputs to the executable gate below. The
near-field-only global profile has no like-for-like historical run, so it is
reported on the same 90-case test split and compared alongside the new
pairwise run rather than being assigned a misleading historical threshold.

For lower-is-better history columns, the executable gate is
`tools/compare_training_metrics.py`. It compares the best finite value in each
history through the same `--max-epoch`, reports the epoch of each minimum, and
returns a nonzero status if any candidate regression exceeds the requested
relative tolerance. For example:

```bash
python tools/compare_training_metrics.py BASELINE.csv CANDIDATE.csv \
  --metric val_loss_total --max-epoch 2000 --relative-tolerance 0.01
```
