# HONF Run 1501 executor and Run 1502 development

## Decision summary

This task used only the fixed Run-1501 saved-best-total epoch-444 and exact
epoch-500 checkpoints. Run 1501 maturation continued independently; this
Goal-mode task did not wait for or monitor epoch 5000.

The Run-1501 accounting path is corrected from the backend through evaluator
reduction. The exact support-union block executor preserves the learned
operator and tested first derivatives, but its measured benefit is decided by
latency and memory rather than logical row counts; the historical rectangular
reader remains the default. The sole Run-1502 change is final environmental
source refinement from entmax-1.5 to masked sparsemax. Its frozen eight-case
gate materially reduced environmental overlap without empty-support pathology,
so exactly one matched Run 1502 was launched to the epoch-50 review stop. The
epoch-50 learning/evaluation gate rejected continuation; the run ended normally
at 50 and was not resumed to 150 or 500.

The 90-case split remains a development holdout, not an untouched final
benchmark. Learned incidence is surrogate-model organization, not physical
causality, and group labels are permutation-ambiguous.

## 1. Evidence and accounting corrections

### Corrected contracts

- Module rectangular work is `Q * M_pad`; registered group capacity `K=12` is
  not its denominator. Valid module rows and padding are reported separately.
- Scalar work ledgers and padded rows are summed across inner receiver chunks;
  query-local arrays are concatenated, including equal-width chunks.
- `predict_case` now aggregates Run-1501 ledgers across outer evaluator chunks.
- Rectangular QE geometry rows are the executed `Q * E` rectangle, not the
  positive-support mask. Content-dot rows are geometry rows times the four
  attention heads.
- Timed physical forwards keep routing maps and detailed ledgers off. Untimed
  parity reads retain them. Prepared-P2 GPU decode, full GPU forward, and
  application evaluator scopes remain distinct.
- Population artifacts retain `M_active`, `M_pad`, active source IDs, original
  query-grid indices and bounds, and explicit full-grid versus deterministic
  subset provenance.

The focused synthetic contract has `M_pad=5`, `K=12`, `Q=11`, and inner chunks
`[3,3,3,2]`. Chunked and unchunked predictions and ledgers agree, so the test
would fail if `Q*K` were substituted for `Q*M_pad`.

### Real Q1024/Q8192 cross-check

The exact e500 checkpoint was evaluated on cases 0273 and 0653. The Q1024 pass
used unequal outer chunks (`333,333,333,25`); the Q8192 pass used eight outer
chunks of 1024, while the model retained its native inner chunk of 128.

| Pooled mean over two cases | Q1024 subset | Q8192 original grid |
|---|---:|---:|
| Mean query degree Kq | 3.38916 | 3.36609 |
| Module support RM | 0.84447 | 0.84206 |
| Environment support RE | 0.66580 | 0.66500 |
| Module rectangular rows | 12,288 | 98,304 |
| Module padded rows | 8,192 | 65,536 |
| QE geometry rows | 196,608 | 1,572,864 |
| QE content-dot rows | 786,432 | 6,291,456 |

Every executed-row ledger scales exactly by eight. The small support-statistic
difference is the measured difference between the deterministic Q1024 subset
and the full original grid, not a chunk-accounting inconsistency.

Representative e500 boards use all 8192 original grid coordinates, hide padded
module slots, list active module IDs, use one physical coordinate frame, and
separate dominant group from exact Kq. The maintained renderer emits both PNG
previews and vector PDFs.

## 2. Exact support-union block executor

For each observed query support signature, the executor packs positive group
support into one integer word, gathers queries with that signature, forms the
unique physical-source union once, and evaluates one smaller rectangular QM or
QE block. It never enumerates `2^12` masks, normalizes per group, or duplicates
a physical pair through multiple groups. Numeric `alpha`, source memberships,
`rho`, group controls, source measures, geometry, K/V values, and output biases
remain live. Empty source-type support returns exactly zero.

`rectangular_reference` remains the historical default. `support_blocks` is an
explicit evaluation-time policy; no Run-1501 training process or checkpoint is
modified. The implementation deliberately has no custom kernel or detached
cross-step cache.

### Exactness evidence

Focused tests cover multiple signatures, shared sources, zero support for one
source type, mixed batch sizes and coordinate scales, maps-on/maps-off forward
parity, and connected first gradients. On real predicted-port batches with 32
queries per anchor, the maximum field difference was `1.43e-6`, maximum query
gradient difference `7.15e-7`, and maximum group-code gradient difference
`1.34e-7` across e444/e500 and cases 0273/0653.

### Q8192 benchmark decision

The matched benchmark used GPU 1, receiver chunks of 128, three warmups and ten
synchronized repetitions per scope. Times below are median wall milliseconds;
ratios are `support_blocks / rectangular`.

| Checkpoint / case | Full GPU forward rect / blocks (ratio) | Prepared P2 rect / blocks (ratio) | Evaluator rect / blocks (ratio) | Max output difference |
|---|---:|---:|---:|---:|
| e444 / 0273 | 227.47 / 1479.82 (6.51x) | 169.50 / 1324.53 (7.81x) | 272.85 / 1527.19 (5.60x) | 5.72e-6 |
| e444 / 0653 | 222.39 / 1667.82 (7.50x) | 178.61 / 1505.91 (8.43x) | 263.35 / 1706.93 (6.48x) | 8.34e-6 |
| e500 / 0273 | 226.40 / 1517.03 (6.70x) | 170.21 / 1434.06 (8.43x) | 270.79 / 1575.24 (5.82x) | 9.36e-6 |
| e500 / 0653 | 236.36 / 1684.24 (7.13x) | 167.31 / 1507.34 (9.01x) | 268.25 / 1674.10 (6.24x) | 1.67e-5 |

CUDA-event medians closely matched wall medians. Peak incremental allocation
was also higher for support blocks in every matched scope, by approximately
2.0x--2.6x. This occurred despite the intended row reduction: for example,
e500/0273 reduced QM rows from 98,304 to 22,609 and QE rows from 1,572,864 to
1,108,279; e500/0653 reduced them to 31,300 and 983,631. The exact support-union
algorithm is therefore retained as a tested, opt-in reference implementation,
not selected as the production executor. The rectangular executor remains the
Run-1501/1502 default. No custom-kernel work is justified by this task.

## 3. Frozen Run 1502 diagnostic

The fixed panel was 0273, 0653, 0644, 0686, 0277, 0291, 0680, and 0281, with
Q1024 predicted-port queries on GPU 2. Only final environmental refinement was
recomputed with masked sparsemax. Environmental proposal, module proposal and
final assignment, and query routing retained their Run-1501 normalizers.

| Checkpoint | Parent RE | Candidate RE | Relative RE reduction | Parent / candidate RM | Parent / candidate Kq | Mean output relative RMS |
|---|---:|---:|---:|---:|---:|---:|
| e444 saved best | 0.63966 | 0.44224 | 30.86% | 0.75081 / 0.74995 | 3.27344 / 3.23328 | 0.03049 |
| e500 exact | 0.64621 | 0.45082 | 30.24% | 0.76549 / 0.76532 | 3.28772 / 3.24146 | 0.02949 |

Both checkpoints had zero empty phase groups, empty environment rows, empty
query rows, and nonfinite cases. This is a structural gate, not a retrained
accuracy forecast. The approximately 30% RE reduction with no obvious support
failure justified one managed Run 1502; the roughly 3% frozen output change
was not interpreted as either physical improvement or rejection.

## 4. Run 1502 training evidence

Identity:

```text
run-id: 1502
run-name: sparse_incidence_environment_sparsemax
architecture: sparse_incidence_group_control_honf
sole change: interface_model.environment_refinement_normalizer=sparsemax
device: cuda:2 with ordinary CUDA visibility
```

The run started from scratch with the Run-1501 seed, data, optimizer, losses,
K=12, D=16, geometry, fine kernels, and rectangular executor. It was initially
bounded to epoch 50. No parent warm start, distillation, second candidate, or
normalizer/sparsity sweep was used.

At epoch 10 all recorded losses and 280 floating checkpoint tensors were
finite. Validation total loss was 3.07310 and the parameter update norm was
0.17115, confirming that the requested path was learning rather than merely
loading. The NaN update/gradient cells at unsampled epochs are intentional:
those diagnostics were scheduled only at epochs 1, 2, 5, 10, 20, and 50.

The managed run completed epoch 50 with validation total loss 0.69815, field
MSE 0.38435, temperature MSE 0.21141, parameter update norm 0.17834, and peak
CUDA memory 24,168.6 MiB. The fixed 8-case x 1024-query population comparison
against static Run-1501 epoch 50 was:

| Epoch-50 population measure | Run 1501 entmax-1.5 | Run 1502 sparsemax |
|---|---:|---:|
| Environment support RE | 0.7231 | 0.3985 |
| Module support RM | 0.6951 | 0.7423 |
| Mean query degree Kq | 2.9349 | 3.0050 |
| Mean environment degree | 5.9766 | 2.7995 |

Run 1502 had zero active-module, environment, or query empty supports. Its
singleton fractions were 0% for active modules, 0.065% for environment tokens,
and 0.903% for queries. Thus the structural objective persisted through
training without a support-collapse failure.

The maintained GPU-2 fixed-panel comparison nevertheless failed the learning
and cost gate. Candidate/reference fluid-field relative L2 was
0.4857/0.4560, fluid-temperature physical MAE 2.2319/2.1112, effective-h MAE
1.9178/1.3970, and mean evaluation time 0.3273/0.2811 seconds per case.
Near-interface relative L2 was effectively tied (0.4105/0.4093), while surface
temperature MAE (2.1193/2.1293) and heat-flux MAE (5.6936/5.7487) improved only
slightly. Peak memory was effectively unchanged. Therefore the exact decision
is `stop_before_epoch_150`: the sparse structure is real, but it does not
justify continuation in the face of field/thermal and timing regressions. No
epoch-150 or epoch-500 continuation was launched.

## 5. Deferred work

Mature Run-1501 epoch-5000 evaluation is explicitly out of scope. When that
training eventually finishes, no automatic evaluation, checkpoint selection,
executor benchmark, restart, or process action should occur. A later 90-case
mature comparison requires a new explicit user request.

No Run-1502 continuation beyond epoch 500 is authorized. No custom kernel,
monitoring infrastructure, new provenance/security contract, second
normalizer, temperature sweep, top-k rule, sparsity loss, or historical
architecture change was added.

## 6. Reproducible artifacts

- Accounting Q1024: `diagnostics/generated/run1501_executor_run1502_development/accounting_e500_q1024`
- Accounting Q8192 and full-grid figures: `diagnostics/generated/run1501_executor_run1502_development/accounting_e500_q8192`
- Frozen Run-1502 diagnostic: `diagnostics/generated/run1501_executor_run1502_development/frozen_run1502_q1024`
- Q8192 support-block benchmark: `diagnostics/generated/run1501_executor_run1502_development/run1501_support_blocks_benchmark_q8192.json` (`sha256 a3e294e02b804297c65c135023bd713248510db432a7b3d3549a8329fa799417`)
- Run-1502 epoch-50 review: `diagnostics/generated/run1501_executor_run1502_development/run1502_epoch0050/epoch0050_review.json` (`sha256 55088eb6cda8d288d1f7afd58ea167c642c38074ec24b55240ea9c8de70ef8fa`)
- Run-1502 managed output: `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1502_20260923_004751_sparse_incidence_environment_sparsemax`
