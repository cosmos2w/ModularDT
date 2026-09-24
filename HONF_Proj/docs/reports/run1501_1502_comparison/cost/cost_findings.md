# Cost and execution findings

## Matched inference benchmark

The headline is the complete `predict_case` call: it prepares one case, evaluates the full 8,192-point test grid with outer query batch 32,768, and transfers the returned field to CPU. The four selected best-field checkpoints were measured on the same NVIDIA RTX 6000 Ada (GPU 2), on the same 90 test cases, with checkpoint-native inner receiver chunks, maps off, one warmup, and three synchronized repetitions. The reported mean is the mean of the 90 per-case median wall times; p95 is the 95th percentile of those per-case medians.

| Run (selected epoch) | Parameters (trainable) | `predict_case` mean / p95 (ms) | Incremental peak allocated (MiB) | Relative to Run 1804 |
|---|---:|---:|---:|---:|
| 1404 (4890) | 3.51M (2.47M) | 29.88 / 32.53 | 364.27 | 5.32× faster |
| 1804 (4738) | 5.43M (4.40M) | 158.87 / 171.22 | 53.58 | reference |
| 1501 (4689) | 3.99M (2.95M) | 231.33 / 241.47 | 43.19 | 1.46× slower |
| 1502 (4794) | 3.99M (2.95M) | 231.09 / 244.72 | 43.19 | 1.45× slower |

Run 1404 has the lowest latency, with substantially higher transient allocation than the other three. Runs 1501 and 1502 use the same parameter shapes and configured receiver chunk (128); their measured application latencies differ by about 0.11%, below what this sequential benchmark can establish as a meaningful speed difference. Both new runs are slower than dense Run 1804 in this implementation despite using fewer parameters. Their incremental peak allocated memory is about 19% lower than 1804's.

The direct full-forward means were 27.78, 159.83, 229.14, and 229.89 ms for 1404, 1804, 1501, and 1502. The phase probes measured preparation plus one query at 24.92, 41.54, 63.63, and 64.18 ms, and prepared P2 decode at 6.58, 118.49, 167.88, and 168.14 ms. These probes help locate cost in the pipeline, but are not additive wall-time components: they are separate scopes with different query counts and retained states.

## Training and model size

| Run | Trainable parameters | Recorded train time (h) | Validation time (h) | Total for 5,000 epochs (h) | Trainer-recorded peak CUDA memory (summary MB field) |
|---|---:|---:|---:|---:|---:|
| 1404 | 2.47M | 4.14 | 0.98 | 5.12 | 24,841 |
| 1804 | 4.40M | 23.02 | 2.53 | 25.55 | 27,210 |
| 1501 | 2.95M | 10.77 | 1.38 | 12.15 | 24,280 |
| 1502 | 2.95M | 10.76 | 1.40 | 12.15 | 24,273 |

These are historical run-summary totals, not a same-device training benchmark. Runs span different dates, software states, and visible CUDA device counts. Run 1804 was resumed and its post-resume `metrics.csv` timing rows contain malformed values; its total here comes from the finite `summary.json` `actual_*_wall_seconds` fields. The summary table does not infer a total by summing per-epoch CSV timings. The trainer peak-memory field is also historical and should not be compared as a controlled inference-memory result.

## Support versus executed rows

The pair and row ledgers below came from one untimed maps-on direct forward per case. They are diagnostic counters, not measured FLOPs or proof of GPU kernel work. “Unique pairs”/“logical paths” describe routed support; `fine_rows_forward` and related counters describe the rows the selected implementation reports forwarding.

| Run | Module unique pairs / logical paths | Module rows forwarded (valid denominator) | Environment unique pairs / logical paths | Environment rows forwarded (valid denominator) | Executor |
|---|---:|---:|---:|---:|---|
| 1404 | 47,787 mean dense routes; 98,304 padded pairs evaluated; 0 gathered | 98,304 pairs evaluated, across Q×12 padded module slots | Not exposed by scalar ledger | Not exposed | Classic route selection ratio 1.0 |
| 1804 | Not exposed by scalar ledger | Dense query-module tensor path | Not exposed by scalar ledger | Dense query-environment tensor path | Dense baseline |
| 1501 | 38,114 / 85,325 | 98,304 forwarded (47,787 valid-pair denominator; 50,517 padded rows) | 962,995 / 1,820,504 | 1,572,864 forwarded (1,572,864 valid-pair denominator) | `rectangular_reference`; optimized executor not selected |
| 1502 | 39,690 / 87,267 | 98,304 forwarded (47,787 valid-pair denominator; 50,517 padded rows) | 642,541 / 877,277 | 1,572,864 forwarded (1,572,864 valid-pair denominator) | `rectangular_reference`; optimized executor not selected |

Run 1404's measured dense route count varies with the active module count (24,576–81,920 over the cases); its evaluated-pair count stays at 98,304 and its gathered route count is zero. For Runs 1501 and 1502, environmental unique support falls by about 33% and logical paths by about 52% in 1502, while the reported environment forward rows remain fixed at 1,572,864 per case. Both report 6,291,456 content-dot rows, and both retain the same module padded rectangle. Therefore the support reduction in 1502 did not reduce these executed row counts. Sparsemax may have changed the learned support pattern, but this benchmark does not show a resulting compute reduction or a reliable latency improvement.

Run 1804 does not emit the same scalar ledger. The dense baseline implementation constructs all query-module interactions in `src/honf_forward_core/interface_fields/dense_pairwise.py` (module receiver/source expansions and their concatenation at lines 121–132), and computes query-environment geometry/attention against the full source set at lines 187–205. This is an implementation-shape reading, not a runtime row count.

## Reconciling evaluator timings

The current matched cost pass disables routing maps, warms each scope once, then takes three synchronized repetitions. The separate 1501/1502 accuracy pass requested routing maps, had zero warmups, and timed one `predict_case` call per case. That pass recorded 516.3 ms mean for 1501 best-field and 511.8 ms for 1502 best-field, versus 231.3 and 231.1 ms in the maps-off repeated pass. Routing-map production and one-pass/no-warmup timing make those separate measurements; use the corrected same-protocol cost table for the four-run efficiency comparison. The maps-on pass was an accuracy/evidence collection run, not the headline cost protocol.

An earlier cost attempt is preserved under `superseded/`. It mistakenly set inner `receiver_chunk_size` to the outer evaluator batch size (32,768), overriding native chunk size 128 for 1804/1501/1502. That reduced chunk-loop overhead and changed memory use, so its approximately 0.03–0.05 s 150x readings do not represent the checkpoint-native implementation and are excluded from the comparison. The corrected pass reports 231 ms for the new runs at native chunk 128. Run 1404's corrected `predict_case` mean is 29.88 ms (direct full-forward mean 27.78 ms); a roughly 0.036 s value is close to this legacy-model scope, but is not the corrected 1501/1502 latency.

## Reproduction and artifacts

The exact benchmark protocol and checkpoint paths are stored in `inference_cost_cuda2.json`; raw case/phase results are in `inference_cost_cuda2.csv`. Summary tables are `model_cost_summary.csv`, `inference_phase_summary.csv`, and `execution_ledger_summary.csv`. `figures/inference_cost_comparison.png` plots application/direct latency, preparation/decode probes, transient inference allocation, and historical 5,000-epoch training time.

The benchmark is inference-only: it creates no optimizer, calls no backward pass, and writes no checkpoint/model state. GPU 0 was not used. The first overridden-chunk result remains available only for audit in `superseded/`.
