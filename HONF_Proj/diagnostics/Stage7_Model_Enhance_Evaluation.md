# Stage-7 Model Enhancement Evaluation

## Decision

Retain Run 1401 best-by-field, epoch 4585, as the scientific K=6 baseline and
promote the fused gathered executor as its evaluation/deployment path. The
rank-96 factorized candidate (Run 1600) is stopped and rejected at epoch 500:
its representation stayed healthy and its execution was substantially cheaper,
but it missed both matched-budget accuracy gates by more than 2x. No 2500-,
5000-, or 10000-epoch continuation was launched.

Implementation started from `629b2e47f0953f509e7103db2988f64b4107cbbd`.
The launch-visible implementation commit is
`246d4c5a0f1ee5cf2d91f4b6a03263d4150de3f3`.

## Implementation and configuration

- Added historical-default `pairwise_aggregation_mode=edge_explicit`,
  `pairwise_kernel_mode=legacy_mlp`, and
  `query_module_retained_mass_floor=1.0` configuration fields.
- Added fused query-module aggregation and vectorized retained-beta gathering.
  Truncated beta is never renormalized, and selection occurs before the pair
  kernel. Dense maps are returned only for explicit routing-map requests.
- Kept the accepted legacy parameter path
  `decoder.pairwise_kernel.pair_mlp.*` unchanged. Historical edge-explicit
  dense/gathered and additive research paths remain separate and loadable.
- Added the single factorized candidate: prepared module encoder, relative
  encoder, sigmoid-gated multiplicative interaction, LayerNorm, and output
  projection. Prepared module codes are private runtime state, not parameters
  or public model outputs.
- Added exactly two strict overlays:
  `stage7_fused_query_module.json` and
  `stage7_factorized_gated_r96.json`. The recommended profile remains
  `stage7_structured_context`.
- Extended the maintained golden replay, retained-mass evaluator, and existing
  checkpoint benchmark. Generated evidence uses ignored scratch output or
  managed evaluation manifests.

## Compatibility and parity gates

- Full root and ThermalChannel tests: **281 passed, 1 expected skip**.
- Run 1000 epoch 9655: exact golden replay, strict 237-key load, one optimizer
  group.
- Run 1401 epoch 4585: exact golden replay, strict 237-key load, one optimizer
  group.
- Run 1401 fused-dense versus historical outputs: exact zero difference for
  field grid, internal temperature, interface, and port condition.
- Test A adds no trainable parameters and does not change historical state keys.
- Run 1600 best-field checkpoint: strict 243-key load, no missing/unexpected
  keys; optimizer resume restores one group and 107 state entries.

The dense legacy-kernel fused reference computes beta for routing and uses
32-feature contraction blocks to preserve the accepted floating-point reduction
order without allocating the full `[B,Q,K,H]` edge-context tensor. The sparse
and factorized paths aggregate the selected query-module pairs directly.

## Sparse Run-1401 result

The initial one-case sweep found that all prescribed thresholds
`{0.999, 0.995, 0.99, 0.98}` retained every active ThermalChannel module.
The loosest prescribed setting, beta mass `0.98`, was therefore used for the
required 90-case evaluation.

- Retained beta mass: mean `0.999999993`, p05 `0.999999881`, minimum
  `0.999999762`; no hard-cap violations.
- Active-route removal: `0%`. Current five-module routing is too diffuse for
  actual beta truncation at the prescribed floors.
- Pooled field-MSE degradation: `4.30e-9` relative; case-p95 degradation
  `1.46e-7`; worst case `2.27e-7`.
- Every channel and the near-interface temperature/vorticity metrics remained
  far inside their promotion bounds.
- Full-support gathered parity difference: `0`.
- Prepared decoder on case 0273: median `6.549 -> 4.011 ms` (**38.75% less**)
  and peak allocated `403.2 -> 166.2 MB` (**58.78% less**). This gain comes
  from evaluating active gathered pairs instead of padded/full pair tensors,
  not from removing active five-module routes.

Managed evidence:

- `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1401_20260823_151126_stage7_modern_structured_context/evaluations/retained_mass_pruning/20260824_085935/evaluation_manifest.json`

## Synthetic scaling

The synchronized benchmark covers
`Q={8192,65536,262144,1000000}`,
`M={5,12,32,64,128}`, K=6, hidden width 256, and 8192-query chunks.
It records 100 rows across historical, fused, sparse, factorized, and
factorized+sparse modes. Dense legacy-kernel parity stayed below `4.7e-9`.

At `Q=1,000,000`, `M=128`, beta-0.98 selected `17.9%` of pairs:

| Mode | Median | p95 | Incremental allocated |
|---|---:|---:|---:|
| edge-explicit legacy | 6.497 s | 6.502 s | 3429.1 MB |
| fused sparse legacy | 1.151 s | 1.154 s | 819.5 MB |
| factorized dense R96 | 3.368 s | 3.375 s | 3125.0 MB |
| factorized sparse R96 | 0.741 s | 0.747 s | 579.2 MB |

The sparse legacy path reduces time by 82.3% and allocated memory by 76.1%
at this large shape. Evidence:
`diagnostics/generated/stage7_pairwise_scaling_benchmark.json`.

## Factorized Run 1600 decision

Run 1600 was trained once, from scratch, with fixed K=6, softmax routing,
residual/raw hyperedge state, context fusion, the unchanged Stage-A coupling,
and shared AdamW policy. It stopped at the planned epoch 500.

- Epoch-500 trailing-50 validation field MSE: `5.1624e-2` versus Run 1401
  `2.2987e-2` (**2.246x**, gate <=1.15x).
- Epoch-500 trailing-50 temperature MSE: `3.1167e-2` versus Run 1401
  `1.5089e-2` (**2.066x**, gate <=1.15x).
- Best field MSE through epoch 500: `3.9949e-2` at epoch 500; best temperature
  MSE `2.3660e-2` at epoch 467.
- Topology remained healthy: environment cosine `0.153`, environment rank
  `4.624`, query cosine `0.196`, query rank `4.978`, region separation `0.297`.
- Pair-kernel parameters: `279,809 -> 83,969` (**69.99% less**); total model
  parameters: `3,508,649 -> 3,312,809` (**5.58% less**).
- Matched case-0273 benchmark: prepared median `6.851 -> 3.656 ms`
  (**46.63% less**) and allocated memory `361.7 -> 102.4 MB`
  (**71.69% less**). Full-forward median improves 8.77%.

Run/evaluation evidence:

- `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1600_20260824_090240_stage7_factorized_gated_r96/run_manifest.json`
- `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1600_20260824_090240_stage7_factorized_gated_r96/evaluations/topology_quality/epoch0500_best_field/evaluation_manifest.json`
- `diagnostics/generated/stage7_model_enhance_checkpoint_benchmark.json`

The exact initial launch command was:

```bash
python train.py \
  --config project://src/config_core/forward/stage7_structured_context.json \
  --experiment-overlay project://src/config_core/forward/experiments/stage7_factorized_gated_r96.json \
  --workflow forward --device cuda:2 --epochs 500 \
  --run-id 1600 --run-name stage7_factorized_gated_r96 --yes
```

## Four-entry K=6 comparison and K audit

1. Run 1401 best, historical edge-explicit dense: retained baseline.
2. Run 1401 best, fused gathered beta-0.98: promoted execution path; no active
   route removal on the present five-module split, but lower runtime/memory.
3. Run 1600 best, factorized fused dense: measured and rejected at epoch 500.
4. Run 1600 best, factorized fused sparse: not promoted to full scientific
   evaluation because entry 3 failed the mandatory epoch-500 accuracy gate.

The later K-scaling audit is prepared but not launched. It should use the
retained Run-1401 architecture plus fused execution, train only K=4 and K=8
variants to epoch 500 initially, and admit K=12 only if K=8 shows clear
unsaturated capacity. No audit overlay or run was created in this round, so the
two-overlay limit and minimal formal-run count remain intact.

There is no unresolved correctness or compatibility failure. The unresolved
scientific result is explicit: prescribed beta thresholds do not remove active
modules in the current small-M dataset, and the R96 factorized architecture is
not accurate enough at the matched 500-epoch gate. No scientific behavior of
the accepted Run-1401 model was intentionally changed.
