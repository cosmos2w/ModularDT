# Stage 5 — Soft Organization, Sparse Execution Evaluation

## Decision

Proceed with a matched, two-run comparison:

1. **Model A:** six exchangeable slots, all-soft softmax organization and query routing, dense reference training.
2. **Model B:** six fixed-projection softmax edges with the same modern descriptor-first additive decoder and dense background.

The existing ladder selects **Outcome B** from the Stage-5 Goal Mode plan:

> Stage-2 exchangeable-all-soft is meaningfully organized, while the scheduled Stage-4 family becomes nearly rank-one.

No exchangeable locality, persistent-code residual, fixed-scale locality, physical anchor, or topology-content decoupling change is justified by the evidence. The next controlled experiment should remove the sparsity/selection curriculum from training, not redesign slot identity.

No long training run was launched during this work.

## Provenance and scope

- Branch: `agent/honf-adaptive-correctness`
- Implementation parent SHA: `7a493b96649bbf78192d88c737a48239f28f14c5`
- Dataset identity in every inspected run: `thermal_channel_global_v1`
- Dataset fingerprint: `4224093c22a67af4adfecc8b21d53548e4263ec2254c230dc83c89526b36da05`
- Primary ladder screen: 21 deterministic test cases (`0273`–`0292`, plus `0653`), best and latest checkpoints where available.
- Finalist confirmation: complete 90-case test split for Stage-1 best-field and Stage-2 best-field.
- All evaluation restored checkpoint-owned normalization, channel order, and explicit saved training progress.
- Historical checkpoints and historical run directories were read only.

Machine-readable outputs:

- `diagnostics/stage5_topology_ladder_21cases/topology_quality_summary.json`
- `diagnostics/stage5_topology_ladder_21cases/topology_quality_per_case.csv`
- `diagnostics/stage5_topology_ladder_21cases/topology_quality_per_case_edge.csv`
- `diagnostics/stage5_topology_finalists_full90/topology_quality_summary.json`
- `diagnostics/stage5_pruning_existing_finalists_20cases/retained_mass_pruning_summary.json`

The maintained tools are:

- `diagnostics/evaluate_topology_quality.py`
- `diagnostics/evaluate_retained_mass_pruning.py`

## Existing-run inventory

| Run/checkpoint family | Source SHA | Epochs inspected | Organizer and routing | Decoder/background | Initialization provenance |
|---|---:|---:|---|---|---|
| Run 1000 | `2afa8475` | best 9655; latest 10000 | fixed projection, 6 edges, softmax | residual-concat, context fusion | direct historical training |
| Stage 1 / Run 1005 | `edb29371` | best 499; latest 500 | fixed projection, 6 edges, softmax | descriptor-first, exact additive | partially initialized from Run 0001 context-fusion; physical/additive heads skipped |
| Stage 2 / Run 1007 | `7ed6ee41` | best 547; latest 600 | exchangeable, 6 slots, all selected, softmax | descriptor-first, exact additive | initialized from Run 1005; fixed organizer keys skipped |
| Run 1201 | `2ccbeb79` | best 2466; latest 2500 | exchangeable, scheduled entmax/selection | additive, dense background | Stage-4 uniform-LR control |
| Run 1202 | `2ccbeb79` | best 4684; latest 5000 | exchangeable, scheduled entmax/selection | additive, dense background | Stage-4 split-LR run |
| Run 1203 | `2ccbeb79` | best 2499; latest 2500 | exchangeable, scheduled entmax/selection | additive, pooled background | Stage-4 pooled-background run |

Run 1007's manifest status is `failed` at epoch 600, but both evaluated checkpoints are readable and internally consistent. It is used as structural evidence, not as a completed formal training result.

## Organizer-quality result

### Complete 90-case finalist evaluation

Values below are test-split medians. Lower profile cosine means more distinct edge maps; higher effective rank means more independent edge profiles.

| Metric | Stage 1 fixed-additive | Stage 2 exchangeable-all-soft | Stage-5 provisional gate |
|---|---:|---:|---:|
| Normalized field MSE | 0.2351 | 0.1740 | descriptive only; budgets differ |
| Module row entropy | 0.6854 | 0.8522 | no hard gate |
| Module edge-profile cosine | 0.5252 | 0.7785 | lower is better |
| Module effective rank | 2.5386 | 1.8347 | higher is better |
| Environment row entropy | 0.2392 | 0.5514 | `< 0.65` |
| Environment mean row maximum | 0.8246 | 0.6437 | `> 0.50` |
| Environment edge-profile cosine | 0.0815 | 0.5855 | `< 0.55` |
| Environment effective rank | 3.8050 | 1.9207 | `> 2.5` |
| Largest environment occupancy | 0.4323 | 0.5911 | `< 0.70` |
| Normalized region separation | 0.2842 | 0.1630 | `> 0.10` |
| Query row entropy | 0.3547 | 0.4538 | `< 0.90` |
| Query edge-profile cosine | 0.1438 | 0.2101 | `< 0.80` |
| Query effective rank | 4.7068 | 4.3470 | higher is better |
| Largest query occupancy | 0.2709 | 0.2772 | `< 0.80` |
| Query global-route L1 deviation | 1.2520 | 1.1408 | `> 0.10` |
| Pairwise edge-map cosine | 0.1578 | 0.2411 | `< 0.85` |
| Temperature additive-edge cosine | 0.0992 | 0.1745 | `< 0.90` |

Interpretation:

- Stage 1 decisively preserves structured organization after descriptor-first state and additive decoding. It is a safe modern fixed-organizer fallback.
- Stage 2 is **not rank-one**. Its environment organization is materially weaker than Stage 1 and narrowly misses the provisional environment-profile cosine gate, while its environment effective rank remains below the desired final gate. However, region separation, occupancy, query routing, pairwise maps, and additive fields are all clearly differentiated.
- The Stage-2 evidence is sufficient for the intended Stage-5 experiment: train anonymous roles softly without the later entmax/selection curriculum, then judge the new 1500-epoch checkpoint against the complete acceptance matrix.

### Scheduled Stage-4 collapse

The 21-case best-checkpoint means expose a qualitative discontinuity:

| Metric | Run 1000 | Stage 1 | Stage 2 | Run 1201 | Run 1202 | Run 1203 |
|---|---:|---:|---:|---:|---:|---:|
| Environment profile cosine | 0.084 | 0.082 | 0.577 | 0.922 | 0.481 | 0.312 |
| Environment effective rank | 5.02 | 3.61 | 1.97 | 1.02 | 1.59 | 1.64 |
| Region separation / domain diagonal | 0.322 | 0.284 | 0.164 | 0.008 | 0.023 | 0.237 |
| Query profile cosine | 0.402 | 0.167 | 0.230 | 0.999 | 0.855 | 0.803 |
| Query effective rank | 3.87 | 4.46 | 4.16 | 1.00 | 1.01 | 1.01 |
| Pairwise profile cosine | 0.466 | 0.217 | 0.287 | 0.996 | 0.851 | 0.797 |
| Temperature edge-field cosine | unavailable | 0.120 | 0.192 | 0.997 | 0.853 | 0.798 |

Run 1202's environment profile cosine varies because its surviving-edge count varies by case, but the decisive query/additive effective ranks remain approximately one. Multiple selected edges therefore do not represent multiple independent mechanisms.

Representative corrected figures:

- [Stage-1 physical-order organizer](../diagnostics/stage5_topology_smoke/figures/stage1_1005/0653/organization_matrices_physical_order.png)
- [Stage-2 physical-order organizer](../diagnostics/stage5_topology_smoke/figures/stage2_1007/0653/organization_matrices_physical_order.png)
- [Stage-2 query routing](../diagnostics/stage5_topology_smoke/figures/stage2_1007/0653/routing_attention_maps.png)
- [Stage-2 routing margin](../diagnostics/stage5_topology_smoke/figures/stage2_1007/0653/routing_dominant_margin.png)
- [Stage-2 unsorted physical environment view](../diagnostics/stage5_topology_smoke/figures/stage2_1007/0653/organization_environment_unsorted.png)
- [Stage-2 explicitly sorted environment view](../diagnostics/stage5_topology_smoke/figures/stage2_1007/0653/organization_environment_sorted_by_dominant_edge.png)
- [Stage-2 per-edge physical-field contributions](../diagnostics/stage5_topology_smoke/figures/stage2_1007/0653/topology_field_contributions.png)
- [Run-1202 corrected organizer](../diagnostics/stage5_topology_smoke_1000_1202/figures/stage4_1202/0653/organization_matrices_physical_order.png)

The ordinary organization matrix now preserves physical environment-token order. A second artifact is explicitly named and labeled `sorted_by_dominant_edge`. Routing plots use the effective active-edge mask, so inactive candidates are no longer rendered as mechanisms.

The complete-split per-edge table also records dominant module/environment/query occupancy, mean query probability, pairwise contribution fraction, learned source/region coordinates and scales, and additive energy by physical channel in whole, fluid, near-interface-fluid, and far-field-fluid regions. Stage 2 does contain a strong broad-field edge: one persistent code is the largest whole/fluid `u` contributor in 89 of 90 cases and the largest whole/fluid `v` contributor in 71 of 90. It is not the universal mechanism, however: temperature is most often led by another code (42 of 90 whole-field cases), and no one code uniformly dominates every channel and region. This is a real specialization caveat for the future Model-A checkpoint, not evidence of the Stage-4 all-edge rank-one failure.

## Base, provisional, and final organizer diagnosis

The 21-case best-checkpoint means are:

| Family/pass | Module effective rank | Environment effective rank | Environment profile cosine | Region separation |
|---|---:|---:|---:|---:|
| Stage 1 base | 1.810 | 3.643 | 0.083 | 0.285 |
| Stage 1 final | 2.173 | 3.610 | 0.082 | 0.284 |
| Stage 2 base | 1.462 | 1.862 | 0.598 | 0.154 |
| Stage 2 final | 1.591 | 1.970 | 0.577 | 0.164 |
| Run 1202 base | 1.016 | 1.561 | 0.496 | 0.027 |
| Run 1202 final | 1.018 | 1.592 | 0.481 | 0.023 |

Conclusions:

- Local-response fusion does not collapse a healthy Stage-2 topology; provisional/final organization slightly improves its effective ranks.
- Run 1202 is already collapsed in the base pass. Final reorganization changes assignment values but does not create the rank-one failure.
- Topology-content decoupling is therefore not indicated.
- The best-supported failure class is the scheduled training regime—entmax assignment schedules plus hard adaptive selection—not repeated ThermalChannel reorganization alone. Because the historical stages differ in training budget and initialization, this is a staged localization result rather than a randomized causal estimate.

## Dense versus retained-mass execution

The evaluation-only comparator was validated on 20 cases from both existing additive finalists with query retained mass `0.98`, routed-module retained mass `0.95`, and a minimum of one route. It modifies only process-local routing configuration and the prepared-state execution flag; source checkpoints and state-dict structure are unchanged.

| Metric | Stage 1 fixed-additive | Stage 2 exchangeable-all-soft |
|---|---:|---:|
| Query retained mass p05 | 0.9819 | 0.9813 |
| Routed-module retained mass p05 | 0.9835 | 1.0000 |
| Aggregate fluid MSE degradation | +0.86% | +0.25% |
| Worst fluid-channel MSE degradation | +2.49% | +1.04% |
| Query-edge route reduction | 51.3% | 39.7% |
| Active-module route reduction | 3.2% | 0.0% |
| Median dense decoder time | 9.02 ms | 9.13 ms |
| Median gathered decoder time | 6.55 ms | 6.82 ms |
| Peak allocated memory, dense | 434 MB | 438 MB |
| Peak allocated memory, gathered | 262 MB | 290 MB |
| Full-limit gathered max difference | `2.86e-6` | `2.86e-6` |

Both frozen proxies pass the accuracy, retained-mass, and query-edge route-reduction gates. Neither currently obtains meaningful module-incidence pruning; Stage 2 retains all active modules. Sparse execution is therefore ready as an evaluation/deployment option, but the new formal checkpoints must be re-evaluated rather than inheriting this result.

## Prepared profiles

### Model A — `stage5_exchangeable_soft_organized.json`

- six exchangeable slots;
- all candidates active;
- module/environment/query softmax for all epochs;
- selection and sparsity schedules explicitly disabled;
- no environment or query locality repair;
- descriptor-first exact additive decoder;
- dense query-attention background;
- dense training/reference routing;
- prediction LR `3e-4`, organizer LR `1e-4`;
- milestone checkpoints at epochs 250, 500, 1000, 1500, and 2500.

### Model B — `stage5_fixed_softmax_modern.json`

The same settings as Model A except:

- fixed-projection organizer;
- six fixed hyperedges;
- no exchangeable runtime capacity.

No model-parameter or loss regularizer distinguishes the pair.

## User-controlled run commands

Run these from `HONF_Proj`. They deliberately start from fresh forward-model weights; both use the same frozen Stage-A checkpoint.

### Model A: Run 1301 to 1500 epochs

```bash
python train.py \
  --config src/config_core/forward/adaptive_sparse_additive.json \
  --experiment-overlay src/config_core/forward/experiments/stage5_exchangeable_soft_organized.json \
  --local-checkpoint Trained_Results/ThermalChannel/Local_Module_Runs/thermal_disk/Run_0000_base/best_model.pt \
  --run-id 1301 \
  --run-name stage5_exchangeable_soft_organized \
  --epochs 1500 \
  --device cuda:0 \
  --yes
```

### Model B: Run 1302 to 1500 epochs

```bash
python train.py \
  --config src/config_core/forward/adaptive_sparse_additive.json \
  --experiment-overlay src/config_core/forward/experiments/stage5_fixed_softmax_modern.json \
  --local-checkpoint Trained_Results/ThermalChannel/Local_Module_Runs/thermal_disk/Run_0000_base/best_model.pt \
  --run-id 1302 \
  --run-name stage5_fixed_softmax_modern \
  --epochs 1500 \
  --device cuda:1 \
  --yes
```

### Resume the same runs to 2500 epochs

Set each variable to the exact timestamped directory printed by its original launch. Do not allocate new run IDs.

```bash
RUN_A_DIR=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1301_<timestamp>_stage5_exchangeable_soft_organized

python train.py \
  --config src/config_core/forward/adaptive_sparse_additive.json \
  --experiment-overlay src/config_core/forward/experiments/stage5_exchangeable_soft_organized.json \
  --resume-checkpoint "$RUN_A_DIR/latest_model.pt" \
  --local-checkpoint Trained_Results/ThermalChannel/Local_Module_Runs/thermal_disk/Run_0000_base/best_model.pt \
  --epochs 2500 \
  --device cuda:0 \
  --yes
```

```bash
RUN_B_DIR=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1302_<timestamp>_stage5_fixed_softmax_modern

python train.py \
  --config src/config_core/forward/adaptive_sparse_additive.json \
  --experiment-overlay src/config_core/forward/experiments/stage5_fixed_softmax_modern.json \
  --resume-checkpoint "$RUN_B_DIR/latest_model.pt" \
  --local-checkpoint Trained_Results/ThermalChannel/Local_Module_Runs/thermal_disk/Run_0000_base/best_model.pt \
  --epochs 2500 \
  --device cuda:1 \
  --yes
```

## Acceptance and stopping recommendation

At epoch 1500, evaluate both models on the complete split. Continue both—not only the more accurate one—to 2500 only if both are stable, still improving, and scientifically viable.

The primary open risk is Model A's weaker environment effective rank, not query routing or decoder capacity. If the new all-soft exchangeable run remains near the Stage-2 complete-split values while preserving prediction quality, it is scientifically viable but not yet superior to the fixed organizer. If it becomes rank-one despite removal of the schedules, then exchangeable role formation should return to the gated locality/code-residual research path in a separate round.

## Verification completed

```text
/home/wanglz/miniconda3/envs/ModularDT/bin/python -m pytest -q tests
196 passed in 5.88s

/home/wanglz/miniconda3/envs/ModularDT/bin/python -m pytest -q Case_ThermalChannel/tests
58 passed, 1 skipped in 6.67s
```

The skipped test is the pre-existing inverse joint-training integration check, which reports that its optional local inverse artifacts are unavailable. No inverse code was modified.

Both Stage-5 launch profiles also passed `train.py --dry-run` against the real 690-case dataset manifest, complete 600/90 split, dataset fingerprint, and frozen Stage-A checkpoint. No run directory was created and no training epoch was executed.

## Limitations

- Existing runs have unequal training budgets, source commits, and initialization histories. Their loss values are not a controlled final model comparison.
- Stage 1 was initialized from Run 0001 context-fusion, not Run 1000. Structural preservation is established by output topology, not by identical ancestry.
- Run 1000 has no exact additive `pred_field_by_edge`; its additive metrics are correctly unavailable.
- The retained-mass runtime benchmark is a single fixed 2-D case on GPU 0. Route reduction and accuracy are more portable evidence than the absolute timing.
- Full 90-case evaluation was run for the two existing finalists; the broader best/latest ladder used 21 cases.
