# Stage 7 — Modern Structured Context HONF

## Decision

Stage 7 is ready for one formal experiment: Run 1401. It consolidates the modern repository around the successful Run-1000 scientific path rather than introducing another architecture.

The formal model is:

\[
\text{fixed six-edge soft organizer}
\rightarrow
\text{context-fusion HONF bottleneck}
\rightarrow
\text{field prediction}.
\]

No organizer or decoder prediction code was changed for Stage 7. The historical Run-1000 checkpoint loads with an exact 237-key state-dict match and reproduces the existing case-0653 evaluation numerically under the current evaluator.

## Formal model contract

Run 1401 uses:

- `organizer_mode = fixed_projection` and `num_hyperedges = 6`;
- learned softmax module and environment assignments;
- the fixed organizer's source-centered environment geometry bias;
- learned softmax query routing with the existing learned query-to-hyperedge geometry bias;
- `mechanism_state_mode = residual_concat`;
- `use_hyper_mechanism_encoder = false`;
- `field_assembly_mode = context_fusion`;
- dense routing with no query-edge or query-module truncation;
- one AdamW parameter group at `3e-4`, with weight decay `1e-5`;
- the current frozen Stage-A `best_model.pt` dependency;
- the existing predicted-port, interaction-refinement, and one-pass coupling behavior.

The formal profile is [stage7_structured_context.json](../src/config_core/forward/stage7_structured_context.json).

## Explicit exclusions

The profile does not activate exchangeable slots, adaptive selection, entmax, sparsity schedules, descriptor-first state, the mechanism refinement encoder, additive field assembly, an additive background, gathered training, staged optimization, or topology regularization.

Those implementations remain available for archived experiments and later deployment work. Stage 7 does not delete or refactor them.

## Retained modern infrastructure

The run retains variable module counts, masking and dynamic padding, per-case environment coordinates, the current frozen Stage-A integration, interaction refinement, prepared/chunked decoding, managed run provenance, dataset fingerprints, best-metric checkpoint selectors, latest-checkpoint cadence, milestone checkpoints, host-side efficiency improvements, topology evaluation, and inverse-facing mechanism descriptors.

## Checkpoint policy

The profile saves milestones at:

```text
500, 1000, 2500, 5000, 7500, 10000
```

It also saves best total, best field, best temperature, best predicted, and latest checkpoints. The latest checkpoint cadence is every 10 epochs.

## Formal Run 1401 command

Run from `HONF_Proj` in the `ModularDT` environment:

```bash
python train.py \
  --config src/config_core/forward/stage7_structured_context.json \
  --local-checkpoint Trained_Results/ThermalChannel/Local_Module_Runs/thermal_disk/Run_0000_base/best_model.pt \
  --run-id 1401 \
  --run-name stage7_modern_structured_context \
  --epochs 5000 \
  --device cuda:0 \
  --yes
```

This must be a fresh run. Do not add `--initialize-checkpoint`.

## Same-run resume to 10,000 epochs

After the epoch-5000 decision gate, substitute the actual timestamped Run-1401 directory:

```bash
RUN_DIR=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1401_<timestamp>_stage7_modern_structured_context

python train.py \
  --config src/config_core/forward/stage7_structured_context.json \
  --resume-checkpoint "$RUN_DIR/latest_model.pt" \
  --local-checkpoint Trained_Results/ThermalChannel/Local_Module_Runs/thermal_disk/Run_0000_base/best_model.pt \
  --epochs 10000 \
  --device cuda:0 \
  --yes
```

Resume the same run; do not create a second 1401 run for the extension.

## Epoch-5000 decision gate

Compare Run 1401 with Run 1000 using complete-split evaluation. Preferred structure bands are:

| Metric | Preferred at epoch 5000 | Run-1000 reference |
|---|---:|---:|
| Environment edge-profile cosine | `< 0.20` | `~0.094` |
| Environment effective rank | `> 3.5` | `~5.24` |
| Largest environment occupancy | `< 0.50` | `~0.292` |
| Normalized region separation | `> 0.20` | `~0.304` |
| Query edge-profile cosine | `< 0.55` | `~0.436` |
| Query effective rank | `> 3.0` | `~3.70` |

The preferred matched-budget accuracy condition is a trailing validation field-MSE median no worse than approximately `1.10 ×` Run 1000. Dense full-forward latency should be no worse than approximately `1.10 ×` Run 1000.

If structure is strong and the accuracy trajectory is close at epoch 5000, resume the same run to epoch 10000. Sparse context-fusion execution remains a separate later task.

## Readiness evidence

- Compatibility audit: [Stage7_Run1000_Compatibility_Audit.md](../diagnostics/Stage7_Run1000_Compatibility_Audit.md)
- Focused validation: [Stage7_Focused_Validation.md](../diagnostics/Stage7_Focused_Validation.md)
- Dry-run provenance: [stage7_dry_run_provenance.json](../diagnostics/stage7_dry_run_provenance.json)

No long training run and no smoke-training run were launched while preparing Stage 7.
