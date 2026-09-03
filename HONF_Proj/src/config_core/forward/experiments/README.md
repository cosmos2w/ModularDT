# Forward experiment overlays

These files are strict overlays rather than complete profiles. Apply one with `--experiment-overlay`; the resolver permits changes only to keys already declared by the selected source profile and records the overlay path, hash, and exact JSON in the managed run.

## Current Stage-7 execution and audit overlays

- `stage7_fused_query_module.json`: promoted behavior-equivalent K=6 fused query-module executor for training and evaluation on `stage7_structured_context.json`; training remains dense with the legacy pair kernel and full beta mass, while beta-0.98 gathered pruning remains an evaluation-time override.
- `stage7_k4_fused_audit.json`: completed Run-1602 K-scaling research audit; trained to epoch 5000 and retained as a non-promoted reference because its global accuracy gain came with weaker near-interface fidelity and compressed topology.
- `stage7_k8_fused_audit.json`: rejected Run-1603 K-scaling audit, stopped at epoch 500 after failing the continuation gates.
- `stage7_factorized_gated_r96.json`: rejected Run-1600 factorized-kernel experiment, stopped at epoch 500 and retained only for reproducibility; it is not an active candidate.

`stage7_structured_context.json` remains the historical/scientific Run-1401 architecture profile and numerical checkpoint reference. It is intentionally not rewritten to select fused or sparse execution.

## Case-adaptive residual candidate

`../case_adaptive_residual_context.json` is a complete, standalone research
candidate rather than an overlay. It keeps the Stage-7 physical/training
settings and uses `organizer_mode="case_adaptive_residual"` with
`num_hyperedges=0`, shared sequential residual extraction, soft training
survival, hard case-specific evaluation support, and fused dense
query-module routing. Its 2500-epoch budget and milestone list are explicit;
it does not change `recommended_forward_profile` or enable organizer count
regularization.

## Historical experiment overlays

- `old_parity.json`: disables hyperedge value context.
- `uniform_h_assignment.json`: uses uniform module assignment and query routing with the pairwise-only decoder.
- `global_only.json`: disables the Stage-A dependency and uses global fallback heads with no interaction refinement.
- `stage1_fixed_additive_soft.json`: isolates descriptor-first mechanism state and exact additive field assembly while retaining the six-edge fixed organizer, dense soft routing, ThermalChannel data/losses, frozen Stage A, predicted ports, and one-pass coupling of the comparison profile.
- `stage2_exchangeable_soft.json`: replaces only the fixed organizer with six exchangeable runtime slots while keeping every candidate selected, softmax assignments, no locality, dense execution, exact additive assembly, and the Phase-1 learning rate.
- `stage4_uniform_lr2e4_dense_background.json`: Stage-4 uniform-learning-rate control using the unchanged dense query-to-environment background.
- `stage4_split_lr_dense_background.json`: separates exchangeable-organizer and prediction learning rates while retaining the dense background.
- `stage4_split_lr_pooled_background.json`: combines the split optimizer with the parameter-compatible globally pooled background efficiency experiment.
- `stage5_exchangeable_soft_organized.json`: uses six anonymous candidates with softmax organization/routing for the complete training run, dense additive reference execution, dense residual background, and split prediction-versus-organizer learning rates; retained-mass pruning is evaluation-only.
- `stage5_fixed_softmax_modern.json`: matched Stage-5 fallback using the proven six-edge fixed organizer with the same additive decoder, background, execution path, optimizer rates, and retained milestone epochs.

Use `enhanced_honf_pairwise.json` for older compatibility switches and `adaptive_sparse_additive.json` for the Stage-1–6 additive/adaptive family. CLI run ID, name, device, epoch limit, and evaluation-only routing overrides remain separate provenance.
