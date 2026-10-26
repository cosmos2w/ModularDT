# Interaction-reused original-member routing

The `interaction_reused_member_routing_v1` capability prepares original M and E records through two ordered contextual cycles. In destination-first source-code notation, each cycle applies `mm`, `em`, `ee`, and `me` gates to the actual value messages. The Thermal adapter attaches prescribed heat to its original module before preparation and extracts all native observations from one joint four-channel field head. Historical model classes and checkpoints retain their original interpretation.

Each present M record is an address. Its E-member coefficients come from the last applied `me` gate, and its nonself M-member coefficients come from the last applied `mm` gate. M self-access has coefficient one. These are live applied coefficients, rather than regenerated attention weights or a transposed exchange. Address membership retains original records; it does not pool their values into a group function.

In Route mode, query-conditioned M-address proposals use entmax1.5 with a zero-logit null atom. Positive proposals define a union of eligible original member IDs. A separate cheap learned query–member score combines contextual keys, relative geometry, the live proposal-weighted relation prior, and a dimensionless information measure. M and E each use their own entmax distribution and null atom. Only exact-positive final member weights cause query–member value rows to execute; duplicate address paths to one original ID execute that member once. The two typed contexts then enter one shared physical trunk and joint output head.

Present M members use unit count measure in the final read, including a present module with zero contextual donor measure. Positive E physical measures are divided by their scene's positive mean before the final read, preserving ratios. Contextual donor measures and final-read information measures belong to separate reductions. An all-null M-address proposal or absent M inventory provides zero typed contexts; there is no independent environment-value fallback.

Full mode uses the same contextual formation, common initialized tensors, typed value projections, and joint field head, while reading every eligible original member. It is a predictive-capacity reference. Full/Route training differences do not establish parameter-identical or equal-work causal attribution to grouping. Full candidate scoring, upstream E-to-E exchange, cached scene projections, selected value rows, and complete elapsed time remain separate costs. Decoder read support is not a complete input-dependency graph or evidence of physical causality.

## Training and recovery

Use the `ModularDT` environment and the maintained research CLI from `HONF_Proj`. The qualification configurations bind the existing fixed TRAIN150/DEV22 membership. The scientific configurations start separately from scratch, use TRAIN600-fitted normalization, and stop at epoch 100. Debugging weights and optimizer state are never promoted into the full-data lineages. The first full-data endpoint is preliminary; further epochs require the user's explicit instruction.

```bash
conda activate ModularDT
cd /home/wanglz/Desktop/src/ModularDT/HONF_Proj
export CUDA_VISIBLE_DEVICES=0,1
python tools/honf_research.py train-contextual \
  --config configs/research/thermal_member_routing_r13_3_full_v1.json \
  --device cuda:0 --devices cuda:0 --epochs 100 \
  --output /data2/wanglz/ModularDT/round13_3_20261025/full
python tools/honf_research.py train-contextual \
  --config configs/research/thermal_member_routing_r13_3_route_v1.json \
  --device cuda:1 --devices cuda:1 --epochs 100 \
  --output /data2/wanglz/ModularDT/round13_3_20261025/route
```

New runs require empty output directories. Recovery uses the latest complete maintained checkpoint and restores named optimizer state, RNG, data membership, sampling, and provider state. Keep the same visible-device inventory used at launch when restoring CUDA RNG state. The round report records the actual executed recovery commands and measured costs; commands in this guide describe the maintained interface and do not establish that a run has completed.

Scientific payloads, plots, histories, and checkpoint aliases belong on the mounted DATA2 filesystem. Keep latest, best-field, declared 100-epoch milestones, and the requested final or clean-stop state. Curves refresh every ten epochs and at completion. Preserve existing runs and shared caches; generated artifacts and one-time analysis helpers remain outside Git.

## Evaluation

Evaluate the retained epoch-100 checkpoint with the maintained member-routing evaluator. The fixed detailed panel includes TRAIN M1 and exposed validation M3/M7/M10 examples. Statistical validation covers canonical exposed VAL89 separately from low-M TRAIN; DEV22 is an exposed subset, not independent TEST evidence. The evaluator retains original-ID query traces, numerical field/residual arrays, all nine native observables, fixed-checkpoint interventions, and measured work/cost records.

B-only interventions keep candidate IDs, proposals, and cheap member scores fixed. Geometry controls change the function and report their actual selected counts. The dense candidate-union executor uses the same weights as the selected executor, including zero weights, to check numerical parity while charging the extra value rows. Removing an applied contextual message requires fresh preparation and dependent-stage recomputation. Saved heat/geometry references provide bounded physical diagnostics; this round adds no reference solves, response-fitting loss, or inverse optimizer.
