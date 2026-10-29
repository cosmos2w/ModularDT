# Matched hyperedge receiver readouts

**Archived reference — 29 October 2026.** This guide records a predecessor model or a dated experiment, not current architecture policy or standing launch authorization. Use the [current guide index](../../../README.md); see the [archive review](../README.md) for the reason it was moved.

Round6 adds a receiver-conditioned context residual to the completed Round5 H model. The pool-first control R-P and edge-first candidate R-E use identical parameter names, initial values, geometry features and local covers. The comparison changes where SiLU is applied relative to the weighted reduction. It retains the original contextualizer, moment processor, native field extraction, local source support, source-linear heat law, losses and deterministic query sampling.

## Reader contract

The reader receives the fused completed H state `Z[B,E,128]`, the existing query encoding `q_state[B,Q,128]`, and the original four-slot Thermal or eight-slot Wind cover. It prepares `A_Z Z[B,E,64]` once per scene and computes `A_q q_state[B,Q,64]` once per receiver. All four maps are bias-free; the separate hidden bias has width 64. The residual output has width 128 and enters the existing field and Thermal response heads.

For valid cover weights `w`, actual floating-point row mass `s = sum(w)`, `u = A_Z Z + A_g psi(relative_geometry)` and `v = A_q q_state + b`, the two corrections are:

```text
R-P: W_o SiLU(s v + sum(w u))
R-E: W_o sum(w SiLU(v + u))
```

Both add their correction to the unchanged pooled H context. The shared geometry map contains `asinh(r)` and sine/cosine pairs at frequencies pi, pi/2 and pi/4 for each coordinate. Thermal uses signed query-minus-anchor displacement divided by the existing scene frame scale. Wind uses the supplied signed downstream/crosswind/vertical coordinates in rotor diameters. It applies no second rotation or scale division.

The new parameter counts are 25,536 for Thermal and 25,984 for Wind. Named initialization fixes nonzero input maps, zero hidden bias and exactly zero `W_o`. The initial complete function therefore recovers H while retaining the first-update output-map gradient. There is no extra residual gate, physical-source reader, learned selector or new bank processor.

Local query tiles contain at most 512 receivers. The conditional full-bank diagnostic R-G uses query tiles of 64 and anchor tiles of 32, with fixed positive environment measures normalized over valid anchors. Non-reentrant checkpointing preserves input and parameter higher derivatives without retaining a full query-by-bank hidden tensor. R-G changes only the correction; the original local base context and heat contraction remain local. Global access and its integration weights are distinct from the primary local comparison.

Prepared projections carry scene, frame, tensor-version and owner bindings. An optimizer step or association intervention requires rebuilding every dependent projection. A bound prepared object cannot be shared across model instances or reused after its differentiation graph is consumed.

## Frozen fits and joint confirmation

| Contract | Frozen diagnostic | Conditional joint confirmation |
| --- | --- | --- |
| Source | Exact own-dataset H e2500 | Exact own-dataset H e1000 |
| Main age | New fit age 1 onward | Original absolute age 1001 onward |
| Sampler age | 2500 + fit age | Original absolute age |
| Trainable parameters | New reader, existing field head, Thermal response head | All original H parameters and the new reader |
| Optimizer birth | Fresh AdamW for every trainable fit parameter | Inherited named moments for common H parameters; fresh state for new reader |
| Prepared P, moments and Z | Numerically fixed for identical inputs | May differ as the two models coadapt |
| Required endpoint | 500 full selected-data epochs | Absolute e2500 |

The frozen provider preserves every non-allowlisted parameter and every buffer bitwise, including the normalization epoch. It freezes weights with `requires_grad_(False)` while retaining native geometry and query derivatives. Exact own-run resumes restore that fit's complete weights, optimizer, RNG, provider state and history. They cannot rewind to a historical checkpoint or import the parent optimizer into a fresh frozen fit.

The inherited engine calls its complete native objective stage `joint`. Frozen-probe histories retain that internal stage name while the Round6 phase remains `frozen_probe` and the optimizer contains only its declared readout/head allowlist. Joint confirmation has the separate Round6 phase `joint_confirmation` and trains the complete model.

Each Thermal epoch visits all 150 selected TRAIN cases in four updates, including the final six-case macro, with effective batch 48 and microbatch 24. Each Wind epoch visits all 72 selected rows in five updates, including the final eight-row macro, with effective batch 16 and microbatch 8. Thermal retains Q1024 and the original auxiliaries and scheduled response; Wind training retains Q8192 with its original volume/hub objective. The fixed Q1024 five-role Wind fitting audit is evaluation only. Exposed DEV, independent TEST and full-TRAIN identities remain distinct.

Fresh frozen runs target the fit-100 health stop. Healthy pairs then resume to fit500. Checkpoints follow the original 100-epoch cadence, with latest/best aliases, complete history, 10-epoch loss-curve refresh and tqdm progress. A clean stop saves a complete epoch boundary. Engineering one-/two-epoch clean stops do not constitute scientific fit horizons.

The paired 500-to-1000 extension needs a positive, checksum-bound TRAIN decision and prospective resources for both arms and their evidence. Either arm must improve its all-TRAIN priority contrast by at least 3% from fit300 to fit500, or improve the final 100-epoch native-objective median by at least 3% against epochs301–400, with finite gradients in both arms. DEV outcomes do not authorize this decision.

The local signal requires an R-E/R-P priority contrast at most 0.90, every priority component at most 1.05, the same direction on at least three original TRAIN4 fixtures, the declared nonpriority guards, an R-E/H priority contrast at most 0.95, and an original fixed-support native objective no worse than H. Thermal priorities are v/p; Wind priorities are near/downstream Ux. Missing or zero-denominator ratios remain unavailable. The pure decision reducers report every failed check and do not themselves authorize a process launch.

Only the plan's TRAIN access gate admits an independent equal-age R-G fit. Only a useful local signal, clean native-law checks and prospective resource receipts admit joint confirmation. Joint readers start at zero on H e1000; no learned frozen-fit reader or adapted head is imported. The strict joint recipe has no frozen-parameter policy. A frozen e2500-plus-fit result cannot establish equal-age deployment qualification against D0 e2500. No full-TRAIN launch is implicit in this workflow.

## Entry point and source binding

Use the ModularDT environment. New large outputs belong under `/data2/wanglz/ModularDT` with a symlink from the ignored workspace diagnostics directory. Verify the real mount and its filesystem identity, declare persistent and temporary byte budgets, and reuse the verified native shared cache. A RAM allowance does not authorize a new disk cache. The experiment report records actual resource usage and audited round cleanup.

The maintained entry point is `HONF_Proj/tools/hyperedge_readout_train.py`. Recipe creation pins an exact source checkpoint, parent recipe, native provider identity and coordinate convention. CPU preparation allocates no training run and performs zero optimizer updates:

```bash
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES='' \
  /home/wanglz/miniconda3/envs/ModularDT/bin/python \
  HONF_Proj/tools/hyperedge_readout_train.py make-recipe \
  --parent-checkpoint /absolute/retained/H/epoch_2500_model.pt \
  --reduction edge_first --output /absolute/data2/recipe.json

CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES='' \
  /home/wanglz/miniconda3/envs/ModularDT/bin/python \
  HONF_Proj/tools/hyperedge_readout_train.py train \
  --recipe /absolute/data2/recipe.json --device cpu --stop-after 100 --dry-run
```

GPU dispatch exposes exactly the currently authorized physical GPU1 or GPU2 UUID as logical `cuda:0`; it rejects GPU0 or unrestricted visibility. Configure the inherited numerical policy before CUDA initialization. Actual fits additionally require the current experiment's resource reservation and `--yes`. Start with `--stop-after 100`, then resume that output's current latest checkpoint with `--stop-after 500`. Extension uses `--stop-after 1000 --extension-decision /absolute/decision.json` on the exact fit500 resume. A joint recipe uses `--phase joint_confirmation`, the exact H e1000 parent and `--joint-gate-binding`; its fresh fit explicitly branches from that parent and targets e2500.

The R6 source binding hashes a deterministic inventory of shared/native source files and the two maintained training entry points. It does not depend on Python import order. Scientific runs use a separately retained read-only runtime snapshot. Historical Round5 source snapshots, recipe guards and checkpoints remain intact; no `strict=False` loading or catch-all schema fallback is introduced.

`load_hyperedge_readout_components(checkpoint, runtime_root=..., device="cpu")` reconstructs a saved model and its native adapter from checkpoint metadata without constructing a provider, opening a dataset or creating an optimizer. The CPU-only `inspect-checkpoint` command exposes this path on the CLI. Frozen loading and joint loading retain explicit phase and parent-age identities.

## Numerical evidence and limits

The independent tests cover literal FP32/FP64 reader formulas, identity-activation equivalence, subprobability row mass, masks, duplicate indices, permutations, signed coordinate conventions, parameter and input gradients, gradcheck/gradgradcheck, prepared-state invalidation and native Thermal derivatives. Native full-epoch engineering checks additionally establish fresh optimizer birth, nonzero output/input projection gradients, exact save/resume, unchanged frozen tensors and unchanged query streams. Small tests and startup checks remain distinct from completed scientific horizons.

Optional work counters are disabled on timed paths and collected in separate diagnostic passes. Unavailable statistics are omitted or flagged, never recorded as zero executed work. Selected slots, hidden entries and projection rows describe the executed tensor shapes separately from physical fidelity. The tile-byte counter is the size of one largest width-64 hidden tile; it is not total temporary storage or allocator peak. The uncheckpointed nonlinear-activation byte counter is a structural input/output storage bound; complete retained graph storage requires a separate tensor audit. Complete native call latency, prepared-query latency, new-heat response, forward/extraction memory, target-free backward memory and actual optimizer memory are different measured scopes.

Thermal preserves the original explicit signed source columns, heat-independent flow, zero-heat and affine/JVP/VJP laws, native derivatives and endpoint maxima. Precise checks cast raw factors before contraction; historical ordinary-FP32 subtraction misses retain their original thresholds and interpretation. Wind preserves the native profile, normalizer, rotor-diameter frame, component order and 1.5D local support. Learned associations and intervention responses demonstrate computational use, not physical causality. This diagnostic produces no new inverse design, physical reference solve or independent TEST result.
