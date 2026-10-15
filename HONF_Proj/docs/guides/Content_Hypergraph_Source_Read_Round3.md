# Round3 source-read study protocol

This opt-in study preserves the v2 A2-D contextualizer and dataset-owned native laws. `source_read_development_v1` uses fixed Thermal TRAIN150/DEV22 and Wind TRAIN72/DEV24 memberships with the successful formal samplers and objectives. These runs are new fresh comparisons; previous Wind five-role development training is historical context. The execution contract is [the Round3 plan](../../UpgradePlan/HONF_A2D_Fidelity_and_Sparse_Execution_Round3_Plan.md).

## Identity and execution

Both datasets initialize common named tensors at seed0. Thermal sampling uses seed0, effective48/micro24 and1024 primary fluid queries plus native auxiliaries. Wind sampling uses seed42, effective16/micro8 and8192 queries:6144 volume plus2048 hub-band. Both use constant all-parameter AdamW LR3e-4, weight decay1e-5 and clip1. Transforms are fitted on the declared TRAIN cohort only. Wind's omitted TRAIN M18 stratum remains a limitation. The native five-role accuracy board remains distinct from its two-region training objective and volume-MSE selector.

`reader_execution_backend="legacy"` preserves historical arithmetic. `"reduced_projection"` reduces 128-wide hidden vectors separately for each output head before applying the corresponding output-projection rows, including actual attention-mass bias. Thermal composes its scalar projection differentiably without attention-weighting K. Local projection reuse retains every support/measure factor. These are execution alternatives at unchanged learned weights, subject to native/gradient/optimizer qualification.

Historical A2-D has no internal query tile:128 is the feature width. `reader_query_chunk_size` is a new explicit optional execution setting; absence preserves whole-request decoding. Ordinary decoding prepares the scene once, slices live query tensors and native local-pair indices, concatenates predictions, and lets the native adapter extract observations from the complete union. Explicit diagnostic traces keep the full-request trace path and its global query-index/cap semantics. Repeated affine-response factor retention is a separate interface. Choose a chunk size only after measured qualification of128/512/2048/8192, and bind it before a lineage starts.

D-scale adds only a bias-free source-length displacement projection, zeroed after name-based initialization:256 Thermal or384 Wind parameters. Common parameters and the initial function match D0. The old config omits new default fields, and historical H6/formal-D protocol guards remain intact.

## Stops and retained output

Development stops at literal2500, reviews100/500/1000, and declares every100-epoch checkpoint. Thermal selects `field_score`; Wind selects `val_volume_mse`. Curves refresh every10 epochs; tqdm remains visible in the terminal and retained log. All validation results stay in full history, with separate validation files only at declared milestones and final or clean-stop reviews. Keep latest, best-field and declared milestones.

`source_read_full1000` is a distinct conditional permission. It rejects stops1001/2500/5000 before model/data/run allocation. Its future schedule horizon remains5000 Thermal or2500 Wind, while the approved execution stop is1000. Unchanged D0 cannot obtain another full run through this protocol. Qualification requires literal2500 native evidence, complete costs, native laws, query/plan ownership, real gradients/optimizer/resume, TRAIN-only calibration, the saved Thermal response guard and a remaining-budget forecast. Each qualification is bound to one named output directory and the exact imported source files. No qualification or fresh full-data execution is implied by an inert recipe or CPU dry-run.

Sparse children attach only to the chosen dense host's literal development e500 checkpoint. They inherit common tensors, Adam moments, RNG, dataset identity and absolute age; added parameters live below `direct_reader.source_read` (inside `core` for Thermal). The attachment binds the exact parent SHA, added names and an inspected per-file source patch. Auxiliary fitting freezes inherited tensors and verifies its TRAIN-only work and changed new buffers. Its optimizer moments begin at their actual auxiliary birth. Absolute epochs501–600 transition to the selected function;601 onward executes selected fine rows. Children reach absolute2500 after2000 new physical epochs.

## Resource ownership

The user's Round3 authorization selects physical GPU0 and GPU2, superseding the plan's older GPU0/1 wording for this study. Historical protocols retain their old device scope. GPU1's unrelated job is protected. Use ModularDT Python, FP32/TF32 off and bounded CPU threads. The phase ceiling is20 aggregate GPU-associated process-hours and14 elapsed hours after first scientific GPU work, reserving the final2 hours for report/cleanup; all preparation, failures, teacher reads and evaluations count. Record live forecasts at100/500/1000 and before optional starts.

The declared round root is `/data/wanglz/ModularDT/diagnostics/attention_round3_20261015T152913Z`. New persistent bytes are capped at12GiB there and1GiB on the home filesystem, with20GiB free-space reserve. Reuse exact verified shared catalogue keys; do not create another persistent dataset/geometry cache or copy source data. Wind TEST and unexposed Thermal targets remain locked, and solver usage stays326/326. Cleanup receipts and one-time helpers stay local and ignored.

## Tested preparation commands

Run from the repository root with `/home/wanglz/miniconda3/envs/ModularDT/bin/python`. The CPU preparation does not allocate a run or perform an optimizer update.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/wanglz/miniconda3/envs/ModularDT/bin/python -B HONF_Proj/tools/content_hypergraph_train.py --recipe HONF_Proj/src/config_core/forward/content_hypergraph/thermal_a2-d_source_read_development_d0_v1.json --device cpu --dry-run --stop-after 100
```

The real-data P0 receipts retain both datasets' exact commands, cohort/transform fingerprints and report-fixture reproduction. Engineering/native qualification, training launch commands and final measurements belong in the Round3 evidence index as they are executed; preparation alone is not a new fidelity or speed result.
