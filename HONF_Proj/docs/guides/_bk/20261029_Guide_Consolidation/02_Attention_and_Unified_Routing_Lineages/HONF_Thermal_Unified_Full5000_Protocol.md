# Thermal unified full5000: classic-protocol comparison

**Archived reference — 29 October 2026.** This guide records a predecessor model or a dated experiment, not current architecture policy or standing launch authorization. Use the [current guide index](../../../README.md); see the [archive review](../README.md) for the reason it was moved.

The user's 22 October 2026 instruction replaces the earlier Round11 200-epoch warm-start experiment with fresh U-H and U-D runs through 5,000 epochs, and a prepared U-S configuration. H is assigned physical GPU0 and D physical GPU1; S must remain unlaunched until explicitly requested. The earlier H run is preserved at local epoch18, 234 optimizer updates and 10,800 case visits, with full optimizer/RNG state in `/data2/wanglz/ModularDT/diagnostics/thermal_honf_round11_20261022/H-full-20261022-GPU0/checkpoints/latest_model.pt`. Its original process exited and its writer lock was released; `protocol_status.json` labels that protocol superseded. It is not an initialization source for these runs.

## Protocol and intentional differences

The source of truth is `HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/thermal_a2d_full5000`, specifically its saved `configs/experiment_identity.json`, optimizer contract, normalization payload, and recorded initialization source revision `10d27ecb`. Metadata are recovered without deserializing or copying its trained weights. The actual saved A2-D core has zero latent hyperedges; the generic recipe's stale `num_hyperedges=6` is not used to define the new architecture.

| Item | Saved classic A2-D | New U-D | New U-H | Prepared U-S |
|---|---|---|---|---|
| Membership | Original TRAIN600; exposed canonical VAL89, excluding 0273 | Same exact IDs/order | Same | Same |
| Normalization | Fitted on original TRAIN600 only | Same saved statistics/key | Same | Same |
| Initialization | Fresh seed0 named initialization; empty AdamW | Fresh seed0 U constructor; empty AdamW | Same policy | Same policy |
| Epochs / case visits / updates | 5000 / 3,000,000 / 65,000 | Same authorized horizon | Same | Prepared same horizon |
| Effective batch / microbatch / primary queries | 48 / 24 / 1024 | Same | Same | Same |
| Precision / sampling | FP32, seed0, case_epoch_v1 | Same classic sampling namespace | Same | Same |
| Optimizer | One formal_joint group; AdamW 3e-4 constant, decay1e-5 on all parameters, betas(.9,.999), eps1e-8; clipping1.0 | Same grouping rule and schedule on active U parameters | Same | Same |
| Native losses and response addendum | Equal-case native flow/T/q-proxy; original TRAIN response once per epoch | Same | Same | Same |
| Context / final access | Original A2-D contextual/readout paths | Unified local-record preparation, all-open context and dense access | Applied learned context and group-composed retrieval | Same applied context family, direct record retrieval |
| Removed summary/aggregate shortcuts | Present in the original architecture | Remain removed | Remain removed | Remain removed |
| Existing work objectives | No continuous sparse-work objective | Disabled for dense control | Existing M/E/context constraints | Existing M/E/context constraints |
| Execution | Historical implementation | Dense-weighted, tile1024, values checkpoint | Same selected settings | Prepared same settings |
| Validation/latest/milestones | Every10 / every10 / 10,50,100,250,500,1000,2500,5000 | Same | Same | Same |
| Curves | Historical cadence, later output amendment | PDF + small PNG every10, live tqdm/log | Same | Same |

Native loss weights remain 0.125 each for u/v/p/omega, 1/6 each for fluid/surface/material temperature, q-proxy 0.05 and response 0.1; operator residual is disabled. Surface/material losses retain within-case then equal-active-module aggregation. The final macro has 24 cases; exactly one original-TRAIN response family is attached to its first microbatch, cycling 0001/0318/0333/0348 by `(epoch + final_update_index) % 4`. Existing fluid/material/surface response scales are 0.2871006010/0.5540040764/0.4400270609. Changing packing preserves per-case query draws and macro loss denominators; partitioned microbatch sampling hashes and FP32 accumulation order can change.

The exact saved normalizer key is `7ab3441ba2616c68224f9d829a92e2db4c59658b45d3f1acf4c535333896939b`. All three models start with new neural weights, source-training age zero and sampler epoch1; neither U-e5000 learned weights nor optimizer moments are loaded. Seed0 is explicit in both the global constructor and base `init_seed`. The current name-based initialization functions match the recoverable classic source functions; the old initial checkpoint itself is not available for a direct tensor comparison. Architecture-specific parameters and zero-initialized E-conditioning follow the implemented U constructor.

For H/S, existing fixed TRAIN-calibrated controller scalar hyperparameters are M=1.0071820021, E=1.5921736956 and context=1.3138799667. These scalar settings are reused with fresh dual values/counters/windows; learned router weights and mature dual state are not copied. U-D has only inactive M/E duals. The native controller uses its existing coordinate convention `3000 + local_epoch`, so sparse age equals the new local epoch; this offset represents no inherited training age and no accelerated 40-epoch ramp. Targets follow the existing 200/1000-age schedule and its plateau. These model-specific work constraints are an intentional difference from the classic all-open A2-D objective.

The comparison matches the named A2-D run's data/native training protocol. It is not a pure grouping ablation: U also changes contextual value exchange and removes summary/aggregate paths, and H versus D changes contextual selection plus final retrieval. Earlier 1404/1804 port/interface objectives are not identical to this native A2-D objective; comparisons to those older architectures must retain that qualification. Signed source contraction, zero heat, heat-independent flow and native extraction remain unchanged. No E-access map, new router, solver run or still-locked target exposure is introduced; canonical89 is the already-exposed membership explicitly authorized by the user.

## Throughput and storage

Actual same-case native TrainingEngine probes on an RTX 6000 Ada included forward, native backward, typed-work derivatives, clipping and AdamW. The original eight-case/tile512/tile-checkpoint setting measured 8.039 seconds per effective48 update; 24-case/tile1024/values measured 2.577 seconds, about 3.12 times higher throughput, at 20.83 GiB peak allocation. Tile2048 was essentially tied at 2.553 seconds with 27.74 GiB allocation. No-checkpoint24 exhausted memory. Packing48 was slightly faster on a mixed batch (2.285 seconds, 41.59 GiB) but exhausted memory on the input-selected largest-M panel and downshifted to24; it is not the default. The chosen 24-case setting processed that high-M panel without fallback at 21.08 GiB. An actual saved-state packing comparison preserved per-case queries and the once-per-epoch response family, with parameter-update relative L2 1.41e-5 and maximum absolute difference 1.19e-7. These are execution diagnostics, not new scientific screens or full-run convergence evidence.

All scientific runs use fresh non-overwriting directories beneath the prescribed `/data2`-backed `Trained_Results/ThermalChannel/HONF_Forward_Runs` location. The recorded allowance is 10 GiB per run, 30 GiB for the three runs, with estimated retained payload 4 GiB per run, peak temporary space 2 GiB per run and minimum filesystem headroom 200 GiB. Existing packed data and shared response caches are reused, with zero new persistent dataset-cache allowance. Initial available `/data2` capacity was 3,229,049,573,376 bytes. Curves, histories, checkpoints and logs remain local scientific artifacts and are excluded from Git.

## Launch and resume

Run commands below from `/home/wanglz/Desktop/src/ModularDT` using the ModularDT environment. Physical GPUs are isolated explicitly; each process sees its selected device as `cuda:0`. The actual launches retain tmux sessions and training logs in the normal results location.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/wanglz/miniconda3/envs/ModularDT/bin/python -B -u HONF_Proj/tools/honf_research.py train --config HONF_Proj/configs/research/thermal_u_h_full5000_v1.json --device cuda:0 --devices cuda:0 --epochs 5000 --output /data2/wanglz/ModularDT/relocated_workspace/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/thermal_u_h_full5000_v1
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/wanglz/miniconda3/envs/ModularDT/bin/python -B -u HONF_Proj/tools/honf_research.py train --config HONF_Proj/configs/research/thermal_u_d_full5000_v1.json --device cuda:0 --devices cuda:0 --epochs 5000 --output /data2/wanglz/ModularDT/relocated_workspace/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/thermal_u_d_full5000_v1
```

U-S is prepared in `configs/research/thermal_u_s_full5000_v1.json`; it has no active process or automatic launcher. To resume a launched run after an interruption, use its exact output directory, a permitted device and `resume --stop-after-epoch 5000`. Do not use `train` or source-weight initialization to resume it. The run manifest retains nominal identity; the latest checkpoint retains weights and exact optimizer/RNG state. When execution overrides or OOM recovery change packing/options, `execution_runtime.json` records those changes separately; its absence means the declared defaults still apply. Native CUDA OOM recovery can downshift packing while preserving effective48 and case/query ordering; any such change is recorded rather than hidden. Evaluation and resume reject checkpoints outside the declared run or with a different saved experiment identity.

## Verified launch record, 22 October 2026

Both processes started from local epoch1 with fresh seed0 weights and completed the epoch10 native validation/checkpoint boundary. Each saved checkpoint contains ten history rows, 6,000 case visits, 130 optimizer updates, named AdamW state and RNG state; PDF and PNG loss curves were saved and visually inspected. Training remains active through the authorized epoch5000 horizon. This verifies startup, the first exposed-VAL89 review and resumable state; it is not a completed full5000 result or evidence of final accuracy, sparse savings or a speed advantage over classic A2-D.

| Run | Physical GPU | Launch PID | Retained tmux session | Verified milestone |
|---|---:|---:|---|---|
| `thermal_u_h_full5000_v1` | 0 | 1432831 | `thermal_u_h_full5000_gpu0` | `checkpoints/epoch_0010_model.pt` |
| `thermal_u_d_full5000_v1` | 1 | 1432837 | `thermal_u_d_full5000_gpu1` | `checkpoints/epoch_0010_model.pt` |
| `thermal_u_s_full5000_v1` | Unassigned | None | None | Prepared only; output directory absent |

Initial complete-epoch train medians were 33.315 seconds for H over epochs1–7 and 19.525 seconds for D over epochs1–13, including native response training, work derivatives and optimizer updates but excluding monitoring. Every observed epoch used 25 microbatches for its 13 effective48 updates, with no packing fallback. Peak training allocation was 21.77 GiB for H and 16.86 GiB for D; peak reserved memory was 24.65/17.43 GiB against 47.38 GiB device capacity, leaving 22.73/29.95 GiB beyond the allocator reservation at that observation. These are early-run measurements and do not bound future memory or runtime as learned support changes.

The log files are retained beside the run directories as `thermal_u_h_full5000_v1.training.log` and `thermal_u_d_full5000_v1.training.log`. Each directory contains `logs/progress.json`, `logs/active_process.json`, `configs/experiment_identity.json`, `research_run.json`, `checkpoints/`, and `plots/training/loss_curves.{pdf,png}`. The generic curve title uses the inherited A2-D base-arm tag; the run directory and experiment identity distinguish U-H from U-D, and exposed validation is the authorized canonical89 panel. Loss history retains every native objective even when the generic plot groups slash-named field losses under other provider objectives.

Compact launch, checkpoint, throughput, source-hash and cleanup receipts are retained under `/data2/wanglz/ModularDT/diagnostics/thermal_honf_round11_20261022/throughput_20261022/`. The dirty-tree launch source is retained once in `launch_source/`; subsequent checkpoint-reader hardening changes no active training computation. Round-owned pytest fixtures were removed after checking the exact allowlists, handles and protected checkpoint hash: 33 files containing 16,364 payload bytes, 15 temporary symlinks and 34 directories across the qualification and final focused-test cleanup. All scientific checkpoints, histories, caches and qualification evidence were retained. Available capacity after this cleanup was 3,228,934,832,128 bytes on `/data2` and 104,719,306,752 bytes on `/home`; active training continues to write within its declared budgets. Protected user/HET-MF edits and the three solver-attempt ledger hashes remain unchanged.

The focused research-runner suite passed all 16 tests in 4.43 seconds on CPU, including fresh metadata-only initialization, shared seed0 tensors, runtime packing and OOM restoration, absolute resume horizons, and rejection of foreign or mismatched checkpoints. The actual H/D epoch10 checkpoint reader also verified matching run identities, nonempty per-parameter AdamW states (207/190 entries), next sampler epoch11 and cursor0. No additional scientific fit, screen or solver evaluation was run for closeout.
