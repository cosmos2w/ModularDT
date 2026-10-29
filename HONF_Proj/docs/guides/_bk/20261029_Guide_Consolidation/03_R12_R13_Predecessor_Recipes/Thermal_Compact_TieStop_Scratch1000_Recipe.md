# Thermal compact tie-stop: full TRAIN600 scratch1000 recipe

**Archived reference — 29 October 2026.** This guide records a predecessor model or a dated experiment, not current architecture policy or standing launch authorization. Use the [current guide index](../../../README.md); see the [archive review](../README.md) for the reason it was moved.

The 25 October2026 run-directory cleanup moved this superseded v1 run, stopped at epoch467, to `_bk/10_20261023_Compact_TieStop_and_U_Controls/thermal_compact_tiestop_full1000_scratch_v1`. The compact soft-only successor remains at the forward-run root. See the [current run map](../../../Thermal_Forward_Run_Archive.md). Commands below retain their historical launch namespace and do not authorize restarting this archived experiment.

This recipe starts a new randomly initialized compact hypergraph model and trains it for 1000 complete TRAIN600 epochs. It uses the maintained Round12 retrieval implementation and its stop-outer-probability tie loss, with canonical exposed VAL89 training-side validation. It reads architecture, physical objectives and optimizer hyperparameters from the formal A2-D recipe; it does not read pretrained weights, saved optimizer moments, mature controller state or an existing run. The frozen executable recipe is [`thermal_compact_tiestop_full1000_scratch_v1.json`](https://github.com/cosmos2w/ModularDT/blob/7981c173e8b1d29ee2b3addb7b972d0f15003542/HONF_Proj/configs/research/thermal_compact_tiestop_full1000_scratch_v1.json).

## Initialization and data

Seed0 constructs a fresh Unified U-H core and copies only these newly generated tensors into the compact core to retain the existing architecture and compact parameter-migration contract. Compact compatibility starts at its maintained negative bias; group projections and copied route/readers use their maintained initialization. All TRAIN600 cases and the canonical89 exposed validation cases come from the existing packed dataset and formal task factory. Normalization is fitted on TRAIN600 only and reused within this run. No dataset copy, persistent dataset cache, solver call, control run or additional evaluation campaign is required. `thermal_u_h_full5000_v1` remains untouched and is not an initialization source.

## Frozen training schedule

| Item | Contract |
| --- | --- |
| Horizon | 1000 full epochs, 600 case visits per epoch; no inherited exposure |
| Randomness | Initialization and case/query sampling seed0; maintained `case_epoch_v1` keys |
| Optimizer | Fresh AdamW, constant LR0.0003, weight decay0.00001, betas(0.9,0.999), eps0.00000001; gradient clip1 |
| Packing | Effective48 cases, nominal microbatch24; maintained OOM rollback and repacking |
| Queries | 1024 primary queries per case; existing native objectives and physical source law |
| Warmup | Epochs1–20 use singleton organization, no structural penalty |
| Soft phase | Epochs21–40 use the maintained tying ramp; tie outer probability is detached |
| Calibration | At21, calibrate tie/capacity once using this run's own completed20 state and next TRAIN macro, at most two microbatches; each coefficient targets0.02 of its participating native gradient norm; no optimizer step and caller RNG restored |
| Hard cut | Immediately after40, accept only strict positive compatibility logits; retain before/after40 checkpoints and freeze compatibility policy |
| Recovery | Epochs41–1000 retain the cut and policy freeze while training the remaining native model |
| Work controller | Fresh counters/windows; native coordinate3000 is an open-schedule origin, not pretrained age; local1–1000 maps to3001–4000; fixed fresh M/E/context lambda0 values are config hyperparameters |
| Compute | Explicit `cuda:0`, all CUDA devices remain visible; dense weighted executor, query tile1024, value checkpointing, segment context |

The planned complete fit makes600000 case visits,13000 optimizer updates and614400000 primary queries, excluding ordinary validation and the bounded no-update calibration. Training-side VAL89 validation, latest-state saves and loss-curve refreshes run every10 epochs. Best-field selection uses those ordinary validation reviews. Complete history retains every training epoch and validation review; separate validation files are written only for declared checkpoint milestones and the final or clean stop.

## Retention and output

The new run is `HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/thermal_compact_tiestop_full1000_scratch_v1`. The workspace `Trained_Results` symlink keeps its declared DATA2 binding. An existing destination, including an empty directory, is rejected. Retain milestone20 for calibration provenance; milestone40 immediately before the hard cut and `epoch_0040_after_cut_model.pt` immediately after it; milestones100,200,300,400,500,600,700,800,900,1000; best/latest aliases; full resolved configs, identity, software/source provenance, terminal log, calibration record and complete training history. Boundary snapshots preserve Adam, RNG, sampler and native clocks; activating the cut adds no update. The runner enforces a10GiB output budget,1GiB temporary budget and200GiB minimum free DATA2 headroom.

Run through the maintained entry point from the repository root, with no `CUDA_VISIBLE_DEVICES` override:

```bash
/home/wanglz/miniconda3/envs/ModularDT/bin/python -u HONF_Proj/tools/honf_research.py train \
  --config HONF_Proj/configs/research/thermal_compact_tiestop_full1000_scratch_v1.json \
  --device cuda:0 --devices cuda:0 --epochs 1000 \
  --output HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/thermal_compact_tiestop_full1000_scratch_v1
```

Launch once in a persistent terminal session with tqdm enabled and terminal output retained on DATA2. Perform one immediate startup check and then stop observing the process. This document freezes an executable experiment; startup does not establish completed training, accepted merges, compact-model accuracy or comparison results.
