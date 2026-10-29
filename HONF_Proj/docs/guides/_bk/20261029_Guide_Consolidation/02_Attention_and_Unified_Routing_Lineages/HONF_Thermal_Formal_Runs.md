# Thermal HONF research runs

**Archived reference — 29 October 2026.** This guide records a predecessor model or a dated experiment, not current architecture policy or standing launch authorization. Use the [current guide index](../../../README.md); see the [archive review](../README.md) for the reason it was moved.

**Current full-data comparison:** use [the fresh U full5000 protocol](HONF_Thermal_Unified_Full5000_Protocol.md) and `thermal_u_h_full5000_v1.json`, `thermal_u_d_full5000_v1.json`, or the prepared-only `thermal_u_s_full5000_v1.json`. The user's newer instruction supersedes the 200-epoch warm-start comparison below. The original H experiment stopped cleanly at epoch18 and remains preserved; its source weights/optimizer are not reused by the new fresh runs. The older `_full.json`/`_dev150.json` recipes remain available as separately labelled warm-start/development workflows.

The ordinary Thermal research runner starts one model fit, resumes it from its own latest checkpoint, evaluates a saved checkpoint on the exposed development panel, or summarizes the run history; it delegates model updates, losses, optimizers, sampling, validation, checkpoints, and plots to the maintained `TrainingEngine` and Thermal v2 provider.

## Training memberships

The `*_dev150.json` presets train on the fixed25 TRAIN150 development selection and validate on fixed25 DEV22, with a 150-epoch local horizon and a 50-epoch first segment. The `*_full.json` presets train on original TRAIN600, retain the U-triplet source checkpoint's TRAIN150 normalizer unchanged, validate on the same fixed25 DEV22 panel, and offer a 1,000-epoch local horizon with a 200-epoch first segment. The H preset is the recommended starting model, D is the dense-route control, and S is the direct-route control.

The fixed25 DEV22 panel is the only exposed validation partition used by these presets; the remaining original TEST cases stay locked and are never read. A full preset is a user-ready configuration, not approval to start a formal fit. The 200-epoch segment records reviews at epochs 50, 100, 150, and 200; assess the evidence before extending the same run, and stop at the authorized horizon.

`thermal_honf_full_fresh_train600_normalization.json` is a separate full-TRAIN600 normalization experiment. It fits new statistics on original TRAIN600, constructs a new seeded core from architecture configuration without copying source weights, resets optimizer and controller state, samples from local epoch 1, and maps the fresh continuous-work controller through its native near-open coordinate ramp from e3000 to e4000 over 40 local epochs. Its active M/E/context penalty strengths are explicit lambda0 hyperparameters (1.007182, 1.592174, and 1.313880) calibrated from the U10-H e5000 checkpoint; only these scalar strengths are reused, while mature dual values, windows, counters, and model weights are reset. It retains the same source checkpoint only as the architecture and named optimizer-group template; it is not a warm start.

## Execution and storage

Full-TRAIN presets use 24-case microbatches, effective batch 48, 1,024 primary queries per case, query tile 1,024 and values-only activation checkpointing. They retain `dense_weighted` execution and segment context. The development presets retain eight-case packing, tile512 and tile checkpointing. Full-run defaults were amended after native optimizer measurements showed approximately 3.12 times higher update throughput than the original eight-case/tile512/tile-checkpoint launch; see the throughput amendment below.

The earlier B=2 M1/M12 cost probe at tile512 measured a combined field-plus-work backward, excluding the separate engine VJPs, optimizer and response addenda: 0.848 seconds/612.4 MiB peak allocation for dense-weighted, 0.847 seconds/728.3 MiB for source-order and 0.934 seconds/873.4 MiB for segment. Those measurements remain historical cost evidence. Native global-gradient differences were below 5.2e-5, while some weak/cancelling FP32 per-parameter gradients exceeded 1e-4 relatively despite tiny absolute differences; the independent FP64 reference passed. Dense-weighted executes every value in its rectangular tensors, including closed gates and padding. The case-board's 22% E access describes reference packed execution and does not imply 22% executed values under dense-weighted.

Every preset stores outputs under `/data2/wanglz/ModularDT/diagnostics/thermal_honf_round11_20261022` and declares a 20 GiB output budget, 4 GiB peak temporary-space cap, and 200 GiB minimum free-space headroom. Before each output-producing operation, the runner verifies that `/data2` is mounted, the output stays beneath the configured root on that filesystem, and the requested byte estimates fit those limits. It reuses the existing dataset and shared cache and does not copy the dataset or create a persistent dataset cache.

CUDA ordinals refer to the process-visible device list; set `CUDA_VISIBLE_DEVICES` before starting Python if physical GPU selection is needed. When offering a fallback, include every allowed visible device in `--devices`; the requested `--device` must appear in that list. CUDA OOM recovery within an optimizer update restores its starting model, optimizer, RNG and provider state, then retries the same ordered cases with smaller case packing at most twice. Effective batch, loss normalization and per-case sampling keys stay unchanged. Runtime packing is recorded separately from nominal run identity and restored on resume. A process never migrates a live CUDA computation between devices; use the ordinary resume command to move a saved run.

## Commands

Run the recommended H preset's first segment after the formal fit is authorized; the output path must be new and inside the configured root.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=HONF_Proj/src:HONF_Proj/Case_ThermalChannel/src:HONF_Proj/Case_WindFarm/src \
CUDA_VISIBLE_DEVICES=0,1 \
/home/wanglz/miniconda3/envs/ModularDT/bin/python HONF_Proj/tools/honf_research.py train \
  --config HONF_Proj/configs/research/thermal_honf_full.json \
  --device cuda:0 --devices cuda:0 cuda:1 --epochs 200 \
  --output /data2/wanglz/ModularDT/diagnostics/thermal_honf_round11_20261022/H-full
```

The full D and S controls use their matched presets and distinct output directories; select the fresh-normalization command only when that separate experiment is intended.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=HONF_Proj/src:HONF_Proj/Case_ThermalChannel/src:HONF_Proj/Case_WindFarm/src CUDA_VISIBLE_DEVICES=0,1 /home/wanglz/miniconda3/envs/ModularDT/bin/python HONF_Proj/tools/honf_research.py train --config HONF_Proj/configs/research/thermal_dense_full.json --device cuda:0 --devices cuda:0 cuda:1 --epochs 200 --output /data2/wanglz/ModularDT/diagnostics/thermal_honf_round11_20261022/D-full
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=HONF_Proj/src:HONF_Proj/Case_ThermalChannel/src:HONF_Proj/Case_WindFarm/src CUDA_VISIBLE_DEVICES=0,1 /home/wanglz/miniconda3/envs/ModularDT/bin/python HONF_Proj/tools/honf_research.py train --config HONF_Proj/configs/research/thermal_direct_full.json --device cuda:0 --devices cuda:0 cuda:1 --epochs 200 --output /data2/wanglz/ModularDT/diagnostics/thermal_honf_round11_20261022/S-full
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=HONF_Proj/src:HONF_Proj/Case_ThermalChannel/src:HONF_Proj/Case_WindFarm/src CUDA_VISIBLE_DEVICES=0,1 /home/wanglz/miniconda3/envs/ModularDT/bin/python HONF_Proj/tools/honf_research.py train --config HONF_Proj/configs/research/thermal_honf_full_fresh_train600_normalization.json --device cuda:0 --devices cuda:0 cuda:1 --epochs 200 --output /data2/wanglz/ModularDT/diagnostics/thermal_honf_round11_20261022/H-fresh-train600-normalization
```

For the fixed25 TRAIN150 development protocol, select `thermal_honf_dev150.json`, `thermal_dense_dev150.json`, or `thermal_direct_dev150.json`; the preset defaults to 50 epochs and the full 150-epoch local horizon remains available for explicit extensions.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=HONF_Proj/src:HONF_Proj/Case_ThermalChannel/src:HONF_Proj/Case_WindFarm/src \
CUDA_VISIBLE_DEVICES=0,1 \
/home/wanglz/miniconda3/envs/ModularDT/bin/python HONF_Proj/tools/honf_research.py train \
  --config HONF_Proj/configs/research/thermal_honf_dev150.json \
  --device cuda:0 --devices cuda:0 cuda:1 \
  --output /data2/wanglz/ModularDT/diagnostics/thermal_honf_round11_20261022/H-dev150
```

Resume by additional local epochs without re-reading the original source checkpoint or config file; the run manifest and selected checkpoint carry the resume contract.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=HONF_Proj/src:HONF_Proj/Case_ThermalChannel/src:HONF_Proj/Case_WindFarm/src \
CUDA_VISIBLE_DEVICES=0,1 \
/home/wanglz/miniconda3/envs/ModularDT/bin/python HONF_Proj/tools/honf_research.py resume \
  --run /data2/wanglz/ModularDT/diagnostics/thermal_honf_round11_20261022/H-full \
  --device cuda:0 --devices cuda:0 cuda:1 --additional-epochs 50
```

Evaluate the current best-field checkpoint or latest checkpoint on fixed25 DEV22, then inspect actual visits, updates, validation history, and retained checkpoints with `assess`.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=HONF_Proj/src:HONF_Proj/Case_ThermalChannel/src:HONF_Proj/Case_WindFarm/src CUDA_VISIBLE_DEVICES=0,1 /home/wanglz/miniconda3/envs/ModularDT/bin/python HONF_Proj/tools/honf_research.py evaluate --run /data2/wanglz/ModularDT/diagnostics/thermal_honf_round11_20261022/H-full --device cuda:0 --checkpoint best
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=HONF_Proj/src:HONF_Proj/Case_ThermalChannel/src:HONF_Proj/Case_WindFarm/src /home/wanglz/miniconda3/envs/ModularDT/bin/python HONF_Proj/tools/honf_research.py assess --run /data2/wanglz/ModularDT/diagnostics/thermal_honf_round11_20261022/H-full --dataset thermal
```

Case-limited fits require both `--diagnostic-only` and an explicit `--case-limit` or one or more `--case-id` values. They qualify code paths only and must be reported as diagnostic plumbing evidence, not as a fitted research result.

## Round 11 pipeline qualification

The runner completed a two-case diagnostic fit on physical GPU1 for two local epochs, then resumed its own latest checkpoint on physical GPU0 for one more epoch, evaluated epoch 3 on CPU over DEV22, and summarized the saved run. Across the three epochs it recorded 6 TRAIN case visits and 3 optimizer updates; the three optimizer epochs are a pipeline qualification only, not a model-quality or full-membership result. The run manifest records fixed25 DEV22, no locked TEST reads, inherited TRAIN150 normalization, source and execution identity, complete checkpoint state, and `/data2` storage accounting. A formal full-membership epoch uses 13 optimizer updates for 600 TRAIN cases at effective batch 48, so the first 50-epoch review requires 650 updates.

The measured training engine elapsed times and process wall times are recorded in `research_worker_usage.json` in that diagnostic run; CUDA memory comes from the TrainingEngine's per-epoch peak allocation record. A separate disposable native engine macro on 48 TRAIN150 cases, six case microbatches of eight and Q=1024, including separate work derivatives, clipping and one AdamW update, took 8.833 seconds and 2.363 GiB peak allocation on an RTX 6000 Ada. Scaling that single macro gives approximately 1.59 compute hours to the first full-TRAIN 50-epoch review, or 6.38 hours for 200 epochs. This is a planning extrapolation, excluding validation, plotting, checkpointing, response addenda, differing module mix and contention; it is neither measured TRAIN600 throughput nor a convergence forecast.

## Authorized Round11 full-TRAIN H run

On 22 October 2026 the user explicitly authorized one H full-TRAIN fit on physical GPU0, with GPU1 available for finishing the development work. This supersedes the plan's preparation-only boundary for this one run. Its configured stop is local epoch 200, with reviews at 50/100/150/200; the configured 1,000-epoch horizon supports later explicit continuations and does not authorize them. D, S and fresh-normalization fits remain prepared commands. The optional E-access trial is skipped.

The launched run uses the H preset above, the original U10-H best-field e5000 weights and TRAIN150 normalization, fresh AdamW moments at the source terminal group rates, and the original TRAIN600 population. Its dedicated output directory is `/data2/wanglz/ModularDT/diagnostics/thermal_honf_round11_20261022/H-full-20261022-GPU0`; the startup record and log are retained beside it. `CUDA_VISIBLE_DEVICES=0` and `--device cuda:0 --devices cuda:0` restrict this process to physical GPU0. After startup is verified it runs independently; this delivery does not wait for its training results. Use that exact directory with `assess`, `evaluate` or `resume` when reviewing the run later.
