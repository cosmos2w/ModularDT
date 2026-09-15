# WindFarm larger-sample fresh runs

On 2026-09-13, the user requested stopping continuations 2100/2101 and starting
fresh models with larger batches, more native queries and larger query chunks.
Run 2100 stopped after recorded epoch 631; Run 2101 after epoch 581. Their
checkpoints, CSV histories and loss figures remain in their original directories.
The manifests record `stopped_by_user`; no historical checkpoint was overwritten
as part of stopping. The initial 500-epoch study remains a separate result.

The new runs use scratch initialization (seed 0), AdamW 3e-4 / weight decay 1e-5,
clip norm 1, no AMP/dropout, and the same two architectures, training-only target
normalization, layout-grouped split, native coordinate view and geometry E512.
Batch size is 16, queries per case 8192 (6144 volume + 2048 rotor-height band),
and receiver chunk size 512. There are 27 updates per epoch, with four cases in
the last batch. All 420 training cases are visited each epoch with fresh samples.
The target is 2500 total epochs. No prior model/optimizer state is loaded.

A real one-update disposable batch used 16 training rows with 29–30 turbines.
Both models produced finite losses/gradients and nonzero parameter updates:

| Model | Peak allocated / reserved MiB | Batch wall seconds | Preclip norm | Sampled update norm |
|---|---:|---:|---:|---:|
| Classic | 34435.10 / 35064 | 0.785 | 3.30348 | 0.0158881 |
| Dense | 25779.68 / 28492 | 1.496 | 0.938795 | 0.0158093 |

The probe was run on physical GPU 1 and discarded. This is a resource/backward
check, not a generalization result or a promise of constant GPU utilization.
The first attempt stopped in configuration parsing because the classic core
correctly rejects an interface backend configuration block. A case-owned
`dataset.receiver_chunk_size` execution option fixes that placement while
preserving historical defaults. No memory error or scientific model change
was required. Five focused numerical/render tests passed before launch.

## Actual launches

- Run 2102, physical GPU 1: `/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Trained_Results/WindFarm/HONF_Forward_Runs/Run_2102_20260913_135849_windfarm_classic_k6_b16_q8192`
- Run 2103, physical GPU 2: `/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Trained_Results/WindFarm/HONF_Forward_Runs/Run_2103_20260913_135849_windfarm_dense_pairwise_b16_q8192`

Both launched through root `train.py` and the existing allocator, with no resume
or initialization checkpoint. Both completed real updates from epoch 1. Their
launch source commit is `11e3ca8`.

Each run writes `metrics.csv` every epoch and `plots/loss_history.png` after
epoch 1, every 10 epochs and at completion. Figures distinguish training
objective, volume train/validation MSE and rotor-height-band train/validation
MSE. Validation remains every five epochs with its existing fixed samples.
Best checkpoints use validation volume MSE. `latest_model.pt` updates every
10 epochs; retained epoch checkpoints are 10, 50, 100, 250, 500, 1000, 1500,
2000 and 2500. The source-level plot hook replaces the earlier one-time
postprocessing approach for these new runs.

Actual commands/PIDs/logs are in the ignored
`Case_WindFarm/diagnostics/generated/forward_velocity_capacity/launch.json`;
real batch evidence is `batch_probe.json` in that directory. New profiles are
`src/config_core/forward/windfarm_classic_k6_b16_q8192.json` and
`src/config_core/forward/windfarm_dense_pairwise_b16_q8192.json`, using
`Case_WindFarm/configs/forward_velocity_b16_q8192.json`.

No ongoing agent monitoring is scheduled. These larger-sample results must not
be treated as the same training budget as the initial study: each epoch uses
8 times as many supervised samples, while doubling batch size halves the
approximate optimizer-update count per epoch. There is no new test evaluation
or automatic follow-on run in this launch task.

For future launches, the training workflow saves the figure at
`plots/training/loss_history.png`. Runs 2102/2103 were already running when
this path correction was made; their existing outputs and processes were
left unchanged.
