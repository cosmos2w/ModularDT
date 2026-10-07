# Fresh ordinary D-sep and R-direct formal5000 launch

The user authorized fresh full-dataset training for ordinary D-sep on GPU 0 and R-direct on GPU 2, each for 5,000 epochs in its own tmux session. Ordinary D-sep completed all 5,000 epochs, and fresh R-direct then started with that exact flow endpoint frozen. Both components passed their first-ten-epoch live reviews with finite losses and gradients, all 600 TRAIN cases visited per epoch and 13 optimizer updates per epoch. R-direct remains running toward e5000; its startup review establishes normal training, not mature physical accuracy.

## Run identities and access

All numerical artifacts live outside Git under `/data/wanglz/ModularDT/thermal_formal/rdirect_full5000_20261007_034358`. Both runs have symlinks in `HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs`; existing runs and reference checkpoints were preserved. The launch controller, logs, exact profile copies and receipts are available through the ignored [launch directory](../../diagnostics/generated/formal5000_launch_20261007/launch).

| Stage | Run | Physical GPU / tmux session | Lineage |
|---|---|---|---|
| Ordinary D-sep flow | [Run3901](../../Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_3901_20261007_034358_ordinary_dsep_full5000_v1) | GPU 0 / `honf_3901_dsep_gpu0` | Fresh seed-0 weights, e0→e5000 |
| R-direct thermal | [Run3902](../../Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_3902_20261007_034358_r_direct_full5000_v1) | GPU 2 / `honf_3902_rdirect_gpu2` | Fresh seed-0 thermal weights, exact frozen Run3901 e5000 flow |

The GPU-2 session initially waited without CUDA training and started after the GPU-0 process exited successfully at e5000 and the immutable `epoch_5000_model.pt` existed. A failed or stopped flow process blocks the thermal stage; it does not substitute a development or startup checkpoint. Commands, return codes and failure receipts remain in the launch directory. To inspect either session, use `rtk tmux attach -t honf_3901_dsep_gpu0` or `rtk tmux attach -t honf_3902_rdirect_gpu2`; each session has a `progress` window streaming its corresponding `flow_console_retry1.log` or `thermal_console_retry1.log`. The completed GPU-0 session remains available for log inspection.

The [handoff review](../../diagnostics/generated/formal5000_launch_20261007/launch/handoff_identity_review.json) verified the real D-sep e5000 state: all 38 Adam entries are at step 65,000, with 3,000,000 case visits. The thermal prepared checkpoint was exactly e0 with empty Adam state, and every thermal parameter tensor matched a fresh seed-0 construction. Both components carry exactly identical full-TRAIN dataset, normalization and validation bindings. The frozen flow milestone SHA256 is `aa751d6c51660b4c889f80da842a4e190e1ea80434adf96587312fd1dd769751`; their shared full-TRAIN normalization hash is `b06a65cc1e84176badea9b77431ae5f751ea9672295afd3aad883d0ed61abd86`. The live thermal PID was also verified on physical GPU 2.

## Configuration and later classical comparison

| Setting | Bound formal recipe | Comparison implication |
|---|---|---|
| Dataset | Original packed H5, all 600 original TRAIN cases | Same packed dataset and TRAIN split as Run1804/1502; formal IDs and metadata hashes are explicit |
| Evaluation panels | Canonical89 primary; original90 compatibility | Excludes test 0273, an input-content duplicate of TRAIN 0001, from primary metrics; original90 preserves historical compatibility |
| Normalization | One shared transform fitted on all 600 TRAIN cases only | Preserve each classic checkpoint's native transform; compare denormalized physical errors and use one declared common transform for any standardized comparison |
| Numerical training | Seed 0, FP32, AdamW, effective batch 48 / microbatch 8, 1,024 fluid queries per case | Aligns these configuration values with the classic references; batching order and model objectives still differ |
| Per-component schedule | 5,000 epochs, LR 3e-4 through e2000, cosine to 3e-6 at e5000 | Both component ages must be reported; the composed model has two separately trained stages |
| Flow objective | Ordinary four-field standardized reconstruction | The rejected near-boundary auxiliary is absent |
| Thermal objective | Three-role temperature reconstruction + 0.05 native q proxy + calibrated TRAIN responses and qualified discrete residual | Keeps the reviewed R-direct objective; this differs from the classic joint five-field/port/local-surrogate objective |
| Per-epoch work | 600 visits, 13 optimizer updates, 614,400 fluid queries | Full-data epochs are actual case visits; 65,000 updates per completed component |

The classic config and embedded-normalizer audit is preserved in the [comparison contract](../../diagnostics/generated/formal5000_launch_20261007/launch/classic_comparison_contract.json). The historical normalizers differ from the fresh full-TRAIN transform and remain untouched. Later endpoint comparisons should use literal e5000 states for Run1804, Run1502 and both new components, with selected-best states reported separately. Differences in loss, normalization, component training budget and selection rule must remain visible; alignment of the dataset does not establish an isolated architecture comparison.

The full formal fitters compute canonical89 and original90 validation at the 100-epoch cadence. The maintained offline evaluator now supports explicit formal-panel replay while preserving the strict default DEV22 path. It requires a trusted non-startup R-direct e5000 reference, verifies current full-TRAIN and validation catalog bindings, and keeps the classic checkpoint's native transform. An original90 replay reports canonical89 primary aggregates in `primary_excluding_0273` and original90 compatibility aggregates in `compatibility_including_0273`; compare the same key across model families. These TEST rows are monitored during fitting and must be described as exposed validation panels rather than untouched holdout evidence. The new evaluator route has CPU binding/selection tests; real full-panel native physical replay remains pending the completed thermal endpoint.

After both stages finish, the following commands reproduce a common original90 replay and canonical89 primary summary without changing weights or launching training. Each output must be a new, empty directory; the evaluator refuses to overwrite existing evidence. These commands have not been executed in this launch round.

```bash
PROJECT=/home/wanglz/Desktop/src/ModularDT/HONF_Proj
PYTHON=/home/wanglz/miniconda3/envs/ModularDT/bin/python
RUN_ROOT=/data/wanglz/ModularDT/thermal_formal/rdirect_full5000_20261007_034358
FORMAL=$RUN_ROOT/Run_3902_20261007_034358_r_direct_full5000_v1/epoch_5000_model.pt
CLASSICS=$PROJECT/Trained_Results/ThermalChannel/HONF_Forward_Runs
rtk env CUDA_VISIBLE_DEVICES= "$PYTHON" "$PROJECT/tools/thermal_source_response_evaluate.py" \
  --checkpoint "$FORMAL" --mode fields --formal-panel original90 \
  --output-dir "$RUN_ROOT/evaluation/R-direct_e5000_original90" --device cpu
rtk env CUDA_VISIBLE_DEVICES= "$PYTHON" "$PROJECT/tools/thermal_source_response_evaluate.py" \
  --checkpoint "$CLASSICS/Run_1804_20260905_081349_dense_pairwise_field_adaptation/checkpoints/latest.pt" \
  --mode classic-fields --classic-id Run1804_e5000_latest \
  --formal-reference-checkpoint "$FORMAL" --formal-panel original90 \
  --output-dir "$RUN_ROOT/evaluation/Run1804_e5000_original90" --device cpu
rtk env CUDA_VISIBLE_DEVICES= "$PYTHON" "$PROJECT/tools/thermal_source_response_evaluate.py" \
  --checkpoint "$CLASSICS/Run_1502_20260923_004751_sparse_incidence_environment_sparsemax/checkpoints/latest.pt" \
  --mode classic-fields --classic-id Run1502_e5000_latest \
  --formal-reference-checkpoint "$FORMAL" --formal-panel original90 \
  --output-dir "$RUN_ROOT/evaluation/Run1502_e5000_original90" --device cpu
```

## Monitoring and retained checkpoints

Both exact run-local profile copies retain milestone checkpoints only at e100, e500, e1000, e2500 and e5000. Validation, the replaceable latest checkpoint, the best-field alias and the PDF loss curve update every 100 epochs. The first-ten-epoch curve is emitted without saving an e10 checkpoint or adding a validation-selection age. There are no 25-epoch snapshots, alternate selector aliases or automatic checkpoint deletions. Changed profiles, memberships, normalization, source/partner identity or schedules still fail strict resume checks.

The first attempted start stopped before any optimizer update because the JSON identity receipt represented tuple-valued `heat_columns` as a list. The shared receipt guard now compares the exact JSON representation; the separate checkpoint-to-checkpoint identity check remains strict. A regression test verifies acceptance of the round-tripped tuple and rejection of changed columns. The failed launch receipts remain preserved, and Run3901 restarted from the same verified fresh e0 checkpoint with empty Adam state.

## First ten epochs

| Component | e1 loss | e10 loss | Visits / updates / fluid queries | Live review |
|---|---:|---:|---|---|
| Ordinary D-sep | 1.239691 | 0.823594 | 6,000 / 130 / 6,144,000 | Finite losses and positive finite gradient norms; correct GPU 0 mapping; continued |
| R-direct | 1.684328 | 0.318228 | 6,000 / 130 / 6,144,000 | Finite reconstruction, response, q proxy and operator losses and positive finite gradients; verified GPU 2 mapping; continued |

![Ordinary D-sep first-ten-epoch full-TRAIN loss](../../diagnostics/generated/formal5000_launch_20261007/launch/flow_first10_learning.png)

Figure 1. The saved, visually inspected ordinary D-sep loss curve records the first ten full-TRAIN600 epochs, with normalized four-field reconstruction loss decreasing by 33.57%. No validation point is claimed before the first e100 review. The [first-ten-epoch receipt](../../diagnostics/generated/formal5000_launch_20261007/launch/flow_first10_review.json) preserves each epoch's actual case/query/update counts, gradient norms and training times; this verifies normal training startup rather than mature physical accuracy. [PDF master](../../diagnostics/generated/formal5000_launch_20261007/launch/flow_first10_learning.pdf).

![R-direct first-ten-epoch full-TRAIN thermal loss and actual learning rate](../../diagnostics/generated/formal5000_launch_20261007/launch/thermal_first10_learning.png)

Figure 2. The saved, visually inspected R-direct startup curve shows an 81.11% decrease in standardized three-role temperature reconstruction loss over the first ten full-TRAIN600 epochs, with LR held at 3e-4. Reconstruction including the q proxy fell 1.8997→0.4769, response loss fell 1.0320→0.4171, and q proxy fell 4.3078→3.1735. The unweighted operator residual changed 0.00104→0.00284 and remains finite; improvement in every objective term is not claimed. No validation point is claimed before e100. [PDF master](../../diagnostics/generated/formal5000_launch_20261007/launch/thermal_first10_learning.pdf) and [per-epoch review](../../diagnostics/generated/formal5000_launch_20261007/launch/thermal_first10_review.json).

The maintained curves are [flow_learning.pdf](../../Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_3901_20261007_034358_ordinary_dsep_full5000_v1/flow_learning.pdf) in Run3901 and [thermal_learning.pdf](../../Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_3902_20261007_034358_r_direct_full5000_v1/thermal_learning.pdf) in Run3902. The figure index is Figure 1, ordinary D-sep startup; Figure 2, R-direct startup. Both are fixed first-ten-epoch inspection copies; subsequent monitoring refreshes the maintained curves without rewriting those reviews. Generated PDF masters and the two inline PNG companions remain local and ignored.

Ordinary D-sep completed e5000 with final TRAIN normalized four-field MSE 0.00086096 and canonical89 monitoring MSE 0.00186483. Its actual process interval was 3,041.92 s (50.70 min), including 2,928.42 s training, 2.19 s full-panel validation and 9.79 s measured save time; per-epoch JSON logging and setup account for remaining overhead. The five retained milestones, latest and best-field aliases are present. R-direct's first ten epochs averaged 3.2524 s of training, reached 903,380,480 bytes peak allocated memory, and each used 614,400 fluid queries, 109,344 material queries, 54,672 surface queries and 76,800 operator rows. These measurements price actual training work and do not establish mature thermal accuracy or completed thermal5000 execution.

## Result limits and next step

Predictor evidence in this launch record is live training and the explicitly labelled startup review. Mature physical field, material-peak, response and geometry comparisons remain future endpoint analyses; the prior measured limitations remain in the [readiness report](HONF_RDirect_Formal_Readiness_and_Receiver_Local_Organization_Report.md). Organizer results are unchanged: no R-group retraining or sparse-executor experiment was launched. Inverse results are unchanged: no inverse/Wind campaign or new physical solve was launched, and the retained solve count remains 326/326.

The requested first-ten-epoch reviews are complete. Ordinary D-sep has finished, and R-direct is left running toward its configured e5000 endpoint. The earlier readiness estimate was about 6.07 hours of training alone for both stages; the new thermal startup mean projects about 4.52 hours of thermal training alone, excluding monitoring, logging and saves. This remains a forecast, not completed execution or a guarantee. The maintained source and focused negative checks passed 55 CPU tests, Ruff and compilation checks. Generated profiles, figures, run data and the one-round controller stay ignored and local; only maintained source/tests and launch documentation are uploaded.
