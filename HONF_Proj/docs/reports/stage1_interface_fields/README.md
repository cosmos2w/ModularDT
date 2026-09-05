# HONF sparse-interface program: Stage 1 handoff

## Outcome

Stage 1 implemented and trained two matched non-legacy interface-field baselines:

- `dense_pairwise_field`: a contextualized dense pairwise/message-passing adaptation, not an exact Graph Kernel Network or neural-operator reproduction.
- `geometry_latent_field`: a geometry-aware Set-Transformer/Perceiver-style adaptation, not an exact UPT or Transolver reproduction.

Both use a new `encode_case` / `prepare` / `read` boundary alongside `legacy_honf`. They do not instantiate fake hyperedges or pass through the Run-1401 organizer. Both use the same encoders, eight-token coarse route, compact local correction, field/port heads, 24-by-8 environmental grid, frozen Stage-A operator, autonomous predicted ports, one physical refinement, losses, optimizer, batch size 48, 1024 sampled field points, and seed 0. Their weights are independent. Baseline topology is reported as not applicable.

Both requested 500-epoch trainings completed sequentially on GPU 0. The matched 20-case evaluation used exact epoch-500 checkpoints and a subset selected before evaluation from physical module-count and spacing/clustering descriptors. This is the existing development holdout, not a pristine test set.

## Implementation

The architecture selector and typed interface configuration live in `src/honf_forward_core/config.py`. Historical profiles still default to `legacy_honf`, retain their organizer/decoder configuration, and serialize exactly as before. New-family resolved configurations omit legacy organizer and decoder keys.

The reusable core is in `src/honf_forward_core/interface_fields/`:

- `core.py` owns the architecture factory and encode/prepare/read/decode facade.
- `common.py` owns typed coarse cross-attention, continuous coarse read, gather-before-MLP compact local correction, and the common field head.
- `dense_pairwise.py` performs simultaneous signed MM, ME, and EM contextual updates from pre-update states, followed by nonlinear dense query-module messages and a quadrature/geometric environmental attention read. Activation checkpointing is declared in the profile and used only as memory management.
- `latent_attention.py` uses 16 learned main seeds at deterministic 4-by-4 reference coordinates, separate geometry-biased module/environment cross-attentions, two pre-normalized latent blocks, and geometry-biased receiver attention. Its disclosed total memory is 8 coarse + 16 main = 24 latents.
- `types.py` contains the small encoded/prepared/read dataclasses.

`Case_ThermalChannel/src/channelthermal/interface_field_coupling.py` reuses the case adapter and `LocalSurrogateCoupling` helpers. The selected backend supplies initial per-port contexts, provisional outside-temperature reads, and the final cached prepared state. The frozen local network remains differentiable with respect to its port inputs. `PortConditionHead` accepts `[B,M,P,H]` context on the new path and preserves the historical `[B,M,H]` broadcast path. Checkpoint loading remains strict and trusted-resource handling and run allocation were not loosened.

Training now records FP64-accumulated pre-clip gradient norms by encoder/backend/head/local-coupling group, clipping scale, actual parameter-update norms, per-epoch train/validation time, and peak CUDA allocation. Comparison output includes physical near-interface (within 2.5 module radii) and far-fluid errors, compact routing maps, and correctly labelled dense-influence/latent-attention plots.

## Execution record

| Model | Managed run | Epochs | Trainable scalars | Train seconds | Validation seconds | Measured epoch work | Manifest wall time | Peak CUDA MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Dense pairwise adaptation | `Run_1804_20260905_081349_dense_pairwise_field_adaptation` | 500 | 4,395,409 | 8,912.34 | 976.09 | 9,888.42 (2.747 h) | 9,967.15 (2.769 h) | 27,208.66 |
| Geometry latent adaptation | `Run_1801_20260905_110056_geometry_latent_field_adaptation` | 500 | 4,542,489 | 8,057.86 | 891.80 | 8,949.67 (2.486 h) | 9,039.02 (2.511 h) | 23,039.53 |

The complete managed run paths are `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1804_20260905_081349_dense_pairwise_field_adaptation` and `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1801_20260905_110056_geometry_latent_field_adaptation`.

Run 1800 was allocated for the proposed dense ID and failed before completing epoch 1 with a real later-bucket CUDA OOM. Its failed manifest was preserved at `Run_1800_20260905_081027_dense_pairwise_field_adaptation`. A first substitute, Run 1803, exposed a second peak-memory site and was also preserved as failed. Run 1802 was intentionally left available for Stage 2. After removing a duplicated final query read and enabling declared dense-MLP activation checkpointing, the exact failing buckets and a worst observed 12-module batch ran successfully; the allocator then created completed replacement Run 1804. No run was overwritten or resumed under a different configuration.

The first gradient snapshot was large for both fresh models and was clipped, then settled without a finite-value gate:

| Model / epoch | Pre-clip total | Encoder | Backend | Head | Local coupling | Clip scale | Actual update norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense / 1 | 547.461 | 246.558 | 94.809 | 477.653 | 42.218 | 0.001827 | 0.5646 |
| Dense / 20 | 22.313 | 8.770 | 4.067 | 19.731 | 3.888 | 0.04482 | 0.09184 |
| Dense / 500 | 0.569 | 0.343 | 0.213 | 0.397 | 0.058 | 1.000 | 0.1148 |
| Latent / 1 | 449.255 | 204.533 | 65.332 | 392.228 | 43.424 | 0.002226 | 0.5853 |
| Latent / 20 | 26.633 | 7.312 | 4.434 | 25.074 | 2.731 | 0.03755 | 0.1104 |
| Latent / 500 | 1.449 | 0.906 | 0.410 | 1.045 | 0.136 | 0.6903 | 0.09279 |

The full sparse schedule is in each run's `metrics.csv`: epochs 1, 2, 5, 10, 20, and every 50 thereafter.

## Epoch-500 histories

| Model | Endpoint validation total | Best validation total <=500 | Endpoint validation field | Best validation field <=500 |
|---|---:|---:|---:|---:|
| Run 1401 legacy HONF | 0.05887 | 0.04474 (epoch 466) | 0.02360 | 0.01841 (epoch 466) |
| Dense pairwise adaptation | 0.04518 | 0.03011 (epoch 493) | 0.01662 | 0.01339 (epoch 493) |
| Geometry latent adaptation | 0.05693 | 0.05028 (epoch 490) | 0.02721 | 0.02466 (epoch 490) |

The aligned history figure and inspectable table are in the comparison directory at `figures/training_histories.png` and `tables/training_history_summary.csv`.

## Matched 20-case physical comparison

The deterministic case manifest is [`development_cases.json`](development_cases.json). It contains cases 0273, 0653, 0277, 0291, 0298, 0680, 0281, 0654, 0643, 0644, 0276, 0667, 0274, 0293, 0657, 0665, 0282, 0292, 0688, and 0284, covering 3, 5, 7, and 10 active modules.

Mean normalized L2 errors at the exact epoch-500 endpoints:

| Model | Fluid field | Near interface | Far fluid | Internal cell | Internal temperature | Surface temperature | Normal flux |
|---|---:|---:|---:|---:|---:|---:|---:|
| Run 1401 legacy HONF | 0.10940 | 0.10089 | 0.11911 | 0.10970 | 0.11102 | 0.12445 | 0.44467 |
| Dense pairwise adaptation | **0.09352** | **0.08596** | **0.10077** | 0.11597 | 0.12229 | 0.13636 | 0.46178 |
| Geometry latent adaptation | 0.13744 | 0.10679 | 0.16568 | **0.10562** | **0.10462** | **0.10883** | **0.41862** |

Relative to Run 1401, dense improves mean fluid/near/far field error by 14.5%/14.8%/15.4%, while its local/interface errors are 3.8% to 10.2% worse. Latent improves internal-cell, internal-temperature, surface-temperature, and flux errors by 3.7%/5.8%/12.6%/5.9%, while its fluid field is 25.6% worse overall and 39.1% worse in the far field. This is evidence of a global-field versus local-interface tradeoff, not a qualification decision.

For the selected layouts, `M` ranges from 3 to 10 (mean 5.1), `E=192`, and every final grid has `Q=8192`. Counts below are actual receiver/source or message pairs executed by the defined paths, excluding channel-width constants:

| Per-case work | Mean | Min | Max | 20-case total |
|---|---:|---:|---:|---:|
| Dense typed preparation messages per pass: M(M-1)+2ME | 1,984 | 1,158 | 3,930 | 39,684 |
| Latent main preparation score pairs per pass: L(M+E)+2L^2 | 3,666 | 3,632 | 3,744 | 73,312 |
| Shared coarse preparation score pairs per pass | 1,641 | 1,624 | 1,680 | 32,816 |
| Dense final main read pairs: Q(M+E) | 1,614,643 | 1,597,440 | 1,654,784 | 32,292,864 |
| Latent final main read pairs: QL | 131,072 | 131,072 | 131,072 | 2,621,440 |
| Shared coarse final read pairs: QG | 65,536 | 65,536 | 65,536 | 1,310,720 |
| Gathered local final-read neighbours | 2,305 | 1,342 | 4,525 | 46,094 |

The physical loop prepares three times (P0/P1/P2), and also reads ports for autonomous prediction and refinement; the table isolates comparable per-pass preparation and final field-read work. Dense final main read uses 12.3 times as many source/receiver pairs as latent for these cases. The measured training costs above include the complete physical loop and are the stronger end-to-end cost evidence.

Run 1401 alone reports legacy organizer topology (six active edges in this subset's summary). Both adaptations have `topology_applicable=false` and no fabricated organizer values. Dense environmental-influence and geometry-latent query-attention maps for anchors 0273 and 0653 are in `figures/interaction/` under explicit adaptation labels; the corresponding head-averaged arrays are in `debug_npz/`.

Comparison root:

`Trained_Results/ThermalChannel/HONF_Forward_Runs/CompareModels/Stage1_Run1401_1804_1801_Epoch500_20Case`

## Verification and exact commands

Actual forward/backward batches exercised the real packed dataset, frozen Stage-A operator, one refinement, and nonzero gradients through local-operator inputs into the port predictor. Focused tests cover finite outputs, chunk-independent reads, coordinate gradients, joint module permutation, attention-map shape, empty organizer output, and clean new-family serialization. New checkpoints also passed strict save/reload, and the final comparison strictly loaded all three endpoint checkpoints.

```bash
PYTHONPATH=src:Case_ThermalChannel/src /home/wanglz/miniconda3/envs/ModularDT/bin/python -m pytest -q tests Case_ThermalChannel/tests
```

Result: `332 passed, 1 skipped`; the skip is the existing inverse joint-training integration test whose local inverse artifacts are unavailable.

```bash
PYTHONPATH=src:Case_ThermalChannel/src /home/wanglz/miniconda3/envs/ModularDT/bin/python tools/select_stage1_development_cases.py --dataset /data/wanglz/ModularDT/1_ChannelThermal/Processed_ChannelThermal_Dataset/packed_dataset.h5 --split test --count 20 --output docs/reports/stage1_interface_fields/development_cases.json

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:Case_ThermalChannel/src /home/wanglz/miniconda3/envs/ModularDT/bin/python train.py --config src/config_core/forward/dense_pairwise_interface_context.json --run-id 1804 --device cuda:0 --yes

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:Case_ThermalChannel/src /home/wanglz/miniconda3/envs/ModularDT/bin/python train.py --config src/config_core/forward/geometry_latent_interface_context.json --device cuda:0 --yes
```

The first dense invocation omitted `--run-id` and allocated proposed Run 1800; the two preserved pre-epoch failures motivated only the disclosed memory-management implementation, not a batch/query reduction. The latent profile allocated its proposed Run 1801 directly.

```bash
/home/wanglz/miniconda3/envs/ModularDT/bin/python tools/plot_stage1_training_histories.py --series 'Run 1401 legacy HONF=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1401_20260823_151126_stage7_modern_structured_context/metrics.csv' --series 'Dense pairwise adaptation=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1804_20260905_081349_dense_pairwise_field_adaptation/metrics.csv' --series 'Geometry latent adaptation=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1801_20260905_110056_geometry_latent_field_adaptation/metrics.csv' --max-epoch 500 --output-figure Trained_Results/ThermalChannel/HONF_Forward_Runs/CompareModels/Stage1_Run1401_1804_1801_Epoch500_20Case/figures/training_histories.png --output-table Trained_Results/ThermalChannel/HONF_Forward_Runs/CompareModels/Stage1_Run1401_1804_1801_Epoch500_20Case/tables/training_history_summary.csv
```

```bash
PYTHONPATH=src:Case_ThermalChannel/src /home/wanglz/miniconda3/envs/ModularDT/bin/python evaluate.py --config src/config_core/forward/dense_pairwise_interface_context.json --workflow compare --checkpoint-path Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1401_20260823_151126_stage7_modern_structured_context/epoch_0500_model.pt --checkpoint-path Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1804_20260905_081349_dense_pairwise_field_adaptation/epoch_0500_model.pt --checkpoint-path Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1801_20260905_110056_geometry_latent_field_adaptation/epoch_0500_model.pt --label 'Run 1401 legacy HONF @500' --label 'Dense pairwise adaptation @500' --label 'Geometry latent adaptation @500' --case-list docs/reports/stage1_interface_fields/development_cases.json --split test --device cuda:0 --query-batch-size 32768 --local-port-condition-mode predicted --return-routing-maps --save-debug-npz --output-dir Trained_Results/ThermalChannel/HONF_Forward_Runs/CompareModels/Stage1_Run1401_1804_1801_Epoch500_20Case
```

## Known limitations and Stage-2 handoff

- This is one seed on an established development holdout. It does not establish final generalization or publication-level superiority.
- Dense activation checkpointing is necessary at the matched batch/query sizes and contributes to its measured wall time. Its final read remains dense in modules and environmental samples.
- The 24-token total latent communication bottleneck is scalable but loses far-field accuracy at this budget; its stronger local/interface results do not prove adequate global physics.
- The common eight-token route is an explicit low-rank approximation. Context-norm diagnostics can reveal reliance, but Stage 1 does not prove that arbitrary nonlocal physics fits through it.
- ThermalChannel validation is two-dimensional. The core relative-coordinate operations are dimension-derived, but no three-dimensional physical generalization claim is made.

Stage 2 can proceed regardless of this tradeoff. Implement `sparse_interface_honf` within the existing interface-field facade, adding real sparse geometry/incidence construction in `supports.py` and joint group preparation/read in `group_operator.py`. Reuse the same ThermalChannel coupling helper so P0/P1/P2 share layout support indices while group states refresh; keep the common coarse/local paths, dimensions, features, losses, predicted ports, frozen Stage-A operator, and one refinement unchanged. Add dense-reference arithmetic/gradient tests only for the sparse sums, then execute one 500-epoch GPU-0 run through the allocator using proposed Run 1802. Do not add a support-matched control or extend training without an explicit request. Reuse this case manifest and endpoint comparison protocol for the Stage-2 closeout.
