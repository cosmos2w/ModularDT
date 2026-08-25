# Stage 6 Frozen-Screen Evaluation

## Decision

**Do not promote S1, S2, or S3.** The frozen evidence does not support a larger descriptor-content residual or the proposed query-locality prior as the Stage-6 scientific intervention.

- S1 (`content scale = 0.70`) nearly doubled pooled field MSE on both screened checkpoints and made the epoch-1500 query profiles less distinct.
- S2 (`scale = 0.35`, bounded Gaussian query locality `0.25`) produced only small, inconsistent routing changes. On the full split it improved epoch-1500 MSE by 2.85%, but worsened the best checkpoint by 2.59%; best-checkpoint `v` MSE worsened by 8.57%.
- Neither variant approached the frozen-screen query targets or useful retained-mass route reduction.
- S3 was not run because neither component was individually promising, as required by the plan.

The only evidence-supported final profile is therefore **S0**, the exact Run-1302 fixed-softmax control: descriptor-first scale `0.35`, learned query geometry retained, and no added query-locality prior. This is a controlled negative Stage-6 result. The final profile is safe and reproducible, but a long run with it would be a Run-1302 replication rather than a validated role-consistency upgrade.

## Scope and provenance

Frozen checkpoints:

| Checkpoint | Epoch | SHA-256 |
|---|---:|---|
| Run 1302 milestone | 1500 | `5ffbf6b30443f237c26c84d7010d75d89c3e5f44c5262e3ffd6110f48704192f` |
| Run 1302 best by field MSE | 2497 | `a7dd82d16597e6db6c1883b368cd54e8be90e774c007ce08d408607400427789` |

The deterministic screen used test cases `0273`–`0292`. The confirmation used the complete 90-case test split. All variants used identical checkpoint tensors, dataset fingerprint `4224093c22a67af4adfecc8b21d53548e4263ec2254c230dc83c89526b36da05`, checkpoint-owned normalization, and channel order `u, v, p, omega, temperature`. Evaluation ran on `cuda:2`.

No weights were updated. The evaluation-only overrides preserve all 251 state-dict keys.

## Deterministic 20-case screen

MSE is pooled normalized fluid-domain MSE. Structural values are medians across cases. Lower cosine/entropy and higher rank/spatial standard deviation indicate more differentiated query roles.

| Variant | Fluid MSE | vs S0 | Query cosine | Query rank | Query entropy | Query spatial std | Pairwise cosine | Routes removed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Epoch 1500 S0 | 0.003438 | — | 0.786 | 1.904 | 0.921 | 0.0770 | 0.733 | 0.74% |
| Epoch 1500 S1 | 0.006803 | **+97.9%** | 0.813 | 1.776 | 0.934 | 0.0702 | 0.745 | 0.70% |
| Epoch 1500 S2 | 0.003406 | -0.9% | 0.781 | 1.931 | 0.923 | 0.0779 | 0.735 | 0.84% |
| Best S0 | 0.001783 | — | 0.851 | 1.579 | 0.944 | 0.0591 | 0.788 | 0.20% |
| Best S1 | 0.003633 | **+103.7%** | 0.847 | 1.655 | 0.943 | 0.0613 | 0.796 | 0.28% |
| Best S2 | 0.001831 | +2.7% | 0.858 | 1.561 | 0.948 | 0.0577 | 0.786 | 0.14% |

S1 fails on accuracy and does not improve the epoch-1500 topology. S2 is neutral-to-negative on the best checkpoint and its small epoch-1500 changes are far below the suggested screen targets: query cosine `<0.65`, rank `>2.5`, entropy `<0.88`, and route reduction `>10–15%`.

## Complete 90-case confirmation

S0 and the least-bad intervention, S2, were promoted to the full split for confirmation.

### Accuracy

| Checkpoint | S0 pooled MSE | S2 pooled MSE | S2 change | S0 case p95 | S2 case p95 |
|---|---:|---:|---:|---:|---:|
| Epoch 1500 | 0.004331 | 0.004208 | -2.85% | 0.00776 | 0.00750 |
| Best | **0.001786** | 0.001832 | **+2.59%** | 0.00383 | 0.00382 |

Per-channel S2 changes relative to S0:

| Checkpoint | `u` | `v` | `p` | `omega` | `temperature` |
|---|---:|---:|---:|---:|---:|
| Epoch 1500 | -0.07% | -1.53% | -4.45% | +0.23% | -4.59% |
| Best | +2.44% | **+8.57%** | +2.57% | +0.73% | +2.82% |

The best-checkpoint result is the more relevant frozen deployment test. It rejects S2 because the aggregate field worsens and one channel exceeds the 5% screening allowance.

### Query, pairwise, and additive structure

| Metric, median | Run 1000 reference | Epoch-1500 S0 | Epoch-1500 S2 | Best S0 | Best S2 |
|---|---:|---:|---:|---:|---:|
| Query profile cosine ↓ | 0.436 | 0.716 | **0.705** | 0.784 | **0.775** |
| Query effective rank ↑ | 3.702 | 2.170 | **2.256** | 1.966 | **2.011** |
| Query entropy ↓ | 0.763 | 0.901 | **0.900** | 0.928 | **0.923** |
| Query spatial std ↑ | 0.155 | 0.0919 | **0.0946** | 0.0792 | **0.0805** |
| Pairwise-map cosine ↓ | 0.462 | 0.632 | **0.627** | 0.707 | **0.700** |

S2 moves each median in the intended direction on the full split, but the movement is small: best query cosine changes by only `-0.0092`, and rank by `+0.045`. It does not cross any frozen-screen target and does not reverse the late topology drift.

Per-channel additive edge-map cosine at the best checkpoint:

| Variant | `u` | `v` | `p` | `omega` | `temperature` |
|---|---:|---:|---:|---:|---:|
| S0 | 0.789 | 0.749 | 0.731 | 0.785 | 0.725 |
| S2 | 0.789 | **0.744** | **0.724** | **0.782** | **0.716** |

These changes are again too small to meet the intended `<0.60` mechanism-diversity target. Environment organization is unchanged by construction: best-checkpoint environment cosine is about `0.258`, effective rank `1.49`, largest occupancy `0.857`, and region separation `0.247`.

### Retained-mass pruning and runtime

The same evaluation-only floors were used throughout: query mass `0.98`, routed-module mass `0.95`, and at least one retained route.

| Checkpoint | Variant | Query routes removed | Aggregate MSE change | Query retained mass p05 | Dense decoder median |
|---|---|---:|---:|---:|---:|
| Epoch 1500 | S0 | 1.87% | +0.69% | 0.9855 | 8.97 ms |
| Epoch 1500 | S2 | 1.92% | +0.57% | 0.9852 | 9.14 ms |
| Best | S0 | 1.09% | +0.22% | 0.9924 | 9.06 ms |
| Best | S2 | 1.07% | +0.14% | 0.9927 | 9.17 ms |

The dedicated fixed-case benchmark used case `0273`, 8192 queries, 10 warmups, and 40 synchronized iterations on the same GPU:

| Variant | Prepared decoder median | Full forward median | Incremental peak allocation | Parameters |
|---|---:|---:|---:|---:|
| S0 | 8.894 ms | 28.328 ms | 410.2 MB | 3,979,458 |
| S2 | 9.010 ms | 28.367 ms | 410.2 MB | 3,979,458 |

S2's full-forward overhead is only 0.14%, so efficiency is not the reason for rejection. The scientific effect is simply too weak and inconsistent.

## Interpretation

The fixed organizer's useful environment geometry is not being preserved downstream, but the tested frozen score adjustments do not repair that loss:

1. Increasing descriptor content changes a representation that downstream weights were not trained to consume and severely damages prediction.
2. The existing bounded locality term is cheap and numerically sound, but strength `0.25` is too weak to materially restructure a checkpoint whose learned content scores are already diffuse; it also creates checkpoint-dependent field trade-offs.
3. Because no intervention reaches the screen targets, training S3 would conflate two unsupported changes.
4. The late Run-1302 drift remains the strongest clue: topology is healthier at epoch 1500 than at the best field checkpoint, while field error continues to improve. The next scientific work should focus on preserving formed roles during optimization, not on forcing a frozen geometric bias.

## Evidence artifacts

- `stage6_frozen_screen_20_topology/`
- `stage6_frozen_screen_20_accuracy/`
- `stage6_frozen_screen_20_pruning/`
- `stage6_frozen_screen_full90_topology/`
- `stage6_frozen_screen_full90_accuracy/`
- `stage6_frozen_screen_full90_pruning/`
- `stage6_frozen_benchmark.json`
- `stage6_profile_dry_run.json`

All are evaluation-only outputs; no training was launched.
