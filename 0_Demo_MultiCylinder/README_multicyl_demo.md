# Multi-cylinder PhiFlow Demo

Updated: 2026-04-24

This demo covers the periodic multi-cylinder inert wake workflow:

1. simulate wake fields with PhiFlow
2. preprocess raw cases into a packed HDF5 dataset
3. train a hypergraph-organized neural field surrogate
4. reconstruct full flow fields from saved checkpoints

## Current Dynamic-Residual Upgrade

This revision targets the residual-dynamics plateau with four scoped changes:

1. Local top-k refinement is now wake-aware. Module and environment tokens are
   ranked by periodic proximity, hyperedge wake relevance, and learned local
   attention bias.
2. Dynamic mode capacity was increased modestly with more hyperedges, wider
   dynamic tokens, and a second local refinement layer.
3. A single dynamic-energy loss matches masked residual energy, defaulting to
   the omega channel, to discourage residual/vorticity amplitude collapse.
4. Phase-window batches replace part of each training item with the same xy
   locations sampled across multiple tau values from `canonical_cycle`, so the
   residual branch sees phase-dependent motion rather than isolated snapshots.
5. `loss_curve.png` is now a 2x2 figure: total, field, residual, and dynamic
   energy.

## Periodic Boundary Assumption

The PhiFlow simulator in this demo uses periodic boundaries. The surrogate keeps
that assumption by design:

- all organizer and decoder proximity features use periodic minimum-image
  geometry
- wake direction uses directed periodic downstream distance in `+x`
- the model is not an open-channel approximation

Two periodic x-semantics matter throughout the code:

- minimum-image periodic distance: shortest wrapped separation, used for
  proximity and top-k local gathering
- directed periodic downstream distance: `(x_dst - x_src) mod Lx`, used for
  wake direction, wake relevance, and wake-focused sampling

## Model Summary

The forward path is:

`structure -> organizer -> behavior / dynamic memory -> mean branch + residual branch -> pred_field`

Key outputs from `src/model.py`:

- `pred_field`, `pred_mean`, `pred_residual`
- `freq_pred`
- `A_me`, `A_mh`, `A_eh`
- `hyper_source_coords`
- `hyper_wake_coords`
- `hyper_wake_axis`
- `hyper_wake_extent`
- `hyper_strength`
- `dynamic_global_token`
- `dynamic_hyper_base`
- optional `hyper_phase_offsets`, `hyper_phase_sin_coeff`, `hyper_phase_cos_coeff`

### Organizer

The organizer now returns both source-centered and wake-centered hyperedge
geometry.

- `hyper_source_coords` are circular weighted means of cylinder coordinates from
  `A_mh`
- `hyper_wake_coords` are circular weighted means of environment-token
  coordinates from `A_eh`
- `hyper_wake_axis` points from source center to wake center under periodic
  geometry
- `hyper_wake_extent` is a lightweight weighted wake spread

This makes hyperedges represent interaction regions rather than only
source-cylinder clusters.

### Dynamic Hyper Tokens

The residual branch no longer relies on a mostly static hyper token alone.
Dynamic hyper memory is phase-conditioned with lightweight low-rank harmonic
modulation:

- `dynamic_hyper_base`: per-hyperedge base dynamic memory
- harmonic coefficients: low-rank `sin/cos` phase responses per hyperedge
- query-time phase context: combined with hyperedge relevance weights over
  `K_h`, without building a dense `[B, Q, K_h, D]` tensor

The decoder keeps spatial and phase encoders separate:

- spatial encoder for `(x, y)`
- harmonic phase encoder for `tau`

### Decoder

Mean branch:

- spatial query encoding
- shallow global read over structured memory
- predicts smooth mean structure

Residual branch:

- spatial encoding + phase encoding + dynamic global token + phase-conditioned
  hyper context
- wake-aware global attention over all memory
- top-k local refinement over wake-relevant module/env tokens plus all hyper
  tokens

Local module/env gathering still preserves periodic minimum-image distance, but
the default ranking also uses hyperedge wake relevance and learned attention
bias. Hyperedge relevance uses wake-centered geometry.

## Training Changes

This revision improves dynamic reconstruction by changing representation and
sampling, not by adding a large loss stack.

### Cylinder-order randomization

Training now randomizes the valid cylinder order each epoch before padding to
`max_num_cylinders`. This reduces slot/index memorization and improves
generalization across layouts. Validation stays deterministic by default.

### Wake-focused point sampling

Training point subsampling now supports `dataset.point_sampling_mode =
"wake_focused"`.

The sampler still starts from a contiguous chunk, then mixes:

- uniform points
- near-cylinder points using periodic minimum-image distance
- downstream wake points using directed periodic downstream distance in `+x`
- high-`|omega|` points

Validation stays deterministic uniform by default unless explicitly configured
otherwise.

### Losses

The objective stays intentionally simple:

- `field_mse`
- `residual_mse`
- light `freq_mse`
- dynamic-energy loss on residual energy, defaulting to omega
- light direct organizer supervision

Recommended defaults:

- `field_mse_weight = 1.0`
- `residual_mse_weight = 1.25`
- `mean_mse_weight = 0.0`
- `freq_mse_weight = 0.05`
- `dynamic_energy_weight = 0.02`
- `organizer_me_weight = 0.01`
- `organizer_mm_weight = 0.05`
- `organizer_consistency_weight = 0.05`

No spectral, gradient, adversarial, diffusion, or flow-matching loss was added
in this revision.

## Validation And Checkpoint Selection

Validation logs:

- `val_total_loss`
- `val_field_mse`
- `val_mean_mse`
- `val_residual_mse`
- `val_freq_mse`
- `val_dynamic_energy`
- `val_residual_focus`

By default:

`val_residual_focus = val_residual_mse + residual_focus_field_weight * val_field_mse`

Best-checkpoint selection now follows `validation.best_metric_name`, which
defaults to `val_residual_focus`.

## Organizer Visualization Diagnostics

`src/evaluate.py` writes several organization diagnostics for each evaluated
case. The old ambiguous physical-plus-tripartite figure has been split into
clearer outputs:

1. `organization_physical_*.png`
   - physical-domain overlay plus a hyperedge summary table
   - cylinder locations, environment tokens, source centers, wake centers, and
     learned wake-axis arrows
   - source-to-wake and cylinder-to-hyperedge links use periodic shortest-image
     geometry

2. `organization_matrices_*.png`
   - `A_mh` heatmap for cylinder/module to hyperedge assignment
   - `A_eh` heatmap for environment-token to hyperedge assignment
   - per-hyperedge spatial maps showing where each hyperedge owns environment
     tokens

3. `organization_sankey_*.png`
   - abstract tripartite graph for topology/debugging
   - `C_i` means cylinder/module
   - `H_k` means learned interaction hyperedge/group
   - `EnvGroup_k` means environment tokens dominated by `H_k`
   - hyperedge and environment-group rows use collision-avoiding vertical
     layout, so labels do not all inherit the same wake-center y coordinate

4. `organization_summary_*.csv` and `organization_summary_*.json`
   - machine-readable hyperedge summary with strength, source/wake centers,
     wake axis, extent, top cylinders, and top environment tokens

Useful evaluator arguments:

- `--organization-view {all,physical,matrices,sankey}` selects which diagnostic
  figures to render; the default is `all`
- `--organization-threshold` controls which soft assignment edges are drawn
- `--topk-me-links` controls optional light cylinder-to-environment links
- `--organization-topk-cylinders` controls how many cylinder memberships appear
  in the summary table/export
- `--organization-topk-env` controls how many environment tokens appear in the
  summary table/export
- `--organization-min-gap` controls vertical spacing in the Sankey layout
- `--no-organization-table` hides the side table in the physical view

## Default Config Highlights

The default config lives in `Config_Train/train_config_template.json`.

Important defaults in this revision:

- `num_env_tokens_x = 24`
- `num_env_tokens_y = 8`
- `num_hyperedges = 6`
- `hidden_dim / behavior_dim / latent_dim / dynamic_token_dim = 80`
- `phase_fourier_frequencies = 2`
- `use_phase_conditioned_dynamic_tokens = true`
- `dynamic_phase_harmonics = 3`
- `dynamic_phase_rank = 12`
- `use_wake_centered_hyper_geometry = true`
- `perceiver_num_layers_local = 2`
- `perceiver_refine_topk_env = 24`
- `dynamic_energy_weight = 0.02`
- `randomize_cylinder_order = true`
- `point_sampling_mode = "wake_focused"`
- `use_phase_window_batches = true`
- `best_metric_name = "val_residual_focus"`
- `residual_focus_field_weight = 0.25`

## Workflow

Run commands below from `0_Demo_MultiCylinder/`.

### 1. Simulate raw inert cases

```bash
python src/simulate_multicylinder_phiflow.py --config-json config_inert.json
```

### 2. Preprocess cases into a packed dataset

```bash
python src/preprocess_inert_multicyl_dataset.py \
  --input-root ./Data_Saved \
  --output-root ./Data_Saved/Processed_Inert_Dataset \
  --device cuda:0
```

### 3. Train the surrogate

```bash
python src/train.py --config train_config_template.json --device cuda:0
```

### 4. Evaluate a trained checkpoint

```bash
python src/evaluate.py --case-id 0002 --dataset-case-id 0161 --dataset-split test
```

### 5. Evaluate the generative model

Stage-2 generative evaluation has two modes. Snapshot mode preserves the old
single-phase workflow:

```bash
python src/evaluate_gen.py --stage 2 --case-id gen001 --split test --phase-index 0 --n-samples 4
```

Cycle mode evaluates one case across selected canonical phase bins and writes a
compressed `cycle_reconstruction.npz`, `cycle_metrics.json`,
`per_phase_metrics.csv`, `cycle_omega.gif`, `cycle_omega_gt_generated.gif`,
and `cycle_montage_omega.png`:

```bash
python src/evaluate_gen.py --stage 2 --case-id gen001 --split test --cycle --n-samples 4
```

The stage-2 rectified-flow model is trained on phase snapshots, so cycle mode
generates one tau-conditioned stochastic field per phase. `--cycle-noise-mode`
controls the initial latent coupling across phases:

- `independent`: fresh latent noise for every phase
- `shared`: one latent noise field reused for all phases in a sample
- `harmonic`: sinusoidal latent interpolation over tau for smoother samples

For memory control, use `--phase-chunk-size` and `--sample-chunk-size`.

## Directory Layout

```text
0_Demo_MultiCylinder/
├── Config_Train/
├── MODEL_EXPLAIN.txt
├── README_multicyl_demo.md
└── src/
```
