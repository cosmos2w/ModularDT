# Multi-cylinder PhiFlow Demo

This demo covers the inert multi-cylinder wake workflow:

1. simulate wake fields with PhiFlow
2. preprocess raw cases into a packed HDF5 dataset
3. train a hypergraph-organized neural field surrogate
4. reconstruct full flow fields from saved checkpoints

The current surrogate path is built around the packed dataset written by
`src/preprocess_inert_multicyl_dataset.py`. The training and evaluation scripts
still use the same packed-HDF5 / point-chunk pipeline; this revision focuses on
model architecture and training objectives, not on rewriting the dataset path.

## What Changed

This revision replaces the older multi-decoder setup with one clean primary
architecture aimed at better dynamic wake reconstruction.

- The organizer still produces module, environment, and hyperedge states, but it
  now also returns `hyper_coords` and `hyper_strength` so the decoder can treat
  hyperedges as explicit local dynamic memory.
- The old pooled dynamic summary is no longer the main residual carrier. The
  behavior path now emits:
  `behavior_latent`, `mean_latent`, `dynamic_global_token`,
  `dynamic_hyper_tokens`, `freq_pred`, and optional `hyper_phase_offsets`.
- The decoder is now split into:
  a light global mean branch for smooth structure and
  a hierarchical residual branch for localized, phase-sensitive wake dynamics.
- `(x, y)` and `tau` are encoded separately.
  Spatial queries use Fourier features; phase uses harmonic `sin/cos` features.
- Default losses now emphasize `field + residual`, while `mean_mse_weight`
  defaults to `0.0`.
- Validation now logs `val_residual_mse` as a first-class metric and checkpoint
  selection defaults to `validation.best_metric_name = "val_residual_focus"`,
  which means:
  `val_residual_mse + 0.25 * val_field_mse`.

## Model Overview

The forward model is now:

`structure -> organizer -> organized memory -> behavior / dynamic token generation -> mean branch + residual branch -> pred_field`

### Organizer Outputs

`src/model.py` keeps the organizer as the structure encoder. It returns:

- `module_state`
- `env_state`
- `hyper_state`
- `A_me`, `A_mh`, `A_eh`
- `module_coords_norm`
- `env_coords`
- `hyper_coords`
- `hyper_strength`
- `global_features`
- `cyl_mask`

`hyper_coords` are derived from `A_mh`-weighted module coordinates. This gives
the residual decoder a stable hyperedge location without concatenating raw
geometry into the output heads.

### Dynamic Memory

The behavior head now keeps one smooth summary plus structured dynamic memory:

- `behavior_latent`: compact global summary
- `mean_latent`: smooth/global context for the mean branch
- `dynamic_global_token`: one global residual-memory token
- `dynamic_hyper_tokens`: one dynamic token per hyperedge
- `freq_pred`: dominant-frequency prediction
- `hyper_phase_offsets` (optional): lightweight per-hyperedge phase offsets

The key design change is that residual decoding reads `dynamic_hyper_tokens`
directly instead of depending mostly on one monolithic pooled latent.

### Decoder Structure

The decoder is a single hierarchical Perceiver-style path.

Mean branch:

- query init uses spatial encoding plus smooth/global context
- one shallow global read over structured memory
- predicts only the smooth mean field

Residual branch:

- query init uses spatial encoding, phase encoding, global dynamic context, and
  a phase-aware summary of dynamic hyper tokens
- Stage 1: global read over module / env / hyper / dynamic-hyper / global tokens
- Stage 2: local refinement over gathered top-k nearby env tokens, top-k nearby
  module tokens, all hyper tokens, all dynamic-hyper tokens, and the dynamic
  global token

Relative geometry is used only as attention bias for query-to-module and
query-to-environment reads. Raw query coordinates and raw relative geometry are
not re-injected into the final output heads.

## Training Defaults

The default config is in `Config_Train/train_config_template.json`.

Recommended starting model settings:

- `hidden_dim = 64`
- `behavior_dim = 64`
- `latent_dim = 64`
- `dynamic_token_dim = 64`
- `num_hyperedges = 4`
- `num_env_tokens_x = 24`
- `num_env_tokens_y = 8`
- `message_passing_steps = 3`
- `structure_fourier_frequencies = 1`
- `spatial_query_fourier_frequencies = 3`
- `phase_harmonics = 2`
- `decoder_hidden_dim = 128`
- `perceiver_num_layers_global = 1`
- `perceiver_num_layers_local = 1`
- `perceiver_num_heads = 4`
- `perceiver_head_dim = 16`
- `perceiver_ffn_mult = 2`
- `perceiver_dropout = 0.05`
- `perceiver_refine_topk_env = 16`
- `perceiver_refine_topk_mod = 4`
- `perceiver_query_chunk_size = 1024`
- `use_dynamic_hyper_tokens = true`
- `use_hyper_phase_offsets = true`

Recommended starting loss weights:

- `field_mse_weight = 1.0`
- `mean_mse_weight = 0.0`
- `residual_mse_weight = 1.25`
- `freq_mse_weight = 0.05`
- `organizer_sparsity_weight = 0.0`
- `organizer_entropy_weight = 0.0`
- `organizer_me_weight = 0.01`
- `organizer_mm_weight = 0.05`
- `organizer_consistency_weight = 0.05`

The point of this revision is to improve representation and decoding, not to add
more smoothing losses.

## Validation And Checkpoint Selection

Validation supports both point-chunk evaluation and canonical full-grid
evaluation. Both now log:

- `val_total_loss`
- `val_field_mse`
- `val_mean_mse`
- `val_residual_mse`
- `val_freq_mse`

Best-checkpoint selection is configurable through:

- `validation.best_metric_name`

Supported default:

- `val_residual_focus`

which is computed as:

`val_residual_mse + 0.25 * val_field_mse`

This makes checkpoint selection visibly care about dynamic wake quality rather
than selecting only by smooth global fit.

## Memory Tuning Advice

The new decoder is designed to stay moderate in memory use, but a few settings
matter a lot:

- `dataset.points_per_item` and `training.batch_size` control total query count
  per forward pass
- `model.perceiver_query_chunk_size` controls global-attention query chunking
- `model.perceiver_refine_topk_env` and `model.perceiver_refine_topk_mod`
  control local refinement cost
- `validation.query_batch_size` controls full-grid reconstruction chunking

If GPU memory is tight:

1. reduce `dataset.points_per_item`
2. reduce `training.batch_size`
3. keep `perceiver_num_layers_global = 1`
4. keep `perceiver_num_layers_local = 1`
5. reduce `perceiver_refine_topk_env` before increasing model width

## Directory Layout

Run commands below from `0_Demo_MultiCylinder/`.

```text
0_Demo_MultiCylinder/
├── Config_Train/
├── Configs/
├── Data_Saved/
├── README_multicyl_demo.md
├── MODEL_EXPLAIN.txt
└── src/
```

Typical generated data layout:

```text
Data_Saved/
├── case_0001_YYYYMMDD_HHMMSS_multicyl/
├── case_0002_YYYYMMDD_HHMMSS_multicyl/
└── Processed_Inert_Dataset/
    ├── global_case_index.csv
    ├── packed_dataset.h5
    ├── train/
    └── test/
```

Training outputs are written under:

```text
Saved_Model/
└── Case0001_YYYYMMDD_HHMMSS/
    ├── best_model.pt
    ├── latest_model.pt
    ├── loss_history.csv
    ├── loss_curve.png
    └── resolved_train_config.json
```

## Environment

Install the required Python packages:

```bash
pip install phiflow torch numpy pandas matplotlib imageio tqdm h5py scipy pillow
```

If your PyTorch build has CUDA support, preprocessing and training can run on
GPU.

## Workflow

### 1. Simulate raw inert cases

```bash
python src/simulate_multicylinder_phiflow.py --config-json config_inert.json
```

### 2. Preprocess raw cases into a packed dataset

```bash
python src/preprocess_inert_multicyl_dataset.py \
  --input-root ./Data_Saved \
  --output-root ./Data_Saved/Processed_Inert_Dataset \
  --device cuda:0 \
  --phase-bins 24 \
  --save-cycles 2 \
  --points-per-phase-bin 0 \
  --sampling-mode uniform \
  --save-full-canonical-cycles
```

Keep `--save-full-canonical-cycles` enabled if you want canonical-cycle
validation and full-grid evaluation later.

### 3. Train the surrogate

```bash
python src/train.py --config train_config_template.json --device cuda:0
```

### 4. Evaluate / reconstruct full grids

`src/evaluate.py` can load a saved checkpoint and call
`reconstruct_full_grid(...)` on the revised model path.

## Notes

- This round does not try to preserve older decoder families or older config
  schema.
- The packed dataset / point-chunk interface is intentionally unchanged.
- The residual branch is where sharp vortices should now be learned; the mean
  branch is deliberately lightweight and smooth.
