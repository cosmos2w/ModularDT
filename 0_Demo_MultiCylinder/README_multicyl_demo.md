# Multi-cylinder PhiFlow Demo

This demo covers the full inert multi-cylinder workflow:

1. simulate wake fields with PhiFlow
2. preprocess raw cases into a packed HDF5 dataset
3. train a PyTorch forward surrogate model
4. evaluate checkpoints by reconstructing full flow fields

The current surrogate-model path is built around the packed dataset written by
[`src/preprocess_inert_multicyl_dataset.py`]
The training and evaluation scripts expect the HDF5 structure produced there,
including per-case `sampled_points`, `mean_field`, `x_grid`, `y_grid`, and
canonical-cycle data for validation and reconstruction.

## Contents

- `src/multicyl_common.py` - shared configuration, path handling, layout sampling, and utility helpers
- `src/simulate_multicylinder_phiflow.py` - raw PhiFlow simulation / data generation
- `src/visualize_multicylinder_case.py` - visualization, GIF export, and QoI extraction for raw cases
- `src/plot_domain_shape.py` - computational-domain preview from a selected config
- `src/preprocess_inert_multicyl_dataset.py` - converts raw cases into processed per-case outputs and `packed_dataset.h5`
- `src/inspect_packed_h5_case.py` - quick inspection utility for the packed HDF5 dataset
- `src/model.py` - hypergraph-organized neural-field surrogate model
- `src/train.py` - training loop with logging, validation, and checkpointing
- `src/evaluate.py` - checkpoint loading and full-grid reconstruction / plotting
- `Configs/` - simulation config files
- `Config_Train/` - training config templates and training-config backups

## Assumptions

- The solver uses periodic boundaries for the computational domain.
- Cylinders have fixed radius; the design varies through cylinder count and locations.
- The current surrogate path targets inert cases only.
- Preprocessing converts time-resolved simulations into a phase-aligned canonical-cycle representation and point samples for neural-field supervision.

## Environment

Install the required Python packages into your environment:

```bash
pip install phiflow torch numpy pandas matplotlib imageio tqdm h5py scipy pillow
```

If your PyTorch install has CUDA support, both preprocessing and training can run on GPU.

## Directory layout

Run the commands below from `0_Demo_MultiCylinder/`.

```text
0_Demo_MultiCylinder/
├── Config_Train/
├── Configs/
├── Data_Saved/
├── Domain_shape/
├── README_multicyl_demo.md
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
    ├── resolved_train_config.json
    └── evaluation/
```

## Data root and symlink

`Data_Saved/` is designed to be the top-level location for simulation outputs and processed datasets. If you want the actual files stored elsewhere, create `Data_Saved` as a symbolic link.

Example:

```bash
cd /home/wanglz/Desktop/src/ModularDT/0_Demo_MultiCylinder
mkdir -p /data/wanglz/ModularDT/0_MultiCylinder
ln -s /data/wanglz/ModularDT/0_MultiCylinder Data_Saved
```

If `Data_Saved` already exists as a regular directory, rename or remove it first.

## Simulation configs

Default simulation configs are stored in `Configs/`:

- `Configs/config_inert.json`
- `Configs/config_active.json`

The simulation script accepts a config filename or path with `--config-json`. Relative config names are resolved from `Configs/`.

Whenever a config file is used in a simulation run, the script copies it into `Configs/Config_bk/` with the case id and timestamp in the filename.

## Device selection

You can select CPU or GPU either in the JSON config or from the command line.

Example config block:

```json
"execution": {
  "device": "cpu",
  "gpu_id": 0
}
```

Example CLI override:

```bash
python src/simulate_multicylinder_phiflow.py \
  --config-json config_inert.json \
  --device gpu \
  --gpu-id 1 \
  --case-id 0001
```

## Workflow

### 1. Simulate raw inert cases

```bash
python src/simulate_multicylinder_phiflow.py --config-json config_inert.json
```

You can also use direct overrides:

```bash
python src/simulate_multicylinder_phiflow.py \
  --case-id 0001 \
  --mode inert \
  --num-cylinders 4 \
  --re 100 \
  --seed 7 \
  --device cpu
```

Each run creates a case directory directly under `Data_Saved/`:

```text
Data_Saved/
└── case_0001_20260417_132015_multicyl/
    ├── case_config.json
    ├── frame_index.csv
    ├── plots/
    └── scene/
```

### 2. Preprocess raw cases into a packed dataset

The preprocessing script converts raw PhiFlow outputs into:

- per-case structure and summary files
- canonical-cycle tensors
- point-sampled training tuples
- one packed HDF5 dataset at `Data_Saved/Processed_Inert_Dataset/packed_dataset.h5`

Example:

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

Notes:

- `--save-full-canonical-cycles` should remain enabled if you want validation in `train.py` and reconstruction plots in `evaluate.py`.
- If `input-root` contains `train/` and `test/` subdirectories, those split labels are preserved in the packed HDF5 metadata.
- The packed HDF5 case groups contain `sampled_points`, `mean_field`, `rms_field`, `x_grid`, `y_grid`, and optionally `canonical_cycle` and `phase_bin_centers`.

You can inspect a packed case with:

```bash
python src/inspect_packed_h5_case.py
```

### 3. Train the surrogate model

Training configuration lives in `Config_Train/train_config_template.json`. The main fields are:

- `dataset.packed_h5_path` - path to the HDF5 dataset
- `dataset.train_split` and `dataset.val_split` - split labels read from HDF5 case metadata
- `dataset.points_per_item` - point-chunk size per dataset item
- `dataset.train_point_fraction` - fraction of points kept from each training chunk for one epoch
- `dataset.min_points_per_sample` - lower bound on retained points per chunk after sub-sampling
- `dataset.resample_each_epoch` - whether to refresh the random point subset every epoch
- `model.*` - neural architecture parameters from [`src/model.py`](/home/wanglz/Desktop/src/ModularDT/0_Demo_MultiCylinder/src/model.py)
- `training.max_physical_queries_per_step` - cap on the number of decoder queries in one physical forward pass
- `training.*` - optimizer, scheduler, mixed precision, and epoch-level training settings
- `validation.*` - canonical-cycle validation settings

Start training with:

```bash
python src/train.py --config train_config_template.json
```

Or point to a custom config path:

```bash
python src/train.py --config /absolute/path/to/train_config.json --device cuda:0
```

Training behavior:

- `train.py` reads chunked point samples from `sampled_points`
- each training chunk can be randomly sub-sampled, which is useful for neural-field training when full chunk density is unnecessary
- `mean_field` is sampled on the same grid locations to form `mean_targets`
- the model predicts `pred_mean`, `pred_residual`, `pred_field`, and a dominant frequency
- validation reconstructs selected canonical-cycle phases on full grids
- loss history, loss curves, and checkpoints are written once per epoch
- checkpoints are saved to `Saved_Model/Case{case_id}_{timestamp}/`
- the resolved config is copied to both the run directory and `Config_Train/Configs_bk/`

The current training script expects:

- `cases/<case_id>/sampled_points/{x,y,tau,u,v,p,omega}`
- `cases/<case_id>/mean_field`
- `cases/<case_id>/x_grid`
- `cases/<case_id>/y_grid`
- validation cases to also contain `canonical_cycle` and `phase_bin_centers`

### 4. Evaluate a checkpoint and rebuild flow fields

`evaluate.py` finds the most recent run directory matching `Saved_Model/Case{case_id}_*` and loads `best_model.pt` by default.

Example:

```bash
python src/evaluate.py \
  --case-id 0001 \
  --dataset-case-id 0001 \
  --dataset-split test \
  --tau 0.25 \
  --device cuda:0
```

Useful options:

- `--latest` loads `latest_model.pt` instead of `best_model.pt`
- `--dataset-case-id` selects which packed-dataset case to reconstruct
- `--dataset-split` chooses the search split if `--dataset-case-id` is omitted
- `--query-batch-size` controls decoder chunk size during full-grid reconstruction
- `--output-dir` overrides the default output directory under the run folder

Evaluation outputs:

- a quicklook PNG comparing ground truth, prediction, and error for `u`, `v`, `p`, and `omega`
- a GIF over the canonical cycle
- a compressed `.npz` with reconstructed arrays
- a JSON summary with the selected run, case id, phase index, MSE, and frequency comparison

## Visualization and domain preview

Visualize a raw case:

```bash
python src/visualize_multicylinder_case.py case_0001_YYYYMMDD_HHMMSS_multicyl
```

Preview the computational domain from a config:

```bash
python src/plot_domain_shape.py --config-json config_inert.json
python src/plot_domain_shape.py --config-json config_active.json
```

The domain preview is saved into `Domain_shape/`.

## Notes

- Relative simulation config names are resolved from `Configs/`.
- Relative training config names are resolved from `Config_Train/`.
- The training and evaluation scripts are now anchored to the demo directory, so they work when launched from either `0_Demo_MultiCylinder/` or the repo root.
- If `packed_dataset.h5` was generated without canonical cycles, training can still use point samples, but canonical validation and `evaluate.py` reconstruction comparisons will not work.
- GPU execution requires a CUDA-enabled PyTorch install and a working NVIDIA driver.

## Recommended next steps

1. Preview the domain with `plot_domain_shape.py`.
2. Simulate a small inert batch and inspect one raw case.
3. Preprocess into `packed_dataset.h5` and confirm the train/test split labels.
4. Train with a short run first to validate loss curves and checkpoint writing.
5. Use `evaluate.py` on a held-out packed-dataset case to confirm reconstruction quality.
