# Demo 1: Channel Thermal Modular Design

This demo starts the project direction of **generative modular design by
hypergraph organized neural fields**. It creates raw simulation data and packed
HDF5 datasets for a 2-D channel with heated circular solid modules.

No training scripts are included yet. The goal is to prepare the global and
local data contracts that later forward models can learn from.

## What Is Different From `0_Demo_MultiCylinder`

`0_Demo_MultiCylinder` is a periodic wake benchmark. This demo is nonperiodic:

- left boundary: velocity/temperature inlet
- right boundary: outlet-style zero-gradient temperature and pressure reference
- top/bottom boundaries: no-slip, isothermal walls
- circular modules: larger solid bodies, not just wake obstacles
- module interiors: have internal heat generation
- learning targets: both the global environment field and module-internal
  temperature fields

The code follows the earlier demo's conventions where useful: nested dataclass
configs, materialized layout backup, `case_config.json`, `frame_index.csv`,
timestamped case directories, and packed `packed_dataset.h5` outputs.

## Global Channel Thermal Problem

The global simulator is in `src/simulate_channelthermal.py`.

For this first version, the flow is a stable NumPy channel approximation rather
than a full CFD solve. It builds a steady laminar channel field with obstacle
wake deficits, no-slip module cells, a pressure drop, and vorticity. The thermal
field is evolved explicitly on one shared grid:

- fluid cells: advection plus diffusion
- solid module cells: diffusion plus internal heat generation
- fluid/solid interface: approximate coupling by diffusion across neighboring
  cells in the shared grid

Saved per frame:

- `u`, `v`, `p`, `omega`, `temperature`
- `module_mask`, `module_id`
- `module_internal_temperature`
- `module_internal_mask`
- `interface_response`
- `interface_feature_names`

The interface response includes angle, normal, surface temperature, outside
temperature, estimated normal heat flux, and local normal/tangential velocity.

## Local Module Surrogate Problem

The local simulator is in `src/simulate_local_module_thermal.py`.

It solves one circular solid module in normalized local coordinates with steady
conduction:

```text
-k_s Laplacian(T) = q
```

with Robin boundary functions:

```text
-k_s dT/dn = h(theta) * (T_surface - T_env(theta))
```

`T_env(theta)` and `h(theta)` are sampled from low-frequency Fourier/random
modes. This is intended for Stage-A local surrogate data.

## Why Steady / Quasi-Steady First

Demo 1 saves transient raw frames, but the first packed global dataset averages
the final heat-active window into a steady target. This keeps the first training
task focused on structure-conditioned thermal response rather than phase-cycle
dynamics. Later versions can add true transient states once the global organizer
and local module surrogate are established.

## Data Layout

Raw global cases are written under:

```bash
1_Demo_ChannelThermal/Data_Saved/case_<id>_<timestamp>_<tag>/
```

Key files:

- `case_config.json`
- `frame_index.csv`
- `grid.npz`
- `scene/frame_000000.npz`, ...
- `plots/`

Packed global dataset:

```bash
1_Demo_ChannelThermal/Data_Saved/Processed_ChannelThermal_Dataset/packed_dataset.h5
```

Packed local module dataset:

```bash
1_Demo_ChannelThermal/Data_Saved/Processed_LocalModule_Dataset/packed_dataset.h5
```

## How To Run

From this directory:

```bash
cd 1_Demo_ChannelThermal
```

Run one global case:

```bash
python src/simulate_channelthermal.py --config-json config_channelthermal.json --device cpu
```

Run the global batch launcher:

```bash
python src/launch_dataset_batch.py
```

Preprocess global cases:

```bash
python src/preprocess_channelthermal_dataset.py \
  --input-root ./Data_Saved \
  --output-root ./Data_Saved/Processed_ChannelThermal_Dataset
```

Visualize the latest raw global case:

```bash
python src/visualize_channelthermal_case.py
```

Visualize a processed global case:

```bash
python src/visualize_channelthermal_case.py \
  --processed-h5 ./Data_Saved/Processed_ChannelThermal_Dataset/packed_dataset.h5
```

Run one local module case:

```bash
python src/simulate_local_module_thermal.py --config-json config_local_module.json
```

Run the local module batch launcher:

```bash
python src/launch_local_module_batch.py
```

Preprocess local module cases:

```bash
python src/preprocess_local_module_dataset.py \
  --input-root ./Data_Saved/LocalModule_Raw \
  --output-root ./Data_Saved/Processed_LocalModule_Dataset
```

Visualize the latest raw local module case:

```bash
python src/visualize_local_module_thermal.py
```

Visualize a processed local module case:

```bash
python src/visualize_local_module_thermal.py \
  --processed-h5 ./Data_Saved/Processed_LocalModule_Dataset/packed_dataset.h5
```

## Future Training Plan

Stage A: train a small local module surrogate from port tokens and module
parameters to internal temperature/interface targets.

Stage B: freeze the local surrogate and train a global hypergraph organizer that
maps channel layout, module properties, and environment samples to the global
steady field.

Stage C: jointly fine-tune the local surrogate and global organizer so module
interior predictions and global interface responses improve together.
