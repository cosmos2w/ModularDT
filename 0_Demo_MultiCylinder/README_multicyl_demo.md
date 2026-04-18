# Multi-cylinder PhiFlow Demo

This demo simulates 2-D periodic multi-cylinder wake cases with PhiFlow on the PyTorch backend, saves field data for later analysis, and provides utilities for domain preview and post-processing.

## Contents

- `src/multicyl_common.py` — shared configuration, path handling, runtime estimation, layout sampling, and utility helpers.
- `src/simulate_multicylinder_phiflow.py` — simulation / data-generation script.
- `src/visualize_multicylinder_case.py` — visualization, GIF export, and QoI extraction.
- `src/plot_domain_shape.py` — vivid computational-domain preview from a selected config.
- `Configs/` — default JSON config files.
- `Config_bk/` — archived copies of configs used in simulation runs.
- `Data_Saved/` — data root for generated cases. This is intended to be a symbolic link.
- `Domain_shape/` — saved PNG previews of the configured domain.

## Assumptions

- The solver uses periodic boundaries for the computational domain.
- Cylinders have fixed radius; the design varies through cylinder count, locations, and in active mode, per-cylinder heat-source strengths.
- Active mode is one-way coupled: the flow drives temperature advection and diffusion, but temperature does not feed back into the velocity field.
- The thermal source is modeled as a Gaussian shell around each cylinder for a lightweight initial workflow.

## Environment

Install the required Python packages into your environment:

```bash
pip install phiflow torch numpy pandas matplotlib imageio tqdm
```

If your machine has a CUDA-enabled PyTorch install, importing `phi.torch.flow` uses the PyTorch backend on GPU when requested.

## Directory layout

The current demo layout is:

```text
0_Demo_MultiCylinder/
├── Config_bk/
├── Configs/
├── Data_Saved/
├── Domain_shape/
├── README_multicyl_demo.md
└── src/
```

### Data root and symlink

`Data_Saved/` is designed to be the single top-level location for simulation outputs:

```text
Data_Saved/
├── case_0001_YYYYMMDD_HHMMSS_multicyl/
├── case_0002_YYYYMMDD_HHMMSS_multicyl/
└── ...
```

It should not create `Data_Saved/Data_Saved/...`.

If you want the actual files stored elsewhere, create `Data_Saved` as a symbolic link. Example:

```bash
cd /home/wanglz/Desktop/src/ModularDT/0_Demo_MultiCylinder
mkdir -p /data/wanglz/ModularDT/0_MultiCylinder
ln -s /data/wanglz/ModularDT/0_MultiCylinder Data_Saved
```

If `Data_Saved` already exists as a regular directory, rename or remove it first.

## Config files

Default config files are stored in `Configs/`:

- `Configs/config_inert.json`
- `Configs/config_active.json`

The simulation script accepts a config filename or path with `--config-json`. Relative config names are resolved from `Configs/`.

Whenever a config file is used in a simulation run, the script copies it into `Config_bk/` and renames it to include the case id and timestamp, for example:

```text
Config_bk/config_inert_case_0001_20260417_141935.json
```

## Device selection

You can select CPU or GPU either in the JSON config or directly from the command line.

### In the config file

```json
"execution": {
  "device": "cpu",
  "gpu_id": 0
}
```

Valid values:

- `device`: `cpu` or `gpu`
- `gpu_id`: GPU index used when `device` is `gpu`

### From the command line

```bash
python src/simulate_multicylinder_phiflow.py \
  --config-json config_inert.json \
  --device gpu \
  --gpu-id 1
  --case-id 0001
```

CLI values override the config file.

## Simulation usage

Run from `0_Demo_MultiCylinder/`.

### 1) Inert case from config

```bash
python src/simulate_multicylinder_phiflow.py \
  --config-json config_inert.json
```

### 2) Active case from config

```bash
python src/simulate_multicylinder_phiflow.py \
  --config-json config_active.json
```

### 3) Inert case with direct CLI overrides

```bash
python src/simulate_multicylinder_phiflow.py \
  --case-id 0001 \
  --mode inert \
  --num-cylinders 4 \
  --re 100 \
  --seed 7 \
  --device cpu
```

### 4) Active case on a selected GPU

```bash
python src/simulate_multicylinder_phiflow.py \
  --config-json config_active.json \
  --case-id 0002 \
  --device gpu \
  --gpu-id 0
```

## Simulation reporting

The simulation script now actively reports runtime milestones:

- when the run starts
- which config file was loaded
- where the config backup was saved
- the prepared configuration summary
- the selected device
- the created case directory
- the runtime summary
- every saved frame
- final completion with output location

It also shows `tqdm` progress bars for:

- total simulation steps
- total saved frames

## Output layout

Each simulation run creates a case directory directly under `Data_Saved/`:

```text
Data_Saved/
└── case_0001_20260417_132015_multicyl/
    ├── case_config.json
    ├── frame_index.csv
    ├── plots/
    └── scene/
        ├── Velocity_000000.npz
        ├── Pressure_000000.npz
        ├── Vorticity_000000.npz
        └── ...
```

## Visualization usage

You can pass either a case directory name or a path rooted at `Data_Saved/`.

### Plot selected frames

```bash
python src/visualize_multicylinder_case.py case_0001_YYYYMMDD_HHMMSS_multicyl
```

### Save a GIF and full QoI time series

```bash
python src/visualize_multicylinder_case.py \
  case_0001_YYYYMMDD_HHMMSS_multicyl \
  --save-gif \
  --gif-field vorticity \
  --gif-dpi 90 \
  --fps 10
```

Accepted path forms include:

- `case_0001_YYYYMMDD_HHMMSS_multicyl`
- `Data_Saved/case_0001_YYYYMMDD_HHMMSS_multicyl`
- an absolute path to the case directory

## Domain preview usage

Use the domain-plotting utility to preview the computational setup from a config file:

```bash
python src/plot_domain_shape.py --config-json config_inert.json
python src/plot_domain_shape.py --config-json config_active.json
```

The resulting PNG is saved into `Domain_shape/` with a timestamped name such as:

```text
Domain_shape/domain_config_inert_case_0001_20260417_135240.png
```

## Notes

- Relative config names are resolved from `Configs/`.
- Relative data case names are resolved from `Data_Saved/`.
- If a config uses `"root_dir": "./Data_Saved"`, outputs still go directly into `Data_Saved/case_...`, not `Data_Saved/Data_Saved/case_...`.
- GPU execution requires a CUDA-enabled PyTorch install and a working NVIDIA driver.

## Recommended next steps

1. Run `plot_domain_shape.py` first to confirm the geometry and layout in the selected config.
2. Run one inert case and confirm the output lands directly in `Data_Saved/case_...`.
3. Visualize the saved fields and check that the cadence captures multiple wake structures cleanly.
