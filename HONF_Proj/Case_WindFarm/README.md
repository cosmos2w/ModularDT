# WindFarm preprocessing case

`Case_WindFarm` is the preprocessing and exploratory-visualisation workspace
for the uploaded wind-farm CFD data.  This round deliberately stops before a
HONF model, training workflow, or checkpoint contract is introduced.  The
case package is responsible for making the external arrays inspectable,
recording leakage-safe layout groups, and producing reproducible data-quality
summaries and representative field views.

## Data hand-off

The source directory is external to Git and must not be copied into this
workspace.  Keep one local browsing link at:

```text
Dataset/links/wind_farm -> <data-owner wind_farm directory>
```

The link is intentionally local and is not a repository asset.  The uploaded
directory currently contains the canonical `family_tensor.npz` bundle and the
ragged `family_volume/` directory, together with source plotting helpers and
reference images.  Only the two data resources are described by the committed
manifest; scripts and images remain external reference material.

Copy `Dataset/dataset_locations.example.json` to the ignored
`Dataset/dataset_locations.local.json` when a tool needs an explicit location
map.  The example uses `project://` paths through the local link so that no
machine-specific path is committed.

## Workspace layout

```text
Case_WindFarm/
├── README.md
├── pyproject.toml
├── configs/
│   ├── case_default.json
│   └── case_config.schema.json
├── Dataset/
│   ├── README.md
│   ├── PHYSICS_AND_DATA.md
│   ├── dataset_manifest.json
│   ├── dataset_locations.example.json
│   └── links/                 local symlink, ignored
├── artifacts/README.md        no model artifacts in this round
└── src/windfarm/              preprocessing-only readers/plots
```

Install the small case package from this directory with:

```bash
python -m pip install -e .
```

The source package uses read-only memory maps for the full-volume arrays.  Do
not load the complete ragged `U.npy` array into memory merely to inspect one
run; use the per-run offsets and exact coordinate axes documented in
`Dataset/PHYSICS_AND_DATA.md`.

## Run the preprocessing pass

From `Case_WindFarm`, validate the ragged arrays, write a compact case table,
sample field statistics with bounded memory, and create a deterministic
70/15/15 layout-group split:

```bash
python scripts/inspect_windfarm.py \
  --dataset-root Dataset/links/wind_farm \
  --output-dir diagnostics/generated \
  --sample-points-per-run 512
```

Render the maintained low/median/high wake-loss showcase with one shared speed
scale, turbine overlays, hub-height and vertical cuts, and the oblique 3-D
view:

```bash
python scripts/visualize_windfarm.py \
  --dataset-root Dataset/links/wind_farm \
  --output-dir diagnostics/generated \
  --cases 167,133,534 \
  --field speed \
  --vmin 5 --vmax 12
```

The generated JSON, CSV, split indices, manifest, and PNGs are ignored by Git.
The evidence-backed data inventory and category analysis are in
[`docs/WIND_FARM_DATASET_REPORT.md`](docs/WIND_FARM_DATASET_REPORT.md).

## Analysis boundary

The compact bundle has 600 rows but only 200 independent layouts: each layout
is evaluated at wind directions 270, 285, and 300 degrees.  Any future split
or summary comparing layouts must group by `layout_index`; row-wise random
splits leak geometry across directions.  Invalid padded cells must be excluded
with `valid_hub` or `valid_vert` masks.  The full-volume export is ragged and
uses cell offsets, so its `run_shape` and offset invariants must be checked
before reshaping.

Recommended cut planes and camera settings are recorded in the preprocessing
profile and the data contract.  The representative showcase should include a
hub-height horizontal plane plus the two orthogonal vertical sections through
the farm centre; a single oblique 3-D view is not sufficient for this highly
anisotropic domain.

See `Dataset/PHYSICS_AND_DATA.md` for the array schemas, physical units,
case categories, integrity checks, and visualisation conventions.  Generated
tables, figures, derived arrays, and the local path map are not committed.
