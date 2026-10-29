# WindFarm physical and data contract

This document records the observed schema of the external wind-farm upload.
It is a preprocessing and visualisation contract only.  It does not define a
model input, a training objective, or a checkpoint format.

## 1. Physical scope and coordinate frame

The data are steady CFD simulations for a family of wind-farm layouts.  The
compact package declares:

| Quantity | Value |
|---|---:|
| Reference inflow speed `U_ref` | 9 m/s |
| Rotor diameter `D_m` | 80 m |
| Hub height `hub_height_m` | 70 m (`0.875D`) |
| Wind-direction categories | 270°, 285°, 300° |
| Compact x extent | -20D to 30D (`-1600` to `2400` m) |
| Compact y extent | -19D to 19D (`-1520` to `1520` m) |
| Compact z extent | 0D to 6.25D (`0` to `500` m) |

The supplied plotting helpers label compact `x` as downstream and `y` as
crosswind.  `z` is vertical.  Turbine centres are stored per case in the
wind-farm frame and can differ across the three direction rows of one layout;
do not reuse one row's coordinates for its other directions.  Treat the
three `wd_deg` values as categorical operating conditions unless a separate
source document establishes a meteorological angle convention.

The compact regular grid has spacing `0.125D` in all three axes:

```text
x_D: 401 points, -20.0 ... 30.0
y_D: 305 points, -19.0 ... 19.0
z_D:  51 points,   0.0 ... 6.25
```

## 2. Logical resources

The committed `dataset_manifest.json` defines two IDs:

| ID | Storage | Intended use |
|---|---|---|
| `wind_farm_tensor_v1` | `family_tensor.npz` | Fast metadata, masked raster inspection, and scalar/category analysis. |
| `wind_farm_volume_v1` | `family_volume/` | Per-run cell-centred 3-D fields with exact ragged coordinates. |

The compact bundle is 954,965,970 bytes and has SHA-256
`8c29cfad0d0e241c94aa54d4b6b02fdc0e9f7f2d1c5aa2d8f16a1309c0d2fc4c` in the
observed upload.  The full-volume directory contains approximately 59.6 GB of
logical NumPy payloads, so its SHA-256 is intentionally not computed in the
manifest.  Verify its individual files only when a copy or replacement is
made.

## 3. Compact tensor bundle

`family_tensor.npz` contains 600 rows, ordered in blocks of three:

```text
gen_0000_wd270, gen_0000_wd285, gen_0000_wd300,
gen_0001_wd270, gen_0001_wd285, gen_0001_wd300,
...
```

There are 200 unique layouts (`layout_index` 0–199), exactly three rows per
layout.  The full field arrays are padded to common raster shapes; use their
validity masks for every field statistic or visual loss.

| Array | Shape | Meaning |
|---|---:|---|
| `U_hub` | `(600, 305, 401, 2)` | Hub-height `[Ux, Uy]` in m/s, ordered `[case, y, x, component]`. |
| `valid_hub` | `(600, 305, 401)` | Simulated-cell mask for `U_hub`; 0 denotes padded space. |
| `rotor_hub` | `(600, 305, 401)` | Rasterized rotor/actuator-disk mask. |
| `U_vert` | `(600, 2, 51, 401, 2)` | `[Ux, Uz]` on two vertical x–z planes, ordered `[case, plane, z, x, component]`. |
| `valid_vert` | `(600, 2, 51, 401)` | Simulated-cell mask for `U_vert`. |
| `x_D`, `y_D`, `z_D` | `(401,)`, `(305,)`, `(51,)` | Regular coordinates in rotor diameters. |
| `turbine_xy_D` | `(600, 30, 2)` | Padded turbine centres; first `n_turbines` rows are active and other rows are NaN. |
| `n_turbines` | `(600,)` | Active turbine count, 6–30. |
| `wd_deg` | `(600,)` | Direction category, 270/285/300 degrees. |
| `wake_loss_pct` | `(600,)` | Scalar wake-loss target, observed range 2.285–37.898%. |
| `case`, `layout` | `(600,)` | Readable row and layout identifiers. |
| `layout_index` | `(600,)` | The only safe grouping key for a layout split. |
| `descriptors` | `(600, 6)` | Columns named by `descriptor_names`. |
| `sampled_params` | `(600, 4)` | Generator columns named by `sampled_param_names`. |
| `domain_inside_window` | `(600,)` | Source quality flag; all observed rows are 1. |
| `gap_filled_nodes` | `(600,)` | Count of grid nodes filled during source raster preparation. |
| `vert_plane_y_m` | `(2,)` | Vertical-plane locations, observed as 0 m and 400 m. |

The remaining scalar metadata are `calib_ratio` (observed approximately
0.769758), `U_ref`, `D_m`, and `hub_height_m`.  Array dtypes, required keys,
and exact shapes are machine-readable in the manifest.

### Descriptor columns

The source names the six layout descriptors as:

```text
[N, min_sep_D, mean_nn_D, density, anisotropy, nn_dispersion]
```

`N` is the active turbine count.  The other columns are source-defined layout
statistics in a rotor-diameter-scaled geometry; retain their names and order
without inferring a formula from the column name alone.

The four generator metadata columns are:

```text
[req_min_sep_D, sep_capped, slack, clump]
```

`sep_capped` is a 0/1 flag in the observed data.  The other three values are
source generator parameters and should be treated as metadata, not simulated
flow fields.

## 4. Full-volume ragged fields

`family_volume/` follows the same 600-run order as the compact bundle.  Meshes
have different horizontal dimensions, so cell data are concatenated along one
dimension and addressed by offsets.  The observed export has 2,484,440,512
cells in total and 583 distinct `(nx, ny, nz)` shapes; `nz=64` for all observed
runs.

| Array | Shape | Units and meaning |
|---|---:|---|
| `U.npy` | `(2484440512, 3)` | Cell-centred `[Ux, Uy, Uz]`, m/s. |
| `p.npy` | `(2484440512,)` | Kinematic pressure, m²/s². |
| `k.npy` | `(2484440512,)` | Turbulent kinetic energy, m²/s². |
| `epsilon.npy` | `(2484440512,)` | Turbulent dissipation rate, m²/s³. |
| `run_shape.npy` | `(600, 3)` | `(nx, ny, nz)` per run. |
| `run_cell_offsets.npy` | `(601,)` | Flat field start/end offsets. |
| `run_x_offsets`, `run_y_offsets`, `run_z_offsets` | `(601,)` each | Coordinate-axis start/end offsets. |
| `x_cell_m`, `y_cell_m`, `z_cell_m` | ragged | Exact per-run cell-centre axes in metres. |
| `case.npy` | `(600,)` | Readable run names. |
| `layout_index.npy` | `(600,)` | Layout grouping key. |
| `wd_deg.npy` | `(600,)` | Direction category. |
| `source_time.npy` | `(600,)` | Selected OpenFOAM final-time directory; observed range 529–896. |
| `completed.npy` | `(600,)` | Build-integrity marker; every observed value is 1. |

For run `i`, let `(nx, ny, nz) = run_shape[i]`, and let `[a,b)` be
`run_cell_offsets[i:i+2]`.  The safe reconstruction is:

```python
U_i = U[a:b].reshape(nz, ny, nx, 3)
p_i = p[a:b].reshape(nz, ny, nx)
k_i = k[a:b].reshape(nz, ny, nx)
epsilon_i = epsilon[a:b].reshape(nz, ny, nx)
```

The x coordinate varies fastest in this C-order layout.  The corresponding
axis slices come from their own `run_*_offsets` arrays; they must not be
reconstructed from global minima or from another run.

The observed cell-centre extents are approximately `x=-1594...2394 m`,
`y=-1506...1468 m`, and `z=1.8...492.8 m`.  These are cell-centre extents,
not necessarily exact boundary locations.

## 5. Case categories and coverage

### Wind direction

Each of the three directions has 200 rows:

| `wd_deg` | Rows | Layout groups |
|---:|---:|---:|
| 270 | 200 | 200 |
| 285 | 200 | 200 |
| 300 | 200 | 200 |

The wake-loss target has the following observed summary:

| Direction | Minimum % | Mean % | Maximum % |
|---:|---:|---:|---:|
| 270 | 2.285 | 19.326 | 37.898 |
| 285 | 3.939 | 18.352 | 37.287 |
| 300 | 3.323 | 17.357 | 36.425 |

### Layout size

The 200 layouts span 6–30 active turbines.  Counts below are **layouts**;
each contributes three compact rows and three full-volume runs:

| `N` | layouts | `N` | layouts | `N` | layouts | `N` | layouts | `N` | layouts |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 3 | 7 | 10 | 8 | 8 | 9 | 9 | 10 | 7 |
| 11 | 10 | 12 | 6 | 13 | 9 | 14 | 7 | 15 | 11 |
| 16 | 8 | 17 | 9 | 18 | 7 | 19 | 9 | 20 | 9 |
| 21 | 7 | 22 | 8 | 23 | 9 | 24 | 8 | 25 | 8 |
| 26 | 8 | 27 | 9 | 28 | 9 | 29 | 8 | 30 | 4 |

Other useful observed ranges are:

```text
min_sep_D       2.507 ... 7.187
mean_nn_D       2.513 ... 7.360
density         0.00849 ... 0.14138
anisotropy      1.051 ... 2.950
nn_dispersion   0.00000 ... 0.47518
clump           0.00068 ... 0.99583
```

These are coverage descriptors, not guarantees of a uniform design of
experiments.  Summaries should report the actual group counts and direction
categories alongside any aggregate statistic.

## 6. Integrity, masks, and split rules

Preprocessing should fail early when any of the following is violated:

1. The compact bundle is missing a manifest-required key or has a changed
   shape/dtype.
2. `layout_index` is not 0–199 with three rows per layout, or the three
   directions are not 270/285/300.
3. `domain_inside_window` contains a zero when the analysis assumes the
   complete source window.
4. A compact field statistic includes `valid_hub==0` or `valid_vert==0`
   padding, or treats `rotor_hub` as a validity mask.
5. `completed.npy` contains a value other than 1.
6. Any volume offset array has length other than 601, is non-monotone, or
   disagrees with its corresponding array length.
7. A volume run violates
   `run_cell_offsets[i+1] - run_cell_offsets[i] == nx * ny * nz`.
8. `case`, `layout_index`, or `wd_deg` order disagrees between compact and
   volume metadata.

There is no source train/validation/test split.  A later split must use
`layout_index` as the grouping key and keep all three wind directions for a
layout in one partition.  This is required even for exploratory baselines;
row-wise random splitting would expose the same geometry in both partitions.

## 7. Visualisation contract

The 3-D domain is much wider in x/y than in z.  Use physical or D-scaled
equal-aspect coordinates and show cut planes rather than relying on a single
opaque volume rendering.

### Canonical cuts

For each representative case, render:

1. **Hub-height horizontal cut:** `z=70 m` (`z_D=0.875`), showing `|U|` or
   `Ux` with turbine centres overlaid from the active rows of
   `turbine_xy_D`.
2. **Streamwise vertical cut:** `y=0 m`, showing `x–z` and wake recovery.
3. **Crosswind vertical cut:** `x=0 m`, showing `y–z` and vertical structure.
4. **Optional second vertical plane:** `y=400 m`, which is the second plane
   present in `U_vert` and `vert_plane_y_m`.

For fields from `family_volume`, choose the nearest cell-centre plane to these
locations and state the actual selected coordinate in the figure metadata.
For `U_vert`, use the stored plane index rather than interpolating between
planes.  Always apply the corresponding validity mask for compact rasters.

### View angle and colour policy

The source helper's reproducible oblique view is approximately:

```text
elevation = 25 degrees
azimuth   = -55 degrees
```

Small changes are acceptable when they improve turbine/wake visibility, but
the selected angles and slice coordinates belong in the showcase caption or
metadata.  Use a shared colour scale when comparing cases or wind directions;
for signed components use a zero-centred scale, and for speed use a robust
percentile range.  Do not let padded zeros determine colour limits.

The maintained first showcase uses three data-driven rows spanning target
percentiles, layout sizes, and directions:

| Row | Index | Layout category | Direction | Wake-loss context |
|---|---:|---:|---:|---|
| `gen_0055_wd300` | 167 | `N=8` | 300° | Approximately the 10th percentile (`9.607%`). |
| `gen_0044_wd285` | 133 | `N=17` | 285° | Median target (`18.396%`). |
| `gen_0178_wd270` | 534 | `N=22` | 270° | Approximately the 90th percentile (`27.272%`). |

These rows are deliberately not the minimum/maximum turbine-count cases: the
selection gives a more representative low/median/high-response comparison
while varying wind direction.  Additional cases may be added for a dedicated
layout-count study.

## 8. Provenance and versioning

`dataset_manifest.json` is the source of truth for the observed upload.  A
change to array order, coordinate convention, mask meaning, target definition,
or required keys requires a new schema name and logical dataset ID.  Derived
tables and figures should record the manifest version, source logical ID, case
row/index, selected cut coordinates, and preprocessing code revision.

The raw data, local symlink, local location map, generated arrays, figures,
and any future checkpoints remain outside Git.  This case currently has no
model artifacts; see `../artifacts/README.md`.
