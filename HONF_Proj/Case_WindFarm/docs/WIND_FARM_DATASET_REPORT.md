# Wind-farm dataset audit and preprocessing hand-off

## Scope and result

This is a read-only audit of the uploaded directory exposed through
`Dataset/links/wind_farm` (the audited source resolved to
`/data/data_transfer/forLinzheng/wind_farm`).  It covers the compact
`family_tensor.npz` bundle and the ragged `family_volume/` export.  No raw data,
figures, or model/training artifacts are copied into this repository.

The upload is structurally usable for preprocessing and exploratory
visualisation:

| Resource | Observed contents | Audit result |
|---|---|---|
| `family_tensor.npz` | 26 uncompressed `.npy` members; 600 rows; compact geometry, targets, masks, hub/vertical rasters | All headers, metadata, masks, and field values inspected successfully. |
| `family_volume/` | 17 individual `.npy` arrays; 600 ragged runs; 2,484,440,512 cells | Offset/shape/axis invariants pass; all `completed.npy` markers are 1; a chunked mmap finite-value scan passes. |
| `family_tensor/` | README describes an individual-array directory | **Absent in the source upload.** The only compact resource currently present is `family_tensor.npz`. |

The NPZ is 954,965,970 bytes on disk (954,962,740 bytes of member payload),
and the full-volume `.npy` files occupy 59,628,041,920 bytes (about 59.6 GB,
55.5 GiB).  The large volume must remain external and be accessed by read-only
memory maps and per-run offsets.

## Audit method

The compact bundle was inspected by reading ZIP member headers and loading only
bounded metadata.  The five large compact members were addressed through
read-only mappings of their stored NPZ payloads for mask coverage and field
statistics; they were not copied into a single working array.  The full-volume
arrays were opened with `numpy.load(..., mmap_mode="r")`.  Structural checks
covered array lengths, monotone offsets, shape products, axis lengths, case
ordering, and completion markers.  Full-volume value finiteness was checked in
bounded chunks (8 million scalar rows for scalar fields and 2 million rows for
`U`), not by loading any 30 GB field into RAM.

This report records both checks that cover the whole compact/full-volume
resources and checks that are intentionally samples.  The finite-value scan of
the full-volume fields is whole-array; representative value ranges below are
therefore not merely endpoint samples.

## Compact bundle schema: `family_tensor.npz`

The NPZ members are ZIP-stored (not compressed), but an NPZ archive is not a
portable per-array mmap interface.  Standard `np.load(...)["U_hub"]` materialises
the selected member.  The preprocessing reader therefore treats the compact
bundle as a metadata/raster source and uses explicit bounded access; it does not
assume the missing `family_tensor/` directory exists.

### Field, grid, and geometry arrays

| Array | Shape | Dtype/order | Meaning |
|---|---:|---|---|
| `U_hub` | `(600, 305, 401, 2)` | `float32`, C | Hub-height `(Ux, Uy)` raster, case/y/x/component. |
| `valid_hub` | `(600, 305, 401)` | `uint8`, C | Binary hub-plane validity mask. |
| `rotor_hub` | `(600, 305, 401)` | `uint8`, C | Binary rasterised rotor locations. |
| `U_vert` | `(600, 2, 51, 401, 2)` | `float32`, C | Two vertical `(Ux, Uz)` rasters, case/plane/z/x/component. |
| `valid_vert` | `(600, 2, 51, 401)` | `uint8`, C | Binary vertical validity mask. The two plane masks are identical in this upload. |
| `x_D` | `(401,)` | `float32`, C | Hub/vertical grid x-coordinate in rotor diameters, `-20` to `30`. |
| `y_D` | `(305,)` | `float32`, C | Hub grid y-coordinate in rotor diameters, `-19` to `19`. |
| `z_D` | `(51,)` | `float32`, C | Vertical grid z-coordinate in rotor diameters, `0` to `6.25`. |
| `turbine_xy_D` | `(600, 30, 2)` | `float32`, C | Padded turbine centres `(x_D, y_D)`; first `n_turbines` slots are finite and remaining slots are `NaN`. |

### Case metadata and targets

| Array | Shape | Dtype/order | Meaning |
|---|---:|---|---|
| `n_turbines` | `(600,)` | `int16`, C | Number of real turbines, 6–30. |
| `wd_deg` | `(600,)` | `float32`, C | Wind direction; exactly 270, 285, or 300 degrees. |
| `wake_loss_pct` | `(600,)` | `float32`, C | Scalar wake-loss target in percent. |
| `case` | `(600,)` | `<U20`, C | Case ID, `gen_NNNN_wdDDD`. |
| `layout` | `(600,)` | `<U12`, C | Layout ID, `gen_NNNN`. |
| `layout_index` | `(600,)` | `int16`, C | Leakage-safe physical-layout group ID, 0–199. |
| `descriptors` | `(600, 6)` | `float32`, Fortran | Layout descriptors; columns are named by `descriptor_names`. |
| `descriptor_names` | `(6,)` | `<U14`, C | `N`, `min_sep_D`, `mean_nn_D`, `density`, `anisotropy`, `nn_dispersion`. |
| `sampled_params` | `(600, 4)` | `float32`, Fortran | Generation parameters; columns are named by `sampled_param_names`. |
| `sampled_param_names` | `(4,)` | `<U14`, C | `req_min_sep_D`, `sep_capped`, `slack`, `clump`. |

### Constants and preprocessing markers

| Array | Shape | Dtype | Observed value/meaning |
|---|---:|---|---|
| `calib_ratio` | `()` | `float32` | `0.76975799`. Preserve as provenance; do not silently replace it with a guessed calibration. |
| `U_ref` | `()` | `float32` | `9.0 m/s`. |
| `D_m` | `()` | `float32` | `80.0 m`. |
| `hub_height_m` | `()` | `float32` | `70.0 m`. |
| `domain_inside_window` | `(600,)` | `uint8` | All 600 entries are 1. |
| `gap_filled_nodes` | `(600,)` | `int32` | 546 zeros, 54 positive entries; range 0–888, mean 20.49. Treat as a provenance/quality marker. |
| `vert_plane_y_m` | `(2,)` | `float32` | `[0.0, 400.0] m`, the two vertical-plane locations. |

## Case population and categories

The row order is deterministic and grouped by physical layout:

```text
gen_0000_wd270, gen_0000_wd285, gen_0000_wd300,
gen_0001_wd270, gen_0001_wd285, gen_0001_wd300,
...
gen_0199_wd270, gen_0199_wd285, gen_0199_wd300
```

There are 200 layouts and exactly three rows per `layout_index`. Layout-level
descriptors and turbine count are constant within each group. The stored
`turbine_xy_D` coordinates are direction-aligned and therefore rotate between
the three rows, while simulated fields and targets also vary by direction.
The direction counts are 200 each for 270, 285, and 300 degrees.

### Layout population

`n_turbines` takes every integer value from 6 through 30.  Counts below are per
independent layout (not per direction row):

| Turbine-count bin | Layouts | Direction rows |
|---:|---:|---:|
| 6–10 | 37 | 111 |
| 11–15 | 43 | 129 |
| 16–20 | 42 | 126 |
| 21–25 | 40 | 120 |
| 26–30 | 38 | 114 |
| **Total** | **200** | **600** |

There are 10,800 finite turbine slots and 7,200 padded slots in the compact
array.  Valid turbine coordinates span `x_D = -14.997..14.998` and
`y_D = -14.881..14.405`.  Padded slots are `NaN`; using a zero-filled padded
array would create artificial turbines.

The six descriptor ranges (minimum, median, maximum over the 200 layouts) are:

| Descriptor | Min | Median | Max |
|---|---:|---:|---:|
| `N` | 6 | 18 | 30 |
| `min_sep_D` | 2.5069 | 4.0651 | 7.1871 |
| `mean_nn_D` | 2.5125 | 4.1515 | 7.3602 |
| `density` | 0.00849 | 0.03649 | 0.14138 |
| `anisotropy` | 1.0511 | 1.5220 | 2.9499 |
| `nn_dispersion` | ~0 | 0.01013 | 0.47518 |

The sampled-generation parameters have ranges `req_min_sep_D = 2.5069..7.9651`,
`slack = 1.1057..1.8950`, and `clump = 0.00068..0.99583`.
`sep_capped` is binary: 113 layouts have value 1 and 87 have value 0.

### Target distribution

Across all 600 rows, `wake_loss_pct` is 2.2852–37.8980%, with quartiles
13.5249%, 18.3989%, and 22.9078%.  Direction-conditioned summaries are:

| Wind direction | Min | Median | Mean | Max |
|---:|---:|---:|---:|---:|
| 270° | 2.2852 | 19.1671 | 19.3263 | 37.8980 |
| 285° | 3.9388 | 18.0514 | 18.3515 | 37.2870 |
| 300° | 3.3232 | 17.2696 | 17.3567 | 36.4255 |

The direction effect is not a guaranteed monotone ordering within a layout:
82 groups have `270 >= 285 >= 300`, 17 have the reverse ordering, and 101
groups have a mixed ordering.  Direction must remain an explicit category.

### Compact-grid coverage and value checks

The normalized compact grids are regular at `0.125 D` spacing:

| Grid | Shape | Coordinate range | Validity result |
|---|---:|---|---|
| Hub | `(305, 401)` | `y_D=-19..19`, `x_D=-20..30` | 38,430,440 of 73,383,000 cells valid (52.370%); per case 22,338–107,164, mean 64,050.7. |
| Vertical, each plane | `(51, 401)` | `z_D=0..6.25`, `x_D=-20..30` | 10,181,079 of 12,270,600 cells valid per plane (82.971%); per case 11,067–20,094. |
| Rotor overlay | `(305, 401)` | hub grid | 190,528 positive cells (0.260%); binary, not a continuous area fraction. |

`valid_hub`, `valid_vert`, and `rotor_hub` contain only 0 and 1.  All compact
field values inspected (inside and outside masks) are finite, but outside-mask
values are not a safe substitute for masking: invalid hub `Ux` values are
nonzero in 15,141–99,967 cells per case, and invalid `Uy` values are also
nonzero.  Apply the validity mask to losses and plots, and show it explicitly
when a plot could otherwise imply a simulated domain where none exists.

For valid cells, the observed compact-field ranges/statistics are:

| Field | Min | Max | Mean | Std |
|---|---:|---:|---:|---:|
| Hub `Ux` [m/s] | 4.9800 | 9.7682 | 8.6424 | 0.7159 |
| Hub `Uy` [m/s] | -1.1709 | 1.1487 | -0.0019 | 0.0887 |
| Vertical `Ux` [m/s] | 0.0000 | 12.0500 | 10.2195 | 2.0732 |
| Vertical `Uz` [m/s] | -1.2708 | 0.9365 | 0.0042 | 0.0412 |

## Full-volume schema: `family_volume/`

The volume export contains final cell-centred fields in a concatenated ragged
cell dimension.  Each run is reshaped as `(nz, ny, nx)` or `(nz, ny, nx, 3)`
with x varying fastest, exactly as described by the source README.

| Array | Shape | Dtype | Role |
|---|---:|---|---|
| `U.npy` | `(2484440512, 3)` | `float32` | Cell-centred `(Ux, Uy, Uz)`; about 29.8 GB. |
| `p.npy` | `(2484440512,)` | `float32` | Kinematic pressure; about 9.94 GB. |
| `k.npy` | `(2484440512,)` | `float32` | Turbulent kinetic energy; about 9.94 GB. |
| `epsilon.npy` | `(2484440512,)` | `float32` | Dissipation rate; about 9.94 GB. |
| `run_cell_offsets.npy` | `(601,)` | `int64` | Cell start/end offsets per run. |
| `run_shape.npy` | `(600, 3)` | `int32` | `(nx, ny, nz)` per run. |
| `run_x_offsets.npy` | `(601,)` | `int64` | Ragged x-axis offsets. |
| `run_y_offsets.npy` | `(601,)` | `int64` | Ragged y-axis offsets. |
| `run_z_offsets.npy` | `(601,)` | `int64` | Ragged z-axis offsets. |
| `x_cell_m.npy` | `(181,485,)` | `float32` | Concatenated x cell-centre coordinates in metres. |
| `y_cell_m.npy` | `(126,721,)` | `float32` | Concatenated y cell-centre coordinates in metres. |
| `z_cell_m.npy` | `(38,400,)` | `float32` | Concatenated z cell-centre coordinates in metres. |
| `case.npy` | `(600,)` | `<U20` | Run IDs matching the compact bundle. |
| `layout_index.npy` | `(600,)` | `int16` | Layout grouping matching the compact bundle. |
| `wd_deg.npy` | `(600,)` | `float32` | Direction matching the compact bundle. |
| `source_time.npy` | `(600,)` | `float32` | Final-time directory label supplied by the exporter; 236 unique values, 529–896. |
| `completed.npy` | `(600,)` | `uint8` | Build-integrity marker; all 600 values are 1. |

### Ragged-shape and integrity results

- All 600 `run_shape` rows have positive dimensions. `nx` ranges 197–358,
  `ny` 114–324, and `nz` is fixed at 64. There are 583 unique `(nx, ny, nz)`
  shapes. Cell counts range from 1,451,904 to 6,905,088, with median
  4,079,744, and sum to 2,484,440,512.
- For every run,
  `run_cell_offsets[i+1] - run_cell_offsets[i] == nx * ny * nz`.  The four
  field arrays have matching cell counts and the U component dimension is 3.
- The x/y/z offset lengths match the corresponding `run_shape` dimensions.
  All offsets start at zero and are monotone. Every per-run cell-centre axis is
  strictly increasing.
- The concatenated axis lengths are 181,485 (x), 126,721 (y), and 38,400 (z).
  Every run has 64 z centres. x and y spacings are approximately 11 m and
  9 m, respectively; z spacing is nonuniform. Physical extents vary by run:
  x start/end ranges are -1594.3..-658.6 m and 1468.7..2394.4 m; y start/end
  ranges are -1506.0..-512.4 m and 485.0..1467.9 m; z spans 1.80..492.80 m.
- `case`, `layout_index`, and `wd_deg` in `family_volume/` match the compact
  bundle row-for-row (600/600 exact comparisons).
- A chunked full scan found no non-finite values. Observed full-volume ranges
  are `U=-1.3770..12.0501 m/s`, `p=-17.3821..22.6978 m²/s²`,
  `k=0.9524..4.7540 m²/s²`, and
  `epsilon=0.0010168..0.564189 m²/s³`. These ranges are diagnostic only; do
  not use them as training normalization without considering physical masks,
  axes, and the intended target.

Because x and y axes and cell counts vary per run, the volume cannot be stacked
as one dense `(case, z, y, x, component)` tensor without padding/resampling.
Use `run_shape`, `run_cell_offsets`, and the three axis-offset arrays together.

## Leakage-safe grouping and preprocessing cautions

1. **Split by `layout_index`, never by row.** The three direction rows share
   exactly the same turbine geometry and layout descriptors. A row-wise random
   split puts the same physical farm in train and validation/test.
2. **Keep compact and volume rows joined by case ID.** They are two views of
   the same 600 runs, not independent samples. Use `case` plus
   `layout_index`/`wd_deg` as an integrity join check.
3. **Use `n_turbines` before reading geometry.** The remaining padded
   `turbine_xy_D` slots are NaN. Do not impute them to zero unless the model
   contract explicitly encodes a mask.
4. **Mask compact fields.** Invalid hub/vertical cells can contain finite,
   nonzero values. Use `valid_hub`/`valid_vert` in visualizations and any field
   loss; keep the mask available as an input feature if a raster model is ever
   introduced.
5. **Do not mix coordinate conventions silently.** Compact grids are in rotor
   diameters and normalized fixed raster coordinates; full-volume axes are
   per-run cell-centre coordinates in metres. `D_m=80` and `hub_height_m=70`
   are the supplied conversion constants.
6. **Do not infer broader operating coverage.** The upload contains one
   reference speed (`U_ref=9 m/s`), the supplied neutral-boundary-layer CFD
   family, and only directions 270/285/300 degrees. Results outside that
   envelope are extrapolations.
7. **Volume fields are cell-centred, not meshes.** The export contains fields,
   coordinates, and reshape order but no connectivity/topology. Do not claim
   mesh-level geometric operations from these arrays alone.

## Representative cases and visualization contract

The first showcase set is deliberately stratified by wake-loss regime and
direction, using the exact row IDs below. The two stress cases cover the
observed target extremes and are useful for checking color-scale robustness.

| Row | Case | Layout/group | Wind direction | Turbines | Wake loss | Why it is useful |
|---:|---|---:|---:|---:|---:|---|
| 167 | `gen_0055_wd300` | `gen_0055` / 55 | 300° | 8 | 9.6069% | Low-loss, sparse/high-separation layout; low-regime showcase. |
| 133 | `gen_0044_wd285` | `gen_0044` / 44 | 285° | 17 | 18.3961% | Near-median target and intermediate occupancy. |
| 534 | `gen_0178_wd270` | `gen_0178` / 178 | 270° | 22 | 27.2722% | High-regime target with denser geometry and a different direction. |
| 210 | `gen_0070_wd270` | `gen_0070` / 70 | 270° | 27 | 37.8980% | Maximum observed wake loss; minimum `min_sep_D` (2.5069 D). |
| 357 | `gen_0119_wd270` | `gen_0119` / 119 | 270° | 8 | 2.2852% | Minimum observed wake loss; regular, widely separated layout. |

For a 3-D full-volume showcase, use the per-run cell-centre axes and render
three orthogonal cuts rather than relying on one oblique view:

- horizontal hub-height-like plane: nearest available `z=70 m`;
- streamwise vertical plane: nearest available `y=0 m`;
- crosswind vertical plane: nearest available `x=0 m`;
- optional 3-D camera: elevation about 25°, azimuth about -55°, with equal
  physical units and a shared field scale when comparing cases.

The exact selected cell centres should be printed in the figure caption because
the ragged axes are not identical. Overlay compact turbine centres after
converting `turbine_xy_D * D_m`, and show the two cut lines on the horizontal
panel. For compact-raster figures, show `valid_hub`/`rotor_hub` alongside the
field so the padded domain is not mistaken for simulated flow.

The preprocessing visualizer writes the following stable, derived paths under
`Case_WindFarm/diagnostics/generated/`:

| Relative link from this report | Intended contents |
|---|---|
| [`../diagnostics/generated/windfarm_case_0167_gen_0055_wd300.png`](../diagnostics/generated/windfarm_case_0167_gen_0055_wd300.png) | Low-loss row 167: hub, vertical, and oblique 3-D cuts. |
| [`../diagnostics/generated/windfarm_case_0133_gen_0044_wd285.png`](../diagnostics/generated/windfarm_case_0133_gen_0044_wd285.png) | Near-median row 133 with the same cut policy and color limits. |
| [`../diagnostics/generated/windfarm_case_0534_gen_0178_wd270.png`](../diagnostics/generated/windfarm_case_0534_gen_0178_wd270.png) | High-loss row 534 with the same cut policy and color limits. |
| [`../diagnostics/generated/windfarm_visualizations.json`](../diagnostics/generated/windfarm_visualizations.json) | Exact case IDs, requested/actual cut coordinates, camera, and output paths. |

These are derived visual outputs only; they should remain untracked. The source
plotting helpers and uploaded reference PNGs are provenance hints, not dataset
members or committed figures.

## Preprocessing hand-off

The data are ready for the next preprocessing stage after the local symlink is
created and the generated figures are rendered. The committed reader contract
should retain the following invariants: read full-volume arrays with
`mmap_mode="r"`, slice by run offsets, validate `run_shape` before reshape,
preserve `layout_index` grouping, preserve compact validity masks, and report
the absent `family_tensor/` directory rather than silently assuming individual
compact files exist. No training/model code is required for this round.

See [`../Dataset/PHYSICS_AND_DATA.md`](../Dataset/PHYSICS_AND_DATA.md) for the
case-package contract once it is generated.
