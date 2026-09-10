# WindFarm datasets

The wind-farm data are external resources and are not copied into this
repository.  The committed manifest names the resources by stable logical IDs;
the machine-local location map resolves those IDs to the hand-off directory or
to the local symlink under `Dataset/links/`.

## Configure the local link and map

Create the local link once, pointing it at the data-owner directory:

```bash
ln -s /path/to/wind_farm Dataset/links/wind_farm
```

Then copy the example map and keep the copied file untracked:

```bash
cp Dataset/dataset_locations.example.json \
   Dataset/dataset_locations.local.json
```

The example map resolves through `project://Case_WindFarm/Dataset/links/` and
therefore remains portable.  An absolute path is also acceptable in a local
map.  Do not put a machine-specific path in a committed profile or manifest.

## Logical resources

| ID | Resource | Contents | Approximate logical size |
|---|---|---|---:|
| `wind_farm_tensor_v1` | `family_tensor.npz` | 600 compact cases, layout metadata, masked hub/vertical velocity rasters, and scalar wake-loss targets | 955 MB |
| `wind_farm_volume_v1` | `family_volume/` | 600 ragged OpenFOAM runs with `U`, `p`, `k`, `epsilon`, cell-centre axes, offsets, and run metadata | 59.6 GB |

The compact bundle is convenient for metadata and raster inspection but
decompressing a field array can require substantial memory.  The volume arrays
must be opened with `mmap_mode="r"` and sliced by run.  `U.npy` alone is about
29.8 GB; it should never be loaded in full for a single-case plot.

The source upload also contains plotting scripts and reference PNGs.  They are
useful provenance hints but are not part of either logical dataset and are not
used as inputs to the preprocessing contract.  The current upload provides
the compact NPZ bundle rather than a separate `family_tensor/` directory of
individual compact arrays; the manifest records the observed layout.

## Splitting and safety

There are no source-provided train/validation/test partitions.  The 600 rows
are three wind-direction evaluations of 200 layouts.  If a downstream study
needs partitions, generate them by `layout_index`, keeping all three
directions together.  Do not split rows independently.

Every full-volume run must have `completed.npy == 1`.  Before reshaping a run,
check that:

```text
run_cell_offsets[i+1] - run_cell_offsets[i]
    == nx * ny * nz
```

and apply the same offset logic to the three coordinate-axis arrays.  See
`PHYSICS_AND_DATA.md` for the complete field and category contract.
