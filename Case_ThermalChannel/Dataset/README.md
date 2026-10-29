# ThermalChannel datasets

Large HDF5 files are external resources and are not committed to this source
repository.  Training configuration selects a logical dataset ID; the case
plugin resolves that ID through `dataset_manifest.json` and a machine-local
location map.

## Configure a machine

1. Copy `dataset_locations.example.json` to
   `dataset_locations.local.json`.
2. Replace the example values with paths to the two packed HDF5 files.
3. Run a launch dry-run from the project root:

   ```bash
   python train.py --config project://src/config_core/forward/enhanced_honf_pairwise.json --dry-run
   ```

The local map is intentionally ignored by Git. Paths may be absolute or use
`project://` to anchor them at `HONF_Proj`. As an alternative, set
`HONF_DATA_ROOT` when both resources live below one common root in the
manifest's `Processed_LocalModule_Dataset/` and
`Processed_ChannelThermal_Dataset/` relative locations.

`tools/link_datasets.py` can create optional browsing links under `links/`.
The loader does not depend on those links.

See `PHYSICS_AND_DATA.md` for physical meaning and the files under `schemas/`
for the exact packed-HDF5 contracts.
