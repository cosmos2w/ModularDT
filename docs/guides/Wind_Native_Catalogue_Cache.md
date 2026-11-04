# Persistent Wind native role catalogues

Wind providers can reuse a persistent geometry catalogue for both full TRAIN420/VALID90 and segmented `wind_shared_fixed24_v1` TRAIN72/DEV24 training. This removes repeated native coordinate, role-mask, quadrature and cumulative-distribution construction. It does not remove metadata loading, target normalization/calibration, model initialization or checkpoint loading; a warm process still verifies array checksums and maps the catalogue files.

The default shared directory is `Case_WindFarm/diagnostics/generated/native_role_catalogues_v1`, which is ignored by Git. Set `HONF_WIND_CATALOGUE_CACHE_DIR=/absolute/cache/path` to relocate it. The native relation-anchored provider requires verified prebuilt entries: it neither builds missing geometry nor writes repairs during training. Existing joint/unified and standalone callers retain their original optional persistent-cache and memory-fallback policies; `HONF_WIND_CATALOGUE_CACHE_DIR=off` continues to select memory-only operation for those legacy callers. It does not enable fallback for a strict relation-anchored provider.

## One-time CPU preparation

Run from the repository root using the ModularDT environment. Omitting both modes prints the selected partition binding. `--verify` reads required prebuilt entries and fails on a miss without building, locking or writing. `--prepare` creates only geometry cache files, uses no GPU or optimizer, never visits original TEST rows, and never samples or reads velocity target values. Native views map field headers while constructing cases; the preparation loop does not access `run.U`. Run preparation with sufficient host memory and disk budget; construction retains its existing per-case peak memory requirement.

```bash
conda run -n ModularDT python tools/wind_native_catalogue_cache.py --scope segmented
conda run -n ModularDT python tools/wind_native_catalogue_cache.py --scope segmented --prepare
conda run -n ModularDT python tools/wind_native_catalogue_cache.py --scope full --prepare
conda run -n ModularDT python tools/wind_native_catalogue_cache.py --scope full --verify
```

The full preparation covers 510 original TRAIN/VALID rows, including the 96 segmented rows. Preparing segmented first is also valid: full preparation verifies and reuses entries already prepared, then builds only missing geometry. Both scopes reuse the existing seed-42 group split; segmented preparation additionally verifies the fixed manifest hash and exact row/direction membership. Neither command selects a new subset or starts training. `--data-root`, `--derived-root`, `--manifest-path` and `--cache-dir` allow explicit existing bindings. Use the same directory for preparation and subsequent training, through the environment variable when overriding the default.

Progress goes to stderr and the final JSON reports selected rows, partition fingerprint, preparation seconds, payload array bytes, build count and verified disk-hit count. The tool retains only the current case in memory. Catalogue files can require substantial disk space across the full native dataset: payload bytes are reported explicitly; storage exhaustion fails visibly rather than substituting an unverified catalogue. Independent source-row geometries are not assumed equal across wind directions.

## Geometry and numerical safeguards

Each catalogue key binds layout identity, exact native axes and grid shape, turbine diameter, hub height and active turbine centers. Native float32 geometry keeps its previous sampling identity; higher-precision centers are hashed without rounding. Geometry changes within a process still trigger the original row-mutation guard. The store additionally binds role definitions, role constants, NumPy version and the relevant geometry/catalogue implementation digest. A changed policy or implementation creates a separate namespace.

Only float32 native coordinates, uint32 non-volume cell indices, float64 role CDFs and role support volumes are persisted. Arrays retain their original precision and order. A warm load verifies geometry, recipe, cell count, inventory, shapes, dtypes, byte lengths and SHA256 payload checksums before use, then opens read-only memory maps. Velocity targets, query draws, RNG state, normalization, losses, model parameters and checkpoint state are never stored. Full and segmented memberships remain independently sealed; shared geometry does not grant access to another partition's targets.

Verified numerical-recipe identity can read the existing compatible v1 namespace in place; its geometry, role policy, NumPy version and payload checksums are still checked. No payload is copied or relabeled. Legacy provider identities retain their original source-based recipe identity. Descriptor exhaustion (`EMFILE`/`ENFILE`) propagates as the actual operating-system error rather than a false cache miss. Memory-mapped files consume descriptors: budget the process file limit for the simultaneously retained catalogue arrays and other data handles before a large workload.

Per-entry process locks prevent two simultaneous launches from constructing the same geometry. Complete entries are published through an atomic directory rename after payload/manifest synchronization. Incomplete staging directories cannot be loaded. Invalid entries are retained under `.invalid-*` and rebuilt under the lock; no dataset files or scientific checkpoints are deleted. The existing bounded memory LRU remains active, with verified disk reuse after eviction or an oversized memory bypass.

## Validation and limits

CPU tests compare memory-only, cold persisted and warm persisted draws exactly, including coordinates, role masses, sampled cell indices and freshly changed targets. Independent subprocess tests verify concurrent single-build publication and a later launch whose catalogue builder is forbidden. Negative tests cover geometry/precision changes, changed recipes, corrupt bytes, missing/truncated payloads, invalid bindings and incomplete staging. The prebuild selection tests cover both sealed scopes and reject target access in geometry preparation.

Availability depends on the local prepared store; compatible entries are verified on first access and those reads cost time. Missing geometry requires explicit preparation for strict callers. A real shared-cache check forbids builder/writer execution and verifies a mapped disk hit; synthetic fixtures cover corruption, geometry precision, compatibility and resource errors. These checks establish data reuse, not full-dataset startup acceleration or reduced peak host memory.
