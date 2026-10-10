# Persistent Wind native role catalogues

Future joint and unified Wind providers reuse a persistent geometry catalogue for both full TRAIN420/VALID90 and segmented `wind_shared_fixed24_v1` TRAIN72/DEV24 training. This removes repeated native coordinate, role-mask, quadrature and cumulative-distribution construction after the first preparation. It does not remove metadata loading, target normalization/calibration, model initialization or checkpoint loading; a warm process still verifies array checksums and maps the catalogue files. Existing running processes retain their original behavior.

The default shared directory is `Case_WindFarm/diagnostics/generated/native_role_catalogues_v1`, which is ignored by Git. Set `HONF_WIND_CATALOGUE_CACHE_DIR=/absolute/cache/path` to relocate it, or set `HONF_WIND_CATALOGUE_CACHE_DIR=off` to retain the original memory-only behavior. Joint and unified providers use this directory automatically without changing the scientific recipe or dataset identity. Standalone diagnostic callers retain their memory-only default and can explicitly pass `persistent_dir` when needed.

## One-time CPU preparation

Run from `HONF_Proj` using the ModularDT environment. Omitting `--prepare` validates and prints the selected partition binding without building entries. Preparation creates only geometry cache files, uses no GPU or optimizer, never visits original TEST rows, and never samples or reads velocity target values. Native views map field headers while constructing cases; the preparation loop does not access `run.U`. Run preparation when host memory and storage I/O are available; the native construction still has its existing per-case peak memory requirement.

```bash
conda run -n ModularDT python tools/wind_native_catalogue_cache.py --scope segmented
conda run -n ModularDT python tools/wind_native_catalogue_cache.py --scope segmented --prepare
conda run -n ModularDT python tools/wind_native_catalogue_cache.py --scope full --prepare
```

The full preparation covers 510 original TRAIN/VALID rows, including the 96 segmented rows. Preparing segmented first is also valid: full preparation verifies and reuses entries already prepared, then builds only missing geometry. Both scopes reuse the existing seed-42 group split; segmented preparation additionally verifies the fixed manifest hash and exact row/direction membership. Neither command selects a new subset or starts training. `--data-root`, `--derived-root`, `--manifest-path` and `--cache-dir` allow explicit existing bindings. Use the same directory for preparation and subsequent training, through the environment variable when overriding the default.

Progress goes to stderr and the final JSON reports selected rows, partition fingerprint, preparation seconds, payload array bytes, build count and verified disk-hit count. The tool retains only the current case in memory. Catalogue files can require substantial disk space across the full native dataset: payload bytes are reported explicitly; storage exhaustion fails visibly rather than substituting an unverified catalogue. Independent source-row geometries are not assumed equal across wind directions.

## Geometry and numerical safeguards

Each catalogue key binds layout identity, exact native axes and grid shape, turbine diameter, hub height and active turbine centers. Native float32 geometry keeps its previous sampling identity; higher-precision centers are hashed without rounding. Geometry changes within a process still trigger the original row-mutation guard. The store additionally binds role definitions, role constants, NumPy version and the relevant geometry/catalogue implementation digest. A changed policy or implementation creates a separate namespace.

Only float32 native coordinates, uint32 non-volume cell indices, float64 role CDFs and role support volumes are persisted. Arrays retain their original precision and order. A warm load verifies geometry, recipe, cell count, inventory, shapes, dtypes, byte lengths and SHA256 payload checksums before use, then opens read-only memory maps. Velocity targets, query draws, RNG state, normalization, losses, model parameters and checkpoint state are never stored. Full and segmented memberships remain independently sealed; shared geometry does not grant access to another partition's targets.

Per-entry process locks prevent two simultaneous launches from constructing the same geometry. Complete entries are published through an atomic directory rename after payload/manifest synchronization. Incomplete staging directories cannot be loaded. Invalid entries are retained under `.invalid-*` and rebuilt under the lock; no dataset files or scientific checkpoints are deleted. The existing bounded memory LRU remains active, with verified disk reuse after eviction or an oversized memory bypass.

## Validation and limits

CPU tests compare memory-only, cold persisted and warm persisted draws exactly, including coordinates, role masses, sampled cell indices and freshly changed targets. Independent subprocess tests verify concurrent single-build publication and a later launch whose catalogue builder is forbidden. Negative tests cover geometry/precision changes, changed recipes, corrupt bytes, missing/truncated payloads, invalid bindings and incomplete staging. The prebuild selection tests cover both sealed scopes and reject target access in geometry preparation.

The implementation has not been prepopulated or timed on all real native rows as part of this change. The first missing geometry still requires construction, and strict checksum verification incurs disk reads on first access in each process. The current training jobs are untouched; a prepared directory benefits future launches. No real full-dataset startup speedup or reduced peak host-memory claim is inferred from the small CPU fixtures.
