# Prebuilt Wind native role catalogues

New Wind training consumes prebuilt shared geometry catalogues for original TRAIN420/VALID90 and fixed segmented TRAIN72/DEV24. Training never builds catalogue geometry, publishes entries or silently substitutes a RAM build. A missing, incompatible or corrupt entry fails with its layout/geometry identity and the explicit preparation command. Existing running processes retain their loaded source and behavior.

The shared directory is `Case_WindFarm/diagnostics/generated/native_role_catalogues_v1`, bound by the workspace symlink to its declared DATA2 location. `HONF_WIND_CATALOGUE_CACHE_DIR=/absolute/cache/path` can select another explicitly prepared store. Disabling the persistent store does not enable memory-only training. Reuse the verified shared store; do not copy it into individual runs.

## Explicit CPU preparation and verification

Run from the repository root using the ModularDT environment. The default command only prints the selected partition binding. Only `--prepare` permits construction; `--verify` checks every required catalogue without building, writing or copying payloads. Both modes stream one case at a time, and their final JSON separates verified disk hits, legacy v1 hits, builds, writes and missing/incompatible entries.

```bash
conda run -n ModularDT python tools/wind_native_catalogue_cache.py --scope full
conda run -n ModularDT python tools/wind_native_catalogue_cache.py --scope full --verify
conda run -n ModularDT python tools/wind_native_catalogue_cache.py --scope full --prepare
conda run -n ModularDT python tools/wind_native_catalogue_cache.py --scope segmented --verify
```

Full scope covers the 510 original TRAIN/VALID rows, including the 96 segmented rows. Both scopes reuse the existing seed-42 group split; segmented scope also verifies the fixed manifest and exact row/direction membership. Preparation and verification use only geometry: no query draws, velocity targets, optimizer, GPU or original TEST rows. Native views map field headers when constructing cases, but the loop does not access `run.U`. `--data-root`, `--derived-root`, `--manifest-path` and `--cache-dir` select explicit existing bindings. Record a disk budget and verify the DATA2 mount/binding before preparation; full payloads are substantial.

## Numerical identity and in-place v1 reuse

The implementation identity covers the canonical Python syntax of the actual catalogue dependencies: role masks, catalogue construction, native coordinates, downstream envelope, support weights, centre weights and axis validation. Comments, docstrings and unrelated plotting functions are excluded. Role definitions/constants and NumPy version remain in the recipe. Changing numerical code or policy changes the recipe namespace.

The verified existing v1 recipe is accepted only for its explicitly audited numerical implementation digest. Loading checks the numerical namespace first, then that known compatible v1 namespace; it validates the original v1 manifest against its original recipe. This compatibility path neither edits manifests nor creates copied payloads or replacement aliases. A numerical dependency change disables v1 compatibility. The old whole-file implementation digest is retained solely as the identifier of those existing payloads.

Each geometry key still binds layout identity, exact native axes/grid shape, turbine diameter, hub height and active turbine centres. Float32 geometry retains its historical sampling identity; higher-precision centres are hashed without rounding. In-process geometry mutation remains rejected. Every disk load verifies recipe, geometry, cell count, array inventory, shapes, dtypes, byte lengths, SHA256 checksums, role support and CDF bounds before returning read-only memory maps.

Catalogue coordinates remain float32, non-volume indices uint32, and role CDFs/support measures float64. Arrays retain their original ordering and precision. Targets, sampled queries, RNG, normalization, losses and checkpoints are never stored in this cache. Shared geometry does not grant access to another partition's targets. The bounded RAM LRU retains verified mappings and reopens disk entries after eviction or an oversized-entry bypass.

Only explicit preparation uses per-entry process locks and atomic publication. It may archive an invalid numerical-namespace entry before rebuilding it. Training does not acquire these write locks, archive entries or repair them implicitly.

## Measured coverage and regressions

The R15-2 repair verified all 510 real TRAIN420/VALID90 entries in 107.7 seconds: 510 disk hits, all from compatible v1 payloads, zero builds, zero writes and zero missing entries. The retained arrays total 70,311,713,312 bytes; this is existing shared payload size, not newly allocated disk space. The prior tuning replay established exact coordinates, CDFs, indices, supports and Q8192 draws on four real cases. These checks establish catalogue compatibility and coverage; they do not measure complete training speed or model quality.

`Case_WindFarm/tests/test_prebuilt_native_role_cache.py` forbids builder/writer execution during training, rejects missing/corrupt/incompatible entries, checks geometry precision, verifies in-place v1 reuse and exercises an actual shared disk hit. Relation-anchored training tests separately keep Q8192 training independent of the fixed Q4096/batch4 validation panel. Missing geometry must be prepared explicitly before launching a new lineage.
