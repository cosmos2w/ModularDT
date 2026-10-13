# Fresh matched P/R1/R2 comparison

This guide records the maintained preparation path for the segmented P/R1/R2 comparison. It defines recipes and audit commands; the [multiview comparison report](../reports/HONF_Multiview_Native_Routing_20261012.md) and [evidence index](../reports/HONF_Multiview_Native_Evidence_Index_20261012.md) record the measured campaign and its limits. The shared native-routing implementation and existing `TrainingEngine` remain the only model and optimizer paths.

| Dataset | Fixed protocol | Seed | Primary queries and native objectives | Effective/microbatch cases | Output law |
|---|---|---:|---|---:|---|
| ThermalChannel | `fixed25_v1`, TRAIN150 / exposed DEV22 | 0 | Q1024 plus the established auxiliary terms and TRAIN-only calibration | 48 / 8 | `native_curl_cell_centred_v1` flow and source-resolved affine temperature |
| WindFarm | `wind_shared_fixed24_v1`, TRAIN72 / DEV24 | 42 | Q4096 across the five fixed velocity roles | 24 / 8 | TRAIN-profile-scaled nonlinear three-component velocity |

The maintained recipes at `src/config_core/forward/joint_regional/fresh_multiview_{thermal,wind}_v1.json` freeze H128/message128, two pair rounds, the original 2500-epoch objective and schedule, `case_epoch_v1` sampling, and checkpoints/field selection every100 epochs. Review boundaries are100,500,1000,1500,2500. P, R1 and R2 use identical selected memberships, common initialized model tensors, TRAIN-fitted transforms and calibration, case/query streams, optimizer settings, effective batches and selection opportunities. R1 keeps the literal scalar P-R checkpoint schema and shapes; R2 alone declares `native_interaction_routing_v2` with two64-feature blocks.

Every fresh receipt binds the active recipe/provider, TRAIN scales and calibration, sampler namespace, complete initial-state digest and common-state digest. The common digest excludes only `core.native_routing.*`, so it compares P, R1 and R2's shared physical/value trunk while the full digest still records each private router. The receipt states `source_checkpoint: null`, `inherited_parameter_names: []`, and `optimizer_state_start: fresh_zero_moments`; no historical model state or AdamW moment is attached. `native_validation_policy: consolidated_route_flags_v1` is explicit in the recipe and receipt, applied before fitting/evaluation and reapplied after exact checkpoint loading; legacy runs keep the default eager validation path.

Wind comparison arms reuse the verified shared role catalogues at `Case_WindFarm/diagnostics/generated/native_role_catalogues_v1` and the exact-keyed TRAIN calibration cache. They must not create an arm-specific catalogue. The Thermal calibration receipt is shared by P/R1/R2 and binds the native-curl law and committed training-source hashes. Preparation and stream checks create only small manifests, recipes and receipts; they do not copy case data or create model checkpoints.

## Allocate and prepare

Run from `HONF_Proj` in the `ModularDT` environment. Choose a new campaign directory under the ignored `diagnostics/generated/multiview_20261012/` root for each dataset; the allocator refuses an existing campaign directory and writes three UUID-owned recipes plus one comparison manifest. It does not create run directories or optimizer state.

```bash
python tools/joint_regional_train.py allocate-matched --task thermal --output-dir diagnostics/generated/multiview_20261012/thermal_fresh_primary_01
python tools/joint_regional_train.py allocate-matched --task wind --output-dir diagnostics/generated/multiview_20261012/wind_fresh_primary_01
```

For Thermal, run the `P` recipe's TRAIN-only native-curl calibration once and use the allocated campaign output directory; then prepare all three arms. Wind needs only the preparation command. `prepare-matched` forces CPU, requires the sealed source set to be clean, checks common tensor and R1/R2 score-trunk hashes, and writes an initialization seal without materializing DEV targets or creating optimizer state.

```bash
python tools/joint_regional_train.py calibrate-thermal --recipe-json diagnostics/generated/multiview_20261012/thermal_fresh_primary_01/recipes/p.json --device cuda:0 --output-dir diagnostics/generated/multiview_20261012/thermal_fresh_primary_01/calibration-preparation
python tools/joint_regional_train.py prepare-matched --comparison-manifest diagnostics/generated/multiview_20261012/thermal_fresh_primary_01/comparison_manifest.json
python tools/joint_regional_train.py prepare-matched --comparison-manifest diagnostics/generated/multiview_20261012/wind_fresh_primary_01/comparison_manifest.json
```

The source identity for fresh runs adds `src/honf_forward_core/interface_fields/native_interaction_routing.py` and both maintained fresh-comparison JSON files to the existing recovery source set, plus the Thermal native-curl implementation. It does not change the source set used to validate retained P/R/C identities. The general `RunStore.propose` requires a caller-supplied run ID and produces a timestamped ConfigBundle directory, while this runner accepts an explicit output path and has no comparison-specific allocator. The fresh allocator therefore owns collision-checked `TMV1-`/`WMV1-` UUIDs inside each immutable campaign manifest instead of introducing a global registry.

## Matched TRAIN cost and stream checks

After the source is sealed and the root has dispatched the measurement, the existing engine exposes the exact epoch sampler without requiring callers to copy its rules: `TrainingEngine.epoch_case_order(provider, epoch)` returns the deterministic case order and dataset namespace; `epoch_update_slices(case_count)` returns the effective-case macro boundaries; and `prepare_update_microbatches(...)` constructs each macro's exact arm/phase/update/microbatch-keyed batches. `TrainingEngine.fit` uses these same methods. A bounded complete-epoch forecast can build each fresh model/provider/engine with `tools/joint_regional_train.py::build`, call the shared `_run_update` once for each of the four Thermal or three Wind epoch-1 macros, and call `_evaluate` once after the DEV evaluation boundary is sealed; these disposable clones create no checkpoint or training history. The three-arm measurement uses 12 Thermal and nine Wind optimizer steps, 21 total, separately accounted from scientific histories.

An example measurement driver should resolve Thermal recipes with `_bind_calibration_receipt`, build each arm from its immutable recipe, construct its ordinary declared AdamW groups and epoch-1 schedule, and use the sampler methods above for every macro before calling `_run_update`. It may call `_evaluate(model, provider, arm, "joint", 100)` once to time the exact full validation panel after explicit DEV materialization is authorized. The preparation seal must exist first. This probe is a cost measurement, not a result used to choose or amend an architecture; retain each arm's complete TRAIN, validation and process timing.

After all three actual runs have saved their first reviewable history, audit the observed epoch-1 case order, per-update query hashes, masks and denominator streams with the standalone read-only diagnostic. It reads RunStore artifacts from `configs/experiment_identity.json` and `metrics/history.json`, accepts a complete unambiguous legacy flat pair, rejects conflicting dual layouts, and leaves the sealed runner source unchanged:

```bash
python tools/diagnostics/audit_fresh_matched.py streams --comparison-manifest PATH_TO_CAMPAIGN/comparison_manifest.json
python tools/diagnostics/audit_fresh_matched.py forecast --output-dir PATH_TO_ALLOCATED_P_RUN --horizon 2500
```

The stream audit requires all three histories to contain the same complete epoch range, verifies case order, query stream and case/update/microbatch counters at every saved epoch, and checks the full mask and denominator receipts at epoch1. It writes an immutable `matched_stream_audit_eNNNN_<history-digest>.json` receipt for each shared completed age, so later reviews preserve earlier audit results. The forecast uses measured complete TRAIN-epoch and full-DEV validation durations from the run history and reports all remaining TRAIN epochs plus scheduled reviews. The older `joint_regional_train.py validate-matched-streams` and `forecast` subcommands expect flat artifact paths and do not read the maintained RunStore layout; use this standalone audit for fresh RunStore outputs.

## Reviewed fitting and forecast commands

Fitting requires a separately dispatched run and a new empty UUID-owned output directory from the manifest. Start all three arms at the same stop, review the saved P/R1/R2 physical fields and counters together, and extend them together only after the review decision. The runner accepts only the common 100-epoch selection ages and exact-resumes each run from its own latest checkpoint.

```bash
python tools/joint_regional_train.py start --recipe-json PATH_TO_CAMPAIGN/recipes/p.json --device cuda:0 --output-dir PATH_TO_ALLOCATED_P_RUN --stop-after 100
python tools/joint_regional_train.py start --recipe-json PATH_TO_CAMPAIGN/recipes/p_r.json --device cuda:0 --output-dir PATH_TO_ALLOCATED_R1_RUN --stop-after 100
python tools/joint_regional_train.py start --recipe-json PATH_TO_CAMPAIGN/recipes/p_r2.json --device cuda:0 --output-dir PATH_TO_ALLOCATED_R2_RUN --stop-after 100
python tools/joint_regional_train.py resume --recipe-json PATH_TO_CAMPAIGN/recipes/p.json --device cuda:0 --output-dir PATH_TO_ALLOCATED_P_RUN --stop-after 500
python tools/diagnostics/audit_fresh_matched.py forecast --output-dir PATH_TO_ALLOCATED_P_RUN --horizon 2500
```

Repeat exact resume for the remaining common review ages only after each review. The standalone `forecast` command uses measured full TRAIN-epoch and full-DEV review durations from the run history, adds all remaining TRAIN epochs and scheduled validation reviews, and records observed case visits, optimizer updates, work counts, objective terms and current best-field age. The final selection remains a field-only `field_score` minimum over the same 25 saved100-epoch checkpoints for every arm.

No second seed is included in these primary recipes. A separate `conditional_confirmation` receipt permits only Thermal1 or Wind43 and only P plus the frozen preferred routing arm, after all six literal e2500 endpoints and mandatory physical/cost measurements are complete. It binds an immutable eligibility decision and enforces the physical/robustness pass, Q8192 price at most1.30 P, no unresolved implementation failure, and Wind preference when both datasets qualify. Primary recipes without this receipt retain the exact Thermal0/Wind42 guards. Confirmation reuses the primary TRAIN-fitted coefficients/scales while drawing independently initialized common tensors and observed case/query streams; it cannot attach a primary checkpoint or use its optimizer state.
