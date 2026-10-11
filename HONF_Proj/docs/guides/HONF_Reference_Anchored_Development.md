# Reference-Anchored HONF Development

Retain a useful trained function before changing its training recipe or execution. The Run2202 compatibility loader, optional affine executor and native diagnostic APIs serve different purposes: preservation of a historical predictor, evaluation of the same computation at lower cost, and controlled investigation of learning. Their qualification and scientific findings are recorded in the [2026-10-11 consolidation report](../reports/HONF_Reference_Anchored_Consolidation_20261011.md).

## Preserve the mature Wind function

The [migration module](../../Case_WindFarm/src/windfarm/migration.py) extends the existing 52-tensor Run2202 direct-to-P bridge. It accepts the literal hash-bound e5000 source checkpoint, exact TRAIN normalizer/profile, run identity and saved source-state record. It checks learned keys, shapes, dtypes, feature definitions and E8 source/environment interpretation. Its output embeds the direct weights, normalizer/profile, full experiment identity, training history, sampler/RNG state, original optimizer moments and 14 excluded training-only refinement tensors. The excluded tensors and moments are archival provenance; the migration is an inference artifact rather than a P optimizer resume checkpoint.

Create an artifact in an ignored, explicitly selected destination with the CPU-only CLI:

```bash
conda run -n ModularDT python Case_WindFarm/tools/migrate_wind_direct.py \
  --checkpoint /absolute/path/to/Run_2202_20261008_194400_w3_h128_full_train420_q4096_e5000/checkpoints/epoch_5000_model.pt \
  --normalization /absolute/path/to/the/exact/train_only_normalization.json \
  --output /absolute/path/to/ignored/wind_run2202_direct_p_e5000.pt
```

The loader reads only that migration file:

```python
from windfarm.migration import load_run2202_migration

model, provenance = load_run2202_migration(migration_path)
model = model.to(device)
prepared = model.prepare_scene(native_wind_scene)
velocity_mps = model.predict_physical(prepared, receivers_D, chunk_size=512)
```

Supply the existing native Wind scene representation and rotor-diameter receiver coordinates. Preserve physical source IDs, ordering, dimensions, geometry-only environment measures and feature construction. A geometry or configuration change requires a fresh prepared context. The migrated e5000 model inherits full TRAIN420 exposure and its complete prior learning history; its age and exposure must remain visible in comparisons with fresh or subset-trained P models.

## Select the affine executor explicitly

`factorized_affine_execution=True` selects an optional executor in the shared context/read core and the interaction-preserving Wind/Thermal adapters. The setting adds no learned tensors. It applies slices of each existing first-layer weight to source, global, query and relative features separately, sums the preactivations with the original bias once, and then applies the original nonlinearities and remaining layers. It also factors the first affine projections in existing MM/ME/EM context messages. Masks, donor measures, source sums, output transforms, native-curl computation and zero-offset affine temperature laws remain part of the original computation.

Construct the adapter with the flag when serializing a selected execution mode. Existing configurations omit the flag and retain historical execution. `core.set_factorized_affine_execution(...)` is available for controlled same-weight checks; switching invalidates already prepared contexts. Rebuild the context after switching, changing weights, moving the model or changing input configuration. The adapters export the current flag, and their strict loaders accept its optional presence.

Floating-point reassociation can change FP32 results. Qualification must include native fields, derivatives and actual optimizer updates under fixed criteria, followed by complete costs at identical receiver/output scopes. A passing FP64 algebra check cannot qualify a failed FP32 deployment case. Keep existing selected-versus-dense and B48-versus-B8 failures separate from the new executor comparison. The consolidation report states the qualified workloads and any rejected cases; enabling the flag does not expand that numerical coverage automatically.

The 2026-10-11 native GPU checks rejected factoring for Wind training gradients/AdamW and Thermal complete physical outputs, including interface flux and some curl/endpoint-derived outputs. Its measured cost changes were small and mixed. Keep the flag false for authoritative training and deployment; the optional path is retained for controlled execution research with explicit failure coverage.

## Diagnose the objective at the optimizer boundary

The [comparison CLI](../../tools/diagnostics/reference_anchored_diagnose.py) supports `freeze-panels`, `loss`, `gradients`, `native-metrics`, `counterfactual-updates` and `fit-branch`. All paths, identities, factories and panel selections are explicit inputs. Metadata-only freezing retains source TRAIN separation and input strata. Wind freezing requires `--wind-data-root` and joins actual native layout/count/direction metadata; the fixed manifest's sorted turbine-count category list is not a per-layout array. Selection opens no field arrays and verifies all three directions and common TRAIN membership. Its panel fingerprint binds the selected cases before model errors are inspected. For native diagnosis, the context factory has the contract `factory(config, checkpoint_payload) -> (model, provider, engine)` and may return a fourth data-access provenance mapping; it must construct an existing provider without optimization or refitting calibration. The CLI restores model and provider state strictly and rejects provider identity drift.

```bash
conda run -n ModularDT python tools/diagnostics/reference_anchored_diagnose.py gradients \
  --checkpoint /absolute/path/to/a/saved/checkpoint.pt \
  --context-factory your_context_module:create_context \
  --factory-config /absolute/path/to/context.json \
  --panel /absolute/path/to/fixed_train_batches.json \
  --output /absolute/path/to/ignored/gradient_receipt.json
```

The panel declares the checkpoint age, sampler age, phase, execution arm, ordered TRAIN case IDs and microbatch keys. `gradients` uses the exact provider `LossTerm` coefficients and target-only macro denominators, then measures per-term/shared-parameter gradients, common-basis cosines, global clipping and named optimizer-state provenance without stepping. `loss` measures the fixed-state objective. `counterfactual-updates` requires an explicit optimizer-update count equal to its number of interventions, restores identical state for each cloned arm, and evaluates the actual AdamW displacement on a disjoint TRAIN probe. Its `d_v` is the selected probe loss gradient dotted with that displacement; state whether the selected loss includes the provider coefficient. Independently cloned updates establish counterfactual effects; per-loss AdamW updates cannot be added as an exact decomposition.

The old Wind Full objective includes base-message approximation against detached fine messages. Its physical main/full replay is identical supervision when both physical tensors coincide. Measure the base path's gradients into shared context, the auxiliary contribution to global clipping, its recovered message scale, and its named moments before attributing a fresh P gap to architecture. Saved calibration replay checks the native TRAIN binding and receipt dimensions; ordinary fresh-provider calibration remains the default.

Thermal already has separate nonlinear flow and affine temperature source reads. Diagnose native-curl omega, u/v/p, temperature regions, interface flux, response and operator terms along their actual parameter paths. Use cloned checkpoint moments, exact native denominators and disjoint TRAIN effects to select a single diagnostic. Removing omega supervision must preserve the curl output and every other loss coefficient. Any blocked-gradient or flow-only arm is a diagnostic control whose sacrificed channels remain reportable.

## Keep short diagnostics as explicit children

The CLI's `fit-branch` mode performs a zero-step preflight by default; `--execute` starts only its hash-bound declared child. It verifies the identical sampling namespace and named AdamW moments. `TrainingEngine.fit(..., branch_from_checkpoint=..., diagnostic_age_branch=...)` provides a bounded objective-diagnostic child without loosening exact resume. The declaration pins the source hash/age, fitting case keys, loss-weight overrides, TRAIN evidence hash, review ages and an update limit of at most 400 per arm. The provider retains its exact parent identity plus a nested `diagnostic` mapping containing only `fit_case_keys` and `loss_weight_overrides`; the overrides must also appear in the sealed TRAIN evidence's `allowed_loss_weight_overrides` list. Scientific engine settings, named parameter groups, optimizer schedules, initialization/data/sampling identity and selector configuration remain equal to the parent; only run labels and monitoring/checkpoint presentation can change. Strict model loading prevents an undeclared forward architecture change.

The child starts at the source's next absolute sampler/schedule age with its exact parameters, moments, RNG and provider state. Its history counts only new visits and updates; parent exposure and historical counts remain in `branch_parent`. Every child epoch checks its frozen membership and update limit. Exact child continuation accepts only predeclared review stops and cannot exceed the declared horizon. A shorter diagnostic fitting inventory does not erase inherited full-data exposure, and child scheduler ages are not full-source dataset epochs.

Choose the comparison from TRAIN evidence, freeze both arms' initialization/samples and review at 100/200/400 new updates when those budgets are declared. Reuse saved exposed DEV measurements for unchanged checkpoints. Report actual visits, macro updates, query records, elapsed time, inherited exposure and output scope. The default Thermal development protocol remains [fixed stratified quarter data](Thermal_Model_Development_Protocol.md); formal full-data training requires a separate explicit request.

## Report four separate decisions

State whether the mature Wind function is retained, whether same-function execution is qualified and cheaper, what measured evidence limits current joint learning, and which specific organizer learning or computation-reuse experiment is justified. Learned graph support establishes executed information flow rather than physical causality. Full-access and sparse work must be reported separately, and complete cost includes preprocessing, context preparation, native curl, all requested output roles, synchronization and CPU return. Prepared-read timing is a distinct reuse scenario rather than an additive decomposition of a complete call.

Retain compact physical/cost receipts, protected hashes and selected PDF figures in ignored evidence directories. Embed small raster companions only when the report display requires them, inspect the rendered figures, and preserve one-line paragraphs/captions. Audit every outgoing commit object, including artifacts added and later removed, before the repository's pre-push gate. Durable source, tests, configuration and documentation are the upload boundary.
