# Native interaction routing and the destination-only control

The versioned P-R and P-C families attach small trainable modules to the existing two-round source/environment context. P-R changes the donor weights of the existing MM, ME and EM reductions; P-C retains those weights and modulates each reduced aggregate from its destination state and prescribed global context. Individual physical-source reads, native output laws and all historical default paths are retained. These families have no regional bank or final-query router.

## Families and configuration

| Dataset/output law | Family | Routing configuration location |
|---|---|---|
| Wind nonlinear velocity | `wind_native_interaction_routing_v1` | `standalone_model_config.core.native_routing` |
| Thermal affine temperature / nonlinear flow | `thermal_native_interaction_routing_v1` | `model_config.native_routing` |
| Thermal native cell-centred curl | `thermal_native_curl_interaction_routing_v1` | `model_config.native_routing` |

The nested object uses schema `native_interaction_routing_v1`, mode `P-R` or `P-C`, the parent's initialization seed, and policy `learned_density` or `destination_gain`. P-R fixes `score_hidden=32` and `score_bound=2.0`. The reviewed H128/message128 control uses `control_hidden=32` for Thermal and `34` for Wind, chosen from parameter counts before fitting. Legacy configurations omit this object and continue to execute the original reduction path. Keep `regional_anchors=0` and `factorized_affine_execution=False` for the native-routing comparison.

## Numerical and physical contract

For each typed destination row, P-R preserves the actual old weights `w0`, including measure, presence and MM self-exclusion. With `S=sum(w0)`, scores `s=2*tanh(raw/2)`, `e=expm1(s)` and `mean_e=sum(w0*e)/S`, its density is `rho=1+(e-mean_e)/(1+mean_e)` and its donor weights are `w=w0*rho`. Empty rows use a finite safe denominator and retain zero mass. At the zero-final-layer initialization, the density is exactly one while its final-layer gradient remains live. The original source measure still governs final physical-source reads; heat powers are never donor masses.

P-C uses `aggregate*(1+tanh(g(destination,context)))` with zero final weights and bias. Its modulator sees no individual donor states, donor geometry or source IDs. Both modules use a private initialization RNG so inherited parameters and the continuation sampling RNG are preserved. All inherited and added parameters remain trainable during scientific fitting. Newly added AdamW states begin absent; inherited named states are copied exactly.

Thermal context and flow remain independent of heat forcing, temperature remains affine in the heat powers at a fixed prepared state, and native-curl/proxy definitions are unchanged. Wind remains nonlinear and rejects an exact affine-increment interface. Prepared contexts reject incompatible owners, parameter/input versions, dtypes, devices or routing generations, including a disable/reenable cycle.

## Architecture-child continuation

`tools/diagnostics/native_routing_training.py::build_saved_context` constructs the source-bound model, provider and existing TrainingEngine from explicit paths and a CPU-loaded P/e1000 checkpoint. It binds the segmented T4111 or W2401 checkpoint, recipe, cohort, TRAIN-fitted normalization and saved calibration; it does not fit or refit these quantities. The constructor does not install checkpoint model weights or named moments: those are attached and checked by the existing TrainingEngine architecture-child path. A separate fresh-start check can use the source metadata and saved TRAIN transforms while retaining fresh common initialization and empty moments; it must be labelled separately. The constructor's default keeps exposed DEV targets unmaterialized. Scientific monitoring must explicitly materialize the frozen Thermal DEV22 panel, while Wind loads its fixed DEV batches through the existing validation path.

Pass `architecture_child_initialization` to `TrainingEngine.fit` together with `branch_from_checkpoint`, the literal new arm and a child identity that changes only the versioned provider and enumerated added parameter names. The declaration requires schema `native_interaction_routing_child_v1`, exact source SHA256/epoch/arm, target mode, sorted added names, review epochs, final epoch, update cap and reason. The runtime verifies inherited weights, moments, groups, absolute schedule, membership, sampling and objective contracts before stepping. The checkpoint's `branch_parent` preserves the original identity, exposure and inherited counts; child history counts only new work. Subsequent calls resume the saved child exactly at declared reviews. The independent diagnostic-age branch remains restricted to its original identical-architecture contract.

The retained top-level recipe still describes the original P parent. The new provider configuration, checkpoint arm and architecture-child receipt identify the active R/C model; the Thermal engine loader verifies their agreement before loading the child without consulting its parent file. Provider-state attachment maps only the literal arm label and requires every objective/calibration field to remain exact before calling the unchanged strict state loader. The routed development provider rejects full-data flags and a changed horizon before data access.

## Executed traces and interventions

Prepare with `routing_trace=True` to collect detached diagnostic records from both rounds. Each record includes the legacy route name, explicit donor/destination types, native IDs, presence and measures, self-exclusion, `w0`, scores, density, actual weights, weighted-message norms and aggregate. ME means environment donor to physical-source destination; EM means physical-source donor to environment destination. Trace collection performs CPU copies and is excluded from ordinary inference timings.

The P-R `native_route_intervention` accepts `density_reset` or `equal_measure_reassignment` and an optional selection of `MM`, `ME`, `EM`. Reassignment moves densities only among positive-weight donors with identical measure/presence, ordered by native ID; donor values and IDs remain fixed. Every intervention rebuilds the complete two-round context before final physical reads. Report skipped singleton buckets, changed donors, physical errors and exact restoration of the ordinary call. These interventions diagnose model dependence; they do not identify physical causal edges.

Ordinary P-R executes every eligible nonlinear value message. Positive normalized weights and concentrated support do not establish eliminated work. Compare complete calls at identical queries, native outputs, CPU staging/return and checkpoint scope; report cold setup, memory, training time and numerical failures separately. Development evidence uses exposed fixed cohorts, and neither a routed graph nor a successful short qualification authorizes a formal run.

The [native-routing development report](../reports/HONF_Native_Interaction_Routing_Development_Report.md) records the measured warm children, physical misses, complete costs and evidence limitations; the [evidence index](../reports/HONF_Native_Interaction_Routing_Evidence_Index.md) points to the ignored numerical artifacts.
