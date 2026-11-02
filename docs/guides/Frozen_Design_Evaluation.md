# Frozen-model design evaluation

The case-owned differentiable evaluators expose native physical observations while keeping the loaded model in evaluation mode with parameter gradients disabled. Load a literal checkpoint through its maintained strict loader and preserve its saved transforms, original member IDs, controls and field ordering. Design coordinates remain live tensors; reference fields and observations never become scene inputs.

## Case interfaces

`channelthermal.inverse.relation_anchored.ThermalDesignEvaluator(model, record)` evaluates `evaluate_design(centers, heat=None, read_mode="full")`. Centers have shape `[M,2]` in original module order and native physical units. The result contains native fields, the normalized smooth maximum over valid moving material receivers, the sampled maximum, native fluid-band pressure difference, work counts and the prepared organization. Its temperature scale comes from the checkpoint's internal-temperature transform. Moving interface/material receivers and geometry-dependent masks follow the current centers.

`windfarm.inverse.relation_anchored.WindDesignEvaluator(model, case)` evaluates `evaluate_design(centers, coordinates, query_ids=None, read_mode="full", return_trace=False)`. Centers and query coordinates are `[M,3]` and `[Q,3]` in the adapter's diameter-normalized frame. The returned velocity is physical m/s, followed by the core output and the prepared scene/organization pair. For inverse identification, scale observed m/s residuals by the saved dimensionless TRAIN standard deviations multiplied by the saved physical reference velocity.

Both interfaces require explicit interpretation of `full` versus `restricted` reads. Full Pi is organization-weighted access. Restricted P95 support is a separate execution choice with discrete support changes; small finite differences and automatic derivatives should be compared with that distinction recorded.

## Exact preparation reuse

Wind's `prepare(centers)` and `evaluate_prepared(prepared, coordinates, ...)` let several receiver blocks use the same unchanged scene organization. Combine their objectives before one backward pass, or rebuild after consuming the autograd graph. Thermal's maintained native evaluation already serves the fluid, interface and material receiver union from one preparation/decoding path; compare against that operational reference.

Rebuild after changing layout, heat, materials, operating conditions or weights. An old prepared state is not a supported prediction interface for another design. Detached organization summaries can guide proposals, but every predicted design still receives its own complete preparation.

## Detached proposal guidance

`honf_runtime.inverse_optimization.ProposalAdam(initial_xy, lr, arm)` owns a live `xy` leaf. Evaluate the case objective using that tensor, then call `step(loss, project, learned_B=None, mu_e=None)`. The callback `project(old, proposed)` returns feasible coordinates and case-owned diagnostics without additional model evaluations. The helper consumes one backward pass and returns objective/gradient counts, graph diagnostics, ordinary/proposed/realized displacement, gradient-dot-realized displacement and its own elapsed time. The caller measures complete forward-inclusive cost and retains the full history.

| Arm | Proposal transform |
| --- | --- |
| A0 | Ordinary Adam. |
| A1 | Learned co-service graph at the initial design, held as detached metadata. |
| A2 | Learned co-service graph refreshed from the current preparation every 10 design optimization epochs. |
| A3 | Geometry affinity refreshed on the same cadence, with the initial median pair distance held as bandwidth. |

For movable original M IDs, `learned_graph(B, mu_e)` uses their columns of the actually executed full Pi and normalized physical E measures. It removes diagonal co-service, forms a symmetric normalized Laplacian with zero isolated rows/columns, and uses `inverse(I + L)`. The helper applies that transform separately to the two displacement columns and matches the ordinary displacement's RMS before the common feasibility callback. Graph construction is detached; it does not change the forward model or imply physical causality.

Keep identical starts, learning rates, objective definitions, receivers, epoch budgets and feasibility handling across arms. Extract graph metadata only when required, and separate tracing, plotting and numerical diagnostics from uninstrumented timing. Report graph extraction/matrix costs, preparation/decode work, memory and complete elapsed time separately. A graph-guided result is useful only when the measured task outcome justifies its added cost.

## Reference limits

Thermal reference verification must solve each nominated geometry with the declared physical generator and score the same native receiver definitions. Stored fields sampled at moved locations do not supply new-geometry truth. Wind layout identification from saved CFD observations measures recovery and held-out observation matching; without a matching moved-layout CFD case it does not establish physical layout-response, power or AEP improvement. Learned organization and directed interventions describe dependence of model computations, not physical causal edges.
