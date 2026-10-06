# Thermal source-response interface

The opt-in `thermal_source_response_v1` family predicts a shared native
temperature grid from a learned geometry-dependent source kernel. A frozen
`D-sep` reader supplies u/v/p/omega. It does not execute the old thermal
wrapper, Stage-A, or a physical generator. Run3801 remains a protected
incumbent; the new family has independently measured accuracy.

## Inputs and physical roles

`SourceResponseOperator` lives in the shared core. The Thermal adapter owns
units, the affine-forcing capability, native grid interpolation, material
coordinates, surface/outside extraction, conductivity and port ratios.
Module geometry/materials, operating conditions and environment geometry
condition the coefficients. Current heating enters only `apply_forcing` or
`apply_increment`. Environmental tokens are context donors, not separately
actuated heat sources. Moving geometry rebuilds the prepared operator.

The direct and grouped modes use the same contextual encoder and near-branch
architectures and initial tensors, then train separate weights.
The direct far read evaluates each source separately; the grouped far read
uses signed receiver functions and normalized source memberships. Its exact
near correction subtracts the far response within the near support. Source
IDs survive preparation, interpolation and kernel export.

Fluid, surface, outside and material temperatures use the same learned grid.
The q-normal proxy uses the native outside-minus-surface rule. h-proxy and
h-effective ratios retain their native epsilon, clipping and validity rules;
they need not be affine. Material maxima follow field reconstruction.
Historical initial-port metrics are not applicable because there is no
P0/P1/P2 thermal trajectory in this family.

## Loading and replay

Use `channelthermal.source_response.load_source_response_model(path, device)`.
The historical loader rejects the new capability explicitly. Loading checks
the flow checkpoint digest, dependency capability, normalization and primary
development membership. The thermal checkpoint contains the new thermal
weights and the literal frozen flow-partner path/digest; it does not contain a
second complete thermal model.

The case model supports native samples and already saved response records:

```python
model, metadata = load_source_response_model(checkpoint, device)
prepared = model.prepare_record(saved_record, device=device)
prediction = model.apply_record(prepared)
increment = model.apply_record_increment(prepared, physical_delta_heat)
kernels = model.thermal.export_native_kernels(prepared["thermal"])
```

Prepared states belong to the exact geometry and receiver catalogue. Mutation
invalidates them. Autograd-bearing preparation belongs to its owning request
graph; it is not a persistent detached cache for geometry derivatives.
`apply_record_increment` returns the imposed zero flow increment, so test
independence with actual cold flow calls and derivatives as well.

## Paired training and continuation

`tools/thermal_source_response_fit.py --prepare-only` seals one common recipe,
including parent and flow digests, response-array digests, fixed25_v1
membership, common initialization, role budgets, gradient calibration and the
operator-qualification decision. Fit modes are `direct` and `group`; both
require `--recipe`, `--resume`, `--parent`, `--flow-checkpoint`,
`--operator-decision`, `--output`, `--device` and `--stop-after`.

The declared horizon is 2500 new thermal epochs. AdamW uses 3e-4 through epoch
1000 and cosine decay to 3e-6 at 2500. Every epoch visits the same 150 primary
TRAIN cases; 22 exposed DEV cases select the best three-temperature-role
monitoring checkpoint. Saved positive responses from original TRAIN
0001/0318/0333/0348 are an explicitly separate addendum. They provide one
solved heat direction per family. No DEV responses set coefficients or
construct training stencils.

The optional discrete balance is a training-only local stencil. Stored TRAIN
velocity is a supervision coefficient and never an encoder input. Operator
applications do not solve, invert, factorize or integrate the physical
problem. Their counts and neural receiver work are distinct from the physical
reference ledger.

Only declared 100-epoch milestones, latest and one best-field alias are kept.
Resume validates the complete sealed recipe and same arm. For a graceful
stop, use the existing stop-request interface; do not overwrite a checkpoint
or change a running recipe.

## Evaluation and interpretation

`tools/thermal_source_response_evaluate.py` measures all22 native fields or
saved `fit`, `development`, and `counted` response cohorts. The evaluator
separates physical finite-response errors from cold/prepared equality.
`tools/thermal_source_response_cost.py` measures complete cold native calls,
actual input VJPs, prepared multiple-forcing reuse and model-only bounded
compression. Formation, near work, storage and application costs remain
explicit. A reduced number of response modes is not an executor-speed claim.

`tools/thermal_source_response_gradients.py` records component norms and pooled
gradient cosines on the recipe's fixed TRAIN calibration cases and response
families. These are checkpoint probes, not measurements at every ordinary
optimizer boundary, and they do not adjust the frozen coefficients.
`tools/thermal_source_response_interventions.py` replays existing geometry
endpoints and records native kernel/heat/query/geometry derivative checks.
Its strict arithmetic comparisons use the retained tolerances and keep
failures separate from physical prediction errors.

Use the round report for exact selected checkpoint paths, actual fitted ages,
resource receipts and remaining qualification. Affine algebra does not prove
the learned slope is correct. Source kernels are estimates constrained by
available data and a qualified operator; memberships are representation-local
and do not establish physical causality. Saved-pool comparisons are not a new
inverse search or independently validated designs.
