# ThermalChannel dependency-separated flow reader

`channelthermal.dependency_flow` is an explicit opt-in composition. It calls
the selected Run3801 Add500 thermal model once and replaces only the four
flow channels with a new learned reader. The order remains
`u, v, p, omega, temperature`; material/interface/port predictions and Stage-A
remain inherited. Thermal parameters are frozen, and their input derivatives
remain live. This primary fit cannot repair Run3801's unresolved0291 thermal
response signs.

The case capability is
`channelthermal_analytic_wake_prescribed_flow_v1`. It applies to the audited
analytic-wake benchmark configuration, not arbitrary coupled multiphysics.
The shared `GeometryFlowField` has two source-context updates and nonlinear
relative-geometry source reads. It uses no case IDs, observed fields, predicted
ports, thermal hidden states or analytic generator. All active sources are
read densely; this is not a sparse thermal executor.

The ThermalChannel adapter whitelists module centers/mask/radius, Reynolds
number, inlet velocity, physical kinematic viscosity and domain lengths.
Material diffusivity/conductivity columns do not enter. `D-sep` constructs its
experimental heating slot from new fixed zero tensors without reading heat.
`D-open` supplies existing TRAIN-normalized own-module heat. Both arms use
identical parameter shapes and seed0 tensors; the encoder heat-column weights
start at zero. No heat-null penalty, gradient suppression or fallback is used.

## Fit and selection

Run `tools/thermal_dependency_flow_fit.py` in the ModularDT environment with
`--parent` set to the maintained selected Run3801 `epoch_0500_model.pt`,
`--policy D-sep` or `D-open`, and an ignored `--output` directory. Start each
arm with `--epochs 100`; inspect physical fields and dependency/geometry
checks before continuing with `--epochs 500 --resume <arm>/latest_model.pt`.
Optional1000 requires the authorized evidence and whole-round budget review.

The trainer binds the inherited fixed25_v1 manifest150 TRAIN/22 exposed DEV
and exact parent TRAIN normalization. A flow epoch visits all150 cases with
Q1024, microbatch8 and effective48: four AdamW updates,153600 query rows.
Only four normalized flow channels are optimized with equal-case,
equal-channel quadrature MSE. AdamW uses3e-4, weight decay1e-5, clipping1.0
and the same cosine schedule horizon1000. Each arm selects its minimum
all22 normalized flow score at saved100-epoch reviews. Exact endpoints remain
separate from that selector.

Flow labels are existing packed H5 observations, prepared in CPU memory for
training. They are never queried by model inference. Fixed receiver streams
reuse the maintained seed/case/epoch sampler formula. Validation uses the
same fixed1024 receiver selection at every review. Checkpoints retain only
declared100-epoch milestones, latest and best-field aliases, with atomic
writing, trusted loading and an explicit new dependency identity. Creating
`<arm>/stop_requested` uses the existing epoch-boundary save/acknowledgement
mechanism. Resume rejects changed parent digest, policy, reader dimensions,
membership, query budgets, seed or optimizer schedule.

## Loading and prepared inference

`load_dependency_model(path, device)` checks the explicit dependency identity,
case capability, exact parent digest and unchanged role order. It restores
the thermal parent through the maintained strict loader, then strictly loads
the new flow state. Candidate checkpoints reference the preserved parent
instead of duplicating a complete thermal wrapper.

`model.prepare_flow(structure)`, `read_flow(prepared, query_xy)` and
`forward_flow(structure, query_xy)` expose actual flow inputs and source
states for numerical audits. Ordinary `forward(..., return_prepared_state=True)`
returns both lane states; `decode_prepared` always combines their outputs.
`evaluate_heat_batch(structures, query_xy)` accepts only an explicit identical
layout/context request. D-sep reuses its prepared flow/query result within
that request, while every heat endpoint still calls the thermal predictor.
There is no durable cache or case-ID lookup.
Heat batches reject a shared `local_module_params` tensor because its own-heat
entry may be stale. Each thermal endpoint rebuilds local module parameters
from its current physical heat allocation and predicted ports.

The retained thermal execution tile is a separate runtime setting. Apply the
same qualified tile to the parent and candidate when pricing complete calls;
report cold calls separately from fixed-layout heat reuse. A flow epoch cost
does not estimate complete thermal training or formal execution.

`tools/thermal_flow_dependency_evaluation.py --checkpoint <selected-child.pt>
--output-dir <fresh-ignored-directory> --device cuda:0` measures ordinary
composed heat changes, actual flow inputs/states, heat/geometry/context/query
VJPs and AD/FD agreement on the declared native panel. Its all22Q1024 and
fixed-fourQ8192 checks are measured development evidence; they do not launch
a reference solver or perform training. Preserve each measured output
directory when repeating the checks at a different checkpoint.
