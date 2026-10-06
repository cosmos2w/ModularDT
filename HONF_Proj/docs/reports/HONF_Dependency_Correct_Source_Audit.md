# ThermalChannel dependency and saved-response audit

The source audit confirms that the active benchmark flow labels depend on
geometry and prescribed flow context, while heating drives the shared-grid
thermal solution. It also confirms that effective ports are derived thermal
quantities. The audit changes the physical/data description; it changes no
labels or thermal weights. The matched flow-reader fit and complete cost
measurements are reported separately in the dependency-correct forward report.

**Predictor:** Run3801 Add500 remains the retained thermal predictor. Its
0291 mean fluid-temperature response signs remain wrong: reference minus/plus
are -0.02891885/+0.02891378; the retained Add500 predicts
+0.01042027/-0.01550809. Flow separation does not correct these errors.

**Organizer:** the inherited thermal organization remains learned and
heat-conditioned. A known zero heat-to-flow dependency can be encoded by an
input signature. This audit establishes no new unique organizer advantage.

**Inverse:** the source supports heat-independent flow for this benchmark,
and approximately affine fixed-layout point temperatures in the saved
references. The response evidence excites only one balanced heat direction
per layout. Wrong thermal signs and unexcited directions remain limits on
broader inverse use. No inverse run/search or physical solve was executed.

## Actual generator and packed configurations

The active reference adapter resolves its generator in
`Case_ThermalChannel/src/channelthermal/interaction_evidence/reference_adapter.py:344`
to the locally present
`../1_Demo_ChannelThermal/src/simulate_channelthermal.py`. The inspected source
is an analytic wake approximation plus a shared thermal grid, not coupled
Navier–Stokes CFD. All 690 embedded packed case configurations were read
directly, without invoking the generator. Every case uses `analytic_wake`,
projection disabled, `nu=null`, `viscosity_scale=1`, `u_in=1`, zero inlet/wall
temperatures, `solid_alpha=0.01`, `fluid_alpha=0.02`, and
`solid_k=fluid_k=1`. Seventeen prescribed Reynolds numbers occur. Temperature
stopping is enabled, with absolute/relative tolerances 1e-4/1e-5, a 10-frame
window, and maximum solve time 200. The global dataset and atlas configurations
identify the same generator variant.

In this table, **S** means source-confirmed dependence, **0S** means source
confirms absence, and **M** means also verified by saved heat-only arrays.
Query geometry means physical sampling of a generated state, rather than
changing the grid solve.

| Output role | Layout/domain geometry | Prescribed context | Material coefficients | Heating | Predicted neural ports | Receiver geometry |
|---|---|---|---|---|---|---|
| u/v | S | S: inlet; Re irrelevant without projection | 0S thermal coefficients | 0S/M | 0S | S |
| p | S | S: inlet, Re-derived viscosity | 0S thermal coefficients | 0S/M | 0S | S |
| omega | S: velocity gradients/grid | S: inlet | 0S thermal coefficients | 0S/M | 0S | S |
| Fluid/shared-grid T | S | S: prescribed advection, inlet/wall T | S: fluid/solid alpha | S/M | 0S generator; S neural predictor | S |
| Surface/material T | S | S | S: thermal alpha | S/M | 0S generator; S neural predictor | S |
| Outside T | S | S | S: thermal alpha | S | 0S generator; S neural predictor | S: radius+sampling offset |
| q_normal proxy | S | S | S: alpha through T, harmonic k during extraction | S/M | 0S generator; S neural predictor | S |
| u_normal/u_tangent | S | S | 0S thermal coefficients | 0S | 0S | S: angle and exterior coordinates |
| h_proxy/h_effective | S | S | S: T and extraction k | S via ratio/epsilon/clipping | 0S generator; S neural predictor | S |

Velocity is parabolic with obstacle deficits and transverse perturbations,
then no-slip masks and wall taper are applied. Pressure combines the prescribed
viscous drop and obstacle perturbations; subtracting its outlet mean fixes
the gauge. Vorticity is computed from the resulting velocity and zeroed in
solid cells (`build_channel_flow`, generator lines172–221). The optional
projection (`project_flow_field`, lines251–294) reads no heat or temperature;
all 690 inspected cases leave it disabled. Viscosity is
`viscosity_scale*u_in*(2*radius)/Re` unless `flow.nu` overrides it
(`_helpers_forward/channelthermal_common.py:432`). Thus admissible new learned
flow inputs are centers/presence, radius, domain lengths, physical receiver
coordinates, prescribed inlet/Re and viscosity. Solid conductivity,
diffusivity, heating, outside/surface temperature, effective ports and local
response states are excluded from the independent flow reader.

Thermal diffusion is a centered finite-volume linear flux operator with fixed
alpha; upwind advection uses fixed prescribed u/v. Heat is deposited as the
module heat value at its solid cells. The domain boundary operator is fixed
Dirichlet at inlet/walls and zero-gradient at outlet
(`enforce_temperature_boundaries`:330,
`diffusion_rhs`:340, `advection_rhs`:358, `heat_source_field`:396,
`step_temperature`:405, coefficient assembly:612). No buoyancy, radiation or
temperature-dependent coefficients enter these inspected routines. At a
common integration step count, the temperature update is affine in heat.
The input-dependent convergence checks (`check_converged`:464) and FP32
storage (`save_frame`:582) qualify an exact steady/saved-state claim.

`extract_interface_response`:506 samples boundary and exterior temperatures
from that shared grid, not separate fluid/solid interface solves. Its proxy
flux is `-k_interface*(T_outside-T_surface)/delta`, with harmonic k.
`h_proxy=abs(q_normal)/(abs(T_surface-T_outside)+1e-6)` is a thermal ratio.
`preprocess_channelthermal_dataset.py:607` derives h_effective from
`q_normal/(T_surface-T_outside)` using a signed denominator epsilon, finite
checks, clipping and a validity mask. Without these safeguards the ideal
ratio cancels the temperature jump in this generator; with them the stored
quantity can still vary with heating. It is not an independent prescribed
heat-transfer coefficient. No port-independence claim is made.

## Neural dependency inventory and canonical APIs

Physical heat is normalized with the parent TRAIN-only mean/std, then enters
the four own-source heat features in
`input_adapter.py:162–182`. `source_local_v3` sets four global heat statistics
to fixed zeros (`input_adapter.py:217–222`), but retains own-source heat.
The module encoder therefore remains heat-sensitive. Its tokens drive P0,
fine/coarse/local interaction values, global planning/calibration/reference
means and continuous initial port contexts. The port head also reads source
heat explicitly (`local_coupling.py:136`). Stage-A receives physical heat and
predicted outside T/h summaries, produces internal/interface responses, and
fuses them into module states. P1 probes exterior temperatures and refines
T/h; a second Stage-A evaluation supplies P2 fused states and the final
five-channel field decoder (`interface_field_coupling.py:705–929`). This
explains the old thermal-to-flow leakage even with zero global heat slots.
Frozen parameter gradients do not remove any of these input sensitivities.

On TRAIN0001/M3 and0348/M10, nominal H5, typed-response and prepared-query
interfaces were evaluated at identical physical inputs and all 8192 original
fluid receivers. Both use the exact selected Run3801 `epoch_0500_model.pt`
resolved from the maintained prior selected-counted mapping. At common packed
M12 padding and P64 ports, the following actual tensors agree exactly:
normalized source heat, centers, material/context tensors, module features,
global context, local receiver coordinates, angle/normal features, and
1036 receiver-reference coordinates/weights/roles. Nominal versus typed
physical five-field outputs have zero maximum difference; prepared versus
full normalized outputs also have zero difference. Source module slots and
typed receiver joins retain their existing identities. Observed nominal
teacher values are withheld by predicted-port mode; typed placeholders
contain no observed values. No target was supplied as a predictor input.

The initial diagnostic comparison failed after 8 complete calls because the
ordinary nominal collator cropped M3 active slots while the typed adapter
retained M12 capacity. The comparison was corrected to common fixed padding,
without changing historical readers. The successful run made 16 complete
calls and 2 prepared decodes on CPU in 21.61s. The cumulative 24-call diagnosis
ceiling includes the failed attempt. Both compared APIs have P64 and equal
input-only reference catalogues under matched settings; this does not prove
that every historical API configuration used the same catalogue. No source
repair was justified, and this result does not explain0291's wrong signs.

## Saved linearity and actual excitation rank

Central closure is measured as `T_plus+T_minus-2*T_base`, after widening saved
FP32 arrays to FP64 and using the original valid receiver masks. All statistics
use unclipped arrays. The thermal parent is the selected Add500 checkpoint.

| Family scope | Case | M | Solved balanced rank / M-1 | Reference fluid-T closure RMS | Add500 closure RMS |
|---|---:|---:|---:|---:|---:|
| FIT | 0001 | 3 | 1/2 | 3.10410e-7 | 0.0573183 |
| FIT | 0318 | 5 | 1/4 | 3.16324e-6 | 0.0996146 |
| FIT | 0333 | 7 | 1/6 | 1.57809e-5 | 0.0257689 |
| FIT | 0348 | 10 | 1/9 | 6.61454e-7 | 0.0257996 |
| DEV | 0304 | 3 | 1/2 | 3.12821e-6 | 0.119986 |
| DEV | 0320 | 5 | 1/4 | 3.57555e-7 | 0.0520030 |
| DEV | 0335 | 7 | 1/6 | 4.36435e-7 | 0.0311534 |
| DEV | 0350 | 10 | 1/9 | 1.23726e-5 | 0.130814 |
| Fixed audit | 0291 | 5 | 1/4 | 1.26082e-5 | 0.0178700 |
| Fixed audit | 0294 | 7 | 1/6 | 4.42661e-7 | 0.0289958 |
| Fixed audit | 0687 | 10 | 1/9 | 7.39003e-7 | 0.000184389 |

Temperature units are dataset units. Closure for surface and material
temperatures, full increment matrices and singular values are retained in
`audit/saved_linearity_rank.json`. Rank uses 1e-6 absolute tolerance to exclude
input-quantization noise; opposite transfer endpoints supply one direction,
not two. There are two, four, six and nine free fixed-total heating directions
for M3/M5/M7/M10. Thousands of field receivers or module maxima do not add
input-direction rank. The saved arrays constrain only their solved directional
responses at each layout. Shared geometry-conditioned learning can transfer
information between layouts; this evidence cannot uniquely identify the full
source-response operator for each layout. Maximum-temperature selection may
be nonlinear even when point temperatures are affine.

The inherited thermal training used 150 fixed25_v1 primary TRAIN cases plus
three auxiliary original-TRAIN response anchors (0001/0318/0333);0348 is
already primary TRAIN. DEV0304/0320/0335/0350 and fixed-audit cases did not
enter those refinement gradients. This historical thermal exposure is
separate from the new 150-case flow-only fit. All reported validation and
response panels are exposed development evidence, not independent tests.

0277 has no valid baseline, so it remains secondary-only and has no central
closure claim. Its two saved endpoints still span exactly one balanced
direction out of two available for M3; this is endpoint-span evidence rather
than a baseline-relative response. Its actual increment is retained separately
as `secondary_0277_rank` in the saved JSON. Saved final-iterate convergence warnings and storage noise are
descriptive qualifications, not certified grid-error or sign-error bounds.

## Bounded port localization

The unchanged Run3801 was called only on existing TRAIN response families0001
and0348. Normal calls capture initial/refined T/h, local responses and final
fields. Interventions hold either initial predicted ports or refined predicted
ports at their normal baseline values, then recompute the remaining path.
These are model-side localization experiments, not solved physical
counterfactuals. No validation labels, new surrogate or training are involved.

| TRAIN family/direction | Physical mean fluid-T response | Normal Add500 | Hold initial T/h | Hold refined T/h |
|---|---:|---:|---:|---:|
| 0001 minus | +0.00966444 | +0.0266857 | +0.0232091 | +0.0397762 |
| 0001 plus | -0.00966444 | -0.000893514 | +0.000838855 | -0.00225274 |
| 0348 minus | +0.0111616 | -0.000156043 | -0.000968954 | +0.00297923 |
| 0348 plus | -0.0111616 | +0.00494613 | +0.00649742 | +0.00459158 |

Holding initial ports does not repair either wrong0348 direction. Holding
refined ports changes the minus sign, but leaves plus wrong and does not
restore the measured magnitude. Direct source features, local heat and fused
response paths remain live during these interventions, so this is evidence
that port feedback contributes to the problem without identifying it as the
sole cause. The baseline refined h means are 15.25298/15.31557 for0001/0348,
versus initial means 3.00296/2.98432. These actual intermediate values, complete
local outputs and all full-grid predictions are saved in
`audit/native_port_audit_arrays.npz`. Exact parent weights and buffers were
compared before/after and are unchanged. The 24-call ceiling is exhausted;
no further localization calls were made.

## Selected inspected figures

The short figure index is: **0291 full responses** (both directions, native
fluid field/residual), and **closure/rank/ports** (saved numerical diagnostics
and bounded TRAIN intervention). PDF masters and small raster companions are
kept in the local ignored artifact root. Both raster companions were visually
inspected; all 8192 query coordinates were joined to their exact128x64 grid
positions before rendering. Solid receivers are blank and statistics use
unclipped valid-fluid arrays.

![Both0291 full-grid thermal responses and residuals preserve the unresolved wrong mean signs](../../diagnostics/generated/dependency_correct_20261006/audit/thermal_0291_full_response.png)

Reference minus/plus means are -0.02891885/+0.02891378 versus retained
Add500 +0.01042027/-0.01550809; response residual RMS is 0.066168/0.079100.
This is existing counted fixed-audit evidence from the analytic-wake/shared-grid
generator, not an independent test or new solver result. Freezing thermal
weights retains these errors.

![Saved central closure and rank with TRAIN0348 baseline-port interventions](../../diagnostics/generated/dependency_correct_20261006/audit/thermal_linearity_rank_ports.png)

Reference fluid-T closure RMS is 3.10e-7–1.58e-5 across 11 complete families;
Add500 is 1.84e-4–0.130814. Every layout has solved rank 1. The0348 intervention
changes one response sign while the other stays wrong. These measurements
support a geometry-conditioned, source-resolved thermal-response modeling
question, not a claim that the ports alone explain the defect.

## Unexecuted next question and resource disposition

The next thermal-model question is whether a whole-layout-conditioned affine
point-temperature response representation can preserve the strong field
predictor while fitting independently excited source responses, with derived
effective ports and nonlinear maxima handled explicitly. No such model or
training portfolio was introduced in this round.

One economical future reference request would reuse the four FIT baselines
and their existing balanced direction, adding M-2 independent one-sided
balanced directions at each layout. This costs exactly 1+3+5+8=17 new attempts;
a fresh complete baseline-plus-basis request would cost
(1+2)+(1+4)+(1+6)+(1+9)=25. Using the measured existing symmetric heat-endpoint
times for 0001/0318/0333/0348 (means 2.74347/3.95684/5.02755/7.05044s), the 17
additional directions project to approximately 96.2 CPU seconds of generator
execution, excluding setup/storage and differences in convergence. Adding a
common-mode direction would require four more attempts when total-heat
variation is authorized. These are unexecuted estimates, not a launch request
or permission. The allowance remains326/326, with zero new attempts.

**A/B/C:** A remains the previously weak unique organizer evidence; this audit
does not improve it. B is now source-backed: heat drives thermal tokens,
ports, Stage-A and final shared thermal decoding, while active physical flow
has no heat input. C remains exposed development response evidence with one
direction per layout and explicit0291 failures. No new formal/Wind run,
inverse design/search, physical solve, restart or optimization update occurred
in this audit. The matched flow experiment has its own measured conclusions.

All one-time scripts, arrays, logs and generated figures are local under
`/data/wanglz/ModularDT/thermal_development/dependency_correct_20261006/audit`.
The figure links use the existing ignored-local-artifact convention; restoring
that root is required to render them in a fresh clone. Only this report and
the physical/data clarification are durable audit changes.
