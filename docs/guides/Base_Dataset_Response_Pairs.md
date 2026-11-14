# Response supervision from existing physical-field samples

A case with saved physical fields can supervise changes without running another simulator. Load two existing TRAIN scenes, predict each scene, and compare the predicted change with the change between their saved references. Store their case IDs and the comparison definition; do not copy their fields into a second dataset. Native field training remains available without any response provider.

For example, two saved scenes have pressure drops 1.0 and 1.4 in the case's pressure units. Predictions of 1.1 and 1.3 give a predicted change of 0.2 rather than the saved change of 0.4. The additional error is -0.2. The case must also say which inputs changed: this comparison could involve different layouts, heating or inlet conditions. It does not automatically measure the effect of moving one object while holding everything else fixed.

## Implementation status and default path

This guide defines the shared interface and scientific policy to implement in R17-3. The existing `honf_runtime.state_response` module already supplies generic difference reductions; the native-pair provider, batching and common recorder described here are implementation targets, not an available new command or configuration flag. Existing Thermal `state_response` recipes use the separate edited-scene provider documented in [state-response training](State_Response_Training.md). Preserve that provider and its checkpoints for historical reproduction.

The supported development order is native field loading and prediction, optional comparisons drawn from that same dataset, then any explicitly requested specialized intervention study. No extra edited-scene H5, physical solve, synthetic field label or Thermal-specific manifest may become a requirement of the shared native-pair interface. Existing extra-response data must not silently enter fitting, normalization, response scales or calibration in a base-only experiment.

## What the comparison learns

For an observable extracted by the case, write its prediction error in scene i as e_i = predicted value minus saved reference value. A normalized pair penalty is `(e_B - e_A)^2 / s^2`, where s is a positive scale obtained from TRAIN. Absolute supervision of the two scenes is still necessary: the same wrong offset in both predictions disappears from their difference.

This penalty uses existing information. Algebraically, `(e_B - e_A)^2 = e_A^2 + e_B^2 - 2 e_A e_B`; it changes which relative errors and correlations training penalizes, rather than supplying a new physical observation. For uniformly weighted comparisons of every pair, it reduces to a scaled variance of the existing residuals. A sparse, input-selected set is useful for controlling work and emphasizing available input changes, but cannot create a physical intervention absent from the base samples.

The adoption question has two parts: whether a base-only recipe provides useful field accuracy at acceptable cost, and whether adding its difference penalty improves change prediction over a matched absolute-field control. Compare separate fresh models with identical initialization, native case visits, auxiliary absolute labels, physical query positions, sampling and schedules. Only the difference contribution changes. A historical response-trained checkpoint is a practical reference, not that causal control; initializing from it would inherit the extra corpus's influence.

## Small case interface

Extend the existing native case adapter rather than introducing another dataset framework. The following are the required responsibilities; concrete class names should follow the maintained provider interfaces when implemented.

| Operation | Case supplies | Shared code supplies |
| --- | --- | --- |
| Describe an existing sample | Original case ID/split, input-only compatibility keys, physical coordinate frame, prescribed-input descriptor and original member IDs if present | Split-safe selection, deterministic pair scheduling and compact case-ID references |
| Prepare a comparison | Two existing native sample references; query positions and transformations; valid masks and physical measures; named observables and their units; any verified member correspondence | Loading through existing readers, bounded batching, prediction reuse and accounting for extra queries |
| Extract observations | Differentiable predicted observations and matching saved observations, using the same extraction, support and units | Normalized absolute/difference reductions, declared weights, TRAIN-only scale/calibration plumbing and ordinary metric records |

The adapter does not serialize the whole source dataset, own another optimizer, or implement its own logging tree. Shared code must not import Thermal, Wind or geothermal modules to calculate a loss. A minimal pair record contains the two original IDs, split, input-change description and reference to its observation recipe. Large masks, coordinates and fields stay in their original shared source, or in one explicitly budgeted reusable input artifact if computation genuinely requires it.

For a new dataset, reuse its native loader and add the input description and observation extraction. If it has no valid way to compare two fields at common positions, it can still compare compatible scalar observables. If it supplies neither, native training remains supported and response supervision is unavailable; do not invent correspondence or replace missing targets with zero.

## Positions, objects and physical measures

- A field difference requires predictions and references at the same physical positions in the same coordinate frame. Intersect the two physical validity masks and use the case's physical weights. Equal array shapes, sample indices or normalized coordinate values alone are insufficient.
- A temperature attached to a material point or a turbine-local measurement requires original identity and an explicit local-coordinate correspondence. An input-space assignment used to find similar layouts does not establish persistent object identity. Compare scene-level observables when such identity is absent.
- Scalar observables may use each scene's own support when that defines the same physical quantity, such as each scene's inlet-minus-outlet pressure drop. A regional temperature comparison uses one defined physical region and common valid positions in that region. Explain which operation applies.
- Keep interpolation, frame transformations and pressure reference conventions case-owned. Preserve source dtypes and coordinates; using FP64 for a reduction does not turn an FP32 saved field into a higher-precision observation.
- Fit normalizers, response scales and gradient balance from native TRAIN data only. State physical units, scale floors and zero-variation components separately. Validation and TEST pairs must contain only cases in their own declared panels; never pair across splits.

## Pair selection and execution

Choose pairs using prescribed inputs, without model errors, target differences, response signs or design outcomes. Useful choices include existing repeated layouts with different operating conditions, or similar layouts within compatible domain/material/control groups. Record all inputs that actually differ. Exclude identical-input duplicates from informative response edges while retaining their ordinary native supervision.

Use a sparse pair list or reproducible sampler, not a materialized all-pairs field dataset. Keep every declared native case visit; an unpairable sample still contributes its native loss. A small rotating set of pairs can receive additional observation queries while the rest retain the ordinary native sampling. Both comparison arms execute the same scene-preparation and prediction operations at the same inputs/positions and receive the same absolute targets; only the added difference calculation changes. Their prediction values can differ as their weights learn. Co-locate paired samples when practical, reuse scene preparation and existing predictions, and charge any second decode explicitly. Do not retain an entire epoch's autograd graphs merely to pair distant batches.

Store the sampling method, seed, original membership reference and required sampler state. Per-epoch SHA summaries cannot restore sampling and are not a default substitute for that state. Record actual case/pair counts and useful executed-work totals. Use the [shared artifact contract](Run_Artifacts_and_Monitoring.md); no case-specific logging framework is needed.

## Thermal and other cases

Thermal's native dataset includes saved full physical fields, material temperatures and port observations. Existing helpers calculate the material-temperature smooth maximum J and inlet-minus-outlet pressure drop P in `channelthermal.native_functionals`. The regional helper in `thermal_native_port_task` is a historical example defined from its supplied baseline layout; a native-pair adapter must explicitly choose the region-defining geometry rather than assume symmetry or generic suitability. Each scene uses its own material and pressure-band validity for J/P; regional differences use one stated region and identical physical coordinates valid in both scenes. Arbitrary native layouts must not be forced into the historical baseline-plus-four-edits family class or subtracted by module slot.

The inspected Thermal base TRAIN data contains repeated-layout comparisons whose Reynolds number changes, and independently generated layouts whose heating also changes. These are different kinds of supervision. Neither provides a general fixed-heat, layout-only edit dataset. The analytic-wake benchmark's particular dependencies do not authorize a generic heat-independent flow model or a specialized prediction branch.

Wind has repeated layouts under different wind directions, but its saved coordinates and velocity components may be wind-aligned. A case must define whether observations refer to a wind-relative or Earth-fixed position and transform positions and vectors accordingly. Preserve source turbine slots; do not invent persistent identity metadata. Geothermal cases can use the same interface for pressure/temperature at declared physical positions or well observables; their meshes, well identities and integration weights belong in their adapter. Supporting this interface does not itself establish response accuracy in either application.

R17-3 should demonstrate the interface with Thermal and a small Wind no-training example, plus a non-Thermal ragged/coordinate fixture. Additional case-specific training or new simulations require their own scoped task.

## Verification and decisions

Check direct native labels against adapter-extracted labels, source precision, masks/weights, role separation, deterministic pair schedules and predictions reused at aligned positions. A prediction equal to the saved reference must give zero absolute and difference loss; a common erroneous offset must remain visible to the absolute loss. Check differentiability and reject invalid alignment. Use a small interrupted/resumed example to verify actual model, optimizer, RNG and sampler state.

Report native accuracy, change errors in physical units, normalized aggregates, near-zero response handling and measured complete cost separately. Count contributing cases/layouts as well as correlated pairs. Apply approximately 10% matched model-evaluation coverage by default. If pair supervision ties or loses, retain the simpler base-only recipe; that does not make an extra physical-response corpus a mandatory fallback. A controlled-edit claim or verified inverse design requires the corresponding evidence and cannot be inferred from smaller field error.
