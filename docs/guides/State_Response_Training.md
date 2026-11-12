# State and finite-response training

State-response training augments the native physical-field objective with supervised edited endpoints and finite responses. A response is the difference between complete predictions at a baseline and an edited feasible design. The case owns both design construction and physical observations; the model is prepared again after an edit so geometry, records, organization and query coverage correspond to that endpoint. This is joint predictor training, without a separate response-only prediction head.

## Objective and defaults

The new `state_response` objective is enabled by default and is adjustable. Set `enabled` to `false` for a native-only baseline. When enabled, the selected case provider must have the paired labels it needs; it must fail explicitly when they are unavailable. Existing saved recipes retain their own objectives, schedules and coefficients, and their checkpoints use the original loading and continuation semantics.

The generic configuration is `honf_runtime.state_response.ResponseObjectiveConfig`. Its default endpoint and response outer weights are `1.0`, the field-response weight is `1.0`, and the default named functional-weight mapping is empty. The default warmup is 10 epochs followed by a 40-epoch linear ramp. The default `gradient_norm` balance targets endpoint and response component gradient norms of `0.25` and `0.10` times the native gradient norm, respectively. These ratios and outer weights are configurable. The `fixed` mode normally uses the configured outer weights directly; an explicit calibration file supplies component coefficients that are multiplied by those weights. A case supplies any named physical functionals and their relative weights.

A generic recipe fragment can set the objective as follows; it must be combined with a compatible case provider and its required data and functional settings:

```json
{
  "training_profile": "state_response",
  "state_response": {
    "enabled": true,
    "endpoint_weight": 1.0,
    "response_weight": 1.0,
    "field_response_weight": 1.0,
    "warmup_epochs": 10,
    "ramp_epochs": 40,
    "coefficient_mode": "gradient_norm",
    "gradient_norm_ratios": {"endpoint": 0.25, "response": 0.10}
  }
}
```

The ramp multiplies both auxiliary contributions: it is zero through epoch 10 and reaches one at epoch 50 with the defaults. In `gradient_norm` mode, the case obtains the native and auxiliary gradient norms from a declared TRAIN calibration sample. For each component, the effective outer coefficient is its outer weight times the target ratio times the native norm divided by the component norm. Active components with positive target ratios require positive measured norms; missing or invalid calibration is an error. Scales, balance, ramp and family exposure are training settings and must never be fitted from validation or TEST labels. Save them with the recipe and resumable run state.

## Adjusting the recipe

Start with the recorded native objective and TRAIN-fitted response scales. The gradient-norm ratios set the initial relative influence of endpoint and response learning; the outer weights multiply that calibration and the epoch ramp. Set `response_weight=0` to isolate endpoint supervision, or `enabled=false` for the native-only baseline. If native fidelity deteriorates, reduce the responsible outer weight or target ratio in a new recipe; increase it only when matched withheld response and decision results justify the tradeoff. Scale floors represent the case's physical units and label resolution, while engineering constraint margins are separate request inputs. Do not reinterpret a numerical-resolution threshold as an engineering tolerance.

Changing observations, masks, action coverage or functional definitions requires revisiting TRAIN-only scales and calibration. Preserve these choices in a new run identity rather than silently changing a resumed objective. A collaborator can add a functional through the case's observation/extraction code and `functional_weights`; the shared loss math does not need a Thermal-specific branch. The default profile includes endpoint and finite-response fitting, without decision-ranking loss; the shared decision helpers are optional building blocks.

## Case integration

The shared module provides `normalized_response_mse` for an aligned field response on a nonempty common mask, `normalized_delta_mse` for case-supplied functionals, and equal, weighted or equal-family reductions. `validate_aligned_receiver_ids` verifies the already-aligned original receiver IDs; the case must perform any reordering and check coordinates and physical validity. The field loss compares the predicted endpoint difference with the observed difference using a positive TRAIN-fitted scale. Each absolute endpoint is compared on its own physical mask, while the finite difference uses common support.

A case adapter supplies the source dataset and TRAIN-family selection, native observations and units, original IDs, masks and geometry rules, endpoint target extraction, response-scale fitting, sampling and family exposure. It also defines any scalar functionals and their units. Its provider must report these settings in the saved training identity and restore them on continuation. Generic response learning does not assume Thermal ports, angular resolution, a frozen local operator, or named temperature and pressure functionals.

| Shared implementation | Case-owned implementation |
| --- | --- |
| Core joint-field computation, prescribed record features, and scoped full/restricted reads | Mapping physical geometry, materials and controls into those records; native units and masks |
| Endpoint/difference reductions, coefficient calibration, auxiliary updates, history and checkpoint state | Paired TRAIN corpus, receiver alignment, scale fitting, native observation extraction and family exposure |
| Candidate-pool feasibility, ranking, violations and regret accounting | Named physical functionals, engineering caps, sensitivities, geometry validity and independent physical verification |

The native training entry currently wires the enabled profile to Thermal. Other cases need their own paired-data provider; an unsupported case with the objective enabled raises an error. The case-neutral helpers can be reused independently, and disabling the objective lets supported native-only case workflows continue.

## Thermal provider

`channelthermal.training.thermal_joint_response_task.build_thermal_joint_response_config` constructs a new `state_response` recipe from a compatible native Thermal base recipe. Its case-specific `thermal_joint_response` block owns the FIT manifest, scale source, family sampling and native port options. The default Thermal response mixture is `field_response_weight=0.8`, with `functional_weights={"J": 0.1, "P": 0.1}`. J is a normalized smooth maximum over valid material temperatures; P is the difference between mean pressures in the leftmost and rightmost 8% of the domain. The default smooth-maximum parameter is `tau=0.4149511814117432` in the dataset's temperature units.

After warmup, the default is four FIT families per active epoch, in a reproducible order set by `family_seed=0`. Set `families_per_epoch` no larger than the declared FIT set; the example manifest contains one family and must be expanded or paired with a smaller count. Native ports default to 64 angles and an outside-temperature loss weight of `0.05`; `angle_count` must be a positive divisor of the saved 64-angle catalogue. This provider supervises native observations directly and has no frozen local operator. The separate [Thermal port/local guide](Thermal_Port_Local_Integration.md) documents the existing P64 composition and its external frozen checkpoint.

The portable `thermal_response_families_v1` manifest contains `cases` and `actions` arrays. Every case declares a unique `case_id` belonging to the native TRAIN selection, `role: "fit"`, its complete physical `source_config`, and the baseline `source_case_dir`. Every action declares the same `case_id`, a unique `action`, `role: "fit"`, its ordered edited `centers`, `raw_case_dir` and finite `physical_delta` values for both `J` and `P`. Paths inside the manifest may be relative to the manifest directory. Optional `actual_solver_config` and `numerical_policy` travel together when describing a recorded solver endpoint. The provider verifies original TRAIN controls and physical endpoint provenance; no validation or TEST family may enter this manifest.

For an enabled objective, supply the FIT manifest and exactly one scale source to the builder: `scale_floors` to fit response RMS scales from the FIT families, `response_scales` for explicit TRAIN-derived scales, or `calibration_path` for a TRAIN-derived file containing `response_scales`, `coefficients`, `native_case_ids` and `fit_family_ids`. Each scale map covers `flow/u`, `flow/v`, `flow/p`, `flow/omega`, `temperature/fluid`, `temperature/surface`, `temperature/material`, `q_proxy`, `J` and `P`; each value or floor must be positive. Automatic gradient balancing uses the first selected TRAIN native update and FIT family and saves the coefficients in provider state. The supplied native case IDs and FIT family IDs must be nonempty, unique members of the selected TRAIN membership and FIT manifest, respectively; calibration must have TRAIN-only provenance.

```python
from channelthermal.training.thermal_joint_response_task import build_thermal_joint_response_config

# Supply a compatible native recipe and a complete positive TRAIN-only scale map.
recipe = build_thermal_joint_response_config(
    base_recipe,
    fit_manifest="path/to/train_fit_families.json",
    scale_floors=training_scale_floors,
)
```

The example variables and paths stand for local resources. Supply a compatible base recipe and replace them before serializing and passing the result to `tools/relation_anchored_train.py`. `build_thermal_joint_response_config(base_recipe, enabled=False)` selects the native-only baseline and needs no response files. Existing recipes lacking `training_profile` retain their historical objective.

For a compatible Thermal dataset with a different population, set `training.membership` to `catalog_train`, supply `dataset.data_path`, and omit `dataset.manifest_path`. The dataset catalogue supplies all original `train` IDs by default; an optional `training.train_case_ids` selects a unique subset of that split. `evaluation.validation_split` names a distinct catalogue split and defaults to `validation`. Explicitly expose a source split marked `test` only when authorized. This path reuses the dataset's existing catalogue identity and does not require the fixed25 manifest or 600 TRAIN cases. Set `storage.formal_forward_run_alias` to `false` for this membership; the formal alias is reserved for the established `full_train600` lineage. The current Thermal adapter still requires its native 128-by-64 field grid and saved 64-angle physical port catalogue; it is not a general arbitrary-grid reader.

The new Thermal profile defaults to the first approximately 10% of its declared validation cases, selected without using model errors. Set `evaluation.sample_seed` for reproducible random selection or provide `evaluation.validation_case_ids` for a fixed panel; keep the same IDs across compared models. Apply the same approximately 10% default to TRAIN, TEST and diagnostic evaluations. A full-population evaluation requires an explicit request.

The [generic defaults](../../configs/state_response_defaults.json), [Thermal recipe example](../../configs/examples/thermal_relation_anchored_state_response.json) and [family-manifest template](../../configs/examples/thermal_response_families.example.json) show the corresponding saved configuration. Copy and adapt the paths, complete source configurations, native TRAIN case IDs and actual physical endpoint labels before use; the template's null J/P values are intentionally unfilled and cannot be used for fitting. The example's `1e-6` scale floors are illustrative in the corresponding native units; they are not claims about measurement precision or constraint margins. No paired labels or trained checkpoints are distributed with the repository.

## Execution and storage

Use `tools/relation_anchored_train.py` for profile inspection, fitting, exact continuation and native evaluation. A fresh fit retains the resolved recipe in `configs/resolved_config.json`; continuation uses that same recipe and run directory. The older `honf_research.py resume` constructors have different identity contracts and reject these profile runs. Loss histories retain every recorded epoch, curves refresh every 10 epochs, and a configured wall-clock budget requests a clean save before its reserve is consumed.

Declare `storage.output_root`, `max_output_bytes`, `max_peak_temporary_bytes` and `minimum_free_headroom_bytes` before fitting. The runtime checks the mounted DATA2 filesystem and workspace bindings, existing root usage, retained-checkpoint estimates, atomic-save temporary space and headroom, and records the result under the run's `logs/`. The output-byte limit covers the configured root, including retained runs. The optional Thermal formal alias applies only to `full_train600`; other case/population recipes set `formal_forward_run_alias=false`.

## Interpretation

Compare native field fidelity, endpoint accuracy, finite-response magnitudes and resolved signs/orderings separately on matched panels. On the fitted families, the R16-10 response-trained Thermal model improved J and P response errors, while validation field metrics and some new-family results worsened. That measured tradeoff motivates an adjustable profile, not a universal accuracy or inverse-design guarantee. Evaluate edited designs against an independent physical reference and report work and runtime measurements separately. See [physical response evaluation](Case_Response_Evaluation.md) for masks, alignment and evidence requirements.
