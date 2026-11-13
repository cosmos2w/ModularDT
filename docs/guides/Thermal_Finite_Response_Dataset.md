# Thermal finite-response dataset

`thermal_finite_responses_v1` stores a source-preserving response corpus alongside the existing `thermal_channel_global_v1` physical-field dataset. The H5 is a separate scientific dataset, not a replacement for native-state training. Configure `finite_responses` in the ignored `Case_ThermalChannel/Dataset/dataset_locations.local.json`; `Case_ThermalChannel/Dataset/dataset_locations.example.json` shows the binding name. The optional `Case_ThermalChannel/Dataset/links/finite_responses.h5` link is discoverable but the reader accepts any explicitly supplied path.

`tools/thermal_pack_finite_responses.py --help` lists the required source manifests, native H5, baseline panel, original attempt ledgers and storage limits. Inventory first with `--inventory-only`; `--convergence-only`, `--controls-only` and `--binding-only` provide read-only preflight checks. A build requires a fresh destination, a receipt path and `--audit`. It streams to a sibling temporary file, audits the closed file and publishes using a no-clobber atomic link. Preserve the source files and the resulting coverage receipt. Source paths stored in the H5 are provenance; endpoint reading does not open them.

## Layout and roles

- `/meta` records the native dataset identity, original channel names and units, coordinate and q-proxy conventions, source manifests and ledgers, schema, creation details and representation coverage. `/index` preserves ordered family/action identifiers, roles, native mapping and population eligibility.
- `/families` contains prescribed family metadata, action metadata, source-saved physical deltas when present, and separately named FP64 functional reductions and deltas. Saved values can come from an action manifest or an attributed historical physical evaluation; the original manifest metadata is not rewritten. `/endpoints` preserves the source configuration and solver configuration, original final-frame and grid arrays, full frame-index convergence evidence, and exact observation-record metadata and typed arrays when a matching saved record exists.
- `/catalogues` contains one copy of each mapped native case, including its saved sampled coordinates, repeated rows, weights, strata and historical baseline fields. `/populations` retains ordered TRAIN150/TRAIN600 native and FIT identities, attributed historical TRAIN-only normalizers and response scales where supplied, and marks new objective coefficients pending calibration. `/reference_baselines` binds the input-selected TRAIN15/VAL9 physical baselines to endpoint keys without adding fitting families.
- The direct role names are `fit`, `response_holdout`, `historical_new_withheld`, `native_seen` and `fresh_layout`. Native split and action eligibility are separate: a withheld action can belong to a native TRAIN layout. Fresh layouts have an explicitly absent native-case mapping when none exists. Restrict every training query to both `role='fit'` and the declared population.

An absent `observation_record` means no exact saved binding was found in the inspected ledgers. It does not mean that the raw endpoint or native case is absent, and a different physical solve is not a replacement. Endpoint source arrays retain their saved dtype; an FP64 reduction of an FP32 frame is not an FP64 observation. The q channel is the saved heat-flux proxy in dataset units, not an independently solved conservative flux. Fluid, material, physical outside-stencil and teacher-h masks remain distinct.

## Reading effective views

```python
from channelthermal.data.finite_responses import FiniteResponseStore

responses = FiniteResponseStore("Case_ThermalChannel/Dataset/links/finite_responses.h5")
for family in responses.family_ids(role="fit", population="TRAIN150", source="original"):
    for action in responses.actions(family, role="fit", population="TRAIN150"):
        baseline = responses.endpoint(family, role="fit", population="TRAIN150")
        edited = responses.endpoint(family, action, role="fit", population="TRAIN150")
        saved_delta = responses.physical_delta(family, action, role="fit", population="TRAIN150", kind="saved")
```

The default `r17_1_raw` view reproduces the maintained raw endpoint dictionary, including source-precision configuration and original FP32 values. `kind='derived'` requests the separately documented FP64 functional reduction; it does not replace the saved label. `physical_delta_provenance` identifies each saved label's source and extraction or the derived functional recipe. `observation_record` exposes exact saved record metadata and arrays and raises when unavailable. `reference_baseline(native_id)` accesses the saved TRAIN15/VAL9 source-precision baseline independently of source paths; `reference_provenance` and `reference_frame_index` expose its saved convergence evidence.

`legacy_view_components` explicitly identifies the R16-10 source-specific native-baseline/record-edit or raw/raw components. `legacy_pair` builds the case-owned batches using a native-state provider whose original case is checked against the packed catalogue. For an original FIT family, `stage='response_pair'` selects the earlier native-H5 baseline and record-backed edited values; `stage='r16_10'` selects the later R16-10 objective's raw-baseline rebuild and record-backed edit. Both keep outside-port observations from raw arrays. The new-family legacy path uses raw endpoint values at both stages. Missing records required by a legacy profile raise an error; readers never silently switch its target representation. The store opens independent read-only H5 handles for each operation and can be used by separate loader processes.
