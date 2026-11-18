# Local multisecant response training

The optional Thermal profile predicts the ordinary physical fields for several existing TRAIN scenes at common fluid positions. It compares those predictions with their saved references, first as absolute fields and optionally as differences between scenes. A geometry-aligned input chart determines the bounded relative emphasis of the available scene contrasts; it does not create a physical edit or a separate response head. The uniform response and state-anchor controls use the same scenes, physical positions and absolute labels. Native field training continues to work without this option.

`honf_runtime.local_response_contrasts` owns the case-neutral contrast and reduction. `channelthermal.training.thermal_local_response` owns Thermal compatibility, complete prescribed-input charts, shared observations and the adapter. The input chart uses original scene values and a geometry-only assignment; the assignment does not establish persistent material identity. The observation measure reuses saved receiver lists, including a field-gradient-selected stratum, and is not uniform physical area. Broad nearest-input neighborhoods describe finite changes, not verified derivatives or fixed-control interventions.

## Prepare a shared definition and calibration

Use a finished literal native-only `epoch_0010_model.pt`, a verified output root, and the scientific Python environment. The commands below show one TRAIN population; use the distinct TRAIN600 native prefix and new TRAIN600 definitions when preparing a formal child. Set `SOURCE` to the intended existing prefix and `ROUND_ROOT` to authorized storage before execution. No command here performs a physical solve or makes another field dataset.

```bash
PYTHON=python
SOURCE=/path/to/native/run/checkpoints/epoch_0010_model.pt
ROUND_ROOT=/path/to/authorized/round/storage
DEFINITIONS="$ROUND_ROOT/train150_input_definitions.json"
CALIBRATION="$ROUND_ROOT/train150_calibration.json"

"$PYTHON" tools/thermal_local_response_prepare.py definitions --source "$SOURCE" --output "$DEFINITIONS"
"$PYTHON" tools/thermal_local_response_prepare.py recipe --source "$SOURCE" --input-definitions "$DEFINITIONS" --variant state_anchors --name Neighborhood_State_Anchors_R17_7 --output-root "$ROUND_ROOT" --output "$ROUND_ROOT/anchors_recipe.json"
"$PYTHON" tools/thermal_local_response_prepare.py recipe --source "$SOURCE" --input-definitions "$DEFINITIONS" --variant uniform_response --name Uniform_Neighborhood_Response_R17_7 --output-root "$ROUND_ROOT" --output "$ROUND_ROOT/uniform_recipe.json"
"$PYTHON" tools/thermal_local_response_prepare.py recipe --source "$SOURCE" --input-definitions "$DEFINITIONS" --variant multisecant_response --name Local_Multisecant_Response_R17_7 --output-root "$ROUND_ROOT" --output "$ROUND_ROOT/multisecant_recipe.json"
"$PYTHON" tools/thermal_local_response_prepare.py calibrate --source "$SOURCE" --recipe "$ROUND_ROOT/anchors_recipe.json" --device cuda:0 --output "$CALIBRATION"
```

Choose the CUDA device at runtime after checking its ownership; `cuda:0` is an example, not a scientific requirement. The calibration measures native, absolute, uniform and conditioned gradients from the same source state and selected TRAIN neighborhoods. One coefficient targets an initial absolute gradient norm of 0.25 of the native norm; another targets the uniform response norm at 0.10 of native. Both response variants use that same response coefficient. The selected neighborhoods, source checkpoint, input definition, scales and objective are checked when importing the record. Calibration is a balance measurement, not evidence of model improvement.

## Initialize, resume, and evaluate

Initialize each new child from that same checkpoint and calibration, substituting the corresponding recipe and run name. The initializer checks that native data, predictor, loss, sampling and optimizer schedule are unchanged and inherits the committed first ten epochs. The child's future auxiliary columns start blank; old runs retain their saved definitions.

```bash
RUN="$ROUND_ROOT/Uniform_Neighborhood_Response_R17_7"
RECIPE="$ROUND_ROOT/uniform_recipe.json"
"$PYTHON" tools/field_response_branch.py --recipe "$RECIPE" --source "$SOURCE" --calibration "$CALIBRATION" --output "$RUN"
"$PYTHON" tools/relation_anchored_train.py fit --recipe "$RUN/configs/resolved_config.json" --output "$RUN" --resume latest --stop-after 100 --device cuda:0
"$PYTHON" tools/relation_anchored_train.py fit --recipe "$RUN/configs/resolved_config.json" --output "$RUN" --resume latest --stop-after 500 --device cuda:0
"$PYTHON" tools/relation_anchored_train.py evaluate --recipe "$RUN/configs/resolved_config.json" --output "$RUN" --checkpoint epoch_0500_model.pt --device cuda:0
```

Each later segment requires its own authorization and keeps the saved scientific horizon and optimizer state. Routine validation reports a native-only loss subtotal: the new neighborhood losses are not evaluated on that panel, so the full validation objective is unavailable. The existing physical comparison tool `tools/thermal_field_response_evaluate.py` evaluates declared matched panels and saved physical differences; its `--local-contrast-diagnostics` option adds the available small-panel contrast diagnostic. Use its `--help` for the required panel, partition, stage and output arguments. Record actual preparation and decoding work and complete process time; the nominal number of scene-position presentations does not establish a speedup. The historical field-response calibration importer is a separate legacy path and does not gain the new profile's structured binding checks.
