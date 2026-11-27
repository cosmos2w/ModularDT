# Thermal organization-focused response training

This guide describes the R17-12 comparison and the reusable preparation and recovery commands. It does not authorize additional training. Each model predicts all native physical fields from the same original case inputs and native observation rows. The response arms additionally compare predicted and observed changes between existing paired training scenes. They differ in whether that response gradient can update all trainable parameters or only the parameters that govern relationships and access. Native field losses update the complete model in every arm. Both response arms use the same all-parameter epoch-zero calibration and gradient-composition rule.

For the present Thermal coupled-native binding, the auxiliary difference compares velocity components, pressure, vorticity and fluid temperature at genuinely shared physical observation positions. The original extraction retains the velocity dependencies needed for vorticity. Surface, material and outside-temperature records have no declared member correspondence between independent base scenes, so they retain their ordinary native state supervision without invented response labels. Their responses to saved physical edits are measured separately at evaluation; those edited reference arrays never enter fitting or calibration.

`observation_response.reader_policy` selects `dual` or `sampled_equal_probability` independently of `response_update_scope`, which selects `disabled`, `all_parameters` or `organization_parameters`. The two enabled scopes require `response_gradient_composition: {"kind": "native_priority_projection", "maximum_native_norm_ratio": 0.1}`. Omitting these options preserves the historical variant's execution and ordinary summed-gradient behavior. An explicitly disabled response keeps the matched pair-aware native samples without extracting a response loss or constructing its gradient graph.

Ownership follows the actual module and parameter identities, and the resolved names are saved once in the provider definition. Relationship-exclusive query/key projections, geometry scores, applied-gate calibrations, member scores and typed read strengths belong to organization. A tensor also used to form values belongs to content/shared, including tied parameters, the joint query encoder, and contextual key normalization: that normalization feeds both the key and value projections. Native losses continue updating both sets.

For each effective batch, the engine accumulates the complete native gradient and the response gradient on its selected parameter set separately. It removes the response component opposing the native gradient on that same set, then caps the remaining response norm at 0.1 times the complete native gradient norm. A zero complete native norm or zero projected response gives no correction. Missing gradients remain missing when the correction is zero, preserving AdamW's inactive-parameter behavior. The normal global clip and optimizer advance run once after composition. The recorded TRAIN total is a nominal weighted loss sum; the projected update is not its gradient, and this local projection does not guarantee an improving AdamW displacement or held-out field error. Recorded projection/cap fractions divide their counts by `native_priority_updates`; cosine averages use their explicitly defined count.

The three fresh TRAIN150 recipes target 1,000 epochs apiece, with reviews at 100, 500 and 1,000. A fresh TRAIN600 native reference and one candidate target 500 apiece, with reviews at 100 and 500. Nominate the organization-focused candidate unless the completed TRAIN150 evidence gives a clear reason to select all-parameter response instead. Stop at the declared final age; any later extension requires separate authorization. An epoch visits the declared training membership once. Qualification epochs are additional work.

## Constructing matched recipes

`tools/thermal_coupled_campaign_prepare.py` derives recipes from a saved R17-11 coupled-native template and the saved formal R17-4 schedule source. Select the template from the intended cohort. The preparer keeps the template's dataset, native membership, case pairs, calibration cases, query budgets and normalization; it writes only small JSON files and does not prepare the model or copy data. Provide a verified DATA2 round output directory and its declared byte limits. Examples use descriptive placeholders chosen by the operator:

```bash
python tools/thermal_coupled_campaign_prepare.py --template TRAIN150_TEMPLATE.json --schedule-source R17_4_FORMAL_RECIPE.json --cohort train150 --output-root ROUND_OUTPUT --recipe-dir ROUND_OUTPUT/recipes --initial-state ROUND_OUTPUT/qualification/Common_Train150_Epoch_Zero_R17_12.pt --round-name R17_12 --max-output-bytes OUTPUT_BUDGET --peak-temporary-bytes TEMPORARY_BUDGET --minimum-free-bytes FREE_HEADROOM
python tools/thermal_coupled_campaign_prepare.py --template TRAIN600_TEMPLATE.json --schedule-source R17_4_FORMAL_RECIPE.json --cohort train600 --output-root ROUND_OUTPUT --recipe-dir ROUND_OUTPUT/recipes --initial-state ROUND_OUTPUT/qualification/Common_Train600_Epoch_Zero_R17_12.pt --round-name R17_12 --max-output-bytes OUTPUT_BUDGET --peak-temporary-bytes TEMPORARY_BUDGET --minimum-free-bytes FREE_HEADROOM
```

The defaults produce a native reference, all-parameter response and organization-focused response for TRAIN150, plus a native reference and organization-focused response for TRAIN600. If the completed comparison supports the alternative, use `--candidate-scope all_parameters` when constructing the formal pair in a fresh recipe directory. The source schedule has learning rate 0.0003 through epoch 1,000, cosine decay to 0.00003 at epoch 2,000, and 0.00003 thereafter. The clock is absolute: 500- or 1,000-epoch stops do not shorten it. The independent sampled-reader policy draws full and restricted access with equal probability once per effective batch; both members of a pair use that reader. Scheduled validation uses both readers. Query tile 512 uses FP32 and checkpointed execution; default activation-retention fraction is zero. GPU choice belongs only to the runtime invocation.

Prepare one actual zero-epoch model, empty optimizer state, RNG and response calibration for each cohort. Use an enabled response recipe as the calibration carrier; calibration always measures all parameters, even when the carrier's later updates are organization-focused. All arms in that cohort load the same file, including the native reference. The calibration compares raw native and response gradients over the declared input-only training panel, without an optimizer advance. Do not reuse a TRAIN150 initial state or its normalization for TRAIN600.

```bash
python tools/observation_response_prepare.py --recipe TRAIN150_ALL_PARAMETER_RECIPE.json --fresh-epoch-zero --output COMMON_TRAIN150_STATE.pt --device RUNTIME_DEVICE
python tools/observation_response_prepare.py --recipe TRAIN600_RESPONSE_RECIPE.json --fresh-epoch-zero --output COMMON_TRAIN600_STATE.pt --device RUNTIME_DEVICE
```

## Qualification and execution

After the shared state exists, the following check runs two uninterrupted epochs and a separate two-epoch interrupted/resumed lineage. It reloads the saved configuration in a fresh process and compares the model, optimizer, RNG, sampler, provider, schedule, recording columns, losses, validation, next draws and loss-curve files. Charge all four epochs to technical qualification. Choose a new empty output under the declared round storage root.

```bash
python tools/verify_coupled_native_recovery.py --recipe ORGANIZATION_RESPONSE_RECIPE.json --output-root ROUND_OUTPUT/qualification/Organization_Focused_Recovery --device RUNTIME_DEVICE
```

For any one of the saved recipes, start and continue the same lineage as follows. Change the next stop only within the authorized horizon; use the saved configuration for every resume. Launch independent model processes only on devices available under the current runtime policy. Review startup timing once, then use sparse read-only monitoring. Keep the existing complete histories, ten-epoch curve cadence, validation and checkpoint milestones.

For formal TRAIN600 runs, the physical run directory must be a direct child of `storage.output_root`, for example `ROUND_OUTPUT/Native_State_Reference_Train600_R17_12`. A convenient `RUN_DIRECTORY` symlink may point to that directory. A nested physical path such as `ROUND_OUTPUT/runs/NAME` does not satisfy the formal-alias policy when the configured root is `ROUND_OUTPUT`; keep the physical payload at the declared root and preserve the usual formal alias under `Trained_Results/ThermalChannel/HONF_Forward_Runs/`.

```bash
python tools/relation_anchored_train.py fit --recipe RECIPE.json --output RUN_DIRECTORY --stop-after 100 --device RUNTIME_DEVICE
python tools/relation_anchored_train.py fit --recipe RUN_DIRECTORY/configs/resolved_config.json --output RUN_DIRECTORY --resume latest --stop-after 500 --device RUNTIME_DEVICE
python tools/relation_anchored_train.py fit --recipe RUN_DIRECTORY/configs/resolved_config.json --output RUN_DIRECTORY --resume latest --stop-after 1000 --device RUNTIME_DEVICE
```

The last command applies only to the declared TRAIN150 horizon. For a future explicitly authorized stop beyond a saved horizon, use the same saved recipe and latest checkpoint with `--extend-horizon --stop-after AUTHORIZED_TOTAL_EPOCHS`. This extends the run without stretching or restarting the saved optimizer schedule; the command itself is not authorization.

## Matched physical evaluation

At TRAIN150 reviews use the saved input-selected TRAIN15 panel in `diagnostics/generated/r17_11_efficient_coupled_response/panels/native_panels.json` and the maintained assessment9. Run each command for both `restricted` and `full` reader values, writing separate outputs. The field evaluator also computes natural-scene differences and optional J/P and field-change diagnostics on the same physical positions. Assessment9 is historically exposed development evidence, not independent TEST data.

```bash
python tools/thermal_field_response_evaluate.py --recipe RUN_DIRECTORY/configs/resolved_config.json --run RUN_DIRECTORY --checkpoint CHECKPOINT --panels diagnostics/generated/r17_11_efficient_coupled_response/panels/native_panels.json --stage development --split train --read-mode READER --thermal-functionals --local-contrast-diagnostics --device RUNTIME_DEVICE --output TRAIN15_RESULT.json
python tools/thermal_field_response_evaluate.py --recipe RUN_DIRECTORY/configs/resolved_config.json --run RUN_DIRECTORY --checkpoint CHECKPOINT --assessment9 --stage development --split validation --read-mode READER --thermal-functionals --local-contrast-diagnostics --device RUNTIME_DEVICE --output ASSESSMENT9_RESULT.json
```

## Occasional loss-family gradient audit

At a declared review, run the following once for each saved checkpoint. It uses the same input-selected TRAIN15 scenes and native sampling epoch 1 for both readers, retains the checkpoint's actual pair identities and common initial response calibration, sets the organization clock to the checkpoint epoch with update index zero, and writes a new compact JSON record. Replace the checkpoint, output and runtime device with the chosen review values.

```bash
python tools/thermal_loss_family_gradient_audit.py --checkpoint RUN_DIRECTORY/checkpoints/epoch_0100_model.pt --panels diagnostics/generated/r17_11_efficient_coupled_response/panels/native_panels.json --device RUNTIME_DEVICE --output ROUND_OUTPUT/evidence/MODEL_GRADIENT_AUDIT_EPOCH_0100.json
```

The record gives raw weighted loss-family gradient norms and cosines for all parameters and for the resolved organization and content/shared subsets. Native reductions use the actual ordered loss weights and denominators on all 15 scenes; response averages only configured eligible pairs wholly inside the panel and uses the saved calibration and checkpoint-epoch ramp. Both pair members use the same reader. For the present TRAIN150 pairing the panel contains five complete pairs; the TRAIN600 pairing contains one, which limits its response comparison. An outside-panel pair member can affect the existing native sampling plan but is not decoded. The native-reference response is a diagnostic calculation only: the audit performs no fitting or optimizer update. These fixed-panel gradients are not the actual 48-case update, and precede projection, clipping and AdamW. It records restoration, preparation and gradient time plus forward-work and gradient-call counts; the forward counters do not include checkpoint recomputation during gradients. The provider and sampler recovery state, model state and random streams are restored; any temporary metadata warmed during this standalone process is discarded when it exits.

At the final formal review, the specifically authorized original held-out coverage uses the maintained 89-case native evaluator once per model for both readers. It excludes the overlapping case 0273 and preserves all per-field means, tails and near-object results. It does not evaluate every TRAIN600 case or create new blind test evidence.

```bash
python tools/thermal_native_checkpoint_evaluate.py --checkpoint FORMAL_EPOCH_500_MODEL.pt --all-original-held-out --read-mode both --near-object-errors --query-tile-size 1024 --device RUNTIME_DEVICE --output ORIGINAL_HELDOUT89_RESULT.json
```

The previously exposed 80-edit/19-family corpus and saved seven-candidate inverse request remain separate from training. `tools/thermal_saved_response_benchmark.py` restores a saved native checkpoint and uses the same fixed historical family/action identities for every model. It checks their saved physical source paths, records each baseline's current native-training membership separately, and persists both readers' model choices before the physical outcome join. The fixed historical scales and response roles (`fit`, `original_withheld`, `historical_new_withheld`, `native_seen`, `fresh_layout`) are preserved. Field results include the zero-change reference error, predicted and reference change magnitudes, resolved direction and ratio, and separate newly valid fluid-state errors. No physical solves or new labels are needed.

Prepare the panel once from the existing R17-4 reference board files, their source recipe and scale records, and the original saved strict-request protocol. Each placeholder is a path to the named existing artifact; all output belongs under the declared round storage budget.

```bash
python tools/thermal_saved_response_benchmark.py prepare --reference-boards REFERENCE_BOARD_DIRECTORY --historical-recipe HISTORICAL_CORPUS_RECIPE.json --training-inputs HISTORICAL_TRAIN600_SCALES.json --strict-protocol HISTORICAL_STRICT_PROTOCOL.json --strict-scales-identity HISTORICAL_STRICT_SCALE_IDENTITY.json --historical-new-manifest HISTORICAL_NEW_ACTIONS.json --native-seen-manifest NATIVE_SEEN_CORPUS.json --fresh-layout-manifest FRESH_LAYOUT_CORPUS.json --output ROUND_OUTPUT/evidence/fixed_saved_response_panel_complete.json
python tools/thermal_saved_response_benchmark.py evaluate --panel ROUND_OUTPUT/evidence/fixed_saved_response_panel_complete.json --checkpoint RUN_DIRECTORY/checkpoints/epoch_1000_model.pt --output-dir ROUND_OUTPUT/evidence/MODEL_RESPONSE_REVIEW --device RUNTIME_DEVICE
```

For an intermediate review of the original and historical-new withheld roles, retain their exact three families and 16 edits using the following narrower invocation. The final formal checkpoint is `epoch_0500_model.pt`; the TRAIN150 final checkpoint is `epoch_1000_model.pt`. A complete evaluation includes all five roles and the strict request.

```bash
python tools/thermal_saved_response_benchmark.py evaluate --panel ROUND_OUTPUT/evidence/fixed_saved_response_panel_complete.json --checkpoint RUN_DIRECTORY/checkpoints/epoch_0500_model.pt --output-dir ROUND_OUTPUT/evidence/MODEL_WITHHELD_REVIEW --role original_withheld --role historical_new_withheld --skip-strict-request --device RUNTIME_DEVICE
```

Report predictor, relation/use, physical response, cost and inverse-choice outcomes separately. A changed access pattern or a faster query-only replay is not evidence of physical causality or complete training speed. Measure complete new-run cost and account for two gradient calculations; compare complete work and memory on the same device.
