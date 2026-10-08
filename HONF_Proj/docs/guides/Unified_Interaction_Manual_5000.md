# Manual full-data recipes for unified interaction refinement

The unified development Goal does not launch a formal run. These commands provide a separate, manually chosen 5000-epoch full-data workflow through the same `TrainingEngine`; they do not resume or promote fixed25/fixed24 weights. Choose one dataset and one execution arm after reviewing the development report. The commands below are capabilities, not evidence that a new full-data fit has completed.

Run from `HONF_Proj` in the `ModularDT` environment. Every preparation needs a new recipe directory outside preserved scientific runs. Thermal binds the completed immutable formal3901 D-sep checkpoint, reads formal3902 architecture metadata, and initializes fresh fine/base/router weights and optimizer moments. Its original600 TRAIN cases fit fresh normalization; canonical89 monitoring excludes the known TRAIN duplicate0273 while original90 compatibility results remain separately labelled. Wind uses original420 TRAIN direction rows,90 validation rows and90 TEST metadata rows from the seed42 layout split; normalization, height profile and physical role scales are freshly fitted on TRAIN only. Neither task starts new physical solves.

## Preparation and metadata validation

Metadata-only preparation verifies source membership and preserved bindings without fitting full-data transforms, constructing a training model or starting an optimizer. Its recipe is deliberately not ready to train. Use a different path for subsequent full preparation.

```bash
CUDA_VISIBLE_DEVICES="" python tools/unified_interaction_formal.py prepare --task thermal --arm adaptive_detail --metadata-only --recipe diagnostics/generated/manual_thermal_metadata/recipe.json
CUDA_VISIBLE_DEVICES="" python tools/unified_interaction_formal.py dry-run --recipe diagnostics/generated/manual_thermal_metadata/recipe.json
CUDA_VISIBLE_DEVICES="" python tools/unified_interaction_formal.py prepare --task wind --arm adaptive_detail --metadata-only --recipe diagnostics/generated/manual_wind_metadata/recipe.json
CUDA_VISIBLE_DEVICES="" python tools/unified_interaction_formal.py dry-run --recipe diagnostics/generated/manual_wind_metadata/recipe.json
```

Full preparation reads bound full-TRAIN targets to fit the new transforms and declared calibration, and writes a ready recipe. It performs no optimizer update. These commands are for an explicitly authorized future formal preparation; the development Goal executed only metadata-only checks. Replace `adaptive_detail` with `full_detail` if that is the chosen candidate rather than launching both as another portfolio.

```bash
CUDA_VISIBLE_DEVICES="" python tools/unified_interaction_formal.py prepare --task thermal --arm adaptive_detail --recipe diagnostics/generated/manual_thermal5000/recipe.json
CUDA_VISIBLE_DEVICES="" python tools/unified_interaction_formal.py dry-run --recipe diagnostics/generated/manual_thermal5000/recipe.json
CUDA_VISIBLE_DEVICES="" python tools/unified_interaction_formal.py prepare --task wind --arm adaptive_detail --recipe diagnostics/generated/manual_wind5000/recipe.json
CUDA_VISIBLE_DEVICES="" python tools/unified_interaction_formal.py dry-run --recipe diagnostics/generated/manual_wind5000/recipe.json
```

## Manual startup benchmark

A separately labelled startup benchmark visits the entire bound TRAIN population for three complete epochs and uses the exposed development validation panel for its stop review. It creates a real optimizer and checkpoint, so it requires explicit future authorization and was not executed during development. Its run identity differs from the formal run. Use its measured preparation, epoch, validation, save and process times to forecast the H64/message64 5000-epoch candidate; the older H32 pilot is not a runtime estimate for this model.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/unified_interaction_formal.py start --recipe diagnostics/generated/manual_thermal5000/recipe.json --run-dir diagnostics/generated/manual_thermal5000/startup_three_epochs --device cuda:0 --startup-benchmark --stop-after 3
CUDA_VISIBLE_DEVICES=2 python tools/unified_interaction_formal.py start --recipe diagnostics/generated/manual_wind5000/recipe.json --run-dir diagnostics/generated/manual_wind5000/startup_three_epochs --device cuda:0 --startup-benchmark --stop-after 3
```

## Start, status, clean stop and exact resume

Start a new empty run directory after approval of the recipe and measured forecast. The common stages are all-fine warmup through500, open gates through600, soft refinement through800 and hard refinement thereafter; fine learning-rate hold ends at2000 and the final horizon is5000. Native losses, source laws, physical normalization and sampling stay dataset-owned. Monitoring and declared checkpoints occur every100 epochs, with an explicitly requested stop also saved.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/unified_interaction_formal.py start --recipe diagnostics/generated/manual_thermal5000/recipe.json --run-dir diagnostics/generated/manual_thermal5000/formal --device cuda:0 --stop-after 5000
CUDA_VISIBLE_DEVICES=2 python tools/unified_interaction_formal.py start --recipe diagnostics/generated/manual_wind5000/recipe.json --run-dir diagnostics/generated/manual_wind5000/formal --device cuda:0 --stop-after 5000
CUDA_VISIBLE_DEVICES="" python tools/unified_interaction_formal.py status --run-dir diagnostics/generated/manual_wind5000/formal
CUDA_VISIBLE_DEVICES="" python tools/unified_interaction_formal.py clean-stop --run-dir diagnostics/generated/manual_wind5000/formal --note "manual budget review"
CUDA_VISIBLE_DEVICES=2 python tools/unified_interaction_formal.py resume --recipe diagnostics/generated/manual_wind5000/recipe.json --run-dir diagnostics/generated/manual_wind5000/formal --checkpoint diagnostics/generated/manual_wind5000/formal/latest_model.pt --device cuda:0 --stop-after 5000
```

The stop request is acknowledged after a complete epoch; inspect the acknowledgement and terminal process before resuming. Exact resume checks the recipe, dataset/normalization binding, execution arm, schedule, optimizer group membership, checkpoint age, provider state and RNG. Use the matching Thermal recipe/run paths and GPU0 for its resume. A startup checkpoint must retain `--startup-benchmark` when resumed and cannot silently become the formal run. A changed membership, fresh transform or different schedule requires a new labelled run. Keep all recipes, figures, arrays and checkpoints in ignored local storage; commit only maintained code, tests and documentation.

The maintained implementation is [unified_interaction_formal.py](../../tools/unified_interaction_formal.py), with dataset factories [Thermal](../../Case_ThermalChannel/src/channelthermal/training/unified_formal.py) and [Wind](../../Case_WindFarm/src/windfarm/training/unified_formal.py). The [development guide](Unified_Interaction_Training_and_Refinement.md) describes the shared numerical contracts and retained histories.
