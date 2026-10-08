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

Full preparation reads bound full-TRAIN targets to fit the new transforms and declared calibration, and writes a ready recipe. It performs no optimizer update. The development Goal executed only metadata-only checks; full preparation and training require their own user authorization. Replace `adaptive_detail` with `full_detail` if that is the chosen candidate. Keep historical comparison receipts alongside each run: matching full-data membership and normalization does not make different architectures, objectives, query budgets or update counts a matched comparison.

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

Start a new empty run directory after explicit user authorization of the full-data fit. The common stages are all-fine warmup through500, open gates through600, soft refinement through800 and hard refinement thereafter; fine learning-rate hold ends at2000 and the final horizon is5000. All-fine warmup describes the forward execution route; it does not freeze router/gate parameters or suppress their existing declared gradients. Native losses, source laws, physical normalization and sampling stay dataset-owned. The defaults for newly prepared recipes validate at100 and every500 epochs, retain immutable checkpoints at100/500/1000/2000/2500/5000, and update loss curves every100 epochs. The engine writes `last.pt` and an identical regular-file compatibility copy, `latest_model.pt`, every100 epochs and at validation reviews and explicit/clean stops; the first completed epoch also writes both copies and curves. Best-field selection uses only actual validation reviews and replaces one alias; it cannot select unvalidated checkpoints.

The preparation options `--validation-every`, `--validation-epochs`, `--checkpoint-epochs`, `--latest-every` and `--curve-every` are sealed into the recipe and exact-resume identity. An explicit final stop saves its checkpoint; a requested clean stop saves latest state and evaluates the completed epoch without adding an undeclared milestone. Existing sealed recipes without these options retain their original100-epoch monitoring behavior. Curves replace `loss_curves.pdf` and a small `loss_curves.png` companion; their training values are equal macro-update means, including the partial final update. The curves use logarithmic y axes and mask nonpositive or nonfinite points for display only; the saved history is left unchanged. Compact native validation metrics remain separate from TRAIN objectives, and routine training does not export physical field arrays or figures.

At recipe-default training microbatches (Thermal8 and Wind4), a full5000 horizon visits600 Thermal cases per epoch in75 microbatches and13 optimizer updates, including a24-case final update:3,000,000 case visits and65,000 updates total. Wind visits420 direction rows per epoch in105 microbatches and18 updates, including a12-case final update:2,100,000 case visits and90,000 updates total. `progress.json` records completed optimizer updates within a cold first epoch; `history.json` and the final summary retain completed-epoch counters. These counters, query budgets, source/partition hashes, units and validation IDs must accompany later comparisons with historical runs.

The formal Wind provider allows a lazy128GiB host cache for native geometry catalogues, shared by TRAIN and the first validation construction; it does not preallocate that capacity or change query sampling. Check available host memory before launch. The capacity is recorded in the provider identity and checked on exact resume, but the resident catalogue entries are process-local and cold-rebuilt after resume. TRAIN transform preparation retains its existing64GiB cache; cold catalogue construction can take several minutes before the first optimizer update and before the first completed-epoch curve.

When stderr is attached to a terminal, the engine displays a `tqdm` progress bar there. In tmux, stdout may append to a console log while stderr remains attached to the pane, for example with `>> diagnostics/generated/manual_wind5000/wind_console.log`; do not merge stderr into that file if the live bar should remain visible.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/unified_interaction_formal.py start --recipe diagnostics/generated/manual_thermal5000/recipe.json --run-dir Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_NEW_thermal_adaptive_full5000 --device cuda:0 --stop-after 5000
CUDA_VISIBLE_DEVICES=2 python tools/unified_interaction_formal.py start --recipe diagnostics/generated/manual_wind5000/recipe.json --run-dir Trained_Results/WindFarm/HONF_Forward_Runs/Run_NEW_wind_adaptive_full5000 --device cuda:0 --stop-after 5000
CUDA_VISIBLE_DEVICES="" python tools/unified_interaction_formal.py status --run-dir Trained_Results/WindFarm/HONF_Forward_Runs/Run_NEW_wind_adaptive_full5000
CUDA_VISIBLE_DEVICES="" python tools/unified_interaction_formal.py clean-stop --run-dir Trained_Results/WindFarm/HONF_Forward_Runs/Run_NEW_wind_adaptive_full5000 --note "manual budget review"
CUDA_VISIBLE_DEVICES=2 python tools/unified_interaction_formal.py resume --recipe diagnostics/generated/manual_wind5000/recipe.json --run-dir Trained_Results/WindFarm/HONF_Forward_Runs/Run_NEW_wind_adaptive_full5000 --checkpoint Trained_Results/WindFarm/HONF_Forward_Runs/Run_NEW_wind_adaptive_full5000/latest_model.pt --device cuda:0 --stop-after 5000
```

The stop request is acknowledged after a complete epoch; inspect the acknowledgement and terminal process before resuming. Exact resume checks the recipe, dataset/normalization binding, execution arm, schedule, optimizer group membership, checkpoint age, provider state and RNG. Use the matching Thermal recipe/run paths and GPU0 for its resume. A startup checkpoint must retain `--startup-benchmark` when resumed and cannot silently become the formal run. A changed membership, fresh transform or different schedule requires a new labelled run. Keep all recipes, figures, arrays and checkpoints in ignored local storage; commit only maintained code, tests and documentation.

### Explicit training-microbatch continuation

`--microbatch-cases` is available only on `resume` and requires an explicit value; it changes training microbatching while preserving the effective batch, schedule, saved optimizer moments, RNG state and completed history. For a memory-bounded continuation, use Thermal16 with effective48 and Wind8 with effective24. The macro-update counts remain13 and18 per epoch, while the training microbatch counts become38 Thermal and53 Wind, down from75 and105 at recipe defaults. Past history rows retain their original counters; from the resumed epoch onward, query seeds can change because the microbatch index is part of the sampling key, and FP32 accumulation order can differ. Treat this as an explicitly amended trajectory rather than bitwise-identical replay. Validation grouping remains8 cases for Thermal and4 for Wind. The native formal Wind task accepts at most24 cases per training microbatch; fixed24 development remains capped at4.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/unified_interaction_formal.py resume --recipe diagnostics/generated/manual_thermal5000/recipe.json --run-dir Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_NEW_thermal_adaptive_full5000 --checkpoint Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_NEW_thermal_adaptive_full5000/latest_model.pt --device cuda:0 --stop-after 5000 --microbatch-cases 16 >> diagnostics/generated/manual_thermal5000/thermal_console.log
CUDA_VISIBLE_DEVICES=2 python tools/unified_interaction_formal.py resume --recipe diagnostics/generated/manual_wind5000/recipe.json --run-dir Trained_Results/WindFarm/HONF_Forward_Runs/Run_NEW_wind_adaptive_full5000 --checkpoint Trained_Results/WindFarm/HONF_Forward_Runs/Run_NEW_wind_adaptive_full5000/latest_model.pt --device cuda:0 --stop-after 5000 --microbatch-cases 8 >> diagnostics/generated/manual_wind5000/wind_console.log
```

The maintained implementation is [unified_interaction_formal.py](../../tools/unified_interaction_formal.py), with dataset factories [Thermal](../../Case_ThermalChannel/src/channelthermal/training/unified_formal.py) and [Wind](../../Case_WindFarm/src/windfarm/training/unified_formal.py). The [development guide](Unified_Interaction_Training_and_Refinement.md) describes the shared numerical contracts and retained histories.
