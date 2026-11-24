# Native relation-anchored case training

The shared entry `tools/relation_anchored_train.py` builds a case-owned model, native sampler, physical objective and saved transforms. It supports the Wind joint-velocity adapter, the existing Thermal native field adapter and the one-pass Thermal port/local composition. Installation and public core selection are described in the [README](../../README.md).

## Configure a case

Wind's [native adapter guide](../../Case_WindFarm/docs/relation_anchored.md) supplies a portable recipe template. The generic `wind_relation_anchored_v1` recipe declares the 3-D environment lattice, query counts, packing, validation workload and epoch horizon. Historical recipe versions retain their original interpretation. The native geometry catalogue must be prepared before fitting; [the cache guide](Wind_Native_Catalogue_Cache.md) explains verification and resource usage.

Thermal native recipes continue to use the [maintained workflow](Thermal_Relation_Anchored_Training.md) and require no extra response corpus. The existing edited-scene `state_response` profile adds configurable absolute/difference fitting when explicitly selected with its paired TRAIN data; its [guide](State_Response_Training.md) documents the actual settings and native-only opt-out. For new response work, prefer the [base-data pairing interface](Base_Dataset_Response_Pairs.md), an R17-3 implementation target. Recipes without these profiles retain their recorded objective. The [port/local composition](Thermal_Port_Local_Integration.md) accepts the existing versioned C0/P64 recipe and its declared R14 source and compatible external local checkpoint. Read its saved recipe from an existing run or supply a matching configured recipe; source data and model artifacts are not distributed with this repository. The local operator is frozen, with gradients through its predicted outside-temperature inputs. Target-derived port temperatures enter supervised loss only.

For an explicit within-role access control, set `member_routing.model.within_role_access_prior` to `measure_only`. The model first selects the usual full and restricted original-member rows using its learned relations. Within each row and M/E role, it retains the selected members and their total prior mass, and distributes that mass in proportion to the scene's physical member measures. Subsequent query scoring, contextual exchange, role gates, member values and joint output are unchanged. A role with positive prior mass but no selected physical measure is undefined and raises an error. Omitting the option preserves historical learned priors and checkpoint behavior.

To compare a learned-prior checkpoint on exactly the same prepared states and selected members, call `core.decode(prepared, query, intervention={"within_role_access_prior": "measure_only"})` or pass the same intervention to `decode_paired`. This changes the within-role proportions only; it does not remove learned support, role masses or contextual organization. Preparation and query work counts identify the replacement work separately; complete execution time and memory require measurement.

For a saved `observation_response` checkpoint, `tools/observation_access_prior_evaluate.py` accepts `--recipe`, `--run`, `--checkpoint`, `--pair-panel`, `--device` and `--output`. The panel JSON must supply a name and explicit scene pairs; the command does not choose a population. It samples each pair once, reuses each scene/frame preparation for both policies, saves the selected original-member IDs and role masses, and reports physical component estimates and executed work. Paired full/restricted timing and standalone restricted timing are separate and include physical scaling and transfer to CPU. The allocator peaks and single ordered call times describe this evaluation, not a controlled speed comparison.

## Inspect, fit and resume

Configure dataset, calibration, source-checkpoint and output paths for the local installation. `inspect` builds the actual provider and reports membership, parameters and optimizer groups without fitting. Provider preparation can read data and calibration; it is not a model-free syntax check.

```bash
python tools/relation_anchored_train.py inspect --recipe /path/to/recipe.json --device cpu
python tools/relation_anchored_train.py fit --recipe /path/to/recipe.json \
  --output /data2/your-project/native_run --stop-after 100 --device cuda:0
python tools/relation_anchored_train.py fit --recipe /path/to/recipe.json \
  --output /data2/your-project/native_run --resume latest --stop-after 200 --device cuda:0
python tools/relation_anchored_train.py evaluate --recipe /path/to/recipe.json \
  --output /data2/your-project/native_run --checkpoint latest --device cpu
```

`--stop-after` is an execution stop within the recipe's epoch horizon; an epoch visits the declared primary membership once. Query tiles and physical microbatches are execution settings, while query draws, native masks, effective reductions and TRAIN-only transforms retain their scientific meaning. Use the engine's declared checkpoint and history cadence. Total TRAIN and total validation curves retain their separately recorded objectives and reductions.

## Explicit execution-horizon extension

For a supported constant-schedule Wind run, `fit --resume latest --stop-after 2500 --extend-horizon` can extend the execution horizon while retaining the original recipe and optimizer schedule. The engine accepts only the explicit horizon and future 100-epoch milestone changes; hidden data, objective, transform, packing or optimizer changes are rejected. Generic Wind recipes use the same explicit mechanism; historical versions retain their saved rules.

The preserved full-TRAIN600 P64 continuation supports its original 100→500 extension. Other Thermal horizons and experimental arms are not registered by this entry. Existing R14 and other historical workflows keep their own loaders and resume rules. Supplying a compatible public name never rewrites a checkpoint's architecture, normalization or external-asset dependencies.

For design-response measurements use [case response evaluation](Case_Response_Evaluation.md). Rebuild organization for an edited design, compare against aligned physical references, and report response accuracy independently of state reconstruction. The training entry makes no claim that a trained model is suitable for inverse optimization.
