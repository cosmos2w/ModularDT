# `src_new` Code Evaluation and Revision Plan

## Executive assessment

The current implementation contains a coherent model worth preserving: a ChannelThermal input adapter builds module and environment tokens, a reusable HONF core organizes them into hyperedges, a query decoder predicts the global field, and an optional Stage-A local surrogate feeds internal/interface physics back into the module tokens. The most valuable architectural seam is already present as `encode_and_organize()` followed by `decode_queries()`.

The repository is not yet cleanly transferable or production-ready. The main obstacles are not the core HONF equations; they are ownership and contract problems around them:

- training, evaluation, and comparison do not make checkpoint normalization metadata the single source of truth;
- mixed Stage-A data is normalized in incompatible spaces and only one normalizer is saved;
- configuration accepts unknown keys silently, contains settings with no effect, and has overlapping mode/boolean authorities;
- the full wrapper, dataset readers, and CLI scripts expose very large legacy dictionaries instead of explicit typed inputs and outputs;
- production behavior, ablations, compatibility paths, diagnostics, plotting, and inverse-design export are interleaved;
- evaluation recomputes the entire case representation for every query chunk;
- duplicated files and helpers obscure which implementation is authoritative;
- current tests protect several valuable behaviors, but they are executable scripts tied to local data/checkpoints rather than a layered, CI-friendly test suite.

The recommended direction is therefore an extraction and consolidation, not a rewrite of the model mathematics. Freeze current validated behavior, fix correctness risks first, establish typed/versioned contracts, then move the enhanced HONF path into an installable core package with ChannelThermal-specific adapters and optional local-coupling plugins around it.

## Audit scope and evidence

This audit covered every Python source line currently under `src_new`:

- 38 Python files;
- 10,721 lines total;
- all top-level training, evaluation, and comparison entry points;
- both dataset modules;
- every helper, model, package initializer, and test script;
- the untracked working-tree file `compare_models.py`;
- the adjacent `Configs_new` profiles where needed to determine whether settings are operational.

The audit used file-by-file reading, import/call/reference searches, AST parsing, duplicate-file comparison, config-to-code searches, and execution of the existing regression scripts. No Python source was changed.

Verification performed in the `ModularDT` conda environment:

- all 38 Python files parse successfully;
- `test_hypergraph_plan_stability.py` passes;
- `smoke_global_modes.py` passes for global-head, teacher, predicted, and mixed modes when given an existing valid config explicitly;
- `test_local_checkpoint_compat.py` strictly loads the existing Stage-A checkpoint and passes finite-output checks;
- `test_newhonf_hardening.py` passes its decoder-call, frozen-surrogate, mechanism-path, and self-contained checkpoint round-trip checks.

These passes establish a useful behavioral baseline. They do not cover the normalization, mixed-dataset, configuration, data-loading, or chunked-inference issues identified below.

## Current model and dependency structure

The operational dependency direction is:

```text
train.py / evaluate.py / compare_models.py
    -> GlobalChannelThermalDataset
    -> ChannelThermalHONFModel
         -> ChannelThermalInputAdapter
         -> ChannelThermalEnvironmentBuilder
         -> HONFNeuralField
              -> HypergraphOrganizerCore
              -> HypergraphFieldDecoder
         -> LocalSurrogateCoupling
              -> LocalModuleSurrogate (optional Stage A)
         -> GlobalFallbackHeads
    -> losses, diagnostics, checkpoint metadata, plots, plan export

train_local.py / evaluate_local.py
    -> LocalModuleDataset or GlobalModuleAlignmentDataset
    -> LocalModuleSurrogate
```

The intended reusable portion is `_models_core`, but it is not fully domain-neutral. Its configuration assumes a two-dimensional rectangular domain and module radius, and `honf_decoder.py` contains ChannelThermal boundary features. Conversely, the ChannelThermal wrapper correctly owns material, heat, port, interface, local-surrogate, and channel-geometry behavior.

## Behavioral invariants to preserve

The following behavior should be protected before structural changes begin:

1. Active module slots are masked throughout encoding, organization, local response, and final fusion.
2. Present rows of `A_mh` and rows of `A_eh` are normalized; inactive `A_mh` rows carry zero mass.
3. The organizer produces `A_mh [B,M,K]`, `A_eh [B,E,K]`, hyperedge states, source/region coordinates, masses, and strengths.
4. The enhanced decoder routes each query through `alpha_qk [B,Q,K]` and can add both hyperedge value context and H-routed pairwise module detail.
5. The local surrogate retains strict compatibility with existing Stage-A checkpoint parameter names and output behavior until a versioned migration loader is available.
6. A frozen local surrogate remains in evaluation mode after `ChannelThermalHONFModel.train()`.
7. Flux/refinement residual heads begin as near-identity corrections through their zero or small-gate initialization.
8. Standard model forward performs one final global-field decode when optional consistency/refinement probes are disabled.
9. Global checkpoints remain self-contained: the embedded local model config, local weights, normalization metadata, and global model weights can load without the original external local checkpoint.
10. Hypergraph-plan export remains canonical under a permutation of hyperedge indices and preserves stable module slot indexing.
11. The five global output channels retain their current order unless a schema version explicitly changes it.

## Concrete tensor flow

Use these symbols throughout the cleanup:

| Symbol | Meaning |
|---|---|
| `B` | batch size |
| `M` | padded module slots |
| `P` | interface/port angular samples |
| `L` | local internal-disk query points |
| `Q` | global field query points |
| `E` | environment tokens, currently `num_env_tokens_x * num_env_tokens_y` |
| `K` | hyperedges |
| `D` | HONF hidden dimension |
| `F` | predicted field channels, currently 5 |

### Global data to HONF inputs

`GlobalChannelThermalDataset.__getitem__()` currently emits a nested dictionary containing:

- module centers `[M,2]`, heat powers `[M]`, presence mask `[M]`, material descriptors `[6]`, Reynolds number `[1]`, inlet velocity `[1]`, and domain lengths;
- sampled global queries `[Q,2]`, field targets `[Q,F]`, and point weights `[Q]`;
- interface conditions `[M,P,8]`, teacher local port tokens `[M,P,5]`, and interface targets `[M,P,2]`;
- local parameters `[M,7]`, local queries `[L,2]`, and internal targets `[M,L]`;
- comparison-only structure targets and optional full grids.

After batching, `ChannelThermalInputAdapter` converts the physical/domain inputs to:

- `module_features [B,M,10]`: signed/absolute heat at dataset and case-relative scales, active flag, and five material/radius descriptors;
- `global_context [B,14]`: flow inputs, module/heat aggregates, domain lengths, and six material descriptors;
- unchanged centers `[B,M,2]`, presence `[B,M]`, and heat `[B,M]`.

`ChannelThermalEnvironmentBuilder` creates:

- cell-centered environment coordinates `[B,E,2]`;
- seven ChannelThermal environment features `[B,E,7]` describing normalized position, wall/inlet/outlet distances, and centerline proximity.

### HONF encoding and organization

`HONFNeuralField.encode_and_organize()` performs:

1. `global_context [B,14] -> global_token [B,D]` through a lazy MLP.
2. `module_features [B,M,10] -> feature state [B,M,D]`.
3. normalized/Fourier module centers `[B,M,*] -> position state [B,M,D]`.
4. feature plus position state, masked by `module_present`, gives `module_tokens [B,M,D]`.
5. environment position/Fourier features plus the ChannelThermal environment features are encoded to `env_tokens [B,E,D]`; global context is added when enabled.
6. `HypergraphOrganizerCore` maps module and environment tokens to:
   - `A_mh [B,M,K]`;
   - `A_eh [B,E,K]`;
   - `hyper_state [B,K,D]`;
   - hyperedge module/environment masses `[B,K]`;
   - source and region coordinates `[B,K,2]`;
   - strength and mechanism diagnostics.

The organizer state is case-static for a fixed design and should therefore be computed once, not once per query chunk.

### Optional ChannelThermal local coupling

The wrapper uses base module tokens plus module-environment context, heat, and the global token to predict:

- `pred_port_tokens [B,M,P,5] = [theta, cos(theta), sin(theta), T_env, h]`.

Teacher, mixed, or predicted mode selects the local ports. The Stage-A model then receives flattened active/padded module batches:

- module parameters `[B*M,7]`;
- port tokens `[B*M,P,5]`;
- internal queries `[B*M,L,2]`.

It produces:

- internal temperature `[B,M,L,1]`;
- interface prediction `[B,M,P,2]` for surface temperature and normal flux;
- module response latent `[B,M,D_local]`.

`LocalSurrogateCoupling` optionally converts normalization spaces, applies the selected flux correction, reduces local outputs to a six-value response summary `[B,M,6]`, projects the latent and summary to `[B,M,D]`, and fuses them into the module tokens. One optional interaction-refinement pass decodes outside temperatures at port locations, refines the ports, reruns Stage A, and fuses the refreshed response.

The final organizer is built from the fused module tokens and unchanged environment tokens.

### Query decoding

`HypergraphFieldDecoder` receives query coordinates `[B,Q,2]`, final organizer state, and global token. In the validated enhanced path it computes:

- encoded query state `[B,Q,D]`;
- query-to-hyperedge attention `alpha_qk [B,Q,K]`;
- optional hyperedge value context `[B,Q,D]`;
- H-routed pairwise module context `[B,Q,D]` via an intermediate proportional to `[B,Q,M,*]`;
- optional global and near-module context `[B,Q,D]`;
- final field prediction `[B,Q,F]`.

Routing maps and scalar diagnostics are currently mixed into the same output dictionary. The revised API should return predictions by default and materialize dense diagnostics only on explicit request.

## Highest-priority findings

### P0: fix before treating new training or comparison results as authoritative

#### 1. Mixed Stage-A normalization is mathematically inconsistent

`MixedLocalDataset` concatenates `LocalModuleDataset` and `GlobalModuleAlignmentDataset`, but each child applies its own normalizer. `GlobalModuleAlignmentDataset` also fits a new normalizer from the selected split. The mixed model therefore sees two different normalized meanings for the same feature columns, while the mixed wrapper exposes and checkpoints only the first dataset's normalizer.

Consequences:

- identical normalized values represent different physical quantities by source;
- validation can use statistics fitted from a different split than training;
- the saved checkpoint cannot faithfully invert or reproduce the global-alignment portion.

Required revision: fit one training-only Stage-A normalizer over the combined physical-space training sources, inject that same immutable transform into every train/validation dataset, and save its schema, statistics, source fingerprint, and fit split in the checkpoint.

#### 2. Inference normalization is owned by the current HDF5 file, not the checkpoint

Global and local evaluators instantiate datasets that read normalizer arrays from the currently selected HDF5 file. Predictions are denormalized with that dataset normalizer even though checkpoints already embed their training statistics. This works only while the evaluation HDF5 normalization group is exactly compatible with the checkpoint. Dataset overrides, regenerated files, and cross-dataset comparisons can silently change the input/output transform.

Required revision: construct inference transforms from checkpoint metadata, validate the current HDF5 schema/fingerprint against the checkpoint, and use raw physical samples plus the checkpoint transform. Refuse incompatible inputs unless the user explicitly chooses a documented conversion.

#### 3. Configurations contain false or ambiguous controls

- `train.py` defaults to `./Configs_new/train_global_honf_template.json`, but that file does not exist at the root of `Configs_new`; a similarly named file is under `_tests`.
- the entire `checkpointing` section is ignored; all checkpoint variants are handled unconditionally;
- dataset config `require_converged` is not passed into filtering and has no operational effect;
- local `local_synthetic_weight` and `global_alignment_weight` are ignored; mixed data is plain concatenation;
- `local_surrogate_interface_target_weights` is unused;
- unknown dataclass/config keys are silently discarded, so misspellings are not errors;
- `max_num_modules`, `use_dynamic_tokens`, `use_local_surrogate_patch`, `field_names`, and `heat_scale` are stored but not used by the current model path;
- `material_param_dim` suggests a variable width, but the adapter always truncates/pads to six material values;
- `decoder_mode` and independent booleans overlap. In particular, names beginning with `no_hyper` do not themselves disable hyper context, and `_uses_global()` reduces to `use_global_context` for every non-`hyper_only` mode.

Required revision: introduce a strict, versioned schema; reject unknown fields; validate combinations; select one production decoder strategy explicitly; make experiment variants separate named profiles; and add tests proving every retained setting changes behavior as documented.

#### 4. Hypergraph comparison can score heuristic targets as if they were solved targets

When solved structure targets are absent, the dataset creates geometry-only fallback targets and marks `has_solved_structure_targets = 0`. `compare_models.hypergraph_metrics()` still computes target-agreement metrics whenever the fallback keys exist. Presentation plots can therefore report “hypergraph target agreement” against heuristic constructions rather than solved-field supervision.

Required revision: do not compute or aggregate solved-target agreement unless the solved-target flag is true. Report heuristic diagnostics under a separately named namespace with their construction and grid resolution recorded.

### P1: address in the first structural revision

#### 5. Chunked evaluation recomputes all case-static work

`evaluate.predict_case()` invokes the full `ChannelThermalHONFModel.forward()` for every query chunk. Each chunk repeats input adaptation, environment creation, base organization, port prediction, Stage-A inference, optional refinement, and final organization. Only query decoding needs to repeat.

Required API:

```text
prepared = model.prepare_case(case_inputs, coupling_options)
chunk_prediction = model.decode(prepared, query_xy, diagnostics=False)
local_prediction = prepared.local_prediction
```

This also gives inverse-design and serving code a stable reusable case representation.

#### 6. Mutually exclusive modules are always instantiated and saved

The wrapper always constructs local-coupling heads and fallback heads. `LocalSurrogateCoupling` always constructs port, flux, refinement, and fusion heads even when the selected config cannot use them. In global-only training, fallback/port losses are still computed and multiplied by zero; zero gradients can still create optimizer state and expose parameters to AdamW weight decay.

Required revision: instantiate a selected `InternalModel`/`CouplingStrategy` only when enabled, and skip loss computation entirely when its effective weight is zero. Keep old state names in a compatibility loader rather than in the production object graph.

#### 7. Dataset samples are much larger than each task needs

Global training ignores `point_group`, full internal-temperature grids, `rms_field`, and all structure targets. Only comparison consumes structure targets. Local training defaults `include_grid=True`, so batches load `local_grid` and `local_mask` even though training does not use them. The global reader also loads full internal grids to derive masked points for every sample.

Required revision: use task-specific projections/collators (`global_train`, `global_eval`, `comparison`, `local_train`, `local_eval`) and make expensive arrays explicitly opt-in.

#### 8. Training repeats avoidable work and I/O

- predicted validation is run separately even when the effective validation mode is already fully predicted, yielding a duplicate full pass;
- all auxiliary losses and diagnostics are computed even when their weights are zero;
- seven plots are regenerated from the full CSV every epoch;
- global-only forward still predicts unused port and fallback outputs;
- inactive padded modules are run through Stage A and masked only afterward;
- environment and local query grids are rebuilt repeatedly.

Required revision: reuse equivalent validation results, gate optional work, plot on a configurable cadence, gather only active modules for Stage A, and cache immutable geometry tensors.

#### 9. Random point sampling is not reproducible

The random training branch calls `np.random.default_rng()` without the configured seed. `set_seed()` therefore does not reproduce sampled point subsets, especially across workers/epochs.

Required revision: derive sampling RNG state from a documented tuple such as `(training_seed, epoch, case_id, worker_id)` and test exact replay.

#### 10. Package boundaries rely on path mutation

Most imports are top-level absolute imports such as `_helpers...`; direct scripts mutate `sys.path` through `_bootstrap_imports.py`, and tests mutate it again. This prevents normal installation, makes module names collision-prone, and forces `compare_models.py` to import implementation functions from `evaluate.py`.

Required revision: create an installable package, use package-relative imports internally, and expose console entry points. Evaluation services shared by CLI tools should live in a library module, not another executable.

## Detailed component evaluation

### Core HONF model

Strengths:

- `HONFNeuralField` already separates case encoding/organization from query decoding.
- organizer masking and normalization behavior is clear and testable.
- the enhanced pairwise path has a scientifically meaningful routing structure rather than an unrestricted direct memory shortcut.
- optional dense routing maps are already guarded by a flag.

Revision needs:

- move `channel_boundary_features()` out of `honf_decoder.py` into a domain query-feature adapter;
- replace fixed `domain_length_x`, `domain_length_y`, and `module_radius` assumptions with a geometry policy or explicitly scope the core to 2-D object/environment fields;
- consolidate the three Fourier-feature implementations and multiple MLP/LazyMLP implementations, choosing and documenting the `pi` versus `2*pi` convention;
- remove the unused `query_time` local in encoding and either fully support time or move time modes to a separate extension;
- handle per-case environment coordinates correctly rather than deriving encoded coordinates from `env_coords[0]`;
- remove redundant dictionary updates/copies and return a typed `OrganizedCase`;
- collapse the 12 decoder modes to one validated production path plus deliberately located experiment strategies;
- avoid a full `[B,Q,M,*]` materialization where possible by query/module chunking or sparse edge/module routing;
- separate prediction tensors, scalar diagnostics, and dense diagnostic maps.

### ChannelThermal adapter and wrapper

Strengths:

- domain feature engineering is mostly in the correct layer;
- the wrapper prevents a discarded preliminary field decode before local fusion;
- local response fusion and optional one-step interaction refinement are explicit;
- global target normalization is available for physical port-temperature probes.

Revision needs:

- split the 545-line wrapper into `prepare_inputs`, `predict_ports`, `run_local_coupling`, `organize`, and `decode` services;
- replace the dual structure-dict/keyword API and silent zero defaults for `re`, `u_in`, and materials with one validated input contract;
- use per-case domain geometry consistently in the adapter, environment builder, organizer, decoder, port probes, and visualization;
- cache the environment grid for fixed geometry;
- reduce redundant environment features (`x_norm` equals inlet distance, `y_norm` equals bottom-wall distance) unless ablation evidence justifies them;
- make the temperature channel a schema lookup instead of hard-coded index 4;
- make active-edge thresholds config/schema values rather than hard-coded `0.05` in multiple files;
- define autonomous predicted-port behavior consistently through refinement. Current teacher/mixed auxiliary predicted outputs do not necessarily undergo the same refinement as the actual selected ports;
- prevent teacher-derived interface summaries from leaking into autonomous parameter construction;
- move plan export out of the model method into a small service consuming `OrganizedCase`;
- return a small prediction object, with legacy dictionaries produced only by a compatibility adapter.

### Local surrogate and physical coupling

Strengths:

- existing checkpoint compatibility is proven by a strict-load regression;
- the Perceiver-like port encoder plus coordinate decoder is a reasonable reusable Stage-A pattern;
- normalization conversion, frozen-module behavior, zero-initialized corrections, and local/global fusion capture important physics workflow requirements.

Revision needs:

- replace magic 5-port, 7-parameter, 2-interface, and 6-summary column positions with named schema objects;
- set cross-attention `need_weights=False` because returned weights are discarded;
- stop returning port states and Perceiver latents unless diagnostics request them;
- remove or implement the unused `interface_query_theta` parameter;
- gather active modules before calling the local surrogate rather than evaluating padded inactive modules;
- validate port mode, mixed ratio, flux mode, and blend alpha instead of silently defaulting/clipping downstream;
- instantiate correction/refinement heads only for the chosen strategy;
- make normalization state a versioned persistent object. It is currently stored in non-persistent buffers and reserialized through parallel checkpoint metadata;
- explicitly mark tensor spaces (`physical`, `local_normalized`, `global_normalized`) so flux/interface diagnostics cannot mix units silently;
- decide whether Stage-B fine-tuning of Stage A is supported. If not, remove the option from production config; if yes, test optimizer/checkpoint behavior.

### Data layer

Strengths:

- compatibility fallbacks for older HDF5 schemas are valuable;
- lazy per-worker HDF5 reopening through `__getstate__` is appropriate;
- feature-name lookup is preferable to relying solely on fixed indices;
- masks and raw/processed interface targets are retained for evaluation.

Revision needs:

- `_data/channelthermal_datasets.py` and `_data/local_module_datasets.py` have identical implementation bodies; keep one authoritative implementation, then split it by actual responsibility;
- move legacy HDF5 fallbacks into a versioned `H5SchemaAdapter`;
- replace nested untyped samples with dataclasses/protocols and task-specific collators;
- filter convergence when requested, instead of only reporting convergence counts;
- inspect schema consistency across all cases rather than inferring optional datasets solely from the first case;
- centralize material, port, local-parameter, and interface feature construction, which is currently repeated across datasets and coupling code;
- cache local disk query coordinates/masks and avoid loading full grids for point training;
- make structure-target loading comparison-only;
- record normalization provenance and prevent fitting transforms on validation/test splits;
- add explicit file closing/context-manager support for long-lived processes;
- remove the redundant `indices` reconstruction and the unused `_choose_points()` wrapper after migration.

### Training, evaluation, and comparison

Strengths:

- checkpoint payloads contain extensive model, normalization, feature-name, and provenance metadata;
- losses cover field, local temperature, interface, port supervision, smoothness, global consistency, and generic organizer regularization;
- evaluation produces physical-field, local/interface, routing, organization, and plan artifacts;
- comparison uses the same selected case indices for all models and records a manifest.

Revision needs:

- move duplicated field loss into the retained training-loss module;
- unify global/local run-ID resolution and checkpoint selection rules;
- honor checkpointing config and save atomically so comparison never reads a partially written file;
- default comparison to fail closed rather than silently substituting a different “best” checkpoint; allow fallback only with an explicit flag;
- avoid loading a checkpoint multiple times during retry/model construction;
- make dataset/schema fingerprints mandatory in multi-model comparison;
- preserve input model order in summaries/figures rather than alphabetically resorting labels;
- report physical and normalized metrics with explicit units/statistics; standardized aggregate relative L2 can be unstable when the normalized target norm is small;
- share plotting and error-metric implementations between local/global evaluators;
- move evaluation logic to library services so `compare_models.py` does not import from the `evaluate.py` executable;
- align local evaluator's default saved root with `train_local.py`'s configured output root or require the run directory explicitly.

### Helpers and visualization

The helpers fall into four different categories that should not share one generic folder:

1. reusable tensor/runtime utilities: device transfer, seeding, AMP, masking, small layers;
2. ChannelThermal metrics and visualization;
3. experiment diagnostics and hypergraph-plan export;
4. checkpoint/config/path compatibility.

The current `model_utils.py` is a broad grab bag spanning all four concerns. Split it by responsibility, keep path resolution out of model libraries, and ensure plotting imports remain outside training/model import paths.

## Explicit redundancy and dead-code inventory

Static in-tree reference analysis identified the following cleanup candidates. Public use outside this tree must be checked before deletion.

### Exact or near-exact duplication

- `_data/channelthermal_datasets.py` and `_data/local_module_datasets.py`: identical implementation apart from the module docstring.
- `train.field_loss()` and `_helpers/training_losses.weighted_field_mse()`.
- local plot functions in `evaluate_local.py` and `_helpers/local_module_plots.py`.
- `error_metrics()` in `evaluate_local.py` and `_helpers/evaluation_plots.py`.
- `FourierEncoder`/`FourierFeatures` in `model_utils.py`, `honf_core.py`, and `honf_decoder.py`, with inconsistent frequency conventions.
- MLP/LazyMLP implementations in `model_utils.py`, `honf_core.py`, and `honf_decoder.py`.
- fixed-theta token construction in local model/coupling paths.
- material/local-parameter and teacher-port feature construction in data and coupling modules.
- thin checkpoint functions in `_helpers/checkpointing.py` duplicate direct calls in the trainers.

### No current in-tree caller

- all functions in `_helpers/checkpointing.py`;
- both functions in `_helpers/local_module_plots.py`;
- `_helpers/training_losses.weighted_field_mse()`;
- `model_utils.masked_mse()`, `masked_softmax()`, and `save_loss_curve()`;
- `channelthermal_input_adapter.feature_metadata()`;
- `model_local.build_local_model_from_config()` beyond package export;
- `evaluate_local.l2_error()`;
- `compare_models.checkpoint_dataset_config()`;
- `CaseConfig` and `AblationConfig` in `honf_types.py`;
- `UnifiedHypergraphNeuralField`, which is only an alias/export.

### Unused imports

- `Iterable` in both duplicated dataset modules and `model_utils.py`;
- `torch.nn.functional as F` in `honf_decoder.py` and `honf_organizer.py`;
- `json` in `compare_models.py`.

### Data produced or loaded but unused by the associated production path

- global-training `point_group`;
- full `module_internal_temperature` grid;
- `rms_field` in current evaluators;
- all `structure_targets` in training;
- local-training `local_grid` and `local_mask` under the current default;
- raw adapter inputs echoed in adapter/wrapper output dictionaries;
- large organizer token/state dictionaries when only predictions/scalars are needed;
- attention weights requested by the Stage-A cross-attention block and immediately discarded.

## File-by-file disposition

Every Python file in scope is listed below.

| File | Current role | Recommended disposition and principal finding |
|---|---|---|
| `_bootstrap_imports.py` | mutates `sys.path` for direct scripts | Remove after packaging; replace with installed console entry points. |
| `train.py` | global trainer, config resolver, losses, checkpointing, metrics, plotting | Split into library trainer plus CLI; fix false config controls, duplicate validation, zero-weight work, deterministic sampling, and checkpoint policy. |
| `train_local.py` | Stage-A trainer and mixed-dataset wrapper | Split CLI/trainer/data composition; replace mixed normalization and implement declared source weights. |
| `evaluate.py` | global checkpoint loading, inference, metrics, visualization, plan export | Split into inference service, metrics, artifacts, and CLI; use checkpoint transforms and prepare/decode chunking. |
| `evaluate_local.py` | Stage-A evaluation and duplicated plots/metrics | Reuse shared local evaluation library; consume checkpoint normalizer; support explicit source/schema selection. |
| `compare_models.py` | untracked multi-checkpoint comparison tool | Keep as an experiment CLI after extracting shared inference; fail closed on checkpoint substitution and separate solved from heuristic structure metrics. |
| `_data/__init__.py` | exports global dataset | Replace with explicit public data contracts/loaders after package split. |
| `_data/channelthermal_datasets.py` | all local/global HDF5 readers, normalization, transforms, sampling | Retain behavior but divide into schema adapter, transforms, local/global datasets, normalization, and collators. |
| `_data/local_module_datasets.py` | duplicate of the previous file | Delete after imports move to the authoritative data package. |
| `_helpers/__init__.py` | package marker | Remove generic helper namespace or expose a very small intentional public API. |
| `_helpers/model_utils.py` | paths, JSON, seeding, device/AMP, tensors, layers, plotting | Split into runtime, serialization, tensor ops, and NN layers; remove unused functions after external-use check. |
| `_helpers/checkpointing.py` | unused thin checkpoint wrappers | Merge into versioned checkpoint service or remove. |
| `_helpers/training_losses.py` | unused field loss duplicate | Make authoritative loss module and delete trainer copy, or remove. |
| `_helpers/evaluation_plots.py` | global/local metrics and ChannelThermal plots | Keep in `visualization/channelthermal`; separate pure metrics from plotting and remove hard-coded channel assumptions. |
| `_helpers/local_module_plots.py` | unused local plots duplicating evaluator | Merge with retained visualization module, then remove file. |
| `_helpers/honf_diagnostics.py` | scalar routing/organization diagnostics and regularization | Keep, but accept typed diagnostics and separate pure metrics from training regularization. |
| `_helpers/hypergraph_plan.py` | canonical static plan extraction/validation/serialization | Keep as a versioned artifact service; it has useful synthetic regression coverage. |
| `_helpers/organizer_viz_channelthermal.py` | large ChannelThermal organizer presentations | Keep optional visualization package; it must not be a model dependency. |
| `_helpers/routing_viz_channelthermal.py` | dense routing map export and plots | Keep optional diagnostics package; centralize JSON/NPZ artifact schemas. |
| `_models_core/__init__.py` | core exports, including legacy alias | Export only stable typed core API; remove alias after migration. |
| `_models_core/honf_types.py` | config and generic batch dictionary wrapper | Replace experiment remnants and `Any` fields with strict production config and tensor protocols/dataclasses. |
| `_models_core/honf_core.py` | encoders plus organize/decode seam | Keep and simplify; fix per-batch env coordinates and duplicate feature/layer implementations. |
| `_models_core/honf_organizer.py` | module/environment-to-hyperedge organization | Keep as core; make geometry policy explicit and return typed minimal state. |
| `_models_core/honf_decoder.py` | 12 decoder variants, routing, pairwise/global/direct/near contexts | Retain validated enhanced path; move domain boundary features out and relocate ablations to experiments. |
| `_models_channelthermal/__init__.py` | ChannelThermal public exports | Keep a small adapter/model API after package rename. |
| `_models_channelthermal/channelthermal_config.py` | wraps core and flattened ChannelThermal settings | Replace with strict nested schema shared by CLI/checkpoint/model construction. |
| `_models_channelthermal/channelthermal_input_adapter.py` | physical inputs to module/global features | Keep as domain adapter; validate schemas and remove unused metadata helper or make it part of the schema. |
| `_models_channelthermal/channelthermal_environment.py` | ChannelThermal env grid/features | Keep and cache; remove redundant columns if validation permits. |
| `_models_channelthermal/channelthermal_full_model.py` | 545-line orchestration and legacy output contract | Decompose around `prepare_case`/`decode`; move legacy input/output adaptation to compatibility layer. |
| `_models_channelthermal/local_coupling.py` | port prediction, Stage-A attachment/normalization, physics correction, refinement, fusion | Keep domain-specific behavior; split strategies, type spaces, validate settings, and run active modules only. |
| `_models_channelthermal/internal_fallback_heads.py` | global-only comparison fallback for local/internal outputs | Move to experiments/ablation and instantiate only when selected. |
| `_models_local/__init__.py` | Stage-A exports | Keep only stable model/config API; compatibility loader belongs elsewhere. |
| `_models_local/model_local.py` | copied Stage-A Perceiver-like surrogate | Preserve behavior through migration; remove discarded attention/aux work and formalize port/local schemas. |
| `_tests/__init__.py` | describes script-style tests | Replace with normal test package and markers for data/checkpoint integration tests. |
| `_tests/smoke_global_modes.py` | external-data smoke of four wrapper modes | Convert to pytest integration test; provide a tiny synthetic fixture and separately mark real-data coverage. |
| `_tests/test_hypergraph_plan_stability.py` | synthetic plan canonicalization/round-trip | Keep and convert to directly discoverable pytest functions. |
| `_tests/test_local_checkpoint_compat.py` | strict test against a dated local checkpoint path | Keep as optional compatibility test using an artifact fixture/version, not a hard-coded run directory. |
| `_tests/test_newhonf_hardening.py` | decoder efficiency, frozen local behavior, mechanism paths, checkpoint round-trip | Keep and split into focused unit/integration tests; extend to organizer count and chunked prepare/decode equivalence. |

## Proposed target repository structure

```text
pyproject.toml
src/channelthermal_honf/
    core/
        config.py
        contracts.py
        encoders.py
        organizer.py
        decoder.py
        model.py
    domains/channelthermal/
        schema.py
        input_adapter.py
        environment.py
        query_features.py
        model.py
        coupling/
            ports.py
            local_surrogate.py
            flux.py
            refinement.py
    local_model/
        config.py
        model.py
    data/
        contracts.py
        normalization.py
        sampling.py
        hdf5_schema.py
        global_dataset.py
        local_dataset.py
        collate.py
    training/
        global_trainer.py
        local_trainer.py
        losses.py
        checkpoints.py
        metrics_log.py
    evaluation/
        inference.py
        metrics.py
        comparison.py
        artifacts.py
        hypergraph_plan.py
    visualization/
        fields.py
        local.py
        organizer.py
        routing.py
    compatibility/
        checkpoint_v1.py
        hdf5_v1.py
        legacy_api.py
    cli/
        train_global.py
        train_local.py
        evaluate_global.py
        evaluate_local.py
        compare.py
configs/
    production/
    experiments/
tests/
    unit/
    integration/
    compatibility/
```

The core package should depend only on PyTorch and core contracts. It must not import ChannelThermal, HDF5, plotting, path-resolution, CLI, or checkpoint compatibility code.

## Proposed stable contracts

Use explicit dataclasses or equivalent typed structures instead of nested `Dict[str, Any]`.

```text
ChannelThermalCaseInputs
    flow: re [B,1], u_in [B,1]
    modules: centers [B,M,2], heat [B,M], present [B,M], materials [B,6]
    geometry: per-case domain description
    interface: optional physical port/interface inputs

GlobalTargets
    query_xy [B,Q,2]
    field [B,Q,F]
    point_weight [B,Q]

LocalInputs
    module_params [N,7]
    ports [N,P,5]
    internal_query [N,L,2]

OrganizedCase
    global_token [B,D]
    module_tokens [B,M,D]
    env_tokens [B,E,D]
    assignments/masses/geometry needed by the decoder

PreparedChannelThermalCase
    organized: OrganizedCase
    local_prediction: optional typed local result
    predicted_ports: optional typed port result
    schema/normalization provenance

FieldPrediction
    values [B,Q,F]
    optional ScalarDiagnostics
    optional DenseRoutingMaps
```

Every transformed tensor should carry or be governed by an explicit space contract. At minimum, APIs and field names must distinguish physical, global-normalized, and local-normalized values.

## Revision sequence and exit gates

### Phase 0: freeze the baseline

Actions:

- select one production profile and one global-only profile;
- record dataset/checkpoint hashes and exact resolved config;
- save golden outputs for a tiny set of cases in predicted mode;
- capture organizer tensors, local outputs, final fields, and backward gradients;
- convert current regression scripts to callable tests without changing model behavior.

Exit gate: current checkpoints reproduce golden outputs within documented tolerances on CPU and GPU where available.

### Phase 1: correct data/config semantics

Actions:

- implement one training-fitted normalizer for mixed Stage A;
- drive inference transforms from checkpoint metadata and validate dataset schemas;
- make random point sampling reproducible;
- implement `require_converged` filtering;
- introduce strict versioned config validation;
- fix the missing default global config path;
- honor checkpointing and dataset-source weight settings;
- make decoder mode semantics single-authority and test them;
- suppress solved-target metrics for heuristic structure targets.

Exit gate: normalization round-trip, mixed-source consistency, config-effect, convergence-filter, deterministic-sampling, and comparison-label tests all pass.

### Phase 2: establish package and data contracts

Actions:

- create installable package/console entry points;
- replace bootstrap path mutation and executable-to-executable imports;
- create typed case, local, organizer, prepared-case, and prediction contracts;
- add versioned HDF5 and checkpoint adapters;
- project only task-required data in each collator.

Exit gate: training/evaluation entry points run from outside the repository root, legacy checkpoints load through the compatibility adapter, and type/shape validation fails clearly on malformed inputs.

### Phase 3: consolidate duplicates without changing behavior

Actions:

- delete the duplicate dataset module after import migration;
- unify error metrics, local/global plots, field loss, checkpoint helpers, Fourier encoders, MLPs, fixed-theta construction, and feature construction;
- remove unused imports and no-caller functions after external-use search;
- move experiment-only config/classes/modes out of production modules.

Exit gate: golden forward/backward outputs and compatibility tests still pass; static duplicate/dead-code checks have an explicit allowlist only for compatibility code.

### Phase 4: decompose the wrapper and optimize inference

Actions:

- implement `prepare_case()` and `decode()`;
- make chunked evaluation reuse prepared case state;
- make diagnostics opt-in;
- instantiate only selected fallback/coupling strategies;
- gather active modules for Stage A;
- cache fixed environment/local geometry;
- reuse predicted validation when equivalent and skip zero-weight loss branches;
- reduce plotting/checkpoint I/O cadence.

Exit gate: chunked and unchunked outputs match, organizer/Stage-A call-count tests prove one case preparation, peak memory and wall-clock benchmarks improve, and standard forward retains its one-final-decode invariant.

### Phase 5: finish core/domain separation

Actions:

- move ChannelThermal boundary features and rectangular geometry policies out of the decoder;
- define the supported generic geometry protocol;
- make per-case geometry consistent throughout encoding, routing, and probes;
- trim production decoder to the validated enhanced path;
- keep ablations under `experiments` with explicit strategy selection;
- replace legacy result dictionaries with a compatibility adapter.

Exit gate: a minimal synthetic second domain can use the core without importing ChannelThermal code, while the ChannelThermal golden outputs remain equivalent.

### Phase 6: remove compatibility scaffolding deliberately

Actions:

- publish a checkpoint/data migration command and schema changelog;
- define a support window for v1 checkpoints and HDF5 files;
- remove legacy aliases, silent defaults, dead fields, old modes, and copied state-layout constraints only after migration fixtures exist;
- archive historical configs with result provenance rather than leaving them in the production config directory.

Exit gate: all supported artifacts either migrate successfully or fail with a precise version error; production modules contain no legacy-only branches.

## Required test matrix for the revised repository

### Unit tests

- masks and row normalization for `A_mh`/`A_eh`, including zero active modules;
- organizer permutation properties and hypergraph-plan canonicalization;
- each retained decoder strategy and every config combination validation;
- checkpoint normalizer round-trip and mismatched-dataset rejection;
- mixed-source normalization using one transform;
- deterministic point sampling across epochs/workers;
- physical/local/global normalization conversions for temperature and flux;
- active-module gather/scatter equivalence;
- port curriculum and ratio validation;
- per-case environment geometry and boundary features;
- metric masking and solved-versus-heuristic structure labels.

### Integration tests

- local training step with forward/backward/checkpoint resume;
- global predicted-mode training step with and without local coupling;
- teacher and mixed debug modes;
- prepare/decode versus legacy full-forward equivalence;
- query chunk-size invariance;
- self-contained global checkpoint load without external Stage-A path;
- CLI invocation from an arbitrary working directory;
- comparison across models with identical dataset/schema fingerprints.

### Compatibility tests

- strict or mapped load of each supported legacy checkpoint version;
- old HDF5 schema fallbacks, including `h_effective` reconstruction;
- golden outputs for the selected existing Stage-A and global checkpoints;
- explicit failure for unknown config keys, unsupported schemas, and incompatible field orders.

## Deletion safety checklist

Do not remove a candidate merely because static analysis finds no in-tree caller. Before deletion:

1. search the entire repository, notebooks, scripts, and documented commands;
2. identify whether inverse-design or external workflows import it;
3. add or confirm a replacement API;
4. preserve checkpoint/data migration behavior where state layout is involved;
5. run golden forward, backward, inference, artifact, and compatibility tests;
6. record the removal in a migration manifest.

In particular, do not delete the fallback heads, old decoder modes, legacy result keys, or copied Stage-A state layout until their role in historical experiment reproduction and checkpoint loading has been replaced by explicit experiment/compatibility modules.

## Recommended definition of “clean and transferable”

The revision is complete when:

- the HONF core can be installed and imported without ChannelThermal or path bootstrapping;
- one strict config path constructs the same model for training, evaluation, and checkpoint loading;
- every tensor crossing a subsystem has a documented shape, meaning, mask, and normalization space;
- a case is prepared once and decoded for arbitrary query chunks;
- production modules contain only the validated path, while experiments and compatibility are separately named;
- current supported checkpoints and datasets are versioned and reproducible;
- unused and duplicate implementations are removed;
- a synthetic unit suite runs without repository datasets, and marked integration/compatibility suites validate real artifacts;
- metrics and comparison artifacts state their physical/normalized units and whether targets are solved or heuristic;
- a second domain can supply its own input, environment, geometry, and query-feature adapters without editing the HONF organizer or decoder.
