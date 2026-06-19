# Staged Migration Manifest

This manifest was written before revising the copied implementation. Prompt 1
created a standalone global-field HONF path under `src_new/` and
`Configs_new/`; Prompts 2 and 3 added the copied Stage-A local surrogate stack
and ChannelThermal-specific physical coupling without importing old source at
runtime.

Pre-training hardening revisions:

- `src_new/_models_core/honf_core.py` now exposes `encode_and_organize()` and
  `decode_queries()` so ChannelThermal can avoid discarded pre-fusion field
  decodes.
- `src_new/_models_channelthermal/channelthermal_full_model.py` uses full
  organizer dictionaries for final, port-global, and refinement decoder calls;
  reduced legacy organizer aliases are created only for outputs/plots.
- `src_new/_models_channelthermal/local_coupling.py` keeps a frozen attached
  local surrogate in eval mode even when coupling heads are trained.
- `src_new/train.py` resolves authoritative nested config sections, clips
  gradients, saves concrete `config_resolved.json`, and embeds local surrogate
  config/normalization/provenance in global checkpoints.
- `src_new/evaluate.py` loads embedded local surrogate metadata before loading
  the global state dict and validates critical state-dict mismatches.
- Smoke and regression utilities are grouped under `src_new/_tests/`.

Diagnostics and inverse-readiness revisions:

- `src_new/_helpers/honf_diagnostics.py` logs scalar HONF health metrics during
  training without retaining dense query-routing maps.
- `src_new/train.py` deprecates ambiguous legacy organizer entropy weights in
  favor of disabled-by-default generic `loss.organizer_regularization`
  controls.
- `src_new/evaluate.py` uses predicted mode as the primary organization,
  routing, and hypergraph-plan export source when `--local-port-condition-mode
  both` is requested; teacher outputs are retained with a suffix.
- `src_new/_helpers/hypergraph_plan.py` exports a canonical schema-versioned
  inverse-ready plan with loader and validator utilities.
- `Configs_new/train_global_honf_validated_core.json` preserves the exact
  validated enhanced-HONF sandbox core settings and adds the full local
  coupling stack.
- `Configs_new/train_global_honf_old_parity.json` mirrors old comparison data,
  normalization, seed, optimizer, local-checkpoint, and Smooth L1 loss choices
  where practical.

| Source file copied | Destination file | Reason copied | Required revisions |
| --- | --- | --- | --- |
| `Unified_Forward_Model_0/src/unified_types.py` | `src_new/_models_core/honf_types.py` | Validated HONF configuration and batch container. | Rename to HONF terminology, add CORE HONF docstring, use relative imports, add optional generic environment coordinates/features to `BatchData`. |
| `Unified_Forward_Model_0/src/unified_organizer.py` | `src_new/_models_core/honf_organizer.py` | Validated A_me/A_mh/A_eh organizer, mechanism descriptors, and hyperedge diagnostics. | Rename imports to local HONF modules and add CORE HONF docstring. |
| `Unified_Forward_Model_0/src/unified_decoder.py` | `src_new/_models_core/honf_decoder.py` | Validated routing-first hyperedge field decoder, optional c_H value context, H-routed pairwise kernel, global and near contexts. | Rename imports to local HONF modules, add CORE HONF docstring, keep direct module/environment decoder disabled by config. |
| `Unified_Forward_Model_0/src/unified_model_core.py` | `src_new/_models_core/honf_core.py` | Validated standalone HONF module/env encoders plus organizer/decoder composition. | Rename classes/imports to HONF names, add CORE HONF docstring, minimally accept generic external environment coordinates/features from a wrapper. |
| `1_Demo_ChannelThermal/src/_helpers_forward/channelthermal_datasets.py` | `src_new/_data/channelthermal_datasets.py` | Stable packed HDF5 dataset reader preserving split logic, normalization stats, point sampling, and batch keys. | Rename helper imports to `src_new._helpers.model_utils`, add CHANNELTHERMAL-SPECIFIC docstring, keep HDF5 behavior intact. |
| `1_Demo_ChannelThermal/src/_helpers_forward/channelthermal_model_utils.py` | `src_new/_helpers/model_utils.py` | Stable path, normalization, checkpoint-load, tensor/device, AMP, MLP, and plotting utilities. | Add CHANNELTHERMAL-SPECIFIC docstring and update `DEMO_ROOT` for the `src_new` location. |
| `1_Demo_ChannelThermal/src/_helpers_forward/channelthermal_model_utils.py` | `src_new/_helpers/checkpointing.py` | Checkpoint and run-directory support needed by new training/evaluation scripts. | Provide a compact standalone checkpoint API for NewHONF runs. |
| `1_Demo_ChannelThermal/src/_helpers_forward/organizer_viz_channelthermal.py` | `src_new/_helpers/organizer_viz_channelthermal.py` | Existing presentation organizer visualizations expected by legacy evaluation. | Add CHANNELTHERMAL-SPECIFIC docstring and keep visualization inputs compatible with new organizer aliases. |
| `1_Demo_ChannelThermal/src/evaluate.py` | `src_new/evaluate.py` | Legacy CLI, checkpoint lookup, case selection, query chunking, field quicklook, internal/interface plots, and organizer output behavior. | Revise imports/model loading for `ChannelThermalHONFModel`, move reusable plot helpers to `src_new/_helpers/evaluation_plots.py`, and support predicted local-port evaluation outputs. |
| `1_Demo_ChannelThermal/src/train.py` | `src_new/train.py` | Legacy training behavior for Run_ID normalization, timestamped run folders, checkpoints, metrics, summaries, and loss curves. | Revise to train `ChannelThermalHONFModel` with field, internal, interface, port, port-global, and predicted autonomous validation losses while keeping only weak generic organizer regularization by default. |
| New compatibility adapter | `src_new/_models_channelthermal/channelthermal_input_adapter.py` | Maps physical ChannelThermal inputs into generic HONF tensors. | Document feature columns; avoid unrestricted coordinate copies in module features. |
| New environment adapter | `src_new/_models_channelthermal/channelthermal_environment.py` | Builds ChannelThermal wall/inlet/outlet environment coordinates/features outside the reusable core. | Keep wall/inlet/outlet semantics out of CORE HONF. |
| New full-model wrapper | `src_new/_models_channelthermal/channelthermal_full_model.py` | Provides legacy forward signature and output dictionary while using the standalone HONF core. | Return internal/interface/port tensors from either the copied local surrogate path or global fallback heads, plus organizer aliases expected by evaluation. |
| New fallback heads | `src_new/_models_channelthermal/internal_fallback_heads.py` | Preserves old global fallback internal/interface head shapes for comparison runs when local surrogate is disabled. | Keep available through `internal_prediction_mode="global_head"`. |
| New config template | `Configs_new/train_global_honf_template.json` | Production Prompt-3 training configuration for standalone enhanced HONF plus ChannelThermal coupling. | Add `_note_*` explanations and separate core HONF, local coupling, physical correction, loss, curriculum, checkpointing, and path blocks. |
| New diagnostics helper | `src_new/_helpers/honf_diagnostics.py` | Scalar HONF collapse/routing/context diagnostics and generic anti-collapse regularization. | Keep outputs scalar during training; avoid dense routing tensors unless evaluation explicitly requests them. |
| New plan helper | `src_new/_helpers/hypergraph_plan.py` | Canonical inverse-ready static organization export. | Add schema metadata, canonical H ordering, loader, validator, summary, and exclude query-dependent routing/module tokens. |
| New routing helper | `src_new/_helpers/routing_viz_channelthermal.py` | Optional query-dependent routing maps for NewHONF evaluation. | Save routing NPZ/JSON and opt-in PNGs only when `--return-routing-maps` is enabled. |
| New validated config | `Configs_new/train_global_honf_validated_core.json` | Reproduce the validated enhanced-HONF core settings inside the standalone NewHONF stack. | Add local coupling/physical correction sections without changing the copied core hyperparameters. |
| New parity config | `Configs_new/train_global_honf_old_parity.json` | Compare against the old ChannelThermal global model under matching data/normalization/loss settings. | Keep NewHONF H-routed decoder; do not restore old direct-memory attention. |
