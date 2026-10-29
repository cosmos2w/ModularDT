# Adding a physical case

A case translates physical inputs and targets into a reusable HONF model and interprets its outputs. The case owns data and physical semantics; the core owns generic interaction, organization and field-prediction mechanisms. `M` and `E` identify input roles, not separate output responsibilities. A new model should predict the requested physical fields jointly, with prescribed geometry, materials, boundary conditions and controls entering their owning records before interaction.

## Data and adapter contract

Define a documented schema for stable case and original-member identities, variable-size module and environment records, coordinates, physical measures, prescribed controls, valid masks, query locations and target channels. Fixed-size padded batches must carry masks; ragged arrays need explicit offsets or lengths. Preserve native geometry and physical units so outputs can be mapped back to real coordinates and meaningful observables. Targets and post-solve diagnostics must not enter the model as inputs.

Define TRAIN, validation and test membership before fitting, accounting for repeated geometry or other dependent cases. Fit normalization only on TRAIN and save the normalizer with the checkpoint. Use deterministic sampling keyed to the chosen seed, epoch and case identity. The case owns inverse normalization, target extraction, physically meaningful losses, fieldwise metrics and plots. Some historical models use earlier normalization conventions; load their stored state faithfully rather than silently replacing it.

The historical generic batch container is [`BatchData`](../src/honf_forward_core/config.py), containing module centers, presence, features, global context, query coordinates and optional environment, geometry, measure and identity fields. A model family may use a more specialized adapter instead; follow that family's actual interface. For a concrete independent example, electro-thermal's `HETMFTask` binds `HETMFDataset` and `ElectroThermalHONFAdapter` to `honf_runtime.field_training`; its [handoff](../Case_ElectroThermal/docs/COWORKER_HANDOFF.md) describes its HDF5 schema and joint fields.

## Managed plugin workflow

The generic historical `train.py` and `evaluate.py` entry points discover a `CasePlugin` through a factory named in the case configuration, such as `channelthermal.plugin:create_plugin`. The runtime protocol lives in [`case_protocol.py`](../src/honf_runtime/case_protocol.py). A plugin supplies `case_id`, `display_name`, `version`, `validate_config`, `inspect_launch`, `train`, and `evaluate`. Launch inspection validates external resources; training receives the composed configuration, common `WorkflowRequest`, and reserved run directory; evaluation receives the common request and case-specific arguments.

The plugin package owns its manifest and readers, `BatchData` mapping, case configuration, dataset-derived dimensions, physical wrappers and interpretation, losses and post-processing, optional local modules, and case-specific checkpoint metadata or migration. Reusable core code must not import a case, and generic dispatch must not branch on case IDs. A tiny synthetic plugin should demonstrate installation, dry-run, batch adaptation, inference and loss without changes to core or runtime.

A case-owned optional local sub-module can describe itself with `LocalModuleSpec`, including input, port, query and target schemas; latent width; datasets; model and checkpoint factories; workflows; coupling adapter; and freeze/embedding policy. It is optional: a new case does not need Thermal's Stage-A surrogate or its output law.

The standalone electro-thermal `python -m hetmf` workflow uses the shared field-training engine and its own task interface; it is not automatically a `CasePlugin` dispatch through `train.py`. Likewise, research-family commands in [`honf_research.py`](../tools/honf_research.py) have their own explicitly documented provider and checkpoint contracts. Choose the integration that matches the selected model, and preserve old identifiers and checkpoints for existing workflows.

## Evidence and inverse design

Document the case schema and example location map, and test resource validation, splits and normalization, fixed and ragged masks, adapter and field shapes, reproducible sampling, compatible checkpoint loading, and evaluation artifacts for the workflows actually supported. A brief synthetic forward/backward pass can establish interface health; it cannot establish physical accuracy or convergence. Keep machine paths, datasets, trained weights and generated output outside Git.

Expose native-unit field errors and any physical residuals separately. Every reconstruction field plot must visibly annotate fieldwise L2 (identify relative L2) or RMSE on the error panel, or on the reconstruction panel when no error panel is present; use the plotted physical mask and state units. If no physical reference exists, label the error unavailable. Measure selected records, executed pair calculations, preparation cost and complete runtime before claiming sparse work or speed savings. A learned relation is not by itself physical causality.

The generic inverse contracts describe requests, candidate designs and verification results; the case owns geometry constraints, physical objectives and a verifier capable of checking candidates against its physical reference. A plausible sampled candidate or a surrogate score alone is not a verified design. State the exact model, checkpoint, dataset and validation scope for any inverse result.
