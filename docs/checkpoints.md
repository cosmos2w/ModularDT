# Checkpoint contract

Training may write these selectors: `best`, `best_by_field_mse`,
`best_by_temperature_mse`, `best_predicted`, and `latest`. A missing requested
selector is an error unless the specific comparison command is explicitly
given `--allow-checkpoint-fallback`.

ThermalChannel forward checkpoints retain model weights, the resolved
architecture, training normalizer statistics, epoch/optimizer/scaler state and
best metrics needed by resume, plus the frozen Stage-A model and its local
normalizer. Therefore global evaluation does not need the original external
Stage-A file. Local checkpoints retain their architecture and fitted
normalization contract.

Resume must use a compatible case, model family, architecture, feature/channel
schema, normalization, and dataset identity. It restores optimizer/scaler and
epoch state; it is not the same as `--local-checkpoint`, which either initializes
Stage A or selects the frozen dependency for a new global run.

PyTorch `.pt` files can execute pickle payloads during loading. This project
treats run checkpoints as trusted local artifacts; never load a checkpoint from
an untrusted source. Model state keys preserve reference names so historical
ThermalChannel checkpoints can be reconstructed through the compatibility
loader.

Current checkpoints declare `checkpoint_schema_version=1`, `case_id`,
`model_family`, and `workflow`; local artifacts also declare their module ID.
`tools/migrate_checkpoint.py` can add only this metadata to a trusted historical
checkpoint. Inspect first with `--dry-run`, then use an explicit new `--output`
path. Existing files are never overwritten and tensor names are never changed.
