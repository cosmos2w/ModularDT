# ThermalChannel case plugin

This installable package owns every steady heated-channel concern: packed-HDF5
readers and normalization, physical-to-core adapters, environment features,
the coupled model wrapper, Stage-A thermal-disk surrogate, Stage-B losses,
Robin-flux correction, training workflows, metrics, plots, routing diagnostics,
hypergraph-plan export, and multi-model comparison.

The parent project owns the case-neutral HONF organizer/decoder, configuration
composition, plugin discovery, run store, confirmation, and generic entry
points. This dependency direction is intentional.

## Lifecycle

1. Map `thermal_disk_local_v1` and `thermal_channel_global_v1` in
   `Dataset/dataset_locations.local.json`.
2. Train `local_module_thermal_disk.json` through the parent `train.py`.
3. Select a Stage-A checkpoint. Supply it with `--local-checkpoint` or install
   it as `artifacts/thermal_disk.pt`.
4. Train either maintained global decoder profile.
5. Evaluate through the parent `evaluate.py`; post-processing is stored below
   the source run.

The global checkpoint embeds the frozen local surrogate and its normalization,
so later evaluation is self-contained. Teacher/mixed port conditioning remains
available for diagnostics; maintained global training uses autonomous predicted
ports.

## Package map

```text
src/channelthermal/
├── data/                 global/local datasets and normalization
├── local_surrogate/      Stage-A architecture and checkpoint contract
├── evaluation_tools/     plots, organizer views, routing, plan export
├── workflows/            case implementations called by the plugin
├── input_adapter.py      physical sample to generic BatchData
├── environment.py        channel boundary/environment tokens
├── local_coupling.py     Stage-A attachment and port construction
├── fallback_heads.py     no-surrogate internal/interface fallback
├── model.py              complete coupled case wrapper
├── config.py             resolved case/model configuration
├── workflows/train_*.py  case-specific objectives and training loops
├── resources.py          logical dataset registry and validation
└── plugin.py             generic runtime boundary
```

Use the root `README.md` for commands and `Dataset/PHYSICS_AND_DATA.md` for the
governing physics and exact tensor schema.

`local_surrogate/spec.py` exposes the machine-readable `LocalModuleSpec`: its
feature schemas, dataset IDs, latent width, factories/workflows, freeze policy,
and parent-checkpoint embedding policy. New case-specific sub-modules should
provide the same contract instead of adding special cases to the runtime.
