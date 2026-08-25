# Trained results

Training creates case- and model-family-specific run directories here.  Run
artifacts are local/generated data and are ignored by Git; only this contract
and empty directory markers belong in the source repository.

```text
<CaseID>/
├── HONF_Forward_Runs/
├── HONF_Inverse_Runs/
├── Local_Module_Runs/<LocalModuleID>/
└── Baselines/<BaselineID>/Runs/
```

New managed forward/local runs use the canonical artifact tree below. Legacy
root checkpoint names remain available for command compatibility; plots and
post-processing outputs are written only once.

```text
Run_<id>_<timestamp>_<name>/
├── checkpoints/
├── metrics/
├── plots/{training,diagnostics}/
├── evaluations/single_case/<case>_<timestamp>/
│   ├── summary.json
│   ├── evaluation_manifest.json
│   └── {fields,organization,routing,topology,plans,metrics,arrays,diagnostics}/
├── comparisons/
├── configs/
├── environment/
├── logs/
└── run_manifest.json
```
