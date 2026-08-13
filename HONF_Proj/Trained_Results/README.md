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
