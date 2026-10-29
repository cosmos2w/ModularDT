# Case-local artifacts

This directory is the portable landing place for external, untracked model
artifacts used by the maintained ThermalChannel configuration.

The default forward profile expects:

```text
thermal_disk.pt
```

Create it by training the local module and copying or symlinking the selected
checkpoint here, or pass its location directly to `train.py` with
`--local-checkpoint`. Checkpoints are intentionally ignored by Git.
