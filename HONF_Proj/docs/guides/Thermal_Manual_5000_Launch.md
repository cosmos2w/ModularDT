# Provisional manual Thermal 5000-epoch launches

Choose the final one or two architecture names only after the common 1000-epoch review. No finalist has been selected by this guide, and no 5000-epoch training has been launched. The launcher prepares one explicitly selected profile per invocation.

Run from the ModularDT environment with the native packages available:

```bash
export PYTHONPATH="$PWD/src:$PWD/Case_ThermalChannel/src"
```

The examples below assume the current directory is `HONF_Proj`. Replace `FINALIST`, `NEW_RUN_ID`, and `EXACT_E1000_CHECKPOINT` after reviewing saved physical metrics, responses, actual hard/soft work, and the retained epoch-1000 state.

For continuation, first validate without writes:

```bash
python tools/thermal_campaign_long_run.py \
  --profile src/config_core/forward/thermal_campaign/FINALIST_e1000.json \
  --run-id NEW_RUN_ID --physical-gpu 1 \
  --output-root /data/wanglz/ModularDT/thermal_manual5000 \
  --parent-checkpoint EXACT_E1000_CHECKPOINT --dry-run
```

Add `--prepare` in place of `--dry-run` to create one profile and a new standard RunStore workspace. The unchanged epoch-1000 checkpoint is copied into that workspace, and its parent path is recorded in the manifest. The finished parent folder remains unchanged. Run the printed `validate_command`, inspect its resolved data/local-surrogate/output paths, and then use the printed `launch_command` only after manual authorization. Select physical GPU1 or GPU2 explicitly; both appear to the trainer as logical `cuda:0`.

Continuation runs epochs1001–5000 with saved weights, optimizer, RNG, calibration, curriculum and the unchanged absolute 5000-epoch horizon. Physical denominator policy2 is already active and must match the parent. Preparation compares the complete native model configuration, including CaseThermal physics and Stage-A binding, resolving auto dimensions from a one-query metadata sample. Only duration and run placement may change. It uses the existing trusted checkpoint loader and performs no predictive replay. Tests check state preservation using small synthetic checkpoints and never construct a training optimizer. A real retained epoch-1000 parser/load and small-query replay remains required when that checkpoint exists.

For fresh initialization, replace `--parent-checkpoint ...` with `--fresh --seed 0` and choose an unused run ID. Preparation writes one profile; the ordinary trainer reserves its new workspace when launched. Fresh seed0 reproduces matched initial weights in a separate run. It is not an independent-seed replication claim. The objective remains policy1 through epoch100 and policy2 from epoch101, matching the final screened continuation. Nonzero fresh replication seeds remain deferred.

No launcher invocation chooses architectures, starts a subprocess trainer, or automatically continues another arm. For later recovery, pass the new workspace's top-level native `latest_model.pt` or `epoch_NNNN_model.pt` to the ordinary trainer and keep its generated profile. The native workflow places resumed outputs beside the selected checkpoint; these top-level paths preserve the intended workspace. Evaluations retain the original parent lineage.
