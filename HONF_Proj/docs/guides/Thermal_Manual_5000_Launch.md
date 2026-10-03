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

Preparation rejects missing or incomplete calibration rather than recalibrating
at epoch1001. Every parent must preserve the explicit policy1→2 amendment at
e100/101, five finite response samples, their saved gradient norms and bounded
median coefficient. An H parent additionally needs completed structural
calibration v2 over all five training-M strata, consistent sample provenance,
norms and coefficients, and selection state e1000/horizon5000. Dense and B-fine
legitimately have no organizer selection or structural calibration. The actual
B-fine e500 metadata passed calibration and native-configuration review;
the complete launcher correctly rejects it as an e1000 parent. This partial
software check does not verify a future finalist recipe. The actual B-fine
exact1000 checkpoint subsequently passes the complete read-only native
configuration/calibration/optimizer/RNG guard with all 120 active optimizer
states. No profile or workspace is created and no 5000 training is launched
by that check. Actual selected H finalist preparation and small-query replay
remain required before the final manual handoff.

For fresh initialization, replace `--parent-checkpoint ...` with `--fresh --seed 0` and choose an unused run ID. Preparation writes one profile; the ordinary trainer reserves its new workspace when launched. Fresh seed0 reproduces matched initial weights in a separate run. It is not an independent-seed replication claim. The objective remains policy1 through epoch100 and policy2 from epoch101, matching the final screened continuation. Nonzero fresh replication seeds remain deferred.

No launcher invocation chooses architectures, starts a subprocess trainer, or automatically continues another arm. For later recovery, pass the new workspace's top-level native `latest_model.pt` or `epoch_NNNN_model.pt` to the ordinary trainer and keep its generated profile. The native workflow places resumed outputs beside the selected checkpoint; these top-level paths preserve the intended workspace. Evaluations retain the original parent lineage.
