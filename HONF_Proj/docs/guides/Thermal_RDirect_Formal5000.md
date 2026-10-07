# Manual R-direct full-TRAIN research recipe

**Authorized launch update, 2026-10-07.** The user subsequently requested the two fresh formal stages. Run3901 completed ordinary D-sep e5000 on GPU 0; Run3902 started fresh R-direct on GPU 2 with that exact flow endpoint frozen, passed its first-ten-epoch review and was left running toward thermal e5000. Actual run identities, tmux sessions, checkpoint retention and the first-ten-epoch evidence are recorded in the [formal launch record](../reports/_bk/20261007_035824Z_Thermal_RDirect_Formal5000_Launch_20261007.md). The startup-only measurements below describe the earlier readiness round and are preserved as historical evidence.

**Model and launch decision.** R-direct with a freshly trained D-sep flow reader is the preferred response-family research reference for a manually launched full-TRAIN run. R-group remains an independently trained control, Run3801 remains the incumbent, and this recipe does not claim general inverse-design or grouping superiority. During the earlier readiness round, the formal5000 training stages were not launched; its only new full-TRAIN fits were explicitly labelled startup benchmarks capped at three epochs. The subsequent authorized formal launch is recorded above.

## Bound recipe

The formal workflow identity is `formal_full_train_v1`: it binds all 600 original TRAIN case IDs from the packed-H5 catalog, fits one global normalizer on those same 600 cases only, and carries the identical dataset, normalizer, and validation bindings in the flow and thermal checkpoints. Fixed25_v1 remains the 150/22 development protocol; its exposed DEV22 panel is used only for the bounded startup checks described below.

| Stage | Fresh component and schedule | Per-epoch work | Validation and selection |
|---|---|---|---|
| D-sep flow | Fresh `ThermalFlowReader("D-sep")`, 5,000 epochs, 3e-4 through epoch 2,000, then cosine decay to 3e-6 | All 600 TRAIN cases; 1,024 fluid queries/case; effective batch 48, microbatch 8, 13 optimizer updates | Full run monitors canonical89 primary and original90 compatibility at the declared cadence; no thermal parameters are trained |
| R-direct thermal | Fresh R-direct weights, fixed exact D-sep e5000 partner, 5,000 epochs, same 3e-4/2,000-epoch hold/cosine-to-3e-6 schedule | All 600 TRAIN cases; 1,024 fluid queries/case, 32 material-temperature queries/module, surface stride 4, 128 qualified operator rows/case; effective batch 48, microbatch 8, 13 optimizer updates | Full run selects by the three-role standardized temperature selector on canonical89; it also records the common five-field standardized MSE and physical per-channel RMSE on canonical89 and original90 |

The R-direct objective averages standardized fluid-temperature, surface-temperature, and material-temperature reconstruction losses, adds the configured 0.05 q-proxy term, and adds the separately gradient-calibrated TRAIN response and qualified TRAIN-only discrete-balance terms. The response anchors remain the saved original-TRAIN families 0001, 0318, 0333, and 0348; the operator qualification is a consistency constraint on those saved TRAIN fields, not independent physical validation. The flow reader stays frozen during thermal fitting, and the fresh formal flow checkpoint must be exactly epoch 5000 before full formal thermal preparation or fitting is accepted.

The formal validation binding derives canonical89 by excluding training duplicate 0273 from the original90 test rows. Canonical89 is the primary thermal selector; original90 is retained as the compatibility panel. Their shared full-TRAIN normalizer is used for flow denormalization and the five-field standardized score. Startup checkpoints are a separate `startup_benchmark` identity, may reach only e3, and may use DEV22 only; the loader rejects startup/formal mixing and cannot promote startup weights into the formal5000 identity.

## Prepare and launch manually

These commands create a new, externally stored formal run and do not launch from this guide automatically. Use a unique output root for each manually authorized run; keep it outside tracked source paths. The `rtk` prefix follows this checkout's command policy.

```bash
PROJECT=/home/wanglz/Desktop/src/ModularDT/HONF_Proj
PYTHON=/home/wanglz/miniconda3/envs/ModularDT/bin/python
PARENT=/data/wanglz/ModularDT/thermal_development/response_refinement_20261005/ThermalChannel/HONF_Forward_Runs/Run_3801_20261005_214458_thermal_response_refinement25_h-add_v1/epoch_0500_model.pt
ATLAS=$PROJECT/diagnostics/generated/interactions/physical_response_atlas_20260926/families
OPERATOR=$PROJECT/diagnostics/generated/response_operator_20261006/qualification/operator_decision.json
FORMAL_RUN_ROOT=/data/wanglz/ModularDT/thermal_development/manual_RDirect_formal5000_20261007
FLOW=$FORMAL_RUN_ROOT/D-sep-formal5000
THERMAL=$FORMAL_RUN_ROOT/R-direct-formal5000
```

Prepare a new formal D-sep flow checkpoint. This writes an epoch-zero checkpoint and binding receipt only; it does not train.

```bash
rtk "$PYTHON" "$PROJECT/tools/thermal_formal_dsep_flow_fit.py" \
  --parent "$PARENT" \
  --profile-file "$PROJECT/src/config_core/forward/thermal_source_response/d-sep_full5000.json" \
  --output "$FLOW" --run-identity formal5000 --prepare-only
```

Start the full D-sep stage manually on GPU 0. Keep this launch running in one terminal. From a second terminal, read `active_process.json` and `history.json` to monitor it; create `stop_requested` to stop at the next completed epoch boundary. The script saves the latest optimizer state before acknowledging the request.

```bash
rtk env CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$PROJECT/tools/thermal_formal_dsep_flow_fit.py" \
  --parent "$PARENT" \
  --profile-file "$PROJECT/src/config_core/forward/thermal_source_response/d-sep_full5000.json" \
  --output "$FLOW" --run-identity formal5000 \
  --resume "$FLOW/latest_model.pt" --stop-after 5000 --device cuda:0
```

While that launch is still running, issue the stop request and read status from a second terminal:

```bash
rtk touch "$FLOW/stop_requested"
rtk cat "$FLOW/active_process.json"
```

Resume the same D-sep identity from its latest checkpoint after the process reports `stopped_resumable`. Keep the profile, run identity, source checkpoint, output path, and device-visible GPU mapping unchanged.

```bash
rtk env CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$PROJECT/tools/thermal_formal_dsep_flow_fit.py" \
  --parent "$PARENT" \
  --profile-file "$PROJECT/src/config_core/forward/thermal_source_response/d-sep_full5000.json" \
  --output "$FLOW" --run-identity formal5000 \
  --resume "$FLOW/latest_model.pt" --stop-after 5000 --device cuda:0
```

After D-sep reaches exactly e5000, prepare the fresh R-direct thermal stage. This seals its calibration, data bindings, source hashes, and operator decision; it does not train.

```bash
rtk "$PYTHON" "$PROJECT/tools/thermal_source_response_fit.py" \
  --parent "$PARENT" --flow-checkpoint "$FLOW/latest_model.pt" \
  --profile-file "$PROJECT/src/config_core/forward/thermal_source_response/r-direct_full5000.json" \
  --output "$THERMAL" --run-identity formal5000 \
  --atlas-directory "$ATLAS" --operator-decision "$OPERATOR" --prepare-only
```

Start R-direct manually on GPU 2 and keep the launch running in one terminal. From a second terminal, monitor `active_process.json`, `history.json`, and the 100-epoch validation files; request an epoch-boundary stop by creating `stop_requested` in the thermal run directory. Resume with the same recipe, exact e5000 flow partner, output directory, identity, profile, and device mapping.

```bash
rtk env CUDA_VISIBLE_DEVICES=2 "$PYTHON" "$PROJECT/tools/thermal_source_response_fit.py" \
  --parent "$PARENT" --flow-checkpoint "$FLOW/latest_model.pt" \
  --profile-file "$PROJECT/src/config_core/forward/thermal_source_response/r-direct_full5000.json" \
  --output "$THERMAL" --run-identity formal5000 \
  --atlas-directory "$ATLAS" --operator-decision "$OPERATOR" \
  --recipe "$THERMAL/formal_recipe.json" --resume "$THERMAL/latest_model.pt" \
  --stop-after 5000 --device cuda:0
```

While that launch is still running, issue the stop request and read status from a second terminal:

```bash
rtk touch "$THERMAL/stop_requested"
rtk cat "$THERMAL/active_process.json"
```

```bash
rtk env CUDA_VISIBLE_DEVICES=2 "$PYTHON" "$PROJECT/tools/thermal_source_response_fit.py" \
  --parent "$PARENT" --flow-checkpoint "$FLOW/latest_model.pt" \
  --profile-file "$PROJECT/src/config_core/forward/thermal_source_response/r-direct_full5000.json" \
  --output "$THERMAL" --run-identity formal5000 \
  --atlas-directory "$ATLAS" --operator-decision "$OPERATOR" \
  --recipe "$THERMAL/formal_recipe.json" --resume "$THERMAL/latest_model.pt" \
  --stop-after 5000 --device cuda:0
```

Both fitters reject changed profile hashes, memberships, normalization values, schedules, source identities, and cross-scope resumes. Full formal monitoring uses canonical89/original90; the DEV22 scope is enabled only by the explicit startup flag and `startup_flow_*`/`startup_thermal_*` identities. Checkpoint monitoring and latest state are saved every 100 epochs, and only the declared best-field alias and epoch milestones are retained.

## Bounded startup evidence and measured price

The startup runs were stored outside Git at `/data/wanglz/ModularDT/thermal_development/formal5000_manual_20261006`. D-sep started from its fresh epoch-zero checkpoint and completed e1–e3 over all 600 TRAIN cases; it used 13 updates and 614,400 sampled fluid queries per epoch, with 0.715 s mean measured training time/epoch, 0.011 s DEV22 validation at e3, 0.485 s final save, 247,451,648 bytes peak allocated GPU memory, and 9.323 s process wall time. Its e3 DEV22 standardized four-field MSE was 1.1128; this is startup evidence, not a full-population score.

D-sep completed e1–e3 in one uninterrupted process, so this startup did not measure a live D-sep stop/restart. A separate strict e3→e3 no-op resume loaded the same startup identity and acknowledged a pre-existing stop request with status `already_completed`; it made zero case visits and optimizer updates in 6.658 s, removed the request, and did not rewrite the checkpoint. The latest checkpoint SHA256 remained `85bc0ef60004876ef11a275f4627b3c399feea9e81879a6376525325e82d31a5`, preserving all 38 flow parameter tensors and 38 Adam state entries at step 39. This verifies completed-checkpoint loading and no-op status handling, not a continued D-sep resume. The GPU receipts are in `/data/wanglz/ModularDT/thermal_development/formal5000_manual_20261006/D-sep-startup/resource_sessions.json`; the separate CPU comparison receipt is in `/data/wanglz/ModularDT/thermal_development/formal5000_manual_20261006/D-sep-startup-e3-noop-review/resource_sessions.json`.

R-direct started from fresh epoch-zero thermal weights with the e3 startup flow frozen, completed e1–e2, received a real `stop_requested` file after e1, reported `stopped_resumable` at e2, then strictly resumed from e2 and completed e3. The three epochs each visited 600 TRAIN cases, made 13 updates, used 614,400 fluid queries and 76,800 qualified operator rows, and recorded 10.959 s aggregate training time; the two DEV22 validation passes took 0.126 s total and the two measured checkpoint saves took 0.998 s total. The resumed process loaded model/optimizer state in 0.502 s and completed e3 with all 52 optimizer parameter states at step 39. Peak allocated memory was 896,102,400 bytes. The DEV22 three-role selector changed from 0.5071 at e2 to 0.4307 at e3, and the common five-field standardized MSE changed from 1.0009 to 0.9965; e3 physical RMSE in native dataset units for u/v/p/omega/temperature was 0.3183/0.0537/0.1700/1.1327/5.2885. These exposed-panel measurements verify training and composition plumbing only.

Across the four CUDA-visible process intervals, measured GPU-associated wall time was 48.1 s; the actual model-training intervals account for 13.1 s, while data preparation, checkpoint loading, validation, and saving are itemized separately in each `resource_sessions.json`. A linear forecast from measured startup training epochs is about 59.6 min for 5,000 D-sep epochs plus 5 h 4 min for 5,000 R-direct epochs on one GPU, or about 6.07 hours of training alone, before full canonical89/original90 monitoring and the later, larger checkpoint-save overhead. The full-panel monitoring overhead was deliberately not measured during DEV22-only startup. This estimate prices a future manual run but does not establish a six-hour end-to-end completion time; no formal5000 training was started.

## Scope limits

The near-curl D-sep refinement was not selected: the matched +500 review remained worse than ordinary D-sep on fluid-u by 12.68% and showed no omega benefit. No R-group retraining, inverse search, wind campaign, forced K compression, new physical solve, or automatic formal5000 launch is part of this recipe. `solver_attempts` remains unchanged at 326/326, and all historical model checkpoints and evaluations remain preserved.
