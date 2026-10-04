# Manual Thermal 5,000-epoch recipes after the completed 500 review

The user closed this campaign at 500 epochs on 2026-10-04. **No further campaign training or 5,000-epoch launch was performed.** The [final conclusions](../reports/HONF_Shared_Core_Thermal_500_Epoch_Conclusions.md) select one exploratory organizer, **H-tree**, for a possible later manual experiment: it is the strongest new organizer on physical temperature fidelity and has useful conditional graph actions. Fresh Dense is more accurate; Tree has no measured sparse speedup and its inverse readiness is mixed. Longer training is not established to help.

Continuation and fresh seed 0 below are two alternatives for the same architecture, not independent finalists. Existing Local exact 1000 preparations remain supplementary historical artifacts.

## Actual preparation and bounded verification

The durable [launcher](../../tools/thermal_campaign_long_run.py) accepts `--completed-stage 500` or `1000` (default 1000); its 50 focused CPU tests passed. Actual Tree 500 continuation/fresh preparation and both printed native `train.py --dry-run` commands exited 0 at source `5c06293fcbcc157ac54918fbfa06089fb69cec64`. Preparation checked complete native configuration, case physics and frozen Stage-A binding; only duration and placement changed. Preparation and parsers constructed no model or optimizer.

Run2503 contains a byte-identical copy of the original Run2203 **top-level exact 500** parent, not selected467 or recovered Run2303 state:

```text
/data/wanglz/ModularDT/shared_core_campaign_20261002/HONF_Forward_Runs/Run_2203_20261002_220934_thermal_h-tree_v1/epoch_0500_model.pt
SHA256 0a2d68b9af8706cf9608f504c9afc517b1ad2914b81bd7bf4bed476cad0b5ada
```

The copy preserves 185 active Adam states at step 6500, all four RNG streams, physical policy 2, completed response/structural calibration and selection epoch 500/horizon 5000. Continuation starts at **501**, with the absolute schedule horizon 5000 unchanged. Every epoch must visit all 600 training and 90 exposed-development cases; batch caps are absent, AMP is disabled and predicted ports remain active throughout. Fresh seed 0 retains policy 1 through 100 and policy 2 from 101, matching the original amendment.

An actual **CPU1 case0274, 14-query native forward** from the copied parent exited 0 at 2026-10-04 02:49:04 UTC. P0/P1/P2 each prepared once, all seven nonempty prediction/latent outputs were finite, and 316 persistent model tensors/4,287,933 scalars (including frozen Stage-A109) stayed bitwise unchanged. Three disabled outputs were explicitly zero-width. All 801 source/parent/recipe guards stayed unchanged. No optimizer, RNG restore, first-gradient or training update occurred. This tests loading and bounded forward execution; future optimization, GPU gradients, full-dataset fidelity, convergence and 5,000 completion remain untested.

Receipts: [preparation/parser exits](../../diagnostics/generated/shared_core_campaign_20261002/tmp/manual_h_tree500_preparation_20261004/actual_preparation_execution_receipt.json), [native replay](../../diagnostics/generated/shared_core_campaign_20261002/tmp/manual_h_tree500_Q14_native_replay/bounded_replay_receipt.json), [saved-array/state review](../../diagnostics/generated/shared_core_campaign_20261002/tmp/manual_h_tree500_preparation_20261004/Q14_replay_saved_review.json), [50-test receipt](../../diagnostics/generated/shared_core_campaign_20261002/tmp/manual500_helper_tests_20261004/implementation_test_receipt.json).

## Ready exact 500 continuation: Run2503

The profile and copied parent below are already prepared in a new standard RunStore workspace; the original Run2203 is unchanged. **Manual launch only; this command was not executed.** Replacing `--yes` with `--dry-run` gives the exact parser command that passed.

```bash
rtk proxy env CUDA_VISIBLE_DEVICES=2 \
  PYTHONPATH=/home/wanglz/Desktop/src/ModularDT/HONF_Proj/src:/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Case_ThermalChannel/src \
  /home/wanglz/miniconda3/envs/ModularDT/bin/python \
  /home/wanglz/Desktop/src/ModularDT/HONF_Proj/train.py \
  --config /data/wanglz/ModularDT/thermal_manual5000/launch_profiles/2503_thermal_h-tree_continue500_e5000.json \
  --device cuda:0 \
  --resume-checkpoint /data/wanglz/ModularDT/thermal_manual5000/ThermalChannel/HONF_Forward_Runs/Run_2503_20261003_224123_thermal_h-tree_continue500_e5000/continuation_parent_epoch_500_model.pt \
  --yes
```

Physical GPU2 maps to logical `cuda:0`. A later GPU1 placement uses `CUDA_VISIBLE_DEVICES=1` with the same logical device; record placement and rerun the parser check. Inspect current contention before launching, retain affected timing and leave unrelated jobs running.

## Ready fresh alternative: Run2603, seed 0

Fresh preparation wrote the profile only; the ordinary trainer reserves a new Run2603 workspace on launch. Its parser check with `--dry-run` passed. Seed 0 matches initial weights in a distinct run; it is **not independent-seed replication** or a promise to reproduce the historical trajectory after source revisions. **Manual launch only; this command was not executed:**

```bash
rtk proxy env CUDA_VISIBLE_DEVICES=2 \
  PYTHONPATH=/home/wanglz/Desktop/src/ModularDT/HONF_Proj/src:/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Case_ThermalChannel/src \
  /home/wanglz/miniconda3/envs/ModularDT/bin/python \
  /home/wanglz/Desktop/src/ModularDT/HONF_Proj/train.py \
  --config /data/wanglz/ModularDT/thermal_manual5000/launch_profiles/2603_thermal_h-tree_fresh_seed0_e5000.json \
  --device cuda:0 --yes
```

To recreate either preparation elsewhere, use tracked profile `src/config_core/forward/thermal_campaign/h-tree_e500.json` with `tools/thermal_campaign_long_run.py --completed-stage 500 --physical-gpu 2 --output-root ... --run-id ... --prepare`, supplying the original exact 500 parent or `--fresh --seed 0`. Choose unused run/output identities. The helper refuses existing preparations, copies the parent into a new workspace and never starts the trainer. Generated profiles and outputs remain in ignored external artifact storage.

## Cost, review and recovery

Tree's saved stage 500 epochs 481–500 had median train+validation elapsed **102.013223 seconds/epoch**. At that historical rate, 4,500 additional epochs would take **127.52 hours**, excluding setup, checkpoint/preview IO, evaluations, interruptions and future contention/source changes. This is a conditional elapsed-time extrapolation, not exclusive GPU time or a completion guarantee. The campaign was closed because waiting longer was unacceptable; these recipes do not schedule that cost.

Profiles retain all existing best-field/best-temperature/best-predicted selectors, latest saves every 25 epochs, milestones every 25 through 1000 then 2500/5000, and previews every 50. If manually launched later, review coverage, finite physical/gradient metrics, responses and organizer utility at 750/1000 before deciding whether to continue toward 2500/5000. Both recipes use repaired Tree soft-training normalization from `d9a3646`; hard quadrature measures and physical loss floors are preserved. A continuation from 500 under repaired source is a declared new lineage segment, not retroactive evidence about historical 500 behavior.

Ctrl-C creates no extra checkpoint. For a later interruption, inspect the **new** workspace's actual saved epoch and resume with its own top-level `latest_model.pt` or `epoch_NNNN_model.pt` and generated profile. Never resume from a selection/evaluation subfolder or overwrite the finished original 500 workspace. Wind training remains paused; these recipes do not launch Wind, paired heads, CFD validation or extra portfolio candidates.
