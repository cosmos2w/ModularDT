# Relation-guided member-access continuation

**Archived reference — 29 October 2026.** This guide records a predecessor model or a dated experiment, not current architecture policy or standing launch authorization. Use the [current guide index](../../../README.md); see the [archive review](../README.md) for the reason it was moved.

Resume a v3 member-access run through the maintained `tools/honf_member_access.py` entrypoint. An explicit horizon extension may also declare checkpoint milestones and a curve interval. These output controls preserve weights, named optimizer moments, RNG, data, normalization, native objectives, query sampling, validation cadence and the optimizer schedule. The provider retains the source checkpoint's original recipe, including its historical output declarations; the engine and continuation manifest carry the requested execution outputs. Historical v1/v2 recovery remains unchanged.

The user-directed continuation of `thermal_r13_4_guide_train600_100_v3` starts from its completed epoch 100 and stops at epoch 500 on physical GPU 1. It retains the existing epoch-12/100 scientific checkpoints, adds milestones 200 and 500, and maintains latest/best-field states. No epoch-300/400 milestone is requested. Curves refresh every epoch; validation remains every 100 epochs and at the existing final/clean-stop review.

The curve setting follows the epoch-based monitoring convention in PhyCoFlow's `src/phycoflow_reconstruction/training/monitoring.py`; its readiness configurations use an interval of one epoch. This run explicitly overrides the historical ten-epoch drawing interval. It overwrites the current PNG/PDF curves with complete history rather than accumulating per-epoch image files. Fresh configurations and other runs retain their declared settings.

From `/home/wanglz/Desktop/src/ModularDT/HONF_Proj` in the ModularDT environment:

```bash
export CUDA_VISIBLE_DEVICES=0,1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/data2/wanglz/ModularDT/round13_4_20261026/continuation_e100_to500/tmp
export MPLCONFIGDIR="$TMPDIR/mpl"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
/home/wanglz/miniconda3/envs/ModularDT/bin/python -u tools/honf_member_access.py resume \
  --run /home/wanglz/Desktop/src/ModularDT/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/thermal_r13_4_guide_train600_100_v3 \
  --device cuda:1 --stop-after-epoch 500 --extend-horizon \
  --checkpoint-epochs 200 500 --curve-every 1 --wall-clock-budget-seconds 36000
```

The cumulative wall-clock allowance is an execution setting, recorded separately from the unchanged recipe and constant AdamW schedule. It prevents the original allowance from stopping the newly authorized 400 additional epochs prematurely. The existing native-health pause and writer safeguards remain enabled.

After an interruption, recover the same run with `resume --run <same-run> --device cuda:1 --stop-after-epoch 500`. The saved execution horizon, output policy and cumulative allowance are restored; the explicit output overrides are needed only when extending the horizon again. Do not fresh-train into this run directory. Runtime logs and the continuation receipt are local under `HONF_Proj/diagnostics/round13_4_20261026/continuation_e100_to500`, and current loss curves remain under the run's `plots/training` directory.
