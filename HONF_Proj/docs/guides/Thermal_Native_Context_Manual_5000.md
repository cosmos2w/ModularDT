# Manual native-context full-data comparison

The completed [Tree-C/Global-C development report](../reports/_bk/20261005_105606Z_HONF_Native_Context_Organizer_Confirmation_Report.md) recommends a fresh comparative learning experiment. Tree500 improves six core means and 18 of 24 role means over matched Global500, but loses fluid pressure and vorticity, retains low-M thermal misses and costs more. Both retained Dense-D25 500 and 1000 have lower means than both new arms on all eight core roles, with the feature, initializer and objective differences disclosed in the report. Route-uniform controls nearly preserve Tree's thermal accuracy. The full-data pair should test whether its trained support/organization tradeoff persists beyond the exposed quarter panel; it is not definitive HONF or inverse confirmation.

The two maintained recipes are [Tree-C](../../src/config_core/forward/thermal_native_context/tree-c_full5000.json), proposed Run3501, and [Global-C](../../src/config_core/forward/thermal_native_context/global-c_full5000.json), proposed Run3502. These commands are for the user to launch manually after choosing available GPUs. No formal optimizer or training run was created during preparation.

From `/home/wanglz/Desktop/src/ModularDT/HONF_Proj`:

```bash
rtk proxy env CUDA_VISIBLE_DEVICES=1 /home/wanglz/miniconda3/envs/ModularDT/bin/python train.py --config src/config_core/forward/thermal_native_context/tree-c_full5000.json --device cuda:0 --yes
```

```bash
rtk proxy env CUDA_VISIBLE_DEVICES=2 /home/wanglz/miniconda3/envs/ModularDT/bin/python train.py --config src/config_core/forward/thermal_native_context/global-c_full5000.json --device cuda:0 --yes
```

Both start fresh at seed0 on the original 600 TRAIN cases, with their own training-only global normalization and source-local heat scale. The 90 validation cases share that transform. Frozen Stage-A retains its checkpoint-owned normalization. No quarter manifest, selected-quarter statistics, parent, trained initialization checkpoint or optimizer state is inherited. The preparation fit has 12 finite statistic arrays and normalized heat maximum 1.7521239519; this different basis does not by itself prove leakage in the historical packed statistics.

Keep native coarse/local context, fine physical values, `source_local_v3`, predicted ports, FP32 Q1024, effective batch48/micro8, learning rate0.0003, weight decay1e-5 and gradient clip1. Absolute stop and schedule horizon are5000. Save declared100-epoch milestones, latest, best-field selection and plots every100 epochs; select over saved checkpoints consistently, without a temperature-only selector. The absolute exploration/calibration schedule is preserved rather than stretched. Each full epoch visits600 cases; future counts and runtime must be measured from the actual formal trainer.

Both use physical-denominator policy1 through100 and policy2 from101. Only the existing exposed TRAIN0348 response anchor is enabled after100 on the maintained auxiliary cadence. Extra null-training coefficient is0. Tree starts its own TRAIN600 measure-policy2 structural calibration from26, maximum coefficient0.001 and `local_context_shadow_v1`; Global has no frontier penalty and uses ordinary task gradients. The compatibility predicted-consistency coefficient0.05 is inactive in pure predicted-port mode; the primary predicted-port losses and port-global consistency are separate active components.

Both exact recipes passed the real CLI CPU dry-run. Bounded native preparation used eight whole wrappers per arm, including fresh canonical-reference materialization and comparisons on input-only TRAIN0003(M1)/0228(M12), with zero optimizer construction, backward, update or validation-sample calls. Tree's finite identity comparisons and pre-cleanup weight/Stage-A/gradient guards passed, maximum difference8.94e-8. Its final restoration receipt is unavailable because the diagnostic used the wrong removable-handle accessor after removing owned hooks. The accessor was corrected and generic CPU normal/exception ownership checks passed; no additional Tree wrappers were spent. Global's corrected native replay exited0 with weights, all module modes, RNG and owned hooks restored. This limitation is retained rather than labelling both attempts successful.

Future formal evaluation must explicitly use `--evaluation-scope formal-full --stage 5000` at the actual completed/selected checkpoint. Canonical89 excluding duplicate0273 and original90 compatibility are separate previously exposed scopes. Keep statistical coverage separate from a small declared detail panel. No formal validation forward was performed here. Retain Run1804 as a historical original600/packed-normalization reference, not a matched fresh baseline.

Whole-wrapper physical agreement, tighter representation misses, known heat-null leakage and absent held positive heat-response truth remain distinct in the report. The prepared12-solve request needs separate authorization; these recipes add no solver, inverse generator or design-search campaign.
