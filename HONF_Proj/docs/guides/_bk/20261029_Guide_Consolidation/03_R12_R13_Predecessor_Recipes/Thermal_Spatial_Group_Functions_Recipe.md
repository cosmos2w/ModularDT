# Thermal spatial group functions: maturation and full-TRAIN recipe

**Archived reference — 29 October 2026.** This guide records a predecessor model or a dated experiment, not current architecture policy or standing launch authorization. Use the [current guide index](../../../README.md); see the [archive review](../README.md) for the reason it was moved.

This guide records the maintained Round13-1 maturation and full-TRAIN recipe for the two Thermal spatial-group models. Both original development runs stopped at epoch 100; those checkpoints remain the literal source states for full-state continuation to development epoch 500, followed by fresh TRAIN600 fits from their matching epoch-500 endpoints. The companion [Round13 development report](../../../../reports/HONF_Round13_Spatial_Group_Functions_Development_Report.md) documents the original short-run evidence and does not claim mature model potential.

The 25 October run-directory cleanup archived the original F13 epoch100 runs under `_bk/13_20261025_Spatial_Group_Development100` and the balanced-singleton recovery under `_bk/12_20261024_Interaction_Group_Recovery`. At the user’s follow-up request, both development500 parents and both full-data Round13-1 runs were also moved to `_bk/14_20261025_Round13_1_Maturation`; none remain at the forward-run root. The [run archive guide](../../../Thermal_Forward_Run_Archive.md) gives exact locations. All recorded source-path strings and scientific configs remain provenance. Commands below document the executed experiments and previously tested continuation path; an exact replay now requires restoring original physical run and parent paths before execution, rather than rewriting checkpoint identities. No continuation is scheduled.

## Model and learning recipe

The new core builds one deterministic balanced spatial hierarchy over Thermal's 24×8 environment catalogue and learns a context-conditioned cut that is shared by both interaction stages. Both arms retain MM/ME/E→E, prescribed controls in the input, the early group→M relay, a direct all-group receiver read, and one joint physical-field trunk. The unused final group→M update and M-first proposals/postings are removed. Each selected group carries spatial information through the computation: F13-0 keeps the mean-valued group function, while F13-1 adds the two first-order spatial slope modes. Both arms use one conditional-utility head trained from actual paired native losses to choose when a hierarchy node should be subdivided; hard topology itself is not differentiated, and the learned cut is not a physical-causality claim.

Every fourth main optimizer macro, the task selects an input-stratified TRAIN case and an internal node by module-count and size/depth bands, including nodes below stopped ancestors. It compares the node as a parent group with a descendant split, alternating one child level and two levels, while preserving off-path decisions and recomputing the shared relay and all dependent stage-two/native outputs. The two physical losses receive equal half-weights normalized by the actual macro case count; the signed normalized loss difference trains only the utility head with detached features and label, Huber delta 1, and coefficient 1. There is no REINFORCE term, learned work price, target-K controller, or additional optimizer.

Primary TRAIN uses deterministic scale exploration on one quarter of case/update keys, drawing uniformly over available tree depths and using the same exploration keys in both arms. The other primary exposures use the learned cut. Exploration is disabled for response supervision and evaluation. Every fourth main update also receives one TRAIN0348 response pair, alternating heat and single-geometry perturbations; the coefficient is inherited as 23.47364739373233 from the literal source checkpoint, with the existing 10% shared-gradient cap. No response recalibration campaign or extra response cohort is introduced.

## Parents, initialization, and fixed inputs

Both development arms adapt the literal balanced-singleton R12 epoch-100 endpoint at `/data2/wanglz/ModularDT/relocated_workspace/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/thermal_interaction_groups_balanced_dev_v2/checkpoints/epoch_0100_model.pt`. The compatible-tensor mapper copies the shared encoders, interaction stages, group transform/reader, and field path, while obsolete frontier/proposal tensors are dropped and the new utility head is initialized for this task. F13-0 and F13-1 share the same mapped initialization; each run creates a fresh AdamW optimizer with empty moments. The literal source's fitted TRAIN150 normalizer and heat scale 1.3164895030821728 are inherited exactly without refitting.

Development uses the fixed `fixed25_train150_dev22` membership and manifest SHA256 `933b0138ba2f8447a1ecadfe31fd0bb2cb4a05607d3ac3d9f0dc79419f196044`; its selected TRAIN and exposed DEV identities are checked against that manifest. The source data remains the existing packed Thermal dataset at `/data/wanglz/ModularDT/1_ChannelThermal/Processed_ChannelThermal_Dataset/packed_dataset.h5`, with manifest `/data/wanglz/ModularDT/thermal_development/fixed25_v1/manifest.json`. Each development epoch visits all 150 selected TRAIN cases in four main updates at effective batch 48 and microbatch 24, with Q1024 primary fluid queries. Development validation is the fixed exposed DEV22 panel, not an independent TEST result.

The original configs remain valid records of the 100-epoch setup: [F13-0 development](../../../../../configs/research/thermal_spatial_groups_f13_0_dev_v1.json), [F13-1 development](../../../../../configs/research/thermal_spatial_groups_f13_1_dev_v1.json), and the original [F13-0](../../../../../configs/research/thermal_spatial_groups_f13_0_fulltrain_v1.json) and [F13-1](../../../../../configs/research/thermal_spatial_groups_f13_1_fulltrain_v1.json) full-TRAIN profiles. The Round13-1 profiles are [F13-0 development maturation](../../../../../configs/research/thermal_spatial_groups_f13_0_dev_mature_v1.json), [F13-1 development maturation](../../../../../configs/research/thermal_spatial_groups_f13_1_dev_mature_v1.json), [F13-0 full-TRAIN](../../../../../configs/research/thermal_spatial_groups_f13_0_fulltrain_mature_v1.json), and [F13-1 full-TRAIN](../../../../../configs/research/thermal_spatial_groups_f13_1_fulltrain_mature_v1.json). Their DATA2 run roots are `thermal_spatial_groups_r13_1_dev_f13_0`, `thermal_spatial_groups_r13_1_dev_f13_1`, `thermal_spatial_groups_r13_1_full_f13_0`, and `thermal_spatial_groups_r13_1_full_f13_1` beneath the existing Thermal run root.

## Qualification actually completed

The initial native forward check evaluated both F13 arms and the literal G-EE parent on fitted TRAIN case `0006` (M1) and case `0631` (M12). It verified equal common F13 tensors, exact inherited normalizer and heat scale, and fresh optimizer state; it was a two-case engineering check, not a DEV22 qualification or a performance-selection result. The retained [initial-forward receipt](../../../../../diagnostics/round13_spatial_20261025/initial_forward.json) contains the per-case evidence.

A disposable F13-1 run used the maintained runner for local epoch 1 and resumed the same run to epoch 2. The completed qualification covers 300 primary case visits, 8 main optimizer updates, 2 paired probes, 2 response pairs, final-macro case counts `[6, 6]`, one visible CUDA RNG stream, and 155 named Adam entries; both epochs were finite and the saved/resumed boundary completed. This engineering replay is not a scientific arm and does not count toward either 100-epoch development endpoint. Its compact [qualification receipt](../../../../../diagnostics/round13_spatial_20261025/qualification_receipt.json) and logs preserve the execution evidence.

The original F13-0 and F13-1 profiles completed epoch 100 in their original run roots. Round13-1 development continuation starts from each literal epoch-100 checkpoint in a new namespace and resumes its saved model, named Adam state, provider counters, complete history, RNG streams, sampler state, and field-selection state. It preserves the original constant `3e-4` schedule through its recorded epoch-100 horizon while extending the observation horizon and checkpoint milestones to 500; it does not copy the source checkpoint payload into the new run. The Round13-1 full-TRAIN profiles then start a new optimizer and provider clock from their matching development epoch-500 model, inherit its TRAIN150 normalizer and heat scale, train on TRAIN600, and expose canonical VAL89 with case `0273` excluded. The first requested full-data stop is epoch 500; the plan permits one later matched extension or amendment only after mature evidence and within its reserve.

## Development commands

Run from the HONF project directory with the `ModularDT` environment. Expose one physical GPU to the process and select its logical `cuda:0`; the configs contain no physical UUID or immutable visibility binding. These commands continue the saved epoch-100 development states into their new `/data2` namespaces.

```bash
cd /home/wanglz/Desktop/src/ModularDT/HONF_Proj

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n ModularDT python tools/honf_research.py continue-contextual \
  --config configs/research/thermal_spatial_groups_f13_0_dev_mature_v1.json \
  --device cuda:0 --epochs 500 \
  --output /data2/wanglz/ModularDT/relocated_workspace/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/thermal_spatial_groups_r13_1_dev_f13_0

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n ModularDT python tools/honf_research.py continue-contextual \
  --config configs/research/thermal_spatial_groups_f13_1_dev_mature_v1.json \
  --device cuda:0 --epochs 500 \
  --output /data2/wanglz/ModularDT/relocated_workspace/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/thermal_spatial_groups_r13_1_dev_f13_1
```

Each continuation keeps the original development membership, normalization, optimizer schedule, and epoch-100 history, with monitoring checkpoints at 100, 200, 300, 400, and 500, 10-epoch loss-curve refresh, the existing validation cadence, and tqdm progress. Review the saved trajectory at the declared milestones; the 500-epoch endpoint is the minimum maturity observation horizon, not a convergence or physical-validity certificate.

## Full-TRAIN training

The full-TRAIN profiles are separately labelled `full_train600`/`VAL89` runs with a 1000-epoch absolute horizon and a first 500-epoch segment. Each starts from its matching F13 development epoch-500 checkpoint, creates a fresh AdamW optimizer and provider clock, and inherits the TRAIN150-fitted normalizer and heat scale recorded by that development lineage. It does not refit normalization on TRAIN600. The plan authorizes the first 500 epochs; any use of the remaining horizon requires the single conditional reserve decision described in the Round13-1 work plan.

```bash
cd /home/wanglz/Desktop/src/ModularDT/HONF_Proj

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n ModularDT python tools/honf_research.py train-contextual \
  --config configs/research/thermal_spatial_groups_f13_0_fulltrain_mature_v1.json \
  --device cuda:0 --epochs 500 \
  --output /data2/wanglz/ModularDT/relocated_workspace/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/thermal_spatial_groups_r13_1_full_f13_0

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n ModularDT python tools/honf_research.py train-contextual \
  --config configs/research/thermal_spatial_groups_f13_1_fulltrain_mature_v1.json \
  --device cuda:0 --epochs 500 \
  --output /data2/wanglz/ModularDT/relocated_workspace/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/thermal_spatial_groups_r13_1_full_f13_1
```

If either full-TRAIN fit stops before its requested epoch 500, resume that same run with `resume-contextual` to the absolute local epoch 500. A cleanly completed 500-epoch result can be evaluated on canonical VAL89; the remaining horizon is not an automatic launch.

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n ModularDT python tools/honf_research.py resume-contextual \
  --run /data2/wanglz/ModularDT/relocated_workspace/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/thermal_spatial_groups_r13_1_full_f13_0 \
  --device cuda:0 --stop-after-epoch 500
```

The same resume command applies to the `thermal_spatial_groups_r13_1_full_f13_1` root. A later process may select a different physical GPU by changing `CUDA_VISIBLE_DEVICES` while keeping exactly one device visible and continuing to pass `--device cuda:0`.

These commands establish the authorized Round13-1 training path; completion, measured gains and misses, and any conditional reserve choice must be reported from saved evidence rather than inferred from successful startup.

## Route T runtime allocation after the evaluated full-TRAIN500 endpoints

Route T is recorded in [the versioned reserve recipe](../../../../../configs/research/thermal_spatial_groups_r13_1_reserve_t_v1.json): both original full-TRAIN runs continue from their own literal epoch-500 checkpoints to the absolute epoch-1000 horizon with model, named Adam state, provider state, RNG, sampler, schedule, and scientific configs unchanged. The per-arm absolute cumulative provider-elapsed caps are 48,354 seconds for F13-0 and 50,352 seconds for F13-1; each retains the existing 600-second clean-save reserve and has up to eight additional provider-hours beyond its saved source elapsed. These caps are execution allocations, not changes to the scientific recipe, and GNU-time process charges remain separately measured.

For the first continuation after the epoch-500 evaluation, pass the arm's absolute cumulative cap to `resume-contextual`; this resumes the existing run directory and does not create a new arm or reset optimizer/provider/RNG state.

```bash
cd /home/wanglz/Desktop/src/ModularDT/HONF_Proj

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n ModularDT python tools/honf_research.py resume-contextual \
  --run /data2/wanglz/ModularDT/relocated_workspace/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/thermal_spatial_groups_r13_1_full_f13_0 \
  --device cuda:0 --stop-after-epoch 1000 --wall-clock-budget-seconds 48354

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n ModularDT python tools/honf_research.py resume-contextual \
  --run /data2/wanglz/ModularDT/relocated_workspace/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/thermal_spatial_groups_r13_1_full_f13_1 \
  --device cuda:0 --stop-after-epoch 1000 --wall-clock-budget-seconds 50352
```

If either resumed fit needs another process after a clean interruption, omit `--wall-clock-budget-seconds` to reuse the same absolute cap stored in that run's `run_manifest.json`; the runner never adds a fresh allowance. An explicit later value is also absolute cumulative provider elapsed time, not additional time, and is recorded in the manifest. The provider clock starts at its maintained budget binding after provider construction, so report it separately from whole-process elapsed charge. These commands authorize work only through epoch 1000 and do not claim either endpoint is complete.

## Recorded user stop and historical continuation commands

Both development500 and full-TRAIN500 fits and their native VAL89 evaluations completed. At the user's subsequent stop request, Route T clean-stopped with exit0 at full-data epoch779 for F13-0 and755 for F13-1. Their latest/best checkpoints, complete histories, final clean-stop validation and refreshed curves are retained in the same full-data run roots. The matched scientific endpoint remains500; the unequal reserve stops were not given additional full-native evaluation. The [mature report](../../../../reports/HONF_Round13-1_Mature_Spatial_Group_Functions_Report.md) records the completed measurements and unfinished1000 work.

All four Round13-1 runs are now archived at the user’s request. No automatic continuation remains. The commands below retain the original physical run names as a record of the tested continuation interface; they are not directly usable against the archived tree. A future exact replay would first restore all required original run/parent paths under explicit user direction. Remaining horizons are221/245 epochs. The explicit-cap resume path was exercised by the actual500→partial continuation, and omitted-cap reuse passed focused runner tests; archival does not reset states or renew compute allowances. These commands were not launched after the stop/archive requests.

```bash
cd /home/wanglz/Desktop/src/ModularDT/HONF_Proj

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n ModularDT python tools/honf_research.py resume-contextual \
  --run /data2/wanglz/ModularDT/relocated_workspace/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/thermal_spatial_groups_r13_1_full_f13_0 \
  --device cuda:0 --stop-after-epoch 1000

CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n ModularDT python tools/honf_research.py resume-contextual \
  --run /data2/wanglz/ModularDT/relocated_workspace/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/thermal_spatial_groups_r13_1_full_f13_1 \
  --device cuda:0 --stop-after-epoch 1000
```
