# Thermal spatial group functions: development and handoff recipe

This guide records the maintained Round13 recipe for the two new Thermal spatial-group models, the verified short-run boundary, and the user-controlled development and full-TRAIN commands. Both development arms have now completed epoch 100; the companion [Round13 development report](../reports/HONF_Round13_Spatial_Group_Functions_Development_Report.md) is the source for detailed field results, measured cost, and any model recommendation, and this guide makes no model-selection claim.

## Model and learning recipe

The new core builds one deterministic balanced spatial hierarchy over Thermal's 24×8 environment catalogue and learns a context-conditioned cut that is shared by both interaction stages. Both arms retain MM/ME/E→E, prescribed controls in the input, the early group→M relay, a direct all-group receiver read, and one joint physical-field trunk. The unused final group→M update and M-first proposals/postings are removed. Each selected group carries spatial information through the computation: F13-0 keeps the mean-valued group function, while F13-1 adds the two first-order spatial slope modes. Both arms use one conditional-utility head trained from actual paired native losses to choose when a hierarchy node should be subdivided; hard topology itself is not differentiated, and the learned cut is not a physical-causality claim.

Every fourth main optimizer macro, the task selects an input-stratified TRAIN case and an internal node by module-count and size/depth bands, including nodes below stopped ancestors. It compares the node as a parent group with a descendant split, alternating one child level and two levels, while preserving off-path decisions and recomputing the shared relay and all dependent stage-two/native outputs. The two physical losses receive equal half-weights normalized by the actual macro case count; the signed normalized loss difference trains only the utility head with detached features and label, Huber delta 1, and coefficient 1. There is no REINFORCE term, learned work price, target-K controller, or additional optimizer.

Primary TRAIN uses deterministic scale exploration on one quarter of case/update keys, drawing uniformly over available tree depths and using the same exploration keys in both arms. The other primary exposures use the learned cut. Exploration is disabled for response supervision and evaluation. Every fourth main update also receives one TRAIN0348 response pair, alternating heat and single-geometry perturbations; the coefficient is inherited as 23.47364739373233 from the literal source checkpoint, with the existing 10% shared-gradient cap. No response recalibration campaign or extra response cohort is introduced.

## Parents, initialization, and fixed inputs

Both development arms adapt the literal balanced-singleton R12 epoch-100 endpoint at `/data2/wanglz/ModularDT/relocated_workspace/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/thermal_interaction_groups_balanced_dev_v2/checkpoints/epoch_0100_model.pt`. The compatible-tensor mapper copies the shared encoders, interaction stages, group transform/reader, and field path, while obsolete frontier/proposal tensors are dropped and the new utility head is initialized for this task. F13-0 and F13-1 share the same mapped initialization; each run creates a fresh AdamW optimizer with empty moments. The literal source's fitted TRAIN150 normalizer and heat scale 1.3164895030821728 are inherited exactly without refitting.

Development uses the fixed `fixed25_train150_dev22` membership and manifest SHA256 `933b0138ba2f8447a1ecadfe31fd0bb2cb4a05607d3ac3d9f0dc79419f196044`; its selected TRAIN and exposed DEV identities are checked against that manifest. The source data remains the existing packed Thermal dataset at `/data/wanglz/ModularDT/1_ChannelThermal/Processed_ChannelThermal_Dataset/packed_dataset.h5`, with manifest `/data/wanglz/ModularDT/thermal_development/fixed25_v1/manifest.json`. Each development epoch visits all 150 selected TRAIN cases in four main updates at effective batch 48 and microbatch 24, with Q1024 primary fluid queries. Development validation is the fixed exposed DEV22 panel, not an independent TEST result.

The versioned configs bind the arms and paths directly: [F13-0 development](../../configs/research/thermal_spatial_groups_f13_0_dev_v1.json), [F13-1 development](../../configs/research/thermal_spatial_groups_f13_1_dev_v1.json), [F13-0 full-TRAIN handoff](../../configs/research/thermal_spatial_groups_f13_0_fulltrain_v1.json), and [F13-1 full-TRAIN handoff](../../configs/research/thermal_spatial_groups_f13_1_fulltrain_v1.json). Development output roots are `/data2/wanglz/ModularDT/relocated_workspace/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/thermal_spatial_groups_r13_f13_0` and the matching `_f13_1` path; the full-TRAIN roots are their `/fulltrain` children. Each run declares a 2 GiB output budget, 512 MiB peak temporary budget, and 200 GiB minimum free-space headroom on `/data2`.

## Qualification actually completed

The initial native forward check evaluated both F13 arms and the literal G-EE parent on fitted TRAIN case `0006` (M1) and case `0631` (M12). It verified equal common F13 tensors, exact inherited normalizer and heat scale, and fresh optimizer state; it was a two-case engineering check, not a DEV22 qualification or a performance-selection result. The retained [initial-forward receipt](../../diagnostics/round13_spatial_20261025/initial_forward.json) contains the per-case evidence.

A disposable F13-1 run used the maintained runner for local epoch 1 and resumed the same run to epoch 2. The completed qualification covers 300 primary case visits, 8 main optimizer updates, 2 paired probes, 2 response pairs, final-macro case counts `[6, 6]`, one visible CUDA RNG stream, and 155 named Adam entries; both epochs were finite and the saved/resumed boundary completed. This engineering replay is not a scientific arm and does not count toward either 100-epoch development endpoint. Its compact [qualification receipt](../../diagnostics/round13_spatial_20261025/qualification_receipt.json) and logs preserve the execution evidence.

Both production development profiles have completed their full epoch-100 horizon and saved the matching F13-0 and F13-1 endpoints in the configured development roots. The disposable boundary check remains a two-epoch resume qualification; no full-TRAIN600 training run has been launched, and the full-TRAIN commands below remain user-controlled handoff recipes.

## Development commands

Run from the HONF project directory with the `ModularDT` environment. Expose one physical GPU to the process and select its logical `cuda:0`; the configs contain no physical UUID or immutable visibility binding. Each command below is a manual recipe and writes only to its configured `/data2` output root.

```bash
cd /home/wanglz/Desktop/src/ModularDT/HONF_Proj

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n ModularDT python tools/honf_research.py train-contextual \
  --config configs/research/thermal_spatial_groups_f13_0_dev_v1.json \
  --device cuda:0 --epochs 100 \
  --output /data2/wanglz/ModularDT/relocated_workspace/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/thermal_spatial_groups_r13_f13_0

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n ModularDT python tools/honf_research.py train-contextual \
  --config configs/research/thermal_spatial_groups_f13_1_dev_v1.json \
  --device cuda:0 --epochs 100 \
  --output /data2/wanglz/ModularDT/relocated_workspace/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/thermal_spatial_groups_r13_f13_1
```

Each development profile has an absolute 100-epoch horizon and a 100-epoch initial segment. Review the compact training summaries at local epochs 25 and 50 without making extra checkpoint copies; at 100, review the retained endpoint and the results in the companion report. These profiles keep 100-epoch milestone checkpoints and best-field/latest handling at the review cadence, 10-epoch loss-curve refresh, internal validation cadence, and tqdm progress. There are no automatic extensions, portfolio launches, ratio approval gates, extra seeds, or extra training arms; historical ratio lines may be described but do not block a user-directed run or handoff.

## Full-TRAIN handoff and manual continuation

The full-TRAIN profiles are separately labelled `full_train600`/`VAL89` runs with a 1000-epoch absolute horizon and a first 200-epoch segment. Each starts from its matching F13 development epoch-100 checkpoint, copies that trained model, inherits the same TRAIN150-fitted normalizer and heat scale explicitly, and creates a fresh AdamW optimizer. It does not refit normalization on TRAIN600. Both endpoints were independently reconstructed and strict-loaded on CPU for this handoff check, and the matching source paths are fixed in the configs above.

```bash
cd /home/wanglz/Desktop/src/ModularDT/HONF_Proj

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n ModularDT python tools/honf_research.py train-contextual \
  --config configs/research/thermal_spatial_groups_f13_0_fulltrain_v1.json \
  --device cuda:0 --epochs 200 \
  --output /data2/wanglz/ModularDT/relocated_workspace/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/thermal_spatial_groups_r13_f13_0/fulltrain

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n ModularDT python tools/honf_research.py train-contextual \
  --config configs/research/thermal_spatial_groups_f13_1_fulltrain_v1.json \
  --device cuda:0 --epochs 200 \
  --output /data2/wanglz/ModularDT/relocated_workspace/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/thermal_spatial_groups_r13_f13_1/fulltrain
```

The first-segment stop is user-controlled. If the first 200-epoch segment is interrupted before reaching 200, resume its latest state to that same requested stop. `--stop-after-epoch` is an absolute epoch in that run. Further segments require a new user decision and a finite budget; the 1000-epoch configuration horizon does not launch or authorize them automatically.

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n ModularDT python tools/honf_research.py resume-contextual \
  --run /data2/wanglz/ModularDT/relocated_workspace/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/thermal_spatial_groups_r13_f13_0/fulltrain \
  --device cuda:0 --stop-after-epoch 200
```

The same resume command applies to the `_f13_1/fulltrain` root. A later process may select a different physical GPU by changing `CUDA_VISIBLE_DEVICES` while keeping exactly one device visible and continuing to pass `--device cuda:0`; the one-stream qualification verifies this visibility shape and the configs do not pin UUIDs. Do not launch a full-TRAIN run or continue beyond a reviewed segment automatically.

The guide's commands establish an executable handoff, not completed formal training or a winner between F13-0 and F13-1; use the linked development report for current results, measured cost, and the evidence-based recommendation when available.
