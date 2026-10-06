# Thermal physical response refinement

`tools/thermal_response_refinement.py` prepares a matched separable-interface H-add and joint-interface H-joint refinement. It inherits each arm's exact `epoch_0100_model.pt` from the retained receiver fit mapping, restores the embedded frozen Stage-A and global/local normalization, and creates a fresh optimizer at refinement age zero. Preparation performs no learning, model evaluation, or solver call.

The default identities are Run3801 and Run3802. Their histories are physical backbone 1000, interface prefit 100, refinement 0. The primary fixed25_v1 manifest remains unchanged: 150 primary training cases and 22 exposed validation cases. The separately declared response addendum uses original-TRAIN families 0001/0318/0333/0348; response-development families 0304/0320/0335/0350 and fixed-audit families 0277/0291/0294/0687 never enter refinement gradients or coefficient calibration. In the retained cohort, 0348 is primary training and the other three fit anchors are extra auxiliary families. This is development evidence, not an independent test.

Prepare both arms only after saving the shared, pooled, training-only calibration JSON. It must contain positive finite `response_coefficient` and `null_coefficient`; preserve the measured gradient norms and training-panel provenance in the same JSON. Preparation copies this identical declaration into both configurations and binds saved response NPZ/JSON content hashes to resume identity.

```bash
/home/wanglz/miniconda3/envs/ModularDT/bin/python tools/thermal_response_refinement.py \
  --prepared-fits /data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/continuation/prepared_interface_fits.json \
  --calibration-json /data/wanglz/ModularDT/thermal_development/response_refinement_20261005/preparation/shared_train_calibration.json \
  --output-root /data/wanglz/ModularDT/thermal_development/response_refinement_20261005
```

Run the command from `HONF_Proj`. The calibration and output paths are explicit local choices; the file must already exist. The tool writes ordinary native run directories, `attached_refinement0000.pt`, `forward_refinement_attachment.json`, 100-epoch profiles, and `prepared_response_refinements.json`. That mapping supplies the exact launch/resume commands. It never launches them. The two tracked JSON files under `src/config_core/forward/thermal_response_refinement/` are unbound override templates, not direct training configurations.

The maintained `query_physical_v1` allowlist trains `core.backend.tensor_query_interaction.*`; the fine `mm_message`, `me_message`, `em_message`, `module_update`, `env_update`, `query_module_message`, `query_module_output`, `env_query`, `env_attention`, and `env_geometry_bias` components; `core.common.field_head`; and `local_coupling.port_head`/`port_refinement_head`. All other parameters remain frozen, including encoders, coarse/local context, inherited global organizer/calibration, and Stage-A. Normalization buffers remain fixed. AdamW uses organizer 1e-4, physical 1e-5, and weight decay 1e-5; the attachment receipt lists exact parameter names and scalar counts.

`query_admission_mode="soft"` fixes the numerical admission/donor formulation independently of refinement age while retaining learned input-dependent memberships. Plans remain fresh functions of each design input and are shared through that wrapper's physical phases. Old checkpoints retain the default `curriculum` formulation, and historical `interface_fit` guards reject a silent switch to this refinement mode.

The initial profiles stop at 100 new epochs. Resume the exact saved milestone after the paired 100 review supports continuation to 500; any 1000 extension requires the separately declared evidence and remaining budget. The following are the exact commands used for the reviewed 100-to-500 continuation in this round, run from `HONF_Proj`:

```bash
CUDA_VISIBLE_DEVICES=1 /home/wanglz/miniconda3/envs/ModularDT/bin/python train.py \
  --config /data/wanglz/ModularDT/thermal_development/response_refinement_20261005/preparation/h-add_refinement100.json \
  --resume-checkpoint /data/wanglz/ModularDT/thermal_development/response_refinement_20261005/ThermalChannel/HONF_Forward_Runs/Run_3801_20261005_214458_thermal_response_refinement25_h-add_v1/epoch_0100_model.pt \
  --epochs 500 --device cuda:0 --yes
CUDA_VISIBLE_DEVICES=2 /home/wanglz/miniconda3/envs/ModularDT/bin/python train.py \
  --config /data/wanglz/ModularDT/thermal_development/response_refinement_20261005/preparation/h-joint_refinement100.json \
  --resume-checkpoint /data/wanglz/ModularDT/thermal_development/response_refinement_20261005/ThermalChannel/HONF_Forward_Runs/Run_3802_20261005_214502_thermal_response_refinement25_h-joint_v1/epoch_0100_model.pt \
  --epochs 500 --device cuda:0 --yes
```

These commands document the existing run histories; do not launch a duplicate trainer over a live or completed run. For another authorized refinement, use its exact mapped paths and supported saved age. Ordinary resume rejects changed trainable scope, admission mode, normalization, primary identity, response addendum/content, objective declaration, coefficients, or parent lineage. It retains optimizer state after initial attachment. Keep only latest, best-field, and declared 100-epoch milestones/plots; numerical artifacts and figures stay in ignored local paths.

No new solver attempt, inverse generator, design search, or formal full-data run belongs to this workflow. The cumulative physical-reference ledger remains 326/326. CPU tests and preparation receipts establish software contracts; actual learning, physical responses, runtime, organizer utility, and stored-pool decisions require their separately recorded measurements under the [response refinement plan](../../UpgradePlan/HONF_Response_Competent_Forward_Refinement_Plan.md).
