# Thermal port and response adaptation

The Thermal-owned `thermal_port_response_v1` provider composes the joint R14 predictor with the declared frozen local operator. One global preparation supplies both the direct joint fields and the complete predicted 64-angle outside-temperature sequence. Prescribed harmonic-conductivity/native-offset coefficients condition the local operator. The newly computed local response is not fed back into that inference's global state. The signed q observable remains the saved finite-offset proxy.

## Candidates and evidence

| Recipe arm | Supervision alongside the inherited native objective |
| --- | --- |
| C0 | Local material/surface/q targets and raw direct/local consistency at the ordinary 16 sampled interface angles |
| P64 | Outside temperature at all 64 angles and reference-anchored 64-angle compatibility; the reference-conditioned local call is stopped-gradient and used only in the supervised loss |
| R64 | P64 plus four TRAIN-origin response packs per epoch; each pack shares a baseline between a saved single geometry edit and a balanced heat edit |

Full and restricted training reads share preparation but retain separate predicted ports and local outputs. The frozen local parameters stay in evaluation mode; gradients through its predicted conditioning inputs remain active. Original member IDs, joint u/v/p/temperature prediction, native masks and physical units are preserved. The composition introduces no field head or wider module record.

Response normalization uses only FIT single/heat labels, separately by edit class and physical role, with a floor of 0.01 times the inherited absolute role scale. Heat-to-flow null supervision uses 0.05 times the inherited absolute flow scales. Each thermal physical role averages its direct/local readouts before the four physical roles are averaged. Pair-specific q masks remain distinct from temperature masks. The response sampling namespace does not consume the primary query generator. Four baseline/single/heat packs mean 12 additional physical design evaluations per epoch before branch accounting.

The loss history distinguishes the active-pack response mean from its actual contribution averaged across all optimizer updates. Report predictor fields, ports/local composition, finite responses, and organization/guidance separately. Exposed development families are selection evidence, and the frozen local asset has historical TRAIN600/DEV22/VAL89 exposure.

## Execution

Use the maintained runner and an explicit recipe. The C0 grid recipes in `configs/research/` declare a 100-epoch TRAIN600 adaptation from literal R14 epoch 5000, fresh constant-rate AdamW, inherited TRAIN transforms and independent run identity. G192, G96 and G72 create 24×8, 12×8 and 12×6 environmental cells; native 128×64 receivers and the physical extraction offset are preserved. Source checkpoint metadata is not rewritten.

```bash
python tools/relation_anchored_train.py inspect --recipe CONFIG.json --device cuda:0
python tools/relation_anchored_train.py fit --recipe CONFIG.json --device cuda:0 --output RUN --stop-after 100
python tools/relation_anchored_train.py fit --recipe CONFIG.json --device cuda:0 --output RUN --resume latest --stop-after 100
```

Verify the explicitly authorized physical GPU UUID and ownership before launch, map it through `CUDA_VISIBLE_DEVICES`, and use mounted DATA2 outputs under the round's combined disk budget. Formal TRAIN600 aliases belong in `Trained_Results/ThermalChannel/HONF_Forward_Runs/`; qualification outputs remain in diagnostics. The device argument refers to the visible ordinal. Initial and qualification runs receive sparse Luna monitoring under the repository instructions.

`load_thermal_port_response_checkpoint` reconstructs the strict adapter and original source age plus this run's fitting age from checkpoint metadata without reopening the dataset. The frozen local checkpoint remains an explicitly declared external asset. A same-lineage resume restores optimizer, RNG, provider state and complete history through the maintained engine.

Only one selected P64/R64 formal lineage may use its authorized completed-100-to-500 continuation. The runner's `--extend-horizon` changes the execution endpoint and future 100-epoch checkpoints while retaining the original constant optimizer schedule, recipe, data and losses. This support does not authorize every candidate or a control to continue. Complete the required 100-epoch review and guide analysis first; after a justified final continuation, verify resume/startup once and finish without attaching a monitor.

For offline organization reuse, use the [Thermal transfer-probe guide](Thermal_Organization_Transfer_Probe.md). Its full-mode field-plus-gradient and aligned/permuted organization controls test incremental information on saved feasible edits; learned support alone does not establish physical causality or a valid inverse design.
