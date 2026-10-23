# Round10 reusable core and endpoint guide

Round10 separates physical adapters, the trained retrieval core, execution backends, and dataset-specific evaluation. Use a literal endpoint together with its own source manifest, recipe, normalization/provider identity, native receiver policy, and recorded run lineage. A current workspace import alone does not reproduce a frozen experiment. The [final report](../reports/_bk/20261018_20261023_Round10_Continuous_Unified/20261021_143334Z_HONF_Round10_Continuous_Unified_Organization_Report.md) and [evidence index](../reports/_bk/20261018_20261023_Round10_Continuous_Unified/20261021_143334Z_HONF_Round10_Continuous_Unified_Organization_Evidence_Index.md) give the tested endpoints and exact local artifacts.

## Capabilities and model-card summary

| Mode | Trained evidence | Physical capability and limits | Appropriate use |
|---|---|---|---|
| Legacy dense M and dense M+E | Literal non-U e5000 endpoints and native boards | Preserve the original Thermal physical readout and Wind physical-velocity adapters; E utility is not independent environmental physics | Required same-age controls and retained standalone reconstruction |
| Mature hard H9-ME/S9-ME | Literal e5000 endpoints; fixed TRAIN/DEV fields | Original hard-routing functions and their historical numerical/seam limits remain identifiable | Assess maturation without changing historical architecture or data |
| Continuous C10-H/C10-S | Literal e5000 endpoints, native boards, and complete-cycle cost/memory measurements | Exact-positive native retrieval and E-conditioned M response; no causal interpretation of learned support; H/S work is not jointly matched | Evaluate final field fidelity, actual original-record reads, continuity, and complete runtime/memory separately |
| Unified local-input U-D/U-H/U-S | Six separately authorized exploratory e3000→e5000 fits active; earlier one-epoch qualifications remain engineering evidence | Applied typed contextual preparation with removed aggregate/global value paths; final trained field quality remains unmeasured | Monitor the exact separate Thermal/Wind stages and evaluate retained milestones; original automatic-admission gate remains unpassed |

All reported DEV metrics are exposed development evidence. Wind TEST remains locked. Saved-candidate ranking is limited to its compatible physical reference catalogue, and a good rank is not evidence of valid generated designs. Full access and all-open mathematical controls must remain separately labelled. Counter unavailability is not zero computation, and selected value-row reduction is not an elapsed-time or peak-memory measurement.

## Interfaces to retain

`honf_forward_core.continuous_joint.model.ContinuousJointCore` owns the continuous routing and native decode equations; its typed reader applies selected original values with the trained coefficients. Dataset adapters retain native physical extraction, geometry, normalization, receiver identities, and response laws. `honf_runtime.unified_training.TrainingEngine` retains optimizer/schedule/case/query semantics. `honf_runtime.joint_native_evaluation` computes physical errors and typed work; its denominator-alias correction changes reporting only, and saved original boards must stay immutable.

Standalone endpoint packages are authenticated by the 12-member export closure. Its initial isolated loaders and saved-input native replay follow-up are completed tests, rather than file-existence checks. Reuse the authenticated package/source identity instead of exporting duplicate checkpoint families. Do not replace a literal e5000 endpoint with `best` or attach another endpoint's cost to it. A deployment or new scientific training campaign needs its own scope and resource authorization.

## Tested read-only CLI examples

The following commands were actually executed successfully using the ModularDT Conda interpreter with CUDA hidden. They inspect supported interfaces or retained status and do not launch training, create a package, reserve GPU capacity, or run evaluation. Their captured return codes and output are in the [CLI check receipt](../../diagnostics/generated/non_U_final_postprocessing_v1/reusable_core_CLI_checks_v1.json).

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /home/wanglz/miniconda3/envs/ModularDT/bin/python -B \
  /home/wanglz/Desktop/src/ModularDT/HONF_Proj/tools/continuous_joint_train.py --help
```

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /home/wanglz/miniconda3/envs/ModularDT/bin/python -B \
  /home/wanglz/Desktop/src/ModularDT/HONF_Proj/tools/continuous_joint_train.py status \
  --output /data2/wanglz/ModularDT/diagnostics/continuous_joint_round10_20261018T153522Z/runs/continuous_read/thermal/H
```

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /home/wanglz/miniconda3/envs/ModularDT/bin/python -B \
  /home/wanglz/Desktop/src/ModularDT/HONF_Proj/tools/joint_native_evaluate.py --help
```

The public parser's available commands do not override a stage's scope or resource allocation. Actual Round10 operations use their immutable scientific sources plus separately audited resource/device/bootstrap compositions; invoking a historical raw command directly can select an obsolete resource alias. Use the retained completed receipts for reproduction and inspect their exact source/argv/input bindings before a replay. Operational compatibility projections must remain explicit and cannot manufacture missing scientific or completion evidence.

The public continuous native evaluator accepts relocated response-reference path labels only when both labels name the same physical file and its bytes match the checkpoint's saved SHA256. Wind cache path labels require the same physical directory. All remaining provider/normalizer identity fields must match exactly; equal duplicate contents in a different file do not qualify. The evaluator exposes the validated aliases in its provenance. Fresh CPU provider reloads of both literal C10-H e5000 endpoints passed after the `/data2` relocation; this is loader compatibility, not a new physical measurement or a change to the frozen fitting source.

## Final evidence and future work

Prefer the final e5000 reconstruction, original-ID organization/support, continuity, response, saved-candidate decision, and measured cost figures. Intermediate training snapshots support lineage and monitoring but are not additional independent results. Preserve selected numerical arrays, declared checkpoint milestones/latest/best aliases, raw failures, manifests, and reusable caches. New derived figures and one-time helpers belong under ignored `/data2` storage with declared budgets and cleanup receipts.

The [U feasibility proposal](../../UpgradePlan/ThemalChannel/20261018_20261023_Round10_Continuous_Unified/20261021_050444Z_HONF_Unified_Organization_Next_Stage_Feasibility_Proposal.md) is the historical planning document. Later [Thermal](/data2/wanglz/ModularDT/thermal_U_triplet_next_stage_20261021/receipts/user_launch_authorization.json) and [Wind](/data2/wanglz/ModularDT/wind_U_triplet_next_stage_20261021/receipts/user_launch_authorization.json) authorizations launched complete exploratory triplets with the original gate explicitly unpassed. Preserve those frozen trajectories and their separate resource ledgers. The current integrated execution completes their diagnostics and exports without silently certifying old qualification failures or launching replacement training.
