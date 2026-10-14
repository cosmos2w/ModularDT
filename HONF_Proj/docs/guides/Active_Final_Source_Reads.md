# Active final source reads

The opt-in read plan removes absent or explicitly omitted nonlinear final source–receiver reads before the expensive source MLP. The default call still uses the historical dense path. It preserves each original source measure, source identity, nonlinear head and source-axis reduction order; it never renormalizes over retained donors. Every source/environment context route and Thermal affine temperature/interface/material read stays dense. No new model parameters, buffers, checkpoint schemas or model registry are introduced.

The 2026-10-13 TRAIN deletion study did not qualify active-read omission for either primary host. Use full access for those hosts. The `selected` interface supports controlled research and numerical checks; it is not a released sparse policy or an inverse-ready derivative path. See [the experiment report](../reports/_bk/20261013_184402Z_HONF_R1_Consolidation_and_Sparse_Reads_20261013.md) for measured bounds, preserved failures, and host identities.

## Common core

Use an existing prepared context and its exact receiver tensors. A plan binds the model object and parameter versions, prepared context, original source IDs, receiver tensors/features/IDs, policy generation, receiver generation and admitted pair versions. Rebuild after changing any of those objects or weights. Restoring parameter bytes after an in-place edit does not restore tensor versions and does not make an old plan valid again.

```python
# This example uses a supported direct InteractionPreservingJointCore.
# prepared, receivers, features and receiver_ids come from the native caller.
model.bind_read_policy_generation("full_access_v1", 0)
plan = model.prepare_read_plan(
    prepared, receivers, features,
    mode="active_dense",
    policy_identity="full_access_v1", policy_generation=0,
    receiver_generation=0, receiver_ids=receiver_ids,
)
prediction = model.predict_fields(  # Use predict(...) for a plain NonlinearFieldReadout.
    prepared, receivers, features,
    read_plan=plan, receiver_ids=receiver_ids,
    read_policy_identity="full_access_v1", read_policy_generation=0,
    receiver_generation=0,
)
print(plan.active_pair_count, plan.evaluated_pair_count,
      plan.padding_pair_count, plan.omitted_active_pair_count)
```

For a declared diagnostic subset, pass `mode="selected", selected_pairs=mask`, where `mask` is a Boolean `[B,Q,M]` tensor on the native device and excludes every absent source. The caller must fix selection before observing evaluation errors. Neither physical source IDs nor case IDs are learned numerical features. The primitive validates admission; it does not supply a learned scorer or certify a subset's physical accuracy.

If every physical pair is admitted and there is no padding, execution may dispatch directly to the legacy reader. Otherwise it gathers admitted features, evaluates only admitted value rows in bounded row batches, stores an admitted-value buffer, then scatters weighted rows into the original source-axis workspace for reduction. That dense reduction workspace and admitted-value buffer have real memory costs. Count actual hooked MLP rows and complete model work; a mask count alone is insufficient. Query tiling changes batching and scratch capacity, not the selected support.

## Wind physical outputs

The Wind adapter requires the exact feature tensor used to construct the plan; it must not reconstruct an equal-valued but differently bound feature object during decoding. The maintained physical conversion still applies the checkpoint-owned velocity normalizer and prescribed profile in the original order.

```python
from windfarm.joint_regional import wind_receiver_features

features = wind_receiver_features(prepared.scene, receivers_D)
model.core.bind_read_policy_generation("full_access_v1", 0)
plan = model.core.prepare_read_plan(
    prepared.core_context, receivers_D, features,
    mode="active_dense", receiver_ids=receiver_ids,
    policy_identity="full_access_v1", policy_generation=0,
    receiver_generation=0,
)
velocity_mps = model.predict_physical(
    prepared, receivers_D, receiver_features=features,
    read_plan=plan, receiver_ids=receiver_ids,
    read_policy_identity="full_access_v1", read_policy_generation=0,
    receiver_generation=0,
)
```

The primary execution host is the strict self-contained migration of Wind2202 P e5000 with its original E8 context and TRAIN420 normalization. Its archival optimizer moments are not resumable field-training state. The fresh Wind P/R1 e2500 pair is a separate learning comparison. There is no mature R1 execution host in this round.

## Thermal native curl and affine outputs

Prepare the native geometry, solid mask and complete curl stencil first. Admission applies to `flow_query_receivers`, including required neighbor nodes; requested primary query count alone does not price that work. Temperature, K, surface/material support and heat increments use the existing dense affine path. Current heating is excluded from the flow policy.

```python
import torch

receiver_ids = prepared.native_curl_stencil.node_indices
if receiver_ids.ndim == 1:
    receiver_ids = receiver_ids[None]
model.core.bind_read_policy_generation("full_access_v1", 0)
plan = model.core.prepare_read_plan(
    prepared.context, prepared.flow_query_receivers, None,
    mode="active_dense", receiver_ids=receiver_ids,
    policy_identity="full_access_v1", policy_generation=0,
    receiver_generation=0,
)
outputs = model.apply_native(
    prepared, physical_heat,
    flow_read_plan=plan, flow_receiver_ids=receiver_ids,
    read_policy_identity="full_access_v1", read_policy_generation=0,
    receiver_generation=0,
)
increments = model.apply_native(
    prepared, delta_heat_fp64, increment=True,
    accumulation_dtype=torch.float64,
)
```

Do not attach a flow plan to `increment=True`; that combination is rejected. Compute nonlinear peak changes from the separately evaluated endpoints, never from the maximum of a linear increment. Dense smooth inference remains the default for geometry derivatives and inverse consumers; fixed-support VJP parity does not qualify derivatives through changing support.

## Exact Wind continuation handoff

The existing pair has completed its original horizon. P is `WMV1-963fb63fbd7a4685a645d0cf5dd9bc40`; R1 is `WMV1-621070cb7f8e479684adbcd2be68fd06`. Both retain their literal e2000 states and new e2100–e2500 milestones, latest/best aliases, moments and complete observed streams. No further resume or horizon extension is authorized here.

The retained executable handoff delegates to the maintained `tools/joint_regional_train.py resume` and `TrainingEngine` in the clean sealed commit `8c13d09b24c404991bef2df7dbff0f15990ab238`; its only runtime adaptation binds already verified shared input/cache paths in memory. It checks the original source seal, recipe, allocated run ID, exact latest checkpoint, named AdamW moments, exposed membership, counters and prior stop records before fitting. Current source code must not be substituted or its guards weakened to resume these historical states.

These read-only commands inspect the retained handoff and attained endpoints from the project workspace using the ModularDT environment:

```bash
conda run -n ModularDT python -B /data/wanglz/ModularDT/diagnostics/r1_sparse_reads_20261013_115851/qualification/r1_exact_resume_wind_8c13.py --help
conda run -n ModularDT python -B -c 'import json,pathlib; p=pathlib.Path("/data/wanglz/ModularDT/diagnostics/r1_sparse_reads_20261013_115851/receipts/wind_r1_e2500_matched_history_review_v1.json"); print(json.dumps(json.loads(p.read_text()),indent=2))'
```

The performed block interface was `--arm p|p_r --stop-after 2100|2200|2300|2400|2500`, with a reviewed immutable `--preflight-only` receipt before each root resource-wrapper launch. Those commands now reject restarting a completed block. They are a reproducible handoff of the executed continuation, not instructions to reset latest aliases, overwrite prior states, or start another field fit. The sealed checkout and campaign runners remain local scientific artifacts; source data, models and generated evidence are excluded from Git.

## Focused qualification

The focused checks below run without CUDA or optimizer steps and cover admission, stale identities, source/receiver VJPs and direct-mode forwarding. Native CPU FP64 and GPU FP32 qualification receipts in the report additionally bind the real low/high-M hosts. Inference qualification does not establish sparse training acceleration.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=HONF_Proj/src:HONF_Proj/Case_WindFarm/src:HONF_Proj/Case_ThermalChannel/src conda run -n ModularDT python -B -m pytest -q -p no:cacheprovider HONF_Proj/tests/test_shared_interaction_core.py -k 'read_plan or active_dense_plan or selected_plan or all_active_plan'
```
