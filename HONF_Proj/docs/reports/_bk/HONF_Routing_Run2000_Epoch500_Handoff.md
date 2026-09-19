# Run 2000: paused analysis handoff

The user requested a pause after preparing the epoch-500 analysis tools. **Leave the existing training process running to its authorized epoch 500. Wait for the user's instruction before executing the final analysis below.** Do not restart training, launch another run, or continue beyond 500.

The implementation and pretraining checks are recorded in the [draft report](HONF_Routing_Run2000_Epoch500_Report.md). Its pending sections must be completed from actual endpoint artifacts; it is not a completed epoch-500 assessment. The [comparison draft](HONF_Dynamic_Sparse_Routing_Comparison_Report.md) keeps Goals 2 and 3 pending.

## Existing execution

- Run: `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_2000_20260915_225542_routed_module_hubs`.
- Training log: `diagnostics/generated/interface_operator_study/dynamic_sparse_routing/run2000/training.log`.
- Physical GPU 2, logical `cuda:0`; fresh initialization; ordinary allocator; requested terminal epoch 500.
- At epoch 50, validation field MSE was 0.232619, first-batch preclip/update norms were 2.94970 / 0.131763, and router norms were 0.0154323 / 0.0266564. Losses were finite and updates active. This is early evidence, not final quality assessment or proof of monotonic learning.
- Completed tests: 33 focused routing/integration tests, eight endpoint-reducer tests, and 124 compatibility tests. Exactly two disposable real optimizer checks and the corrected actual pretraining profile completed before the sole managed launch.
- Pretraining numerical artifacts, smoke results, and commands are under the study directory. Do not rerun optimizer checks or retrain baselines.
- CPU tool-readiness artifacts are retained under `numerics/tool_checks/`: Legacy real/tiny-synthetic execution with realized E counts, exact active-only Dense QM agreement, and disabled omitted-audit behavior all report zero failures. Their original temporary output paths remain in the recorded execution metadata. The corrected routing preview was visually inspected for phase separation, padding, and actual epoch-10-to-50 turnover.

Concurrent work for other tasks appeared during preparation of this handoff, including mean-shift routing and generalization of the shared reducer/training plotter. Those edits were preserved and are outside this task's implementation and test claims. Run-2000 CLI defaults remain compatible. Do not revert or include unrelated staged/unstaged work in a Goal-1-only commit; inspect the current tree when resuming.

On resumption, inspect the existing process, allocator manifest, contiguous metrics through epoch 500, and actual checkpoint epoch metadata. If training has genuinely failed, report the failure before considering any further training. The commands here do not restart or continue training.

## Analysis commands: prepared, not executed

Run from `/home/wanglz/Desktop/src/ModularDT/HONF_Proj` after the user asks to continue and GPU 2 is free of the existing training process. All model measurements are sequential on physical GPU 2.

```bash
routing_python=/home/wanglz/miniconda3/envs/ModularDT/bin/python
routing_run=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_2000_20260915_225542_routed_module_hubs
routing_study=diagnostics/generated/interface_operator_study/dynamic_sparse_routing/run2000
routing_driver=tools/diagnostics/run_dynamic_sparse_routing_study.py
routing_exec=(rtk proxy env CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src:Case_ThermalChannel/src:tools:tools/diagnostics OMP_NUM_THREADS=4 "$routing_python")

"${routing_exec[@]}" "$routing_driver" endpoint --device cuda:0 \
  --exact-checkpoint "run2000=$routing_run/epoch_0500_model.pt" \
  --best-checkpoint "run2000=$routing_run/best_by_field_mse_model.pt" \
  --evaluation-dir "$routing_study/endpoint500/comparison" \
  --output "$routing_study/endpoint500/endpoint.json"

"${routing_exec[@]}" tools/diagnostics/reduce_routing_endpoint.py \
  --endpoint-table "$routing_study/endpoint500/comparison/tables/per_case_metrics.csv" \
  --output "$routing_study/endpoint500/analysis"

"${routing_exec[@]}" tools/diagnostics/render_routing_training.py \
  --run-dir "$routing_run" --output "$routing_study/figures/training"

"${routing_exec[@]}" "$routing_driver" interventions --device cuda:0 \
  --checkpoint "exact500=$routing_run/epoch_0500_model.pt" --query-count 8192 \
  --output "$routing_study/interventions/exact500.json"

"${routing_exec[@]}" "$routing_driver" ledger --device cuda:0 \
  --checkpoint "exact500=$routing_run/epoch_0500_model.pt" --query-count 8192 \
  --no-missed-source-audit --save-maps --maps-dir "$routing_study/routing/selected" \
  --csv "$routing_study/routing/full_grid.csv" --output "$routing_study/routing/full_grid.json"

"${routing_exec[@]}" "$routing_driver" ledger --device cuda:0 \
  --checkpoint "exact500=$routing_run/epoch_0500_model.pt" --query-count 32 \
  --case-id 0273 --case-id 0653 --csv "$routing_study/routing/missed_sources_32.csv" \
  --output "$routing_study/routing/missed_sources_32.json"

for routing_case in 0273 0653 0283 0298 0302; do
  "${routing_exec[@]}" "$routing_driver" turnover --device cuda:0 --case-id "$routing_case" \
    --checkpoint "epoch10=$routing_run/epoch_0010_model.pt" \
    --checkpoint "epoch50=$routing_run/epoch_0050_model.pt" \
    --checkpoint "epoch100=$routing_run/epoch_0100_model.pt" \
    --checkpoint "epoch250=$routing_run/epoch_0250_model.pt" \
    --checkpoint "epoch500=$routing_run/epoch_0500_model.pt" \
    --output "$routing_study/routing/turnover_$routing_case.json"
done

"${routing_exec[@]}" tools/diagnostics/routing_derivative_checks.py anchors --device cuda:0 \
  --checkpoint "exact500=$routing_run/epoch_0500_model.pt" --case-id 0273 --case-id 0298 \
  --query-count 32 --output "$routing_study/numerics/physical_derivatives_epoch500.json"
```

The endpoint command evaluates both policies over the same complete 90-case development population, using predicted ports and checkpoint-owned normalization. Accuracy keeps configured receiver chunk 128. Saved-best means validation-field selection through 500; record its actual epoch. The reducer reuses existing exact-500 Legacy, Dense, and Regional tables, validates matching populations/denominators/strata, and writes pooled, equal-case, channel, physical, stratum, tail, and paired summaries. Do not substitute mature parent best weights.

## Process-isolated timing

Each following profile invocation loads one model in its own process. Default real anchors are 0273/0653 with 8,192 queries, two warmups and five repeats. Large shapes use one warmup and three repeats. Main inference uses receiver chunk 2048 and outer query batch 32768. Legacy has no equivalent internal receiver-chunk override; report its actual native behavior, not a fictitious chunk value. The separate `full_physical_outer_batched` scope includes physical preparation and the prescribed outer decode loop; `full_forward` remains the direct all-query call.

```bash
routing_dense=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1804_20260905_081349_dense_pairwise_field_adaptation/epoch_0500_model.pt
routing_legacy=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1401_20260823_151126_stage7_modern_structured_context/epoch_0500_model.pt
for routing_spec in "routed=$routing_run/epoch_0500_model.pt" "dense=$routing_dense" "legacy=$routing_legacy"; do
  routing_label=${routing_spec%%=*}
  "${routing_exec[@]}" "$routing_driver" profile --device cuda:0 --checkpoint "$routing_spec" \
    --large-shape 32 768 65536 --large-shape 128 3072 262144 \
    --output "$routing_study/timing/epoch500_$routing_label.json"
done

"${routing_exec[@]}" "$routing_driver" profile --device cuda:0 \
  --checkpoint "routed=$routing_run/epoch_0500_model.pt" --case-id 0273 \
  --receiver-chunk-size 128 --no-active-only-reference \
  --output "$routing_study/timing/epoch500_routed_chunk128.json"

"${routing_exec[@]}" "$routing_driver" profile --device cuda:0 \
  --checkpoint "routed=$routing_run/epoch_0500_model.pt" --case-id 0273 \
  --large-shape 32 768 65536 --warmups 0 --repetitions 1 \
  --large-warmups 0 --large-repetitions 1 --no-active-only-reference --trace \
  --trace-dir "$routing_study/timing/epoch500_traces" \
  --output "$routing_study/timing/epoch500_trace_attribution.json"
```

Use the five/three-repeat unprofiled rows for latency conclusions, not the trace-attribution invocation. Keep actual allocation failures as measured limitations. Do not silently lower the shape or introduce a route cap. Separate active M from Mpack, raw paths from unique pairs, source fanout from Dk, all-valid-hub Dk from occupied-only Dk, and source/query active K. Separate live, incremental peak, total peak allocated, and reserved memory. Active-only inherited Dense QM timings remove source padding and require output agreement; they are not an end-to-end routing speedup.

Read parameter counts from fully materialized checkpoint models, checkpoint bytes from the actual final files, and optimizer tensor storage from the saved optimizer state. Do not infer optimizer storage by multiplying parameter count by an assumed constant. The early CPU inventory is `numerics/parameter_counts.json`; update its final-checkpoint observations when analysis resumes.

## Figures and final report

Use `tools/diagnostics/render_routing_diagnostics.py` with `--ledger "$routing_study/routing/full_grid.json"`, `--figure-dir "$routing_study/figures/routing"`, `--strict`, and five repeatable `--turnover "$routing_study/routing/turnover_CASE.json"` arguments. Supply `--endpoint-links PATH` with a JSON object mapping each anchor to the existing endpoint field plot paths. Check the actual generated paths after evaluation; do not invent them or copy baseline field arrays. The renderer accepts existing local files or HTTP(S) links, never numeric ledger values as links. Inspect the generated routing, field, and training PNGs and the index links.

Complete the draft report and comparison with exact versus selected epochs, full accuracy and thermal/tail evidence, phase interventions, signed derivative results, sparse-work counts, timing/memory and profile attribution, commands/artifact paths, and deviations. Position AD/FD steps are `0.01r` and `0.005r`; heat steps are the previously fixed `0.01` and `0.005` normalized input units. Report physical conversions from checkpoint statistics and finite-difference cancellation honestly. A missing omitted pair in an actually dense route is an unavailable formal direct-omission probe, not a failed zero-effect theorem; the separate production-reader numerical fixture covers the conditional identity.

Recheck whether trustworthy responses to the existing 16 independent reference requests have arrived. Do not start solver development. Formal `barrier benefit: Evidence Missing` remains unless appropriate solver-labelled layouts actually exist. No route map or model-versus-itself derivative is a physical validation certificate.

Treat 500 as an early assessment. Give an evidence-based recommendation and an exact **unexecuted** same-run continuation command only if warranted. Do not launch continuation, sweeps, another managed run, Goal 2, or Goal 3. Preserve the user's pre-existing staged documentation moves when committing only this task's files.


## Recovery on 2026-09-16

The user requested verification and continuation to epoch 500 after observing the epoch-100 milestone. Inspection found original PID 3684940 and tool session absent, GPU 2 idle, metrics through epoch 123, and the log ending during epoch 124. The old running manifest was stale. No traceback or kernel OOM evidence was found around the stop; cause is unconfirmed.

Resumed the same managed directory from best_model.pt at epoch 123, the newest complete saved state, restoring model, optimizer and RNG through the existing workflow. latest_model.pt was only epoch 120. No new run was allocated. Physical GPU 2, terminal epoch 500. Launch used subprocess.Popen(start_new_session=True, stdin=DEVNULL), with stdout/stderr appended to resume_epoch123_to500.log in this study directory, independent of the interactive command session. Launch PID: 4107994. Startup log confirms epoch 124 / 500. Final analysis remains deferred.

Executed from HONF_Proj with CUDA_VISIBLE_DEVICES=2, PYTHONPATH=src:Case_ThermalChannel/src, OMP_NUM_THREADS=4:
```bash
/home/wanglz/miniconda3/envs/ModularDT/bin/python -u train.py --config project://src/config_core/forward/routing_module_hubs_context.json --workflow forward --device cuda:0 --epochs 500 --resume-checkpoint Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_2000_20260915_225542_routed_module_hubs/best_model.pt --yes
```
