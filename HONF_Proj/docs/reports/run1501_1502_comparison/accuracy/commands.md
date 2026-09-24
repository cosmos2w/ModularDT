# Accuracy evaluation commands and provenance

## Fresh Run 1501/1502 inference

Executed from `/home/wanglz/Desktop/src/ModularDT/HONF_Proj`:

```bash
rtk proxy env PYTHONPATH=src:Case_ThermalChannel/src /home/wanglz/miniconda3/envs/ModularDT/bin/python -m channelthermal.workflows.compare_models \
  --checkpoint-path Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1501_20260922_211056_sparse_incidence_adaptive_honf/best_by_field_mse_model.pt \
  --checkpoint-path Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1501_20260922_211056_sparse_incidence_adaptive_honf/best_model.pt \
  --checkpoint-path Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1502_20260923_004751_sparse_incidence_environment_sparsemax/best_by_field_mse_model.pt \
  --label Run1501_best_field_e4689 \
  --label Run1501_best_total_e4877 \
  --label Run1502_best_field_e4794 \
  --split test --case-ratio 1.0 --query-batch-size 32768 \
  --local-port-condition-mode predicted --return-routing-maps --save-debug-npz \
  --debug-case-id 0273 --debug-case-id 0653 \
  --anchor-case-id 0273 --anchor-case-id 0653 \
  --skip-figures --device cuda:2 \
  --output-dir docs/reports/run1501_1502_comparison/accuracy/evaluation \
  --dataset /data/wanglz/ModularDT/1_ChannelThermal/Processed_ChannelThermal_Dataset/packed_dataset.h5 \
  --saved-root Trained_Results/ThermalChannel/HONF_Forward_Runs
```

Run 1502's saved best-total and best-field model-state dictionaries were
verified bit-identical on CPU. The three-model pass therefore covers the
Run 1502 best-total variant without another forward evaluation. GPU 0 was
occupied by active WindFarm training; I used idle GPU 2 and did not change
`CUDA_VISIBLE_DEVICES`. The fresh pass took 2m51s and GPU 2 was released after
it completed.

## Reused baseline evaluation

Run 1404 and Run 1804 use their best-field checkpoint tables from
`diagnostics/generated/run1404_1406_1407_1804_best5000_accuracy_20260920/`.
The exact original CLI arguments are preserved in that directory's
`comparison_manifest.json`. The manifest records full `test` split evaluation,
case ratio 1.0, query batch size 32,768, predicted port conditions, the same
packed HDF5 dataset, and 90 cases. Its model labels are
`Run1404_best_field` and `Run1804_best_field`.

The baseline inference was evaluated on GPU 0 on September 20. To decide
whether to repeat it, I compared the evaluator and shared prediction paths to
the current source. `compare_models.py` has no changes since that evaluation.
Later edits to `evaluation/prepared.py` preserve routing/interaction diagnostics
across decoder query chunks; they do not alter `pred_field_grid`. Later
`interface_field_coupling.py`/core additions dispatch only to opt-in newer
architectures. Run 1404 uses the `legacy_honf` default, Run 1804 explicitly
uses `dense_pairwise_field`, and both architecture branches predate the
September 20 evaluation. The coordinate-scale broadcast normalization change
in the core is equivalent for the baseline tensor ranks. Exact matched case
IDs, target counts, and target SSE across the 13 reported quantities provide an
additional cross-evaluation consistency check. On that evidence, a new
baseline GPU pass was unnecessary.

## Rebuild CSVs and figures

Executed from `/home/wanglz/Desktop/src/ModularDT/HONF_Proj`:

```bash
rtk proxy env PYTHONPATH=src:Case_ThermalChannel/src /home/wanglz/miniconda3/envs/ModularDT/bin/python docs/reports/run1501_1502_comparison/accuracy/summarize_accuracy.py
```

This CPU-side script validates exact 90-case ID/order matching, joins the saved
per-case metrics, reconciles all target counts and SSEs, computes paired wins
and 10,000 paired case-bootstrap intervals (seed 20260923), and writes the
tables, JSON summary, PNG, and PDF. It does not load or modify model weights
except reading candidate checkpoint metadata on CPU.
