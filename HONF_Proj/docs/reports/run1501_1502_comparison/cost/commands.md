# Cost benchmark reproduction

The benchmark command below reproduces the completed protocol saved in
`inference_cost_cuda2.json`. Run from `HONF_Proj/`. It uses the same 90 case
IDs selected by the accuracy pass and checkpoints at the paths recorded in the
JSON. Each checkpoint retains its own configured inner receiver chunk; the
outer `predict_case` query batch remains 32,768.

```bash
rtk proxy env PYTHONPATH=src:Case_ThermalChannel/src /home/wanglz/miniconda3/envs/ModularDT/bin/python \
  docs/reports/run1501_1502_comparison/cost/benchmark_inference_cost.py \
  --checkpoint 1404=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1404_20260916_092508_routing_only_pairwise/best_by_field_mse_model.pt \
  --checkpoint 1804=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1804_20260905_081349_dense_pairwise_field_adaptation/best_by_field_mse_model.pt \
  --checkpoint 1501=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1501_20260922_211056_sparse_incidence_adaptive_honf/best_by_field_mse_model.pt \
  --checkpoint 1502=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1502_20260923_004751_sparse_incidence_environment_sparsemax/best_by_field_mse_model.pt \
  --case-list docs/reports/run1501_1502_comparison/accuracy/evaluation/logs/selected_cases.csv \
  --output docs/reports/run1501_1502_comparison/cost/inference_cost_cuda2.json \
  --dataset /data/wanglz/ModularDT/1_ChannelThermal/Processed_ChannelThermal_Dataset/packed_dataset.h5 \
  --split test --query-count 8192 --query-batch-size 32768 \
  --mixed-teacher-ratio 0.5 --warmups 1 --repetitions 3 --device cuda:2
```

Rebuild summary tables and the figure from the saved JSON without using a GPU:

```bash
rtk proxy python docs/reports/run1501_1502_comparison/cost/summarize_cost_evidence.py
```

The superseded first-pass output is not reproduced by this command. Its
mistaken inner-chunk override and exclusion from conclusions are documented in
`superseded/README.md`.
