# Forward upgrade reference

This document freezes the behavior present before the topology-adaptive forward
upgrade. It is an acceptance reference, not a model-quality claim and not a
training result.

## Source state

- Captured: 2026-08-17 (America/New_York).
- Commit: `2afa84759858931a236321e0086750734466dcec`.
- Branch: `main`.
- Starting tracked worktree: `.gitignore` was already modified outside this
  task. The change was left untouched.
- Reference profile: `src/config_core/forward/enhanced_honf_pairwise.json`.
- Existing computation: six fixed projection edges, residual-concatenation
  mechanism path when enabled, context-fusion field decoding, softmax routing,
  and dense pair/edge execution.

## Static validation baseline

Commands were run from `HONF_Proj` in the maintained `ModularDT` conda
environment. GPU commands were restricted to physical GPU 0.

```bash
CUDA_VISIBLE_DEVICES=0 rtk conda run -n ModularDT pytest -q tests Case_ThermalChannel/tests
```

Result: `116 passed, 1 skipped in 14.32s`. The single skip was
`Case_ThermalChannel/tests/test_inverse_joint_training_integration.py:31`
because its optional local inverse integration artifacts were unavailable.

```bash
rtk python -m compileall -q src Case_ThermalChannel/src train.py evaluate.py
```

Result: exit code 0 with no diagnostics.

```bash
rtk python -c 'import json, pathlib, subprocess; files=[pathlib.Path(p) for p in subprocess.check_output(["git","ls-files","*.json"], text=True).splitlines()]; [json.loads(p.read_text()) for p in files]; print(f"validated {len(files)} tracked JSON files")'
```

Result: `validated 24 tracked JSON files`.

## Fixed-case checkpoint reference

The available autonomous Run 0002 checkpoint and packed ThermalChannel dataset
were evaluated without training:

```bash
CUDA_VISIBLE_DEVICES=0 rtk conda run -n ModularDT python evaluate.py \
  --config src/config_core/forward/enhanced_honf_pairwise.json \
  --workflow forward \
  --checkpoint Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_0002_20260813_140013_official_enhanced_honf_pairwise/best_predicted_model.pt \
  --dataset ../1_Demo_ChannelThermal/Data_Saved/Processed_ChannelThermal_Dataset/packed_dataset.h5 \
  --device cuda:0 --case-index 0 \
  --output-dir Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_0002_20260813_140013_official_enhanced_honf_pairwise/evaluations/forward_upgrade_goal0 \
  --organization-view all --routing-view summary --export-hypergraph-plan
```

Result: exit code 0. The selected test case was `0273`. Its fluid-field metrics
were MSE `0.0038193664274251728`, RMSE `0.061801022867143326`, MAE
`0.02098678718928614`, and relative L2 `0.024788913909518158` over 39,870
values. The existing schema-v2 plan contains 12 padded module slots, 192
environment tokens, six fixed hyperedges, and six active hyperedges. Its
`module_count=12` records padded width rather than the three active modules;
the topology-signature upgrade must remove this ambiguity without changing the
existing plan schema.

The generated reference artifacts are intentionally ignored by Git and remain
under:

```text
Trained_Results/ThermalChannel/HONF_Forward_Runs/
  Run_0002_20260813_140013_official_enhanced_honf_pairwise/
  evaluations/forward_upgrade_goal0/0273_20260817_194949/
```

Primary SHA-256 references:

| Artifact | SHA-256 |
|---|---|
| `best_predicted_model.pt` | `29f26aeb4c9d105da089548af5cc3cb10af073eb620c574058e91d911c674485` |
| `evaluation_outputs_predicted.npz` | `6b59da07ca45036986369578e903643e1f164c48f467b12a7fccb5453fdbe7fc` |
| `metrics_predicted.csv` | `2a39a6639beafb6f1ba19302c7967b8b9ade9494ed8f87e3a448cf1b43e7b0c2` |
| `hypergraph_plan.npz` | `dc45979e133ec361787376a5815c931035e5daf426cca0b45eaa7d67ea8fb6bd` |
| `hypergraph_diagnostics.json` | `3c2a3bbcb32a98a7fe03848ce88b48443c631f1a792aca899225b14e38e63753` |
| `summary_compact.json` | `87d7e942c7f40bc13c8a8ee0a59a533fb107a517c32b9be897b3502ea0967f93` |

## Parameter and dense-routing baseline

Loading the same self-contained checkpoint on CUDA produced:

| Count | Value |
|---|---:|
| Total model parameters | 3,507,625 |
| Trainable parameters | 2,472,486 |
| Reusable forward-core parameters | 1,692,190 |
| Local-coupling parameters | 1,710,984 |

The current prepared decoder was timed with CUDA events on case `0273`, using
five warmups and 30 synchronized repetitions at `Q=8192`, 12 padded module
slots (three active), and six fixed edges. This is the dense reference path;
peak memory is process allocation observed after resetting CUDA peak stats, not
an isolated tensor-only allocation.

| Dense prepared decode metric | Value |
|---|---:|
| Median latency | 22.932545 ms |
| p95 latency | 30.695423 ms |
| Median query throughput | 357,221.586 queries/s |
| Peak CUDA memory allocated | 451,502,592 bytes |
| Dense query-module routes | 98,304 |
| Dense query-edge routes | 49,152 |

The timing command loaded the checkpoint with
`channelthermal.workflows.evaluate_forward.load_model`, reconstructed the
checkpoint-owned normalizer and test sample, prepared the physical case once,
and timed `model.decode_prepared(prepared, query_xy)` with
`torch.cuda.Event`. No optimizer, backward pass, or training loop was run.

## Goal 0 compatibility statement

No model/configuration source was changed in Goal 0. The only tracked task
change is this reference document; the goal-plan completion note is a local
planning record. Generated evaluation artifacts are ignored. The pre-upgrade
checkpoint path and fixed-projection/context-fusion behavior therefore remain
unchanged at this boundary.
