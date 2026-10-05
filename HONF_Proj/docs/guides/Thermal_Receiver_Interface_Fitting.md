# Frozen Thermal receiver-interface fitting

`tools/thermal_interface_fit.py` prepares the receiver-interaction experiment from the selected/exact Run3601 G-fast epoch-1000 checkpoint. It does not start training. This entry is separate from the epoch-500 matched continuation entry; neither entry accepts the other's lineage.

```bash
conda run --no-capture-output -n ModularDT python tools/thermal_interface_fit.py \
  --parent-checkpoint /absolute/path/Run_3601_.../epoch_1000_model.pt \
  --output-root /data/.../receiver_interaction/continuation
```

The parent must use `fixed25_v1`, with 150 selected training cases and 22 development-validation cases, the saved selected-training normalization basis, `source_local_v3`, the native physical loss policy 2, and the specialized G-fast reader. Preparation checks every inherited model weight and buffer for exact equality after loading. It retains the parent's RNG state, normalization metadata and response calibration. The parent remains intact.

The two children use identical new parameter tensors. `H-add` executes source-only and receiver-only query corrections; `H-joint` additionally executes their centered source/receiver interaction. Only `core.backend.tensor_query_interaction.*` has `requires_grad=True`. Every inherited physical component, including the input/output adapters and Stage-A local coupling, is frozen. Forward operations remain differentiable with respect to physical inputs and query coordinates. New AdamW moments start empty, with learning rate `3e-4`, weight decay `1e-5` and no scheduler; inherited optimizer moments are deliberately excluded.

Preparation writes ordinary native run directories, an attachment receipt and `prepared_interface_fits.json` below the ignored output root. That map contains the exact per-arm config, checkpoint and launch command. Launch those commands through the ModularDT environment with physical GPUs 1 and 2 assigned respectively. The initial profiles stop at fit epoch 100. Extension to fit epoch 500 or 1000 requires the experiment's measured review and remaining budget.

Checkpoint `epoch` and `selection_state.epoch` are interface-fit age, starting at zero. The inherited organizer stays at physical epoch 1000; native objective scheduling and the existing train-only response exposure use physical-objective epoch `1000 + fit_epoch`. The dataset/query sampler uses the same fit age in both arms. This distinction is recorded in CSV and the native attachment state. A fit epoch still visits all 150 selected cases, with query count 1024, effective batch 48 and microbatch 8.

The maintained checkpoint writer saves declared fit-age milestones every 100 epochs, latest state and the best-field alias selected over those same saved milestones. A resume retains the declared source, model configuration, dataset membership, normalizers, fit horizon and new-parameter inventory. The optimizer contains only the new interface parameters. A normal epoch-boundary stop can be requested by creating `stop_requested` in that child run directory; inspect its acknowledgment and saved resumable state before resuming.

Measured validation uses all 22 fixed development cases and the original native physical roles. Detailed fields and interaction exports use only the fixed four representative cases. The new control paths carry their actual source-only, receiver-only and joint terms; physical values continue through the dense native fine-source paths. An interface support plot therefore does not establish sparse physical execution or physical causality. All generated checkpoints, figures, numerical exports and one-time diagnostics stay under ignored output paths.
