# Query tiling

`query_tile_size` is an execution setting shared by `relation_anchored` and `relation_anchored_active`. It bounds the query dimension of each decoder chunk for every scene in the batch. Every query is still evaluated, and its organized access to original members is preserved.

## Effect

| Smaller tiles | Larger tiles |
| --- | --- |
| Usually lower peak live memory | Usually higher peak live memory |
| More chunks, small operations and kernel launches | Fewer chunks; often better throughput until a plateau |
| More checkpoint bookkeeping when activation checkpointing is enabled | Less per-chunk bookkeeping; checkpointed chunks are still recomputed during backward |

Tiling preserves the model equations and physical objective. It does not change the number or locations of sampled queries, scene batch size, retained directory mass or optimizer schedule. Floating-point reduction order can change, so numerical agreement is measured rather than assumed bitwise. Activation checkpointing is a separate execution choice: it trades recomputation for lower activation storage. Keep that choice fixed when testing tile size.

The best tile depends on your GPU, precision, scene batch size, query count, member counts and organized support. Results from another dataset do not establish a suitable value for yours.

## Set it before training

For a new shared core, pass the setting explicitly; `512` below is an example to test locally:

```python
from honf_forward_core.models import build_model, relation_anchored_config

config = relation_anchored_config(
    "relation_anchored", base_model_config,
    input_dataset="my_case", query_tile_size=512,
)
core = build_model("relation_anchored", config)
```

For the maintained Thermal recipes, edit `member_routing.model.query_tile_size` in your explicit recipe and launch with `--config`. The shared constructor currently defaults to 256; explicit recipes can choose another value. Save the resolved model configuration with the checkpoint.

## Test on your own inputs

1. Select a small input-only panel containing typical scenes and the largest expected member/query workloads. Keep the exact cases, query coordinates, masks, weights, precision, batch size, checkpointing and objective fixed across candidates. Use disposable model/optimizer copies for training-update replays.
2. Start with a tile that fits comfortably, then double it, for example 256, 512, 1024 and 2048. Stop increasing when memory becomes restrictive or throughput gains flatten. These are trial values, not a recommended setting for every dataset.
3. Warm up each candidate and repeat a complete training-update replay, including forward, loss and backward. Include the parameter update and data preparation consistently if they are part of the quantity you report. Synchronize CUDA immediately before and after the timed region. Forward-only timing misses checkpoint recomputation and backward memory.
4. Record mean time, variability, `torch.cuda.max_memory_allocated()` and `torch.cuda.max_memory_reserved()` after resetting peak statistics. Allocated memory describes live tensors; reserved memory also includes reusable allocator cache. Check changing and largest expected batches, and retain room for validation and other required work.
5. From the same initial weights and inputs, compare field outputs, weighted loss and parameter gradients against the smaller tile. Record absolute/relative differences and finite-gradient checks in the precision you use. Restore the same weights and optimizer state between candidates; disable stochastic layers or restore their random state for this comparison.
6. Choose a tile near the best measured throughput with adequate memory headroom. Confirm it over a few ordinary epochs, retaining continuous loss history and curves. Report your cases, batch size, precision, checkpointing, timing boundary and memory measurements with the selected value.

No convergence experiment is needed to measure execution cost. The short numerical comparison checks the tiling change; it does not establish final field accuracy on a new dataset.

## Existing checkpoints

Ordinary loading and resume restore the saved tile value. Editing a fresh-run recipe does not change an existing lineage's saved configuration. The current Thermal CLI has no query-tile override argument.

For the maintained detail-preserving training workflow, a runner can reconstruct the model/provider from a copy of the saved recipe with only the tile changed, then call `TrainingEngine.fit(..., resume_checkpoint=..., allow_query_tile_change=True)`. The opt-in checks consistent changes in the experiment recipe, provider recipe and adapter core, and requires all other identity fields to match. It restores the saved model, optimizer, RNG, sampler and history; subsequent checkpoints record the new tile. Use an epoch-boundary clean stop before changing the active run. The maintained identity layout is required; a custom dataset runner with a different checkpoint format needs its own explicit resume validation.
