# Query tiling

`query_tile_size` is an execution setting shared by `relation_anchored` and `relation_anchored_active`. It bounds the query dimension of each decoder chunk for every scene in the batch. Every query is still evaluated, and its organized access to original members is preserved.

## Observation budget, decoded positions and tile size

A scene can request 1024 native fluid observations in one training epoch and still require more than 1024 joint-field predictions. Interpolation, curl stencils and member material/surface/outside observations can require additional physical positions. The adapter constructs their complete query union and retains each observation's extraction from it. Increasing the tile size does not increase that observation budget or remove any of those dependencies.

| Quantity | Meaning | What to record |
| --- | --- | --- |
| Native observations per case per epoch | The saved row draws and other prescribed native observations presented on that case's visit | Case IDs, row IDs, component/extraction definitions and the per-case budgets; 600 visits with 1024 fluid rows give 614400 fluid-row presentations |
| Unique decoded physical positions | The union needed by all extraction dependencies within each case | Each case's valid union count, including interpolation and derivative support; preserve the original observation pullbacks when positions coalesce |
| Dense decoder slots and padding | A dense batch pads each case to its largest query-union length | Total slots are batch size times padded query length; padding-only rows equal that total minus the valid union counts and can still reach the joint head |
| Query-tile size | The maximum positions per case handled together by a decoder call | Tile size, actual number of calls and the final partial tile; this is independent of the native row budget |

For example, a hypothetical 24-case microbatch whose largest union contains 3100 positions has 74400 dense decoder slots and seven decoder calls at tile 512. Padding-only rows require the other 23 valid union counts to calculate. All prescribed observations still use their complete extractions. A sampled reader executes one access policy; the usual anchored full/restricted pair executes two heads while sharing some work. The `all_native` capability aliases one output instead. Report actual score/value/head counts rather than deriving them from the tile size or number of nonzero access weights. Returned work counters describe the initial forward and exclude backward checkpoint recomputation.

The number of joint output components is another independent model dimension. Tiling and activation retention preserve the declared N-field schema, original M/E members, access weights, masks, physical measures and extraction definitions; they do not select output physics or pool away members.

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

For a new recipe used by `tools/relation_anchored_train.py`, set `member_routing.model.query_tile_size` explicitly and pass that file with `--recipe`. The shared constructor currently defaults to 256; explicit recipes can choose another value. Set the value before initializing a new scientific lineage and save its resolved model configuration with the checkpoint.

## Test on your own inputs

1. Declare the complete training membership, replay epoch, source checkpoint, epoch/device-hour and storage limits. Use disposable model/optimizer/provider copies. Keep exact cases, native draws, extraction dependencies, reader choices, FP32 or other declared precision, microbatch/effective batch, objectives and schedule fixed. Use the same approximately 10% evaluation panels across settings; do not infer permission for full-population inference from a complete training replay.
2. Repeat the unchanged execution policy first. Restore model, optimizer, RNG and provider/sampler state before every replay. Establish its actual repeat variability in outputs, gradients and optimizer applications before attributing a candidate difference to tiling or retention.
3. Compare a small declared set of tile sizes with checkpointing fixed. Warm each setting and interleave repeated complete epochs on the same device. Time ordinary data assembly, organization/index construction, decoding, loss, backward and optimization, with CUDA synchronized around that entire region. Keep the provider-cache and CUDA allocator starting conditions comparable, and state whether unused allocator blocks were released; a controlled cold-allocator replay does not establish sustained warm-allocator latency. Record state restoration, startup/cache preparation and instrumented parity probes separately; include all GPU occupancy, warmups and failures in the process cost.
4. Compare outputs on the matched panel, ordered weighted losses, pre/post-clipping gradients and optimizer displacements/moments. Keep detailed tensor-copy probes outside timed repetitions. Report maximum absolute and relative-norm differences alongside the self-repeat control and precision tolerances; a near-zero tensor entry makes a pointwise relative maximum alone misleading. Separate unchanged-input single-application differences from later training-trajectory differences, and select using the declared acceleration and fidelity priorities. A componentwise diagnostic exceedance alone does not establish a physical-quality loss.
5. Reset and record allocated and reserved peaks for complete training epochs and actual scheduled validation separately. Validation can request more positions or use a different inference tile and both reader policies. An ordinary forward-only probe or a small training query panel does not establish validation headroom. Reserved bytes include allocator cache and are not live tensor bytes; neither measurement includes every non-PyTorch allocation on the device.
6. Only after the tile comparison, test a small declared set of activation-retention fractions at a promising tile. Recheck self-repeat/numerical fidelity and counterbalance the timings against that tile's checkpointed reference. Count any interrupted or out-of-memory replay as consumed work. Do not reduce observations, native dependencies, microbatch size or precision to make a candidate fit.
7. Select the lowest measured complete-epoch latency with acceptable numerical behavior and measured validation headroom. If the difference is within repeat variability, retain the simpler or lower-memory setting and call the timing inconclusive. No target occupancy or arbitrary speed percentage defines success. Save complete concise histories and first/ten-epoch/final curves, labeling independently restored replay epochs rather than presenting them as a learning trajectory.

No convergence experiment is needed to measure execution cost. The short numerical comparison checks the tiling change; it does not establish final field accuracy on a new dataset.

## Measured R17-11 example

The TRAIN600 study restored the same epoch-1000 model, optimizer, RNG and provider state before every epoch-1001 replay on one RTX 6000 Ada. It kept FP32, microbatch 24, effective batch 48, 1024 native fluid rows per case, the complete extraction dependencies and sampled-reader draws fixed. Physical-output checks used TRAIN60; actual validation used the saved VAL9 of 89 cases with both readers. Timings cover complete epochs with matched cold allocator starts and warmed provider metadata, excluding instrumented probes, restoration and validation. Those other phases remain included in the process cost.

| Training tile, checkpointed | Median epoch [range], seconds; three repeats | Peak allocated / reserved, GiB | Numerical comparison with tile 128 |
| --- | --- | --- | --- |
| 128 | 29.500 [29.389, 29.756] | 3.715 / 5.229 | Bitwise self-repeat |
| 256 | 19.965 [19.962, 20.032] | 4.970 / 5.879 | Later gradients and optimizer applications exceed recorded tolerances |
| 512 | 16.362 [16.359, 16.376] | 7.504 / 28.229 | Later gradients and optimizer applications exceed recorded tolerances |

All tiles presented 614400 fluid rows and needed 1959978 unique physical positions plus 190782 padding rows, giving 2150760 dense head slots. Batched decoder calls were 714, 362 and 185. Thus a native-row budget, a decoded-position count and a tile size describe different quantities. Score/value pair counts differed by about 0.001% across the resulting trajectories, so identical query/head counts do not imply identical learned-access work.

**Select tile 512 with checkpointing for the next explicitly configured run on this workload.** It reduced median complete-epoch latency by **44.5%** versus tile 128 (29.500 to 16.362 seconds), with actual validation completing and about 18.572 GiB estimated free headroom at its reserved peak. Acceleration is the selection priority. The first optimizer application agreed within tolerance, while later numerical trajectories diverged; that diagnostic is reported rather than used as a requirement for bitwise substitution. The precise arithmetic/support mechanism was not isolated. These are FP32 executions of the same equations, observation budgets, member-access policy and objective; the replay does not establish mature field-quality equivalence.

At tile 128, two timed repeats per retention setting gave:

| Native activation retention | Median epoch [range], seconds | Peak allocated / reserved, GiB | Estimated free memory at validation reserved peak, GiB |
| --- | --- | --- | --- |
| 0 | 29.921 [29.767, 30.074] | 3.715 / 5.230 | 41.268 |
| 0.125 | 28.920 [28.816, 29.025] | 6.042 / 6.824 | 39.977 |
| 0.25 | 28.000 [27.988, 28.011] | 8.531 / 12.852 | 33.947 |

The secondary result at **tile 128 with native retention 0.25** reduced median epoch latency by 6.4% against its within-stage checkpointed control, with bitwise agreement on the checked physical outputs, every pre/post-clipping gradient, all optimizer displacements, model/Adam states and ordered losses. It retained 182 of 714 batched native tiles after per-call rounding. The unchanged control also matched the earlier tile-128 replay's ordered losses. Actual validation completed; its live allocated peak was 0.586 GiB, while training left 12.852 GiB reserved and about 33.947 GiB estimated free headroom. This option uses less reserved memory but takes longer than checkpointed tile 512. Retention at tile 512 was not measured; do not combine the two gains or recommend that untested combination as a measured winner.

The study completed 22 replay epochs without failures and consumed 0.24886 aggregate device-hours including setup, probes, validation, plots and shutdown. An additional seven-output CPU fixture preserved both reader modes, all output components and parameter/coordinate gradients. The active 3000-epoch lineage retained its saved execution policy. These measurements support this execution choice; they do not change global defaults or establish new learned field quality or sustained warm-allocator speed.

## Retain selected native decoder activations

For a new field-response or observation-response lineage using an anchored core, `execution_config.native_retention_fraction` selects the fraction of native decoder tiles whose activations remain available for backward. The default is zero, which retains the existing checkpointed execution. `tools/relation_anchored_train.py fit --recipe <recipe.json> --native-retention-fraction 0.25 ...` supplies the same runtime setting and saves it in the resolved configuration. Reload that saved configuration when resuming. Device selection and allocator limits remain runtime arguments.

Tiles are selected deterministically at evenly distributed indices, counting the final partial tile. For `n` tiles and requested fraction `f`, the retained count is `floor(n*f + 0.5)`, bounded by `n`. Record this actual count: a small fraction may round to zero, and a retained tile covers its whole batch slice. The scope covers ordinary native predictions; the separate common-position auxiliary decoder remains checkpointed. It changes storage and backward recomputation, while retaining every physical prediction and loss term. Retention does not reduce no-gradient validation storage. Check numerical agreement on physical support and complete-epoch time on the same device before selection; reduced padding or one faster dense batch does not establish a complete-epoch gain.

The observation-response builder records nonzero applied retention in its provider identity and rejects unsupported cores. This makes the actual execution policy verifiable on reload. A historical observation-response configuration that contained a nonzero but previously ignored key has a different applied identity and cannot silently resume as if retention had already been enabled. Omitted and zero settings preserve the previous behavior and identity. A disposable probe can explicitly scope native predictions with `configure_native_retention(provider, fraction)`; that does not authorize changing an existing lineage's saved execution policy.

## Existing checkpoints

Ordinary loading and resume restore the saved tile value. Editing a fresh-run recipe does not change an existing lineage's saved configuration. The Thermal training CLI has no query-tile override argument. The separate `thermal_native_checkpoint_evaluate.py` inference tool accepts `--query-tile-size` for its evaluation copy and records the applied value; it does not change the checkpoint or training configuration.

For the maintained detail-preserving training workflow, a runner can reconstruct the model/provider from a copy of the saved recipe with only the tile changed, then call `TrainingEngine.fit(..., resume_checkpoint=..., allow_query_tile_change=True)`. The opt-in checks consistent changes in the experiment recipe, provider recipe and adapter core, and requires all other identity fields to match. It restores the saved model, optimizer, RNG, sampler and history; subsequent checkpoints record the new tile. Use an epoch-boundary clean stop before changing the active run. The maintained identity layout is required; a custom dataset runner with a different checkpoint format needs its own explicit resume validation.

That opt-in is specific to its supported identity layout; it is not a general override for observation-response lineages. A selected replay setting is a recommendation for an explicitly configured future run. Preserve active jobs, their original configuration and checkpoint aliases unless a separate authorized continuation path has been verified.
