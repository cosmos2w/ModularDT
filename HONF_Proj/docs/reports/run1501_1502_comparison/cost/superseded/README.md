# Superseded inference pass

`inference_cost_cuda2_inner_chunk_overridden.json` and its CSV preserve the
first four-checkpoint pass for audit. That pass incorrectly used the evaluator
outer `query_batch_size=32768` as a runtime override for each model's inner
`receiver_chunk_size`. Runs 1804, 1501, and 1502 are configured with an inner
chunk of 128; the legacy 1404 path uses its own default. The override changed
the measured execution and inflated memory for the sparse-incidence models.

Do not use these measurements in the comparison. The corrected run preserves
each checkpoint's native inner receiver chunk while keeping the same outer
query batch, cases, GPU, checkpoint, and timing repetitions.
