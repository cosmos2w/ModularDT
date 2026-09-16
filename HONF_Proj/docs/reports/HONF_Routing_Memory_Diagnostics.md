# Run 2000 / Run 2100 memory investigation

Investigation date: 2026-09-16. Scope: the existing module-hub and three-step
mean-shift runs, their shared sparse executor, and training/validation tensor
lifetimes. Disposable diagnostics use physical GPU 0. The existing runs on
physical GPUs 2 and 1 retain their settings, weights, optimizer, and history.

**Conclusion:** bounded real training/validation replays reproduce increasing
reserved memory without a persistent tensor leak. Duplicate scalar path
expansion and variable allocation sizes are the main measured memory costs.
The earlier silent exits cannot be attributed to OOM from the available evidence.
Both existing runs continue toward their configured epoch-500 endpoint.

## Findings from the existing runs

An increasing `nvidia-smi` reading does not establish a tensor leak. It includes
the CUDA allocator's reserved memory and other device allocations. The existing
`metrics.csv` records **peak allocated tensor memory**, reset each epoch before
training and sampled after validation; it does not record reserved memory or the
live allocation floor after an epoch.

At the initial inspection, physical GPU 1 used about 47,901 MiB and GPU 2 about
43,775 MiB. Both resumed processes were advancing. Their peak tensor allocations
were substantially lower and did not show cumulative growth:

| Run | Epoch window | Min / mean / max peak allocated MiB |
|---|---|---:|
| 2000 | 1–20 | 33,456 / 34,258 / 35,050 |
| 2000 | 81–100 | 30,203 / 30,788 / 31,310 |
| 2000 | 174–193 | 29,915 / 30,291 / 30,631 |
| 2100 | 1–20 | 37,065 / 37,433 / 38,161 |
| 2100 | 81–100 | 32,945 / 33,758 / 34,874 |
| 2100 | 153–172 | 32,911 / 33,720 / 34,193 |

These observations constrain the diagnosis; they do not prove that no transient
OOM ever occurred. No causal attribution of the earlier silent stops is justified
without a contemporaneous exit status, traceback, or attributable kernel event.

Four read-only samples over 174 seconds (09:02:03–09:04:57 EDT) found exactly
flat device/process memory while the CSVs advanced: Run 2000 epochs 194→197,
Run 2100 epochs 174→176. Their leaf cgroups had no OOM kills or memory limits;
the host had about 473 GiB available. The targeted kernel journal contained no
OOM, NVRM, or Xid matches, and systemd-oomd had no entries. Direct `dmesg` access
was denied. An ancestor cgroup's cumulative 11 OOM kills has no victim/time
attribution here and cannot be assigned to these runs. The old stops therefore
remain unexplained; neither OOM nor terminal/session termination is established.

Run 2100 is currently under `honf-run2100-resume.service`, with no service
restarts. Run 2000 is an orphaned process in its original session scope. The
different launch lifetimes matter for reliable execution but do not explain a
past stop without an exit record.

At the final check (13:17:45 UTC), Run 2000 had completed epoch **208** and
Run 2100 epoch **187**; both original resumed PIDs were alive with `--epochs 500`.
GPU 1 still had only 619 MiB unreserved device memory, although much of its
process allocation can be allocator cache. No process was restarted or altered
by this investigation. See `final_live_status.json` in the artifact directory.

## Mechanisms that deserve attention

1. **Duplicate path expansion is expensive even when fine messages are unique.**
   `interface_fields/routing_index/pair_join.py` expands positive query/hub/source
   paths, sorts them, and coalesces their priors. Fine QM/QE execution is delayed
   until after coalescing, as required, but scalar raw-path indices and values
   still scale with the number of paths. Several indices remain saved for
   backward. Dense positive support can therefore be costly without any leak.
2. **Checkpointing fine pair tiles does not checkpoint the scalar compiler.**
   Fine gathered features are checkpointed inside `_tile` in
   `interface_fields/routed_pairwise.py`. Each receiver chunk's compiler graph
   can nevertheless remain live until the physical loss backward pass. Smaller
   fine tiles alone cannot eliminate that cost.
3. **Dynamic shapes encourage allocator cache growth.** Different module counts,
   sparse supports, receiver counts, and train/validation phases request different
   allocation sizes. Cached blocks are reusable memory, but fragmentation and
   limited headroom remain real risks on a nearly full GPU.
4. **There are bounded lifetime inefficiencies.** Context norm/fraction diagnostics
   in `interface_fields/core.py` retain autograd branches despite being used only
   for reporting. The training loop retains the previous output while evaluating
   the next forward, and clears gradients after that forward. P0/P1 prepared
   objects and optional raw source logits also outlive their last direct use.
   None of these observations alone shows accumulation across epochs; physical
   refinement still requires its loss-connected tensors for backward.
5. **Mean-shift adds bounded live work.** Its three updates retain differentiable
   source/candidate calculations. The source samples stay fixed within the
   updates and remain in autograd, as specified. Detaching them would change the
   algorithm and is not an acceptable memory fix.

Detailed source references are in
[`code_lifetime_audit.md`](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/comparison/memory_diagnostics/code_lifetime_audit.md).

## Isolated scalar compiler measurement

Fixed B=48, receiver chunk Q=128, E=192, fully positive scalar memberships;
this is a compiler diagnostic, not additional model training or a bandwidth
sweep. All cases represent 1,179,648 unique pairs.

| Positive hubs | Raw paths | Live after forward MiB | Peak allocated MiB |
|---:|---:|---:|---:|
| 1 | 1,179,648 | 99.8 | 171.8 |
| 6 | 7,077,888 | 420.4 | 833.4 |
| 12 | 14,155,776 | 801.3 | 1,647.3 |

Gradients were finite. After backward, deleting the tensors, and garbage
collection, allocated memory returned to **zero** in every case. Reserved
memory remained cached until the diagnostic's final `empty_cache()` call.
This demonstrates expensive temporary work and cache retention, not a persistent
leak in the isolated compiler. It is not a substitute for the real-model replay.

Artifacts live under
[`comparison/memory_diagnostics`](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/comparison/memory_diagnostics).
`pair_join_memory.json` contains the original probe; the maintained reproduction
script is [`diagnose_pair_join_memory.py`](../../tools/diagnostics/diagnose_pair_join_memory.py).

## Real-model replay

Both epoch-100 checkpoints completed a disposable replay on physical GPU 0 with
their restored optimizer and canonical physical `run_epoch` path: B=48, Q=1024,
two repetitions of an identical small-module training batch, four alternating
small/large training batches, and three validation batches. Training spans M=1
and M=12; validation spans M=3 and M=10. Short case pools repeat indices to fill
B=48, so these are shape/lifetime diagnostics, not accuracy estimates. Update
diagnostics were enabled. Model weights are restored between the fixed and
alternating patterns; no trained weights are saved.

| Replay | Largest peak allocated MiB | Reserved after large batches MiB | Live floor after each phase MiB | Reserved after final cache release MiB |
|---|---:|---:|---:|---:|
| Run 2000, native allocator | 31,211.4 | 34,456 | 70.99 | 100 |
| Run 2000, expandable segments | 31,047.6 | 31,264 | 70.99 | 96 |
| Run 2100, native allocator | 34,307.1 | 37,666 | 70.99 | 100 |
| Run 2100, expandable segments | 34,204.3 | 34,506 | 70.99 | 96 |

Both models returned to **the same 70.99 MiB live floor after every phase**.
Garbage collection did not lower it further. All tracked output weak references
were dead after `run_epoch` returned, even before GC. There were zero allocator
retries and zero OOMs. Training outputs' reporting norm tensors had autograd
history; validation norms did not. Thus the measured growth is reproducible as
allocator reservation with changing workloads, without persistent tensor growth
in these bounded replays. This is strong evidence against the proposed common
per-step leak, not a proof about every possible long-run execution.

The largest inactive split allocation was about 4,942 MiB for Run 2000 and
5,072 MiB for Run 2100 during alternating training. Cached memory is reusable,
but variable-size allocation and limited device headroom remain design concerns.
The final diagnostic `empty_cache()` releases unused cache while leaving the
live floor unchanged; it is not part of training or a recommended per-step fix.

A separate process repeated the same Run 2100 checkpoint, batch plans, and
optimizer settings with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
Reserved memory fell by **3,160 MiB (8.4%)**. The largest recorded scalar loss or
MSE difference was `2.98e-8`; all phases completed without retries or OOMs and
the live floor was unchanged. The allocator reported zero inactive split bytes
for expandable segments, whose accounting differs from native split blocks;
this does not establish zero fragmentation in every sense. This bounded result
supports considering expandable segments at a future authorized restart. It
does not establish a long-run benefit or warrant interrupting the active runs.
Run 2000 repeated the same allocator comparison: reserved memory fell by
3,192 MiB (9.3%), the largest reported loss/MSE difference was `1.05e-7`, and
all 326 parameter tensors remained finite after each phase. Both comparisons
used PyTorch 2.6.0+cu124, CUDA 12.4; expandable segments is an option of the
native allocator, not a different model or optimizer.

Artifacts: `run2000_native.json`, `run2100_native.json`,
`run2000_expandable.json`, `run2100_expandable.json`; the earlier
`run2000_quartile_preliminary.json` used less extreme shapes and is retained as
preliminary evidence. The reusable script is
[`diagnose_routing_memory.py`](../../tools/diagnostics/diagnose_routing_memory.py).
Actual commands are in
[`executed_commands.md`](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/comparison/memory_diagnostics/executed_commands.md).

## Recommendations and boundaries

The highest-value executor improvement to investigate is reducing **saved scalar
path state**, not shrinking the already-checkpointed feature tiles. One exact
alternative is to construct the positive pair union using integer support, then
evaluate each retained pair's live prior
`Pi[q,i] = omega[i] * sum_k A[i,k] * d[q,k]` in checkpointed pair tiles. That
preserves gradients through A, d and omega, source/query sparsemax, and delayed
unique-pair QM/QE execution, while avoiding saved differentiable state for every
duplicate path. Its forward union construction still needs a bounded compiler;
this is a design recommendation, not an implemented or benchmarked improvement.
It requires support, prior, prediction, and gradient equivalence tests before use.

Smaller opportunities are to detach reporting-only context norms at their input,
clear completed batch outputs before the next forward, and clear gradients before
the next forward. Do not detach any loss-connected routing/source tensors, add
per-batch `empty_cache()` as a supposed leak fix, or change the formal routing
strategy/bandwidth to hide memory pressure. Existing run artifacts and checkpoints
must remain comparable.
