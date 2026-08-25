# Stage 7 repository cleanup closeout

## Revisions

- cleanup start: `3b37e341d906c440cd8bfd869d7e72483830bf32`;
- cleanup implementation end: `62c64c229e596977131eecf77984891988ac7e10`;
- branch: `agent/honf-adaptive-correctness`.

The closeout is committed separately so it can name the immutable cleanup
implementation SHA. That documentation-only commit does not alter source or
numerical evidence.

## Major source map

| Primary facade | Before | After | Extracted implementation |
|---|---:|---:|---|
| `honf_forward_core/organizer.py` | 1,604 lines | 303 lines | `organization/helpers.py`, `organization/exchangeable.py` |
| `honf_forward_core/decoder.py` | 1,367 lines | 644 lines | `decoding/pairwise.py`, non-registering `decoding/research.py` mixin |
| `channelthermal/model.py` | 774 lines | 552 lines | non-registering `model_support.py` mixin |
| `workflows/train_forward.py` | 1,945 lines | 778 lines | `training/{epoch,optimizer,checkpoints,reporting}.py` |
| `workflows/evaluate_forward.py` | 1,054 lines | 452 lines | `evaluation/{loading,prepared,results}.py` |

Fixed organizer and decoder parameters remain registered directly on their
historical facades. Exchangeable parameters remain under
`core.organizer.exchangeable.*`. Maintained offline evaluators moved to
`tools/diagnostics`; compatibility wrappers remain at their former paths.

## Freeze and golden references

- checkpoint/hash manifest: `docs/experiments/stage1_7_freeze_manifest.json`;
- public config/checkpoint/plan/topology schemas:
  `tests/fixtures/forward_cleanup/public_schemas.json`;
- compact numerical replay and state/optimizer inventories:
  `tests/fixtures/forward_cleanup/golden_replay.json`;
- replay command: `python tools/diagnostics/replay_forward_golden.py`;
- profile registry: `src/config_core/forward/profile_registry.json`.

The manifest inventories Runs 1000, 1005, 1007, 1102, 1202, 1301, 1302,
1304, and 1401 without storing checkpoint bytes in Git. Run 1401 best-by-field
epoch 4585 is marked current; Run 1000 best-by-field epoch 9655 is marked the
historical compatibility reference.

## Numerical and lifecycle gates

- Run 1000 case 0653: exact replay, strict `237/237` keys, one optimizer group.
- Run 1401 epoch 4585 case 0653: exact replay, strict `237/237` keys, one
  optimizer group.
- All nine maintained best-field checkpoint families strict-loaded with zero
  missing or unexpected keys; observed key counts were 237, 251, 261, 261,
  261, 261, 251, 245, and 237 respectively.
- The ordered model-parameter digest for Runs 1000 and 1401 is unchanged from
  the start SHA:
  `e0464cf77c92bc383c52157b6227f114f08e4cb8234435d49adb1f0766549261`.
- Both accepted optimizer states loaded after exact group-count, parameter
  order, and group-inventory validation.
- Stage-1 additive closure, Stage-2 exchangeability, Stage-3 selection and
  gathered parity, ThermalChannel teacher/predicted/mixed coupling, prepared
  decoding, artifact lifecycle, historical configuration, topology, and
  inverse-facing plan contracts passed in focused and full suites.
- All 19 registry profiles/overlays resolve against their declared base;
  frozen JSON files were not edited to add registry metadata.
- Bounded launch lifecycle: Stage-7 CPU dry-run with diagnostic Run ID 9901,
  one epoch/two train batches/one validation batch requested; validation passed
  and no run was started or written.

Full root plus ThermalChannel result: `267 passed, 1 skipped in 15.32s`. The
single skip is the pre-existing local inverse integration test whose external
artifacts are unavailable; all available inverse contract tests passed.

## Performance

The initial absolute remeasurement was slower because three unrelated GPU jobs
had started after the accepted Stage-7 benchmark. The cleanup was therefore
compared sequentially against a detached start-SHA worktree under the same
current load:

| Checkpoint | Prepared decode change | Full-forward change | Peak allocation |
|---|---:|---:|---:|
| Run 1000 best | -14.31% | -1.40% | unchanged, 362.41 MiB |
| Run 1401 best | -11.77% | +3.20% | unchanged, 362.41 MiB |

Both full-forward changes are inside the 5% guardrail. Exact inputs, historical
absolute measurements, paired measurements, and the contention caveat are in
`docs/experiments/stage7_cleanup_performance.json`.

## Deferred cleanup and archival candidates

No historical modes, checkpoints, runs, or evidence were deleted or archived.
Candidates for a separately approved later action are generated per-case
CSV/NPZ/figure trees, exact duplicate reports or plot aliases, superseded stage
drafts, and diagnostic compatibility wrappers after a migration window.
Lower-priority structural work on `compare_models.py`, data loading,
topology-signature internals, and aesthetic test/config moves is also deferred.

No scientific model behavior was intentionally changed. No loss, routing,
selection, normalization, data, Stage-A coupling, or inverse-facing contract
was changed. No long training was launched.
