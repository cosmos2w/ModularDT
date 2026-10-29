# Run artifacts and loss monitoring

Training producers use `honf_runtime.run_layout.RunLayout` for both HONF and baseline families. A new run writes each artifact into its category directly, while checkpoint selectors and readers accept the historical root layout. The training schedule, selected cases, normalization, native losses, and checkpoint selection policy remain dataset-owned contracts.

| Directory | Contents |
| --- | --- |
| `configs/` | Resolved configuration, source configuration, recipes, experiment identity and initialization inventories |
| `checkpoints/` | Rolling latest state, declared best selectors and retained epoch milestones |
| `metrics/` | Training history, routing metrics and run summaries |
| `plots/training/` | Loss monitoring PDF and a small PNG companion where supported |
| `plots/diagnostics/` | Physical fields, residuals, routing and support diagnostics |
| `evaluations/validation/` | Scheduled held-out validation receipts, identified by checkpoint epoch |
| `evaluations/` | Explicit post-training evaluations, including partition and checkpoint provenance |
| `comparisons/` | Matched comparison results |
| `logs/` | Process status, progress, stop acknowledgments, stage transitions and resource records |
| `environment/` | Software and source-state snapshots |

The root holds the run manifest or layout marker and operational controls such as `.training.lock`, `CLEAN_STOP_REQUEST.json` or `stop_requested`. Producers must not create root checkpoint copies or populate the category directories by copying results after training. `RunStore.finalize_artifacts` inventories existing artifacts rather than duplicating them. Familiar checkpoint filenames such as `latest_model.pt`, `best_by_field_mse_model.pt` and `epoch_0100_model.pt` remain available inside `checkpoints/`. For canonical Wind forward runs, `best_model.pt` is a compatibility link to the single best-field checkpoint because both names select the same metric; distinct Thermal selector states remain separate.

## Recording and comparison

Record TRAIN losses at each completed epoch with their actual case visits, optimizer updates, query/work counters, phase, sampling identity and elapsed time. At the configured monitoring epochs, evaluate the frozen held-out panel and save native accuracy metrics and the provider's measurable validation loss terms. Use the declared monitoring and retention cadence; this layout change does not authorize additional epochs, a larger validation population or a new training run.

TRAIN optimization losses and held-out inference losses must use explicit labels for reduction, coefficient and normalization. TRAIN histories preserve the arithmetic mean of macro-update means, including the final partial update; validation pools loss numerators and valid-element denominators over the held-out panel. Validation uses hard inference routing and evaluation mode, while adaptive TRAIN prediction losses can include fine-path replay. Plot sums of same-named terms as loss summaries with these differences visible, rather than claiming identical objectives or reductions. Train-only replay, response-anchor supervision, derivative terms or routing regularizers that are unavailable on the held-out panel must remain marked unavailable; do not fill missing terms with zeros or present a partial validation subtotal as the full training objective.

Name each loss in plain language, explain what it measures directly in the figure, and retain detailed scaling and coefficient definitions in the monitoring metadata sidecar. Use readable panel titles and legends, common epoch limits, and log axes only for positive quantities. Group the objective contributions separately from physical accuracy and work penalties. Physical metrics retain their native units; standardized MSE and work penalties are dimensionless. Preserve raw zero, negative or nonfinite records and explicitly annotate values masked from logarithmic plots. The graph should identify TRAIN and exposed VALIDATION, the execution phase, and the selected case population. Repeatedly monitored source test rows are exposed validation evidence, not an untouched independent TEST set.

Legacy histories may contain native validation accuracy without validation objective terms. Re-render these histories using only saved measurements, label the missing validation losses, and retain the original numerical history. A new plot cannot recover losses that were never recorded.

The common loss monitor presents the recorded weighted objective or the sum of terms shared by TRAIN and validation, the exposed validation field selector, and the available field prediction losses. Thermal joint histories separate flow and temperature terms; other model families show only the prediction, physical or regularization terms they actually recorded. Work counts and organizer health remain diagnostics rather than default loss panels. TRAIN values are sampled during fitting; validation is a fixed-model held-out measurement at its saved review epochs. Epoch is the horizontal unit, and panel metadata records the actual definitions, weighting, paths and reduction limits rather than treating unlike measurements as interchangeable.

The historical Thermal CSV producer maps its recorded `loss_total`, `val_loss_total`, `val_field_mse` and available component metrics into the same populated-panel organization. CSV columns and numerical histories remain unchanged, absent columns do not produce curves, and the companion metadata identifies the source and notes that its reductions and prediction paths follow the historical run configuration. In some legacy dense-pairwise histories, expanded rows with a mismatched CSV header have ambiguous validation fields and are omitted from the common validation curves; header-aligned rows are retained. The JSON renderer accepts historical full histories and the compact history reader restores the full row view from `history_metadata.json`; direct consumers of compact rows must use `read_training_history`.

## Compact history and necessary records

New maintained contextual/member-routing runs use compact JSON output by default. A resume preserves the existing history format unless `--compact-history` or `--no-compact-history` is explicit. These options affect serialization and figure presentation only; they are excluded from scientific identity, optimizer state, sampling, loss definitions, validation cadence and checkpoint selection. Changes to an existing run's output format use an explicit output-only audit receipt. The shared engine keeps its original expanded-history default for historical callers.

| Recording level | Retained content and reason |
| --- | --- |
| Every completed epoch | Actual weighted TRAIN terms, epoch/case visits, learning rate, elapsed time, sampling identity, changing execution counts and measured resource values. These establish trajectory and reproducible exposure. |
| Existing scheduled reviews | Every recorded validation value, including case-level values, numerators/denominators, native objectives and selectors; actual organizer health and gradient observations at their existing cadence. Nothing is replaced by a smoothed curve or discarded to save space. |
| Once per run, with explicit row overrides | Fixed arm/phase and reader-context defaults, names of always-zero or inapplicable work counters, objective definitions, normalization/reduction and prediction-path descriptions. Existing run configs remain the source for data, normalization and optimizer declarations. |
| Removed duplication | A validation objective identical to the copy already in that row, identical paired-reader context, repeated static descriptions, repeated zero counters and verbose whitespace. A later phase/context/count change is stored as a row override. |

`metrics/history.json` remains a JSON list with one compact epoch object per source line. `metrics/history_metadata.json` supplies shared definitions; keep both files together. The maintained `read_training_history(path)` restores the exact historical full-row view, including absent keys and all numerical values. Shared defaults stay stable and objective definitions append, so publishing the metadata before atomically replacing history does not invalidate the prior history. Checkpoints continue carrying their complete expanded history and training state. A missing or unsupported metadata sidecar is not permission to guess measurements.

On a saved relation-anchored history through epoch 2697, the exact round trip reduced approximately 73.3 MB to 44.2 MB including metadata (about 39.7%), without removing validation, organizer-health or changing work values. Per-case validation and real health evidence still account for substantial size; removing them would change the scientific record rather than merely its presentation. This is file-size evidence, not an end-to-end training-speed claim. Existing legacy CSV histories are not rewritten by this format change.

## Existing runs

Unmarked legacy runs continue writing at their existing paths on resume. New runs and runs explicitly marked with `artifact_layout_version: 1` use the canonical tree. Readers and named checkpoint selectors prefer the active producer location, then fall back to the other layout; this prevents stale category copies from overriding a resumed legacy run's current root records. An exact shared-engine resume still requires the current latest checkpoint, its sealed identity and an advancing stop epoch; relocating artifacts does not relax those checks.

`tools/standardize_run_artifacts.py RUN_DIR` inspects one run and prints a checksum-backed relocation plan without changing its artifacts. `--apply` takes the shared training lock, rejects live owners or processes using the run, refuses conflicting destinations, relocates recognized root files, verifies every checksum and saves `logs/artifact_relocation.json`. It retains checkpoint compatibility symlinks for historical explicit file references and leaves unrecognized artifacts untouched. When historical `last.pt` and `latest_model.pt` are byte-identical, it shares their inode after checksum verification while retaining both references. It does not prune scientific checkpoint states or compute retrospective metrics. Review the receipt and the unclassified file list before applying it to another historical run.

All generated results, preview plots and relocation receipts remain in ignored local result paths. Commit and push only maintained code, tests and documentation after auditing the entire outgoing history.
