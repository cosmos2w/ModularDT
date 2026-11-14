# Run artifacts and loss monitoring

A run must keep the information needed to restore training, draw its curves and assess its declared scientific question. Write configuration once, numerical histories as CSV, small process records as JSON, and physical arrays in their existing shared input files. Case adapters supply metric values and definitions; they should not implement another file-management framework.

## Implementation status

The shared trainer supports an opt-in `csv_v3` policy for providers that declare their complete scheduled scalar columns. It appends training and validation tables, keeps the numerical history outside checkpoints, and reads it for exact resume and curves. The Thermal native-pair provider opts in; the Wind refinement provider accepts `training_output_policy: csv_v3` and declares its native-role, base, router and work terms. Generic providers still use `categorized_v2`; historical producers, saved recipes and resume paths keep their existing behavior.

The run status and Thermal research assessment commands read the format declared by the run marker. The development evaluator can reuse a current-epoch validation receipt only when the run, arm, ordered panel, routing path and numeric receipt match the checkpoint-bound CSV; otherwise it recomputes. CSV child/fork conversion is unsupported. These implementation checks do not establish scientific performance or authorize training.

## File ownership and formats

| Location | What belongs here | Writing rule |
| --- | --- | --- |
| `configs/resolved_config.json` | Actual model, optimizer, objective and schedule, plus data/normalizer/membership references | Once at creation; record an authorized change explicitly |
| `configs/metric_definitions.json` | Loss terms, coefficients, units, reductions, prediction paths and sampling descriptions not already expressed by the resolved configuration | Once; reference rather than copy definitions into every row or plot |
| `configs/` | Small additional input definitions or references genuinely needed by the case | No second expanded configuration disguised as experiment identity |
| `checkpoints/` | Actual resumable latest state, one best-field selector and declared milestones/final state | Existing declared cadence; atomic replacement; no duplicate selector payloads |
| `metrics/history.csv` | One scalar training row per completed epoch | Append, with one stable header |
| `metrics/validation.csv` | All scalar results from actual scheduled validation | Append to this one table; missing measurements are not zero |
| `metrics/fit_summary.json` | Final/clean-stop reason, completed epoch, selected checkpoint references, concise metrics and total costs | Write at an actual stop or completion |
| `logs/console.log` | Terminal output and exception traces | During the process |
| `logs/active_process.json`, `logs/progress.json` | PID/host, run reference, times, current completed epoch and actual process state | Small records, replaced when state changes; no full identity |
| `logs/events.jsonl` | Launch, resume, authorized change, clean stop, failure and recovery events | Only when an event occurs |
| `plots/training/` | Initial/current/final loss-curve PDF masters and small PNG previews | First available history, every 10 epochs, validation, pause and completion |
| `plots/diagnostics/` | Actually requested physical-field, organization or other diagnostic figures | On demand |
| `evaluations/` | Detailed checkpoint evaluations and their provenance | Declared milestones/final review, on demand |
| `environment/` | Code revision, relevant local source changes and software/runtime information | Once, with changes recorded if necessary |

Create a category directory when its first artifact is written. Empty pre-created directories are not evidence of a measurement. Matched cross-model comparisons belong to the round's comparison location, rather than an empty `comparisons/` in every run. Keep the root's layout marker and existing operational controls such as `.training.lock`, `CLEAN_STOP_REQUEST.json` and stop acknowledgments where their readers require them. Write outputs into their category directly; do not copy a second tree at finalization.

Large fixed masks, coordinate arrays, region index ranges, reference fields and sampling tables are scientific inputs. Reference their existing dataset/profile and original IDs. If a new reusable input file is necessary, save one HDF5/NPZ with a declared owner and byte budget, not copies in process status, summaries and checkpoints. Preserve exact values and physical definitions. Do not replace numerical recovery state with hashes, or replace necessary provenance with a short but uninterpretable name.

## Numerical tables

Training uses a wide CSV table whose columns are determined by the actual objective at run creation. A typical beginning is:

```csv
epoch,train_loss_total,loss_u,loss_v,loss_temperature,learning_rate,train_cases,train_seconds
```

Declare columns for all scheduled phases and components up front, leaving inactive values blank; if a workflow genuinely cannot know its scalar inventory, use an explicitly described long scalar table rather than rewriting headers during training. Add the actually used weighted components and the few changing work/resource scalars needed for the declared question. Store raw components separately when they are needed to interpret weighting. Define each term once. Use a shared writer, ordinary CSV quoting and round-trip numerical precision, such as Python float representation; do not truncate values to make the file look tidy. Leave unmeasured values empty, retain real zeroes, and keep case/member IDs as strings so leading zeroes survive reading. No JSON objects or arrays belong inside numerical cells.

Validation can use a long table with `epoch,partition,case_id,member_id,metric,value,count`. Empty case/member IDs identify an aggregate where its definition says so; named metrics and their units/reductions are defined once. Preserve the per-case or per-member values actually required by the protocol, without making every possible diagnostic a default. Append all scheduled validation results to this table. Separate detailed validation exports are produced only at declared milestones and final/clean-stop reviews, not for every ordinary validation call.

Do not collect default case-order/query-sampling SHA summaries, repeated allocator descriptions, unchanged executor settings, redundant epoch offsets or reconstructible family-ID lists. Store fixed descriptions once. Keep useful peak memory and declared computation totals; enable detailed exposure, gradient, memory or sampling audits only for a concrete question. Disable routine telemetry collection in the engine and case provider, not merely in a serializer; preserve hashes used internally for deterministic sampling namespaces and seed mapping. Preserve explicitly required scientific, data, security and exact-resume checks, including any existing opt-in stream proof.

A shared recorder needs only to append training scalars, append validation scalars, read recorded history and write small status/events. Objective definitions and naming come from the provider/configuration. Avoid a generalized database, event bus, compression/expansion scheme or new identity-freezing service.

## Checkpoints and interruption recovery

A new checkpoint format retains model parameters, optimizer state, normalization needed by inference, RNG state, sampler/provider and relevant auxiliary state, completed epoch, schedule position, selectors and cumulative counters needed for resume. Include a compact configuration/data reference and necessary small configuration for loading the model. Do not embed the entire history, full region arrays or a second large experiment identity. Standalone inference must still work with its declared input resources; exact continuation also needs the run's original configuration and records.

Keep the existing checkpoint cadence. Flush complete CSV records before the scheduled atomic checkpoint replacement; the checkpoint's completed epoch defines the resume boundary. On an actual interruption, only the resuming writer under the existing exclusive run lock may reconcile torn trailing records and rows beyond the owned latest checkpoint, preserving useful uncommitted rows once as recovery evidence and recording the action. Status and plot readers stay read-only and can ignore an incomplete final line while a writer is appending. Do not silently invent missing committed measurements. Resume from saved optimizer/RNG/sampler state rather than a hash or an inferred epoch. Implement and test this path with the producer and readers together; do not introduce a new journal or checkpoint frequency solely for logging.

Make best-selector and milestone updates recoverable with that same boundary. For a new producer, commit the atomic latest state before publishing selectors that refer to it; retain the previous valid best until the replacement is committed. Store the small selector epoch/score in resume state so a writer interrupted between latest and selector publication can finish the reference from saved state. Test that interruption window without adding another checkpoint cadence or discarding the previous best.

Keep one latest state, one best-field selector, declared 100-epoch milestones and the requested final/clean-stop state. Preserve additional historical selectors when they have different scientific meanings. Avoid duplicate checkpoint copies merely to support alternate filenames; existing named aliases can reference one payload where the selection is identical.

## Loss curves and interpretation

Show the full recorded weighted TRAIN objective and the full measurable validation objective in separate panels from the first curve. Leave the validation panel present and marked pending until measured. Mark unavailable terms explicitly; a partial shared-term subtotal may be an additional named series, never a substitute for the complete objective. A method without a scalar objective must not be assigned an invented total. Keep a field-selection score separate from losses.

TRAIN optimization losses and held-out inference losses require explicit reductions, coefficients and prediction-path labels. The current shared trainer averages macro-update means, including the final partial update; validation pools numerators and valid-element denominators across its panel. Hard inference routing and evaluation mode can differ from adaptive TRAIN paths and replay. A case must preserve these distinctions rather than forcing unlike values onto one purported objective.

Use an equal-sized subplot grid, leaving an unused cell when needed; do not stretch the last panel. Distinguish sampled fitting losses, fixed-model validation, physical errors and gradient/organization diagnostics. Preserve meaningful zero, negative and nonfinite records; explain values excluded from logarithmic axes. Use readable names, physical units and epoch axes. Refer to one shared metric-definition file rather than duplicating the calibration profile beside every curve.

Save the first curves as soon as history exists, refresh every 10 completed epochs and additionally at validation, pause and completion. If a producer accepts the legacy `runtime.plot_every_steps` key, interpret it as an epoch interval with default 10; do not imply this key is already implemented by every workflow. Apply the cadence to separate case workflows as well as the shared trainer; automatic plotting does not authorize extra evaluations or conversational polling. Enable tqdm by default while retaining an explicit opt-out. Follow the project instructions for sparse read-only Luna monitoring after actual launches.

## Historical formats and consumers

Do not migrate, rewrite, prune or change the resume behavior of active/historical runs to apply this contract. Existing JSON, compact JSON plus `history_metadata.json`, CSV and legacy figure paths remain readable. The common reader must distinguish their real formats; new CSV rows do not need the old deletion-and-expansion mechanism. Existing exact-resume identity checks and checkpoint semantics remain intact for historical formats.

Update real new-run consumers together: checkpoint resume, curve generation, status, assessment, validation-cache lookup and any supported child/fork command. A checkpoint-to-validation association must still identify the actual epoch, panel and prediction path. Do not declare a command compatible merely because a lower-level CSV reader exists. Unsupported historical fork formats can retain their original path rather than gaining speculative conversion code.

Legacy histories that lack validation loss terms cannot supply them retrospectively; label them unavailable and keep the original evidence. The historical Thermal CSV producer and compact JSON reader retain their saved field definitions. Readers should use the actual producer location so a stale copied category does not override an existing run's current records.

`tools/standardize_run_artifacts.py RUN_DIR` remains an explicit historical relocation tool, read-only by default. Its existence does not authorize applying it to a run. Preserve its lock, ownership, conflict, checksum and compatibility safeguards when an actual relocation is requested.

## Proportionate verification

Before adopting the new producer, verify CSV precision/missing values and string IDs; no repeated identity/arrays or default hash collection; exact uninterrupted-versus-resumed model/optimizer/RNG/sampler behavior; torn/uncommitted CSV recovery; truthful loss totals and plot cadence; actual new-run status/evaluation consumers; and unchanged historical readers. Demonstrate a small Thermal and Wind integration without new physical simulations. Use measured artifact sizes and save/plot/runtime costs to judge the benefit, not smaller JSON alone.

All generated results, figures and recovery receipts remain in ignored local DATA2 paths under the project's storage/cleanup rules. Commit and push maintained code, tests and documentation only after auditing the full outgoing commit range.
