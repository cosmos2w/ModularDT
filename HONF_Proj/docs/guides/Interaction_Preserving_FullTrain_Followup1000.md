# Interaction-preserving full-TRAIN follow-up at 1,000 epochs

This protocol supports the user-authorized follow-up after the segmented interaction-preserving recovery comparison. It is a fresh full-TRAIN experiment with its own membership, normalization, calibration, source binding, and checkpoint lineage. The execution protocol is `interaction_preserving_full_train_followup1000_v1`; `formal_full` remains false so legacy formal recipes retain their original identity.

The selected pair is Thermal T4203/P-H with vorticity derived from native velocity curl and Wind W2501/P with the shared nonlinear velocity core. Thermal flow and temperature remain jointly trained; no external learned flow model supplies their fields. P-family support in the runner does not authorize a model portfolio. The segmented recovery decision and a current resource forecast must identify the actual pair before execution.

## Population and calibration

Thermal uses all 600 original TRAIN cases and the established canonical89 exposed validation panel. The original H5 `test` label denotes the already exposed validation population here; canonical89 excludes historical duplicate 0273. Normalization is fitted only on the 600 TRAIN cases. Fresh P calibration uses the full TRAIN population and the four existing TRAIN response families 0001/0318/0333/0348, with zero optimizer updates and no validation target materialization. A separate receipt must bind this population, the normalization, native geometry-mask catalogs, initial direct weights, and committed training sources; quarter-data calibration is rejected. Stored native geometry masks are receiver metadata and do not enter learned context.

Wind uses all 420 original TRAIN direction rows across 140 layouts and the fixed fullVALID90 panel across 30 layouts. The original 90 TEST rows across 30 layouts remain locked. The profile, normalization, and componentwise role scales are fitted on TRAIN only in the separate follow-up cache namespace. Model environment tokens remain E8; the normalization-fit view is a distinct preparation detail.

## Execution boundary

The optimizer retains the selected mechanism, loss weights, queries, microbatch size, effective case batch, initialization seed, and gradient clipping. Its declared schedule has a 5,000-epoch horizon: 20 warmup epochs from 3e-5 to 3e-4, a hold through epoch 1,000, and cosine decay thereafter toward 3e-6. Declaring that schedule does not authorize training beyond epoch 1,000.

Fresh training must stop at epoch 100. An exact continuation of the same latest checkpoint can advance from 100 to 500 and then from 500 to 1,000 after the required reviews. The runner rejects skipped reviews, cross-population resumes, branches, learned parents, and extensions above 1,000. The recipe requires an explicit `root_protocol_go=true` before calibration or fitting. `validate-followup` checks recipe metadata without dataset access; full-follow-up `dry-run` performs no optimizer update. Preparation and startup receipts are not completed training evidence.

Monitoring, latest state, best-field state, and curves are saved every 100 epochs. Retain declared milestones and the best/latest aliases. Freeze the field-only selector over the ten monitoring checkpoints before the final native validation comparison; response, graph, and full-native errors must not determine selection.

## Review and comparison

Record actual TRAIN-case visits, optimizer updates, microbatches, primary queries, elapsed time, source identities, and checkpoint hashes at each review. Native-curl stencil unions and padding are executed work, separate from primary supervision counts. Charge preparation, failures, profiling, and both GPU lifetimes against the campaign ceiling; preserve the final elapsed-time reserve.

Compare the completed follow-up with saved Thermal 1404/1804/3906 canonical89 physical results and saved Wind 2102/2103/2204 fullVALID90 Q8192 physical results. Use the same saved case/query identities and physical reducers, and report the older checkpoints' selected and endpoint ages. These historical models have different architectures, objectives, schedules, and training ages, so physical comparisons do not isolate a causal architecture effect. WindTEST and the exhausted 326/326 solver allowance remain unchanged.

The user decides whether to extend to 5,000 after seeing the completed 1,000-epoch comparison. This guide provides no 5,000-epoch execution approval, organizer promotion, donor reduction, new solve, or inverse campaign.
