# HONF Round9 — matched e3500 evaluation and decision stop

**Status:** authorized amendment during execution on October 17, 2026; measurements are pending until the named artifacts are verified. **Scope:** all four Thermal and four Wind arms. This amendment supersedes the remaining training horizon and pending audit ages in the [original Round9 work plan](HONF_Round9_Joint_Native_Hypergraph_Work_Plan.md). It preserves that document and the completed evidence as history.

The user amended Round9 to stop every arm at literal epoch 3500, evaluate the matched models, complete the report and cleanup, commit and push durable work, and wait for a decision. Epochs 4000–5000 are not authorized by this amendment. The resulting study must be described as an e3500 decision stop, not completion of the original e5000 horizons.

## Training and identity

Complete the currently admitted e3000–3200 block, review all eight arms together, and admit only the matched e3200–3500 block if the existing scientific and resource checks pass. Stop each child through the native `--stop-after 3500` mechanism. Preserve weights, named AdamW moments, RNG state, data membership, normalization, absolute-epoch case/query streams, native losses, auxiliary/response calls, protections, router policy and historical checkpoints. Retain the original recipe's constant learning-rate policy and horizon metadata; that metadata does not authorize training beyond the explicit stop.

| Executed component per dataset | New epochs | Thermal updates | Wind updates |
|---|---:|---:|---:|
| D9-M, D8 e2500 to e3500 | 1,000 | 4,000 | 5,000 |
| Shared dense D9-ME warm-up, e2500 to e3000 | 500 | 2,000 | 2,500 |
| D9-ME child, e3000 to e3500 | 500 | 2,000 | 2,500 |
| H9-ME child, e3000 to e3500 | 500 | 2,000 | 2,500 |
| S9-ME child, e3000 to e3500 | 500 | 2,000 | 2,500 |
| Actually executed total | 3,000 | 12,000 | 15,000 |

The amended target is 27,000 scientific optimizer updates, with the shared warm-up charged once. Each terminal lineage has 1,000 new case/query epochs after D8. H9-ME and S9-ME each receive only 500 router-training epochs: 200 initial dense-estimator epochs and 300 balanced-cadence epochs. Their original deterministic training phase has not begun. Insufficient sparsity at this age is inconclusive, not proof that the architecture or longer prescribed training fails.

At sparse age 500, the unchanged schedule has removable-row targets M ≈ 0.912578 and E ≈ 0.787735, relaxation temperature 0.5 and noise scale ≈ 0.625782. Training audits must preserve that stochastic policy and the saved macro clock. Deployed validation/inference remains deterministic. Neither policy may be silently substituted for the other to improve accuracy, sparsity or cost.

## Applicable e3500 audits

Retarget pending jobs to new, explicitly labelled e3500 manifests. Preserve prior manifests, source seals, execution receipts and the completed e3000 boards; mark unexecuted e4000/e5000 job plans as superseded, not completed. Do not execute both old and replacement jobs. No additional e4000 board is needed.

- Run complete fixed development-validation boards and all selected development-TRAIN endpoint audits for all eight literal e3500 checkpoints. Keep Thermal M1/M2 strata visible in selected TRAIN, individual physical roles, case tails and Wind layout summaries. Compare H and S with both same-age dense controls.
- Audit actual typed M/E nonlinear reads, IDs, protections, floors, padding and checkpoint recomputation. Distinguish deterministic logical support, stochastic training support, selected-step execution and dense-estimator work.
- Retain the fair same-weight/mask low/high-M physical memory probes, complete native inference and reuse scopes, and full four-epoch native training cycles. Replay policy/query epochs e3497–3500 from e3500 source weights and optimizer/RNG state as discarded engineering work. Label instrumentation and subsequent trajectory differences explicitly. Preserve the completed twenty-epoch lifetime checks as earlier engineering evidence.
- Include the trained S9-ME control in applicable field, read, inference/runtime, E-path and physical-response comparisons. Report unequal work rather than forcing a posthoc cutoff.
- Run applicable saved-reference responses, typed membership/value controls, legal input-shift adaptation, seam and invariance audits. Keep rebuilt-input, fixed-route, final-read and physical-reference evidence separate. Preserve source-column identity, signed Thermal response coefficients, native units and physical-provider limits.
- Keep saved-design ranking conditional on its declared field, E-effect, actual omission and measured-memory prerequisites. Failure to qualify at e3500 does not authorize a replacement solver, target synthesis or a new inverse campaign.

All original GPU2, precision, common microbatch, elapsed-time, engineering-update, disk-budget and headroom limits remain in force. Retargeting does not reset any usage ledger. At amendment time, 483 engineering updates had been recorded; the pending 116-update audit allocation would bring the total to 599 of the unchanged 600-update cap. Verify actual journals before execution and closeout. TEST remains locked and no new solver attempts are authorized.

## Report, decision and closeout

Begin the report by explaining the implemented joint typed reader and the amended early decision stop. Give exact checkpoint paths/hashes, source ancestry, actual cases/updates/queries and measured elapsed time. Present predictor gains and misses per field against both dense controls, including tails and populated strata; report E-conditioning utility through measured effects and reference errors, not projection amplitude alone.

Present organizer results as executed information flow, input dependence and physical-response agreement separately. Report actual and logical M/E work, memory baselines/peaks, selected-step and complete-cycle costs, full-access fallback and complete native-call latency. A speedup is not required. Dense contextualization means omitted final records may still influence selected states; no causal completeness or independent environmental excitation is established by the read graph.

Retain the important numerical evidence and visually inspected figure families, with explicit e3500 and partition labels. Use the repository's single-line report paragraphs, relative figure embeds, PDF masters and small required raster companions. Complete exact-allowlist cleanup, preserve protected artifacts and historical evidence, report removed files/bytes and remaining capacity, and commit/push durable work with the full outgoing-history audit.

Recommend one of stopping this line of work, extending specified matched comparisons under a new explicit budget, or proposing a separately labelled full-data run for a useful candidate. State what evidence supports that recommendation and what remains unresolved after 500 router-training epochs. Do not start an extension or a full-data run; wait for the user's decision after the e3500 report and closeout.
