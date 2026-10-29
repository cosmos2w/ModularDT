# HONF consolidation and compatibility work plan

The current research branch is preserved at `7981c173e8b1d29ee2b3addb7b972d0f15003542` as `cosmos2w/historical-honf-r13-20261029`. Work proceeds on `cosmos2w/honf-consolidation-test` in an isolated DATA2 worktree. The original checkout used by ongoing training is unchanged; this work does not monitor, stop, restart or extend that training. The milestone is a retained Git reference, not a claim that server-side branch protection was installed.

## Scope and preservation

The maintained model catalogue has six names: `hypergraph_routed`, `dense_pairwise`, `content_attention`, `unified_hypergraph`, `relation_anchored` and `relation_anchored_active`. These correspond to development identities 1404,1804,A2-D,U-H,S,X. Public names describe information flow; original serialized configuration IDs, state tensor keys, normalization, role definitions and run identities remain unchanged. Previous-main six-hyperedge and dense checkpoints retain their loaders, visualization and resume paths. Historical case-specific physical heads remain compatibility implementations and are not generic HONF laws.

Local tests, datasets, post-processing code and evidence are protected. Retired implementation files and tests are moved into the ignored DATA2 `local_history` with their original repository paths; no locally saved material is deleted. The original source checkout and historical branch also retain the complete previous source. Existing scientific checkpoint, run, history, numerical evidence and shared-cache payloads are never migrated or rewritten for this cleanup.

## Staged implementation

1. Audit exact previous-main and six-family construction, import, configuration, checkpoint, optimizer, RNG/sampler and visualization contracts before removing code. Record the concrete live dependencies rather than pruning by historical filename alone.
2. Establish a small public `honf_forward_core.models` catalogue and `build_model(name, config)` interface over the original strict constructors. Keep historical serialized IDs and existing case/CLI entry points. A public name selects a core, not a physical normalization or a new checkpoint format.
3. Simplify the legacy pairwise dispatcher while proving state inventory, forward and gradient parity against previous main and the historical milestone. Collapse the anchored research inheritance into its actually used implementation and private numerical primitives; retain original-member access, applied relation provenance and union-score sharing. Reduce public research/task dispatch to the selected families.
4. Archive obsolete intermediate model/configuration/launcher/test modules after the maintained import closure is explicit. Preserve all local originals and test material. Keep shared foundation modules that the selected models truly execute; a module's historical name alone does not justify removal.
5. Test strict real-checkpoint construction, configuration round trips, fixed-input predictions and gradients, optimizer ordering/state and next-epoch resume behavior. Exercise previous-main six-hyperedge/dense, all six public constructors, case adapters, post-processing/organization extraction, and the inverse-facing loading paths actually supported. Do not silently claim a new model is supported by an old inverse verifier.
6. Rewrite the README around the HONF information path, six public model names, short class-specific equations, callable interfaces, core/case ownership and collaborator data onboarding. Explain fixed/ragged record identity, physical measures, coordinate frames, TRAIN-only normalization, targets kept out of model inputs and case-owned physical losses/verification. Demonstrate literal latest-checkpoint identity separately from saved older figures and best-field selectors; never relabel saved epoch500/1600 figures as current latest output.
7. Verify local figure links, decoded PNG previews, PDF masters, concise captions and rendered Markdown. Follow the three global organization views and disclose exact checkpoint/partition/control/mass provenance. Generated figures/data and one-time helpers stay local and ignored. Audit the entire outgoing range, run the repository artifact hook, commit and push only the test branch, verify its remote tip and report actual cleanup/storage results.

## Review boundaries

This is consolidation of existing methods, not a new architecture experiment. Numerical changes require separate evidence and must not hide inside a cleanup. Small deterministic CPU fixtures and bounded checkpoint/visualization checks establish compatibility; they do not establish scientific convergence or independent TEST performance. Preserve mature histories and negative results. New datasets and physical inverse acceptance remain owned by their case adapters and verifiers.

## Storage

The new worktree and outputs use the mounted DATA2 filesystem. The declared new retained allowance is 2 GiB, temporary allowance 512 MiB and estimated peak 1 GiB, with minimum free headroom of 100 GiB on DATA2 and 20 GiB on home. No dataset copy is allowed. Retired source moves and existing output bindings are recorded locally; tests use the round's DATA2 temporary root. Any necessary budget revision must be explicit before materializing additional payloads.
