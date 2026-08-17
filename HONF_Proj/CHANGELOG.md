# Changelog

## Unreleased

### Hierarchical inverse generator and evaluation workflow

- Audited the initial inverse release surface, reorganized the root README as a
  beginner-first forward-to-inverse workflow, and expanded the mathematical
  reference with exact request and compact-plan definitions, HDF5 shapes,
  rectified-flow/training equations, verifier/ranking semantics, and explicit
  first-version limitations. Public
  loading now validates checkpoint schema/provenance, tensor-only correction
  fails before an unverifiable proposal, and invalid candidate counts fail
  before any HONF call.
- Moved the four ThermalChannel inverse CLI launchers out of the project root
  into `Case_ThermalChannel/scripts/inverse/`. They remain thin dispatchers to
  the unchanged installed workflow modules; all documented commands and smoke
  checks now use the case-owned paths.
- Added a modest two-stage conditional rectified-flow hierarchy: a
  permutation-invariant request/context encoder, fixed-canonical-edge compact
  plan flow, padded-slot layout flow with presence/count heads, optional
  canonical/Hungarian/Sinkhorn matching, and an optional single bounded joint
  corrector. Activity and strength stay derived from mass features.
- Added the four explicit trainer stages, mixed teacher/generated plan policy,
  flow-dominance caps, sparse differentiable frozen-HONF consistency bridge,
  atomic inverse checkpoints, familiar checkpoint aliases, metrics, resolved
  config, and complete dataset/forward/schema/normalization provenance.
- Added strict generated-`D,c` forward input construction, including the
  immutable Stage-A physical parameters needed to reproduce the dataset-backed
  autonomous path without borrowing source boundary values. A GPU test proves
  exact compact-plan/port parity, and a one-batch GPU test proves gradients
  reach inverse layout outputs while all HONF parameters remain frozen.
- Added exact raw/corrected candidate populations, explicit HONF-call ledgers,
  request/geometry/plan scoring, request-first diversity-aware ranking,
  JSON/CSV/compressed-NPZ serialization, five request examples, predicted-only
  top-candidate plots, and population plots. Success is reported before
  reranking.
- Added an analytic one-pass ThermalChannel endpoint parameterization for edge
  bounds, minimum x separation, and optional heat range. It is part of decode,
  not an iterative repair/search loop; the first GPU evaluation produced
  geometry-valid raw and corrected candidates with a bounded correction.
- Added an explicit verified one-pass trust region. All correction proposals
  remain available as a proposal-only group; each lineage accepts its proposal
  only when exact request violation improves and geometry remains valid. The
  serializable API, summaries, CSVs, manifests, and comparison plots now report
  raw, proposal, accepted, and final-ranked populations separately.
- Fixed corrector-only supervision so a sampled module with no feasible target
  slot is no longer pulled toward zero padding, and inactive slots no longer
  dilute the target loss. A high-fidelity `32x16` corrector continuation keeps
  plan/layout flows frozen and preserves the one-pass topology contract.

### Inverse dataset ABI and diagnostics

- Added deterministic split-before-augmentation, per-case/per-variant seeds,
  2–4 active requests, at most one regional token, train-only statistics,
  canonical active-first layouts, strict case-major HDF5 I/O, a flattened
  public reader, atomic writes, manifests, split hashes, and the four required
  reread-based diagnostics.
- Added a config-driven GPU builder and early/dry-run resource reporting. A
  three-case partial artifact completed against a SHA-pinned immutable snapshot
  after the live forward checkpoint was observed changing concurrently; the
  provenance guard correctly refused to mix incompatible checkpoint bytes.
- Completed the full 690-case dataset build with 540/60/90 inverse
  train/validation/test cases, 16 variants per case, no split leakage, and
  dataset SHA-256
  `e4acbd96ffa28f2a204551b04522192d0e9ce2555ec4dbaaff4832bf94897989`.

### Inverse bounded milestone evidence

- Completed the bounded four-stage full-data diagnostic schedule and empirical
  matching audit. Canonical plan distance remained order-sensitive, while
  Sinkhorn matching reduced held-out median planned/realized distance to the
  release range; matching therefore remains optional but is enabled in the
  diagnostic checkpoint by recorded evidence.
- A 16-request held-out audit of 64 raw candidates and 64 one-pass proposals
  passes all five hierarchy milestones on the high-fidelity corrector
  checkpoint: request-sensitive plan RMS `0.0790`, raw geometry validity
  `1.000`, layout diversity `0.998`, median Sinkhorn plan distance `0.1365`,
  raw request-term satisfaction `0.3646`, `45.31%` of raw candidates satisfying
  at least half their active terms, and accepted one-pass request-violation
  improvement `10.70%`. The accepted fraction is `0.453`; mean
  proposal correction magnitude is `0.0165`, accepted median center movement is
  `0.0188` of the domain diagonal, and median active-heat movement is `0.0139`
  training standard deviations.
- The final checkpoint SHA-256 is
  `592e6e532c7538dd08515dac73c3b15003a146410325a62877c4e56aba77389d`;
  the persisted 16-request audit SHA-256 is
  `18128339ddd68da22a5773dbf71e50202c02356bba99fb0a7e2a1c5907030ed8`.
- Completed one-candidate GPU correction-enabled and CPU correction-disabled
  evaluation smokes. They produced the required JSON/CSV/NPZ/plot inventory
  with explicit SHA-256 manifests and exact two-call/one-call ledgers.
- These runs are deliberately bounded diagnostics. They establish the intended
  generative hierarchy but do not claim converged production-scale training.
- Final validation passed 95 CPU tests plus both physical-GPU-1 frozen-verifier
  integration tests. All inverse modules compile, 25 inverse JSON files parse,
  no runtime import reaches the old inverse source tree, and the diff check is
  clean.

### Inverse data contracts and frozen verifier

- Added case-neutral contracts, affine normalization helpers, and the strict
  schema-v1 request codec with a four-token cap, seven ThermalChannel
  functionals, four relations, separate geometry constraints, JSON
  round-tripping, tensorization, and readable summaries.
- Added ThermalChannel context/design canonicalization, exact geometry and
  functional evaluation, and compact-plan schema v1. The compact target uses
  canonical fixed-edge order and derives organizer activity/strength from the
  mass columns rather than generating contradictory duplicate variables.
- Added a frozen verifier that uses the maintained self-contained checkpoint
  loader, checkpoint normalization, autonomous predicted ports, chunked
  decoding, the final post-fusion organizer, and canonical full-plan exporter.
  A real physical-GPU-0 replay produced finite outputs and a deterministic
  `6x12` realized compact plan; the focused contract suite passes 17 tests.

### Inverse hierarchy planning audit

- Reviewed `INVERSE_CODING.md` against the maintained forward loader,
  autonomous predicted-port path, final organizer, canonical plan exporter,
  runtime/plugin boundaries, packed ThermalChannel grids, and release
  dependencies before starting implementation.
- Simplified the plan-flow state from 11 to 10 independent continuous
  attributes: hyperedge strength and activity now follow the forward
  organizer's exact mass-derived definitions, preventing contradictory plans.
- Removed priority/weight double counting, made immutable base-plan fidelity
  explicit across correction, and replaced impractically exact diversity ties
  with small recorded near-tie bands.
- Required generated-design verification to construct inputs from `D,c`
  without borrowing a hidden dataset case and to inherit the packed dataset's
  recorded `128x64` global, `64x64` internal, and 64-point interface contracts.
- Replaced the proposed evaluation-only HDF5 dependency with versioned JSON,
  CSV, and compressed NPZ artifacts so the reusable inverse core remains on its
  NumPy/Torch dependency boundary.
- Added quantitative held-out evidence gates for the five end-to-end hierarchy
  milestones. The unchanged forward baseline passed 49 tests in `ModularDT`.

## 0.1.0 — 2026-08-12

- Extracted the checkpoint-compatible HONF forward core and generic runtime.
- Added the installable ThermalChannel case plugin and standardized Stage-A
  local-module contract.
- Added strict split configuration, named dataset resources, manifests, schema
  documentation, confirmation-gated training, real resume state, canonical run
  provenance, and explicit checkpoint selection.
- Preserved global/local evaluation, plots, routing diagnostics, hypergraph-plan
  export, and aligned multi-model comparison.
- Added case-injected query features, neutral near-context scale, strict
  experiment overlays, historical checkpoint migration, and equal-budget
  performance tooling.

See `VALIDATION.md` for the numerical compatibility evidence and complete rerun
record.
