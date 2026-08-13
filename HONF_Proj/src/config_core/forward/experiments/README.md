# Archived experiment overlays

These files preserve established ablations without copying the complete core
and case profiles. Apply one with `--experiment-overlay`; the resolver permits
changes only to keys already declared by the selected source profiles and
records the overlay path/hash and exact JSON in the run.

- `old_parity.json`: disables hyperedge value context.
- `uniform_h_assignment.json`: uniform module assignment and query routing with
  the pairwise-only decoder.
- `global_only.json`: disables the Stage-A dependency and uses global fallback
  heads with no interaction refinement.

Use `enhanced_honf_pairwise.json` as the base profile for these archived
switches. CLI run ID/name/epoch overrides remain separate provenance.
