# Run1505 e500 effective-K diagnostics

This folder contains a frozen-checkpoint diagnosis for the 90 cases in the
dataset's `test` split, reused as a development set. The exact Run1505 epoch
500 checkpoint is taken from the formation artifact. P2 routing was replayed
on every native `64 × 128 = 8192` grid point, in 1024-query decode chunks by
default. The figures and tables are diagnostic evidence for the 500-epoch run.

## Measures

- `K_plan` is the configured proposal capacity (12).
- `occupied_proposals` is the count of active proposal slots in a phase plan.
- `compact_R` is the number of valid classes after that phase's fusion plan.
- `compact_Kq` counts compact classes with positive query mass.
- `virtual_constituent_support` sums each supported class's proposal
  multiplicity. It counts original proposals represented by positive compact
  mass, not compact classes.
- `K_eff = 1 / sum(alpha_j^2)` is the participation ratio of normalized
  compact-class mass `alpha`. It describes mass concentration; it is neither
  positive support nor a proposal count.
- `dominant_compact_class` is the argmax mass class, with the first class used
  on an exact tie. Case-level dominant/support counts are reported separately
  for all grid points and for fluid points (`module_mask == 0`).

`case_phase_summary.csv` has 270 sampled P0/P1/P2 rows from the supplied Q=1024
formation artifact and 90 separately labeled full-grid P2 rows. P0/P1/P2 plans
are recomputed from phase-specific state, so read phase differences as separate
states rather than a cumulative compression path. The sampled phase Kq means
use only rows with a valid route; their query row counts differ by phase.

`case_effective_k_accuracy_association.png` links routing summaries to the
exact paired e500 field and port-temperature errors. It is descriptive and
does not establish a causal effect of routing. No training, checkpoint, or
production model files are modified.

## Files

- `effective_k_distribution_p2_fullgrid.png` and `.pdf`: all-grid and fluid
  query distributions, plus case means stratified by compact R.
- `phase_organizer_summary.png` and `.pdf`: case-level R and sampled Kq across
  P0/P1/P2.
- `compact_class_utilization.png` and `.pdf`: fluid-grid positive-support and
  dominant-class counts against P2 compact R, with case counts by R.
- `case_effective_k_accuracy_association.png` and `.pdf`: paired case-level
  error association.
- `case_level.csv`: 90 full-grid P2 case summaries and paired errors.
- `case_phase_summary.csv`: 270 sampled phase rows plus 90 full-grid P2 rows.
- `query_level_p2_fullgrid.npz`: compact per-query K_eff/Kq/virtual support,
  dominant compact class, P2 class mapping/multiplicity, and fluid mask in
  dataset case order. Raw mass matrices are omitted; rerun the script from the
  exact checkpoint to regenerate them if needed.
- `summary.json`: provenance, definitions, and population summaries.

## Reproduction

```bash
/home/wanglz/miniconda3/envs/ModularDT/bin/python HONF_Proj/tools/diagnostics/render_run1505_effective_k.py
```

The script validates the exact checkpoint identity, epoch, architecture, test
case order, 90 cases, 8192 queries per case, P2 plan agreement with the saved
formation artifact, and fluid-mask counts against the paired evaluator.

To rebuild the figures and README from saved CSV/JSON/NPZ data without rerunning
checkpoint inference:

```bash
/home/wanglz/miniconda3/envs/ModularDT/bin/python HONF_Proj/tools/diagnostics/render_run1505_effective_k.py --render-only
```
