# Case-local artifacts

This directory is reserved for external, untracked artifacts associated with
the WindFarm preprocessing case.  There are no model checkpoints or training
outputs in this round.

Generated profile tables, derived arrays, figure files, and showcase metadata
should live below a local output directory (for example `artifacts/generated/`)
and must not be committed.  Each generated artifact should retain the source
logical dataset ID, manifest schema version, case row/index, selected cut
coordinates, and preprocessing code revision.

The source dataset itself is accessed through `Dataset/links/wind_farm`; do not
copy it into this directory.  If a future model workflow adds checkpoints,
define their schema and lifecycle before placing them here.
