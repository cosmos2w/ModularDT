# Forward compatibility contract

Run 1401 epoch 4585 is the current scientific baseline. Run 1000 remains the
historical golden compatibility reference. Both must strict-load all 237 state
keys and exactly match the case-0653 golden replay fixture.

Stage-1--6 fixed/additive, exchangeable, adaptive, scheduled-entmax, and
gathered modes remain loadable research modes. Historical configuration files
are immutable; their status and expected base profiles are recorded separately
in `src/config_core/forward/profile_registry.json`.

Training resume preserves parameter order, optimizer group membership, group
count, and optimizer state. Evaluation uses strict state loading. The canonical
run/evaluation layout has one producer location; historical aliases remain
readable and historical run directories are not rewritten.
