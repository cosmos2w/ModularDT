# Thermal and Wind relation-anchored training

The case providers use `tools/relation_anchored_train.py` and the shared `TrainingEngine`. Case code owns native sampling, physical transforms, loss terms and metrics; the shared core and its joint field prediction are unchanged. These recipes describe execution settings, not evidence of model convergence or permission to start a run.

## Recipes and entry point

Thermal recipes `configs/research/thermal_r15_2_joint_train{150,600}_v1.json` select direct joint prediction, while `thermal_r15_2_localports_train{150,600}_v1.json` enable the case-owned port head. The declared segmented schedule horizon is 200 epochs and the formal horizon is 500; both initially stop at 100. Wind recipes `configs/research/wind_relation_anchored_r15_2_w{0,1}.json` select `4x4x4` and `8x4x2` E grids respectively, with a 200-epoch schedule and an initial stop at 100. Configure the selected measured tile and physical microbatch before starting a lineage; retain its recipe and effective batch during resume.

From the repository root, replace the placeholders with the selected recipe and authorized run directory:

```sh
python tools/relation_anchored_train.py inspect --recipe RECIPE.json --device cpu
python tools/relation_anchored_train.py fit --recipe RECIPE.json --device cuda:0 --output RUN --stop-after 100
python tools/relation_anchored_train.py fit --recipe RECIPE.json --device cuda:0 --output RUN --resume latest --stop-after 200
python tools/relation_anchored_train.py evaluate --recipe RECIPE.json --device cuda:0 --output RUN --checkpoint latest
```

An extension requires its own authorization; the example does not authorize it. Resume uses the same recipe and run's latest state, with strict model and optimizer-group loading and restoration of provider state, RNG, sampler and history. A saved 500-epoch schedule can resume to an authorized later stop without resetting that schedule. An epoch visits the declared training membership once. `case_epoch_v1` keys native queries by dataset membership, epoch, physical case and role, rather than arm or microbatch position. Record actual GPU identity and ownership under the project instructions before execution.

The declared ordinary validation cadence is every 100 epochs; the engine also validates at a requested stop or clean stop. Latest state is saved every epoch, with 100-epoch milestones and a best-native-field selector. Curves first appear after the first completed epoch, refresh every 10 epochs and at a stop, and include weighted total TRAIN and validation losses plus components. Before the first validation the validation panel is marked pending. Progress is enabled by default and can be disabled in the recipe for unattended consumers. Totals use different TRAIN and validation reductions and need their prediction paths and term coverage read alongside them.

## Thermal objective and local readouts

Thermal preserves the native 16-angle, stride-4 field sampler and averages the full and restricted organized native losses during training. The port candidate adds equally weighted normalized outside-temperature and h losses on restricted 64-angle reads from the same prepared organization. The TRAIN-only channel scales, valid-port masks and per-channel valid-case reductions are saved in provider identity and loss metadata. Validation uses the restricted native prediction and restricted port read. The best-field selector uses native field accuracy rather than the augmented loss total.

Port targets are supervision and do not enter the scene or prescribed controls. The head consumes the organized outside prediction, angular geometry and prescribed heat/material properties. The h target is the saved epsilon-regularized, clipped effective coefficient with its validity mask; the nonnegative head does not add an upper bound because the packed metadata does not establish its numerical clip limit. These quantities do not constitute independently measured convection physics.

The local readout loads the declared epoch-4975 asset strictly with its own embedded transforms and frozen weights. It derives all four h/temperature summaries from the selected ports and evaluates present modules in one batched call. The case helper `channelthermal.local_ports_evaluation.evaluate_local_port_readouts` compares direct prediction with predicted-port, oracle-port and TRAIN-mean-h local readouts on the same batch; oracle ports use solved targets and are diagnostic only. The CLI `evaluate` command reports native validation and port-loss metrics; call the case helper explicitly for the matched local-readout comparison. A terminal local result does not by itself establish a seamless coupled field or inverse readiness.

## Wind objective and cost

The Wind bridge uses the fixed TRAIN72/DEV24 layout-group membership, saved TRAIN-only normalizer and background profile, and the five-role objective: volume, hub slab, downstream envelope, near turbine and background, each with weight 0.2 and the existing per-component role scales. It predicts the three velocity components jointly. The two E64 grids retain the same physical support and represented volume. Ragged scenes are packed into one preparation/decode per microbatch with masked module padding and original IDs; work reports include the padding slots and actual core operations. Output and gradient parity with sequential execution is tested, while throughput and memory must be measured on the actual workload.

An explicit existing role-scale calibration may be used when the file-based dataset fingerprint differs only after the saved and current non-fingerprint recipe bindings agree and the current deterministic 12-row TRAIN calibration panel reproduces every nonvolatile saved calibration field. The provider records both fingerprints and the panel-only scope. This replay verifies those calibration observations; it does not establish identity of the complete native dataset or authorize access to locked TEST data. The normal read-only loader remains strict for other workflows.

For execution comparisons preserve initial state, native rows, queries, objective, precision and effective batch. Measure complete preparation, prediction, loss and backward time and peak allocated/reserved memory; report preparation and decode work counters separately. Query tiles and microbatch packing can change numerical reduction order, so compare outputs, losses and gradients with appropriate numerical tolerances. Report actual executed member reads and pair calculations separately from memory or wall time; none alone establishes end-to-end speedup or physical causality.
