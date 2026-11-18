# ThermalChannel inverse commands

These files are deliberately thin command-line launchers. Reusable inverse
models live in `honf_inverse_core`; ThermalChannel physics and workflow
implementations live in `channelthermal.inverse` and
`channelthermal.workflows`.

Run commands from the `HONF_Proj` directory after installing both packages:

```bash
python -m pip install -e . -e ./Case_ThermalChannel
python Case_ThermalChannel/scripts/inverse/build_inverse_dataset.py --help
python Case_ThermalChannel/scripts/inverse/train_inverse_hierarchical.py --help
python Case_ThermalChannel/scripts/inverse/evaluate_inverse_hierarchical.py --help
python Case_ThermalChannel/scripts/inverse/audit_inverse_hierarchy.py --help
```

The launchers contain no model behavior. Moving them does not change dataset
building, training, sampling, frozen-HONF verification, or artifact formats.

## Frozen placement benchmark

Run the native finite-pool comparison from the repository root.

```bash
python tools/thermal_placement_benchmark.py --panel /path/to/placement-panel.json \
  --checkpoint B0=/path/to/b0.pt --checkpoint R=/path/to/r.pt \
  --budget 4 --tau 0.4149511814117432 \
  --pressure-allowance 0.1 --j-resolution 0.01 --full-pool --format markdown
```

The numeric arguments illustrate the interface. Use the smoothing scale of
the supplied reference labels and choose the pressure allowance and
temperature resolution in the dataset's native units. J is smooth peak
material temperature. P is the inlet-to-outlet pressure drop.

The tool reads an explicit [response-family panel](../../../configs/examples/thermal_response_families.example.json).
Each case contains its prescribed source configuration and ordered module
slots. Each action provides edited centers in that same order. Module count,
disk geometry, operating context, and per-module heating come from the source.
Supply either all reference J/P deltas for a family or omit them all for a
surrogate-only run. An optional panel-level `tau` checks the label convention.
Panel roles describe exposure and are summarized independently.

Each checkpoint evaluates the same supplied cases, edits, seed and budgets. The proposal policies use frozen weights and require no fitting. The geometry check and baseline forward are shared. Each model policy uses at most the same number of fresh candidate forwards, including probes. Set the budget below the size of the legal pool to measure the value of shortlisting.

| Policy | How it chooses edits |
| --- | --- |
| Uniform sample | Draws from the legal pool without replacement. |
| Hot-receiver proximity | Favors moving modules near the anchor's high-temperature material receivers. |
| HONF environment access | Uses the same receiver weights and follows their prepared E-address directory weights back to the original modules. |
| Signed finite response | Uses single-coordinate probes to estimate combined moves by adding predicted J/P changes. It checks the actual combined layout with a fresh forward. |
| Reference-only poll | Selects the first single-coordinate move in the shared seeded order, uses zero model forwards, and reads at most one edited reference label to decide whether to keep the move. |

For an active finite-response comparison, include combined moves and their exact single-coordinate component moves as separate legal actions in the panel. Keep original slot order and use the same edited coordinate values in each component and its combined move. For example, to test moving module A right and module B down together, supply A-right, B-down, and A-right plus B-down. A budget of at least three permits two probes and one combined move. Probe predictions remain eligible for the final decision without a second forward. A family falls back to the uniform shortlist if the budget or bank cannot support any complete set of probes. The JSON records the probes, prioritized moves, estimates and fallback used.

Each scored model layout is built and evaluated freshly. The baseline prediction anchors J/P deltas. The model selects one ranked candidate under the pressure allowance or keeps the baseline. With complete matched labels the replay reads the selected edit's reference result and keeps it only if the pressure increase is within the allowance and J decreases by more than its resolution. The reference baseline must be feasible. The pressure allowance specifies a rise relative to that baseline in native units.

`--format markdown` shows mean forwards and probes, the fraction of labeled families with an accepted improvement, pressure failures among checked edits, and mean cooling. The loss columns distinguish improvement missed by shortlisting from improvement lost in the final decision, screening and verification. Both compare against finite-pool reference quantities. The default JSON adds selected actions and deltas, actual counted model work, timings and predicted hard temperature maxima. With `--full-pool`, each checkpoint separately scores every edit for a matched predictor comparison. This diagnostic reports the direction of resolved temperature changes, ordering of feasible edits and pressure error, with its extra forwards outside the policy budgets.

The policy replay uses stored labels for reference checks and the offline diagnostics use the full set of supplied labels. These labels should record matched converged references from a declared family partition and observation convention. Supply the same held-out panel for each checkpoint. Neither the replay nor the predictor diagnostic executes the reference solver, so reference solver runtime and prospective optimizer performance need a live comparison.

### Yield curves and critical decisions

Replace `--budget 4` with `--budgets 1 2 3 4` to compare policies across increasing candidate-forward budgets. Each point repeats the one-edit replay from the original baseline with the same cases, seed, scorer and reference rule. The curve reports mean verified cooling over labeled families in each panel role, including zero when the baseline is retained. The reference-only poll uses zero model forwards throughout. The Markdown report includes a chart, exact sampled values and a collapsible family-level event table. It marks first accepted improvements, changed recommendations, pressure failures, signed-probe activation and changes in cooling.

Curve points are separate budget experiments. A larger shortlist can change the recommendation and reduce reference-verified cooling. The JSON preserves actual forwards, probes, reference checks, both loss terms, per-budget results and critical steps. `--full-pool` scores the unrestricted action pool once per checkpoint and family at the largest budget, outside the policy budgets. [Worked yield curves](../../../design_investigation/Case_ThermalChannel/README.md#cooling-yield-versus-forward-budget)

### Fixed downstream temperature constraints

Add `regional_caps` at the top level of the panel to protect one or several fluid footprints while minimizing material `J` under the pressure allowance.

```json
{
  "regional_caps": {
    "lower_intake_peak": {"bounds": [0.85, 0.06, 1.0, 0.43], "reduction": "max", "allowance": 0.04},
    "upper_intake_mean": {"bounds": [0.85, 0.57, 1.0, 0.98], "reduction": "mean", "allowance": 0.01},
    "outlet_temperature_spread": {"bounds": [0.94, 0.07, 1.0, 0.93], "reduction": "std", "allowance": 0.0}
  }
}
```

Bounds are `[x0, y0, x1, y1]` normalized by channel size. Each rectangle must remain fluid across the baseline and every edit, and must contain valid native grid receivers. The allowance is a nonnegative maximum increase from a reference-feasible baseline. Use `std` for population standard deviation over the selected fluid receivers. A zero `std` allowance keeps outlet temperature uniformity at least as good as at the baseline. Include a matched delta under each region name alongside `J` and `P` on each labeled action. Extract the reference values using the same mask and reduction. These constraints apply to the model screen, reference check and finite-pool comparisons. Use a common legal action pool when comparing requests with different intake geometry.

To save a combined field and yield figure, run a budget sweep with one checkpoint and append `--figure-family FAMILY --figure-policy uniform --figure-budget 3 --figure-out /data2/wanglz/ModularDT/path/placement.pdf`. The plot uses the baseline and recommended layout at that curve point, including a rejected recommendation when applicable. It marks the inspected budget and the policy's critical decisions for the chosen family on the role-level curve. The native field passes made for display are recorded separately from the policy budget. Provide converged raw endpoint directories in the panel's `source_case_dir` and `raw_case_dir` for a reference field and signed error. The RMSE uses the plotted common valid fluid mask. When a matched field reference is unavailable, the figure labels the error unavailable and displays the predicted change over common fluid receivers. Use a new destination on verified DATA2. The writer records a 16 MiB limit for the figure and keeps a minimum 1 GiB of free space.
