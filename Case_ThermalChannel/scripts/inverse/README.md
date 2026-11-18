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
  --budget 2 --tau 0.4149511814117432 \
  --pressure-allowance 0.1 --j-resolution 0.01
```

The numeric arguments illustrate the interface. Use the smoothing scale of
the supplied reference labels and choose the pressure allowance and
temperature resolution in the dataset's native units.

The tool reads an explicit [response-family panel](../../../configs/examples/thermal_response_families.example.json).
Each case contains its prescribed source configuration and ordered module
slots. Each action provides edited centers in that same order. Module count,
disk geometry, operating context, and per-module heating come from the source.
Supply either all reference J/P deltas for a family or omit them all for a
surrogate-only run. An optional panel-level `tau` checks the label convention.
Panel roles describe exposure and are summarized independently.

Three methods shortlist the same number of edits from each legal pool.
Uniform samples without replacement. Proximity scores the moved modules by
their distance to hot material receivers. Environment access uses the same
receiver weights, distributes them through the native interpolation stencils
and E-address cover, and accumulates learned access weight back to the
original modules. Both guided methods use the fraction of each edit's total
movement at each module. A fresh native HONF pass scores each shortlisted
layout. The shared baseline prediction anchors temperature and pressure
deltas. The pressure allowance is measured relative to a feasible baseline.
Set the candidate budget below the pool size to measure a shortlisting effect.

The JSON on stdout includes candidate and baseline work, timings, predicted
hard temperature maxima, and reference outcomes when matched labels are
provided. Reference deltas do not enter selection. A reference result must
satisfy the pressure allowance and improve J beyond its stated resolution
to replace the baseline. Regret is relative to the best feasible action in
the supplied pool. This replay consumes already supplied reference labels.
It is distinct from evaluation of a live optimizer with a solver budget.
