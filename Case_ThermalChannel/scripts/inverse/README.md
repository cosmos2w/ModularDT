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
