# Migration from Demo 1

The original `1_Demo_ChannelThermal/src_HONF_CL` remains an immutable reference
until the modular implementation and complete reruns are accepted. No dataset
or historical run is copied into the release source tree.

Command mapping:

| Previous command | Modular command |
|---|---|
| `src_HONF_CL/train_local.py` | `train.py --config src/config_core/forward/local_module_thermal_disk.json` |
| `src_HONF_CL/train.py` | `train.py --config <core-profile>` |
| `src_HONF_CL/evaluate_local.py` | `evaluate.py --workflow local_module --config <local-profile>` |
| `src_HONF_CL/evaluate.py` | `evaluate.py --workflow forward --config <core-profile>` |
| `src_HONF_CL/compare_models.py` | `evaluate.py --workflow compare --config <core-profile>` |

Old combined JSON files are replaced by a core launch profile plus
`Case_ThermalChannel/configs/case_default.json`. Old `Data_Saved` paths are
replaced by logical IDs and an ignored location map. Old `Saved_Model_*`
directories are replaced by the structured `Trained_Results/ThermalChannel`
tree. The model parameter names, decoder options, Stage-A architecture,
normalization behavior, physical losses, checkpoint selectors, and all
post-processing capabilities are retained.
