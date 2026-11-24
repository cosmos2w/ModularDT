# HONF guides

The [project README](../../README.md) introduces the six model families, equations, saved examples and installation. These guides describe reusable practices rather than authorizing new experiments. Historical plans, reports and predecessor manuals remain local and in the frozen research branch.

| Guide | Purpose |
| --- | --- |
| [Global organization visualization](Global_Hypergraph_Organization_Visualization.md) | Actual relation provenance, three standard views and saved-evidence plotting. |
| [Run artifacts and monitoring](Run_Artifacts_and_Monitoring.md) | Shared CSV/small-JSON recording target, resumable state, loss panels and historical formats. |
| [Query tiling](Query_Tiling.md) | Decoder speed/memory tradeoffs, testing on your own inputs and explicit resume scope. |
| [Thermal development protocol](Thermal_Model_Development_Protocol.md) | Input-only membership, TRAIN-only normalization and fair epoch budgets. |
| [Thermal training budgets](Thermal_Training_Budget_Conventions.md) | Historical training lengths, staged TRAIN150/600 decisions, operational handoff and cost forecasts. |
| [Thermal relation-anchored training](Thermal_Relation_Anchored_Training.md) | Current public model selection, recipes and continuation. |
| [Native case training](Case_Relation_Anchored_Training.md) | Wind and Thermal provider dispatch, native objectives and identity-preserving continuation. |
| [Wind relation-anchored interface](../../Case_WindFarm/docs/relation_anchored.md) | Three-dimensional coverage, ragged turbine batches and native calibration. |
| [Thermal port/local integration](Thermal_Port_Local_Integration.md) | One-pass angular ports, frozen local readout and strict checkpoint loading. |
| [Response pairs from the base dataset](Base_Dataset_Response_Pairs.md) | No-new-solve supervision, the shared adapter target, physical correspondence and fair comparisons. |
| [General observation and response training](General_Observation_Response_Training.md) | Named native observation blocks, coupled-field sampling, executable fixture and collaborator binding. |
| [Target contrasts and state priority](Targeted_Response_and_State_Priority.md) | Bounded combinations of saved fields, explicit support and optional response-gradient projection. |
| [State and finite-response training](State_Response_Training.md) | Existing edited-scene provider, shared reductions, TRAIN-only calibration and historical compatibility. |
| [Thermal finite-response dataset](Thermal_Finite_Response_Dataset.md) | Optional historical edited-scene corpus, source precision, roles and direct reader. |
| [Physical response evaluation](Case_Response_Evaluation.md) | Fresh design preparation, physical-mask comparisons and resolved response metrics. |
| [Wind native catalogue cache](Wind_Native_Catalogue_Cache.md) | Geometry/cache identity and native precision. |
| [Root layout and compatibility](../Root_Layout_and_Compatibility.md) | Promoted paths, installation and old checkpoints. |

The [project instructions](../../AGENTS.md) govern local storage and authorized execution. For a new case, use [case integration](../case_plugin.md), the [core contract](../../src/honf_forward_core/README.md), and the [electro-thermal handoff](../../Case_ElectroThermal/docs/COWORKER_HANDOFF.md). M/E identify input roles; case adapters own physical units, normalization, geometry and verification.
