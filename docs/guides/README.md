# HONF guides

The [project README](../../README.md) introduces the six model families, equations, saved examples and installation. These guides describe reusable practices rather than authorizing new experiments. Historical plans, reports and predecessor manuals remain local and in the frozen research branch.

| Guide | Purpose |
| --- | --- |
| [Model definition and notation](../../Model_Explain.md) | Implementation-bound M/E/C formulation, exchange, native hyperedges and query reading; canonical symbols for new model descriptions. |
| [Global organization visualization](Global_Hypergraph_Organization_Visualization.md) | Actual relation provenance, three standard views and saved-evidence plotting. |
| [Run artifacts and monitoring](Run_Artifacts_and_Monitoring.md) | Complete histories, meaningful loss panels, selectors and old layouts. |
| [Query tiling](Query_Tiling.md) | Decoder speed/memory tradeoffs, testing on your own inputs and explicit resume scope. |
| [Thermal development protocol](Thermal_Model_Development_Protocol.md) | Input-only membership, TRAIN-only normalization and fair epoch budgets. |
| [Thermal relation-anchored training](Thermal_Relation_Anchored_Training.md) | Current public model selection, recipes and continuation. |
| [Thermal and Wind case inputs](Thermal_Wind_Case_Inputs.md) | Input ownership, native units, differentiable Wind integration and frozen Thermal local ports. |
| [Thermal and Wind relation-anchored training](Case_Relation_Anchored_Training.md) | Case recipes, exact resume, objectives, local readouts and measured execution costs. |
| [Case response evaluation](Case_Response_Evaluation.md) | Rebuilt-design measurements, existing reference coverage and later response-study definitions. |
| [Wind native catalogue cache](Wind_Native_Catalogue_Cache.md) | Geometry/cache identity and native precision. |
| [Root layout and compatibility](../Root_Layout_and_Compatibility.md) | Promoted paths, installation and old checkpoints. |

The [project instructions](../../AGENTS.md) govern local storage and authorized execution. For a new case, use [case integration](../case_plugin.md), the [core contract](../../src/honf_forward_core/README.md), and the [electro-thermal handoff](../../Case_ElectroThermal/docs/COWORKER_HANDOFF.md). M/E identify input roles; case adapters own physical units, normalization, geometry and verification.
