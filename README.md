# Hypergraph Organized Neural Fields

**HONF learns how objects interact, then uses those relationships to predict physical fields.** Describe a system with object records, geometry and prescribed conditions; query the trained model at the locations where you need a prediction. The same core mechanisms can support different physical cases through a dataset-specific adapter.

<p align="center"><img src="docs/assets/readme/honf_showcase.png" alt="HONF showcase: overlapping original-member hyperedges, spatial access regions, and velocity and temperature reconstruction" width="1100"></p>

A thermal-channel example of relation-anchored HONF: overlapping relationships, their spatial access regions, and reconstructed fields. Temperature uses a cool-to-warm palette; signed errors use a dark diverging palette and display native-unit RMSE on the fluid mask. The overlapping sets preserve original membership; their ellipse shapes are illustrative. [PDF master](docs/assets/readme/honf_showcase.pdf) · [Figure details](docs/assets/readme/README.md).

## The idea

Many physical systems contain a variable number of interacting components: heat sources, turbines, material regions, boundary elements or other objects. HONF represents them as individual records rather than requiring one fixed arrangement. A record can carry coordinates, material properties, physical measures and prescribed controls.

The model contextualizes these records through information exchange. In its organized families, it also prepares relationships that guide which original members can contribute to a field query. A hyperedge describes an overlapping set of members and their permitted interactions; it does not require replacing those members with a single pooled feature.

Write the input records as $V=M\cup E$, where $M$ and $E$ identify input-object roles—for example, components and their environment. They do not assign different output fields to different roles. At a query location $q$, the overall computation is:

```math
Z=\mathrm{Exchange}_{\theta}(M,E;G),\qquad
\widehat{\mathbf{y}}(q)=\mathrm{Read}_{\theta}(q,Z,\mathcal{O}).
```

Here $G$ contains contextual relations, $Z$ contains contextualized records, and $\mathcal{O}$ is the selected model's organization, when present. The reader combines accessible information to predict the requested field channels. This is a schematic description: the families below differ in their exchange, organization and readout. A learned relationship describes model information flow, not physical causality.

## Model families

| Public identifier | Main distinction |
| --- | --- |
| `hypergraph_routed` | Shared latent hyperedges organize routing and original-member pair reads. |
| `dense_pairwise` | Dense pairwise and environmental reads, without hypergraph routing. |
| `content_attention` | Contextual record exchange followed by direct attention to member content. |
| `unified_hypergraph` | Learned hypergraph organization governs access to contextualized original members. |
| `relation_anchored` | Reuses contextual relations to build an environment-addressed member directory for joint field prediction. |
| `relation_anchored_active` | Uses the relation-anchored reader with access measures normalized over the active domain rather than the full inventory. |

Choose a family together with a case adapter and its configuration. Input dimensions, physical output laws and normalization are case-dependent; trained weights are not interchangeable across cases or families. Existing checkpoints retain their saved interfaces. See the [model registry](src/honf_forward_core/models.py) and [checkpoint compatibility guide](docs/Root_Layout_and_Compatibility.md).

For programmatic construction, use the public registry with a configuration exported by the matching core's `get_config()` method or configuration object's `to_dict()` method:

```python
import json
from pathlib import Path
from honf_forward_core.models import build_model

config = json.loads(Path("core_config.json").read_text())
core = build_model("relation_anchored", config)
```

This initializes a core model. To restore trained predictions, use the matching case loader with the saved configuration and normalization.

## Quick start

Python 3.10+ and a suitable PyTorch installation are required. From the repository root, install the shared package and the case you want to use:

```bash
python -m pip install -e ".[visualization]"
python -m pip install -e ./Case_ElectroThermal
# Other case packages: ./Case_ThermalChannel or ./Case_WindFarm
python -m hetmf train --help
```

Datasets and trained checkpoints are supplied separately. The electro-thermal case provides a concrete HDF5-based workflow; with a compatible dataset, inspect the configuration before training:

```bash
python -m hetmf train \
  --config Case_ElectroThermal/configs/train_six_hyperedge_standard.yaml \
  --dataset /path/to/standard.h5 --output /data2/your-project/new-run \
  --device cpu --dry-run
```

The dry run reads a small portion of the supplied dataset and checks the model integration without fitting. To train, replace `--dry-run` with `--epochs 100 --yes` and choose the appropriate device. This case's trainer currently requires a mounted `/data2` output filesystem and the free-space headroom declared in its configuration. A run groups its configuration, checkpoints, training history and validation outputs; the electro-thermal normalizer is stored in the matching HDF5 dataset.

```bash
# Continue the same run to 200 total epochs using its original config and dataset.
python -m hetmf train \
  --config Case_ElectroThermal/configs/train_six_hyperedge_standard.yaml \
  --dataset /path/to/standard.h5 --output /data2/your-project/new-run \
  --resume /data2/your-project/new-run/latest.pt --epochs 200 --device cuda:0 --yes

# Evaluate a saved checkpoint on the development split.
python -m hetmf evaluate --run /data2/your-project/new-run \
  --dataset /path/to/standard.h5 --checkpoint best_field.pt \
  --split dev --device cuda:0 --queries 1024
```

An epoch visits the declared training population once. Resume with the same data, model configuration and saved normalization. `latest.pt` contains the most recent continuation state; `best_field.pt` is selected by validation performance and may come from an earlier epoch. See the [electro-thermal walkthrough](Case_ElectroThermal/docs/COWORKER_HANDOFF.md), [Thermal data guide](Case_ThermalChannel/Dataset/PHYSICS_AND_DATA.md), and [Wind data guide](Case_WindFarm/Dataset/PHYSICS_AND_DATA.md) for each case's data and commands. Thermal workflows also expose their family-specific interfaces through `python tools/honf_research.py --help`.

## Adapting HONF to your dataset

The shared core handles information exchange, organization and field reading. Your **case adapter** defines what the records mean, how physical data enters the model, and how predictions return to native units. Add a case package rather than embedding dataset assumptions in the core.

1. **Define the data schema.** Give each scene stable object IDs, coordinates, prescribed features and controls, physical measures, query coordinates and target channels. Use masks for padded arrays or offsets for ragged arrays; retain the native geometry and units.
2. **Define independent splits and normalization.** Separate training, validation and test scenes before fitting. Keep dependent configurations together where appropriate, fit normalization on training data only, and save it with the run. Targets and post-solve quantities must not become input features.
3. **Implement the reader and adapter.** Map your schema to the chosen family's actual batch contract. The shared [`BatchData`](src/honf_forward_core/config.py) container and the existing case packages are useful references; some families use a specialized adapter. Set output channels, coordinate transforms and inverse normalization explicitly.
4. **Supply the learning task.** Define query sampling, physical losses, fieldwise metrics and native-unit visualization. The electro-thermal `HETMFTask` illustrates integration with the shared field-training engine; the [case plugin guide](docs/case_plugin.md) also describes the managed workflow interface.
5. **Check a small end-to-end example.** Verify input masks, a forward prediction, loss calculation, checkpoint loading and continuation before a full fit. Compare reconstructed fields with references on the physical domain, and label each field's RMSE or relative L2 error on its plot.

For inverse problems, the inverse core supplies request, design and verification interfaces. Your case supplies the design variables, admissible geometry and controls, physical objectives and reference verifier. Verify proposed candidates against that physical workflow before treating them as valid designs.

## Example outputs

<p align="center"><img src="docs/assets/readme/hypergraph_routed_fields.png" alt="Hypergraph-routed thermal-channel example showing reference, prediction and signed error for velocity and temperature" width="1000"></p>

A hypergraph-routed thermal-channel reconstruction. Reference and prediction share a scale within each row; RMSE is computed over 7,464 fluid points in packed native units. These examples illustrate model outputs rather than a controlled comparison between families. [PDF master](docs/assets/readme/hypergraph_routed_fields.pdf).

<p align="center"><img src="docs/assets/readme/relation_anchored_loss.png" alt="Example relation-anchored training history with sampled training losses and validation measurements" width="1000"></p>

A saved relation-anchored training history. Sampled training terms and validation measurements are shown separately because their reductions differ. [PDF master](docs/assets/readme/relation_anchored_loss.pdf).

The [figure collection](docs/assets/readme/README.md) provides full-resolution field and organization views, checkpoint provenance and interpretation details. The [organization visualization guide](docs/guides/Global_Hypergraph_Organization_Visualization.md) explains spatial access regions, original-member service maps and overlapping hyperedges.

## Repository map

| Location | Purpose |
| --- | --- |
| `src/honf_forward_core/` | Reusable model families and field-prediction mechanisms |
| `src/honf_inverse_core/` | Inverse requests, design and verification contracts |
| `src/honf_runtime/` | Shared workflows, training services and run artifacts |
| `Case_ThermalChannel/`, `Case_WindFarm/`, `Case_ElectroThermal/` | Dataset readers, physical adapters and case workflows |
| `tools/` | Command-line workflows and visualization tools |
| `docs/` | Integration guides and curated README figure resources |

Continue with the [guide index](docs/guides/README.md), [case integration guide](docs/case_plugin.md), or [installation and compatibility guide](docs/Root_Layout_and_Compatibility.md).
