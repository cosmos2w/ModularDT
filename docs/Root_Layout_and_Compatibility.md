# Project layout and compatibility

HONF occupies the repository root. The shared Python packages live in `src/`, physical data and adapters live in `Case_*/`, and reusable commands live in `tools/`. Case packages own units, normalization, geometry, targets and verification; shared model code does not infer those semantics from a model-family name.

## Installation and commands

Install the core and the case you use from an editable source checkout:

```bash
python -m pip install -e '.[dev,visualization]' -e ./Case_ThermalChannel
# Other case packages:
python -m pip install -e ./Case_WindFarm -e ./Case_ElectroThermal
python train.py --help
python evaluate.py --help
python tools/honf_research.py --help
python -m hetmf train --help
python -m pytest -q
```

Managed training/evaluation entry points use the source checkout because configurations and case schemas live beside the source. Core wheels expose Python model packages; external datasets, checkpoints, research reports and all source-tree launch profiles are not bundled. Configure each case's ignored dataset-location map from its checked-in example. Dry-runs inspect configuration and integration without starting an optimizer or a training run.

## Existing checkpoints and resume

The six public model-family names describe model choices. Package names, configuration keys, serialized architecture IDs, tensor names, normalization and physical laws remain part of the checkpoint contract. Use the matching case loader, original model configuration and saved normalization. Existing absolute external checkpoint/data paths are not rewritten; `project://` paths resolve relative to this repository root, independent of the shell's current directory.

Fresh `relation_anchored` and `relation_anchored_active` configurations share `DetailAnchoredMemberRoutingCore` in the mechanism module `interaction_reused_member_routing/detail.py`; the public configuration constructor selects the corresponding access measure. Loading uses a concentrated saved-schema dispatcher: historical anchored schemas reconstruct `AnchoredMemberRoutingCore`, while the detail-preserving schema reconstructs the shared Detail core. Saved architecture IDs and state-dictionary names remain exact. The former `r14` import path is a small re-export of that same class, rather than another implementation. Thermal adapters retain their saved adapter schema and normalizer and use this dispatcher when restoring their core.

Resume preserves case membership, input identity, optimizer/scaler, RNG, sampler, selection, schedule and complete history. Latest, best-field and literal epoch checkpoints remain distinct. A compatible load is not a completed continuation. See the [compatibility contract](architecture/compatibility.md) and [run-artifact guide](guides/Run_Artifacts_and_Monitoring.md).

The maintained contract suite covers all six families, original six-hyperedge/dense next-update resume, relation-anchored and unified-hypergraph task handoffs, case adapters and field/organization visualization. A historical numerical-digest test marks only its exact known mismatch as expected; a new mismatch still fails. CUDA and external-data checks require their declared resources. Consult the actual test output for the environment being used.

## Documentation figures and external resources

The [README asset index](assets/readme/README.md) lists four selected PNG/PDF pairs. They live under `docs/assets/readme/`, use relative links, preserve aspect ratio and state checkpoint source and figure purpose. Detailed field figures show measured physical errors; the opening figure is a conceptual information-flow illustration. Update this same directory for future README revisions. Numerical arrays, weights, full histories, generated reports and one-time scripts remain external to Git. The artifact hook allows only the curated README figure directory as an exception.

Datasets, checkpoints and output locations are machine-owned resources. Configure them explicitly on each host; do not assume that a maintainer's storage paths exist elsewhere. Existing workspace aliases may retain earlier directory names without changing their payloads. Local `docs/reports/` and `UpgradePlan/` archives are ignored and are not part of the source distribution.
