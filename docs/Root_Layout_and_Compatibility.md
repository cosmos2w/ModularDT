# Root layout and compatibility

The HONF project now occupies the repository root. Its content is selected from `cosmos2w/honf-consolidation-test` at `282fb3f26008eabbedba9479e5a1fed291700423`; the publication commit has the old main `49b1bf3369b0418b1f23eefd559ff5f436fb6eb7` as its sole parent. Main advances by ordinary fast-forward. The intervening research commits are not merged or copied into main's ancestry.

The complete old main history is retained on `cosmos2w/historical-main-20261029`. The complete research milestone remains on `cosmos2w/historical-honf-r13-20261029`. Existing source worktrees, checkpoints, histories, test code, numerical data and postprocessing materials stay local. `docs/reports/` and `UpgradePlan/` are explicitly local-only directories and are excluded from the new main tree.

## Installation and commands

From the repository root, install the core and the case you use:

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

The documented training/evaluation entry points use an editable source checkout because configurations and case schemas live beside the source. A core wheel exposes the Python model packages; it does not bundle external datasets, checkpoints, research reports or all source-tree launch profiles. Configure each case's ignored dataset-location map from its checked-in example. A dry-run must not start an optimizer or write a run. Training and dataset generation always require separately selected data, budget and output locations.

## Existing checkpoints and resume

Package names, configuration keys, serialized architecture IDs, tensor names, normalization and physical laws are preserved. The directory lift changes source/document locations rather than checkpoint contents. Use the matching case loader for each saved model; a public family name alone does not load an adapter. Existing absolute external checkpoint/data paths remain external and are not rewritten. `project://` paths now resolve relative to this repository root, independent of the shell's current directory.

Resume preserves case membership, input/normalization identity, optimizer/scaler, RNG, sampler, selection, schedule and complete history. Latest, best-field and literal epoch checkpoints remain distinct. A compatible load is not a completed continuation. Existing S/X long-running jobs retain their original source checkout and receive no new observer or restart from this publication.

The published contract suite covers all six families, strict original six-hyperedge/dense next-update resume, S/X and U-H task handoffs, case adapters and field/organization visualization. A single exact historical SHA mismatch is marked expected only for the already observed digest; any new digest mismatch still fails. CUDA/external-data checks may skip when their required resources are absent. Historical campaign-specific suites remain local with their source branch rather than presenting retired experimental workflows as current tests.

## Documentation figures and local artifacts

The [README asset index](assets/readme/README.md) lists the eight user-approved PNG/PDF figure pairs, totaling about 3.4 MB. They are versioned only under `docs/assets/readme/`, use relative links, preserve aspect ratio and carry source/checkpoint/error interpretation. Update that same directory for future README revisions. Numerical arrays, weights, full histories, generated reports and one-time scripts remain excluded. The artifact hook recognizes the root and historical nested layouts and allows only this narrow curated-figure exception.

On the maintained workstation, `diagnostics` and `Trained_Results` remain symlinks to the existing DATA2 roots whose physical paths still contain the historical `HONF_Proj` name. The Wind native catalogue uses its existing DATA2 target. The directory lift does not move, duplicate or rewrite those physical payloads. Other hosts should use their own explicit external-resource maps and output settings rather than copying workstation paths.

## Publication validation

The root-layout regression suite passed 587 tests in the independent installed environment, with one CUDA-dependent skip and one exact known historical digest marked expected. All six retained real checkpoint families loaded strictly on CPU through the lifted implementation: Hypergraph-routed epoch 4890, Dense pairwise epoch 4738, Content-attention and Unified hypergraph epoch 5000, frozen Relation-anchored epoch 2813 and completed Active-domain relation-anchored TRAIN150 epoch 999. These are load checks, not newly measured physical accuracy or completed long-run claims. Legacy six-hyperedge/dense exact next-update resume and S/X/U-H task handoffs are covered by the contract suite.

An independent DATA2 environment validated editable core/case installations, seven command inspections including a real read-only managed dry-run, and four wheel builds/imports from outside the checkout. The predictor implementation files remain byte-identical to the final test branch. The only existing Python changes are a project-root guard, a path-description docstring and the precise historical-digest expected-failure handling; the root-layout test is new. No dataset or checkpoint was migrated, no new training or physical-validation campaign ran, and existing long continuations were not observed.
