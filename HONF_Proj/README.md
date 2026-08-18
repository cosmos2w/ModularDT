# HONF Project

`HONF_Proj` is the installable implementation of the Hypergraph Organized Neural Field (HONF), a neural operator for continuous field prediction around a variable-size set of interacting modules. The repository contains a reusable forward core, a hierarchical inverse core, a strict configuration/runtime layer, and the complete ThermalChannel case with a reusable Stage-A thermal-disk surrogate.

The checkpoint-compatible default forward profile is `enhanced_honf_pairwise`; the opt-in upgraded profile is `adaptive_sparse_additive`. The default preserves fixed-projection organization, context-fusion decoding, softmax routing, and dense execution for old saved configurations and checkpoints. The upgraded profile uses exchangeable candidate edges, descriptor-first mechanism states, entmax15 routing with a bounded Gaussian environment/query locality bias, gathered pre-MLP execution, and an exact background-plus-edge additive field.

## 1. Quick default forward run with the local Stage-A checkpoint

Run commands from `HONF_Proj` in the `ModularDT` environment. This dry-run validates the default profile, dataset, GPU 0, output location, and the exact local checkpoint without creating a run:

```bash
conda activate ModularDT
python train.py \
  --config src/config_core/forward/enhanced_honf_pairwise.json \
  --local-checkpoint Trained_Results/ThermalChannel/Local_Module_Runs/thermal_disk/Run_0000_base/latest_model.pt \
  --run-id 9100 \
  --device cuda:0 \
  --dry-run
```

Use an unused numeric run ID. The verified local checkout already contains run ID `0002`, so the profile’s built-in ID cannot be reused for a new launch. To perform a fast one-batch correctness smoke, run:

```bash
python train.py \
  --config src/config_core/forward/enhanced_honf_pairwise.json \
  --local-checkpoint Trained_Results/ThermalChannel/Local_Module_Runs/thermal_disk/Run_0000_base/latest_model.pt \
  --run-id 9100 \
  --epochs 1 \
  --max-train-batches 1 \
  --max-val-batches 1 \
  --device cuda:0 \
  --yes
```

This smoke checks end-to-end construction, strict Stage-A loading, one training batch, one validation batch, checkpoint writing, and plotting hooks; it is not meaningful model training. For a normal default run, keep the same command but remove `--epochs 1`, `--max-train-batches 1`, and `--max-val-batches 1`. Review the dry-run before adding `--yes` because the full profile requests 10,000 epochs.

The `--local-checkpoint` argument takes precedence over the `best_model.pt` path stored in `Case_ThermalChannel/configs/case_default.json`. The requested `latest_model.pt` is a trusted local epoch-6357 Stage-A artifact with dimensions 7 input parameters, 5 port-token values, 2 interface outputs, hidden/latent width 128, 16 port latents, 4 heads, 4 attention layers, 6 coordinate Fourier frequencies, and dropout 0.0. Stage B loads its normalizers, copies the model, and freezes it.

## 2. Architecture in one view

```text
physical design + operating context + global queries
                    │
                    ├── ThermalChannel input adapter: module/global features
                    ├── ThermalChannel environment builder: 24 x 8 tokens
                    ▼
              reusable HONF encoders
                    ▼
           organizer and predicted ports
                    ▼
       frozen Stage-A thermal-disk surrogate
                    ▼
    local response fusion + one port refinement
                    ▼
              final organizer
                    ▼
      query routing + continuous field decoder
                    ▼
 [u, v, p, omega, T] + internal T + interface [T, qn]
```

The reusable core never imports ThermalChannel code. The case package owns physical features, datasets, Stage-A coupling, Robin flux correction, field names, losses, evaluation views, and inverse functional definitions. The top-level launchers discover that package through `channelthermal.plugin:create_plugin`.

## 3. Forward profiles and concrete settings

Both maintained profiles use hidden width 256, dropout 0.0, LayerNorm, a (24\times8) environment grid, four Fourier frequencies, six nominal mechanism edges, the enhanced hypergraph-plus-pairwise decoder family, predicted local ports, frozen Stage A, and one local/global refinement pass.

| Setting | `enhanced_honf_pairwise.json` | `adaptive_sparse_additive.json` |
|---|---|---|
| Purpose | default and checkpoint-compatible | upgraded adaptive sparse architecture |
| Organizer | `fixed_projection` | `exchangeable_slots` |
| Edge extent | 6 fixed projections | runtime candidate capacity 8 |
| Active selection | all six | all viable candidates during warmup; then minimum 1 with quality/coverage/novelty (initial reference 6) |
| Selection thresholds | n.a. | warmup 200 epochs, coverage 0.95, token mass 0.50, maximum redundancy 0.85 |
| Candidate viability | n.a. | module and environment normalized mass fractions each above 0.01 |
| Module/environment/query normalizer | softmax | entmax15 with alpha 1.5 |
| Locality | none | bounded Gaussian environment/query bias, strength 1.0, radius cap 3.0 |
| Mechanism state | `residual_concat`, mechanism encoder off | `descriptor_first`, content residual scale 0.35 |
| Field assembly | `context_fusion` | exact `edge_additive` |
| Execution | dense | gathered before pair and edge MLPs |
| Query limits | unlimited | at most 8 modules and 3 edges per query |
| Topology signature flag | false | true |
| Default run ID | 0002 | 0003 |

The upgraded organizer uses shared parameters and deterministic sinusoidal candidate codes; it has no learned edge-index embedding. Its runtime edge capacity can change without changing parameter shapes. Selection progress is serialized, nonviable candidates are excluded, and selected token assignments use detached one-hot fallback before row renormalization when sparse support would otherwise vanish. Entmax15 provides exact zero routes, while `routing_execution="gathered"` renormalizes retained module incidence and query-edge probability before avoiding expensive MLPs on unselected pairs. Dense multiplication by zero is not counted as computational sparsity.

The upgraded field is exactly

$$\widehat U(q)=U_{background}(q,g,E)+\sum_{k=1}^{K_{cap}}a_k\alpha_{qk}U_k(q,t_k,c^{pair}_{qk}).$$

The background sees query, global, and environment context but no module memory. Optional `pred_field_background` and `pred_field_by_edge` outputs close exactly to `pred_field`. The shipped case profile has organizer regularization disabled and therefore has no default edge-count penalty.

See [Model_Explain.md](Model_Explain.md) for the complete equations, Stage-A coupling, topology signature, inverse flow, and code-to-math map.

## 4. Backward compatibility

`UnifiedForwardConfig` gives saved configurations without upgraded fields the strict mode defaults `organizer_mode="fixed_projection"`, `mechanism_state_mode="residual_concat"`, `field_assembly_mode="context_fusion"`, softmax module/environment/query assignment, and `routing_execution="dense"`. Only modules required by the selected modes are instantiated, so old state-dict paths remain valid and old checkpoints do not acquire unexpected upgraded parameters.

Historical forward checkpoints that contain `core_honf.max_num_modules` still load. That value is treated as migration metadata, not a runtime capacity, and their fourteen-value `legacy_v1` global feature transform uses a fixed saved denominator. Current `padding_invariant_v2` checkpoints use eighteen global features that do not depend on batch padding width.

Stage-A/local coupling remains in `Case_ThermalChannel`; neither forward profile changes its ownership or execution order. Existing self-contained Stage-B checkpoints embed the Stage-A state and local/global normalizers, so later Stage-B evaluation does not require the original external Stage-A file.

## 5. Installation and data

Python 3.10 or newer is required. Install PyTorch for the local CUDA platform, then install the core and ThermalChannel package:

```bash
cd HONF_Proj
conda activate ModularDT
python -m pip install -e . -e ./Case_ThermalChannel
```

Install test dependencies when developing:

```bash
python -m pip install -e '.[dev]'
```

Datasets are external. Copy the location template and edit both paths:

```bash
cp Case_ThermalChannel/Dataset/dataset_locations.example.json Case_ThermalChannel/Dataset/dataset_locations.local.json
```

| Dataset ID | Purpose | Current manifest split |
|---|---|---|
| `thermal_disk_local_v1` | isolated and globally aligned Stage-A disk responses | 919 train, 115 test |
| `thermal_channel_global_v1` | coupled channel fields and local responses | 600 train, 90 test |

The local location map is ignored by Git. `HONF_DATA_ROOT` can instead point to a directory that contains `Processed_LocalModule_Dataset/` and `Processed_ChannelThermal_Dataset/`.

Validate the packed files and manifest hashes with:

```bash
python tools/inspect_dataset.py --dataset-id thermal_disk_local_v1 --sha256
python tools/inspect_dataset.py --dataset-id thermal_channel_global_v1 --sha256
```

The ThermalChannel data schema and physical definitions are in [PHYSICS_AND_DATA.md](Case_ThermalChannel/Dataset/PHYSICS_AND_DATA.md). Generated PyTorch checkpoints are pickle-based trusted artifacts; do not load an untrusted `.pt` file.

## 6. Configuration composition

A launch composes one core profile under `src/config_core`, the referenced case profile, an optional strict experiment overlay, and allow-listed CLI overrides. Core profiles own generic architecture, optimization, checkpoint policy, and run identity; the ThermalChannel profile owns data, feature semantics, Stage-A dependency, local coupling, physical corrections, and losses.

`project://...` paths resolve from `HONF_Proj`, while `config://...` paths resolve from the file that contains them. Unknown fields and ownership violations fail before training. Managed runs save source configurations, CLI overrides, the deterministic resolved configuration, hashes, software information, and Git state.

The default data/training values are 1024 sampled global points per case, train/validation batch size 48, four workers, dynamic module padding, module-count bucketing, normalized inputs and targets, learning rate (3\times10^{-4}), weight decay (10^{-5}), gradient clipping at 1.0, seed 0, AMP disabled, and 10,000 nominal epochs.

## 7. Train the upgraded forward profile

Validate an upgraded launch with the same Stage-A dependency:

```bash
python train.py \
  --config src/config_core/forward/adaptive_sparse_additive.json \
  --local-checkpoint Trained_Results/ThermalChannel/Local_Module_Runs/thermal_disk/Run_0000_base/latest_model.pt \
  --run-id 9101 \
  --device cuda:0 \
  --dry-run
```

Add `--yes` only when the validated destination and full training budget are intended. A one-batch smoke may use the same three bounding options shown in Section 1.

## 8. Evaluate forward checkpoints and export topology

Training normally writes the following selectors:

| Selector | File | Criterion |
|---|---|---|
| `best` | `best_model.pt` | total validation objective |
| `best_by_field_mse` | `best_by_field_mse_model.pt` | global field MSE |
| `best_by_temperature_mse` | `best_by_temperature_mse_model.pt` | temperature MSE |
| `best_predicted` | `best_predicted_model.pt` | autonomous predicted-port validation |
| `latest` | `latest_model.pt` | latest resumable optimizer/model state |

`best_predicted` is the normal autonomous deployment checkpoint and the required basis for inverse dataset provenance. Evaluate a managed default run on GPU 0 with:

```bash
python evaluate.py \
  --config src/config_core/forward/enhanced_honf_pairwise.json \
  --workflow forward \
  --run-id 0002 \
  --checkpoint best_predicted \
  --device cuda:0 \
  --case-index 0 \
  --organization-view all \
  --routing-view summary
```

Evaluate an explicit checkpoint by replacing `--run-id` and the selector with `--checkpoint /absolute/path/to/checkpoint.pt`. Add `--export-topology-signature` to request dense evaluation diagnostics needed for the schema-v3 unordered topology signature and per-edge field closure:

```bash
python evaluate.py \
  --config src/config_core/forward/adaptive_sparse_additive.json \
  --workflow forward \
  --checkpoint /absolute/path/to/best_predicted_model.pt \
  --device cuda:0 \
  --case-index 0 \
  --organization-view all \
  --routing-view summary \
  --export-topology-signature
```

The evaluation directory then contains `topology_signature.npz`, `topology_signature_summary.json`, structure diagnostics, and ThermalChannel topology views. The generic schema is `honf_topology_signature` version 3; edge order is canonicalized only for deterministic serialization, while comparisons use Hungarian active-set matching plus relation error.

## 9. Resume and run storage

Resume a managed run in place with its exact profile and latest checkpoint:

```bash
python train.py \
  --config src/config_core/forward/enhanced_honf_pairwise.json \
  --resume-checkpoint Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_####_<timestamp>_<name>/latest_model.pt \
  --device cuda:0 \
  --yes
```

Resume validates case, workflow, model family, immutable model/data/loss sections, feature schemas, dataset identity, and normalization. It restores model, optimizer, AMP scaler, epoch, best metrics, and Python, NumPy, Torch, and CUDA random states. `--local-checkpoint` starts a new Stage-B run with a selected Stage-A dependency; it is not a resume option.

Managed output families are:

```text
Trained_Results/ThermalChannel/
├── Local_Module_Runs/thermal_disk/
├── HONF_Forward_Runs/
├── Inverse_Dataset_Builds/
└── HONF_Inverse_Runs/
```

Run IDs are unique within a family and are never overwritten silently.

## 10. Unordered topology and inverse design

The accepted forward topology schema records active masks, edge descriptors, module/environment incidences, edge relations, reference-query routing, per-field contributions, query-grid provenance, field names, case ID, and forward checkpoint SHA-256. `src/honf_forward_core/evaluation/topology_signature.py` provides validation, canonical serialization, Hungarian comparison, module-affinity reconstruction, query-to-module influence reconstruction, and compact summaries.

The opt-in inverse profile `src/config_core/inverse/train_inverse_topology_set_template.json` consumes schema-v3 topology sets. Its plan flow uses shared token projections and permutation-equivariant set self-attention with no learned edge-index embedding, supports runtime topology capacity, and trains with Sinkhorn set matching. Its layout flow uses set cross-attention from physical module slots to active plan tokens. It requires the exact SHA-256 of the forward checkpoint that created the topology dataset.

The earlier indexed compact-plan flow and ordered-flat layout conditioner remain available for compatible inverse artifacts. Mode-specific modules are instantiated only for the chosen inverse profile, and the fixed-width corrector is intentionally unavailable in exchangeable-set mode.

Inverse design remains a bounded research workflow. Generated candidates are ranked only after evaluation by a frozen autonomous forward checkpoint; neither forward nor inverse predictions replace a high-fidelity solver or engineering validation.

## 11. Testing

Run the focused forward-upgrade tests on GPU 0 when CUDA is required:

```bash
CUDA_VISIBLE_DEVICES=0 pytest -q \
  tests/test_forward_upgrade_config.py \
  tests/test_forward_additive.py \
  tests/test_exchangeable_organizer.py \
  tests/test_sparse_routing.py \
  tests/test_gathered_routing.py \
  tests/test_topology_signature.py \
  Case_ThermalChannel/tests/test_inverse_topology_set.py \
  Case_ThermalChannel/tests/test_topology_signature_visualization.py
```

Run the complete feasible suite with:

```bash
CUDA_VISIBLE_DEVICES=0 pytest -q
```

The forward upgrade plan, acceptance gates, commands, and recorded results are maintained in `HONF_Forward_Upgrade_Codex_Goal_Plan.md`.

## 12. Repository layout

```text
HONF_Proj/
├── train.py                              generic forward/local training dispatcher
├── evaluate.py                           generic forward/local/compare dispatcher
├── src/
│   ├── honf_forward_core/                reusable encoders, organizers, routing, decoder, evaluation
│   ├── honf_inverse_core/                request encoder, indexed/set flows, matching, sampling
│   ├── honf_runtime/                     strict config composition, plugins, paths, checkpoints, runs
│   └── config_core/                      forward and inverse profiles plus schemas
├── Case_ThermalChannel/
│   ├── src/channelthermal/               case adapter, Stage A, coupling, losses, workflows, plots
│   ├── configs/case_default.json         physical data, coupling, loss, and evaluation settings
│   ├── Dataset/                          manifests, schemas, local path map
│   └── scripts/inverse/                  ThermalChannel inverse launchers
├── Trained_Results/                      generated local, forward, and inverse artifacts
├── docs/                                 configuration, extension, checkpoint, and result contracts
└── tests/                                reusable and case integration tests
```

## 13. Extension boundary

To add another physical case, implement the plugin protocol, physical-to-generic input adapter, environment/query features, dataset loaders, named field/loss policy, and evaluation hooks. Reuse `honf_forward_core` without importing the new case into it. Add a case-owned local surrogate and coupling only if that physics needs one, and add inverse functionals only after the forward checkpoint and topology schema are stable.

Further contracts are in `docs/CASE_EXTENSION.md`, `docs/CONFIG_CONTRACT.md`, `docs/CHECKPOINT_CONTRACT.md`, and `docs/RESULT_CONTRACT.md`.
