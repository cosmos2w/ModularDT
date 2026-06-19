# ChannelThermal NewHONF

`src_new/` is the standalone ChannelThermal NewHONF stack. It keeps the CORE
HONF reusable, copies the Stage-A local surrogate stack for checkpoint
compatibility, and adds ChannelThermal-specific physical coupling around the
new H-routed decoder. It does not import the old `src/` tree or
`Unified_Forward_Model_0` at runtime.

The core is split into `encode_and_organize()` and `decode_queries()`. The
ChannelThermal wrapper uses the base encode/organize pass, fuses any local
response into module tokens, recomputes the final organizer, and decodes the
environmental field only once for standard forwards. Auxiliary port-global
temperature probes are requested only when their loss or diagnostics need them.

Main field equation:

```text
U(q) = f_out[
  c_H(q)
  + g_pair c_pair(q)
  + c_global(q)
  + c_near(q)
]
```

## CORE Reusable HONF

`_models_core/` contains generic module and environment encoders, the
hypergraph organizer (`A_me`, `A_mh`, `A_eh`), mechanism descriptors, optional
hyperedge value context, learned query-to-H routing, and the H-routed pairwise
field decoder. It only consumes generic tensors.

## CHANNELTHERMAL-SPECIFIC

`_models_channelthermal/` maps Re, inlet velocity, module centers, heat powers,
module masks, and materials into generic HONF inputs. Wall, inlet, outlet, and
centerline environment features are constructed outside the core. The
ChannelThermal wrapper attaches the copied local surrogate, predicts/uses port
conditions, applies corrected-physics interface flux, and can run one
interaction-refinement pass before the final HONF field decode.

## Commands

Compile:

```bash
conda run -n ModularDT python -m py_compile src_new/train.py src_new/evaluate.py
```

Smoke and compatibility utilities live under `src_new/_tests/` so the top
level only exposes the main train/evaluate entrypoints:

```bash
conda run -n ModularDT python src_new/_tests/test_local_checkpoint_compat.py --device cpu

conda run -n ModularDT python src_new/_tests/smoke_global_modes.py \
  --config Configs_new/train_global_honf_template.json \
  --device cpu \
  --points 32

conda run -n ModularDT python src_new/_tests/test_newhonf_hardening.py \
  --config Configs_new/train_global_honf_template.json \
  --device cpu \
  --points 32

conda run -n ModularDT python src_new/_tests/test_hypergraph_plan_stability.py
```

CPU smoke train:

```bash
conda run -n ModularDT python src_new/train.py \
  --config Configs_new/train_global_honf_template.json \
  --device cpu \
  --max-train-batches 1 \
  --max-val-batches 1 \
  --epochs 1
```

Evaluate one case:

```bash
conda run -n ModularDT python src_new/evaluate.py \
  --Run_ID 0001 \
  --checkpoint best_predicted \
  --saved-root ./Saved_Model_NewHONF \
  --case-index 0 \
  --device cpu \
  --local-port-condition-mode predicted \
  --temperature-display-mode composite_internal \
  --organization-view all
```

Evaluate with routing maps and inverse-ready hypergraph export:

```bash
conda run -n ModularDT python src_new/evaluate.py \
  --Run_ID 0001 \
  --checkpoint best_predicted \
  --saved-root ./Saved_Model_NewHONF \
  --case-index 0 \
  --device cpu \
  --query-batch-size 4096 \
  --local-port-condition-mode predicted \
  --temperature-display-mode composite_internal \
  --organization-view all \
  --return-routing-maps \
  --routing-view all \
  --export-hypergraph-plan
```

## Organization Versus Routing

Static organization is the case-level hypergraph produced before query
decoding:

- `A_mh`: soft module-to-hyperedge assignment.
- `A_eh`: soft environment-token-to-hyperedge assignment.
- `hyper_source_coords`, `hyper_region_coords`: readable source and thermal
  region coordinates.
- `hyper_module_mass`, `hyper_env_mass`, `hyper_strength`: compact edge
  summaries.

Query-dependent routing is recomputed for every query point during decoding:

- `alpha_qk` / `query_hyper_attention`: query-to-hyperedge attention.
- `c_H`: directly readable hyperedge value context.
- `c_pair`: H-routed query-module pairwise detail.
- `pairwise_edge_contribution`: `||g_pair alpha_qk edge_pair_context_qk||`.

Normal training and evaluation do not retain dense `[Q,K]` routing arrays.
They are emitted only with `--return-routing-maps`.

## Compact Plan Export

`--export-hypergraph-plan` writes `hypergraph_plan.npz` and
`hypergraph_plan_summary.json`. The plan stores static organization variables
for inverse-design seeding: `A_mh`, `A_eh`, source/region coordinates, masses,
strengths, and the active hyperedge mask.

The export is canonicalized with active hyperedges first, then source/region
coordinates and strength. `edge_permutation` records the original H-index
provenance. Use `_helpers.hypergraph_plan.load_hypergraph_plan()` and
`validate_hypergraph_plan()` to round-trip and validate saved plans. The
model wrapper also exposes `ChannelThermalHONFModel.extract_hypergraph_plan()`
for inverse code that already has `organizer_aux`.

The export intentionally excludes dense learned `hyper_state`,
query-dependent `alpha_qk`, and module tokens. `alpha_qk` is recomputed by the
forward decoder for each query grid, and module tokens are recomputed from the
generated physical design.

Every evaluation writes `hypergraph_diagnostics.json`. It summarizes static
organization health, optional routing summaries, and base-versus-final H
changes after local-response fusion (`A_mh`, `A_eh`, source/region shifts, and
mass/strength shifts). When `--local-port-condition-mode both` is used, teacher
outputs are saved with a `_teacher` suffix, but organization plots, routing
maps, and hypergraph-plan exports use predicted mode as the primary inverse
source.

## Training Diagnostics

Training logs scalar HONF diagnostics in `metrics.csv` for both train and
validation:

- active and soft-active edge counts.
- `A_mh`/`A_eh` entropy and mass concentration.
- query-attention entropy, effective edge count, and max attention.
- pairwise gate/context norm, `c_H` value-context norm, total hyper context,
  and non-hyper context norm.
- flags for whether hyper value context and pairwise kernel are active.

Dense `[Q,K]` routing maps are not retained during training. The old ambiguous
`organizer_entropy_weight` path is deprecated; use
`loss.organizer_regularization` for optional generic anti-collapse experiments.
All generic regularization effects are disabled by default.

## Config And Checkpoints

`Configs_new/train_global_honf_template.json` uses authoritative nested
sections:

- `model.core_honf`: reusable HONF settings.
- `model.channelthermal`: adapter dimensions, internal prediction mode, and
  fallback-head dimensions.
- `model.local_coupling`: local surrogate enable/path/freeze/latent settings.
- `model.physical_correction`: flux mode, refinement, and port-global probe
  settings.

Legacy duplicate keys are parsed with warnings, but the nested
`local_coupling` and `physical_correction` sections win. Training writes a
`config_resolved.json` with concrete dataset-derived dimensions and no
`"auto"` model fields.

Global checkpoints are self-contained for evaluation: they store the local
surrogate config, local normalization stats, frozen flag, and original external
checkpoint path as provenance. The local model state itself is included in the
global `model_state_dict`.

Additional profiles:

- `Configs_new/train_global_honf_validated_core.json` pins the enhanced HONF
  sandbox settings from `Run_0009_20260617_154601` and adds the full
  ChannelThermal coupling stack.
- `Configs_new/train_global_honf_old_parity.json` matches old comparison data
  normalization, split, point count, seed, optimizer scale, local checkpoint,
  and Smooth L1 interface/log-h losses where practical.
