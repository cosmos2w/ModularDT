# ChannelThermal NewHONF

`src_new/` is the standalone ChannelThermal NewHONF stack. It keeps the CORE
HONF reusable, copies the Stage-A local surrogate stack for checkpoint
compatibility, and adds ChannelThermal-specific physical coupling around the
new H-routed decoder. It does not import the old `src/` tree or
`Unified_Forward_Model_0` at runtime.

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

The export intentionally excludes dense learned `hyper_state`,
query-dependent `alpha_qk`, and module tokens. `alpha_qk` is recomputed by the
forward decoder for each query grid, and module tokens are recomputed from the
generated physical design.
