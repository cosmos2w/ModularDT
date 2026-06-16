# Unified Forward Model 0

This sandbox progressively rebuilds the forward model to identify which parts
of the hypergraph-organized neural field (HONF) are core, case-specific, or
redundant. It loads data from the existing demo datasets when available, but
all outputs, diagnostics, and ablation records are written under
`Unified_Forward_Model_0/`.

## Core Notation

- `D`: design/layout
- `c`: context/operating condition
- `H`: learned module-environment hypergraph organization
- `U`: full predicted field/state

## Target Core Architecture

```text
D, c
  -> module tokens
  -> environment tokens
  -> hypergraph organizer H
  -> hypergraph-centric neural field decoder
  -> U
```

## Initial Ablation Ladder

- `A0 hyper_only`
- `A1 hyper_plus_global`
- `A2 hyper_plus_direct_residual`
- `A3 hyper_plus_near_module`
- `A4 current_like`

## Case-Specific Patches

MultiCylinder:

- periodic geometry
- phase `tau`
- dynamic hyper tokens
- mean/residual split

ChannelThermal:

- nonperiodic geometry
- local surrogate
- port prediction
- interface/internal heads

## Strong Rule

Existing demo folders are not modified. This sandbox is used to test
simplifications before back-porting.

## External Naive Baselines

The naive ChannelThermal baselines test whether simple MLP-style neural fields
can learn the layout-to-field map without learned organization `H`.

- `NB0_flat_layout_mlp`: a slot-order-dependent lower baseline that flattens
  module layout/features and concatenates them with each query point.
- `NB1_query_deepsets_mlp`: a permutation-invariant query-conditioned DeepSets
  baseline that pools simple module-relative embeddings per query.

Neither baseline uses environment tokens, `A_mh`, `A_eh`, `A_me`, hyperedge
source/region coordinates, local surrogates, port/interface heads, or HONF
decoder internals.

## Mechanism-Aware Sparse HONF

The enhanced HONF path treats each hyperedge `H_k` as a generic field
mechanism rather than only a pooled token. Each hyperedge carries a source
coordinate, a field-region coordinate, source-to-region displacement,
mass/strength features, and a learned hyperedge state. These mechanism
descriptors are generic source-region geometry and mass summaries, not
case-specific thermal rules.

For each query point `q`, the decoder can use dense attention over all
hyperedges or top-k sparse query-to-H attention. `hyper_attention_topk=0`
preserves the dense softmax behavior; `hyper_attention_topk=2` is the
recommended sparse interpretable setting for the mechanism-aware comparison.

In `enhanced_honf_pairwise` mode, query-module pairwise geometry remains routed
through `H`: module pairs are first aggregated by `A_mh`, then selected by
query-to-H attention. This differs from the strong query-pair baseline, which
directly pools all query-module pairs without using `H`.

Recommended final comparison:

- `H_clean_newH_sparseK2_capacity`
- `H_enhanced_newH_sparseK2_pairwise_capacity`
- `NH2_no_hyper_current_like_direct005_capacity`
- `NB_query_pair_deepsets_full_capacity`

Example:

```bash
python src/ablation_runner.py \
  --config configs/unified_forward_channelthermal_template.json \
  --plan configs/ablation_plan_honf_mechanism_enhanced.json \
  --case channelthermal
```
