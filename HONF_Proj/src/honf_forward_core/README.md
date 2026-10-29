# HONF forward core

The forward core owns reusable interaction, organization and query prediction. Case adapters own physical record schemas, units, geometry, controls, normalization, target extraction, physical laws and plotting. Core modules must not import a case package. See the [project README](../../README.md) for the maintained six families and their equations, and [case integration](../../docs/case_plugin.md) for adding data and physics.

## Public model selection

[`models.py`](models.py) exposes immutable `MODEL_FAMILIES`, `model_family(name)` and `build_model(name, config)`. Public families are `hypergraph_routed`, `dense_pairwise`, `content_attention`, `unified_hypergraph`, `relation_anchored` and `relation_anchored_active`. Development aliases remain accepted at this boundary; serialized architecture IDs, configuration keys and state-dictionary names are preserved. Construction takes the selected core's original strict configuration and initializes a bare core. Loading a trained case still requires that case's loader, saved normalizer and matching adapter.

The original six-slot and dense cores retain established main-branch interfaces. `HONFNeuralField` accepts `BatchData`; `InterfaceFieldCore` retains `encode_case(batch)`, `prepare(encoded, encoded.module_tokens)` and `decode_queries(prepared, query_xy, query_features)`. `ContentHypergraphCoreV2` uses its own `SceneBatch` and `QueryBatch` contracts. Unified and anchored cores preserve their original scene/query interfaces. The [electro-thermal handoff](../../Case_ElectroThermal/docs/COWORKER_HANDOFF.md) demonstrates six-slot, dense and content-attention adapters without Thermal's local-surrogate dependency.

## Compatibility and extension

The legacy batch fields include `module_centers`, `module_present`, `module_features`, `global_context`, `env_coords`, `env_features`, `query_xy`, optional `query_features`/`query_time`, optional `target_field`, and JSON-safe metadata. `query_xy` is an established name for query coordinates. Masks and original identities remain explicit. Contemporary case adapters may use a family-specific scene contract instead.

Reusable blocks in `nn.py` preserve historical MLP dropout indices and Fourier feature order. Neutral `coordinate_scale` and `periodic_axes` settings describe coordinates; historical domain and geometry aliases remain loadable. Existing physical heads and local coupling are reconstructed faithfully for their checkpoints, without making Thermal-specific output laws generic defaults.

Relation-anchored S/X directly share the anchored core and reusable member-routing primitives. Their access-measure policies remain distinct. Applied contextual relations, prepared directory support and query contributions have separate provenance; organization-controlled reads preserve original members. Intermediate research variants are excluded from maintained public selection, while required shared implementation and historical configuration types remain internal dependencies. The historical milestone branch preserves the complete predecessor source.
