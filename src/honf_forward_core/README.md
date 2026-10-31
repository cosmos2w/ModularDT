# HONF forward core

The forward core owns reusable interaction, organization and query prediction. Case adapters own physical record schemas, units, geometry, controls, normalization, target extraction, physical laws and plotting. Core modules must not import a case package. See the [project README](../../README.md) for the maintained six families and their equations, and [case integration](../../docs/case_plugin.md) for adding data and physics.

Use [Model_Explain.md](../../Model_Explain.md) for the implementation-bound core formulation and the canonical mathematical notation. It separates global conditions, contextual coefficients, prepared access weights and final query contributions, and maps each quantity to this implementation.

## Public model selection

[`models.py`](models.py) exposes immutable `MODEL_FAMILIES`, `model_family(name)`, `build_model(name, config)` and `relation_anchored_config(name, base_model_config, ...)`. Public families are `hypergraph_routed`, `dense_pairwise`, `content_attention`, `unified_hypergraph`, `relation_anchored` and `relation_anchored_active`. For a fresh anchored model, pass the content core configuration to `relation_anchored_config` and its result to `build_model`; the optional arguments set the input dataset, query tile size, retained mass and initial EE width. Development aliases remain accepted at this boundary. Construction from a saved configuration preserves its schema, architecture and parameter inventory. Loading a trained case still requires that case's loader, saved normalizer and matching adapter.

`query_tile_size` bounds decoder chunks along each scene's query dimension while evaluating all queries. Larger values trade more live memory for fewer small operations and launches; benchmark full forward/backward execution on your own inputs. See [query tiling](../../docs/guides/Query_Tiling.md) for selection and resume guidance.

The original six-slot and dense cores retain established main-branch interfaces. `HONFNeuralField` accepts `BatchData`; `InterfaceFieldCore` retains `encode_case(batch)`, `prepare(encoded, encoded.module_tokens)` and `decode_queries(prepared, query_xy, query_features)`. `ContentHypergraphCoreV2` uses its own `SceneBatch` and `QueryBatch` contracts. Unified and anchored cores preserve their original scene/query interfaces. The [electro-thermal handoff](../../Case_ElectroThermal/docs/COWORKER_HANDOFF.md) demonstrates six-slot, dense and content-attention adapters without Thermal's local-surrogate dependency.

## Compatibility and extension

The legacy batch fields include `module_centers`, `module_present`, `module_features`, `global_context`, `env_coords`, `env_features`, `query_xy`, optional `query_features`/`query_time`, optional `target_field`, and JSON-safe metadata. `query_xy` is an established name for query coordinates. Masks and original identities remain explicit. Contemporary case adapters may use a family-specific scene contract instead.

Reusable blocks in `nn.py` preserve historical MLP dropout indices and Fourier feature order. Neutral `coordinate_scale` and `periodic_axes` settings describe coordinates; historical domain and geometry aliases remain loadable. Existing physical heads and local coupling are reconstructed faithfully for their checkpoints, without making Thermal-specific output laws generic defaults.

Fresh relation-anchored and active-domain relation-anchored models share the detail-preserving anchored implementation. A case adapter supplies continuous E-cover indices and weights; selected original M/E members provide query-relative value messages, and the access-measure policies distinguish the two families. Applied contextual relations, prepared directory support and query contributions have separate provenance. Historical anchored configurations dispatch to their original implementation, while existing detail configurations retain their original schema and state names. Intermediate research variants are excluded from maintained public selection, while required shared implementation and historical configuration types remain internal dependencies.
