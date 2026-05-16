# Design Prior Configuration Notes

The behavior-aware design prior is a target-agnostic latent atlas:

```text
z, context -> layout + planned hypergraph + behavior descriptor
```

It is trained separately from the KPI-conditioned inverse generator. Field
functional design tasks are solved later by searching over `z` with the frozen
forward HONF verifier.

## Main Training Hyperparameters

- `latent_dim`: compact atlas dimension. Default is `32`.
- `hidden_dim`: MLP width for encoder/decoder networks.
- `kl_weight`: VAE regularization weight. Default is `1e-3`; keep this small
  early to avoid posterior collapse.
- `behavior_recon_weight`: reconstruction weight for compact behavior
  descriptors.
- `hypergraph_recon_weight`: reconstruction weight for planned/realized
  hypergraph vectors.
- `geometry_weight`: differentiable validity pressure for overlaps, channel
  boundaries, and inactive heat slots.

Suggested sweeps:

| Parameter | Useful range |
| --- | --- |
| `latent_dim` | `16`, `32`, `64` |
| `hidden_dim` | `128`, `256`, `512` |
| `kl_weight` | `1e-4` to `1e-2` |
| `behavior_recon_weight` | `0.25` to `2.0` |
| `hypergraph_recon_weight` | `0.1` to `1.5` |
| `geometry_weight` | `0.01` to `0.2` |

## Guided Search Hyperparameters

- `latent_cem_iterations`: number of latent CEM updates.
- `latent_cem_population`: candidates per CEM iteration.
- `latent_cem_elite_frac`: fraction of candidates used to update the proposal.
- `prior_energy_weight`: penalty for moving far from the learned prior.
- `hypergraph_consistency_weight`: penalty for planned-vs-realized organization
  mismatch.
- `geometry_penalty_weight`: penalty for repaired or invalid decoded layouts.

Suggested sweeps:

| Parameter | Useful range |
| --- | --- |
| `latent_cem_population` | `64` to `512` |
| `latent_cem_iterations` | `5` to `20` |
| `latent_cem_elite_frac` | `0.10` to `0.25` |
| `prior_energy_weight` | `0.001` to `0.1` |
| `hypergraph_consistency_weight` | `0.0` to `5.0` |
| `geometry_penalty_weight` | `0.5` to `5.0` |

## Raw Layout CEM vs Latent Guided Search

Use raw layout CEM when you need a direct optimizer baseline. It searches
normalized module positions and active masks without using the learned atlas.
This is useful for measuring whether the latent prior genuinely improves
sample efficiency or feasibility.

Use latent guided search when the design prior has learned a meaningful
behavior-aware manifold. The guided search optimizes in compact latent space,
then verifies every decoded candidate with the same frozen forward HONF. This
is the intended generative inverse-design method.

Use unguided prior sampling only as a prior-quality baseline. It should produce
plausible layouts, but it is not expected to satisfy arbitrary downstream
field-functional objectives by itself.
