# HONF forward core

`honf_forward_core.organizer.HypergraphOrganizerCore` and
`honf_forward_core.decoder.HypergraphFieldDecoder` remain the stable public
facades. The accepted path is fixed six-edge softmax organization, raw/residual
hyperedge state, and dense context fusion. Its checkpoint-visible layers remain
registered directly on those facades.

Optional Stage-1--6 mechanisms are implemented under `organization/` and
`decoding/`. Exchangeable slots retain their historical
`organizer.exchangeable.*` ownership. Additive and gathered execution use a
plain mixin that owns no `nn.Module` state, so no parameter was reparented.
