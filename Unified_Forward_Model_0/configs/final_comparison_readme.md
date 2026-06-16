# Final Comparison

Notation: `D` is layout/module structure, `c` is operating context, `H` is learned hypergraph organization, `U` is the predicted physical field, and `q` is a query point.

- `H_clean_capacity`: clean capacity-matched HONF using learned `H`, global context, near-module context, query Fourier/boundary features, and position Fourier features.
- `H_enhanced_pairwise`: HONF plus an `H`-routed query-module pairwise kernel. Pairwise geometry is aggregated by `A_mh` and query-to-hyperedge attention rather than exposed as an unrestricted direct shortcut.
- `NH2_no_hyper_current_like_direct005_capacity`: no-`H` shortcut upper-bound style model with global, near-module, and gated direct module/env residual context.
- `NB_query_pair_deepsets_full_capacity`: strong direct query-module pairwise neural field baseline.

Main question: Does `H_enhanced_pairwise` close the gap to `NB_query_pair_deepsets_full_capacity` while preserving interpretable `H`?
