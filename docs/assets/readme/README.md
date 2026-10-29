# README figure resources

This directory is the user-approved home for the eight selected README figures. Each figure has a PNG preview and a one-page PDF master. The homepage uses relative links and explicit display widths; retain aspect ratio and avoid enlarging a small diagram. Future README figure replacements belong here as well. Do not add numerical arrays, trained weights, histories, diagnostic scripts or unrelated report figures.

| Stem | Saved source and meaning |
| --- | --- |
| `hypergraph_routed_fields` | Thermal hypergraph-routed latest, epoch 5000, exposed VAL0678, native reference/prediction/residual for streamwise velocity and fluid temperature. |
| `hypergraph_routed_organization` | Same checkpoint and scene; actual six-slot M/E assignments, original IDs and dense positive membership. |
| `relation_anchored_fields` | Frozen relation-anchored latest at epoch 2813, exposed VAL0678, restricted joint95, saved TRAIN-only normalization. |
| `relation_anchored_regions` | Same relation-anchored scene; prepared global directory service regions and module participation. |
| `relation_anchored_member_services` | Same relation-anchored scene; original M2/M3 service and separately scaled actual contextual M receivers. |
| `relation_anchored_hyperedges` | Same relation-anchored scene; three exact native original-member intersections, with display threshold 0.02 and positive M/own-E exceptions. |
| `hypergraph_routed_loss` | Hypergraph-routed history through epoch 5000; recorded training and validation reductions. |
| `relation_anchored_loss` | Earlier preserved relation-anchored history through epoch 2697; not a current progress assertion. |

The two field figures use the same 7,464 physical-fluid points. Their residual panels show RMSE in packed native units: hypergraph-routed velocity/temperature 0.01170/0.29471 and relation-anchored 0.01941/0.57633. Different checkpoint ages, normalizers and historical prediction laws prevent a controlled model ranking. Source geometry units are not silently converted to SI. Organization figures show learned model access, not physical causality.

All field figures must display fieldwise L2 (identify relative L2) or RMSE on an error panel, otherwise on the reconstruction panel. Compute the value on the plotted physical mask and in stated units; a caption alone is insufficient. If no reference exists, state that the error is unavailable. Keep this rule, saved-checkpoint identity, meaningful alternative text and the separately linked PDF master when updating figures.
