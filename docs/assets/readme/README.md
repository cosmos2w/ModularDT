# README figure collection

Four curated figures accompany the project homepage. Each has a PNG preview and a single-page PDF master. All use the relation-anchored shared detail-preserving mechanism with inventory-normalized access, from TRAIN600 run `thermal_r14_detail_train600_v1` at epoch 500. The two input-selected layouts belong to the exposed validation population. Coordinates and fields retain the packed ThermalChannel dataset's native units.

| Figure | Preview and master | Source and content |
| --- | --- | --- |
| Conceptual opening | [PNG](honf_showcase.png) · [PDF](honf_showcase.pdf) | Scene 0684/M10: original physical objects, selected native-set imagery, actual directed module gates, and velocity/temperature prediction textures. |
| Prepared access regions | [PNG](relation_anchored_regions.png) · [PDF](relation_anchored_regions.pdf) | Scene 0684/M10: nonself directory winners and original-module participation ring sectors. |
| Native overlapping hyperedges | [PNG](relation_anchored_hyperedges.png) · [PDF](relation_anchored_hyperedges.pdf) | Scene 0277/M3: exact original-member intersections, measured MM arrows and a physical-layout locator. |
| Physical reconstruction | [PNG](relation_anchored_fields.png) · [PDF](relation_anchored_fields.pdf) | Scene 0684/M10: speed and fluid-temperature reference, prediction and signed error. |

## Opening illustration

The opening image presents object-to-relation-to-field information flow. Its field textures use the saved prediction; the branch arrows are schematic workflow arrows. The central sets use four actual rows h(E126), h(E67), h(E188) and h(E5), selected with the common native-profile rule. Their abstract geometry preserves displayed membership; its area and distance have no physical scale. This is a conceptual introduction, so it carries no residuals, reconstruction metrics, quantitative colorbars or dense mathematical annotations.

## Organization figures

Spatial cell colors identify the strongest positive nonself directory member. Each original M member's ring sectors show its normalized service over those same display regions. Own-E entries are omitted only for the region-label calculation; the saved directory remains unchanged.

The detailed native-set figure selects h(E66), h(E166) and h(E11) automatically from all 192 address rows, using shared-profile resolution 0.5 and coverage quantile 0.9. These rows cover 99.0% of row profiles within that distance. With the visual E threshold 0.02, every actual positive M and own-E membership is retained; displayed row masses are 76.4%, 65.8% and 70.2%, respectively. Each set contains 3M+1E original members after the declared display simplification. Native rows remain overlapping memberships rather than three learned physical groups.

Dark arrows use recorded cycle-1 MM gates with donor→receiver direction. The figure nominates the strongest positive incoming donor per M receiver and displays at most eight connections; positive nonself coefficients use fixed stroke bins below 1/3, from 1/3 to below 2/3, and from 2/3 through 1. The locator uses original coordinates and module footprints. Selection, original IDs, weights, omitted members and exact ellipse intersections follow the [global organization specification](../../guides/Global_Hypergraph_Organization_Visualization.md).

## Reconstruction figure

Speed is $\sqrt{u^2+v^2}$; fluid temperature is the stored physical temperature channel. Metrics use the 7,474 receivers satisfying `fluid_valid AND fluid_target_weight > 0`. Reference and prediction share a scale within each field. Speed uses `viridis`, temperature `inferno`, and signed error uses the dark diverging `berlin` palette. Error limits are symmetric and include the full displayed residual range.

| Observable | Native-unit RMSE | Relative L2 |
| --- | ---: | ---: |
| Speed | 0.02203846 | 3.2419% |
| Fluid temperature | 0.55410568 | 4.3053% |

Both errors are recomputed from the saved plotted prediction/reference and the stated mask; the corresponding values appear inside each error panel. Geometry holes remain white. The opening conceptual image is a separate presentation of prediction textures.

## Source and updates

The literal source checkpoint is `thermal_r14_detail_train600_v1/checkpoints/epoch_0500_model.pt`. Physical arrays and restricted-directory/contextual arrays were exported together for the source layouts, with the checkpoint's saved train-fitted normalization. The local R14 evidence retains those arrays, exact selection/connection mappings and `readme_asset_provenance.json`; numerical data, checkpoints and one-time renderers remain outside this curated directory.

Retain only these selected PNG/PDF pairs here. Use relative links, meaningful alternative text, explicit display widths and preserved aspect ratios. Inspect PNGs, rasterized PDF masters and the rendered README after replacement. Detailed reconstruction figures carry their measured physical error; conceptual overview artwork follows its explicitly stated introductory purpose.
