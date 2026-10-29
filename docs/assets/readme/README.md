# README figure collection

Curated examples accompany the project homepage. Each image has a PNG preview and a single-page PDF master; links below open the detailed views. Use explicit display widths and retain aspect ratios when embedding them. This directory contains illustration assets, not datasets, trained weights or numerical exports.

| Figure | Preview and master | Source and meaning |
| --- | --- | --- |
| Opening showcase | [PNG](honf_showcase.png) · [PDF](honf_showcase.pdf) | Relation-anchored field reconstruction, spatial access regions and exact overlapping original-member sets, composed for an introductory overview. |
| Hypergraph-routed fields | [PNG](hypergraph_routed_fields.png) · [PDF](hypergraph_routed_fields.pdf) | Thermal-channel checkpoint at epoch 5000, validation scene 0678; native reference, prediction and signed error for streamwise velocity and fluid temperature. |
| Hypergraph-routed organization | [PNG](hypergraph_routed_organization.png) · [PDF](hypergraph_routed_organization.pdf) | Same checkpoint and scene; six shared slots with actual M/E assignments and original record IDs. |
| Relation-anchored fields | [PNG](relation_anchored_fields.png) · [PDF](relation_anchored_fields.pdf) | Saved checkpoint at epoch 2813, validation scene 0678; restricted joint95 read and training-only normalization. |
| Spatial access regions | [PNG](relation_anchored_regions.png) · [PDF](relation_anchored_regions.pdf) | Same relation-anchored scene; prepared directory service regions and original-module participation. |
| Original-member service | [PNG](relation_anchored_member_services.png) · [PDF](relation_anchored_member_services.pdf) | Same scene; M2/M3 directory service and separately scaled contextual receiver maps. |
| Overlapping hyperedges | [PNG](relation_anchored_hyperedges.png) · [PDF](relation_anchored_hyperedges.pdf) | Same scene; three exact native original-member intersections with the display rule described below. |
| Hypergraph-routed training | [PNG](hypergraph_routed_loss.png) · [PDF](hypergraph_routed_loss.pdf) | Recorded sampled training terms and validation measurements through epoch 5000. |
| Relation-anchored training | [PNG](relation_anchored_loss.png) · [PDF](relation_anchored_loss.pdf) | Saved history through epoch 2697; an earlier snapshot than the field example. |

## Field displays

The opening showcase and both detailed field figures use the same 7,464 physical-fluid points. Geometry and fields retain the packed dataset's native units. Reference and prediction share their color scale within each field; signed errors use a separate symmetric scale. Temperature uses `coolwarm`, velocity uses `viridis`, and signed error uses the dark diverging `berlin` palette. The two complete field views retain reference coordinates; the showcase omits their repeated axes for readability.

The error panels display RMSE computed from the plotted prediction and reference: hypergraph-routed velocity/temperature 0.01170/0.29471 and relation-anchored 0.01941/0.57633. These saved examples have different checkpoint ages, normalizers and physical prediction laws, so they do not provide a controlled model ranking. The showcase simplifies presentation without changing the numerical fields or filling the solid-object mask.

## Organization displays

The showcase reuses the relation-anchored checkpoint and scene above. Its spatial map colors each of 192 environment addresses by the strongest nonself member in the prepared directory; the nine displayed leading members are all M objects. Module ring sectors show each original M member's service across those regions. The own-E entry is omitted only when assigning display regions. The participation table remains available in the detailed spatial-access figure.

The overlapping rows h(E120), h(E104) and h(E85) contain 5M+1E, 7M+2E and 10M+1E, respectively. An E display threshold of 0.02 retains all positive M and own-E entries; these three rows retain 100% of their directory mass. Vertices lie inside exactly their displayed member sets. Ellipse shape, area and abstract distance are illustrative; they carry no physical scale. These are representative directory rows, not three discovered physical groups. Learned organization describes model access rather than physical causality.

## Updating these assets

Retain PNG previews, PDF masters, meaningful alternative text and the source/metric description when replacing figures. Every reconstruction plot must visibly show fieldwise L2 (identify relative L2) or RMSE computed on its stated physical mask and units. Place it on the error panel when present, otherwise on the reconstruction panel; if no reference exists, label the error unavailable. Numerical arrays, histories, checkpoints and one-time rendering scripts belong outside this image directory.
