# Global hypergraph organization: visual guide and working protocol

HONF organization should first be shown as the reusable information interface prepared for an entire physical system. Use three primary expressions: **whole-layout access regions for spatial intuition; original-member service maps for weights; overlapping native hyperedges for sharing and intersection**. A query figure then explains one invocation of that interface. This guide defines their meanings, illustrates them with saved relation-anchored evidence, and provides a reusable CPU plotting command and an agent execution procedure.

The figures must explain the model that actually ran. They do not require a new architecture, a forced number of groups, geometric-neighbor constraints, or replacing original records with pooled features. M/E remain input-object roles. A region labeled by an E supplier can jointly use several M members, and a remote contextualized E record can serve a regional collection of addresses.

## Start here

| Need | Read or run |
| --- | --- |
| Understand the three views | Sections 1–4 and the packaged README examples. |
| Generate a matched comparison | Section 6: `tools/plot_honf_organization.py`. |
| Export or adapt scene evidence | Section 5: array schema and actual-relation sources. |
| Review a figure or delegate to an agent | Sections 7–8: presentation rules and execution sequence. |
| Trace a numerical claim | The generated `evidence.json`, containing original IDs, permutations, member weights and exact intersections. |

The maintained implementation is [honf_viz](../../src/honf_viz/organization_data.py); the entry point is [plot_honf_organization.py](../../tools/plot_honf_organization.py). It reads saved NPZ/JSON evidence without loading a checkpoint, a model, torch, training targets or query arrays. Plotting dependencies are the optional `visualization` extra in [pyproject.toml](../../pyproject.toml): NumPy, Matplotlib and SciPy. Use the existing project environment; do not install or copy datasets just to redraw figures.

## 1. The global object and the quantities that must stay separate

For a fixed physical scene `s`, let `V_s = M_s ∪ E_s` contain original records with their identities and geometry. An E-addressed organization prepares a directory `R[a,i]` before query decoding. Its restricted directory is `C[a,i] = R[a,i] · 1[i ∈ candidate(a)]`, preserving the actual stored weights. The native hyperedge at address `a` is `h_a = {i : C[a,i] > 0}`, and its weighted incidence is `I[i,a] = C[a,i]`. All directory rows together form a scene-wide organization even though individual queries later invoke different rows.

This object depends on the prescribed physical inputs, including layout, materials, boundaries and controls. It is not necessarily a geometry-only graph. In the relation-anchored examples, the E-receiver directory reuses actual final-cycle E←M and E←E relations plus separately recorded structural self access and the model's access measure. Do not replace those equations with a visualization's normalization.

| Quantity | Question answered | Appropriate evidence |
| --- | --- | --- |
| Applied contextual gate `G` | Which directed relationship affected contextual exchange? | Actual captured gate, original receiver/donor IDs and contextual stage. |
| Executed attention coefficient | How was a particular message weighted in a head? | Actual coefficient including attention, measure and normalization; keep its head/reduction definition. |
| Prepared directory `R`, restricted directory `C` | Which original members may an address access, with what prior weight? | Scene-prepared rows before query scoring; keep full/restricted modes separate. |
| Query coefficient `β(q,i)` | How did one query actually use permitted members? | Native read trace including locator, type strength and within-role preference. |
| Field/design response | What happens to a physical objective when inputs change? | Field evaluation, perturbation or sensitivity, not graph degree alone. |

The hypergraph-routed interface had six explicitly parameterized shared hyperedges. The current anchored interface has a fine-grained library of E-address rows. Both admit a global view; the latter does not automatically contain a small learned group-assignment variable. Display colors, similarity ordering and representative selection summarize the library without claiming that the model merged it.

## 2. Whole-layout access regions and joint M/E participation

Color each E coverage cell by its leading retained nonself member: `g(a) = argmax_(i≠own(a)) C[a,i]`. The winner can be M or E. An address without a positive nonself entry stays neutral; do not assign a fictitious winner. Remove own E only from this display copy, since otherwise structural self access can obscure sharing. Preserve the original C in evidence and model execution.

Show how every original M participates in those regions using `W[m,g] = Σ_(a:g(a)=g) C[a,m] / Σ_a C[a,m]`. A module's ring sectors and table row encode the same fractions; a positive-service row sums to one, while a module with no service has a zero row. This is a descriptive joint incidence view, not a hard assignment of a module to its geometric neighborhood. It does not renormalize C itself.


Add a full incidence view in both physical order and shared similarity order. Physical columns follow y/x coordinate order. Similarity ordering uses average-linkage cosine clustering with optimal leaf ordering on equal-normalized model profiles; keep M/E member bands identifiable, preserve every displayed value and nonzero, and retain the permutation to original IDs. Apply the same ordering to matched models. Do not cut the hierarchy at a desired group count or rearrange values to manufacture blocks.


A shared resource legitimately spans several regions. A clean block diagonal is one possible pattern, not a success requirement. The spatial map explains where source preferences change; the matrix explains how memberships overlap.

## 3. Original-member service and actual contextual neighborhoods

A donor service field is `S_i(a) = C[a,i]`: where the original member `i` is available to E addresses and at what directory weight. Show all addresses, including zero entries, on one absolute scale across compared models. Use a clearly labeled marker for the supplier's actual location. Source position and service region can differ because contextual records already contain exchanged information.

Select a small common donor panel independently within M and E roles, using pooled scene-wide nonself directory mass across compared models. Selection uses the entire directory, not attractive query examples or model errors. The reusable command defaults to two M and two E suppliers. A role with zero shared service can still be displayed as an explicitly zero-service example; do not invent access to make a panel interesting.

The directory has E receiver rows only. To show additional M receivers, overlay actual contextual `MM[:,m]` for an M donor or `ME[:,e]` for an E donor. Use M squares and dashed donor→receiver arrows, with a separate G colorbar and explicit direction. Do not pretend these are M-receiver rows of C. Every positive M gate is drawn; no extra connectivity threshold is applied to these arrows.



For a receiver-centered interpretation, combine actual incoming gate rows rather than directory columns: an M receiver gets `[MM[m,:], ME[m,:]]`, and an E receiver gets `[EM[a,:], EE[a,:]]`. This exposes all four exchange routes in a consistent quantity. E-cell color marks incoming E gates; highlighted M symbols and arrows mark incoming M gates. A red receiver symbol has a different meaning from a donor marker and must be identified in the legend.


## 4. Native overlapping hyperedges

Use original M/E members as vertices and actual directory rows as hyperedges. Large, labeled M squares distinguish modules; smaller E circles retain original identities; a colored rim can mark a hyperedge's own E. Select at most three representative rows from the complete prepared directory for a readable exact-intersection diagram. The full global incidence remains the authoritative whole-scene view; a few selected hyperedges are an explanatory sample, not the entire topology.

The default selection removes own E only from selection profiles, L2-normalizes each model's rows, concatenates the models equally, and chooses representative actual rows by deterministic medoid refinement. Abstract vertices are then placed inside exactly the ellipses corresponding to their memberships. Verify every inside/outside pattern numerically. Tilted ellipses are suitable for a small number of edges; ellipse area, abstract distance and arrangement do not encode physical extent, strength or causality. Keep a small locator for the true anchor coordinates.

A visual E threshold is allowed to reduce clutter, but it does not change the model's execution coverage. Always retain actual positive M memberships and actual positive own-E memberships. Never add a zero-weight own E. Record omitted original IDs and weights, and compute each selected edge's shown-mass fraction `η_a = Σ_(i displayed) C[a,i] / Σ_i C[a,i]`. Do not renormalize the displayed weights or report η as prediction fidelity. If an exact-intersection panel exceeds its declared display capacity, show fewer representative edges or choose and disclose another E display threshold; do not silently delete vertices.


A dual service-set picture is optional, but its meaning differs: E-address hyperedges become vertices, and one original member becomes the hyperedge of addresses it serves. Label it explicitly as a dual. Do not add original M symbols to a dual diagram while pretending that they have the same vertex meaning as the E addresses, and do not interpret transpose/reverse indexing as an executed reverse exchange.

## 5. Evidence input and adapter responsibilities

The plotting tool expects one NPZ per scene/checkpoint with adjacent metadata JSON. It accepts one or two matched models. Required arrays follow the table below; padding is accepted only in M slots and is packed out by the loader. Original IDs can be noncontiguous integers or strings, stored without pickled object arrays. The command never reads `query_ids`, `query_xy`, β or targets, even when they coexist in a historical export.

| Array | Shape | Meaning |
| --- | --- | --- |
| `module_xy`, `module_ids`, `module_mask` | `[M_slots,2]`, `[M_slots]`, `[M_slots]` | Original 2D physical/projected coordinates, IDs and active-slot mask. |
| `environment_xy`, `environment_ids` | `[E,2]`, `[E]` | Original active E coordinates and address order. |
| `C_restricted` | `[E,M_slots+E]` | Actual restricted directory weights, with columns in M-slot order then E order. |
| `context_cycleN_mm/em/ee/me` | Receiver×donor, with M-slot axes where appropriate | Optional actual applied contextual gates, not attention logits or reconstructed transposes. |
| `domain_bounds` | `[4]` | Optional adapter bounds `[xmin,xmax,ymin,ymax]`. |
| `environment_polygons` | `[E,vertices,2]` | Optional physical coverage polygons; do not infer them from model colors. |
| `radius` | Scalar or `[M_slots]` | Optional circular M geometry; omit when circles are not the adapter's geometry. |

The adjacent JSON records at least the literal case, partition, epoch, checkpoint, coordinate units/projection, prescribed controls or their existing scene identity, directory mode/coverage, and route-stage provenance. Existing relation-anchored exports retain this information; units absent from historical metadata must be described as source geometry units rather than invented. Keep prescribed controls fixed when comparing models. Record whether any environment features or normalization differ. Existing manifests/IDs suffice; no new hashes or frozen-baseline service is needed.

For new exports, prepare the scene once using its native adapter and evaluation model; do not pass unknown field targets into the model. Capture actual route snapshots while the encoder runs, including their receiver/donor IDs and stage. In the maintained implementation, `NativeRouteSnapshot.applied_incidence` is the applied G source in [route_capture.py](../../src/honf_forward_core/native_retrieval/route_capture.py), and MM/EM/EE/ME execute in [staged_attention_v2.py](../../src/honf_forward_core/content_hypergraph/staged_attention_v2.py). The anchored preparation stores `anchored_applied_EM`, `anchored_applied_EE`, `anchored_route_provenance` and `e_access_directory` in [anchored.py](../../src/honf_forward_core/interaction_reused_member_routing/anchored.py).

The packed full/restricted directory contains offsets and original-member indices. Export C by assigning each row's `access_weight` to its `member_index` column, preserving the original M slots and E order. Use the stored directory; do not recompute G→R conversion with a plotting-specific measure. Capture MM/ME from the actual staged snapshots if needed; EM/EE alone cannot establish them. The CLI can render directory views without all four gate blocks and explicitly reports that contextual panels were omitted. A model with a different organization schema needs a transparent adapter export, not fabricated E rows.

Adapter-owned polygons take precedence. Otherwise a complete Cartesian coordinate inventory can be drawn as cells using midpoint boundaries; irregular coordinates are drawn as points without inventing coverage cells. The core plotting package has no Thermal domain size, grid count, module count or role-specific output-physics assumption. Three-dimensional cases require an explicitly documented 2D projection or slice from the adapter; the command does not silently flatten them. Noncircular module outlines require an appropriate adapter extension rather than a fictitious radius. Different layouts should have separate reports unless a deliberate original-ID/geometry alignment is supplied by a suitable comparison implementation.

## 6. Reproduce the reference and use the API

Run the plotter on saved scene evidence using explicit local input/output paths. This reads arrays only; it does not fit or load a model. Supply one or more aligned scenes, with their sidecars, as documented by `--help`:

```bash
python tools/plot_honf_organization.py --help
```

The [packaged README examples](../../README.md#relation-anchored-organization) show the three standard views from a frozen relation-anchored checkpoint at epoch 2813 on exposed VAL0678. Their [image index](../assets/readme/README.md) states original IDs, units, source age, thresholds and limitations. Other comparison galleries and full numerical evidence remain local; they are not silently relabeled as these examples.

The Python entry points support composition in another report generator. Use the installed editable package or add `src` to the import path or use the installed package with its visualization dependencies:

```python
from pathlib import Path
from honf_viz.organization_data import load_scene, require_matched
from honf_viz.organization_overview import render_overview, render_incidence
from honf_viz.organization_services import render_donor_services, render_incoming_neighborhoods
from honf_viz.organization_hyperedges import render_native_hyperedges

scenes = [load_scene("candidate", Path("/data2/path/to/scene.npz"))]
require_matched(scenes)
output = Path("/data2/path/to/budgeted-visualization")
overview = render_overview(scenes, output)
incidence = render_incidence(scenes, output)
services = render_donor_services(scenes, output, per_role=2)
hyperedges = render_native_hyperedges(scenes, output, e_threshold=0.02, edges=3)
# Each FigureArtifact returns stem, title, caption and numerical evidence.
# Call render_incoming_neighborhoods only when all four actual gate blocks exist.
```

## 7. Presentation rules

Keep figure interiors clean. Retain only a main title, useful panel titles, physical axes/units when known, original IDs, a necessary symbol/direction legend and quantitative colorbars. Do not place redundant subtitles, pale-gray secondary explanation, checkpoint paragraphs, method notes, threshold warnings or a block of caveats around the panels. Put those details in readable prose captions. Do not shrink many tiny labels to force a large graph onto a page; select a declared illustrative panel and retain the full global matrix and numerical evidence.

Use shared member colors and absolute quantitative scales within each comparison. A color is not a model group unless such a group actually exists. Keep exact zero distinct from small positive values; a logarithmic color floor is a color choice, not a support threshold. Label supplier, receiver and query markers distinctly. A dashed arrow must have a stated direction. Use two separate scales when C and G share a panel. Keep M/E input roles separate from predicted physical channels.

Every caption states source layout/case, partition, epoch, directory or contextual quantity, full/restricted mode, coordinate units/projection, example-selection rule, relevant self handling, any threshold/omission and the one main conclusion. For native ellipse diagrams, include retained mass and state that abstract geometry is illustrative. For sorting, state that values and support are preserved. Keep contextual stages distinct and never infer physical causality from a service field.

Retain PDF masters. Embed PNG companions with relative paths and meaningful alt text, and link the PDF separately. Include a short figure index. Each prose paragraph and caption occupies one continuous Markdown source line. Render and inspect actual PDFs, PNGs and the full report after the final regeneration; verify decoded images, relative links, IDs, scale consistency and exact diagram memberships. Do not stop at file existence or numerical logs.

## 8. Agent execution sequence

1. Read this guide and the relevant model's preparation/read code. Identify the exact scene, controls, checkpoint epoch, original inventories, full/restricted mode and actual contextual stages. Reuse saved evidence when it matches; no new training is needed to redraw it.
2. Verify DATA2 and ignored symlink bindings. Record free space, retained/temporary byte budgets and headroom; reuse source arrays and shared caches. Preserve active runs, user edits and historical evidence.
3. Export or validate only the bounded scene evidence needed. Align original IDs and geometry across models. Missing gate blocks remain missing; do not reconstruct reverse exchanges or add value paths to improve a diagram.
4. Run the CPU command. Keep all three primary views. Include physical/sorted incidence and four-route neighborhoods when they clarify the question. Use matched examples/scales and preserve unmodified source weights.
5. Inspect the figures and write the specific interpretation from their evidence. Distinguish scene-wide topology, one query's activation and physical/design response. For inverse-design interpretation, compare actual layout perturbations and field/objective changes; broad service alone is not a sensitivity ranking.
6. Validate original-ID crosswalks, exact sorted values/support, soft-participation sums, threshold omissions and ellipse intersections. Save a readable Markdown report with all selected PNG panels and PDF links, and inspect the rendered result.
7. Remove only round-owned temporary previews/caches and superseded outputs after validating retained replacements. Preserve historical/selected evidence. Record cleanup bytes, retained artifacts and free space. Commit/push durable code, tests, documentation and project instructions through existing safeguards; do not upload scene arrays, one-time diagnostics or other generated figures. The explicitly approved curated README PNG/PDF pairs live in `docs/assets/readme/`.

The maintained project reminders are in the repository [AGENTS.md](../../AGENTS.md). This protocol is a reporting and interpretation workflow, not a new training admission mechanism. Its purpose is to make learned organization visible without changing what HONF is asked to learn.
