# Dataset and checkpoint web demos: design and reuse guide

This guide describes how to turn an existing dataset and trained checkpoint into a local, sponsor-facing scientific web demo. The reference implementation is [HONF / Wind Lab](../../../Case_WindFarm/demo/README.md), developed through the source revision `7fd5a5d`. Its central idea is to make a real reconstruction directly explorable: a large physical scene, a few standard quantitative sections, editable supported inputs, and a second view into the learned organization. Reuse the trained model and its native adapter; build a small presentation layer around them.

The maintained source is `docs/guides/Dataset_Checkpoint_Web_Demo/README.md`. A local reading copy under `HONF_Proj/docs/guides/Dataset_Checkpoint_Web_Demo/` preserves the requested workspace location with relative links adapted to that directory.

The method applies to other physical datasets, but Wind Lab is a working case implementation rather than an already implemented case-neutral demo framework. Its turbine geometry, vector fields, coordinate conventions, input limits and profile-derived deficit must be replaced or adapted for another case. A convincing demo preserves the model's actual capabilities while making them easy to see and manipulate.

## Start here

| Need | Read |
| --- | --- |
| Understand the design progression and intended experience | Sections 1 and 4. |
| Connect a new dataset and checkpoint | Sections 2–3 and 11. |
| Reuse field rendering, scientific sections or editing | Sections 5–7. |
| Show learned organization without changing its meaning | Section 8 and the [global organization guide](../Global_Hypergraph_Organization_Visualization.md). |
| Keep the demo lightweight and convenient to launch | Sections 9–10. |
| Verify and deliver a bounded implementation | Section 12. |

## 1. The development progression

The initial requirement was an attractive desktop demonstration of three-dimensional reconstruction and learned hypergraph organization, using the latest suitable full-data Wind checkpoint. The useful refinements made that scene more quantitative, more editable and easier to operate. The table records the design decisions worth carrying forward.

| Stage | Resulting behavior | Reusable lesson |
| --- | --- | --- |
| Establish real inference | Load the existing dataset, saved model configuration and TRAIN-fitted transforms; render physical predictions. | Resolve data and model identity before polishing the scene. A display cannot repair an incompatible loader or an unsupported control. |
| Make the field legible | Add real object geometry, sampled sections, integrated streamlines and direct camera controls. | Give the central physical view most of the page and make the important structure visible at the default view. |
| Add scientific anchors | Place three conventional section maps beside the 3D scene, with coordinates, units, colorbars and paired-reference errors. | A visually strong scene benefits from a small set of familiar quantitative views that share its data and scale. |
| Improve contrast and organization | Add quantity-specific palettes, local wake-focused color limits, stronger slice visibility, spatial links and exact overlapping hyperedges. | Improve display mapping and selection without changing field values, memberships or the model architecture. |
| Make editing explicit | Introduce a separate gridded design mode, object placement tools, boundary handles and resizable map views. | Use a visible task mode and direct manipulation; distinguish physical input edits from camera or display changes. |
| Finish operation and lifecycle | Provide one-command remote opening, coalesced reconstruction requests, bounded rendering work and verified resource cleanup. | A demo is complete when it is repeatable and usable through a full session, including reopening. |

Use this progression as a short delivery sequence, not a reason to build every possible renderer or interaction. Start with one representative case and one meaningful physical view, verify the entire inference path, then add the small set of interactions that explain the selected model.

## 2. Establish the scientific interface before the UI

### 2.1 Choose and identify the source

Inspect the saved run identity, actual training membership, checkpoint epoch, model configuration, normalization and native dataset reader. A run name, file modification time or a `latest` alias alone does not establish that a checkpoint was fitted on the full training population. Select the intended existing checkpoint, verify that its saved state loads through the supported historical loader, and pin its literal path and epoch for the demo. Subsequent demonstrations should not silently follow a changing alias.

Keep dataset and checkpoint payloads in place. Reuse the existing native reader, memory mappings and verified shared caches. Obtain selectable examples from the permitted partitions; Wind Lab exposes TRAIN and VALID and leaves TEST targets locked. Select illustrative scenes using declared input properties such as module count and direction, rather than searching for the lowest prediction error. No new training is needed to make this presentation layer.

### 2.2 Let model capabilities determine controls

List which inputs the checkpoint actually consumes, their units, training range, legal edit range and dependence on geometry. An editable input must reach its owning case object before model preparation. A quantity that was fixed during training should be shown as fixed unless the saved model genuinely supports it. Keep unsupported controls absent or visibly disabled. Legal input validation does not establish in-distribution accuracy or design reliability.

| Wind Lab example | Delivered meaning | What another case must supply |
| --- | --- | --- |
| `wind_r15_2_w2_train420_q8192_b24_full1000_v2/checkpoints/epoch_2500_model.pt` | Saved epoch 2500; 420 TRAIN and 90 VALID cases. The run name's earlier horizon is not the saved endpoint. | Its own literal checkpoint, epoch, population and loader. |
| Joint `Ux`, `Uy`, `Uz` prediction | Velocity in m/s; magnitude and deficit are derived display quantities. | Its actual jointly predicted channels, units and derivations. |
| Coordinates in `D` | Rotor diameter `D = 80 m`; hubs at `0.875 D`. | Adapter-owned physical coordinates, scales, object shapes and measures. |
| Wind direction | One of 270°, 285° or 300° in the existing direction-specific wind-aligned frame. Changing the category does not rotate the layout. | The real control representation and any required coordinate transformation. |
| Inflow speed | Fixed 9 m/s training reference; arbitrary speed editing is unavailable. | A truthful editable or fixed control. |
| Turbine layout and domain | 1–30 turbines are accepted; training layouts contained 6–30. Edited layouts have no new CFD reference. | Valid identities, topology, bounds and a clear reference-availability rule. |

Use a compact model-information panel and durable documentation for these details. The main scene should carry physical names, units, essential status and direct actions; avoid filling it with methodological subtitles or repeated explanations.

## 3. A small frontend/backend architecture

Wind Lab uses React and Vite for the application, Three.js for the physical and organization scenes, Canvas 2D for scientific maps, and a Python FastAPI/PyTorch service for inference. Python owns case interpretation and physical transforms. The browser owns interaction state and rendering. The backend serves the built frontend, so the demonstration needs one local URL; the Vite development server proxies `/api` during development.

```text
Dataset metadata and target-free case inputs ──> case adapter ──> saved model
                                                                  │
Browser scene edits ──> validated API request ──> prepare + decode ─┤
                                                                  ▼
Browser 3D view / maps / organization <── physical result + provenance
                  ▲
Native reference and physical mask ── paired evaluation only
```

The model receives prescribed inputs, never unknown target fields. Reference retrieval and error calculation accompany an eligible native reconstruction after inference. Keep organization captured from the same prepared scene as the prediction. Wind Lab prepares once per reconstruction, decodes bounded query chunks and converts outputs with the checkpoint's saved transforms; it does not add a plotting-specific correction branch.

### 3.1 Keep the API explicit

| Endpoint | Responsibility |
| --- | --- |
| `GET /api/meta` | Checkpoint identity, available case metadata, units, default scene, supported controls and limits. Wind Lab loads metadata here and lazily loads weights on the first reconstruction. |
| `GET /api/cases/{id}` | A target-free scene: original IDs, object geometry, domain, controls and partition. |
| `POST /api/reconstruct` | Validate the selected case and edited inputs; return the scene, physical samples, available reference, organization and provenance. |

The existing Wind request contains `case_id`, optional `turbines`, `domain`, `wind`, and `resolution: [nx, ny, nz]`. Its response has five main objects: `scene`, `grid`, `organization`, `errors`, and `provenance`. This is a useful small pattern to adapt, not a requirement to call every future object a turbine or encode every dataset as a regular volume.

| Response information | Required meaning |
| --- | --- |
| Scene | The actual submitted geometry, original object IDs, physical domain, prescribed controls and units. |
| Grid or native sampling geometry | Explicit coordinates, shape/connectivity, ordering, predicted physical fields and any physical validity mask. |
| Reference | Paired values at exactly the same coordinates, with the same channel interpretation and declared physical mask, or an explicit unavailable state in the payload. |
| Organization | Actual member identities, roles, coordinates, row membership/weights, relation direction, preparation stage and display-selection metadata when available. |
| Provenance | Dataset case and partition, literal run/checkpoint/epoch, reference availability, numerical sampling and measured inference timing. |

For the Wind volume, axes may be nonuniform. Values are flattened x-major, z-fast, component-last: component `c` at `(ix, iy, iz)` has offset `((ix * ny + iy) * nz + iz) * C + c`. Carry the explicit axes to the browser and interpolate in physical coordinates. Do not reconstruct cell centers from box bounds or assume that a domain face is a returned native sample plane.

Keep UI convenience limits separate from server validation. Wind's browser editor uses horizontal extents of 4–100 D with 0.5 D center clearance; the server separately checks finite inputs, unique IDs, valid hub height, increasing bounds, centers inside support, altitude, trained direction/speed and query budget. Both layers need to agree on accepted actions, but they serve different purposes.

### 3.2 Reference eligibility follows the scene

Wind Lab returns paired CFD values only when positions, original IDs/order, bounds and direction match the native case within the implemented numerical tolerance. It samples native cell centers and extracts references at the same indices. Edited scenes instead use a grid over the submitted support and return null reference fields, errors and CFD mask. Restoring the exact native scene can restore its reference; a visually similar edited scene is insufficient.

For another dataset, define this rule through its native observation/extraction interface. Preserve real solid/fluid masks and invalid regions. Wind's sampled native-volume mask is all true; this does not establish generic mask support throughout the current frontend. Its scientific maps handle nontrivial masks, but the current 3D slice/volume error readout does not consult one. A port with solids or invalid cells must make 3D interpolation, triangles, paths and metrics mask-aware before reuse. Also avoid duplicate native coordinates when a requested resolution exceeds the native axis length: cap the sample count or use a documented native interpolation path.

## 4. Desktop composition and mode switching

Prioritize the desktop demonstration. Wind Lab uses a compact top status bar, a left scene/control panel, a large central 3D workspace, a right scientific-map sidebar and a small footer for provenance and inference status. Its layout has a desktop minimum width of 1180 px. Mobile layout was not an objective of this implementation.

Use one coherent visual system across modes: dark navy backgrounds, clear typography, restrained mint/cyan accents, readable object silhouettes and physical color scales. Translucent control panels should preserve scene visibility without making labels hard to read. Keep the field itself visually dominant. Use code-native object geometry and lighting so physical objects remain readable as the camera moves. A future case can choose another palette while preserving this hierarchy.

| Mode | Visual context | Primary actions |
| --- | --- | --- |
| **Flow field** | Physical objects, reconstructed field, streamlines and selected section. | Orbit, zoom, fit, select a quantity, move a section, adjust contrast, pause animation. |
| **Module design** | A distinct blue-toned background, floor grid, object layout and relevant handles. | Add, move or delete an object; resize the supported domain. |
| **Learned hypergraph** | The same application shell and visual vocabulary, exposing the prepared organization. | Start with focused spatial links; inspect an address or switch to abstract Venn sets. |

This mode switch provides the intended experience of looking beneath the field into its information organization. Keep physical geometry recognizable in the spatial view, but signal abstract coordinates clearly in the set view. Mode changes, camera moves and quantity selection should reuse the current inference result. Only physical input changes or model sampling changes require new inference.

Use short English action labels, physical symbols and units. Group controls by task rather than exposing implementation flags. Provide visible selection, hover/focus feedback, accessible names, keyboard operation for key actions and a clear reset/back path. Camera orbit must remain usable while reconstruction runs; a loading state should identify pending field data without blocking the editor.

## 5. Make physical structure visible without altering it

### 5.1 Quantities and color mapping

Give each physical quantity an appropriate palette. Wind Lab uses distinct sequential palettes for speed and downstream velocity, and zero-centered diverging palettes for crosswind velocity, vertical velocity and downstream deficit. A new case should choose scales from its physical meaning: temperature magnitude, signed residual and material identity should not share an undifferentiated rainbow mapping. Use the same selected quantity, units and labeled limits in 3D and the accompanying section maps.

Whole-volume color limits can hide the phenomenon of interest. In Wind, vertical profile variation made near-rotor velocity deficits look uniform. **Wake focus** therefore uses the 2nd–98th percentiles of actual grid values in the union of rotor neighborhoods: −0.5 to +8 D downstream, ±1.5 D laterally and ±0.65 D about hub height. Signed quantities retain symmetric limits about zero; too few local samples fall back to the full field. **Full range** remains available. Endpoint clipping changes colors only, never field values or reported errors. For another case, define a physically meaningful region of interest rather than copying the rotor bands.

Downstream deficit is the saved TRAIN-fitted profile `Ux(z)` minus predicted `Ux` at that height. It is not a uniform 9 m/s subtraction. Document derived quantities and compute them consistently for prediction, reference, maps and legends. A visual improvement must not introduce a synthetic wake, analytical correction or manually painted feature into model output.

### 5.2 Sections, paths and object rendering

Sample movable XY/XZ/YZ sections from the returned physical axes, with the selected position shared between the 3D plane and its inspector. Preserve orientation and units when mapping physical `(x, y, z)` into renderer coordinates. Vertical exaggeration is a labeled display transformation; changing the physical domain is a model input edit. Do not conflate the two kinds of stretching.

Use opacity deliberately. A nearly transparent plane can hide the reconstruction even when its scalar colors are correct. Wind Lab increases alpha from 0.13 to 0.80 according to the actual positive profile-relative deficit, preserving the selected field as the color quantity. This improves wake visibility while retaining objects and spatial context. It is a case-specific display strategy, not a generic additional physical channel or a probability scale.

Seed streamlines near the real objects and include limited upstream context. Integrate through the predicted vector field rather than drawing arbitrary parallel decorative lines; Wind uses midpoint integration in physical coordinates. Reuse sampled path coordinates and colors for moving streaks. Rotor motion and particles illustrate a reconstructed steady field; they do not establish transient forecasting. A scalar-only checkpoint needs a meaningful scalar renderer, not invented velocity animation.

## 6. Scientific section maps as part of the interface

Keep conventional maps beside the central scene to support scientific inspection. Wind Lab shows a hub-height XY plane and XZ/YZ sections through the turbine-layout center. The maps have transparent backgrounds, physical aspect ratios, concise plane titles, modest axes, light gray contours and individual colorbars sharing the selected physical limits. Avoid pasting white-background report images into the dark workbench.

Use continuous rendering from the underlying returned samples. When a canvas grows or a crop changes, resample the physical field for that displayed area and device-pixel size, within a bounded work budget. Enlarging an existing PNG cannot restore detail. Interpolation adds display resolution; it does not create new model predictions, native CFD resolution or evidence at unobserved scales. If more model samples are desired, request them separately and distinguish that action from zooming.

| Map state | Wind contour levels | Maximum contour mesh budget |
| --- | --- | --- |
| Compact section | 8 | 7,000 cells |
| Rendered plot at least 440 px wide | 16 | 14,000 cells |
| Fullscreen map grid | 20 | 24,000 cells |
| Focused single map | 28 | 32,000 cells |

Contour density and mesh resolution should increase together as the display enlarges; line width can decrease so extra detail remains light. Wind's raster sampling is capped at 128,000 points per panel. Masked cells stay transparent and do not enter contours or errors. The exact counts are measured implementation choices to adapt to another dataset's complexity and machine, not universal scientific thresholds.

The right divider resizes the sidebar from a useful compact width up to half of the window. Dragging beyond that limit by 32 px opens an opaque full map workspace. Clicking a map focuses it; the wheel zooms at the pointer and dragging pans the crop. Wind allows 1–24× map zoom. **Show in 3D** selects the corresponding physical section. **Back** or Escape restores the default sidebar width and map crops. Fullscreen layout gives XY a large area beside two vertical panels instead of merely stretching three narrow thumbnails.

With a paired reference, compute the selected-field RMSE from the actual displayed crop and valid paired samples, and draw it on the reconstruction panel. The 3D overlay uses the displayed slice or sampled volume. Wind Lab follows the explicitly requested interactive-UI convention of showing no error annotation when reference data is absent; the response still records reference availability and the interface identifies edited model exploration. This UI choice does not change the repository's separate annotation rules for exported scientific figures and reports. Never invent an error or silently reuse a native reference after editing its inputs.

## 7. Direct physical editing with clear update semantics

Design mode makes editing a deliberate task. **Turbine layout** exposes **Add**, **Move** and **Delete**. Add places an object on a valid floor click; Move drags a selected object in physical coordinates; Delete removes the clicked original object. A camera drag must not be interpreted as a placement click. Keep IDs stable across surviving objects, show the current selection and allow precise coordinate entry when useful. Adapt this interaction to actual object geometry for another case; a volume, wall or irregular mesh may need a different placement constraint.

**Domain sizing** exposes X/Y edge handles and extent sliders. Wind keeps the domain center fixed, so moving one edge changes the full extent by twice that edge displacement. Bounds enforce the editor's limits and retain turbine centers at least 0.5 D inside the horizontal support. Expansion and contraction must not silently remove objects. Only expose dimensions and shape changes that the native adapter can consume; changing a box around an arbitrary mesh is not automatically a valid physical-domain edit.

Separate input state, valid inference state and presentation state. Wind debounces scene edits by 650 ms, allows one active reconstruction per mounted page and retains only the newest waiting scene. On input or sampling changes, invalidate stale work immediately; accept a response only if it still matches the latest request and scene signature. Hide outdated field/map pixels while new data is pending, rather than displaying an old field on new geometry. Quantity changes, crop changes, camera motion and vertical exaggeration should not submit additional inference requests.

Aborting a browser request does not necessarily cancel CPU inference already running on the server. Coalescing new work prevents obsolete edits from forming a long server queue. Metadata/case fetches can be canceled on replacement or teardown. The backend still validates and serializes model execution. The current single-flight policy applies to one mounted page; it is not a multi-client scheduling service. Add such infrastructure only for an actual multi-user requirement.

## 8. Learned organization: spatial intuition and exact sets

Follow the [global hypergraph organization guide](../Global_Hypergraph_Organization_Visualization.md) for definitions, evidence and formal figure production. M/E are input-object roles, not responsibilities for separate output physics. Keep original IDs, physical measures and geometry. Distinguish applied contextual gates, prepared access-directory weights and final query contributions; preserve receiver←donor direction and stage provenance. A transposed lookup is not an executed reverse exchange.

### 8.1 Spatial links by default

Enter **Learned hypergraph** with **Spatial links** and **Focused links** selected. Place original M objects and E anchors at their real coordinates, distinguish roles with shape/size, and use subdued links with a limited bright backbone and restrained motion. Wind exports all native directory rows; its spatial view represents 12 evenly selected E-address rows, and its address selector can inspect any one of those represented rows. Venn selection can draw from the complete directory. **All links** restores every positive connection of the represented rows, not every row of the entire directory at once.

Filter only for display. Wind preserves every positive M membership and positive own-E entry, then retains the strongest other resolvable E members toward 95% of each represented row's drawable positive weight. Report visible membership counts and retained weight against the full positive row weight; unresolved coordinates can lower full coverage and must remain accounted for. Never renormalize the model's directory to make a cleaner picture. Keep the full positive support available.

### 8.2 Venn sets for overlapping membership

Provide an abstract set view as a complementary mode. Choose a small number of representative actual native rows and place each original member inside exactly the sets to which it belongs. Wind compares non-self, L2-normalized membership profiles, refines medoids and adds a farthest profile until the 90th-percentile residual reaches 0.5 or a three-set display cap is reached. A selected address pins the first row. This is a display selection, not a learned group-count result.

Preserve every positive membership of the selected rows, including small positive M and own-E entries; Wind's Venn view applies no E display threshold. Verify exact intersections geometrically and numerically. Make M symbols prominent and connect them using only the relation actually available. Wind's M-to-E arrows depict prepared-directory access; curve shape is schematic. If actual contextual M-to-M exchange is needed, capture its real applied coefficients and stage rather than inventing them from directory overlap.

Venn coordinates, sphere areas and distances are abstract. Disable physical dragging and physical top-view presets there; provide a clear face-on fit and restore the physical camera when returning to spatial mode. Use consistent colors and materials to maintain the visual transition, while keeping the coordinate meaning unambiguous. These two interactive views do not replace the formal guide's complete gallery of whole-layout access, original-member service and native overlapping hyperedges. Learned support alone proves neither physical causality nor sparse execution savings; the Wind demo uses the checkpoint's full-read prediction.

## 9. Bounded rendering, inference and session lifetime

Mount one persistent 3D scene and update only objects affected by a change. Cache physical paths and their colors outside the animation loop; update existing buffer attributes instead of allocating arrays, colors, materials or geometry every frame. Keep reference sampling, range calculation, graph rebuilding and React state changes out of steady playback. Bound raster points, contour work, query counts and any palette cache; use weak keys when a cache should not retain discarded result grids.

Dispose replaced geometries, materials and textures, disconnect observers, remove listeners and cancel queued animation/pointer callbacks on teardown. Treat shared resources, such as Three.js sprite geometry, according to their shared lifetime rather than disposing them during every group rebuild. Release the renderer and WebGL context when the scene is removed. Suspend 3D animation/rendering for hidden tabs and opaque fullscreen maps; resuming should not jump the simulation clock by the hidden duration.

Use one lazily loaded evaluation model and fixed read-only dataset mappings on the backend when they fit the local demonstration. Wind runs on CPU with bounded query chunks and no per-case result history. Its browser holds the original/current scene and current result in page memory, without localStorage, sessionStorage, IndexedDB or a service worker. Reload or close discards edits and results; reopening starts from the default dataset scene. Ordinary browser asset caching is a separate lifetime.

Closing the page intentionally leaves the reusable tmux inference server running, with its model and dataset mappings resident. A submitted reconstruction may finish after its tab closes. This is fixed service residency, not a saved history of demonstrations. Full backend memory release requires stopping that owned service. Keep this lifetime clear in operator documentation; do not add automatic session tracking, heartbeat daemons or persistent scene storage merely to support a local demo.

Check resource behavior over a bounded interval and repeated interactions. The delivered Wind build was audited at 1600 × 1000 in Chromium with software rendering: three minutes of playback kept GPU resource counts and DOM nodes unchanged, JS heap was 12.5–13.1 MB, and browser process PSS was 428–438 MiB. Twelve mode/field/map cycles, slowed rapid edits, reload and tab reopening were also checked. Treat these as dated measurements for that build and environment, not as universal memory budgets or an arbitrarily long leak-free guarantee for future cases. A stable JS heap alone does not measure GPU memory or backend residency.

## 10. One local URL and a remote-friendly launcher

Provide a normal launcher and a convenience opener. Wind's [run.sh](../../../Case_WindFarm/demo/run.sh) reuses the project environment and locked Node dependencies, builds the frontend, and starts the localhost inference server. Its [open.sh](../../../Case_WindFarm/demo/open.sh) reuses a healthy service or starts a dedicated `honf-wind-demo-<port>` tmux session, waits for metadata and requests browser opening from the active VS Code Remote terminal.

```bash
# Current Wind reference implementation; replace the case path for a port.
/home/wanglz/Desktop/src/ModularDT/Case_WindFarm/demo/open.sh

# Start the service without requesting browser opening.
/home/wanglz/Desktop/src/ModularDT/Case_WindFarm/demo/open.sh --no-open
```

Browser dispatch uses the injected `$BROWSER` helper only with a valid `VSCODE_IPC_HOOK_CLI` socket. An ordinary SSH shell, an old tmux environment or the standalone Tunnel CLI may not provide that route. Keep a direct fallback: **Ports → Forward a Port → 8765 → Open in Browser**, using the forwarded address actually shown by VS Code. A successful server startup or helper dispatch alone is not evidence that the user's local browser opened.

Check service identity as well as port availability. The current Wind opener recognizes a Wind-like metadata response, not an exact checkpoint fingerprint; operators still inspect `/api/meta` for the intended run and epoch. A port to another case should update that recognition to its own existing identity fields and reject an unrelated occupied service. Keep localhost binding for the local demonstration. Publishing a multi-user externally reachable service is a separate deployment task.

Store dependencies, builds, browser-test downloads and disposable caches in the verified DATA2 output location, with ignored workspace symlinks where appropriate. Record free space, a hard disk budget, estimated temporary peak and minimum headroom before materialization. Wind used a 700 MB retained budget, 850 MB estimated peak and 500 GB DATA2 free-space floor; choose budgets for the actual new case. Reuse existing datasets, checkpoints and exact shared caches instead of copying them for the demo.

## 11. Porting the reference to another dataset

| Keep the pattern | Replace or verify for the new case |
| --- | --- |
| Thin Python adapter around an existing evaluation path | Dataset reader, checkpoint loader/configuration, saved transforms, M/E construction and query/output interpretation. |
| Explicit metadata and target-free case endpoint | Permitted partitions, existing case IDs, object roles, physical measures, units, prescribed controls and actual capability limits. |
| Scene/result/presentation state separation | Editable owning inputs, result invalidation and native-reference eligibility. Omit design mode if the checkpoint cannot accept meaningful edits. |
| Large physical view with conventional quantitative companions | Real geometry, scalar/vector/tensor fields, dimensions, masks, topology, physical slices and a suitable renderer. Keep 2D data 2D unless an explicitly labeled display extrusion is useful. |
| Quantity-specific palettes and shared scales | Field signs, valid derived quantities, physical ROI for contrast and clipping disclosure. |
| Linked crop/section interactions | Native axes or mesh connectivity, interpolation, masking, aspect ratios and paired error calculation. |
| Faithful organization inspection | Actual available directory/gates, original IDs, relation direction/stage and representative display selection. Omit unavailable organization rather than constructing a decorative graph. |
| Bounded runtime and one-command opening | Device/thread/query limits, environment paths, service identity, port, output location and measured resource budgets. |

The minimal case boundary owns loading an allowed input scene, describing capabilities, validating edits, preparing the saved model's input/query objects, decoding physical fields, retrieving aligned references/masks, and exposing real organization/provenance where available. Implement only the seam needed by the next case; the current code has no general case registry or interchangeable frontend adapter API. Extract shared modules after concrete reuse demonstrates a useful common interface. Follow the repository's [case/core boundary](../../case_plugin.md) rather than moving Wind-specific profile, geometry or field-head assumptions into a shared HONF predictor.

### Source map

| Reference source | Responsibility |
| --- | --- |
| [backend/app.py](../../../Case_WindFarm/demo/backend/app.py) | Saved-model loading, case access, API validation, physical prediction/reference, organization and provenance. |
| [App.jsx](../../../Case_WindFarm/demo/src/App.jsx) | Scene/result/view state, modes, supported controls, latest-scene request coalescing and field evidence. |
| [FlowScene.jsx](../../../Case_WindFarm/demo/src/FlowScene.jsx) | Three.js camera, object geometry, physical slices/paths, picking/editing, organization scene and resource lifecycle. |
| [flowMath.js](../../../Case_WindFarm/demo/src/flowMath.js) | Physical interpolation, quantities/palettes, contrast limits, streamline integration and spatial membership selection. |
| [ModuleDesignPanel.jsx](../../../Case_WindFarm/demo/src/ModuleDesignPanel.jsx) and [designMath.js](../../../Case_WindFarm/demo/src/designMath.js) | Editing tools, physical-domain handles, center-preserving sizing and object bounds. |
| [ScientificWorkspace.jsx](../../../Case_WindFarm/demo/src/ScientificWorkspace.jsx) | Resizable sidebar, fullscreen/focus transitions, crop state and default-layout restoration. |
| [ScientificPanels.jsx](../../../Case_WindFarm/demo/src/ScientificPanels.jsx) and [scientificPlotMath.js](../../../Case_WindFarm/demo/src/scientificPlotMath.js) | Transparent maps, physical crop sampling, mask-aware contours and paired crop RMSE. |
| [organizationVenn.js](../../../Case_WindFarm/demo/src/organizationVenn.js) | Representative native-row selection, exact set intersections and abstract set geometry. |
| [package.json](../../../Case_WindFarm/demo/package.json), [vite.config.js](../../../Case_WindFarm/demo/vite.config.js), [run.sh](../../../Case_WindFarm/demo/run.sh) and [open.sh](../../../Case_WindFarm/demo/open.sh) | Reproducible dependencies/build, development proxy, static serving and operator entry points. |

## 12. A bounded build and verification sequence

1. **Identify and qualify one saved source.** Record the literal checkpoint/epoch, dataset partitions, transforms, controls, field units, geometry and reference rules. Load one permitted case through the supported existing adapter. Preserve checkpoints, histories and active jobs.
2. **Connect one complete physical path.** Implement metadata, a target-free scene and one reconstruction response. Verify coordinate order, inverse normalization, native paired references and masks before styling. Bound query and payload sizes. No retraining or new full-dataset export is required.
3. **Build the desktop core.** Establish the physical scene, readable object geometry, camera controls and one useful default section. Add quantity selection, physically meaningful contrast and standard maps sharing its values and scale. Keep controls and visible prose concise.
4. **Add supported interactions.** Implement explicit design tools and domain controls only where the saved model accepts them. Keep display changes local, debounce/coalesce physical edits, invalidate stale results and provide reset/back behavior.
5. **Expose actual organization.** Export the prepared global object once with original IDs and provenance. Add a clear spatial view and a small exact overlapping-set view. Verify display filtering, membership intersections and retained mass without changing model support.
6. **Test numerical semantics.** Check physical interpolation on nonuniform axes, shared section coordinates, derived fields, reference/mask pairing and crop RMSE. Check tiny positive/zero/self memberships, original-ID preservation, filtering mass and exact Venn inside/outside patterns. The Wind [test command](../../../Case_WindFarm/demo/package.json) runs focused math tests for these implemented paths; new meshes or masks need their own meaningful checks.
7. **Test the rendered desktop workflow.** Open the actual served app at representative desktop sizes; inspect readability, transparency, legends, enlarged contours and layout. Exercise orbit/fit, all quantities, section movement, real add/move/delete, both domain handles, sidebar resize, fullscreen/focus/zoom/pan, Back/Escape, spatial/Venn switching and reset. Confirm submitted physical coordinates and latest-result acceptance, not only button labels. Check console and network errors. Use the available browser automation plugin first, or a documented Playwright fallback when it is absent.
8. **Measure a complete session.** Compare post-warm-up JS heap, browser process memory, DOM/listener counts and GPU resource counts during a fixed playback interval and repeated interactions. Check backend memory/handles separately, slow transport to expose stale-request queues, verify hidden/fullscreen suspension, and test reload plus close/reopen. Inspect browser storage and output directories for unwanted accumulation. Report the actual duration, workload, rendering environment and remaining fixed residency.
9. **Deliver an operational entry point.** Verify build/assets/API health, exact checkpoint metadata, launcher reuse, unrelated-port handling and remote-browser fallback. Keep source/lockfiles/docs in Git and generated outputs on DATA2. Retain a small set of inspected screenshots and compact QA evidence locally; static scientific exports follow the [organization guide](../Global_Hypergraph_Organization_Visualization.md) and repository figure rules.
10. **Close the round.** Remove verified round-owned browser downloads, temporary profiles, duplicated build/test outputs and superseded visual exports after retained evidence is checked. Preserve source data, checkpoints, reusable caches and unrelated work; record removed bytes, retained artifacts and free space. Audit the entire outgoing commit range and push durable source/tests/docs through the existing repository safeguards.

The final handoff should state what the selected model demonstrably reconstructs, which inputs can be edited, how physical and organization views relate, how to launch the demo, and what is retained after the page closes. Report measured gains and material limits directly. A polished interactive display is a presentation of saved-model behavior; physical causality, transient forecasting, sparse runtime savings and inverse-design reliability require their own evidence.
