# Saved organization and finite-response diagnostics

The commands below compare one already solved Thermal edit with its original scene. The shared core also provides `capture_applied_relationships`, `select_context_relations`, `prepare_relation_removal` and `finite_read_decomposition` in `honf_forward_core.interaction_reused_member_routing.organization_response`. These are evaluation tools; they do not authorize training, new physical solves, a TEST evaluation or an extension of any run.

## What is measured

The core captures actual directed contextual gates and the per-head coefficients actually used in contextual value reductions, in the order each stage ran. It records prepared E-address weights and final query-to-original-member coefficients separately. The stage records include receiver and donor roles, IDs and the source state version. The final query coefficients already combine address coverage, conditional preferences, read strength and coalescing. None of these weights is a signed physical effect.

For fixed common physical query positions and persistent original IDs, the finite difference of each typed member context is divided into an access-coefficient change and a message change. The union includes every original member actually read on either side. Missing-side coefficients are zero, but both messages are recomputed from their respective prepared scenes; the saved work record charges both the missing-side calculations and repeated selected-member calculations. The message term includes upstream contextual changes. A symmetric allocation through the existing nonlinear joint field head adds exactly to the finite joint output change up to numerical roundoff; the intermediate contexts are computational counterfactuals, not solved physical states.

Moving surface or material positions and a changed query encoding are not attributed using these two terms. Insertion, removal, incompatible member identity and a changed role are unavailable. The case adapter must establish the identity of edited members from saved records; geometric nearest-neighbor matching is not sufficient. Interpret exact reconstruction residuals at the recorded model precision. The function rejects stale prepared states and unsupported nonlinear-hyperedge heads.

The Thermal entry point samples a few common-fluid physical cell centres from input geometry, including inlet and outlet bands. Its physical metrics compare native, denormalized `u`, `v`, pressure and fluid temperature against the saved physical arrays on those positions. Its sampled pressure-band means are diagnostics and do not replace the complete native pressure functional. It does not compute vorticity, moving surface/material effects, complete native fields, J, P or inverse design results. Use the established physical-evaluation tools for those measurements.

## Capture a saved physical edit

Run from the repository root after assigning the actual checkpoint, the existing `finite_responses_v2.h5` pack, and an ignored output path on the verified DATA2 filesystem. Supply `--device` as a runtime choice. The provider loads the exposed validation case `0679` only to reconstruct the checkpoint's native data binding; `--loader-validation-case` can select another existing exposed validation case. The actual physical comparison is the explicit family and action.

```bash
python tools/thermal_organization_response_diagnostic.py \
  --checkpoint "$CHECKPOINT" --pack "$PACK" \
  --family 0081 --action single_plus --role fit \
  --read-mode restricted --query-count 32 --relations 4 --seed 0 \
  --device "$DEVICE" --output "$OUTPUT/0081_single_plus_restricted.json"
```

For the declared bounded TRAIN150 panel, the existing pack contains FIT families `0081`, `0129`, `0169`, `0251`, with respectively 2, 4, 6 and 12 modules. All four have saved `single_plus` and `single_minus` edits. These are examples of input-selected, common family identities; use the same fixed list, edit identities, query count and seed for all models. One invocation captures one edit and one reader. `--relations 4` chooses a high-mass, a low-mass and seeded positive-mass controls on TRAIN input alone, then freshly prepares both scenes for each named removal. Mass is the mean of the actual applied coefficients over attention heads for that directed relation. It also measures within-role measure-only access and removes one selected original member, retaining structural own-E entries as defined by the existing intervention. It records the nominal and changed field and finite-response errors, actual gates and coefficients, executed decoder work and all additional preparations. Fewer positive relations produce fewer interventions. Relation importance for a prediction is not proof that it improves physical accuracy or reflects physical causation.

Select the relations once on the agreed native reference, then pass its JSON as `--relations-from` for the same family in the other reader, edit or model. This preserves contextual stage, direction and original IDs, and reuses the selected original-member removal. The current model's actual original and edited gates and coefficients are recorded separately, including zero support. For example:

```bash
python tools/thermal_organization_response_diagnostic.py \
  --checkpoint "$OTHER_CHECKPOINT" --pack "$PACK" \
  --family 0081 --action single_minus --role fit \
  --read-mode full --query-count 32 --seed 0 \
  --relations-from "$REFERENCE_OUTPUT/0081_single_plus_restricted.json" \
  --device "$DEVICE" --output "$OUTPUT/0081_single_minus_full.json"
```

The JSON stores provenance, physical metrics, actual work counts, and array references. The adjacent compressed NPZ holds detached stage gates, prepared access, selected query/member coefficients, original messages and counterfactual allocations. The two files are a pair and must stay together. Creation refuses to overwrite either file. Do not select interventions using held-out physical outcomes.

## Shared exposed reference scene and global views

Use the packed reference baseline to capture the same exposed validation scene for each checkpoint. `0679` is a historically exposed development-validation example, not a blind TEST. The capture contains actual contextual stages, full and restricted prepared E-address rows, and final query coefficients. It makes no finite-response attribution for a scene without an edit.

```bash
python tools/thermal_organization_reference_capture.py \
  --checkpoint "$CHECKPOINT" --pack "$PACK" --case 0679 \
  --read-mode restricted --query-count 32 --seed 0 --device "$DEVICE" \
  --output "$OUTPUT/0679_reference.json"
```

Convert either side of a saved-edit capture, or the `before` side of a reference capture, to the established global organization format. This bridge reads the saved arrays without loading a model or fabricating reverse relations. The plotting tool produces access regions, original-member service views and native overlapping hyperedges from actual saved records.

```bash
python tools/organization_response_to_global.py \
  --capture "$OUTPUT/0679_reference.json" --side before --output "$OUTPUT/0679_global.npz"
python tools/plot_honf_organization.py \
  --scene "Current=$OUTPUT/0679_global.npz" --output "$OUTPUT/0679_global_figures"
```

Use separate global galleries for before and edited layouts because their physical member coordinates differ. Same-layout model comparisons can pass two `--scene` arguments with matching IDs, coordinates and checkpoint ages. The global plotter reports its display thresholds and retained mass; a visual threshold is not a training or physical support change. Retain figure PDF masters, PNG previews where needed, relative report links and the numerical capture in the round's ignored DATA2 locations.
