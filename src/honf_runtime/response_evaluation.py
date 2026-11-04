"""Small in-memory response measurements on freshly prepared case inputs.

Callers own legality, original IDs, reference matching and input selection. Returned
traces describe executed learned access, not physical causality. No fitting,
reference lookup, persistence, or solver invocation is performed here.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch


@dataclass(frozen=True)
class ResponseDesign:
    """A named input design and the case-specific inputs needed to rebuild it."""

    name: str
    inputs: Any


def _sync(model: Any) -> None:
    parameter = next(model.parameters(), None)
    if parameter is not None and parameter.is_cuda:
        torch.cuda.synchronize(parameter.device)


def _organization_evidence(organization: Any) -> dict[str, Any]:
    return {
        "applied_stages": getattr(organization, "context_stage_traces", ()),
        "stage_metadata": getattr(organization, "context_stage_metadata", ()),
        "directory": getattr(organization, "e_access_directory", None),
        "directory_route_provenance": getattr(organization, "anchored_route_provenance", {}),
        "scene": organization.scene,
    }


def evaluate_response_designs(
    model: Any,
    designs: Sequence[ResponseDesign],
    build: Callable[[Any], tuple[Any, Any]],
    native: Callable[[Any, Any, Any], Any],
) -> dict[str, Any]:
    """Evaluate a panel via a case adapter's actual scene/query constructors.

    ``build`` must freshly construct (scene, query) from each design. ``native``
    extracts physical fields and masks from (scene, query, actual core output).
    Results retain the native tensors, original IDs, actual prepared organization,
    decoder trace and executed work. Timings include tracing and are not a
    benchmark of uninstrumented execution. Matched reference comparisons belong
    to the caller and require common masks and coordinate conventions.
    """
    if not designs or len({design.name for design in designs}) != len(designs):
        raise ValueError("A response panel needs uniquely named designs.")
    rows = []
    with torch.no_grad():
        for design in designs:
            _sync(model)
            start = perf_counter()
            scene, query = build(design.inputs)
            _sync(model)
            built = perf_counter()
            organization = model.core.prepare(scene, return_trace=True)
            _sync(model)
            prepared = perf_counter()
            output = model.core.decode(organization, query, return_trace=True)
            result = native(scene, query, output)
            _sync(model)
            ended = perf_counter()
            rows.append({
                "name": design.name,
                "module_ids": scene.module_ids,
                "environment_ids": scene.environment_ids,
                "module_mask": scene.module_mask,
                "environment_mask": scene.environment_mask,
                "query_coordinates": query.coordinates,
                "query_mask": query.query_mask,
                "native": result,
                "organization": _organization_evidence(organization),
                "query_trace": output.trace,
                "work": {"prepare": dict(organization.work_counts), "decode": dict(output.work_counts)},
                "seconds": {"build": built - start, "prepare": prepared - built,
                            "decode_and_native": ended - prepared, "total": ended - start},
            })
    return {"designs": rows, "reference": "not supplied", "interpretation": "software response mechanics"}


def evaluate_thermal_response_designs(
    model: Any, designs: Sequence[ResponseDesign], *, trace_query_indices: Sequence[int] = (0,),
) -> dict[str, Any]:
    """Run Thermal native extraction for complete, freshly prepared designs.

    Each input mapping supplies structure, fluid_xy and optionally physical_heat,
    local_query_points, ntheta and scene_key, as for ``prepare_native``. Masks
    remain those of the native role extractor and the rebuilt scene/query.
    """
    rows = []
    if not designs or len({design.name for design in designs}) != len(designs):
        raise ValueError("A response panel needs uniquely named designs.")
    with torch.no_grad():
        for design in designs:
            args = dict(design.inputs)
            structure, fluid_xy = args.pop("structure"), args.pop("fluid_xy")
            heat = args.pop("physical_heat", None)
            _sync(model)
            start = perf_counter()
            prepared = model.prepare_native(structure, fluid_xy, physical_heat=heat,
                                            capture_context_trace=True, **args)
            _sync(model)
            built = perf_counter()
            native, counts, _output = model.apply_native_with_work_counts(prepared, heat)
            _sync(model)
            decoded = perf_counter()
            from honf_forward_core.content_hypergraph.query_chunks_v1 import select_query_rows_v1
            selected_query = select_query_rows_v1(prepared.query_batch, trace_query_indices)
            output = model.core.decode(prepared.organization, selected_query, return_trace=True)
            _sync(model)
            ended = perf_counter()
            scene, query = prepared.scene_batch, prepared.query_batch
            rows.append({
                "name": design.name, "module_ids": scene.module_ids,
                "environment_ids": scene.environment_ids, "module_mask": scene.module_mask,
                "environment_mask": scene.environment_mask, "query_coordinates": query.coordinates,
                "query_mask": query.query_mask, "native": native,
                "native_solid_mask": prepared.native_solid_mask,
                "receiver_valid_masks": {name: stencil.valid for name, stencil in prepared.stencils.items()},
                "organization": _organization_evidence(prepared.organization),
                "query_trace": output.trace, "trace_query_indices": tuple(trace_query_indices),
                "work": {"prepare": dict(prepared.organization.work_counts), "decode": counts,
                         "trace_decode": dict(output.work_counts)},
                "seconds": {"build_and_prepare": built - start,
                            "decode_and_native": decoded - built, "trace": ended - decoded,
                            "total_with_trace": ended - start},
            })
    return {"designs": rows, "reference": "not supplied", "interpretation": "software response mechanics"}
