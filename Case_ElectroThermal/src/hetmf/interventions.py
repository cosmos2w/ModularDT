"""Input-frozen electro-thermal counterfactual quartets.

The four endpoints of a quartet differ only at the declared intervention sites.
No solution field is inspected when choosing an intervention.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import numpy as np


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def physical_input_hash(scene: dict) -> str:
    """Hash only the physical problem, excluding provenance and solved targets."""
    env = scene["environment"]
    controls = {key: env[key] for key in ("s", "kappa", "beta")}
    controls["renderer"] = env.get("renderer", "log_bilinear_centers_border_v1")
    contacts = [{key: contact[key] for key in ("x", "y", "r", "gamma", "u")} for contact in scene["contacts"]]
    return _digest({"environment": controls, "contacts": contacts})


def _legal_move(contacts: list[dict], index: int, distance: float = 0.025) -> tuple[float, float]:
    contact = contacts[index]
    x, y, radius = (float(contact[k]) for k in ("x", "y", "r"))
    for dx, dy in ((distance, 0), (0, distance), (-distance, 0), (0, -distance),
                   (distance / math.sqrt(2), distance / math.sqrt(2))):
        xx, yy = x + dx, y + dy
        if min(xx, yy, 1 - xx, 1 - yy) < radius + 0.025:
            continue
        if any(math.hypot(xx - float(c["x"]), yy - float(c["y"])) < radius + float(c["r"]) + 0.015
               for j, c in enumerate(contacts) if j != index):
            continue
        return xx, yy
    raise ValueError("no legal fixed-distance pad displacement")


def _patch(env: dict, field: str, box: tuple[int, int, int, int], factor: float) -> dict:
    out = deepcopy(env)
    y0, y1, x0, x1 = box
    arr = np.asarray(out[field], dtype=np.float64)
    if arr.shape != (12, 12):
        raise ValueError("material controls must have shape 12x12")
    arr[y0:y1, x0:x1] *= factor
    if not np.isfinite(arr).all() or (arr <= 0).any():
        raise ValueError("invalid material patch")
    out[field] = arr.tolist()
    return out


def _intervene(scene: dict, kind: str, axis: str) -> tuple[dict, dict]:
    out = deepcopy(scene)
    contacts = out["contacts"]
    env = out["environment"]
    if kind == "me_patch_move":
        if axis == "A":
            contacts[0]["x"], contacts[0]["y"] = _legal_move(contacts, 0)
            spec = {"type": "pad_displacement", "contact": contacts[0]["id"], "to": [contacts[0]["x"], contacts[0]["y"]]}
        else:
            out["environment"] = _patch(env, "s", (4, 6, 4, 6), 1.15)
            spec = {"type": "electrical_control_patch", "box_yx": [4, 6, 4, 6], "factor": 1.15}
    elif kind == "me_rotate_gamma":
        if axis == "A":
            out["environment"]["s"] = np.rot90(np.asarray(env["s"], dtype=np.float64)).tolist()
            spec = {"type": "electrical_control_rotation", "quarter_turns": 1, "unchanged": ["kappa", "beta"]}
        else:
            contacts[0]["gamma"] = float(contacts[0]["gamma"]) * 1.15
            spec = {"type": "contact_gamma_factor", "contact": contacts[0]["id"], "factor": 1.15}
    elif kind == "mm_voltages":
        if len(contacts) < 3:
            raise ValueError("voltage interaction needs at least three contacts")
        index = 0 if axis == "A" else 1
        contacts[index]["u"] = float(contacts[index]["u"]) + (0.075 if axis == "A" else -0.06)
        spec = {"type": "voltage_increment", "contact": contacts[index]["id"], "delta": 0.075 if axis == "A" else -0.06}
    elif kind == "mm_moves":
        index = 0 if axis == "A" else 1
        contacts[index]["x"], contacts[index]["y"] = _legal_move(contacts, index)
        spec = {"type": "pad_displacement", "contact": contacts[index]["id"], "to": [contacts[index]["x"], contacts[index]["y"]]}
    elif kind == "ee_electrical":
        box = (1, 3, 1, 3) if axis == "A" else (8, 10, 8, 10)
        out["environment"] = _patch(env, "s", box, 1.15)
        spec = {"type": "electrical_control_patch", "box_yx": list(box), "factor": 1.15}
    elif kind == "ee_cross":
        box = (1, 3, 1, 3) if axis == "A" else (8, 10, 8, 10)
        field = "s" if axis == "A" else "beta"
        out["environment"] = _patch(env, field, box, 1.15)
        spec = {"type": "control_patch", "field": field, "box_yx": list(box), "factor": 1.15}
    else:
        raise ValueError(f"unknown quartet kind: {kind}")
    return out, spec


def make_counterfactuals(base: dict, kind: str, quartet_id: str) -> dict:
    """Return one input-only quartet, retaining root IDs and exact unchanged values."""
    if kind not in {"me_patch_move", "me_rotate_gamma", "mm_voltages", "mm_moves", "ee_electrical", "ee_cross"}:
        raise ValueError(f"unknown quartet kind: {kind}")
    endpoints = []
    a_only, spec_a = _intervene(base, kind, "A")
    b_only, spec_b = _intervene(base, kind, "B")
    specs = {"A": spec_a, "B": spec_b}
    for a, b in ((0, 0), (1, 0), (0, 1), (1, 1)):
        scene = deepcopy(base)
        if a:
            scene = deepcopy(a_only)
        if b:
            if not a:
                scene = deepcopy(b_only)
            elif kind in ("me_patch_move", "me_rotate_gamma", "mm_voltages", "mm_moves"):
                if kind == "me_patch_move":
                    scene["environment"] = deepcopy(b_only["environment"])
                elif kind == "me_rotate_gamma":
                    scene["contacts"][0]["gamma"] = b_only["contacts"][0]["gamma"]
                elif kind == "mm_voltages":
                    scene["contacts"][1]["u"] = b_only["contacts"][1]["u"]
                else:
                    scene["contacts"][1]["x"] = b_only["contacts"][1]["x"]
                    scene["contacts"][1]["y"] = b_only["contacts"][1]["y"]
            else:
                # EE axes touch disjoint control regions; overlay exactly the
                # B patch, retaining all A values bit for bit.
                field = "beta" if kind == "ee_cross" else "s"
                scene["environment"][field][8:10] = deepcopy(b_only["environment"][field][8:10])
        if kind == "mm_moves":
            from .layouts import validate_contacts
            validate_contacts(scene["contacts"])
        scene["state_id"] = f"{quartet_id}_{a}{b}"
        scene["quartet_id"] = quartet_id
        scene["quartet_endpoint"] = f"{a}{b}"
        scene["interaction_family_id"] = kind
        scene["input_hash"] = physical_input_hash(scene)
        endpoints.append(scene)
    return {"quartet_id": quartet_id, "kind": kind, "interventions": specs, "endpoints": endpoints,
            "root_state_id": base["state_id"]}
