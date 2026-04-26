from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def _as_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim > 0 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _physical_xy(norm_xy: np.ndarray, lx: float, ly: float) -> Dict[str, float]:
    return {"x": float(norm_xy[0] * lx), "y": float(norm_xy[1] * ly)}


def build_hypergraph(aux: Dict[str, Any], cylinders: List[Dict[str, float]], lx: float, ly: float) -> Dict:
    result = {
        "cylinders": [
            {"id": f"C{i}", "x": float(cyl["x"]), "y": float(cyl["y"])}
            for i, cyl in enumerate(cylinders)
        ],
        "env_tokens": [],
        "hyperedges": [],
        "links": [],
    }

    env_coords = _as_numpy(aux.get("env_coords"))
    A_eh = _as_numpy(aux.get("A_eh"))
    if env_coords is not None:
        for i, coord in enumerate(env_coords):
            group = None
            confidence = None
            if A_eh is not None and i < A_eh.shape[0]:
                group = int(np.argmax(A_eh[i]))
                confidence = float(np.max(A_eh[i]))
            token = {"id": int(i), **_physical_xy(coord, lx, ly), "group": group, "confidence": confidence}
            result["env_tokens"].append(token)

    hyper_source = _as_numpy(aux.get("hyper_source_coords"))
    hyper_wake = _as_numpy(aux.get("hyper_wake_coords"))
    hyper_axis = _as_numpy(aux.get("hyper_wake_axis"))
    hyper_strength = _as_numpy(aux.get("hyper_strength"))
    A_mh = _as_numpy(aux.get("A_mh"))

    num_hyper = 0
    for arr in (hyper_source, hyper_wake, hyper_axis, hyper_strength):
        if arr is not None and arr.ndim >= 1:
            num_hyper = max(num_hyper, int(arr.shape[0]))
    if A_mh is not None and A_mh.ndim == 2:
        num_hyper = max(num_hyper, int(A_mh.shape[1]))
    if A_eh is not None and A_eh.ndim == 2:
        num_hyper = max(num_hyper, int(A_eh.shape[1]))

    for h in range(num_hyper):
        source = _physical_xy(hyper_source[h], lx, ly) if hyper_source is not None and h < len(hyper_source) else None
        wake = _physical_xy(hyper_wake[h], lx, ly) if hyper_wake is not None and h < len(hyper_wake) else None
        axis = None
        if hyper_axis is not None and h < len(hyper_axis):
            axis = {"x": float(hyper_axis[h][0]), "y": float(hyper_axis[h][1])}
        strength = None
        if hyper_strength is not None and h < len(hyper_strength):
            strength = float(np.asarray(hyper_strength[h]).reshape(-1)[0])

        top_cylinders = []
        if A_mh is not None and A_mh.ndim == 2 and h < A_mh.shape[1]:
            order = np.argsort(-A_mh[:, h])
            for idx in order[: min(4, len(order))]:
                weight = float(A_mh[idx, h])
                if weight > 0:
                    top_cylinders.append({"id": f"C{int(idx)}", "weight": weight})
                    result["links"].append(
                        {"source": f"C{int(idx)}", "target": f"H{h}", "type": "cylinder-hyperedge", "weight": weight}
                    )

        result["hyperedges"].append(
            {
                "id": f"H{h}",
                "strength": strength,
                "source": source,
                "wake": wake,
                "axis": axis,
                "top_cylinders": top_cylinders,
            }
        )

    if A_eh is not None and A_eh.ndim == 2:
        for e in range(A_eh.shape[0]):
            top_h = int(np.argmax(A_eh[e]))
            weight = float(A_eh[e, top_h])
            result["links"].append({"source": f"E{e}", "target": f"H{top_h}", "type": "env-hyperedge", "weight": weight})

    return result
