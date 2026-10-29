"""Deterministic, input-only HET-MF pilot and formal manifest design.

Formal plans describe proposed inputs only. Building or resolving a plan never
solves a PDE and does not authorize formal generation.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json

from .interventions import make_counterfactuals, physical_input_hash


CONTACT_COUNTS = (2, 3, 4, 6, 8, 12)
ENV_FAMILIES = ("correlated_mosaic", "layered_barrier", "inclusions", "corridors")
LAYOUT_FAMILIES = ("dispersed", "rows", "two_cluster")
PROFILE_SCALE = {"compact": 1, "standard": 2, "expanded": 4}
REPRESENTATIVE_SPECS = (
    ("R00", "uniform", 2, "dispersed", "balanced"),
    ("R01", "uniform", 6, "dispersed", "multilevel"),
    ("R02", "correlated_mosaic", 4, "dispersed", "balanced"),
    ("R03", "correlated_mosaic", 8, "rows", "multilevel"),
    ("R04", "layered_barrier", 6, "rows", "balanced"),
    ("R05", "barrier_gap", 4, "dispersed", "balanced"),
    ("R06", "corridors", 6, "rows", "multilevel"),
    ("R07", "inclusions", 8, "two_cluster", "balanced"),
    ("R08", "thermal_parent_R02", 4, "dispersed", "balanced"),
    ("R09", "cooling_parent_R02", 4, "dispersed", "balanced"),
    ("R10", "correlated_mosaic", 12, "two_cluster", "multilevel"),
    ("R11", "composite", 12, "dispersed", "multilevel"),
)


def _seed(root_seed: int, *parts: object) -> int:
    payload = json.dumps([int(root_seed), *parts], separators=(",", ":"), sort_keys=True)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def hash_plan(plan: dict) -> str:
    """Canonical SHA-256, excluding a stored copy of the digest itself."""
    content = {key: value for key, value in plan.items() if key != "plan_hash"}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _physics():
    from .environment import generate_environment
    from .layouts import generate_layout
    return generate_environment, generate_layout


def _environment(seed: int, family: str, root_id: str, **kwargs) -> dict:
    generate_environment, _ = _physics()
    env = generate_environment(seed, family, **kwargs)
    env["root_id"] = root_id
    return env


def _layout(seed: int, m: int, family: str, voltage_family: str, root_id: str) -> dict:
    _, generate_layout = _physics()
    layout = generate_layout(seed, m, family, voltage_family)
    layout["root_id"] = root_id
    return layout


def _scene(env: dict, layout: dict, state_id: str, split: str, **metadata) -> dict:
    scene = {"state_id": state_id, "split": split, "environment_root_id": env["root_id"],
             "layout_root_id": layout["root_id"], "environment": deepcopy(env),
             "contacts": deepcopy(layout["contacts"]), "layout_family": layout["family"],
             "voltage_family": layout.get("voltage_family", "balanced"), **metadata}
    scene["input_hash"] = physical_input_hash(scene)
    return scene


def _representatives(root_seed: int) -> list[dict]:
    reps = []
    for index, (name, family, m, layout_family, voltage_family) in enumerate(REPRESENTATIVE_SPECS):
        if family in ("thermal_parent_R02", "cooling_parent_R02"):
            parent = reps[2]
            env = deepcopy(parent["environment"])
            layout = {"root_id": parent["layout_root_id"], "family": parent["layout_family"],
                      "voltage_family": parent["voltage_family"], "contacts": deepcopy(parent["contacts"])}
            field = "kappa" if name == "R08" else "beta"
            # A strictly local change preserves the entire electrical problem.
            for iy in range(4, 7):
                for ix in range(4, 7):
                    env[field][iy][ix] *= 1.2 if field == "kappa" else 1.25
            env["family"] = parent["environment"]["family"]
        else:
            kwargs = {"contrast_s": 100.0} if name == "R11" else {}
            env = _environment(_seed(root_seed, "pilot", "representative", name, "environment"), family,
                               f"pilot:representative:{name}:environment", **kwargs)
            layout = _layout(_seed(root_seed, "pilot", "representative", name, "layout"), m,
                             layout_family, voltage_family, f"pilot:representative:{name}:layout")
        scene = _scene(env, layout, name, "pilot", representative_id=name,
                       diagnostic="out_of_core" if name == "R11" else "core_or_control")
        if name in ("R08", "R09"):
            scene["electrical_parent_state_id"] = "R02"
        reps.append(scene)
    return reps


def _throughput(root_seed: int) -> list[dict]:
    families = (["correlated_mosaic"] * 26 + ["layered_barrier"] * 13 +
                ["inclusions"] * 13 + ["corridors"] * 12)
    scenes = []
    for env_index, family in enumerate(families):
        env = _environment(_seed(root_seed, "pilot", "throughput", "environment", env_index), family,
                           f"pilot:throughput:environment:{env_index:03d}")
        for within in range(4):
            state_index = env_index * 4 + within
            m = CONTACT_COUNTS[(env_index * 4 + within) % len(CONTACT_COUNTS)]
            layout_family = LAYOUT_FAMILIES[(env_index + within) % len(LAYOUT_FAMILIES)]
            voltage_family = "balanced" if (env_index + within) % 2 == 0 or m == 2 else "multilevel"
            layout = _layout(_seed(root_seed, "pilot", "throughput", "layout", state_index), m,
                             layout_family, voltage_family, f"pilot:throughput:layout:{state_index:03d}")
            scenes.append(_scene(env, layout, f"P{state_index:03d}", "pilot", panel="throughput"))
    return scenes


def build_pilot_plan(seed: int = 20261017) -> dict:
    """Freeze twelve representative, six quartet, and 256 throughput inputs."""
    reps = _representatives(seed)
    quartet_kinds = ("me_patch_move", "me_rotate_gamma", "mm_voltages", "mm_moves",
                     "ee_electrical", "ee_cross")
    parents = (reps[2], reps[6], reps[1], reps[7], reps[3], reps[4])
    quartets = [make_counterfactuals(parent, kind, f"Q{index:02d}")
                for index, (parent, kind) in enumerate(zip(parents, quartet_kinds))]
    throughput = _throughput(seed)
    # F00 is the representative parent and is stored by reference to avoid a
    # second physical copy. The other 18 endpoints are distinct pilot states.
    quartet_new = [endpoint for quartet in quartets for endpoint in quartet["endpoints"]
                   if endpoint["quartet_endpoint"] != "00"]
    scenes = reps + quartet_new + throughput
    if len(scenes) != 286 or len({scene["state_id"] for scene in scenes}) != 286:
        raise AssertionError("pilot state accounting failed")
    plan = {"schema": "hetmf_input_plan_v1", "kind": "pilot", "seed": int(seed),
            "formal_generation_authorized": False, "representative_ids": [s["state_id"] for s in reps],
            "refinement_ids": [s["state_id"] for s in reps],
            "refinement_grids": [48, 96, 192], "optional_reference_grid": 384,
            "quartets": [{key: value for key, value in q.items() if key != "endpoints"} |
                         {"endpoint_state_ids": [q["root_state_id"] if e["quartet_endpoint"] == "00"
                                                 else e["state_id"] for e in q["endpoints"]],
                          "endpoint_input_hashes": [e["input_hash"] for e in q["endpoints"]]}
                         for q in quartets],
            "throughput_ids": [s["state_id"] for s in throughput], "scenes": scenes,
            "counts": {"representatives": 12, "quartets": 6, "quartet_endpoints": 24,
                       "throughput": 256, "distinct_states": 286}}
    plan["plan_hash"] = hash_plan(plan)
    return plan


def replay_pilot_layout_diagnostics(plan: dict) -> dict:
    """Replay layout proposals beside a frozen pilot plan without editing it.

    This sidecar restores geometry-attempt accounting from the keyed layout
    namespace and refuses any mismatch with the saved physical contacts.
    """
    if plan.get("kind") != "pilot" or plan.get("plan_hash") != hash_plan(plan):
        raise ValueError("valid frozen pilot plan required")
    by_root = {}
    for scene in plan["scenes"]:
        root = scene["layout_root_id"]
        previous = by_root.get(root)
        if previous is not None and previous["contacts"] != scene["contacts"]:
            # Quartet descendants intentionally edit contacts. Their parent
            # source is the representative whose ID names the root.
            if scene.get("quartet_endpoint") is not None:
                continue
            raise ValueError(f"inconsistent ordinary layout root: {root}")
        if previous is None or (previous.get("quartet_endpoint") is not None and scene.get("quartet_endpoint") is None):
            by_root[root] = scene
    diagnostics = []
    for root, scene in sorted(by_root.items()):
        if root.startswith("pilot:representative:") and root.endswith(":layout"):
            representative = root.split(":")[2]
            seed = _seed(plan["seed"], "pilot", "representative", representative, "layout")
        elif root.startswith("pilot:throughput:layout:"):
            index = int(root.rsplit(":", 1)[1])
            seed = _seed(plan["seed"], "pilot", "throughput", "layout", index)
        else:
            raise ValueError(f"unknown pilot layout root namespace: {root}")
        _, generate_layout = _physics()
        replay = generate_layout(seed, len(scene["contacts"]), scene["layout_family"], scene["voltage_family"])
        if replay["contacts"] != scene["contacts"]:
            raise ValueError(f"frozen layout replay mismatch: {root}")
        diagnostics.append({"layout_root_id": root, "seed": seed,
                            "m": len(scene["contacts"]), "family": scene["layout_family"],
                            "voltage_family": scene["voltage_family"],
                            "proposal_count": replay["proposal_count"],
                            "rejected_proposals": replay["rejected_proposals"],
                            "minimum_pairwise_gap": replay["minimum_pairwise_gap"],
                            "contacts_sha256": hashlib.sha256(json.dumps(scene["contacts"], sort_keys=True,
                                                                    separators=(",", ":")).encode()).hexdigest()})
    return {"schema": "hetmf_pilot_layout_replay_v1", "pilot_plan_hash": plan["plan_hash"],
            "layout_roots": len(diagnostics),
            "geometry_proposals": sum(item["proposal_count"] for item in diagnostics),
            "rejected_geometry_proposals": sum(item["rejected_proposals"] for item in diagnostics),
            "layouts": diagnostics}


def _formal_counts(profile: str) -> dict:
    if profile not in PROFILE_SCALE:
        raise ValueError(f"unknown profile: {profile}")
    scale = PROFILE_SCALE[profile]
    return {"train_environments": 120 * scale, "dev_environments": 20 * scale,
            "test_environments": 20 * scale, "ordinary_train": 1440 * scale,
            "ordinary_dev": 240 * scale, "ordinary_test": 240 * scale,
            "panel": 256 * scale, "quartets": 32 * scale,
            "counterfactual_states": 128 * scale, "total_states": 2304 * scale}


def _catalogue(root_seed: int, split: str, nenv: int) -> tuple[dict, dict, list[dict]]:
    envs, layouts, rows = {}, {}, []
    block_env, block_layout = (120, 40) if split == "train" else (20, 8)
    if nenv % block_env:
        raise AssertionError("profile environment count must respect nested blocks")
    per_m = (nenv // block_env) * block_layout
    for ei in range(nenv):
        family = ("correlated_mosaic", "layered_barrier", "correlated_mosaic",
                  "inclusions", "corridors")[ei % 5]
        env_id = f"formal:{split}:environment:{ei:04d}"
        envs[env_id] = _environment(_seed(root_seed, "formal", split, "environment", ei), family, env_id)
    for mi, m in enumerate(CONTACT_COUNTS):
        for li in range(per_m):
            layout_id = f"formal:{split}:layout:m{m}:{li:04d}"
            family = LAYOUT_FAMILIES[(li + mi) % len(LAYOUT_FAMILIES)]
            voltage = "balanced" if m == 2 or (li + mi) % 2 == 0 else "multilevel"
            layouts[layout_id] = _layout(_seed(root_seed, "formal", split, "layout", m, li),
                                         m, family, voltage, layout_id)
    for ei in range(nenv):
        env_id = f"formal:{split}:environment:{ei:04d}"
        for m in CONTACT_COUNTS:
            # Two distinct partners per stratum; every layout appears exactly
            # 2*nenv/per_m = 6 times in this split.
            for local in (0, 1):
                block = ei // block_env
                li = block * block_layout + (2 * (ei % block_env) + local) % block_layout
                layout_id = f"formal:{split}:layout:m{m}:{li:04d}"
                rows.append(_row(envs, layouts, env_id, layout_id, split,
                                 f"formal:{split}:e{ei:04d}:m{m}:l{local}"))
    return envs, layouts, rows


def _row(envs: dict, layouts: dict, env_id: str, layout_id: str, split: str,
         state_id: str, **metadata) -> dict:
    scene = _scene(envs[env_id], layouts[layout_id], state_id, split)
    return {"state_id": state_id, "split": split, "environment_root_id": env_id,
            "layout_root_id": layout_id, "input_hash": scene["input_hash"], **metadata}


def resolve_state(plan: dict, row: dict | str) -> dict:
    """Expand one catalogue row to the full, immutable physical input."""
    if isinstance(row, str):
        row = next((item for item in plan["states"] if item["state_id"] == row), None)
        if row is None:
            raise KeyError("state ID absent from plan")
    env = plan["environments"][row["environment_root_id"]]
    layout = plan["layouts"][row["layout_root_id"]]
    scene = _scene(env, layout, row["state_id"], row["split"])
    if "input_override" in row:
        scene["environment"] = deepcopy(row["input_override"]["environment"])
        scene["contacts"] = deepcopy(row["input_override"]["contacts"])
        scene["input_hash"] = physical_input_hash(scene)
    if scene["input_hash"] != row["input_hash"]:
        raise ValueError("formal input hash mismatch")
    return scene


def build_formal_plan(profile: str = "standard", seed: int = 20261017) -> dict:
    """Build proposed catalogue and exact rows; never dispatch physical solves."""
    counts = _formal_counts(profile)
    scale = PROFILE_SCALE[profile]
    envs, layouts, rows = {}, {}, []
    for split in ("train", "dev", "test"):
        e, l, r = _catalogue(seed, split, counts[f"{split}_environments"])
        envs.update(e); layouts.update(l); rows.extend(r)
    train_env = [k for k in envs if k.startswith("formal:train:environment:")]
    train_layout = [k for k in layouts if k.startswith("formal:train:layout:")]
    panel_each = 64 * scale
    # The first three panels declare their component-exposure semantics.
    for i in range(panel_each):
        env_id = f"formal:panel:new_environment:{i // 4:04d}"
        if env_id not in envs:
            family = ENV_FAMILIES[(i // 4) % len(ENV_FAMILIES)]
            envs[env_id] = _environment(_seed(seed, "formal", "panel", "new_environment", i // 4), family, env_id)
        block, local = divmod(i, 64)
        m = CONTACT_COUNTS[local % len(CONTACT_COUNTS)]
        lid = f"formal:train:layout:m{m}:{block * 40 + (local * 7) % 40:04d}"
        rows.append(_row(envs, layouts, env_id, lid, "panel_new_environment",
                         f"formal:panel:new_environment:{i:04d}", environment_seen_in_train=False,
                         layout_seen_in_train=True))
    for i in range(panel_each):
        lid = f"formal:panel:new_layout:{i:04d}"
        m = CONTACT_COUNTS[i % len(CONTACT_COUNTS)]
        layouts[lid] = _layout(_seed(seed, "formal", "panel", "new_layout", i), m,
                               LAYOUT_FAMILIES[i % 3], "balanced" if m == 2 or i % 2 == 0 else "multilevel", lid)
        block, local = divmod(i, 64)
        eid = f"formal:train:environment:{block * 120 + (local * 17) % 120:04d}"
        rows.append(_row(envs, layouts, eid, lid, "panel_new_layout", f"formal:panel:new_layout:{i:04d}",
                         environment_seen_in_train=True, layout_seen_in_train=False))
    core_pairs = {(r["environment_root_id"], r["layout_root_id"]) for r in rows if r["split"] == "train"}
    for i in range(panel_each):
        block, local = divmod(i, 64)
        eid = f"formal:train:environment:{block * 120 + (local * 17) % 120:04d}"
        m = CONTACT_COUNTS[i % len(CONTACT_COUNTS)]
        original_ei = int(eid.rsplit(":", 1)[1])
        li = (original_ei // 120) * 40 + (2 * (original_ei % 120) + 2) % 40
        lid = f"formal:train:layout:m{m}:{li:04d}"
        if (eid, lid) in core_pairs:
            raise AssertionError("new-pair panel reused a TRAIN pair")
        rows.append(_row(envs, layouts, eid, lid, "panel_new_pair", f"formal:panel:new_pair:{i:04d}",
                         environment_seen_in_train=True, layout_seen_in_train=True))
    challenges = ("m10", "m16", "high_contrast", "composite")
    for challenge in challenges:
        for i in range(panel_each // 4):
            eid = f"formal:panel:extrapolation:{challenge}:environment:{i:04d}"
            kwargs = {"contrast_s": 100.0} if challenge == "high_contrast" else {}
            family = "composite" if challenge == "composite" else ENV_FAMILIES[i % len(ENV_FAMILIES)]
            envs[eid] = _environment(_seed(seed, "formal", "panel", challenge, "environment", i), family, eid, **kwargs)
            m = 10 if challenge == "m10" else 16 if challenge == "m16" else CONTACT_COUNTS[i % len(CONTACT_COUNTS)]
            lid = f"formal:panel:extrapolation:{challenge}:layout:{i:04d}"
            layouts[lid] = _layout(_seed(seed, "formal", "panel", challenge, "layout", i), m,
                                   LAYOUT_FAMILIES[i % 3], "balanced" if m == 2 or i % 2 == 0 else "multilevel", lid)
            rows.append(_row(envs, layouts, eid, lid, "panel_extrapolation",
                             f"formal:panel:extrapolation:{challenge}:{i:04d}", challenge=challenge,
                             environment_seen_in_train=False, layout_seen_in_train=False))
    # Audit quartets own their roots. Each endpoint is a separate physical state.
    quartet_kinds = ("me_patch_move", "me_patch_move", "mm_voltages", "ee_electrical")
    quartets = []
    for qi in range(counts["quartets"]):
        family = quartet_kinds[qi % len(quartet_kinds)]
        eid = f"formal:audit:environment:{qi:04d}"
        lid = f"formal:audit:layout:{qi:04d}"
        envs[eid] = _environment(_seed(seed, "formal", "audit", "environment", qi), ENV_FAMILIES[qi % 4], eid)
        m = CONTACT_COUNTS[(qi % 5) + 1] if family == "mm_voltages" else CONTACT_COUNTS[qi % 6]
        layouts[lid] = _layout(_seed(seed, "formal", "audit", "layout", qi), m,
                               LAYOUT_FAMILIES[qi % 3], "balanced" if m == 2 else "multilevel", lid)
        base = _scene(envs[eid], layouts[lid], f"formal:audit:q{qi:04d}:base", "audit")
        quartet = make_counterfactuals(base, family, f"formal:audit:q{qi:04d}")
        quartets.append({"quartet_id": quartet["quartet_id"], "kind": family,
                         "interventions": quartet["interventions"],
                         "endpoint_state_ids": [ep["state_id"] for ep in quartet["endpoints"]]})
        for ep in quartet["endpoints"]:
            # Store intervention endpoints as full input overrides because
            # catalogue roots intentionally refer to the unperturbed parents.
            rows.append({"state_id": ep["state_id"], "split": "audit", "environment_root_id": eid,
                         "layout_root_id": lid, "input_hash": ep["input_hash"],
                         "input_override": {"environment": ep["environment"], "contacts": ep["contacts"]},
                         "quartet_id": quartet["quartet_id"], "quartet_endpoint": ep["quartet_endpoint"]})
    split_counts = Counter(r["split"] for r in rows)
    if len(rows) != counts["total_states"]:
        raise AssertionError(f"formal count mismatch: {len(rows)} vs {counts['total_states']}")
    plan = {"schema": "hetmf_input_plan_v1", "kind": "formal_proposal", "profile": profile,
            "seed": int(seed), "formal_generation_authorized": False, "counts": counts,
            "split_counts": dict(split_counts), "environments": envs, "layouts": layouts,
            "states": rows, "quartets": quartets,
            "core_root_rule": "train/dev/test disjoint on environment and layout axes",
            "pilot_namespace_disjoint": True}
    plan["plan_hash"] = hash_plan(plan)
    return plan
