"""R15-3 dense Wind lattice workloads on the unchanged relation-anchored core."""
from __future__ import annotations

from .relation_anchored_task import _build_wind_relation_anchored_from_validated


GRIDS = ((8, 4, 4), (8, 8, 4), (12, 12, 4), (12, 12, 8))


def validate_wind_r15_3_recipe(recipe):
    value = dict(recipe)
    required = {"schema": "wind_relation_anchored_r15_3_v1", "dataset": "wind",
                "architecture": "relation_anchored", "seed": 42,
                "primary_queries": 8192, "effective_batch_size": 24,
                "precision": "fp32", "optimizer_bias_norm_exclusion": True,
                "validation_queries": 4096, "validation_microbatch_size": 4}
    if any(value.get(k) != v for k, v in required.items()):
        raise ValueError("Wind R15-3 recipe differs from its declared scientific workload.")
    if value.get("protocol") not in ("development", "full_train1000"):
        raise ValueError("Wind R15-3 protocol must be development or full_train1000.")
    if tuple(value.get("environment_shape", ())) not in GRIDS:
        raise ValueError("Wind R15-3 requires a declared lattice.")
    if value.get("microbatch_size") not in (1, 3, 6, 12, 24):
        raise ValueError("Wind R15-3 microbatch must be 1, 3, 6, 12, or 24.")
    if value.get("query_tile_size") not in (256, 512, 1024):
        raise ValueError("Wind R15-3 query tile must be 256, 512, or 1024.")
    if type(value.get("total_epochs")) is not int or value["total_epochs"] <= 0:
        raise ValueError("Wind R15-3 total_epochs must be positive.")
    if value.get("gradient_accumulation") != (value["microbatch_size"] != 24):
        raise ValueError("Wind R15-3 accumulation must match its physical batch.")
    cache = (128 if value["protocol"] == "full_train1000" else 32) * 1024**3
    if value.get("catalogue_cache_ram_bytes") != cache:
        raise ValueError("Wind R15-3 requires the existing read-only catalogue RAM bound.")
    return value


def build_wind_r15_3_task(recipe, device):
    value = validate_wind_r15_3_recipe(recipe)
    return _build_wind_relation_anchored_from_validated(
        value, device, full=value["protocol"] == "full_train1000", tuned=True)
