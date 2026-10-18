"""Thermal's explicitly versioned local-record neural input capability."""
from dataclasses import replace
import hashlib
import json

SCHEMA = "local_records_context_v1"
ENVIRONMENT_FEATURES = (
    "x_over_Lx", "y_over_Ly", "nearest_module_distance_over_radius_minus_one",
    "module_gaussian_occupancy", "distance_from_lower_x", "distance_from_lower_y",
    "distance_to_upper_x", "distance_to_upper_y")
OPERATING_FEATURES = (
    "Re_over_100", "inlet_velocity", "nu_times_100", "solid_alpha_times_100",
    "fluid_alpha_times_100", "solid_k", "fluid_k", "module_radius", "Lx_over_12",
    "Ly_over_6", "module_count_over_12", "module_count_per_domain_area",
    "reserved_zero_0", "reserved_zero_1")
DROPPED_ENVIRONMENT = ENVIRONMENT_FEATURES[2:4]
DROPPED_OPERATING = OPERATING_FEATURES[10:12]


def feature_contract():
    payload = dict(schema=SCHEMA, dataset="thermal", environment_features=list(ENVIRONMENT_FEATURES),
                   operating_features=list(OPERATING_FEATURES),
                   dropped_environment_features=list(DROPPED_ENVIRONMENT),
                   dropped_operating_features=list(DROPPED_OPERATING),
                   retained_environment_features=[x for x in ENVIRONMENT_FEATURES if x not in DROPPED_ENVIRONMENT],
                   retained_operating_features=[x for x in OPERATING_FEATURES if x not in DROPPED_OPERATING],
                   module_features="unchanged own-record geometry and prescribed case-global material constants",
                   treatment="zero reserved slots; preserve TRAIN scales, units, physical inventory and domain")
    payload["sha256"] = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return payload


def local_records_scene(scene):
    if scene.environment_features.shape[-1] != 8 or scene.operating_features is None or scene.operating_features.shape[-1] != 14:
        raise ValueError("Thermal local-record capability requires the named E8/context14 feature inventory.")
    environment = scene.environment_features.clone()
    operating = scene.operating_features.clone()
    environment[..., 2:4] = 0
    operating[..., 10:12] = 0
    return replace(scene, environment_features=environment, operating_features=operating)
