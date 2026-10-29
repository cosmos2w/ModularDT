"""Wind's local-record value transform preserves its actual support domain."""
from dataclasses import replace
import hashlib
import json

SCHEMA = "local_records_context_v1"
OPERATING_FEATURES = (
    "wind_direction_one_hot_270", "wind_direction_one_hot_285", "wind_direction_one_hot_300",
    "turbine_count_fraction", "reference_speed_fraction",
    "support_lower_x_over_positional_scale_D", "support_lower_y_over_positional_scale_D",
    "support_lower_z_over_positional_scale_D", "support_extent_x_over_positional_scale_D",
    "support_extent_y_over_positional_scale_D", "support_extent_z_over_positional_scale_D")


def feature_contract():
    payload = dict(schema=SCHEMA, dataset="wind", operating_features=list(OPERATING_FEATURES),
                   dropped_operating_features=["turbine_count_fraction"], dropped_environment_features=[],
                   retained_operating_features=[x for x in OPERATING_FEATURES if x != "turbine_count_fraction"],
                   environment_features="unchanged own prescribed boundary distances and absolute height",
                   module_features="unchanged own rotor radius and hub height",
                   treatment="zero reserved count slot; preserve direction, speed, true domain, TRAIN scales and units")
    payload["sha256"] = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return payload


def local_records_scene(scene):
    if scene.environment_features.shape[-1] != 7 or scene.operating_features is None or scene.operating_features.shape[-1] != 11:
        raise ValueError("Wind local-record capability requires the named E7/context11 feature inventory.")
    operating = scene.operating_features.clone()
    operating[..., 3] = 0
    return replace(scene, operating_features=operating)
