"""ThermalChannel request vocabulary for inverse design.

Physical design ``D`` is the modular channel layout, context ``c`` contains
flow/material/domain values, request ``R`` uses exactly the names below,
compact plan ``G`` is the desired fixed-edge mechanism, and ``G_hat`` is the
realized plan recovered by frozen HONF verification.
"""

from __future__ import annotations


REQUEST_SCHEMA_NAME = "thermalchannel_inverse_request"
REQUEST_SCHEMA_VERSION = 1

REQUEST_TYPES = (
    "environment_temperature_max",
    "pressure_drop",
    "outlet_temperature_nonuniformity",
    "internal_temperature_max",
    "internal_temperature_spread",
    "regional_temperature_mean",
    "regional_temperature_max",
)

REQUEST_TYPE_TO_ID = {name: index for index, name in enumerate(REQUEST_TYPES)}

REGIONAL_REQUEST_TYPES = (
    "regional_temperature_mean",
    "regional_temperature_max",
)

NONREGIONAL_REQUEST_TYPES = tuple(name for name in REQUEST_TYPES if name not in REGIONAL_REQUEST_TYPES)

FUNCTIONAL_UNITS = {
    "environment_temperature_max": "temperature",
    "pressure_drop": "pressure",
    "outlet_temperature_nonuniformity": "temperature",
    "internal_temperature_max": "temperature",
    "internal_temperature_spread": "temperature",
    "regional_temperature_mean": "temperature",
    "regional_temperature_max": "temperature",
}


__all__ = [
    "FUNCTIONAL_UNITS",
    "NONREGIONAL_REQUEST_TYPES",
    "REGIONAL_REQUEST_TYPES",
    "REQUEST_SCHEMA_NAME",
    "REQUEST_SCHEMA_VERSION",
    "REQUEST_TYPES",
    "REQUEST_TYPE_TO_ID",
]
