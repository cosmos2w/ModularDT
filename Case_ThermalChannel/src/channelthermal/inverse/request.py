"""ThermalChannel binding for the versioned structured request language.

Physical design ``D`` and operating context ``c`` are supplied separately.
Request ``R`` contains the supported functional set and geometry block. Compact
plan ``G`` and realized plan ``G_hat`` are conditioned/evaluated using the
normalized tensors produced here.
"""

from __future__ import annotations

from typing import Mapping

from honf_inverse_core.normalization import ScalarStats
from honf_inverse_core.request_schema import RequestCodec

from .vocabulary import REGIONAL_REQUEST_TYPES, REQUEST_SCHEMA_NAME, REQUEST_TYPES


def make_request_codec(
    normalization: Mapping[str, ScalarStats] | None = None,
) -> RequestCodec:
    """Create the strict ThermalChannel schema-v1 request codec."""

    return RequestCodec(
        schema_name=REQUEST_SCHEMA_NAME,
        request_types=REQUEST_TYPES,
        regional_types=REGIONAL_REQUEST_TYPES,
        normalization=normalization,
    )


__all__ = ["make_request_codec"]
